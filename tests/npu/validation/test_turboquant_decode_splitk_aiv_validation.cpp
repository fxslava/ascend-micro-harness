// Validation ladder for turboquant_single_token_decode_splitk_aiv: AIV-only,
// M <= 8 query rows, d = 256 channels (directive 3B).
//
// One launch per rung. Every rung checks, through
// tq_splitk_rig::AivRig::Validate (shared with the benchmark, see
// tests/npu/turboquant_splitk_rigs.hpp):
//
//   * the partition protocol -- every worker published, each wrote the
//     partition its own task index names, every live lane of every record it
//     wrote came back finite, and the exported m/l blocks are replicated;
//   * exactly `requests * heads` workers emitted a context, and they are the
//     first `requests * heads` blocks of the grid -- the reducers;
//   * the per-worker census is not torn across its eight replicated lanes,
//     and each worker's page count equals Partition's for its split;
//   * the final context: normalized, butterfly-rotated and gated, in BF16,
//     against splitk_decode_test::Fixture::expected -- the independent FP64
//     full-context oracle, which applies DenseRotate rather than the kernel's
//     factorization, so the egress ordering is checked rather than assumed.
//
// This test's own part is the matrix. M runs over 1, 2, 4, 6 and 8: 1 is the
// single-token boundary, 8 is the kernel's maximum, and 6 is the default the
// rest of the harness uses. K runs over 2, 4, 8 and 16. K = 1 is deliberately
// absent: with one split the kernel takes the fused path, writes no partition
// and never reaches the barrier, so it exercises none of what this test is
// for. The benchmark covers K = 1.
//
// Every rung runs TWICE, once per egress basis:
//
//   EGRESS_UNROTATE = true   the context in the model's channel basis, graded
//                            against `Fixture::expected`, which applies
//                            DenseRotate rather than the kernel's
//                            factorization;
//   EGRESS_UNROTATE = false  the context in the cache's rotated basis, graded
//                            against `Fixture::expected_rotated`, which gates
//                            the normalized accumulator where it already is.
//
// The second line is not the first one rotated. The sigmoid gate is
// elementwise and does not commute with Pi, so the fixture computes both in
// FP64 from the same softmax, and a disagreement between the two legs is a
// statement about the egress and nothing else.
//
// WHY THE REQUEST COUNT VARIES WITH K
//
// The reduction runs inside the launch, behind `AscendC::SyncAll`, and that
// barrier completes only if every launched block is resident at once. So a
// split launch may not span hardware waves, and `requests * heads * K` is
// capped at the number of blocks one wave holds, which for this AIV-only
// operator is the 64 vector subcores of dav-3510 rather than its 32 Cube
// cores (turboquant_splitk_contract.hpp). The ladder therefore takes the
// LARGEST request count each K admits rather than a fixed one: two requests at
// K = 2, 4 and 8, one at K = 16, so K = 8 lands exactly ON the bound at
// 2 x 4 x 8 = 64 and K = 16 lands on it at 1 x 4 x 16 = 64. Both of those are
// full-wave grids, and both were unreachable while the bound was 32.
//
// Concurrency is still real rather than nominal: every rung runs all four
// heads and every split of every head at once, so the whole grid is in flight
// together, which is the condition the barrier and the disjointness of the
// partitions both have to hold under. At the two 64-block rungs the whole
// machine is in that condition at once, which is the configuration the raised
// bound exists to make legal and the one a barrier that did not span both
// subcores of a cluster would hang on.

#include "../turboquant_splitk_rigs.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <exception>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using tq_npu::check;

// Narrowing the matrix. With no arguments the full ladder runs, which is what
// the acceptance command does; the flags exist because on CAModel every rung
// is a real launch and a 40-launch sweep is not an iteration loop. A narrowed
// run is a narrowed claim, and the report says which rungs produced it.
//
//   --rows 1,8      --splitk 8,16      --basis model|rotated|both
struct Matrix {
    std::vector<int> rows{1, 2, 4, 6, 8};
    std::vector<int> splits{2, 4, 8, 16};
    bool model = true, rotated = true;
};

std::vector<int> int_list(std::string_view text)
{
    std::vector<int> out;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto item = text.substr(0, comma);
        int value = 0;
        const auto parsed =
            std::from_chars(item.data(), item.data() + item.size(), value);
        tq_npu::require(parsed.ec == std::errc{} &&
                            parsed.ptr == item.data() + item.size() && value > 0,
                        "expected a comma-separated list of positive integers");
        out.push_back(value);
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
        tq_npu::require(!text.empty(), "trailing comma");
    }
    tq_npu::require(!out.empty(), "empty list");
    return out;
}

Matrix parse(int argc, char** argv)
{
    Matrix m;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag{argv[i]};
        tq_npu::require(i + 1 < argc, "option requires a value");
        const std::string_view value{argv[++i]};
        if (flag == "--rows") m.rows = int_list(value);
        else if (flag == "--splitk") m.splits = int_list(value);
        else if (flag == "--basis") {
            m.model = value == "model" || value == "both";
            m.rotated = value == "rotated" || value == "both";
            tq_npu::require(m.model || m.rotated,
                            "--basis must be model, rotated or both");
        } else throw std::runtime_error("unknown option");
    }
    return m;
}

// Real operator geometry: 128-token pages, four KV heads, 256 channels. Four
// pages so that every split of K = 8 still owns a page at M = 8, and so the
// uneven partition at K = 4 and 8 is exercised too. At K = 16 the four pages
// leave twelve splits with an EMPTY page range, which is the neutral-element
// path reached by geometry rather than by a special case -- and it is reached
// there by 48 of the 64 blocks at once.
constexpr int kMaxRequests = 2;
constexpr int kHeads = splitk_decode_test::kHeads;
constexpr int kPageTokens = splitk_decode_test::kPageTokens;
constexpr int kPages = 4;

// The largest request count this K admits inside one hardware wave.
int RequestsFor(int k)
{
    for (int requests = kMaxRequests; requests >= 1; --requests)
        if (turboquant_splitk::GridFitsOneWave(requests, kHeads, k)) return requests;
    return 0;
}

int run(int argc, char** argv)
{
    const Matrix matrix = parse(argc, argv);
    tq_npu::Session session;

    const std::vector<int>& rows = matrix.rows;
    const std::vector<int>& splits = matrix.splits;
    std::vector<std::pair<tq_splitk_rig::Basis, const char*>> bases;
    if (matrix.model) bases.push_back({tq_splitk_rig::Basis::kModel, "model"});
    if (matrix.rotated) bases.push_back({tq_splitk_rig::Basis::kRotated, "rotated"});

    int launches = 0, reducers = 0, workers = 0, full_wave = 0;
    for (const int m : rows) {
        for (const int k : splits) {
            const int requests = RequestsFor(k);
            tq_npu::require(requests >= 1,
                            "no request count fits one wave at this K and head count");
            // One fixture per (M, K): the request count varies with K, so the
            // fixture's own extent does too. Both egress legs share it, which
            // is what makes them comparable -- same cache bytes, same query,
            // same gate, same softmax.
            splitk_decode_test::Fixture f(kPages * kPageTokens, requests, kHeads, m,
                                          kPageTokens);
            const int grid = requests * kHeads * k;
            for (const auto& [basis, basis_name] : bases) {
                tq_splitk_rig::AivRig rig(f, k, basis);
                rig.Reset();
                check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
                rig.Launch(session.stream, k);
                check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");
                const auto census = rig.Validate(k);

                // Validate() has already required that every worker published
                // its own partition and that the reducers are the blocks that
                // emitted the contexts, so these are the aggregate shapes,
                // reported rather than re-asserted. `export_ticks` is the
                // figure that replaced the lock region: the worst partial
                // write of the rung.
                const double slowest =
                    census.export_ticks.empty()
                        ? 0.0
                        : *std::max_element(census.export_ticks.begin(),
                                            census.export_ticks.end());
                reducers += requests * kHeads;
                workers += grid;
                ++launches;
                full_wave += static_cast<int>(
                    grid == static_cast<int>(turboquant_splitk::kMaxAivBlocks));
                std::printf("  ok  M=%-2d K=%-2d %-7s  %d requests, grid %3d%s, "
                            "%2d reducers, slowest export %.0f ticks\n",
                            m, k, basis_name, requests, grid,
                            grid == static_cast<int>(turboquant_splitk::kMaxAivBlocks)
                                ? " (full wave)" : "",
                            requests * kHeads, slowest);
            }
        }
    }

    std::printf("PASS AIV split-K decode validation: %d launches over %zu row "
                "counts x %zu split counts x %zu egress bases, %d of them full "
                "%u-block waves; %d reducers and %d partition writers in total; "
                "disjoint partitions, replicated exported state, one emitter per "
                "head and a drained census all checked, and each leg's gated BF16 "
                "context matches its own FP64 full-context oracle (d=%d, %d pages "
                "of %d tokens, every grid within the one-wave bound of %u)\n",
                launches, rows.size(), splits.size(), bases.size(), full_wave,
                turboquant_splitk::kMaxAivBlocks, reducers, workers,
                splitk_decode_test::kChannels, kPages, kPageTokens,
                turboquant_splitk::kMaxAivBlocks);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL AIV split-K decode validation: %s\n", error.what());
        return 1;
    }
}
