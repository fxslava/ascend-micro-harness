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
// rest of the harness uses. K runs over 2, 4 and 8. K = 1 is deliberately
// absent: with one split the kernel takes the fused path, writes no partition
// and never reaches the barrier, so it exercises none of what this test is
// for. The benchmark covers K = 1.
//
// WHY THE REQUEST COUNT VARIES WITH K
//
// The reduction runs inside the launch, behind `AscendC::SyncAll`, and that
// barrier completes only if every launched block is resident at once. So a
// split launch may not span hardware waves, and `requests * heads * K` is
// capped at the physical block count (turboquant_splitk_contract.hpp). The
// ladder therefore takes the LARGEST request count each K admits rather than
// a fixed one: two requests at K = 2 and K = 4, one at K = 8, each landing on
// or under the bound. Pinning requests at two and letting K = 8 overflow it
// would not test a wider shape -- the host refuses that launch.
//
// Concurrency is still real rather than nominal: every rung runs all four
// heads and every split of every head at once, so the whole grid is in flight
// together, which is the condition the barrier and the disjointness of the
// partitions both have to hold under.

#include "../turboquant_splitk_rigs.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

using tq_npu::check;

// Real operator geometry: 128-token pages, four KV heads, 256 channels. Four
// pages so that every split of K = 8 still owns a page at M = 8, and so the
// uneven partition at K = 4 and 8 is exercised too.
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

int run()
{
    tq_npu::Session session;

    const std::vector<int> rows = {1, 2, 4, 6, 8};
    const std::vector<int> splits = {2, 4, 8};

    int launches = 0, reducers = 0, workers = 0;
    for (const int m : rows) {
        for (const int k : splits) {
            const int requests = RequestsFor(k);
            tq_npu::require(requests >= 1,
                            "no request count fits one wave at this K and head count");
            // One fixture per (M, K): the request count varies with K, so the
            // fixture's own extent does too.
            splitk_decode_test::Fixture f(kPages * kPageTokens, requests, kHeads, m,
                                          kPageTokens);
            tq_splitk_rig::AivRig rig(f, k);
            rig.Reset();
            check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
            rig.Launch(session.stream, k);
            check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");
            const auto census = rig.Validate(k);

            // Validate() has already required that every worker published its
            // own partition and that the reducers are the blocks that emitted
            // the contexts, so these are the aggregate shapes, reported rather
            // than re-asserted. `export_ticks` is the figure that replaced the
            // lock region: the worst partial write of the rung.
            const double slowest =
                census.export_ticks.empty()
                    ? 0.0
                    : *std::max_element(census.export_ticks.begin(),
                                        census.export_ticks.end());
            reducers += requests * kHeads;
            workers += requests * kHeads * k;
            ++launches;
            std::printf("  ok  M=%-2d K=%-2d  %d requests, grid %3d, %2d reducers, "
                        "slowest export %.0f ticks\n",
                        m, k, requests, requests * kHeads * k, requests * kHeads,
                        slowest);
        }
    }

    std::printf("PASS AIV split-K decode validation: %d launches over M=1/2/4/6/8 "
                "x K=2/4/8, %d reducers and %d partition writers in total; "
                "disjoint partitions, replicated exported state, one emitter per "
                "head and a drained census all checked, and the gated BF16 "
                "context matches the FP64 full-context oracle (d=%d, %d pages of "
                "%d tokens, grids within the one-wave bound of %u)\n",
                launches, reducers, workers, splitk_decode_test::kChannels,
                kPages, kPageTokens, turboquant_splitk::kMaxCoResidentBlocks);
    return 0;
}

}  // namespace

int main()
{
    try {
        return run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL AIV split-K decode validation: %s\n", error.what());
        return 1;
    }
}
