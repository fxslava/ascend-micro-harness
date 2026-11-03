// Validation ladder for turboquant_paged_decode_splitk_aic: MIX_AIC_1_2,
// M = 16 packed query rows, d = 256 channels (directive 3A).
//
// One launch per rung. Every rung checks, for all 16 query rows of every
// (request, head, split) item, against splitk_decode_test::ReferenceAicPartial
// -- the independent FP64 oracle the CPU probe test_aic_splitk_decode_probe
// also uses:
//
//   * the emitted partial state (m, l), including that its eight replicated
//     lanes are not torn;
//   * the partial accumulator Acc over all 256 channels;
//   * the rotated query Q_rot, bit-exact against the BF16 reference;
//   * for a split whose page range is empty, the exact neutral element
//     (m == -3e38f, l == 0, Acc == 0) and an untouched Q_rot canary.
//
// Those checks are tq_splitk_rig::AicRig::Validate, shared with the benchmark
// rather than restated here: see tests/npu/turboquant_splitk_rigs.hpp.
//
// The ladder is the part that is this test's own. Page counts 1, 4 and 8 give
// the boundary canary, a plain multi-page range and a multi-temporal-split
// range; 3, 5 and 7 are the non-power-of-2 counts, which are what make
// Partition hand out ranges of unequal length and -- once splits exceed the
// page count -- empty ranges, so the neutral-element path above is reached by
// the geometry rather than by a special case. Every rung runs at K = 1, 2, 4
// and 8.
//
// A nonzero page_begin is included because a window that is ignored would
// otherwise pass every check above: the kernel would simply compute the full
// range, and the oracle it is compared against is handed the same window.
// The CPU probe pins that the window is numerically observable; this pins that
// the kernel honours it.
//
// THE PREROTATED INGRESS LEG
//
// A third section runs `turboquant_paged_decode_splitk_prerotated_aic`, the
// INGRESS_ROTATE = false specialization, over a narrower ladder. It is narrower
// on purpose: the ingress is one prologue per item and the partial-state oracle
// does not change with it, so what the leg has to establish is that the
// prologue is GONE and that nothing downstream noticed. The rig does that with
// three statements per rung --
//
//   * the query operand is the rotated query, which is bit for bit what the
//     rotating leg's own prologue produces, so the FP64 partial oracle is the
//     same oracle and every (m, l, Acc) check is the same check;
//   * `pi_table` is flooded with the canary rather than filled, so a prologue
//     that still ran would build its L0A out of 0x7FC0BEEF and no partial
//     state would come near that oracle;
//   * `qrot` must come back untouched, every word still 0xAAAA, because the
//     prologue is its only writer.
//
// Whole-range and windowed, K = 1 and 4, at one page, four and eight: enough
// to cover a single-split launch, a multi-split one, a nonzero page_begin and
// the empty-range neutral element on a leg whose L1 arena is loaded
// differently from the contract leg's.

#include "../turboquant_splitk_rigs.hpp"
#include "../../../kernels/ops/turboquant/turboquant_decode_tail.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <exception>
#include <string_view>
#include <vector>

namespace {

using splitk_decode_test::kAicRows;
using tq_npu::check;

// Narrowing the ladder. With no arguments everything below runs, which is what
// the acceptance command does; the flags exist because on CAModel every rung is
// a real MIX launch over a 16-row panel. A narrowed run is a narrowed claim,
// and the report says which rungs produced it.
//
//   --section ladder|tail|prerotated|all    --pages 1,4    --splitk 1,4
struct Selection {
    bool ladder = true, tail = true, prerotated = true;
    std::vector<int> pages, splits;  // empty: each section's own default
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

Selection parse(int argc, char** argv)
{
    Selection selection;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag{argv[i]};
        tq_npu::require(i + 1 < argc, "option requires a value");
        const std::string_view value{argv[++i]};
        if (flag == "--pages") selection.pages = int_list(value);
        else if (flag == "--splitk") selection.splits = int_list(value);
        else if (flag == "--section") {
            selection.ladder = value == "ladder" || value == "all";
            selection.tail = value == "tail" || value == "all";
            selection.prerotated = value == "prerotated" || value == "all";
            tq_npu::require(selection.ladder || selection.tail ||
                                selection.prerotated,
                            "--section must be ladder, tail, prerotated or all");
        } else throw std::runtime_error("unknown option");
    }
    return selection;
}

struct Rung {
    int pages;
    const char* note;
};

// Real operator geometry: 128-token physical pages, 16 packed query rows,
// 256 channels. Two requests and two heads so that the (request, head, split)
// item index the kernel derives from its block id is exercised in all three
// dimensions rather than only in the split.
constexpr int kRequests = 2;
constexpr int kHeads = 2;
constexpr int kPageTokens = splitk_decode_test::kPageTokens;

int run(int argc, char** argv)
{
    const Selection selection = parse(argc, argv);
    tq_npu::Session session;

    const std::vector<Rung> full_ladder = {
        {1, "boundary canary: one page"},
        {4, "four pages"},
        {8, "eight pages, multi-temporal split"},
        {3, "non-power-of-2"},
        {5, "non-power-of-2"},
        {7, "non-power-of-2"},
    };
    std::vector<Rung> ladder;
    for (const Rung& rung : full_ladder)
        if (selection.pages.empty() ||
            std::find(selection.pages.begin(), selection.pages.end(), rung.pages) !=
                selection.pages.end())
            ladder.push_back(rung);
    const std::vector<int> splits =
        selection.splits.empty() ? std::vector<int>{1, 2, 4, 8} : selection.splits;

    int rungs = 0, empty_splits = 0, windowed = 0;
    for (const Rung& rung : (selection.ladder ? ladder : std::vector<Rung>{})) {
        // The fixture is built once per page count and reused across K: it is
        // the expensive part, and K changes only the partition.
        splitk_decode_test::Fixture f(rung.pages * kPageTokens, kRequests, kHeads,
                                      kAicRows, kPageTokens);
        for (const int k : splits) {
            // begin == 0 covers the whole range; a second pass drops the first
            // page, which is the window the kernel has to honour. One page
            // cannot be windowed and still leave work, so it is whole-range only.
            const int windows = rung.pages > 1 ? 2 : 1;
            for (int window = 0; window < windows; ++window) {
                const int begin = window == 0 ? 0 : 1;
                const int count = rung.pages - begin;

                tq_splitk_rig::AicRig rig(f, k, begin, count);
                rig.Reset();
                check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
                rig.Launch(session.stream, 0);
                check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");
                rig.Validate(0);

                for (int split = 0; split < k; ++split)
                    if (turboquant_splitk::Partition(count, k, split).count == 0)
                        ++empty_splits;
                windowed += begin != 0;
                ++rungs;
                std::printf("  ok  pages=%-2d begin=%d count=%-2d K=%-2d  %s\n",
                            rung.pages, begin, count, k, rung.note);
            }
        }
    }

    // ---------------------------------------------------------- the tail
    // The same ladder at context lengths that are NOT a multiple of the
    // 16-token strip. The Cube's per-page strip count stays at capacity here,
    // so what is under test is the Vector core's three-way choice per strip:
    // accumulate it whole, accumulate it masked, or skip it while still
    // exchanging its cross-core flags. The last of those is why the dead-page
    // rungs below exist -- a skipped strip that failed to acknowledge the
    // Cube's announcement would hang rather than return a wrong number.
    struct TailRung {
        int pages, dead;
        const char* note;
    };
    const std::vector<TailRung> tails = {
        {1, 1, "one page, 15 live lanes in the last strip"},
        {1, 15, "one page, a single live lane in the last strip"},
        {1, 127, "one page, a single live token in the whole range"},
        {4, 1, "four pages, 15 live lanes in the last strip"},
        {4, 15, "four pages, a single live lane in the last strip"},
        {4, 129, "four pages, a whole dead page and a partial strip"},
        {5, 273, "non-power-of-2, two dead pages and a partial strip"},
    };

    int tail_rungs = 0, dead_splits = 0, partial_strips = 0;
    for (const TailRung& rung :
         (selection.tail ? tails : std::vector<TailRung>{})) {
        const int capacity = rung.pages * kPageTokens;
        const int live = capacity - rung.dead;
        tq_npu::require(live > 0,
                        "an unaligned rung must keep at least one live token");
        splitk_decode_test::Fixture f(capacity, kRequests, kHeads, kAicRows,
                                      kPageTokens, live);
        tq_npu::require(f.seq_len == live && f.tokens == capacity,
                        "fixture did not separate live tokens from capacity");

        const std::uint32_t last_strip =
            turboquant_decode_tail::StripsForTokens(
                static_cast<std::uint32_t>(live)) - 1u;
        const std::uint32_t active = turboquant_decode_tail::ActiveInStrip(
            static_cast<std::uint64_t>(last_strip) * 16u,
            static_cast<std::uint32_t>(live));
        if (active != 16u) ++partial_strips;

        for (const int k : splits) {
            const int windows = rung.pages > 1 ? 2 : 1;
            for (int window = 0; window < windows; ++window) {
                const int begin = window == 0 ? 0 : 1;
                const int count = rung.pages - begin;

                tq_splitk_rig::AicRig rig(f, k, begin, count);
                rig.Reset();
                check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
                rig.Launch(session.stream, 0);
                check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");
                rig.Validate(0);

                // A split with pages to walk but no live token in any of them:
                // every strip skipped, every flag still exchanged, and the
                // published partial must be the exact neutral element -- the
                // same one an empty range produces, reached by a different
                // path through the kernel.
                for (int split = 0; split < k; ++split) {
                    const auto range = turboquant_splitk::Partition(count, k, split);
                    if (range.count == 0) continue;
                    std::uint32_t live_here = 0;
                    for (std::uint32_t p = 0; p < range.count; ++p)
                        live_here += turboquant_decode_tail::ActiveInPage(
                            static_cast<std::uint32_t>(begin) + range.begin + p,
                            kPageTokens, static_cast<std::uint32_t>(live));
                    if (live_here == 0u) ++dead_splits;
                }
                ++tail_rungs;
            }
        }
        std::printf("  ok  S=%-4d of %-4d capacity, %2u strips, %2u live lanes in "
                    "the last  %s\n",
                    live, capacity, last_strip + 1u, active, rung.note);
    }

    if (selection.ladder)
    std::printf("PASS AIC split-K decode validation: %d launches over page counts "
                "1/3/4/5/7/8 x K=1/2/4/8, %d of them windowed, %d empty temporal "
                "splits reaching the neutral element; M=%d rows, d=%d, %d requests "
                "x %d heads, all against the FP64 partial oracle\n",
                rungs, windowed, empty_splits, kAicRows,
                splitk_decode_test::kChannels, kRequests, kHeads);
    if (selection.tail)
    std::printf("PASS AIC split-K decode at unaligned context lengths: %d launches "
                "over %zu rungs, %d of which end on a partial strip, and %d split "
                "ranges with pages to walk but no live token in any of them -- "
                "every strip skipped, every cross-core flag still exchanged, and "
                "the exact neutral element published\n",
                tail_rungs, tails.size(), partial_strips, dead_splits);

    // ------------------------------------------- the prerotated ingress leg
    const std::vector<int> prerotated_pages =
        selection.pages.empty() ? std::vector<int>{1, 4, 8} : selection.pages;
    const std::vector<int> prerotated_splits =
        selection.splits.empty() ? std::vector<int>{1, 4} : selection.splits;
    int prerotated_rungs = 0, prerotated_windowed = 0;
    for (const int pages : (selection.prerotated ? prerotated_pages
                                                 : std::vector<int>{})) {
        splitk_decode_test::Fixture f(pages * kPageTokens, kRequests, kHeads,
                                      kAicRows, kPageTokens);
        for (const int k : prerotated_splits) {
            const int windows = pages > 1 ? 2 : 1;
            for (int window = 0; window < windows; ++window) {
                const int begin = window == 0 ? 0 : 1;
                const int count = pages - begin;

                tq_splitk_rig::AicRig rig(f, k, begin, count,
                                          tq_splitk_rig::Ingress::kPrerotated);
                rig.Reset();
                check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
                rig.Launch(session.stream, 0);
                check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");
                rig.Validate(0);

                prerotated_windowed += begin != 0;
                ++prerotated_rungs;
                std::printf("  ok  prerotated pages=%-2d begin=%d count=%-2d K=%-2d  "
                            "pi_table flooded, qrot untouched\n",
                            pages, begin, count, k);
            }
        }
    }
    if (selection.prerotated)
    std::printf("PASS AIC split-K prerotated ingress: %d launches over page counts "
                "1/4/8 x K=1/4, %d of them windowed; the 64-Mmad prologue, the "
                "pi_table read and the qrot write are all absent, and every "
                "partial (m, l, Acc) still matches the SAME FP64 oracle the "
                "rotating leg is graded against\n",
                prerotated_rungs, prerotated_windowed);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL AIC split-K decode validation: %s\n", error.what());
        return 1;
    }
}
