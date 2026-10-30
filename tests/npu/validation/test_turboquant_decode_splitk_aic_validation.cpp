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

#include "../turboquant_splitk_rigs.hpp"
#include "../../../kernels/ops/turboquant/turboquant_decode_tail.hpp"

#include <cstdio>
#include <exception>
#include <vector>

namespace {

using splitk_decode_test::kAicRows;
using tq_npu::check;

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

int run()
{
    tq_npu::Session session;

    const std::vector<Rung> ladder = {
        {1, "boundary canary: one page"},
        {4, "four pages"},
        {8, "eight pages, multi-temporal split"},
        {3, "non-power-of-2"},
        {5, "non-power-of-2"},
        {7, "non-power-of-2"},
    };
    const std::vector<int> splits = {1, 2, 4, 8};

    int rungs = 0, empty_splits = 0, windowed = 0;
    for (const Rung& rung : ladder) {
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
    for (const TailRung& rung : tails) {
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

    std::printf("PASS AIC split-K decode validation: %d launches over page counts "
                "1/3/4/5/7/8 x K=1/2/4/8, %d of them windowed, %d empty temporal "
                "splits reaching the neutral element; M=%d rows, d=%d, %d requests "
                "x %d heads, all against the FP64 partial oracle\n",
                rungs, windowed, empty_splits, kAicRows,
                splitk_decode_test::kChannels, kRequests, kHeads);
    std::printf("PASS AIC split-K decode at unaligned context lengths: %d launches "
                "over %zu rungs, %d of which end on a partial strip, and %d split "
                "ranges with pages to walk but no live token in any of them -- "
                "every strip skipped, every cross-core flag still exchanged, and "
                "the exact neutral element published\n",
                tail_rungs, tails.size(), partial_strips, dead_splits);
    return 0;
}

}  // namespace

int main()
{
    try {
        return run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL AIC split-K decode validation: %s\n", error.what());
        return 1;
    }
}
