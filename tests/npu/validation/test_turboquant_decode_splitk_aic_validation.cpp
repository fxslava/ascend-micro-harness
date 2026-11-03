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
// The third section is the multi-page L1 reuse regression, and it is named
// rather than left inside the cross product above. See the comment at it.

#include "../turboquant_splitk_rigs.hpp"
#include "../../../kernels/ops/turboquant/turboquant_decode_tail.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <string>
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

// ULP budgets for the multi-page L1 reuse rungs below. Chosen from measured
// data, not picked for comfort.
//
// `reports/decode_splitk_aic_strip_pipeline.md` section 5 is the CAModel
// evidence for the revision these rungs guard. On values of order 1, where one
// FP32 ULP is 1.1920929e-07:
//
//   * the accumulator's mean |deviation| across the 2-, 4- and 8-page rungs is
//     4.35e-08 to 6.21e-08, so about 0.4 to 0.5 ULP;
//   * `m` is bit-exact on every rung, so 0 ULP;
//   * `l`'s worst rung reaches 0.00135x of its 1e-4 * max(1, l) bound at
//     l around 90, which is about 1.2e-05 absolute against a 7.63e-06 ULP at
//     that magnitude -- roughly 1.6 ULP.
//
// And the defect these rungs exist to catch produced a mean |deviation| of
// 1.022830e-03 on the same geometry, which is about 8600 ULP. So 64 and 256
// sit two to three orders of magnitude ABOVE the measured noise -- enough that
// a different device's rounding cannot false-alarm -- and one to two orders
// BELOW the defect's own signature. A budget that did not separate those two
// numbers would not be worth asserting.
//
// WHICH accumulator ULP the 256 holds, and why it is not the obvious one.
// Running these rungs on CAModel (reports/decode_splitk_aic_npu_multipage.md
// section 5) showed the per-channel ULP distance -- deviation over the spacing
// at THAT CHANNEL's value -- is not an error measure for an accumulator:
//
//   strips    worst per-channel ULP    |got - ref| there    |ref| there
//        4                      48            1.78e-07         0.0577
//        8                      98            4.54e-08         0.00624
//       32                     247            1.15e-07         0.00397
//
// The deviations do not grow. What grows is the ULP number, because the
// unluckiest of 4096 channels keeps landing closer to zero -- a row is a sum
// with cancellation, so some channel always nearly cancels, and there is no
// lower bound on how nearly. That number is unbounded for a bit-correct
// kernel, and at 32 strips it had already reached 247 of 256. So it is
// reported (`AicRig::margins.acc_ulp`) and NOT capped.
//
// The 256 is held against `acc_scaled_ulp`, the same deviation over the
// spacing at the ROW's accumulation scale -- the magnitude the arithmetic
// actually worked at, which is what the summation error is bounded by.
//
// NOTE these are budgets for PHYSICAL HARDWARE, and no hardware run has
// exercised them: there is no NPU on the host this was written on. If a device
// run trips one, the failure message names the row, channel, value and
// reference, which is what a budget set from simulation owes its first real
// run. See reports/decode_splitk_aic_npu_multipage.md.
constexpr double kStateUlpBudget = 64.0;
constexpr double kAccScaledUlpBudget = 256.0;

// The block table as the launch sees it, for the log: a rung whose point is a
// scattered mapping should show the mapping it got.
std::string TableImage(const splitk_decode_test::Fixture& f)
{
    std::string out = "[";
    for (int page = 0; page < f.pages; ++page) {
        if (page) out += ",";
        out += std::to_string(f.table[static_cast<std::size_t>(page)]);
    }
    return out + "]";
}

// Which sections to run. Every section by default: this is a gate, and a
// default that silently ran less than everything would be worse than no
// selector at all.
struct Sections {
    bool ladder = true, tail = true, l1_reuse = true;
};

int run(const Sections& sections)
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
    for (const Rung& rung : sections.ladder ? ladder : std::vector<Rung>{}) {
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
    for (const TailRung& rung : sections.tail ? tails : std::vector<TailRung>{}) {
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

    // ------------------------ the multi-page L1 reuse regression
    // A NAMED rung for the hazard `reports/decode_splitk_aic_page_race.md`
    // diagnosed and `reports/decode_splitk_aic_strip_pipeline.md` closed: at
    // four or more carried pages AIV1 restaged the L1 key page for sequence
    // page p + 1 over page p's before the Cube's MTE1 had read it, and the
    // mode-4 edge meant to prevent it blocked only on its first use of an id.
    //
    // The first ladder above already reaches pages >= 4 at K = 1, so why this
    // exists: there, those values appear only as two coordinates of a cross
    // product, and the geometry that actually reproduced the defect is named
    // nowhere. A future narrowing of that ladder -- dropping K = 1 as
    // redundant, say, or trimming the page counts -- would take this coverage
    // with it and nothing would say so. Four properties here are NOT reachable
    // from the ladder above at all:
    //
    //   * requests = 1, heads = 1, splits = 1, so the ENTIRE page window is
    //     carried by one physical core and every page transition contends the
    //     same L1 key ring. At two requests and two heads the work spreads
    //     over four blocks and any one core carries fewer consecutive pages,
    //     which is precisely the pressure the defect needed;
    //   * block_size = 16, one strip per page. The ladder hardcodes
    //     kPageTokens and never reaches this, yet it is the geometry the
    //     CAModel sweep used to expose the defect. At one strip per page every
    //     strip lands in ring slot 0, so a page boundary and a slot wrap
    //     coincide on every page rather than every eighth one;
    //   * a SCATTERED block table, monotone in neither direction. The
    //     fixture's default reverse permutation already catches a kernel that
    //     ignores the table, but a kernel that walks its pages backwards reads
    //     a correct-looking sequence out of it. A scatter makes a page mix-up
    //     land on unrelated data;
    //   * ULP budgets. The relative bounds the ladder uses have an additive
    //     floor of 1e-6 on the accumulator, and the 0xAA canary is
    //     -3.03e-13 -- so a channel whose reference is near zero passes the
    //     relative bound while holding a word no worker ever wrote. The ULP
    //     reading and the explicit canary scan are what close that.
    struct RaceRung {
        int page_tokens, pages;
        const char* note;
    };
    // Smallest first. Each rung is a real launch, and on CAModel a 128-token
    // page is eight strips, so the 8-page full-page rung is 64 strips and tens
    // of minutes. Ordering them this way means a run that is cut short still
    // leaves evidence for the rungs that completed.
    const std::vector<RaceRung> race = {
        {16, 4, "one strip per page, four carried"},
        {16, 8, "one strip per page, eight carried"},
        {kPageTokens, 4, "full pages, four carried"},
        {kPageTokens, 8, "full pages, eight carried"},
    };

    int race_rungs = 0;
    double worst_state_ulp = 0, worst_acc_ulp = 0, worst_acc_scaled_ulp = 0;
    for (const RaceRung& rung : sections.l1_reuse ? race : std::vector<RaceRung>{}) {
        splitk_decode_test::Fixture f(rung.pages * rung.page_tokens, 1, 1, kAicRows,
                                      rung.page_tokens, -1,
                                      splitk_decode_test::TailFill::kDirty,
                                      splitk_decode_test::PageMap::kScattered);
        tq_npu::require(f.pages == rung.pages && f.page_tokens == rung.page_tokens,
                        "the L1 reuse fixture did not get the geometry it asked for");
        // The scatter has to BE scattered, or this is the ladder again under a
        // different name. Checked rather than trusted, because the permutation
        // comes from a seeded shuffle and a bad seed could hand back either
        // degenerate case.
        bool identity = true, reverse = true;
        for (int page = 0; page < f.pages; ++page) {
            identity &= f.table[static_cast<std::size_t>(page)] ==
                        static_cast<std::uint32_t>(page);
            reverse &= f.table[static_cast<std::size_t>(page)] ==
                       static_cast<std::uint32_t>(f.pages - 1 - page);
        }
        tq_npu::require(!identity && !reverse,
                        "an L1 reuse rung needs a block table that is neither the "
                        "identity nor the reverse permutation");

        tq_splitk_rig::AicRig rig(f, 1, 0, rung.pages);
        rig.state_ulp_cap = kStateUlpBudget;
        // Deliberately NOT rig.acc_ulp_cap -- see kAccScaledUlpBudget above.
        rig.acc_scaled_ulp_cap = kAccScaledUlpBudget;
        rig.Reset();
        check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
        rig.Launch(session.stream, 0);
        check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");
        rig.Validate(0);
        worst_state_ulp = std::max(worst_state_ulp, rig.margins.state_ulp);
        worst_acc_ulp = std::max(worst_acc_ulp, rig.margins.acc_ulp);
        worst_acc_scaled_ulp =
            std::max(worst_acc_scaled_ulp, rig.margins.acc_scaled_ulp);
        ++race_rungs;
        std::printf("  ok  L1 reuse: block_size=%-3d strips=%d pages=%d K=1 "
                    "table=%-18s  %s\n",
                    rung.page_tokens, rung.page_tokens / 16, rung.pages,
                    TableImage(f).c_str(), rung.note);
    }

    std::printf("SECTIONS RUN: ladder=%s tail=%s l1-reuse=%s%s\n",
                sections.ladder ? "yes" : "NO", sections.tail ? "yes" : "NO",
                sections.l1_reuse ? "yes" : "NO",
                (sections.ladder && sections.tail && sections.l1_reuse)
                    ? "" : "  -- NARROWED, this is not the full gate");
    std::printf("PASS AIC split-K decode validation: %d launches over page counts "
                "1/3/4/5/7/8 x K=1/2/4/8, %d of them windowed, %d empty temporal "
                "splits reaching the neutral element; M=%d rows, d=%d, %d requests "
                "x %d heads, all against the FP64 partial oracle\n",
                rungs, windowed, empty_splits, kAicRows,
                splitk_decode_test::kChannels, kRequests, kHeads);
    std::printf("PASS AIC multi-page L1 reuse regression: %d launches at "
                "requests=1 heads=1 K=1 -- one core carrying every page -- over "
                "block_size 128 and 16 at 4 and 8 carried pages, each through a "
                "scattered block table; worst_state_ulp %.2f of a %.0f budget, "
                "worst_acc_scaled_ulp %.2f of a %.0f budget, zero canary words "
                "and zero nonfinite words in any published partition "
                "(worst per-channel acc ULP %.2f, reported not capped)\n",
                race_rungs, worst_state_ulp, kStateUlpBudget,
                worst_acc_scaled_ulp, kAccScaledUlpBudget, worst_acc_ulp);
    std::printf("PASS AIC split-K decode at unaligned context lengths: %d launches "
                "over %zu rungs, %d of which end on a partial strip, and %d split "
                "ranges with pages to walk but no live token in any of them -- "
                "every strip skipped, every cross-core flag still exchanged, and "
                "the exact neutral element published\n",
                tail_rungs, tails.size(), partial_strips, dead_splits);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    Sections sections;
    for (int arg = 1; arg < argc; ++arg) {
        const std::string option = argv[arg];
        if (option == "--only" && arg + 1 < argc) {
            const std::string which = argv[++arg];
            sections = Sections{false, false, false};
            if (which == "ladder") sections.ladder = true;
            else if (which == "tail") sections.tail = true;
            else if (which == "l1-reuse") sections.l1_reuse = true;
            else {
                std::fprintf(stderr, "unknown section %s; expected ladder, tail "
                             "or l1-reuse\n", which.c_str());
                return 2;
            }
        } else {
            std::fprintf(stderr,
                         "usage: %s [--only ladder|tail|l1-reuse]\n"
                         "  Default: every section. --only narrows, which is for "
                         "iterating on the\n  CAModel twin where each rung is a "
                         "real launch; a narrowed run is NOT\n  the gate.\n",
                         argv[0]);
            return 2;
        }
    }
    try {
        return run(sections);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL AIC split-K decode validation: %s\n", error.what());
        return 1;
    }
}
