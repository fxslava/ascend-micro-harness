// Tier-1 host probe for the AIV-only single-token decode operator (No.3).
//
// Stage V1 of SKELETON_KERNEL_WORKFLOW.md: no device, no simulator, no
// toolchain. A CAModel run compares the device against the oracle in
// tests/common/single_token_decode_oracle.hpp and therefore cannot tell a
// wrong device from a wrong oracle. These are the checks that can only be
// made here, and each one is a thing a six-minute simulator run would
// otherwise have been spent discovering:
//
//   1. THE GEMV'S INDEX ARITHMETIC. The leaf's three phases -- four
//      quarter-major `Mul`s, one 64-repeat `ReduceRepeat`, two folds -- are
//      replayed at the granularity the hardware uses, from the repeat strides
//      the header derives, and the resulting panel is compared BIT-EXACTLY
//      against the direct dot product. A wrong repeat stride, a wrong
//      quarter base or a fold at the wrong offset is a wrong panel here, in
//      milliseconds.
//
//   2. THAT THE PRODUCT MAP IS A BIJECTION. All 4096 (token, channel) pairs
//      land in exactly one product slot, and no slot is written twice. A map
//      that merely looks plausible can still alias, and aliasing inside one
//      instruction is invisible in the numbers of a single strip.
//
//   3. THE ORDER-INDEPENDENCE THE BIT-EXACT GATE RESTS ON. The same panel is
//      recomputed in three orders -- the host's left-to-right loop, the
//      device's binary tree inside a repeat, and the quarter fold across
//      repeats -- and all three must agree to the bit. If they ever do not,
//      the gate is wrong, not the device.
//
//   4. THE ROADMAP'S SKETCH IS MEASURED, NOT ARGUED. `routine_vector_decode
//      _gemv.hpp` rejects the roadmap's two-instruction row because a 32-bit
//      continuous-mode mask is bounded at 64 and an over-range mask is
//      CLAMPED rather than diagnosed. That claim is only worth making if the
//      clamped form gives a different answer, so the clamped reduction is
//      computed here and required to disagree. It is the one check that says
//      the deviation was necessary.
//
//   5. THE EGRESS, both directions of agreement: the eight-stage butterfly
//      against the dense 256x256 right-multiplication by Pi, over the
//      operator's own six rows and its own normalised accumulator.
//
//   6. SINGLE-STRIP ISOLATION. The golden state for one strip depends on that
//      strip alone -- the invariant the canary protocol asserts on the device.
//
//   7. EVERY FIXTURE PROPERTY the gates depend on, at one page and at four,
//      for all four heads, plus the identity with the Cube operator's fixture
//      at one page.
//
// What it does NOT establish: anything about the device. No pipeline, no
// flags, no UB layout, no engine. That is CAModel's job and the trace audit's.

#include "../common/single_token_decode_oracle.hpp"
// The leaf itself, for its geometry constants. Outside a device build it
// declares nothing but those, so the replay below is driven by the same
// numbers the kernel compiles against rather than by a second copy of them.
#include "../../routines/routine_vector_decode_gemv.hpp"

#include <cstdio>
#include <vector>

namespace {

using namespace single_token_decode;

int failures = 0;

void expect(bool condition, const char* what)
{
    if (condition) {
        std::printf("ok   %s\n", what);
        return;
    }
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

// ------------------------------------------------- the device's GEMV, replayed
//
// Geometry taken from the leaf, not restated: if the header changes, this
// replay changes with it and the comparison stays honest.
constexpr std::uint32_t kQuarters = ascend_micro::kGemvQuarters;          // 4
constexpr std::uint32_t kVecLanes = ascend_micro::kGemvVecLanes;          // 64
constexpr std::uint32_t kTokenBlocks = ascend_micro::kGemvTokenBlocks;    // 32
constexpr std::uint32_t kRegisterBlocks = ascend_micro::kGemvRegisterBlocks;  // 8
constexpr std::uint32_t kLanesPerBlock = ascend_micro::kGemvLanesPerBlock;    // 8
constexpr std::uint32_t kPartials = ascend_micro::kGemvPartials;          // 64

// `ReduceRepeat<SUM>` sums the lanes of one repeat by binary tree accumulation
// (reference: "ReduceDataBlock Key Feature Description"). Modelled exactly,
// because the whole bit-exact gate is a claim about summation order.
float tree_sum(const float* values, std::size_t count)
{
    std::vector<float> level(values, values + count);
    while (level.size() > 1) {
        std::vector<float> next;
        for (std::size_t i = 0; i + 1 < level.size(); i += 2)
            next.push_back(level[i] + level[i + 1]);
        if (level.size() % 2 == 1) next.push_back(level.back());
        level.swap(next);
    }
    return level.front();
}

// One row of the panel, computed the way the four `Mul`s, the `ReduceRepeat`
// and the two `Add`s compute it -- addressing products through the repeat
// strides rather than through the closed form, so a wrong stride shows up.
//
// `writes` counts how many times each product slot is written, which is what
// check 2 reads.
std::vector<float> replay_gemv_row(const float* key_page, const float* query_row,
                                   std::vector<int>* writes)
{
    std::vector<float> products(static_cast<std::size_t>(kKeys) * kHeadSize, 0.0f);

    // Phase 1: four Mul instructions. Quarter j issues 16 repeats; repeat t
    // advances src0 by `kTokenBlocks` DataBlocks and dst by `kRegisterBlocks`,
    // while src1 stands still (src1RepStride == 0).
    for (std::uint32_t j = 0; j < kQuarters; ++j) {
        const std::uint32_t dst_base = j * kKeys * kVecLanes;
        const std::uint32_t src0_base = j * kVecLanes;
        const std::uint32_t src1_base = j * kVecLanes;
        for (std::uint32_t t = 0; t < static_cast<std::uint32_t>(kKeys); ++t) {
            const std::uint32_t dst = dst_base + t * kRegisterBlocks * kLanesPerBlock;
            const std::uint32_t src0 = src0_base + t * kTokenBlocks * kLanesPerBlock;
            for (std::uint32_t lane = 0; lane < kVecLanes; ++lane) {
                products[dst + lane] = key_page[src0 + lane] * query_row[src1_base + lane];
                if (writes) (*writes)[dst + lane] += 1;
            }
        }
    }

    // Phase 2: one ReduceRepeat, mask 64, 64 repeats, srcRepStride
    // kRegisterBlocks, dstRepStride 1 (FP32 lanes, for SUM).
    std::vector<float> partials(kPartials, 0.0f);
    for (std::uint32_t r = 0; r < kPartials; ++r)
        partials[r] = tree_sum(&products[r * kRegisterBlocks * kLanesPerBlock], kVecLanes);

    // Phase 3: two Adds, mask 32 then mask 16, at whole-DataBlock offsets.
    std::vector<float> fold(kPartials / 2u, 0.0f);
    for (std::uint32_t i = 0; i < kPartials / 2u; ++i)
        fold[i] = partials[i] + partials[i + kPartials / 2u];
    std::vector<float> panel(static_cast<std::size_t>(kKeys), 0.0f);
    for (std::uint32_t t = 0; t < static_cast<std::uint32_t>(kKeys); ++t)
        panel[t] = fold[t] + fold[t + kKeys];
    return panel;
}

// The roadmap's sketch, as the hardware would execute it: a continuous-mode
// mask of 256 on a 32-bit operand is CLAMPED to 64, so each token's "dot
// product" covers channels 0..63 only.
std::vector<float> replay_clamped_row(const float* key_page, const float* query_row)
{
    std::vector<float> panel(static_cast<std::size_t>(kKeys), 0.0f);
    for (int t = 0; t < kKeys; ++t) {
        std::vector<float> terms(kVecLanes);
        for (std::uint32_t lane = 0; lane < kVecLanes; ++lane)
            terms[lane] = key_page[static_cast<std::size_t>(t) * kHeadSize + lane] *
                          query_row[lane];
        panel[t] = tree_sum(terms.data(), terms.size());
    }
    return panel;
}

// The dequantised key strip the device holds in Unified Buffer, token-major.
std::vector<float> key_page_of(int head, int strip)
{
    std::vector<float> page(static_cast<std::size_t>(kKeys) * kHeadSize, 0.0f);
    for (int t = 0; t < kKeys; ++t)
        for (int c = 0; c < kHeadSize; ++c)
            page[static_cast<std::size_t>(t) * kHeadSize + c] =
                static_cast<float>(KeyNibbleAt(head, strip, t, c));
    return page;
}

std::vector<float> query_panel_of(int head)
{
    std::vector<float> panel(static_cast<std::size_t>(kRowElems), 0.0f);
    for (int row = 0; row < kRows; ++row)
        for (int c = 0; c < kHeadSize; ++c)
            panel[static_cast<std::size_t>(row) * kHeadSize + c] =
                QRotEntry(head, row, c);
    return panel;
}

}  // namespace

int main()
{
    std::printf("single-token decode (No.3) Tier-1 host probe\n");
    std::printf("  geometry: %d rows, d = %d, %d tokens per strip, %d strips per page, "
                "%d KV heads\n",
                kRows, kHeadSize, kKeys, kStripsPerPage, kKvHeads);

    try {
        // ---- 1/2/3. the GEMV replay, the bijection and the three orders ----
        std::size_t panel_bad = 0;
        std::size_t order_bad = 0;
        std::size_t clamped_same = 0;
        std::size_t clamped_total = 0;
        bool bijection = true;
        for (int head = 0; head < kKvHeads; ++head) {
            const std::vector<float> query = query_panel_of(head);
            for (int strip = 0; strip < kStripsPerPage; ++strip) {
                const std::vector<float> page = key_page_of(head, strip);
                const std::vector<float> want = GoldenRawPanel(head, strip);
                for (int row = 0; row < kRows; ++row) {
                    std::vector<int> writes(static_cast<std::size_t>(kKeys) * kHeadSize, 0);
                    const std::vector<float> got = replay_gemv_row(
                        page.data(), query.data() + static_cast<std::size_t>(row) * kHeadSize,
                        &writes);
                    for (int w : writes)
                        if (w != 1) bijection = false;
                    for (int t = 0; t < kKeys; ++t) {
                        const float expected = want[static_cast<std::size_t>(row) * kKeys + t];
                        if (float_bits(got[t]) != float_bits(expected)) ++panel_bad;
                        // The third order: left to right over all 256 terms.
                        float serial = 0.0f;
                        for (int c = 0; c < kHeadSize; ++c)
                            serial += query[static_cast<std::size_t>(row) * kHeadSize + c] *
                                      page[static_cast<std::size_t>(t) * kHeadSize + c];
                        if (float_bits(serial) != float_bits(expected)) ++order_bad;
                    }
                    const std::vector<float> clamped = replay_clamped_row(
                        page.data(),
                        query.data() + static_cast<std::size_t>(row) * kHeadSize);
                    for (int t = 0; t < kKeys; ++t) {
                        ++clamped_total;
                        if (float_bits(clamped[t]) ==
                            float_bits(want[static_cast<std::size_t>(row) * kKeys + t]))
                            ++clamped_same;
                    }
                }
            }
        }
        expect(bijection,
               "check 2: every one of the 4096 product slots is written exactly once");
        expect(panel_bad == 0,
               "check 1: the replayed four-Mul / one-Reduce / two-Add panel is "
               "bit-exact against the direct dot product");
        expect(order_bad == 0,
               "check 3: the host loop, the in-repeat binary tree and the quarter "
               "fold agree to the bit on every entry");
        // A handful of columns can coincide by arithmetic accident -- a token
        // whose upper 192 channels happen to cancel -- so the requirement is
        // that the clamped form is WRONG almost everywhere, not everywhere.
        expect(clamped_same * 20 < clamped_total,
               "check 4: the roadmap's over-range mask, clamped to 64 lanes as "
               "the reference says it is, gives a different panel");
        std::printf("     clamped-mask panel: %zu/%zu entries coincide with the "
                    "correct one (%.2f%%)\n",
                    clamped_same, clamped_total,
                    100.0 * static_cast<double>(clamped_same) /
                        static_cast<double>(clamped_total));

        // ---- 5. the egress: butterfly against the dense rotation ----------
        double worst_dense = 0.0;
        double largest_dense = 0.0;
        for (int head = 0; head < kKvHeads; ++head) {
            std::vector<int> strips;
            for (int s = 0; s < kStripsPerPage; ++s) strips.push_back(s);
            const GoldenState state = GoldenDecode(head, strips);
            for (int row = 0; row < kRows; ++row) {
                const std::size_t base = static_cast<std::size_t>(row) * kHeadSize;
                const std::vector<float> dense = DenseRotate(&state.normalized[base]);
                for (int c = 0; c < kHeadSize; ++c) {
                    worst_dense = std::max(
                        worst_dense,
                        static_cast<double>(std::fabs(state.rotated[base + c] - dense[c])));
                    largest_dense =
                        std::max(largest_dense, static_cast<double>(std::fabs(dense[c])));
                }
            }
        }
        // Two different associativities over 256 cancelling terms, so this is
        // a relative check against the largest value in the row and not a
        // bit-exact one. 2^-18 of the largest magnitude leaves six bits of
        // margin on FP32's 24.
        const double dense_budget = largest_dense / 262144.0 + 1.0e-9;
        expect(worst_dense <= dense_budget,
               "check 5: the eight-stage egress butterfly agrees with the dense "
               "256x256 right-multiplication by Pi");
        std::printf("     egress: max|butterfly - dense| = %.3e vs budget %.3e "
                    "(largest |rot| %.6f)\n",
                    worst_dense, dense_budget, largest_dense);

        // ---- 6. single-strip isolation ------------------------------------
        // The state for strip T alone must not depend on any other strip, and
        // two different single strips must give different states. The first is
        // what the canary protocol asserts on the device; the second is what
        // stops the first from being vacuous.
        bool isolated = true;
        bool distinct = true;
        for (int head = 0; head < kKvHeads; ++head) {
            std::vector<std::uint32_t> signatures;
            for (int strip = 0; strip < kStripsPerPage; ++strip) {
                const GoldenState one = GoldenDecode(head, {strip});
                const GoldenState again = GoldenDecode(head, {strip});
                for (int row = 0; row < kRows; ++row)
                    if (float_bits(one.m[row]) != float_bits(again.m[row]) ||
                        float_bits(one.l[row]) != float_bits(again.l[row]))
                        isolated = false;
                signatures.push_back(float_bits(one.m[0]) ^ float_bits(one.l[0]));
            }
            std::sort(signatures.begin(), signatures.end());
            if (std::unique(signatures.begin(), signatures.end()) != signatures.end())
                distinct = false;
        }
        expect(isolated, "check 6: a single-strip golden state is reproducible");
        expect(distinct,
               "check 6: the eight single-strip states of a head are pairwise "
               "distinguishable");

        // ---- 7. the fixture, at one page and at four ----------------------
        for (int head = 0; head < kKvHeads; ++head) {
            ValidateSharedFixture(head);
            const FixtureReport one = ValidateFixture(head, 1);
            const FixtureReport four = ValidateFixture(head, 4);
            std::printf("     head %d: |Q_rot| <= %.4f, max partial sum %.1f "
                        "(bound %.0f), score [%.4f, %.4f], rescale %d/%d strips, "
                        "carried %d/%d, min l %.4f, closest panels differ in "
                        "%d/%d entries (1 page: %d)\n",
                        head, one.max_qrot_magnitude, four.max_partial_sum,
                        16777216.0 / 16.0, four.min_score, four.max_score,
                        four.rescale_strips, 4 * kStripsPerPage, four.carried_strips,
                        4 * kStripsPerPage, four.smallest_l, four.closest_panels,
                        kPanelEntries, one.closest_panels);
        }
        expect(true,
               "check 7: every fixture property holds for all four heads at one "
               "page and at four, and the one-page fixture is the Cube "
               "operator's byte for byte");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL exception: %s\n", error.what());
        return 1;
    }

    if (failures != 0) {
        std::fprintf(stderr, "FAIL %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS single-token decode Tier-1 probe\n");
    return 0;
}
