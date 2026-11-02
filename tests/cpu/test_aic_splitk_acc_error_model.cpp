// How much of the AIC Split-K partial's deviation from the FP64 oracle is the
// kernel's own FP32 arithmetic?
//
// This re-executes the decode recurrence on the host in FP32, phase by phase
// and strip by strip in the order routine_vector_decode_recurrence.hpp issues
// it, and compares the result against the SAME oracle `AicRig::Validate` uses.
// It exists because a device deviation at four pages is otherwise
// unattributable: "FP32 drift over a longer carry" and "a defect in the carry"
// predict the same sign and the same growth direction, and only a model of the
// first can rule it out.
//
// The answer it establishes, at every geometry below, is that the FP32
// recurrence lands three to four ORDERS OF MAGNITUDE inside the rig's
// tolerance. So FP32 accumulation is not an explanation for a deviation near
// that tolerance, and the remaining candidates are the device's own `Exp`
// approximation (which enters `acc` amplified by the cancellation in a signed
// contraction, and `l` -- a sum of positive terms -- unamplified) or a real
// defect.
//
// It also reports the per-channel condition number, sum|terms| / |acc|, which
// is the factor that turns a relative error on one probability into a relative
// error on the accumulator. The rig's `acc` bound is written against
// max|acc| and does not contain that factor; this is where the number to put
// there would come from.
#include "../common/aic_splitk_decode_fixture.hpp"

#include <cmath>
#include <cstdio>

namespace test = splitk_decode_test;
using test::Fixture;
using ascend_micro::sim::require;

namespace {
constexpr int kChannels = 256;
constexpr int kRows = 16;

struct Emulated {
    std::vector<float> m, l, acc;
    // Sum of |contribution| per (row, channel), carried at the running m
    // scale exactly as `acc` is, so the two are directly comparable.
    std::vector<double> abs_terms;
};

// The kernel's recurrence in FP32. Every phase is in the order
// `vector_decode_recur_query` calls it: scale the row, reduce the maximum,
// fold it into the carried maximum, centre and exponentiate, rescale the
// denominator, fold the value scale, contract, rescale the accumulator.
Emulated Emulate(const Fixture& f, int request, int head, unsigned begin, unsigned pages)
{
    Emulated out;
    out.m.assign(kRows, -3.0e38f);
    out.l.assign(kRows, 0.0f);
    out.acc.assign(kRows * kChannels, 0.0f);
    out.abs_terms.assign(kRows * kChannels, 0.0);
    const int strips_per_page = f.page_tokens / 16;
    const std::size_t value_plane = f.scales.size() / 2;

    for (int row = 0; row < kRows; ++row) {
        const auto q_base = ((request * f.heads + head) * kRows + row) * kChannels;
        for (unsigned p = 0; p < pages; ++p) {
            const auto physical = f.table[request * f.pages + begin + p];
            for (int strip = 0; strip < strips_per_page; ++strip) {
                const auto base = (static_cast<std::size_t>(physical) * f.heads + head) *
                                  f.page_tokens * 128 +
                                  static_cast<std::size_t>(strip) * 2048;
                const auto scale0 = (static_cast<std::size_t>(physical) * f.heads + head) *
                                    f.page_tokens + static_cast<std::size_t>(strip) * 16;

                // The Cube's score panel: bf16 x exact-integer nibbles,
                // accumulated in FP32 over the 256-channel contraction.
                float scaled[16], tile_max = -3.0e38f;
                for (int t = 0; t < 16; ++t) {
                    float dot = 0.0f;
                    for (int c = 0; c < kChannels; ++c)
                        dot += single_token_decode::bfloat16_to_float(f.q[q_base + c]) *
                               static_cast<float>(
                                   Fixture::Unpack(f.key.data() + base, t * kChannels + c));
                    scaled[t] = dot * f.scales[scale0 + t];
                    tile_max = std::max(tile_max, scaled[t]);
                }
                const float m_new = std::max(out.m[row], tile_max);
                const float alpha = std::exp(out.m[row] - m_new);

                float prob[16], p_sum = 0.0f;
                for (int t = 0; t < 16; ++t) prob[t] = std::exp(scaled[t] - m_new);
                for (int t = 0; t < 16; ++t) p_sum += prob[t];
                const float l_new = out.l[row] * alpha + p_sum;

                float p_scaled[16];
                for (int t = 0; t < 16; ++t)
                    p_scaled[t] = prob[t] * f.scales[value_plane + scale0 + t];
                for (int c = 0; c < kChannels; ++c) {
                    float weighted = 0.0f;
                    double absolute = 0.0;
                    for (int t = 0; t < 16; ++t) {
                        const auto v = static_cast<float>(
                            Fixture::Unpack(f.value.data() + base, c * 16 + t));
                        weighted += v * p_scaled[t];
                        absolute += std::abs(static_cast<double>(v) * p_scaled[t]);
                    }
                    const auto at = row * kChannels + c;
                    out.acc[at] = out.acc[at] * alpha + weighted;
                    out.abs_terms[at] = out.abs_terms[at] * alpha + absolute;
                }
                out.m[row] = m_new;
                out.l[row] = l_new;
            }
        }
    }
    return out;
}

struct Worst {
    double margin = 0, deviation = 0, tolerance = 0, condition = 0;
    int row = 0, channel = 0;
};

// The rig's own bounds, kept here verbatim so the two cannot drift apart
// silently. If one is retuned, this probe is where the evidence comes from.
double AccTolerance(const test::AicPartial& ref, int row)
{
    double scale = 0;
    for (int c = 0; c < kChannels; ++c)
        scale = std::max(scale, std::abs(ref.acc[row * kChannels + c]));
    return 2e-4 * scale + 1e-5 * ref.l[row] + 1e-6;
}

void Report(int context, int page_tokens)
{
    const int pages = context / page_tokens;
    Fixture f(context, 1, 1, kRows, page_tokens);
    const auto ref = test::ReferenceAicPartial(f, 0, 0, 0, static_cast<unsigned>(pages));
    const auto emu = Emulate(f, 0, 0, 0, static_cast<unsigned>(pages));

    double worst_m = 0, worst_l = 0;
    Worst worst;
    for (int row = 0; row < kRows; ++row) {
        worst_m = std::max(worst_m, std::abs(emu.m[row] - ref.m[row]) /
                                        (2e-6 * std::max(1.0, std::abs(ref.m[row]))));
        worst_l = std::max(worst_l, std::abs(emu.l[row] - ref.l[row]) /
                                        (1e-4 * std::max(1.0, ref.l[row])));
        const double tolerance = AccTolerance(ref, row);
        for (int c = 0; c < kChannels; ++c) {
            const auto at = row * kChannels + c;
            const double deviation = std::abs(emu.acc[at] - ref.acc[at]);
            if (deviation / tolerance > worst.margin) {
                worst.margin = deviation / tolerance;
                worst.deviation = deviation;
                worst.tolerance = tolerance;
                worst.condition = emu.abs_terms[at] /
                                  std::max(1e-30, std::abs(ref.acc[at]));
                worst.row = row;
                worst.channel = c;
            }
        }
    }

    // The largest cancellation anywhere in the partial, which is the factor a
    // condition-aware `acc` bound would have to carry.
    double worst_condition = 0, worst_relative = 0;
    for (int row = 0; row < kRows; ++row)
        for (int c = 0; c < kChannels; ++c) {
            const auto at = row * kChannels + c;
            if (std::abs(ref.acc[at]) > 1e-6)
                worst_condition = std::max(worst_condition,
                                           emu.abs_terms[at] / std::abs(ref.acc[at]));
            if (emu.abs_terms[at] > 0)
                worst_relative = std::max(worst_relative,
                                          std::abs(emu.acc[at] - ref.acc[at]) /
                                              emu.abs_terms[at]);
        }

    std::printf("context=%-5d page_tokens=%-4d pages=%-3d | FP32 recurrence vs FP64 oracle\n",
                context, page_tokens, pages);
    std::printf("  margins (fraction of the rig's own bound): m=%.6fx l=%.6fx acc=%.6fx\n",
                worst_m, worst_l, worst.margin);
    std::printf("  worst acc: row=%-3d c=%-4d deviation=%.4e tolerance=%.4e condition=%.1fx\n",
                worst.row, worst.channel, worst.deviation, worst.tolerance, worst.condition);
    std::printf("  worst cancellation anywhere: %.1fx ; worst |deviation|/sum|terms|: %.3e\n\n",
                worst_condition, worst_relative);

    // The claim this probe exists to make, asserted rather than printed: the
    // kernel's own FP32 arithmetic is nowhere near the rig's bounds, so a
    // device deviation at those bounds is not FP32 accumulation.
    require(worst.margin < 0.01,
            "FP32 recurrence is within 1% of the rig's acc bound: the bound can no "
            "longer distinguish arithmetic from a defect");
    require(worst_l < 0.01 && worst_m < 0.01,
            "FP32 recurrence is within 1% of the rig's m/l bounds");
}

// The emulation above reads the caches through the SAME index arithmetic as
// `ReferenceAicPartial`, so agreement between the two says nothing about
// whether either addresses the cache correctly. This does: it takes the
// oracle's whole-window partial, normalizes, rotates and gates it, and
// compares against `Fixture::expected`, which the fixture builds in its own
// constructor from its own independent loop over the same bytes.
//
// It matters because the alternative explanation for a device deviation at a
// page count the oracle has never been cross-checked at is that the ORACLE is
// wrong there, and the device right. `test_aic_splitk_decode_probe` covers
// three 16-token pages and 8/32/128 128-token pages; the page-depth ladder
// runs one, two, three and four 16-token pages, so four was uncovered.
void CrossCheckOracle(int context, int page_tokens)
{
    const int pages = context / page_tokens;
    Fixture f(context, 1, 1, kRows, page_tokens);
    const auto full = test::ReferenceAicPartial(f, 0, 0, 0, static_cast<unsigned>(pages));
    double worst = 0;
    for (int row = 0; row < kRows; ++row) {
        std::vector<float> normalized(kChannels);
        for (int c = 0; c < kChannels; ++c)
            normalized[c] = static_cast<float>(full.acc[row * kChannels + c] / full.l[row]);
        const auto rotated = single_token_decode::DenseRotate(normalized.data());
        const auto base = row * kChannels;
        for (int c = 0; c < kChannels; ++c) {
            const float result = rotated[c] / (1.0f + std::exp(-f.gate[base + c]));
            worst = std::max(worst, static_cast<double>(
                                        std::abs(result - f.expected[base + c])));
        }
    }
    std::printf("oracle cross-check  context=%-5d pages=%-3d | worst |oracle - fixture "
                "generator| = %.3e\n", context, pages, worst);
    require(worst < 2e-6,
            "the AIC oracle disagrees with the fixture's own generator: a device "
            "deviation at this geometry cannot be attributed until this is resolved");
}
}  // namespace

int main()
{
    try {
        std::puts("AIC Split-K partial: FP32 arithmetic error model\n");
        // The 16-token page geometry is the page-depth ladder's, one to four
        // pages; the 128-token geometry is a full eight-strip page.
        for (int context : {16, 32, 48, 64}) Report(context, 16);
        for (int context : {128, 256}) Report(context, 128);
        std::puts("Oracle addressing, against the fixture's independent generator:");
        for (int context : {16, 32, 48, 64}) CrossCheckOracle(context, 16);
        for (int context : {128, 256}) CrossCheckOracle(context, 128);
        std::puts("\nPASS: the FP32 recurrence stays below 1% of every rig bound, and "
                  "the oracle agrees with the fixture's own generator, at every "
                  "geometry above.");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
