// Tier-1 host probe for the Split-K synchronisation oracle
// (tests/common/l2_atomic_oracle.hpp). No device, no simulator, milliseconds.
//
// A CAModel run compares the device against that oracle, so a wrong oracle is
// indistinguishable from a wrong kernel. Five properties are settled here,
// and each is a property the CAModel gate depends on:
//
//   1. the uniform fixture's combine really is order-independent and
//      bit-exact, for every split count and repetition count the probe
//      sweeps, over randomised interleavings -- this is what lets the device
//      check be `==` rather than a tolerance, and so what makes it a lost
//      update detector;
//   2. the uniform fixture's closed form agrees with the step-by-step
//      combine, which is the only thing that makes having both worthwhile;
//   3. the staggered fixture actually exercises the rescale: its factors are
//      not 1, its carried maximum changes more than once, and the result
//      depends on the arrival order (so a device check against one fixed
//      order would be wrong, and the runner replays the ticket log instead);
//   4. the staged RegGather egress agrees with the dense 256 x 256
//      right-multiplication by Pi, which is what makes the Mode C reference a
//      reference and not a second copy of the device's algorithm;
//   5. the shortfall note names a split whose update was dropped.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "../common/l2_atomic_oracle.hpp"

namespace {

int failures = 0;

void check(bool condition, const char* what)
{
    if (!condition) {
        std::printf("  FAIL %s\n", what);
        ++failures;
    }
}

std::vector<std::uint32_t> arrival_order(std::uint32_t splits, std::uint32_t reps,
                                         std::uint32_t seed)
{
    std::vector<std::uint32_t> order;
    for (std::uint32_t r = 0; r < reps; ++r)
        for (std::uint32_t k = 0; k < splits; ++k) order.push_back(k);
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    return order;
}

// ---- 1 and 2: the uniform fixture -------------------------------------
void test_uniform_is_order_independent()
{
    std::printf("uniform fixture: order independence and the closed form\n");
    for (std::uint32_t splits : {2u, 4u, 8u}) {
        for (std::uint32_t reps : {1u, 2u, 4u}) {
            const l2_probe_oracle::State want =
                l2_probe_oracle::uniform_expected(splits, reps);
            for (std::uint32_t seed = 0; seed < 16; ++seed) {
                const auto order = arrival_order(splits, reps, seed);
                const l2_probe_oracle::State got = l2_probe_oracle::combine_sequence(
                    l2_probe_oracle::Fixture::Uniform, order);
                bool exact = got.m == want.m && got.l == want.l;
                for (std::uint32_t c = 0; c < l2_probe_oracle::kChannels; ++c)
                    exact = exact && got.acc[c] == want.acc[c];
                if (!exact) {
                    std::printf(
                        "  FAIL K=%u reps=%u seed=%u: m %.6g/%.6g l %.6g/%.6g "
                        "acc[0] %.6g/%.6g acc[255] %.6g/%.6g\n",
                        splits, reps, seed, static_cast<double>(got.m),
                        static_cast<double>(want.m), static_cast<double>(got.l),
                        static_cast<double>(want.l), static_cast<double>(got.acc[0]),
                        static_cast<double>(want.acc[0]),
                        static_cast<double>(got.acc[255]),
                        static_cast<double>(want.acc[255]));
                    ++failures;
                    break;
                }
            }
        }
    }
    // The property the bit-exactness rests on: every rescale factor is
    // exactly 1.0f, which is only true because every split shares one m.
    const l2_probe_oracle::State a = l2_probe_oracle::tile_state(
        l2_probe_oracle::Fixture::Uniform, 0);
    const l2_probe_oracle::State b = l2_probe_oracle::tile_state(
        l2_probe_oracle::Fixture::Uniform, 7);
    check(a.m == b.m, "uniform splits share one maximum");
    check(std::exp(a.m - b.m) == 1.0f, "the uniform rescale factor is exactly 1");
    // And the largest value stays inside FP32's exact integer range.
    const l2_probe_oracle::State big = l2_probe_oracle::uniform_expected(8u, 8u);
    check(big.acc[l2_probe_oracle::kChannels - 1] <= 16777216.0f,
          "the widest uniform sum is exactly representable");
    std::printf("  acc[255] at K=8 reps=8 is %.0f, inside 2^24\n",
                static_cast<double>(big.acc[l2_probe_oracle::kChannels - 1]));
}

// ---- 3: the staggered fixture exercises what it claims to --------------
void test_staggered_exercises_the_rescale()
{
    std::printf("staggered fixture: the rescale path\n");
    const auto fixture = l2_probe_oracle::Fixture::Staggered;
    // Distinct maxima, so no two splits share a rescale factor.
    std::vector<float> maxima;
    for (std::uint32_t k = 0; k < 8u; ++k)
        maxima.push_back(l2_probe_oracle::tile_state(fixture, k).m);
    std::vector<float> sorted = maxima;
    std::sort(sorted.begin(), sorted.end());
    check(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
          "every staggered split carries a distinct maximum");
    check(std::all_of(maxima.begin(), maxima.end(), [](float m) { return m < 0.0f; }),
          "every staggered maximum is negative");
    // Not monotone in the split index, so a reduction ordered by block index
    // cannot pass by accident.
    check(!std::is_sorted(maxima.begin(), maxima.end()) &&
              !std::is_sorted(maxima.rbegin(), maxima.rend()),
          "the staggered maxima are not monotone in the split index");

    // The carried maximum must change more than once in index order, or the
    // branch where the CARRIED value wins is never taken.
    std::uint32_t changes = 0;
    float carried = l2_probe_oracle::identity().m;
    for (std::uint32_t k = 0; k < 8u; ++k) {
        const float m = l2_probe_oracle::tile_state(fixture, k).m;
        if (m > carried) {
            carried = m;
            ++changes;
        }
    }
    check(changes >= 2u, "the carried maximum is replaced at least twice");
    check(carried == *std::max_element(maxima.begin(), maxima.end()),
          "the carried maximum ends at the fixture maximum");

    // The factors are genuinely not 1, and the result genuinely depends on
    // the order -- which is why the device check replays the ticket log.
    const auto order_a = arrival_order(8u, 2u, 1u);
    const auto order_b = arrival_order(8u, 2u, 2u);
    const auto got_a = l2_probe_oracle::combine_sequence(fixture, order_a);
    const auto got_b = l2_probe_oracle::combine_sequence(fixture, order_b);
    check(got_a.m == got_b.m, "the combined maximum is order independent");
    bool differs = false;
    for (std::uint32_t c = 0; c < l2_probe_oracle::kChannels; ++c)
        differs = differs || got_a.acc[c] != got_b.acc[c];
    check(differs, "the staggered accumulator depends on the arrival order");
    double worst = 0.0;
    for (std::uint32_t c = 0; c < l2_probe_oracle::kChannels; ++c) {
        const double denom = std::max(1e-30, std::abs(static_cast<double>(got_a.acc[c])));
        worst = std::max(worst, std::abs(static_cast<double>(got_a.acc[c] - got_b.acc[c])) /
                                    denom);
    }
    std::printf("  two interleavings differ by at most %.3g relative\n", worst);
}

// ---- 4: the egress reference against the dense matrix ------------------
void test_egress_matches_the_dense_rotation()
{
    std::printf("Mode C reference: staged egress against the dense matrix\n");
    l2_probe_oracle::State combined = l2_probe_oracle::uniform_expected(4u, 1u);
    const std::vector<float> signs = rot_oracle::canonical_signs();
    std::vector<float> norm(l2_probe_oracle::kChannels);
    for (std::uint32_t c = 0; c < l2_probe_oracle::kChannels; ++c)
        norm[c] = combined.acc[c] / combined.l;

    const std::vector<float> staged = rot_oracle::algo_reg_gather(
        rot_oracle::Direction::Egress, norm.data(), signs.data(), 1.0f / 16.0f);
    const std::vector<float> dense = rot_oracle::dense_right(norm.data(), signs.data());

    double worst = 0.0;
    for (std::uint32_t c = 0; c < l2_probe_oracle::kChannels; ++c) {
        const double want = static_cast<double>(dense[c]) / 16.0;
        const double denom = std::max(1.0, std::abs(want));
        worst = std::max(worst, std::abs(static_cast<double>(staged[c]) - want) / denom);
    }
    check(worst < 1e-6, "the staged egress matches the dense right-multiplication");
    std::printf("  worst relative difference %.3g over %u channels\n", worst,
                l2_probe_oracle::kChannels);

    // The gate must actually change the answer, or Mode C would pass with the
    // gate omitted entirely. Only the channels the rotation leaves non-zero
    // can move: a right-multiplication by Pi of a linear ramp is mostly
    // zeros, and sigmoid(gate) . 0 is 0 for every gate. So the property is
    // that EVERY non-zero channel moves, and that there are enough of them
    // for the check to mean something.
    const std::vector<float> gated = l2_probe_oracle::egress_reference(combined);
    std::uint32_t nonzero = 0, moved = 0;
    for (std::uint32_t c = 0; c < l2_probe_oracle::kChannels; ++c) {
        if (staged[c] == 0.0f) {
            check(gated[c] == 0.0f, "a zero channel stays zero through the gate");
            continue;
        }
        ++nonzero;
        if (gated[c] != staged[c]) ++moved;
    }
    check(nonzero >= 8u, "the rotation leaves enough non-zero channels to gate");
    check(moved == nonzero, "the gate moves every non-zero channel");
    std::printf("  the gate moved %u of %u non-zero channels\n", moved, nonzero);
    const std::vector<float> gate = l2_probe_oracle::gate_vector();
    check(*std::min_element(gate.begin(), gate.end()) < 0.0f &&
              *std::max_element(gate.begin(), gate.end()) > 0.0f,
          "the gate spans both signs");

    // bf16 rounding, the device's CAST_RINT: round to nearest, ties to even.
    check(l2_probe_oracle::round_bf16(1.0f) == 1.0f, "bf16 keeps an exact value");
    const float half_ulp = l2_probe_oracle::bf16_to_float(0x3F81u);
    check(l2_probe_oracle::round_bf16(half_ulp) == half_ulp,
          "bf16 round-trips a representable value");
}

// ---- 5: the shortfall note --------------------------------------------
void test_shortfall_names_a_dropped_split()
{
    std::printf("diagnostics: the uniform shortfall note\n");
    const std::uint32_t splits = 8u, reps = 1u;
    const auto want = l2_probe_oracle::uniform_expected(splits, reps);
    const std::string clean =
        l2_probe_oracle::uniform_shortfall_note(want.acc[0], splits, reps);
    check(clean.find("shortfall 0.0") != std::string::npos,
          "a complete reduction reports no shortfall");
    // Drop split 5's contribution and see whether the note names it.
    const float dropped = want.acc[0] - static_cast<float>(1u << 5);
    const std::string note =
        l2_probe_oracle::uniform_shortfall_note(dropped, splits, reps);
    check(note.find("split(s) 5") != std::string::npos,
          "a single dropped split is named");
    std::printf("  %s\n", note.c_str());
}

}  // namespace

int main()
{
    std::printf("Tier-1 Split-K synchronisation oracle probe\n");
    test_uniform_is_order_independent();
    test_staggered_exercises_the_rescale();
    test_egress_matches_the_dense_rotation();
    test_shortfall_names_a_dropped_split();
    if (failures != 0) {
        std::printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("\nOK Tier-1 Split-K oracle probe\n");
    return 0;
}
