#pragma once

// Host oracle for the Split-K synchronisation probe
// (routines/routine_probe_l2_atomic_sync.hpp).
//
// A CAModel run compares the device against this oracle, so a wrong oracle is
// indistinguishable from a wrong kernel. What is settled here, with no device
// and no simulator: the log-sum-exp combine, the two fixtures and the
// property that makes one of them bit-exact, and the tail-split egress
// reference.
//
// The order-independence that the bit-exact gate rests on
// ------------------------------------------------------
// The combine is
//
//   m'   = max(m, m_t)
//   a    = exp(m  - m'),  a_t = exp(m_t - m')
//   l'   = l . a + l_t . a_t
//   Acc' = Acc . a + Acc_t . a_t
//
// which is associative and commutative in exact arithmetic but NOT in FP32:
// the rescale factors depend on the order the splits arrive in, and the
// spinlock does not fix an order. So a probe that only checked the combined
// accumulator against one canonical order could not distinguish a lost update
// from a different -- and equally valid -- interleaving.
//
// The `uniform` fixture removes that ambiguity rather than tolerating it.
// Give every split the SAME m, and both factors are exp(0) = 1.0f exactly for
// every step in every order, so the combine degenerates to
//
//   l' = l + l_t,   Acc' = Acc + Acc_t
//
// a plain sum. Make every addend a small integer and the sum is exact in FP32
// whatever the order. The addends are then chosen so the sum NAMES its
// contributors: split k contributes 2^k . (c + 1) to channel c, so the
// combined accumulator is (2^K - 1)(c + 1) and a missing split k leaves
// (2^K - 1 - 2^k)(c + 1). A lost update is therefore not merely detected, it
// is identified -- which is the whole point of a mutual-exclusion probe,
// because a lost update is exactly what a broken critical section produces.
//
// The largest value is (2^8 - 1) . 256 = 65280, well inside FP32's exact
// integer range, so the gate is bit-exact at every K up to 8.
//
// The `staggered` fixture is the one that exercises the rescale at all: each
// split carries a distinct NEGATIVE m, which is also what makes the hardware
// atomic maximum's handling of negative floats observable. Its accumulator is
// checked to a tolerance, with the order taken from the device's own ticket
// log so the oracle replays the interleaving that actually happened.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "rotate_butterfly_oracle.hpp"

namespace l2_probe_oracle {

constexpr std::uint32_t kChannels = 256;
constexpr std::uint32_t kStateLanes = 8;
constexpr std::uint32_t kStateElems = 2u * kStateLanes;            // 16
constexpr std::uint32_t kRecordElems = kStateElems + kChannels;    // 272
constexpr std::uint32_t kMOffset = 0;
constexpr std::uint32_t kLOffset = kStateLanes;
constexpr std::uint32_t kAccOffset = kStateElems;

// Mirrors routine_probe_l2_atomic_sync.hpp's metric slice.
constexpr std::uint32_t kMetricSlots = 24;
constexpr std::uint32_t kMetricLanesPerSlot = 8;
constexpr std::uint32_t kMetricElems = kMetricSlots * kMetricLanesPerSlot;
enum Metric : std::uint32_t {
    kMetricBlock = 0,
    kMetricMode = 1,
    kMetricReps = 2,
    kMetricRegion = 3,
    kMetricPhase0 = 4,
    kMetricPhase1 = 5,
    kMetricPhase2 = 6,
    kMetricPhase3 = 7,
    kMetricPhase4 = 8,
    kMetricRepBase = 16,
};

enum class Fixture { Uniform, Staggered };

// One split's state, in the device's record layout.
struct State {
    float m = 0.0f;
    float l = 0.0f;
    std::vector<float> acc;
    State() : acc(kChannels, 0.0f) {}
};

// The identity of the combine: m = -inf would make exp(m - m') a NaN on the
// first step through inf - inf, so the carried maximum starts at the same
// large negative the decode operator seeds its state with.
constexpr float kMNegative = -3.0e38f;

inline State identity()
{
    State s;
    s.m = kMNegative;
    s.l = 0.0f;
    return s;
}

// --------------------------------------------------------------- fixtures
// Split k's contribution. `uniform` gives every split the same m so the
// combine is an exact sum; `staggered` gives each a distinct negative m so
// the rescale runs.
inline State tile_state(Fixture fixture, std::uint32_t split)
{
    State s;
    const float weight = static_cast<float>(1u << split);
    for (std::uint32_t c = 0; c < kChannels; ++c)
        s.acc[c] = weight * static_cast<float>(c + 1);
    s.l = weight;
    if (fixture == Fixture::Uniform) {
        // One shared maximum. Negative, because decode m is, and because the
        // atomic maximum over equal negatives must still land on that value.
        s.m = -4.0f;
    } else {
        // Distinct, negative, and NOT monotone in the split index: a reduction
        // that happened to be ordered by block would look correct against a
        // monotone fixture even if it ignored the carried maximum.
        static const float kM[8] = {-2.0f, -9.0f, -5.0f, -1.0f,
                                    -12.0f, -3.0f, -7.0f, -6.0f};
        s.m = kM[split % 8u];
    }
    return s;
}

inline std::vector<float> tile_array(Fixture fixture, std::uint32_t splits)
{
    std::vector<float> out(static_cast<std::size_t>(splits) * kRecordElems, 0.0f);
    for (std::uint32_t k = 0; k < splits; ++k) {
        const State s = tile_state(fixture, k);
        float* rec = out.data() + static_cast<std::size_t>(k) * kRecordElems;
        for (std::uint32_t i = 0; i < kStateLanes; ++i) {
            rec[kMOffset + i] = s.m;   // lane-replicated, as the device carries it
            rec[kLOffset + i] = s.l;
        }
        for (std::uint32_t c = 0; c < kChannels; ++c) rec[kAccOffset + c] = s.acc[c];
    }
    return out;
}

// --------------------------------------------------------------- the combine
// One read-modify-write step, in the device's exact instruction order: the
// maximum, the two exponentials, then the two weighted sums. Computed in FP32
// throughout, because the device is FP32 throughout and a double-precision
// oracle would mask a real divergence.
inline State combine_step(const State& curr, const State& tile)
{
    State out;
    out.m = curr.m > tile.m ? curr.m : tile.m;
    const float a_curr = std::exp(curr.m - out.m);
    const float a_tile = std::exp(tile.m - out.m);
    out.l = curr.l * a_curr + tile.l * a_tile;
    for (std::uint32_t c = 0; c < kChannels; ++c)
        out.acc[c] = curr.acc[c] * a_curr + tile.acc[c] * a_tile;
    return out;
}

// The combined state after `reps` rounds of all `splits` splits, in the given
// arrival order. `order` holds split indices, one per acquisition; passing the
// device's ticket log replays the interleaving that actually happened.
inline State combine_sequence(Fixture fixture, const std::vector<std::uint32_t>& order)
{
    State acc = identity();
    for (std::uint32_t split : order) acc = combine_step(acc, tile_state(fixture, split));
    return acc;
}

// The uniform fixture's closed form: every factor is exactly 1.0f, so the
// result is the plain sum of the contributions and does not depend on `order`
// at all. Derived independently of `combine_step` on purpose -- an oracle that
// reused the same loop could not catch an error in the loop.
inline State uniform_expected(std::uint32_t splits, std::uint32_t reps)
{
    if (splits > 8u) throw std::invalid_argument("uniform fixture covers K <= 8");
    State out;
    out.m = tile_state(Fixture::Uniform, 0).m;
    const float mask = static_cast<float>((1u << splits) - 1u);
    const float rounds = static_cast<float>(reps);
    out.l = mask * rounds;
    for (std::uint32_t c = 0; c < kChannels; ++c)
        out.acc[c] = mask * rounds * static_cast<float>(c + 1);
    return out;
}

// What a uniform-fixture shortfall says about which splits lost an update.
//
// Channel 0's exact value is reps . sum_k 2^k, so the shortfall against the
// expectation is sum_k (lost_k . 2^k) with 0 <= lost_k <= reps. That
// decomposition is NOT unique once a split can lose more than one update --
// two lost updates from split 0 and one from split 1 both show as 2 -- so
// this reports the shortfall and offers the at-most-one-per-split reading as
// a hypothesis rather than a conclusion. It is a diagnostic aid for a failing
// run; the gate itself is the bit-exact comparison against
// `uniform_expected`, which needs none of this.
inline std::string uniform_shortfall_note(float measured_channel0,
                                          std::uint32_t splits, std::uint32_t reps)
{
    const float expected =
        static_cast<float>((1u << splits) - 1u) * static_cast<float>(reps);
    const float shortfall = expected - measured_channel0;
    char head[160];
    std::snprintf(head, sizeof head,
                  "channel 0: measured %.1f, expected %.1f, shortfall %.1f",
                  static_cast<double>(measured_channel0),
                  static_cast<double>(expected), static_cast<double>(shortfall));
    std::string note(head);
    if (shortfall <= 0.0f) return note;
    auto remaining = static_cast<std::uint32_t>(shortfall + 0.5f);
    std::string splits_note;
    for (std::uint32_t k = splits; k-- > 0;) {
        const std::uint32_t weight = 1u << k;
        if (remaining >= weight) {
            remaining -= weight;
            splits_note += (splits_note.empty() ? "" : ", ") + std::to_string(k);
        }
    }
    if (remaining == 0u && !splits_note.empty())
        note += "; consistent with one lost update from split(s) " + splits_note;
    return note;
}

// --------------------------------------------------------- Mode C reference
// The tail split's egress: normalise, rotate by Pi on the right, gate, and
// round to bf16. `algo_reg_gather` is the per-stage mirror of the device's
// RegGather leaf, not the dense matrix, so this reference follows the device's
// associativity rather than an idealised one.
inline std::vector<float> gate_vector()
{
    std::vector<float> gate(kChannels);
    // Spread across the sigmoid's interesting range, both signs, and never
    // exactly zero -- a gate of 0 would hide a missing factor of 0.5.
    for (std::uint32_t c = 0; c < kChannels; ++c)
        gate[c] = -3.0f + 6.0f * (static_cast<float>(c % 61u) + 0.5f) / 61.0f;
    return gate;
}

inline std::vector<float> egress_reference(const State& combined)
{
    const std::vector<float> signs = rot_oracle::canonical_signs();
    std::vector<float> norm(kChannels);
    for (std::uint32_t c = 0; c < kChannels; ++c) norm[c] = combined.acc[c] / combined.l;
    const std::vector<float> rotated = rot_oracle::algo_reg_gather(
        rot_oracle::Direction::Egress, norm.data(), signs.data(),
        1.0f / 16.0f);
    const std::vector<float> gate = gate_vector();
    std::vector<float> out(kChannels);
    for (std::uint32_t c = 0; c < kChannels; ++c)
        out[c] = rotated[c] / (1.0f + std::exp(-gate[c]));
    return out;
}

// bf16 round-to-nearest-even, for comparing against the device's CAST_RINT.
inline float round_bf16(float value)
{
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    const std::uint32_t lower = bits & 0xFFFFu;
    std::uint32_t upper = bits >> 16;
    if (lower > 0x8000u || (lower == 0x8000u && (upper & 1u) != 0u)) ++upper;
    const std::uint32_t rounded = upper << 16;
    float out;
    std::memcpy(&out, &rounded, 4);
    return out;
}

inline float bf16_to_float(std::uint16_t bits)
{
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &wide, 4);
    return out;
}

}  // namespace l2_probe_oracle
