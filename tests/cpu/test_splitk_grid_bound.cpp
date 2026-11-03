// Host gate for the two shared declarations the rotation work rests on: the
// AIV single-wave grid bound, and the pair of egress oracles.
//
// This is the part of REFINEMENT_OF_COMBAT_KERNELS.md §6.1 that survives on a
// machine with no CANN toolkit and no 950PR, and it is only worth anything
// because the KERNEL and the HOST LAUNCHER share these declarations rather
// than each carrying a copy:
//
//   turboquant_splitk::GridFitsOneWave   the device's early return in
//                                        turboquant_single_token_decode_splitk_aiv.cpp
//                                        and the host launcher's refusal are
//                                        the same function, compiled here from
//                                        a host translation unit;
//   Fixture::expected                    the egress oracle for
//                                        EGRESS_UNROTATE = true;
//   Fixture::expected_rotated            the egress oracle for
//                                        EGRESS_UNROTATE = false.
//
// WHAT THE SECOND HALF IS FOR
//
// A bypass leg graded against a bypassed oracle establishes nothing: if the
// oracle line for the rotated basis were wrong in the same way the kernel was,
// both would agree and both would be wrong. So the two oracle lines are
// checked against EACH OTHER here, through the dense 256x256 rotation, on the
// host, before either is used to grade a device run:
//
//   expected[c] = DenseRotate(expected_rotated . sigmoid)[c] / sigmoid[c]
//
// -- that is, strip the gate off the rotated-basis line, rotate it densely,
// and the gate must put it back onto the model-basis line. `DenseRotate` sums
// over the input channel with a different associativity from the kernel's
// eight-stage butterfly, so this ties the two lines together without assuming
// either factorization.
//
// And the two lines must actually DIFFER, by a wide margin: a bypass that
// silently emitted the model basis anyway would pass a check whose two
// references happened to coincide.
#include "../common/splitk_decode_fixture.hpp"
#include "../../include/turboquant_splitk_contract.hpp"

#include <cmath>
#include <cstdio>

using ascend_micro::sim::require;
namespace contract = turboquant_splitk;

namespace {

void grid_bound()
{
    // The bound is the AIV subcore count, not the Cube core count: 32 clusters
    // of two vector subcores each. A split launch of exactly that many blocks
    // is one wave; one block more is two.
    require(contract::kMaxAivBlocks == 64u, "dav-3510 holds 64 AIV blocks");
    require(contract::kMaxAivBlocks ==
                contract::kAiCoreClusters * contract::kAivSubcoresPerCluster,
            "the bound must be derived from the cluster geometry");

    // Exhaustive over every operand triple the ABI admits, against the
    // predicate's own definition stated independently here.
    int admitted = 0, refused = 0, full_wave = 0;
    for (std::uint32_t requests = 1; requests <= 64; ++requests)
        for (std::uint32_t heads = 1; heads <= 8; ++heads)
            for (std::uint32_t splits = 1; splits <= contract::kMaxSplits; ++splits) {
                const std::uint64_t grid =
                    static_cast<std::uint64_t>(requests) * heads * splits;
                const bool want =
                    splits <= 1u || grid <= contract::kMaxAivBlocks;
                const bool got = contract::GridFitsOneWave(requests, heads, splits);
                require(got == want, "grid bound disagrees with its definition");
                if (got) ++admitted; else ++refused;
                if (got && splits > 1u && grid == contract::kMaxAivBlocks) ++full_wave;
            }
    require(refused > 0, "no triple is refused: the bound is not being applied");
    require(full_wave > 0, "no triple lands exactly on the bound");

    // The three shapes the raise exists for, and the first one past it.
    require(contract::GridFitsOneWave(1, 4, 16), "K=16 at H=4, B=1 must fit");
    require(contract::GridFitsOneWave(2, 4, 8), "K=8 at H=4, B=2 must fit");
    require(contract::GridFitsOneWave(4, 4, 4), "K=4 at H=4, B=4 must fit");
    require(!contract::GridFitsOneWave(2, 4, 16), "a 128-block split grid must not");
    require(!contract::GridFitsOneWave(1, 5, 16), "an 80-block split grid must not");

    // 65 is the first grid past the wave, and it must be refused however the
    // factors are arranged. A bound applied to the wrong axis would let one of
    // these through.
    require(!contract::GridFitsOneWave(65, 1, 2) ||
                65u * 1u * 2u <= contract::kMaxAivBlocks,
            "the request axis escapes the bound");
    for (std::uint32_t splits = 2; splits <= contract::kMaxSplits; ++splits) {
        const std::uint32_t requests =
            contract::kMaxAivBlocks / splits + 1u;  // one block past the wave
        require(!contract::GridFitsOneWave(requests, 1, splits),
                "a grid one block past the wave was admitted");
    }

    // A single-split launch never reaches the barrier, so no grid bounds it.
    for (std::uint32_t requests = 1; requests <= 256; requests *= 2)
        require(contract::GridFitsOneWave(requests, 8, 1),
                "a single-split launch must never be refused");

    std::printf("  ok  grid bound: %d triples admitted, %d refused, %d land exactly "
                "on the %u-block wave\n",
                admitted, refused, full_wave, contract::kMaxAivBlocks);
}

// The two egress oracles, tied together through the dense rotation.
void egress_oracles()
{
    // Real operator geometry, two requests and all four heads, so the row base
    // arithmetic the oracle shares with the kernel is exercised rather than
    // collapsed to a single row.
    splitk_decode_test::Fixture f(4 * splitk_decode_test::kPageTokens, 2,
                                  splitk_decode_test::kHeads, 6,
                                  splitk_decode_test::kPageTokens);
    constexpr int kChannels = splitk_decode_test::kChannels;
    const std::size_t rows = f.expected.size() / kChannels;
    require(rows > 0, "the fixture produced no rows");
    require(f.expected_rotated.size() == f.expected.size(),
            "the two oracle lines must cover the same rows");

    double worst_roundtrip = 0.0, worst_separation = 0.0;
    double largest_model = 0.0, largest_rotated = 0.0;
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * kChannels;
        // Strip the gate off the rotated-basis line to recover Acc / l.
        std::vector<float> normalized(kChannels);
        for (int c = 0; c < kChannels; ++c) {
            const double sigmoid =
                1.0 / (1.0 + std::exp(-static_cast<double>(f.gate[base + c])));
            normalized[c] =
                static_cast<float>(f.expected_rotated[base + c] / sigmoid);
            largest_rotated = std::max<double>(
                largest_rotated, std::abs(f.expected_rotated[base + c]));
            largest_model =
                std::max<double>(largest_model, std::abs(f.expected[base + c]));
        }
        // Rotate it densely and gate it again: that must be the model-basis
        // line, which is how the two oracles are shown to describe one softmax
        // in two bases rather than two unrelated computations.
        const auto rotated = single_token_decode::DenseRotate(normalized.data());
        for (int c = 0; c < kChannels; ++c) {
            const double sigmoid =
                1.0 / (1.0 + std::exp(-static_cast<double>(f.gate[base + c])));
            const double got = rotated[c] * sigmoid;
            const double reference = f.expected[base + c];
            const double scale = std::max(1.0, std::abs(reference));
            worst_roundtrip =
                std::max(worst_roundtrip, std::abs(got - reference) / scale);
            worst_separation = std::max(
                worst_separation,
                std::abs(static_cast<double>(f.expected_rotated[base + c]) -
                         reference));
        }
    }
    // FP32 storage of `expected_rotated`, the gate division and the dense
    // 256-term sum are the whole of the error here; the rotation itself is
    // exact for dyadic inputs.
    require(worst_roundtrip < 1e-5,
            "the rotated-basis oracle does not rotate onto the model-basis one");
    // And they are not the same tensor. If they were, a kernel that ignored
    // EGRESS_UNROTATE entirely would pass both legs.
    require(worst_separation > 0.1 * std::max(largest_model, largest_rotated),
            "the two oracle lines are too close to tell the egress legs apart");
    std::printf("  ok  egress oracles: dense round trip within %.3e relative over "
                "%zu rows; the two bases differ by up to %.4f against maxima "
                "%.4f (model) and %.4f (rotated)\n",
                worst_roundtrip, rows, worst_separation, largest_model,
                largest_rotated);
}

}  // namespace

int main()
{
    try {
        grid_bound();
        egress_oracles();
        std::printf("PASS split-K grid bound and egress oracle gates\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL split-K grid bound gate: %s\n", error.what());
        return 1;
    }
}
