#pragma once

// The egress of paged decode attention, on AIV0: normalise the accumulator,
// rotate it back out of the cache's channel basis, and gate it.
// Ascend 950PR / arch35 / dav-3510 / CANN 9.2.
//
// Why a rotation is needed at all
// -------------------------------
// The KV cache stores values already rotated by Pi, because that is what the
// prefill operator wrote (turboquant_reshape_and_cache_*.cpp). So the
// accumulator this kernel builds lives in the rotated channel basis:
//
//   Acc[q] = sum_t P[q][t] . (Pi . V[t]) = Pi . ctx[q]
//
// and since Pi Pi^T = I exactly, ctx[q] = Acc[q] . Pi -- a right-multiplication
// by the same matrix, which is the operator's egress stage.
//
// Why eight butterflies and not a second Cube pass
// ------------------------------------------------
// Pi = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32) / 16, and right-multiplying
// by it decomposes into
//
//   strides 32, 64, 128     the 8-point Hadamard across channel groups
//   elementwise             the sign diagonal
//   strides 1, 2, 4, 8, 16  the 32-point Hadamard inside each group
//   scale                   1 / 16
//
// one radix-2 butterfly stage per stride, where a stage at stride s is
//
//   out[i] = in[i ^ s] + sgn_s(i) . in[i],   sgn_s(i) = (i & s) ? -1 : +1.
//
// That is 8 passes over 256 FP32 lanes per query, exact in FP32 for dyadic
// inputs, run ONCE at the end of the strip sequence rather than per strip.
//
// This is the opposite choice from the prefill operator's ingress, which folds
// the whole rotation into the Cube as 64 Mmads against resident blocks, and
// the two are consistent: that measurement
// (reports/reshape_and_cache_a8.md) was about a rotation in the hot loop, once
// per cached page, where four full-panel Vector passes cost the Cube a third
// of its cycles waiting. Here the rotation is outside the loop entirely, the
// Cube is already finished, and putting it back on the Cube would need another
// cross-core handshake and another Fixpipe to return the result. One pass over
// 16 x 256 lanes at the end is not the same trade.
//
// Why three of the eight stages need a Gather and five do not
// -----------------------------------------------------------
// An FP32 DataBlock is 8 lanes, and every vector operand must start 32-byte
// aligned. For s >= 8 the partner lane i ^ s is a whole number of DataBlocks
// away, so the stage is two repeat-form instructions over contiguous runs --
// `Add` for the lower half of each 2s-lane group and `Sub` for the upper half,
// with the second source offset by s elements, which is s * 4 >= 32 bytes and
// therefore aligned.
//
// For s in {1, 2, 4} the partner is inside the DataBlock. A tensor offset of
// 4, 8 or 16 bytes is not 32-byte aligned, so the contiguous form is
// unavailable, and the masked form cannot reach it either: a mask selects
// lanes within a repeat but both operands still start at their own aligned
// bases. `Gather` is the one API whose source footprint is per-element, so
// those three stages pay a Gather plus a sign multiply. The offset and sign
// vectors are 256 entries each, three of each, loaded once -- 6 KiB, against
// the 16 KiB a single 4096-entry offset vector would have cost.

#include <cstdint>
#include <utility>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#endif

namespace ascend_micro {

constexpr std::uint32_t kEgressChannels = 256;
constexpr std::uint32_t kEgressVecLanes = 64;      // FP32 lanes in one repeat
constexpr std::uint32_t kEgressLanesPerBlock = 8;  // FP32 per 32-byte DataBlock

// The stage order, and where the sign diagonal sits. Mirrored exactly by
// `EgressButterfly` in tests/common/paged_decode_oracle.hpp, which the Tier-1
// probe checks against the dense 256 x 256 right-multiplication.
constexpr std::uint32_t kEgressStages = 8;
constexpr std::uint32_t kEgressStrideOf[kEgressStages] = {32, 64, 128, 1, 2, 4, 8, 16};
constexpr std::uint32_t kEgressSignBeforeStage = 3;

// The three sub-DataBlock stages, in the order they appear above.
constexpr std::uint32_t kEgressGatherStages = 3;
constexpr std::uint32_t kEgressGatherStrideOf[kEgressGatherStages] = {1, 2, 4};

constexpr float kEgressScale = 1.0f / 16.0f;  // 1 / sqrt(256)

// Compile-time shape of a block-aligned stage.
template <std::uint32_t S>
constexpr std::uint32_t kEgressLanesPerRepeat = S < kEgressVecLanes ? S : kEgressVecLanes;
template <std::uint32_t S>
constexpr std::uint32_t kEgressRepeats =
    (kEgressChannels / 2u) / kEgressLanesPerRepeat<S>;
template <std::uint32_t S>
constexpr std::uint32_t kEgressRepStrideBlocks =
    S <= kEgressVecLanes ? (2u * S / kEgressLanesPerBlock)
                         : (kEgressVecLanes / kEgressLanesPerBlock);

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

// A 0-ULP divide, the same configuration routine_vector_normalize_store.hpp
// uses, so the normalisation here and there round identically.
constexpr AscendC::DivConfig kEgressPreciseDiv = {
    AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};

struct NativeEgressRotateEngine {
    // ---- inputs ----------------------------------------------------------
    AscendC::LocalTensor<float> acc;      // kTokens * kEgressChannels, rotated basis
    AscendC::LocalTensor<float> l_prev;   // kTokens replicated 32-byte blocks
    // One query row at a time. The whole [16 x 256] gate would be 16 KiB of
    // Unified Buffer for a tensor each query touches once, so the operator
    // refills this row per query over a single MTE2_V edge instead.
    AscendC::LocalTensor<float> gate;     // kEgressChannels, one query row
    AscendC::LocalTensor<float> pi_sign;  // kEgressChannels, +-1
    // Three offset vectors and three sign vectors, one pair per sub-block
    // stage, each kEgressChannels entries, concatenated in stride order.
    AscendC::LocalTensor<std::uint32_t> gather_offsets;
    AscendC::LocalTensor<float> gather_signs;

    // ---- scratch ---------------------------------------------------------
    AscendC::LocalTensor<float> l_broadcast;  // kEgressChannels
    AscendC::LocalTensor<float> buffer_a;     // kEgressChannels
    AscendC::LocalTensor<float> buffer_b;     // kEgressChannels
    AscendC::LocalTensor<float> permuted;     // kEgressChannels
    AscendC::LocalTensor<float> signed_tmp;   // kEgressChannels
    AscendC::LocalTensor<float> denominator;  // kEgressChannels

    // ---- outputs ---------------------------------------------------------
    AscendC::LocalTensor<bfloat16_t> context;  // kTokens * kEgressChannels
    // The rotated, pre-gate row, kept per query. Not needed by the operator,
    // but it is the only place the rotation can be separated from the gate
    // from outside the kernel, and a six-minute CAModel run that can only say
    // "the egress is wrong" is a run mostly wasted.
    AscendC::LocalTensor<float> rotated;  // kTokens * kEgressChannels

    static constexpr std::int32_t kChannelLanes =
        static_cast<std::int32_t>(kEgressChannels);
    static constexpr std::int32_t kVecLanes = static_cast<std::int32_t>(kEgressVecLanes);

    // ---- the current query's views, set by select_query ------------------
    AscendC::LocalTensor<float> acc_row;           // kEgressChannels
    AscendC::LocalTensor<float> l_cur;             // one replicated block
    AscendC::LocalTensor<bfloat16_t> context_row;  // kEgressChannels
    AscendC::LocalTensor<float> rotated_row;       // kEgressChannels

    std::uint32_t acc_base = 0;
    std::uint32_t l_base = 0;
    std::uint32_t context_base = 0;
    std::uint32_t rotated_base = 0;
    AscendC::TPosition position = AscendC::TPosition::VECCALC;

    // Runtime per-query views, for the same two reasons the recurrence engine
    // uses them: forming a base address from a loop counter reads no Unified
    // Buffer, so it is not the scalar trip section 1.2 forbids; and unrolling
    // sixteen copies of the eight-stage rotation below made the AIV
    // translation unit uncompilable in practice.
    __aicore__ inline void select_query(std::uint32_t query)
    {
        acc_row = {position, acc_base + query * kEgressChannels * 4u, kEgressChannels};
        l_cur = {position, l_base + query * 8u * 4u, 8u};
        context_row = {position, context_base + query * kEgressChannels * 2u,
                       kEgressChannels};
        rotated_row = {position, rotated_base + query * kEgressChannels * 4u,
                       kEgressChannels};
    }

    // ctx = acc / l. `Duplicate`'s tensor form broadcasts lane 0 of its source
    // to `count` lanes; on dav-3510 it reads a whole 256-byte vector register
    // from the source base regardless of count, so the caller must keep that
    // much addressable Unified Buffer behind the last query's state block. It
    // performs no scalar read and needs no TPipe, so it is a footprint
    // constraint and not a scalar hazard -- the operator static_asserts it.
    // ctx = (acc / l) / 16. The 1/16 normalisation of Pi is applied HERE
    // rather than after the butterflies: the rotation is linear, so scaling
    // its input is identical to scaling its output, and doing it on the way in
    // keeps every operand of every later stage distinct from its destination.
    //
    // That is not a style preference. The first revision ended `rotate()` with
    // `Muls(buffer_b, buffer_b, 1/16)` and computed the sigmoid denominator
    // with `Exp(denominator, denominator)` and `Adds(denominator, denominator,
    // 1)` -- three in-place vector operations, a shape that appears nowhere
    // else in this harness. With them the device matched none of nine
    // candidate egress formulations, which is what two independent faults
    // looks like. Every destination below is distinct from its sources.
    __aicore__ inline void normalise()
    {
        AscendC::Duplicate(l_broadcast, l_cur, kChannelLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div<float, kEgressPreciseDiv>(permuted, acc_row, l_broadcast,
                                               kChannelLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(buffer_a, permuted, kEgressScale, kChannelLanes);
    }

    // A block-aligned stage. Within each 2S-lane group the lower S lanes get
    // lo + hi and the upper S lanes get lo - hi; `dst` must not alias `src`,
    // because the lower half's write would otherwise clobber an operand of the
    // upper half's.
    template <std::uint32_t S>
    __aicore__ inline void block_stage(const AscendC::LocalTensor<float>& dst,
                                       const AscendC::LocalTensor<float>& src) const
    {
        constexpr std::int32_t mask =
            static_cast<std::int32_t>(kEgressLanesPerRepeat<S>);
        constexpr std::uint8_t repeats = static_cast<std::uint8_t>(kEgressRepeats<S>);
        constexpr std::uint8_t stride =
            static_cast<std::uint8_t>(kEgressRepStrideBlocks<S>);
        const AscendC::BinaryRepeatParams params(1, 1, 1, stride, stride, stride);
        AscendC::Add(dst, src, src[S], mask, repeats, params);
        AscendC::Sub(dst[S], src, src[S], mask, repeats, params);
    }

    // A sub-DataBlock stage. `Index` selects the stride's offset and sign
    // vectors out of the concatenated tables.
    template <std::uint32_t Index>
    __aicore__ inline void gather_stage(const AscendC::LocalTensor<float>& dst,
                                        const AscendC::LocalTensor<float>& src) const
    {
        constexpr std::uint32_t table = Index * kEgressChannels;
        AscendC::Gather(permuted, src, gather_offsets[table], 0u, kEgressChannels);
        AscendC::Mul(signed_tmp, src, gather_signs[table], kChannelLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(dst, permuted, signed_tmp, kChannelLanes);
    }

    // The whole rotation, ping-ponging between the two buffers. Eight stages
    // starting in buffer_a leave the result in buffer_b.
    __aicore__ inline void rotate()
    {
        block_stage<32>(buffer_b, buffer_a);
        AscendC::PipeBarrier<PIPE_V>();
        block_stage<64>(buffer_a, buffer_b);
        AscendC::PipeBarrier<PIPE_V>();
        block_stage<128>(buffer_b, buffer_a);
        AscendC::PipeBarrier<PIPE_V>();
        // The sign diagonal, between the group stages and the inner stages.
        AscendC::Mul(buffer_a, buffer_b, pi_sign, kChannelLanes);
        AscendC::PipeBarrier<PIPE_V>();
        gather_stage<0>(buffer_b, buffer_a);  // stride 1
        AscendC::PipeBarrier<PIPE_V>();
        gather_stage<1>(buffer_a, buffer_b);  // stride 2
        AscendC::PipeBarrier<PIPE_V>();
        gather_stage<2>(buffer_b, buffer_a);  // stride 4
        AscendC::PipeBarrier<PIPE_V>();
        block_stage<8>(buffer_a, buffer_b);
        AscendC::PipeBarrier<PIPE_V>();
        block_stage<16>(buffer_b, buffer_a);
        AscendC::PipeBarrier<PIPE_V>();
        // Publish the rotated row before the gate touches it. Adds by 0.0f is
        // an exact FP32 copy for every finite value, signed zeros included.
        AscendC::Adds(rotated_row, buffer_b, 0.0f, kChannelLanes);
    }

    // out = y . sigmoid(gate) = y / (1 + exp(-gate)).
    //
    // Written as the division rather than as a separate sigmoid and a multiply:
    // the identity removes a reciprocal, a vector of ones and a temporary, and
    // the denominator is computed in place in the gate row, which is what
    // "in-place sigmoid(gate)" asks for. One `Exp` (<= 1 ULP, documented) and
    // one 0-ULP `Div`, so the gated output carries about 2 ULP against an FP32
    // host oracle and is not bit-comparable -- unlike the score panel, which
    // is.
    __aicore__ inline void gate_and_cast()
    {
        AscendC::Muls(signed_tmp, gate, -1.0f, kChannelLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(permuted, signed_tmp, kChannelLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(denominator, permuted, 1.0f, kChannelLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div<float, kEgressPreciseDiv>(buffer_a, buffer_b, denominator,
                                               kChannelLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(context_row, buffer_a, AscendC::RoundMode::CAST_RINT,
                      kChannelLanes);
    }
};

// One query's egress. The caller loops over the queries, calling
// `select_query` and refilling `gate` before each one, and owns the MTE2 edge
// that makes `gate` readable and the MTE3 edge that publishes `context`.
template <class Engine>
__aicore__ inline void vector_egress_query(Engine& engine)
{
    engine.normalise();
    AscendC::PipeBarrier<PIPE_V>();
    engine.rotate();
    AscendC::PipeBarrier<PIPE_V>();
    engine.gate_and_cast();
}

#endif  // __CCE_AICORE__

}  // namespace ascend_micro
