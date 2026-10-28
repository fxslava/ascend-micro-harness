#pragma once

// The streaming online-softmax recurrence for paged decode attention, widened
// from one query row over 16 channels to 16 query rows over 256 channels.
// Ascend 950PR / arch35 / dav-3510 / CANN 9.2.
//
// Relation to routine_vector_online_softmax.hpp
// ---------------------------------------------
// That leaf carries one query's (m, l, Acc) with Acc 16 channels wide, and its
// replicated-state invariant -- every logically scalar quantity occupying a
// whole 32-byte DataBlock with the value in all 8 lanes -- is what lets the
// whole recurrence run without a single scalar read of Unified Buffer. This
// leaf keeps that invariant exactly and changes two things:
//
//   * the state is an ARRAY of replicated blocks, m_prev[q] at byte offset
//     q * 32. A 32-byte stride keeps every query's state 32-byte aligned, so
//     the per-query operand is a whole DataBlock from an aligned base, which
//     is what rule A1 demands and what a 4-byte-per-query vector would
//     violate;
//   * the weighted value accumulation contracts 16 keys into 256 channels
//     instead of 16, and does it without a Gather. See below.
//
// The two API avoidances that leaf documents carry over unchanged and are not
// re-derived here: `ReduceRepeat` instead of `ReduceSum`/`ReduceMax`, because
// those end in a library-internal scalar readback through a null `TPipe`; and
// `Brcb` instead of `Duplicate` for state broadcast, because `Duplicate`'s
// tensor form lowers to an unmasked 256-byte register load.
//
// The weighted value accumulation, without a Gather
// -------------------------------------------------
// The contraction is
//
//   Acc[q][c] += sum_k P[q][k] . V[c][k]
//
// with V stored channel-major -- [256 channels x 16 keys], key axis contiguous
// -- which is the layout the operator's `value_cache` contract specifies and
// is load-bearing twice over: it makes the per-channel reduction over keys a
// contiguous `ReduceRepeat`, and it makes the 2048-byte strip burst one
// contiguous MTE2 read.
//
// The obvious way to form the elementwise product is to replicate P[q] 256
// times with a `Gather` over a 4096-entry offset vector, which is what the
// 16-channel leaf does at panel scale. At 256 channels that offset vector is
// 16 KiB of Unified Buffer and the replicated P another 16 KiB, for a pattern
// that is simply periodic. The repeat form of `Mul` expresses it directly: a
// repeat of 64 FP32 lanes covers exactly four channels' worth of 16 keys, so
// with P[q] pre-replicated into ONE 64-lane block and `src1RepStride = 0`,
//
//   Mul(weighted, value_page, p_x4, 64, 64, {1, 1, 1, 8, 8, 0})
//
// computes all 4096 products in one instruction, re-reading the same 256 bytes
// of P on every repeat. That removes 32 KiB of Unified Buffer and the
// `Gather`'s offset ingress, and it is why `p_x4` exists.
//
// Scales live outside this leaf
// -----------------------------
// Neither the key's nor the value's per-token scale is applied to a 4096-entry
// page. The key scale multiplies the score panel's 16 columns, which is exact
// because the scales are powers of two and which keeps the Cube's key operand
// an exact small integer -- the property that makes the score panel
// bit-checkable. The value scale folds into P before the contraction, 16
// multiplies instead of 4096, and the denominator still sees the unscaled P.
// The caller supplies both as 16-lane vectors.

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#endif

namespace ascend_micro {

constexpr std::uint32_t kDecodeKeys = 16;        // keys per strip
constexpr std::uint32_t kDecodeChannels = 256;   // head size
constexpr std::uint32_t kDecodeStateLanes = 8;   // one FP32 DataBlock
constexpr std::uint32_t kDecodeBcLanes = 64;     // one Brcb repeat, 256 bytes
constexpr std::uint32_t kDecodeVecLanes = 64;    // FP32 lanes in one repeat

// FP32 elements in one DataBlock, and the per-channel reduction's shape.
constexpr std::uint32_t kDecodeLanesPerBlock = 8;
constexpr std::uint32_t kDecodeKeyBlocks = kDecodeKeys / kDecodeLanesPerBlock;  // 2

// `ReduceRepeat`'s repeatTime is bounded by 255, and the per-channel reduction
// needs 256 repeats, so it runs as two halves.
constexpr std::uint32_t kDecodeReduceHalf = kDecodeChannels / 2;  // 128

static_assert(kDecodeVecLanes % kDecodeKeys == 0,
              "a 64-lane repeat must cover a whole number of 16-key channels, "
              "or the src1RepStride = 0 broadcast of p_x4 does not line up");

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

struct NativeDecodeRecurrenceEngine {
    // ---- per-strip inputs, all UB resident -------------------------------
    AscendC::LocalTensor<float> panel;        // 16 x 16 ND, the Cube's output
    AscendC::LocalTensor<float> key_scale;    // kDecodeKeys
    AscendC::LocalTensor<float> value_scale;  // kDecodeKeys
    AscendC::LocalTensor<float> value_page;   // kDecodeChannels * kDecodeKeys

    // ---- carried state, one replicated 32-byte block per query ----------
    AscendC::LocalTensor<float> m_prev;  // kTokens * kDecodeStateLanes
    AscendC::LocalTensor<float> l_prev;  // kTokens * kDecodeStateLanes
    AscendC::LocalTensor<float> acc;     // kTokens * kDecodeChannels

    // ---- per-query scratch, reused across queries ------------------------
    AscendC::LocalTensor<float> scaled_row;    // kDecodeKeys
    AscendC::LocalTensor<float> centered;      // kDecodeKeys
    AscendC::LocalTensor<float> p;             // kDecodeKeys
    AscendC::LocalTensor<float> p_scaled;      // kDecodeKeys
    AscendC::LocalTensor<float> p_x4;          // kDecodeVecLanes
    AscendC::LocalTensor<float> m_new;         // kDecodeStateLanes
    AscendC::LocalTensor<float> l_new;         // kDecodeStateLanes
    AscendC::LocalTensor<float> alpha;         // kDecodeStateLanes
    AscendC::LocalTensor<float> alpha_delta;   // kDecodeStateLanes
    AscendC::LocalTensor<float> l_scaled;      // kDecodeStateLanes
    AscendC::LocalTensor<float> tile_max_raw;  // kDecodeStateLanes, lane 0 only
    AscendC::LocalTensor<float> p_sum_raw;     // kDecodeStateLanes, lane 0 only
    AscendC::LocalTensor<float> tile_max_bc;   // kDecodeBcLanes
    AscendC::LocalTensor<float> m_bc;          // kDecodeBcLanes
    AscendC::LocalTensor<float> p_sum_bc;      // kDecodeBcLanes
    AscendC::LocalTensor<float> alpha_bc;      // kDecodeBcLanes
    AscendC::LocalTensor<float> weighted;      // kDecodeChannels * kDecodeKeys
    AscendC::LocalTensor<float> weighted_sum;  // kDecodeChannels
    AscendC::LocalTensor<float> acc_scaled;    // kDecodeChannels

    static constexpr std::int32_t kKeyLanes = static_cast<std::int32_t>(kDecodeKeys);
    static constexpr std::int32_t kKeyBlocks =
        static_cast<std::int32_t>(kDecodeKeyBlocks);
    static constexpr std::int32_t kStateLanes =
        static_cast<std::int32_t>(kDecodeStateLanes);
    static constexpr std::int32_t kChannelLanes =
        static_cast<std::int32_t>(kDecodeChannels);
    static constexpr std::int32_t kVecLanes =
        static_cast<std::int32_t>(kDecodeVecLanes);

    // ---- the current query's views, set by select_query ------------------
    AscendC::LocalTensor<float> panel_row;  // kDecodeKeys
    AscendC::LocalTensor<float> m_cur;      // kDecodeStateLanes
    AscendC::LocalTensor<float> l_cur;      // kDecodeStateLanes
    AscendC::LocalTensor<float> acc_cur;    // kDecodeChannels

    // Base byte addresses the views are derived from.
    std::uint32_t panel_base = 0;
    std::uint32_t m_base = 0;
    std::uint32_t l_base = 0;
    std::uint32_t acc_base = 0;
    AscendC::TPosition position = AscendC::TPosition::VECCALC;

    // Point the four per-query views at query `query`.
    //
    // The query index is a RUNTIME value and these are runtime address
    // computations. That is not the scalar trip the zero-scalar-compute
    // invariant forbids: section 1.2 prohibits reading a UB *value* into the
    // scalar unit (`GetValue`, a predicate on UB data, an index derived from
    // computed UB data), because such a read needs a V_S fence and a TPipe.
    // Forming a base address from a loop counter is exactly what the scalar
    // unit is for -- section 1.1 calls it the instruction dispatcher -- and it
    // reads no Unified Buffer at all.
    //
    // It is also what keeps this kernel compilable. Unrolling the 16 queries
    // at every one of the 8 strip sites instantiated 128 copies of the body
    // below, and the AIV translation unit did not finish compiling in 25
    // minutes. One body per strip compiles in a fraction of that and emits the
    // same instruction stream per query.
    //
    // Every offset stays 32-byte aligned for any query: a panel row is 64
    // bytes, a state block 32, an accumulator row 1024.
    __aicore__ inline void select_query(std::uint32_t query)
    {
        panel_row = {position, panel_base + query * kDecodeKeys * 4u, kDecodeKeys};
        m_cur = {position, m_base + query * kDecodeStateLanes * 4u, kDecodeStateLanes};
        l_cur = {position, l_base + query * kDecodeStateLanes * 4u, kDecodeStateLanes};
        acc_cur = {position, acc_base + query * kDecodeChannels * 4u, kDecodeChannels};
    }

    // ---------------------------------------------------------------- phases
    // Phase 1. Apply the key's per-token scale to the panel row, take the row
    // maximum, and fold it into the carried maximum.
    __aicore__ inline void update_max_and_alpha()
    {
        AscendC::Mul(scaled_row, panel_row, key_scale, kKeyLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceRepeat<AscendC::ReduceType::MAX, float>(
            tile_max_raw, scaled_row, kKeyLanes, 1, 1, 1, kKeyBlocks,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        // A reduction writes lane 0 only and Brcb consumes a whole DataBlock,
        // so the two need intra-pipe ordering. PipeBarrier<PIPE_V> is intra
        // pipe and consumes no event id, so it does not breach leaf purity.
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(tile_max_bc, tile_max_raw, 1, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(m_new, m_cur, tile_max_bc, kStateLanes);
        AscendC::Sub(alpha_delta, m_cur, m_new, kStateLanes);
        AscendC::Exp(alpha, alpha_delta, kStateLanes);
    }

    // Phase 2. Centre the row on the new maximum and exponentiate.
    __aicore__ inline void exponentiate_scores()
    {
        AscendC::Brcb(m_bc, m_new, 1, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(centered, scaled_row, m_bc, kKeyLanes);
        AscendC::Exp(p, centered, kKeyLanes);
    }

    // Phase 3. The denominator, rescaled by alpha.
    __aicore__ inline void update_denominator()
    {
        AscendC::ReduceRepeat<AscendC::ReduceType::SUM, float>(
            p_sum_raw, p, kKeyLanes, 1, 1, 1, kKeyBlocks,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(p_sum_bc, p_sum_raw, 1, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(l_scaled, l_cur, alpha, kStateLanes);
        AscendC::Add(l_new, l_scaled, p_sum_bc, kStateLanes);
    }

    // Phase 4. Fold the value scale into P, replicate P into one 64-lane
    // block, form all 4096 products in one repeat-form Mul, reduce over the
    // 16 contiguous keys of each channel, and rescale the accumulator.
    __aicore__ inline void update_accumulator()
    {
        AscendC::Mul(p_scaled, p, value_scale, kKeyLanes);
        AscendC::PipeBarrier<PIPE_V>();
        // p_x4[j] = p_scaled[j % 16] for j < 64: four repeats of 16 lanes,
        // destination advancing two DataBlocks, source standing still.
        AscendC::Adds(p_x4, p_scaled, 0.0f, kKeyLanes, 4,
                      AscendC::UnaryRepeatParams(1, 1, kKeyBlocks, 0));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(weighted, value_page, p_x4, kVecLanes,
                     static_cast<std::uint8_t>(kDecodeChannels * kDecodeKeys /
                                               kDecodeVecLanes),
                     AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 0));
        AscendC::PipeBarrier<PIPE_V>();
        // One FP32 result per channel, from 16 contiguous lanes. 256 repeats
        // exceeds the repeatTime bound of 255, so this runs as two halves.
        AscendC::ReduceRepeat<AscendC::ReduceType::SUM, float>(
            weighted_sum, weighted, kKeyLanes,
            static_cast<std::int32_t>(kDecodeReduceHalf), 1, 1, kKeyBlocks,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::ReduceRepeat<AscendC::ReduceType::SUM, float>(
            weighted_sum[kDecodeReduceHalf], weighted[kDecodeReduceHalf * kDecodeKeys],
            kKeyLanes, static_cast<std::int32_t>(kDecodeReduceHalf), 1, 1, kKeyBlocks,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::Brcb(alpha_bc, alpha, 1, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(acc_scaled, acc_cur, alpha_bc, kVecLanes,
                     static_cast<std::uint8_t>(kDecodeChannels / kDecodeVecLanes),
                     AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 0));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(acc_cur, acc_scaled, weighted_sum, kChannelLanes);
    }

    // Phase 5. Commit the carried state. An Adds by 0.0f is an exact FP32 copy
    // for every finite value, signed zeros included.
    __aicore__ inline void commit_state()
    {
        AscendC::Adds(m_cur, m_new, 0.0f, kStateLanes);
        AscendC::Adds(l_cur, l_new, 0.0f, kStateLanes);
    }
};

// One query's whole recurrence. The caller loops over the 16 queries, calling
// `select_query` before each one.
template <class Engine>
__aicore__ inline void vector_decode_recur_query(Engine& engine)
{
    engine.update_max_and_alpha();
    engine.exponentiate_scores();
    engine.update_denominator();
    engine.update_accumulator();
    engine.commit_state();
}

// The whole strip: all kDecodeQueries queries in order.
constexpr std::uint32_t kDecodeQueries = 16;

template <class Engine>
__aicore__ inline void vector_decode_recur_strip(Engine& engine)
{
    for (std::uint32_t query = 0; query < kDecodeQueries; ++query) {
        engine.select_query(query);
        vector_decode_recur_query(engine);
    }
}

#endif  // __CCE_AICORE__

}  // namespace ascend_micro
