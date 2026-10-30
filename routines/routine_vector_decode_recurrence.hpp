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

    // ---- tail-mask scratch, for a partial strip --------------------------
    // Four operands, primed once per kernel by `prime_tail_mask` and then read
    // without being rewritten. The three float buffers each need a 256-byte
    // WRITE pad, not 64 bytes: `CreateVecIndex` and `Duplicate` both write a
    // whole vector register regardless of the lane count, which is the same
    // rule this header already records for `Brcb`.
    AscendC::LocalTensor<float> lane_index;     // kDecodeKeys: 0, 1, ... 15
    AscendC::LocalTensor<float> masked_fill;    // kDecodeKeys of kMaskedScore
    AscendC::LocalTensor<float> zero_fill;      // kDecodeKeys of +0.0f
    AscendC::LocalTensor<std::uint8_t> tail_sel;  // one bit per key lane

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
    //
    // Split into `scale_row` and `reduce_max_and_alpha` so that a PARTIAL
    // strip can mask the score row between the two. Both halves are
    // `__aicore__ inline` and are called back to back below, so the aligned
    // path issues the same instructions in the same order as before the split
    // -- which matters because `vector_decode_recur_strip`, the Cube decode
    // operator's frozen entry point, goes through here unchanged.
    __aicore__ inline void scale_row()
    {
        AscendC::Mul(scaled_row, panel_row, key_scale, kKeyLanes);
    }

    // Primed once per kernel, before the first strip. Reads nothing that
    // changes, so it is hoisted out of the strip loop entirely.
    //
    // `lane_index` is the key-lane ramp the predicate compares against;
    // `masked_fill` and `zero_fill` are the two constants the selects pull
    // from. All three are built by APIs that take a plain element count, which
    // is the form that does not touch the hardware mask registers.
    __aicore__ inline void prime_tail_mask(float masked_score)
    {
        AscendC::CreateVecIndex(lane_index, 0.0f, kDecodeKeys);
        AscendC::Duplicate(masked_fill, masked_score, kKeyLanes);
        AscendC::Duplicate(zero_fill, 0.0f, kKeyLanes);
    }

    // Phase 1m-a. The predicate for this strip: lane < active, one bit per key
    // lane. Computed ONCE per strip, in the strip-level entry point, because
    // all 16 queries of a strip share the same live lane count.
    //
    // `active` reaches the vector unit as a scalar operand of `Compares`, not
    // as a mask register and not through Unified Buffer, so this is the same
    // permitted kind of scalar work as `select_query`'s address arithmetic.
    __aicore__ inline void select_tail(std::uint32_t active)
    {
        AscendC::Compares(tail_sel, lane_index, static_cast<float>(active),
                          AscendC::CMPMODE::LT, kDecodeKeys);
    }

    // Phase 1m-0. Replace the dead lanes of BOTH per-token scales with +0,
    // before either multiply reads them.
    //
    // Without this the kernel is numerically correct but still feeds NaN into
    // two F32 vector multiplies per partial strip, which the CAModel reports
    // as `vec_err_idata_inf_nan_t0` -- 306 of them in the S=1/S=17 run that
    // otherwise passed. The results were discarded by the Selects that follow,
    // so the context came out clean, but an operator that hands NaN to the
    // vector unit and relies on overwriting the answer is one assertion away
    // from a failure, and it makes the model's error log useless for spotting
    // a real hazard.
    //
    // Masking in place is safe: the recurrence is the last reader of both
    // scale vectors in this ring slot, and the slot's MTE2 refill edge is not
    // released until after it.
    //
    // Zero is the right fill for both. The key scale then makes the dead
    // score lane exactly 0 -- which is why `mask_tail_scores` below is still
    // needed to push it to `masked_score` -- and the value scale makes the
    // dead probability lane exactly 0 before the contraction.
    __aicore__ inline void mask_tail_scales()
    {
        AscendC::Select(key_scale, tail_sel, key_scale, zero_fill,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, kDecodeKeys);
        AscendC::Select(value_scale, tail_sel, value_scale, zero_fill,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, kDecodeKeys);
    }

    // Phase 1m-b. Force the dead score lanes to `masked_score` by SELECTION.
    //
    // This replaces a bit-masked `Duplicate`, and the replacement is the whole
    // point of this revision. The masked write rested on an unstated property
    // -- that a vector instruction leaves its deselected destination lanes
    // untouched -- and a physical 950PR run disproved it: at S = 1 the context
    // came back non-finite, the canary having survived into the softmax. The
    // CANN 9.2 dav_3510 implementation shows why it was never a safe bet; the
    // bit-mask form allocates a temporary UB buffer and issues its own V_S
    // fence rather than being the single gated instruction it looks like.
    //
    // A `Select` writes EVERY lane, choosing per lane between two sources from
    // a predicate. There is no "preserved" lane to depend on, so the canary
    // cannot survive by omission. It is also not arithmetic: the dead lane's
    // NaN is discarded rather than multiplied, which a keep-mask could not do.
    //
    // REQUIRES `select_tail` to have run for this strip, and `0 < active < 16`.
    // A fully live strip takes the unmasked path; an entirely dead strip must
    // not be walked at all -- with every lane masked `m_new` would be
    // `masked_score`, the centred row exactly 0, `exp(0)` exactly 1, and the
    // strip would add 16 to l instead of 0.
    __aicore__ inline void mask_tail_scores()
    {
        AscendC::Select(scaled_row, tail_sel, scaled_row, masked_fill,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, kDecodeKeys);
    }

    // Phase 4m. Clamp the dead probability lanes to exactly +0, before the
    // value contraction reads them.
    //
    // This is the SECOND NaN path, and the one masking the scores does not
    // close. `p_scaled = p * value_scale` reads the per-token VALUE scale, and
    // a never-written cache slot holds the 0x7FC0BEEF canary there too. The
    // score mask makes `p` exactly 0 on a dead lane, but `0 * NaN` is NaN, so
    // the product was poisoned and the NaN then spread over all 256 channels
    // through `p_x4` and the repeat-form `Mul`. The denominator l was fine --
    // it sums `p`, not `p_scaled` -- which is exactly why this leaked as a
    // non-finite CONTEXT with a plausible l.
    //
    // Selecting +0 here means the contraction multiplies the value page by a
    // true zero, so an uninitialised V line cannot interact with it at all.
    __aicore__ inline void clamp_tail_probabilities()
    {
        AscendC::Select(p_scaled, tail_sel, p_scaled, zero_fill,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, kDecodeKeys);
    }

    __aicore__ inline void reduce_max_and_alpha()
    {
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

    __aicore__ inline void update_max_and_alpha()
    {
        scale_row();
        AscendC::PipeBarrier<PIPE_V>();
        reduce_max_and_alpha();
    }

    // The same phase for a partial strip. The predicate is already built, so
    // this is one extra instruction and one extra barrier over the aligned
    // path, on the one strip that needs them.
    __aicore__ inline void update_max_and_alpha_masked()
    {
        // `mask_tail_scales` has already run for this strip, in the entry
        // point, so the scales reaching `scale_row` are finite.
        scale_row();
        AscendC::PipeBarrier<PIPE_V>();
        mask_tail_scores();
        AscendC::PipeBarrier<PIPE_V>();
        reduce_max_and_alpha();
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
    // Split into `fold_value_scale` and `contract_values` so that a partial
    // strip can clamp the probabilities between the two. Both halves are
    // `__aicore__ inline` and called back to back below, so the aligned path
    // issues the same instructions in the same order as before the split.
    __aicore__ inline void fold_value_scale()
    {
        AscendC::Mul(p_scaled, p, value_scale, kKeyLanes);
    }

    __aicore__ inline void contract_values()
    {
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

    __aicore__ inline void update_accumulator()
    {
        fold_value_scale();
        AscendC::PipeBarrier<PIPE_V>();
        contract_values();
    }

    // The same phase for a partial strip, with the dead probability lanes
    // clamped to +0 before anything reads them.
    __aicore__ inline void update_accumulator_masked()
    {
        fold_value_scale();
        AscendC::PipeBarrier<PIPE_V>();
        clamp_tail_probabilities();
        AscendC::PipeBarrier<PIPE_V>();
        contract_values();
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

// The whole strip at a RUNTIME row count (roadmap D1). Added for the
// single-token decode operator, which carries 6 x (1 + k) rows per
// (request, kv_head) and is 6 rows at k = 0.
//
// Additive on purpose. `vector_decode_recur_strip` above is left exactly as
// the Cube decode operator calls it -- same symbol, same compile-time bound,
// same emitted instruction stream -- because that operator and its verified
// eight-strip ladder are frozen. D1 will later rewrite the bound in place and
// retire this entry point; until it does, the two coexist and the 16-row path
// is not touched by anything the 6-row path needs.
//
// Only the bound differs. Every phase below already addresses its operands
// through `select_query`'s runtime base addresses, so nothing else in the
// engine is row-count dependent -- but the CALLER's buffers are: `m_prev`,
// `l_prev` and `acc` must hold `rows` entries, and `Brcb`/`Duplicate` read a
// whole 256-byte register from their source base regardless of the lane count,
// so the caller still owns the read pad behind the last row's state block.
template <class Engine>
__aicore__ inline void vector_decode_recur_rows(Engine& engine, std::uint32_t rows)
{
    for (std::uint32_t query = 0; query < rows; ++query) {
        engine.select_query(query);
        vector_decode_recur_query(engine);
    }
}

// One query's recurrence over a PARTIAL strip: the first `active` key lanes
// are live, the rest are masked out of the softmax entirely.
//
// TWO phases differ, not one, and the second was learned the hard way. Phase 1
// forces the dead score lanes to `masked_score`, so `exp` underflows them to
// +0 and the denominator adds nothing. Phase 4 then clamps the dead
// PROBABILITY lanes to +0, because `p_scaled = p * value_scale` reads the
// value scale of a dead slot -- a 0x7FC0BEEF canary -- and `0 * NaN` is NaN.
// Masking the scores alone left that path open, and a physical 950PR run at
// S = 1 found it as a non-finite context.
//
// Phases 2, 3 and 5 are the aligned ones, untouched. The caller must have run
// `select_tail(active)` for this strip.
template <class Engine>
__aicore__ inline void vector_decode_recur_query_masked(Engine& engine)
{
    engine.update_max_and_alpha_masked();
    engine.exponentiate_scores();
    engine.update_denominator();
    engine.update_accumulator_masked();
    engine.commit_state();
}

// The partial strip at a RUNTIME row count: the single-token decode
// operators' tail strip. `active` must be in 1..15; a full strip takes
// `vector_decode_recur_rows` and a dead strip is not walked.
//
// The predicate is built once here rather than per query: every query of a
// strip shares the same live lane count, so 16 `Compares` would compute the
// same 16 bits sixteen times.
template <class Engine>
__aicore__ inline void vector_decode_recur_rows_masked(Engine& engine,
                                                       std::uint32_t rows,
                                                       std::uint32_t active)
{
    engine.select_tail(active);
    AscendC::PipeBarrier<PIPE_V>();
    // Per strip, not per query: the two scale vectors are strip inputs shared
    // by every query, so masking them inside the query loop would redo the
    // same two Selects `rows` times -- and the second pass would be masking
    // already-masked data.
    engine.mask_tail_scales();
    AscendC::PipeBarrier<PIPE_V>();
    for (std::uint32_t query = 0; query < rows; ++query) {
        engine.select_query(query);
        vector_decode_recur_query_masked(engine);
    }
}

// The partial strip at the frozen 16-query bound, for the Cube decode
// operator. Added so that the tail fix is reachable from that operator without
// changing its aligned path; wiring its strip loop and launcher to a `seq_len`
// is a separate change and has NOT been made.
template <class Engine>
__aicore__ inline void vector_decode_recur_strip_masked(Engine& engine,
                                                        std::uint32_t active)
{
    engine.select_tail(active);
    AscendC::PipeBarrier<PIPE_V>();
    engine.mask_tail_scales();
    AscendC::PipeBarrier<PIPE_V>();
    for (std::uint32_t query = 0; query < kDecodeQueries; ++query) {
        engine.select_query(query);
        vector_decode_recur_query_masked(engine);
    }
}

#endif  // __CCE_AICORE__

}  // namespace ascend_micro
