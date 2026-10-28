#pragma once

// The score GEMV for single-token paged decode (roadmap kernel No.3).
// Ascend 950PR / arch35 / dav-3510 / CANN 9.2.
//
// What it computes
// ----------------
// One query row against one 16-token key strip:
//
//   R[t] = sum_c Q_rot[c] . K[t][c],   c = 0..255,  t = 0..15
//
// and leaves the 16 results as 16 CONTIGUOUS FP32 lanes, which is exactly the
// panel row `routine_vector_decode_recurrence.hpp` indexes at
// `panel_base + query * 16 * 4`. The per-token key scale is NOT applied here:
// the recurrence's phase 1 multiplies the 16 score columns by it, which is one
// exact multiply per column instead of 4096 over the page, and leaves the raw
// panel an exact small multiple of the query's quantum -- the property that
// makes the panel bit-checkable against a host oracle.
//
// Why the key layout makes this cheap
// -----------------------------------
// `K` is token-major inside the strip (nibble index = token * 256 + channel),
// so a token's 256 channels are contiguous and a per-token reduction is a
// contiguous reduction. This is the second payoff of that layout, alongside
// the Nz(K) = Zn(K^T) identity that serves the Cube in kernel No.4.
//
// Why this is NOT two instructions per row
// ----------------------------------------
// The roadmap sketches the row as
//
//   Mul(prod, key_page_fp32, q_bcast, 64, 64, {1,1,1,8,8,0})
//   ReduceRepeat over 256 contiguous lanes x 16 repeats
//
// and neither line is expressible on this part. Both fail on the same bound,
// from the same page of the reference (`SIMD_compute/mask.md`, "Mask
// Overview"): a repeat is one 256-byte vector register, so for a 32-bit
// operand the continuous-mode mask is `mask in [0, 64]` -- and, critically,
// "if mask exceeds the range, it is calculated as the maximum mask value of
// the corresponding data type". A `ReduceRepeat` asking for 256 lanes is
// therefore NOT a compile error and NOT a runtime fault: it silently reduces
// the first 64 channels of each token and returns a plausible, wrong panel.
// That is the same silent-truncation class as the unlisted
// `Cast(bfloat16_t <- int16_t)` the decode operator lost a cycle to.
//
// The `Mul` fails differently and worse. With 64 repeats over a token-major
// page, repeat r covers token r/4, channels (r%4)*64..+63, so the broadcast
// operand has to supply query channel quarter `r % 4` -- a period-4 wrap.
// `src1RepStride = 0` does not wrap, it stands still, so every repeat would be
// multiplied by query channels 0..63. The repeat-stride trick the value
// contraction uses (`p_x4`, src1RepStride = 0) works there because the value
// page is channel-major with the 16-key axis contiguous, which makes the
// broadcast operand's period exactly 64 lanes. The key page is token-major, so
// its period is 256 and the same trick does not transfer.
//
// What is expressible, and is what this leaf issues, is seven instructions:
//
//   4 x Mul            one per channel quarter j, 16 repeats, src1 standing
//                      still on q[64j..64j+63]. Products land QUARTER-MAJOR:
//                      prod[j*1024 + t*64 + c'].
//   1 x ReduceRepeat   mask 64, 64 repeats -> partial[j*16 + t], the quarter
//                      dot products, 64 contiguous lanes.
//   2 x Add            fold the four quarters: mask 32 then mask 16.
//
// The quarter-major product layout is the whole trick. It is what makes the
// 64 first-level reductions one instruction (the partials are contiguous and
// in repeat order) and what makes the second level two 32-byte-aligned `Add`s
// instead of a strided reduction the API cannot express: summing four lanes at
// stride 4 would need a source repeat stride of half a DataBlock, and summing
// four lanes at stride 16 is just `p + p[32]` then `+ u[16]`, both at 128- and
// 64-byte offsets.
//
// Reduction order does not matter here, and that is a fixture property
// -------------------------------------------------------------------
// `ReduceRepeat<SUM>` accumulates within a repeat as a binary tree, and the
// two folds above impose their own associativity ((j0+j2) + (j1+j3)). Neither
// matches a host oracle's left-to-right loop. The panel is still compared
// BIT-EXACTLY, because on the verification fixture every partial sum is
// exactly representable: Q_rot entries are multiples of 1/16 bounded by 1, the
// nibbles are integers in [-7, 7], so every partial sum is a multiple of 1/16
// bounded by 256 * 7 = 1792, i.e. an integer multiple of 1/16 needing at most
// 15 mantissa bits. Every summation order gives the same FP32 value. On
// production data it does not, and the panel is then accurate rather than
// reproducible -- the bit-exact gate is a property of the fixture, stated here
// so it is not mistaken for a property of the arithmetic.

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#endif

namespace ascend_micro {

// ---------------------------------------------------------------- geometry
constexpr std::uint32_t kGemvKeys = 16;       // tokens per strip
constexpr std::uint32_t kGemvChannels = 256;  // d, one KV head
constexpr std::uint32_t kGemvVecLanes = 64;   // FP32 lanes in one repeat
constexpr std::uint32_t kGemvQuarters = kGemvChannels / kGemvVecLanes;  // 4
constexpr std::uint32_t kGemvLanesPerBlock = 8;  // FP32 per 32-byte DataBlock

// Products, one strip's worth: 16 tokens x 256 channels.
constexpr std::uint32_t kGemvProductElems = kGemvKeys * kGemvChannels;  // 4096
// First-level partials: one per (quarter, token).
constexpr std::uint32_t kGemvPartials = kGemvQuarters * kGemvKeys;  // 64

// The repeat strides, in DataBlocks, that the four `Mul`s need.
//   src0 advances one whole token per repeat
//   dst  advances one whole 64-lane register per repeat
constexpr std::uint32_t kGemvTokenBlocks = kGemvChannels / kGemvLanesPerBlock;    // 32
constexpr std::uint32_t kGemvRegisterBlocks = kGemvVecLanes / kGemvLanesPerBlock;  // 8

static_assert(kGemvChannels % kGemvVecLanes == 0,
              "the channel axis must be a whole number of vector registers");
static_assert(kGemvPartials == kGemvVecLanes,
              "the first-level partials must fill exactly one vector register, "
              "or the two folds below are not the right shape");
// Both folds take a tensor offset, and every vector operand base must be
// 32-byte aligned (rule A1). 32 and 16 FP32 lanes are 128 and 64 bytes.
static_assert((kGemvPartials / 2u) % kGemvLanesPerBlock == 0 &&
                  (kGemvPartials / 4u) % kGemvLanesPerBlock == 0,
              "the quarter folds must land on DataBlock boundaries");
// A repeat stride field is uint8_t.
static_assert(kGemvTokenBlocks <= 255 && kGemvRegisterBlocks <= 255,
              "a repeat stride must fit in uint8_t");

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

struct NativeDecodeGemvEngine {
    // ---- inputs, UB resident --------------------------------------------
    // The dequantised key strip, token-major: key_page[t * 256 + c].
    AscendC::LocalTensor<float> key_page;
    // The rotated query panel, `rows` rows of kGemvChannels, row-major. One
    // row is 1024 bytes, so every row base is 32-byte aligned for any row.
    AscendC::LocalTensor<float> query;

    // ---- scratch ---------------------------------------------------------
    // kGemvProductElems. Written quarter-major, read once.
    AscendC::LocalTensor<float> products;
    // kGemvPartials, plus the 256-byte read pad rule A3 demands behind any
    // operand a level-2 continuous instruction reads (both folds read a whole
    // register from `partials[32]` and `fold[16]`).
    AscendC::LocalTensor<float> partials;
    AscendC::LocalTensor<float> fold;

    // ---- output ----------------------------------------------------------
    // `rows` x kGemvKeys, row-major: the raw panel the recurrence reads.
    AscendC::LocalTensor<float> panel;

    // Base byte addresses the per-row views are derived from.
    std::uint32_t query_base = 0;
    std::uint32_t panel_base = 0;
    AscendC::TPosition position = AscendC::TPosition::VECCALC;

    static constexpr std::int32_t kFullMask = static_cast<std::int32_t>(kGemvVecLanes);
    static constexpr std::int32_t kHalfMask =
        static_cast<std::int32_t>(kGemvPartials / 2u);  // 32
    static constexpr std::int32_t kQuarterMask =
        static_cast<std::int32_t>(kGemvKeys);  // 16

    // ---- the current row's views, set by select_row ----------------------
    AscendC::LocalTensor<float> query_row;  // kGemvChannels
    AscendC::LocalTensor<float> panel_row;  // kGemvKeys

    // Point the two per-row views at row `row`. A runtime index forming a
    // runtime base address reads no Unified Buffer, so it is not the scalar
    // trip the zero-scalar-compute invariant forbids -- the same argument
    // `routine_vector_decode_recurrence.hpp::select_query` makes, and for the
    // same second reason: one body per strip compiles, six unrolled copies at
    // eight strip sites did not.
    __aicore__ inline void select_row(std::uint32_t row)
    {
        query_row = {position, query_base + row * kGemvChannels * 4u, kGemvChannels};
        panel_row = {position, panel_base + row * kGemvKeys * 4u, kGemvKeys};
    }

    // Phase 1. All 4096 products, quarter-major, in four instructions.
    //
    // Quarter j: 16 repeats, repeat t computes
    //   products[j * 1024 + t * 64 + c'] = key_page[t * 256 + j * 64 + c']
    //                                      . query_row[j * 64 + c']
    // with the query operand standing still across the repeats
    // (src1RepStride = 0), so the 256 bytes of that quarter are read once.
    __aicore__ inline void multiply() const
    {
        const AscendC::BinaryRepeatParams params(
            1, 1, 1, static_cast<std::uint8_t>(kGemvRegisterBlocks),
            static_cast<std::uint8_t>(kGemvTokenBlocks), 0);
        for (std::uint32_t j = 0; j < kGemvQuarters; ++j) {
            const std::uint32_t lane = j * kGemvVecLanes;
            AscendC::Mul(products[j * kGemvKeys * kGemvVecLanes], key_page[lane],
                         query_row[lane], kFullMask,
                         static_cast<std::uint8_t>(kGemvKeys), params);
        }
    }

    // Phase 2. The 64 quarter dot products, in one instruction.
    //
    // mask 64 reduces one whole register per repeat and 64 repeats cover the
    // page, so partials[j * 16 + t] is the quarter-j dot product of token t.
    // `dstRepStride` is in units of the dst data type for SUM (reference
    // table 3), so 1 means consecutive FP32 lanes.
    __aicore__ inline void reduce_quarters() const
    {
        AscendC::ReduceRepeat<AscendC::ReduceType::SUM, float>(
            partials, products, kFullMask, static_cast<std::int32_t>(kGemvPartials), 1,
            1, static_cast<std::int32_t>(kGemvRegisterBlocks),
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    }

    // Phase 3. Fold the four quarters into the panel row, in two instructions.
    //
    //   fold[j*16 + t] = partials[j*16 + t] + partials[(j+2)*16 + t]   j = 0,1
    //   panel_row[t]   = fold[t] + fold[16 + t]
    //
    // Both source offsets are whole DataBlocks (128 and 64 bytes). `fold` must
    // not alias `partials`: the first instruction's destination overlaps its
    // own second source otherwise.
    __aicore__ inline void fold_quarters() const
    {
        AscendC::Add(fold, partials, partials[kGemvPartials / 2u], kHalfMask);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(panel_row, fold, fold[kGemvKeys], kQuarterMask);
    }
};

// One query row's score GEMV. The caller has already called `select_row`.
//
// The three phases are separated by intra-pipe barriers and nothing else:
// `PipeBarrier<PIPE_V>` consumes no event id, so this leaf issues no HardEvent
// and takes no flag, which is what lets the operator above it be a straight
// line with no cross-core synchronisation of any kind.
template <class Engine>
__aicore__ inline void vector_decode_gemv_row(Engine& engine)
{
    engine.multiply();
    AscendC::PipeBarrier<PIPE_V>();
    engine.reduce_quarters();
    AscendC::PipeBarrier<PIPE_V>();
    engine.fold_quarters();
}

// The whole panel: `rows` rows against one strip. `rows` is a runtime bound
// (6 for the single-token decode operator, 6 x (1 + k) in general), and the
// loop is a real loop for the reason the recurrence's is.
template <class Engine>
__aicore__ inline void vector_decode_gemv_strip(Engine& engine, std::uint32_t rows)
{
    for (std::uint32_t row = 0; row < rows; ++row) {
        engine.select_row(row);
        vector_decode_gemv_row(engine);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

#endif  // __CCE_AICORE__

}  // namespace ascend_micro
