#pragma once

// The Split-K final reduction as a Cube GEMV: the layout, flag and budget contract.
//
// Dependency-free on purpose (§6.1). The device kernel and the host gate include THIS
// declaration rather than each carrying a copy, so the host compile of
// tests/cpu/test_gemm_reduce_geometry.cpp is evidence about the kernel's arithmetic and
// not about a transcription of it.
//
// WHAT THE STAGE COMPUTES
//
// TurboQuantPartialReducer (fused_decode/vector/turboquant_vector_service.h) merges S split
// partials for one (token, head) with a sequential online-softmax rescale. Each split s wrote
// an unnormalised accumulator o_s[0..D) plus its own statistics m_s (running max) and
// l_s (sum of exp(x - m_s)) into the workspace row
//
//     workspace[((token * numHeads + head) * numSplits + s) * kPartialStride]
//
// whose first D floats are o_s and whose tail carries m_s at lane kPartialMaxLane and l_s at
// lane kPartialSumLane. The merged head is
//
//     m*     = max_s m_s
//     w_s    = exp(m_s - m*)
//     out[d] = (sum_s w_s * o_s[d]) / (sum_s w_s * l_s)
//
// so the NUMERATOR is exactly an M=1, K=S, N=D matmul of the [1,S] weight row against the
// [S,D] partial matrix, and the denominator is a [1,S]x[S,1] dot the vector unit does in one
// 8-lane multiply plus a reduce. Note w_s = exp(m_s - m*) and NOT exp(m_s - m*) * l_s: the
// partials this operator stores are unnormalised, so l_s belongs in the denominator only.
//
// This is algebraically the same value as the Vector reducer's two-term recurrence, but it is
// NOT the same floating-point expression: the recurrence rescales the running accumulator S-1
// times and sums in split order, the GEMV sums S products in the Cube's own order inside one
// fp32 accumulator. The difference is measured, not asserted -- see the report.
//
// WHY THE CUBE IS FREE TO DO IT
//
// In turboquant_fused_decode_opt.cpp the reduction phase is
//
//     op.Process(tasksPerBlock);
//     if (op.NeedsReduction()) { AscendC::SyncAll<false>(); if ASCEND_IS_AIV { ... Reduce ... } }
//
// The reduction is guarded by `if ASCEND_IS_AIV`, so the Cube is idle for the whole phase
// while the busier engine does every rescale. That idleness, not a FLOP shortage, is what
// this stage spends.
//
// WHAT IT DOES NOT DO
//
// It does not overlap the reduction with the attention phase. Token t's combine needs every
// split of token t complete, which is a grid-wide dependency that the SyncAll above is
// discharging; removing it needs a per-token completion counter in L2, which is a different
// probe. The overlap this stage buys is WITHIN the reduction phase: the Cube's GEMV for task r
// runs against the vector unit's statistics merge for task r+2 and its normalise of r-2, and
// against the AIC's own MTE2 fetch of task r+1's partials.

#include <cstdint>

// The helpers below are compiled into BOTH the kernel and the host gate, which is the whole
// point of �6.1: the two share this declaration instead of each carrying a copy. BiSheng treats
// a plain function as __host__ only, so a device caller needs the attribute, and the host gate
// must not see it.
#if defined(__CCE_AICORE__)
#define ASCEND_MICRO_REDUCE_FN __aicore__ inline constexpr
#else
#define ASCEND_MICRO_REDUCE_FN constexpr
#endif

namespace ascend_micro::gemm_reduce {

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

// A Cube C0 block is 32 bytes on every DaVinci generation, so a b32 operand carries 8 elements
// along K per block and the b32 fractal is 16 x 8 (512 bytes). This is the whole reason the
// K = S operand is viable: the decode's split count is capped at kMaxSequenceSplits = 8
// (fused_decode/common/turboquant_layout.h), so K = S <= 8 lands inside ONE C0 block and the
// GEMV needs no K tiling and no multi-block zero padding at all. At S = 8 it is exact.
constexpr std::uint32_t kC0Bytes = 32;
constexpr std::uint32_t kFloatBytes = 4;
constexpr std::uint32_t kC0Floats = kC0Bytes / kFloatBytes;  // 8
constexpr std::uint32_t kFractalRows = 16;
constexpr std::uint32_t kFractalBytes = kFractalRows * kC0Bytes;  // 512

// The operator's real geometry, not an illustrative shape.
constexpr std::uint32_t kHeadSize = 256;
constexpr std::uint32_t kMaxSplits = 8;
constexpr std::uint32_t kPartialTail = 16;  // kPartialTail = 2 * kFp32PerBlock
constexpr std::uint32_t kPartialMaxLane = 0;
constexpr std::uint32_t kPartialSumLane = 8;
constexpr std::uint32_t kPartialStride = kHeadSize + kPartialTail;  // 272 floats

// Mmad's K direction for float aligns to ceil(K/8)*8 on Ascend 950PR -- kDirectionAlign
// supports only `false` there, which is that rule. So a split count below kC0Floats leaves
// K lanes S..7 live in the hardware's eyes.
ASCEND_MICRO_REDUCE_FN std::uint32_t kAlignedSplits(std::uint32_t splits)
{
    return (splits + kC0Floats - 1) / kC0Floats * kC0Floats;
}
static_assert(kAlignedSplits(1) == 8 && kAlignedSplits(8) == 8);
// Every legal split count is one C0 block: no K tiling, one Mmad.
static_assert(kAlignedSplits(kMaxSplits) == kC0Floats);

// Those padding lanes are a CORRECTNESS hazard and not merely waste, because §3 floods every
// GM payload with 0x7FC0BEEF (an fp32 quiet NaN) and 0 * NaN is NaN, not 0. Two independent
// guards, both exercised by the S < 8 rungs of the ladder:
//   * the weight row's lanes S..7 are zeroed, so a padding partial cannot reach the sum, and
//   * DataCopy's DN2NZ conversion zero-fills the destination when dValue * sizeof(T) is not
//     32-byte aligned, which is exactly the S < 8 case.
// Relying on either alone would be relying on an undocumented half of the pair.
constexpr bool kZeroPadWeightLanes = true;

// ---------------------------------------------------------------------------------------------
// The B operand's road into L1
// ---------------------------------------------------------------------------------------------
//
// Mmad wants B as K x N in Zn in L0B. The workspace already holds [S, D] row-major, which IS
// K x N, so the only question is the fractal layout. Two facts settle it without a transposing
// load:
//
//   * DataCopy GM->L1 with Dn2NzParams performs an inline DN->NZ conversion, which the API
//     reference states is "equivalent to the ND-to-ZN format conversion". Feeding the ND
//     [S, D] rows therefore lands the ZN image of that same K x N matrix in L1, with the S
//     axis as C0. It is an Ascend 950PR/950DT-only path and it supports float.
//   * Load2DV2 with ifTranspose = false leaves both the intra- and inter-fractal layout alone
//     (Zn2Zn), so the L0B operand is the Zn cube Mmad asked for.
//
// So the reduction's B operand takes the SAME non-transposed load the score GEMM's N x K key
// tile takes (TurboQuantCubeMm::StageRightToL0bFromNk), and none of the transposed-load
// constraints apply to it. That matters: for b32 WITH transpose, Load2DV2 requires kStep to be
// a multiple of 2, and the b32 transpose merges two consecutive 16x8 fractals into a 16x16
// square, which demands a source Zn cube with Col == 16 -- a condition a K = S <= 8 operand
// cannot meet. Avoiding the transpose avoids all of it.
//
// The resulting L1 image is [N rows][C0 floats] contiguous.
constexpr std::uint32_t kPartialL1Floats = kHeadSize * kC0Floats;              // 2048
constexpr std::uint32_t kPartialL1Bytes = kPartialL1Floats * kFloatBytes;      // 8192
static_assert(kPartialL1Bytes == kHeadSize * kC0Bytes);
static_assert(kPartialL1Bytes % kFractalBytes == 0, "Load2DV2 moves whole 512-byte fractals");
// Exactly the payload: 8 splits x 256 floats, nothing padded away at the maximum split count.
static_assert(kPartialL1Bytes == kMaxSplits * kHeadSize * kFloatBytes);

// Dn2NzParams for one (token, head): dValue is the C0 axis (the splits), nValue the N axis,
// srcDValue the source row stride, which is kPartialStride and not kHeadSize because the
// statistics tail shares the row.
// This toolkit's Dn2NzParams names the per-matrix strides srcDnMatrixStride and
// dstNzMatrixStride, not the API reference's ...CubeStride; both are 0 for a single matrix.
constexpr std::uint32_t kDn2NzDnNum = 1;
constexpr std::uint32_t kDn2NzNValue = kHeadSize;
constexpr std::uint32_t kDn2NzSrcDValue = kPartialStride;
constexpr std::uint32_t kDn2NzDstNzC0Stride = kHeadSize;
constexpr std::uint32_t kDn2NzDstNzNStride = 1;
static_assert(kDn2NzNValue <= 16384 && kDn2NzDstNzC0Stride <= 16384 && kDn2NzSrcDValue >= 1,
              "Dn2NzParams value ranges");

// The A operand. GEMV requires the 1 x K row to be 512-byte aligned with K contiguous
// elements, so the weight slot is a whole fractal even though only S floats of it are read.
constexpr std::uint32_t kWeightL1Bytes = kFractalBytes;  // 512
static_assert(kWeightL1Bytes >= kMaxSplits * kFloatBytes);
static_assert(kWeightL1Bytes % 512 == 0, "GEMV's 1xK left cube needs a 512-byte base");

// The C operand. M = 1 is GEMV's own case: Mmad enables it automatically, and Ascend 950PR is
// the one product that can switch it off with MmadParams::disableGemv. The prototype carries
// both legs because they land DIFFERENT cubes in L0C and the Fixpipe has to match:
//   * GEMV on  -- C is 1 x N split into N/16 sub-vectors, each occupying 1024 bytes of L0C of
//     which only the low 64 bytes are valid.
//   * GEMV off -- C is a conventional M-unaligned cube, M padded to kFractalRows, which is the
//     shape the production Fixpipe path in this operator already emits.
// kGemvDisabled is the default precisely because it reuses the audited Fixpipe parameters; the
// enabled leg is a measurement variant.
constexpr bool kGemvDisabledDefault = true;
constexpr std::uint32_t kProductL0cBytes = kFractalRows * kHeadSize * kFloatBytes;  // 16384
constexpr std::uint32_t kGemvL0cBytes = (kHeadSize / kFractalRows) * 1024;           // 16384
static_assert(kProductL0cBytes == kGemvL0cBytes,
              "both legs reserve the same L0C, so the slot budget does not depend on the knob");

// ---------------------------------------------------------------------------------------------
// The stage's ring and its flags
// ---------------------------------------------------------------------------------------------

// Two slots: one the Cube is multiplying while the vector unit prepares the next task's
// weights and retires the previous task's output. A third buys nothing -- the Cube's GEMV is
// one Mmad and the stage is bound by the GM fetch, not by Cube occupancy.
constexpr std::uint32_t kReduceSlots = 2;

// One AIV->AIC place and one AIC->AIV place per slot, and no more, because each doubles as the
// other direction's free credit:
//
//   weights_ready[s]  AIV -> AIC, initially 0, primed by the prologue.
//                     Posted by the AIV's MTE3 store of the weight row into L1 slot s, which
//                     the AIV reaches only AFTER its Finish of the task two earlier read UB
//                     product slot s. So this one token says both "the weights are in L1" and
//                     "product slot s is yours again".
//   product_ready[s]  AIC -> AIV, initially 0.
//                     Posted by the AIC's Fixpipe into UB product slot s. Consumed by the
//                     AIV's Finish, after which the same AIV step is what frees the weights.
//
// The reuse cycle weights_ready[s] -> product_ready[s] -> weights_ready[s] is primed by the
// prologue's unconditional sets for the first min(R, kReduceSlots) tasks, so it is marked
// before it is ever waited on, and both places drain to zero (§1).
constexpr std::uint32_t kReducePlacesPerSlot = 2;
constexpr std::uint32_t kReduceFlagCount = kReduceSlots * kReducePlacesPerSlot;  // 4

// WHERE THOSE IDS COME FROM, which is the stage's sharpest constraint.
//
// A mode-4 flag id is four bits and the toolkit reserves 11..14, leaving 0..10. The attention
// schedule already spends 0..9 (3 slot-ready + 3 slot-free + 2 scores + context + probs) and
// kTaskJoinFlag is 10. There is NO free id. The reduction stage therefore RECYCLES ids, which
// is legal here only because AscendC::SyncAll<false>() and the attention phase's own epilogue
// stand between the two phases and every attention place is drained to zero before the first
// reduction set -- and which drags in §6.6's precondition: on a recycled id, every
// announcement needs its SetFlag/WaitFlag<HardEvent::x_S> fence on the pipe whose work it
// reports, a priming set excepted, and the audit that matters is tick order per id rather than
// per-id counts, because an unfenced announcement balances every count and still releases a
// wait early.
constexpr std::uint16_t kReduceFlagFirst = 0;
constexpr std::uint16_t kReduceFlagWeightsReady = kReduceFlagFirst;                      // 0..1
constexpr std::uint16_t kReduceFlagProductReady = kReduceFlagWeightsReady + kReduceSlots;  // 2..3
constexpr std::uint16_t kReduceFlagReservedFirst = 11;
constexpr std::uint16_t kReduceFlagLast = kReduceFlagProductReady + kReduceSlots - 1;
static_assert(kReduceFlagLast < kReduceFlagReservedFirst,
              "a longer schedule must not renumber a channel into a library-reserved id");
static_assert(kReduceFlagCount == kReduceFlagLast - kReduceFlagFirst + 1,
              "the id budget is an object, not a comment: every place owns exactly one id");

// ---------------------------------------------------------------------------------------------
// Budgets
// ---------------------------------------------------------------------------------------------
//
// These are what the stage ADDS. TPipe allocates for the whole kernel, so the reduction's
// buffers coexist with the attention phase's even though the phases do not.

constexpr std::uint32_t kReduceL1Bytes = kReduceSlots * (kPartialL1Bytes + kWeightL1Bytes);
constexpr std::uint32_t kReduceL0aBytes = kFractalBytes;
constexpr std::uint32_t kReduceL0bBytes = kPartialL1Bytes;
constexpr std::uint32_t kReduceL0cBytes = kProductL0cBytes;
// Per slot: the product landing the Fixpipe writes. Plus one shared statistics arena holding
// the S maxima, the S sums, the weight row and its reduction scratch.
constexpr std::uint32_t kReduceProductPayloadBytes = kHeadSize * kFloatBytes;  // 1024
// AKA3003. The payload alone is exactly 32 UB blocks, so floor(offset/32) mod 16 repeats and
// the two product slots would present the same bank to both ports: the slot the Fixpipe is
// filling and the slot the vector unit is normalising. One block of skew per slot breaks it,
// the same remedy kBankSkewBytes applies to the attention ring's ingest buffers. The skew is
// an allocation tail, never payload.
constexpr std::uint32_t kReduceBankSkewBytes = kC0Bytes;
constexpr std::uint32_t kReduceProductUbBytes =
    kReduceProductPayloadBytes + kReduceBankSkewBytes;  // 1056
constexpr std::uint32_t kReduceStatsUbBytes = 4 * kC0Bytes;  // 128
constexpr std::uint32_t kReduceUbBytes =
    kReduceSlots * kReduceProductUbBytes + kReduceStatsUbBytes;

// The attention phase's own L1 claim, restated from turboquant_fused_decode_opt.hpp's assert so
// the sum is checked here rather than inferred.
constexpr std::uint32_t kAttentionL1Bytes = 16 * 256 + 16 * 64 + 2 * 3 * 64 * 256;
static_assert(kAttentionL1Bytes + kReduceL1Bytes <= 512 * 1024, "L1 budget");
static_assert(kReduceL0aBytes <= 64 * 1024 && kReduceL0bBytes <= 64 * 1024, "L0A/L0B budget");
static_assert(kReduceL0cBytes <= 128 * 1024, "L0C budget");
// Every allocation is a whole 32-byte block, and the product slots land in different UB banks
// (AKA3003: floor(offset/32) mod 16 must differ for a paired slot).
static_assert(kReduceProductUbBytes % kC0Bytes == 0 && kReduceStatsUbBytes % kC0Bytes == 0);
static_assert(kReduceProductUbBytes / kC0Bytes % 16 != 0,
              "paired product slots must not present the same bank");
// The skew is checked against the payload it separates, which is what makes it a skew and not
// a number: it is not a universal bank-conflict proof for every repeat/stride pattern.
static_assert(kReduceProductUbBytes > kReduceProductPayloadBytes &&
                  kReduceBankSkewBytes % kC0Bytes == 0,
              "the skew is a whole block of tail past the payload");

// ---------------------------------------------------------------------------------------------
// The schedule, as arithmetic both roles compute identically
// ---------------------------------------------------------------------------------------------

ASCEND_MICRO_REDUCE_FN std::uint32_t kReducePrologue(std::uint32_t tasks)
{
    return tasks < kReduceSlots ? tasks : kReduceSlots;
}
ASCEND_MICRO_REDUCE_FN std::uint32_t kReduceSlotOf(std::uint32_t task) { return task % kReduceSlots; }
// Number of (token, head) combines this block owns, clamped like production's Reduce().
ASCEND_MICRO_REDUCE_FN std::uint32_t kReduceTaskCount(std::uint32_t total, std::uint32_t block,
                                         std::uint32_t perBlock)
{
    const std::uint64_t start = std::uint64_t{block} * perBlock;
    if (start >= total) return 0;
    const std::uint64_t end = start + perBlock;
    return static_cast<std::uint32_t>((end > total ? total : end) - start);
}
static_assert(kReduceTaskCount(16, 0, 8) == 8 && kReduceTaskCount(16, 2, 8) == 0);
static_assert(kReduceTaskCount(5, 0, 4) == 4 && kReduceTaskCount(5, 1, 4) == 1);
// An empty Split-K worker is a real rung: it must still drain, so a zero task count is legal
// and is not a reason to skip the phase's join.
static_assert(kReducePrologue(0) == 0 && kReducePrologue(1) == 1 && kReducePrologue(9) == 2);

ASCEND_MICRO_REDUCE_FN std::uint64_t kPartialRowOffset(std::uint32_t token, std::uint32_t head,
                                          std::uint32_t split, std::uint32_t numHeads,
                                          std::uint32_t numSplits)
{
    return ((std::uint64_t{token} * numHeads + head) * numSplits + split) * kPartialStride;
}

ASCEND_MICRO_REDUCE_FN bool kValidReduceGeometry(std::uint32_t headSize, std::uint32_t splits)
{
    return headSize == kHeadSize && splits >= 1 && splits <= kMaxSplits &&
           kAlignedSplits(splits) == kC0Floats;
}
static_assert(kValidReduceGeometry(kHeadSize, 1) && kValidReduceGeometry(kHeadSize, 8));
static_assert(!kValidReduceGeometry(kHeadSize, 9) && !kValidReduceGeometry(128, 4));

// ---------------------------------------------------------------------------------------------
// The probe ABI
// ---------------------------------------------------------------------------------------------

// Three legs over one workspace. kGemm and kGemmGemvMode share a body and differ in nothing but
// MmadParams::disableGemv, so a behavioural difference between them isolates that knob; kVector
// is the shipping TurboQuantPartialReducer and is the numerical reference.
enum class ReduceLeg : std::uint32_t {
    kGemm = 0,
    kGemmGemvMode = 1,
    kVector = 2,
};

struct ReduceLaunch {
    std::uint32_t grid = 1;
    std::uint32_t tokens = 1;
    std::uint32_t heads = 4;
    std::uint32_t head_size = kHeadSize;
    std::uint32_t splits = 8;
    std::uint32_t per_block = 4;
    // Diagnostic only: a negative limit combines every token, which is what the pure
    // multi-split rungs want; a real limit drives the fused-vs-combined mix.
    std::int32_t fused_limit = -1;
    ReduceLeg leg = ReduceLeg::kGemm;
};

struct ReduceBuffers {
    void *workspace = nullptr;
    void *output = nullptr;
    // Optional: when null every (token, head) is combined.
    void *lengths = nullptr;
};

}  // namespace ascend_micro::gemm_reduce
