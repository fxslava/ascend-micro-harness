/*
 * The Split-K final reduction as a Cube GEMV. PROTOTYPE, SIMULATOR ONLY.
 *
 * This header lives under tests/ on purpose: nothing in kernels/ includes it, so the variant
 * cannot link into production, the same isolation tests/sim/kernel/fused_decode_opt_kernel.cpp
 * uses. It calls the production vector service for every piece of arithmetic it shares with
 * TurboQuantPartialReducer -- NormalizeHeads, WriteHeads and the output-stage finisher -- and
 * reimplements nothing but the combine itself, which is the thing under test.
 *
 * The geometry, flag ids and budgets come from include/turboquant_gemm_reduce.hpp, which the
 * host gate compiles too.
 */

#ifndef ASCEND_MICRO_TURBOQUANT_GEMM_REDUCE_SERVICE_H
#define ASCEND_MICRO_TURBOQUANT_GEMM_REDUCE_SERVICE_H

#include "kernel_operator.h"

#include "../../../include/turboquant_gemm_reduce.hpp"
#include "../../../kernels/ops/turboquant/fused_decode/common/turboquant_common.h"
#include "../../../kernels/ops/turboquant/fused_decode/vector/turboquant_vector_service.h"

namespace ascend_micro {
namespace gemm_reduce {

using vllm_ascend::turboquant::kFp32PerBlock;
using vllm_ascend::turboquant::kSubBlockSyncMode;
using vllm_ascend::turboquant::kVectorNegInf;
using vllm_ascend::turboquant::MixBlockIdx;
using vllm_ascend::turboquant::NormalizeHeads;
using vllm_ascend::turboquant::SyncEvent;
using vllm_ascend::turboquant::SyncMatrixToFixpipe;
using vllm_ascend::turboquant::SyncMte1ToMatrix;
using vllm_ascend::turboquant::kGatherSrcBase;
using vllm_ascend::turboquant::SyncMte2ToVector;
using vllm_ascend::turboquant::SyncScalarToVector;
using vllm_ascend::turboquant::SyncVectorToMte3;
using vllm_ascend::turboquant::WriteHeads;

// The stage's cross-core ids. They RECYCLE the attention schedule's: a mode-4 id is four bits,
// the toolkit reserves 11..14, the attention ring already spends 0..9 and the task join is 10,
// so there is no free id to take. Recycling is sound here only because SyncAll<false>() and
// the attention phase's own epilogue stand between the phases and every attention place is
// drained to zero first -- and it drags in Âsection 6.6's precondition, which is why every
// announcement below carries its x_S fence and why the trace audit checks tick order per id
// rather than per-id counts.
// One id per slot, per direction. Mode 2 is a block-wide notification and not a per-subcore
// mailbox: AIV -> AIC, BOTH vector subcores set the id and the Cube waits ONCE (which is what
// turboquant_fused_decode_opt.cpp's CubeScores does on kOptFlagSlotReady); AIC -> AIV, the
// Cube sets once and each subcore waits once. The token model keeps a place per subcore
// because that is what forbids a double set; the hardware wait count is this.
__aicore__ inline constexpr uint16_t FlagWeightsReady(uint32_t slot)
{
    return static_cast<uint16_t>(kReduceFlagWeightsReady + slot);
}
__aicore__ inline constexpr uint16_t FlagProductReady(uint32_t slot)
{
    return static_cast<uint16_t>(kReduceFlagProductReady + slot);
}

// Ascend 950PR is the one product whose Mmad can switch GEMV off. The two legs land different
// cubes in L0C, so the Fixpipe has to match: with GEMV on, C is 1 x N in sub-vectors of 1024
// bytes each holding 64 valid bytes; with it off, C is a conventional M-unaligned cube with M
// padded to 16, which is the shape this operator's audited Fixpipe path already emits. The
// default is off for exactly that reason; the enabled leg is a measurement variant.
template <typename scalar_t, bool GEMV_DISABLED = kGemvDisabledDefault>
class TurboQuantGemmReducer {
public:
    __aicore__ inline explicit TurboQuantGemmReducer(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *workspace, __gm__ void *output, const uint32_t numHeads,
                                const uint32_t headSize, const uint32_t numSplits)
    {
        numHeads_ = numHeads;
        headSize_ = headSize;
        numSplits_ = numSplits;
        alignedSplits_ = kAlignedSplits(numSplits);

        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(output));

        // L1, allocated by BOTH roles so the two cores agree on the addresses: the vector
        // subcore writes the weight row by MTE3 and the Cube reads it by MTE1.
        for (uint32_t slot = 0; slot < kReduceSlots; ++slot) {
            pipe_->InitBuffer(weightL1Buf_[slot], kWeightL1Bytes);
            pipe_->InitBuffer(partialL1Buf_[slot], kPartialL1Bytes);
        }
        if ASCEND_IS_AIC {
            pipe_->InitBuffer(operandL0aBuf_, kReduceL0aBytes);
            pipe_->InitBuffer(operandL0bBuf_, kReduceL0bBytes);
            pipe_->InitBuffer(productL0cBuf_, kReduceL0cBytes);
        }
        // Allocated by BOTH roles, like the L1 buffers above and for the same reason: the
        // Cube's Fixpipe destination is an address in the vector core's UB, so the AIC has to
        // walk the same TPipe allocation to know where it is. Guarding these behind
        // `if ASCEND_IS_AIV` is what the first Cube run of this probe did, and the Fixpipe
        // then targeted an unallocated tensor -- su_ccu_mpu_err_t0, a B32 scalar access at a
        // null base on cubecore0 -- while the arithmetic still came out right, which is
        // exactly why a run is not evidence until its exception dumps are read.
        for (uint32_t slot = 0; slot < kReduceSlots; ++slot) {
            // kReduceProductUbBytes carries kReduceBankSkewBytes of tail past the headSize
            // payload, so the slot the Fixpipe is filling and the slot the vector unit is
            // normalising do not present the same UB bank (AKA3003). The skew is never
            // payload.
            pipe_->InitBuffer(productUbBuf_[slot], kReduceProductUbBytes);
        }
        pipe_->InitBuffer(statsUbBuf_, kStatsUbBytes);
        pipe_->InitBuffer(outUbBuf_, headSize * sizeof(scalar_t));
        // Init boundary: drains every pipe before the stage's first GM read, and before the
        // recycled ids are primed.
        AscendC::PipeBarrier<PIPE_ALL>();
        PrimeVectorState();
    }

    // `total` is numTokens * numHeads; `perBlock` the production tiling's reduceTasksPerBlock.
    // `select` answers "does this (token, head) need a combine at all", so a fused token is
    // skipped exactly as production's Reduce() skips it. `finisher` is the shared output stage.
    template <typename Select, typename Finisher>
    __aicore__ inline void Reduce(const uint32_t total, const uint32_t perBlock, Select &select,
                                  Finisher &finisher)
    {
        const uint32_t start = MixBlockIdx() * perBlock;
        if (start >= total) {
            // An empty worker still drains: "this queue is empty" is what the worker boundary
            // needs established, and nothing below was primed, so there is nothing to retire.
            return;
        }
        uint32_t end = start + perBlock;
        if (end > total) {
            end = total;
        }

        // Compact the block's live tasks first, so the ring's slot arithmetic is over a dense
        // index. A fused token contributes no task and must not consume a slot generation.
        uint32_t tasks = 0;
        for (uint32_t task = start; task < end && tasks < kMaxTasksPerBlock; ++task) {
            if (select.NeedsCombine(task / numHeads_)) {
                taskIds_[tasks++] = task;
            }
        }
        if (tasks == 0) {
            return;
        }

        const uint32_t prologue = kReducePrologue(tasks);
        for (uint32_t r = 0; r < prologue; ++r) {
            Fetch(r);
            Stats(r);
        }
        for (uint32_t r = 0; r < tasks; ++r) {
            Gemm(r, tasks);
            Finish(r, finisher);
            if (r + kReduceSlots < tasks) {
                Fetch(r + kReduceSlots);
                Stats(r + kReduceSlots);
            }
        }
    }

private:
    // One UB arena for the statistics, in 8-float blocks. Every vector destination below is a
    // whole-block base: a vector store to a 4-byte-aligned address raises
    // vec_err_instr_misalign on this core, which is how the single-element lane packing this
    // first carried was found. The lane gather is an AscendC::Gather instead.
    //
    //   0   gathered m_s, padding lanes kVectorNegInf
    //   1   gathered l_s, padding lanes zero
    //   2   the weight row w_s, the 32 bytes MTE3 stores into L1
    //   3   m* broadcast over 8 lanes
    //   4   w_s * l_s
    //   5-6 ReduceMax / ReduceSum work
    //   8-9 l*, one block per slot: produced by Stats(r) and read by Finish(r) two tasks later,
    //       so a single shared block would be overwritten before its read
    //   10  NormalizeHeads' sums
    //   11  NormalizeHeads' invs
    //   12  Gather byte offsets of the maxima
    //   13  Gather byte offsets of the masses
    //   16+ the 64-byte landing of each split's statistics tail, kMaxSplits of them
    static constexpr uint32_t kLaneMax = 0 * kFp32PerBlock;
    static constexpr uint32_t kLaneSum = 1 * kFp32PerBlock;
    static constexpr uint32_t kLaneWeight = 2 * kFp32PerBlock;
    static constexpr uint32_t kLaneStar = 3 * kFp32PerBlock;
    static constexpr uint32_t kLaneWeighted = 4 * kFp32PerBlock;
    static constexpr uint32_t kLaneScratch = 5 * kFp32PerBlock;
    static constexpr uint32_t kLaneTotal = 8 * kFp32PerBlock;  // + slot
    static constexpr uint32_t kLaneSums = 10 * kFp32PerBlock;
    static constexpr uint32_t kLaneInvs = 11 * kFp32PerBlock;
    static constexpr uint32_t kLaneMaxOffsets = 12 * kFp32PerBlock;
    static constexpr uint32_t kLaneSumOffsets = 13 * kFp32PerBlock;
    static constexpr uint32_t kLaneTail = 16 * kFp32PerBlock;
    // Each split's tail gets its own 64-byte landing before anything reads it, so an unmasked
    // register read of one split's block cannot touch another split's DMA landing (section 2).
    static constexpr uint32_t kTailBlocksPerSplit = 2;
    static constexpr uint32_t kStatsUbFloats =
        kLaneTail + kMaxSplits * kTailBlocksPerSplit * kFp32PerBlock;
    static constexpr uint32_t kStatsUbBytes = kStatsUbFloats * sizeof(float);
    static_assert(kLaneTotal + kReduceSlots * kFp32PerBlock <= kLaneSums, "l* blocks overlap");
    static_assert(kStatsUbBytes % 32 == 0, "every UB allocation is a whole 32-byte block");
    static_assert(kStatsUbBytes >= kReduceStatsUbBytes, "the contract's stats budget is a floor");
    // The byte stride between two splits' tail landings, which is what the Gather tables step.
    static constexpr uint32_t kTailStrideBytes =
        kTailBlocksPerSplit * kFp32PerBlock * sizeof(float);
    // The ring index arithmetic is over a dense live-task index, so the compacted id list is
    // what bounds a block's share.
    static constexpr uint32_t kMaxTasksPerBlock = 64;

    __aicore__ inline AscendC::LocalTensor<float> Stats() { return statsUbBuf_.template Get<float>(); }

    // Two one-time jobs on the vector subcore that owns the stage.
    //
    // The weight slot is a whole 512-byte fractal because GEMV wants its 1 x K row 512-byte
    // aligned, but only the first numSplits_ floats are ever rewritten. Zeroing the rest once
    // matters for correctness and not tidiness: section 3 floods GM with 0x7FC0BEEF, an fp32 quiet
    // NaN, and a stale NaN in a lane the Mmad reads would poison the whole product.
    //
    // The Gather tables are byte offsets into the statistics landings, one per split. They are
    // written by the SCALAR unit and handed to the vector pipe through S_V, which is this
    // repository's idiom for "a lane the vector pipe cannot address because it is not on a
    // 32-byte boundary" (SyncScalarToVector). This is table setup at Init, not per-element
    // arithmetic, so it is the address scheduling section 2 calls control flow rather than the scalar
    // dequant/GEMV/softmax computation section 2 forbids.
    __aicore__ inline void PrimeVectorState()
    {
        if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        const AscendC::LocalTensor<float> stats = Stats();
        AscendC::Duplicate(stats, 0.0f, kStatsUbFloats);
        SyncVectorToMte3();
        for (uint32_t slot = 0; slot < kReduceSlots; ++slot) {
            AscendC::DataCopy(weightL1Buf_[slot].template Get<float>(), stats,
                              kWeightL1Bytes / sizeof(float));
        }
        SyncEvent<AscendC::HardEvent::MTE3_S>();

        const AscendC::LocalTensor<uint32_t> maxOffsets =
            stats[kLaneMaxOffsets].template ReinterpretCast<uint32_t>();
        const AscendC::LocalTensor<uint32_t> sumOffsets =
            stats[kLaneSumOffsets].template ReinterpretCast<uint32_t>();
        for (uint32_t split = 0; split < kMaxSplits; ++split) {
            const uint32_t base = split * kTailStrideBytes;
            maxOffsets.SetValue(split, base + kPartialMaxLane * sizeof(float));
            sumOffsets.SetValue(split, base + kPartialSumLane * sizeof(float));
        }
        SyncScalarToVector();
        }
    }

    __aicore__ inline uint64_t PartialRow(const uint32_t task, const uint32_t split) const
    {
        return kPartialRowOffset(task / numHeads_, task % numHeads_, split, numHeads_, numSplits_);
    }

    // ------------------------------------------------------------------------------------
    // AIC MTE2: the partial cube, GM -> L1, in the layout Mmad's B operand needs
    // ------------------------------------------------------------------------------------
    //
    // Mmad wants B as K x N in Zn. The workspace already holds [S, D] row-major, which IS
    // K x N, so only the fractal layout is in question, and the inline DN2NZ conversion
    // settles it: the API reference states DN->NZ is "equivalent to the ND-to-ZN format
    // conversion", so feeding the ND rows lands the ZN image of that same matrix in L1 with
    // the split axis as C0. A non-transposed Load2DV2 then leaves it alone (Zn2Zn).
    //
    // That is the whole reason this operand is viable. The transposed b32 path merges two
    // consecutive 16x8 fractals into a 16x16 square and so needs a source Zn cube with
    // Col == 16, which a K = S <= 8 operand cannot provide; it also constrains kStep to a
    // multiple of 2. Avoiding the transpose avoids all of it.
    __aicore__ inline void Fetch(const uint32_t r)
    {
        if ASCEND_IS_AIC {
        const uint32_t slot = kReduceSlotOf(r);
        if (r >= kReduceSlots) {
            // The cube of task r - 2 must have reached L0B before this overwrites it. Posted
            // by Gemm only when a later fetch will consume it.
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(slot));
        }
        AscendC::Dn2NzParams params;
        params.dnNum = kDn2NzDnNum;
        params.nValue = static_cast<uint16_t>(headSize_);
        // dValue is the C0 axis: the splits. Below kC0Floats the conversion zero-fills the
        // destination to the 32-byte boundary, which is the second of the two independent
        // guards on the padding lanes (the first is the zeroed weight lanes).
        params.dValue = numSplits_;
        params.srcDnMatrixStride = 0;
        // Not headSize_: the statistics tail shares the row, so the source row stride is the
        // full partial stride.
        params.srcDValue = kPartialStride;
        params.dstNzC0Stride = static_cast<uint16_t>(headSize_);
        params.dstNzNStride = kDn2NzDstNzNStride;
        params.dstNzMatrixStride = 0;
        AscendC::DataCopy(partialL1Buf_[slot].template Get<float>(),
                          workspaceGm_[PartialRow(taskIds_[r], 0)], params);
        SyncEvent<AscendC::HardEvent::MTE2_MTE1>();
        }
    }

    // ------------------------------------------------------------------------------------
    // AIV0: the statistics merge, and the weight row into L1
    // ------------------------------------------------------------------------------------
    //
    // The sequential online rescale the Vector reducer runs is replaced by a two-pass form:
    // every m_s is known up front, so m* is one reduction and the S weights are ONE 8-lane
    // Exp rather than S dependent Exp pairs. That restructuring is available to a vector-only
    // reducer too; it is not what the Cube buys. What the Cube buys is the 256-wide part --
    // the two BroadcastMuls and the Add per split, four repeats each -- which leaves the
    // vector pipe entirely.
    __aicore__ inline void Stats(const uint32_t r)
    {
        if ASCEND_IS_AIV {
        const uint32_t slot = kReduceSlotOf(r);
        if (AscendC::GetSubBlockIdx() != 0) {
            // AIV1 carries no data. It still posts the credit that paces the Cube against it:
            // the Cube waits once per subcore, so without this post it could run two slot
            // generations ahead of AIV1's outstanding product wait and drop a SET.
            AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(FlagWeightsReady(slot));
            return;
        }

        const AscendC::LocalTensor<float> stats = Stats();
        const AscendC::LocalTensor<float> maxes = stats[kLaneMax];
        const AscendC::LocalTensor<float> sums = stats[kLaneSum];
        const AscendC::LocalTensor<float> weight = stats[kLaneWeight];
        const AscendC::LocalTensor<float> star = stats[kLaneStar];
        const AscendC::LocalTensor<float> weighted = stats[kLaneWeighted];
        const AscendC::LocalTensor<float> scratch = stats[kLaneScratch];
        const AscendC::LocalTensor<float> tail = stats[kLaneTail];

        // Padding lanes: a finite -FLT_MAX, not -inf, so that m_s - m* underflows to -inf and
        // Exp gives exactly 0 instead of the NaN an inf - inf would produce when every split
        // of a task is empty.
        AscendC::Duplicate(maxes, kVectorNegInf, kFp32PerBlock);
        AscendC::Duplicate(sums, 0.0f, kFp32PerBlock);

        for (uint32_t split = 0; split < numSplits_; ++split) {
            const uint64_t row = PartialRow(taskIds_[r], split) + headSize_;
            // One 64-byte burst per split carries both lanes: the max sits at kPartialMaxLane
            // and the sum at kPartialSumLane, 8 floats apart, inside the same tail.
            AscendC::DataCopy(tail[split * kTailBlocksPerSplit * kFp32PerBlock], workspaceGm_[row],
                              kTailBlocksPerSplit * kFp32PerBlock);
        }
        SyncMte2ToVector();

        // The Duplicates above wrote the lanes the Gathers below overwrite.
        AscendC::PipeBarrier<PIPE_V>();

        // Pack lane 0 of each split's landing into contiguous lanes.
        //
        // The obvious way to do this -- one single-element Adds per split, writing maxes[s] --
        // is ILLEGAL on this core, and the simulator says so rather than silently working: a
        // vector store whose destination is not 32-byte aligned raises
        // vec_err_instr_misalign_t0, and maxes[1] is four bytes in. Vector repeat strides are
        // per-repeat and in whole blocks, so no repeat pattern expresses a 4-byte pick either.
        // Gather does: it takes a table of BYTE offsets and writes its result contiguously
        // from a block-aligned base, which is the primitive the KV codec's expand already uses.
        AscendC::Gather(maxes, tail, stats[kLaneMaxOffsets].template ReinterpretCast<uint32_t>(),
                        kGatherSrcBase, numSplits_);
        AscendC::Gather(sums, tail, stats[kLaneSumOffsets].template ReinterpretCast<uint32_t>(),
                        kGatherSrcBase, numSplits_);
        AscendC::PipeBarrier<PIPE_V>();

        // m* over all kC0Floats lanes: the padding lanes are kVectorNegInf, so reducing the
        // whole block is the same answer as reducing the live prefix and needs no mask.
        AscendC::ReduceMax<float>(scratch, maxes, scratch[kFp32PerBlock], kFp32PerBlock, false);
        AscendC::Brcb(star, scratch, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
        AscendC::Sub(weight, maxes, star, kFp32PerBlock);
        AscendC::Exp(weight, weight, kFp32PerBlock);
        AscendC::Mul(weighted, weight, sums, kFp32PerBlock);
        // l* lands in this slot's own block: it is produced here but consumed by Finish two
        // tasks later, so a single shared lane would be overwritten before its read.
        AscendC::ReduceSum<float>(stats[kLaneTotal + slot * kFp32PerBlock], weighted, scratch,
                                  kFp32PerBlock);

        SyncVectorToMte3();
        AscendC::DataCopy(weightL1Buf_[slot].template Get<float>(), weight, kFp32PerBlock);
        // Âsection 6.6's fence: the announcement must be preceded by the x_S pair on the pipe whose
        // work it reports. On a recycled id this is the precondition, not a nicety -- an
        // unfenced announcement balances every count and still releases the Cube onto a
        // weight row MTE3 has not landed.
        SyncEvent<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(FlagWeightsReady(slot));
        }
    }

    // ------------------------------------------------------------------------------------
    // AIC: the GEMV
    // ------------------------------------------------------------------------------------
    __aicore__ inline void Gemm(const uint32_t r, const uint32_t tasks)
    {
        if ASCEND_IS_AIC {
        const uint32_t slot = kReduceSlotOf(r);
        // One wait, not one per subcore: a mode-2 notification is satisfied when both vector
        // subcores have set the id.
        AscendC::CrossCoreWaitFlag(FlagWeightsReady(slot));

        const AscendC::LocalTensor<float> leftL0a = operandL0aBuf_.template Get<float>();
        const AscendC::LocalTensor<float> rightL0b = operandL0bBuf_.template Get<float>();
        const AscendC::LocalTensor<float> product = productL0cBuf_.template Get<float>();

        // A: 1 x K, M x K Nz, not transposed. mStep counts 16-element M steps and kStep
        // 32-byte K steps, so both are 1 for a single 1 x 8 float row.
        AscendC::LoadData2DParamsV2 aParams;
        aParams.mStartPosition = 0;
        aParams.kStartPosition = 0;
        aParams.mStep = 1;
        aParams.kStep = 1;
        aParams.srcStride = 1;
        aParams.dstStride = 1;
        aParams.ifTranspose = false;
        AscendC::LoadData(leftL0a, weightL1Buf_[slot].template Get<float>(), aParams);

        // B: the Zn cube DN2NZ already built. Non-transposed, so the b32 transpose
        // constraints do not apply: mStep walks N in 16s, kStep walks K in C0 blocks, and
        // K <= kC0Floats makes that exactly one.
        AscendC::LoadData2DParamsV2 bParams;
        bParams.mStartPosition = 0;
        bParams.kStartPosition = 0;
        bParams.mStep = static_cast<uint16_t>(headSize_ / kFractalRows);
        bParams.kStep = 1;
        bParams.srcStride = static_cast<uint16_t>(headSize_ / kFractalRows);
        bParams.dstStride = static_cast<uint16_t>(headSize_ / kFractalRows);
        bParams.ifTranspose = false;
        AscendC::LoadData(rightL0b, partialL1Buf_[slot].template Get<float>(), bParams);

        SyncMte1ToMatrix();

        AscendC::MmadParams mmad;
        mmad.m = 1;
        mmad.n = static_cast<uint16_t>(headSize_);
        // The hardware aligns float K to ceil(K/8)*8 regardless, so passing the aligned
        // extent states what it will do. The lanes beyond numSplits_ are zero in both
        // operands, by the weight row and by DN2NZ's own zero fill.
        mmad.k = static_cast<uint16_t>(alignedSplits_);
        // cmatrixInitVal, not the doc's ccubeInitVal: this toolkit's MmadParams names it the
        // former. true zeroes C, which is what a single-Mmad GEMV wants.
        mmad.cmatrixInitVal = true;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // disableGemv exists only on the arch that has it to disable. dav-3510 is the target,
        // so the knob is real here; the #if keeps the variant compilable elsewhere.
        mmad.disableGemv = GEMV_DISABLED;
#else
        static_cast<void>(GEMV_DISABLED);
#endif
        AscendC::Mmad(product, leftL0a, rightL0b, mmad);

        // The cube credit, posted only when a later fetch will consume it. Unconditional, it
        // leaves the id set on a short worker for the next worker's prologue to find.
        if (r + kReduceSlots < tasks) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(slot));
        }

        SyncMatrixToFixpipe();
        AscendC::Fixpipe<float, float, vllm_ascend::turboquant::kFixpipeToUb>(
            productUbBuf_[slot].template Get<float>(), product,
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>(
                static_cast<uint16_t>(headSize_), 1, static_cast<uint16_t>(kFractalRows),
                headSize_));
        // Drains the Fixpipe before the announcement releases the vector unit onto its rows:
        // the Âsection 6.6 fence for this id, on the pipe that did the work.
        AscendC::PipeBarrier<PIPE_FIX>();
        SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(FlagProductReady(slot));
        }
    }

    // ------------------------------------------------------------------------------------
    // AIV0: normalise, finish, write -- all shared production code
    // ------------------------------------------------------------------------------------
    template <typename Finisher>
    __aicore__ inline void Finish(const uint32_t r, Finisher &finisher)
    {
        if ASCEND_IS_AIV {
        const uint32_t slot = kReduceSlotOf(r);
        AscendC::CrossCoreWaitFlag(FlagProductReady(slot));
        if (AscendC::GetSubBlockIdx() != 0) {
            // AIV1's whole job: consume its copy of the token. Without this the phase ends
            // with the flag set, which the next worker's prologue would then mistake for its
            // own credit.
            return;
        }

        const AscendC::LocalTensor<float> stats = Stats();
        const AscendC::LocalTensor<float> acc = productUbBuf_[slot].template Get<float>();
        const uint32_t task = taskIds_[r];
        const uint32_t token = task / numHeads_;
        const uint32_t head = task % numHeads_;

        // The same close-out TurboQuantPartialReducer runs, on the same shared helpers: the
        // 1/(l* + eps) scale, then the output stage, then the cast and the GM burst.
        NormalizeHeads(acc, stats[kLaneTotal + slot * kFp32PerBlock], 1, headSize_,
                       stats[kLaneSums], stats[kLaneInvs]);
        finisher.FinishHeads(acc, token, head, 1);
        WriteHeads<scalar_t>(acc, 1, headSize_, outUbBuf_.template Get<scalar_t>(), outputGm_,
                             (static_cast<uint64_t>(token) * numHeads_ + head) * headSize_);
        }
    }

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> weightL1Buf_[kReduceSlots];
    AscendC::TBuf<AscendC::TPosition::B1> partialL1Buf_[kReduceSlots];
    AscendC::TBuf<AscendC::TPosition::A2> operandL0aBuf_;
    AscendC::TBuf<AscendC::TPosition::B2> operandL0bBuf_;
    AscendC::TBuf<AscendC::TPosition::CO1> productL0cBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> productUbBuf_[kReduceSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> statsUbBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outUbBuf_;
    AscendC::GlobalTensor<float> workspaceGm_;
    AscendC::GlobalTensor<scalar_t> outputGm_;
    uint32_t taskIds_[kMaxTasksPerBlock] = {};
    uint32_t numHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t alignedSplits_ = kC0Floats;
};

}  // namespace gemm_reduce
}  // namespace ascend_micro

#endif
