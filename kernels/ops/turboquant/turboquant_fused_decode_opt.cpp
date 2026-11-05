/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The modulo-scheduled fused TurboQuant decode: the same arithmetic as turboquant_fused_decode.cpp's
// kv4fp8 instance, on a three-stage software pipeline instead of the two-stage one.
//
// WHAT CHANGES, AND WHY
//
// The shipping ring schedule (TurboQuantFusedDecode::StageTilesRing) already runs the KV read a tile ahead
// of the stage, so MTE2 overlaps the tile it feeds. What it cannot overlap is the Cube against the vector
// unit: a tile's score GEMM, its softmax, its context GEMM and its accumulate form one chain, and with a
// single Fixpipe score window the next tile's score GEMM cannot start until this tile's softmax has
// consumed -- and overwritten, in place -- the window it would write. The Cube idles through every
// softmax; the vector unit idles through every GEMM pair.
//
// This file breaks the chain with a second score window. In the steady state, iteration k is
//
//   MTE2        reads tile k + kCubePipelineSlots into ingest ring slot k % kCubePipelineSlots
//   AIC Cube    GemmScores(tile k + 1) into score window (k + 1) % kCubeScoreWindows,
//               then GemmContext(tile k) into the context window
//   AIV Vector  the online softmax of tile k out of window k % kCubeScoreWindows, the exponent decay,
//               the L1 stage of tile k + 2, and the accumulate of tile k
//
// and all three are in flight at once: the score GEMM of k + 1 writes the window the softmax of k is not
// reading, the context GEMM of k runs against the vector unit's decay and L1 stage, and the MTE2 burst for
// k + kCubePipelineSlots runs against both.
//
// The accumulate of tile k stays in iteration k rather than sliding to k + 1, which is what keeps the
// online-softmax state single-buffered: alpha and the probability scale are per-tile, held in one state
// field each, and deferring the accumulate would need a second copy of both. Deferring it buys nothing
// here anyway -- the Cube is already covered by the next tile's score GEMM.
//
// WHAT THE SCHEDULE COSTS
//
// Per ring slot: kCubeTileRows * packedBytes of UB for the packed plane and kCubeTileRows * scaleSlot
// floats for the scale lanes (about 10 KB together at headSize 256), plus 2 * kCubeTileRows * headSize
// bytes of L1 for the key and value operands (32 KB). Three slots is 30 KB of the 256 KB UB partition and
// 96 KB of the 512 KB L1, so neither buffer is what bounds the ring: the cross-core flag space is. A flag
// id is four bits wide and the toolkit reserves 11..14, and one id per slot and per window spends
// 2 * kCubePipelineSlots + kCubeScoreWindows + 2 = 10 of the 11 that are left (turboquant_layout.h).
//
// Harness integration: local service snapshot, 64-token KV4/FP8 geometry,
// Vector microroutines and strict CAModel runner. See reports/fused_decode_opt.md.

#include "../../../include/turboquant_fused_decode_opt.hpp"
#include "fused_decode/types.h"

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"

#include "fused_decode/common/turboquant_codec_mx.h"
#include "fused_decode/cube/turboquant_cube_service.h"
#include "fused_decode/cube/turboquant_query_basis.h"
#include "fused_decode/vector/turboquant_vector_service.h"

using vllm_ascend::turboquant::CeilDiv;
using vllm_ascend::turboquant::DecodeUnpack;
using vllm_ascend::turboquant::DecodeUnpackGathers;
using vllm_ascend::turboquant::DecodeUnpackIsLloydMax;
using vllm_ascend::turboquant::kBankSkewBytes;
using vllm_ascend::turboquant::kCubePipelineSlots;
using vllm_ascend::turboquant::kCubeScoreWindows;
using vllm_ascend::turboquant::kCubeTileM;
using vllm_ascend::turboquant::kCubeTileRows;
using vllm_ascend::turboquant::kOptFlagContextReady;
using vllm_ascend::turboquant::kOptFlagProbsReady;
using vllm_ascend::turboquant::kOptFlagReservedFirst;
using vllm_ascend::turboquant::kOptFlagScoresReady;
using vllm_ascend::turboquant::kOptFlagSlotFree;
using vllm_ascend::turboquant::kOptFlagSlotReady;
using vllm_ascend::turboquant::kSubBlockSyncMode;
using vllm_ascend::turboquant::kVectorSubcoresPerBlock;
using vllm_ascend::turboquant::MixBlockIdx;
using vllm_ascend::turboquant::SyncVectorToMte2;
using vllm_ascend::turboquant::TurboQuantCubeMm;
using vllm_ascend::turboquant::TurboQuantMode;
using vllm_ascend::turboquant::TurboQuantModeCodec;
using vllm_ascend::turboquant::TurboQuantPartialReducer;
using vllm_ascend::turboquant::TurboQuantQueryBasis;
using vllm_ascend::turboquant::TurboQuantTaskHeads;
using vllm_ascend::turboquant::TurboQuantVectorDecodeService;

namespace {

// A block-local join between tasks and before returning to the scheduler.
// No other participant can reuse the remote Fixpipe landings before this join.
__aicore__ inline void DrainFusedDecodeTask()
{
    constexpr uint16_t flag = ascend_micro::fused_decode_opt::kTaskJoinFlag;
    if ASCEND_IS_AIC {
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::M_S>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(flag);
        AscendC::CrossCoreWaitFlag(flag);
        // Close the MTE1 -> Matrix -> Scalar chain explicitly, including
        // empty tasks, before the join channel can be reused.
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE1_M>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::M_S>();
    }
    if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag(flag);
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE2_S>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::V_S>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(flag);
    }
}

constexpr uint32_t PipelineGcd(const uint32_t a, const uint32_t b)
{
    return b == 0 ? a : PipelineGcd(b, a % b);
}

// The period of the schedule's whole index state: the ring slot repeats every kCubePipelineSlots tiles and
// the score window every kCubeScoreWindows, so unrolling the loop body this many times makes every index
// in it a compile-time constant. Six, for a three-slot ring and two windows.
constexpr uint32_t kPipelinePhases =
    kCubePipelineSlots * kCubeScoreWindows / PipelineGcd(kCubePipelineSlots, kCubeScoreWindows);

// The modulo-scheduled decode. MODE and UNPACK carry the same meaning they do on TurboQuantFusedDecode,
// and PRE_ROTATED the same two query bases: true takes the rotated fp32 query, false rotates the raw query
// once in a prologue and selects the output stage per launch (cube/turboquant_query_basis.h).
//
// Read RunCubeSteadyPhase and RunVectorSteadyPhase together: every flag is one
// edge of the marked graph they form, and the only reason the loop needs no draining barrier is that each
// edge is point-to-point. Both cores derive the same slot and window index from the same template
// parameter, so neither is ever told which one the other picked.
template <TurboQuantMode MODE, typename scalar_t, bool PRE_ROTATED = true,
          DecodeUnpack UNPACK = DecodeUnpack::kNative>
class TurboQuantFusedDecodeOpt {
public:
    static constexpr uint32_t kSlots = kCubePipelineSlots;
    static constexpr uint32_t kWindows = kCubeScoreWindows;
    static constexpr uint32_t kPhases = kPipelinePhases;

    using Mm = TurboQuantCubeMm<MODE, kSlots>;
    using Codec = TurboQuantModeCodec<MODE, DecodeUnpackGathers(UNPACK), DecodeUnpackIsLloydMax(UNPACK)>;
    using Vector =
        TurboQuantVectorDecodeService<MODE, scalar_t, Mm, Codec, UNPACK, kSlots, kWindows, kBankSkewBytes>;
    using Reducer = TurboQuantPartialReducer<scalar_t>;
    using Basis = TurboQuantQueryBasis<scalar_t, !PRE_ROTATED>;

    // The ring ingest and the NZ-tiled plane this schedule streams are kv4fp8's. A codebook mode reads its
    // slots row-major and expands them in bands, which is a different ingest and a different pipeline.
    static_assert(Codec::WorkBufferWords(256, 32) * sizeof(float) ==
        ascend_micro::fused_decode_opt::kUbSizes[27], "codec workspace budget drift");
    static_assert(Vector::kBatched, "the modulo schedule is the batched (kv4fp8) ingest only");
    static_assert(Vector::kIngestSlots == kSlots, "the UB ring and the L1 ring have to be the same depth");
    static_assert(Vector::kScoreWindows == kWindows, "the vector service did not take the window count");
    static_assert(Mm::kSlots == kSlots, "the Cube's L1 ring and the ingest ring have to agree");
    static_assert(kOptFlagProbsReady < kOptFlagReservedFirst,
                  "the schedule's flag ids run into the ids the toolkit reserves for its own barriers");
    static_assert(kSlots == 3 && kWindows == 2 && kPhases == 6, "three-zone schedule geometry");
    static_assert(kWindows >= 2, "one score window is the lock-step schedule, not this one");
    static_assert(kSlots >= 3, "a two-slot ring cannot keep a read, a stage and a GEMM pair in flight");

    __aicore__ inline explicit TurboQuantFusedDecodeOpt(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *queryRot, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *scaleCache, __gm__ void *blockTables, __gm__ void *contextLens,
                                __gm__ void *modeTables, __gm__ void *workspace, __gm__ void *output,
                                const uint32_t numTokens, const uint32_t numHeads, const uint32_t numKvHeads,
                                const uint32_t headSize, const uint32_t blockSize, const uint32_t maxBlocksPerSeq,
                                const uint32_t numSplits, const uint32_t headsPerTask,
                                const uint32_t fusedContextLimit, const float scale, const float invSqrtLen)
    {
        numTokens_ = numTokens;
        numHeads_ = numHeads;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        blockSize_ = blockSize;
        maxBlocksPerSeq_ = maxBlocksPerSeq;
        numSplits_ = numSplits;
        fusedContextLimit_ = fusedContextLimit;
        groupHeads_ = numHeads / numKvHeads;
        headsPerTask_ = headsPerTask < 1 ? 1 : headsPerTask;
        if (headsPerTask_ > groupHeads_) {
            headsPerTask_ = groupHeads_;
        }
        if (headsPerTask_ > kCubeTileM) {
            headsPerTask_ = kCubeTileM;
        }
        chunksPerGroup_ = CeilDiv(groupHeads_, headsPerTask_);

        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTables));
        contextLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(contextLens), numTokens);

        mm_.Init(pipe_, headSize_, kCubeTileRows);
        vector_.Init(pipe_, queryRot, keyCache, valueCache, scaleCache, modeTables, workspace, output, numHeads,
                     numKvHeads, headSize, blockSize, numSplits, scale, invSqrtLen);
        mm_.InitUnitScales(pipe_);
        // No launch-boundary flush. The lock-step decode drains every pipe here, which is the one place a
        // PipeBarrier<PIPE_ALL> is cheap but also the one place it hides what it is for. On this
        // instantiation there is nothing in flight to drain: Init only allocates (compile-time offsets, no
        // traffic), kv4fp8's affine codec copies no table at all, InitUnitScales is fp4-only, and the two
        // instantiations whose codec does copy a 64 B table fence it inside their own Init
        // (turboquant_codec_mx.h). The first thing to move data after this point is StageQuery, which
        // posts its own MTE2 -> V edge.
    }

    __aicore__ inline void InitProbe(uint32_t page, uint32_t tile)
    {
        probePage_ = page;
        probeTile_ = tile;
    }

    // The raw-query decode's two basis stages: InitBasis after Init, RotateQuery before Process.
    __aicore__ inline void InitBasis(__gm__ void *query, __gm__ void *piSigns, __gm__ void *rotTables,
                                     __gm__ void *h16, __gm__ void *gate, __gm__ void *queryRot,
                                     const uint32_t cubeChunkVectors, const uint32_t outputStage,
                                     const float invSqrtLen)
    {
        static_assert(Basis::kEnabled, "a pre-rotated decode has no query rotation");
        basis_.Init(pipe_, query, piSigns, rotTables, h16, gate, queryRot, numHeads_, headSize_, cubeChunkVectors,
                    outputStage, invSqrtLen);
    }

    __aicore__ inline void RotateQuery(const uint32_t vectorsPerBlock)
    {
        static_assert(Basis::kEnabled, "a pre-rotated decode has no query rotation");
        basis_.Rotate(numTokens_ * numHeads_, vectorsPerBlock);
    }

    __aicore__ inline void Process(const uint32_t tasksPerBlock)
    {
        const uint32_t tasks = numTokens_ * numKvHeads_ * numSplits_ * chunksPerGroup_;
        const uint32_t start = MixBlockIdx() * tasksPerBlock;
        uint32_t end = start + tasksPerBlock;
        if (end > tasks) {
            end = tasks;
        }
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t chunk = task % chunksPerGroup_;
            const uint32_t splitTask = task / chunksPerGroup_;
            const uint32_t split = splitTask % numSplits_;
            const uint32_t tokenKvHead = splitTask / numSplits_;
            ComputeTask(tokenKvHead / numKvHeads_, tokenKvHead % numKvHeads_, split, chunk);
        }
    }

    // CAModel-only reduction-network probe schedule. Split zero processes the first 64-token
    // tile of page zero through the real attention pipeline. Every other split publishes the
    // neutral state using the production partial layout and immediately joins the same task
    // drain. The global context length remains full-sized, so every block takes the reduction.
    __aicore__ inline void ProcessReductionProbe(const uint32_t tasksPerBlock)
    {
        const uint32_t tasks = numTokens_ * numKvHeads_ * numSplits_ * chunksPerGroup_;
        const uint32_t start = MixBlockIdx() * tasksPerBlock;
        uint32_t end = start + tasksPerBlock;
        if (end > tasks) end = tasks;
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t chunk = task % chunksPerGroup_;
            const uint32_t splitTask = task / chunksPerGroup_;
            const uint32_t split = splitTask % numSplits_;
            const uint32_t tokenKvHead = splitTask / numSplits_;
            const uint32_t token = tokenKvHead / numKvHeads_;
            const uint32_t kvHead = tokenKvHead % numKvHeads_;
            const uint32_t offset = chunk * headsPerTask_;
            const uint32_t rows = groupHeads_ - offset < headsPerTask_ ? groupHeads_ - offset : headsPerTask_;
            const TurboQuantTaskHeads heads = Vector::Heads(kvHead * groupHeads_ + offset, rows);
            if (split == 0) {
                ComputeTask(token, kvHead, split, chunk, static_cast<int32_t>(kCubeTileRows));
            } else {
                vector_.WriteNeutralPartial(token, split, heads);
                DrainFusedDecodeTask();
            }
        }
    }

    // Grid-uniform, like the lock-step decode's: every block reads every token's length, so the SyncAll
    // the caller guards with this is entered by all of them or by none.
    __aicore__ inline bool NeedsReduction()
    {
        if (numSplits_ <= 1) {
            return false;
        }
        for (uint32_t token = 0; token < numTokens_; ++token) {
            if (!IsFused(contextLenGm_.GetValue(token))) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void Reduce(Reducer &reducer, const uint32_t reduceTasksPerBlock)
    {
        const uint32_t total = numTokens_ * numHeads_;
        const uint32_t start = MixBlockIdx() * reduceTasksPerBlock;
        uint32_t end = start + reduceTasksPerBlock;
        if (end > total) {
            end = total;
        }
        if (start >= end) {
            return;
        }
        const uint32_t firstHalf = CeilDiv(end - start, kVectorSubcoresPerBlock);
        const bool secondSubcore = AscendC::GetSubBlockIdx() != 0;
        const uint32_t from = secondSubcore ? start + firstHalf : start;
        const uint32_t to = secondSubcore ? end : start + firstHalf;
        for (uint32_t task = from; task < to; ++task) {
            const uint32_t token = task / numHeads_;
            if (!IsFused(contextLenGm_.GetValue(token))) {
                reducer.Reduce(token, task % numHeads_, basis_);
            }
        }
    }

#if defined(ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE) && ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE
    // The Cube GEMM reduction's entry. Three things differ from Reduce() above, and all three
    // are why it needs its own method rather than a different reducer object:
    //
    //   * It is driven by BOTH roles, not `if ASCEND_IS_AIV`. The Cube does the contraction, so
    //     the caller must not guard the call -- and both roles have to walk the same TPipe
    //     allocation, because the Fixpipe's destination is an address in the vector cores' UB.
    //   * It does its own task slicing. Reduce() hands one (token, head) at a time; the batched
    //     form needs runs of consecutive live tasks, because dnNum walks one uniform source
    //     stride and a fused token's id gap has to end a run. So it takes the whole share and
    //     the predicate, and batches internally.
    //   * The two vector subcores split each batch's rows rather than the share's tasks. The
    //     ceil/floor halving here is the Fixpipe's, not this loop's.
    //
    // IsFused is read through the same contextLenGm_ Reduce() reads, so a token this skips is
    // exactly a token that one skips.
    struct GemmCombineSelect {
        // Non-const: GlobalTensor::GetValue is not a const member in this toolkit.
        TurboQuantFusedDecodeOpt *op;
        __aicore__ inline bool NeedsCombine(const uint32_t token) const
        {
            return !op->IsFused(op->contextLenGm_.GetValue(token));
        }
    };

    template <typename GemmReducer>
    __aicore__ inline void ReduceGemm(GemmReducer &reducer, const uint32_t reduceTasksPerBlock)
    {
        GemmCombineSelect select{this};
        reducer.Reduce(numTokens_ * numHeads_, reduceTasksPerBlock, select, basis_);
    }

    // The attention phase's transient UB, for the reducer to alias its panels onto. Handed out
    // as a reference rather than an address so the reducer's own static_assert is what checks
    // it is long enough.
    __aicore__ inline auto &ReduceGemmSharedPool() { return vector_.TransientPool(); }
#endif

private:
    struct TileCursor {
        uint32_t block = 0;
        uint32_t rows = 0;
        uint32_t base = 0;
        uint32_t physical = 0;
        uint32_t valid = 0;
        bool active = false;
    };

    // Everything a phase of either half needs that is not its phase index. The cursors advance across
    // phases, so they are carried by reference; holding the rest alongside them keeps an unrolled phase
    // from reloading the task's extents from the frame on every tile.
    struct PipelineTask {
        TileCursor read;
        TileCursor consume;
        uint32_t token = 0;
        uint32_t contextLen = 0;
        uint32_t blockEnd = 0;
        uint32_t numTiles = 0;
        uint32_t kvHead = 0;
        uint32_t tile = 0;
    };

    __aicore__ inline bool IsFused(const int32_t contextLen) const
    {
        return numSplits_ <= 1 || contextLen <= static_cast<int32_t>(fusedContextLimit_);
    }

    // THE TASK'S ONE JOIN. Every path out of a task leaves through the DrainFusedDecodeTask()
    // below, including the paths that do no work at all:
    //
    //   * the duplicate splits of a FUSED token, which this used to leave by a bare `return`
    //     in front of the join -- symmetric across the roles, because both read the same
    //     contextLen word, but the one task exit that did not pass the rendezvous;
    //   * an UNMAPPED split, whose block range is empty because blockStart >= blockEnd;
    //   * a split whose every block-table entry is negative, so numTiles == 0 and neither role
    //     may touch a pipeline flag.
    //
    // The last two already fell through to the join; the first does now. The cost is one
    // mode-2 rendezvous per skipped split of a fused token, and what it buys is that the
    // per-task flag protocol is identical on every exit: the trace auditor's
    // dispatch == completion count then holds over the whole task stream without a case
    // analysis, and a future divergence between the two roles' view of `fused` cannot leave
    // one of them one task ahead of the other. RunTask holds the work and has no join.
    __aicore__ inline void ComputeTask(const uint32_t token, const uint32_t kvHead, const uint32_t split,
                                       const uint32_t chunk, const int32_t contextOverride = -1)
    {
        const int32_t fullContextLen = contextLenGm_.GetValue(token);
        const int32_t contextLen = contextOverride >= 0 ? contextOverride : fullContextLen;
        const bool fused = IsFused(fullContextLen);
        if (!fused || split == 0) {
            RunTask(token, kvHead, split, chunk, contextLen, fused);
        }
        DrainFusedDecodeTask();
    }

    __aicore__ inline void RunTask(const uint32_t token, const uint32_t kvHead, const uint32_t split,
                                   const uint32_t chunk, const int32_t contextLen, const bool fused)
    {
        const uint32_t offset = chunk * headsPerTask_;
        const uint32_t rows = groupHeads_ - offset < headsPerTask_ ? groupHeads_ - offset : headsPerTask_;
        const TurboQuantTaskHeads heads = Vector::Heads(kvHead * groupHeads_ + offset, rows);

        if ASCEND_IS_AIV {
            // Nothing drains the vector unit between tasks, and the previous task's vector work may still
            // read the query buffer and an ingest slot. This task's reads must land behind it.
            SyncVectorToMte2();
            vector_.BeginTask(heads);
        }

        const uint32_t seqBlocks = contextLen > 0 ? CeilDiv(static_cast<uint32_t>(contextLen), blockSize_) : 0;
        const uint32_t blocksPerSplit = CeilDiv(seqBlocks, fused ? 1u : numSplits_);
        const uint32_t blockStart = split * blocksPerSplit;
        uint32_t blockEnd = blockStart + blocksPerSplit;
        if (blockEnd > seqBlocks) {
            blockEnd = seqBlocks;
        }

        // The split's block table, fetched once, here, by both roles. This is the earliest
        // point at which the range exists: BeginTask runs before the split's extent is known
        // and on the vector role only, while the AIC needs the same entries because CountTiles
        // is the one predicate both roles branch on.
        PrefetchBlockTable(token, blockStart, blockEnd);

        if (blockStart < blockEnd) {
            PipelineTask task;
            task.token = token;
            task.contextLen = static_cast<uint32_t>(contextLen);
            task.blockEnd = blockEnd;
            task.kvHead = kvHead;
            task.numTiles = CountTiles(token, task.contextLen, blockStart, blockEnd);
            task.read.block = blockStart;
            task.consume.block = blockStart;

            if ASCEND_IS_AIV {
                vector_.StageQuery(mm_, token, kvHead, heads);
                const uint32_t tail = task.contextLen % kCubeTileRows;
                if (heads.mine > 0 && blockEnd == seqBlocks && tail != 0) {
                    vector_.ComputeTailMask(tail);
                }
            }
            // The one condition both cores branch on, so the bypass stays lock-step. A block range can
            // hold no tile at all -- every blockTable entry in it negative, which is what a padding or an
            // unmapped block carries -- and then neither core may touch a flag: an AIV that skipped the
            // pipeline while the AIC still entered it would leave the AIC waiting on a slot nothing is
            // ever going to stage, which is a hung launch rather than a wrong answer. CountTiles is safe
            // to branch on here because both cores run it over the same GM words with the same indices.
            if (task.numTiles > 0) {
                if ASCEND_IS_AIV {
                    RunVectorPipeline(task, heads);
                }
                if ASCEND_IS_AIC {
                    RunCubePipeline(task, heads.rows);
                }
            }
        }

        if ASCEND_IS_AIV {
            vector_.StageTaskOutput(token, split, heads, fused, basis_);
        }
    }

    // THE BLOCK TABLE, OUT OF THE PIPELINE.
    //
    // CountTiles and NextTile both used to read blockTableGm_ with GetValue -- a scalar load
    // from Global Memory -- and NextTile runs INSIDE the steady loop, once per tile on each of
    // the two cursors. So one logical page's entry was fetched three times per task (the count,
    // on both roles; the read cursor; the consume cursor) and two of those three fetches sat
    // between a softmax and the next tile's stage, on the scalar queue, inside the dependence
    // chain the three-zone schedule exists to keep full.
    //
    // The task prologue now sweeps [blockStart, blockEnd) once into this scalar cache and both
    // walkers read local memory. The sweep is still GetValue rather than a DataCopy into UB on
    // purpose: the Cube GEMM reduction's overlay already spends the only transient UB this
    // operator has, and a UB landing would make the AIC walk a TPipe allocation the AIV does
    // not -- the fault that made that reduction's first Cube run target an unallocated tensor.
    //
    // Both roles prefetch, identically, over the same GM words in the same order, so the
    // numTiles both cores branch on comes from one read each rather than from two reads that
    // could in principle see different bytes.
    //
    // A split longer than the cache falls back to the direct read, per block, exactly as the
    // old code always did -- correct, and slow in the same way. 64 entries covers every
    // geometry the validation ladder and the benchmark's split sweeps reach: 16384 context
    // tokens at 128-token pages is 128 blocks, so splits >= 2 is inside the cache and
    // splits == 1 at that context is the one sweep row that falls back. 256 bytes of scalar
    // local memory, which is why it is 64 entries and not 256.
    static constexpr uint32_t kBlockCacheEntries = 64;

    __aicore__ inline void PrefetchBlockTable(const uint32_t token, const uint32_t blockStart,
                                              const uint32_t blockEnd)
    {
        blockCacheBase_ = blockStart;
        if (blockEnd <= blockStart) {
            blockCacheCount_ = 0;
            return;
        }
        const uint32_t span = blockEnd - blockStart;
        blockCacheCount_ = span < kBlockCacheEntries ? span : kBlockCacheEntries;
        const uint64_t row = static_cast<uint64_t>(token) * maxBlocksPerSeq_;
        for (uint32_t i = 0; i < blockCacheCount_; ++i) {
            blockCache_[i] = blockTableGm_.GetValue(row + blockStart + i);
        }
    }

    __aicore__ inline int32_t BlockTableEntry(const uint32_t token, const uint32_t block)
    {
        if (block >= blockCacheBase_ && block - blockCacheBase_ < blockCacheCount_) {
            return blockCache_[block - blockCacheBase_];
        }
        return blockTableGm_.GetValue(static_cast<uint64_t>(token) * maxBlocksPerSeq_ + block);
    }

    __aicore__ inline uint32_t CountTiles(const uint32_t token, const uint32_t contextLen, const uint32_t blockStart,
                                          const uint32_t blockEnd)
    {
        uint32_t tiles = 0;
        for (uint32_t block = blockStart; block < blockEnd; ++block) {
            if (probePage_ != UINT32_MAX && block != probePage_) continue;
            const int32_t physical = BlockTableEntry(token, block);
            if (physical < 0) {
                continue;
            }
            uint32_t rows = blockSize_;
            const uint32_t consumed = block * blockSize_;
            if (consumed + rows > contextLen) {
                rows = contextLen - consumed;
            }
            if (probePage_ == UINT32_MAX) tiles += CeilDiv(rows, kCubeTileRows);
            else if (probeTile_ * kCubeTileRows < rows) ++tiles;
        }
        return tiles;
    }

    __aicore__ inline bool NextTile(TileCursor &cursor, const uint32_t token, const uint32_t contextLen,
                                    const uint32_t blockEnd)
    {
        if (cursor.active) {
            cursor.base += kCubeTileRows;
            if (cursor.base < cursor.rows && probePage_ == UINT32_MAX) {
                cursor.valid = cursor.rows - cursor.base;
                if (cursor.valid > kCubeTileRows) {
                    cursor.valid = kCubeTileRows;
                }
                return true;
            }
            cursor.active = false;
            ++cursor.block;
        }
        while (cursor.block < blockEnd) {
            if (probePage_ != UINT32_MAX && cursor.block != probePage_) { ++cursor.block; continue; }
            const int32_t physical = BlockTableEntry(token, cursor.block);
            if (physical >= 0) {
                uint32_t rows = blockSize_;
                const uint32_t consumed = cursor.block * blockSize_;
                if (consumed + rows > contextLen) {
                    rows = contextLen - consumed;
                }
                if (rows > 0) {
                    cursor.physical = static_cast<uint32_t>(physical);
                    cursor.rows = rows;
                    cursor.base = probePage_ == UINT32_MAX ? 0 : probeTile_ * kCubeTileRows;
                    cursor.valid = rows - cursor.base < kCubeTileRows ? rows - cursor.base : kCubeTileRows;
                    cursor.active = true;
                    return true;
                }
            }
            ++cursor.block;
        }
        return false;
    }

    // Short tasks exist in isolated probes and empty/short Split-K workers.
    // Their bounded dispatch is outside the three-zone fast path.
    __aicore__ inline void RunCubeShortPipeline(PipelineTask &, const uint32_t rows,
                                               const uint32_t count)
    {
        CubeScores<0>(rows);
        if (count > 1) CubeScores<1>(rows);
        CubeContext<0>(rows);
        if (count > 2) CubeScores<2>(rows);
        if (count > 1) CubeContext<1>(rows);
        if (count > 2) CubeContext<2>(rows);
    }

    __aicore__ inline void RunVectorShortPipeline(PipelineTask &task, const TurboQuantTaskHeads &heads)
    {
        ReadPipelineTile<0>(task);
        if (task.numTiles > 1) ReadPipelineTile<1>(task);
        StageToL1(task, 0);
        if (task.numTiles > 2) ReadPipelineTile<2>(task);
        if (task.numTiles > 1) StageToL1(task, 1);
        VectorSoftmax<0>(task, heads);
        vector_.ComputeAccumulatorDecay(heads);
        if (task.numTiles > 2) StageToL1(task, 2);
        VectorAccumulate(heads);
        if (task.numTiles > 1) {
            VectorSoftmax<1>(task, heads);
            vector_.ComputeAccumulatorDecay(heads);
            VectorAccumulate(heads);
        }
        if (task.numTiles > 2) {
            VectorSoftmax<2>(task, heads);
            vector_.ComputeAccumulatorDecay(heads);
            VectorAccumulate(heads);
        }
    }

    // Deep tasks have three zones. Short/empty workers retain the bounded
    // compatibility path; no boundary predicate enters a steady phase.
    __aicore__ inline void RunCubePipeline(PipelineTask &task, const uint32_t rows)
    {
        if (task.numTiles < kSlots + 1) {
            RunCubeShortPipeline(task, rows, task.numTiles);
            return;
        }
        // Slot 2 has never been staged. Prime its initial free credit so even
        // phase zero uses the same unconditional stage wait as every wrap.
        ReleaseCubeSlot<2>();
        CubeScores<0>(rows); // tick -1; ticks -3/-2 belong to MTE2/MTE3.
        const uint32_t steady_limit = task.numTiles - kSlots;
        while (task.tile + kPhases <= steady_limit) {
            RunCubeSteadyPhase<0>(rows);
            RunCubeSteadyPhase<1>(rows);
            RunCubeSteadyPhase<2>(rows);
            RunCubeSteadyPhase<3>(rows);
            RunCubeSteadyPhase<4>(rows);
            RunCubeSteadyPhase<5>(rows);
            task.tile += kPhases;
        }
        RunCubeSteadyTail<0>(rows, steady_limit - task.tile);
    }

    __aicore__ inline void RunVectorPipeline(PipelineTask &task, const TurboQuantTaskHeads &heads)
    {
        if (task.numTiles < kSlots + 1) {
            RunVectorShortPipeline(task, heads);
            return;
        }
        // Exactly three fill ticks; no length checks in the prologue.
        ReadPipelineTile<0>(task);                         // -3: read T0
        ReadPipelineTile<1>(task); StageToL1(task, 0);     // -2: read T1, stage T0
        ReadPipelineTile<2>(task); StageToL1(task, 1);     // -1: read T2, stage T1
        const uint32_t steady_limit = task.numTiles - kSlots;
        while (task.tile + kPhases <= steady_limit) {
            RunVectorSteadyPhase<0>(task, heads);
            RunVectorSteadyPhase<1>(task, heads);
            RunVectorSteadyPhase<2>(task, heads);
            RunVectorSteadyPhase<3>(task, heads);
            RunVectorSteadyPhase<4>(task, heads);
            RunVectorSteadyPhase<5>(task, heads);
            task.tile += kPhases;
        }
        RunVectorSteadyTail<0>(task, heads, steady_limit - task.tile);
    }

    template <uint32_t PHASE>
    __aicore__ inline void CubeScores(const uint32_t rows)
    {
        constexpr uint32_t kSlot = PHASE % kSlots;
        constexpr uint32_t kWindow = PHASE % kWindows;
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagSlotReady + kSlot));
        mm_.template GemmScores<true>(vector_.Scores(kWindow), mm_.L1Key(kSlot), rows, headSize_, kCubeTileRows);
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(
            static_cast<uint16_t>(kOptFlagScoresReady + kWindow));
    }

    template <uint32_t PHASE>
    __aicore__ inline void CubeContext(const uint32_t rows)
    {
        constexpr uint32_t kSlot = PHASE % kSlots;
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagProbsReady));
        mm_.template GemmContext<true>(vector_.Context(), mm_.L1Value(kSlot), rows, kCubeTileRows, headSize_);
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(static_cast<uint16_t>(kOptFlagContextReady));
    }

    template <uint32_t PHASE>
    __aicore__ inline void ReleaseCubeSlot()
    {
        constexpr uint32_t kSlot = PHASE % kSlots;
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(
            static_cast<uint16_t>(kOptFlagSlotFree + kSlot));
    }

    template <uint32_t PHASE>
    __aicore__ inline void ReadPipelineTile(PipelineTask &task)
    {
        constexpr uint32_t kSlot = PHASE % kSlots;
        NextTile(task.read, task.token, task.contextLen, task.blockEnd);
        vector_.ReadTileToSlot(task.read.physical, task.read.base, task.kvHead, kSlot);
    }

    template <uint32_t PHASE>
    __aicore__ inline void VectorSoftmax(PipelineTask &task, const TurboQuantTaskHeads &heads)
    {
        constexpr uint32_t kSlot = PHASE % kSlots;
        constexpr uint32_t kWindow = PHASE % kWindows;
        NextTile(task.consume, task.token, task.contextLen, task.blockEnd);
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagScoresReady + kWindow));
        vector_.ComputeSoftmaxAndStageProbs(mm_, task.consume.valid, heads, kSlot, kWindow);
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(static_cast<uint16_t>(kOptFlagProbsReady));
    }

    __aicore__ inline void VectorAccumulate(const TurboQuantTaskHeads &heads)
    {
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagContextReady));
        vector_.ComputeAccumulate(heads);
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunCubeSteadyPhase(const uint32_t rows)
    {
        CubeScores<PHASE + 1>(rows);
        CubeContext<PHASE>(rows);
        ReleaseCubeSlot<PHASE>();
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunVectorSteadyPhase(PipelineTask &task, const TurboQuantTaskHeads &heads)
    {
        constexpr uint32_t kStageSlot = (PHASE + 2) % kSlots;
        VectorSoftmax<PHASE>(task, heads);
        SyncVectorToMte2();
        ReadPipelineTile<PHASE>(task);
        vector_.ComputeAccumulatorDecay(heads);
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagSlotFree + kStageSlot));
        StageToL1(task, kStageSlot);
        VectorAccumulate(heads);
    }

    // The remainder dispatch selects the drain's phase statically as well.
    // It never resets phase to zero at N-3, including all five ragged residues.
    template <uint32_t PHASE>
    __aicore__ inline void RunCubeSteadyTail(const uint32_t rows, const uint32_t remaining)
    {
        if constexpr (PHASE < kPhases - 1) {
            if (PHASE < remaining) {
                RunCubeSteadyPhase<PHASE>(rows);
                RunCubeSteadyTail<PHASE + 1>(rows, remaining);
                return;
            }
        }
        RunCubeDrain<PHASE>(rows);
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunVectorSteadyTail(PipelineTask &task, const TurboQuantTaskHeads &heads,
                                              const uint32_t remaining)
    {
        if constexpr (PHASE < kPhases - 1) {
            if (PHASE < remaining) {
                RunVectorSteadyPhase<PHASE>(task, heads);
                RunVectorSteadyTail<PHASE + 1>(task, heads, remaining);
                return;
            }
        }
        RunVectorDrain<PHASE>(task, heads);
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunCubeDrain(const uint32_t rows)
    {
        CubeScores<PHASE + 1>(rows); CubeContext<PHASE>(rows);     // N-3: no refill credit
        CubeScores<PHASE + 2>(rows); CubeContext<PHASE + 1>(rows); // N-2
        CubeContext<PHASE + 2>(rows);                            // N-1: no scores
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunVectorDrain(PipelineTask &task, const TurboQuantTaskHeads &heads)
    {
        constexpr uint32_t kStageSlot = (PHASE + 2) % kSlots;
        VectorSoftmax<PHASE>(task, heads);                       // N-3: no read
        vector_.ComputeAccumulatorDecay(heads);
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagSlotFree + kStageSlot));
        StageToL1(task, kStageSlot);
        VectorAccumulate(heads);
        VectorSoftmax<PHASE + 1>(task, heads);                   // N-2: no stage
        vector_.ComputeAccumulatorDecay(heads);
        VectorAccumulate(heads);
        VectorSoftmax<PHASE + 2>(task, heads);                   // N-1: final softmax/context
        vector_.ComputeAccumulatorDecay(heads);
        VectorAccumulate(heads);
    }

    // The slot's MTE2 read is already posted on the slot's own event id; StageSlotToL1 waits exactly that
    // one, unpacks the plane in UB in NZ order and bursts it into L1. Both subcores set the slot's token,
    // because the AIC's single wait needs one from each of them.
    __aicore__ inline void StageToL1(PipelineTask &task, const uint32_t slot)
    {
        vector_.StageSlotToL1(mm_, task.kvHead, slot);
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(static_cast<uint16_t>(kOptFlagSlotReady + slot));
    }

    uint32_t probePage_ = UINT32_MAX, probeTile_ = 0;
    // The prefetched block table of the task in flight. blockCacheCount_ == 0 means every
    // lookup falls back to GM, which is what an empty or unmapped split leaves behind.
    int32_t blockCache_[kBlockCacheEntries] = {};
    uint32_t blockCacheBase_ = 0;
    uint32_t blockCacheCount_ = 0;
    AscendC::TPipe *pipe_;
    Mm mm_;
    Vector vector_;
    AscendC::GlobalTensor<int32_t> blockTableGm_;
    AscendC::GlobalTensor<int32_t> contextLenGm_;
    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t groupHeads_ = 1;
    uint32_t headsPerTask_ = 1;
    uint32_t chunksPerGroup_ = 1;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t fusedContextLimit_ = 0;
    // Last, so a pre-rotated decode's members keep their offsets; empty when PRE_ROTATED.
    Basis basis_;
};

}

// ------------------------------------------------------------------------------------------------
// The reduction phase, in one place for both launch macros
// ------------------------------------------------------------------------------------------------
//
// Default: the shipping vector reducer, guarded by `if ASCEND_IS_AIV` because the Cube has no
// part in it -- which is the idleness the Cube GEMM form spends.
//
// ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1 selects TurboQuantBatchedGemmReducer instead: M = 16
// packed query rows against 16-channel slices of D, one dense Mmad per slice, the product split
// between the two vector subcores by a dual-destination Fixpipe. Three differences are visible
// right here and all three are load-bearing:
//
//   * NO `if ASCEND_IS_AIV`. The Cube does the contraction, and both roles have to walk the
//     same TPipe allocation because the Fixpipe's destination is an address in the vector
//     cores' UB. Guarding this is what made the first Cube run of the probe target an
//     unallocated tensor -- su_ccu_mpu_err_t0 on cubecore0 -- while still printing the right
//     numbers.
//   * op.ReduceGemm() rather than op.Reduce(): the batched form slices its own share, because
//     dnNum walks one uniform source stride and a fused token's task-id gap has to end a run.
//   * The reducer is constructed AFTER SyncAll<false>(), as before, but it is not free: see the
//     UB deficit below.
//
// THE SWITCH'S UB COMES FROM AN OVERLAY, NOT FROM HEADROOM. The reducer's panels do not fit
// what this operator has left: it claims 27264 UB bytes against the vector reducer's 3360, a
// net 23904, and only 4256 are free under the 192 KB window -- a 19648-byte deficit. So the
// reducer allocates from a TBufPool aliased onto the attention phase's transient pool, which
// SyncAll<false>() above has made dead. The alias costs zero bytes: InitBufPool's shared
// overload sets the new pool's startAddr_ to the shared one's and does not advance the hardware
// pool. include/turboquant_gemm_reduce.hpp carries the arithmetic and the static_assert that
// the reducer fits inside that pool.
#if defined(ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE) && ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE
#include "fused_decode/cube/turboquant_gemm_reduce_service.h"

// OVERLAY SAFETY: what is checked here, and the one thing no compiler can check.
//
// ReduceGemm's two panel slots live in a TBufPool aliased onto the attention phase's transient
// pool -- kUbSizes[15..17], which are nzOperandBuf_ (the tile ingest's NZ staging operand),
// accBuf_ (the per-task online-softmax accumulator) and queryWorkBuf_ (the query staging). The
// reduction therefore WRITES the same UB bytes the attention phase read. Four checks below,
// and all four are extent checks:
//
//   1. the pool's length is what the contract header says it is, so the two headers that
//      restate it cannot drift (this was the only check here before);
//   2. the reducer's UB claim fits inside the pool the OPERATOR allocates -- kTransientPoolBytes,
//      not the reducer's own restatement of it, so this holds even while the restatement is
//      being changed, and it is the use-site counterpart of the reducer header's own fit
//      assert. (The extent itself needs no assert here: kTransientPoolBytes is DEFINED as
//      kUbSizes[15] + kUbSizes[16] + kUbSizes[17] in the contract header and pinned to 32768
//      there, and check 1 ties the reducer's copy to it. A pool one allocation longer would
//      move kUbEnd and every offset after it, and the Cube's dual-destination Fixpipe windows
//      at UB offsets 0 and kScoreBytes are addressed by offset on three cores with nothing in
//      the toolchain checking that they agree.)
//   3. the launch validator rejects a split count the reduction's partial fractal cannot
//      hold. kMaxSplits is baked into the fractal geometry (one 8-float C0 per query row), so
//      a valid_launch that admitted splits > kMaxSplits would hand the reducer a source
//      stride it has no lane for. valid_launch is constexpr, so the bound is checked by
//      calling it rather than by restating it.
//
// NOT CHECKED, AND NOT CHECKABLE HERE: that those three buffers are DEAD when the reduction
// starts. Nothing in the toolchain verifies liveness. What establishes it is the
// AscendC::SyncAll<false>() in the launch macros -- every block's attention phase has retired
// before any block reduces -- plus the fact that no method reachable from ReduceGemm reads
// them. A future change that reads nzOperandBuf_, accBuf_ or queryWorkBuf_ after the attention
// phase breaks this SILENTLY, with no diagnostic and correct-looking output at geometries that
// do not reduce. The gate that would catch it is a multi-split CAModel rung with its canary
// and exact-comparison checks -- section 9 of
// reports/gemm_batched_reduction_camodel_20261104.md -- not this translation unit.
static_assert(ascend_micro::fused_decode_opt::kTransientPoolBytes ==
                  ascend_micro::gemm_reduce::kGemmSharedPoolBytes,
              "turboquant_gemm_reduce.hpp restates the transient pool's length; update "
              "kGemmSharedPoolBytes there to match kTransientPoolBytes");
static_assert(ascend_micro::gemm_reduce::kGemmReducerUbBytes <=
                  ascend_micro::fused_decode_opt::kTransientPoolBytes,
              "the batched reducer's panels do not fit the attention phase's transient pool");
static_assert(ascend_micro::fused_decode_opt::valid_launch(
                  1, 1, 16, 1, 256, 128, 8, ascend_micro::gemm_reduce::kMaxSplits, 16, 8, 16),
              "the launch validator refuses the reduction's own maximum split count");
static_assert(!ascend_micro::fused_decode_opt::valid_launch(
                  1, 1, 16, 1, 256, 128, 8, ascend_micro::gemm_reduce::kMaxSplits + 1, 16, 9, 16),
              "the launch validator admits more splits than the Cube GEMM reduction's partial "
              "fractal has lanes for");

#define ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE(TYPE)                                                                    \
    do {                                                                                                             \
        ascend_micro::gemm_reduce::TurboQuantBatchedGemmReducer<TYPE> reducer(&pipe);                                 \
        reducer.Init(workspace, output, numHeads, headSize, numSplits, op.ReduceGemmSharedPool());                    \
        op.ReduceGemm(reducer, reduceTasksPerBlock);                                                                  \
    } while (0)
#else
// THE DEFAULT PHASE'S ROLE CONTRACT, and why its asymmetry is not a hang.
//
// The reducer object and the Reduce() call stay behind `if ASCEND_IS_AIV`: the Cube has no part
// in the vector recurrence, and -- unlike the Cube GEMM form above -- the reducer's destination
// is the vector cores' own UB, so the two roles do not have to walk the same TPipe allocation.
// The AIC allocates nothing and does nothing in this phase.
//
// That asymmetry was audited for a cross-core join deadlock and does not have one:
//
//   * Both roles reach the launch macro's DrainFusedDecodeTask() after this phase, and that
//     drain is a balanced two-phase rendezvous on ONE flag id. CrossCoreSetFlag(ISASI) mode 2
//     (CANN 9.2, Ascend 950PR): the AIC's set releases the CrossCoreWaitFlag on BOTH AIVs, and
//     the two AIVs' sets TOGETHER release the AIC's wait. So one AIC set and one set from each
//     subcore is exactly the balance DrainFusedDecodeTask() posts, in both directions.
//   * The AIC therefore announcing its join while the AIVs are still reducing is the intended
//     order, not a race: the AIC then blocks in its own wait until both AIVs have finished
//     reducing AND drained MTE3, which the AIV leg's SyncEvent<MTE3_S>() in front of its set is
//     what guarantees.
//   * Nothing on TurboQuantPartialReducer's path takes a cross-core flag or a SyncAll of its
//     own -- its only barrier is a PipeBarrier inside its own Init -- so the AIC's absence
//     cannot block the AIVs either.
//
// What a future change must preserve is that last bullet. A cross-core primitive introduced
// anywhere under op.Reduce() turns this guard into a hang, and the gate that would show it is
// a multi-split CAModel rung and its trace audit, not a build. The join itself stays in the
// launch macros, once, after the phase, so both forms of this macro are followed by the same
// rendezvous and neither has to post one of its own.
#define ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE(TYPE)                                                                    \
    do {                                                                                                             \
        if ASCEND_IS_AIV {                                                                                           \
            TurboQuantPartialReducer<TYPE> reducer(&pipe);                                                           \
            reducer.Init(workspace, output, numHeads, headSize, numSplits);                                          \
            op.Reduce(reducer, reduceTasksPerBlock);                                                                 \
        }                                                                                                            \
    } while (0)
#endif

#define ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT(NAME, MODE, TYPE)                                                      \
    extern "C" __global__ __aicore__ void NAME(                                                                      \
        GM_ADDR queryRot, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables,             \
        GM_ADDR contextLens, GM_ADDR modeTables, GM_ADDR workspace, GM_ADDR output, uint32_t numTokens,              \
        uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,      \
        uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,              \
        uint32_t fusedContextLimit, float scale, float invSqrtLen)                                                   \
    {                                                                                                                \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); \
        AscendC::InitSocState(); \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantFusedDecodeOpt<MODE, TYPE> op(&pipe);                                                               \
        op.Init(queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,  \
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,       \
                fusedContextLimit, scale, invSqrtLen);                                                                \
        op.Process(tasksPerBlock);                                                                                    \
        if (op.NeedsReduction()) {                                                                                    \
            AscendC::SyncAll<false>();                                                                                \
            ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE(TYPE);                                                                 \
        }                                                                                                             \
        DrainFusedDecodeTask(); \
    }

// The raw-query entry, argument for argument the same as the lock-step decode's, so a benchmark or an
// msprof run can swap one launcher for the other without touching its tiling.
#define ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT_RAW_QUERY(NAME, MODE, TYPE)                                            \
    extern "C" __global__ __aicore__ void NAME(                                                                      \
        GM_ADDR query, GM_ADDR piSigns, GM_ADDR rotTables, GM_ADDR h16, GM_ADDR gate, GM_ADDR queryRot,              \
        GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables, GM_ADDR contextLens,          \
        GM_ADDR modeTables, GM_ADDR workspace, GM_ADDR output, uint32_t numTokens, uint32_t numHeads,                \
        uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,     \
        uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,                                 \
        uint32_t prologueVectorsPerBlock, uint32_t prologueCubeChunkVectors, uint32_t outputStage,                   \
        uint32_t fusedContextLimit, float scale, float invSqrtLen)                                                   \
    {                                                                                                                \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); \
        AscendC::InitSocState(); \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantFusedDecodeOpt<MODE, TYPE, false> op(&pipe);                                                        \
        op.Init(queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,  \
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,       \
                fusedContextLimit, scale, invSqrtLen);                                                                \
        op.InitBasis(query, piSigns, rotTables, h16, gate, queryRot, prologueCubeChunkVectors, outputStage,           \
                     invSqrtLen);                                                                                     \
        op.RotateQuery(prologueVectorsPerBlock);                                                                       \
        op.Process(tasksPerBlock);                                                                                     \
        if (op.NeedsReduction()) {                                                                                     \
            AscendC::SyncAll<false>();                                                                                  \
            ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE(TYPE);                                                                  \
        }                                                                                                              \
        DrainFusedDecodeTask(); \
    }

// kv4fp8 only: the NZ-tiled ring ingest the schedule is built on, and the one mode the fused decode is
// validated in at all (TURBOQUANT_TESTS.md 13.25).
ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT(turboquant_mm_fused_decode_opt_kv4fp8_half, TurboQuantMode::KV4_FP8, half)
ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT_RAW_QUERY(turboquant_mm_fused_decode_opt_raw_query_kv4fp8_half,
                                                TurboQuantMode::KV4_FP8, half)

#elif defined(ASCEND_MICRO_CANN_LAUNCH)
#include "kernel_operator.h"
#include <stdexcept>
extern "C" __global__ __aicore__ void turboquant_mm_fused_decode_opt_kv4fp8_half(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, float, float);
extern "C" __global__ __aicore__ void turboquant_mm_fused_decode_opt_raw_query_kv4fp8_half(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float);

namespace vllm_ascend {

void turboquant_mm_fused_decode_opt_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim,
                                         void *queryRot, void *keyCache, void *valueCache, void *scaleCache,
                                         void *blockTables, void *contextLens, void *modeTables, void *workspace,
                                         void *output, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                         uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                         uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock,
                                         uint32_t reduceTasksPerBlock, uint32_t fusedContextLimit, float scale,
                                         float invSqrtLen)
{
    if (!ascend_micro::fused_decode_opt::valid_launch(blockDim, numTokens, numHeads,
            numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
            headsPerTask, tasksPerBlock, reduceTasksPerBlock))
        throw std::invalid_argument("unsupported fused decode geometry or incomplete task coverage");
    if (type != AscendType::FP16 || blockDim == 0 ||
        static_cast<turboquant::TurboQuantMode>(mode) != turboquant::TurboQuantMode::KV4_FP8) {
        return;
    }
    turboquant_mm_fused_decode_opt_kv4fp8_half<<<blockDim, nullptr, stream>>>(
        queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,
        numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,
        tasksPerBlock, reduceTasksPerBlock, fusedContextLimit, scale, invSqrtLen);
}

void turboquant_mm_fused_decode_opt_raw_query_impl(
    int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *query, void *piSigns, void *rotTables,
    void *h16, void *gate, void *queryRot, void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
    void *contextLens, void *modeTables, void *workspace, void *output, uint32_t numTokens, uint32_t numHeads,
    uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
    uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock, uint32_t prologueVectorsPerBlock,
    uint32_t prologueCubeChunkVectors, uint32_t outputStage, uint32_t fusedContextLimit, float scale,
    float invSqrtLen)
{
    if (uint64_t{blockDim} * prologueVectorsPerBlock < uint64_t{numTokens} * numHeads ||
        (prologueCubeChunkVectors != 0 && prologueCubeChunkVectors != 16) || outputStage > 2)
        throw std::invalid_argument("unsupported raw query basis tiling");
    if (!ascend_micro::fused_decode_opt::valid_launch(blockDim, numTokens, numHeads,
            numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
            headsPerTask, tasksPerBlock, reduceTasksPerBlock))
        throw std::invalid_argument("unsupported fused decode geometry or incomplete task coverage");
    if (type != AscendType::FP16 || blockDim == 0 ||
        static_cast<turboquant::TurboQuantMode>(mode) != turboquant::TurboQuantMode::KV4_FP8) {
        return;
    }
    turboquant_mm_fused_decode_opt_raw_query_kv4fp8_half<<<blockDim, nullptr, stream>>>(
        query, piSigns, rotTables, h16, gate, queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens,
        modeTables, workspace, output, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq,
        numSplits, headsPerTask, tasksPerBlock, reduceTasksPerBlock, prologueVectorsPerBlock,
        prologueCubeChunkVectors, outputStage, fusedContextLimit, scale, invSqrtLen);
}

}

// The raw-query entry, through the same convenience ABI. It exists because the only launcher
// this harness had called the PRE-ROTATED kernel, so nothing could reach the query prologue or
// the output stage at all -- neither a test nor a benchmark nor an msprof run.
extern "C" void launch_fused_decode_opt_raw_query(void* stream,
    const ascend_micro::fused_decode_opt::Launch* c,
    const ascend_micro::fused_decode_opt::RawQueryBuffers* b)
{
    if (!c || !b) throw std::invalid_argument("null fused decode raw-query launch descriptor");
    vllm_ascend::turboquant_mm_fused_decode_opt_raw_query_impl(4, vllm_ascend::AscendType::FP16,
        stream, c->grid, b->raw_query, b->pi_signs, b->rot_tables, b->h16, b->gate,
        b->decode.query, b->decode.key, b->decode.value, b->decode.scales, b->decode.block_table,
        b->decode.context_lengths, b->decode.mode_tables, b->decode.workspace, b->decode.output,
        c->tokens, c->heads, c->kv_heads, c->head_size, c->block_size, c->max_blocks, c->splits,
        c->heads_per_task, c->tasks_per_block, c->reduce_tasks_per_block,
        c->prologue_vectors_per_block, c->prologue_cube_chunk_vectors, c->output_stage,
        c->fused_context_limit, c->scale, c->inv_sqrt_len);
}

extern "C" void launch_fused_decode_opt(void* stream,
    const ascend_micro::fused_decode_opt::Launch* c,
    const ascend_micro::fused_decode_opt::Buffers* b)
{
    if (!c || !b) throw std::invalid_argument("null fused decode launch descriptor");
    vllm_ascend::turboquant_mm_fused_decode_opt_impl(4, vllm_ascend::AscendType::FP16,
        stream, c->grid, b->query, b->key, b->value, b->scales, b->block_table,
        b->context_lengths, b->mode_tables, b->workspace, b->output, c->tokens,
        c->heads, c->kv_heads, c->head_size, c->block_size, c->max_blocks, c->splits,
        c->heads_per_task, c->tasks_per_block, c->reduce_tasks_per_block,
        c->fused_context_limit, c->scale, c->inv_sqrt_len);
}

#endif
