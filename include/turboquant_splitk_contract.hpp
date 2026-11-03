#pragma once

#include <cstddef>
#include <cstdint>

namespace turboquant_splitk {
constexpr std::uint32_t kChannels = 256;
constexpr std::uint32_t kStateLanes = 8;
constexpr std::uint32_t kMaxRows = 8;
constexpr std::uint32_t kMaxSplits = 16;
constexpr std::uint32_t kRecordFloats = 272;

// One query row's partial softmax state. `m` and `l` are replicated across a
// whole 32-byte DataBlock -- the decode operators' replicated-state invariant,
// which is what lets the merge run with no scalar read of Unified Buffer --
// and `acc` is the unnormalized accumulator in the cache's rotated basis.
struct RowState {
    float m_global[kStateLanes];
    float l_global[kStateLanes];
    float acc_global[kChannels];
};

// ----------------------------------------------- the partition workspace
//
// The standardized intermediate layout both decode paths export into, and the
// only thing the reduction reads:
//
//     [requests][heads][splits][kMaxStateRows] of RowState
//
// A task owns one `[kMaxStateRows] of RowState` partition and nothing else, so
// every worker's write is a disjoint, conflict-free MTE3 burst at
// `workspace + item * kPartitionFloats` and no two workers ever address the
// same line. That is the whole point of the layout: it replaces the Global
// Memory test-and-set the first revision of the Vector path used, whose
// in-lock merge serialized every worker of a head through one cache line.
//
// The row bound is the Cube path's M, not the Vector path's. The two operators
// have different row bounds -- the Vector kernel's Unified Buffer arena is
// sized for `kMaxRows` because the dispatch rule sends rows >= 12 to the Cube
// operator -- and a shared stride has to hold the larger of them. A Vector
// task writes `rows * kRecordFloats` floats at the base of its partition and
// leaves the rest of it untouched; the reduction reads exactly as many.
constexpr std::uint32_t kMaxStateRows = 16;
constexpr std::uint32_t kPartitionFloats = kMaxStateRows * kRecordFloats;

// The grid bound an in-launch reduction imposes. `AscendC::SyncAll` is a
// barrier across the LAUNCHED blocks, and on dav-3510 it only completes if
// every one of them is resident at the same time: a grid that runs in two
// hardware waves parks wave 0 on the barrier while wave 1 has not been
// scheduled yet, and the launch hangs rather than returning a wrong answer.
// So a split launch, which is the only configuration that reaches the barrier,
// is capped at the number of blocks one hardware wave holds.
//
// WHAT A BLOCK IS, AND WHY THE BOUND IS 64 RATHER THAN 32
//
// The only operator that reaches this barrier is the Vector Split-K decode,
// and it is `KERNEL_TYPE_AIV_ONLY`. On an AIV-only launch a block IS a vector
// subcore, not an AI Core cluster: dav-3510 has 32 Cube cores and each of them
// fronts TWO vector subcores, so a single wave spans 64 independent blocks.
// The CAModel for `Ascend950PR_9599` enumerates exactly that -- 32 clusters of
// `cubecore0`, `veccore0`, `veccore1`, 96 cores in all, 64 of them AIV -- and
// `reports/decode_ring_pipeline/summary.json` records 95 idle cores beside the
// one this harness's single-block probes drive.
//
// 32 was the bound this header shipped with, and it came from the AI Core
// block count, which is the right number for a `KERNEL_TYPE_MIX_AIC_1_2`
// operator and the wrong one here. It cost half the machine and it put
// K = 16 at H = 4, B = 1 and K = 8 at B = 2 out of reach for no hardware
// reason. The barrier the Vector kernel raises is `SYNC_AIV_ONLY_ALL`
// (`SyncAll<true>`, the `isAIVOnly` specialization), which is the AIV-only
// participant set and has no Cube core in it to wait for.
//
// A MIX operator that ever wants an in-launch barrier needs the cluster count
// instead, so it does not get to borrow this constant: it is named for the
// engine whose wave it describes.
//
// A single-split launch never reaches the barrier -- it emits its context
// directly from the decode pass -- and so is not bounded here.
constexpr std::uint32_t kAivSubcoresPerCluster = 2;
constexpr std::uint32_t kAiCoreClusters = 32;
constexpr std::uint32_t kMaxAivBlocks = kAiCoreClusters * kAivSubcoresPerCluster;

#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr bool GridFitsOneWave(std::uint32_t requests, std::uint32_t heads,
                               std::uint32_t splits)
{
    return splits <= 1u ||
           static_cast<std::uint64_t>(requests) * heads * splits <= kMaxAivBlocks;
}

// Floats the host must allocate, and the task's own offset inside them. The
// task index is the block index of the split decode grid, which is why the
// partition axis is minor: `item = (request * heads + head) * splits + split`.
constexpr std::size_t WorkspaceFloats(std::uint32_t requests, std::uint32_t heads,
                                      std::uint32_t splits)
{
    return static_cast<std::size_t>(requests) * heads * splits * kPartitionFloats;
}

#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr std::uint64_t PartitionOffset(std::uint32_t head_item, std::uint32_t split,
                                        std::uint32_t splits)
{
    return (static_cast<std::uint64_t>(head_item) * splits + split) * kPartitionFloats;
}

// The worker census. Every field is an eight-lane replicated DataBlock, so the
// whole struct is one contiguous MTE3 burst of nine blocks written by the
// Vector pipe: there is no SIMT lane and no cache-bypassing store left in this
// protocol, because no worker reads another worker's control any more.
// Replication is also what makes a torn publication detectable -- a reader
// that finds two lanes of one field disagreeing saw a partial burst.
//
// `finish` is sampled after the partial-state MTE3 burst on an exporting
// worker, and after the final out_context burst on a worker that also reduced.
struct alignas(32) WorkerMetrics {
    // 1 once the worker published, either a partition or a fused context.
    std::uint32_t exported[8];
    // The partition this worker wrote, or kNoPartition on the fused path.
    std::uint32_t partition[8];
    // 1 when this block also ran a reduction task after the barrier.
    std::uint32_t reduced[8];
    // Logical pages this worker's split covered.
    std::uint32_t pages[8];
    // 1 on the worker that wrote this head's out_context: the reducer under
    // Split-K, the single decode worker when splits == 1.
    std::uint32_t tail[8];
    std::uint32_t finish_hi[8], finish_lo[8];
    // The non-blocking export region: state pack through retired MTE3 burst.
    // The direct replacement for the lock region the atomic protocol measured,
    // and the figure that shows the critical section is gone rather than moved.
    std::uint32_t export_hi[8], export_lo[8];
};

constexpr std::uint32_t kMetricBlocks = 9;
constexpr std::uint32_t kMetricLanes = kMetricBlocks * 8;

constexpr std::uint32_t kNoPartition = 0xFFFFFFFFu;

struct PageRange { std::uint32_t begin, count; };
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr PageRange Partition(std::uint32_t pages, std::uint32_t splits,
                              std::uint32_t split)
{
    const std::uint32_t width = (pages + splits - 1u) / splits;
    const std::uint32_t begin = split * width;
    const std::uint32_t remaining = begin < pages ? pages - begin : 0u;
    return {begin, remaining < width ? remaining : width};
}

// The grid bound as a compile-time object rather than a comment, so a later
// edit that renumbers it has to break a build. The three rungs below are the
// shapes the 64-block wave is for: K = 16 at H = 4 B = 1, K = 8 at B = 2, and
// the largest grid any legal operand triple can reach, which must still fit.
static_assert(kMaxAivBlocks == 64u, "dav-3510: 32 clusters x 2 AIV subcores");
static_assert(GridFitsOneWave(1, 4, 16), "K=16 at H=4, B=1 is a 64-block grid");
static_assert(GridFitsOneWave(2, 4, 8), "K=8 at H=4, B=2 is a 64-block grid");
static_assert(!GridFitsOneWave(2, 4, 16), "a 128-block split grid spans two waves");
static_assert(GridFitsOneWave(1, 4, 1) && GridFitsOneWave(64, 4, 1),
              "a single-split launch never reaches the barrier");
// The split axis alone can never overflow the wave, so a one-head, one-request
// launch is admissible at every K the ABI accepts.
static_assert(kMaxSplits <= kMaxAivBlocks && GridFitsOneWave(1, 1, kMaxSplits),
              "the maximum split count must fit one wave on its own");

static_assert(sizeof(RowState) == kRecordFloats * sizeof(float), "row ABI");
static_assert(kRecordFloats * sizeof(float) % 32 == 0,
              "a row record must be a whole number of 32-byte bursts");
static_assert(kPartitionFloats * sizeof(float) % 32 == 0,
              "a partition base must be 32-byte aligned");
static_assert(kMaxRows <= kMaxStateRows, "the shared stride must hold both paths");
static_assert(sizeof(WorkerMetrics) == kMetricLanes * sizeof(std::uint32_t),
              "metric ABI: nine replicated DataBlocks, no padding");
static_assert(sizeof(WorkerMetrics) % 32 == 0,
              "the census must be a whole number of replicated DataBlocks");
}  // namespace turboquant_splitk
