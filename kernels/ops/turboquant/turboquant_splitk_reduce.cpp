#include <cstdint>

// The Split-K reduction of the TurboQuant decode pipeline, for a PHYSICAL
// device. Ascend 950PR / arch35 / dav-3510 / CANN 9.2, KERNEL_TYPE_AIV_ONLY.
//
// Relationship to the CAModel probe
// ---------------------------------
// `kernels/camodel/l2_atomic_synthetic.cpp` is the cycle-accurate probe that
// answered where the reduction should live; its figures are in
// `reports/l2_atomic_contention_split_k.md`. This is the same two protocols
// written for silicon, and it is a separate translation unit on purpose:
//
//   * the probe's published instruction budget was measured against an
//     UNBOUNDED spin, so adding the watchdog this file needs would have
//     invalidated the numbers rather than extended them;
//   * hardware needs K up to 16, absolute timestamps for the tail, and a
//     watchdog; CAModel needs none of those and pays for all of them;
//   * the probe carries Mode C (the tail egress), which a reduction benchmark
//     does not exercise.
//
// Both share `routines/routine_probe_l2_atomic_sync.hpp`: the state record,
// the `L2AtomicProbeEngine` combine, and -- for this file -- the two
// watchdog-bounded SIMT entry points the routine adds additively.
//
// What it does NOT do, by design
// ------------------------------
// There is **no combine kernel**. Mode A writes each split's state to its own
// slice and stops; Mode B accumulates in place under a Global Memory
// spinlock, with the K splits contending for one line. That is the point: the
// benchmark is measuring L2 line bouncing and lock serialisation across
// physical cores, and a second pass that tidied up afterwards would be
// measured work that the question does not ask about.
//
// The watchdog, and what it guarantees
// ------------------------------------
// Every acquire is bounded by `max_attempts` failed exchanges. On expiry the
// split gives up WITHOUT the lock, records that it did, and the release skips
// clearing a lock it never held. So a launch cannot hang: the worst case is
// `max_attempts` atomic round trips per acquisition, after which every split
// retires having published a timeout the host can see. A run with any timeout
// is not a valid measurement and the benchmark says so rather than averaging
// it in.
//
// `max_attempts == 0` disables the bound and reproduces the probe exactly,
// which is what a CAModel comparison run uses.

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "../../../routines/routine_probe_l2_atomic_sync.hpp"

namespace {

using ascend_micro::kProbeAccOffset;
using ascend_micro::kProbeChannels;
using ascend_micro::kProbeLOffset;
using ascend_micro::kProbeMetricElems;
using ascend_micro::kProbeMetricLanesPerSlot;
using ascend_micro::kProbeMOffset;
using ascend_micro::kProbeRecordElems;
using ascend_micro::kProbeStateElems;
using ascend_micro::kProbeStateLanes;
using ascend_micro::kSplitKMaxSplits;

// ---------------------------------------------------------------- UB arena
// Rule AKA3003, as everywhere in this harness: every operand base 32-byte
// aligned, a DataBlock of bank skew between live slots, and a vector
// register of addressable UB behind the last operand.
constexpr std::uint32_t kBlock = 32u;
constexpr std::uint32_t kBankSkew = 32u;
constexpr std::uint32_t kVecReg = 256u;
constexpr std::uint32_t kUbWindow = 192u * 1024u;
constexpr std::uint32_t kF32 = 4u;

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkew;
}

constexpr std::uint32_t kStatePairBytes = kProbeStateElems * kF32;  // 64
constexpr std::uint32_t kChannelBytes = kProbeChannels * kF32;      // 1024
constexpr std::uint32_t kLaneBytes = kProbeStateLanes * kF32;       // 32

constexpr std::uint32_t kUbStateTile = 0u;
constexpr std::uint32_t kUbAccTile = NextSlot(kUbStateTile, kStatePairBytes);
constexpr std::uint32_t kUbStateCurr = NextSlot(kUbAccTile, kChannelBytes);
constexpr std::uint32_t kUbAccCurr = NextSlot(kUbStateCurr, kStatePairBytes);
constexpr std::uint32_t kUbStateNew = NextSlot(kUbAccCurr, kChannelBytes);
constexpr std::uint32_t kUbAccNew = NextSlot(kUbStateNew, kStatePairBytes);
constexpr std::uint32_t kUbAlphaCurr = NextSlot(kUbAccNew, kChannelBytes);
constexpr std::uint32_t kUbAlphaTile = NextSlot(kUbAlphaCurr, kLaneBytes);
constexpr std::uint32_t kUbStateTmp = NextSlot(kUbAlphaTile, kLaneBytes);
constexpr std::uint32_t kUbStateTmp2 = NextSlot(kUbStateTmp, kLaneBytes);
constexpr std::uint32_t kUbAlphaCurrBc = NextSlot(kUbStateTmp2, kLaneBytes);
constexpr std::uint32_t kUbAlphaTileBc = NextSlot(kUbAlphaCurrBc, kChannelBytes);
constexpr std::uint32_t kUbTermA = NextSlot(kUbAlphaTileBc, kChannelBytes);
constexpr std::uint32_t kUbTermB = NextSlot(kUbTermA, kChannelBytes);
constexpr std::uint32_t kUbMetrics = NextSlot(kUbTermB, kChannelBytes);
constexpr std::uint32_t kArenaEnd = kUbMetrics + kProbeMetricElems * kF32;

static_assert(kUbStateTile % kBlock == 0 && kUbAccTile % kBlock == 0 &&
                  kUbStateCurr % kBlock == 0 && kUbAccCurr % kBlock == 0 &&
                  kUbStateNew % kBlock == 0 && kUbAccNew % kBlock == 0 &&
                  kUbAlphaCurr % kBlock == 0 && kUbAlphaTile % kBlock == 0 &&
                  kUbStateTmp % kBlock == 0 && kUbStateTmp2 % kBlock == 0 &&
                  kUbAlphaCurrBc % kBlock == 0 && kUbAlphaTileBc % kBlock == 0 &&
                  kUbTermA % kBlock == 0 && kUbTermB % kBlock == 0 &&
                  kUbMetrics % kBlock == 0,
              "every UB operand base must be 32-byte aligned");
static_assert((kUbAccCurr - kUbStateCurr) % kVecReg != 0u &&
                  (kUbTermB - kUbTermA) % kVecReg != 0u,
              "live slot strides must not be a multiple of 256 B");
static_assert(kArenaEnd + kVecReg <= kUbWindow, "UB arena leaves no read pad");

// Event ids; 6 and 7 are library-reserved. Each route is set and waited inside
// one body, so none needs priming and none can leak.
constexpr std::int32_t kEvtIngress = 0;
constexpr std::int32_t kEvtFetch = 1;
constexpr std::int32_t kEvtStore = 2;
constexpr std::int32_t kEvtDrain = 3;
constexpr std::int32_t kEvtLock = 4;
constexpr std::int32_t kEvtUnlock = 5;

__aicore__ inline dim3 LockThreads() { return dim3{32u, 1u, 1u}; }

// A scalar reaches Global Memory as eight replicated int32 lanes of one
// DataBlock, written by `Duplicate` from a scalar: no UB subscript, so the
// Zero Scalar Compute Invariant holds.
__aicore__ inline void PublishMetric(std::uint32_t slot, std::int64_t value)
{
    AscendC::LocalTensor<std::int32_t> dst{
        AscendC::TPosition::VECCALC,
        kUbMetrics + slot * kProbeMetricLanesPerSlot * kF32,
        kProbeMetricLanesPerSlot};
    AscendC::Duplicate(dst, static_cast<std::int32_t>(value),
                       static_cast<std::int32_t>(kProbeMetricLanesPerSlot));
}

// An absolute 64-bit system cycle as two int32 slots. The counter is global
// across cores on this part, so these are directly comparable between splits,
// which is what makes a tail delta meaningful.
__aicore__ inline void PublishStamp(std::uint32_t hi_slot, std::uint32_t lo_slot,
                                    std::int64_t stamp)
{
    PublishMetric(hi_slot, static_cast<std::int32_t>(
                               static_cast<std::uint64_t>(stamp) >> 32));
    PublishMetric(lo_slot, static_cast<std::int32_t>(
                               static_cast<std::uint64_t>(stamp) & 0xFFFFFFFFu));
}

// The Duplicate read window, written finite before anything reads it: the
// tensor form reads a whole 256-byte vector register from its source base
// whatever the count, and a bank-skew gap sits inside that window.
__aicore__ inline void ZeroDuplicateWindow(std::uint32_t base)
{
    AscendC::LocalTensor<float> window{AscendC::TPosition::VECCALC, base, 128u};
    AscendC::Duplicate(window, 0.0f, 128);
}

__aicore__ inline void BindEngine(ascend_micro::L2AtomicProbeEngine& engine)
{
    constexpr auto ub = AscendC::TPosition::VECCALC;
    engine.m_tile = {ub, kUbStateTile + kProbeMOffset * kF32, kProbeStateLanes};
    engine.l_tile = {ub, kUbStateTile + kProbeLOffset * kF32, kProbeStateLanes};
    engine.pair_tile = {ub, kUbStateTile, kProbeStateElems};
    engine.acc_tile = {ub, kUbAccTile, kProbeChannels};
    engine.m_curr = {ub, kUbStateCurr + kProbeMOffset * kF32, kProbeStateLanes};
    engine.l_curr = {ub, kUbStateCurr + kProbeLOffset * kF32, kProbeStateLanes};
    engine.pair_curr = {ub, kUbStateCurr, kProbeStateElems};
    engine.acc_curr = {ub, kUbAccCurr, kProbeChannels};
    engine.m_new = {ub, kUbStateNew + kProbeMOffset * kF32, kProbeStateLanes};
    engine.l_new = {ub, kUbStateNew + kProbeLOffset * kF32, kProbeStateLanes};
    engine.pair_new = {ub, kUbStateNew, kProbeStateElems};
    engine.acc_new = {ub, kUbAccNew, kProbeChannels};
    engine.alpha_curr = {ub, kUbAlphaCurr, kProbeStateLanes};
    engine.alpha_tile = {ub, kUbAlphaTile, kProbeStateLanes};
    engine.alpha_curr_bc = {ub, kUbAlphaCurrBc, kProbeChannels};
    engine.alpha_tile_bc = {ub, kUbAlphaTileBc, kProbeChannels};
    engine.term_a = {ub, kUbTermA, kProbeChannels};
    engine.term_b = {ub, kUbTermB, kProbeChannels};
    engine.state_tmp = {ub, kUbStateTmp, kProbeStateLanes};
    engine.state_tmp2 = {ub, kUbStateTmp2, kProbeStateLanes};
}

// This split's partial state, as the same 2 + 32 bursts the modes publish.
__aicore__ inline void LoadTile(ascend_micro::L2AtomicProbeEngine& engine,
                                const AscendC::GlobalTensor<float>& tiles,
                                std::uint32_t split)
{
    const std::uint32_t base = split * kProbeRecordElems;
    AscendC::DataCopy(engine.pair_tile, tiles[base + kProbeMOffset], kProbeStateElems);
    AscendC::DataCopy(engine.acc_tile, tiles[base + kProbeAccOffset], kProbeChannels);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtIngress);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtIngress);
}

__aicore__ inline void PublishCommon(std::uint32_t split, std::uint32_t mode,
                                     std::uint32_t reps, std::int64_t begin,
                                     std::int64_t end)
{
    PublishMetric(ascend_micro::kMetricBlock, split);
    PublishMetric(ascend_micro::kMetricMode, mode);
    PublishMetric(ascend_micro::kMetricReps, reps);
    PublishMetric(ascend_micro::kMetricRegion, end - begin);
    PublishStamp(ascend_micro::kMetricBeginHi, ascend_micro::kMetricBeginLo, begin);
    PublishStamp(ascend_micro::kMetricEndHi, ascend_micro::kMetricEndLo, end);
}

__aicore__ inline void DrainMetrics(const AscendC::GlobalTensor<std::int32_t>& metrics,
                                    const AscendC::LocalTensor<std::int32_t>& slice,
                                    std::uint32_t split)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStore);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStore);
    AscendC::DataCopy(metrics[split * kProbeMetricElems], slice, kProbeMetricElems);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
}

// ================================================= workspace: no sharing
// This split's state to this split's slice. Two MTE3 bursts, no fetch, no
// combine, no lock -- the write-out latency with the combination removed.
__aicore__ inline void run_workspace(__gm__ float* gm_tiles, __gm__ float* gm_workspace,
                                     __gm__ std::int32_t* gm_metrics,
                                     std::uint32_t split, std::uint32_t reps)
{
    constexpr auto ub = AscendC::TPosition::VECCALC;
    AscendC::GlobalTensor<float> tiles, workspace;
    AscendC::GlobalTensor<std::int32_t> metrics;
    tiles.SetGlobalBuffer(gm_tiles);
    workspace.SetGlobalBuffer(gm_workspace);
    metrics.SetGlobalBuffer(gm_metrics);
    AscendC::LocalTensor<std::int32_t> slice{ub, kUbMetrics, kProbeMetricElems};

    ascend_micro::L2AtomicProbeEngine engine;
    BindEngine(engine);
    engine.workspace = workspace;
    engine.record_base = split * kProbeRecordElems;

    LoadTile(engine, tiles, split);

    AscendC::PipeBarrier<PIPE_ALL>();
    const std::int64_t begin = AscendC::GetSystemCycle();
    for (std::uint32_t rep = 0; rep < reps; ++rep) {
        engine.store_tile();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const std::int64_t end = AscendC::GetSystemCycle();

    PublishCommon(split, 0u, reps, begin, end);
    PublishMetric(ascend_micro::kMetricAcquired, 0);
    PublishMetric(ascend_micro::kMetricTimeouts, 0);
    PublishMetric(ascend_micro::kMetricSpins, 0);
    DrainMetrics(metrics, slice, split);
}

// ============================================ atomic: one line, K writers
// Per repetition: the atomic scalar maximum, the watchdog-bounded acquire, the
// read-modify-write on the shared line, the conditional release.
//
// The four edges, none of which is a barrier. `asc_vf_call` dispatches the VF
// asynchronously on the Vector pipe, so without `V_MTE2` the MTE2 fetch --
// on a queue nothing else gates -- would read the shared record before the
// lock was held. `MTE3_V` proves the write-back RETIRED before the release
// clears the lock; the fence inside the release orders accesses and does not
// wait for another pipe's outstanding burst.
//
// Unlike the CAModel probe, there is no `PipeBarrier<PIPE_ALL>` inside the
// repetition: the probe needed one per phase to read the cycle counter, and
// those barriers also happened to order the VF against MTE2. Here the region
// is timed once, end to end, so the flag edges are load-bearing on their own
// -- which is the configuration the probe's report lists as untested and the
// one a production combine would actually use.
__aicore__ inline void run_atomic(__gm__ float* gm_tiles, __gm__ float* gm_workspace,
                                  __gm__ std::uint32_t* gm_lock,
                                  __gm__ std::uint32_t* gm_ticket,
                                  __gm__ std::uint32_t* gm_ticket_log,
                                  __gm__ std::uint32_t* gm_spins,
                                  __gm__ std::uint32_t* gm_acquired,
                                  __gm__ float* gm_atomic_m,
                                  __gm__ std::int32_t* gm_metrics, std::uint32_t split,
                                  std::uint32_t reps, std::uint32_t max_attempts)
{
    constexpr auto ub = AscendC::TPosition::VECCALC;
    AscendC::GlobalTensor<float> tiles, workspace;
    AscendC::GlobalTensor<std::int32_t> metrics;
    tiles.SetGlobalBuffer(gm_tiles);
    workspace.SetGlobalBuffer(gm_workspace);
    metrics.SetGlobalBuffer(gm_metrics);
    AscendC::LocalTensor<std::int32_t> slice{ub, kUbMetrics, kProbeMetricElems};

    ascend_micro::L2AtomicProbeEngine engine;
    BindEngine(engine);
    engine.workspace = workspace;
    engine.record_base = 0u;  // every split contends for head slot 0

    ZeroDuplicateWindow(kUbAlphaCurr);
    AscendC::PipeBarrier<PIPE_V>();
    LoadTile(engine, tiles, split);

    __gm__ float* gm_tile_m = gm_tiles + split * kProbeRecordElems + kProbeMOffset;

    AscendC::PipeBarrier<PIPE_ALL>();
    const std::int64_t begin = AscendC::GetSystemCycle();

    for (std::uint32_t rep = 0; rep < reps; ++rep) {
        const std::uint32_t slot = split * reps + rep;

        asc_vf_call<ascend_micro::probe_state_max_vf>(LockThreads(), gm_atomic_m,
                                                      gm_tile_m);
        asc_vf_call<ascend_micro::splitk_lock_acquire_vf>(
            LockThreads(), gm_lock, gm_ticket, gm_ticket_log, gm_spins, gm_acquired,
            slot, max_attempts);

        // MTE2 may not start before the acquire VF has retired.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtLock);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtLock);
        engine.fetch_record();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtFetch);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtFetch);
        engine.combine();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStore);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStore);
        engine.store_record();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtUnlock);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtUnlock);

        asc_vf_call<ascend_micro::splitk_lock_release_vf>(LockThreads(), gm_lock,
                                                          gm_acquired, slot);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const std::int64_t end = AscendC::GetSystemCycle();

    // The per-slot acquire/spin records live in Global Memory, written by the
    // VFs; the host sums them. What this slice publishes is the timing and the
    // repetition count, so a host that reads only the metrics still knows how
    // many acquisitions to expect.
    PublishCommon(split, 1u, reps, begin, end);
    PublishMetric(ascend_micro::kMetricAcquired, reps);
    PublishMetric(ascend_micro::kMetricTimeouts, 0);
    PublishMetric(ascend_micro::kMetricSpins, 0);
    DrainMetrics(metrics, slice, split);
}

}  // namespace

#define SPLITK_ENTRY(NAME, ...)                                                       \
    extern "C" __global__ __aicore__ void turboquant_splitk_reduce_##NAME(            \
        __gm__ float* tiles, __gm__ float* workspace, __gm__ std::uint32_t* lock,     \
        __gm__ std::uint32_t* ticket, __gm__ std::uint32_t* ticket_log,               \
        __gm__ std::uint32_t* spins, __gm__ std::uint32_t* acquired,                  \
        __gm__ float* atomic_m, __gm__ std::int32_t* metrics, uint32_t reps,          \
        uint32_t splits, uint32_t max_attempts)                                       \
    {                                                                                 \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                               \
        AscendC::InitSocState();                                                      \
        if (reps == 0u || reps > ascend_micro::kProbeMaxReps) return;                 \
        if (splits == 0u || splits > ascend_micro::kSplitKMaxSplits) return;          \
        const std::uint32_t split =                                                   \
            static_cast<std::uint32_t>(AscendC::GetBlockIdx());                       \
        if (split >= splits) return;                                                  \
        __VA_ARGS__;                                                                  \
    }

SPLITK_ENTRY(workspace, (void)lock; (void)ticket; (void)ticket_log; (void)spins;
             (void)acquired; (void)atomic_m; (void)max_attempts;
             run_workspace(tiles, workspace, metrics, split, reps))

SPLITK_ENTRY(atomic,
             run_atomic(tiles, workspace, lock, ticket, ticket_log, spins, acquired,
                        atomic_m, metrics, split, reps, max_attempts))

#undef SPLITK_ENTRY

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

#include <stdexcept>

#define SPLITK_LAUNCH(NAME)                                                            \
    extern "C" __global__ __aicore__ void turboquant_splitk_reduce_##NAME(             \
        __gm__ float*, __gm__ float*, __gm__ std::uint32_t*, __gm__ std::uint32_t*,    \
        __gm__ std::uint32_t*, __gm__ std::uint32_t*, __gm__ std::uint32_t*,           \
        __gm__ float*, __gm__ std::int32_t*, uint32_t, uint32_t, uint32_t);            \
    extern "C" void launch_turboquant_splitk_reduce_##NAME(                            \
        void* stream, std::uint32_t splits, float* tiles, float* workspace,            \
        std::uint32_t* lock, std::uint32_t* ticket, std::uint32_t* ticket_log,         \
        std::uint32_t* spins, std::uint32_t* acquired, float* atomic_m,                \
        std::int32_t* metrics, std::uint32_t reps, std::uint32_t max_attempts)         \
    {                                                                                  \
        if (splits == 0u || splits > 16u)                                              \
            throw std::invalid_argument("split-K reduce splits must be 1..16");        \
        if (reps == 0u || reps > 8u)                                                   \
            throw std::invalid_argument("split-K reduce reps must be 1..8");           \
        turboquant_splitk_reduce_##NAME<<<splits, nullptr, stream>>>(                  \
            tiles, workspace, lock, ticket, ticket_log, spins, acquired, atomic_m,     \
            metrics, reps, splits, max_attempts);                                      \
    }

SPLITK_LAUNCH(workspace)
SPLITK_LAUNCH(atomic)
#undef SPLITK_LAUNCH

#endif
