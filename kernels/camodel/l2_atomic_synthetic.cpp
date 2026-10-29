#include <cstdint>

// CAModel synthetic contention probe for the Split-K reduction of the Cube
// decode pipeline. Ascend 950PR_9599 (dav-3510, __NPU_ARCH__ 3510),
// CANN 9.2.0-beta.2, RUN_MODE=sim, KERNEL_TYPE_AIV_ONLY.
//
// The leaf under test is routines/routine_probe_l2_atomic_sync.hpp; read its
// header first -- it carries the state-record layout, the reason the lock is a
// SIMT VF, and the two fixtures.
//
// One kernel, three entry points
// ------------------------------
//   camodel_l2_atomic_mode_a   Mode A, collision-free Split-K workspace
//   camodel_l2_atomic_mode_b   Mode B, shared slot under a GM spinlock
//   camodel_l2_atomic_mode_c   Mode C, the tail split's egress flush
//
// Three entries rather than one `mode` argument, for the same reason the
// butterfly benchmark has three: the comparison between the modes is a
// comparison of engine censuses, and it is only like for like if the three
// share the fixture, the grid, the UB arena, the event ids and the
// surrounding DMA. Everything outside the measured region is identical by
// construction here; a runtime branch would also have been identical, but it
// would have left both branches' instructions in one trace.
//
// What `splits` means
// -------------------
// `splits` is K, the number of temporal splits contending, and it is the
// LAUNCH dimension: under KERNEL_TYPE_AIV_ONLY every AIV is its own block
// (GetBlockIdx() is the AIV index and GetSubBlockNum() is 1 -- observed by the
// scatter ingress and single-token decode operators, which publish it rather
// than assuming it), so a launch of K blocks is K cores running concurrently.
// In Mode B all K address the SAME record; in Mode A each addresses its own.
//
// What this probe deliberately is NOT
// -----------------------------------
// It is not a decode kernel. There is no Cube, no L1, no KV cache, no score
// GEMM and no strip schedule: the tile state arrives pre-seeded from the host
// in exactly the shape AIV0 would have left it in, because the question is
// what the SYNCHRONISATION costs and a probe that recomputed the state would
// put the state's cost inside the measurement. The companion harness
// tests/camodel/test_camodel_cube_single_tile_baseline.cpp measures the other
// term against the frozen production operator.
//
// Why the measurement is a system-counter read and not only a trace
// ----------------------------------------------------------------
// `AscendC::GetSystemCycle()` is a PIPE_S read of the system counter,
// supported on Ascend 950PR, and the documented way to time a region is to
// drain the pipes around it. So every measured region here is bracketed by
// `PipeBarrier<PIPE_ALL>()`, and the consequence is explicit: the numbers are
// region LATENCIES with the pipeline drained, not throughputs under overlap.
// That is the right instrument for this question -- a Split-K sync is a
// serial interval between two compute intervals, and what the crossover needs
// is its latency -- but it means these cycles must not be compared against a
// pipelined per-strip interval without saying which is which. The report
// does. The trace audit is kept as the independent check, because the counter
// is a device-reported number and the instruction dumps are not.

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "../../routines/routine_probe_l2_atomic_sync.hpp"
#include "../../routines/routine_vector_rotate_butterfly.hpp"

namespace {

using ascend_micro::kProbeAccOffset;
using ascend_micro::kProbeChannels;
using ascend_micro::kProbeLOffset;
using ascend_micro::kProbeMaxReps;
using ascend_micro::kProbeMetricElems;
using ascend_micro::kProbeMetricLanesPerSlot;
using ascend_micro::kProbeMOffset;
using ascend_micro::kProbeRecordElems;
using ascend_micro::kProbeStateElems;
using ascend_micro::kProbeStateLanes;
using ascend_micro::ProbeMode;

// ---------------------------------------------------------------- UB arena
// Rule AKA3003: every operand base 32-byte aligned, one DataBlock of bank
// skew between live slots, and VECTOR_REG_WIDTH of addressable UB behind the
// last operand, because a level-2 vector op reads a whole vector register per
// source unmasked and `Duplicate`'s tensor form reads one from its base
// whatever the count.
constexpr std::uint32_t kBlock = 32u;
constexpr std::uint32_t kBankSkew = 32u;
constexpr std::uint32_t kVecReg = 256u;  // VECTOR_REG_WIDTH on dav-3510
constexpr std::uint32_t kUbWindow = 192u * 1024u;

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkew;
}

constexpr std::uint32_t kF32 = 4u;
constexpr std::uint32_t kB16 = 2u;
constexpr std::uint32_t kStatePairBytes = kProbeStateElems * kF32;  // 64
constexpr std::uint32_t kChannelBytes = kProbeChannels * kF32;      // 1024
constexpr std::uint32_t kLaneBytes = kProbeStateLanes * kF32;       // 32

// The three state records. Each (m, l) pair is ONE 64-byte slot with m at +0
// and l at +32 and no skew between them: that adjacency is what lets the pair
// move as a single 2-DataBlock burst in each direction.
constexpr std::uint32_t kUbStateTile = 0u;
constexpr std::uint32_t kUbAccTile = NextSlot(kUbStateTile, kStatePairBytes);
constexpr std::uint32_t kUbStateCurr = NextSlot(kUbAccTile, kChannelBytes);
constexpr std::uint32_t kUbAccCurr = NextSlot(kUbStateCurr, kStatePairBytes);
constexpr std::uint32_t kUbStateNew = NextSlot(kUbAccCurr, kChannelBytes);
constexpr std::uint32_t kUbAccNew = NextSlot(kUbStateNew, kStatePairBytes);

// Rescale scratch.
constexpr std::uint32_t kUbAlphaCurr = NextSlot(kUbAccNew, kChannelBytes);
constexpr std::uint32_t kUbAlphaTile = NextSlot(kUbAlphaCurr, kLaneBytes);
constexpr std::uint32_t kUbStateTmp = NextSlot(kUbAlphaTile, kLaneBytes);
constexpr std::uint32_t kUbStateTmp2 = NextSlot(kUbStateTmp, kLaneBytes);
constexpr std::uint32_t kUbAlphaCurrBc = NextSlot(kUbStateTmp2, kLaneBytes);
constexpr std::uint32_t kUbAlphaTileBc = NextSlot(kUbAlphaCurrBc, kChannelBytes);
constexpr std::uint32_t kUbTermA = NextSlot(kUbAlphaTileBc, kChannelBytes);
constexpr std::uint32_t kUbTermB = NextSlot(kUbTermA, kChannelBytes);

// Mode C's egress operands. `kUbNorm` is the butterfly's source and
// `kUbRotated` its destination; both are addressed as raw __ubuf__ pointers by
// the register-resident leaf, which loads and stores whole 64-lane registers
// from a 32-byte aligned base -- the same configuration the butterfly
// benchmark runs today.
constexpr std::uint32_t kUbPiSign = NextSlot(kUbTermB, kChannelBytes);
constexpr std::uint32_t kUbGateRow = NextSlot(kUbPiSign, kChannelBytes);
constexpr std::uint32_t kUbLBroadcast = NextSlot(kUbGateRow, kChannelBytes);
constexpr std::uint32_t kUbNorm = NextSlot(kUbLBroadcast, kChannelBytes);
constexpr std::uint32_t kUbRotated = NextSlot(kUbNorm, kChannelBytes);
constexpr std::uint32_t kUbGateTmp = NextSlot(kUbRotated, kChannelBytes);
constexpr std::uint32_t kUbGateDen = NextSlot(kUbGateTmp, kChannelBytes);
constexpr std::uint32_t kUbGated = NextSlot(kUbGateDen, kChannelBytes);
constexpr std::uint32_t kUbContext = NextSlot(kUbGated, kChannelBytes);

// The metric slice, last so the arena's read pad covers it too.
constexpr std::uint32_t kUbMetrics = NextSlot(kUbContext, kProbeChannels * kB16);
constexpr std::uint32_t kArenaEnd = kUbMetrics + kProbeMetricElems * kF32;

// ------------------------------------------------------------------ V0 gates
static_assert(kUbStateTile % kBlock == 0 && kUbAccTile % kBlock == 0 &&
                  kUbStateCurr % kBlock == 0 && kUbAccCurr % kBlock == 0 &&
                  kUbStateNew % kBlock == 0 && kUbAccNew % kBlock == 0 &&
                  kUbAlphaCurr % kBlock == 0 && kUbAlphaTile % kBlock == 0 &&
                  kUbStateTmp % kBlock == 0 && kUbStateTmp2 % kBlock == 0 &&
                  kUbAlphaCurrBc % kBlock == 0 && kUbAlphaTileBc % kBlock == 0 &&
                  kUbTermA % kBlock == 0 && kUbTermB % kBlock == 0 &&
                  kUbPiSign % kBlock == 0 && kUbGateRow % kBlock == 0 &&
                  kUbLBroadcast % kBlock == 0 && kUbNorm % kBlock == 0 &&
                  kUbRotated % kBlock == 0 && kUbGateTmp % kBlock == 0 &&
                  kUbGateDen % kBlock == 0 && kUbGated % kBlock == 0 &&
                  kUbContext % kBlock == 0 && kUbMetrics % kBlock == 0,
              "every UB operand base must be 32-byte aligned");
// Rule A2: no two live slots on one bank phase.
static_assert((kUbAccCurr - kUbStateCurr) % kVecReg != 0u &&
                  (kUbTermB - kUbTermA) % kVecReg != 0u &&
                  (kUbRotated - kUbNorm) % kVecReg != 0u,
              "live slot strides must not be a multiple of 256 B");
// Rule A3, and `Duplicate`'s 256-byte source read from each alpha block.
static_assert(kArenaEnd + kVecReg <= kUbWindow, "UB arena leaves no read pad");
static_assert(kUbAlphaCurr + kVecReg <= kUbWindow &&
                  kUbAlphaTile + kVecReg <= kUbWindow &&
                  kUbStateCurr + kProbeLOffset * kF32 + kVecReg <= kUbWindow,
              "a Duplicate source needs a whole vector register behind it");
// Rule A4 / the task's accounting: both directions move 2 + 32 DataBlocks.
static_assert(kStatePairBytes / kBlock == 2u && kChannelBytes / kBlock == 32u,
              "the record moves as a 2-burst state pair and a 32-burst accumulator");

// The same 0-ULP divide configuration the production egress uses, so Mode C's
// normalisation and gate round identically to the operator's. At namespace
// scope and not inside the body that uses it: as a function-local object it is
// not accepted as the explicit `config` template argument, which is also why
// routine_vector_egress_rotate.hpp declares its copy this way.
constexpr AscendC::DivConfig kPreciseDiv = {AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};

// Event ids. 6 and 7 are library-reserved; every route below is set and
// waited inside one body, so none needs priming and none can leak.
constexpr std::int32_t kEvtIngress = 0;  // MTE2_V: the fixture reached UB
constexpr std::int32_t kEvtFetch = 1;    // MTE2_V: the shared record reached UB
constexpr std::int32_t kEvtStore = 2;    // V_MTE3: the combined record is in UB
constexpr std::int32_t kEvtDrain = 3;    // MTE3_S: the outflow retired
constexpr std::int32_t kEvtLock = 4;     // V_MTE2: the acquire VF retired
constexpr std::int32_t kEvtUnlock = 5;   // MTE3_V: the write-back retired

// One warp per SIMT dispatch; lane 0 does the work.
__aicore__ inline dim3 LockThreads() { return dim3{32u, 1u, 1u}; }

// ------------------------------------------------------------ metric writes
// A scalar reaches Global Memory as eight replicated int32 lanes of one
// DataBlock, written by `Duplicate` from a scalar. No UB subscript, no scalar
// read of UB, and the eight-fold replication is itself a check: a slot whose
// lanes disagree was not written by one instruction.
__aicore__ inline void PublishMetric(std::uint32_t slot, std::int64_t value)
{
    AscendC::LocalTensor<std::int32_t> dst{
        AscendC::TPosition::VECCALC,
        kUbMetrics + slot * kProbeMetricLanesPerSlot * kF32,
        kProbeMetricLanesPerSlot};
    AscendC::Duplicate(dst, static_cast<std::int32_t>(value),
                       static_cast<std::int32_t>(kProbeMetricLanesPerSlot));
}

// ------------------------------------------------ the Duplicate read window
// `Duplicate`'s tensor form broadcasts lane 0 of its source, but on dav-3510
// it READS a whole 256-byte vector register from the source base whatever the
// count. The arena already asserts that much UB is addressable behind every
// such source; what it cannot assert is that those bytes are INITIALISED, and
// a 32-byte bank-skew gap sits inside every one of these windows.
//
// Reading an uninitialised gap into a broadcast register is a real hazard --
// the bytes can hold any bit pattern, including a NaN -- so every Duplicate
// source's whole read window is written with a finite value before anything
// reads it. 512 bytes from each window base covers the 256 the instruction
// reads with room to spare and stays well inside the following 1024-byte
// accumulator slot; the record fetch and the arithmetic then overwrite
// everything in the window except the gap, which keeps the zero.
//
// This is a precaution on its own terms. It is NOT the fix for the Inf/NaN
// diagnostic the Mode C rung raises -- that was the first hypothesis and the
// trace refuted it. See §5 of reports/l2_atomic_contention_split_k.md: the
// flag survived this change unaltered, so the skew gap was not its cause.
constexpr std::uint32_t kDuplicateWindowLanes = 128u;  // 512 B

__aicore__ inline void ZeroDuplicateWindow(std::uint32_t base)
{
    AscendC::LocalTensor<float> window{AscendC::TPosition::VECCALC, base,
                                       kDuplicateWindowLanes};
    AscendC::Duplicate(window, 0.0f,
                       static_cast<std::int32_t>(kDuplicateWindowLanes));
}

// --------------------------------------------------------------- the engine
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

// The fixture ingress, identical in all three modes: this core's own record
// out of the seeded tile array, as the same 2 + 32 bursts the modes use.
__aicore__ inline void LoadTile(ascend_micro::L2AtomicProbeEngine& engine,
                                const AscendC::GlobalTensor<float>& tiles,
                                std::uint32_t block)
{
    const std::uint32_t base = block * kProbeRecordElems;
    AscendC::DataCopy(engine.pair_tile, tiles[base + kProbeMOffset], kProbeStateElems);
    AscendC::DataCopy(engine.acc_tile, tiles[base + kProbeAccOffset], kProbeChannels);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtIngress);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtIngress);
}

// ======================================================= Mode A: no sharing
// This core's state to this core's slice. Two MTE3 bursts and the drain that
// proves they retired, and nothing else: no fetch, no combine, no lock.
__aicore__ inline void run_mode_a(__gm__ float* gm_tiles, __gm__ float* gm_workspace,
                                  __gm__ std::int32_t* gm_metrics, std::uint32_t block,
                                  std::uint32_t reps)
{
    constexpr auto ub = AscendC::TPosition::VECCALC;
    AscendC::GlobalTensor<float> tiles, workspace;
    AscendC::GlobalTensor<std::int32_t> metrics;
    tiles.SetGlobalBuffer(gm_tiles);
    workspace.SetGlobalBuffer(gm_workspace);
    metrics.SetGlobalBuffer(gm_metrics);
    AscendC::LocalTensor<std::int32_t> metric_slice{ub, kUbMetrics, kProbeMetricElems};

    ascend_micro::L2AtomicProbeEngine engine;
    BindEngine(engine);
    engine.workspace = workspace;
    // The dedicated slice: gm_workspace + split_id * sizeof(State).
    engine.record_base = block * kProbeRecordElems;

    LoadTile(engine, tiles, block);

    AscendC::PipeBarrier<PIPE_ALL>();
    const std::int64_t region_begin = AscendC::GetSystemCycle();
    std::int64_t store_cycles = 0;
    for (std::uint32_t rep = 0; rep < reps; ++rep) {
        AscendC::PipeBarrier<PIPE_ALL>();
        const std::int64_t t0 = AscendC::GetSystemCycle();
        engine.store_tile();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
        AscendC::PipeBarrier<PIPE_ALL>();
        const std::int64_t t1 = AscendC::GetSystemCycle();
        store_cycles += t1 - t0;
        if (rep < kProbeMaxReps)
            PublishMetric(ascend_micro::kMetricRepBase + rep, t1 - t0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const std::int64_t region_end = AscendC::GetSystemCycle();

    PublishMetric(ascend_micro::kMetricBlock, block);
    PublishMetric(ascend_micro::kMetricMode,
                  static_cast<std::int64_t>(ProbeMode::CollisionFree));
    PublishMetric(ascend_micro::kMetricReps, reps);
    PublishMetric(ascend_micro::kMetricRegion, region_end - region_begin);
    PublishMetric(ascend_micro::kMetricPhase0, store_cycles);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStore);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStore);
    AscendC::DataCopy(metrics[block * kProbeMetricElems], metric_slice,
                      kProbeMetricElems);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
}

// ================================================== Mode B: the shared slot
// K cores, one record. Per repetition: the atomic scalar maximum, then the
// lock, then the read-modify-write, then the release.
//
// The four edges that make this correct, and none of them is a barrier
// -------------------------------------------------------------------
//   V_MTE2  (kEvtLock)    the acquire VF issues on the Vector pipe and
//                         `asc_vf_call` dispatches it asynchronously, so
//                         without this the MTE2 fetch below -- on a queue
//                         nothing else gates -- would read the shared record
//                         BEFORE the lock was held. This is the edge that
//                         makes the critical section critical.
//   MTE2_V  (kEvtFetch)   the fetched record is readable by the combine.
//   V_MTE3  (kEvtStore)   the combined record is final in UB.
//   MTE3_V  (kEvtUnlock)  the write-back has RETIRED, so the release VF may
//                         clear the lock word. The fence inside the release
//                         orders accesses; it does not wait for another
//                         pipe's outstanding burst, which is why the retire
//                         has to be proved here instead.
__aicore__ inline void run_mode_b(__gm__ float* gm_tiles, __gm__ float* gm_workspace,
                                  __gm__ std::uint32_t* gm_lock,
                                  __gm__ std::uint32_t* gm_ticket,
                                  __gm__ std::uint32_t* gm_ticket_log,
                                  __gm__ std::uint32_t* gm_spins,
                                  __gm__ float* gm_atomic_m,
                                  __gm__ std::int32_t* gm_metrics, std::uint32_t block,
                                  std::uint32_t reps)
{
    constexpr auto ub = AscendC::TPosition::VECCALC;
    AscendC::GlobalTensor<float> tiles, workspace;
    AscendC::GlobalTensor<std::int32_t> metrics;
    tiles.SetGlobalBuffer(gm_tiles);
    workspace.SetGlobalBuffer(gm_workspace);
    metrics.SetGlobalBuffer(gm_metrics);
    AscendC::LocalTensor<std::int32_t> metric_slice{ub, kUbMetrics, kProbeMetricElems};

    ascend_micro::L2AtomicProbeEngine engine;
    BindEngine(engine);
    engine.workspace = workspace;
    engine.record_base = 0u;  // every split contends for head slot 0

    // Every Duplicate source's read window, before the arithmetic reaches it.
    ZeroDuplicateWindow(kUbAlphaCurr);
    AscendC::PipeBarrier<PIPE_V>();

    LoadTile(engine, tiles, block);

    // This split's m, for the atomic reduction, as a Global Memory ADDRESS and
    // not as a value: the VF reads it with `asc_ldcg` itself. Resolving it
    // here with `GlobalTensor::GetValue` instead cost a library-emitted
    // MTE2 -> S synchronisation on reserved event id 7, which the trace audit
    // rejects; see the VF's own header comment.
    __gm__ float* gm_tile_m = gm_tiles + block * kProbeRecordElems + kProbeMOffset;

    AscendC::PipeBarrier<PIPE_ALL>();
    const std::int64_t region_begin = AscendC::GetSystemCycle();
    std::int64_t atomic_cycles = 0, acquire_cycles = 0, critical_cycles = 0,
                 release_cycles = 0;

    for (std::uint32_t rep = 0; rep < reps; ++rep) {
        const std::uint32_t slot = block * reps + rep;

        // ---- the scalar reduction: one hardware atomic, no lock ---------
        AscendC::PipeBarrier<PIPE_ALL>();
        const std::int64_t t0 = AscendC::GetSystemCycle();
        asc_vf_call<ascend_micro::probe_state_max_vf>(LockThreads(), gm_atomic_m,
                                                      gm_tile_m);
        AscendC::PipeBarrier<PIPE_ALL>();
        const std::int64_t t1 = AscendC::GetSystemCycle();

        // ---- spinlock acquisition --------------------------------------
        asc_vf_call<ascend_micro::probe_lock_acquire_vf>(
            LockThreads(), gm_lock, gm_ticket, gm_ticket_log, gm_spins, slot);
        AscendC::PipeBarrier<PIPE_ALL>();
        const std::int64_t t2 = AscendC::GetSystemCycle();

        // ---- the critical section --------------------------------------
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
        // The write-back must be retired before the lock is cleared.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtUnlock);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtUnlock);
        AscendC::PipeBarrier<PIPE_ALL>();
        const std::int64_t t3 = AscendC::GetSystemCycle();

        // ---- release ----------------------------------------------------
        asc_vf_call<ascend_micro::probe_lock_release_vf>(LockThreads(), gm_lock);
        AscendC::PipeBarrier<PIPE_ALL>();
        const std::int64_t t4 = AscendC::GetSystemCycle();

        atomic_cycles += t1 - t0;
        acquire_cycles += t2 - t1;
        critical_cycles += t3 - t2;
        release_cycles += t4 - t3;
        if (rep < kProbeMaxReps)
            PublishMetric(ascend_micro::kMetricRepBase + rep, t4 - t0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const std::int64_t region_end = AscendC::GetSystemCycle();

    PublishMetric(ascend_micro::kMetricBlock, block);
    PublishMetric(ascend_micro::kMetricMode,
                  static_cast<std::int64_t>(ProbeMode::SharedAtomic));
    PublishMetric(ascend_micro::kMetricReps, reps);
    PublishMetric(ascend_micro::kMetricRegion, region_end - region_begin);
    PublishMetric(ascend_micro::kMetricPhase1, atomic_cycles);
    PublishMetric(ascend_micro::kMetricPhase2, acquire_cycles);
    PublishMetric(ascend_micro::kMetricPhase3, critical_cycles);
    PublishMetric(ascend_micro::kMetricPhase4, release_cycles);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStore);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStore);
    AscendC::DataCopy(metrics[block * kProbeMetricElems], metric_slice,
                      kProbeMetricElems);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
}

// ============================================== Mode C: the tail-split flush
// Only the designated tail split runs this, which is what makes it the
// terminal mode: the combine has already happened, the shared record holds
// the whole context's (m, l, Acc), and what remains is the part of the
// single-page egress that does NOT commute with a cross-block combine.
//
//   O   = Acc / l                     the normalisation
//   O.Pi                              the FWHT butterfly, RegGather leaf
//   O.Pi . sigmoid(gate)              the gate
//   bf16                              the cast, then one 512-byte burst
//
// The rotation is the published `rotate_butterfly_gather_vf` entry point in
// the Egress direction, with Pi's 1/16 orthonormalisation passed as the
// leaf's own `scale` so it folds into four register multiplies rather than
// becoming a separate Unified Buffer pass.
__aicore__ inline void run_mode_c(__gm__ float* gm_workspace, __gm__ float* gm_pi_sign,
                                  __gm__ float* gm_gate,
                                  __gm__ bfloat16_t* gm_out_context,
                                  __gm__ std::int32_t* gm_metrics, std::uint32_t block,
                                  std::uint32_t reps)
{
    constexpr auto ub = AscendC::TPosition::VECCALC;
    constexpr std::int32_t kChannels = static_cast<std::int32_t>(kProbeChannels);

    AscendC::GlobalTensor<float> workspace, pi_sign_gm, gate_gm;
    AscendC::GlobalTensor<bfloat16_t> out_context;
    AscendC::GlobalTensor<std::int32_t> metrics;
    workspace.SetGlobalBuffer(gm_workspace);
    pi_sign_gm.SetGlobalBuffer(gm_pi_sign);
    gate_gm.SetGlobalBuffer(gm_gate);
    out_context.SetGlobalBuffer(gm_out_context);
    metrics.SetGlobalBuffer(gm_metrics);

    AscendC::LocalTensor<std::int32_t> metric_slice{ub, kUbMetrics, kProbeMetricElems};
    AscendC::LocalTensor<float> l_curr{ub, kUbStateCurr + kProbeLOffset * kF32,
                                       kProbeStateLanes};
    AscendC::LocalTensor<float> acc_curr{ub, kUbAccCurr, kProbeChannels};
    AscendC::LocalTensor<float> pi_sign{ub, kUbPiSign, kProbeChannels};
    AscendC::LocalTensor<float> gate{ub, kUbGateRow, kProbeChannels};
    AscendC::LocalTensor<float> l_broadcast{ub, kUbLBroadcast, kProbeChannels};
    AscendC::LocalTensor<float> norm{ub, kUbNorm, kProbeChannels};
    AscendC::LocalTensor<float> rotated{ub, kUbRotated, kProbeChannels};
    AscendC::LocalTensor<float> gate_tmp{ub, kUbGateTmp, kProbeChannels};
    AscendC::LocalTensor<float> gate_den{ub, kUbGateDen, kProbeChannels};
    AscendC::LocalTensor<float> gated{ub, kUbGated, kProbeChannels};
    AscendC::LocalTensor<bfloat16_t> context{ub, kUbContext, kProbeChannels};

    // The l broadcast's read window, zeroed BEFORE the record lands: the
    // fetch overwrites the state pair and the accumulator inside the window
    // and leaves the bank-skew gap holding the zero.
    ZeroDuplicateWindow(kUbStateCurr);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtIngress);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtIngress);

    // The combined record, the sign vector and this split's gate row.
    AscendC::DataCopy(AscendC::LocalTensor<float>{ub, kUbStateCurr, kProbeStateElems},
                      workspace[kProbeMOffset], kProbeStateElems);
    AscendC::DataCopy(acc_curr, workspace[kProbeAccOffset], kProbeChannels);
    AscendC::DataCopy(pi_sign, pi_sign_gm[0], kProbeChannels);
    AscendC::DataCopy(gate, gate_gm[0], kProbeChannels);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtIngress);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtIngress);

    __ubuf__ float* norm_base = reinterpret_cast<__ubuf__ float*>(norm.GetPhyAddr());
    __ubuf__ float* rotated_base =
        reinterpret_cast<__ubuf__ float*>(rotated.GetPhyAddr());
    __ubuf__ float* sign_base = reinterpret_cast<__ubuf__ float*>(pi_sign.GetPhyAddr());

    AscendC::PipeBarrier<PIPE_ALL>();
    const std::int64_t region_begin = AscendC::GetSystemCycle();
    std::int64_t norm_cycles = 0, rot_cycles = 0, gate_cycles = 0, store_cycles = 0;

    for (std::uint32_t rep = 0; rep < reps; ++rep) {
        AscendC::PipeBarrier<PIPE_ALL>();
        const std::int64_t t0 = AscendC::GetSystemCycle();
        // O = Acc / l. `Duplicate`'s tensor form broadcasts lane 0 of the
        // replicated l block across all 256 channels.
        AscendC::Duplicate(l_broadcast, l_curr, kChannels);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div<float, kPreciseDiv>(norm, acc_curr, l_broadcast, kChannels);
        AscendC::PipeBarrier<PIPE_ALL>();
        const std::int64_t t1 = AscendC::GetSystemCycle();

        // The butterfly, one row, x . Pi, with the 1/16 folded in.
        ascend_micro::rotate_butterfly_gather_vf<ascend_micro::RotateDirection::Egress>(
            rotated_base, norm_base, sign_base, ascend_micro::kRotOrthoScale, 1u);
        AscendC::PipeBarrier<PIPE_ALL>();
        const std::int64_t t2 = AscendC::GetSystemCycle();

        // out = y . sigmoid(gate) = y / (1 + exp(-gate)), then bf16.
        AscendC::Muls(gate_tmp, gate, -1.0f, kChannels);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gate_den, gate_tmp, kChannels);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(gate_tmp, gate_den, 1.0f, kChannels);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div<float, kPreciseDiv>(gated, rotated, gate_tmp, kChannels);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(context, gated, AscendC::RoundMode::CAST_RINT, kChannels);
        AscendC::PipeBarrier<PIPE_ALL>();
        const std::int64_t t3 = AscendC::GetSystemCycle();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStore);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStore);
        AscendC::DataCopy(out_context[0], context, kProbeChannels);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
        AscendC::PipeBarrier<PIPE_ALL>();
        const std::int64_t t4 = AscendC::GetSystemCycle();

        norm_cycles += t1 - t0;
        rot_cycles += t2 - t1;
        gate_cycles += t3 - t2;
        store_cycles += t4 - t3;
        if (rep < kProbeMaxReps)
            PublishMetric(ascend_micro::kMetricRepBase + rep, t4 - t0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const std::int64_t region_end = AscendC::GetSystemCycle();

    // ---- the egress intermediates, for diagnosis, OUTSIDE the timed loop --
    // The model raises `vec_err_idata_inf_nan_t0` on the 0-ULP divide's
    // classification path in this mode and not in Mode A, Mode B or the
    // production operator's identical egress, so the question "which operand
    // register holds a non-finite value" has to be answerable from the host
    // rather than from a theory. These three bursts publish the two divides'
    // numerator/denominator pairs into workspace slices 1..3, which Mode C
    // otherwise never touches, and the runner scans them. They are emitted
    // after the measured region so the phase cycles stay comparable with the
    // other modes.
    AscendC::DataCopy(workspace[1u * kProbeRecordElems + kProbeAccOffset],
                      l_broadcast, kProbeChannels);
    AscendC::DataCopy(workspace[2u * kProbeRecordElems + kProbeAccOffset], rotated,
                      kProbeChannels);
    AscendC::DataCopy(workspace[3u * kProbeRecordElems + kProbeAccOffset], gate_tmp,
                      kProbeChannels);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);

    PublishMetric(ascend_micro::kMetricBlock, block);
    PublishMetric(ascend_micro::kMetricMode,
                  static_cast<std::int64_t>(ProbeMode::TailEgress));
    PublishMetric(ascend_micro::kMetricReps, reps);
    PublishMetric(ascend_micro::kMetricRegion, region_end - region_begin);
    PublishMetric(ascend_micro::kMetricPhase0, norm_cycles);
    PublishMetric(ascend_micro::kMetricPhase1, rot_cycles);
    PublishMetric(ascend_micro::kMetricPhase2, gate_cycles);
    PublishMetric(ascend_micro::kMetricPhase3, store_cycles);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStore);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStore);
    AscendC::DataCopy(metrics[block * kProbeMetricElems], metric_slice,
                      kProbeMetricElems);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
}

}  // namespace

// The three entry points share one argument list so the runner can drive them
// through one launcher signature and one fixture. `splits` is K and bounds the
// grid; `reps` is the per-split repetition count.
#define PROBE_ENTRY(NAME, ...)                                                      \
    extern "C" __global__ __aicore__ void camodel_l2_atomic_##NAME(                  \
        __gm__ float* tiles, __gm__ float* workspace, __gm__ std::uint32_t* lock,    \
        __gm__ std::uint32_t* ticket, __gm__ std::uint32_t* ticket_log,              \
        __gm__ std::uint32_t* spins, __gm__ float* atomic_m, __gm__ float* pi_sign,  \
        __gm__ float* gate, __gm__ bfloat16_t* out_context,                          \
        __gm__ std::int32_t* metrics, uint32_t reps, uint32_t splits)                 \
    {                                                                                \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                              \
        AscendC::InitSocState();                                                     \
        if (reps == 0u || reps > ascend_micro::kProbeMaxReps) return;                \
        if (splits == 0u || splits > ascend_micro::kProbeMaxSplits) return;           \
        const std::uint32_t block =                                                  \
            static_cast<std::uint32_t>(AscendC::GetBlockIdx());                      \
        if (block >= splits) return;                                                 \
        __VA_ARGS__;                                                                 \
    }

PROBE_ENTRY(mode_a, (void)lock; (void)ticket; (void)ticket_log; (void)spins;
            (void)atomic_m; (void)pi_sign; (void)gate; (void)out_context;
            run_mode_a(tiles, workspace, metrics, block, reps))

PROBE_ENTRY(mode_b, (void)pi_sign; (void)gate; (void)out_context;
            run_mode_b(tiles, workspace, lock, ticket, ticket_log, spins, atomic_m,
                       metrics, block, reps))

// Mode C is the TAIL split's work, so exactly one core does it: the last
// split of the launch. The others return having touched nothing, which is
// also what makes the runner's "no other slice was written" check meaningful.
PROBE_ENTRY(mode_c, (void)tiles; (void)lock; (void)ticket; (void)ticket_log;
            (void)spins; (void)atomic_m;
            if (block + 1u != splits) return;
            run_mode_c(workspace, pi_sign, gate, out_context, metrics, block, reps))

#undef PROBE_ENTRY

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

#include <stdexcept>

#define PROBE_LAUNCH(NAME)                                                            \
    extern "C" __global__ __aicore__ void camodel_l2_atomic_##NAME(                   \
        __gm__ float*, __gm__ float*, __gm__ std::uint32_t*, __gm__ std::uint32_t*,   \
        __gm__ std::uint32_t*, __gm__ std::uint32_t*, __gm__ float*, __gm__ float*,   \
        __gm__ float*, __gm__ bfloat16_t*, __gm__ std::int32_t*, uint32_t, uint32_t); \
    extern "C" void launch_camodel_l2_atomic_##NAME(                                  \
        void* stream, std::uint32_t splits, float* tiles, float* workspace,           \
        std::uint32_t* lock, std::uint32_t* ticket, std::uint32_t* ticket_log,        \
        std::uint32_t* spins, float* atomic_m, float* pi_sign, float* gate,           \
        std::uint16_t* out_context, std::int32_t* metrics, std::uint32_t reps)        \
    {                                                                                 \
        if (splits == 0u || splits > 8u)                                              \
            throw std::invalid_argument("l2 atomic probe splits must be 1..8");       \
        if (reps == 0u || reps > 8u)                                                  \
            throw std::invalid_argument("l2 atomic probe reps must be 1..8");         \
        camodel_l2_atomic_##NAME<<<splits, nullptr, stream>>>(                        \
            tiles, workspace, lock, ticket, ticket_log, spins, atomic_m, pi_sign,     \
            gate, reinterpret_cast<bfloat16_t*>(out_context), metrics, reps, splits); \
    }

PROBE_LAUNCH(mode_a)
PROBE_LAUNCH(mode_b)
PROBE_LAUNCH(mode_c)
#undef PROBE_LAUNCH

#endif
