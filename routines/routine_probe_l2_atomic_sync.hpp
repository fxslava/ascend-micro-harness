#pragma once

// The Split-K synchronisation probe for the Cube decode pipeline.
// Ascend 950PR / arch35 / dav-3510 / CANN 9.2, AIV-only.
//
// What this leaf is for
// ---------------------
// `kernels/ops/turboquant/turboquant_paged_decode_attention.cpp` accumulates
// the streaming online-softmax state (m, l, Acc) in the Unified Buffer of
// AIV0 (`kUbMPrev`, `kUbLPrev`, `kUbAcc`) across the eight-strip schedule of
// one 128-token page, and emits a PARTIAL softmax per page. A multi-thousand
// token context therefore needs a temporal (Split-K) reduction across pages,
// and that reduction is not in that kernel -- its report says so explicitly.
//
// The design question this leaf exists to answer is where the reduction
// should live. Two shapes are possible, and they differ by orders of
// magnitude in what they cost:
//
//   Mode A  every split writes its own state to its own slice of a Global
//           Memory workspace and a separate pass combines them. No two cores
//           ever address the same line, so there is no contention to measure
//           and the cost is one pair of MTE3 bursts.
//   Mode B  the splits combine IN PLACE into one shared head slot under a
//           Global Memory spinlock. One line, K writers, a read-modify-write
//           critical section each.
//
// and one terminal shape:
//
//   Mode C  the tail split -- the one that observes the last contribution --
//           normalises, rotates back out of the cache's channel basis and
//           gates, which is the egress the single-page operator already does
//           and the only part of it that does NOT commute with a cross-block
//           combine.
//
// Simulating a full 1024- or 4096-token sequence on CAModel is cycle
// prohibitive (the eight-strip single-page run is already about fifteen
// minutes on this host), so the question is answered by composition instead:
// this leaf measures T_sync(K) in isolation, the single-tile Cube probe
// measures T_compute_strip, and the crossover follows arithmetically. What
// that composition assumes is stated in the report rather than here.
//
// The state record
// ----------------
// Laid out to mirror AIV0's resident state exactly, so the bursts this leaf
// issues are the bursts a real combine would issue:
//
//   m    one lane-replicated 32-byte DataBlock   (kProbeStateLanes = 8 FP32)
//   l    one lane-replicated 32-byte DataBlock
//   Acc  kProbeChannels = 256 FP32              = 1024 B = 32 DataBlocks
//
// 272 FP32 = 1088 B = 34 DataBlocks in all. m and l are carried as replicated
// blocks and not as single floats for the reason section 1.2 of
// MICROROUTINE_DESIGN_GUIDELINES.md gives: a logically scalar value that the
// Vector pipe produces must never be read back by the Scalar unit, so it is
// carried as a whole DataBlock from a 32-byte aligned base and every
// operation on it stays on the Vector pipe. The record is contiguous in
// Global Memory but is moved as TWO bursts in each direction -- 2 DataBlocks
// for (m, l) and 32 for Acc -- which is what makes the 32-burst accumulator
// transfer legible in a trace as one instruction rather than hidden inside a
// 34-block one.
//
// Why the lock lives in a SIMT VF
// -------------------------------
// There is no SIMD atomic on this part. `asc_atomic_exch`, `asc_atomic_cas`,
// `asc_atomic_max` and `asc_atomic_add` are SIMT API interfaces declared
// `__simt_callee__` (CANN 9.2.0-beta.2,
// asc/include/simt_api/device_atomic_functions.h, guarded
// `#if (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102)`), so they are
// reachable only from a `__simt_vf__` body dispatched by `asc_vf_call`. Two
// consequences are load-bearing:
//
//   * `asc_vf_call` is a no-op unless `SPLIT_CORE_VEC` is defined, and
//     simt_api/common_functions.h defines it for `__DAV_VEC__ && __NPU_ARCH__
//     == 3510` -- i.e. in the AIV leg and nowhere else. A lock in a Cube-side
//     code path would compile to nothing at all, silently. That is why this
//     probe is `KERNEL_TYPE_AIV_ONLY` and why nothing here is reachable from
//     an `ASCEND_IS_AIC` branch.
//   * the VF issues on the Vector pipeline and `asc_vf_call` dispatches it
//     asynchronously, so it orders NOTHING with respect to the MTE2 fetch and
//     MTE3 write-back that make up the critical section. The caller owns that
//     ordering, and this header's contract names the exact edges (see
//     `probe_mode_b_*` below). A `DataSyncBarrier` is not one of them.
//
// Every pointer these VFs dereference is `__gm__`. That is deliberate: a
// `__ubuf__` argument to a SIMT VF must live in the declared dynamic UB
// region or the first SIMT UB access faults with AIV error 341 on every core,
// which in turn forces `dyn_ub_size` into the launch chevron and a single UB
// allocation mechanism onto the whole kernel. The lock, the ticket counter,
// the spin census and the atomically reduced maximum are all Global Memory
// words, so this leaf needs none of that and the kernel keeps the repository's
// hand-placed `LocalTensor` idiom.
//
// What the two atomic reductions are, and why both
// -----------------------------------------------
//   * the scalar maximum is a true hardware atomic: `asc_atomic_max` on a
//     `__gm__ float*`. One instruction, no lock, no critical section. It is
//     measured separately because a cross-split m reduction needs nothing
//     more than this, and because its result is independently checkable --
//     the final value must be the maximum over the splits, bit for bit, in
//     any interleaving.
//   * the vector accumulation cannot be a single atomic, because the rescale
//     factor exp(m_curr - m_new) depends on the value being replaced. That is
//     what forces the read-modify-write, and therefore the lock.
//
// A NOTE on `asc_atomic_max` over negative floats. The published contract
// says the interface takes `__gm__ float*` and returns the initial value; it
// does not say how the comparison is implemented, and a sign-magnitude
// integer comparison would order negatives backwards. Decode m values are
// negative in general, so the staggered fixture here uses negative m
// deliberately and the runner checks the reduced maximum bit-exactly. If the
// part orders negatives wrongly, this probe reports it rather than inheriting
// the assumption.
//
// Numerics
// --------
// The combine is the standard log-sum-exp merge
//
//   m_new    = max(m_curr, m_tile)
//   a_curr   = exp(m_curr - m_new),  a_tile = exp(m_tile - m_new)
//   l_new    = l_curr . a_curr + l_tile . a_tile
//   Acc_new  = Acc_curr . a_curr + Acc_tile . a_tile
//
// one `Exp` (<= 1 ULP, documented) and three multiplies per term, so the
// accumulator carries about 2 ULP per combine step against an FP32 host
// oracle and is NOT bit-comparable in general. The probe therefore drives two
// fixtures: a `uniform` one where every split carries the SAME m, which makes
// both rescale factors exactly 1.0f and the whole combine an exact integer
// sum -- bit-comparable, order-independent, and so a lost update detector
// rather than a tolerance check -- and a `staggered` one with distinct
// negative m, which is the only fixture that exercises the rescale at all and
// is checked against the oracle to a stated tolerance.

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#endif

#if defined(__CCE_AICORE__)
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#endif

namespace ascend_micro {

// ---------------------------------------------------------------- geometry
constexpr std::uint32_t kProbeChannels = 256;   // d, one KV head
constexpr std::uint32_t kProbeStateLanes = 8;   // FP32 per 32-byte DataBlock
// The state record, in FP32 elements: m block, l block, then the accumulator.
constexpr std::uint32_t kProbeStateElems = 2u * kProbeStateLanes;      // 16
constexpr std::uint32_t kProbeRecordElems = kProbeStateElems + kProbeChannels;
constexpr std::uint32_t kProbeRecordBytes = kProbeRecordElems * 4u;    // 1088
constexpr std::uint32_t kProbeAccBytes = kProbeChannels * 4u;          // 1024
constexpr std::uint32_t kProbeAccBursts = kProbeAccBytes / 32u;        // 32
// Element offsets inside the record.
constexpr std::uint32_t kProbeMOffset = 0u;
constexpr std::uint32_t kProbeLOffset = kProbeStateLanes;
constexpr std::uint32_t kProbeAccOffset = kProbeStateElems;

static_assert(kProbeRecordBytes % 32u == 0u,
              "the state record must be a whole number of DataBlocks");
static_assert(kProbeAccBursts == 32u,
              "the accumulator transfer is the task's 32-burst MTE3 payload");

// The widest split count the probe sweeps. K is a runtime launch dimension;
// this bounds only the host-side fixtures and the per-block metric slices.
constexpr std::uint32_t kProbeMaxSplits = 8u;
// Per-split repetitions whose cycle counts are recorded individually. Four is
// the smallest count that puts a split past the two-slot ring depth the
// decode operator uses, which section 3.1 of the guidelines requires before a
// reported interval is called steady state.
constexpr std::uint32_t kProbeMaxReps = 8u;

// One metric slice per block. Each metric occupies a whole 32-byte DataBlock
// of eight replicated int32 lanes, because the only way to publish a scalar
// from the Vector pipe without a UB subscript is `Duplicate` from a scalar
// into a DataBlock-aligned destination (section 1.2 and the AIV-only
// operators' launch probes do exactly this). The host reads lane 0 of each
// slot and may use the other seven as a corruption check: a slot whose eight
// lanes disagree was not written by one `Duplicate`.
constexpr std::uint32_t kProbeMetricSlots = 24u;
constexpr std::uint32_t kProbeMetricLanesPerSlot = 8u;
constexpr std::uint32_t kProbeMetricElems =
    kProbeMetricSlots * kProbeMetricLanesPerSlot;              // 192 int32
constexpr std::uint32_t kProbeMetricBytes = kProbeMetricElems * 4u;  // 768
enum ProbeMetric : std::uint32_t {
    kMetricBlock = 0,     // GetBlockIdx(), so an idle core is distinguishable
    kMetricMode = 1,      // the mode this launch ran
    kMetricReps = 2,      // repetitions performed
    kMetricRegion = 3,    // cycles, whole measured region, all repetitions
    kMetricPhase0 = 4,    // cycles, summed: Mode A store / Mode C normalise
    kMetricPhase1 = 5,    // cycles, summed: atomic maximum / Mode C rotate
    kMetricPhase2 = 6,    // cycles, summed: lock acquisition / Mode C gate
    kMetricPhase3 = 7,    // cycles, summed: critical section / Mode C store
    kMetricPhase4 = 8,    // cycles, summed: lock release
    kMetricRepBase = 16,  // kProbeMaxReps slots of per-repetition cycles
};
static_assert(kMetricRepBase + kProbeMaxReps <= kProbeMetricSlots,
              "the per-repetition cycle slots must fit the metric slice");
static_assert(kProbeMetricBytes % 32u == 0u,
              "the metric slice must be a whole number of DataBlocks");

// Modes, matching the task's naming.
enum class ProbeMode : std::uint32_t {
    CollisionFree = 0,  // Mode A
    SharedAtomic = 1,   // Mode B
    TailEgress = 2,     // Mode C
};

// ------------------------------------------------ the hardware split-K path
// Everything above is the CAModel probe, whose instruction budget and cycle
// figures `reports/l2_atomic_contention_split_k.md` records. Everything below
// is ADDITIVE and exists for the physical-device benchmark
// (`tests/npu/bench/bench_turboquant_decode_splitk.cpp`): new constants and
// new entry points, with no line of the probe's own surface changed. That is
// deliberate and it is the repository's own precedent -- the recurrence leaf
// gained `vector_decode_recur_rows` the same way -- because the probe's
// published numbers were measured against the unbounded spin and an edit to it
// would silently invalidate them.
//
// Two things the hardware path needs that the probe does not.
//
// **A watchdog.** On CAModel a lost lock hangs a simulation someone can kill.
// On silicon it hangs a device, and the task requires that it cannot. So the
// hardware acquire below bounds its attempts and, on expiry, gives up WITHOUT
// taking the lock, records that it did, and leaves the lock untouched for
// whoever actually holds it. The release reads that same per-slot record and
// clears the lock only if this split really acquired it, so a timed-out split
// cannot release a lock it never held -- which would be worse than hanging,
// because it would corrupt the protocol silently and still produce numbers.
//
// The bound is an ATTEMPT count, not a cycle budget: `asc_get_system_cycle` is
// `__aicore__` and a SIMT VF may call only `__simt_callee__` helpers, so there
// is no clock to read inside the spin. The benchmark converts: one failed
// attempt is one atomic round trip, measured on CAModel at a few hundred
// ticks, so a bound of N attempts is on the order of N * 500 cycles. The
// report states the conversion rather than implying a wall-clock guarantee.
//
// **Absolute timestamps.** The probe publishes per-phase cycle DELTAS, which
// cannot express a tail: a split's delta says how long it took, not when it
// finished. The system counter is global across cores on this part, so the
// hardware path publishes absolute entry and exit stamps as (hi, lo) int32
// pairs in the slots the probe left free, and the host computes the spread
// between the first and last split to finish.
constexpr std::uint32_t kSplitKMaxSplits = 16u;
static_assert(kSplitKMaxSplits >= kProbeMaxSplits,
              "the hardware sweep must cover at least the probe's split range");

// The free metric slots, 9..15. The probe uses 0..8 and 16..23.
enum SplitKMetric : std::uint32_t {
    kMetricBeginHi = 9,   // absolute system cycle at region entry, high word
    kMetricBeginLo = 10,
    kMetricEndHi = 11,    // absolute system cycle at region exit
    kMetricEndLo = 12,
    kMetricAcquired = 13,  // acquisitions this split actually obtained
    kMetricTimeouts = 14,  // acquisitions the watchdog gave up on
    kMetricSpins = 15,     // failed exchanges, summed over the repetitions
};
// Both sides cast to a common type on purpose: `SplitKMetric` and
// `ProbeMetric` are distinct enumerations, and comparing them directly is
// -Wenum-compare, which the physical-device leg builds with -Werror even
// though the simulator leg does not.
static_assert(static_cast<std::uint32_t>(kMetricSpins) <
                  static_cast<std::uint32_t>(kMetricRepBase),
              "the hardware metric slots must not overrun the probe's "
              "per-repetition slots");

#if defined(__CCE_AICORE__)

// ===================================================================== SIMT
// The Global Memory synchronisation primitives. One warp per dispatch, with
// lane 0 doing the work: a thread block of 32 is the documented
// recommendation (blockDim a multiple of the 32-thread warp), and a lock is
// single-threaded by definition, so the other 31 lanes retire on the guard.
//
// Every store here is `asc_stcg`, which stores through L2/Global Memory
// WITHOUT populating the per-core SIMT DataCache. That is required and not a
// precaution: the DataCache is not coherent with Global Memory, so a plain
// thread store of the spin census or the ticket could sit in one core's
// DataCache where neither the host nor another core can see it. The atomics
// themselves operate in L2 and need no such treatment.

// Test-and-set acquisition, the shape the task specifies: spin on
// `asc_atomic_exch` until it hands back the free value. Each failed attempt
// is a full atomic round trip to the shared line, which is precisely the
// contention this probe is built to measure -- a test-and-TEST-and-set that
// spins on a plain `asc_ldcg` read and only then attempts the exchange would
// measure a different (and cheaper) protocol, and the report says so rather
// than quietly substituting it.
//
// `ticket` is read inside the critical section and logged per acquisition, so
// mutual exclusion is witnessed independently of the arithmetic: across the
// whole launch the logged tickets must be a permutation of 0 .. K*reps-1.
__simt_vf__ __launch_bounds__(32) inline void probe_lock_acquire_vf(
    __gm__ std::uint32_t* lock, __gm__ std::uint32_t* ticket,
    __gm__ std::uint32_t* ticket_log, __gm__ std::uint32_t* spin_census,
    std::uint32_t slot)
{
    if (threadIdx.x != 0u) return;
    std::uint32_t spins = 0u;
    while (asc_atomic_exch(lock, 1u) != 0u) {
        ++spins;
    }
    // Held. Take a ticket, which both orders the acquisitions for the host
    // check and proves the counter was reached under the lock.
    const std::uint32_t mine = asc_atomic_add(ticket, 1u);
    asc_stcg(ticket_log + slot, mine);
    asc_stcg(spin_census + slot, spins);
}

// Release. The fence is what makes the data the critical section wrote
// visible before the lock word that advertises it is cleared; the caller is
// separately responsible for having RETIRED that write on the MTE3 queue
// before this VF is dispatched, because a fence orders accesses and does not
// wait for another pipe's outstanding burst.
__simt_vf__ __launch_bounds__(32) inline void probe_lock_release_vf(
    __gm__ std::uint32_t* lock)
{
    if (threadIdx.x != 0u) return;
    asc_threadfence();
    (void)asc_atomic_exch(lock, 0u);
}

// The scalar reduction: one hardware atomic, no lock.
//
// This split's m is read HERE, by a thread, with `asc_ldcg` -- not by the
// Scalar unit on the SIMD side and not out of Unified Buffer. Three reasons,
// and the first two are hard constraints:
//
//   * a scalar read of a Unified Buffer value the Vector pipe produced needs
//     a V_S fence, which costs a full vector drain and needs a TPipe this
//     kernel does not have (guidelines section 1.2). The UB copy of m is a
//     lane-replicated DataBlock and stays on the Vector pipe.
//   * a `GlobalTensor::GetValue` on the SIMD side would do it without
//     touching UB, and the decode operator resolves its page that way -- but
//     the library implements that scalar Global Memory read with an MTE2 -> S
//     synchronisation on a RESERVED event id, which the trace audit rejects
//     and should: ids 6 and 7 belong to the library. Measured, on this probe,
//     before this VF read the value itself.
//   * `asc_ldcg` loads through L2/Global Memory WITHOUT populating the
//     per-core SIMT DataCache, which is what a read of possibly-shared state
//     has to do -- the DataCache is not coherent with Global Memory.
__simt_vf__ __launch_bounds__(32) inline void probe_state_max_vf(
    __gm__ float* gm_m, __gm__ float* gm_tile_m)
{
    if (threadIdx.x != 0u) return;
    const float m_tile = asc_ldcg(gm_tile_m);
    (void)asc_atomic_max(gm_m, m_tile);
}

// ------------------------------------------- the hardware SIMT entry points
// Watchdog-bounded counterparts of the two lock VFs above. They are separate
// functions rather than a parameter on the originals so that the probe's
// measured instruction budget is untouched; see the note beside
// `kSplitKMaxSplits`.

// Bounded test-and-set. Returns nothing; publishes what happened:
//
//   acquired[slot] = 1 and the lock is HELD      -- the normal path
//   acquired[slot] = 0 and the lock is NOT held  -- the watchdog gave up
//
// `max_attempts` of 0 means unbounded, which reproduces the probe's behaviour
// exactly and is what a CAModel run uses; the benchmark always passes a bound.
// The caller must pair this with `splitk_lock_release_vf` and the SAME
// `acquired` slot, or a timed-out split will release a lock it never held.
__simt_vf__ __launch_bounds__(32) inline void splitk_lock_acquire_vf(
    __gm__ std::uint32_t* lock, __gm__ std::uint32_t* ticket,
    __gm__ std::uint32_t* ticket_log, __gm__ std::uint32_t* spin_census,
    __gm__ std::uint32_t* acquired, std::uint32_t slot,
    std::uint32_t max_attempts)
{
    if (threadIdx.x != 0u) return;
    std::uint32_t spins = 0u;
    std::uint32_t got = 0u;
    while (true) {
        if (asc_atomic_exch(lock, 1u) == 0u) {
            got = 1u;
            break;
        }
        ++spins;
        if (max_attempts != 0u && spins >= max_attempts) break;
    }
    if (got != 0u) {
        // Held. The ticket orders the acquisitions for the host's mutual
        // exclusion check, exactly as in the probe.
        const std::uint32_t mine = asc_atomic_add(ticket, 1u);
        asc_stcg(ticket_log + slot, mine);
    } else {
        // Not held. A ticket is deliberately NOT taken: the ticket sequence
        // must stay a permutation over the acquisitions that really happened,
        // or the host's permutation check would fail for a reason that has
        // nothing to do with exclusion. The sentinel marks the gap instead.
        asc_stcg(ticket_log + slot, 0xFFFFFFFFu);
    }
    asc_stcg(acquired + slot, got);
    asc_stcg(spin_census + slot, spins);
}

// Release, conditional on this split actually holding the lock. The flag is
// read with `asc_ldcg` inside the VF -- through L2/Global Memory, bypassing
// the per-core SIMT DataCache, which is not coherent with Global Memory -- so
// no SIMD-side scalar read is needed and no reserved event id is consumed.
__simt_vf__ __launch_bounds__(32) inline void splitk_lock_release_vf(
    __gm__ std::uint32_t* lock, __gm__ std::uint32_t* acquired,
    std::uint32_t slot)
{
    if (threadIdx.x != 0u) return;
    if (asc_ldcg(acquired + slot) == 0u) return;  // never held it
    asc_threadfence();
    (void)asc_atomic_exch(lock, 0u);
}

// ===================================================================== SIMD
// Engine state. One instance per core; every tensor is a hand-placed
// `LocalTensor` over the kernel's UB arena, as everywhere else in this
// harness, and no `TPipe`, `TBuf` or `LocalMemAllocator` appears in the
// translation unit -- the three share a UB base and mixing two of them
// aliases silently.
// Each of the three (m, l) pairs below is TWO ADJACENT replicated DataBlocks
// in one 64-byte slot, m first, with no bank skew between them: that
// adjacency is what lets the pair move as a single 2-DataBlock burst, and the
// kernel static_asserts both the alignment of the slot and the 32-byte
// spacing of the two views inside it.
struct L2AtomicProbeEngine {
    // ---- the resident tile state, this core's contribution ---------------
    AscendC::LocalTensor<float> m_tile;     // slot + 0, one replicated DataBlock
    AscendC::LocalTensor<float> l_tile;     // slot + 32
    AscendC::LocalTensor<float> pair_tile;  // the 16-element view of both
    AscendC::LocalTensor<float> acc_tile;   // kProbeChannels

    // ---- the fetched shared state ---------------------------------------
    AscendC::LocalTensor<float> m_curr;
    AscendC::LocalTensor<float> l_curr;
    AscendC::LocalTensor<float> pair_curr;
    AscendC::LocalTensor<float> acc_curr;

    // ---- the combined state, written back --------------------------------
    AscendC::LocalTensor<float> m_new;
    AscendC::LocalTensor<float> l_new;
    AscendC::LocalTensor<float> pair_new;
    AscendC::LocalTensor<float> acc_new;

    // ---- rescale scratch -------------------------------------------------
    AscendC::LocalTensor<float> alpha_curr;     // replicated block
    AscendC::LocalTensor<float> alpha_tile;     // replicated block
    AscendC::LocalTensor<float> alpha_curr_bc;  // kProbeChannels
    AscendC::LocalTensor<float> alpha_tile_bc;  // kProbeChannels
    AscendC::LocalTensor<float> term_a;         // kProbeChannels
    AscendC::LocalTensor<float> term_b;         // kProbeChannels
    AscendC::LocalTensor<float> state_tmp;      // replicated block
    AscendC::LocalTensor<float> state_tmp2;     // replicated block

    // ---- Global Memory views ---------------------------------------------
    // The shared head slot (Mode B) and this core's private slice (Mode A)
    // are the same record shape at different addresses, so one view serves
    // both and the mode decides the element offset.
    AscendC::GlobalTensor<float> workspace;
    std::uint32_t record_base = 0;  // elements into `workspace`

    static constexpr std::int32_t kLanes = static_cast<std::int32_t>(kProbeStateLanes);
    static constexpr std::int32_t kChannels = static_cast<std::int32_t>(kProbeChannels);

    // m_new = max(m_curr, m_tile), then the two rescale factors, then the
    // combine. Every destination is distinct from every source of the same
    // instruction: the egress leaf's header records what three in-place
    // vector operations cost to debug, and this leaf does not repeat it.
    __aicore__ inline void combine() const
    {
        AscendC::Max(m_new, m_curr, m_tile, kLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(state_tmp, m_curr, m_new, kLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(alpha_curr, state_tmp, kLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(state_tmp, m_tile, m_new, kLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(alpha_tile, state_tmp, kLanes);
        AscendC::PipeBarrier<PIPE_V>();

        // l_new = l_curr . a_curr + l_tile . a_tile, on the replicated block.
        AscendC::Mul(state_tmp, l_curr, alpha_curr, kLanes);
        AscendC::Mul(state_tmp2, l_tile, alpha_tile, kLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(l_new, state_tmp, state_tmp2, kLanes);

        // The accumulator. `Duplicate`'s tensor form broadcasts lane 0 of its
        // source across `count` lanes; on dav-3510 it reads a whole 256-byte
        // vector register from the source base regardless of count, so the
        // caller keeps that much addressable UB behind each alpha block --
        // the kernel static_asserts it.
        AscendC::Duplicate(alpha_curr_bc, alpha_curr, kChannels);
        AscendC::Duplicate(alpha_tile_bc, alpha_tile, kChannels);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(term_a, acc_curr, alpha_curr_bc, kChannels);
        AscendC::Mul(term_b, acc_tile, alpha_tile_bc, kChannels);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(acc_new, term_a, term_b, kChannels);
    }

    // The two MTE2 bursts that fetch the shared record: 2 DataBlocks of
    // (m, l) and the 32-DataBlock accumulator. The state pair moves through
    // `pair_curr`, the 16-element view spanning both replicated blocks, and
    // not through `m_curr` with a count of 16 -- the two address the same
    // bytes, but only the first says so.
    __aicore__ inline void fetch_record() const
    {
        AscendC::DataCopy(pair_curr, workspace[record_base + kProbeMOffset],
                          kProbeStateElems);
        AscendC::DataCopy(acc_curr, workspace[record_base + kProbeAccOffset],
                          kProbeChannels);
    }

    // The two MTE3 bursts that publish it.
    __aicore__ inline void store_record() const
    {
        AscendC::DataCopy(workspace[record_base + kProbeMOffset], pair_new,
                          kProbeStateElems);
        AscendC::DataCopy(workspace[record_base + kProbeAccOffset], acc_new,
                          kProbeChannels);
    }

    // Mode A's publication: this core's OWN tile state to its OWN slice. No
    // fetch, no combine, no lock -- the same two bursts and nothing else, so
    // the difference between Mode A and Mode B is exactly the synchronisation
    // and the read-modify-write and not a different burst shape.
    __aicore__ inline void store_tile() const
    {
        AscendC::DataCopy(workspace[record_base + kProbeMOffset], pair_tile,
                          kProbeStateElems);
        AscendC::DataCopy(workspace[record_base + kProbeAccOffset], acc_tile,
                          kProbeChannels);
    }
};

#endif  // __CCE_AICORE__

}  // namespace ascend_micro
