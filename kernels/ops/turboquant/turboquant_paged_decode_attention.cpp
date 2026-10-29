// The legacy camodel_* entry keeps the fixed validation ABI described below.
// turboquant_paged_decode_splitk_aic adds runtime request/head/split geometry,
// page_begin + page_count windows, carried multi-page state and partial output.
//
// Fused TurboQuant paged DECODE attention, Ascend 950PR / arch35 / dav-3510 /
// CANN 9.2, MIX_AIC_1_2: one Cube core and BOTH Vector cores.
//
//   batch / query tokens  M = 16          head dimension  d = 256
//   vLLM page             128 tokens = 8 strips of 16      KV heads  4
//   grid                  4 KV heads x 8 temporal blocks = 32 tasks,
//                         one per physical AIC
//
//   engine        owner   work
//   ------------  ------  ------------------------------------------------
//   MTE2          AIC     Q and the Pi table, prologue only (GM -> L1)
//   MTE1/M/FIX    AIC     the 64-Mmad ingress rotation, then one Mmad and one
//                         Fixpipe per strip
//   MTE2          AIV1    packed int4 key strip ingress (GM -> UB)
//   Vector        AIV1    key nibble -> bf16
//   MTE3          AIV1    UB -> L1 direct key staging (the CV path)
//   MTE2          AIV0    packed int4 value strip and the two scale planes
//   Vector        AIV0    value nibble -> FP32, the online-softmax recurrence,
//                         the egress rotation and the sigmoid gate
//   MTE3          AIV0    one write-out, after the strip sequence
//
// Leaves, used through their published entry points:
//   routines/routine_stage_page_zn_bf16.hpp        ND -> Nz ingress
//   routines/routine_fractal_load_zn_bf16.hpp      L1 -> L0, byte identity
//   routines/routine_cube_decode_score.hpp         rotation + score GEMM
//   routines/routine_turboquant_dequant4.hpp       int4 -> bf16 / FP32
//   routines/routine_cv_transfer_ub_to_l1.hpp      UB -> L1, ND -> Nz
//   routines/routine_vector_decode_recurrence.hpp  the (m, l, Acc) recurrence
//   routines/routine_vector_egress_rotate.hpp      Acc . Pi, then the gate
//
//
// The eight-strip modulo schedule
// ===============================
// Each physical 128-token page is eight [16 token x 256 channel] strips, and
// strip s occupies five pipeline stages across five consecutive columns:
//
//  T ->  [T0 ][T1 ][T2 ][T3 ][T4 ][T5 ][T6 ][T7 ][T8 ][T9 ][T10][T11]
//  MTE2  LdK0 LdK1 LdK2 LdK3 LdK4 LdK5 LdK6 LdK7  --   --   --   --   AIV1
//  AIV1   --  DqK0 DqK1 DqK2 DqK3 DqK4 DqK5 DqK6 DqK7  --   --   --   dequant + CV
//  MTE2   --   --  LdV0 LdV1 LdV2 LdV3 LdV4 LdV5 LdV6 LdV7  --   --   AIV0
//  AIC    --   --  QK0^ QK1^ QK2^ QK3^ QK4^ QK5^ QK6^ QK7^  --   --   Mmad + Fixpipe
//  AIV0   --   --   --  DqV0 DqV1 DqV2 DqV3 DqV4 DqV5 DqV6 DqV7  --   dequant
//  AIV0   --   --   --   --  S0V0 S1V1 S2V2 S3V3 S4V4 S5V5 S6V6 S7V7  recurrence
//
// so column j runs, all mutually independent:
//
//   AIV1  MTE2 ingress of strip j,    Vector dequant + CV stage of strip j - 1
//   AIC   Mmad and Fixpipe of strip j - 2
//   AIV0  MTE2 ingress of strip j - 2, dequant of strip j - 3,
//         recurrence of strip j - 4
//
// Twelve columns cover eight strips; the schedule is a fold over a
// compile-time index sequence with runtime guards, so every ring slot index is
// a constant expression. Ring depth is **two** everywhere -- the "double
// window" the score panels are named for -- which makes eight strips wrap
// every ring four times, so every buffer-reuse back edge below carries real
// contention rather than being merely primed and drained.
//
// Two deviations from the schedule as specified, and why
// ------------------------------------------------------
// 1. **Q is staged in the Cube prologue, not spread over T0..T7.** The
//    specified Gantt draws eight `LdQ` cells on the AIC's GM -> L1 row, but Q
//    is strip invariant and the first `Q.K0^T` lands in T2, so a Q load in T3
//    or later could not be an input to it. What is genuinely eight-wide about
//    the ingress rotation is its *Mmad* nest -- eight source groups per output
//    group -- and that runs in the prologue, before T0. The MTE2 row is
//    therefore idle on the AIC during the strip sequence, which is what frees
//    that queue for the two Vector cores' packed ingress.
// 2. **Mmad and Fixpipe of one strip are issued together, attributed to
//    T(s+2) and T(s+3).** They are paired by matching `unitFlag = 3`, which is
//    the hardware mechanism for exactly this overlap: the Fixpipe drains each
//    fractal as the Mmad finishes it. Splitting them into two software stages
//    would need an explicit FIX_M token per strip and would forbid the
//    pairing, so the overlap the Gantt shows is obtained the way the hardware
//    offers it rather than by a second schedule column.
//
//
// The marked graph
// ================
// Vertices are pipeline stages, edges carry tokens. Every directed cycle holds
// exactly one token against a two-slot ring, so the graph is live and bounded
// (liveness: sum of tokens on any cycle >= 1; boundedness: <= slot count).
//
// Forward edges
//   AIV1  MTE2 -> V      MTE2_V       packed key page readable
//   AIV1  V    -> MTE3   V_MTE3       bf16 key page readable by the CV burst
//   AIV1  MTE3 -> AIC    kFlagKeyReady    L1 key page staged      (cross core)
//   AIV0  MTE2 -> V      MTE2_V       packed value page readable
//   AIV0  V    -> AIC    kFlagValueReady  value page dequantised  (cross core)
//   AIC   FIX  -> AIV0   kFlagScoreReady  score window written    (cross core)
//
// Back edges, all of them buffer reuse, each primed once and drained once
//   AIV1  V    -> MTE2   V_MTE2       packed key slot free after the dequant
//   AIV1  MTE3 -> V      MTE3_V       bf16 key page free after the CV burst
//   AIV0  V    -> MTE2   V_MTE2       packed value and scale slot free, set by
//                                     the RECURRENCE -- the dequant is not the
//                                     slot's last reader (see RecurStrip)
//   AIV0  V    -> AIC    kFlagScoreFree   score window free       (cross core)
//   AIC   M    -> MTE1   M_MTE1       L0B key slot free
//   AIC   FIX  -> M      FIX_M        L0C score window free
//
// Two rings need **no** edge at all, and that is a property of the engine
// assignment rather than an omission:
//
//   * the dequantised FP32 value ring. Its producer (the dequant) and its
//     consumer (the recurrence) are both Vector operations on AIV0, and the
//     Vector queue retires in issue order, so the RAW and the WAR are both
//     already expressed by the order the two stages are emitted in. A
//     HardEvent here would be a no-op that still had to be primed and drained.
//   * the dequant scratch, single buffered for the same reason.
//
// A mode-4 set is a scalar instruction, so it needs a drain
// --------------------------------------------------------
// `CrossCoreSetFlag<4, PIPE_x>` lowers to `set_intra_block(PIPE_x, id)`, which
// issues on the FLOWCTRL unit -- the scalar stream. The `PIPE_x` argument does
// NOT hold the set back until pipe x has drained, and on the prefill operator
// that cost 1583 cycles of run-ahead and wrong numbers
// (MICROROUTINE_DESIGN_GUIDELINES.md section 6.2a). So every cross-core set
// whose meaning is "my earlier work on pipe x is finished" is preceded here by
//
//   SetFlag<HardEvent::x_S>(id); WaitFlag<HardEvent::x_S>(id);
//
// which stalls the scalar stream behind pipe x. There is no MTE1_S route, so
// the Cube's L1-slot-free edge rides the FIX_S drain: the Fixpipe is paired to
// the Mmad by unitFlag = 3 and the Mmad consumed L0B, so a retired Fixpipe
// proves the L1 -> L0 load has read the L1 page. Cross-core *waits* need no
// drain: `wait_intra_block` also stalls the scalar stream.
//
// Flag ids, and the one risk this kernel carries
// ----------------------------------------------
// The id space is per sub block: the library's `GetIntraFlagId` adds
// `subBlockId * INTRA_NUM` with `INTRA_NUM == 16`, so ids 0..15 are AIV0's
// channel and 16..31 are AIV1's, and a participant names the peer's channel by
// the offset. Renumbering AIV1 into 0..15 deadlocks the kernel; that is
// measured, not assumed (section 6.2d). Reserved ids 13 and 14 and their +16
// images are avoided, and no raw id reaches 11.
//
//   kFlagValueReady   0..1    AIV0 -> AIC
//   kFlagScoreFree    4..5    AIV0 -> AIC
//   kFlagScoreReady   8..9    AIC  -> AIV0
//   kFlagKeyReady    16..17   AIV1 -> AIC
//
// Four channels in the widest direction (AIV0 -> AIC), two from AIC to AIV0,
// two from AIV1 to AIC and none from AIC to AIV1 -- inside the task's cap of
// six per direction.
//
// **The risk.** Section 6.2c records a measured case where a mode-4 id reused
// across ring slots released a wait 2179 cycles before the announcement it was
// waiting for: a stale set from an earlier announcement on the same id. NOTE,
// 2026-10-29: section 6.2c has since been corrected. That stale release was
// section 6.2a's hazard -- an announcement issued on the scalar unit ahead of
// the pipeline work it reported -- and not a non-consuming latch; recycled
// mode-4 ids ARE sound when every announcement is drained first, which
// tests/research/kernel_cluster_sync_probe.cpp measured on silicon and on
// CAModel. So the id counts below are no longer a hazard in themselves. What
// this kernel has NOT had is scripts/audit_cross_core_fences.py run over it,
// which is the gate for the precondition; do that before relying on the
// reasoning in the rest of this comment. With two-slot rings over eight
// strips each id here carries **four** announcements. The 8-strip rung measured that risk coming true on
// kFlagL1Free -- see the kL1End comment for the ticks -- and that edge is now
// gone, its ring replaced by one L1 page per strip. The three surviving
// channels still carry four announcements each, so the same risk applies to
// them in principle; what makes them sound is not the id count but that each
// one's consumer is measurably slower than its producer, and the trace audit
// now checks the buffers those edges protect directly rather than trusting
// the flags. That configuration is exactly the one the guidelines
// call unproven, and the specified contract -- at most six channels per
// direction, ids below 11 -- cannot be met any other way at this strip count:
// one id per announcement would need 8 for the key edge alone.
//
// What this kernel does about it is make the reuse *ordered* and the ordering
// *measurable*:
//
//   every id is half of a two-edge cycle, and the set for strip s + 2 is
//   issued only after a wait on the paired opposite-direction flag for strip
//   s, which the peer issues only after its wait on this very id consumed the
//   strip-s announcement.
//
// So on each id the global sequence must be SET, WAIT, SET, WAIT, ... with no
// two sets in flight. `scripts/analyze_paged_decode_trace.py` checks that
// interleaving per id across both core dumps, which turns the assumption into
// an observation for the runs actually performed. It does not prove the
// hardware clears a level on a wait, and the report says so.
//
//
// Cache format contract
// =====================
//   key_cache    [num_blocks][kv_heads][8 strips][2048 B]
//                nibble index inside a strip = token * 256 + channel.
//                Token major, so the dequantised UB page is the ND matrix
//                K[token][channel], whose Nz image is byte for byte the Zn
//                image of K^T that the Cube B operand needs -- the KV cache is
//                already stored in the fractal order the Cube wants, which is
//                what a production paged cache does.
//   value_cache  [num_blocks][kv_heads][8 strips][2048 B]
//                nibble index inside a strip = channel * 16 + token.
//                Channel major, 2048-byte contiguous burst per strip. The
//                token axis being contiguous is what makes the weighted value
//                accumulation a per-channel ReduceRepeat over 16 lanes.
//   scale_cache  [2][num_blocks][kv_heads][128] FP32, planar: plane 0 the key
//                scale per token, plane 1 the value scale.
//   block_table  [temporal_blocks], logical temporal block -> physical page.
//
// Scales are applied outside the 4096-entry pages, both times exactly:
//
//   * the key scale multiplies the 16 score columns after the Fixpipe. The
//     Cube's key operand stays an exact small integer in bf16 and the scales
//     are powers of two, so the whole score panel is an exact multiple of a
//     power of two and is checked bit-for-bit rather than against a tolerance.
//     16 FP32 multiplies replace 4096 bf16 ones.
//   * the value scale folds into the probabilities before the contraction, 16
//     multiplies instead of 4096, with the denominator still summing the
//     unscaled probabilities.
//
//
// What this kernel emits, and what a 1024-token answer still needs
// ================================================================
// Each of the 32 tasks owns one 128-token page, so with a temporal split the
// eight blocks of one KV head hold eight partial softmaxes of one 1024-token
// context. This kernel writes, per block:
//
//   out_qrot     the rotated query, so the 64-Mmad prologue is checkable on
//                its own, bit-exactly, without the strip pipeline
//   out_scores   all eight raw score panels, bit-exactly
//   out_state    m and l, as the replicated 32-byte blocks they are carried in
//   out_acc      the raw accumulator, still in the cache's rotated basis
//   out_context  the fused egress: Acc . Pi, gated by sigmoid(gate), bf16
//
// The rotation is linear and commutes with a cross-block log-sum-exp combine;
// **the sigmoid gate does not**. So for a genuine 1024-token result the combine
// must happen before the gate, and `out_context` is the exact answer only when
// a block's page is the whole context for its head -- which is precisely the
// regime the single-tile canary protocol runs in. `out_acc` and `out_state`
// are what a combine would consume. The combine itself is a separate reduction
// and is NOT in this kernel; the report states that rather than implying a
// 1024-token number this kernel does not produce.

#include <cstdint>
#include <utility>
#include "../../../include/turboquant_splitk_contract.hpp"

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "../../../routines/routine_stage_page_zn_bf16.hpp"
#include "../../../routines/routine_fractal_load_zn_bf16.hpp"
#include "../../../routines/routine_cube_decode_score.hpp"
#include "../../../routines/routine_turboquant_dequant4.hpp"
#include "../../../routines/routine_cv_transfer_ub_to_l1.hpp"
#include "../../../routines/routine_vector_decode_recurrence.hpp"
#include "../../../routines/routine_vector_egress_rotate.hpp"
#endif

namespace paged_decode {

// ------------------------------------------------------------------ geometry
// Restated here rather than shared with tests/common/paged_decode_oracle.hpp:
// a device kernel does not include test headers. The runner checks the two
// agree on everything observable from the host.
constexpr std::uint32_t kTokens = 16;
constexpr std::uint32_t kHeadSize = 256;
constexpr std::uint32_t kGroup = 32;
constexpr std::uint32_t kGroups = kHeadSize / kGroup;
constexpr std::uint32_t kStrips = 8;
constexpr std::uint32_t kBlockSize = kStrips * kTokens;  // 128 tokens per page
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kTemporalBlocks = 8;
constexpr std::uint32_t kGrid = kKvHeads * kTemporalBlocks;  // 32
constexpr std::uint32_t kPanel = kTokens * kHeadSize;        // 4096
constexpr std::uint32_t kPackedBytes = kPanel / 2;           // 2048
constexpr std::uint32_t kScores = kTokens * kTokens;         // 256
constexpr std::uint32_t kTableBlocks = 2u * kGroups;         // 16
constexpr std::uint32_t kTableElems = kTableBlocks * kGroup * kGroup;
// Physical pages in the cache the host allocates. Only the scale cache's
// planar stride depends on it -- the packed caches are addressed by a byte
// offset the caller computes -- but it has to match the host exactly, so the
// runner asserts the agreement.
constexpr std::uint32_t kNumPages = kTemporalBlocks + 1u;  // 9

// Ring depth, and the column count the schedule needs for kStrips strips.
constexpr std::uint32_t kStageDepth = 2;
constexpr std::uint32_t kPipelineLag = 4;  // recurrence of strip s runs at T(s+4)
constexpr std::uint32_t kColumns = kStrips + kPipelineLag;  // 12

// The three sub-DataBlock egress stages each need an offset and a sign vector.
constexpr std::uint32_t kGatherStages = 3;

static_assert(kHeadSize == kGroups * kGroup, "the groups must tile the head");
static_assert(kStrips >= 2u * kStageDepth,
              "eight strips must wrap a two-slot ring at least twice, or no "
              "buffer-reuse back edge is ever contended");

}  // namespace paged_decode

#if defined(__CCE_AICORE__)

// Mode-4 intra-block flags, a Fixpipe that picks a Vector core, and the direct
// UB -> L1 channel all need a non-1:1 AIC:AIV ratio. At 1:1 there is one
// Vector core per Cube core, the sub-block index is meaningless, and the same
// DataCopy UB -> L1 silently becomes a KFC relay through Global Memory, which
// would reintroduce the workspace this kernel does not have.
#if defined(__MIX_CORE_AIC_RATION__) && (__MIX_CORE_AIC_RATION__ == 1)
#error "MIX_AIC_1_2 is required: a 1:1 AIC:AIV ratio has no second Vector core"
#endif

namespace paged_decode {

// ---------------------------------------------------------------- UB arena
// Rule AKA3003: every operand base 32-byte aligned, one DataBlock of bank skew
// between live slots, and VECTOR_REG_WIDTH of addressable UB behind the last
// operand, because a level-2 vector op reads a whole vector register per
// source, unmasked.
//
// One map serves both Vector cores -- each has its own 256 KB Unified Buffer
// and touches only the slots its role uses -- so there is a single set of
// alignment and capacity assertions, and the Cube can form its Fixpipe
// destination from the same constants the recurrence reads.
constexpr std::uint32_t kBlock = 32u;
constexpr std::uint32_t kBankSkewBytes = 32u;
constexpr std::uint32_t kReadPadBytes = 256u;  // VECTOR_REG_WIDTH on dav-3510
constexpr std::uint32_t kUbWindow = 192u * 1024u;

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkewBytes;
}

constexpr std::uint32_t kF32 = 4u;
constexpr std::uint32_t kB16 = 2u;
constexpr std::uint32_t kI16 = 2u;
constexpr std::uint32_t kStateLanes = ascend_micro::kDecodeStateLanes;  // 8
constexpr std::uint32_t kBcLanes = ascend_micro::kDecodeBcLanes;        // 64

constexpr std::uint32_t kScoreBytes = kScores * kF32;    // 1024
constexpr std::uint32_t kKeyPageBytes = kPanel * kB16;   // 8192
constexpr std::uint32_t kValuePageBytes = kPanel * kF32; // 16384
constexpr std::uint32_t kScalePairBytes = 2u * kTokens * kF32;  // 128

constexpr std::uint32_t kRingScore = kScoreBytes + kBankSkewBytes;      // 1056
constexpr std::uint32_t kRingPacked = kPackedBytes + kBankSkewBytes;    // 2080
constexpr std::uint32_t kRingKey = kKeyPageBytes + kBankSkewBytes;      // 8224
constexpr std::uint32_t kRingValue = kValuePageBytes + kBankSkewBytes;  // 16416
constexpr std::uint32_t kRingScale = kScalePairBytes + kBankSkewBytes;  // 160

// The Fixpipe landing ring stays at UB 0 so the Cube side can form a raw
// __ubuf__ pointer without depending on any Vector-side allocation. It is
// kStageDepth windows deep and no deeper: that is what makes the score-free
// back edge a real WAR guard once the ring wraps.
constexpr std::uint32_t kUbScoreRing = 0u;
constexpr std::uint32_t kUbPackedKRing = NextSlot(kUbScoreRing, kStageDepth * kRingScore);
constexpr std::uint32_t kUbPackedVRing =
    NextSlot(kUbPackedKRing, kStageDepth * kRingPacked);
constexpr std::uint32_t kUbKeyRing = NextSlot(kUbPackedVRing, kStageDepth * kRingPacked);
constexpr std::uint32_t kUbValueRing = NextSlot(kUbKeyRing, kStageDepth * kRingKey);
constexpr std::uint32_t kUbScaleRing = NextSlot(kUbValueRing, kStageDepth * kRingValue);

// Dequant scratch: ONE int16 plane, single buffered. Reuse across strips is a
// same-pipe Vector dependency, which the Vector queue orders in issue order.
// The key path needs no scratch at all -- int4b_t -> bfloat16_t is a single
// listed Cast straight into the ring slot -- and the value path needs this one
// plane only because int4b_t -> float is not a listed pair and has to go
// through int16_t. Against the four-plane unsigned unpack that is three fewer
// 4096-element passes per strip and 24 KiB of Unified Buffer.
constexpr std::uint32_t kUbDqSigned = NextSlot(kUbScaleRing, kStageDepth * kRingScale);

// Carried state: one replicated 32-byte block per query, so every query's
// operand is a whole DataBlock from a 32-byte aligned base.
constexpr std::uint32_t kUbMPrev = NextSlot(kUbDqSigned, kPanel * kI16);
constexpr std::uint32_t kUbLPrev = NextSlot(kUbMPrev, kTokens * kStateLanes * kF32);
constexpr std::uint32_t kUbAcc = NextSlot(kUbLPrev, kTokens * kStateLanes * kF32);

// Per-query recurrence scratch.
constexpr std::uint32_t kUbMNew = NextSlot(kUbAcc, kTokens * kHeadSize * kF32);
constexpr std::uint32_t kUbLNew = NextSlot(kUbMNew, kStateLanes * kF32);
constexpr std::uint32_t kUbAlpha = NextSlot(kUbLNew, kStateLanes * kF32);
constexpr std::uint32_t kUbAlphaDelta = NextSlot(kUbAlpha, kStateLanes * kF32);
constexpr std::uint32_t kUbLScaled = NextSlot(kUbAlphaDelta, kStateLanes * kF32);
constexpr std::uint32_t kUbTileMaxRaw = NextSlot(kUbLScaled, kStateLanes * kF32);
constexpr std::uint32_t kUbPSumRaw = NextSlot(kUbTileMaxRaw, kStateLanes * kF32);
constexpr std::uint32_t kUbTileMaxBc = NextSlot(kUbPSumRaw, kStateLanes * kF32);
constexpr std::uint32_t kUbMBc = NextSlot(kUbTileMaxBc, kBcLanes * kF32);
constexpr std::uint32_t kUbPSumBc = NextSlot(kUbMBc, kBcLanes * kF32);
constexpr std::uint32_t kUbAlphaBc = NextSlot(kUbPSumBc, kBcLanes * kF32);
constexpr std::uint32_t kUbScaledRow = NextSlot(kUbAlphaBc, kBcLanes * kF32);
constexpr std::uint32_t kUbCentered = NextSlot(kUbScaledRow, kTokens * kF32);
constexpr std::uint32_t kUbP = NextSlot(kUbCentered, kTokens * kF32);
constexpr std::uint32_t kUbPScaled = NextSlot(kUbP, kTokens * kF32);
constexpr std::uint32_t kUbPx4 = NextSlot(kUbPScaled, kTokens * kF32);
constexpr std::uint32_t kUbWeighted = NextSlot(kUbPx4, kBcLanes * kF32);
constexpr std::uint32_t kUbWeightedSum = NextSlot(kUbWeighted, kPanel * kF32);
constexpr std::uint32_t kUbAccScaled = NextSlot(kUbWeightedSum, kHeadSize * kF32);

// Score archive, one slot per strip, written by the recurrence from the panel
// it has just consumed and read back by the single end-of-sequence outflow.
// The Fixpipe ring is only kStageDepth windows deep, so without this the
// panels of strips 0..kStrips-3 would be overwritten before the sequence ends
// and only the last two would be checkable. The copy is a Vector Muls by 1.0f,
// exact for every finite value including signed zeros, so an archived panel is
// the Fixpipe output bit for bit.
constexpr std::uint32_t kUbScoreArchive = NextSlot(kUbAccScaled, kHeadSize * kF32);

// Egress operands.
constexpr std::uint32_t kUbLBroadcast = NextSlot(kUbScoreArchive, kStrips * kRingScore);
constexpr std::uint32_t kUbEgressA = NextSlot(kUbLBroadcast, kHeadSize * kF32);
constexpr std::uint32_t kUbEgressB = NextSlot(kUbEgressA, kHeadSize * kF32);
constexpr std::uint32_t kUbPermuted = NextSlot(kUbEgressB, kHeadSize * kF32);
constexpr std::uint32_t kUbSignedTmp = NextSlot(kUbPermuted, kHeadSize * kF32);
constexpr std::uint32_t kUbDenominator = NextSlot(kUbSignedTmp, kHeadSize * kF32);
constexpr std::uint32_t kUbGateRow = NextSlot(kUbDenominator, kHeadSize * kF32);
constexpr std::uint32_t kUbPiSign = NextSlot(kUbGateRow, kHeadSize * kF32);
constexpr std::uint32_t kUbGatherOffsets = NextSlot(kUbPiSign, kHeadSize * kF32);
constexpr std::uint32_t kUbGatherSigns =
    NextSlot(kUbGatherOffsets, kGatherStages * kHeadSize * kF32);
constexpr std::uint32_t kUbContext =
    NextSlot(kUbGatherSigns, kGatherStages * kHeadSize * kF32);
// The rotated, pre-gate rows. They exist so the runner can separate the
// rotation from the gate; the 24 KiB the signed dequant freed pays for them.
constexpr std::uint32_t kUbRotated = NextSlot(kUbContext, kTokens * kHeadSize * kB16);
constexpr std::uint32_t kArenaEnd = kUbRotated + kTokens * kHeadSize * kF32;

// ------------------------------------------------------------------ V0 gates
static_assert(kUbScoreRing % kBlock == 0 && kUbPackedKRing % kBlock == 0 &&
              kUbPackedVRing % kBlock == 0 && kUbKeyRing % kBlock == 0 &&
              kUbValueRing % kBlock == 0 && kUbScaleRing % kBlock == 0 &&
              kUbDqSigned % kBlock == 0 &&
              kUbMPrev % kBlock == 0 && kUbLPrev % kBlock == 0 &&
              kUbAcc % kBlock == 0 && kUbWeighted % kBlock == 0 &&
              kUbWeightedSum % kBlock == 0 && kUbAccScaled % kBlock == 0 &&
              kUbScoreArchive % kBlock == 0 && kUbEgressA % kBlock == 0 &&
              kUbEgressB % kBlock == 0 && kUbPermuted % kBlock == 0 &&
              kUbSignedTmp % kBlock == 0 && kUbDenominator % kBlock == 0 &&
              kUbGateRow % kBlock == 0 && kUbPiSign % kBlock == 0 &&
              kUbGatherOffsets % kBlock == 0 && kUbGatherSigns % kBlock == 0 &&
              kUbContext % kBlock == 0 && kUbRotated % kBlock == 0,
              "every UB operand base must be 32-byte aligned");
static_assert(kRingScore % kBlock == 0 && kRingPacked % kBlock == 0 &&
              kRingKey % kBlock == 0 && kRingValue % kBlock == 0 &&
              kRingScale % kBlock == 0,
              "every ring slot stride must be a whole number of DataBlocks");
static_assert(kRingScore % 256u != 0u && kRingPacked % 256u != 0u &&
              kRingKey % 256u != 0u && kRingValue % 256u != 0u &&
              kRingScale % 256u != 0u,
              "a 256-byte ring stride puts every slot on one bank phase");
static_assert(kArenaEnd + kReadPadBytes <= kUbWindow,
              "UB arena leaves no VECTOR_REG_WIDTH read pad behind the state");
// Duplicate's tensor form over-reads a whole 256-byte vector register from its
// source base, and the egress points it at the LAST query's state block.
static_assert(kUbLPrev + (kTokens - 1u) * kStateLanes * kF32 + kReadPadBytes <= kUbWindow,
              "Duplicate over-reads 256 bytes from the last l state block");
static_assert(kBcLanes * kF32 == 8u * kBlock, "a Brcb repeat writes eight DataBlocks");
static_assert(kPackedBytes % kBlock == 0 && kScalePairBytes % kBlock == 0 &&
              (kTokens * kHeadSize * kB16) % kBlock == 0,
              "every MTE2/MTE3 length must be whole 32-byte bursts");
static_assert(kPanel % ascend_micro::kDequant4MinElements == 0,
              "a dequant count must be a multiple of 64 nibbles");

// Slot addresses are variable templates, not constexpr functions: in the CCE
// dialect a plain constexpr function is __host__, so an __aicore__ function may
// not call one. A variable template is a compile-time value.
template <std::uint32_t Slot>
constexpr std::uint32_t kScoreSlot = kUbScoreRing + Slot * kRingScore;
template <std::uint32_t Slot>
constexpr std::uint32_t kPackedKSlot = kUbPackedKRing + Slot * kRingPacked;
template <std::uint32_t Slot>
constexpr std::uint32_t kPackedVSlot = kUbPackedVRing + Slot * kRingPacked;
template <std::uint32_t Slot>
constexpr std::uint32_t kKeySlot = kUbKeyRing + Slot * kRingKey;
template <std::uint32_t Slot>
constexpr std::uint32_t kValueSlot = kUbValueRing + Slot * kRingValue;
template <std::uint32_t Slot>
constexpr std::uint32_t kScaleSlot = kUbScaleRing + Slot * kRingScale;
template <std::uint32_t Strip>
constexpr std::uint32_t kArchiveSlot = kUbScoreArchive + Strip * kRingScore;

// ---------------------------------------------------------------- L1 layout
// L1 operands need 32-byte alignment; the 512-byte rule applies to L0A/L0B.
constexpr std::uint32_t kL1PiTable = 0u;
constexpr std::uint32_t kL1Q = NextSlot(kL1PiTable, kTableElems * kB16);
constexpr std::uint32_t kL1QRot = NextSlot(kL1Q, kPanel * kB16);
constexpr std::uint32_t kL1KeyRing = NextSlot(kL1QRot, kPanel * kB16);
constexpr std::uint32_t kL1SlotStride = kRingKey;
// One page per strip, NOT a kStageDepth ring. Measured, on the 8-strip rung:
// AIV1 staged slot 0 for strips 0, 2, 4 and 6 at ticks 6754, 8703, 10653 and
// 12608, while the Cube read that same L1 address at 7088, 26994, 66481 and
// 106697 -- so the reads for strips 2 and 4 both returned strip 6's keys and
// 480 of 2048 score entries were wrong. The kFlagL1Free back edge that was
// supposed to bound AIV1 is a mode-4 counter with the right id, the right
// pipe (its wait gates MTE3 and the burst is MTE3) and the right count, and
// the Cube's announcement is correctly drained behind its Fixpipe -- yet
// AIV1's third and fourth waits did not block. That is the stale-token
// mechanism MICROROUTINE_DESIGN_GUIDELINES.md section 6.2c records.
//
// kStrips pages cost 8 * 8224 = 65792 B of a 512 KB L1 and remove the WAR
// edge outright: every strip writes its own page and every page is read once,
// so there is no reuse for a back edge to order. The UB landing rings and the
// score windows keep theirs, because those are intra-core events and bound
// correctly. This is a documented deviation from the task's section 3, which
// asks for an L1 reuse back edge; the edge it asks for does not bound here.
constexpr std::uint32_t kL1End = kL1KeyRing + kStrips * kL1SlotStride;

static_assert(kL1PiTable % kBlock == 0 && kL1Q % kBlock == 0 &&
              kL1QRot % kBlock == 0 && kL1KeyRing % kBlock == 0 &&
              kL1SlotStride % kBlock == 0,
              "every L1 operand base must be 32-byte aligned");
static_assert(kL1End <= 512u * 1024u,
              "the per-strip L1 key pages exceed the L1 buffer");

template <std::uint32_t Strip>
constexpr std::uint32_t kL1KeyPage = kL1KeyRing + Strip * kL1SlotStride;

// ------------------------------------------------- cross-core flag ids
template <std::uint32_t Slot>
constexpr std::uint16_t kFlagValueReady = 0u + Slot;   //  0..1  AIV0 -> AIC
template <std::uint32_t Slot>
constexpr std::uint16_t kFlagScoreFree = 4u + Slot;    //  4..5  AIV0 -> AIC
template <std::uint32_t Slot>
constexpr std::uint16_t kFlagScoreReady = 8u + Slot;   //  8..9  AIC  -> AIV0
template <std::uint32_t Slot>
constexpr std::uint16_t kFlagKeyReady = 16u + Slot;    // 16..17 AIV1 -> AIC
// Full-page reuse handshake for the dynamic temporal entry. Mode-4 has
// sixteen logical IDs per vector sub-block; 10 and 16+10 are distinct targets.
constexpr std::uint16_t kFlagPageDrainedV0 = 10u;
constexpr std::uint16_t kFlagPageDrainedV1 = 26u;
constexpr std::int32_t kEvtPageDrain = 5;
static_assert(kFlagPageDrainedV0 < 11u && kFlagPageDrainedV1 - 16u < 11u,
              "page-completion flags must remain in the application ID range");

// kFlagL1Free (AIC -> AIV1) is gone with the L1 ring it used to order; see
// the kL1End comment. The legacy entry has no AIC -> AIV1 channel; the
// dynamic entry uses only the full-page completion channel above.

static_assert(kFlagValueReady<kStageDepth - 1u> < 11u &&
              kFlagScoreFree<kStageDepth - 1u> < 11u &&
              kFlagScoreReady<kStageDepth - 1u> < 11u,
              "sub block 0 flag ids must stay below the library-reserved 11..14");
static_assert(kFlagKeyReady<kStageDepth - 1u> - 16u < 11u,
              "sub block 1 flag ids must stay below 11 in their own half");

// Intra-core HardEvent ids. 6 and 7 are reserved, so ring slots take 0..1 and
// the prologue and epilogue take 3.
constexpr std::int32_t kEvtStatic = 3;

// --------------------------------------------- compile-time column schedule
template <std::uint32_t Column>
constexpr std::uint32_t kSlotOf = Column % kStageDepth;
using ColumnSeq = std::make_integer_sequence<std::uint32_t, kColumns>;

constexpr auto kUb = AscendC::TPosition::VECCALC;

// ============================================================ AIV1: key path
// MTE2 ingress of the packed key strip, Vector dequantisation to bf16, and the
// direct UB -> L1 staging burst. Touches Global Memory exactly once per strip,
// to read the 2048-byte packed page.
struct KeyEngine {
    AscendC::GlobalTensor<std::int8_t> key_cache;
    std::uint32_t strip_base_bytes = 0;  // page and head offset, resolved once
};

template <std::uint32_t Slot, std::uint32_t Strip>
__aicore__ inline void StageKeyStrip(KeyEngine& engine, std::uint32_t strip)
{
    static_assert(Slot < kStageDepth && Strip < kStrips);
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);

    ascend_micro::NativeDequant4Engine<bfloat16_t> dq;
    dq.packed = {kUb, kPackedKSlot<Slot>, kPackedBytes};
    dq.output = {kUb, kKeySlot<Slot>, kPanel};
    // No row scale, and so no bf16 scalar anywhere in this kernel: the key's
    // per-token scale is applied to the 16 score columns on AIV0, which keeps
    // this operand an exact small integer for the Cube.

    ascend_micro::NativeCvTransferEngine<bfloat16_t> cv;
    cv.ub_position = kUb;
    cv.l1_position = AscendC::TPosition::B1;
    cv.ub_base_bytes = kKeySlot<Slot>;
    cv.l1_base_bytes = kL1KeyPage<Strip>;

    // Two back edges serialise this slot body against its own incarnation
    // kStageDepth strips earlier, and they gate two different queues:
    //
    //   V_MTE2  the dequant of strip - kStageDepth has read the packed page,
    //           so MTE2 may refill it. WaitFlag<V_MTE2> stalls the MTE2 queue,
    //           which is the queue that must not run ahead: nothing else gates
    //           it, because the Vector queue is throttled by the L1-free flag.
    //   MTE3_V  the CV burst of strip - kStageDepth has read the bf16 page, so
    //           the dequant may overwrite it. WaitFlag<MTE3_V> stalls V.
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt);
    AscendC::DataCopy(dq.packed, engine.key_cache[engine.strip_base_bytes +
                                                  strip * kPackedBytes],
                      kPackedBytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evt);
    ascend_micro::turboquant_dequant4_signed(dq, kPanel);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evt);

    // No L1 back edge: this strip owns L1 page Strip for the whole launch, so
    // there is nothing to wait for. AIV1 still cannot run away -- MTE3_V above
    // holds its Vector queue behind the CV burst of the strip kStageDepth
    // back, which is what bounds the two UB rings.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evt);
    // Nz(16, 256) for bf16: C0 is 16 elements, so a row is 16 column strips,
    // adjacent strips are 16 DataBlocks apart and rows inside a strip are 1.
    ascend_micro::cv_transfer_nd2nz<kTokens, kHeadSize, kHeadSize, 16u, 1u>(cv);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evt);
    // Drain MTE3 into the scalar pipe before announcing the L1 page: the
    // cross-core set is a scalar instruction and would otherwise overtake the
    // staging burst it is reporting. This is the one edge left that the Cube
    // depends on for data, and with AIV1 no longer throttled it is also the
    // one whose id now carries four unconsumed announcements at a time; trace
    // check 3 is what verifies the page was written before the Cube read it.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evt);
    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(kFlagKeyReady<Slot>);
}

template <std::uint32_t Column>
__aicore__ inline void StepColumnKey(KeyEngine& engine, std::uint32_t strips)
{
    // Column j stages strip j - 1; strip j's MTE2 ingress is inside that same
    // body, one column early, which is what the schedule's two MTE2 rows show.
    if constexpr (Column >= 1u && Column - 1u < kStrips) {
        if (Column - 1u < strips)
            StageKeyStrip<kSlotOf<Column - 1u>, Column - 1u>(engine, Column - 1u);
    }
}

template <std::uint32_t... Columns>
__aicore__ inline void StepColumnsKey(KeyEngine& engine, std::uint32_t strips,
                                      std::integer_sequence<std::uint32_t, Columns...>)
{
    (StepColumnKey<Columns>(engine, strips), ...);
}

__aicore__ inline void run_key_core(__gm__ std::int8_t* gm_key_cache,
                                    std::uint32_t strip_base_bytes,
                                    std::uint32_t strips)
{
    KeyEngine engine;
    engine.key_cache.SetGlobalBuffer(gm_key_cache);
    engine.strip_base_bytes = strip_base_bytes;

    // Feedback cycle priming: every ring slot starts drained, on both the MTE2
    // refill edge and the dequant overwrite edge.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);

    StepColumnsKey(engine, strips, ColumnSeq{});

    // Drain every primed token and every token the Cube handed back, so no
    // event bit and no intra-block flag leaks into the next launch. These
    // waits also retire the in-flight MTE3 bursts before the kernel returns.
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
}

// ====================================================== AIV0: value path
struct ValueEngine {
    AscendC::GlobalTensor<std::int8_t> value_cache;
    AscendC::GlobalTensor<float> scale_cache;
    AscendC::GlobalTensor<float> gate;
    std::uint32_t strip_base_bytes = 0;
    std::uint32_t key_scale_base = 0;    // elements into plane 0
    std::uint32_t value_scale_base = 0;  // elements into plane 1
    std::uint32_t gate_base = 0;
};

// MTE2 only: the packed value strip and the two scale slices.
template <std::uint32_t Slot>
__aicore__ inline void LoadValueStrip(ValueEngine& engine, std::uint32_t strip)
{
    static_assert(Slot < kStageDepth);
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);

    AscendC::LocalTensor<std::int8_t> packed{kUb, kPackedVSlot<Slot>, kPackedBytes};
    AscendC::LocalTensor<float> key_scale{kUb, kScaleSlot<Slot>, kTokens};
    AscendC::LocalTensor<float> value_scale{kUb, kScaleSlot<Slot> + kTokens * kF32,
                                            kTokens};

    // The dequant of strip - kStageDepth has read the packed page.
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt);
    AscendC::DataCopy(packed, engine.value_cache[engine.strip_base_bytes +
                                                 strip * kPackedBytes],
                      kPackedBytes);
    AscendC::DataCopy(key_scale,
                      engine.scale_cache[engine.key_scale_base + strip * kTokens],
                      kTokens);
    AscendC::DataCopy(value_scale,
                      engine.scale_cache[engine.value_scale_base + strip * kTokens],
                      kTokens);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evt);
}

// Vector only: nibble -> FP32, straight into the value ring slot. No scale is
// applied; the value's per-token scale folds into the probabilities.
template <std::uint32_t Slot>
__aicore__ inline void DequantValueStrip(ValueEngine&)
{
    static_assert(Slot < kStageDepth);
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);

    ascend_micro::NativeDequant4Engine<float> dq;
    dq.packed = {kUb, kPackedVSlot<Slot>, kPackedBytes};
    dq.signed_nibbles = {kUb, kUbDqSigned, kPanel};
    dq.output = {kUb, kValueSlot<Slot>, kPanel};
    // As on the key path: the value's per-token scale folds into the 16
    // probabilities, so the page is just the nibble value in FP32.

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evt);
    ascend_micro::turboquant_dequant4_signed(dq, kPanel);
    // NO SetFlag<V_MTE2> here, deliberately. LoadValueStrip lands THREE things
    // in this slot -- the packed value page, the key scale slice and the value
    // scale slice -- and the dequant is only the first one's last reader. The
    // two scale slices are read by the recurrence, which runs a column later,
    // so the release belongs at the end of RecurStrip and is set there.
    //
    // Releasing it here cost the 4-strip rung: strip s + kStageDepth's MTE2
    // refilled the slot while the recurrence of strip s still had to read its
    // scales, and because the two are on different queues the program order
    // inside the column ordered nothing. m and l were wrong by 2097152 ULP
    // and acc by 4010429 while every raw score panel stayed bit-exact -- the
    // signature of a corrupted key scale, since m and l have no other input.
    // The 2-strip rung passed because no slot is reused at ring depth.
    //
    // The quorum flag: the Cube multiplies a strip only once both Vector cores
    // have finished their half of the staging, so neither can be outrun. The
    // value page is not an operand of the GEMM, so this edge bounds run-ahead
    // rather than carrying data.
    AscendC::SetFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagValueReady<Slot>);
}

// The recurrence over one strip: all 16 queries, then the archive copy and the
// two release edges.
template <std::uint32_t Slot, std::uint32_t Strip>
__aicore__ inline void RecurStrip(ValueEngine&)
{
    static_assert(Slot < kStageDepth && Strip < kStrips);
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);

    ascend_micro::NativeDecodeRecurrenceEngine es;
    es.position = kUb;
    es.panel_base = kScoreSlot<Slot>;
    es.m_base = kUbMPrev;
    es.l_base = kUbLPrev;
    es.acc_base = kUbAcc;
    es.panel = {kUb, kScoreSlot<Slot>, kScores};
    es.key_scale = {kUb, kScaleSlot<Slot>, kTokens};
    es.value_scale = {kUb, kScaleSlot<Slot> + kTokens * kF32, kTokens};
    es.value_page = {kUb, kValueSlot<Slot>, kPanel};
    es.m_prev = {kUb, kUbMPrev, kTokens * kStateLanes};
    es.l_prev = {kUb, kUbLPrev, kTokens * kStateLanes};
    es.acc = {kUb, kUbAcc, kTokens * kHeadSize};
    es.scaled_row = {kUb, kUbScaledRow, kTokens};
    es.centered = {kUb, kUbCentered, kTokens};
    es.p = {kUb, kUbP, kTokens};
    es.p_scaled = {kUb, kUbPScaled, kTokens};
    es.p_x4 = {kUb, kUbPx4, kBcLanes};
    es.m_new = {kUb, kUbMNew, kStateLanes};
    es.l_new = {kUb, kUbLNew, kStateLanes};
    es.alpha = {kUb, kUbAlpha, kStateLanes};
    es.alpha_delta = {kUb, kUbAlphaDelta, kStateLanes};
    es.l_scaled = {kUb, kUbLScaled, kStateLanes};
    es.tile_max_raw = {kUb, kUbTileMaxRaw, kStateLanes};
    es.p_sum_raw = {kUb, kUbPSumRaw, kStateLanes};
    es.tile_max_bc = {kUb, kUbTileMaxBc, kBcLanes};
    es.m_bc = {kUb, kUbMBc, kBcLanes};
    es.p_sum_bc = {kUb, kUbPSumBc, kBcLanes};
    es.alpha_bc = {kUb, kUbAlphaBc, kBcLanes};
    es.weighted = {kUb, kUbWeighted, kPanel};
    es.weighted_sum = {kUb, kUbWeightedSum, kHeadSize};
    es.acc_scaled = {kUb, kUbAccScaled, kHeadSize};

    AscendC::LocalTensor<float> panel{kUb, kScoreSlot<Slot>, kScores};
    AscendC::LocalTensor<float> archived{kUb, kArchiveSlot<Strip>, kScores};

    AscendC::CrossCoreWaitFlag<4, PIPE_V>(kFlagScoreReady<Slot>);
    ascend_micro::vector_decode_recur_strip(es);
    AscendC::PipeBarrier<PIPE_V>();
    // The RAW panel, before the key scale, so the runner can check what the
    // Cube produced rather than what the recurrence derived from it. Muls by
    // 1.0f is an exact FP32 copy and stays on the Vector pipe, so the
    // cross-core set below, issued on PIPE_V, is ordered after it.
    AscendC::Muls(archived, panel, 1.0f, static_cast<std::int32_t>(kScores));
    AscendC::PipeBarrier<PIPE_V>();
    // The recurrence is the last reader of everything LoadValueStrip put in
    // this slot, so the slot's MTE2 refill edge is released here and not in
    // DequantValueStrip. Set on the Vector queue after the barrier above, so
    // every read of the scales and of the value page has retired.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evt);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagScoreFree<Slot>);
}

template <std::uint32_t Column>
__aicore__ inline void StepColumnValue(ValueEngine& engine, std::uint32_t strips)
{
    // The recurrence is emitted FIRST, before this column's MTE2 ingress, and
    // the order is load-bearing rather than cosmetic.
    //
    // RecurStrip(C - kPipelineLag) sets V_MTE2 for its slot and
    // LoadValueStrip(C - 2) waits that same slot's V_MTE2 -- the two differ by
    // kStageDepth strips, so they are the same ring slot and the same event
    // id. An intra-core SetFlag/WaitFlag pair is not a semaphore that waits
    // for a set still to come: the scalar stream drops the set into the
    // producing pipe's queue and the wait into the consuming pipe's, so the
    // SET has to be reached first in program order.
    //
    // With the load emitted first the model said, on AIV0 and twice per run,
    //   has no matched set_flag, pipe:1, trigger_pipe:4, flag_id:0
    //   execute_set_flag already has same set_flag
    // and let the wait through -- the wait that is the entire point of the
    // section 4.5 fix. The numbers came out correct anyway on that run, which
    // is exactly why this is checked by the trace audit (check 1b) and not by
    // the output comparison.
    //
    // Issuing the recurrence first costs nothing: the scalar stream does not
    // wait for the Vector queue to drain, so this column's DataCopy is issued
    // a few scalar instructions later and executes on an MTE2 queue that is
    // idle either way.
    if constexpr (Column >= kPipelineLag && Column - kPipelineLag < kStrips) {
        if (Column - kPipelineLag < strips)
            RecurStrip<kSlotOf<Column - kPipelineLag>, Column - kPipelineLag>(engine);
    }
    if constexpr (Column >= 3u && Column - 3u < kStrips) {
        if (Column - 3u < strips) DequantValueStrip<kSlotOf<Column - 3u>>(engine);
    }
    if constexpr (Column >= 2u && Column - 2u < kStrips) {
        if (Column - 2u < strips) LoadValueStrip<kSlotOf<Column - 2u>>(engine, Column - 2u);
    }
}

template <std::uint32_t... Columns>
__aicore__ inline void StepColumnsValue(ValueEngine& engine, std::uint32_t strips,
                                        std::integer_sequence<std::uint32_t, Columns...>)
{
    (StepColumnValue<Columns>(engine, strips), ...);
}

// ---------------------------------------------------------------- the egress
// One query at a time: refill the gate row, rotate, gate, cast. The gate row
// is single buffered -- the egress runs once, after the strip sequence, so
// there is nothing left to overlap it with and a resident [16 x 256] gate
// would cost 16 KiB to save nothing.
//
// Single buffered means TWO edges, not one
// ----------------------------------------
// MTE2_V alone says "this query's gate row has arrived". It says nothing about
// when the row may be refilled, and the MTE2 queue runs far ahead of the
// Vector queue -- it is the queue with nothing else gating it. Without the
// V_MTE2 edge below, MTE2 overwrites the row while the Vector pipe is still
// gating the previous query with it.
//
// That is not a precaution. The first revision had only the MTE2_V edge and
// the canary run measured the consequence exactly: the factor the device
// applied to query 0 was sigmoid(gate[query 2]) to within bf16 rounding, so
// MTE2 was running exactly two iterations ahead of the consumer. The rotation
// in the same run was correct to 8.9e-08, which is what made the gate the only
// remaining suspect and the factor worth measuring.
//
// This is the same hazard class, on the same queue, as the missing packed-page
// WAR edge that MICROROUTINE_DESIGN_GUIDELINES.md section 0 records for the
// fused attention operator. It is primed once, carried once per query, and
// drained once.
__aicore__ inline void EgressQueries(ascend_micro::NativeEgressRotateEngine& eg,
                                     ValueEngine& engine)
{
    AscendC::LocalTensor<float> gate_row{kUb, kUbGateRow, kHeadSize};
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtStatic);  // row starts free
    for (std::uint32_t query = 0; query < kTokens; ++query) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtStatic);
        AscendC::DataCopy(gate_row,
                          engine.gate[engine.gate_base + query * kHeadSize], kHeadSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtStatic);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtStatic);
        eg.select_query(query);
        ascend_micro::vector_egress_query(eg);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtStatic);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtStatic);
}

__aicore__ inline void run_value_core(__gm__ std::int8_t* gm_value_cache,
                                      __gm__ float* gm_scale_cache,
                                      __gm__ float* gm_gate,
                                      __gm__ std::uint32_t* gm_gather_offsets,
                                      __gm__ float* gm_gather_signs,
                                      __gm__ float* gm_pi_sign,
                                      __gm__ float* gm_out_scores,
                                      __gm__ float* gm_out_state,
                                      __gm__ float* gm_out_acc,
                                      __gm__ bfloat16_t* gm_out_context,
                                      __gm__ float* gm_out_rotated,
                                      std::uint32_t strip_base_bytes,
                                      std::uint32_t key_scale_base,
                                      std::uint32_t value_scale_base,
                                      std::uint32_t gate_base,
                                      std::uint32_t strips, bool partial = false,
                                      bool initialize = true, bool finalize = true)
{
    ValueEngine engine;
    engine.value_cache.SetGlobalBuffer(gm_value_cache);
    engine.scale_cache.SetGlobalBuffer(gm_scale_cache);
    engine.gate.SetGlobalBuffer(gm_gate);
    engine.strip_base_bytes = strip_base_bytes;
    engine.key_scale_base = key_scale_base;
    engine.value_scale_base = value_scale_base;
    engine.gate_base = gate_base;

    AscendC::GlobalTensor<float> out_scores, out_state, out_acc, out_rotated;
    AscendC::GlobalTensor<bfloat16_t> out_context;
    out_scores.SetGlobalBuffer(gm_out_scores);
    out_state.SetGlobalBuffer(gm_out_state);
    out_acc.SetGlobalBuffer(gm_out_acc);
    out_context.SetGlobalBuffer(gm_out_context);
    out_rotated.SetGlobalBuffer(gm_out_rotated);

    // The egress tables, read once. They are strip invariant, so they are on
    // the prologue's MTE2 and not in any column.
    AscendC::LocalTensor<std::uint32_t> gather_offsets{
        kUb, kUbGatherOffsets, kGatherStages * kHeadSize};
    AscendC::LocalTensor<float> gather_signs{kUb, kUbGatherSigns,
                                             kGatherStages * kHeadSize};
    AscendC::LocalTensor<float> pi_sign{kUb, kUbPiSign, kHeadSize};
    AscendC::GlobalTensor<std::uint32_t> gm_offsets;
    AscendC::GlobalTensor<float> gm_signs, gm_signs_pi;
    gm_offsets.SetGlobalBuffer(gm_gather_offsets);
    gm_signs.SetGlobalBuffer(gm_gather_signs);
    gm_signs_pi.SetGlobalBuffer(gm_pi_sign);
    if (!partial) {
        AscendC::DataCopy(gather_offsets, gm_offsets, kGatherStages * kHeadSize);
        AscendC::DataCopy(gather_signs, gm_signs, kGatherStages * kHeadSize);
        AscendC::DataCopy(pi_sign, gm_signs_pi, kHeadSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtStatic);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtStatic);
    }

    // Seed the replicated-state invariant from immediate scalars, which never
    // read UB and so carry none of the tensor-source hazards. m starts at a
    // large negative rather than -inf: the first strip's Max selects the tile
    // maximum either way, and exp(m_prev - m_new) is then 0 rather than a
    // NaN-producing inf - inf.
    AscendC::LocalTensor<float> m_prev{kUb, kUbMPrev, kTokens * kStateLanes};
    AscendC::LocalTensor<float> l_prev{kUb, kUbLPrev, kTokens * kStateLanes};
    AscendC::LocalTensor<float> acc{kUb, kUbAcc, kTokens * kHeadSize};
    if (initialize) {
        AscendC::Duplicate(m_prev, -3.0e38f, static_cast<std::int32_t>(kTokens * kStateLanes));
        AscendC::Duplicate(l_prev, 0.0f, static_cast<std::int32_t>(kTokens * kStateLanes));
        AscendC::Duplicate(acc, 0.0f, static_cast<std::int32_t>(kTokens * kHeadSize));
    }

    if (strips != 0u) {
        // Feedback cycle priming: every packed value slot is free, and every score
        // window in the Fixpipe ring is free for the Cube to write.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagScoreFree<0>);
        AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagScoreFree<1>);

        StepColumnsValue(engine, strips, ColumnSeq{});

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
    }
    if (partial) {
        if (finalize) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStatic);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStatic);
            AscendC::DataCopy(out_state, m_prev, kTokens * kStateLanes);
            AscendC::DataCopy(out_state[kTokens * kStateLanes], l_prev, kTokens * kStateLanes);
            AscendC::DataCopy(out_acc, acc, kPanel);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtStatic);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtStatic);
        }
        return;
    }

    // ---- the egress, once, after the whole strip sequence ---------------
    ascend_micro::NativeEgressRotateEngine eg;
    eg.position = kUb;
    eg.acc_base = kUbAcc;
    eg.l_base = kUbLPrev;
    eg.context_base = kUbContext;
    eg.rotated_base = kUbRotated;
    eg.acc = acc;
    eg.l_prev = l_prev;
    eg.gate = {kUb, kUbGateRow, kHeadSize};
    eg.pi_sign = pi_sign;
    eg.gather_offsets = gather_offsets;
    eg.gather_signs = gather_signs;
    eg.l_broadcast = {kUb, kUbLBroadcast, kHeadSize};
    eg.buffer_a = {kUb, kUbEgressA, kHeadSize};
    eg.buffer_b = {kUb, kUbEgressB, kHeadSize};
    eg.permuted = {kUb, kUbPermuted, kHeadSize};
    eg.signed_tmp = {kUb, kUbSignedTmp, kHeadSize};
    eg.denominator = {kUb, kUbDenominator, kHeadSize};
    eg.context = {kUb, kUbContext, kTokens * kHeadSize};
    eg.rotated = {kUb, kUbRotated, kTokens * kHeadSize};

    EgressQueries(eg, engine);

    // ---- the single outflow --------------------------------------------
    // Nothing above this point stores to Global Memory. The archive is read
    // back as one strided burst, kStrips panels of kScoreBytes separated by the
    // archive bank skew, so every entry of every panel is checkable without
    // the kernel ever dumping a panel per strip.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStatic);

    AscendC::LocalTensor<float> score_archive{kUb, kUbScoreArchive, kStrips * kScores};
    AscendC::DataCopyParams panels;
    // Exactly the panels this launch filled, not all kStrips of them. The
    // archive is not initialised, so copying the unfilled slots would publish
    // whatever those Unified Buffer bytes happened to hold -- and the canary
    // protocol's temporal-isolation check is precisely that the archive words
    // beyond the strip window come back untouched.
    panels.blockCount = static_cast<std::uint16_t>(strips);
    panels.blockLen = static_cast<std::uint16_t>(kScoreBytes / kBlock);
    panels.srcStride = static_cast<std::uint16_t>((kRingScore - kScoreBytes) / kBlock);
    panels.dstStride = 0;
    AscendC::DataCopy(out_scores, score_archive, panels);

    AscendC::LocalTensor<bfloat16_t> context{kUb, kUbContext, kTokens * kHeadSize};
    AscendC::DataCopy(out_context, context, kTokens * kHeadSize);
    AscendC::DataCopy(out_acc, acc, kTokens * kHeadSize);
    AscendC::LocalTensor<float> rotated{kUb, kUbRotated, kTokens * kHeadSize};
    AscendC::DataCopy(out_rotated, rotated, kTokens * kHeadSize);
    AscendC::DataCopy(out_state[0], m_prev, kTokens * kStateLanes);
    AscendC::DataCopy(out_state[kTokens * kStateLanes], l_prev, kTokens * kStateLanes);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtStatic);
}

// ============================================================== AIC: Cube
struct CubeEngine {
    __ca__ bfloat16_t* l0a;
    __cb__ bfloat16_t* l0b;
    __cc__ float* l0c;
};

template <std::uint32_t Slot, std::uint32_t Strip>
__aicore__ inline void GemmStrip(CubeEngine& engine)
{
    static_assert(Slot < kStageDepth && Strip < kStrips);
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);
    auto* l0b_slot = engine.l0b + Slot * kHeadSize * kTokens;
    auto* l0c_window = engine.l0c + Slot * kScores;
    auto* ub_scores = reinterpret_cast<__ubuf__ float*>(kScoreSlot<Slot>);

    // Double-handshake quorum: one flag from each Vector core, so a strip is
    // multiplied only once AIV1 has staged its key fractal into L1 page Strip
    // and AIV0 has dequantised the matching value page.
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(kFlagKeyReady<Slot>);
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(kFlagValueReady<Slot>);

    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(evt);  // L0B slot free
    ascend_micro::score_strip_stage_l0(kL1KeyPage<Strip>, l0b_slot);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(evt);

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(evt);  // L0C window free
    // Eight accumulating Mmads of k = 32 over the 256-channel contraction.
    ascend_micro::score_strip_compute(engine.l0a, l0b_slot, l0c_window);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(evt);  // L0B slot free
    // The Mmads feed one Fixpipe, so their L0C dependency is an explicit
    // M_FIX token rather than a unitFlag pairing. Set and waited inside this
    // body, so it needs no priming and leaks nothing.
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(evt);
    // The score window is waited here rather than in front of the multiplies:
    // only the Fixpipe touches Unified Buffer, so the eight Mmads for this
    // strip overlap the recurrence of the strip two back instead of queueing
    // behind it. The recurrence must have consumed the previous occupant of
    // this window before the Fixpipe overwrites it.
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagScoreFree<Slot>);
    ascend_micro::score_strip_fixpipe(l0c_window, ub_scores, false);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(evt);
    // One drain for both outgoing flags. A retired Fixpipe proves the Mmad
    // retired, and the Mmad waited MTE1_M, so it also proves the L1 -> L0 load
    // has read the L1 key slot and the staging core may refill it. There is no
    // MTE1_S route to express that more tightly.
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(evt);
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagScoreReady<Slot>);
}

template <std::uint32_t Column>
__aicore__ inline void StepColumnCube(CubeEngine& engine, std::uint32_t strips)
{
    if constexpr (Column >= 2u && Column - 2u < kStrips) {
        if (Column - 2u < strips) GemmStrip<kSlotOf<Column - 2u>, Column - 2u>(engine);
    }
}

template <std::uint32_t... Columns>
__aicore__ inline void StepColumnsCube(CubeEngine& engine, std::uint32_t strips,
                                       std::integer_sequence<std::uint32_t, Columns...>)
{
    (StepColumnCube<Columns>(engine, strips), ...);
}

__aicore__ inline void run_cube_core(__gm__ bfloat16_t* gm_q,
                                     __gm__ bfloat16_t* gm_pi_table,
                                     __gm__ bfloat16_t* gm_out_qrot,
                                     std::uint32_t q_base,
                                     std::uint32_t strips, std::uint32_t pages = 1u,
                                     bool temporal = false)
{
    if (pages == 0u) return;
    using namespace AscendC::Te;
    // L0A holds the eight Q group fractals and then Q_rot, both 4096 bf16
    // elements. L0B holds the 16 resident rotation blocks and then the two
    // key slots. L0C holds the 16 x 256 rotated panel and then the two score
    // windows. Each later use fits inside the earlier allocation.
    __ca__ bfloat16_t l0_a[kPanel] __attribute__((aligned(512)));
    __cb__ bfloat16_t l0_b[kTableElems] __attribute__((aligned(512)));
    __cc__ float l0_c[kPanel] __attribute__((aligned(1024)));
    static_assert(kStageDepth * kHeadSize * kTokens <= kTableElems,
                  "the key slots must fit inside the L0B table allocation");
    static_assert(kStageDepth * kScores <= kPanel,
                  "the score windows must fit inside the L0C panel allocation");

    CubeEngine engine{l0_a, l0_b, l0_c};

    // ---- prologue: the ingress rotation --------------------------------
    // Both operands reach L1 as Nz images. The page conversion is a no-op for
    // the 16 x 256 panel and a real rearrangement for the 32 x 32 blocks; see
    // routine_stage_page_zn_bf16.hpp.
    ascend_micro::stage_page_nz<kTokens, kHeadSize>(gm_q + q_base, kL1Q, kHeadSize);
    ascend_micro::stage_blocks_nz<kTableBlocks, kGroup>(gm_pi_table, kL1PiTable);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtStatic);

    ascend_micro::rotate_query_stage_l0(kL1PiTable, kL1Q, l0_a, l0_b);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtStatic);

    ascend_micro::rotate_query_compute(l0_a, l0_b, l0_c);
    // 64 accumulating Mmads feed the transfer below, so the L0C dependency is
    // an explicit M_FIX token rather than a unitFlag pairing.
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEvtStatic);

    // Q_rot out to Global Memory as bf16 ND: both the runner's bit-exact check
    // on the ingress rotation and the A operand's own source, because the
    // route back in is the same GM -> L1 ND -> Nz ingress the query uses.
    ascend_micro::rotate_query_to_gm(gm_out_qrot, l0_c);
    // There is no FIX_MTE2 route. Draining FIX into the scalar stream stalls
    // instruction issue, so the MTE2 read below is issued only after the
    // Fixpipe has retired and its Global Memory write has committed.
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtStatic);

    ascend_micro::stage_page_nz<kTokens, kHeadSize>(gm_out_qrot, kL1QRot, kHeadSize);
    // The Mmads must have retired before L0A is overwritten with Q_rot.
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEvtStatic);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtStatic);

    ascend_micro::load_query_rotated(kL1QRot, l0_a);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtStatic);

    for (std::uint32_t page = 0; page < pages; ++page) {
        // ---- feedback cycle priming ----------------------------------------
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);

        StepColumnsCube(engine, strips, ColumnSeq{});

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
        AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagScoreFree<0>);
        AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagScoreFree<1>);
        if (temporal) {
            // The final score-free waits prove AIV0 consumed every score.
            // Drain FIX before announcing that all eight L1 key pages and
            // this page's mode-4 announcements can be reused.
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtPageDrain);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtPageDrain);
            AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagPageDrainedV0);  // AIV0 page drained
            AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagPageDrainedV1);  // AIV1 page drained
        }
    }

}

}  // namespace paged_decode

// `active_block` selects one grid block, or kGrid for all of them; `strips` and
// `strip_begin` select a window of a page's eight strips. Together they are the
// single-tile canary protocol's isolation knob: one block, one strip, every
// other Global Memory region left as the host filled it.
//
// The gate is identical on all three cores of a block, because it is a function
// of the block index alone. A gate that differed between the AIC and the AIVs
// of one block would deadlock the launch.
extern "C" __global__ __aicore__ void camodel_paged_decode_attention(
    __gm__ bfloat16_t* q, __gm__ bfloat16_t* pi_table, __gm__ std::int8_t* key_cache,
    __gm__ std::int8_t* value_cache, __gm__ float* scale_cache,
    __gm__ std::uint32_t* block_table, __gm__ float* gate,
    __gm__ std::uint32_t* gather_offsets, __gm__ float* gather_signs,
    __gm__ float* pi_sign, __gm__ bfloat16_t* out_qrot, __gm__ float* out_scores,
    __gm__ float* out_state, __gm__ float* out_acc, __gm__ bfloat16_t* out_context,
    __gm__ float* out_rotated, uint32_t strips, uint32_t strip_begin,
    uint32_t active_block)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using namespace paged_decode;

    if (strips == 0u || strips > kStrips || strip_begin + strips > kStrips) return;

    // GetBlockIdx() ranges over [0, numBlocks) on the AIC and [0, 2*numBlocks)
    // on an AIV at a 1:2 ratio, and GetSubBlockNum() is 1 on the AIC and 2 on
    // an AIV, so this quotient is the same block index on all three cores.
    const std::uint32_t block = static_cast<std::uint32_t>(
        AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());
    if (block >= kGrid) return;
    if (active_block < kGrid && block != active_block) return;

    const std::uint32_t kv_head = block / kTemporalBlocks;
    const std::uint32_t temporal = block % kTemporalBlocks;

    AscendC::GlobalTensor<std::uint32_t> table;
    table.SetGlobalBuffer(block_table);
    const std::uint32_t page = table.GetValue(temporal);

    // Byte offset of (page, head) strip 0 inside the packed caches, and element
    // offsets of the two scale planes and of this head's query and gate.
    const std::uint32_t strip_base_bytes =
        ((page * kKvHeads) + kv_head) * kStrips * kPackedBytes;
    const std::uint32_t scale_plane = kNumPages * kKvHeads * kBlockSize;
    const std::uint32_t scale_base = ((page * kKvHeads) + kv_head) * kBlockSize +
                                     strip_begin * kTokens;
    const std::uint32_t q_base = kv_head * kPanel;
    const std::uint32_t out_base = block * kPanel;

    if ASCEND_IS_AIC {
        run_cube_core(q, pi_table, out_qrot + out_base, q_base, strips);
    } else if (AscendC::GetSubBlockIdx() == 0) {
        run_value_core(value_cache, scale_cache, gate, gather_offsets, gather_signs,
                       pi_sign, out_scores + block * kStrips * kScores,
                       out_state + block * 2u * kTokens * kStateLanes,
                       out_acc + out_base, out_context + out_base,
                       out_rotated + out_base,
                       strip_base_bytes + strip_begin * kPackedBytes, scale_base,
                       scale_plane + scale_base, q_base, strips);
    } else {
        run_key_core(key_cache, strip_base_bytes + strip_begin * kPackedBytes, strips);
    }
}

// Dynamic single-token schema with M=16 packed shared-KV query rows. The
// legacy entry above keeps its validation ABI and standalone egress behavior.
extern "C" __global__ __aicore__ void turboquant_paged_decode_splitk_aic(
    __gm__ bfloat16_t* q, __gm__ bfloat16_t* pi_table,
    __gm__ std::int8_t* key, __gm__ std::int8_t* value, __gm__ float* scales,
    __gm__ uint32_t* table, __gm__ bfloat16_t* qrot,
    __gm__ float* state, __gm__ float* acc, uint32_t requests, uint32_t heads,
    uint32_t block_size, uint32_t cache_pages, uint32_t seq_pages,
    uint32_t splits, uint32_t page_begin, uint32_t page_count)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using namespace paged_decode;
    if (requests == 0u || heads == 0u || splits == 0u || splits > 16u ||
        block_size == 0u || block_size > kBlockSize || block_size % kTokens != 0u ||
        seq_pages == 0u || seq_pages > cache_pages || page_begin > seq_pages ||
        page_count > seq_pages - page_begin) return;
    const auto item = static_cast<uint32_t>(AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());
    if (item >= requests * heads * splits) return;
    const uint32_t head_item = item / splits;
    const uint32_t request = head_item / heads, head = head_item % heads;
    const auto range = turboquant_splitk::Partition(page_count, splits, item % splits);
    const uint32_t strips = block_size / kTokens;
    if ASCEND_IS_AIC {
        // Q rotation is amortized over the entire page range, not repeated
        // per page. The modulo-8 strip pipeline is primed/drained per page.
        run_cube_core(q, pi_table, qrot + item * kPanel, head_item * kPanel,
                      strips, range.count, true);
    } else if (AscendC::GetSubBlockIdx() == 0) {
        // Empty temporal workers publish a neutral partial without any
        // mode-4 flags, query rotation, or normalization of a zero sum.
        const uint32_t iterations = range.count == 0u ? 1u : range.count;
        for (uint32_t p = 0; p < iterations; ++p) {
            const uint32_t physical = range.count == 0u ? 0u :
                table[request * seq_pages + page_begin + range.begin + p];
            const uint32_t base = (physical * heads + head) * block_size;
            run_value_core(value, scales, nullptr, nullptr, nullptr, nullptr,
                nullptr, state + item * 2u * kTokens * kStateLanes,
                acc + item * kPanel, nullptr, nullptr, base * kHeadSize / 2u,
                base, cache_pages * heads * block_size + base, 0u,
                range.count == 0u ? 0u : strips, true, p == 0u, p + 1u == iterations);
            if (range.count != 0u) {
                AscendC::CrossCoreWaitFlag<4, PIPE_V>(kFlagPageDrainedV0);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtPageDrain);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtPageDrain);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(kEvtPageDrain);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(kEvtPageDrain);
            }
        }
    } else {
        for (uint32_t p = 0; p < range.count; ++p) {
            const uint32_t physical = table[request * seq_pages + page_begin + range.begin + p];
            const uint32_t base = (physical * heads + head) * block_size * kHeadSize / 2u;
            run_key_core(key, base, strips);
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(kFlagPageDrainedV1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtPageDrain);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtPageDrain);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtPageDrain);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtPageDrain);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(kEvtPageDrain);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(kEvtPageDrain);
        }
    }
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

#include <cassert>
#include <stdexcept>

extern "C" __global__ __aicore__ void camodel_paged_decode_attention(
    __gm__ bfloat16_t*, __gm__ bfloat16_t*, __gm__ std::int8_t*, __gm__ std::int8_t*,
    __gm__ float*, __gm__ std::uint32_t*, __gm__ float*, __gm__ std::uint32_t*,
    __gm__ float*, __gm__ float*, __gm__ bfloat16_t*, __gm__ float*, __gm__ float*,
    __gm__ float*, __gm__ bfloat16_t*, __gm__ float*, uint32_t, uint32_t, uint32_t);

extern "C" void launch_camodel_paged_decode_attention(
    void* stream, std::uint16_t* q, std::uint16_t* pi_table, std::int8_t* key_cache,
    std::int8_t* value_cache, float* scale_cache, std::uint32_t* block_table,
    float* gate, std::uint32_t* gather_offsets, float* gather_signs, float* pi_sign,
    std::uint16_t* out_qrot, float* out_scores, float* out_state, float* out_acc,
    std::uint16_t* out_context, float* out_rotated, std::uint32_t strips,
    std::uint32_t strip_begin, std::uint32_t active_block)
{
    assert(strips >= 1u && strips <= paged_decode::kStrips);
    if (strips < 1u || strips > paged_decode::kStrips ||
        strip_begin + strips > paged_decode::kStrips)
        throw std::invalid_argument("paged decode strips must select a window of 0..8");
    camodel_paged_decode_attention<<<paged_decode::kGrid, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(q), reinterpret_cast<bfloat16_t*>(pi_table),
        key_cache, value_cache, scale_cache, block_table, gate, gather_offsets,
        gather_signs, pi_sign, reinterpret_cast<bfloat16_t*>(out_qrot), out_scores,
        out_state, out_acc, reinterpret_cast<bfloat16_t*>(out_context), out_rotated,
        strips, strip_begin, active_block);
}

extern "C" __global__ __aicore__ void turboquant_paged_decode_splitk_aic(
    __gm__ bfloat16_t*, __gm__ bfloat16_t*, __gm__ std::int8_t*,
    __gm__ std::int8_t*, __gm__ float*, __gm__ uint32_t*, __gm__ bfloat16_t*,
    __gm__ float*, __gm__ float*, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t);

extern "C" void launch_turboquant_paged_decode_splitk_aic(
    void* stream, std::uint16_t* q, std::uint16_t* pi_table, std::int8_t* key,
    std::int8_t* value, float* scales, std::uint32_t* table,
    std::uint16_t* qrot, float* state, float* acc, std::uint32_t requests,
    std::uint32_t heads, std::uint32_t block_size, std::uint32_t cache_pages,
    std::uint32_t seq_pages, std::uint32_t splits, std::uint32_t page_begin,
    std::uint32_t page_count)
{
    if (!requests || !heads || !splits || splits > 16 || !seq_pages ||
        !block_size || block_size > paged_decode::kBlockSize || block_size % 16 ||
        seq_pages > cache_pages || page_begin > seq_pages || page_count > seq_pages - page_begin)
        throw std::invalid_argument("invalid AIC temporal split geometry");
    turboquant_paged_decode_splitk_aic<<<requests * heads * splits, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(q), reinterpret_cast<bfloat16_t*>(pi_table),
        key, value, scales, table, reinterpret_cast<bfloat16_t*>(qrot), state, acc,
        requests, heads, block_size, cache_pages, seq_pages, splits, page_begin, page_count);
}

#endif
