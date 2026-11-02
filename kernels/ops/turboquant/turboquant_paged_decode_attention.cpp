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

#include <cstdint>
#include <utility>
#include "../../../include/turboquant_splitk_contract.hpp"

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "turboquant_decode_tail.hpp"
#include "../../../routines/routine_stage_page_zn_bf16.hpp"
#include "../../../routines/routine_fractal_load_zn_bf16.hpp"
#include "../../../routines/routine_cube_decode_score.hpp"
#include "../../../routines/routine_turboquant_dequant4.hpp"
#include "../../../routines/routine_cv_transfer_ub_to_l1.hpp"
#include "../../../routines/routine_vector_decode_recurrence.hpp"
#include "../../../routines/routine_vector_egress_rotate.hpp"
#endif

namespace paged_decode {

constexpr std::uint32_t kTokens         = 16;
constexpr std::uint32_t kHeadSize       = 256;
constexpr std::uint32_t kGroup          = 32;
constexpr std::uint32_t kGroups         = kHeadSize / kGroup;
constexpr std::uint32_t kStrips         = 8;
constexpr std::uint32_t kBlockSize      = kStrips * kTokens;
constexpr std::uint32_t kKvHeads        = 4;
constexpr std::uint32_t kTemporalBlocks = 8;
constexpr std::uint32_t kGrid           = kKvHeads * kTemporalBlocks;
constexpr std::uint32_t kPanel          = kTokens * kHeadSize;
constexpr std::uint32_t kPackedBytes    = kPanel / 2;
constexpr std::uint32_t kScores         = kTokens * kTokens;
constexpr std::uint32_t kTableBlocks    = 2u * kGroups;
constexpr std::uint32_t kTableElems     = kTableBlocks * kGroup * kGroup;
constexpr std::uint32_t kNumPages       = kTemporalBlocks + 1u;

constexpr std::uint32_t kStageDepth     = 2;
constexpr std::uint32_t kPipelineLag    = 4;
constexpr std::uint32_t kColumns        = kStrips + kPipelineLag;
constexpr std::uint32_t kGatherStages   = 3;

static_assert(kHeadSize == kGroups * kGroup,        "the groups must tile the head");
static_assert(kStrips   >= 2u * kStageDepth,        "eight strips must wrap a two-slot ring at least twice");

}  // namespace paged_decode

#if defined(__CCE_AICORE__)

#if defined(__MIX_CORE_AIC_RATION__) && (__MIX_CORE_AIC_RATION__ == 1)
#error "MIX_AIC_1_2 is required: a 1:1 AIC:AIV ratio has no second Vector core"
#endif

namespace paged_decode {

constexpr std::uint32_t kBlock          = 32u;
constexpr std::uint32_t kBankSkewBytes  = 32u;
constexpr std::uint32_t kReadPadBytes   = 256u;
constexpr std::uint32_t kUbWindow       = 192u * 1024u;

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}

constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkewBytes;
}

constexpr std::uint32_t kF32            = 4u;
constexpr std::uint32_t kB16            = 2u;
constexpr std::uint32_t kI16            = 2u;
constexpr std::uint32_t kStateLanes     = ascend_micro::kDecodeStateLanes;
constexpr std::uint32_t kBcLanes        = ascend_micro::kDecodeBcLanes;

constexpr std::uint32_t kScoreBytes     = kScores * kF32;
constexpr std::uint32_t kKeyPageBytes   = kPanel * kB16;
constexpr std::uint32_t kValuePageBytes = kPanel * kF32;
constexpr std::uint32_t kScalePairBytes = 2u * kTokens * kF32;

constexpr std::uint32_t kRingScore      = kScoreBytes     + kBankSkewBytes;
constexpr std::uint32_t kRingPacked     = kPackedBytes    + kBankSkewBytes;
constexpr std::uint32_t kRingKey        = kKeyPageBytes   + kBankSkewBytes;
constexpr std::uint32_t kRingValue      = kValuePageBytes + kBankSkewBytes;
constexpr std::uint32_t kRingScale      = kScalePairBytes + kBankSkewBytes;

constexpr std::uint32_t kUbScoreRing    = 0u;
constexpr std::uint32_t kUbPackedKRing  = NextSlot(kUbScoreRing,    kStageDepth * kRingScore);
constexpr std::uint32_t kUbPackedVRing  = NextSlot(kUbPackedKRing,  kStageDepth * kRingPacked);
constexpr std::uint32_t kUbKeyRing      = NextSlot(kUbPackedVRing,  kStageDepth * kRingPacked);
constexpr std::uint32_t kUbValueRing    = NextSlot(kUbKeyRing,      kStageDepth * kRingKey);
constexpr std::uint32_t kUbScaleRing    = NextSlot(kUbValueRing,    kStageDepth * kRingValue);
constexpr std::uint32_t kUbDqSigned     = NextSlot(kUbScaleRing,    kStageDepth * kRingScale);

constexpr std::uint32_t kUbMPrev        = NextSlot(kUbDqSigned,     kPanel * kI16);
constexpr std::uint32_t kUbLPrev        = NextSlot(kUbMPrev,        kTokens * kStateLanes * kF32);
constexpr std::uint32_t kUbAcc          = NextSlot(kUbLPrev,        kTokens * kStateLanes * kF32);

constexpr std::uint32_t kUbMNew         = NextSlot(kUbAcc,          kTokens * kHeadSize * kF32);
constexpr std::uint32_t kUbLNew         = NextSlot(kUbMNew,         kStateLanes * kF32);
constexpr std::uint32_t kUbAlpha        = NextSlot(kUbLNew,         kStateLanes * kF32);
constexpr std::uint32_t kUbAlphaDelta   = NextSlot(kUbAlpha,        kStateLanes * kF32);
constexpr std::uint32_t kUbLScaled      = NextSlot(kUbAlphaDelta,   kStateLanes * kF32);
constexpr std::uint32_t kUbTileMaxRaw   = NextSlot(kUbLScaled,      kStateLanes * kF32);
constexpr std::uint32_t kUbPSumRaw      = NextSlot(kUbTileMaxRaw,   kStateLanes * kF32);
constexpr std::uint32_t kUbTileMaxBc    = NextSlot(kUbPSumRaw,      kStateLanes * kF32);
constexpr std::uint32_t kUbMBc          = NextSlot(kUbTileMaxBc,    kBcLanes * kF32);
constexpr std::uint32_t kUbPSumBc       = NextSlot(kUbMBc,          kBcLanes * kF32);
constexpr std::uint32_t kUbAlphaBc      = NextSlot(kUbPSumBc,       kBcLanes * kF32);
constexpr std::uint32_t kUbScaledRow    = NextSlot(kUbAlphaBc,      kBcLanes * kF32);
constexpr std::uint32_t kUbCentered     = NextSlot(kUbScaledRow,    kTokens * kF32);
constexpr std::uint32_t kUbP            = NextSlot(kUbCentered,     kTokens * kF32);
constexpr std::uint32_t kUbPScaled      = NextSlot(kUbP,            kTokens * kF32);
constexpr std::uint32_t kUbPx4          = NextSlot(kUbPScaled,      kTokens * kF32);
constexpr std::uint32_t kUbWeighted     = NextSlot(kUbPx4,          kBcLanes * kF32);
constexpr std::uint32_t kUbWeightedSum  = NextSlot(kUbWeighted,     kPanel * kF32);
constexpr std::uint32_t kUbAccScaled    = NextSlot(kUbWeightedSum,  kHeadSize * kF32);

constexpr std::uint32_t kUbScoreArchive = NextSlot(kUbAccScaled,    kHeadSize * kF32);
constexpr std::uint32_t kUbLBroadcast   = NextSlot(kUbScoreArchive, kStrips * kRingScore);
constexpr std::uint32_t kUbEgressA      = NextSlot(kUbLBroadcast,   kHeadSize * kF32);
constexpr std::uint32_t kUbEgressB      = NextSlot(kUbEgressA,      kHeadSize * kF32);
constexpr std::uint32_t kUbPermuted     = NextSlot(kUbEgressB,      kHeadSize * kF32);
constexpr std::uint32_t kUbSignedTmp    = NextSlot(kUbPermuted,     kHeadSize * kF32);
constexpr std::uint32_t kUbDenominator  = NextSlot(kUbSignedTmp,    kHeadSize * kF32);
constexpr std::uint32_t kUbGateRow      = NextSlot(kUbDenominator,  kHeadSize * kF32);
constexpr std::uint32_t kUbPiSign       = NextSlot(kUbGateRow,      kHeadSize * kF32);
constexpr std::uint32_t kUbGatherOffsets= NextSlot(kUbPiSign,       kHeadSize * kF32);
constexpr std::uint32_t kUbGatherSigns  = NextSlot(kUbGatherOffsets,kGatherStages * kHeadSize * kF32);
constexpr std::uint32_t kUbContext      = NextSlot(kUbGatherSigns,  kGatherStages * kHeadSize * kF32);
constexpr std::uint32_t kUbRotated      = NextSlot(kUbContext,      kTokens * kHeadSize * kB16);

constexpr std::uint32_t kUbLaneIndex    = NextSlot(kUbRotated,      kTokens * kHeadSize * kF32);
constexpr std::uint32_t kUbMaskedFill   = NextSlot(kUbLaneIndex,    kReadPadBytes);
constexpr std::uint32_t kUbZeroFill     = NextSlot(kUbMaskedFill,   kReadPadBytes);
constexpr std::uint32_t kUbTailSel      = NextSlot(kUbZeroFill,     kReadPadBytes);
constexpr std::uint32_t kArenaEnd       = kUbTailSel + kBlock;

static_assert(kUbScoreRing     % kBlock == 0 && kUbPackedKRing   % kBlock == 0 && kUbPackedVRing   % kBlock == 0 && kUbKeyRing       % kBlock == 0 && kUbValueRing     % kBlock == 0 && kUbScaleRing     % kBlock == 0 && kUbDqSigned     % kBlock == 0 && kUbMPrev        % kBlock == 0 && kUbLPrev        % kBlock == 0 && kUbAcc          % kBlock == 0 && kUbWeighted     % kBlock == 0 && kUbWeightedSum  % kBlock == 0 && kUbAccScaled    % kBlock == 0 && kUbScoreArchive % kBlock == 0 && kUbEgressA      % kBlock == 0 && kUbEgressB      % kBlock == 0 && kUbPermuted     % kBlock == 0 && kUbSignedTmp    % kBlock == 0 && kUbDenominator  % kBlock == 0 && kUbGateRow      % kBlock == 0 && kUbPiSign       % kBlock == 0 && kUbGatherOffsets% kBlock == 0 && kUbGatherSigns  % kBlock == 0 && kUbContext      % kBlock == 0 && kUbRotated      % kBlock == 0, "every UB operand base must be 32-byte aligned");
static_assert(kRingScore       % kBlock == 0 && kRingPacked      % kBlock == 0 && kRingKey         % kBlock == 0 && kRingValue       % kBlock == 0 && kRingScale       % kBlock == 0, "every ring slot stride must be a whole number of DataBlocks");
static_assert(kRingScore       % 256u   != 0u && kRingPacked     % 256u   != 0u && kRingKey        % 256u   != 0u && kRingValue      % 256u   != 0u && kRingScale      % 256u   != 0u, "a 256-byte ring stride puts every slot on one bank phase");
static_assert(kArenaEnd + kReadPadBytes <= kUbWindow, "UB arena leaves no VECTOR_REG_WIDTH read pad behind the state");

static_assert(kTokens <= turboquant_splitk::kMaxStateRows, "the shared partition stride must hold this operator's M");
static_assert(kHeadSize == turboquant_splitk::kChannels && kStateLanes == turboquant_splitk::kStateLanes, "the shared row record is this operator's own row geometry");
static_assert(turboquant_splitk::kRecordFloats == 2u * kStateLanes + kHeadSize, "the row record is [m, l, acc] and nothing else");
static_assert(kStateLanes * kF32 % kBlock == 0u && kHeadSize * kF32 % kBlock == 0u && (turboquant_splitk::kRecordFloats - kStateLanes) * kF32 % kBlock == 0u && (turboquant_splitk::kRecordFloats - kHeadSize) * kF32 % kBlock == 0u, "the partition burst's lengths and pitches must be whole DataBlocks");
static_assert(kTokens <= 0xFFFFu && kHeadSize * kF32 / kBlock <= 0xFFFFu && (turboquant_splitk::kRecordFloats - kStateLanes) * kF32 / kBlock <= 0xFFFFu, "the partition burst must fit the DataCopyParams fields");
static_assert(kUbLPrev + (kTokens - 1u) * kStateLanes * kF32 + kReadPadBytes <= kUbWindow, "Duplicate over-reads 256 bytes from the last l state block");
static_assert(kBcLanes * kF32 == 8u * kBlock, "a Brcb repeat writes eight DataBlocks");
static_assert(kPackedBytes % kBlock == 0 && kScalePairBytes % kBlock == 0 && (kTokens * kHeadSize * kB16) % kBlock == 0, "every MTE2/MTE3 length must be whole 32-byte bursts");
static_assert(kPanel % ascend_micro::kDequant4MinElements == 0, "a dequant count must be a multiple of 64 nibbles");

// Ring slot addresses, in two colours of the same `base + slot * stride`.
//
// The per-page schedule knows its slot at compile time, so it uses the variable
// templates. The continuous strip pipeline resolves `slot = strip % kStageDepth`
// at RUN time and needs functions -- and in the CCE dialect a plain `constexpr`
// function is `__host__`, so one an `__aicore__` function may call has to carry
// `__aicore__` itself. That colour is also why the templates below cannot be
// written in terms of these functions: a namespace scope initialiser is not a
// device context. The two forms name the same base and the same stride, so the
// only way they can disagree is by editing one and not the other.
__aicore__ inline constexpr std::uint32_t ScoreSlotOf(std::uint32_t slot)   { return kUbScoreRing    + slot * kRingScore; }
__aicore__ inline constexpr std::uint32_t PackedKSlotOf(std::uint32_t slot) { return kUbPackedKRing  + slot * kRingPacked; }
__aicore__ inline constexpr std::uint32_t PackedVSlotOf(std::uint32_t slot) { return kUbPackedVRing  + slot * kRingPacked; }
__aicore__ inline constexpr std::uint32_t KeySlotOf(std::uint32_t slot)     { return kUbKeyRing      + slot * kRingKey; }
__aicore__ inline constexpr std::uint32_t ValueSlotOf(std::uint32_t slot)   { return kUbValueRing    + slot * kRingValue; }
__aicore__ inline constexpr std::uint32_t ScaleSlotOf(std::uint32_t slot)   { return kUbScaleRing    + slot * kRingScale; }
__aicore__ inline constexpr std::uint32_t ArchiveSlotOf(std::uint32_t strip){ return kUbScoreArchive + strip * kRingScore; }

template <std::uint32_t Slot>  constexpr std::uint32_t kScoreSlot   = kUbScoreRing    + Slot * kRingScore;
template <std::uint32_t Slot>  constexpr std::uint32_t kPackedKSlot = kUbPackedKRing  + Slot * kRingPacked;
template <std::uint32_t Slot>  constexpr std::uint32_t kPackedVSlot = kUbPackedVRing  + Slot * kRingPacked;
template <std::uint32_t Slot>  constexpr std::uint32_t kKeySlot     = kUbKeyRing      + Slot * kRingKey;
template <std::uint32_t Slot>  constexpr std::uint32_t kValueSlot   = kUbValueRing    + Slot * kRingValue;
template <std::uint32_t Slot>  constexpr std::uint32_t kScaleSlot   = kUbScaleRing    + Slot * kRingScale;
template <std::uint32_t Strip> constexpr std::uint32_t kArchiveSlot = kUbScoreArchive + Strip * kRingScore;

constexpr std::uint32_t kL1PiTable    = 0u;
constexpr std::uint32_t kL1Q          = NextSlot(kL1PiTable, kTableElems * kB16);
constexpr std::uint32_t kL1QRot       = NextSlot(kL1Q,       kPanel * kB16);
constexpr std::uint32_t kL1KeyRing    = NextSlot(kL1QRot,    kPanel * kB16);
constexpr std::uint32_t kL1SlotStride = kRingKey;
constexpr std::uint32_t kL1End        = kL1KeyRing + kStrips * kL1SlotStride;

static_assert(kL1PiTable  % kBlock == 0 && kL1Q       % kBlock == 0 && kL1QRot   % kBlock == 0 && kL1KeyRing % kBlock == 0 && kL1SlotStride % kBlock == 0, "every L1 operand base must be 32-byte aligned");
static_assert(kL1End <= 512u * 1024u, "the per-strip L1 key pages exceed the L1 buffer");

__aicore__ inline constexpr std::uint32_t L1KeyPageOf(std::uint32_t page)
{
    return kL1KeyRing + page * kL1SlotStride;
}

template <std::uint32_t Strip>
constexpr std::uint32_t kL1KeyPage = kL1KeyRing + Strip * kL1SlotStride;

// The continuous strip pipeline reuses the first kStageDepth of those pages as
// a RING addressed by slot, with an explicit cross-core WAR edge behind it. The
// per-page schedule's one-page-per-strip arrangement carried no such edge --
// its premise was that "every page is read once", which holds inside a page and
// fails the moment a second sequence page reuses the address. That is the
// defect reports/decode_splitk_aic_page_race.md section 5 measures.
static_assert(kStageDepth <= kStrips,
              "the streaming L1 key ring is carved out of the per-strip pages");

template <std::uint32_t Slot> constexpr std::uint16_t kFlagValueReady   = 0u + Slot;
template <std::uint32_t Slot> constexpr std::uint16_t kFlagScoreFree    = 4u + Slot;
template <std::uint32_t Slot> constexpr std::uint16_t kFlagScoreReady   = 8u + Slot;
template <std::uint32_t Slot> constexpr std::uint16_t kFlagKeyReady     = 16u + Slot;

// The event id the continuous strip pipeline's cross-core drains ride on. It
// is distinct from the per-slot kEvtStatic and from the slot ids 0..1, so a
// drain pair can sit inside a strip body without colliding with that body's own
// HardEvent places.
constexpr std::int32_t  kEvtStreamDrain    = 5;

// The mode-4 page-completion flags kFlagPageDrainedV0 / V1 are GONE. They were
// a per-page drain cascade standing where a per-strip L1 WAR back edge should
// have been, and on this toolkit an AIC -> AIV mode-4 wait blocks only on its
// FIRST use of an id: the four waits measured 23146, 3, 3 and 3 ticks against
// announcements ~20600 ticks apart, so the edge did not bound its consumer at
// all. reports/decode_splitk_aic_page_race.md sections 4, 5 and 7. The
// replacement is the mode-2 pair above, and it is per strip, not per page.
static_assert(kFlagValueReady<kStageDepth - 1u> < 11u && kFlagScoreFree<kStageDepth - 1u> < 11u && kFlagScoreReady<kStageDepth - 1u> < 11u, "sub block 0 flag ids must stay below the library-reserved 11..14");
static_assert(kFlagKeyReady<kStageDepth - 1u> - 16u < 11u, "sub block 1 flag ids must stay below 11 in their own half");

// ----------------------------------------------------- the mode-2 channels
// The continuous strip pipeline's whole cross-core discipline: one readiness
// quorum and one release broadcast. Mode 2 is the "AIC and ALL AIVs" fused
// mode, so the AIC's counter advances only once BOTH Vector cores have
// announced, and one AIC announcement releases both of them
// (`key_feature_description.md`, "Full-Core Synchronization Between AIC and AIV
// in a Single AI Core"). That quorum is the property this operator needs and
// mode 4, which addresses one AIV at a time, cannot express.
//
// Both participants name the SAME id. There is no +16 rebase as in mode 4,
// because mode 2 does not address an individual AIV.
//
// The budget, asserted rather than commented, because every claim here is a
// silent-wrong-answer away from being wrong: an id outside 0..15 is truncated
// to its low four bits, and an id another API also uses aliases two channels.
struct Mode2IdBudget {
    static constexpr std::uint16_t kIds = 16u;           // 0..15 per core
    // SyncAll's hardware path claims 11..14 (SyncAll.md, "Constraints").
    static constexpr std::uint16_t kSyncAllFirst = 11u;
    static constexpr std::uint16_t kSyncAllLast  = 14u;
    // The Matmul high-level API claims [0, 2N - 1] for N objects, N <= 4. This
    // operator defines no Matmul object -- it issues Mmad directly -- but the
    // budget is asserted against the worst case, so adding one later cannot
    // silently alias a channel.
    static constexpr std::uint16_t kMatmulLast = 7u;
    // Each id is a counter in 0..15; overflowing it is a reported error, not a
    // wrap. kMaxOutstanding below is this pipeline's worst case.
    static constexpr std::uint16_t kCounterMax = 15u;

    static constexpr bool Free(std::uint16_t id)
    {
        return id < kIds && id > kMatmulLast &&
               (id < kSyncAllFirst || id > kSyncAllLast);
    }
};

// Readiness, both AIVs -> AIC, consumed by ONE quorum wait per strip.
constexpr std::uint16_t kSync2StageReady      = 8u;
// The same channel, named so that section 6.6's fence audit can tell the
// PRIMING sets -- "the score slots start empty" -- from announcements of
// earlier pipe work, which are the only ones that need a drain (section 6.2a).
constexpr std::uint16_t kSync2StageReadyPrime = kSync2StageReady;
// Release, AIC -> both AIVs, one broadcast per retired strip.
constexpr std::uint16_t kSync2StepRetired     = 9u;

static_assert(Mode2IdBudget::Free(kSync2StageReady),
              "the readiness channel must not alias SyncAll or Matmul");
static_assert(Mode2IdBudget::Free(kSync2StepRetired),
              "the release channel must not alias SyncAll or Matmul");
static_assert(kSync2StageReady != kSync2StepRetired,
              "readiness and release must never share an id: conflating data "
              "readiness with buffer release is the fault this replaces");

// Worst-case outstanding announcements on either channel. Each AIV's n-th
// readiness announcement is itself behind the AIC's (n - kStageDepth)-th
// release, and the AIC's n-th release is behind its n-th readiness wait, so
// neither side can run more than kStageDepth strips ahead of the other, plus
// the one announcement in flight.
constexpr std::uint16_t kMaxOutstanding = static_cast<std::uint16_t>(kStageDepth + 1u);
static_assert(kMaxOutstanding <= Mode2IdBudget::kCounterMax,
              "a mode-2 flag counter overflow is a reported error, not a wrap");

// Which pipe each participant may name. The library gates both calls on
// IsSplitVectorPipe / IsSplitCubePipe (CANN 9.2
// impl/basic_api/kernel_event.h:257, :263), and on __NPU_ARCH__ == 3510 the two
// sets are nearly disjoint: a call naming the other side's pipe compiles to
// NOTHING, silently and with no diagnostic. PIPE_S is excluded because mode 2
// does not support it (CrossCoreSetFlag_ISASI.md, "Supported Pipeline Type").
constexpr bool Mode2AivPipe(pipe_t pipe)
{
    return pipe == PIPE_V || pipe == PIPE_MTE2 || pipe == PIPE_MTE3;
}

constexpr bool Mode2AicPipe(pipe_t pipe)
{
    return pipe == PIPE_MTE1 || pipe == PIPE_MTE2 || pipe == PIPE_FIX || pipe == PIPE_M;
}

// The AIC waits on MTE1 because MTE1 is the pipe that READS the L1 key page.
// One wait is enough for the Fixpipe too: the Fixpipe is chained to the Mmad
// (M_FIX) and the Mmad to the L1 -> L0B load (MTE1_M), so blocking MTE1 orders
// the whole triple. It announces on FIX, the last pipe it drains.
constexpr pipe_t kSync2AicWait      = PIPE_MTE1;
constexpr pipe_t kSync2AicAnnounce  = PIPE_FIX;
// AIV0's recurrence is Vector work and reads the score panel, so it waits on V
// and announces on V.
constexpr pipe_t kSync2Aiv0Wait     = PIPE_V;
constexpr pipe_t kSync2Aiv0Announce = PIPE_V;
// AIV1 waits on MTE3, the pipe that WRITES the L1 key page, and announces on
// MTE3, the work the announcement reports.
constexpr pipe_t kSync2Aiv1Wait     = PIPE_MTE3;
constexpr pipe_t kSync2Aiv1Announce = PIPE_MTE3;

static_assert(Mode2AicPipe(kSync2AicWait) && Mode2AicPipe(kSync2AicAnnounce),
              "an AIC mode-2 call naming a Vector-only pipe compiles to nothing");
static_assert(Mode2AivPipe(kSync2Aiv0Wait) && Mode2AivPipe(kSync2Aiv0Announce) &&
                  Mode2AivPipe(kSync2Aiv1Wait) && Mode2AivPipe(kSync2Aiv1Announce),
              "an AIV mode-2 call naming a Cube-only pipe compiles to nothing");

constexpr std::int32_t kEvtStatic = 3;

template <std::uint32_t Column>
constexpr std::uint32_t kSlotOf = Column % kStageDepth;

// The runtime forms, as masks. Exact because both are powers of two, and a
// mask rather than a modulo because the scalar unit is the whole core's
// instruction dispatcher: a cycle it spends on address arithmetic is a cycle
// it is not issuing the next burst.
static_assert((kStageDepth & (kStageDepth - 1u)) == 0u,
              "the slot index is formed by masking, so the ring depth must be a power of two");
static_assert((kStrips & (kStrips - 1u)) == 0u,
              "the archive index is formed by masking, so the strip count must be a power of two");

__aicore__ inline constexpr std::uint32_t SlotOfStrip(std::uint32_t strip)
{
    return strip & (kStageDepth - 1u);
}

__aicore__ inline constexpr std::uint32_t ArchiveOfStrip(std::uint32_t strip)
{
    return strip & (kStrips - 1u);
}
using ColumnSeq = std::make_integer_sequence<std::uint32_t, kColumns>;

constexpr auto kUb = AscendC::TPosition::VECCALC;

struct KeyEngine {
    AscendC::GlobalTensor<std::int8_t> key_cache;
    std::uint32_t                      strip_base_bytes = 0;
};

template <std::uint32_t Slot, std::uint32_t Strip>
__aicore__ inline void StageKeyStrip(KeyEngine& engine, std::uint32_t strip)
{
    static_assert(Slot < kStageDepth && Strip < kStrips);
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);

    ascend_micro::NativeDequant4Engine<bfloat16_t> dequant_engine;
    dequant_engine.packed = {kUb, kPackedKSlot<Slot>, kPackedBytes};
    dequant_engine.output = {kUb, kKeySlot<Slot>,     kPanel};

    ascend_micro::NativeCvTransferEngine<bfloat16_t> transfer_engine;
    transfer_engine.ub_position   = kUb;
    transfer_engine.l1_position   = AscendC::TPosition::B1;
    transfer_engine.ub_base_bytes = kKeySlot<Slot>;
    transfer_engine.l1_base_bytes = kL1KeyPage<Strip>;

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt);
    AscendC::DataCopy(dequant_engine.packed, engine.key_cache[engine.strip_base_bytes + strip * kPackedBytes], kPackedBytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evt);
    ascend_micro::turboquant_dequant4_signed(dequant_engine, kPanel);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evt);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evt);
    ascend_micro::cv_transfer_nd2nz<kTokens, kHeadSize, kHeadSize, 16u, 1u>(transfer_engine);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evt);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evt);
    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(kFlagKeyReady<Slot>);
}

template <std::uint32_t Column>
__aicore__ inline void StepColumnKey(KeyEngine& engine, std::uint32_t strips)
{
    if constexpr (Column >= 1u && Column - 1u < kStrips) {
        if (Column - 1u < strips) {
            StageKeyStrip<kSlotOf<Column - 1u>, Column - 1u>(engine, Column - 1u);
        }
    }
}

template <std::uint32_t... Columns>
__aicore__ inline void StepColumnsKey(KeyEngine& engine, std::uint32_t strips, std::integer_sequence<std::uint32_t, Columns...>)
{
    (StepColumnKey<Columns>(engine, strips), ...);
}

__aicore__ inline void run_key_core(__gm__ std::int8_t* gm_key_cache, std::uint32_t strip_base_bytes, std::uint32_t strips)
{
    KeyEngine engine;
    engine.key_cache.SetGlobalBuffer(gm_key_cache);
    engine.strip_base_bytes = strip_base_bytes;

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);

    StepColumnsKey(engine, strips, ColumnSeq{});

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
}

struct ValueEngine {
    AscendC::GlobalTensor<std::int8_t> value_cache;
    AscendC::GlobalTensor<float>        scale_cache;
    AscendC::GlobalTensor<float>        gate;
    std::uint32_t                      strip_base_bytes = 0;
    std::uint32_t                      key_scale_base   = 0;
    std::uint32_t                      value_scale_base = 0;
    std::uint32_t                      gate_base        = 0;
};

template <std::uint32_t Slot>
__aicore__ inline void LoadValueStrip(ValueEngine& engine, std::uint32_t strip)
{
    static_assert(Slot < kStageDepth);
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);

    AscendC::LocalTensor<std::int8_t> packed{kUb,      kPackedVSlot<Slot>,                    kPackedBytes};
    AscendC::LocalTensor<float>        key_scale{kUb,   kScaleSlot<Slot>,                      kTokens};
    AscendC::LocalTensor<float>        value_scale{kUb, kScaleSlot<Slot> + kTokens * kF32,     kTokens};

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt);
    AscendC::DataCopy(packed,      engine.value_cache[engine.strip_base_bytes + strip * kPackedBytes], kPackedBytes);
    AscendC::DataCopy(key_scale,   engine.scale_cache[engine.key_scale_base   + strip * kTokens],     kTokens);
    AscendC::DataCopy(value_scale, engine.scale_cache[engine.value_scale_base + strip * kTokens],     kTokens);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evt);
}

template <std::uint32_t Slot>
__aicore__ inline void DequantValueStrip(ValueEngine&)
{
    static_assert(Slot < kStageDepth);
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);

    ascend_micro::NativeDequant4Engine<float> dequant_engine;
    dequant_engine.packed         = {kUb, kPackedVSlot<Slot>, kPackedBytes};
    dequant_engine.signed_nibbles = {kUb, kUbDqSigned,        kPanel};
    dequant_engine.output         = {kUb, kValueSlot<Slot>,   kPanel};

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evt);
    ascend_micro::turboquant_dequant4_signed(dequant_engine, kPanel);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagValueReady<Slot>);
}

// Every operand of the recurrence that depends only on the ring slot. Shared
// by the per-page schedule and the continuous strip pipeline, so the two cannot
// drift: the carried (m, l, Acc) sit at FIXED UB offsets in both, which is the
// ordering property tests/cpu/test_decode_page_loop_model.py checks.
__aicore__ inline void FillRecurEngine(ascend_micro::NativeDecodeRecurrenceEngine& recur_engine,
                                       std::uint32_t slot)
{
    recur_engine.position     = kUb;
    recur_engine.panel_base   = ScoreSlotOf(slot);
    recur_engine.m_base       = kUbMPrev;
    recur_engine.l_base       = kUbLPrev;
    recur_engine.acc_base     = kUbAcc;
    recur_engine.panel        = {kUb, ScoreSlotOf(slot),                  kScores};
    recur_engine.key_scale    = {kUb, ScaleSlotOf(slot),                  kTokens};
    recur_engine.value_scale  = {kUb, ScaleSlotOf(slot) + kTokens * kF32, kTokens};
    recur_engine.value_page   = {kUb, ValueSlotOf(slot),                  kPanel};
    recur_engine.m_prev       = {kUb, kUbMPrev,                           kTokens * kStateLanes};
    recur_engine.l_prev       = {kUb, kUbLPrev,                           kTokens * kStateLanes};
    recur_engine.acc          = {kUb, kUbAcc,                             kTokens * kHeadSize};
    recur_engine.scaled_row   = {kUb, kUbScaledRow,                       kTokens};
    recur_engine.centered     = {kUb, kUbCentered,                        kTokens};
    recur_engine.p            = {kUb, kUbP,                               kTokens};
    recur_engine.p_scaled     = {kUb, kUbPScaled,                         kTokens};
    recur_engine.p_x4         = {kUb, kUbPx4,                             kBcLanes};
    recur_engine.m_new        = {kUb, kUbMNew,                            kStateLanes};
    recur_engine.l_new        = {kUb, kUbLNew,                            kStateLanes};
    recur_engine.alpha        = {kUb, kUbAlpha,                           kStateLanes};
    recur_engine.alpha_delta  = {kUb, kUbAlphaDelta,                      kStateLanes};
    recur_engine.l_scaled     = {kUb, kUbLScaled,                         kStateLanes};
    recur_engine.tile_max_raw = {kUb, kUbTileMaxRaw,                      kStateLanes};
    recur_engine.p_sum_raw    = {kUb, kUbPSumRaw,                         kStateLanes};
    recur_engine.tile_max_bc  = {kUb, kUbTileMaxBc,                       kBcLanes};
    recur_engine.m_bc         = {kUb, kUbMBc,                             kBcLanes};
    recur_engine.p_sum_bc     = {kUb, kUbPSumBc,                          kBcLanes};
    recur_engine.alpha_bc     = {kUb, kUbAlphaBc,                         kBcLanes};
    recur_engine.weighted     = {kUb, kUbWeighted,                        kPanel};
    recur_engine.weighted_sum = {kUb, kUbWeightedSum,                     kHeadSize};
    recur_engine.acc_scaled   = {kUb, kUbAccScaled,                       kHeadSize};
    recur_engine.lane_index   = {kUb, kUbLaneIndex,                       kTokens};
    recur_engine.masked_fill  = {kUb, kUbMaskedFill,                      kTokens};
    recur_engine.zero_fill    = {kUb, kUbZeroFill,                        kTokens};
    recur_engine.tail_sel     = {kUb, kUbTailSel,                         kTokens};
}

template <std::uint32_t Slot, std::uint32_t Strip>
__aicore__ inline void RecurStrip(ValueEngine&, std::uint32_t active)
{
    static_assert(Slot < kStageDepth && Strip < kStrips);
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);

    ascend_micro::NativeDecodeRecurrenceEngine recur_engine;
    FillRecurEngine(recur_engine, Slot);

    AscendC::LocalTensor<float> panel{kUb,    kScoreSlot<Slot>,    kScores};
    AscendC::LocalTensor<float> archived{kUb, kArchiveSlot<Strip>, kScores};

    recur_engine.prime_tail_mask(turboquant_decode_tail::kMaskedScore);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::CrossCoreWaitFlag<4, PIPE_V>(kFlagScoreReady<Slot>);
    if (active == kTokens) {
        ascend_micro::vector_decode_recur_strip(recur_engine);
    } else if (active != 0u) {
        ascend_micro::vector_decode_recur_strip_masked(recur_engine, active);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(archived, panel, 1.0f, static_cast<std::int32_t>(kScores));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evt);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagScoreFree<Slot>);
}

template <std::uint32_t Column>
__aicore__ inline void StepColumnValue(ValueEngine& engine, std::uint32_t strips, std::uint32_t live_tokens)
{
    if constexpr (Column >= kPipelineLag && Column - kPipelineLag < kStrips) {
        if (Column - kPipelineLag < strips) {
            RecurStrip<kSlotOf<Column - kPipelineLag>, Column - kPipelineLag>(
                engine, turboquant_decode_tail::ActiveInStrip(
                            static_cast<std::uint64_t>(Column - kPipelineLag) * turboquant_decode_tail::kStripTokens,
                            live_tokens));
        }
    }
    if constexpr (Column >= 3u && Column - 3u < kStrips) {
        if (Column - 3u < strips) DequantValueStrip<kSlotOf<Column - 3u>>(engine);
    }
    if constexpr (Column >= 2u && Column - 2u < kStrips) {
        if (Column - 2u < strips) LoadValueStrip<kSlotOf<Column - 2u>>(engine, Column - 2u);
    }
}

template <std::uint32_t... Columns>
__aicore__ inline void StepColumnsValue(ValueEngine& engine, std::uint32_t strips, std::uint32_t live_tokens, std::integer_sequence<std::uint32_t, Columns...>)
{
    (StepColumnValue<Columns>(engine, strips, live_tokens), ...);
}

__aicore__ inline void EgressQueries(ascend_micro::NativeEgressRotateEngine& egress_engine, ValueEngine& engine)
{
    AscendC::LocalTensor<float> gate_row{kUb, kUbGateRow, kHeadSize};
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtStatic);
    for (std::uint32_t query = 0; query < kTokens; ++query) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtStatic);
        AscendC::DataCopy(gate_row, engine.gate[engine.gate_base + query * kHeadSize], kHeadSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtStatic);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtStatic);
        egress_engine.select_query(query);
        ascend_micro::vector_egress_query(egress_engine);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtStatic);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtStatic);
}

__aicore__ inline void run_value_core(__gm__ std::int8_t*   gm_value_cache,
                                      __gm__ float*          gm_scale_cache,
                                      __gm__ float*          gm_gate,
                                      __gm__ std::uint32_t*  gm_gather_offsets,
                                      __gm__ float*          gm_gather_signs,
                                      __gm__ float*          gm_pi_sign,
                                      __gm__ float*          gm_out_scores,
                                      __gm__ float*          gm_out_state,
                                      __gm__ float*          gm_out_acc,
                                      __gm__ float*          gm_partition,
                                      __gm__ bfloat16_t*     gm_out_context,
                                      __gm__ float*          gm_out_rotated,
                                      std::uint32_t          strip_base_bytes,
                                      std::uint32_t          key_scale_base,
                                      std::uint32_t          value_scale_base,
                                      std::uint32_t          gate_base,
                                      std::uint32_t          strips,
                                      std::uint32_t          live_tokens,
                                      bool                   partial    = false,
                                      bool                   initialize = true,
                                      bool                   finalize   = true)
{
    ValueEngine engine;
    engine.value_cache.SetGlobalBuffer(gm_value_cache);
    engine.scale_cache.SetGlobalBuffer(gm_scale_cache);
    engine.gate.SetGlobalBuffer(gm_gate);
    engine.strip_base_bytes = strip_base_bytes;
    engine.key_scale_base   = key_scale_base;
    engine.value_scale_base = value_scale_base;
    engine.gate_base        = gate_base;

    AscendC::GlobalTensor<float>      out_scores;
    AscendC::GlobalTensor<float>      out_state;
    AscendC::GlobalTensor<float>      out_acc;
    AscendC::GlobalTensor<float>      out_rotated;
    AscendC::GlobalTensor<bfloat16_t> out_context;
    out_scores.SetGlobalBuffer(gm_out_scores);
    out_state.SetGlobalBuffer(gm_out_state);
    out_acc.SetGlobalBuffer(gm_out_acc);
    out_context.SetGlobalBuffer(gm_out_context);
    out_rotated.SetGlobalBuffer(gm_out_rotated);

    AscendC::LocalTensor<std::uint32_t> gather_offsets{kUb, kUbGatherOffsets, kGatherStages * kHeadSize};
    AscendC::LocalTensor<float>         gather_signs{kUb,   kUbGatherSigns,   kGatherStages * kHeadSize};
    AscendC::LocalTensor<float>         pi_sign{kUb,        kUbPiSign,        kHeadSize};

    AscendC::GlobalTensor<std::uint32_t> gm_offsets;
    AscendC::GlobalTensor<float>         gm_signs;
    AscendC::GlobalTensor<float>         gm_signs_pi;
    gm_offsets.SetGlobalBuffer(gm_gather_offsets);
    gm_signs.SetGlobalBuffer(gm_gather_signs);
    gm_signs_pi.SetGlobalBuffer(gm_pi_sign);

    if (!partial) {
        AscendC::DataCopy(gather_offsets, gm_offsets,  kGatherStages * kHeadSize);
        AscendC::DataCopy(gather_signs,   gm_signs,    kGatherStages * kHeadSize);
        AscendC::DataCopy(pi_sign,        gm_signs_pi, kHeadSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtStatic);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtStatic);
    }

    AscendC::LocalTensor<float> m_prev{kUb, kUbMPrev, kTokens * kStateLanes};
    AscendC::LocalTensor<float> l_prev{kUb, kUbLPrev, kTokens * kStateLanes};
    AscendC::LocalTensor<float> acc{kUb,    kUbAcc,   kTokens * kHeadSize};

    if (initialize) {
        AscendC::Duplicate(m_prev, -3.0e38f, static_cast<std::int32_t>(kTokens * kStateLanes));
        AscendC::Duplicate(l_prev, 0.0f,     static_cast<std::int32_t>(kTokens * kStateLanes));
        AscendC::Duplicate(acc,    0.0f,     static_cast<std::int32_t>(kTokens * kHeadSize));
    }

    if (strips != 0u) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagScoreFree<0>);
        AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagScoreFree<1>);

        StepColumnsValue(engine, strips, live_tokens, ColumnSeq{});

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
    }

    if (partial) {
        if (finalize) {
            AscendC::GlobalTensor<float> partition;
            partition.SetGlobalBuffer(gm_partition);
            constexpr std::uint32_t kRecord = turboquant_splitk::kRecordFloats;

            AscendC::DataCopyParams lane;
            lane.blockCount = static_cast<std::uint16_t>(kTokens);
            lane.blockLen   = static_cast<std::uint16_t>(kStateLanes * kF32 / kBlock);
            lane.srcStride  = 0;
            lane.dstStride  = static_cast<std::uint16_t>((kRecord - kStateLanes) * kF32 / kBlock);

            AscendC::DataCopyParams band;
            band.blockCount = static_cast<std::uint16_t>(kTokens);
            band.blockLen   = static_cast<std::uint16_t>(kHeadSize * kF32 / kBlock);
            band.srcStride  = 0;
            band.dstStride  = static_cast<std::uint16_t>((kRecord - kHeadSize) * kF32 / kBlock);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStatic);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStatic);
            AscendC::DataCopy(partition,                         m_prev, lane);
            AscendC::DataCopy(partition[kStateLanes],            l_prev, lane);
            AscendC::DataCopy(partition[2u * kStateLanes],       acc,    band);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtStatic);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtStatic);
        }
        return;
    }

    ascend_micro::NativeEgressRotateEngine egress_engine;
    egress_engine.position       = kUb;
    egress_engine.acc_base       = kUbAcc;
    egress_engine.l_base         = kUbLPrev;
    egress_engine.context_base   = kUbContext;
    egress_engine.rotated_base   = kUbRotated;
    egress_engine.acc            = acc;
    egress_engine.l_prev         = l_prev;
    egress_engine.gate           = {kUb, kUbGateRow,    kHeadSize};
    egress_engine.pi_sign        = pi_sign;
    egress_engine.gather_offsets = gather_offsets;
    egress_engine.gather_signs   = gather_signs;
    egress_engine.l_broadcast    = {kUb, kUbLBroadcast, kHeadSize};
    egress_engine.buffer_a       = {kUb, kUbEgressA,    kHeadSize};
    egress_engine.buffer_b       = {kUb, kUbEgressB,    kHeadSize};
    egress_engine.permuted       = {kUb, kUbPermuted,   kHeadSize};
    egress_engine.signed_tmp     = {kUb, kUbSignedTmp,  kHeadSize};
    egress_engine.denominator    = {kUb, kUbDenominator,kHeadSize};
    egress_engine.context        = {kUb, kUbContext,    kTokens * kHeadSize};
    egress_engine.rotated        = {kUb, kUbRotated,    kTokens * kHeadSize};

    EgressQueries(egress_engine, engine);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStatic);

    AscendC::LocalTensor<float> score_archive{kUb, kUbScoreArchive, kStrips * kScores};
    AscendC::DataCopyParams panels;
    panels.blockCount = static_cast<std::uint16_t>(strips);
    panels.blockLen   = static_cast<std::uint16_t>(kScoreBytes / kBlock);
    panels.srcStride  = static_cast<std::uint16_t>((kRingScore - kScoreBytes) / kBlock);
    panels.dstStride  = 0;
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

struct CubeEngine {
    __ca__ bfloat16_t* l0_query;
    __cb__ bfloat16_t* l0_table;
    __cc__ float*      l0_scores;
};

template <std::uint32_t Slot, std::uint32_t Strip>
__aicore__ inline void GemmStrip(CubeEngine& engine)
{
    static_assert(Slot < kStageDepth && Strip < kStrips);
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);

    auto* l0b_slot   = engine.l0_table  + Slot * kHeadSize * kTokens;
    auto* l0c_window = engine.l0_scores + Slot * kScores;
    auto* ub_scores  = reinterpret_cast<__ubuf__ float*>(kScoreSlot<Slot>);

    AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(kFlagKeyReady<Slot>);
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(kFlagValueReady<Slot>);

    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(evt);
    ascend_micro::score_strip_stage_l0(kL1KeyPage<Strip>, l0b_slot);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(evt);

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(evt);
    ascend_micro::score_strip_compute(engine.l0_query, l0b_slot, l0c_window);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(evt);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(evt);
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagScoreFree<Slot>);
    ascend_micro::score_strip_fixpipe(l0c_window, ub_scores, false);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(evt);
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(evt);
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagScoreReady<Slot>);
}

template <std::uint32_t Column>
__aicore__ inline void StepColumnCube(CubeEngine& engine, std::uint32_t strips)
{
    if constexpr (Column >= 2u && Column - 2u < kStrips) {
        if (Column - 2u < strips) {
            GemmStrip<kSlotOf<Column - 2u>, Column - 2u>(engine);
        }
    }
}

template <std::uint32_t... Columns>
__aicore__ inline void StepColumnsCube(CubeEngine& engine, std::uint32_t strips, std::integer_sequence<std::uint32_t, Columns...>)
{
    (StepColumnCube<Columns>(engine, strips), ...);
}

// The Cube's ingress rotation prologue: Q . Pi^T into L1, then into L0A. It
// is identical for both schedules, so both call it.
__aicore__ inline void CubeRotatePrologue(__gm__ bfloat16_t* gm_q,
                                          __gm__ bfloat16_t* gm_pi_table,
                                          __gm__ bfloat16_t* gm_out_qrot,
                                          std::uint32_t      q_base,
                                          CubeEngine&        engine)
{
    ascend_micro::stage_page_nz<kTokens, kHeadSize>(gm_q + q_base, kL1Q, kHeadSize);
    ascend_micro::stage_blocks_nz<kTableBlocks, kGroup>(gm_pi_table, kL1PiTable);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtStatic);

    ascend_micro::rotate_query_stage_l0(kL1PiTable, kL1Q, engine.l0_query, engine.l0_table);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtStatic);

    ascend_micro::rotate_query_compute(engine.l0_query, engine.l0_table, engine.l0_scores);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEvtStatic);

    ascend_micro::rotate_query_to_gm(gm_out_qrot, engine.l0_scores);
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtStatic);

    ascend_micro::stage_page_nz<kTokens, kHeadSize>(gm_out_qrot, kL1QRot, kHeadSize);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEvtStatic);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtStatic);

    ascend_micro::load_query_rotated(kL1QRot, engine.l0_query);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtStatic);
}

__aicore__ inline void run_cube_core(__gm__ bfloat16_t* gm_q,
                                     __gm__ bfloat16_t* gm_pi_table,
                                     __gm__ bfloat16_t* gm_out_qrot,
                                     std::uint32_t      q_base,
                                     std::uint32_t      strips)
{
    using namespace AscendC::Te;

    __ca__ bfloat16_t l0_query[kPanel]       __attribute__((aligned(512)));
    __cb__ bfloat16_t l0_table[kTableElems]  __attribute__((aligned(512)));
    __cc__ float      l0_scores[kPanel]      __attribute__((aligned(1024)));

    static_assert(kStageDepth * kHeadSize * kTokens <= kTableElems, "the key slots must fit inside the L0B table allocation");
    static_assert(kStageDepth * kScores             <= kPanel,      "the score windows must fit inside the L0C panel allocation");

    CubeEngine engine{l0_query, l0_table, l0_scores};

    CubeRotatePrologue(gm_q, gm_pi_table, gm_out_qrot, q_base, engine);

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
}

// =========================================================================
// The continuous strip pipeline, for the Split-K / temporal entry
// =========================================================================
//
// ONE flat sequence over the window's strips, agnostic of page boundaries
// inside the loop. Column c carries the same five stages at the same offsets as
// the per-page schedule at the top of this file -- what changes is that the
// strip index is GLOBAL, so the sequence runs straight through a page boundary
// instead of priming and draining the whole ring at each one:
//
//   AIV1  MTE2 + V + MTE3   stage key strip c - 1  -> L1 key page (c - 1) % 2
//   AIC   MTE1 + M + FIX    GEMM strip c - 2       -> UB score slot c % 2
//   AIV0  MTE2              load value strip c - 2
//   AIV0  V                 dequant value strip c - 3
//   AIV0  V                 recurrence strip c - 4  <- UB score slot c % 2
//
// Three structural differences from the per-page schedule, and why each one is
// there:
//
// 1. The L1 key ring is kStageDepth pages addressed by SLOT, not kStrips pages
//    addressed by STRIP. The per-strip arrangement had no WAR back edge at all;
//    its comment argued "every page is read once", which is true inside one
//    sequence page and false across the page loop, where page p + 1's strip s
//    restages page p's address.
// 2. The intra-core feedback cycles are primed ONCE before the stream and
//    drained ONCE after it. There is no inter-page drain cascade left.
// 3. Cross-core ordering is the two mode-2 channels and nothing else, so ring
//    reuse rests on the token structure rather than on a hardware property of a
//    mode-4 latch.
//
// The rendezvous, per strip
// -------------------------
// Readiness, kSync2StageReady, both AIVs -> AIC, ONE quorum wait:
//   AIV1's n-th announcement = "the L1 key page holds strip n".
//   AIV0's n-th announcement = "the UB score slot strip n will land in is
//     free", i.e. the recurrence of strip n - kStageDepth has retired. Its
//     first kStageDepth announcements are PRIMING sets: the slots start empty.
//   The AIC's n-th wait therefore blocks until BOTH hold, which is exactly the
//   pair of preconditions its n-th GEMM has. Readiness and release never share
//   an id, so neither meaning can stand in for the other.
//
// Release, kSync2StepRetired, AIC -> both AIVs, one broadcast:
//   the AIC's n-th announcement = "strip n is retired": its L1 key page has
//   been read into L0B and its scores have landed in UB.
//   AIV1's wait before staging strip t is its (t - kStageDepth)-th, so it is
//     released by "strip t - kStageDepth retired" -- the strip that owns the L1
//     page strip t is about to overwrite.
//   AIV0's wait before the recurrence of strip t is its t-th, so it is released
//     by "strip t retired" -- the scores it is about to read.
//
// Token census per channel, which is what keeps each counter inside its 0..15
// range and what tests/cpu/test_decode_page_loop_model.py explores
// exhaustively, with `total` the window's strip count:
//
//   readiness  AIV0 min(kStageDepth, total) primes + max(0, total - kStageDepth)
//              post-recurrence announcements = total; AIV1 total; AIC total
//              waits.
//   release    AIC total announcements; AIV0 total waits; AIV1
//              max(0, total - kStageDepth) in the stream + min(kStageDepth,
//              total) in its epilogue = total.
//
// Every wait carries its own x_S scalar drain immediately behind it, on the
// pipe the wait names. A CrossCoreWaitFlag on Ascend 950PR blocks only the
// issuance of that one pipe, so the drain is what converts the wait into a
// stall on instruction ISSUE -- which bounds the whole core rather than one
// queue, and is also the only thing scripts/audit_mode4_tick_order.py can
// measure a stall from.

// The window one Split-K worker owns, in units of strips, plus everything
// needed to turn a global strip index into Global Memory addresses.
struct StripWindow {
    std::uint32_t strips_per_page = 0;  // block_size / kTokens
    std::uint32_t page_begin      = 0;  // first LOGICAL page, range offset included
    std::uint32_t heads           = 0;
    std::uint32_t head            = 0;
    std::uint32_t block_size      = 0;
    std::uint32_t table_base      = 0;  // request * seq_pages
    std::uint32_t scale_plane     = 0;  // cache_pages * heads * block_size
    std::uint32_t live_seq        = 0;
    std::uint32_t total           = 0;  // strips in the window, dead ones included
};

// A monotone cursor over the window's strips. No division and no modulo in the
// loop, and ONE block-table read per page rather than one per strip.
//
// The walk is strictly increasing in the strip index
// (turboquant_decode_tail::WalkIsMonotone, which every caller already
// validates), so `strip / strips_per_page` and `strip % strips_per_page` are a
// counter that rolls over -- an add and a compare instead of a divide. The
// token base rises by exactly kTokens per strip across a page boundary as much
// as inside a page, because strips_per_page * kTokens == block_size, which is
// what lets the tail mask ride the same counter.
//
// Each core keeps its own cursor, and AIV0 keeps TWO: its ingress and its
// recurrence walk the same sequence at different lags, so one cursor cannot
// serve both.
struct StripCursor {
    __gm__ std::uint32_t* table = nullptr;
    std::uint32_t strips_per_page = 0, heads = 0, head = 0, block_size = 0;
    std::uint32_t table_base = 0, page_begin = 0, total = 0;
    std::uint32_t strip = 0, strip_in_page = 0, page_idx = 0, page_base = 0;
    std::uint64_t token_base = 0;

    __aicore__ inline void Resolve()
    {
        const std::uint32_t physical = table[table_base + page_begin + page_idx];
        page_base = (physical * heads + head) * block_size;
    }

    __aicore__ inline void Begin(const StripWindow& window, __gm__ std::uint32_t* gm_table)
    {
        table           = gm_table;
        strips_per_page = window.strips_per_page;
        heads           = window.heads;
        head            = window.head;
        block_size      = window.block_size;
        table_base      = window.table_base;
        page_begin      = window.page_begin;
        total           = window.total;
        strip = 0; strip_in_page = 0; page_idx = 0;
        token_base = static_cast<std::uint64_t>(page_begin) * block_size;
        if (total != 0u) Resolve();
    }

    // Guarded by `total`: past the last strip there is no further page, and
    // resolving one would read the block table outside this worker's window.
    __aicore__ inline void Advance()
    {
        ++strip;
        if (strip >= total) return;
        ++strip_in_page;
        token_base += kTokens;
        if (strip_in_page == strips_per_page) {
            strip_in_page = 0;
            ++page_idx;
            Resolve();
        }
    }

    // Byte offset of this strip's packed int4 panel in the key or value pool.
    // Both pools share the layout, so one accessor serves both -- as the
    // per-page entry already relied on. `kHeadSize / 2` is folded first rather
    // than multiplying by kHeadSize and halving, which keeps the product
    // inside 32 bits for a cache that the other order would overflow.
    __aicore__ inline std::uint32_t PackedBytes() const
    {
        return page_base * (kHeadSize / 2u) + strip_in_page * kPackedBytes;
    }

    // Element offset of this strip's slice of the KEY scale plane; the value
    // plane sits `scale_plane` elements further on.
    __aicore__ inline std::uint32_t ScaleElems() const
    {
        return page_base + strip_in_page * kTokens;
    }

    // Live tokens in this strip, from the running token base rather than from
    // the strip index -- the same value WalkActive would return, without its
    // division.
    __aicore__ inline std::uint32_t Active(std::uint32_t live_seq) const
    {
        return turboquant_decode_tail::ActiveInStrip(token_base, live_seq);
    }
};

// ------------------------------------------------------------------- AIV1
// The engines are built HERE, in the strip body, and not hoisted. Hoisting
// them into a per-slot array was measured 2.1x slower: inline, a LocalTensor's
// base is a constant the compiler folds into an immediate; in an array that
// outlives the body it is memory, and every operand access becomes a load.
// reports/decode_splitk_aic_scalar_issue.md has the numbers.
__aicore__ inline void StreamKeyStrip(KeyEngine& engine, StripCursor& cursor,
                                      std::uint32_t strip)
{
    const std::uint32_t  slot = SlotOfStrip(strip);
    const std::int32_t   evt  = static_cast<std::int32_t>(slot);

    ascend_micro::NativeDequant4Engine<bfloat16_t> dequant_engine;
    dequant_engine.packed = {kUb, PackedKSlotOf(slot), kPackedBytes};
    dequant_engine.output = {kUb, KeySlotOf(slot),     kPanel};

    ascend_micro::NativeCvTransferEngine<bfloat16_t> transfer_engine;
    transfer_engine.ub_position   = kUb;
    transfer_engine.l1_position   = AscendC::TPosition::B1;
    transfer_engine.ub_base_bytes = KeySlotOf(slot);
    transfer_engine.l1_base_bytes = L1KeyPageOf(slot);

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt);
    AscendC::DataCopy(dequant_engine.packed,
                      engine.key_cache[cursor.PackedBytes()], kPackedBytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evt);
    ascend_micro::turboquant_dequant4_signed(dequant_engine, kPanel);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evt);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evt);

    // WAR on the L1 key page. This strip overwrites what strip
    // `strip - kStageDepth` put there, so the AIC must already have read that
    // page into L0B. The wait gates PIPE_MTE3 -- the pipe that writes L1 -- and
    // the MTE3_S drain behind it stalls instruction issue until MTE3 has
    // retired past the wait, which also holds the NEXT strip's MTE2 ingress
    // behind the release. This is the edge that was missing.
    if (strip >= kStageDepth) {
        AscendC::CrossCoreWaitFlag<2, kSync2Aiv1Wait>(kSync2StepRetired);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtStreamDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtStreamDrain);
    }

    ascend_micro::cv_transfer_nd2nz<kTokens, kHeadSize, kHeadSize, 16u, 1u>(transfer_engine);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evt);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evt);
    AscendC::CrossCoreSetFlag<2, kSync2Aiv1Announce>(kSync2StageReady);
}

__aicore__ inline void stream_key_core(__gm__ std::int8_t*   gm_key_cache,
                                       __gm__ std::uint32_t* gm_table,
                                       const StripWindow&    window)
{
    if (window.total == 0u) return;

    KeyEngine engine;
    engine.key_cache.SetGlobalBuffer(gm_key_cache);
    engine.strip_base_bytes = 0;

    StripCursor cursor;
    cursor.Begin(window, gm_table);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);

    for (std::uint32_t strip = 0; strip < window.total; ++strip) {
        StreamKeyStrip(engine, cursor, strip);
        cursor.Advance();
    }

    // The AIC broadcasts one release per strip and this core consumed one per
    // strip from kStageDepth on, so min(kStageDepth, total) releases are still
    // standing. Consume them: a cross-core counter left non-empty at kernel
    // exit leaks into the next launch.
    const std::uint32_t standing =
        window.total < kStageDepth ? window.total : kStageDepth;
    for (std::uint32_t i = 0; i < standing; ++i) {
        AscendC::CrossCoreWaitFlag<2, kSync2Aiv1Wait>(kSync2StepRetired);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtStreamDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtStreamDrain);
    }

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
}

// ------------------------------------------------------------------- AIV0
__aicore__ inline void StreamLoadValueStrip(ValueEngine& engine,
                                            StripCursor& cursor,
                                            std::uint32_t scale_plane,
                                            std::uint32_t strip)
{
    const std::uint32_t slot = SlotOfStrip(strip);
    const std::int32_t  evt  = static_cast<std::int32_t>(slot);

    AscendC::LocalTensor<std::int8_t> packed{kUb,     PackedVSlotOf(slot),                  kPackedBytes};
    AscendC::LocalTensor<float>       key_scale{kUb,  ScaleSlotOf(slot),                    kTokens};
    AscendC::LocalTensor<float>       value_scale{kUb, ScaleSlotOf(slot) + kTokens * kF32,  kTokens};

    const std::uint32_t packed_base = cursor.PackedBytes();
    const std::uint32_t scale_base  = cursor.ScaleElems();

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt);
    AscendC::DataCopy(packed,      engine.value_cache[packed_base], kPackedBytes);
    AscendC::DataCopy(key_scale,   engine.scale_cache[scale_base],  kTokens);
    AscendC::DataCopy(value_scale,
                      engine.scale_cache[scale_plane + scale_base], kTokens);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evt);
}

__aicore__ inline void StreamDequantValueStrip(std::uint32_t strip)
{
    const std::uint32_t slot = SlotOfStrip(strip);
    const std::int32_t  evt  = static_cast<std::int32_t>(slot);

    ascend_micro::NativeDequant4Engine<float> dequant_engine;
    dequant_engine.packed         = {kUb, PackedVSlotOf(slot), kPackedBytes};
    dequant_engine.signed_nibbles = {kUb, kUbDqSigned,         kPanel};
    dequant_engine.output         = {kUb, ValueSlotOf(slot),   kPanel};

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evt);
    ascend_micro::turboquant_dequant4_signed(dequant_engine, kPanel);
}

// The engine is built HERE, per strip, and deliberately NOT hoisted: holding
// it across the loop was measured 2.1x slower, because its operands stop being
// immediates the compiler can fold and become memory the body reloads sixteen
// times. reports/decode_splitk_aic_scalar_issue.md has the measurement.
//
// `active` IS handed in, from the caller's running token base, so the body
// does no division.
__aicore__ inline void StreamRecurStrip(std::uint32_t strip, std::uint32_t active,
                                        std::uint32_t total)
{
    const std::uint32_t slot = SlotOfStrip(strip);
    const std::int32_t  evt  = static_cast<std::int32_t>(slot);

    ascend_micro::NativeDecodeRecurrenceEngine recur_engine;
    FillRecurEngine(recur_engine, slot);

    AscendC::LocalTensor<float> archived{kUb, ArchiveSlotOf(ArchiveOfStrip(strip)), kScores};

    // RAW on the score panel: the AIC's release for THIS strip means its
    // Fixpipe has landed in this slot.
    //
    // No scalar drain behind this wait, unlike AIV1's. The wait names PIPE_V
    // and the recurrence is V work, so the V queue already orders the two; a
    // drain here would only convert the wait into a stall on instruction
    // ISSUE, which held this core's next MTE2 ingress behind a dependency it
    // does not have. On AIV1 that conversion is load-bearing -- it bounds the
    // core whose run-ahead was the defect, and it is the only thing
    // scripts/audit_mode4_tick_order.py can read a stall from -- so AIV1 keeps
    // its drain and this one is gone. Proved safe in the token model first:
    // tests/cpu/test_decode_page_loop_model.py's `aiv0_issue_fence` control is
    // this edge put back, and it must still be ACCEPTED.
    AscendC::CrossCoreWaitFlag<2, kSync2Aiv0Wait>(kSync2StepRetired);

    if (active == kTokens) {
        ascend_micro::vector_decode_recur_strip(recur_engine);
    } else if (active != 0u) {
        ascend_micro::vector_decode_recur_strip_masked(recur_engine, active);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(archived, recur_engine.panel, 1.0f, static_cast<std::int32_t>(kScores));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evt);

    // The score slot is free again. Folded into the readiness quorum rather
    // than given its own channel: the AIC's (strip + kStageDepth)-th GEMM is
    // the one that overwrites this panel, so this IS the announcement its wait
    // needs. Past that point there is no further GEMM to release, and
    // announcing anyway would leave a token on the channel.
    //
    // UNFENCED, ON INSTRUCTION, AND UNDER TEST. Section 6.2a requires a V_S
    // drain here, because a cross-core set issues on the scalar unit and the
    // `pipe` template argument does not order it -- in this operator's own
    // trace the mode-2 set appears on FLOWCTRL, exactly where the mode-4 set
    // appears. The drain was removed on the claim that the mode-2 quorum
    // guarantees visibility at the CCU and makes it redundant.
    //
    // That claim is not settled here. What is at risk if it is wrong: this
    // announcement means "the score panel of strip t is free, I have read
    // it", so an announcement that overtakes the recurrence lets the Cube's
    // Fixpipe for strip t + kStageDepth overwrite a panel still being read.
    // The evidence that decides it is the accumulator against the FP64 oracle
    // on every rung, plus scripts/analyze_announce_overtake.py over the tick
    // order. scripts/audit_cross_core_fences.py reports this as unfenced,
    // which is true and deliberate; see
    // reports/decode_splitk_aic_scalar_issue.md.
    if (strip + kStageDepth < total) {
        AscendC::CrossCoreSetFlag<2, kSync2Aiv0Announce>(kSync2StageReady);
    }
}

__aicore__ inline void stream_value_core(__gm__ std::int8_t*   gm_value_cache,
                                         __gm__ float*         gm_scale_cache,
                                         __gm__ std::uint32_t* gm_table,
                                         __gm__ float*         gm_partition,
                                         const StripWindow&    window)
{
    ValueEngine engine;
    engine.value_cache.SetGlobalBuffer(gm_value_cache);
    engine.scale_cache.SetGlobalBuffer(gm_scale_cache);

    // One cursor for the ingress walk; the recurrence walks the same sequence
    // kPipelineLag columns later, so it carries its own running token base
    // below rather than sharing this one.
    StripCursor ingress;
    ingress.Begin(window, gm_table);

    AscendC::LocalTensor<float> m_prev{kUb, kUbMPrev, kTokens * kStateLanes};
    AscendC::LocalTensor<float> l_prev{kUb, kUbLPrev, kTokens * kStateLanes};
    AscendC::LocalTensor<float> acc{kUb,    kUbAcc,   kTokens * kHeadSize};

    // Seeded ONCE for the whole window. The per-page entry re-entered
    // run_value_core per page with `initialize` true only on the first; a flat
    // stream has one entry, so the carried state is seeded once and every
    // recurrence folds into the same fixed UB offsets in strip order.
    AscendC::Duplicate(m_prev, -3.0e38f, static_cast<std::int32_t>(kTokens * kStateLanes));
    AscendC::Duplicate(l_prev, 0.0f,     static_cast<std::int32_t>(kTokens * kStateLanes));
    AscendC::Duplicate(acc,    0.0f,     static_cast<std::int32_t>(kTokens * kHeadSize));

    if (window.total != 0u) {
        // The tail mask, primed once per core rather than once per strip.
        // routine_vector_decode_recurrence.hpp says so of `prime_tail_mask`:
        // "primed once per kernel, before the first strip. Reads nothing that
        // changes, so it is hoisted out of the strip loop entirely" -- and
        // calling it per strip was against that contract. The engine built
        // here exists only to prime through: the four mask operands are at
        // fixed UB addresses that every engine shares, so priming through any
        // one of them primes all of them. The engines the strip bodies use are
        // built inline, per strip, for the reason StreamRecurStrip records.
        {
            ascend_micro::NativeDecodeRecurrenceEngine priming_engine;
            FillRecurEngine(priming_engine, 0u);
            priming_engine.prime_tail_mask(turboquant_decode_tail::kMaskedScore);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);

        // Priming announcements: the kStageDepth UB score slots start empty, so
        // the AIC's first kStageDepth GEMMs have nothing to wait for on this
        // side. A priming set reports no earlier pipe work and needs no drain
        // (MICROROUTINE_DESIGN_GUIDELINES.md section 6.2a).
        const std::uint32_t primes =
            window.total < kStageDepth ? window.total : kStageDepth;
        for (std::uint32_t i = 0; i < primes; ++i) {
            AscendC::CrossCoreSetFlag<2, kSync2Aiv0Announce>(kSync2StageReadyPrime);
        }

        // The recurrence's own token base, advanced once per recurrence. The
        // tail mask rides this counter instead of a division, for the reason
        // StripCursor records: strips_per_page * kTokens == block_size, so the
        // base rises by kTokens per strip across a page boundary too.
        std::uint64_t recur_token_base =
            static_cast<std::uint64_t>(window.page_begin) * window.block_size;

        for (std::uint32_t column = 0; column < window.total + kPipelineLag; ++column) {
            if (column >= kPipelineLag && column - kPipelineLag < window.total) {
                const std::uint32_t strip = column - kPipelineLag;
                const std::uint32_t active = turboquant_decode_tail::ActiveInStrip(
                    recur_token_base, window.live_seq);
                StreamRecurStrip(strip, active, window.total);
                recur_token_base += kTokens;
            }
            if (column >= 3u && column - 3u < window.total) {
                StreamDequantValueStrip(column - 3u);
            }
            if (column >= 2u && column - 2u < window.total) {
                StreamLoadValueStrip(engine, ingress, window.scale_plane,
                                     column - 2u);
                ingress.Advance();
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
    }

    AscendC::GlobalTensor<float> partition;
    partition.SetGlobalBuffer(gm_partition);
    constexpr std::uint32_t kRecord = turboquant_splitk::kRecordFloats;

    AscendC::DataCopyParams lane;
    lane.blockCount = static_cast<std::uint16_t>(kTokens);
    lane.blockLen   = static_cast<std::uint16_t>(kStateLanes * kF32 / kBlock);
    lane.srcStride  = 0;
    lane.dstStride  = static_cast<std::uint16_t>((kRecord - kStateLanes) * kF32 / kBlock);

    AscendC::DataCopyParams band;
    band.blockCount = static_cast<std::uint16_t>(kTokens);
    band.blockLen   = static_cast<std::uint16_t>(kHeadSize * kF32 / kBlock);
    band.srcStride  = 0;
    band.dstStride  = static_cast<std::uint16_t>((kRecord - kHeadSize) * kF32 / kBlock);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStatic);
    AscendC::DataCopy(partition,                   m_prev, lane);
    AscendC::DataCopy(partition[kStateLanes],      l_prev, lane);
    AscendC::DataCopy(partition[2u * kStateLanes], acc,    band);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtStatic);
}

// -------------------------------------------------------------------- AIC
__aicore__ inline void StreamGemmStrip(CubeEngine& engine, std::uint32_t strip)
{
    const std::uint32_t slot = SlotOfStrip(strip);
    const std::int32_t  evt  = static_cast<std::int32_t>(slot);

    auto* l0b_slot   = engine.l0_table  + slot * kHeadSize * kTokens;
    auto* l0c_window = engine.l0_scores + slot * kScores;
    auto* ub_scores  = reinterpret_cast<__ubuf__ float*>(ScoreSlotOf(slot));

    // ONE quorum wait, for both of this GEMM's cross-core preconditions: the L1
    // key page holds THIS strip's keys (AIV1) and the UB score slot is free
    // (AIV0). Gates PIPE_MTE1, the pipe that reads L1; the Fixpipe is ordered
    // behind it transitively through MTE1_M and M_FIX. MTE1 has no MTE1_S
    // route, so the drain rides M (section 6.2a).
    AscendC::CrossCoreWaitFlag<2, kSync2AicWait>(kSync2StageReady);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtStreamDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtStreamDrain);
    AscendC::SetFlag<AscendC::HardEvent::M_S>(kEvtStreamDrain);
    AscendC::WaitFlag<AscendC::HardEvent::M_S>(kEvtStreamDrain);

    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(evt);
    ascend_micro::score_strip_stage_l0(L1KeyPageOf(slot), l0b_slot);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(evt);

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(evt);
    ascend_micro::score_strip_compute(engine.l0_query, l0b_slot, l0c_window);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(evt);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(evt);
    ascend_micro::score_strip_fixpipe(l0c_window, ub_scores, false);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(evt);

    // The release: this strip's L1 key page has been read into L0B and its
    // scores have landed in UB. MTE1 drains through M, then FIX, so the pipe
    // the announcement is ISSUED on is the last one drained -- which is both
    // section 6.2a's requirement and what section 6.6's audit checks.
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtStreamDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtStreamDrain);
    AscendC::SetFlag<AscendC::HardEvent::M_S>(kEvtStreamDrain);
    AscendC::WaitFlag<AscendC::HardEvent::M_S>(kEvtStreamDrain);
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtStreamDrain);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtStreamDrain);
    AscendC::CrossCoreSetFlag<2, kSync2AicAnnounce>(kSync2StepRetired);
}

__aicore__ inline void stream_cube_core(__gm__ bfloat16_t* gm_q,
                                        __gm__ bfloat16_t* gm_pi_table,
                                        __gm__ bfloat16_t* gm_out_qrot,
                                        std::uint32_t      q_base,
                                        std::uint32_t      total)
{
    if (total == 0u) return;
    using namespace AscendC::Te;

    __ca__ bfloat16_t l0_query[kPanel]       __attribute__((aligned(512)));
    __cb__ bfloat16_t l0_table[kTableElems]  __attribute__((aligned(512)));
    __cc__ float      l0_scores[kPanel]      __attribute__((aligned(1024)));

    static_assert(kStageDepth * kHeadSize * kTokens <= kTableElems, "the key slots must fit inside the L0B table allocation");
    static_assert(kStageDepth * kScores             <= kPanel,      "the score windows must fit inside the L0C panel allocation");

    CubeEngine engine{l0_query, l0_table, l0_scores};

    CubeRotatePrologue(gm_q, gm_pi_table, gm_out_qrot, q_base, engine);

    // Primed once for the whole stream, drained once after it: no per-page
    // cascade, which is what used to have to be got right at every boundary.
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);

    for (std::uint32_t strip = 0; strip < total; ++strip) {
        StreamGemmStrip(engine, strip);
    }

    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
}

}  // namespace paged_decode

extern "C" __global__ __aicore__ void camodel_paged_decode_attention(
    __gm__ bfloat16_t*   q,
    __gm__ bfloat16_t*   pi_table,
    __gm__ std::int8_t*  key_cache,
    __gm__ std::int8_t*  value_cache,
    __gm__ float*        scale_cache,
    __gm__ std::uint32_t* block_table,
    __gm__ float*        gate,
    __gm__ std::uint32_t* gather_offsets,
    __gm__ float*        gather_signs,
    __gm__ float*        pi_sign,
    __gm__ bfloat16_t*   out_qrot,
    __gm__ float*        out_scores,
    __gm__ float*        out_state,
    __gm__ float*        out_acc,
    __gm__ bfloat16_t*   out_context,
    __gm__ float*        out_rotated,
    uint32_t             strips,
    uint32_t             strip_begin,
    uint32_t             active_block)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using namespace paged_decode;

    if (strips == 0u || strips > kStrips || strip_begin + strips > kStrips) return;

    const std::uint32_t block = static_cast<std::uint32_t>(AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());
    if (block >= kGrid) return;
    if (active_block < kGrid && block != active_block) return;

    const std::uint32_t kv_head  = block / kTemporalBlocks;
    const std::uint32_t temporal = block % kTemporalBlocks;

    AscendC::GlobalTensor<std::uint32_t> table;
    table.SetGlobalBuffer(block_table);
    const std::uint32_t page = table.GetValue(temporal);

    const std::uint32_t strip_base_bytes = ((page * kKvHeads) + kv_head) * kStrips * kPackedBytes;
    const std::uint32_t scale_plane      = kNumPages * kKvHeads * kBlockSize;
    const std::uint32_t scale_base       = ((page * kKvHeads) + kv_head) * kBlockSize + strip_begin * kTokens;
    const std::uint32_t q_base           = kv_head * kPanel;
    const std::uint32_t out_base         = block * kPanel;

    if ASCEND_IS_AIC {
        run_cube_core(q, pi_table, out_qrot + out_base, q_base, strips);
    } else if (AscendC::GetSubBlockIdx() == 0) {
        run_value_core(value_cache, scale_cache, gate, gather_offsets, gather_signs, pi_sign,
                       out_scores + block * kStrips * kScores,
                       out_state  + block * 2u * kTokens * kStateLanes,
                       out_acc + out_base, nullptr, out_context + out_base, out_rotated + out_base,
                       strip_base_bytes + strip_begin * kPackedBytes, scale_base, scale_plane + scale_base,
                       q_base, strips, strips * kTokens);
    } else {
        run_key_core(key_cache, strip_base_bytes + strip_begin * kPackedBytes, strips);
    }
}

extern "C" __global__ __aicore__ void turboquant_paged_decode_splitk_aic(
    __gm__ bfloat16_t*   q,
    __gm__ bfloat16_t*   pi_table,
    __gm__ std::int8_t*  key,
    __gm__ std::int8_t*  value,
    __gm__ float*        scales,
    __gm__ uint32_t*     table,
    __gm__ bfloat16_t*   qrot,
    __gm__ float*        workspace,
    uint32_t             requests,
    uint32_t             heads,
    uint32_t             block_size,
    uint32_t             cache_pages,
    uint32_t             seq_pages,
    uint32_t             splits,
    uint32_t             page_begin,
    uint32_t             page_count,
    uint32_t             seq_len)
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
    const uint32_t request   = head_item / heads;
    const uint32_t head      = head_item % heads;
    const auto     range     = turboquant_splitk::Partition(page_count, splits, item % splits);
    const uint32_t strips    = block_size / kTokens;
    const uint32_t live_seq  = seq_len == 0u ? seq_pages * block_size : seq_len;

    if (!turboquant_decode_tail::SeqLenInCapacity(live_seq, cache_pages, block_size)) return;

    // One flat strip stream for the whole window, page boundaries and all. The
    // three cores derive the same `total` from the same geometry, which is what
    // makes their per-strip rendezvous line up by ordinal: strip n's readiness
    // quorum is AIV0's n-th announcement and AIV1's n-th, and strip n's release
    // is the AIC's n-th.
    StripWindow window;
    window.strips_per_page = strips;
    window.page_begin      = page_begin + range.begin;
    window.heads           = heads;
    window.head            = head;
    window.block_size      = block_size;
    window.table_base      = request * seq_pages;
    window.scale_plane     = cache_pages * heads * block_size;
    window.live_seq        = live_seq;
    window.total           = range.count * strips;

    if ASCEND_IS_AIC {
        stream_cube_core(q, pi_table, qrot + item * kPanel, head_item * kPanel,
                         window.total);
    } else if (AscendC::GetSubBlockIdx() == 0) {
        // A worker with no live page still runs, with total = 0: it writes the
        // neutral element (m = -3e38, l = 0, Acc = 0) so the Split-K reduction
        // has a record from every worker. It takes part in no rendezvous, and
        // neither does the Cube, which returns before priming anything.
        stream_value_core(value, scales, table,
                          workspace + static_cast<std::uint64_t>(item) *
                              turboquant_splitk::kPartitionFloats,
                          window);
    } else {
        stream_key_core(key, table, window);
    }
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

#include <cassert>
#include <stdexcept>

#include "turboquant_decode_tail.hpp"

extern "C" __global__ __aicore__ void camodel_paged_decode_attention(
    __gm__ bfloat16_t*   q,               /* [M=16, d=256] Query token vectors in Global Memory */
    __gm__ bfloat16_t*   pi_table,        /* [16 blocks of 32x32] Precomputed Pi rotation table for Cube GEMM */
    __gm__ std::int8_t*  key_cache,       /* [num_blocks, heads, 8 strips, 2048 B] Packed int4 Key cache */
    __gm__ std::int8_t*  value_cache,     /* [num_blocks, heads, 8 strips, 2048 B] Packed int4 Value cache */
    __gm__ float*        scale_cache,     /* [2 planar strides] FP32 per-token scales: plane 0 key, plane 1 value */
    __gm__ std::uint32_t* block_table,    /* [temporal_blocks] Logical-to-physical page translation table */
    __gm__ float*        gate,            /* [M=16, d=256] Projection gate vector for fused egress sigmoid */
    __gm__ std::uint32_t* gather_offsets, /* [3 stages * d=256] Gather permutation indices for egress rotation */
    __gm__ float*        gather_signs,    /* [3 stages * d=256] Gather sign flip vectors for egress rotation */
    __gm__ float*        pi_sign,         /* [d=256] Final sign vector for egress rotation (Acc . Pi) */
    __gm__ bfloat16_t*   out_qrot,        /* [Diagnostic] Rotated Q after Cube prologue (bit-exact check) */
    __gm__ float*        out_scores,      /* [Diagnostic] Raw score panels of all 8 strips straight from Fixpipe */
    __gm__ float*        out_state,       /* [Diagnostic] Intermediate online softmax state buffers (m, l) */
    __gm__ float*        out_acc,         /* [Diagnostic] Raw FP32 accumulator before egress rotation */
    __gm__ bfloat16_t*   out_context,     /* [Output] Final fused egress: (Acc . Pi) * sigmoid(gate) */
    __gm__ float*        out_rotated,     /* [Diagnostic] Rotated context rows prior to sigmoid gating */
    uint32_t             strips,          /* Active strip window count within the page (1..8) */
    uint32_t             strip_begin,     /* Starting strip index inside the 128-token page (0..7) */
    uint32_t             active_block     /* Single-tile isolation block ID for canary tests (or kGrid=32 for all) */
);

extern "C" void launch_camodel_paged_decode_attention(
    void*          stream,
    std::uint16_t* q,
    std::uint16_t* pi_table,
    std::int8_t*   key_cache,
    std::int8_t*   value_cache,
    float*         scale_cache,
    std::uint32_t* block_table,
    float*         gate,
    std::uint32_t* gather_offsets,
    float*         gather_signs,
    float*         pi_sign,
    std::uint16_t* out_qrot,
    float*         out_scores,
    float*         out_state,
    float*         out_acc,
    std::uint16_t* out_context,
    float*         out_rotated,
    std::uint32_t  strips,
    std::uint32_t  strip_begin,
    std::uint32_t  active_block)
{
    assert(strips >= 1u && strips <= paged_decode::kStrips);
    if (strips < 1u || strips > paged_decode::kStrips || strip_begin + strips > paged_decode::kStrips) {
        throw std::invalid_argument("paged decode strips must select a window of 0..8");
    }
    camodel_paged_decode_attention<<<paged_decode::kGrid, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(q),
        reinterpret_cast<bfloat16_t*>(pi_table),
        key_cache, value_cache, scale_cache, block_table, gate, gather_offsets,
        gather_signs, pi_sign, reinterpret_cast<bfloat16_t*>(out_qrot), out_scores,
        out_state, out_acc, reinterpret_cast<bfloat16_t*>(out_context), out_rotated,
        strips, strip_begin, active_block);
}

extern "C" __global__ __aicore__ void turboquant_paged_decode_splitk_aic(
    __gm__ bfloat16_t*   q,           /* [requests * heads, 16, 256] Batched Query token rows in GM */
    __gm__ bfloat16_t*   pi_table,    /* [16 blocks of 32x32] Pi rotation matrix for TurboQuant GEMM */
    __gm__ std::int8_t*  key,         /* Packed int4 KV Key cache pool in Global Memory */
    __gm__ std::int8_t*  value,       /* Packed int4 KV Value cache pool in Global Memory */
    __gm__ float*        scales,      /* Planar FP32 dequantization scale cache (plane 0: K, plane 1: V) */
    __gm__ uint32_t*     table,       /* [requests * seq_pages] Block table mapping sequence pages to physical cache */
    __gm__ bfloat16_t*   qrot,        /* [requests * heads * splits, 16, 256] Staged rotated query buffer in GM */
    __gm__ float*        workspace,   /* Global workspace for strided Split-K partial records: [m(8) | l(8) | acc(256)] */
    uint32_t             requests,    /* Batch size (number of active concurrent decode requests) */
    uint32_t             heads,       /* Number of KV attention heads */
    uint32_t             block_size,  /* Page capacity in tokens (typically 128 = 8 strips of 16) */
    uint32_t             cache_pages, /* Total physical page capacity in the allocated KV cache pool */
    uint32_t             seq_pages,   /* Maximum logical page count allocated per sequence */
    uint32_t             splits,      /* Split-K parallelism factor along the temporal page axis */
    uint32_t             page_begin,  /* Logical starting page index of the processing window */
    uint32_t             page_count,  /* Number of logical pages assigned to this Split-K task */
    uint32_t             seq_len      /* Total live sequence length in tokens for tail masking (0 = full capacity) */
);

extern "C" void launch_turboquant_paged_decode_splitk_aic(
    void*          stream,
    std::uint16_t* q,
    std::uint16_t* pi_table,
    std::int8_t*   key,
    std::int8_t*   value,
    float*         scales,
    std::uint32_t* table,
    std::uint16_t* qrot,
    float*         workspace,
    std::uint32_t  requests,
    std::uint32_t  heads,
    std::uint32_t  block_size,
    std::uint32_t  cache_pages,
    std::uint32_t  seq_pages,
    std::uint32_t  splits,
    std::uint32_t  page_begin,
    std::uint32_t  page_count,
    std::uint32_t  seq_len)
{
    if (!requests || !heads || !splits || splits > 16 || !seq_pages ||
        !block_size || block_size > paged_decode::kBlockSize || block_size % 16 ||
        seq_pages > cache_pages || page_begin > seq_pages || page_count > seq_pages - page_begin) {
        throw std::invalid_argument("invalid AIC temporal split geometry");
    }
    if (seq_len != 0u) {
        if (!turboquant_decode_tail::SeqLenInCapacity(seq_len, cache_pages, block_size)) {
            throw std::invalid_argument("paged_decode_splitk_aic: seqLen exceeds the allocated cache");
        }
        if (!turboquant_decode_tail::WindowCoversSeqLen(seq_len, page_begin, page_count, block_size, 0u, block_size / 16u)) {
            throw std::invalid_argument("paged_decode_splitk_aic: seqLen has live tokens past the end of the page window");
        }
    }
    turboquant_paged_decode_splitk_aic<<<requests * heads * splits, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(q),
        reinterpret_cast<bfloat16_t*>(pi_table),
        key, value, scales, table, reinterpret_cast<bfloat16_t*>(qrot), workspace,
        requests, heads, block_size, cache_pages, seq_pages, splits, page_begin,
        page_count, seq_len);
}

#endif