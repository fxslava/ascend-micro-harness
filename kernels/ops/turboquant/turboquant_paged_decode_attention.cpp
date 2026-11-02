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

template <std::uint32_t Strip>
constexpr std::uint32_t kL1KeyPage = kL1KeyRing + Strip * kL1SlotStride;

template <std::uint32_t Slot> constexpr std::uint16_t kFlagValueReady   = 0u + Slot;
template <std::uint32_t Slot> constexpr std::uint16_t kFlagScoreFree    = 4u + Slot;
template <std::uint32_t Slot> constexpr std::uint16_t kFlagScoreReady   = 8u + Slot;
template <std::uint32_t Slot> constexpr std::uint16_t kFlagKeyReady     = 16u + Slot;

constexpr std::uint16_t kFlagPageDrainedV0 = 10u;
constexpr std::uint16_t kFlagPageDrainedV1 = 26u;
constexpr std::int32_t  kEvtPageDrain      = 5;

static_assert(kFlagPageDrainedV0 < 11u && kFlagPageDrainedV1 - 16u < 11u, "page-completion flags must remain in the application ID range");
static_assert(kFlagValueReady<kStageDepth - 1u> < 11u && kFlagScoreFree<kStageDepth - 1u> < 11u && kFlagScoreReady<kStageDepth - 1u> < 11u, "sub block 0 flag ids must stay below the library-reserved 11..14");
static_assert(kFlagKeyReady<kStageDepth - 1u> - 16u < 11u, "sub block 1 flag ids must stay below 11 in their own half");

constexpr std::int32_t kEvtStatic = 3;

template <std::uint32_t Column>
constexpr std::uint32_t kSlotOf = Column % kStageDepth;
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

template <std::uint32_t Slot, std::uint32_t Strip>
__aicore__ inline void RecurStrip(ValueEngine&, std::uint32_t active)
{
    static_assert(Slot < kStageDepth && Strip < kStrips);
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);

    ascend_micro::NativeDecodeRecurrenceEngine recur_engine;
    recur_engine.position     = kUb;
    recur_engine.panel_base   = kScoreSlot<Slot>;
    recur_engine.m_base       = kUbMPrev;
    recur_engine.l_base       = kUbLPrev;
    recur_engine.acc_base     = kUbAcc;
    recur_engine.panel        = {kUb, kScoreSlot<Slot>,                   kScores};
    recur_engine.key_scale    = {kUb, kScaleSlot<Slot>,                   kTokens};
    recur_engine.value_scale  = {kUb, kScaleSlot<Slot> + kTokens * kF32,  kTokens};
    recur_engine.value_page   = {kUb, kValueSlot<Slot>,                   kPanel};
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

__aicore__ inline void run_cube_core(__gm__ bfloat16_t* gm_q,
                                     __gm__ bfloat16_t* gm_pi_table,
                                     __gm__ bfloat16_t* gm_out_qrot,
                                     std::uint32_t      q_base,
                                     std::uint32_t      strips,
                                     std::uint32_t      pages    = 1u,
                                     bool               temporal = false)
{
    if (pages == 0u) return;
    using namespace AscendC::Te;

    __ca__ bfloat16_t l0_query[kPanel]       __attribute__((aligned(512)));
    __cb__ bfloat16_t l0_table[kTableElems]  __attribute__((aligned(512)));
    __cc__ float      l0_scores[kPanel]      __attribute__((aligned(1024)));

    static_assert(kStageDepth * kHeadSize * kTokens <= kTableElems, "the key slots must fit inside the L0B table allocation");
    static_assert(kStageDepth * kScores             <= kPanel,      "the score windows must fit inside the L0C panel allocation");

    CubeEngine engine{l0_query, l0_table, l0_scores};

    ascend_micro::stage_page_nz<kTokens, kHeadSize>(gm_q + q_base, kL1Q, kHeadSize);
    ascend_micro::stage_blocks_nz<kTableBlocks, kGroup>(gm_pi_table, kL1PiTable);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtStatic);

    ascend_micro::rotate_query_stage_l0(kL1PiTable, kL1Q, l0_query, l0_table);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtStatic);

    ascend_micro::rotate_query_compute(l0_query, l0_table, l0_scores);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEvtStatic);

    ascend_micro::rotate_query_to_gm(gm_out_qrot, l0_scores);
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtStatic);

    ascend_micro::stage_page_nz<kTokens, kHeadSize>(gm_out_qrot, kL1QRot, kHeadSize);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEvtStatic);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtStatic);

    ascend_micro::load_query_rotated(kL1QRot, l0_query);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtStatic);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtStatic);

    for (std::uint32_t page = 0; page < pages; ++page) {
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
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtPageDrain);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtPageDrain);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtPageDrain);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtPageDrain);
            AscendC::SetFlag<AscendC::HardEvent::M_S>(kEvtPageDrain);
            AscendC::WaitFlag<AscendC::HardEvent::M_S>(kEvtPageDrain);
            AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagPageDrainedV0);
            AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagPageDrainedV1);
        }
    }
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

    if ASCEND_IS_AIC {
        run_cube_core(q, pi_table, qrot + item * kPanel, head_item * kPanel, strips, range.count, true);
    } else if (AscendC::GetSubBlockIdx() == 0) {
        const uint32_t iterations = range.count == 0u ? 1u : range.count;
        for (uint32_t p = 0; p < iterations; ++p) {
            const uint32_t physical = range.count == 0u ? 0u : table[request * seq_pages + page_begin + range.begin + p];
            const uint32_t base     = (physical * heads + head) * block_size;
            const uint32_t live_tokens = range.count == 0u ? 0u :
                turboquant_decode_tail::ActiveInPage(page_begin + range.begin + p, block_size, live_seq);

            run_value_core(value, scales, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                workspace + static_cast<std::uint64_t>(item) * turboquant_splitk::kPartitionFloats,
                nullptr, nullptr, base * kHeadSize / 2u, base, cache_pages * heads * block_size + base, 0u,
                range.count == 0u ? 0u : strips, live_tokens, true, p == 0u, p + 1u == iterations);

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
            const uint32_t base     = (physical * heads + head) * block_size * kHeadSize / 2u;
            run_key_core(key, base, strips);

            AscendC::CrossCoreWaitFlag<4, PIPE_V>(kFlagPageDrainedV1);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtPageDrain);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtPageDrain);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtPageDrain);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtPageDrain);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(kEvtPageDrain);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(kEvtPageDrain);
        }
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