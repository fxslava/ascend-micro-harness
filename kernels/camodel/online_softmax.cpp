#include <cstdint>

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "../../routines/routine_vector_online_softmax.hpp"

namespace {

// UB layout, rule AKA3003
// -----------------------
// Every LocalTensor base is 32-byte aligned because that is the smallest
// addressable UB quantum for a vector operand. Slots are separated by a
// kBankSkew gap so that two live operands never land on the same UB bank
// column, which would serialise their load ports.
//
// A level-2 continuous vector op on dav-3510 issues an *unmasked*
// Reg::LoadAlign of one whole vector register from each source base before the
// mask is applied to the ALU result. The read footprint of a source operand is
// therefore kVecReg bytes, not count*sizeof(T). Stores are masked and so are
// exact. kArenaEnd + kVecReg is asserted to stay inside the UB window so that
// the over-read of the last operand cannot leave UB.
constexpr std::uint32_t kBlock = 32;
constexpr std::uint32_t kBankSkew = 32;
constexpr std::uint32_t kVecReg = 256;  // VECTOR_REG_WIDTH on dav-3510
constexpr std::uint32_t kUbWindow = 192u * 1024u;

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
// Next 32-byte aligned slot after a tensor of prev_bytes at prev_base, plus
// one block of bank skew.
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkew;
}

constexpr std::uint32_t kTile = ascend_micro::kOnlineTile;            // 16
constexpr std::uint32_t kState = ascend_micro::kStateLanes;           // 8
constexpr std::uint32_t kBcast = ascend_micro::kBroadcastLanes;       // 64
constexpr std::uint32_t kPanel = kTile * kTile;                       // 256

constexpr std::uint32_t kTileBytes = kTile * sizeof(float);           // 64
constexpr std::uint32_t kStateBytes = kState * sizeof(float);         // 32
constexpr std::uint32_t kBcastBytes = kBcast * sizeof(float);         // 256
constexpr std::uint32_t kPanelBytes = kPanel * sizeof(float);         // 1024

// Tile inputs.
constexpr std::uint32_t kScores = 0;
constexpr std::uint32_t kVByChannel = NextSlot(kScores, kTileBytes);
constexpr std::uint32_t kOffsets = NextSlot(kVByChannel, kPanelBytes);

// Carried state, replicated 32-byte blocks.
constexpr std::uint32_t kMPrev = NextSlot(kOffsets, kPanelBytes);
constexpr std::uint32_t kLPrev = NextSlot(kMPrev, kStateBytes);
constexpr std::uint32_t kMNew = NextSlot(kLPrev, kStateBytes);
constexpr std::uint32_t kLNew = NextSlot(kMNew, kStateBytes);
constexpr std::uint32_t kAlphaDelta = NextSlot(kLNew, kStateBytes);
constexpr std::uint32_t kAlpha = NextSlot(kAlphaDelta, kStateBytes);
constexpr std::uint32_t kLScaled = NextSlot(kAlpha, kStateBytes);

// Reduction destinations, lane 0 only.
constexpr std::uint32_t kTileMaxRaw = NextSlot(kLScaled, kStateBytes);
constexpr std::uint32_t kPSumRaw = NextSlot(kTileMaxRaw, kStateBytes);

// Accumulator lanes.
constexpr std::uint32_t kAccPrev = NextSlot(kPSumRaw, kStateBytes);
constexpr std::uint32_t kAccNew = NextSlot(kAccPrev, kTileBytes);
constexpr std::uint32_t kAccScaled = NextSlot(kAccNew, kTileBytes);

// Tile scratch.
constexpr std::uint32_t kCentered = NextSlot(kAccScaled, kTileBytes);
constexpr std::uint32_t kP = NextSlot(kCentered, kTileBytes);
constexpr std::uint32_t kWeightedSum = NextSlot(kP, kTileBytes);

// Brcb destinations, 256 bytes each.
constexpr std::uint32_t kTileMaxBc = NextSlot(kWeightedSum, kTileBytes);
constexpr std::uint32_t kMBc = NextSlot(kTileMaxBc, kBcastBytes);
constexpr std::uint32_t kPSumBc = NextSlot(kMBc, kBcastBytes);
constexpr std::uint32_t kAlphaBc = NextSlot(kPSumBc, kBcastBytes);

// Panel scratch.
constexpr std::uint32_t kPRepeated = NextSlot(kAlphaBc, kBcastBytes);
constexpr std::uint32_t kWeighted = NextSlot(kPRepeated, kPanelBytes);
constexpr std::uint32_t kArenaEnd = kWeighted + kPanelBytes;

static_assert(kScores % kBlock == 0 && kVByChannel % kBlock == 0 &&
              kOffsets % kBlock == 0 && kMPrev % kBlock == 0 &&
              kLPrev % kBlock == 0 && kMNew % kBlock == 0 &&
              kLNew % kBlock == 0 && kAlphaDelta % kBlock == 0 &&
              kAlpha % kBlock == 0 && kLScaled % kBlock == 0 &&
              kTileMaxRaw % kBlock == 0 && kPSumRaw % kBlock == 0 &&
              kAccPrev % kBlock == 0 && kAccNew % kBlock == 0 &&
              kAccScaled % kBlock == 0 && kCentered % kBlock == 0 &&
              kP % kBlock == 0 && kWeightedSum % kBlock == 0 &&
              kTileMaxBc % kBlock == 0 && kMBc % kBlock == 0 &&
              kPSumBc % kBlock == 0 && kAlphaBc % kBlock == 0 &&
              kPRepeated % kBlock == 0 && kWeighted % kBlock == 0,
              "every UB operand base must be 32-byte aligned");
// Unmasked source over-read of the final operand must stay inside UB.
static_assert(kArenaEnd + kVecReg <= kUbWindow, "UB arena leaves no read pad");
// A Brcb repeat writes 8 DataBlocks; its destination must hold all of them.
static_assert(kBcastBytes == 8u * kBlock, "Brcb destination must be 8 blocks");

}  // namespace

extern "C" __global__ __aicore__ void camodel_online_softmax(
    __gm__ float* scores, __gm__ float* values,
    __gm__ std::uint32_t* offsets, __gm__ float* output)
{
    AscendC::InitSocState();
    constexpr auto ub = AscendC::TPosition::VECCALC;
    ascend_micro::NativeOnlineSoftmaxEngine e;
    e.scores = {ub, kScores, kTile};
    e.v_by_channel = {ub, kVByChannel, kPanel};
    e.p_repeat_offsets = {ub, kOffsets, kPanel};
    e.m_prev = {ub, kMPrev, kState};
    e.l_prev = {ub, kLPrev, kState};
    e.m_new = {ub, kMNew, kState};
    e.l_new = {ub, kLNew, kState};
    e.alpha_delta = {ub, kAlphaDelta, kState};
    e.alpha = {ub, kAlpha, kState};
    e.l_scaled = {ub, kLScaled, kState};
    e.tile_max_raw = {ub, kTileMaxRaw, kState};
    e.p_sum_raw = {ub, kPSumRaw, kState};
    e.acc_prev = {ub, kAccPrev, kTile};
    e.acc_new = {ub, kAccNew, kTile};
    e.acc_scaled = {ub, kAccScaled, kTile};
    e.centered = {ub, kCentered, kTile};
    e.p = {ub, kP, kTile};
    e.weighted_sum = {ub, kWeightedSum, kTile};
    e.tile_max_bc = {ub, kTileMaxBc, kBcast};
    e.m_bc = {ub, kMBc, kBcast};
    e.p_sum_bc = {ub, kPSumBc, kBcast};
    e.alpha_bc = {ub, kAlphaBc, kBcast};
    e.p_repeated = {ub, kPRepeated, kPanel};
    e.weighted = {ub, kWeighted, kPanel};

    AscendC::GlobalTensor<float> gm_scores, gm_values, gm_output;
    AscendC::GlobalTensor<std::uint32_t> gm_offsets;
    gm_scores.SetGlobalBuffer(scores);
    gm_values.SetGlobalBuffer(values);
    gm_offsets.SetGlobalBuffer(offsets);
    gm_output.SetGlobalBuffer(output);

    AscendC::DataCopy(e.p_repeat_offsets, gm_offsets, kPanel);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    // Duplicate from an immediate scalar keeps the value in a GPR and never
    // reads UB, so it carries none of the hazards of the tensor-source form.
    // Seeding all kState lanes establishes the replicated-state invariant.
    AscendC::Duplicate(e.m_prev, 0.0f, static_cast<std::int32_t>(kState));
    AscendC::Duplicate(e.l_prev, 0.0f, static_cast<std::int32_t>(kState));
    AscendC::Duplicate(e.acc_prev, 0.0f, static_cast<std::int32_t>(kTile));
    // scores and v_by_channel are single buffered, so MTE2 must not refill
    // them until the Vector pipe has finished reading the previous tile. That
    // is a V->MTE2 WAR edge and PipeBarrier<PIPE_V> cannot express it: a
    // barrier orders the vector pipe against itself, not against MTE2. Without
    // this handshake MTE2 runs ahead and every tile computes on a later
    // tile's scores. Prime one token so the first iteration finds the slot
    // free; the post-loop wait drains it so no event leaks across launches.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    for (std::uint32_t tile = 0; tile < 4; ++tile) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::DataCopy(e.scores, gm_scores[tile * kTile], kTile);
        AscendC::DataCopy(e.v_by_channel, gm_values[tile * kPanel], kPanel);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        ascend_micro::vector_online_softmax(e);
        AscendC::PipeBarrier<PIPE_V>();
        // Carry the recurrence forward. Adds with an immediate stays in the
        // vector pipe and preserves lane replication.
        AscendC::Adds(e.m_prev, e.m_new, 0.0f, static_cast<std::int32_t>(kState));
        AscendC::Adds(e.l_prev, e.l_new, 0.0f, static_cast<std::int32_t>(kState));
        AscendC::Adds(e.acc_prev, e.acc_new, 0.0f, static_cast<std::int32_t>(kTile));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(gm_output[0], e.m_prev, kState);
    AscendC::DataCopy(gm_output[kState], e.l_prev, kState);
    AscendC::DataCopy(gm_output[2u * kState], e.acc_prev, kTile);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
}
#elif defined(ASCEND_MICRO_CANN_LAUNCH)
extern "C" __global__ __aicore__ void camodel_online_softmax(
    __gm__ float*, __gm__ float*, __gm__ std::uint32_t*, __gm__ float*);
extern "C" void launch_camodel_online_softmax(void* stream, float* scores,
                                                float* values, std::uint32_t* offsets,
                                                float* output)
{
    camodel_online_softmax<<<1, nullptr, stream>>>(scores, values, offsets, output);
}
#endif
