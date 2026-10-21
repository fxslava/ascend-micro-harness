// Fused TurboQuant paged-attention boundary kernel, Ascend 950PR / dav-3510.
//
// Integrates the four verified micro-routine leaves into one MIX-mode kernel
// over kTiles key pages:
//
//   Ingress    routine_turboquant_dequant4.hpp     MTE2 -> UB  (AIV)
//   GEMM       routine_cube_matmul.hpp             L1 -> L0 -> L0C -> UB (AIC)
//   Recurrence routine_vector_online_softmax.hpp   pure Vector (AIV)
//   Outflow    routine_vector_normalize_store.hpp  once at sequence end (AIV)
//
// Geometry. Q carries kDim query rows so the Cube operand geometry matches the
// verified 16x16 engine exactly. The online-softmax recurrence and the outflow
// are per-query leaves, so they are instantiated for query row 0; the Cube
// still produces all kDim rows and the runner checks every one of them against
// the oracle, which is what verifies the dequant + Cube path.
//
// Why K round-trips through GM. The Te copy surface has no UB->L1 transfer
// (CopyGM2L1, CopyGM2UB, CopyL12L0A/B, CopyL12UB, CopyL0C2UB/L1/GM, CopyUB2GM
// and no CopyUB2L1). Dequantisation is Vector work, so it lands in UB, while
// the Cube B operand must reach L1 in ZN. The only available path is therefore
// UB -> GM workspace -> L1(ZN), which is also how a production paged-attention
// kernel stages a dequantised page when the fractal conversion has to happen
// in the GM->L1 converter.
//
// Host-side packing contract (both layouts are the caller's choice, which keeps
// the dequant leaf a pure 1-D vector routine):
//   k_packed nibble index (d * kKeys + key)  = quantised K[key][d]
//     so the staged FP16 page is ND [d][key] == K^T, and S = Q * K^T.
//   v_packed nibble index (ch * kKeys + key) = quantised V[key][ch]
//     which is the channel-major order update_accumulator expects for
//     v_by_channel.

#include <cstdint>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "../../routines/routine_turboquant_dequant4.hpp"
#include "../../routines/routine_cube_matmul.hpp"
#include "../../routines/routine_vector_online_softmax.hpp"
#include "../../routines/routine_vector_normalize_store.hpp"
#endif

namespace fused_attention {
constexpr std::uint32_t kTiles = 3;
constexpr std::uint32_t kDim = 16;    // head dim, channel count, and Cube M
constexpr std::uint32_t kKeys = 16;   // keys per page
constexpr std::uint32_t kPanel = kDim * kKeys;
constexpr std::uint32_t kPackedBytes = kPanel / 2;  // two nibbles per byte
}  // namespace fused_attention

#if defined(__CCE_AICORE__)

namespace fused_attention {

// UB layout, rule AKA3003. Every base is 32-byte aligned with one DataBlock of
// bank skew between live slots, and the arena keeps a full VECTOR_REG_WIDTH of
// addressable UB behind the last operand because level-2 vector ops read a
// whole vector register per source, unmasked.
constexpr std::uint32_t kBlock = 32;
constexpr std::uint32_t kBankSkewBytes = 32;
constexpr std::uint32_t kReadPadBytes = 256;  // VECTOR_REG_WIDTH on dav-3510
constexpr std::uint32_t kUbWindow = 192u * 1024u;
constexpr std::uint32_t kStateLanes = ascend_micro::kStateLanes;      // 8
constexpr std::uint32_t kBroadcastLanes = ascend_micro::kBroadcastLanes;  // 64

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkewBytes;
}

constexpr std::uint32_t kF32 = sizeof(float);
constexpr std::uint32_t kI16 = sizeof(std::int16_t);
constexpr std::uint32_t kF16 = 2u;

// Cube Fixpipe destination. Kept at UB 0 so the AIC side can form the raw
// __ubuf__ pointer without depending on any AIV allocation.
constexpr std::uint32_t kUbScores = 0;

// Ingress staging.
constexpr std::uint32_t kUbKPacked = NextSlot(kUbScores, kPanel * kF32);
constexpr std::uint32_t kUbVPacked = NextSlot(kUbKPacked, kPackedBytes);
constexpr std::uint32_t kUbKSigned = NextSlot(kUbVPacked, kPackedBytes);
constexpr std::uint32_t kUbKMask = NextSlot(kUbKSigned, kPanel * kI16);
constexpr std::uint32_t kUbKUnsigned = NextSlot(kUbKMask, kPanel * kI16);
constexpr std::uint32_t kUbKConv = NextSlot(kUbKUnsigned, kPanel * kI16);
constexpr std::uint32_t kUbKHalf = NextSlot(kUbKConv, kPanel * kF16);
constexpr std::uint32_t kUbVSigned = NextSlot(kUbKHalf, kPanel * kF16);
constexpr std::uint32_t kUbVMask = NextSlot(kUbVSigned, kPanel * kI16);
constexpr std::uint32_t kUbVUnsigned = NextSlot(kUbVMask, kPanel * kI16);
constexpr std::uint32_t kUbVConv = NextSlot(kUbVUnsigned, kPanel * kI16);
// The dequantised value page is double buffered: the ingress of page t+1 runs
// ahead of the recurrence on page t, so the two must not share a slot.
constexpr std::uint32_t kUbVFloat0 = NextSlot(kUbVConv, kPanel * kF32);
constexpr std::uint32_t kUbVFloat1 = NextSlot(kUbVFloat0, kPanel * kF32);

// Online-softmax operands. Logically scalar state is a lane-replicated
// 32-byte DataBlock; reduction destinations hold lane 0 only.
constexpr std::uint32_t kUbOffsets = NextSlot(kUbVFloat1, kPanel * kF32);
constexpr std::uint32_t kUbMPrev = NextSlot(kUbOffsets, kPanel * kF32);
constexpr std::uint32_t kUbLPrev = NextSlot(kUbMPrev, kStateLanes * kF32);
constexpr std::uint32_t kUbMNew = NextSlot(kUbLPrev, kStateLanes * kF32);
constexpr std::uint32_t kUbLNew = NextSlot(kUbMNew, kStateLanes * kF32);
constexpr std::uint32_t kUbAlphaDelta = NextSlot(kUbLNew, kStateLanes * kF32);
constexpr std::uint32_t kUbAlpha = NextSlot(kUbAlphaDelta, kStateLanes * kF32);
constexpr std::uint32_t kUbLScaled = NextSlot(kUbAlpha, kStateLanes * kF32);
constexpr std::uint32_t kUbTileMaxRaw = NextSlot(kUbLScaled, kStateLanes * kF32);
constexpr std::uint32_t kUbPSumRaw = NextSlot(kUbTileMaxRaw, kStateLanes * kF32);
constexpr std::uint32_t kUbAccPrev = NextSlot(kUbPSumRaw, kStateLanes * kF32);
constexpr std::uint32_t kUbAccNew = NextSlot(kUbAccPrev, kDim * kF32);
constexpr std::uint32_t kUbAccScaled = NextSlot(kUbAccNew, kDim * kF32);
constexpr std::uint32_t kUbCentered = NextSlot(kUbAccScaled, kDim * kF32);
constexpr std::uint32_t kUbP = NextSlot(kUbCentered, kDim * kF32);
constexpr std::uint32_t kUbWeightedSum = NextSlot(kUbP, kDim * kF32);
constexpr std::uint32_t kUbTileMaxBc = NextSlot(kUbWeightedSum, kDim * kF32);
constexpr std::uint32_t kUbMBc = NextSlot(kUbTileMaxBc, kBroadcastLanes * kF32);
constexpr std::uint32_t kUbPSumBc = NextSlot(kUbMBc, kBroadcastLanes * kF32);
constexpr std::uint32_t kUbAlphaBc = NextSlot(kUbPSumBc, kBroadcastLanes * kF32);
constexpr std::uint32_t kUbPRepeated = NextSlot(kUbAlphaBc, kBroadcastLanes * kF32);
constexpr std::uint32_t kUbWeighted = NextSlot(kUbPRepeated, kPanel * kF32);

// Outflow operands.
constexpr std::uint32_t kUbLBroadcast = NextSlot(kUbWeighted, kPanel * kF32);
constexpr std::uint32_t kUbNormalized = NextSlot(kUbLBroadcast, kDim * kF32);
constexpr std::uint32_t kUbContext = NextSlot(kUbNormalized, kDim * kF32);
constexpr std::uint32_t kArenaEnd = kUbContext + kDim * kF16;

static_assert(kUbScores % kBlock == 0 && kUbKPacked % kBlock == 0 &&
              kUbVPacked % kBlock == 0 && kUbKHalf % kBlock == 0 &&
              kUbVFloat0 % kBlock == 0 && kUbVFloat1 % kBlock == 0 &&
              kUbOffsets % kBlock == 0 &&
              kUbMPrev % kBlock == 0 && kUbAccPrev % kBlock == 0 &&
              kUbTileMaxBc % kBlock == 0 && kUbPRepeated % kBlock == 0 &&
              kUbWeighted % kBlock == 0 && kUbLBroadcast % kBlock == 0 &&
              kUbNormalized % kBlock == 0 && kUbContext % kBlock == 0,
              "every UB operand base must be 32-byte aligned");
static_assert(kArenaEnd + kReadPadBytes <= kUbWindow,
              "UB arena leaves no VECTOR_REG_WIDTH read pad");
static_assert(kBroadcastLanes * kF32 == 8u * kBlock,
              "a Brcb repeat writes eight DataBlocks");
static_assert(kPanel * kF16 % kBlock == 0 && kDim * kF16 % kBlock == 0,
              "FP16 MTE3 lengths must be whole 32-byte bursts");

// Quantisation scales. Compile-time so no scalar argument is read on device;
// the runner mirrors them exactly.
constexpr float kKeyScale = 0.5f;
constexpr float kValueScale = 0.25f;

// Cross-core flag IDs, mode 4 (AIC <-> AIV0). Mode 4 allows both directions
// with flagId 0..10 on 950PR.
constexpr std::uint16_t kFlagKeyStaged = 0;   // AIV -> AIC, page in GM workspace
constexpr std::uint16_t kFlagScoresReady = 1; // AIC -> AIV, Fixpipe landed in UB
constexpr std::uint16_t kFlagScoresFree = 2;  // AIV -> AIC, UB scores consumed

// Intra-AIV HardEvent IDs. SetFlag/WaitFlag take an int32_t event id; IDs 6
// and 7 are reserved, so static Tensor code keeps to 0..5.
constexpr std::int32_t kEvtPacked = 0;
constexpr std::int32_t kEvtStage = 1;
constexpr std::int32_t kEvtDiag = 2;

// One ingress column: MTE2 the packed pages, dequantise both, and stage the
// FP16 key page into the GM workspace for the Cube GM->L1 converter. The
// dequantised value page lands in the slot the caller nominates, because this
// column runs one page ahead of the recurrence that consumes it.
__aicore__ inline void ingress_page(ascend_micro::NativeDequant4Engine<half>& ek,
                                    ascend_micro::NativeDequant4Engine<float>& ev,
                                    AscendC::GlobalTensor<std::int8_t>& k_packed,
                                    AscendC::GlobalTensor<std::int8_t>& v_packed,
                                    AscendC::GlobalTensor<half>& ws_k,
                                    std::uint32_t tile, std::uint32_t value_slot)
{
    ev.output = {AscendC::TPosition::VECCALC, value_slot, kPanel};
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtPacked);
    AscendC::DataCopy(ek.packed, k_packed[tile * kPackedBytes], kPackedBytes);
    AscendC::DataCopy(ev.packed, v_packed[tile * kPackedBytes], kPackedBytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtPacked);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtPacked);
    // The staging buffer must have been drained by MTE3 before the next
    // dequantisation overwrites it.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtStage);
    ascend_micro::turboquant_dequant4(ek, kPanel);
    ascend_micro::turboquant_dequant4(ev, kPanel);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtPacked);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStage);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStage);
    AscendC::DataCopy(ws_k[tile * kPanel], ek.output, kPanel);
    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(kFlagKeyStaged);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtStage);
}

__aicore__ inline void run_cube_side(__gm__ half* gm_q, __gm__ half* gm_ws_k)
{
    using namespace AscendC::Te;
    __cbuf__ half l1_q[kDim * kKeys] __attribute__((aligned(512)));
    __cbuf__ half l1_k[kDim * kKeys] __attribute__((aligned(512)));
    __ca__ half l0_a[kDim * kKeys] __attribute__((aligned(512)));
    __cb__ half l0_b[kDim * kKeys] __attribute__((aligned(512)));
    __cc__ float l0_c[kDim * kKeys] __attribute__((aligned(1024)));
    auto* ub_scores = reinterpret_cast<__ubuf__ float*>(kUbScores);

    // Q is page invariant, so it reaches L1 once and only the B operand is
    // refreshed per page.
    auto src_q = MakeTensor(MakeMemPtr(gm_q), MakeFrameLayout<NDExtLayoutPtn>(kDim, kKeys));
    auto dst_q = MakeTensor(MakeMemPtr(l1_q), MakeFrameLayout<NZLayoutPtn, half>(kDim, kKeys));
    Copy(MakeCopy(CopyGM2L1{}), dst_q, src_q);

    for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(kFlagKeyStaged);
        auto src_k = MakeTensor(MakeMemPtr(gm_ws_k + tile * kPanel),
                                MakeFrameLayout<NDExtLayoutPtn>(kKeys, kDim));
        auto dst_k = MakeTensor(MakeMemPtr(l1_k),
                                MakeFrameLayout<ZNLayoutPtn, half>(kKeys, kDim));
        Copy(MakeCopy(CopyGM2L1{}), dst_k, src_k);
        AscendC::PipeBarrier<PIPE_ALL>();
        ascend_micro::NativeCubeMatmulEngine<kDim> e{l1_q, l1_k, l0_a, l0_b,
                                                     l0_c, ub_scores};
        ascend_micro::cube_matmul_stage_l0(e);
        AscendC::PipeBarrier<PIPE_ALL>();
        // The previous page's scores must be consumed before Fixpipe
        // overwrites the shared UB panel.
        AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagScoresFree);
        ascend_micro::cube_matmul_compute_dump(e);
        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagScoresReady);
    }
    // Drain the primed free-slot token so nothing leaks across launches.
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagScoresFree);
}

__aicore__ inline void run_vector_side(__gm__ std::int8_t* gm_k_packed,
                                       __gm__ std::int8_t* gm_v_packed,
                                       __gm__ std::uint32_t* gm_offsets,
                                       __gm__ half* gm_ws_k,
                                       __gm__ float* gm_out_scores,
                                       __gm__ float* gm_out_state,
                                       __gm__ float* gm_out_normalized,
                                       __gm__ half* gm_out_context)
{
    constexpr auto ub = AscendC::TPosition::VECCALC;

    ascend_micro::NativeDequant4Engine<half> ek;
    ek.packed = {ub, kUbKPacked, kPackedBytes};
    ek.signed_nibbles = {ub, kUbKSigned, kPanel};
    ek.nibble_mask = {ub, kUbKMask, kPanel};
    ek.unsigned_nibbles = {ub, kUbKUnsigned, kPanel};
    ek.converted = {ub, kUbKConv, kPanel};
    ek.output = {ub, kUbKHalf, kPanel};
    ek.row_scale = static_cast<half>(kKeyScale);

    ascend_micro::NativeDequant4Engine<float> ev;
    ev.packed = {ub, kUbVPacked, kPackedBytes};
    ev.signed_nibbles = {ub, kUbVSigned, kPanel};
    ev.nibble_mask = {ub, kUbVMask, kPanel};
    ev.unsigned_nibbles = {ub, kUbVUnsigned, kPanel};
    ev.converted = {ub, kUbVConv, kPanel};
    ev.output = {ub, kUbVFloat0, kPanel};  // reset per page by ingress_page
    ev.row_scale = kValueScale;

    ascend_micro::NativeOnlineSoftmaxEngine es;
    es.scores = {ub, kUbScores, kKeys};  // query row 0 of the ND score panel
    es.v_by_channel = {ub, kUbVFloat0, kPanel};  // reset per page in the loop
    es.p_repeat_offsets = {ub, kUbOffsets, kPanel};
    es.m_prev = {ub, kUbMPrev, kStateLanes};
    es.l_prev = {ub, kUbLPrev, kStateLanes};
    es.m_new = {ub, kUbMNew, kStateLanes};
    es.l_new = {ub, kUbLNew, kStateLanes};
    es.alpha_delta = {ub, kUbAlphaDelta, kStateLanes};
    es.alpha = {ub, kUbAlpha, kStateLanes};
    es.l_scaled = {ub, kUbLScaled, kStateLanes};
    es.tile_max_raw = {ub, kUbTileMaxRaw, kStateLanes};
    es.p_sum_raw = {ub, kUbPSumRaw, kStateLanes};
    es.acc_prev = {ub, kUbAccPrev, kDim};
    es.acc_new = {ub, kUbAccNew, kDim};
    es.acc_scaled = {ub, kUbAccScaled, kDim};
    es.centered = {ub, kUbCentered, kDim};
    es.p = {ub, kUbP, kDim};
    es.weighted_sum = {ub, kUbWeightedSum, kDim};
    es.tile_max_bc = {ub, kUbTileMaxBc, kBroadcastLanes};
    es.m_bc = {ub, kUbMBc, kBroadcastLanes};
    es.p_sum_bc = {ub, kUbPSumBc, kBroadcastLanes};
    es.alpha_bc = {ub, kUbAlphaBc, kBroadcastLanes};
    es.p_repeated = {ub, kUbPRepeated, kPanel};
    es.weighted = {ub, kUbWeighted, kPanel};

    ascend_micro::NativeNormalizeStoreEngine<half> en;
    en.acc = {ub, kUbAccPrev, kDim};
    en.l = {ub, kUbLPrev, kStateLanes};
    en.l_broadcast = {ub, kUbLBroadcast, kDim};
    en.normalized = {ub, kUbNormalized, kDim};
    en.ub_output = {ub, kUbContext, kDim};
    en.gm_output.SetGlobalBuffer(gm_out_context);

    AscendC::LocalTensor<float> score_panel{ub, kUbScores, kPanel};

    AscendC::GlobalTensor<std::int8_t> k_packed, v_packed;
    AscendC::GlobalTensor<std::uint32_t> offsets;
    AscendC::GlobalTensor<half> ws_k;
    AscendC::GlobalTensor<float> out_scores, out_state, out_normalized;
    k_packed.SetGlobalBuffer(gm_k_packed);
    v_packed.SetGlobalBuffer(gm_v_packed);
    offsets.SetGlobalBuffer(gm_offsets);
    ws_k.SetGlobalBuffer(gm_ws_k);
    out_scores.SetGlobalBuffer(gm_out_scores);
    out_state.SetGlobalBuffer(gm_out_state);
    out_normalized.SetGlobalBuffer(gm_out_normalized);

    AscendC::DataCopy(es.p_repeat_offsets, offsets, kPanel);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtPacked);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtPacked);

    // Seed the replicated-state invariant from immediate scalars, which never
    // read UB and so carry none of the tensor-source hazards.
    AscendC::Duplicate(es.m_prev, 0.0f, static_cast<std::int32_t>(kStateLanes));
    AscendC::Duplicate(es.l_prev, 0.0f, static_cast<std::int32_t>(kStateLanes));
    AscendC::Duplicate(es.acc_prev, 0.0f, static_cast<std::int32_t>(kDim));

    // Prime the free-slot tokens: packed UB buffers are free, the staging
    // buffer is free, and the shared score panel is free.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtPacked);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtStage);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagScoresFree);

    // Prologue: prime the pipeline with page 0 so the Cube has work to do
    // while the recurrence is still idle.
    ingress_page(ek, ev, k_packed, v_packed, ws_k, 0, kUbVFloat0);

    for (std::uint32_t tile = 0; tile < kTiles; ++tile) {
        // ---- Ingress runs one page ahead of the recurrence, so the Cube
        // never waits on the dequantisation of the page it is about to
        // multiply. This is the overlap the three-stage pipeline exists for.
        if (tile + 1 < kTiles) {
            ingress_page(ek, ev, k_packed, v_packed, ws_k, tile + 1,
                         ((tile + 1) % 2 == 0) ? kUbVFloat0 : kUbVFloat1);
        }

        // ---- Recurrence: wait for this page's scores, then pure Vector.
        es.v_by_channel = {ub, (tile % 2 == 0) ? kUbVFloat0 : kUbVFloat1, kPanel};
        AscendC::CrossCoreWaitFlag<4, PIPE_V>(kFlagScoresReady);
        ascend_micro::vector_online_softmax(es);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(es.m_prev, es.m_new, 0.0f, static_cast<std::int32_t>(kStateLanes));
        AscendC::Adds(es.l_prev, es.l_new, 0.0f, static_cast<std::int32_t>(kStateLanes));
        AscendC::Adds(es.acc_prev, es.acc_new, 0.0f, static_cast<std::int32_t>(kDim));
        AscendC::PipeBarrier<PIPE_V>();

        // ---- Diagnostic: dump the whole score panel for oracle comparison.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtDiag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtDiag);
        AscendC::DataCopy(out_scores[tile * kPanel], score_panel, kPanel);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtDiag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtDiag);
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(kFlagScoresFree);
    }
    // Drain the primed intra-core tokens.
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtPacked);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtStage);

    // ---- Outflow: invoked once, after the final page.
    ascend_micro::vector_normalize(en, kDim);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtDiag);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtDiag);
    ascend_micro::vector_store(en, kDim);
    AscendC::DataCopy(out_state[0], es.m_prev, kStateLanes);
    AscendC::DataCopy(out_state[kStateLanes], es.l_prev, kStateLanes);
    AscendC::DataCopy(out_normalized[0], en.normalized, kDim);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDiag);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDiag);
}

}  // namespace fused_attention

extern "C" __global__ __aicore__ void camodel_fused_attention(
    __gm__ half* q, __gm__ std::int8_t* k_packed, __gm__ std::int8_t* v_packed,
    __gm__ std::uint32_t* offsets, __gm__ half* ws_k, __gm__ float* out_scores,
    __gm__ float* out_state, __gm__ float* out_normalized,
    __gm__ half* out_context)
{
    AscendC::InitSocState();
    if ASCEND_IS_AIC {
        fused_attention::run_cube_side(q, ws_k);
    } else {
        if (AscendC::GetSubBlockIdx() == 0) {
            fused_attention::run_vector_side(k_packed, v_packed, offsets, ws_k,
                                             out_scores, out_state,
                                             out_normalized, out_context);
        }
    }
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

extern "C" __global__ __aicore__ void camodel_fused_attention(
    __gm__ half*, __gm__ std::int8_t*, __gm__ std::int8_t*, __gm__ std::uint32_t*,
    __gm__ half*, __gm__ float*, __gm__ float*, __gm__ float*, __gm__ half*);

extern "C" void launch_camodel_fused_attention(
    void* stream, std::uint16_t* q, std::int8_t* k_packed, std::int8_t* v_packed,
    std::uint32_t* offsets, std::uint16_t* ws_k, float* out_scores,
    float* out_state, float* out_normalized, std::uint16_t* out_context)
{
    camodel_fused_attention<<<1, nullptr, stream>>>(
        reinterpret_cast<half*>(q), k_packed, v_packed, offsets,
        reinterpret_cast<half*>(ws_k), out_scores, out_state, out_normalized,
        reinterpret_cast<half*>(out_context));
}

#endif
