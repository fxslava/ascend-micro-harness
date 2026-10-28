#include <cstdint>

// CAModel micro-benchmark for the three randomised-Hadamard butterfly
// implementations in routines/routine_vector_rotate_butterfly.hpp.
// Ascend 950PR_9599 (dav-3510, __NPU_ARCH__ 3510), CANN 9.2.0-beta.2,
// RUN_MODE=sim, KERNEL_TYPE_AIV_ONLY, one AIV by construction.
//
// What this kernel proves, and what it is for
// -------------------------------------------
// The scatter ingress operator (roadmap No.1) is AIV-only and its whole
// arithmetic budget is this leaf, so which permutation strategy the eight
// stages use decides the operator. Three kernels, one per strategy, over the
// same fixture, the same grid and the same surrounding DMA:
//
//   camodel_rotate_butterfly_blocked   Approach A: Unified Buffer operands,
//                                      BinaryRepeatParams strides for s >= 8,
//                                      a level-2 Gather against a 256-entry UB
//                                      offset table for s in {1, 2, 4}. Every
//                                      stage round-trips through UB.
//   camodel_rotate_butterfly_gather    Approach B: four registers resident for
//                                      all eight stages, lane stages as
//                                      Reg::Gather + Reg::MulAddDst.
//   camodel_rotate_butterfly_shuffle   Approach B: same residency, lane stages
//                                      as DeInterleave/Select with no index
//                                      register at all.
//
// The comparison is between the three VECTOR engine censuses, which the three
// kernels make directly comparable: identical GM operands, identical MTE2 and
// MTE3 bursts, identical event ids, and the only difference inside the one
// region between the MTE2_V wait and the V_MTE3 set. `reps` repeats the
// transform over the same source panel so a steady-state per-row cost can be
// separated from the one-off table construction -- the transform is a pure
// function of the source panel, so every repetition writes the same answer and
// the numeric gate still holds.
//
// What it deliberately omits: the quantiser, the nibble packing, the cache
// read-modify-write and any grid wider than one core. Those belong to
// kernels/ops/turboquant/turboquant_reshape_scatter_aiv.cpp, which this
// benchmark exists to choose a leaf for.

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "../../routines/routine_vector_rotate_butterfly.hpp"

namespace {

using ascend_micro::kRotChannels;
using ascend_micro::kRotGatherStages;

// UB layout, rule AKA3003. Every base 32-byte aligned, one DataBlock of bank
// skew between live slots, VECTOR_REG_WIDTH of read pad behind the arena.
constexpr std::uint32_t kBlock = 32;
constexpr std::uint32_t kBankSkew = 32;
constexpr std::uint32_t kVecReg = 256;  // VECTOR_REG_WIDTH on dav-3510
constexpr std::uint32_t kUbWindow = 192u * 1024u;

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkew;
}

// The benchmark's row ceiling is the operator's: a request contributes at most
// blockSize = 16 tokens to one strip.
constexpr std::uint32_t kMaxRows = 16;
constexpr std::uint32_t kRowBytes = kRotChannels * 4u;            // 1024
constexpr std::uint32_t kPanelBytes = kMaxRows * kRowBytes;       // 16384
constexpr std::uint32_t kTableBytes = kRotGatherStages * kRotChannels * 4u;  // 3072

constexpr std::uint32_t kIn = 0;
constexpr std::uint32_t kOut = NextSlot(kIn, kPanelBytes);
constexpr std::uint32_t kSign = NextSlot(kOut, kPanelBytes);
constexpr std::uint32_t kOffsets = NextSlot(kSign, kRowBytes);
constexpr std::uint32_t kGSigns = NextSlot(kOffsets, kTableBytes);
constexpr std::uint32_t kBufA = NextSlot(kGSigns, kTableBytes);
constexpr std::uint32_t kBufB = NextSlot(kBufA, kRowBytes);
constexpr std::uint32_t kPerm = NextSlot(kBufB, kRowBytes);
constexpr std::uint32_t kSignTmp = NextSlot(kPerm, kRowBytes);
constexpr std::uint32_t kArenaEnd = kSignTmp + kRowBytes;

static_assert(kIn % kBlock == 0 && kOut % kBlock == 0 && kSign % kBlock == 0 &&
                  kOffsets % kBlock == 0 && kGSigns % kBlock == 0 && kBufA % kBlock == 0 &&
                  kBufB % kBlock == 0 && kPerm % kBlock == 0 && kSignTmp % kBlock == 0,
              "every UB operand base must be 32-byte aligned");
// Rule A2: the three live scratch rows must not share a bank phase.
static_assert((kBufB - kBufA) % 256u != 0u && (kPerm - kBufB) % 256u != 0u,
              "scratch slot stride must not be a multiple of 256 B");
// Rule A3: level-2 continuous ops read a whole 256-byte vector register from
// each source base, unmasked.
static_assert(kArenaEnd + kVecReg <= kUbWindow, "UB arena leaves no read pad");
// The register-resident bodies address the panels as raw __ubuf__ pointers, so
// a row base must be register-aligned as well as DataBlock-aligned.
static_assert(kRowBytes % kVecReg == 0, "a 256-channel FP32 row is whole vector registers");
static_assert(kPanelBytes % kBlock == 0 && kRowBytes % kBlock == 0,
              "DMA lengths must be whole 32-byte bursts");

// Event ids, static code 0..5 (6/7 reserved). One token in flight per route.
constexpr std::int32_t kEvtInReady = 0;   // MTE2_V: panel, signs and tables in UB
constexpr std::int32_t kEvtOutReady = 1;  // V_MTE3: the rotated panel is in UB
constexpr std::int32_t kEvtDrain = 2;     // MTE3_S: the final outflow retired

enum class Variant : unsigned { Blocked, RegGather, RegShuffle };

template <Variant V>
__aicore__ inline void run_rotate(__gm__ float* gm_in, __gm__ float* gm_out,
                                  __gm__ float* gm_sign, __gm__ std::uint32_t* gm_offsets,
                                  __gm__ float* gm_gsigns, std::uint32_t rows,
                                  std::uint32_t reps)
{
    constexpr auto ub = AscendC::TPosition::VECCALC;
    AscendC::LocalTensor<float> in{ub, kIn, kMaxRows * kRotChannels};
    AscendC::LocalTensor<float> out{ub, kOut, kMaxRows * kRotChannels};
    AscendC::LocalTensor<float> sign{ub, kSign, kRotChannels};
    AscendC::LocalTensor<std::uint32_t> offsets{ub, kOffsets, kRotGatherStages * kRotChannels};
    AscendC::LocalTensor<float> gsigns{ub, kGSigns, kRotGatherStages * kRotChannels};

    AscendC::GlobalTensor<float> g_in, g_out, g_sign, g_gsigns;
    AscendC::GlobalTensor<std::uint32_t> g_offsets;
    g_in.SetGlobalBuffer(gm_in);
    g_out.SetGlobalBuffer(gm_out);
    g_sign.SetGlobalBuffer(gm_sign);
    g_offsets.SetGlobalBuffer(gm_offsets);
    g_gsigns.SetGlobalBuffer(gm_gsigns);

    // Identical ingress for all three variants, so the vector censuses differ
    // only by the leaf. The Blocked tables are copied in even when the variant
    // does not read them: an MTE2 burst the comparison does not care about is
    // cheaper than a comparison that is not like for like.
    AscendC::DataCopy(in, g_in[0], kMaxRows * kRotChannels);
    AscendC::DataCopy(sign, g_sign[0], kRotChannels);
    AscendC::DataCopy(offsets, g_offsets[0], kRotGatherStages * kRotChannels);
    AscendC::DataCopy(gsigns, g_gsigns[0], kRotGatherStages * kRotChannels);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtInReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtInReady);

    // ---- the region under measurement -----------------------------------
    for (std::uint32_t rep = 0; rep < reps; ++rep) {
        if constexpr (V == Variant::Blocked) {
            ascend_micro::NativeRotateBlockedEngine engine;
            engine.pi_sign = sign;
            engine.gather_offsets = offsets;
            engine.gather_signs = gsigns;
            engine.buffer_a = AscendC::LocalTensor<float>{ub, kBufA, kRotChannels};
            engine.buffer_b = AscendC::LocalTensor<float>{ub, kBufB, kRotChannels};
            engine.permuted = AscendC::LocalTensor<float>{ub, kPerm, kRotChannels};
            engine.signed_tmp = AscendC::LocalTensor<float>{ub, kSignTmp, kRotChannels};
            for (std::uint32_t r = 0; r < rows; ++r) {
                const std::uint32_t base = r * kRotChannels;
                AscendC::LocalTensor<float> src{ub, kIn + r * kRowBytes, kRotChannels};
                AscendC::LocalTensor<float> dst{ub, kOut + r * kRowBytes, kRotChannels};
                (void)base;
                engine.rotate<ascend_micro::RotateDirection::Ingress>(dst, src);
                AscendC::PipeBarrier<PIPE_V>();
                engine.scale(dst, 1.0f);
                AscendC::PipeBarrier<PIPE_V>();
            }
        } else {
            __ubuf__ float* in_base = reinterpret_cast<__ubuf__ float*>(in.GetPhyAddr());
            __ubuf__ float* out_base = reinterpret_cast<__ubuf__ float*>(out.GetPhyAddr());
            __ubuf__ float* sign_base = reinterpret_cast<__ubuf__ float*>(sign.GetPhyAddr());
            if constexpr (V == Variant::RegGather) {
                ascend_micro::rotate_butterfly_gather_vf<
                    ascend_micro::RotateDirection::Ingress>(
                    out_base, in_base, sign_base, 1.0f, static_cast<std::uint16_t>(rows));
            } else {
                ascend_micro::rotate_butterfly_shuffle_vf(
                    out_base, in_base, sign_base, 1.0f, static_cast<std::uint16_t>(rows));
            }
        }
    }
    // ---- end of the region under measurement ------------------------------

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtOutReady);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtOutReady);
    // Exactly the rows this launch transformed, not all kMaxRows of them. The
    // output panel is never initialised, so copying the unfilled rows would
    // publish whatever those Unified Buffer bytes happened to hold -- and the
    // runner's "no row beyond the launch's row count was written" check is
    // precisely that the poison beyond the window comes back untouched. At
    // `--rows 16` the two lengths coincide, which is why this only surfaced
    // when the decode shape was finally run.
    AscendC::DataCopy(g_out[0], out, rows * kRotChannels);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
}

}  // namespace

#define ROTATE_ENTRY(NAME, VARIANT)                                                       \
    extern "C" __global__ __aicore__ void camodel_rotate_butterfly_##NAME(                \
        __gm__ float* in, __gm__ float* out, __gm__ float* sign,                          \
        __gm__ std::uint32_t* offsets, __gm__ float* gsigns, uint32_t rows, uint32_t reps) \
    {                                                                                     \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                   \
        AscendC::InitSocState();                                                          \
        run_rotate<VARIANT>(in, out, sign, offsets, gsigns, rows, reps);                  \
    }

ROTATE_ENTRY(blocked, Variant::Blocked)
ROTATE_ENTRY(gather, Variant::RegGather)
ROTATE_ENTRY(shuffle, Variant::RegShuffle)
#undef ROTATE_ENTRY

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

#define ROTATE_LAUNCH(NAME)                                                                \
    extern "C" __global__ __aicore__ void camodel_rotate_butterfly_##NAME(                 \
        __gm__ float*, __gm__ float*, __gm__ float*, __gm__ std::uint32_t*, __gm__ float*, \
        uint32_t, uint32_t);                                                               \
    extern "C" void launch_camodel_rotate_butterfly_##NAME(                                \
        void* stream, float* in, float* out, float* sign, std::uint32_t* offsets,          \
        float* gsigns, std::uint32_t rows, std::uint32_t reps)                             \
    {                                                                                      \
        camodel_rotate_butterfly_##NAME<<<1, nullptr, stream>>>(in, out, sign, offsets,    \
                                                                gsigns, rows, reps);       \
    }

ROTATE_LAUNCH(blocked)
ROTATE_LAUNCH(gather)
ROTATE_LAUNCH(shuffle)
#undef ROTATE_LAUNCH

#endif
