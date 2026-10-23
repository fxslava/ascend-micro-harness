#include <cstdint>

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "../../routines/routine_cube_wta_fixpipe.hpp"

namespace {
constexpr unsigned next_slot(unsigned base, unsigned bytes)
{ return (base + bytes + 31u) / 32u * 32u + 32u; }
constexpr unsigned kState = 0;
constexpr unsigned kWeights = next_slot(kState, 512);
constexpr unsigned kVictory = next_slot(kWeights, 512);
constexpr unsigned kIdentity = next_slot(kVictory, 512);
static_assert(kWeights % 32 == 0 && kVictory % 32 == 0 && kIdentity % 32 == 0);
static_assert(kIdentity + 1024 <= 512 * 1024);
static_assert(1024 <= 64 * 1024 && 2048 <= 128 * 1024);

template <unsigned T, class Out>
__aicore__ inline void run_wta(__gm__ half* input, __gm__ half* weights,
                              __gm__ std::int8_t* identity, __gm__ std::int32_t* output)
{
    using namespace AscendC;
    // Fixed scale 128.0 for every T and sign; no corpus-calibrated scale.
    using Config = ascend_micro::WtaConfig<T, 0x43000000u, Out>;
    LocalTensor<half> state(TPosition::C1, kState, 256);
    LocalTensor<half> w(TPosition::B1, kWeights, 256);
    LocalTensor<Out> victory(TPosition::C1, kVictory, 512);
    LocalTensor<std::int8_t> eye(TPosition::B1, kIdentity, 1024);
    LocalTensor<half> a(TPosition::A2, 0, 256);
    LocalTensor<half> b(TPosition::B2, 0, 256);
    LocalTensor<float> c(TPosition::CO1, 0, 256);
    GlobalTensor<half> gx, gw;
    GlobalTensor<std::int8_t> gi;
    gx.SetGlobalBuffer(input); gw.SetGlobalBuffer(weights); gi.SetGlobalBuffer(identity);
    DataCopy(state, gx, 256);
    DataCopy(w, gw, 256);
    DataCopy(eye, gi, 1024);
    SetFlag<HardEvent::MTE2_MTE1>(1); WaitFlag<HardEvent::MTE2_MTE1>(1);
    LoadData2DParams load;
    load.repeatTimes = 1; load.srcStride = 1; load.ifTranspose = false;
    LoadData(b, w, load);  // Prepacked Zn, no runtime transpose.
    ascend_micro::NativeCubeWtaEngine<Config> engine{a, b, c, state, victory};
    ascend_micro::NativeCubeWtaSync<0> sync;
    ascend_micro::cube_wta_run<Config>(engine, sync);

    // Test-only observation, AFTER the routine. Arch3510 has no L1->GM copy.
    // Read the actual B8 C1 result as 8x32 using an INT8 identity Mmad, then
    // Fixpipe INT32 to GM. This verifies the feedback/output storage without
    // introducing an AIV, a UB hop, scalar tensor reads, or an intermediate GM
    // spill into the routine. For uint8 >127, the host recovers the same byte
    // from its sign-extended INT32 representation. N=16 output is unmerged:
    // 256 packed bytes. LoadData reads 512 bytes, so this wrapper reserves an
    // extra 256 bytes after victory; Mmad consumes only the first eight rows.
    LocalTensor<std::int8_t> qa(TPosition::A2, 0, 512);
    LocalTensor<std::int8_t> qb(TPosition::B2, 0, 1024);
    LocalTensor<std::int8_t> qv(TPosition::C1, kVictory, 512);
    LocalTensor<std::int32_t> qc(TPosition::CO1, 0, 512);
    LoadData(qa, qv, load);
    load.repeatTimes = 2;
    LoadData(qb, eye, load);
    SetFlag<HardEvent::MTE1_M>(1); WaitFlag<HardEvent::MTE1_M>(1);
    MmadParams mp;
    mp.m = 8; mp.n = 32; mp.k = 32; mp.cmatrixInitVal = true; mp.unitFlag = 0;
    Mmad(qc, qa, qb, mp);
    SetFlag<HardEvent::M_FIX>(1); WaitFlag<HardEvent::M_FIX>(1);
    GlobalTensor<std::int32_t> gy;
    gy.SetGlobalBuffer(output);
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fp;
    fp.mSize = 8; fp.nSize = 32; fp.srcStride = 16; fp.dstStride = 32;
    fp.quantPre = QuantMode_t::NoQuant; fp.unitFlag = 0;
    Fixpipe<std::int32_t, std::int32_t, CFG_ROW_MAJOR>(gy, qc, fp);
    SetFlag<HardEvent::FIX_S>(1); WaitFlag<HardEvent::FIX_S>(1);
}
}  // namespace

#define WTA_ENTRY(N, SUFFIX, TYPE) \
extern "C" __global__ __aicore__ void camodel_wta_##N##SUFFIX( \
    __gm__ half* x, __gm__ half* w, __gm__ std::int8_t* eye, __gm__ std::int32_t* y) \
{ \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY); \
    AscendC::InitSocState(); \
    run_wta<N, TYPE>(x, w, eye, y); \
}
#else
#if defined(ASCEND_MICRO_CANN_LAUNCH)
#define WTA_ENTRY(N, SUFFIX, TYPE) \
extern "C" __global__ __aicore__ void camodel_wta_##N##SUFFIX( \
    __gm__ half*, __gm__ half*, __gm__ std::int8_t*, __gm__ std::int32_t*); \
extern "C" void launch_wta_##N##SUFFIX( \
    void* stream, std::uint16_t* x, std::uint16_t* w, std::int8_t* eye, std::int32_t* y) \
{ camodel_wta_##N##SUFFIX<<<1, nullptr, stream>>>( \
    reinterpret_cast<half*>(x), reinterpret_cast<half*>(w), eye, y); }
#endif
#endif

#ifdef WTA_ENTRY
WTA_ENTRY(1, s, std::int8_t)
WTA_ENTRY(2, s, std::int8_t)
WTA_ENTRY(3, s, std::int8_t)
WTA_ENTRY(4, s, std::int8_t)
WTA_ENTRY(5, s, std::int8_t)
WTA_ENTRY(6, s, std::int8_t)
WTA_ENTRY(7, s, std::int8_t)
WTA_ENTRY(8, s, std::int8_t)
WTA_ENTRY(1, u, std::uint8_t)
WTA_ENTRY(2, u, std::uint8_t)
WTA_ENTRY(3, u, std::uint8_t)
WTA_ENTRY(4, u, std::uint8_t)
WTA_ENTRY(5, u, std::uint8_t)
WTA_ENTRY(6, u, std::uint8_t)
WTA_ENTRY(7, u, std::uint8_t)
WTA_ENTRY(8, u, std::uint8_t)
#undef WTA_ENTRY
#endif
