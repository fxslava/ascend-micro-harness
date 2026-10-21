#include <cstdint>

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "../../routines/routine_vector_normalize_store.hpp"

namespace {

// UB layout, rule AKA3003. Every base is 32-byte aligned with one DataBlock of
// bank skew between live slots. kVecReg is the unmasked source read quantum of
// a level-2 vector op on dav-3510: operands are read a whole vector register at
// a time and only the ALU result is masked, so every source base needs kVecReg
// bytes of addressable UB behind it. Duplicate's tensor-source form in
// broadcast_denominator makes that requirement load-bearing for kLDenominator.
constexpr std::uint32_t kBlock = 32;
constexpr std::uint32_t kBankSkew = 32;
constexpr std::uint32_t kVecReg = 256;  // VECTOR_REG_WIDTH on dav-3510
constexpr std::uint32_t kUbWindow = 192u * 1024u;
constexpr std::uint32_t kLanes = 16;      // one normalize tile
constexpr std::uint32_t kStateLanes = 8;  // one replicated DataBlock

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkew;
}

constexpr std::uint32_t kAcc = 0;
constexpr std::uint32_t kLDenominator = NextSlot(kAcc, kLanes * sizeof(float));
constexpr std::uint32_t kLBroadcast =
    NextSlot(kLDenominator, kStateLanes * sizeof(float));
constexpr std::uint32_t kNormalized = NextSlot(kLBroadcast, kLanes * sizeof(float));

constexpr std::uint32_t kUbOutput = NextSlot(kNormalized, kLanes * sizeof(float));

static_assert(kAcc % kBlock == 0 && kLDenominator % kBlock == 0 &&
              kLBroadcast % kBlock == 0 && kNormalized % kBlock == 0,
              "every UB operand base must be 32-byte aligned");
// broadcast_denominator reads a whole vector register from kLDenominator.
static_assert(kLDenominator + kVecReg <= kUbWindow,
              "denominator block needs a full VECTOR_REG_WIDTH read pad");
static_assert(kUbOutput % kBlock == 0 && kUbOutput + kVecReg <= kUbWindow,
              "cast destination must be aligned and inside the UB window");
// FP16/BF16 MTE3 length must be a whole number of 32-byte bursts.
static_assert(kLanes * 2u % kBlock == 0, "MTE3 store length must be 32B aligned");

}  // namespace

template <class Output>
__aicore__ inline void run_normalize(__gm__ float* gm_acc, __gm__ float* gm_l,
                                     __gm__ Output* gm_output)
{
    AscendC::GlobalTensor<float> acc, l;
    acc.SetGlobalBuffer(gm_acc);
    l.SetGlobalBuffer(gm_l);
    ascend_micro::NativeNormalizeStoreEngine<Output> e;
    e.acc = {AscendC::TPosition::VECCALC, kAcc, kLanes};
    e.l = {AscendC::TPosition::VECCALC, kLDenominator, kStateLanes};
    e.l_broadcast = {AscendC::TPosition::VECCALC, kLBroadcast, kLanes};
    e.normalized = {AscendC::TPosition::VECCALC, kNormalized, kLanes};
    e.ub_output = {AscendC::TPosition::VECCALC, kUbOutput, kLanes};
    e.gm_output.SetGlobalBuffer(gm_output);
    AscendC::DataCopy(e.acc, acc, kLanes);
    AscendC::DataCopy(e.l, l, kStateLanes);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    ascend_micro::vector_normalize(e, kLanes);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    ascend_micro::vector_store(e, kLanes);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
}
extern "C" __global__ __aicore__ void camodel_normalize_fp16(
    __gm__ float* acc, __gm__ float* l, __gm__ half* output)
{
    AscendC::InitSocState();
    run_normalize(acc, l, output);
}
extern "C" __global__ __aicore__ void camodel_normalize_bf16(
    __gm__ float* acc, __gm__ float* l, __gm__ bfloat16_t* output)
{
    AscendC::InitSocState();
    run_normalize(acc, l, output);
}
#elif defined(ASCEND_MICRO_CANN_LAUNCH)
extern "C" __global__ __aicore__ void camodel_normalize_fp16(
    __gm__ float*, __gm__ float*, __gm__ half*);
extern "C" __global__ __aicore__ void camodel_normalize_bf16(
    __gm__ float*, __gm__ float*, __gm__ bfloat16_t*);
extern "C" void launch_camodel_normalize_fp16(void* stream, float* acc, float* l,
                                                std::uint16_t* output)
{
    camodel_normalize_fp16<<<1, nullptr, stream>>>(
        acc, l, reinterpret_cast<half*>(output));
}
extern "C" void launch_camodel_normalize_bf16(void* stream, float* acc, float* l,
                                                std::uint16_t* output)
{
    camodel_normalize_bf16<<<1, nullptr, stream>>>(
        acc, l, reinterpret_cast<bfloat16_t*>(output));
}
#endif
