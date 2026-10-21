#include <cstdint>

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "../../routines/routine_turboquant_dequant4.hpp"

extern "C" __global__ __aicore__ void camodel_dequant4(
    __gm__ std::int8_t* input, __gm__ float* output)
{
    AscendC::InitSocState();
    constexpr std::uint32_t count = 64;
    AscendC::GlobalTensor<std::int8_t> gm_input;
    AscendC::GlobalTensor<float> gm_output;
    gm_input.SetGlobalBuffer(input);
    gm_output.SetGlobalBuffer(output);
    ascend_micro::NativeDequant4Engine<float> e;
    e.packed = {AscendC::TPosition::VECCALC, 0, 32};
    e.signed_nibbles = {AscendC::TPosition::VECCALC, 64, count};
    e.nibble_mask = {AscendC::TPosition::VECCALC, 224, count};
    e.unsigned_nibbles = {AscendC::TPosition::VECCALC, 384, count};
    e.converted = {AscendC::TPosition::VECCALC, 544, count};
    e.output = {AscendC::TPosition::VECCALC, 832, count};
    e.row_scale = 0.5f;
    AscendC::DataCopy(e.packed, gm_input, 32);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    ascend_micro::turboquant_dequant4_native(e, count);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(gm_output, e.output, count);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
}
#elif defined(ASCEND_MICRO_CANN_LAUNCH)
extern "C" __global__ __aicore__ void camodel_dequant4(
    __gm__ std::int8_t* input, __gm__ float* output);
extern "C" void launch_camodel_dequant4(void* stream, std::int8_t* input,
                                          float* output)
{
    camodel_dequant4<<<1, nullptr, stream>>>(input, output);
}
#endif
