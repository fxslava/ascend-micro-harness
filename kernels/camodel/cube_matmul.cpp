#include <cstdint>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "../../routines/routine_cube_matmul.hpp"

template <int M>
__aicore__ inline void run_cube(__gm__ half* gm_a, __gm__ half* gm_b,
                                __gm__ float* gm_output)
{
    if ASCEND_IS_AIC {
        using namespace AscendC::Te;
        __cbuf__ half l1_a[M * 16] __attribute__((aligned(512)));
        __cbuf__ half l1_b[16 * 16] __attribute__((aligned(512)));
        __ca__ half l0_a[M * 16] __attribute__((aligned(512)));
        __cb__ half l0_b[16 * 16] __attribute__((aligned(512)));
        __cc__ float l0_c[M * 16] __attribute__((aligned(1024)));
        __ubuf__ float* ub_output = reinterpret_cast<__ubuf__ float*>(0);
        auto src_a = MakeTensor(MakeMemPtr(gm_a), MakeFrameLayout<NDExtLayoutPtn>(M, 16));
        auto src_b = MakeTensor(MakeMemPtr(gm_b), MakeFrameLayout<NDExtLayoutPtn>(16, 16));
        auto dst_a = MakeTensor(MakeMemPtr(l1_a), MakeFrameLayout<NZLayoutPtn, half>(M, 16));
        auto dst_b = MakeTensor(MakeMemPtr(l1_b), MakeFrameLayout<ZNLayoutPtn, half>(16, 16));
        Copy(MakeCopy(CopyGM2L1{}), dst_a, src_a);
        Copy(MakeCopy(CopyGM2L1{}), dst_b, src_b);
        AscendC::PipeBarrier<PIPE_ALL>();
        ascend_micro::NativeCubeMatmulEngine<M> e{l1_a, l1_b, l0_a, l0_b, l0_c, ub_output};
        ascend_micro::cube_matmul_stage_l0(e);
        AscendC::PipeBarrier<PIPE_ALL>();
        ascend_micro::cube_matmul_compute_dump(e);
        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(0);
    } else {
        if (AscendC::GetSubBlockIdx() == 0) {
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(0);
            AscendC::GlobalTensor<float> output;
            output.SetGlobalBuffer(gm_output);
            AscendC::LocalTensor<float> ub(AscendC::TPosition::VECCALC, 0, M * 16);
            AscendC::DataCopy(output, ub, M * 16);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }
}

extern "C" __global__ __aicore__ void camodel_cube16(
    __gm__ half* a, __gm__ half* b, __gm__ float* output)
{
    AscendC::InitSocState();
    run_cube<16>(a, b, output);
}
extern "C" __global__ __aicore__ void camodel_cube32(
    __gm__ half* a, __gm__ half* b, __gm__ float* output)
{
    AscendC::InitSocState();
    run_cube<32>(a, b, output);
}
#elif defined(ASCEND_MICRO_CANN_LAUNCH)
extern "C" __global__ __aicore__ void camodel_cube16(
    __gm__ half* a, __gm__ half* b, __gm__ float* output);
extern "C" __global__ __aicore__ void camodel_cube32(
    __gm__ half* a, __gm__ half* b, __gm__ float* output);
extern "C" void launch_camodel_cube16(void* stream, std::uint16_t* a,
                                        std::uint16_t* b, float* output)
{
    camodel_cube16<<<1, nullptr, stream>>>(reinterpret_cast<half*>(a),
                                          reinterpret_cast<half*>(b), output);
}
extern "C" void launch_camodel_cube32(void* stream, std::uint16_t* a,
                                        std::uint16_t* b, float* output)
{
    camodel_cube32<<<1, nullptr, stream>>>(reinterpret_cast<half*>(a),
                                          reinterpret_cast<half*>(b), output);
}
#endif
