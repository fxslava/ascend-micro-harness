/*
 * Simulator-only probe for the Cube GEMV Split-K reduction, and its launcher.
 *
 * Isolated from production the same way tests/sim/kernel/fused_decode_opt_kernel.cpp is: this
 * translation unit lives under tests/, nothing in kernels/ includes it, and the only symbols it
 * exports are the probe entry points. The combine arithmetic it compares against is the
 * production TurboQuantPartialReducer, instantiated in the SAME kernel from the production
 * header, so the two legs cannot drift apart between the probe and the operator.
 *
 * Two __global__ symbols, differing in nothing but which reducer runs over the same workspace:
 *   turboquant_gemm_reduce_probe     the Cube GEMV stage
 *   turboquant_vector_reduce_probe   the shipping vector reducer, the numerical reference
 * and a third that runs the GEMV leg with Mmad's GEMV mode switched off, so a behavioural
 * difference isolates that knob and nothing else.
 */

#include "../../../include/turboquant_gemm_reduce.hpp"

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"

#include "../../../kernels/ops/turboquant/fused_decode/cube/turboquant_gemm_reduce_service.h"

using ascend_micro::gemm_reduce::TurboQuantBatchedGemmReducer;
using ascend_micro::gemm_reduce::TurboQuantGemmReducer;
using vllm_ascend::turboquant::TurboQuantPartialReducer;
using vllm_ascend::turboquant::TurboQuantRotatedOutput;

namespace {

// Which (token, head) needs a combine. `always` is the pure multi-split case; otherwise the
// decision is production's IsFused, read from the same length vector the operator reads.
//
// The length tensor ALWAYS points at a real allocation, even when `always` is set, and the leg
// is selected by a flag rather than by a null pointer. Passing nullptr here is what the first
// CAModel run of this probe did, and the Cube core raised su_ccu_mpu_err_t0 on a B32 scalar
// load at address 4 -- a null-base GM access from the branch that was not taken. A kernel that
// never receives a null GM address cannot have that failure mode at all.
struct CombineTokens {
    AscendC::GlobalTensor<int32_t> lengths;
    int32_t limit = -1;
    bool always = true;
    __aicore__ inline bool NeedsCombine(const uint32_t token) const
    {
        if (always) {
            return true;
        }
        return lengths.GetValue(token) > limit;
    }
};

__aicore__ inline void DrainReducePhase()
{
    constexpr uint16_t flag = ascend_micro::fused_decode_opt::kTaskJoinFlag;
    if ASCEND_IS_AIC {
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::M_S>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::FIX_S>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE2_S>();
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(flag);
        AscendC::CrossCoreWaitFlag(flag);
        // Close MTE1 -> Matrix -> Scalar explicitly, empty workers included, before the join
        // channel can be reused.
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE1_M>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::M_S>();
    }
    if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag(flag);
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE2_S>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::V_S>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(flag);
    }
}

template <bool GEMV_DISABLED>
__aicore__ inline void RunGemmLeg(GM_ADDR workspace, GM_ADDR output, GM_ADDR lengths,
                                  uint32_t tokens, uint32_t heads, uint32_t splits,
                                  uint32_t perBlock, int32_t fusedLimit)
{
    AscendC::TPipe pipe;
    TurboQuantGemmReducer<half, GEMV_DISABLED> reducer(&pipe);
    reducer.Init(workspace, output, heads, ascend_micro::gemm_reduce::kHeadSize, splits);
    TurboQuantRotatedOutput finisher;
    CombineTokens select;
    select.lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(lengths));
    select.limit = fusedLimit;
    select.always = fusedLimit < 0;
    reducer.Reduce(tokens * heads, perBlock, select, finisher);
    DrainReducePhase();
}

// The dense batched form. Same shape of body as RunGemmLeg -- the probe's two forms differ in
// the reducer they instantiate and in nothing else, so a behavioural difference between them
// isolates the packing and not the harness.
template <bool GEMV_DISABLED>
__aicore__ inline void RunBatchedLeg(GM_ADDR workspace, GM_ADDR output, GM_ADDR lengths,
                                     uint32_t tokens, uint32_t heads, uint32_t splits,
                                     uint32_t perBlock, int32_t fusedLimit)
{
    AscendC::TPipe pipe;
    TurboQuantBatchedGemmReducer<half, GEMV_DISABLED> reducer(&pipe);
    reducer.Init(workspace, output, heads, ascend_micro::gemm_reduce::kHeadSize, splits);
    TurboQuantRotatedOutput finisher;
    CombineTokens select;
    select.lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(lengths));
    select.limit = fusedLimit;
    select.always = fusedLimit < 0;
    reducer.Reduce(tokens * heads, perBlock, select, finisher);
    DrainReducePhase();
}

}  // namespace

extern "C" __global__ __aicore__ void turboquant_gemm_reduce_probe(
    GM_ADDR workspace, GM_ADDR output, GM_ADDR lengths, uint32_t tokens, uint32_t heads,
    uint32_t splits, uint32_t perBlock, int32_t fusedLimit)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    RunGemmLeg<ascend_micro::gemm_reduce::kGemvDisabledDefault>(
        workspace, output, lengths, tokens, heads, splits, perBlock, fusedLimit);
}

// Identical body, one knob: Mmad's M = 1 GEMV mode left enabled.
extern "C" __global__ __aicore__ void turboquant_gemm_reduce_gemv_probe(
    GM_ADDR workspace, GM_ADDR output, GM_ADDR lengths, uint32_t tokens, uint32_t heads,
    uint32_t splits, uint32_t perBlock, int32_t fusedLimit)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    RunGemmLeg<false>(workspace, output, lengths, tokens, heads, splits, perBlock, fusedLimit);
}

// The dense batched form: M = kGemmBatchRows packed query rows against 16-channel slices, both
// vector subcores reducing half the batch each.
extern "C" __global__ __aicore__ void turboquant_batched_reduce_probe(
    GM_ADDR workspace, GM_ADDR output, GM_ADDR lengths, uint32_t tokens, uint32_t heads,
    uint32_t splits, uint32_t perBlock, int32_t fusedLimit)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    RunBatchedLeg<ascend_micro::gemm_reduce::kGemvDisabledDefault>(
        workspace, output, lengths, tokens, heads, splits, perBlock, fusedLimit);
}

// Identical body, one knob. With M = 16 there is no GEMV case for disableGemv to affect, so
// the two batched legs should be indistinguishable -- which is the measurement, because in the
// M = 1 form they were not.
extern "C" __global__ __aicore__ void turboquant_batched_reduce_gemv_probe(
    GM_ADDR workspace, GM_ADDR output, GM_ADDR lengths, uint32_t tokens, uint32_t heads,
    uint32_t splits, uint32_t perBlock, int32_t fusedLimit)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    RunBatchedLeg<false>(workspace, output, lengths, tokens, heads, splits, perBlock, fusedLimit);
}

// The reference leg: the shipping vector reducer over the same workspace, so the numerical
// comparison is against production arithmetic and not against a host reimplementation of it.
extern "C" __global__ __aicore__ void turboquant_vector_reduce_probe(
    GM_ADDR workspace, GM_ADDR output, GM_ADDR lengths, uint32_t tokens, uint32_t heads,
    uint32_t splits, uint32_t perBlock, int32_t fusedLimit)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    if ASCEND_IS_AIV {
        TurboQuantPartialReducer<half> reducer(&pipe);
        reducer.Init(workspace, output, heads, ascend_micro::gemm_reduce::kHeadSize, splits);
        TurboQuantRotatedOutput finisher;
        const uint32_t total = tokens * heads;
        const uint32_t start = vllm_ascend::turboquant::MixBlockIdx() * perBlock;
        uint32_t end = start + perBlock;
        if (end > total) {
            end = total;
        }
        if (start < end) {
            // The same halving between subcores production's Reduce() applies.
            const uint32_t firstHalf = vllm_ascend::turboquant::CeilDiv(
                end - start, vllm_ascend::turboquant::kVectorSubcoresPerBlock);
            const bool second = AscendC::GetSubBlockIdx() != 0;
            const uint32_t from = second ? start + firstHalf : start;
            const uint32_t to = second ? end : start + firstHalf;
            AscendC::GlobalTensor<int32_t> lengthGm;
            if (lengths != nullptr) {
                lengthGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(lengths));
            }
            for (uint32_t task = from; task < to; ++task) {
                const uint32_t token = task / heads;
                if (lengths != nullptr && lengthGm.GetValue(token) <= fusedLimit) {
                    continue;
                }
                reducer.Reduce(token, task % heads, finisher);
            }
        }
    }
    DrainReducePhase();
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

// GM_ADDR is a BiSheng device builtin, not a header macro, so the host leg of the kernel
// library needs this include to declare the entry points at all -- the same include
// op_kernel/turboquant_fused_decode.cpp's own launch branch carries.
#include "kernel_operator.h"

#include <stdexcept>

extern "C" __global__ __aicore__ void turboquant_gemm_reduce_probe(
    GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t, uint32_t, uint32_t, int32_t);
extern "C" __global__ __aicore__ void turboquant_gemm_reduce_gemv_probe(
    GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t, uint32_t, uint32_t, int32_t);
extern "C" __global__ __aicore__ void turboquant_vector_reduce_probe(
    GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t, uint32_t, uint32_t, int32_t);
extern "C" __global__ __aicore__ void turboquant_batched_reduce_probe(
    GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t, uint32_t, uint32_t, int32_t);
extern "C" __global__ __aicore__ void turboquant_batched_reduce_gemv_probe(
    GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t, uint32_t, uint32_t, int32_t);

extern "C" void launch_gemm_reduce_sim(void *stream,
                                       const ascend_micro::gemm_reduce::ReduceLaunch *c,
                                       const ascend_micro::gemm_reduce::ReduceBuffers *b)
{
    if (!c || !b) throw std::invalid_argument("null reduce descriptor");
    const bool batched = c->leg == ascend_micro::gemm_reduce::ReduceLeg::kBatched ||
                         c->leg == ascend_micro::gemm_reduce::ReduceLeg::kBatchedGemvMode;
    const bool geometry = batched
                              ? ascend_micro::gemm_reduce::kValidBatchedGeometry(c->head_size,
                                                                                c->splits)
                              : ascend_micro::gemm_reduce::kValidReduceGeometry(c->head_size,
                                                                               c->splits);
    if (!geometry || !c->grid || c->grid > 32 || !c->tokens || !c->heads || !c->per_block)
        throw std::invalid_argument("unsupported gemm reduce probe geometry");
    switch (c->leg) {
        case ascend_micro::gemm_reduce::ReduceLeg::kGemm:
            turboquant_gemm_reduce_probe<<<c->grid, nullptr, stream>>>(
                b->workspace, b->output, b->lengths, c->tokens, c->heads, c->splits,
                c->per_block, c->fused_limit);
            return;
        case ascend_micro::gemm_reduce::ReduceLeg::kGemmGemvMode:
            turboquant_gemm_reduce_gemv_probe<<<c->grid, nullptr, stream>>>(
                b->workspace, b->output, b->lengths, c->tokens, c->heads, c->splits,
                c->per_block, c->fused_limit);
            return;
        case ascend_micro::gemm_reduce::ReduceLeg::kVector:
            turboquant_vector_reduce_probe<<<c->grid, nullptr, stream>>>(
                b->workspace, b->output, b->lengths, c->tokens, c->heads, c->splits,
                c->per_block, c->fused_limit);
            return;
        case ascend_micro::gemm_reduce::ReduceLeg::kBatched:
            turboquant_batched_reduce_probe<<<c->grid, nullptr, stream>>>(
                b->workspace, b->output, b->lengths, c->tokens, c->heads, c->splits,
                c->per_block, c->fused_limit);
            return;
        case ascend_micro::gemm_reduce::ReduceLeg::kBatchedGemvMode:
            turboquant_batched_reduce_gemv_probe<<<c->grid, nullptr, stream>>>(
                b->workspace, b->output, b->lengths, c->tokens, c->heads, c->splits,
                c->per_block, c->fused_limit);
            return;
    }
    throw std::invalid_argument("unknown reduce leg");
}
#endif
