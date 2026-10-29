#include <cstdint>

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"

namespace research_sync {

// Флаги Mode-4: 0..15 принадлежат AIV0, 16..31 принадлежат AIV1
constexpr std::uint16_t kFlagAiv0ToAic = 0u;
constexpr std::uint16_t kFlagAiv1ToAic = 16u;
constexpr std::uint16_t kFlagAicToAiv0 = 1u;
constexpr std::uint16_t kFlagAicToAiv1 = 17u;

constexpr std::int32_t kEvtDrain = 0;

__aicore__ inline void run_aic(std::uint32_t iters) {
    AscendC::PRINTF("[AIC] Core started\n");
    for (std::uint32_t i = 0; i < iters; ++i) {
        AscendC::PRINTF("[AIC] iter %u: waiting AIV0 (flag %u)...\n", i, kFlagAiv0ToAic);
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(kFlagAiv0ToAic);
        AscendC::PRINTF("[AIC] iter %u: wait AIV0 passed! Waiting AIV1 (flag %u)...\n", i, kFlagAiv1ToAic);
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(kFlagAiv1ToAic);
        AscendC::PRINTF("[AIC] iter %u: quorum reached, draining MTE1...\n", i);

        // Дренаж MTE1 через скалярный барьер перед анонсом
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtDrain);
        AscendC::SetFlag<AscendC::HardEvent::M_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::M_S>(kEvtDrain);

        AscendC::PRINTF("[AIC] iter %u: releasing AIV0 and AIV1\n", i);
        AscendC::CrossCoreSetFlag<4, PIPE_M>(kFlagAicToAiv0);
        AscendC::CrossCoreSetFlag<4, PIPE_M>(kFlagAicToAiv1);
    }
    AscendC::PRINTF("[AIC] Core finished cleanly\n");
}

__aicore__ inline void run_aiv0(std::uint32_t iters) {
    AscendC::PRINTF("[AIV0] Core started\n");
    for (std::uint32_t i = 0; i < iters; ++i) {
        AscendC::PRINTF("[AIV0] iter %u: notifying AIC (flag %u)...\n", i, kFlagAiv0ToAic);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtDrain);
        AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagAiv0ToAic);

        AscendC::PRINTF("[AIV0] iter %u: waiting response from AIC (flag %u)...\n", i, kFlagAicToAiv0);
        AscendC::CrossCoreWaitFlag<4, PIPE_V>(kFlagAicToAiv0);
        AscendC::PRINTF("[AIV0] iter %u: wait AIC passed!\n", i);
    }
    AscendC::PRINTF("[AIV0] Core finished cleanly\n");
}

__aicore__ inline void run_aiv1(std::uint32_t iters) {
    AscendC::PRINTF("[AIV1] Core started\n");
    for (std::uint32_t i = 0; i < iters; ++i) {
        AscendC::PRINTF("[AIV1] iter %u: notifying AIC (flag %u)...\n", i, kFlagAiv1ToAic);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtDrain);
        AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagAiv1ToAic);

        AscendC::PRINTF("[AIV1] iter %u: waiting response from AIC (flag %u)...\n", i, kFlagAicToAiv1);
        AscendC::CrossCoreWaitFlag<4, PIPE_V>(kFlagAicToAiv1);
        AscendC::PRINTF("[AIV1] iter %u: wait AIC passed!\n", i);
    }
    AscendC::PRINTF("[AIV1] Core finished cleanly\n");
}

}  // namespace research_sync

extern "C" __global__ __aicore__ void kernel_cluster_sync_probe(uint32_t iters) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();

    const std::uint32_t block = static_cast<std::uint32_t>(
        AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());

    // Ограничиваемся строго одним физическим блоком AI Core
    if (block != 0u) return;

    if ASCEND_IS_AIC {
        research_sync::run_aic(iters);
    } else if (AscendC::GetSubBlockIdx() == 0) {
        research_sync::run_aiv0(iters);
    } else {
        research_sync::run_aiv1(iters);
    }
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

extern "C" __global__ __aicore__ void kernel_cluster_sync_probe(uint32_t iters);

extern "C" void launch_kernel_cluster_sync_probe(void* stream, uint32_t iters) {
    // 1 блок сетки MIX_AIC_1_2 физически диспетчеризует 1 AIC + 2 AIV
    kernel_cluster_sync_probe<<<1, nullptr, stream>>>(iters);
}

#endif