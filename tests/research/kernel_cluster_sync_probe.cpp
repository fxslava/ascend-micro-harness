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

// Mode-2 ids live in their own 0..15 space per core, independent of mode 4's
// (CrossCoreWaitFlag_ISASI.md: "the flagId of Mode 4 is independent of Modes
// 0, 1, and 2"). Both participants name the SAME id -- unlike mode 4, there is
// no +16 rebase for AIV1, because mode 2 addresses "the AIC and all AIVs"
// rather than one AIV. 8 and 9 stay clear of SyncAll's 11..14 and of the
// Matmul high-level API's worst case [0, 7] at four Matmul objects.
constexpr std::uint16_t kMode2StageReady  = 8u;   // both AIVs -> AIC (quorum)
constexpr std::uint16_t kMode2StepRetired = 9u;   // AIC -> both AIVs (broadcast)

// How many V_S round trips AIV1 burns before its announcement, so the two AIVs
// never arrive together and the AIC's quorum wait has a LATER arrival to be
// released by. Without an asymmetry a quorum and a single-sender latch look the
// same in the trace.
constexpr std::uint32_t kAiv1Lag = 64u;

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

// --------------------------------------------------------------------- mode 2
// The construction `turboquant_paged_decode_splitk_aic` needs, isolated: one
// RAW quorum (both AIVs announce, the AIC does ONE wait) and one WAR broadcast
// (the AIC announces, both AIVs wait), over recycled ids, with an x_S drain in
// front of every announcement and immediately behind every wait.
//
// The drain behind each wait is what makes the stall measurable: a
// CrossCoreWaitFlag on Ascend 950PR blocks only the issuance of the pipe named
// in its template argument, so the scalar `WaitFlag` behind it is the first
// instruction whose logged tick is its release tick
// (scripts/audit_mode4_tick_order.py).
//
// Pipe choice is NOT free. On __NPU_ARCH__ == 3510 the library gates both calls
// on IsSplitVectorPipe / IsSplitCubePipe (impl/basic_api/kernel_event.h:257,
// :263): an AIV may name PIPE_V, PIPE_MTE2 or PIPE_MTE3, an AIC may name
// PIPE_MTE1, PIPE_MTE2, PIPE_FIX or PIPE_M. A mode-2 call naming the other
// side's pipe compiles to NOTHING -- silently, with no diagnostic -- so AIV1
// announcing on PIPE_MTE3 while AIV0 announces on PIPE_V is deliberate here: it
// proves a heterogeneous pipe pair still forms one quorum.
__aicore__ inline void run_aic_mode2(std::uint32_t iters) {
    AscendC::PRINTF("[AIC] mode-2 leg started\n");
    for (std::uint32_t i = 0; i < iters; ++i) {
        AscendC::PRINTF("[AIC] iter %u: one quorum wait on id %u (both AIVs)...\n",
                        i, kMode2StageReady);
        AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(kMode2StageReady);
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtDrain);
        AscendC::PRINTF("[AIC] iter %u: quorum reached\n", i);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtDrain);
        AscendC::SetFlag<AscendC::HardEvent::M_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::M_S>(kEvtDrain);
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtDrain);
        AscendC::PRINTF("[AIC] iter %u: broadcasting release on id %u\n",
                        i, kMode2StepRetired);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(kMode2StepRetired);
    }
    AscendC::PRINTF("[AIC] mode-2 leg finished cleanly\n");
}

__aicore__ inline void run_aiv_mode2(std::uint32_t iters, std::uint32_t lag,
                                     std::uint32_t which) {
    AscendC::PRINTF("[AIV%u] mode-2 leg started\n", which);
    for (std::uint32_t i = 0; i < iters; ++i) {
        for (std::uint32_t burn = 0; burn < lag; ++burn) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtDrain);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtDrain);
        }
        AscendC::PRINTF("[AIV%u] iter %u: announcing id %u\n", which, i, kMode2StageReady);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtDrain);
        if (which == 0u) {
            AscendC::CrossCoreSetFlag<2, PIPE_V>(kMode2StageReady);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(kMode2StageReady);
        }

        AscendC::PRINTF("[AIV%u] iter %u: waiting broadcast id %u...\n",
                        which, i, kMode2StepRetired);
        AscendC::CrossCoreWaitFlag<2, PIPE_V>(kMode2StepRetired);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtDrain);
        AscendC::PRINTF("[AIV%u] iter %u: release observed\n", which, i);
    }
    AscendC::PRINTF("[AIV%u] mode-2 leg finished cleanly\n", which);
}

}  // namespace research_sync

extern "C" __global__ __aicore__ void kernel_cluster_sync_probe(uint32_t iters, uint32_t sync_mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();

    const std::uint32_t block = static_cast<std::uint32_t>(
        AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());

    // Ограничиваемся строго одним физическим блоком AI Core
    if (block != 0u) return;

    if (sync_mode == 2u) {
        if ASCEND_IS_AIC {
            research_sync::run_aic_mode2(iters);
        } else if (AscendC::GetSubBlockIdx() == 0) {
            research_sync::run_aiv_mode2(iters, 0u, 0u);
        } else {
            research_sync::run_aiv_mode2(iters, research_sync::kAiv1Lag, 1u);
        }
        return;
    }

    if ASCEND_IS_AIC {
        research_sync::run_aic(iters);
    } else if (AscendC::GetSubBlockIdx() == 0) {
        research_sync::run_aiv0(iters);
    } else {
        research_sync::run_aiv1(iters);
    }
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

extern "C" __global__ __aicore__ void kernel_cluster_sync_probe(uint32_t iters, uint32_t sync_mode);

extern "C" void launch_kernel_cluster_sync_probe(void* stream, uint32_t iters, uint32_t sync_mode) {
    // 1 блок сетки MIX_AIC_1_2 физически диспетчеризует 1 AIC + 2 AIV
    kernel_cluster_sync_probe<<<1, nullptr, stream>>>(iters, sync_mode);
}

#endif
