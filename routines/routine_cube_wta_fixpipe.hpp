#pragma once

#include <cstdint>
#include <type_traits>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#define ASCEND_WTA_INLINE __aicore__ inline
#else
#define ASCEND_WTA_INLINE inline
#endif

namespace ascend_micro {

// Arch3510, one AIC, sixteen independent rows of sixteen competitors.
// Computes ReLU(V W), with FP16 feedback for steps 0..T-2 and scalar B8
// quantization on step T-1. W is caller supplied, resident in L0B (Zn).
// This is an inhibition primitive, NOT an unconditional argmax/one-hot map.
// Neither convergence in T steps nor a surviving value of exactly 1 follows
// from the recurrence. Ties, negative inputs and quantization can erase winners.
//
// C++17 has no float NTTP: ScaleBits is the IEEE FP32 bit pattern of the scale.
// The hardware ignores its low 13 mantissa bits. Offset is always zero.
template <unsigned T, std::uint32_t ScaleBits = 0x43000000u,
          class Output = std::int8_t>
struct WtaConfig {
    static_assert(T > 0, "WTA needs at least one Mmad");
    static_assert(std::is_same<Output, std::int8_t>::value ||
                  std::is_same<Output, std::uint8_t>::value, "B8 output required");
    static_assert((ScaleBits & 0x80000000u) == 0 &&
                  (ScaleBits & 0x7f800000u) != 0 &&
                  (ScaleBits & 0x7f800000u) != 0x7f800000u,
                  "scale must be positive, finite and normal");
    using output_type = Output;
    static constexpr unsigned steps = T;
    static constexpr std::uint32_t scale_bits = ScaleBits & 0xffffe000u;
    static constexpr std::uint64_t deq_scalar = std::uint64_t(scale_bits) |
        (std::is_same<Output, std::int8_t>::value ? (1ull << 46) : 0ull);
    static constexpr unsigned elements = 16 * 16;
    static constexpr unsigned feedback_bytes = elements * 2;
    // Int8-Channel-Merge.md: an odd final group of 16 channels is NOT merged.
    // N=16 therefore stays a packed 16x16 byte tile (verified on CAModel).
    static constexpr unsigned output_pitch = 16;
    static constexpr unsigned output_bytes = 16 * output_pitch;
};

// Synchronization-free leaves, matching the micro-harness engine interface.
template <class Engine>
ASCEND_WTA_INLINE void cube_wta_load(Engine& e) { e.load_a(); }
template <class Engine>
ASCEND_WTA_INLINE void cube_wta_mmad(Engine& e) { e.mma_dispatch(); }
template <bool Final, class Engine>
ASCEND_WTA_INLINE void cube_wta_fixpipe(Engine& e)
{
    e.template fixpipe<Final>();
}

namespace wta_detail {
template <unsigned Step, class Config, class Engine, class Sync>
ASCEND_WTA_INLINE void step(Engine& e, Sync& sync)
{
    sync.wait_feedback();
    cube_wta_load(e);
    sync.operands_ready();
    cube_wta_mmad(e);
    sync.result_ready();
    cube_wta_fixpipe<Step + 1 == Config::steps>(e);
    sync.publish_feedback();
    if constexpr (Step + 1 < Config::steps) step<Step + 1, Config>(e, sync);
}
}  // namespace wta_detail

// Explicit scheduling driver, NOT a synchronization-free leaf. All steps are
// instantiated at compile time; no tensor-dependent or runtime loop control.
// Chain: FIX_MTE1 -> LoadData -> MTE1_M -> Mmad -> M_FIX -> Fixpipe.
// This serial recurrence also orders all WAR edges: previous Fixpipe retires
// before either L0A or L0C is reused, and Mmad retires before C1 is overwritten.
// FIX_MTE1 has 1+T sets and T+1 waits. Each forward route has T pairs.
template <class Config, class Engine, class Sync>
ASCEND_WTA_INLINE void cube_wta_run(Engine& e, Sync& sync)
{
    sync.publish_feedback();  // Initial C1 ready; caller completed its producer.
    wta_detail::step<0, Config>(e, sync);
    sync.wait_feedback();     // Drain the last feedback token.
    sync.finish();            // Complete Fixpipe before returning to the caller.
}

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#if defined(__NPU_ARCH__) && __NPU_ARCH__ != 3510
#error "cube_wta_fixpipe requires Arch3510"
#endif

template <class Config>
struct NativeCubeWtaEngine {
    // Caller-owned nonaliasing buffers. L0A/B: 512-byte aligned, 256 half;
    // L0C: >=64-byte aligned, 256 float; C1 state/result: >=32-byte aligned,
    // 512 state bytes and 256 result bytes. L0B contains W in Zn; C1 state V0 in Nz.
    // Caller orders the C1 producer before MTE1. Weight loads may be queued on
    // MTE1 before entry (the first MTE1_M fences them), or already completed.
    AscendC::LocalTensor<half> l0a;
    AscendC::LocalTensor<half> l0b;
    AscendC::LocalTensor<float> l0c;
    AscendC::LocalTensor<half> state;
    AscendC::LocalTensor<typename Config::output_type> victory;

    ASCEND_WTA_INLINE void load_a()
    {
        AscendC::LoadData2DParams p;
        p.repeatTimes = 1;
        p.srcStride = 1;
        p.ifTranspose = false;
        AscendC::LoadData(l0a, state, p);
    }
    ASCEND_WTA_INLINE void mma_dispatch()
    {
        AscendC::MmadParams p;
        p.m = 16; p.n = 16; p.k = 16;
        p.cmatrixInitVal = true;
        p.unitFlag = 0;  // Explicit M_FIX; no implicit event allocation.
        AscendC::Mmad(l0c, l0a, l0b, p);
    }
    template <bool Final>
    ASCEND_WTA_INLINE void fixpipe()
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> p;
        p.mSize = 16; p.nSize = 16; p.srcStride = 16;
        p.dstStride = 16;  // ELEMENTS on Arch3510, not 32-byte datablocks.
        p.reluEn = true;
        p.unitFlag = 0;
        if constexpr (Final) {
            p.quantPre = QuantMode_t::QF322B8_PRE;
            p.deqScalar = Config::deq_scalar;
            AscendC::Fixpipe<typename Config::output_type, float, AscendC::CFG_NZ>(
                victory, l0c, p);
        } else {
            p.quantPre = QuantMode_t::F322F16;
            AscendC::Fixpipe<half, float, AscendC::CFG_NZ>(state, l0c, p);
        }
    }
};

// The caller reserves this ID on these four routes for the driver's lifetime.
template <unsigned EventId = 0>
struct NativeCubeWtaSync {
    static_assert(EventId <= 5, "event IDs 6 and 7 are reserved");
    ASCEND_WTA_INLINE void publish_feedback()
    { AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EventId); }
    ASCEND_WTA_INLINE void wait_feedback()
    { AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EventId); }
    ASCEND_WTA_INLINE void operands_ready()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EventId);
    }
    ASCEND_WTA_INLINE void result_ready()
    {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EventId);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EventId);
    }
    ASCEND_WTA_INLINE void finish()
    {
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(EventId);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(EventId);
    }
};
#endif

}  // namespace ascend_micro
#undef ASCEND_WTA_INLINE
