// Ascend 950PR / dav-3510 synchronization probe. This is deliberately not a
// softmax implementation: each diagnostic tile represents one independent
// head, so its single GM store is its final outflow. A production attention
// kernel must keep (m, l, Acc) in Vector/UB across all key tiles and perform
// m'=max(m,max(S)), alpha=exp(m-m'), P=exp(S-m'),
// l'=l*alpha+sum(P), Acc'=Acc*alpha+P*V before its one final store.
// No partial result or Split-K workspace exists in this probe.

#include <cstdint>

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#define PROBE_INLINE __aicore__ inline
#else
#define PROBE_INLINE inline
#endif

namespace ascend_micro::modulo3 {

constexpr unsigned kStageDepth = 3;
constexpr unsigned kBankSkewBytes = 32u;
constexpr unsigned kGuardBytes = 64u;
constexpr unsigned kPayloadWords = 32u;
constexpr unsigned kPayloadBytes = kPayloadWords * sizeof(std::uint32_t);
constexpr unsigned kSlotStrideBytes =
    kGuardBytes + kPayloadBytes + kGuardBytes + kBankSkewBytes;
constexpr unsigned kInputBaseBytes = 0u;
constexpr unsigned kOutputBaseBytes = kStageDepth * kSlotStrideBytes + kBankSkewBytes;
constexpr unsigned kUbBytes = kOutputBaseBytes + kStageDepth * kSlotStrideBytes;
constexpr std::uint32_t kInputCanary = 0x55AA55AAu;
constexpr std::uint32_t kOutputCanary = 0xDEADBEEFu;
constexpr std::uint32_t kCanaryDelta = kOutputCanary - kInputCanary;

static_assert(kPayloadBytes % 32u == 0u && kSlotStrideBytes % 32u == 0u);
static_assert(kSlotStrideBytes % 256u != 0u);
static_assert(kOutputBaseBytes % 32u == 0u && kUbBytes <= 192u * 1024u);

enum class Route : unsigned { Mte2V, VMte2, VMte3, Mte3V };

// One bit per (route, ID), not a counting semaphore. Every set is consumed
// before the next set on that same route/ID. IDs are scoped independently to
// each route; all four routes use precisely 0, 1, and 2.
template <unsigned Slot, class Engine>
PROBE_INLINE void DummyMte2Ingress(Engine& engine, unsigned tile)
{
    static_assert(Slot < kStageDepth);
    engine.template wait<Route::VMte2, Slot>();
    engine.template ingress<Slot>(tile);
    engine.template set<Route::Mte2V, Slot>();
}

template <unsigned Slot, class Engine>
PROBE_INLINE void DummyVectorCompute(Engine& engine, unsigned tile)
{
    static_assert(Slot < kStageDepth);
    engine.template wait<Route::Mte2V, Slot>();
    engine.template vector_compute<Slot>(tile);
    engine.template set<Route::VMte2, Slot>();
    engine.template wait<Route::Mte3V, Slot>();
    engine.template set<Route::VMte3, Slot>();
}

template <unsigned Slot, class Engine>
PROBE_INLINE void DummyMte3Egress(Engine& engine, unsigned tile)
{
    static_assert(Slot < kStageDepth);
    engine.template wait<Route::VMte3, Slot>();
    engine.template egress<Slot>(tile);
    engine.template set<Route::Mte3V, Slot>();
}

// The first call is the only complete column at N=3. The other two calls are
// compile-time ramp-down columns; there is no runtime modulo or hot-loop guard.
template <unsigned InSlot, unsigned VecSlot, unsigned OutSlot,
          bool HasIngress, bool HasVector, class Engine>
PROBE_INLINE void StepColumn(Engine& engine, unsigned base)
{
    static_assert(InSlot < kStageDepth && VecSlot < kStageDepth &&
                  OutSlot < kStageDepth);
    if constexpr (HasIngress) DummyMte2Ingress<InSlot>(engine, base + 2u);
    if constexpr (HasVector) DummyVectorCompute<VecSlot>(engine, base + 1u);
    DummyMte3Egress<OutSlot>(engine, base);
}

template <class Engine>
PROBE_INLINE void RunThreeTiles(Engine& engine)
{
    // Feedback cycles start with one free-input and one free-output token
    // per slot: sum(tokens on each such cycle) >= 1.
    engine.template set<Route::VMte2, 0>();
    engine.template set<Route::VMte2, 1>();
    engine.template set<Route::VMte2, 2>();
    engine.template set<Route::Mte3V, 0>();
    engine.template set<Route::Mte3V, 1>();
    engine.template set<Route::Mte3V, 2>();

    // Ramp up: two MTE2 tiles and the first Vector tile.
    DummyMte2Ingress<0>(engine, 0);
    DummyMte2Ingress<1>(engine, 1);
    DummyVectorCompute<0>(engine, 0);

    // Modulo schedule: 2/1/0 is the complete three-stage column.
    StepColumn<2, 1, 0, true, true>(engine, 0);
    StepColumn<0, 2, 1, false, true>(engine, 1);
    StepColumn<1, 0, 2, false, false>(engine, 2);

    // Every final Vector and MTE3 feedback set is consumed. These waits also
    // wait for the final queued operations to retire before kernel return.
    engine.template wait<Route::VMte2, 0>();
    engine.template wait<Route::VMte2, 1>();
    engine.template wait<Route::VMte2, 2>();
    engine.template wait<Route::Mte3V, 0>();
    engine.template wait<Route::Mte3V, 1>();
    engine.template wait<Route::Mte3V, 2>();
}

#if defined(__CCE_AICORE__)
struct NativeEngine {
    AscendC::GlobalTensor<std::int32_t> input;
    AscendC::GlobalTensor<std::int32_t> output;

    PROBE_INLINE NativeEngine(__gm__ std::int32_t* in,
                              __gm__ std::int32_t* out)
    {
        input.SetGlobalBuffer(in);
        output.SetGlobalBuffer(out);
    }

    template <Route R, unsigned Slot>
    PROBE_INLINE void set()
    {
        if constexpr (R == Route::Mte2V)
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(Slot);
        else if constexpr (R == Route::VMte2)
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(Slot);
        else if constexpr (R == Route::VMte3)
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Slot);
        else
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(Slot);
    }

    template <Route R, unsigned Slot>
    PROBE_INLINE void wait()
    {
        if constexpr (R == Route::Mte2V)
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(Slot);
        else if constexpr (R == Route::VMte2)
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(Slot);
        else if constexpr (R == Route::VMte3)
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Slot);
        else
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(Slot);
    }

    template <unsigned Slot>
    PROBE_INLINE auto in_tensor() const
    {
        constexpr unsigned offset = kInputBaseBytes + Slot * kSlotStrideBytes + kGuardBytes;
        return AscendC::LocalTensor<std::int32_t>(
            AscendC::TPosition::VECCALC, offset, kPayloadWords);
    }

    template <unsigned Slot>
    PROBE_INLINE auto out_tensor() const
    {
        constexpr unsigned offset = kOutputBaseBytes + Slot * kSlotStrideBytes + kGuardBytes;
        return AscendC::LocalTensor<std::int32_t>(
            AscendC::TPosition::VECCALC, offset, kPayloadWords);
    }

    template <unsigned Slot>
    PROBE_INLINE void ingress(unsigned tile)
    {
        AscendC::DataCopy(in_tensor<Slot>(), input[tile * kPayloadWords],
                          kPayloadWords);
    }

    template <unsigned Slot>
    PROBE_INLINE void vector_compute(unsigned)
    {
        // One vector add reads the ingress canary and produces DEADBEEF. It
        // does not synchronize the Scalar core through LocalTensor::GetValue.
        AscendC::Adds(out_tensor<Slot>(), in_tensor<Slot>(),
                      static_cast<std::int32_t>(kCanaryDelta),
                      static_cast<int32_t>(kPayloadWords));
    }

    template <unsigned Slot>
    PROBE_INLINE void egress(unsigned tile)
    {
        AscendC::DataCopy(output[tile * kPayloadWords], out_tensor<Slot>(),
                          kPayloadWords);
    }
};
#endif

}  // namespace ascend_micro::modulo3

#if defined(__CCE_AICORE__)
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache(
    __gm__ std::int32_t* input, __gm__ std::int32_t* output)
{
    AscendC::InitSocState();
    ascend_micro::modulo3::NativeEngine engine(input, output);
    ascend_micro::modulo3::RunThreeTiles(engine);
}
#elif defined(ASCEND_MICRO_CANN_LAUNCH)
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache(
    __gm__ std::int32_t* input, __gm__ std::int32_t* output);

extern "C" void launch_turboquant_reshape_and_cache(void* stream,
                                                      std::int32_t* input,
                                                      std::int32_t* output)
{
    turboquant_reshape_and_cache<<<1, nullptr, stream>>>(input, output);
}
#endif

#undef PROBE_INLINE
