#pragma once

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#endif

namespace ascend_micro {

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#define ASCEND_MICRO_ONLINE_INLINE __aicore__ inline
#else
#define ASCEND_MICRO_ONLINE_INLINE inline
#endif

constexpr std::uint32_t kOnlineTile = 16;

// All state and scratch operands are UB resident. The four phases only issue
// vector instructions; the caller owns input readiness and output visibility.
template <class Engine>
ASCEND_MICRO_ONLINE_INLINE void vector_online_softmax(Engine& engine)
{
    engine.update_max_and_alpha();
    engine.exponentiate_scores();
    engine.update_denominator();
    engine.update_accumulator();
}

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

struct NativeOnlineSoftmaxEngine {
    AscendC::LocalTensor<float> scores;
    AscendC::LocalTensor<float> v_by_channel;
    AscendC::LocalTensor<std::uint32_t> p_repeat_offsets;
    AscendC::LocalTensor<float> m_prev;
    AscendC::LocalTensor<float> l_prev;
    AscendC::LocalTensor<float> acc_prev;
    AscendC::LocalTensor<float> m_new;
    AscendC::LocalTensor<float> l_new;
    AscendC::LocalTensor<float> acc_new;
    AscendC::LocalTensor<float> tile_max;
    AscendC::LocalTensor<float> alpha_delta;
    AscendC::LocalTensor<float> alpha;
    AscendC::LocalTensor<float> m_broadcast;
    AscendC::LocalTensor<float> centered;
    AscendC::LocalTensor<float> p;
    AscendC::LocalTensor<float> p_sum;
    AscendC::LocalTensor<float> l_scaled;
    AscendC::LocalTensor<float> p_repeated;
    AscendC::LocalTensor<float> weighted;
    AscendC::LocalTensor<float> weighted_sum;
    AscendC::LocalTensor<float> alpha_broadcast;
    AscendC::LocalTensor<float> acc_scaled;
    AscendC::LocalTensor<float> reduce_scratch;

    __aicore__ inline void update_max_and_alpha()
    {
        AscendC::ReduceMax(tile_max, scores, reduce_scratch, kOnlineTile);
        AscendC::Max(m_new, m_prev, tile_max, 1);
        AscendC::Sub(alpha_delta, m_prev, m_new, 1);
        AscendC::Exp(alpha, alpha_delta, 1);
    }
    __aicore__ inline void exponentiate_scores()
    {
        AscendC::Duplicate(m_broadcast, m_new, kOnlineTile);
        AscendC::Sub(centered, scores, m_broadcast, kOnlineTile);
        AscendC::Exp(p, centered, kOnlineTile);
    }
    __aicore__ inline void update_denominator()
    {
        AscendC::ReduceSum(p_sum, p, reduce_scratch, kOnlineTile);
        AscendC::Mul(l_scaled, l_prev, alpha, 1);
        AscendC::Add(l_new, l_scaled, p_sum, 1);
    }
    __aicore__ inline void update_accumulator()
    {
        // Caller supplies 16 columns of V contiguously and 256 byte offsets
        // repeating 0,4,...,60 for each column. No scalar index generation.
        AscendC::Gather(p_repeated, p, p_repeat_offsets, 0, 256);
        AscendC::Mul(weighted, p_repeated, v_by_channel, 256);
        AscendC::ReduceRepeat<AscendC::ReduceType::SUM, float>(
            weighted_sum, weighted, 16, 16, 1, 1, 2,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::Duplicate(alpha_broadcast, alpha, kOnlineTile);
        AscendC::Mul(acc_scaled, acc_prev, alpha_broadcast, kOnlineTile);
        AscendC::Add(acc_new, acc_scaled, weighted_sum, kOnlineTile);
    }
};

#endif

#undef ASCEND_MICRO_ONLINE_INLINE
}  // namespace ascend_micro
