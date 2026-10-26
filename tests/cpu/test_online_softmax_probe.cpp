#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "../../include/sim_micro_testbed.hpp"
#include "../../routines/routine_vector_online_softmax.hpp"

using ascend_micro::sim::GuardedBuffer;
using ascend_micro::sim::assert_aligned;
using ascend_micro::sim::float_bits;
using ascend_micro::sim::require;

struct MockOnlineEngine {
    const float* scores;
    const float* v_by_channel;
    const std::uint32_t* offsets;
    float m_prev;
    float l_prev;
    const float* acc_prev;
    float m_new = 0;
    float alpha = 0;
    float l_new = 0;
    float p[16]{};
    float acc_new[16]{};
    unsigned stages = 0;

    void update_max_and_alpha()
    {
        float tile_max = scores[0];
        for (int i = 1; i < 16; ++i) tile_max = std::max(tile_max, scores[i]);
        m_new = std::max(m_prev, tile_max);
        alpha = std::exp(m_prev - m_new);
        ++stages;
    }
    void exponentiate_scores()
    {
        for (int i = 0; i < 16; ++i) p[i] = std::exp(scores[i] - m_new);
        ++stages;
    }
    void update_denominator()
    {
        float sum = 0;
        for (float value : p) sum += value;
        l_new = l_prev * alpha + sum;
        ++stages;
    }
    void update_accumulator()
    {
        for (int col = 0; col < 16; ++col) {
            float sum = 0;
            for (int row = 0; row < 16; ++row) {
                const unsigned p_index = offsets[col * 16 + row] / sizeof(float);
                sum += p[p_index] * v_by_channel[col * 16 + row];
            }
            acc_new[col] = acc_prev[col] * alpha + sum;
        }
        ++stages;
    }
};

int main()
{
    try {
        GuardedBuffer<float> scores(16);
        GuardedBuffer<float> v_by_channel(256);
        GuardedBuffer<std::uint32_t> offsets(256);
        GuardedBuffer<float> m_state(8);
        GuardedBuffer<float> l_state(8);
        GuardedBuffer<float> acc_state(16);
        GuardedBuffer<float> scratch(512);
        for (const void* p : {static_cast<const void*>(scores.data()),
                              static_cast<const void*>(v_by_channel.data()),
                              static_cast<const void*>(offsets.data()),
                              static_cast<const void*>(m_state.data()),
                              static_cast<const void*>(l_state.data()),
                              static_cast<const void*>(acc_state.data()),
                              static_cast<const void*>(scratch.data())})
            assert_aligned(p);
        for (int col = 0; col < 16; ++col)
            for (int row = 0; row < 16; ++row)
                offsets[col * 16 + row] = row * sizeof(float);
        m_state[0] = -std::numeric_limits<float>::infinity();
        l_state[0] = 0;
        for (int col = 0; col < 16; ++col) acc_state[col] = 0;
        float golden_m = m_state[0];
        float golden_l = 0;
        float golden_acc[16]{};
        const auto start = std::chrono::steady_clock::now();
        for (int tile = 0; tile < 4; ++tile) {
            for (int row = 0; row < 16; ++row)
                scores[row] = float(((row * 3 + tile * 7) % 19) - 9) / 4.0f;
            for (int col = 0; col < 16; ++col)
                for (int row = 0; row < 16; ++row)
                    v_by_channel[col * 16 + row] =
                        float(((col * 5 + row * 11 + tile) % 23) - 11) / 32.0f;
            MockOnlineEngine engine{scores.data(), v_by_channel.data(), offsets.data(),
                                    m_state[0], l_state[0], acc_state.data()};
            ascend_micro::vector_online_softmax(engine);
            require(engine.stages == 4, "wrong online stage count");

            float tile_max = scores[0];
            for (int row = 1; row < 16; ++row)
                tile_max = std::max(tile_max, scores[row]);
            const float new_m = std::max(golden_m, tile_max);
            const float alpha = std::exp(golden_m - new_m);
            float p[16];
            float p_sum = 0;
            for (int row = 0; row < 16; ++row) {
                p[row] = std::exp(scores[row] - new_m);
                p_sum += p[row];
            }
            const float new_l = golden_l * alpha + p_sum;
            for (int col = 0; col < 16; ++col) {
                float weighted_sum = 0;
                for (int row = 0; row < 16; ++row)
                    weighted_sum += p[row] * v_by_channel[col * 16 + row];
                golden_acc[col] = golden_acc[col] * alpha + weighted_sum;
                require(float_bits(engine.acc_new[col]) == float_bits(golden_acc[col]),
                        "online Acc mismatch");
                acc_state[col] = engine.acc_new[col];
            }
            require(float_bits(engine.m_new) == float_bits(new_m), "online m mismatch");
            require(float_bits(engine.l_new) == float_bits(new_l), "online l mismatch");
            golden_m = m_state[0] = new_m;
            golden_l = l_state[0] = new_l;
            ASSERT_CANARY_INTACT(scores);
            ASSERT_CANARY_INTACT(v_by_channel);
            ASSERT_CANARY_INTACT(offsets);
            ASSERT_CANARY_INTACT(m_state);
            ASSERT_CANARY_INTACT(l_state);
            ASSERT_CANARY_INTACT(acc_state);
            ASSERT_CANARY_INTACT(scratch);
        }
        const auto us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "PASS online softmax: 4 streaming tiles x 16 channels, "
                     "bit-identical host recurrence, 64-byte canaries intact\n";
        std::cout << "host probe: " << us << " us total (includes oracle/checks)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL online softmax: " << error.what() << '\n';
        return 1;
    }
}
