#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "../../include/sim_micro_testbed.hpp"
#include "../../routines/routine_vector_normalize_store.hpp"

using ascend_micro::sim::GuardedBuffer;
using ascend_micro::sim::assert_aligned;
using ascend_micro::sim::float_to_bfloat16_rne;
using ascend_micro::sim::float_to_half_rne;
using ascend_micro::sim::require;

template <bool Bf16>
struct MockNormalizeStoreEngine {
    const float* acc;
    const float* l;
    float* l_broadcast;
    float* normalized;
    std::uint16_t* ub_output;
    std::uint16_t* gm_output;
    unsigned vector_ops = 0;
    unsigned dma_ops = 0;
    bool vector_complete = false;

    void broadcast_denominator(std::uint32_t count)
    {
        for (std::uint32_t i = 0; i < count; ++i) l_broadcast[i] = l[0];
        ++vector_ops;
    }
    void divide(std::uint32_t count)
    {
        for (std::uint32_t i = 0; i < count; ++i)
            normalized[i] = acc[i] / l_broadcast[i];
        ++vector_ops;
    }
    void cast_output(std::uint32_t count)
    {
        for (std::uint32_t i = 0; i < count; ++i)
            ub_output[i] = Bf16 ? float_to_bfloat16_rne(normalized[i])
                                : float_to_half_rne(normalized[i]);
        ++vector_ops;
    }
    void complete_vector() { vector_complete = true; }
    void mte3_store(std::uint32_t count)
    {
        require(vector_complete, "MTE3 dispatched before vector completion");
        std::memcpy(gm_output, ub_output, count * sizeof(std::uint16_t));
        ++dma_ops;
    }
};

template <bool Bf16>
static void run_case(std::size_t count, float denominator)
{
    GuardedBuffer<float> acc(count);
    GuardedBuffer<float> l(8);
    GuardedBuffer<float> l_broadcast(count);
    GuardedBuffer<float> normalized(count);
    GuardedBuffer<std::uint16_t> ub_output(count);
    GuardedBuffer<std::uint16_t> gm_output(count);
    l[0] = denominator;
    for (std::size_t i = 0; i < count; ++i)
        acc[i] = float(int(i * 17u % 43u) - 21) / 8.0f;
    for (const void* p : {static_cast<const void*>(acc.data()),
                          static_cast<const void*>(l.data()),
                          static_cast<const void*>(l_broadcast.data()),
                          static_cast<const void*>(normalized.data()),
                          static_cast<const void*>(ub_output.data()),
                          static_cast<const void*>(gm_output.data())})
        assert_aligned(p);
    MockNormalizeStoreEngine<Bf16> engine{
        acc.data(), l.data(), l_broadcast.data(), normalized.data(),
        ub_output.data(), gm_output.data()};
    ascend_micro::vector_normalize(engine, std::uint32_t(count));
    engine.complete_vector();  // Caller-owned V->MTE3 boundary.
    ascend_micro::vector_store(engine, std::uint32_t(count));
    require(engine.vector_ops == 3 && engine.dma_ops == 1,
            "wrong vector/MTE3 dispatch count");
    for (std::size_t i = 0; i < count; ++i) {
        const float golden = acc[i] / denominator;
        const std::uint16_t expected = Bf16 ? float_to_bfloat16_rne(golden)
                                            : float_to_half_rne(golden);
        require(gm_output[i] == expected, "normalized output mismatch");
    }
    ASSERT_CANARY_INTACT(acc);
    ASSERT_CANARY_INTACT(l);
    ASSERT_CANARY_INTACT(l_broadcast);
    ASSERT_CANARY_INTACT(normalized);
    ASSERT_CANARY_INTACT(ub_output);
    ASSERT_CANARY_INTACT(gm_output);
}

int main()
{
    try {
        const auto start = std::chrono::steady_clock::now();
        for (float denominator : {2.0f, 3.0f, 16.0f}) {
            run_case<false>(16, denominator);
            run_case<true>(16, denominator);
            run_case<false>(32, denominator);
            run_case<true>(32, denominator);
        }
        const auto us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "PASS normalize/store: 12 FP16/BF16 cases, "
                     "bit-identical host oracle, 64-byte canaries intact\n";
        std::cout << "host probe: " << us << " us total (includes oracle/checks)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL normalize/store: " << error.what() << '\n';
        return 1;
    }
}
