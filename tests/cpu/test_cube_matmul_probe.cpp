#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "../../include/sim_micro_testbed.hpp"
#include "../../routines/routine_cube_matmul.hpp"

using ascend_micro::sim::GuardedBuffer;
using ascend_micro::sim::assert_aligned;
using ascend_micro::sim::float_bits;
using ascend_micro::sim::float_to_half_rne;
using ascend_micro::sim::half_to_float;
using ascend_micro::sim::require;

template <int M>
struct MockCubeMatmulEngine {
    const std::uint16_t* l1_a;
    const std::uint16_t* l1_b;
    std::uint16_t* l0a;
    std::uint16_t* l0b;
    float* l0c;
    float* ub_output;
    unsigned loads = 0;
    unsigned mmads = 0;
    unsigned fixpipes = 0;
    bool l0_ready = false;
    bool l0c_ready = false;

    void load_a() { std::memcpy(l0a, l1_a, M * 16 * 2); ++loads; }
    void load_b() { std::memcpy(l0b, l1_b, 16 * 16 * 2); ++loads; }
    void complete_l0_loads()
    {
        require(loads == 2, "missing L0 stage");
        l0_ready = true;
    }
    void mma_dispatch()
    {
        require(l0_ready, "Cube dispatched before L0 load completion");
        for (int row = 0; row < M; ++row)
            for (int col = 0; col < 16; ++col) {
                float sum = 0;
                for (int k = 0; k < 16; ++k)
                    sum += half_to_float(l0a[row * 16 + k]) *
                           half_to_float(l0b[col * 16 + k]);
                l0c[row * 16 + col] = sum;
            }
        ++mmads;
        l0c_ready = true;
    }
    void fixpipe_to_ub()
    {
        require(l0c_ready, "Fixpipe before Cube");
        std::memcpy(ub_output, l0c, M * 16 * sizeof(float));
        ++fixpipes;
        l0c_ready = false;
    }
};

template <int M>
static void run_case(bool logits)
{
    constexpr std::size_t a_bytes = M * 16 * 2;
    constexpr std::size_t b_bytes = 16 * 16 * 2;
    constexpr std::size_t b_offset = a_bytes + ascend_micro::kCubeBankSkewBytes;
    GuardedBuffer<std::uint16_t> l1((b_offset + b_bytes) / 2);
    GuardedBuffer<std::uint16_t, 512> l0a(M * 16);
    GuardedBuffer<std::uint16_t, 512> l0b(16 * 16);
    GuardedBuffer<float, 1024> l0c(M * 16);
    GuardedBuffer<float> ub(M * 16);
    GuardedBuffer<std::uint16_t> a_nd(M * 16);
    GuardedBuffer<std::uint16_t> b_nd(16 * 16);
    auto* l1_a = l1.data();
    auto* l1_b = reinterpret_cast<std::uint16_t*>(
        reinterpret_cast<std::byte*>(l1.data()) + b_offset);
    assert_aligned(l1_a);
    assert_aligned(l1_b);
    assert_aligned(l0a.data(), 512);
    assert_aligned(l0b.data(), 512);
    assert_aligned(l0c.data(), 1024);
    assert_aligned(ub.data());
    require((reinterpret_cast<std::uintptr_t>(l1_b) -
             reinterpret_cast<std::uintptr_t>(l1_a) - a_bytes) == 32,
            "L1 bank skew missing");
    for (int row = 0; row < M; ++row)
        for (int k = 0; k < 16; ++k)
            a_nd[row * 16 + k] = float_to_half_rne(
                float(((row * 7 + k * 3) % 13) - 6) / 32.0f);
    for (int k = 0; k < 16; ++k)
        for (int col = 0; col < 16; ++col)
            b_nd[k * 16 + col] = float_to_half_rne(
                float(((k * 5 + col * 11) % 17) - 8) / 64.0f);
    std::memcpy(l1_a, a_nd.data(), a_bytes);
    for (int col = 0; col < 16; ++col)
        for (int k = 0; k < 16; ++k)
            l1_b[col * 16 + k] = logits ? b_nd[col * 16 + k]
                                          : b_nd[k * 16 + col];
    MockCubeMatmulEngine<M> engine{l1_a, l1_b, l0a.data(), l0b.data(),
                                    l0c.data(), ub.data()};
    ascend_micro::cube_matmul_stage_l0(engine);
    engine.complete_l0_loads();  // Caller-owned completion boundary.
    ascend_micro::cube_matmul_compute_dump(engine);
    require(engine.loads == 2 && engine.mmads == 1 && engine.fixpipes == 1,
            "wrong Cube instruction count");
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < 16; ++col) {
            float golden = 0;
            for (int k = 0; k < 16; ++k)
                golden += half_to_float(a_nd[row * 16 + k]) *
                          half_to_float(logits ? b_nd[col * 16 + k]
                                                : b_nd[k * 16 + col]);
            require(float_bits(ub[row * 16 + col]) == float_bits(golden),
                    "Cube output mismatch");
        }
    ASSERT_CANARY_INTACT(l1);
    ASSERT_CANARY_INTACT(l0a);
    ASSERT_CANARY_INTACT(l0b);
    ASSERT_CANARY_INTACT(l0c);
    ASSERT_CANARY_INTACT(ub);
    ASSERT_CANARY_INTACT(a_nd);
    ASSERT_CANARY_INTACT(b_nd);
}

int main()
{
    try {
        const auto start = std::chrono::steady_clock::now();
        run_case<16>(true);
        run_case<16>(false);
        run_case<32>(true);
        run_case<32>(false);
        const auto us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "PASS cube matmul: logits/context 16x16 and 32x16, "
                     "bit-identical FP32, 64-byte canaries intact\n";
        std::cout << "host probe: " << us << " us total (includes staging/oracle/checks)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL cube matmul: " << error.what() << '\n';
        return 1;
    }
}
