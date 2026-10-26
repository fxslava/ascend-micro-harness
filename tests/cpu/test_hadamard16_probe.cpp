#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "../../include/sim_micro_testbed.hpp"

using ascend_micro::OutflowMode;
using ascend_micro::cube_hadamard16;
using ascend_micro::kFp16TileBytes;
using ascend_micro::kL1BOffsetBytes;
using ascend_micro::sim::GuardedBuffer;
using ascend_micro::sim::MockCubeEngine;
using ascend_micro::sim::assert_aligned;
using ascend_micro::sim::float_bits;
using ascend_micro::sim::float_to_bfloat16_rne;
using ascend_micro::sim::float_to_half_rne;
using ascend_micro::sim::golden_hadamard16;
using ascend_micro::sim::require;

static std::uint16_t hadamard_coefficient(unsigned row, unsigned column)
{
    // Four H2 Kronecker factors, generated once on the host.
    int product = 1;
    for (unsigned bit = 0; bit < 4; ++bit)
        product *= ((row & column & (1u << bit)) != 0) ? -1 : 1;
    return float_to_half_rne(float(product));
}

static void fill_input(std::uint16_t* x, unsigned case_number)
{
    std::uint32_t state = 0x13579bdfu + case_number;
    for (unsigned i = 0; i < 256; ++i) {
        float value = 0;
        if (case_number == 1) value = (i % 16 == 3) ? 1.0f : 0.0f;
        if (case_number == 2) value = float(int(i % 31) - 15) / 32.0f;
        if (case_number == 3) value = float(int(i % 17) - 8) / 2048.0f;
        if (case_number == 4) {
            state = state * 1664525u + 1013904223u;
            value = float(int((state >> 20) & 2047u) - 1024) / 2048.0f;
        }
        x[i] = float_to_half_rne(value);
    }
}

template <OutflowMode Mode, class Output>
static void run_mode(const std::uint16_t* l0a, const std::uint16_t* l0b,
                     const std::array<float, 256>& golden, float* l0c,
                     GuardedBuffer<Output>& output)
{
    MockCubeEngine engine{l0a, l0b, l0c, output.data()};
    cube_hadamard16<Mode>(engine);
    require(engine.mmads == 1 && engine.fixpipes == 1 && !engine.pending,
            "incorrect Cube/Fixpipe dispatch count");
    for (unsigned i = 0; i < 256; ++i) {
        if constexpr (Mode == OutflowMode::Fp32)
            require(float_bits(output[i]) == float_bits(golden[i]),
                    "FP32 golden mismatch");
        else if constexpr (Mode == OutflowMode::Fp16)
            require(output[i] == float_to_half_rne(golden[i]),
                    "FP16 golden mismatch");
        else
            require(output[i] == float_to_bfloat16_rne(golden[i]),
                    "BF16 golden mismatch");
    }
    ASSERT_CANARY_INTACT(output);
}

int main()
{
    try {
        {
            GuardedBuffer<std::uint32_t> guard_self_test(4);
            auto* tail = reinterpret_cast<std::byte*>(guard_self_test.data()) +
                         guard_self_test.size() * sizeof(std::uint32_t);
            *tail ^= std::byte{1};
            bool detected = false;
            try { ASSERT_CANARY_INTACT(guard_self_test); }
            catch (const std::runtime_error&) { detected = true; }
            *tail ^= std::byte{1};
            require(detected, "canary self-test did not detect corruption");
            ASSERT_CANARY_INTACT(guard_self_test);
        }
        GuardedBuffer<std::uint16_t> gm_x(256);
        GuardedBuffer<std::uint16_t> gm_h(256);
        GuardedBuffer<std::uint16_t> ub_h(256);
        GuardedBuffer<std::uint16_t> l1((kL1BOffsetBytes + kFp16TileBytes) / 2);
        GuardedBuffer<std::uint16_t, 512> l0a(256);
        GuardedBuffer<std::uint16_t, 512> l0b(256);
        GuardedBuffer<float> l0c(256);
        GuardedBuffer<float> out_fp32(256);
        GuardedBuffer<std::uint16_t> out_fp16(256);
        GuardedBuffer<std::uint16_t> out_bf16(256);

        auto* l1_x = l1.data();
        auto* l1_h = reinterpret_cast<std::uint16_t*>(
            reinterpret_cast<std::byte*>(l1.data()) + kL1BOffsetBytes);
        assert_aligned(l1_x);
        assert_aligned(l1_h);
        assert_aligned(l0a.data(), 512);
        assert_aligned(l0b.data(), 512);
        assert_aligned(l0c.data(), 64);
        require((reinterpret_cast<std::uintptr_t>(l1_x) / 32) % 16 !=
                    (reinterpret_cast<std::uintptr_t>(l1_h) / 32) % 16,
                "L1 operand bank skew missing");

        for (unsigned row = 0; row < 16; ++row)
            for (unsigned column = 0; column < 16; ++column)
                ub_h[row * 16 + column] = hadamard_coefficient(row, column);
        std::memcpy(gm_h.data(), ub_h.data(), kFp16TileBytes);

        const auto start = std::chrono::steady_clock::now();
        constexpr unsigned kCases = 5;
        for (unsigned test_case = 0; test_case < kCases; ++test_case) {
            fill_input(gm_x.data(), test_case);
            const auto golden = golden_hadamard16(gm_x.data());

            // Host model of GM->L1 ND2NZ/ND2ZN and L1->L0A/L0B.
            // For one fractal, NZ is row-major and ZN is column-major.
            std::memcpy(l1_x, gm_x.data(), kFp16TileBytes);
            for (unsigned row = 0; row < 16; ++row)
                for (unsigned column = 0; column < 16; ++column)
                    l1_h[column * 16 + row] = gm_h[row * 16 + column];
            std::memcpy(l0a.data(), l1_x, kFp16TileBytes);
            std::memcpy(l0b.data(), l1_h, kFp16TileBytes);

            run_mode<OutflowMode::Fp32>(l0a.data(), l0b.data(), golden,
                                        l0c.data(), out_fp32);
            run_mode<OutflowMode::Fp16>(l0a.data(), l0b.data(), golden,
                                        l0c.data(), out_fp16);
            run_mode<OutflowMode::Bf16>(l0a.data(), l0b.data(), golden,
                                        l0c.data(), out_bf16);

            ASSERT_CANARY_INTACT(gm_x);
            ASSERT_CANARY_INTACT(gm_h);
            ASSERT_CANARY_INTACT(ub_h);
            ASSERT_CANARY_INTACT(l1);
            ASSERT_CANARY_INTACT(l0a);
            ASSERT_CANARY_INTACT(l0b);
            ASSERT_CANARY_INTACT(l0c);
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const auto us = std::chrono::duration<double, std::micro>(elapsed).count();
        std::cout << "PASS 5 input cases x 3 output modes x 256 bit-identical elements; "
                     "all 64-byte canaries intact\n";
        std::cout << "host probe: " << us << " us total, " << us / 15.0
                  << " us per Cube+Fixpipe dispatch (includes staging/golden/checks)\n";
        std::cout << "instruction model per mode: 1 Mmad (4096 FP16 MACs / 8192 FLOPs), "
                     "1 Fixpipe (256 outputs); pre-stage: 2 GM->L1 + 2 L1->L0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
