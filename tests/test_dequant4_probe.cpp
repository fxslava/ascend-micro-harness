#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "../include/sim_micro_testbed.hpp"
#include "../routines/routine_turboquant_dequant4.hpp"

using ascend_micro::sim::GuardedBuffer;
using ascend_micro::sim::assert_aligned;
using ascend_micro::sim::float_bits;
using ascend_micro::sim::float_to_half_rne;
using ascend_micro::sim::half_to_float;
using ascend_micro::sim::require;

template <class Output>
struct MockDequant4Engine {
    const std::int8_t* packed;
    std::int16_t* signed_nibbles;
    std::int16_t* unsigned_nibbles;
    Output* output;
    float scale;
    unsigned instructions = 0;

    void cast_packed_nibbles(std::uint32_t count)
    {
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto byte = static_cast<std::uint8_t>(packed[i / 2]);
            const auto nibble = std::uint8_t((byte >> ((i & 1u) * 4u)) & 15u);
            signed_nibbles[i] = nibble < 8 ? nibble : int(nibble) - 16;
        }
        ++instructions;
    }
    void mask_unsigned_nibbles(std::uint32_t count)
    {
        for (std::uint32_t i = 0; i < count; ++i)
            unsigned_nibbles[i] = signed_nibbles[i] & 15;
        instructions += 2;  // Duplicate mask and vector And.
    }
    void cast_and_scale(std::uint32_t count)
    {
        for (std::uint32_t i = 0; i < count; ++i) {
            const float converted = float(unsigned_nibbles[i]);
            const float result = converted * scale;
            if constexpr (sizeof(Output) == sizeof(float))
                output[i] = result;
            else
                output[i] = float_to_half_rne(result);
        }
        instructions += 2;  // Cast and Muls.
    }
};

template <class Output>
static void verify_row(const GuardedBuffer<std::int8_t>& packed,
                       GuardedBuffer<std::int16_t>& signed_nibbles,
                       GuardedBuffer<std::int16_t>& unsigned_nibbles,
                       GuardedBuffer<Output>& output, float scale)
{
    MockDequant4Engine<Output> engine{packed.data(), signed_nibbles.data(),
                                      unsigned_nibbles.data(), output.data(), scale};
    ascend_micro::turboquant_dequant4(engine, std::uint32_t(output.size()));
    require(engine.instructions == 5, "wrong vector dispatch count");
    for (std::size_t i = 0; i < output.size(); ++i) {
        const auto byte = static_cast<std::uint8_t>(packed[i / 2]);
        const auto golden = float((byte >> (4 * (i % 2))) & 15u) * scale;
        if constexpr (sizeof(Output) == sizeof(float))
            require(float_bits(output[i]) == float_bits(golden), "FP32 mismatch");
        else
            require(output[i] == float_to_half_rne(golden), "FP16 mismatch");
    }
    ASSERT_CANARY_INTACT(output);
}

int main()
{
    try {
        constexpr std::size_t kRows = 4;
        constexpr std::size_t kElements = 128;
        GuardedBuffer<std::int8_t> packed(kRows * kElements / 2);
        GuardedBuffer<std::int16_t> signed_nibbles(kElements);
        GuardedBuffer<std::int16_t> unsigned_nibbles(kElements);
        GuardedBuffer<float> fp32(kElements);
        GuardedBuffer<std::uint16_t> fp16(kElements);
        for (std::size_t i = 0; i < packed.size(); ++i)
            packed[i] = std::int8_t((i * 37u + 0x8fu) & 255u);
        for (const void* p : {static_cast<const void*>(packed.data()),
                              static_cast<const void*>(signed_nibbles.data()),
                              static_cast<const void*>(unsigned_nibbles.data()),
                              static_cast<const void*>(fp32.data()),
                              static_cast<const void*>(fp16.data())})
            assert_aligned(p);
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t row = 0; row < kRows; ++row) {
            const float scale = float(row + 1) / 8.0f;
            // Each row begins at a 64-byte aligned offset.
            GuardedBuffer<std::int8_t> row_packed(kElements / 2);
            for (std::size_t i = 0; i < row_packed.size(); ++i)
                row_packed[i] = packed[row * kElements / 2 + i];
            verify_row(row_packed, signed_nibbles, unsigned_nibbles, fp32, scale);
            verify_row(row_packed, signed_nibbles, unsigned_nibbles, fp16, scale);
            ASSERT_CANARY_INTACT(row_packed);
        }
        const auto us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        ASSERT_CANARY_INTACT(packed);
        ASSERT_CANARY_INTACT(signed_nibbles);
        ASSERT_CANARY_INTACT(unsigned_nibbles);
        ASSERT_CANARY_INTACT(fp32);
        ASSERT_CANARY_INTACT(fp16);
        std::cout << "PASS dequant4: 4 rows x 2 types x 128 bit-identical outputs; "
                     "all 64-byte canaries intact\n";
        std::cout << "host probe: " << us << " us total (includes setup/oracle/checks)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL dequant4: " << error.what() << '\n';
        return 1;
    }
}
