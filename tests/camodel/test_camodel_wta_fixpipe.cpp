#include "camodel_common.hpp"
#include "../cpu/wta_reference.hpp"
#include <cstdlib>

#define DECL(N, S) extern "C" void launch_wta_##N##S(void*, std::uint16_t*, std::uint16_t*, std::int8_t*, std::int32_t*);
DECL(1, s) DECL(2, s) DECL(3, s) DECL(4, s) DECL(5, s) DECL(6, s) DECL(7, s) DECL(8, s)
DECL(1, u) DECL(2, u) DECL(3, u) DECL(4, u) DECL(5, u) DECL(6, u) DECL(7, u) DECL(8, u)
#undef DECL

int main(int argc, char** argv)
{
    try {
        const int steps = argc > 1 ? std::atoi(argv[1]) : 8;
        const bool sign = argc > 2 ? std::atoi(argv[2]) != 0 : true;
        const unsigned seed = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 13;
        const float delta = argc > 4 ? std::strtof(argv[4], nullptr) : .01f;
        wta_test::require(steps >= 1 && steps <= 8 && delta >= .01f && delta <= 1.f,
                          "usage: test_camodel_wta_fixpipe T[1..8] signed[0|1] seed delta[.01..1]");
        // Direct random FP16 input and the alpha=0.25/beta=0.4 inhibition
        // matrix; the fixed probe scale matches Tier 1.
        constexpr std::uint32_t scale = 0x43000000u;
        using Launch = void (*)(void*, std::uint16_t*, std::uint16_t*, std::int8_t*, std::int32_t*);
        const Launch launches[2][8] = {
            {launch_wta_1u, launch_wta_2u, launch_wta_3u, launch_wta_4u,
             launch_wta_5u, launch_wta_6u, launch_wta_7u, launch_wta_8u},
            {launch_wta_1s, launch_wta_2s, launch_wta_3s, launch_wta_4s,
             launch_wta_5s, launch_wta_6s, launch_wta_7s, launch_wta_8s}};
        camodel::Session session;
        camodel::GuardedDeviceBuffer input(512), weights(512), identity(1024), output(1024);
        const auto x = wta_test::fixture(seed, delta);
        const auto w = wta_test::weights();
        std::memcpy(input.host_payload(), x.data(), 512);
        auto* wp = static_cast<std::uint16_t*>(weights.host_payload());
        for (int c = 0; c < 16; ++c)
            for (int k = 0; k < 16; ++k) wp[c * 16 + k] = w[k * 16 + c];
        auto* ip = static_cast<std::int8_t*>(identity.host_payload());
        for (int c = 0; c < 32; ++c)
            for (int k = 0; k < 32; ++k) ip[c * 32 + k] = k == c;
        input.upload(); weights.upload(); identity.upload();
        launches[sign][steps - 1](session.stream,
            static_cast<std::uint16_t*>(input.payload()),
            static_cast<std::uint16_t*>(weights.payload()),
            static_cast<std::int8_t*>(identity.payload()),
            static_cast<std::int32_t*>(output.payload()));
        camodel::check(aclrtSynchronizeStream(session.stream), "WTA synchronize");
        input.download(); weights.download(); identity.download(); output.download();
        input.assert_guards(); weights.assert_guards(); identity.assert_guards(); output.assert_guards();
        const auto expected = wta_test::reference(x, scale, sign);
        const auto* got = static_cast<const std::int32_t*>(output.host_payload());
        // Random FP16 inputs are not dyadic-aligned, so the device's FP32 Mmad
        // accumulation order legitimately differs from the exact oracle in the
        // last bits; Tier-1 sensitivity measured this at 1 B8 ulp. Accept a
        // one-quantum difference, fail beyond it, and report both censuses.
        bool mismatch = false;
        unsigned exact = 0; int max_abs = 0;
        std::array<int, 256> device{};
        for (unsigned i = 0; i < 256; ++i) {
            const int value = sign ? got[i] : static_cast<std::uint8_t>(got[i]);
            device[i] = value;
            const int diff = std::abs(value - expected.quantized[steps - 1][i]);
            if (diff == 0) ++exact;
            if (diff > max_abs) max_abs = diff;
            if (diff > 1) {
                std::fprintf(stderr, "T=%d signed=%d i=%u got=%d expected=%d\n",
                    steps, sign, i, value, expected.quantized[steps - 1][i]);
                mismatch = true;
            }
        }
        if (mismatch) throw std::runtime_error("C1 victory byte differs from oracle by more than one quantum");
        wta_test::Convergence oracle_stats, device_stats;
        oracle_stats.add(x, expected.quantized[steps - 1]);
        device_stats.add(x, device);
        std::printf("PASS native WTA T=%d signed=%d seed=%u delta=%g scale_bits=0x%08x: "
                    "%u/256 exact, max |diff|=%d (bound 1), 512/512 GM guard bytes; "
                    "oracle one_survivor=%u multi=%u zero=%u | device one_survivor=%u multi=%u zero=%u\n",
            steps, sign, seed, delta, scale, exact, max_abs,
            oracle_stats.support_one, oracle_stats.multi, oracle_stats.zero,
            device_stats.support_one, device_stats.multi, device_stats.zero);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL native WTA: %s\n", e.what()); return 1;
    }
}
