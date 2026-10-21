#include "camodel_common.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

extern "C" void launch_camodel_normalize_fp16(void*, float*, float*, std::uint16_t*);
extern "C" void launch_camodel_normalize_bf16(void*, float*, float*, std::uint16_t*);

namespace {
std::uint32_t bits(float value)
{
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
std::uint16_t bf16_rne(float value)
{
    const auto raw = bits(value);
    return static_cast<std::uint16_t>((raw + 0x7fffu + ((raw >> 16) & 1u)) >> 16);
}
std::uint16_t fp16_rne(float value)
{
    const auto raw = bits(value);
    const auto sign = static_cast<std::uint16_t>((raw >> 16) & 0x8000u);
    const int exponent = static_cast<int>((raw >> 23) & 0xffu) - 127 + 15;
    const auto mantissa = raw & 0x7fffffu;
    if (exponent <= 0 || exponent >= 31)
        throw std::runtime_error("test input outside normal half range");
    const auto rounded = mantissa + 0xfffu + ((mantissa >> 13) & 1u);
    return static_cast<std::uint16_t>(sign | (exponent << 10) | (rounded >> 13));
}
void run(bool bf16, camodel::Session& session)
{
    camodel::GuardedDeviceBuffer acc(64), l(32), output(32);
    auto* av = static_cast<float*>(acc.host_payload());
    auto* lv = static_cast<float*>(l.host_payload());
    lv[0] = 2.0f;
    for (int i = 0; i < 16; ++i)
        av[i] = 2.0f + static_cast<float>(i + 1) * 0.00390625f;
    acc.upload(); l.upload();
    if (bf16)
        launch_camodel_normalize_bf16(session.stream,
            static_cast<float*>(acc.payload()), static_cast<float*>(l.payload()),
            static_cast<std::uint16_t*>(output.payload()));
    else
        launch_camodel_normalize_fp16(session.stream,
            static_cast<float*>(acc.payload()), static_cast<float*>(l.payload()),
            static_cast<std::uint16_t*>(output.payload()));
    camodel::check(aclrtSynchronizeStream(session.stream), "normalize synchronize");
    acc.download(); l.download(); output.download();
    acc.assert_guards(); l.assert_guards(); output.assert_guards();
    const auto* got = static_cast<const std::uint16_t*>(output.host_payload());
    for (int i = 0; i < 16; ++i) {
        const auto expected = bf16 ? bf16_rne(av[i] / lv[0]) : fp16_rne(av[i] / lv[0]);
        if (got[i] != expected) {
            std::fprintf(stderr, "normalize %s index=%d got=0x%04x expected=0x%04x\n",
                         bf16 ? "BF16" : "FP16", i, got[i], expected);
            throw std::runtime_error("normalize result mismatch");
        }
    }
    std::printf("PASS native normalize %s: 16/16 bit-exact, 384/384 GM guard bytes intact\n",
                bf16 ? "BF16" : "FP16");
}
}  // namespace

int main()
{
    try {
        camodel::Session session;
        run(false, session);
        run(true, session);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL native normalize: %s\n", error.what());
        return 1;
    }
}
