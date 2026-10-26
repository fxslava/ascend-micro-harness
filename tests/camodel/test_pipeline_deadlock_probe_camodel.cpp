#include <acl/acl.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

extern "C" void launch_turboquant_reshape_and_cache(void* stream,
                                                      std::int32_t* input,
                                                      std::int32_t* output);

namespace {
constexpr unsigned kTiles = 3;
constexpr unsigned kWordsPerTile = 32;
constexpr unsigned kGuardWords = 16;  // 64-byte GM guard bands
constexpr unsigned kPayloadWords = kTiles * kWordsPerTile;
constexpr unsigned kTotalWords = kGuardWords + kPayloadWords + kGuardWords;
constexpr std::uint32_t kInputCanary = 0x55AA55AAu;
constexpr std::uint32_t kOutputCanary = 0xDEADBEEFu;
constexpr std::uint32_t kGuard = 0xA5A5A5A5u;

void check(aclError result, const char* call)
{
    if (result != ACL_SUCCESS) {
        std::fprintf(stderr, "%s failed: %d\n", call, static_cast<int>(result));
        throw std::runtime_error(call);
    }
}
}  // namespace

int main()
{
    void* device_input = nullptr;
    void* device_output = nullptr;
    aclrtStream stream = nullptr;
    try {
        std::array<std::uint32_t, kTotalWords> input{};
        std::array<std::uint32_t, kTotalWords> output{};
        input.fill(kGuard);
        output.fill(kGuard);
        for (unsigned i = 0; i < kPayloadWords; ++i)
            input[kGuardWords + i] = kInputCanary;

        check(aclInit(nullptr), "aclInit");
        check(aclrtSetDevice(0), "aclrtSetDevice");
        check(aclrtCreateStream(&stream), "aclrtCreateStream");
        check(aclrtMalloc(&device_input, sizeof(input), ACL_MEM_MALLOC_NORMAL_ONLY),
              "aclrtMalloc input");
        check(aclrtMalloc(&device_output, sizeof(output), ACL_MEM_MALLOC_NORMAL_ONLY),
              "aclrtMalloc output");
        check(aclrtMemcpy(device_input, sizeof(input), input.data(), sizeof(input),
                          ACL_MEMCPY_HOST_TO_DEVICE), "H2D input");
        check(aclrtMemcpy(device_output, sizeof(output), output.data(), sizeof(output),
                          ACL_MEMCPY_HOST_TO_DEVICE), "H2D output guards");

        auto* in_payload = static_cast<std::int32_t*>(device_input) + kGuardWords;
        auto* out_payload = static_cast<std::int32_t*>(device_output) + kGuardWords;
        launch_turboquant_reshape_and_cache(stream, in_payload, out_payload);
        check(aclrtSynchronizeStream(stream), "kernel synchronize");
        check(aclrtMemcpy(output.data(), sizeof(output), device_output,
                          sizeof(output), ACL_MEMCPY_DEVICE_TO_HOST), "D2H output");
        for (unsigned i = 0; i < kPayloadWords; ++i)
            if (output[kGuardWords + i] != kOutputCanary)
                throw std::runtime_error("output canary mismatch");
        for (unsigned i = 0; i < kGuardWords; ++i)
            if (output[i] != kGuard ||
                output[kGuardWords + kPayloadWords + i] != kGuard)
                throw std::runtime_error("GM guard band corruption");

        check(aclrtFree(device_output), "aclrtFree output");
        check(aclrtFree(device_input), "aclrtFree input");
        check(aclrtDestroyStream(stream), "aclrtDestroyStream");
        check(aclrtResetDevice(0), "aclrtResetDevice");
        check(aclFinalize(), "aclFinalize");
        std::puts("PASS CAModel N=3; output canaries and GM guard bands intact");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL CAModel: %s\n", error.what());
        if (device_output) aclrtFree(device_output);
        if (device_input) aclrtFree(device_input);
        if (stream) aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }
}
