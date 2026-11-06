#include <acl/acl.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

extern "C" void launch_turboquant_reshape_scatter_aiv(
    void*, std::uint32_t, std::uint16_t*, std::uint16_t*, std::int8_t*, std::int8_t*, float*,
    std::int32_t*, float*, std::int32_t*, std::int32_t*, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, float);

namespace {
constexpr std::uint32_t kHeadSize = 256;
constexpr std::uint32_t kBlockSize = 16;
constexpr std::uint32_t kNumBlocks = 2;
constexpr std::uint32_t kProbeLanes = 32;

void check(aclError rc, const char* what)
{
    if (rc != ACL_SUCCESS) throw std::runtime_error(what);
}

struct DeviceBuffer {
    void* ptr = nullptr;
    explicit DeviceBuffer(std::size_t bytes)
    {
        check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
    }
    ~DeviceBuffer() { if (ptr) aclrtFree(ptr); }
};
}

int main(int argc, char** argv)
{
    int warmup = 20;
    int iterations = 200;
    if (argc > 1) warmup = std::max(0, std::atoi(argv[1]));
    if (argc > 2) iterations = std::max(1, std::atoi(argv[2]));

    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    aclrtEvent start = nullptr, stop = nullptr;
    try {
        check(aclrtCreateStream(&stream), "aclrtCreateStream");
        check(aclrtCreateEvent(&start), "aclrtCreateEvent(start)");
        check(aclrtCreateEvent(&stop), "aclrtCreateEvent(stop)");

        DeviceBuffer key(2u * kHeadSize * sizeof(std::uint16_t));
        DeviceBuffer value(2u * kHeadSize * sizeof(std::uint16_t));
        DeviceBuffer key_cache(kNumBlocks * kBlockSize * kHeadSize / 2u);
        DeviceBuffer value_cache(kNumBlocks * kBlockSize * kHeadSize / 2u);
        DeviceBuffer scales(2u * kNumBlocks * kBlockSize * sizeof(float));
        DeviceBuffer slots(32);
        DeviceBuffer signs(kHeadSize * sizeof(float));
        DeviceBuffer cu_seqlens(32);
        DeviceBuffer probe(kProbeLanes * sizeof(std::int32_t));

        std::vector<std::uint16_t> zero_bf16(2u * kHeadSize, 0);
        std::vector<std::int32_t> host_slots(8, -1), host_cu(8, 0), host_probe(kProbeLanes, 0);
        std::vector<float> host_scales(2u * kNumBlocks * kBlockSize, 1.0f);
        std::vector<float> host_signs(kHeadSize, 1.0f);
        host_slots[0] = 0;
        host_cu[0] = 0;
        host_cu[1] = 1;
        check(aclrtMemcpy(key.ptr, zero_bf16.size() * sizeof(std::uint16_t), zero_bf16.data(),
                          zero_bf16.size() * sizeof(std::uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy key");
        check(aclrtMemcpy(value.ptr, zero_bf16.size() * sizeof(std::uint16_t), zero_bf16.data(),
                          zero_bf16.size() * sizeof(std::uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy value");
        check(aclrtMemcpy(slots.ptr, 32, host_slots.data(), 32, ACL_MEMCPY_HOST_TO_DEVICE), "copy slots");
        check(aclrtMemcpy(cu_seqlens.ptr, 32, host_cu.data(), 32, ACL_MEMCPY_HOST_TO_DEVICE), "copy cu_seqlens");
        check(aclrtMemcpy(signs.ptr, host_signs.size() * sizeof(float), host_signs.data(),
                          host_signs.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), "copy signs");
        check(aclrtMemcpy(scales.ptr, host_scales.size() * sizeof(float), host_scales.data(),
                          host_scales.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), "copy scales");
        check(aclrtMemcpy(probe.ptr, host_probe.size() * sizeof(std::int32_t), host_probe.data(),
                          host_probe.size() * sizeof(std::int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy probe");

        const auto launch = [&]() {
            launch_turboquant_reshape_scatter_aiv(
                stream, 1, static_cast<std::uint16_t*>(key.ptr), static_cast<std::uint16_t*>(value.ptr),
                static_cast<std::int8_t*>(key_cache.ptr), static_cast<std::int8_t*>(value_cache.ptr),
                static_cast<float*>(scales.ptr), static_cast<std::int32_t*>(slots.ptr),
                static_cast<float*>(signs.ptr), static_cast<std::int32_t*>(cu_seqlens.ptr),
                static_cast<std::int32_t*>(probe.ptr), 1, 1, kHeadSize, kBlockSize, kNumBlocks,
                0, 1, 1.0f / 16.0f);
        };
        for (int i = 0; i < warmup; ++i) launch();
        check(aclrtSynchronizeStream(stream), "warmup sync");
        check(aclrtRecordEvent(start, stream), "record start");
        for (int i = 0; i < iterations; ++i) launch();
        check(aclrtRecordEvent(stop, stream), "record stop");
        check(aclrtSynchronizeEvent(stop), "benchmark sync");
        float elapsed_ms = 0.0f;
        check(aclrtEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");
        std::printf("scatter launches=%d warmup=%d mean_us=%.3f\n", iterations, warmup,
                    elapsed_ms * 1000.0f / static_cast<float>(iterations));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "scatter benchmark failed: %s\n", e.what());
        if (stop) aclrtDestroyEvent(stop);
        if (start) aclrtDestroyEvent(start);
        if (stream) aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }
    if (stop) aclrtDestroyEvent(stop);
    if (start) aclrtDestroyEvent(start);
    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
