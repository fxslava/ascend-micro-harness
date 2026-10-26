#pragma once

#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void launch_turboquant_reshape_and_cache_a8(
    void*, const std::int8_t*, const std::int8_t*, std::int8_t*, std::int8_t*,
    float*, const std::int32_t*, const float*, const std::int32_t*,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, float);

namespace tq_npu {
constexpr int kBlockSize = 16;
constexpr int kHeadSize = 256;
constexpr int kHeads = 4;
constexpr int kChannels = kHeads * kHeadSize;
constexpr int kGroups = 8;
constexpr int kGroupSize = 32;
constexpr int kHeadBytes = kBlockSize * kHeadSize / 2;
constexpr int kTableBytes = 2 * kGroups * kGroupSize * kGroupSize;
constexpr float kInvSqrtLen = 1.0f / 16.0f;

inline void check(aclError err, const char* op) {
    if (err != ACL_SUCCESS)
        throw std::runtime_error(std::string(op) + " failed: " + std::to_string(err));
}
inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
inline std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
inline int parity(unsigned value) {
    int p = 0;
    while (value) { p ^= static_cast<int>(value & 1u); value >>= 1; }
    return p;
}
inline int h(int i, int j) {
    return parity(static_cast<unsigned>(i & j)) ? -1 : 1;
}
inline int sign(int channel) {
    return ((channel * channel + 3 * channel + 1) % 7 < 3) ? -1 : 1;
}
inline std::uint32_t mix(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}
inline std::int8_t activation(int page, int plane, int head, int token, int channel) {
    const std::uint32_t seed = 0x47C9A3D1u ^
        (static_cast<std::uint32_t>(page) * 0x9E3779B9u) ^
        (static_cast<std::uint32_t>(plane * kHeads + head) * 0x85EBCA6Bu) ^
        (static_cast<std::uint32_t>(token) * 0xC2B2AE35u) ^
        (static_cast<std::uint32_t>(channel) * 0x27D4EB2Du);
    if (page == 0 && token == 0 && channel == 0) return plane == 0 ? -128 : 127;
    if (page == 1 && token == 1 && channel == 1) return plane == 0 ? 127 : -128;
    return static_cast<std::int8_t>(static_cast<int>(mix(seed) & 255u) - 128);
}
inline std::int64_t round_even_shift(std::int64_t x, int shift) {
    if (shift <= 0) return x << (-shift);
    const std::int64_t den = std::int64_t{1} << shift;
    std::int64_t q = x / den;
    const std::int64_t rem = x % den;
    const std::int64_t twice = 2 * (rem < 0 ? -rem : rem);
    if (twice > den || (twice == den && (q & 1) != 0)) q += x >= 0 ? 1 : -1;
    return q;
}

struct Fixture {
    int tokens;
    int blocks;
    std::vector<std::int8_t> key, value, key_expected, value_expected, tables;
    std::vector<float> scales_expected, signs;
    std::vector<std::int32_t> slots;

    explicit Fixture(int n) : tokens(n), blocks(n / kBlockSize),
        key(static_cast<std::size_t>(n) * kChannels),
        value(key.size()),
        key_expected(static_cast<std::size_t>(blocks) * kHeads * kHeadBytes),
        value_expected(key_expected.size()),
        tables(kTableBytes),
        scales_expected(static_cast<std::size_t>(2) * blocks * kHeads * kBlockSize),
        signs(kHeadSize), slots(n) {
        require(n == 16 || n == 512 || n == 1024, "supported contexts: 16, 512, 1024");
        for (int c = 0; c < kHeadSize; ++c) signs[c] = static_cast<float>(sign(c));
        for (int s = 0; s < 2; ++s)
            for (int g = 0; g < kGroups; ++g)
                for (int o = 0; o < kGroupSize; ++o)
                    for (int i = 0; i < kGroupSize; ++i)
                        tables[((s * kGroups + g) * kGroupSize + o) * kGroupSize + i] =
                            static_cast<std::int8_t>((s ? -1 : 1) * sign(g * kGroupSize + o) * h(o, i));
        for (int t = 0; t < n; ++t) {
            slots[t] = t;
            for (int head = 0; head < kHeads; ++head)
                for (int c = 0; c < kHeadSize; ++c) {
                    const std::size_t offset = static_cast<std::size_t>(t) * kChannels + head * kHeadSize + c;
                    key[offset] = activation(t / kBlockSize, 0, head, t % kBlockSize, c);
                    value[offset] = activation(t / kBlockSize, 1, head, t % kBlockSize, c);
                }
        }
        build_oracle();
    }

    void build_oracle() {
        // Exact (H8 tensor I32) diag(signs) (I8 tensor H32).
        std::int64_t inner[kBlockSize][kHeadSize] = {};
        for (int page = 0; page < blocks; ++page)
            for (int plane = 0; plane < 2; ++plane)
                for (int head = 0; head < kHeads; ++head) {
                    const auto& input = plane == 0 ? key : value;
                    for (int t = 0; t < kBlockSize; ++t)
                        for (int g = 0; g < kGroups; ++g)
                            for (int o = 0; o < kGroupSize; ++o) {
                                std::int64_t sum = 0;
                                for (int i = 0; i < kGroupSize; ++i)
                                    sum += h(o, i) * static_cast<int>(input[
                                        static_cast<std::size_t>(page * kBlockSize + t) * kChannels +
                                        head * kHeadSize + g * kGroupSize + i]);
                                inner[t][g * kGroupSize + o] = sum;
                            }
                    for (int t = 0; t < kBlockSize; ++t) {
                        std::int64_t rotated[kHeadSize] = {};
                        std::int64_t absmax = 1;
                        for (int out_g = 0; out_g < kGroups; ++out_g)
                            for (int o = 0; o < kGroupSize; ++o) {
                                std::int64_t sum = 0;
                                for (int g = 0; g < kGroups; ++g)
                                    sum += h(out_g, g) * sign(g * kGroupSize + o) *
                                           inner[t][g * kGroupSize + o];
                                rotated[out_g * kGroupSize + o] = sum;
                                absmax = std::max(absmax, sum < 0 ? -sum : sum);
                            }
                        int exponent = 0;
                        while ((std::int64_t{1} << (exponent + 1)) <= absmax) ++exponent;
                        const std::size_t scale_offset =
                            ((static_cast<std::size_t>(plane) * blocks + page) * kHeads + head) *
                            kBlockSize + t;
                        scales_expected[scale_offset] = std::ldexp(1.0f, exponent - 2) * kInvSqrtLen;
                        auto& packed = plane == 0 ? key_expected : value_expected;
                        const std::size_t base =
                            (static_cast<std::size_t>(page) * kHeads + head) * kHeadBytes;
                        for (int c = 0; c < kHeadSize; ++c) {
                            std::int64_t q = round_even_shift(rotated[c], exponent - 2);
                            q = std::min<std::int64_t>(7, std::max<std::int64_t>(-8, q));
                            const int nib_index = plane == 0 ? t * kHeadSize + c : c * kBlockSize + t;
                            const auto nib = static_cast<std::uint8_t>(q & 15);
                            auto& byte = packed[base + static_cast<std::size_t>(nib_index / 2)];
                            const auto old = static_cast<std::uint8_t>(byte);
                            byte = static_cast<std::int8_t>(old |
                                static_cast<std::uint8_t>((nib_index & 1) ? nib << 4 : nib));
                        }
                    }
                }
    }
};

struct Session {
    aclrtStream stream = nullptr;
    Session() {
        check(aclInit(nullptr), "aclInit");
        check(aclrtSetDevice(0), "aclrtSetDevice");
        check(aclrtCreateStream(&stream), "aclrtCreateStream");
    }
    ~Session() {
        if (stream) aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }
};
struct Buffer {
    void* ptr = nullptr;
    std::size_t bytes;
    explicit Buffer(std::size_t n) : bytes(n) {
        require(n != 0 && n % 32 == 0, "NPU buffer size must be 32-byte aligned");
        check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
        require(reinterpret_cast<std::uintptr_t>(ptr) % 32 == 0, "NPU allocation is unaligned");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (ptr) aclrtFree(ptr); }
    template <typename T> void upload(const std::vector<T>& src) {
        require(src.size() * sizeof(T) == bytes, "upload size mismatch");
        check(aclrtMemcpy(ptr, bytes, src.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
    }
    template <typename T> void download(std::vector<T>& dst) const {
        require(dst.size() * sizeof(T) == bytes, "download size mismatch");
        check(aclrtMemcpy(dst.data(), bytes, ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H");
    }
};

struct A8Device {
    Buffer key, value, key_cache, value_cache, scale_cache, slots, signs, tables;
    explicit A8Device(const Fixture& f)
        : key(f.key.size()), value(f.value.size()),
          key_cache(f.key_expected.size()), value_cache(f.value_expected.size()),
          scale_cache(f.scales_expected.size() * sizeof(float)),
          slots(f.slots.size() * sizeof(std::int32_t)),
          signs(f.signs.size() * sizeof(float)), tables(f.tables.size()) {
        key.upload(f.key); value.upload(f.value);
        slots.upload(f.slots); signs.upload(f.signs); tables.upload(f.tables);
    }
    void launch(aclrtStream stream, const Fixture& f) {
        launch_turboquant_reshape_and_cache_a8(stream,
            static_cast<const std::int8_t*>(key.ptr), static_cast<const std::int8_t*>(value.ptr),
            static_cast<std::int8_t*>(key_cache.ptr), static_cast<std::int8_t*>(value_cache.ptr),
            static_cast<float*>(scale_cache.ptr), static_cast<const std::int32_t*>(slots.ptr),
            static_cast<const float*>(signs.ptr), static_cast<const std::int32_t*>(tables.ptr),
            static_cast<std::uint32_t>(f.tokens), kHeads, kHeadSize, kBlockSize,
            static_cast<std::uint32_t>(f.blocks), kBlockSize, kInvSqrtLen);
    }
};

inline void verify(const Fixture& f, const A8Device& d) {
    std::vector<std::int8_t> key(f.key_expected.size()), value(f.value_expected.size());
    std::vector<float> scales(f.scales_expected.size());
    d.key_cache.download(key); d.value_cache.download(value); d.scale_cache.download(scales);
    std::size_t errors = 0;
    auto compare = [&errors](const std::vector<std::int8_t>& got,
                             const std::vector<std::int8_t>& want, const char* label) {
        for (std::size_t i = 0; i < want.size(); ++i)
            if (got[i] != want[i]) {
                if (errors < 12) std::fprintf(stderr, "%s byte %zu: got %02x expected %02x\n",
                    label, i, static_cast<unsigned>(static_cast<std::uint8_t>(got[i])),
                    static_cast<unsigned>(static_cast<std::uint8_t>(want[i])));
                ++errors;
            }
    };
    compare(key, f.key_expected, "keyCache");
    compare(value, f.value_expected, "valueCache");
    for (std::size_t i = 0; i < scales.size(); ++i)
        if (bits(scales[i]) != bits(f.scales_expected[i])) {
            if (errors < 12) std::fprintf(stderr, "scaleCache[%zu]: got %08x expected %08x\n",
                i, bits(scales[i]), bits(f.scales_expected[i]));
            ++errors;
        }
    if (errors) throw std::runtime_error(std::to_string(errors) + " bit-exact mismatches");
    std::printf("PASS %d tokens: %zu key nibbles, %zu value nibbles, %zu FP32 scales exact\n",
        f.tokens, key.size() * 2, value.size() * 2, scales.size());
}
}  // namespace tq_npu
