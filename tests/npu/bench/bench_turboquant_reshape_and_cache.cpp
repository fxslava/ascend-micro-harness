#include "../turboquant_npu_common.hpp"

#include "types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

namespace vllm_ascend {
void turboquant_reshape_and_cache_impl(
    AscendType type, void* stream, std::uint32_t block_dim, void* key, void* value,
    void* key_cache, void* value_cache, void* scale_cache, void* slot_mapping,
    void* pi_signs, void* tables, std::uint32_t num_tokens,
    std::uint32_t num_kv_heads, std::uint32_t head_size, std::uint32_t block_size,
    std::uint32_t num_blocks, std::uint32_t tokens_per_core, float inv_sqrt_len);
}

namespace {
constexpr int kTokens = 1024;
constexpr int kWarmups = 50;
constexpr int kSamples = 500;
constexpr std::size_t kEvictionBytes = 128u * 1024u * 1024u;
// Our BF16 Cube kernel is driven in windows, like its validation harness: one
// strip per grid block per launch, at most this many blocks at a time. At
// kTokens that makes 64 strips two launches against the A8 kernel's one, and
// the timed sample covers all of them, so the comparison is end-to-end work
// rather than per-dispatch. The launch count is printed next to the timings.
constexpr int kBenchWindow = 32;

std::int32_t float_word(float x) {
    const auto raw = tq_npu::bits(x);
    std::int32_t result = 0;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}
std::vector<std::int32_t> baseline_tables() {
    constexpr int d = tq_npu::kHeadSize;
    std::vector<std::int32_t> t;
    t.reserve(7 * d + 2 * d + 16);
    for (int stage = 0; stage < 3; ++stage) {
        const int stride = 1 << stage;
        for (int c = 0; c < d; ++c)
            t.push_back(float_word(static_cast<float>(1 - 2 * ((c / stride) & 1))));
        for (int c = 0; c < d; ++c) t.push_back(4 * (c ^ stride));
    }
    for (int p = 0; p < d / 2; ++p) t.push_back(8 * p);
    for (int p = 0; p < d / 2; ++p) t.push_back(8 * p + 4);
    for (int b = 0; b < d; ++b) t.push_back(4 * (b / 2));
    for (int b = 0; b < d; ++b) t.push_back(float_word(static_cast<float>(b & 1)));
    constexpr std::array<float, 16> centroids = {
        -2.7325895709951710f, -2.0690172265313920f, -1.6180463860218863f,
        -1.2562311973471796f, -0.9423404564869651f, -0.6567591185324659f,
        -0.3880482994902919f, -0.1283950298511473f, 0.1283950298511473f,
        0.3880482994902919f, 0.6567591185324659f, 0.9423404564869651f,
        1.2562311973471796f, 1.6180463860218863f, 2.0690172265313920f,
        2.7325895709951710f};
    for (float v : centroids) t.push_back(float_word(v));
    tq_npu::require(t.size() == 2320, "baseline codec table size");
    return t;
}
struct Baseline {
    tq_npu::Buffer key, value, key_cache, value_cache, scale_cache, slots, signs, tables;
    explicit Baseline(const tq_npu::Fixture& f)
        : key(f.key.size() * 2), value(f.value.size() * 2),
          key_cache(f.shape.cache_bytes()), value_cache(f.shape.cache_bytes()),
          scale_cache(f.shape.scale_floats() * sizeof(float)),
          slots(f.slots.size() * sizeof(std::int32_t)),
          signs(f.signs.size() * sizeof(float)), tables(2320 * sizeof(std::int32_t)) {
        key.upload(tq_npu::to_bfloat16(f.key));
        value.upload(tq_npu::to_bfloat16(f.value));
        slots.upload(f.slots); signs.upload(f.signs); tables.upload(baseline_tables());
    }
    void launch(aclrtStream stream, const tq_npu::Fixture& f) {
        // The baseline is an AIV-only kernel, so its blockDim is one strip per
        // Vector core and its raw GetBlockIdx() is already the strip index.
        vllm_ascend::turboquant_reshape_and_cache_impl(
            vllm_ascend::AscendType::BF16, stream,
            static_cast<std::uint32_t>(f.strips()),
            key.ptr, value.ptr, key_cache.ptr, value_cache.ptr, scale_cache.ptr,
            slots.ptr, signs.ptr, tables.ptr,
            static_cast<std::uint32_t>(f.shape.num_tokens), tq_npu::kHeads,
            tq_npu::kHeadSize, tq_npu::kBlockSize,
            static_cast<std::uint32_t>(f.shape.num_blocks),
            tq_npu::kBlockSize, f.shape.inv_sqrt_len());
    }
};
struct Events {
    aclrtEvent start = nullptr, end = nullptr;
    Events() {
        tq_npu::check(aclrtCreateEvent(&start), "aclrtCreateEvent start");
        tq_npu::check(aclrtCreateEvent(&end), "aclrtCreateEvent end");
    }
    ~Events() {
        if (start) aclrtDestroyEvent(start);
        if (end) aclrtDestroyEvent(end);
    }
};
struct Stats { double min, median, p99; };
Stats summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return {samples.front(), samples[samples.size() / 2],
            samples[(samples.size() * 99 + 99) / 100 - 1]};
}
template <typename Launch>
Stats measure(aclrtStream stream, Launch&& launch, tq_npu::Buffer& eviction_src,
              tq_npu::Buffer& eviction_dst, bool evict) {
    for (int i = 0; i < kWarmups; ++i) launch();
    tq_npu::check(aclrtSynchronizeStream(stream), "warmup synchronize");
    std::vector<double> timings;
    timings.reserve(kSamples);
    for (int i = 0; i < kSamples; ++i) {
        if (evict) {
            tq_npu::check(aclrtMemcpy(eviction_dst.ptr, eviction_dst.bytes,
                eviction_src.ptr, eviction_src.bytes, ACL_MEMCPY_DEVICE_TO_DEVICE),
                "L2 eviction D2D copy");
        }
        Events events;
        tq_npu::check(aclrtRecordEvent(events.start, stream), "record start");
        launch();
        tq_npu::check(aclrtRecordEvent(events.end, stream), "record end");
        tq_npu::check(aclrtSynchronizeEvent(events.end), "synchronize end event");
        float ms = 0;
        tq_npu::check(aclrtEventElapsedTime(&ms, events.start, events.end),
                      "aclrtEventElapsedTime");
        tq_npu::require(ms > 0, "nonpositive device event duration");
        timings.push_back(static_cast<double>(ms) * 1000.0);
    }
    return summarize(std::move(timings));
}
void report(const char* mode, const char* kernel, const Stats& s, double bytes) {
    std::printf("%-8s %-9s min %.3f us median %.3f us p99 %.3f us "
                "throughput %.2f GB/s\n", mode, kernel,
                s.min, s.median, s.p99, bytes / s.median / 1000.0);
}
}  // namespace

int main() {
    try {
        tq_npu::Fixture f(kTokens);
        tq_npu::Session session;
        tq_npu::A8Device a8(f);
        tq_npu::Bf16Device bf16_cube(f);
        Baseline bf16_vec(f);
        tq_npu::Buffer eviction_src(kEvictionBytes), eviction_dst(kEvictionBytes);
        tq_npu::check(aclrtMemset(eviction_src.ptr, eviction_src.bytes, 0xA5,
                                  eviction_src.bytes), "initialize eviction buffer");
        // Bytes actually moved, not bytes allocated: the cache is deliberately
        // larger than the written pages (numBlocks is 2 * strips + 2, so a
        // stray page offset has somewhere to land where the validation harness
        // catches it), and counting the unwritten pages would halve both
        // throughput figures.
        const double written_cache =
            static_cast<double>(f.strips()) * f.shape.page_bytes();
        const double written_scales =
            static_cast<double>(tq_oracle::kPlanes) * f.strips() * tq_npu::kHeads *
            tq_npu::kBlockSize * sizeof(float);
        const double a8_bytes = 2.0 * f.key.size() + 2.0 * written_cache + written_scales;
        const double bf16_bytes = 4.0 * f.key.size() + 2.0 * written_cache + written_scales;
        const int cube_launches = (f.strips() + kBenchWindow - 1) / kBenchWindow;
        std::printf("%d tokens, %d strips over %d physical pages, %d warmups, "
                    "%d device-event runs\n",
                    f.shape.num_tokens, f.strips(), f.shape.num_blocks, kWarmups,
                    kSamples);
        std::printf("A8 and BF16-vec dispatch one launch of %d blocks; BF16-cube "
                    "dispatches %d launch(es) of at most %d, all inside the timed "
                    "sample\n",
                    f.strips(), cube_launches, kBenchWindow);
        for (bool evict : {false, true}) {
            const char* mode = evict ? "pressure" : "warm";
            const Stats sa = measure(session.stream, [&] { a8.launch(session.stream, f); },
                                     eviction_src, eviction_dst, evict);
            const Stats sc = measure(session.stream,
                                     [&] { bf16_cube.launch(session.stream, f,
                                                            kBenchWindow); },
                                     eviction_src, eviction_dst, evict);
            const Stats sv = measure(session.stream,
                                     [&] { bf16_vec.launch(session.stream, f); },
                                     eviction_src, eviction_dst, evict);
            report(mode, "A8", sa, a8_bytes);
            report(mode, "BF16-cube", sc, bf16_bytes);
            report(mode, "BF16-vec", sv, bf16_bytes);
            // Two comparisons, and only one of them is apples to apples.
            // BF16-cube against BF16-vec is the same input precision and the
            // same quantiser, so it isolates Cube-versus-Vector rotation.
            // Either against A8 also changes the input precision, so it is a
            // wall-time figure for the same geometry and nothing narrower.
            std::printf("%-8s median BF16-vec / BF16-cube: %.3fx (same precision, "
                        "same quantiser -- Cube versus Vector rotation)\n",
                        mode, sv.median / sc.median);
            std::printf("%-8s median BF16-cube / A8: %.3fx, BF16-vec / A8: %.3fx "
                        "(different input precision)\n",
                        mode, sc.median / sa.median, sv.median / sa.median);
        }
        // The Cube form folds the outer factor's signs into the resident blocks,
        // so it spends eight times the multiply-adds of the factored Vector form
        // and buys back the Vector passes. Both Cube paths do the same work; only
        // their input precision, and so their GM traffic, differs.
        constexpr double cube_rotation_ops =
            2.0 * kTokens * 2 * tq_npu::kHeads * tq_npu::kHeadSize * tq_npu::kHeadSize;
        constexpr double vector_rotation_ops =
            2.0 * kTokens * 2 * tq_npu::kHeads * 8 * (tq_npu::kHeadSize / 2);
        std::printf("Rotation-only intensity: A8 Cube %.1f ops/GM byte, "
                    "BF16 Cube %.1f ops/GM byte, BF16 Vector %.1f ops/GM byte. "
                    "GM bytes exclude tables and slot metadata; every estimate "
                    "excludes quantization.\n",
                    cube_rotation_ops / a8_bytes, cube_rotation_ops / bf16_bytes,
                    vector_rotation_ops / bf16_bytes);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL turboquant NPU benchmark: %s\n", e.what());
        return 1;
    }
}
