#include "../turboquant_npu_common.hpp"
#include "../turboquant_audit_report.hpp"
#include "../turboquant_ingest_bench_options.hpp"
#include "../turboquant_native_ingest.hpp"

#include "types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
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
constexpr std::size_t kEvictionBytes = 128u * 1024u * 1024u;
static_assert(tq_ingest_bench::kStripTokens == tq_npu::kBlockSize);
static_assert(tq_ingest_bench::kMaxTokens == tq_npu::kMaxStrips * tq_npu::kBlockSize);
// All three kernels now dispatch one monolithic grid of one strip per block.
// The BF16 Cube kernel used to be driven in capped windows here, so at 64
// strips it paid two dispatches against the other two kernels' one and the
// comparison carried a launch-count difference. It carries the inter-wave
// protocol now (turboquant_wave_protocol.hpp), so every row below is one
// launch of f.strips() blocks and the three kernels are timed on equal terms.

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
using Stats = tq_ingest_audit::Stats;
Stats summarize(std::vector<double> samples) {
    tq_npu::require(!samples.empty(), "no timing samples");
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    const double median = samples.size() % 2 ? samples[middle]
        : samples[middle - 1] + (samples[middle] - samples[middle - 1]) / 2;
    return {samples.front(), median,
            samples[(samples.size() * 99 + 99) / 100 - 1]};
}
template <typename Launch>
Stats measure(aclrtStream stream, Launch&& launch, tq_npu::Buffer& eviction_src,
              tq_npu::Buffer& eviction_dst, bool evict, int warmups, int samples) {
    for (int i = 0; i < warmups; ++i) launch();
    tq_npu::check(aclrtSynchronizeStream(stream), "warmup synchronize");
    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(samples));
    for (int i = 0; i < samples; ++i) {
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
        tq_npu::require(std::isfinite(ms) && ms > 0, "invalid device event duration");
        timings.push_back(static_cast<double>(ms) * 1000.0);
    }
    return summarize(std::move(timings));
}
void report(const char* mode, const char* kernel, const Stats& s, double bytes) {
    std::printf("%-8s %-9s min %.3f us median %.3f us p99 %.3f us "
                "throughput %.2f GB/s\n", mode, kernel,
                s.min, s.median, s.p99, bytes / s.median / 1000.0);
}
void run_case(const tq_ingest_bench::Options& opts, const std::string& model,
              std::uint64_t context, std::uint64_t batch, std::uint64_t chunk,
              std::vector<tq_ingest_audit::Row>& rows) {
    const int strips = static_cast<int>(batch * chunk / tq_npu::kBlockSize);
    // --single: no warmup, one timed sample, so a profiler trace holds exactly
    // one dispatch of each of the three kernels. The same flag and the same
    // meaning as bench_turboquant_decode_splitk's.
    const int warmups = opts.warmups;
    const int samples = opts.samples;
    // The two MIX kernels dispatch the recycled-channel entry point by
    // default; --mode4-single-shot switches them to the fallback that gives
    // every event its own id. Run the benchmark both ways to price recycling:
    // the schedules are identical, so any difference is the id allocation and
    // the arbiter's handling of it. The BF16 Vector baseline is AIV-only and
    // has no mode-4 traffic at all, so it is unaffected and serves as the
    // control row.
    tq_npu::Fixture f(strips * tq_npu::kBlockSize);
    // Explicit horizons/batches allocate a full per-request cache and write
    // the final chunk. The legacy default retains its fragmented mapping.
    // The fixture's cached golden is not used by this timing benchmark;
    // validation runners retain their original fixture and reference.
    if (!opts.contexts.empty() || batch != 1) {
        auto layout = tq_ingest_bench::cache_layout(context, batch, chunk);
        f.shape.num_blocks = layout.blocks;
        f.slots = std::move(layout.slots);
    }
    tq_npu::Session session;
    tq_npu::A8Device a8(f);
    tq_npu::Bf16Device bf16_cube(f);
    Baseline bf16_vec(f);
    std::string native_note = opts.skip_native ? "native skipped" : "native not built";
#if defined(ASCEND_MICRO_HAVE_NATIVE_INGEST)
    std::unique_ptr<tq_ingest_bench::NativeV5> native;
    if (!opts.skip_native) {
        try {
            native = std::make_unique<tq_ingest_bench::NativeV5>(f, bf16_cube);
            native_note.clear();
        } catch (const std::exception& e) {
            native_note = std::string("native unavailable: ") + e.what();
            std::fprintf(stderr, "%s\n", native_note.c_str());
        }
    }
#endif
    tq_npu::Buffer eviction_src(kEvictionBytes), eviction_dst(kEvictionBytes);
    tq_npu::check(aclrtMemset(eviction_src.ptr, eviction_src.bytes, 0xA5,
                              eviction_src.bytes), "initialize eviction buffer");
    // Bytes actually moved, not the entire B*S cache allocation. Only the
    // B*chunk_tokens input rows and their packed pages/scales are touched.
    const std::uint64_t written_cache =
        static_cast<std::uint64_t>(f.strips()) * f.shape.page_bytes();
    const std::uint64_t written_scales =
        static_cast<std::uint64_t>(tq_oracle::kPlanes) * f.strips() * tq_npu::kHeads *
        tq_npu::kBlockSize * sizeof(float);
    const double a8_bytes = 2.0 * f.key.size() + 2.0 * written_cache + written_scales;
    const double bf16_bytes = 4.0 * f.key.size() + 2.0 * written_cache + written_scales;
    const int waves = (f.strips() + 31) / 32;
    std::printf("\nModel label: %s; S=%s; B=%s; chunk_tokens=%s; "
                "fixed 4 KV heads x 256 channels\n", model.c_str(),
                tq_ingest_audit::commas(context).c_str(),
                tq_ingest_audit::commas(batch).c_str(),
                tq_ingest_audit::commas(chunk).c_str());
    std::printf("%d tokens, %d strips over %d physical pages, %d warmups, "
                "%d device-event run%s%s\n",
                f.shape.num_tokens, f.strips(), f.shape.num_blocks, warmups,
                samples, samples == 1 ? "" : "s",
                opts.single ? " (--single)" : "");
    std::printf("All three kernels dispatch one launch of %d grid blocks; the "
                "two MIX kernels span %d physical wave(s) of 32 and use %s\n",
                f.strips(), waves, tq_npu::mode4_name(opts.recycled));
    for (bool evict : {false, true}) {
        const char* mode = evict ? "pressure" : "warm";
        const Stats sa = measure(session.stream,
                                 [&] {
                                     a8.launch(session.stream, f,
                                               opts.recycled);
                                 },
                                 eviction_src, eviction_dst, evict, warmups,
                                 samples);
        const Stats sc = measure(session.stream,
                                 [&] {
                                     bf16_cube.launch(session.stream, f,
                                                      opts.recycled);
                                 },
                                 eviction_src, eviction_dst, evict, warmups,
                                 samples);
        const Stats sv = measure(session.stream,
                                 [&] { bf16_vec.launch(session.stream, f); },
                                 eviction_src, eviction_dst, evict, warmups,
                                 samples);
        report(mode, "A8", sa, a8_bytes);
        report(mode, "BF16-cube", sc, bf16_bytes);
        report(mode, "BF16-vec", sv, bf16_bytes);
        std::optional<Stats> sn;
#if defined(ASCEND_MICRO_HAVE_NATIVE_INGEST)
        if (native) {
            try {
                sn = measure(session.stream, [&] { native->launch(session.stream); },
                             eviction_src, eviction_dst, evict, warmups, samples);
                report(mode, "Native-V5", *sn, 8.0 * f.key.size());
            } catch (const std::exception& e) {
                // A failed optional leg must not poison later measurements.
                tq_npu::check(aclrtSynchronizeStream(session.stream), "native failure synchronize");
                native_note = std::string("native unavailable: ") + e.what();
                std::fprintf(stderr, "%s\n", native_note.c_str());
                native.reset();
            }
        }
#endif
        tq_ingest_audit::Row row;
        row.model = model; row.mode = mode; row.context = context;
        row.batch = batch; row.chunk_tokens = chunk; row.recycled = opts.recycled;
        row.a8 = sa; row.cube = sc; row.vector = sv; row.native = sn;
        row.native_note = native_note;
        row.bytes_written = 2 * written_cache + written_scales;
        row.bytes_moved = static_cast<std::uint64_t>(bf16_bytes);
        rows.push_back(row);
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
    const double cube_rotation_ops = 2.0 * f.shape.num_tokens * 2 *
                                     tq_npu::kHeads * tq_npu::kHeadSize *
                                     tq_npu::kHeadSize;
    const double vector_rotation_ops = 2.0 * f.shape.num_tokens * 2 *
                                       tq_npu::kHeads * 8 *
                                       (tq_npu::kHeadSize / 2);
    std::printf("Rotation-only intensity: A8 Cube %.1f ops/GM byte, "
                "BF16 Cube %.1f ops/GM byte, BF16 Vector %.1f ops/GM byte. "
                "GM bytes exclude tables and slot metadata; every estimate "
                "excludes quantization.\n",
                cube_rotation_ops / a8_bytes, cube_rotation_ops / bf16_bytes,
                vector_rotation_ops / bf16_bytes);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto opts = tq_ingest_bench::parse(argc, argv, tq_npu::kDefaultMode4Reuse);
        if (opts.help) { tq_ingest_bench::usage(argv[0]); return 0; }
        std::vector<tq_ingest_audit::Row> rows;
        for (const auto& model : opts.models)
            for (auto chunk : opts.chunks)
                for (auto context : (opts.contexts.empty()
                        ? std::vector<std::uint64_t>{chunk} : opts.contexts))
                    for (auto batch : opts.batches)
                        run_case(opts, model, context, batch, chunk, rows);
        std::puts("\nIngestion audit (model labels use the fixed 4H x 256 fixture).\n"
                  "Cube cell: median_cube (median_vec / min_cube), all in us.\n"
                  "Speedups: native/cube (native/vector); bandwidth counts packed K/V "
                  "and scale writes only, excluding input reads and unwritten cache.\n");
        tq_ingest_audit::PrintTableA(stdout, rows);
        tq_ingest_audit::AppendPrefillCsvFromEnv(rows);
        return 0;
    } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "Invalid ingest benchmark options: %s\n", e.what());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL turboquant NPU benchmark: %s\n", e.what());
        return 1;
    }
}
