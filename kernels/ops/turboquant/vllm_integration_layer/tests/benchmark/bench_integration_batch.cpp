// Prefill reshape-and-cache benchmark for the staged BF16 writer,
// `turboquant_reshape_and_cache_bf16`.
//
// WHY THIS EXISTS. The staged subtree carries three kernels and, until this file, two
// benchmarks: bench_integration_pipeline for the fused decode operator and
// bench_integration_scatter for the AIV scatter ingress. The prefill batch writer -- the
// third staged kernel, and the one the vLLM ingress path actually calls per prefill step --
// had a validation ladder (tests/validation/test_integration_batch.cpp) and no timed runner
// at all, so nothing in this subtree could say what it costs.
//
// WHAT IT MEASURES. One monolithic dispatch of one 16-token strip per grid block, which is
// how the operator ships: at 33 strips and above the grid is wider than the physical AI Core
// count, so a single launch spans several execution waves and the figure includes the
// inter-wave protocol's drain (op_kernel/common/turboquant_wave_protocol.hpp). There is no
// launch window and no per-wave capping -- a capped runner would time a different operator
// than the one the ladder validates.
//
// EVERY RUNG IS VERIFIED BEFORE IT IS TIMED. The first launch of each rung goes through
// tq_npu::verify, the same bit-exact per-page oracle comparison the validation ladder uses:
// every key byte, value byte and FP32 scale against tests/common/turboquant_oracle.hpp, with
// the pages no strip owns required to still hold their 0xAA canary. A benchmark that times a
// kernel writing the wrong bytes is worse than no benchmark, and on this operator the cheapest
// way to be fast is to skip work. The timed iterations that follow reuse that same device
// state, so they measure the steady-state launch and not the verification.
//
// THE CAMODEL LEG. Built with -DASCEND_MICRO_BENCH_SIM (cmake/SimTargets.cmake registers it
// that way) the binary says so at startup and records NO event times: the simulator's stub
// driver rejects ACL timing events, exactly as bench_integration_pipeline documents. On that
// leg the run is a functional pass over the ladder -- which is still the only leg that can
// RUN at all on a host without a 950PR -- and the latency columns read "NA (CAModel)".
//
// Usage:
//   bench_integration_batch [--strips N[,N...]] [--warmup K] [--iters K]
//                           [--mode4-single-shot] [--csv-out PATH]
// The default ladder is tq_npu::default_ladder(): 1, 32, 33, 48, 64 and 128 strips, which is
// cold start, a full wave, the first block of the second wave, a ragged second wave, two whole
// waves and four.

#include "../npu/turboquant_npu_common.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {
using tq_npu::check;
using tq_npu::require;

struct Options {
    std::vector<int> strips = tq_npu::default_ladder();
    int warmup = 2;
    int iters = 10;
    bool reuse_mode4 = tq_npu::kDefaultMode4Reuse;
    std::string csv_out;
};

void usage(std::FILE* out, const char* argv0)
{
    std::fprintf(out,
                 "Usage: %s [--strips N[,N...]] [--warmup K] [--iters K]\n"
                 "          [--mode4-single-shot] [--csv-out PATH]\n"
                 "  --strips  aligned 16-token strips per rung; one monolithic grid of\n"
                 "            that many blocks per launch (default 1,32,33,48,64,128)\n"
                 "  --warmup  untimed launches per rung, each one verified (default 2)\n"
                 "  --iters   timed launches per rung (default 10)\n"
                 "  --mode4-single-shot  dispatch the fallback channel policy that gives\n"
                 "            every cross-core event its own id, instead of the recycled\n"
                 "            default; the two differ only in id allocation\n"
                 "  --csv-out write one row per rung to PATH\n",
                 argv0);
}

bool parse_int(std::string_view text, int& out)
{
    int value = 0;
    const auto* end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc() || result.ptr != end) return false;
    out = value;
    return true;
}

bool parse_strips(std::string_view text, std::vector<int>& out)
{
    std::vector<int> parsed;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        const std::string_view field = text.substr(0, comma);
        int value = 0;
        if (!parse_int(field, value) || value < 1 || value > tq_npu::kMaxStrips) return false;
        parsed.push_back(value);
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    if (parsed.empty()) return false;
    out = std::move(parsed);
    return true;
}

// 0 continue, 1 exit success, 2 usage error.
int parse(int argc, char** argv, Options& o)
{
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(stdout, argv[0]);
            return 1;
        }
        if (arg == "--mode4-single-shot") {
            o.reuse_mode4 = false;
            continue;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "%.*s needs a value\n", static_cast<int>(arg.size()),
                         arg.data());
            return 2;
        }
        const std::string_view value = argv[++i];
        if (arg == "--strips") {
            if (!parse_strips(value, o.strips)) {
                std::fprintf(stderr, "--strips must be 1..%d values, comma separated\n",
                             tq_npu::kMaxStrips);
                return 2;
            }
        } else if (arg == "--warmup") {
            if (!parse_int(value, o.warmup) || o.warmup < 0) {
                std::fprintf(stderr, "--warmup must be >= 0\n");
                return 2;
            }
        } else if (arg == "--iters") {
            if (!parse_int(value, o.iters) || o.iters < 1) {
                std::fprintf(stderr, "--iters must be >= 1\n");
                return 2;
            }
        } else if (arg == "--csv-out") {
            o.csv_out = std::string(value);
        } else {
            std::fprintf(stderr, "unknown argument: %.*s\n", static_cast<int>(arg.size()),
                         arg.data());
            usage(stderr, argv[0]);
            return 2;
        }
    }
    return 0;
}

double percentile(std::vector<double> samples, double fraction)
{
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    const std::size_t last = samples.size() - 1;
    std::size_t index = static_cast<std::size_t>(fraction * static_cast<double>(last) + 0.5);
    if (index > last) index = last;
    return samples[index];
}

// The bytes one launch actually moves, from the shape rather than from a measurement: the
// BF16 activations read, the two int4 cache planes written, and the FP32 scales. Only the
// pages a strip owns are counted -- the fixture deliberately allocates twice as many, and
// charging the kernel for canary pages it never touches would understate its bandwidth by
// half.
struct Traffic {
    double read_bytes = 0, write_bytes = 0;
    double total() const { return read_bytes + write_bytes; }
};

Traffic traffic_of(const tq_npu::Fixture& f)
{
    const auto strips = static_cast<double>(f.strips());
    const auto page_bytes = static_cast<double>(f.shape.page_bytes());
    const double scale_floats = static_cast<double>(tq_oracle::kPlanes) *
                                f.shape.num_kv_heads * f.shape.block_size;
    Traffic t;
    // Key and value activations, BF16, every token of every head.
    t.read_bytes = 2.0 * static_cast<double>(f.key.size()) * sizeof(std::uint16_t);
    // Key and value cache pages, plus the scale slot of each owned page.
    t.write_bytes = 2.0 * strips * page_bytes + strips * scale_floats * sizeof(float);
    return t;
}

struct Rung {
    int strips = 0;
    int tokens = 0;
    bool timed = false;
    double p50_us = 0, p95_us = 0, min_us = 0;
    double gbs = 0;
    std::uint64_t launches = 0;
};

Rung run_rung(aclrtStream stream, const Options& o, int strips, aclrtEvent start,
              aclrtEvent stop)
{
    const tq_npu::Fixture fixture(strips * tq_npu::kBlockSize);
    tq_npu::Bf16Device device(fixture);

    Rung rung;
    rung.strips = fixture.strips();
    rung.tokens = fixture.tokens();

    // The correctness gate: one launch, then the full bit-exact per-page comparison. verify()
    // throws on any mismatch, stray canary byte or untouched owned page.
    device.launch(stream, fixture, o.reuse_mode4);
    check(aclrtSynchronizeStream(stream), "verification synchronize");
    ++rung.launches;
    tq_npu::verify(fixture, device);

    for (int i = 0; i < o.warmup; ++i) {
        device.launch(stream, fixture, o.reuse_mode4);
        check(aclrtSynchronizeStream(stream), "warmup synchronize");
        ++rung.launches;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(o.iters));
    bool valid_events = true;
    for (int i = 0; i < o.iters; ++i) {
#if defined(ASCEND_MICRO_BENCH_SIM)
        (void)start;
        (void)stop;
        device.launch(stream, fixture, o.reuse_mode4);
        check(aclrtSynchronizeStream(stream), "CAModel post-launch synchronize");
        valid_events = false;
#else
        check(aclrtRecordEvent(start, stream), "record start");
        device.launch(stream, fixture, o.reuse_mode4);
        check(aclrtRecordEvent(stop, stream), "record stop");
        check(aclrtSynchronizeEvent(stop), "synchronize stop event");
        float ms = 0;
        check(aclrtEventElapsedTime(&ms, start, stop), "event elapsed time");
        require(std::isfinite(ms) && ms >= 0, "invalid event duration");
        if (ms <= 0) valid_events = false;
        samples.push_back(static_cast<double>(ms) * 1000.0);
#endif
        ++rung.launches;
    }

    rung.timed = valid_events && !samples.empty();
    if (rung.timed) {
        rung.p50_us = percentile(samples, 0.50);
        rung.p95_us = percentile(samples, 0.95);
        rung.min_us = *std::min_element(samples.begin(), samples.end());
        const Traffic t = traffic_of(fixture);
        rung.gbs = t.total() / (rung.p50_us * 1e-6) / 1e9;
    }
    return rung;
}

}  // namespace

int main(int argc, char** argv)
{
    Options o;
    const int parsed = parse(argc, argv, o);
    if (parsed != 0) return parsed == 1 ? 0 : 2;

    try {
#if defined(ASCEND_MICRO_BENCH_SIM)
        std::puts("CAModel functional run: ACL event timing is unavailable on the simulator,");
        std::puts("so the latency columns are NA. Every rung is still launched and verified.");
#else
        std::puts("Ascend950PR prefill reshape-and-cache benchmark: BF16 input, int4 strip cache.");
#endif
        std::printf("kernel=turboquant_reshape_and_cache_bf16 heads=%d d=%d block=%d "
                    "mode4=%s warmup=%d iters=%d\n",
                    tq_npu::kHeads, tq_npu::kHeadSize, tq_npu::kBlockSize,
                    tq_npu::mode4_name(o.reuse_mode4), o.warmup, o.iters);
        std::puts("One monolithic dispatch of one strip per block; grids above 32 blocks span "
                  "several waves.");

        std::ofstream csv;
        if (!o.csv_out.empty()) {
            csv.open(o.csv_out, std::ios::out | std::ios::trunc);
            require(csv.is_open(), "cannot open --csv-out path");
            csv << "strips,tokens,grid,p50_us,p95_us,min_us,traffic_gbs,launches\n";
        }

        tq_npu::Session session;
        aclrtEvent start = nullptr, stop = nullptr;
#if !defined(ASCEND_MICRO_BENCH_SIM)
        check(aclrtCreateEvent(&start), "aclrtCreateEvent(start)");
        check(aclrtCreateEvent(&stop), "aclrtCreateEvent(stop)");
#endif

        std::puts("+---------+----------+--------+-------------+-------------+-------------+--------------+-----------+");
        std::printf("| %7s | %8s | %6s | %11s | %11s | %11s | %12s | %9s |\n", "Strips",
                    "Tokens", "Grid", "p50 (us)", "p95 (us)", "min (us)", "Traffic GB/s",
                    "Launches");
        std::puts("+---------+----------+--------+-------------+-------------+-------------+--------------+-----------+");

        for (int strips : o.strips) {
            const Rung rung = run_rung(session.stream, o, strips, start, stop);
            if (rung.timed)
                std::printf("| %7d | %8d | %6d | %11.3f | %11.3f | %11.3f | %12.2f | %9llu |\n",
                            rung.strips, rung.tokens, rung.strips, rung.p50_us, rung.p95_us,
                            rung.min_us, rung.gbs,
                            static_cast<unsigned long long>(rung.launches));
            else
                std::printf("| %7d | %8d | %6d | %11s | %11s | %11s | %12s | %9llu |\n",
                            rung.strips, rung.tokens, rung.strips, "NA (CAModel)", "NA", "NA",
                            "NA", static_cast<unsigned long long>(rung.launches));
            if (csv.is_open()) {
                csv << rung.strips << ',' << rung.tokens << ',' << rung.strips << ',';
                if (rung.timed)
                    csv << rung.p50_us << ',' << rung.p95_us << ',' << rung.min_us << ','
                        << rung.gbs << ',';
                else
                    csv << "NA,NA,NA,NA,";
                csv << rung.launches << '\n';
            }
        }
        std::puts("+---------+----------+--------+-------------+-------------+-------------+--------------+-----------+");
        std::puts("Traffic counts BF16 activations read plus owned cache pages and scales "
                  "written; canary pages are excluded.");

#if !defined(ASCEND_MICRO_BENCH_SIM)
        if (stop) aclrtDestroyEvent(stop);
        if (start) aclrtDestroyEvent(start);
#endif
        std::puts("PASS: every rung verified bit-exactly against the shared oracle before timing.");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL prefill reshape-and-cache benchmark: %s\n", error.what());
        return 1;
    }
}
