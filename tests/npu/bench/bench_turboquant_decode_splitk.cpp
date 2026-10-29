// Physical-device benchmark for the TurboQuant Split-K decode path.
// Ascend 950PR / dav-3510 / CANN 9.2. HARDWARE ONLY -- see "Where this can
// run" below.
//
//   --splitk "<list>"   comma-separated K sweep           (default 1,2,4,8,16)
//   --context <N>       context length in tokens          (default 1024)
//   --warmup <N>        untimed iterations                (default 10)
//   --iters <N>         timed iterations                  (default 100)
//   --impl <t>          aic | aiv | both                  (default both)
//   --sync-mode <t>     atomic | workspace | both         (default both)
//   --cache <t>         warm | cold | both                (default warm)
//   --watchdog <N>      spin attempts before giving up    (default 100000)
//
// What it measures, in three separate tables
// ------------------------------------------
// 1. **Decode**, per `--impl`. The production Cube operator
//    (`turboquant_paged_decode_attention`, MIX_AIC_1_2) and the AIV-only one
//    (`turboquant_single_token_decode_aiv`). Independent of K: the number of
//    partial states a decode produces is set by the context, not by the sweep.
// 2. **Reduction**, per (`--sync-mode`, K). `workspace` writes each split's
//    state to its own slice and stops; `atomic` accumulates in place under a
//    Global Memory spinlock with all K splits contending for one line. There
//    is deliberately **no combine kernel**: the question is what L2 line
//    bouncing and lock serialisation cost across physical cores, and a tidy-up
//    pass afterwards would be measured work nobody asked about.
// 3. **End to end**, per (`--impl`, `--sync-mode`, K), with both phases inside
//    one timed sample, because that is the quantity a scheduler sees.
//
// Each table carries min / mean / p50 / p95 / p99 in microseconds from ACL
// device events, the host-side enqueue cost measured separately, and a
// throughput figure over the bytes actually moved.
//
// The tail effect
// ---------------
// Every split stamps the absolute system cycle counter at entry and exit of
// its measured region and publishes both as (hi, lo) int32 pairs. The counter
// is global across cores on this part, so the host can compute
//
//   tail delta = last split to finish - first split to finish
//   span       = last exit - first entry
//
// which is the straggler spread the task asks for. Those stamps are read once,
// after the timed loop, on a dedicated extra iteration: a device-to-host copy
// inside the timed loop would be measuring the copy.
//
// Where this can run, and what was actually verified
// --------------------------------------------------
// This binary needs a 950PR. The development host for this harness has none --
// no `npu-smi`, no `/dev/davinci*`, no driver tree, only the CAModel
// simulator runtimes -- so what is verified here is the clean NPU-mode build,
// exactly as `reports/reshape_and_cache_a8.md` records for the
// reshape-and-cache benchmark ("compile-only -- there is no device here").
// The CLI, the statistics, the tail arithmetic and the watchdog reporting are
// exercised through the simulator twin target `bench_decode_splitk_sim`, whose
// timings are cycle-accurate model ticks and emphatically **not** hardware
// latencies. Any number this binary prints on a real device is a measurement;
// any number its simulator twin prints is a model figure, and it labels which
// it is at startup.
//
// Anti-hang
// ---------
// Every acquire is bounded by `--watchdog` failed exchanges, after which the
// split gives up without the lock and publishes that it did; the release only
// clears a lock the split actually holds. A launch therefore cannot spin
// forever, and any run with a nonzero timeout count is reported as INVALID
// rather than averaged in.

#include "../turboquant_npu_common.hpp"

#include "../../common/paged_decode_oracle.hpp"
#include "../../common/single_token_decode_oracle.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

extern "C" void launch_camodel_paged_decode_attention(
    void*, std::uint16_t*, std::uint16_t*, std::int8_t*, std::int8_t*, float*,
    std::uint32_t*, float*, std::uint32_t*, float*, float*, std::uint16_t*, float*,
    float*, float*, std::uint16_t*, float*, std::uint32_t, std::uint32_t,
    std::uint32_t);

extern "C" void launch_turboquant_single_token_decode_aiv(
    void*, std::uint32_t, std::uint16_t*, std::int8_t*, std::int8_t*, float*,
    std::uint32_t*, float*, float*, float*, float*, float*, float*, std::uint16_t*,
    std::int32_t*, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t);

extern "C" void launch_turboquant_splitk_reduce_workspace(
    void*, std::uint32_t, float*, float*, std::uint32_t*, std::uint32_t*,
    std::uint32_t*, std::uint32_t*, std::uint32_t*, float*, std::int32_t*,
    std::uint32_t, std::uint32_t);

extern "C" void launch_turboquant_splitk_reduce_atomic(
    void*, std::uint32_t, float*, float*, std::uint32_t*, std::uint32_t*,
    std::uint32_t*, std::uint32_t*, std::uint32_t*, float*, std::int32_t*,
    std::uint32_t, std::uint32_t);

namespace {

using tq_npu::Buffer;
using tq_npu::check;
using tq_npu::require;

// ---------------------------------------------------------------- geometry
// Mirrors routines/routine_probe_l2_atomic_sync.hpp. Restated because a host
// translation unit does not include a device routine header.
constexpr std::uint32_t kChannels = 256;
constexpr std::uint32_t kStateLanes = 8;
constexpr std::uint32_t kStateElems = 2u * kStateLanes;            // 16
constexpr std::uint32_t kRecordElems = kStateElems + kChannels;    // 272
constexpr std::uint32_t kRecordBytes = kRecordElems * 4u;          // 1088
constexpr std::uint32_t kMOffset = 0;
constexpr std::uint32_t kLOffset = kStateLanes;
constexpr std::uint32_t kAccOffset = kStateElems;
constexpr std::uint32_t kMaxSplits = 16;
constexpr std::uint32_t kMetricSlots = 24;
constexpr std::uint32_t kMetricLanes = 8;
constexpr std::uint32_t kMetricElems = kMetricSlots * kMetricLanes;
// Slot indices, from the routine's SplitKMetric.
constexpr std::uint32_t kSlotBlock = 0, kSlotRegion = 3;
constexpr std::uint32_t kSlotBeginHi = 9, kSlotBeginLo = 10;
constexpr std::uint32_t kSlotEndHi = 11, kSlotEndLo = 12;

// One reduction per launch is the unit a scheduler issues, so the kernel's
// per-split repetition count is pinned at one here. The CAModel probe sweeps
// it to reach a steady state; a wall-clock benchmark does not want several
// reductions folded into one event window.
constexpr std::uint32_t kReduceReps = 1;

// 128 MiB of device-to-device traffic, enough to evict L2 between iterations.
constexpr std::size_t kEvictionBytes = 128u * 1024u * 1024u;

// The decode operators' fixed temporal geometry: 8 pages of 128 tokens per
// head, so 1024 tokens is the whole context one launch can address. A longer
// context is covered by repeated launches ("waves"), all of them inside the
// timed sample, exactly as the reshape-and-cache benchmark does for its
// windowed Cube path.
//
// What that does and does not model. The work per wave is a real decode over a
// real 1024-token page set, so `--context 4096` issues four times the decode
// work of `--context 1024` and the timing of that work is meaningful. But the
// KV cache fixture has only `kNumBlocks` physical pages and the block table
// maps eight temporal blocks, so the later waves re-read the SAME pages: the
// run never touches 4096 distinct tokens and its cache footprint does not grow
// with the context. A multi-wave figure is therefore a statement about decode
// throughput at that amount of work, not about how the operators behave with a
// genuinely longer sequence resident. The binary prints this at startup rather
// than leaving it to be discovered.
constexpr int kTokensPerPage = paged_decode::kBlockSize;             // 128
constexpr int kPagesPerWave = paged_decode::kTemporalBlocks;         // 8
constexpr int kTokensPerWave = kTokensPerPage * kPagesPerWave;       // 1024

// ------------------------------------------------------------------ options
struct Options {
    std::vector<std::uint32_t> splitk{1, 2, 4, 8, 16};
    int context = 1024;
    int warmup = 10;
    int iters = 100;
    bool aic = true, aiv = true;
    bool atomic = true, workspace = true;
    bool warm = true, cold = false;
    std::uint32_t watchdog = 100000;
};

int parse_int(std::string_view text, long long& out)
{
    const auto r = std::from_chars(text.data(), text.data() + text.size(), out);
    return r.ec == std::errc() && r.ptr == text.data() + text.size() ? 0 : -1;
}

std::vector<std::uint32_t> parse_list(std::string_view text)
{
    std::vector<std::uint32_t> out;
    std::size_t at = 0;
    while (at <= text.size()) {
        const std::size_t comma = text.find(',', at);
        const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
        std::string_view item = text.substr(at, end - at);
        while (!item.empty() && (item.front() == ' ' || item.front() == '"'))
            item.remove_prefix(1);
        while (!item.empty() && (item.back() == ' ' || item.back() == '"'))
            item.remove_suffix(1);
        if (!item.empty()) {
            long long value = 0;
            require(parse_int(item, value) == 0, "--splitk takes integers");
            require(value >= 1 && value <= static_cast<long long>(kMaxSplits),
                    "every --splitk entry must be 1..16");
            out.push_back(static_cast<std::uint32_t>(value));
        }
        if (comma == std::string_view::npos) break;
        at = comma + 1;
    }
    require(!out.empty(), "--splitk produced an empty sweep");
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

Options parse(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const bool has_value = i + 1 < argc;
        auto value = [&]() -> std::string_view {
            require(has_value, "option needs a value");
            return argv[++i];
        };
        if (arg == "--splitk") {
            o.splitk = parse_list(value());
        } else if (arg == "--context") {
            long long v = 0;
            require(parse_int(value(), v) == 0 && v > 0 && v % kTokensPerPage == 0,
                    "--context must be a positive multiple of 128 tokens");
            o.context = static_cast<int>(v);
        } else if (arg == "--warmup") {
            long long v = 0;
            require(parse_int(value(), v) == 0 && v >= 0, "--warmup must be >= 0");
            o.warmup = static_cast<int>(v);
        } else if (arg == "--iters") {
            long long v = 0;
            require(parse_int(value(), v) == 0 && v >= 1, "--iters must be >= 1");
            o.iters = static_cast<int>(v);
        } else if (arg == "--impl") {
            const std::string_view v = value();
            o.aic = v == "aic" || v == "both";
            o.aiv = v == "aiv" || v == "both";
            require(o.aic || o.aiv, "--impl must be aic, aiv or both");
        } else if (arg == "--sync-mode") {
            const std::string_view v = value();
            o.atomic = v == "atomic" || v == "both";
            o.workspace = v == "workspace" || v == "both";
            require(o.atomic || o.workspace,
                    "--sync-mode must be atomic, workspace or both");
        } else if (arg == "--cache") {
            const std::string_view v = value();
            o.warm = v == "warm" || v == "both";
            o.cold = v == "cold" || v == "both";
            require(o.warm || o.cold, "--cache must be warm, cold or both");
        } else if (arg == "--watchdog") {
            long long v = 0;
            require(parse_int(value(), v) == 0 && v >= 0, "--watchdog must be >= 0");
            o.watchdog = static_cast<std::uint32_t>(v);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    return o;
}

// ------------------------------------------------------------- measurement
struct Stats {
    double min = 0, mean = 0, p50 = 0, p95 = 0, p99 = 0;
};

double percentile(const std::vector<double>& sorted, double q)
{
    if (sorted.empty()) return 0.0;
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(pos);
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

Stats summarize(std::vector<double> samples)
{
    require(!samples.empty(), "no samples to summarize");
    std::sort(samples.begin(), samples.end());
    Stats s;
    s.min = samples.front();
    s.mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
             static_cast<double>(samples.size());
    s.p50 = percentile(samples, 0.50);
    s.p95 = percentile(samples, 0.95);
    s.p99 = percentile(samples, 0.99);
    return s;
}

struct Events {
    aclrtEvent start = nullptr, end = nullptr;
    Events()
    {
        check(aclrtCreateEvent(&start), "aclrtCreateEvent start");
        check(aclrtCreateEvent(&end), "aclrtCreateEvent end");
    }
    Events(const Events&) = delete;
    Events& operator=(const Events&) = delete;
    ~Events()
    {
        if (start) aclrtDestroyEvent(start);
        if (end) aclrtDestroyEvent(end);
    }
};

struct Sample {
    Stats device;  // microseconds, from ACL events
    Stats host;    // microseconds, enqueue only
    // True when every ACL event interval came back as exactly zero. That is
    // not a fast kernel, it is an event API that did not measure: the CAModel
    // runtime returns 0 from `aclrtEventElapsedTime` for every interval, with
    // no error at all. Measured on the simulator twin -- a run whose model
    // tick counter advanced by 131,798 and whose host enqueue timings were
    // milliseconds still reported 0.000 us of device time on every line.
    bool device_timing_dead = false;
};

// Times `launch` with ACL events for the device interval and a host clock for
// the enqueue interval. The host figure is taken around the launch calls alone,
// before any synchronisation, so it is the cost of getting the work onto the
// stream and not the cost of waiting for it.
template <typename Launch>
Sample measure(aclrtStream stream, Launch&& launch, int warmup, int iters,
               Buffer* evict_src, Buffer* evict_dst)
{
    for (int i = 0; i < warmup; ++i) launch();
    check(aclrtSynchronizeStream(stream), "warmup synchronize");

    std::vector<double> device_us, host_us;
    device_us.reserve(static_cast<std::size_t>(iters));
    host_us.reserve(static_cast<std::size_t>(iters));
    Events events;
    for (int i = 0; i < iters; ++i) {
        if (evict_src != nullptr) {
            check(aclrtMemcpy(evict_dst->ptr, evict_dst->bytes, evict_src->ptr,
                              evict_src->bytes, ACL_MEMCPY_DEVICE_TO_DEVICE),
                  "L2 eviction D2D copy");
            check(aclrtSynchronizeStream(stream), "eviction synchronize");
        }
        const auto host_begin = std::chrono::steady_clock::now();
        check(aclrtRecordEvent(events.start, stream), "record start");
        launch();
        check(aclrtRecordEvent(events.end, stream), "record end");
        const auto host_end = std::chrono::steady_clock::now();
        check(aclrtSynchronizeEvent(events.end), "synchronize end event");

        float ms = 0;
        check(aclrtEventElapsedTime(&ms, events.start, events.end),
              "aclrtEventElapsedTime");
        require(ms >= 0, "negative device event duration");
        device_us.push_back(static_cast<double>(ms) * 1000.0);
        host_us.push_back(
            std::chrono::duration<double, std::micro>(host_end - host_begin).count());
    }
    const bool dead = std::all_of(device_us.begin(), device_us.end(),
                                  [](double v) { return v == 0.0; });
    return {summarize(std::move(device_us)), summarize(std::move(host_us)), dead};
}

void report(const char* label, const Sample& s, double bytes)
{
    if (s.device_timing_dead) {
        // No device interval at all: report the host figure, which did
        // measure, and say plainly that the device clock did not. Printing
        // 0.000 us here would read as an infinitely fast kernel.
        std::printf("  %-34s dev UNAVAILABLE (aclrtEventElapsedTime returned 0 "
                    "for every interval)   host p50 %7.3f us\n",
                    label, s.host.p50);
        return;
    }
    std::printf("  %-34s dev min %9.3f mean %9.3f p50 %9.3f p95 %9.3f p99 %9.3f us",
                label, s.device.min, s.device.mean, s.device.p50, s.device.p95,
                s.device.p99);
    if (bytes > 0.0 && s.device.p50 > 0.0)
        std::printf("  %8.2f GB/s", bytes / s.device.p50 / 1000.0);
    else
        std::printf("  %8s     ", "-");
    std::printf("  host p50 %7.3f us\n", s.host.p50);
}

// ---------------------------------------------------- the reduction fixture
// Uniform m across the splits, so every rescale factor is exactly 1.0f and the
// accumulated result is a plain sum the host can check in closed form. That is
// not for precision's sake: a benchmark of a kernel that silently dropped
// updates would report excellent numbers, so the atomic mode's result is
// checked once before the timings are believed.
struct ReduceFixture {
    std::vector<float> tiles;
    explicit ReduceFixture(std::uint32_t splits)
        : tiles(static_cast<std::size_t>(kMaxSplits) * kRecordElems, 0.0f)
    {
        for (std::uint32_t k = 0; k < splits; ++k) {
            float* rec = tiles.data() + static_cast<std::size_t>(k) * kRecordElems;
            for (std::uint32_t i = 0; i < kStateLanes; ++i) {
                rec[kMOffset + i] = -4.0f;
                rec[kLOffset + i] = 1.0f;
            }
            for (std::uint32_t c = 0; c < kChannels; ++c) rec[kAccOffset + c] = 1.0f;
        }
    }
};

struct ReduceDevice {
    Buffer tiles, workspace, lock, ticket, ticket_log, spins, acquired, atomic_m,
        metrics;
    explicit ReduceDevice(const ReduceFixture& f)
        : tiles(f.tiles.size() * sizeof(float)),
          workspace(static_cast<std::size_t>(kMaxSplits) * kRecordBytes),
          lock(32), ticket(32),
          ticket_log(static_cast<std::size_t>(kMaxSplits) * kMaxSplits * 4),
          spins(static_cast<std::size_t>(kMaxSplits) * kMaxSplits * 4),
          acquired(static_cast<std::size_t>(kMaxSplits) * kMaxSplits * 4),
          atomic_m(32),
          metrics(static_cast<std::size_t>(kMaxSplits) * kMetricElems * 4)
    {
        tiles.upload(f.tiles);
    }

    // The shared slot starts at the combine identity: m a large negative so
    // the first rescale factor underflows to an exact zero rather than
    // producing a NaN from inf - inf, l and Acc at zero.
    void reset(aclrtStream stream)
    {
        std::vector<float> ws(workspace.bytes / sizeof(float), 0.0f);
        for (std::uint32_t k = 0; k < kMaxSplits; ++k) {
            float* rec = ws.data() + static_cast<std::size_t>(k) * kRecordElems;
            for (std::uint32_t i = 0; i < kStateLanes; ++i) rec[kMOffset + i] = -3.0e38f;
        }
        workspace.upload(ws);
        check(aclrtMemset(lock.ptr, lock.bytes, 0, lock.bytes), "reset lock");
        check(aclrtMemset(ticket.ptr, ticket.bytes, 0, ticket.bytes), "reset ticket");
        check(aclrtMemset(ticket_log.ptr, ticket_log.bytes, 0xFF, ticket_log.bytes),
              "reset ticket log");
        check(aclrtMemset(spins.ptr, spins.bytes, 0, spins.bytes), "reset spins");
        check(aclrtMemset(acquired.ptr, acquired.bytes, 0, acquired.bytes),
              "reset acquired");
        std::vector<float> m(atomic_m.bytes / sizeof(float), -3.0e38f);
        atomic_m.upload(m);
        check(aclrtMemset(metrics.ptr, metrics.bytes, 0, metrics.bytes),
              "reset metrics");
        check(aclrtSynchronizeStream(stream), "reset synchronize");
    }

    void launch(aclrtStream stream, bool atomic, std::uint32_t splits,
                std::uint32_t watchdog)
    {
        auto* fn = atomic ? launch_turboquant_splitk_reduce_atomic
                          : launch_turboquant_splitk_reduce_workspace;
        fn(stream, splits, static_cast<float*>(tiles.ptr),
           static_cast<float*>(workspace.ptr),
           static_cast<std::uint32_t*>(lock.ptr),
           static_cast<std::uint32_t*>(ticket.ptr),
           static_cast<std::uint32_t*>(ticket_log.ptr),
           static_cast<std::uint32_t*>(spins.ptr),
           static_cast<std::uint32_t*>(acquired.ptr),
           static_cast<float*>(atomic_m.ptr),
           static_cast<std::int32_t*>(metrics.ptr), kReduceReps, watchdog);
    }
};

// A split's metric slice, as the host reads it.
struct SplitMetrics {
    std::int32_t block = 0, region = 0;
    std::int64_t begin = 0, end = 0;
    bool written = false;
};

std::int64_t join(std::int32_t hi, std::int32_t lo)
{
    return static_cast<std::int64_t>((static_cast<std::uint64_t>(
                                          static_cast<std::uint32_t>(hi)) << 32) |
                                     static_cast<std::uint32_t>(lo));
}

std::vector<SplitMetrics> read_metrics(const ReduceDevice& d, std::uint32_t splits)
{
    std::vector<std::int32_t> raw(d.metrics.bytes / sizeof(std::int32_t));
    d.metrics.download(raw);
    std::vector<SplitMetrics> out(splits);
    for (std::uint32_t k = 0; k < splits; ++k) {
        const std::int32_t* slice = raw.data() + static_cast<std::size_t>(k) * kMetricElems;
        auto at = [&](std::uint32_t slot) { return slice[slot * kMetricLanes]; };
        out[k].block = at(kSlotBlock);
        out[k].region = at(kSlotRegion);
        out[k].begin = join(at(kSlotBeginHi), at(kSlotBeginLo));
        out[k].end = join(at(kSlotEndHi), at(kSlotEndLo));
        out[k].written = out[k].end != 0;
    }
    return out;
}

// The straggler spread. `tail` is the gap between the first and the last split
// to finish; `span` is from the first entry to the last exit.
struct Tail {
    std::int64_t tail = 0, span = 0, fastest = 0, slowest = 0;
    bool valid = false;
};

Tail tail_of(const std::vector<SplitMetrics>& m)
{
    Tail t;
    std::int64_t first_end = 0, last_end = 0, first_begin = 0;
    bool any = false;
    for (const SplitMetrics& s : m) {
        if (!s.written) continue;
        if (!any) {
            first_end = last_end = s.end;
            first_begin = s.begin;
            t.fastest = t.slowest = s.region;
            any = true;
            continue;
        }
        first_end = std::min(first_end, s.end);
        last_end = std::max(last_end, s.end);
        first_begin = std::min(first_begin, s.begin);
        t.fastest = std::min<std::int64_t>(t.fastest, s.region);
        t.slowest = std::max<std::int64_t>(t.slowest, s.region);
    }
    if (!any) return t;
    t.tail = last_end - first_end;
    t.span = last_end - first_begin;
    t.valid = true;
    return t;
}

// The watchdog and contention census, straight out of Global Memory.
struct LockCensus {
    std::uint64_t acquisitions = 0, timeouts = 0, spins = 0, worst_spin = 0;
};

LockCensus census_of(const ReduceDevice& d, std::uint32_t splits)
{
    const std::size_t words = static_cast<std::size_t>(splits) * kReduceReps;
    std::vector<std::uint32_t> acquired(d.acquired.bytes / 4), spins(d.spins.bytes / 4);
    d.acquired.download(acquired);
    d.spins.download(spins);
    LockCensus c;
    for (std::size_t i = 0; i < words; ++i) {
        if (acquired[i] != 0u)
            ++c.acquisitions;
        else
            ++c.timeouts;
        c.spins += spins[i];
        c.worst_spin = std::max<std::uint64_t>(c.worst_spin, spins[i]);
    }
    return c;
}

// The atomic mode's closed-form check: with uniform m every factor is exactly
// 1, so one reduction over K splits leaves l = K and Acc[c] = K.
bool atomic_result_is_sound(const ReduceDevice& d, std::uint32_t splits)
{
    std::vector<float> ws(d.workspace.bytes / sizeof(float));
    d.workspace.download(ws);
    const float want = static_cast<float>(splits * kReduceReps);
    if (ws[kLOffset] != want) {
        std::printf("    WARNING combine denominator %.1f, expected %.1f: the "
                    "timings below are of a kernel that lost updates\n",
                    static_cast<double>(ws[kLOffset]), static_cast<double>(want));
        return false;
    }
    for (std::uint32_t c = 0; c < kChannels; ++c)
        if (ws[kAccOffset + c] != want) {
            std::printf("    WARNING combine accumulator wrong at channel %u: "
                        "%.1f, expected %.1f\n", c,
                        static_cast<double>(ws[kAccOffset + c]),
                        static_cast<double>(want));
            return false;
        }
    return true;
}

// ------------------------------------------------------- the decode fixtures
// Both decode operators are FROZEN; these build exactly the operands their
// CAModel runners build, with the verification machinery left out. The
// benchmark checks that a decode launch wrote something, not that it wrote the
// right thing: that is what tests/camodel/test_camodel_paged_decode_attention
// .cpp and its twelve rungs are for.
struct AicDecode {
    Buffer q, pi_table, key_cache, value_cache, scale_cache, block_table, gate,
        gather_offsets, gather_signs, pi_sign, out_qrot, out_scores, out_state,
        out_acc, out_context, out_rotated;

    AicDecode()
        : q(static_cast<std::size_t>(paged_decode::kKvHeads) * paged_decode::kTokens *
            paged_decode::kHeadSize * 2),
          pi_table(static_cast<std::size_t>(paged_decode::kTableElems) * 2),
          key_cache(paged_decode::kKeyCacheBytes),
          value_cache(paged_decode::kValueCacheBytes),
          scale_cache(paged_decode::kScaleCacheFloats * 4),
          block_table(static_cast<std::size_t>(paged_decode::kTemporalBlocks) * 4),
          gate(static_cast<std::size_t>(paged_decode::kKvHeads) * paged_decode::kTokens *
               paged_decode::kHeadSize * 4),
          gather_offsets(static_cast<std::size_t>(paged_decode::kEgressGatherCount) *
                         paged_decode::kHeadSize * 4),
          gather_signs(static_cast<std::size_t>(paged_decode::kEgressGatherCount) *
                       paged_decode::kHeadSize * 4),
          pi_sign(static_cast<std::size_t>(paged_decode::kHeadSize) * 4),
          out_qrot(static_cast<std::size_t>(paged_decode::kGrid) *
                   paged_decode::kTokens * paged_decode::kHeadSize * 2),
          out_scores(static_cast<std::size_t>(paged_decode::kGrid) *
                     paged_decode::kStrips * paged_decode::kScores * 4),
          out_state(static_cast<std::size_t>(paged_decode::kGrid) * 2u *
                    paged_decode::kTokens * kStateLanes * 4),
          out_acc(static_cast<std::size_t>(paged_decode::kGrid) * paged_decode::kTokens *
                  paged_decode::kHeadSize * 4),
          out_context(static_cast<std::size_t>(paged_decode::kGrid) *
                      paged_decode::kTokens * paged_decode::kHeadSize * 2),
          out_rotated(static_cast<std::size_t>(paged_decode::kGrid) *
                      paged_decode::kTokens * paged_decode::kHeadSize * 4)
    {
        using namespace paged_decode;
        std::vector<std::uint8_t> keys(kKeyCacheBytes, 0), values(kValueCacheBytes, 0);
        std::vector<float> scales(kScaleCacheFloats, 0.0f);
        for (int temporal = 0; temporal < kTemporalBlocks; ++temporal) {
            const int page = BlockTableEntry(temporal);
            for (int head = 0; head < kKvHeads; ++head)
                for (int strip = 0; strip < kStrips; ++strip) {
                    const CacheFill where{page, head, strip};
                    WriteKeyStrip(keys.data(), where, strip);
                    WriteValueStrip(values.data(), where, strip);
                    WriteScales(scales.data(), where, strip);
                }
        }
        key_cache.upload(keys);
        value_cache.upload(values);
        scale_cache.upload(scales);

        std::vector<std::uint16_t> q_host(q.bytes / 2);
        for (int head = 0; head < kKvHeads; ++head)
            for (int row = 0; row < kTokens; ++row)
                for (int c = 0; c < kHeadSize; ++c)
                    q_host[(static_cast<std::size_t>(head) * kTokens + row) * kHeadSize +
                           c] = ascend_micro::sim::float_to_bfloat16_rne(
                        QueryEntry(row, c));
        q.upload(q_host);

        const RotationTable table = BuildRotationTable();
        std::vector<std::uint16_t> table_host(table.row_major.begin(),
                                              table.row_major.end());
        pi_table.upload(table_host);

        std::vector<float> gate_host(gate.bytes / 4);
        for (int head = 0; head < kKvHeads; ++head)
            for (int row = 0; row < kTokens; ++row)
                for (int c = 0; c < kHeadSize; ++c)
                    gate_host[(static_cast<std::size_t>(head) * kTokens + row) *
                                  kHeadSize + c] = GateEntry(head, row, c);
        gate.upload(gate_host);

        std::vector<std::uint32_t> table_words(kTemporalBlocks);
        for (int t = 0; t < kTemporalBlocks; ++t)
            table_words[t] = static_cast<std::uint32_t>(BlockTableEntry(t));
        block_table.upload(table_words);

        std::vector<std::uint32_t> offsets(gather_offsets.bytes / 4);
        std::vector<float> signs(gather_signs.bytes / 4);
        for (int stage = 0; stage < kEgressGatherCount; ++stage) {
            const int stride = kEgressGatherStride[stage];
            for (int i = 0; i < kHeadSize; ++i) {
                offsets[static_cast<std::size_t>(stage) * kHeadSize + i] =
                    static_cast<std::uint32_t>((i ^ stride) * 4);
                signs[static_cast<std::size_t>(stage) * kHeadSize + i] =
                    (i & stride) ? -1.0f : 1.0f;
            }
        }
        gather_offsets.upload(offsets);
        gather_signs.upload(signs);
        std::vector<float> pi(kHeadSize);
        for (int c = 0; c < kHeadSize; ++c) pi[c] = static_cast<float>(PiSign(c));
        pi_sign.upload(pi);
    }

    void launch(aclrtStream stream)
    {
        launch_camodel_paged_decode_attention(
            stream, static_cast<std::uint16_t*>(q.ptr),
            static_cast<std::uint16_t*>(pi_table.ptr),
            static_cast<std::int8_t*>(key_cache.ptr),
            static_cast<std::int8_t*>(value_cache.ptr),
            static_cast<float*>(scale_cache.ptr),
            static_cast<std::uint32_t*>(block_table.ptr),
            static_cast<float*>(gate.ptr),
            static_cast<std::uint32_t*>(gather_offsets.ptr),
            static_cast<float*>(gather_signs.ptr), static_cast<float*>(pi_sign.ptr),
            static_cast<std::uint16_t*>(out_qrot.ptr),
            static_cast<float*>(out_scores.ptr), static_cast<float*>(out_state.ptr),
            static_cast<float*>(out_acc.ptr),
            static_cast<std::uint16_t*>(out_context.ptr),
            static_cast<float*>(out_rotated.ptr),
            static_cast<std::uint32_t>(paged_decode::kStrips), 0u,
            static_cast<std::uint32_t>(paged_decode::kGrid));
    }
};

struct AivDecode {
    Buffer q_rot, key_cache, value_cache, scale_cache, block_table, gate, pi_sign,
        out_scores, out_state, out_acc, out_rotated, out_context, probe;
    std::uint32_t block_dim;

    AivDecode()
        : q_rot(static_cast<std::size_t>(single_token_decode::kQHeads) *
                single_token_decode::kHeadSize * 2),
          key_cache(paged_decode::kKeyCacheBytes),
          value_cache(paged_decode::kValueCacheBytes),
          scale_cache(paged_decode::kScaleCacheFloats * 4),
          block_table(static_cast<std::size_t>(single_token_decode::kMaxPages) * 4),
          gate(static_cast<std::size_t>(single_token_decode::kQHeads) *
               single_token_decode::kHeadSize * 4),
          pi_sign(static_cast<std::size_t>(single_token_decode::kHeadSize) * 4),
          out_scores(static_cast<std::size_t>(single_token_decode::kItems) *
                     single_token_decode::kMaxStrips *
                     single_token_decode::kPanelEntries * 4),
          out_state(static_cast<std::size_t>(single_token_decode::kItems) * 2u *
                    single_token_decode::kRows * kStateLanes * 4),
          out_acc(static_cast<std::size_t>(single_token_decode::kItems) *
                  single_token_decode::kRowElems * 4),
          out_rotated(static_cast<std::size_t>(single_token_decode::kItems) *
                      single_token_decode::kRowElems * 4),
          out_context(static_cast<std::size_t>(single_token_decode::kItems) *
                      single_token_decode::kRowElems * 2),
          probe(32), block_dim(single_token_decode::kItems)
    {
        using namespace single_token_decode;
        std::vector<std::uint8_t> keys(paged_decode::kKeyCacheBytes, 0);
        std::vector<std::uint8_t> values(paged_decode::kValueCacheBytes, 0);
        std::vector<float> scales(paged_decode::kScaleCacheFloats, 0.0f);
        for (int head = 0; head < kKvHeads; ++head)
            for (int strip = 0; strip < kMaxStrips; ++strip) {
                const int page = PhysicalPageOfStrip(strip);
                const int in_page_strip = StripInPage(strip);
                std::uint8_t* key_strip =
                    keys.data() + paged_decode::KeyStripOffset(page, head, in_page_strip);
                std::uint8_t* value_strip =
                    values.data() +
                    paged_decode::ValueStripOffset(page, head, in_page_strip);
                for (int token = 0; token < kKeys; ++token)
                    for (int c = 0; c < kHeadSize; ++c)
                        paged_decode::PackNibble(
                            key_strip, paged_decode::KeyNibbleIndex(token, c),
                            KeyNibbleAt(head, strip, token, c));
                for (int c = 0; c < kHeadSize; ++c)
                    for (int token = 0; token < kKeys; ++token)
                        paged_decode::PackNibble(
                            value_strip, paged_decode::ValueNibbleIndex(c, token),
                            ValueNibbleAt(head, strip, c, token));
                for (int token = 0; token < kKeys; ++token) {
                    const int at = in_page_strip * kKeys + token;
                    scales[paged_decode::ScaleOffset(0, page, head, at)] =
                        KeyScale(head, strip, token);
                    scales[paged_decode::ScaleOffset(1, page, head, at)] =
                        ValueScaleAt(head, strip, token);
                }
            }
        key_cache.upload(keys);
        value_cache.upload(values);
        scale_cache.upload(scales);

        std::vector<std::uint16_t> qrot_host(q_rot.bytes / 2);
        std::vector<float> gate_host(gate.bytes / 4);
        for (int head = 0; head < kKvHeads; ++head)
            for (int row = 0; row < kRows; ++row)
                for (int c = 0; c < kHeadSize; ++c) {
                    const std::size_t at =
                        (static_cast<std::size_t>(head) * kRows + row) * kHeadSize + c;
                    qrot_host[at] = ascend_micro::sim::float_to_bfloat16_rne(
                        QRotEntry(head, row, c));
                    gate_host[at] = GateEntry(head, QueryRowOf(head, row), c);
                }
        q_rot.upload(qrot_host);
        gate.upload(gate_host);

        std::vector<std::uint32_t> pages(kMaxPages);
        for (int p = 0; p < kMaxPages; ++p)
            pages[p] = static_cast<std::uint32_t>(paged_decode::BlockTableEntry(p));
        block_table.upload(pages);
        std::vector<float> pi(kHeadSize);
        for (int c = 0; c < kHeadSize; ++c)
            pi[c] = static_cast<float>(paged_decode::PiSign(c));
        pi_sign.upload(pi);
    }

    void launch(aclrtStream stream, std::uint32_t pages)
    {
        using namespace single_token_decode;
        launch_turboquant_single_token_decode_aiv(
            stream, block_dim, static_cast<std::uint16_t*>(q_rot.ptr),
            static_cast<std::int8_t*>(key_cache.ptr),
            static_cast<std::int8_t*>(value_cache.ptr),
            static_cast<float*>(scale_cache.ptr),
            static_cast<std::uint32_t*>(block_table.ptr),
            static_cast<float*>(gate.ptr), static_cast<float*>(pi_sign.ptr),
            static_cast<float*>(out_scores.ptr), static_cast<float*>(out_state.ptr),
            static_cast<float*>(out_acc.ptr), static_cast<float*>(out_rotated.ptr),
            static_cast<std::uint16_t*>(out_context.ptr),
            static_cast<std::int32_t*>(probe.ptr),
            static_cast<std::uint32_t>(kItems), static_cast<std::uint32_t>(kKvHeads),
            static_cast<std::uint32_t>(kRows), static_cast<std::uint32_t>(kHeadSize),
            static_cast<std::uint32_t>(kKeys),
            static_cast<std::uint32_t>(paged_decode::kNumBlocks), pages,
            static_cast<std::uint32_t>(kStripsPerPage * pages), 0u, 0u,
            static_cast<std::uint32_t>(kItems));
    }
};

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options o = parse(argc, argv);
        const int waves = (o.context + kTokensPerWave - 1) / kTokensPerWave;
        const int pages = std::min(o.context / kTokensPerPage, kPagesPerWave);

        std::printf("TurboQuant Split-K decode benchmark, Ascend950PR_9599\n");
        std::printf("context %d tokens = %d page(s) of %d, %d decode wave(s) of at "
                    "most %d pages, all inside the timed sample\n",
                    o.context, o.context / kTokensPerPage, kTokensPerPage, waves,
                    kPagesPerWave);
        if (waves > 1)
            std::printf("NOTE a context past %d tokens is EMULATED: the KV cache "
                        "fixture holds %d physical pages and the operators' block "
                        "table maps %d temporal blocks, so the later waves re-read "
                        "the same pages. The decode WORK per wave is real; the token "
                        "count is not. Read this as %d waves of decode, not as %d "
                        "distinct tokens resident.\n",
                        kTokensPerWave, paged_decode::kNumBlocks, kPagesPerWave,
                        waves, o.context);
        std::printf("warmup %d, timed iterations %d, watchdog %u spin attempt(s)%s\n",
                    o.warmup, o.iters, o.watchdog,
                    o.watchdog == 0 ? " (DISABLED -- a lost lock will hang)" : "");
        std::printf("split-K sweep:");
        for (std::uint32_t k : o.splitk) std::printf(" %u", k);
        std::printf("\n");
#if defined(ASCEND_MICRO_BENCH_SIM)
        std::printf("*** SIMULATOR BUILD: every figure below is a cycle-accurate "
                    "CAModel tick, NOT a hardware latency. This build exists to "
                    "exercise the harness where no NPU is present. ***\n");
#endif
        std::printf("\n");

        tq_npu::Session session;
        // 256 MiB of eviction buffers, allocated only when a cold rung asks
        // for them: the simulator twin would otherwise pay for a quarter of a
        // gigabyte it never touches.
        std::unique_ptr<Buffer> evict_src, evict_dst;
        if (o.cold) {
            evict_src.reset(new Buffer(kEvictionBytes));
            evict_dst.reset(new Buffer(kEvictionBytes));
            check(aclrtMemset(evict_src->ptr, evict_src->bytes, 0xA5,
                              evict_src->bytes),
                  "initialize eviction buffer");
        }

        // ------------------------------------------------------- 1. decode
        std::printf("1. decode kernels (independent of K: the context sets how many "
                    "partials a decode produces)\n");
        // Both fixtures allocate the whole KV cache, so they are built only
        // when their kernel is actually in the sweep.
        std::unique_ptr<AicDecode> aic_owned;
        std::unique_ptr<AivDecode> aiv_owned;
        if (o.aic) aic_owned.reset(new AicDecode());
        if (o.aiv) aiv_owned.reset(new AivDecode());
        AicDecode* const aic = aic_owned.get();
        AivDecode* const aiv = aiv_owned.get();

        // Bytes a decode actually moves out of the KV cache: two packed planes
        // of 2048 bytes per (page, head, strip) plus the two scale slices.
        const double decode_bytes =
            static_cast<double>(waves) * paged_decode::kTemporalBlocks *
            paged_decode::kKvHeads * paged_decode::kStrips *
            (2.0 * paged_decode::kPackedBytes + 2.0 * paged_decode::kTokens * 4);

        for (bool cold : {false, true}) {
            if (cold && !o.cold) continue;
            if (!cold && !o.warm) continue;
            Buffer* es = cold ? evict_src.get() : nullptr;
            Buffer* ed = cold ? evict_dst.get() : nullptr;
            const char* tag = cold ? "cold" : "warm";
            if (aic != nullptr) {
                const Sample s = measure(session.stream,
                                     [&] {
                                         for (int w = 0; w < waves; ++w)
                                             aic->launch(session.stream);
                                     },
                                     o.warmup, o.iters, es, ed);
                char label[64];
                std::snprintf(label, sizeof label, "%s aic paged_decode", tag);
                report(label, s, decode_bytes);
            }
            if (aiv != nullptr) {
                const Sample s = measure(session.stream,
                                     [&] {
                                         for (int w = 0; w < waves; ++w)
                                             aiv->launch(session.stream,
                                                         static_cast<std::uint32_t>(pages));
                                     },
                                     o.warmup, o.iters, es, ed);
                char label[64];
                std::snprintf(label, sizeof label, "%s aiv single_token", tag);
                report(label, s, decode_bytes);
            }
        }

        // ---------------------------------------------------- 2. reduction
        std::printf("\n2. split-K reduction, per sync mode and K "
                    "(no combine kernel: the atomic path is measured in place)\n");
        for (bool atomic : {false, true}) {
            if (atomic && !o.atomic) continue;
            if (!atomic && !o.workspace) continue;
            const char* mode = atomic ? "atomic" : "workspace";
            for (std::uint32_t k : o.splitk) {
                ReduceFixture fixture(k);
                ReduceDevice device(fixture);
                device.reset(session.stream);

                for (bool cold : {false, true}) {
                    if (cold && !o.cold) continue;
                    if (!cold && !o.warm) continue;
                    Buffer* es = cold ? evict_src.get() : nullptr;
                    Buffer* ed = cold ? evict_dst.get() : nullptr;
                    // The shared slot accumulates across iterations, which is
                    // fine for timing and useless for checking, so the result
                    // check runs on its own reset launch below.
                    const Sample s = measure(
                        session.stream,
                        [&] { device.launch(session.stream, atomic, k, o.watchdog); },
                        o.warmup, o.iters, es, ed);
                    // Workspace moves one record out per split; atomic moves
                    // one in and one out, on one shared line.
                    const double bytes = static_cast<double>(k) * kRecordBytes *
                                         (atomic ? 2.0 : 1.0);
                    char label[80];
                    std::snprintf(label, sizeof label, "%s %-9s K=%-2u",
                                  cold ? "cold" : "warm", mode, k);
                    report(label, s, bytes);
                }

                // One clean launch for the tail, the census and the result
                // check, outside every timed sample.
                device.reset(session.stream);
                device.launch(session.stream, atomic, k, o.watchdog);
                check(aclrtSynchronizeStream(session.stream), "tail launch sync");
                const Tail t = tail_of(read_metrics(device, k));
                const LockCensus c = census_of(device, k);
                if (t.valid)
                    std::printf("    K=%-2u tail delta %6lld cycle(s), span %6lld, "
                                "per-split region %lld..%lld\n",
                                k, static_cast<long long>(t.tail),
                                static_cast<long long>(t.span),
                                static_cast<long long>(t.fastest),
                                static_cast<long long>(t.slowest));
                else
                    std::printf("    K=%-2u no per-split timestamps were published\n", k);
                if (atomic) {
                    std::printf("    K=%-2u lock: %llu acquisition(s), %llu timeout(s), "
                                "%llu failed exchange(s), worst %llu\n",
                                k, static_cast<unsigned long long>(c.acquisitions),
                                static_cast<unsigned long long>(c.timeouts),
                                static_cast<unsigned long long>(c.spins),
                                static_cast<unsigned long long>(c.worst_spin));
                    if (c.timeouts != 0)
                        std::printf("    K=%-2u INVALID: the watchdog gave up %llu "
                                    "time(s); raise --watchdog or treat the timings "
                                    "above as meaningless\n",
                                    k, static_cast<unsigned long long>(c.timeouts));
                    else if (!atomic_result_is_sound(device, k))
                        std::printf("    K=%-2u INVALID: the combine result is wrong\n", k);
                }
            }
        }

        // --------------------------------------------------- 3. end to end
        if ((o.aic || o.aiv) && (o.atomic || o.workspace)) {
            std::printf("\n3. end to end, decode plus reduction in one timed sample\n");
            for (bool atomic : {false, true}) {
                if (atomic && !o.atomic) continue;
                if (!atomic && !o.workspace) continue;
                const char* mode = atomic ? "atomic" : "workspace";
                for (std::uint32_t k : o.splitk) {
                    ReduceFixture fixture(k);
                    ReduceDevice device(fixture);
                    device.reset(session.stream);
                    if (aic != nullptr) {
                        const Sample s = measure(
                            session.stream,
                            [&] {
                                for (int w = 0; w < waves; ++w) aic->launch(session.stream);
                                device.launch(session.stream, atomic, k, o.watchdog);
                            },
                            o.warmup, o.iters, nullptr, nullptr);
                        char label[80];
                        std::snprintf(label, sizeof label, "aic + %-9s K=%-2u", mode, k);
                        report(label, s, decode_bytes);
                    }
                    if (aiv != nullptr) {
                        const Sample s = measure(
                            session.stream,
                            [&] {
                                for (int w = 0; w < waves; ++w)
                                    aiv->launch(session.stream,
                                                static_cast<std::uint32_t>(pages));
                                device.launch(session.stream, atomic, k, o.watchdog);
                            },
                            o.warmup, o.iters, nullptr, nullptr);
                        char label[80];
                        std::snprintf(label, sizeof label, "aiv + %-9s K=%-2u", mode, k);
                        report(label, s, decode_bytes);
                    }
                }
            }
        }

        std::printf("\nthroughput is over the bytes the phase actually moves: a decode "
                    "reads two 2048-byte packed planes and two 16-float scale slices "
                    "per (page, head, strip); a reduction moves one 1088-byte record "
                    "per split, twice in the atomic mode (read-modify-write).\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL turboquant split-K decode benchmark: %s\n", e.what());
        return 1;
    }
}
