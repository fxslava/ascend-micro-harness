// Single-launch, real page-partitioned Split-K decode benchmark.
// No synthetic partial states or cyclic page waves. Each sample resets its
// head slots outside the ACL event interval, then times exactly one launch.
#include "../turboquant_npu_common.hpp"
#include "../../common/splitk_decode_fixture.hpp"
#include "../../../include/turboquant_splitk_contract.hpp"

#include <charconv>
#include <cmath>
#include <limits>
#include <numeric>
#include <string_view>

extern "C" void launch_turboquant_single_token_decode_splitk_aiv(
    void*, std::uint16_t*, std::int8_t*, std::int8_t*, float*, std::uint32_t*,
    float*, float*, turboquant_splitk::AtomicHeadSlot*,
    turboquant_splitk::WorkerMetrics*, std::uint16_t*, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t);

namespace {
using tq_npu::Buffer;
using tq_npu::check;
using tq_npu::require;
namespace contract = turboquant_splitk;
using splitk_decode_test::Fixture;
using splitk_decode_test::kChannels;

struct Options {
    std::vector<int> contexts{1024, 2048, 4096, 8192, 16384};
    std::vector<int> splits{1, 2, 4, 8, 12, 16};
    int warmup = 5, iters = 30, requests = 1, watchdog = 1000000;
    int heads = 4, rows = 6, page_tokens = 128;
};

int number(std::string_view s)
{
    int n = 0;
    const auto result = std::from_chars(s.data(), s.data() + s.size(), n);
    require(result.ec == std::errc{} && result.ptr == s.data() + s.size(),
            "expected an integer");
    return n;
}

std::vector<int> list(std::string_view s)
{
    std::vector<int> out;
    while (!s.empty()) {
        const auto pos = s.find(',');
        out.push_back(number(s.substr(0, pos)));
        if (pos == std::string_view::npos) break;
        s.remove_prefix(pos + 1);
        require(!s.empty(), "trailing comma");
    }
    require(!out.empty(), "empty list");
    return out;
}

Options parse(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help") {
            std::puts("--context N[,N...] --splitk K[,K...] --warmup N --iters N "
                      "--requests N --watchdog N --heads N --rows N --page-tokens N\n"
                      "Defaults: contexts 1024,2048,4096,8192,16384; K 1,2,4,8,12,16.\n"
                      "Use --page-tokens 16 --heads 1 --context 48 for a short CAModel run.");
            std::exit(0);
        }
        require(i + 1 < argc, "option requires a value");
        const std::string_view value(argv[++i]);
        if (arg == "--context") o.contexts = list(value);
        else if (arg == "--splitk") o.splits = list(value);
        else if (arg == "--warmup") o.warmup = number(value);
        else if (arg == "--iters") o.iters = number(value);
        else if (arg == "--requests") o.requests = number(value);
        else if (arg == "--heads") o.heads = number(value);
        else if (arg == "--rows") o.rows = number(value);
        else if (arg == "--page-tokens") o.page_tokens = number(value);
        else if (arg == "--watchdog") o.watchdog = number(value);
        else throw std::runtime_error("unknown option: " + std::string(arg));
    }
    require(o.page_tokens == 16 || o.page_tokens == 128, "page-tokens must be 16 or 128");
    require(o.heads > 0 && o.heads <= 4 && o.rows > 0 && o.rows <= 8, "heads must be 1..4 and rows 1..8");
    for (int n : o.contexts)
        require(n > 0 && n <= 16384 && n % o.page_tokens == 0,
                "context must be a multiple of page-tokens, at most 16384");
    for (int k : o.splits) require(k >= 1 && k <= 16, "splitk must be 1..16");
    require(o.iters > 0 && o.warmup >= 0 && o.requests > 0 && o.requests <= 16 &&
            o.watchdog > 0, "invalid iteration, batch or watchdog bound");
    return o;
}

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

struct Device {
    const Fixture& f;
    Buffer q, key, value, scales, table, gate, signs, slots, metrics, out;
    Device(const Fixture& fixture, int splits)
        : f(fixture), q(f.q.size() * 2), key(f.key.size()), value(f.value.size()),
          scales(f.scales.size() * 4), table((f.table.size() * 4 + 31) / 32 * 32),
          gate(f.gate.size() * 4), signs(f.signs.size() * 4),
          slots(f.requests * f.heads * sizeof(contract::AtomicHeadSlot)),
          metrics(f.requests * f.heads * splits * sizeof(contract::WorkerMetrics)),
          out(f.q.size() * 2)
    {
        q.upload(f.q); key.upload(f.key); value.upload(f.value);
        scales.upload(f.scales); gate.upload(f.gate); signs.upload(f.signs);
        auto padded = f.table;
        padded.resize(table.bytes / 4, 0);
        table.upload(padded);
        require(reinterpret_cast<std::uintptr_t>(slots.ptr) % 128 == 0,
                "head slot allocation must be 128-byte aligned");
    }
    void Reset()
    {
        check(aclrtMemset(slots.ptr, slots.bytes, 0, slots.bytes), "reset head slots");
        check(aclrtMemset(metrics.ptr, metrics.bytes, 0, metrics.bytes), "reset metrics");
        out.fill_canary();
    }
    void Launch(aclrtStream stream, int splits, int watchdog)
    {
        launch_turboquant_single_token_decode_splitk_aiv(stream,
            static_cast<std::uint16_t*>(q.ptr), static_cast<std::int8_t*>(key.ptr),
            static_cast<std::int8_t*>(value.ptr), static_cast<float*>(scales.ptr),
            static_cast<std::uint32_t*>(table.ptr), static_cast<float*>(gate.ptr),
            static_cast<float*>(signs.ptr), static_cast<contract::AtomicHeadSlot*>(slots.ptr),
            static_cast<contract::WorkerMetrics*>(metrics.ptr),
            static_cast<std::uint16_t*>(out.ptr), f.requests, f.heads, f.rows,
            f.page_tokens, f.requests * f.pages, f.pages, splits, watchdog);
    }
    struct Census {
        std::uint64_t failed = 0, timeouts = 0;
        std::vector<double> tail_ticks, lock_ticks;
    };
    Census Validate(int splits)
    {
        std::vector<contract::WorkerMetrics> m(metrics.bytes / sizeof(contract::WorkerMetrics));
        metrics.download(m);
        std::vector<contract::AtomicHeadSlot> s(f.requests * f.heads);
        slots.download(s);
        Census result;
        for (int head = 0; head < f.requests * f.heads; ++head) {
            std::vector<bool> arrivals(splits + 1, false);
            std::uint64_t first = std::numeric_limits<std::uint64_t>::max(), tail_end = 0;
            int tails = 0;
            for (int split = 0; split < splits; ++split) {
                const auto& w = m[head * splits + split];
                result.failed += w.failed_exchanges;
                result.timeouts += w.timeout;
                if (!w.acquired || w.timeout)
                    std::fprintf(stderr, "INVALID lock: head=%d split=%d acquired=%u "
                                 "timeouts=%u failed_exchanges=%u\n", head, split,
                                 w.acquired, w.timeout, w.failed_exchanges);
                require(w.acquired == 1 && w.timeout == 0, "INVALID: lock watchdog expired");
                require(w.arrival >= 1 && w.arrival <= static_cast<unsigned>(splits) &&
                        !arrivals[w.arrival], "arrival tickets are not a permutation");
                arrivals[w.arrival] = true;
                const auto range = contract::Partition(f.pages, splits, split);
                require(w.pages[0] == range.count, "wrong per-split page count");
                for (int lane = 1; lane < 8; ++lane)
                    require(w.finish_hi[lane] == w.finish_hi[0] &&
                            w.finish_lo[lane] == w.finish_lo[0] &&
                            w.tail[lane] == w.tail[0] && w.pages[lane] == w.pages[0] &&
                            w.lock_region_hi[lane] == w.lock_region_hi[0] &&
                            w.lock_region_lo[lane] == w.lock_region_lo[0],
                            "torn metric payload");
                const std::uint64_t finish = (std::uint64_t{w.finish_hi[0]} << 32) |
                                              w.finish_lo[0];
                require(finish != 0, "missing completion timestamp");
                const std::uint64_t lock_region = (std::uint64_t{w.lock_region_hi[0]} << 32) |
                                                    w.lock_region_lo[0];
                require(lock_region > 0, "missing lock interval");
                result.lock_ticks.push_back(static_cast<double>(lock_region));
                first = std::min(first, finish);
                require(w.tail[0] == static_cast<unsigned>(w.arrival == static_cast<unsigned>(splits)),
                        "wrong tail worker");
                if (w.tail[0]) { ++tails; tail_end = finish; }
            }
            require(tails == 1 && s[head].lock == 0 &&
                    s[head].arrived_count == static_cast<unsigned>(splits), "unfinished head slot");
            require(tail_end >= first, "noncomparable completion timestamps");
            result.tail_ticks.push_back(static_cast<double>(tail_end - first));
        }
        std::vector<std::uint16_t> actual(f.q.size());
        out.download(actual);
        for (std::size_t row = 0; row < actual.size() / kChannels; ++row) {
            float scale = 0;
            for (int c = 0; c < kChannels; ++c)
                scale = std::max(scale, std::abs(f.expected[row * kChannels + c]));
            // BF16 rounding consumes at most 1/256 of a row's maximum.
            const float tolerance = 0.005f * scale + 1e-6f;
            for (int c = 0; c < kChannels; ++c) {
                const auto at = row * kChannels + c;
                const float got = single_token_decode::bfloat16_to_float(actual[at]);
                if (!std::isfinite(got) || std::abs(got - f.expected[at]) > tolerance) {
                    std::fprintf(stderr, "context mismatch row=%zu c=%d got=%.9g ref=%.9g tol=%.9g\n",
                                 row, c, got, f.expected[at], tolerance);
                    throw std::runtime_error("CPU reference validation failed");
                }
            }
        }
        return result;
    }
};

}  // namespace

int main(int argc, char** argv)
{
    try {
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
        const Options o = parse(argc, argv);
#if defined(ASCEND_MICRO_BENCH_SIM)
        std::puts("CAModel functional run: event times/speedups are unavailable; completion deltas are model ticks.");
#else
        std::puts("Ascend950PR true Split-K: one decode+merge+tail-egress launch per sample.");
#endif
        tq_npu::Session session;
        Events events;
        std::printf("requests=%d heads=%d rows=%d page_tokens=%d warmup=%d iters=%d\n",
                    o.requests, o.heads, o.rows, o.page_tokens, o.warmup, o.iters);
        std::puts("Reset/poisoning and validation are outside events. GB/s counts packed K+V and FP32 scales.");
        std::puts("lock_region brackets acquire through release, including contention: a conservative hold-time bound.");
        std::puts("N,K,grid,p50_us,p95_us,speedup,KV_GB_s,tail_p50_ticks,tail_p95_ticks,lock_region_p50_ticks,lock_region_p95_ticks,failed_exchanges,timeouts");
        for (int n : o.contexts) {
            Fixture fixture(n, o.requests, o.heads, o.rows, o.page_tokens);
            // Always measure K=1 first so a sweep omitting 1 still has a real baseline.
            std::vector<int> sweep{1};
            for (int k : o.splits)
                if (std::find(sweep.begin(), sweep.end(), k) == sweep.end()) sweep.push_back(k);
            double baseline = 0;
            for (int k : sweep) {
                Device d(fixture, k);
                for (int i = 0; i < o.warmup; ++i) {
                    d.Reset();
                    d.Launch(session.stream, k, o.watchdog);
                    check(aclrtSynchronizeStream(session.stream), "warmup synchronize");
                    d.Validate(k);
                }
                std::vector<double> times, tails, locks;
                std::uint64_t failures = 0, timeouts = 0;
                for (int i = 0; i < o.iters; ++i) {
                    d.Reset();
                    check(aclrtSynchronizeStream(session.stream), "reset synchronize");
                    check(aclrtRecordEvent(events.start, session.stream), "record start");
                    d.Launch(session.stream, k, o.watchdog);
                    check(aclrtRecordEvent(events.end, session.stream), "record end");
                    check(aclrtSynchronizeEvent(events.end), "synchronize end");
                    float ms = 0;
                    check(aclrtEventElapsedTime(&ms, events.start, events.end), "event elapsed time");
                    require(std::isfinite(ms) && ms >= 0, "invalid event duration");
                    times.push_back(static_cast<double>(ms) * 1000);
                    const auto census = d.Validate(k);
                    failures += census.failed;
                    timeouts += census.timeouts;
                    tails.insert(tails.end(), census.tail_ticks.begin(), census.tail_ticks.end());
                    locks.insert(locks.end(), census.lock_ticks.begin(), census.lock_ticks.end());
                }
                const auto latency = summarize(times), tail = summarize(tails), lock = summarize(locks);
                bool timed = latency.min > 0;
#if defined(ASCEND_MICRO_BENCH_SIM)
                timed = false;
#endif
                if (k == 1) baseline = timed ? latency.p50 : 0;
                const double bytes = static_cast<double>(fixture.key.size() + fixture.value.size() +
                                                          fixture.scales.size() * sizeof(float));
                std::printf("%d,%d,%d,", n, k, o.requests * o.heads * k);
                if (timed && baseline > 0)
                    std::printf("%.3f,%.3f,%.3f,%.3f,", latency.p50, latency.p95,
                                baseline / latency.p50, bytes / latency.p50 / 1000);
                else std::printf("NA,NA,NA,NA,");
                std::printf("%.0f,%.0f,%.0f,%.0f,%llu,%llu\n", tail.p50, tail.p95, lock.p50, lock.p95,
                            static_cast<unsigned long long>(failures),
                            static_cast<unsigned long long>(timeouts));
                std::fflush(stdout);
            }
        }
        std::puts("PASS: every measured launch matched the FP64 full-context CPU reference; one tail per head.");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
