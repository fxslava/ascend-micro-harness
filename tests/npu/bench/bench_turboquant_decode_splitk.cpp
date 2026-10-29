// Single-launch, real page-partitioned Split-K decode benchmark.
// No synthetic partial states or cyclic page waves. Each sample resets its
// head slots outside the ACL event interval, then times exactly one launch.
#include "../turboquant_npu_common.hpp"
#include "../../common/aic_splitk_decode_fixture.hpp"
#include "../../../include/turboquant_splitk_contract.hpp"

#include <charconv>
#include <cmath>
#include <limits>
#include <numeric>
#include <string_view>
#include <fstream>
#include <memory>
#include <iomanip>
#include <sstream>
#include <map>

extern "C" void launch_turboquant_single_token_decode_splitk_aiv(
    void*, std::uint16_t*, std::int8_t*, std::int8_t*, float*, std::uint32_t*,
    float*, float*, turboquant_splitk::AtomicHeadSlot*,
    turboquant_splitk::WorkerMetrics*, std::uint16_t*, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t);

extern "C" void launch_turboquant_paged_decode_splitk_aic(
    void*, std::uint16_t*, std::uint16_t*, std::int8_t*, std::int8_t*, float*,
    std::uint32_t*, std::uint16_t*, float*, float*, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);

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
    bool single = false;
    std::string impl = "aiv", csv_out;
    int page_begin = 0, page_count = -1;
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
                      "--impl aiv|aic|both --single --csv-out PATH --page-begin P --page-count P\n"
                      "Defaults: contexts 1024,2048,4096,8192,16384; K 1,2,4,8,12,16.\n"
                      "Use --page-tokens 16 --heads 1 --context 48 for a short CAModel run.");
            std::exit(0);
        }
        if (arg == "--single") { o.single = true; continue; }
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
        else if (arg == "--impl") o.impl = value;
        else if (arg == "--csv-out") o.csv_out = value;
        else if (arg == "--page-begin") o.page_begin = number(value);
        else if (arg == "--page-count") o.page_count = number(value);
        else if (arg == "--watchdog") o.watchdog = number(value);
        else throw std::runtime_error("unknown option: " + std::string(arg));
    }
    if (o.single) { o.warmup = 0; o.iters = 1; }
    require(o.impl == "aiv" || o.impl == "aic" || o.impl == "both", "impl must be aiv, aic or both");
    require(o.page_begin >= 0 && o.page_count >= -1, "invalid AIC page window");
    require(o.page_tokens == 16 || o.page_tokens == 128, "page-tokens must be 16 or 128");
    require(o.heads > 0 && o.heads <= 4 && o.rows > 0 && o.rows <= 8, "heads must be 1..4 and rows 1..8");
    for (int n : o.contexts) {
        require(n > 0 && n <= 16384 && n % o.page_tokens == 0,
                "context must be a multiple of page-tokens, at most 16384");
        const int pages = n / o.page_tokens;
        require(o.page_begin <= pages && (o.page_count < 0 || o.page_count <= pages - o.page_begin),
                "AIC page window exceeds context");
        if (o.impl != "aic") require(o.page_begin == 0 && (o.page_count == -1 || o.page_count == pages),
                "partial page windows require --impl aic; both compares full contexts");
    }
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
#if !defined(ASCEND_MICRO_BENCH_SIM)
        check(aclrtCreateEvent(&start), "aclrtCreateEvent start");
        check(aclrtCreateEvent(&end), "aclrtCreateEvent end");
#endif
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

struct AicDevice {
    const Fixture& f;
    int splits, begin, pages;
    Buffer q, pi, key, value, scales, table, qrot, state, acc;
    std::vector<splitk_decode_test::AicPartial> reference;
    AicDevice(const Fixture& fixture, int k, int first, int count)
        : f(fixture), splits(k), begin(first), pages(count), q(f.q.size() * 2),
          pi(paged_decode::kTableElems * 2), key(f.key.size()), value(f.value.size()),
          scales(f.scales.size() * 4), table((f.table.size() * 4 + 31) / 32 * 32),
          qrot(f.requests * f.heads * k * splitk_decode_test::kAicAccFloats * 2),
          state(f.requests * f.heads * k * splitk_decode_test::kAicStateFloats * 4),
          acc(f.requests * f.heads * k * splitk_decode_test::kAicAccFloats * 4)
    {
        q.upload(splitk_decode_test::AicRawQuery(f));
        pi.upload(paged_decode::BuildRotationTable().row_major);
        key.upload(f.key); value.upload(f.value); scales.upload(f.scales);
        auto padded = f.table; padded.resize(table.bytes / 4, 0); table.upload(padded);
        for (int request = 0; request < f.requests; ++request)
            for (int head = 0; head < f.heads; ++head)
                for (int split = 0; split < k; ++split) {
                    const auto r = contract::Partition(pages, k, split);
                    reference.push_back(splitk_decode_test::ReferenceAicPartial(
                        f, request, head, begin + r.begin, r.count));
                }
    }
    void Reset() { qrot.fill_canary(); state.fill_canary(); acc.fill_canary(); }
    void Launch(aclrtStream stream, int, int)
    {
        launch_turboquant_paged_decode_splitk_aic(stream,
            static_cast<std::uint16_t*>(q.ptr), static_cast<std::uint16_t*>(pi.ptr),
            static_cast<std::int8_t*>(key.ptr), static_cast<std::int8_t*>(value.ptr),
            static_cast<float*>(scales.ptr), static_cast<std::uint32_t*>(table.ptr),
            static_cast<std::uint16_t*>(qrot.ptr), static_cast<float*>(state.ptr),
            static_cast<float*>(acc.ptr), f.requests, f.heads, f.page_tokens,
            f.requests * f.pages, f.pages, splits, begin, pages);
    }
    Device::Census Validate(int)
    {
        std::vector<float> s(state.bytes / 4), a(acc.bytes / 4);
        std::vector<std::uint16_t> qr(qrot.bytes / 2);
        state.download(s); acc.download(a); qrot.download(qr);
        for (std::size_t item = 0; item < reference.size(); ++item) {
            const auto& ref = reference[item];
            const auto range = contract::Partition(pages, splits, item % splits);
            for (int row = 0; row < 16; ++row) {
                const auto state_base = item * splitk_decode_test::kAicStateFloats;
                const auto acc_base = item * splitk_decode_test::kAicAccFloats + row * kChannels;
                const float m = s[state_base + row * 8], l = s[state_base + 128 + row * 8];
                require(std::isfinite(m) && std::isfinite(l), "nonfinite AIC partial state");
                require(std::abs(m - ref.m[row]) <= 2e-6 * std::max(1.0, std::abs(ref.m[row])),
                        "AIC partial maximum differs from CPU reference");
                require(std::abs(l - ref.l[row]) <= 1e-4 * std::max(1.0, ref.l[row]),
                        "AIC partial denominator differs from CPU reference");
                if (!range.count)
                    require(m == -3e38f && l == 0.0f, "empty AIC split did not write exact neutral state");
                for (int lane = 1; lane < 8; ++lane)
                    require(s[state_base + row * 8 + lane] == m &&
                            s[state_base + 128 + row * 8 + lane] == l, "AIC replicated state torn");
                double scale = 0;
                for (int c = 0; c < kChannels; ++c)
                    scale = std::max(scale, std::abs(ref.acc[row * kChannels + c]));
                for (int c = 0; c < kChannels; ++c) {
                    const float got = a[acc_base + c];
                    if (!range.count)
                        require(got == 0.0f && qr[acc_base + c] == 0xAAAAu,
                                "empty AIC split must zero its accumulator and leave query rotation untouched");
                    if (!std::isfinite(got) || std::abs(got - ref.acc[row * kChannels + c]) >
                        2e-4 * scale + 1e-5 * ref.l[row] + 1e-6) {
                        std::fprintf(stderr, "AIC partial mismatch item=%zu row=%d c=%d got=%.9g ref=%.9g\n",
                                     item, row, c, got, ref.acc[row * kChannels + c]);
                        throw std::runtime_error("AIC accumulator reference failed");
                    }
                    if (range.count)
                        require(qr[acc_base + c] == f.q[(item / splits) * 16 * kChannels + row * kChannels + c],
                                "AIC query rotation differs from BF16 reference");
                }
            }
        }
        return {};
    }
};

struct Measurement {
    bool present = false, timed = false;
    double us = 0, p95 = 0, tail = 0, lock = 0;
    std::uint64_t failed = 0, timeouts = 0, launches = 0;
};

template<class D>
Measurement measure(D& d, const Options& o, int k, aclrtStream stream, Events& events)
{
#if defined(ASCEND_MICRO_BENCH_SIM)
    // The simulator reports no device-event timing. Allocating/recording ACL
    // timing events asks its stub driver for unsupported resource type 5.
    (void)events;
#endif
    Measurement result;
    result.present = true;
    for (int i = 0; i < o.warmup; ++i) {
        d.Reset();
        check(aclrtSynchronizeStream(stream), "warmup pre-launch synchronize");
        d.Launch(stream, k, o.watchdog);
        check(aclrtSynchronizeStream(stream), "warmup post-launch synchronize");
        ++result.launches;
        d.Validate(k);
    }
    std::vector<double> times, tails, locks;
    bool valid_events = true;
    for (int i = 0; i < o.iters; ++i) {
        d.Reset();
        check(aclrtSynchronizeStream(stream), "pre-launch synchronize");
#if !defined(ASCEND_MICRO_BENCH_SIM)
        check(aclrtRecordEvent(events.start, stream), "record start");
#endif
        d.Launch(stream, k, o.watchdog);
#if defined(ASCEND_MICRO_BENCH_SIM)
        check(aclrtSynchronizeStream(stream), "CAModel post-launch synchronize");
        const float ms = 0.0f;
#else
        check(aclrtRecordEvent(events.end, stream), "record end");
        if (o.single) check(aclrtSynchronizeStream(stream), "single post-launch synchronize");
        else check(aclrtSynchronizeEvent(events.end), "synchronize end event");
        float ms = 0;
        check(aclrtEventElapsedTime(&ms, events.start, events.end), "event elapsed time");
#endif
        ++result.launches;
        require(std::isfinite(ms) && ms >= 0, "invalid event duration");
        valid_events = valid_events && ms > 0;
        const auto census = d.Validate(k);
        result.failed += census.failed; result.timeouts += census.timeouts;
        if (o.single) {
            // No percentile sorting or multi-sample accumulation in profiling mode.
            result.us = ms * 1000.0;
            result.tail = census.tail_ticks.empty() ? 0 : *std::max_element(census.tail_ticks.begin(), census.tail_ticks.end());
            result.lock = census.lock_ticks.empty() ? 0 : *std::max_element(census.lock_ticks.begin(), census.lock_ticks.end());
        } else {
            times.push_back(ms * 1000.0);
            tails.insert(tails.end(), census.tail_ticks.begin(), census.tail_ticks.end());
            locks.insert(locks.end(), census.lock_ticks.begin(), census.lock_ticks.end());
        }
    }
    if (!o.single) {
        const auto latency = summarize(times);
        result.us = latency.p50; result.p95 = latency.p95;
        if (!tails.empty()) result.tail = summarize(tails).p50;
        if (!locks.empty()) result.lock = summarize(locks).p50;
    }
    result.timed = valid_events;
#if defined(ASCEND_MICRO_BENCH_SIM)
    result.timed = false;
#endif
    return result;
}

std::string field(double value, bool available, int precision = 3)
{
    if (!available) return "NA";
    std::ostringstream s;
    s << std::fixed << std::setprecision(precision) << value;
    return s.str();
}

void row(int n, int k, int grid, const Measurement& aiv, const Measurement& aic,
         double baseline, double aiv_bytes, double aic_bytes, std::ofstream& csv)
{
    const auto av = field(aiv.us, aiv.timed);
    const auto speed = field(aiv.timed ? baseline / aiv.us : 0, aiv.timed && baseline > 0);
    const auto abw = field(aiv.timed ? aiv_bytes / aiv.us / 1000 : 0, aiv.timed);
    const auto cr = field(aic.us, aic.timed);
    const auto cn = field(aic.us / 16.0, aic.timed);
    const auto cbw = field(aic.timed ? aic_bytes * 16 / aic.us / 1000 : 0, aic.timed);
    const auto tail = field(aiv.tail, aiv.present, 0), lock = field(aiv.lock, aiv.present, 0);
    const auto spins = aiv.present ? std::to_string(aiv.failed) : "NA";
    std::printf("| %11d | %10d | %6d | %12s | %11s | %11s | %17s | %22s | %14s | %12s | %16s | %12s |\n",
                n, k, grid, av.c_str(), speed.c_str(), abw.c_str(), cr.c_str(), cn.c_str(),
                cbw.c_str(), tail.c_str(), lock.c_str(), spins.c_str());
    if (csv.is_open()) {
        csv << n << ',' << k << ',' << grid << ',' << av << ',' << speed << ',' << abw << ','
            << cr << ',' << cn << ',' << cbw << ','
            << field(aic.timed ? aic_bytes / aic.us / 1000 : 0, aic.timed) << ','
            << tail << ',' << lock << ',' << spins << ',' << aiv.timeouts << ','
            << aiv.launches << ',' << aic.launches << '\n';
        csv.flush();
        require(static_cast<bool>(csv), "CSV write failed");
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
        const Options o = parse(argc, argv);
#if defined(ASCEND_MICRO_BENCH_SIM)
        std::puts("CAModel functional run: event times/speedups are unavailable; completion deltas are model ticks.");
#else
        std::puts("Ascend950PR decode benchmark: fused AIV Split-K and AIC M=16 partial-state model.");
#endif
        std::ofstream csv;
        if (!o.csv_out.empty()) {
            csv.open(o.csv_out, std::ios::out | std::ios::trunc);
            require(csv.is_open(), "cannot open --csv-out path");
            csv << "context,splits,grid,aiv_us,aiv_speedup,aiv_kv_gbs,aic_batch_us,aic_model_us_per_req,aic_model_gbs,aic_physical_kv_gbs,tail_ticks,lock_ticks,failed_spins,timeouts,aiv_launches,aic_launches\n";
        }
        if (o.single) std::puts("[PROFILING MODE] Single launch mode active for msprof op.");
        std::puts("AIC model: 16 valid shared-KV query rows per head; raw batch time / 16. Model GB/s includes 16x reuse.");
        std::puts("AIC exports partials; its timing excludes external combine/egress. Crossover is a modeled lower bound.");
        if (o.single) std::puts("Single samples replace p50; tail/lock show maxima. No implicit K=1 launch.");
        std::printf("impl=%s requests=%d heads=%d AIV_rows=%d AIC_rows=16 page_tokens=%d warmup=%d iters=%d\n",
                    o.impl.c_str(), o.requests, o.heads, o.rows, o.page_tokens, o.warmup, o.iters);
        std::puts("+-------------+------------+--------+--------------+-------------+-------------+-------------------+------------------------+----------------+--------------+------------------+--------------+");
        std::printf("| %11s | %10s | %6s | %12s | %11s | %11s | %17s | %22s | %14s | %12s | %16s | %12s |\n",
                    "Context (N)", "Splits (K)", "Grid", o.single ? "AIV (us)" : "AIV p50(us)",
                    "AIV Speedup", "AIV KV GB/s", "AIC Batch Raw(us)",
                    o.single ? "AIC Model(us/req)" : "AIC Model p50(us/req)", "AIC Model GB/s",
                    "Tail (ticks)", o.single ? "Lock (ticks)" : "Lock p50(ticks)", "Failed Spins");
        std::puts("+-------------+------------+--------+--------------+-------------+-------------+-------------------+------------------------+----------------+--------------+------------------+--------------+");
        tq_npu::Session session;
        Events events;
        struct Comparison { int n, k; double aiv, aic; };
        std::vector<Comparison> comparisons;
        std::uint64_t aiv_launches = 0, aic_launches = 0;
        for (int n : o.contexts) {
            std::unique_ptr<Fixture> av, ac;
            if (o.impl != "aic") av.reset(new Fixture(n, o.requests, o.heads, o.rows, o.page_tokens));
            if (o.impl != "aiv") ac.reset(new Fixture(n, o.requests, o.heads, 16, o.page_tokens));
            std::vector<int> sweep;
            if (!o.single && av) sweep.push_back(1);
            for (int k : o.splits)
                if (std::find(sweep.begin(), sweep.end(), k) == sweep.end()) sweep.push_back(k);
            double baseline = 0;
            for (int k : sweep) {
                Measurement aiv, aic;
                double aiv_bytes = 0, aic_bytes = 0;
                if (av) {
                    Device d(*av, k);
                    aiv = measure(d, o, k, session.stream, events);
                    aiv_launches += aiv.launches;
                    if (k == 1 && aiv.timed) baseline = aiv.us;
                    aiv_bytes = static_cast<double>(av->key.size() + av->value.size() + av->scales.size() * 4);
                }
                if (ac) {
                    const int pages = o.page_count < 0 ? ac->pages - o.page_begin : o.page_count;
                    AicDevice d(*ac, k, o.page_begin, pages);
                    aic = measure(d, o, k, session.stream, events);
                    aic_launches += aic.launches;
                    aic_bytes = static_cast<double>(ac->key.size() + ac->value.size() + ac->scales.size() * 4) * pages / ac->pages;
                }
                row(n, k, o.requests * o.heads * k, aiv, aic, baseline, aiv_bytes, aic_bytes, csv);
                if (aiv.timed && aic.timed)
                    comparisons.push_back({n, k, aiv.us, aic.us / 16});
            }
        }
        std::puts("+-------------+------------+--------+--------------+-------------+-------------+-------------------+------------------------+----------------+--------------+------------------+--------------+");
        for (const auto& c : comparisons)
            std::printf("Modeled comparison N=%d K=%d: AIC %.3f us/row vs AIV %.3f us/decode (%s).\n",
                        c.n, c.k, c.aic, c.aiv, c.aic < c.aiv ? "AIC amortized model lower" : "AIV lower");
        std::map<int, int> crossover;
        for (const auto& c : comparisons) {
            if (!crossover.count(c.k)) crossover[c.k] = 0;
            if (c.aic < c.aiv && (crossover[c.k] == 0 || c.n < crossover[c.k])) crossover[c.k] = c.n;
        }
        for (const auto& entry : crossover) {
            if (entry.second)
                std::printf("Modeled crossover K=%d: first tested N=%d (partial-only amortized lower bound).\n",
                            entry.first, entry.second);
            else std::printf("Modeled crossover K=%d: not observed in the tested contexts.\n", entry.first);
        }
        if (o.impl == "both" && comparisons.empty())
            std::puts("Modeled crossover unavailable: device event timing is unavailable.");
        std::printf("PASS: CPU references matched. Kernel launches: AIV=%llu AIC=%llu.\n",
                    static_cast<unsigned long long>(aiv_launches), static_cast<unsigned long long>(aic_launches));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
