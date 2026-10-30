// Single-launch, real page-partitioned Split-K decode benchmark.
// No synthetic partial states or cyclic page waves. Each sample resets its
// head slots outside the ACL event interval, then times exactly one launch.
#include "../turboquant_npu_common.hpp"
#include "../../common/aic_splitk_decode_fixture.hpp"
#include "../../../include/turboquant_splitk_contract.hpp"
#include "../turboquant_splitk_rigs.hpp"

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


namespace {
using tq_npu::Buffer;
using tq_npu::check;
using tq_npu::require;
namespace contract = turboquant_splitk;
using splitk_decode_test::Fixture;
using splitk_decode_test::kChannels;

// The rigs themselves now live in turboquant_splitk_rigs.hpp, shared with the
// NPU and simulator validation tests. These aliases keep the benchmark's own
// names.
using Device = tq_splitk_rig::AivRig;
using AicDevice = tq_splitk_rig::AicRig;

struct Options {
    std::vector<int> contexts{1024, 2048, 4096, 8192, 16384};
    std::vector<int> splits{1, 2, 4, 8, 12, 16};
    int warmup = 5, iters = 30, requests = 1;
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
                      "--requests N --heads N --rows N --page-tokens N\n"
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
    require(o.iters > 0 && o.warmup >= 0 && o.requests > 0 && o.requests <= 16,
            "invalid iteration or batch bound");
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



struct Measurement {
    bool present = false, timed = false;
    // `export_region` is the former `lock`: with the critical section gone,
    // what the census times is the non-blocking partial write that replaced
    // it. The spin and timeout counters went with the lock -- there is
    // nothing left to fail to acquire and no watchdog to expire.
    double us = 0, p95 = 0, tail = 0, export_region = 0;
    std::uint64_t launches = 0;
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
        d.Launch(stream, k);
        check(aclrtSynchronizeStream(stream), "warmup post-launch synchronize");
        ++result.launches;
        d.Validate(k);
    }
    std::vector<double> times, tails, exports;
    bool valid_events = true;
    for (int i = 0; i < o.iters; ++i) {
        d.Reset();
        check(aclrtSynchronizeStream(stream), "pre-launch synchronize");
#if !defined(ASCEND_MICRO_BENCH_SIM)
        check(aclrtRecordEvent(events.start, stream), "record start");
#endif
        d.Launch(stream, k);
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
        if (o.single) {
            // No percentile sorting or multi-sample accumulation in profiling mode.
            result.us = ms * 1000.0;
            result.tail = census.tail_ticks.empty() ? 0 : *std::max_element(census.tail_ticks.begin(), census.tail_ticks.end());
            result.export_region = census.export_ticks.empty() ? 0 : *std::max_element(census.export_ticks.begin(), census.export_ticks.end());
        } else {
            times.push_back(ms * 1000.0);
            tails.insert(tails.end(), census.tail_ticks.begin(), census.tail_ticks.end());
            exports.insert(exports.end(), census.export_ticks.begin(), census.export_ticks.end());
        }
    }
    if (!o.single) {
        const auto latency = summarize(times);
        result.us = latency.p50; result.p95 = latency.p95;
        if (!tails.empty()) result.tail = summarize(tails).p50;
        if (!exports.empty()) result.export_region = summarize(exports).p50;
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
    const auto tail = field(aiv.tail, aiv.present, 0);
    const auto exported = field(aiv.export_region, aiv.present, 0);
    std::printf("| %11d | %10d | %6d | %12s | %11s | %11s | %17s | %22s | %14s | %12s | %16s |\n",
                n, k, grid, av.c_str(), speed.c_str(), abw.c_str(), cr.c_str(), cn.c_str(),
                cbw.c_str(), tail.c_str(), exported.c_str());
    if (csv.is_open()) {
        csv << n << ',' << k << ',' << grid << ',' << av << ',' << speed << ',' << abw << ','
            << cr << ',' << cn << ',' << cbw << ','
            << field(aic.timed ? aic_bytes / aic.us / 1000 : 0, aic.timed) << ','
            << tail << ',' << exported << ','
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
            csv << "context,splits,grid,aiv_us,aiv_speedup,aiv_kv_gbs,aic_batch_us,aic_model_us_per_req,aic_model_gbs,aic_physical_kv_gbs,tail_ticks,export_ticks,aiv_launches,aic_launches\n";
        }
        if (o.single) std::puts("[PROFILING MODE] Single launch mode active for msprof op.");
        std::puts("AIC model: 16 valid shared-KV query rows per head; raw batch time / 16. Model GB/s includes 16x reuse.");
        std::puts("AIC exports partials; its timing excludes external combine/egress. Crossover is a modeled lower bound.");
        if (o.single) std::puts("Single samples replace p50; tail/export show maxima. No implicit K=1 launch.");
        std::printf("impl=%s requests=%d heads=%d AIV_rows=%d AIC_rows=16 page_tokens=%d warmup=%d iters=%d\n",
                    o.impl.c_str(), o.requests, o.heads, o.rows, o.page_tokens, o.warmup, o.iters);
        std::puts("+-------------+------------+--------+--------------+-------------+-------------+-------------------+------------------------+----------------+--------------+------------------+");
        std::printf("| %11s | %10s | %6s | %12s | %11s | %11s | %17s | %22s | %14s | %12s | %16s |\n",
                    "Context (N)", "Splits (K)", "Grid", o.single ? "AIV (us)" : "AIV p50(us)",
                    "AIV Speedup", "AIV KV GB/s", "AIC Batch Raw(us)",
                    o.single ? "AIC Model(us/req)" : "AIC Model p50(us/req)", "AIC Model GB/s",
                    "Tail (ticks)", o.single ? "Export (ticks)" : "Export p50(ticks)");
        std::puts("+-------------+------------+--------+--------------+-------------+-------------+-------------------+------------------------+----------------+--------------+------------------+");
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
        std::puts("+-------------+------------+--------+--------------+-------------+-------------+-------------------+------------------------+----------------+--------------+------------------+");
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
