// Single-launch, real page-partitioned Split-K decode benchmark.
// No synthetic partial states or cyclic page waves. Each sample resets its
// head slots outside the ACL event interval, then times exactly one launch.
//
// KERNEL ISOLATION. By default this executable benchmarks ONE kernel: the fused decode
// operator, turboquant_mm_fused_decode_opt_kv4fp8_half, with the Cube GEMM reduction that
// TQ_GEMM_REDUCE=ON selects for its reduction phase. The legacy decoders -- the AIV-only
// split-K decoder and the AIC M=16 lock-step partial-state model -- are compiled OUT, so
// nothing else launches, nothing else is timed and the two legacy kernel libraries are not
// even linked (cmake/NpuTargets.cmake drops them from the link in this configuration). That is
// what makes an msprof capture of this binary a capture of the operator and not of a sweep
// that happens to contain it.
//
// They are not deleted. Configure with -DTQ_BENCH_LEGACY_DECODERS=ON and the comparison
// columns, the modeled AIC crossover and --impl aiv|aic|both|all all come back exactly as they
// were; the CSV schema is the same either way, with NA in the columns a given build cannot
// fill. Everything under TQ_BENCH_FUSED_OPT_ONLY below is that switch and nothing else.
#if defined(TQ_BENCH_LEGACY_DECODERS) && TQ_BENCH_LEGACY_DECODERS
#define TQ_BENCH_FUSED_OPT_ONLY 0
#else
#define TQ_BENCH_FUSED_OPT_ONLY 1
#endif

#include "../turboquant_npu_common.hpp"
#include "../../common/aic_splitk_decode_fixture.hpp"
#include "../../../include/turboquant_splitk_contract.hpp"
#include "../turboquant_splitk_rigs.hpp"
#include "../turboquant_fused_decode_opt_rig.hpp"

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
#if TQ_BENCH_FUSED_OPT_ONLY
    // The operator's own sweep: contexts 1024..16384, S in {1,2,4,8} (K>8 is outside
    // valid_launch), and 16 query heads over one KV head at D=256 -- the GQA shape the
    // validation ladder and the CAModel evidence are both about. 1 is kept in the split sweep
    // because it is the fused path, which is the baseline every speedup is against and the one
    // geometry that reaches no reduction at all.
    std::vector<int> splits{1, 2, 4, 8};
    int warmup = 5, iters = 30, requests = 1;
    int heads = 1, rows = 16, page_tokens = 128;
#else
    std::vector<int> splits{1, 2, 4, 8, 12, 16};
    int warmup = 5, iters = 30, requests = 1;
    int heads = 4, rows = 6, page_tokens = 128;
#endif
    bool single = false;
#if TQ_BENCH_FUSED_OPT_ONLY
    std::string impl = "fused-opt", csv_out;
#else
    std::string impl = "aiv", csv_out;
#endif
    int page_begin = 0, page_count = -1;
    bool zone_report = false;
    bool Aiv() const { return impl == "aiv" || impl == "both" || impl == "all"; }
    bool Aic() const { return impl == "aic" || impl == "both" || impl == "all"; }
    // `three-zone` is an alias, not a second kernel: the three-zone schedule
    // IS the production fused-opt kernel after the schedule refactor, so both
    // spellings dispatch the same symbol. The alias exists so a sweep can ask
    // for the zone report without also opting into the legacy baselines.
    bool Opt() const { return impl == "fused-opt" || impl == "three-zone" || impl == "all"; }
    bool OptOnly() const { return impl == "fused-opt" || impl == "three-zone"; }
};

// ---------------------------------------------------------------------------
// Zone decomposition of the fused-opt schedule.
//
// This mirrors the kernel's own selection in
// kernels/ops/turboquant/turboquant_fused_decode_opt.cpp: kSlots = 3 L1 ring
// slots, kPhases = 6 unrolled steady phases, steady_limit = numTiles - kSlots,
// the deep path taken at numTiles >= kSlots + 1 and a bounded three-step drain.
// It is a host-side restatement for reporting, so a bench row can say whether
// it actually entered the steady loop. It is not evidence that the kernel took
// that path -- the CAModel trace audit is.
// ---------------------------------------------------------------------------
constexpr int kZoneSlots = 3, kZonePhases = 6, kZoneTileRows = 64;

struct ZoneShape {
    int tiles = 0;      // per-worker numTiles
    bool deep = false;  // the unconditional three-zone path, not the short one
    int periods = 0;    // complete unrolled Period-6 bodies
    int residue = 0;    // ragged steady phases following them
    int drain = 0;      // bounded drain steps
    bool Empty() const { return tiles == 0; }
    // Zone class, for coverage accounting.
    std::string Kind() const
    {
        if (Empty()) return "empty";
        if (!deep) return "short";
        if (periods == 0) return "deep-noperiod";
        return residue ? "deep-ragged" : "deep-exact";
    }
};

ZoneShape zone_shape(int tiles)
{
    ZoneShape z;
    z.tiles = tiles;
    if (tiles <= 0) return z;
    if (tiles < kZoneSlots + 1) return z;  // bounded short/compatibility path
    z.deep = true;
    const int steady = tiles - kZoneSlots;
    z.periods = steady / kZonePhases;
    z.residue = steady % kZonePhases;
    z.drain = kZoneSlots;
    return z;
}

// Per-worker tile counts. Mirrors the kernel's block slicing: a worker owns
// [split * blocksPerSplit, min(start + blocksPerSplit, seqBlocks)). Valid for
// the fused-opt fixture specifically, whose block table maps EVERY logical
// page; CountTiles skips negative entries, so a fixture with holes would need
// the table itself rather than this arithmetic.
//
// RAGGED CONTEXTS: seqBlocks is the CEILING, the last page holds
// context - (seqBlocks - 1) * page_tokens live rows, and a page contributes
// ceil(live / 64) tiles -- never a tile of zero, which is what keeps an
// all-masked strip out of the masking phase.
std::vector<ZoneShape> worker_zones(int context, int page_tokens, int k)
{
    const int seq_blocks = (context + page_tokens - 1) / page_tokens;
    const int workers = k <= 1 ? 1 : k;
    const int per_split = (seq_blocks + workers - 1) / workers;
    std::vector<ZoneShape> out;
    for (int split = 0; split < workers; ++split) {
        const int start = split * per_split;
        const int end = std::min(start + per_split, seq_blocks);
        int tiles = 0;
        for (int block = start; block < end; ++block) {
            const int live = std::min(page_tokens, context - block * page_tokens);
            if (live > 0) tiles += (live + kZoneTileRows - 1) / kZoneTileRows;
        }
        out.push_back(zone_shape(tiles));
    }
    return out;
}

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
    bool explicit_splits = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help") {
#if TQ_BENCH_FUSED_OPT_ONLY
            std::puts("Fused decode operator only. Configure -DTQ_BENCH_LEGACY_DECODERS=ON for\n"
                      "the AIV/AIC comparison columns and --impl aiv|aic|both|all.\n"
                      "Defaults: contexts 1024,2048,4096,8192,16384; K 1,2,4,8;\n"
                      "          --heads 1 --rows 16 (16 query heads over one KV head), D=256,\n"
                      "          --page-tokens 128.");
#endif
            std::puts("--context N[,N...] --splitk K[,K...] --warmup N --iters N "
                      "--requests N --heads N --rows N --page-tokens N\n"
                      "--impl aiv|aic|both|fused-opt|three-zone|all --single --csv-out PATH --page-begin P --page-count P\n"
                      "--zone-report: per-worker three-zone decomposition and zone-class coverage.\n"
                      "three-zone is an alias for fused-opt: one kernel, the refactored schedule.\n"
                      "Defaults: contexts 1024,2048,4096,8192,16384; K 1,2,4,8,12,16.\n"
                      "fused-opt: separate KV4/FP8 fixture; --rows is GQA query heads per KV head; K 1..8.\n"
                      "Its default K sweep is 1,2,4,8. --impl all leaves opt columns NA at K>8.\n"
                      "fused-opt accepts --rows 1..16; legacy/all accepts --rows 1..8.\n"
                      "KNOWN FAILING: fused-opt --rows 1 fails its own CPU attention oracle\n"
                      "(measured on CAModel 2026-11-04, context 128/192, page-tokens 64 and 128,\n"
                      "and with --impl fused-opt as well, so it predates the three-zone work).\n"
                      "--rows 16 passes. The range is left as-is so the failure stays reproducible.\n"
                      "fused-opt page-tokens: 64 or 128; other implementations: 16 or 128.\n"
                      "fused-opt also takes a RAGGED --context (not a multiple of page-tokens\n"
                      "or of the 64-row tile), e.g. --context 79 or 1055; the others do not.\n"
                      "Use --page-tokens 16 --heads 1 --context 48 for a short CAModel run.");
            std::exit(0);
        }
        if (arg == "--single") { o.single = true; continue; }
        if (arg == "--zone-report") { o.zone_report = true; continue; }
        require(i + 1 < argc, "option requires a value");
        const std::string_view value(argv[++i]);
        if (arg == "--context") o.contexts = list(value);
        else if (arg == "--splitk") { o.splits = list(value); explicit_splits = true; }
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
#if TQ_BENCH_FUSED_OPT_ONLY
    require(o.OptOnly(), "this build benchmarks the fused decode operator alone: --impl must be "
                         "fused-opt or three-zone. Configure -DTQ_BENCH_LEGACY_DECODERS=ON to "
                         "restore aiv, aic, both and all");
#else
    require(o.Aiv() || o.Aic() || o.Opt(),
            "impl must be aiv, aic, both, fused-opt, three-zone or all");
#endif
    require(!o.zone_report || o.Opt(), "--zone-report needs fused-opt, three-zone or all");
    if (o.OptOnly() && !explicit_splits) o.splits = {1, 2, 4, 8};
    require(o.page_begin >= 0 && o.page_count >= -1, "invalid AIC page window");
    require(o.Opt() ? (o.page_tokens == 64 || o.page_tokens == 128) :
            (o.page_tokens == 16 || o.page_tokens == 128), "unsupported page-tokens for implementation");
    if (o.Aiv() || o.Aic()) require(o.page_tokens != 64,
            "64-token pages require --impl fused-opt or three-zone");
    require(o.heads > 0 && o.heads <= 4 && o.rows > 0 &&
            o.rows <= (o.OptOnly() ? 16 : 8),
            "heads must be 1..4; rows 1..16 for fused-opt, otherwise 1..8");
    for (int n : o.contexts) {
        // The fused operator takes a RAGGED context: its fixture's page count is the ceiling,
        // the last page holds n - (pages - 1) * page_tokens live tokens, and the kernel masks
        // the tail of the last 64-row tile. The legacy decoders' fixture requires an aligned
        // context, so an unaligned one stays an error for them rather than being rounded.
        require(n > 0 && n <= 16384, "context must be positive and at most 16384");
        require(o.OptOnly() || n % o.page_tokens == 0,
                "an unaligned context needs --impl fused-opt or three-zone");
        const int pages = (n + o.page_tokens - 1) / o.page_tokens;
        require(o.page_begin <= pages && (o.page_count < 0 || o.page_count <= pages - o.page_begin),
                "AIC page window exceeds context");
        if (o.impl != "aic") require(o.page_begin == 0 && (o.page_count == -1 || o.page_count == pages),
                "partial page windows require --impl aic; both compares full contexts");
    }
    for (int k : o.splits) require(k >= 1 && k <= (o.OptOnly() ? 8 : 16),
                                 "splitk must be 1..8 for fused-opt, otherwise 1..16");
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
         const Measurement& opt, int opt_grid, double baseline, double opt_baseline,
         double aiv_bytes, double aic_bytes, double opt_bytes, std::ofstream& csv)
{
    const auto av = field(aiv.us, aiv.timed);
    const auto speed = field(aiv.timed ? baseline / aiv.us : 0, aiv.timed && baseline > 0);
    const auto abw = field(aiv.timed ? aiv_bytes / aiv.us / 1000 : 0, aiv.timed);
    const auto cr = field(aic.us, aic.timed);
    const auto cn = field(aic.us / 16.0, aic.timed);
    const auto cbw = field(aic.timed ? aic_bytes * 16 / aic.us / 1000 : 0, aic.timed);
    const auto tail = field(aiv.tail, aiv.present, 0);
    const auto exported = field(aiv.export_region, aiv.present, 0);
    const auto opt_time = field(opt.us, opt.timed);
    const auto opt_speed = field(opt.timed && opt.us > 0 ? opt_baseline / opt.us : 0,
                                 opt.timed && opt_baseline > 0);
    const auto opt_bw = field(opt.timed ? opt_bytes / opt.us / 1000 : 0, opt.timed);
    const auto opt_blocks = opt.present ? std::to_string(opt_grid) : "NA";
    std::printf("| %11d | %10d | %6d | %12s | %11s | %11s | %17s | %22s | %14s | %12s | %16s | %8s | %13s | %11s | %11s |\n",
                n, k, grid, av.c_str(), speed.c_str(), abw.c_str(), cr.c_str(), cn.c_str(),
                cbw.c_str(), tail.c_str(), exported.c_str(), opt_blocks.c_str(),
                opt_time.c_str(), opt_speed.c_str(), opt_bw.c_str());
    if (csv.is_open()) {
        csv << n << ',' << k << ',' << grid << ',' << av << ',' << speed << ',' << abw << ','
            << cr << ',' << cn << ',' << cbw << ','
            << field(aic.timed ? aic_bytes / aic.us / 1000 : 0, aic.timed) << ','
            << tail << ',' << exported << ','
            << aiv.launches << ',' << aic.launches << ',' << opt_blocks << ','
            << opt_time << ',' << opt_speed << ',' << opt_bw << ',' << opt.launches << '\n';
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
            csv << "context,splits,grid,aiv_us,aiv_speedup,aiv_kv_gbs,aic_batch_us,aic_model_us_per_req,aic_model_gbs,aic_physical_kv_gbs,tail_ticks,export_ticks,aiv_launches,aic_launches,opt_grid,opt_us,opt_speedup,opt_kv_cache_gbs,opt_launches\n";
        }
        if (o.single) std::puts("[PROFILING MODE] Single launch mode active for msprof op.");
#if TQ_BENCH_FUSED_OPT_ONLY
        std::puts("ISOLATED: the fused decode operator is the only kernel in this binary. The AIV and AIC");
        std::puts("columns are NA by construction, not by a missing device; so is the modeled crossover.");
        std::puts("Configure -DTQ_BENCH_LEGACY_DECODERS=ON to benchmark them alongside it again.");
#else
        std::puts("AIC model: 16 valid shared-KV query rows per head; raw batch time / 16. Model GB/s includes 16x reuse.");
        std::puts("AIC exports partials; its timing excludes external combine/egress. Crossover is a modeled lower bound.");
#endif
        if (o.Opt()) {
            std::puts("OPT: KV4/FP8, FP32 pre-rotated GQA queries, FP16 rotated-basis output; complete in-launch Split-K combine.");
            std::printf("OPT GQA query heads per KV head=%d; OPT Grid is the actual dispatch (capped at 32 blocks).\n", o.rows);
            std::puts("OPT speedup uses its own K=1 baseline; cache GB/s excludes fragmented canary pages. Worker timestamps: unavailable.");
            if (o.impl == "all") std::puts("OPT columns are NA for unsupported K>8. Fixtures/precision differ; no cross-implementation speedup is inferred.");
        }
        if (o.single) std::puts("Single samples replace p50; tail/export show maxima. No implicit K=1 launch.");
        std::printf("impl=%s requests=%d heads=%d AIV_rows=%d AIC_rows=16 page_tokens=%d warmup=%d iters=%d\n",
                    o.impl.c_str(), o.requests, o.heads, o.rows, o.page_tokens, o.warmup, o.iters);
        std::puts("+-------------+------------+--------+--------------+-------------+-------------+-------------------+------------------------+----------------+--------------+------------------+----------+---------------+-------------+-------------+");
        std::printf("| %11s | %10s | %6s | %12s | %11s | %11s | %17s | %22s | %14s | %12s | %16s | %8s | %13s | %11s | %11s |\n",
                    "Context (N)", "Splits (K)", "Grid", o.single ? "AIV (us)" : "AIV p50(us)",
                    "AIV Speedup", "AIV KV GB/s", "AIC Batch Raw(us)",
                    o.single ? "AIC Model(us/req)" : "AIC Model p50(us/req)", "AIC Model GB/s",
                    "Tail (ticks)", o.single ? "Export (ticks)" : "Export p50(ticks)",
                    "OPT Grid", o.single ? "OPT Fused(us)" : "OPT p50(us)", "OPT Speedup", "OPT KV GB/s");
        std::puts("+-------------+------------+--------+--------------+-------------+-------------+-------------------+------------------------+----------------+--------------+------------------+----------+---------------+-------------+-------------+");
        tq_npu::Session session;
        Events events;
        struct Comparison { int n, k; double aiv, aic; };
        std::vector<Comparison> comparisons;
        struct ZoneRow { int n, k; std::vector<ZoneShape> workers; };
        std::vector<ZoneRow> zone_rows;
        std::map<std::string, int> zone_classes;
        std::uint64_t aiv_launches = 0, aic_launches = 0, opt_launches = 0;
        for (int n : o.contexts) {
#if !TQ_BENCH_FUSED_OPT_ONLY
            std::unique_ptr<Fixture> av, ac;
#endif
            std::unique_ptr<fused_decode_opt_test::Fixture> opt_fixture;
#if !TQ_BENCH_FUSED_OPT_ONLY
            if (o.Aiv()) av.reset(new Fixture(n, o.requests, o.heads, o.rows, o.page_tokens));
            if (o.Aic()) ac.reset(new Fixture(n, o.requests, o.heads, 16, o.page_tokens));
#endif
            if (o.Opt()) opt_fixture.reset(new fused_decode_opt_test::Fixture(n, o.requests, o.heads, o.rows, o.page_tokens));
            std::vector<int> sweep;
#if TQ_BENCH_FUSED_OPT_ONLY
            if (!o.single && opt_fixture) sweep.push_back(1);
#else
            if (!o.single && (av || opt_fixture)) sweep.push_back(1);
#endif
            for (int k : o.splits)
                if (std::find(sweep.begin(), sweep.end(), k) == sweep.end()) sweep.push_back(k);
            double baseline = 0, opt_baseline = 0;
            for (int k : sweep) {
                Measurement aiv, aic, opt;
                double aiv_bytes = 0, aic_bytes = 0, opt_bytes = 0;
                int opt_grid = 0;
#if !TQ_BENCH_FUSED_OPT_ONLY
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
#endif
                if (opt_fixture && k <= 8) {
                    tq_splitk_rig::FusedOptRig d(*opt_fixture, k);
                    opt = measure(d, o, k, session.stream, events);
                    opt_launches += opt.launches; opt_grid = d.config.grid;
                    if (k == 1 && opt.timed) opt_baseline = opt.us;
                    opt_bytes = opt_fixture->cache_bytes();
                    // The rig's Validate() already compared every launch of
                    // this configuration against the fixture's independent
                    // FP64 attention oracle, and K=1 and K>1 are held to the
                    // SAME oracle, so a Split-K merge that lost or double
                    // counted a worker's partial cannot pass. Record the zone
                    // shape the schedule must have taken for that result.
                    if (o.zone_report) {
                        const auto shapes = worker_zones(n, o.page_tokens, k);
                        for (const auto& z : shapes) ++zone_classes[z.Kind()];
                        zone_rows.push_back({n, k, shapes});
                    }
                }
                row(n, k, o.requests * o.heads * k, aiv, aic, opt, opt_grid,
                    baseline, opt_baseline, aiv_bytes, aic_bytes, opt_bytes, csv);
                if (aiv.timed && aic.timed)
                    comparisons.push_back({n, k, aiv.us, aic.us / 16});
            }
        }
        std::puts("+-------------+------------+--------+--------------+-------------+-------------+-------------------+------------------------+----------------+--------------+------------------+----------+---------------+-------------+-------------+");
        if (o.zone_report) {
            std::puts("");
            std::puts("Three-zone decomposition per Split-K worker. Host restatement of the kernel's own");
            std::puts("selection (kSlots=3, kPhases=6, steady_limit=numTiles-3, deep at numTiles>=4);");
            std::puts("it says which zones a row MUST have used, not that the kernel took them -- only");
            std::puts("the CAModel trace audit shows that. Tiles are 64 tokens (kCubeTileRows).");
            std::puts("+-------------+------------+--------+-------+-------+---------+---------+-------+----------------+");
            std::printf("| %11s | %10s | %6s | %5s | %5s | %7s | %7s | %5s | %14s |\n",
                        "Context (N)", "Splits (K)", "Worker", "Tiles", "Deep", "Periods",
                        "Residue", "Drain", "Zone class");
            std::puts("+-------------+------------+--------+-------+-------+---------+---------+-------+----------------+");
            for (const auto& r : zone_rows)
                for (std::size_t w = 0; w < r.workers.size(); ++w) {
                    const auto& z = r.workers[w];
                    std::printf("| %11d | %10d | %6zu | %5d | %5s | %7d | %7d | %5d | %14s |\n",
                                r.n, r.k, w, z.tiles, z.Empty() ? "NA" : (z.deep ? "yes" : "no"),
                                z.periods, z.residue, z.drain, z.Kind().c_str());
                }
            std::puts("+-------------+------------+--------+-------+-------+---------+---------+-------+----------------+");
            std::puts("Zone-class coverage across this sweep:");
            for (const auto& entry : zone_classes)
                std::printf("  %-14s workers=%d\n", entry.first.c_str(), entry.second);
            // Coverage is reported, and the one class this benchmark exists to
            // exercise is required. A sweep that never left the short bounded
            // path would otherwise report PASS while testing none of the
            // steady loop this refactor introduced.
            const std::size_t deep_classes =
                zone_classes.count("deep-exact") + zone_classes.count("deep-ragged") +
                zone_classes.count("deep-noperiod");
            require(deep_classes > 0, "--zone-report swept no deep three-zone worker: raise --context "
                              "(a worker needs >=4 tiles, i.e. >=256 tokens of its own slice)");
            for (const char* kind : {"short", "deep-exact", "deep-ragged"})
                if (!zone_classes.count(kind))
                    std::printf("NOTE: zone class %s not covered by this sweep.\n", kind);
        }
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
        if (o.Aiv() && o.Aic() && comparisons.empty())
            std::puts("Modeled crossover unavailable: device event timing is unavailable.");
        std::printf("PASS: CPU references matched. Kernel launches: AIV=%llu AIC=%llu OPT=%llu.\n",
                    static_cast<unsigned long long>(aiv_launches), static_cast<unsigned long long>(aic_launches),
                    static_cast<unsigned long long>(opt_launches));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
