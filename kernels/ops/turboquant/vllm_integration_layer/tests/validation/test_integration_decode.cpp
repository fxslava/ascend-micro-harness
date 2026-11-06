// Physical-device validation ladder for the fused TurboQuant decode operator,
// `turboquant_mm_fused_decode_opt_kv4fp8_half`, built with the Cube GEMM reduction that
// -DTQ_GEMM_REDUCE=ON (the default since this change) selects for its reduction phase.
//
// HARDWARE ONLY, and the development host for this harness has no NPU: `npu-smi` and
// `/dev/davinci*` are both absent, so what CI can establish for this file is a clean strict
// build and nothing about the kernel. `add_ascend_micro_sim_targets` is where a probe that can
// run without a device lives; reports/gemm_batched_reduction_camodel_20261104.md is the
// CAModel evidence this ladder is the hardware counterpart of.
//
// WHAT EACH LAUNCH CHECKS
//
// 1. ORACLE. Every emitted FP16 channel of every (token, query head) against
//    fused_decode_opt_test::Fixture::expected -- the fixture's own scalar FP64 attention sum
//    over the whole context, computed from the quantisation levels rather than from the
//    kernel, so it is an independent oracle and not a transcription. The budget is ZERO fp16
//    ULP: every channel must equal the correctly-rounded fp16 of the double-precision combine
//    exactly. Section 8 of the CAModel report measured 0 at S = 1, 4, 5 and 8 on the
//    simulator, which is why the budget is 0 here rather than 1. The residual absolute error
//    against the FP64 value is reported, not asserted -- it is the fp16 quantisation of the
//    oracle itself.
//
// 2. CANARIES, 0x5A5A5A5A, in four places, all required INTACT afterwards:
//    * the 64-byte guard bands on both sides of every one of the nine device allocations
//      (adjacent memory: an MTE burst that overruns a buffer lands here);
//    * every PHYSICAL KV page the block table does not map. The fixture interleaves mapped and
//      unmapped pages -- logical page p of token t lives at physical 2*(t*pages+p)+1 -- so an
//      off-by-one in the paged addressing lands on a canary page rather than on another token's
//      data, and a skipped block-table slot is visible as a page that was read;
//    * the INTER-HEAD PADDING of every mapped page's scale slot. Each row carries eight scale
//      lanes and the kv4fp8 codec reads lanes [head] and [numKvHeads + head]; at one KV head
//      that leaves six lanes per row that nothing may read and nothing may write;
//    * the tail of the output allocation past the last live channel.
//
// 3. WORKSPACE. At S > 1 every float of the live partial extent must come back finite -- a
//    partial the reduction never wrote stays 0x5A5A5A5A, which is ~3.8e15 and not finite once
//    it is combined. At S == 1 the operator takes its fused path, writes the context straight
//    to the output and must not touch the workspace at all, so the WHOLE workspace is required
//    to still be canary. That is the control for a reduction phase that runs when it should
//    not: NeedsReduction() is false at one split.
//
// 4. INPUT IMMUTABILITY. Every input allocation is byte-compared against what was uploaded.
//
// THE LADDER
//
// Nine rungs, each a (context, tokens, KV heads, query heads per KV head) geometry with its own
// split sweep; one launch per (rung, S). D = 256 and 128-token physical pages throughout, so a
// page is two 64-row Cube tiles and a worker's tile count -- which is what selects the
// schedule's path -- is 2 * (pages it owns), less any ragged tail.
//
// ALIGNED, one KV head, 16 query heads:
//
//   * 128 (one page): at S > 1 only the first worker has a block range at all. The others are
//     UNMAPPED SPLITS, blockStart >= blockEnd, which is the early exit that now leaves through
//     ComputeTask's join like every other path.
//   * 1024 (eight pages): S = 8 gives each worker one page, two tiles -- the bounded short
//     pipeline, below the three-slot ring's depth. S <= 4 reaches the steady loop.
//   * 2048 (sixteen pages): every worker of every S is on the deep three-zone path, and at
//     S = 1 one worker walks all 32 tiles, which wraps the three-slot ring ten times. This is
//     the rung that carries the NON-POWER-OF-2 split sweep, S in {3, 5, 7}. blocksPerSplit is
//     ceil(16 / S), so the partition is uneven in a way that differs with S: at S = 3 the
//     workers own 12, 12 and 8 tiles; at S = 5 four own 8 each and the FIFTH IS EMPTY; at
//     S = 7 five own 6, one owns 2 and one is empty. An empty worker still publishes a
//     partial -- BeginTask zeroes the accumulator and sets runMax to -inf, and StageTaskOutput
//     writes that neutral element -- so the reduction has to combine S partials of which some
//     carry l = 0, with its row stride equal to numSplits_ rather than to a power of two and
//     the C0 lanes above numSplits_ zero, or the dense Mmad sums padding into a live row.
//   * 2048 with two tokens: the token dimension of the reduce phase, whose per-token
//     NeedsReduction() loop and whose task ids both have to stay in step across the roles.
//
// RAGGED, where the context divides neither the page nor the 64-row tile. The fixture fills
// every row of every mapped page, tail rows included, so a tail the kernel fails to mask
// carries its own tile's key level and value levels and weights that tile by 64 rows instead
// of by its true count -- which the oracle, weighting each tile by its live rows, rejects:
//
//   * 79 (one page, 79 live rows): one full tile and a 15-row tail. The pure tail-mask rung,
//     and at S = 2 and 3 a ragged worker beside empty ones.
//   * 257 (three pages, the last holding ONE token): five tiles, the last with a single live
//     row. The inter-page boundary switch with a minimal tail -- at S = 2 and 3 the worker
//     that owns page 2 has nothing but that one row, which is the narrowest mask the engine
//     can be given (an all-masked tile must never be staged at all, and the host tiling is what
//     guarantees that: num_tiles is ceil(live rows / 64) per worker, never counting a tile of
//     zero, and it is the one word both roles branch on to skip an empty worker).
//   * 1055 (nine pages, a 31-row tail): 17 tiles. The tail lands on worker 2 at S = 3 and on
//     worker 4 at S = 5 and 7, and at S = 7 two workers are empty besides -- so the short tile
//     is owned by a worker that is neither the first nor the last.
//
// MULTI-KV-HEAD, 4 KV heads x 16 query heads = 64 query heads:
//
//   * 1024 at S in {1, 2, 4, 8}: tasks = tokens * KV heads * S, so S = 8 dispatches a grid of
//     exactly 32 blocks -- the co-resident bound the reduction's SyncAll<false>() needs, hit
//     exactly rather than approached. It is also the rung where the scale cache has NO padding
//     lane left: the codec reads lanes [head] and [4 + head] of eight, so all eight are live
//     and that canary is vacuous here by construction, which this reports rather than skips.
//   * 1055 at S in {3, 7}: 4 KV heads, a ragged tail and a non-power-of-2 split count at once.
//
// WHAT THIS LADDER DOES NOT ESTABLISH: it is one launch per (rung, S), so it says nothing about
// inter-wave behaviour -- every grid here is inside the 32-block co-resident bound
// (turboquant_splitk_contract.hpp), which is also what the reduction's SyncAll<false>()
// requires, so a second execution wave is out of scope by construction. It takes no timing and
// makes no overlap claim; scripts/profile_single_tile_msprof.sh is for that. And it does not
// canary the TAIL ROWS of a mapped page: a masked lane's scale still flows through the Gather
// and the Muls in front of the mask, so a 0x5A5A5A5A scale there (~3.8e15) would be testing the
// masking phase's numerical headroom rather than its logic, and a failure could not be told
// apart from one. Those rows carry ordinary fixture data, which the oracle already rejects if
// they are counted.

#include "../npu/turboquant_npu_common.hpp"
#include "../common/fused_decode_opt_fixture.hpp"
#include "../../include/turboquant_splitk_contract.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <set>
#include <string>
#include <vector>

namespace {

namespace fd = ascend_micro::fused_decode_opt;
using fused_decode_opt_test::Fixture;
using tq_npu::check;
using tq_npu::require;

// The canary this file guards with, as a byte and as the word it forms.
constexpr std::uint8_t kGuardByte = 0x5A;
constexpr std::uint32_t kGuardWord = 0x5A5A5A5Au;
// Guard band on each side of every allocation. 64 bytes keeps every payload 32-byte aligned,
// which tq_npu::Buffer requires, and is wider than one UB block so a single over-length burst
// cannot step past it unseen.
constexpr std::size_t kGuard = 64;

// Real operator geometry. The page size and the ULP budget are the ladder's; the head counts
// vary per rung and live in the rung table.
constexpr int kPageTokens = 128;
constexpr int kUlpBudget = 0;
// Scale lanes per cache row, of which the codec reads two per KV head.
constexpr std::size_t kScaleLanes = 8;

// A device allocation with 0x5A guard bands and a host snapshot of what went into it.
struct Guarded {
    tq_npu::Buffer allocation;
    std::size_t bytes;                   // the payload, past the leading guard
    std::vector<std::uint8_t> snapshot;  // the whole allocation, as uploaded
    explicit Guarded(std::size_t payload)
        : allocation(fused_decode_opt_test::aligned(payload) + 2 * kGuard),
          bytes(fused_decode_opt_test::aligned(payload))
    {
    }
    void* ptr() const { return static_cast<std::uint8_t*>(allocation.ptr) + kGuard; }

    // Upload `payload` into the middle and 0x5A everywhere else -- the guards AND the slack
    // between the payload's end and the aligned length.
    template <class T>
    void Upload(const std::vector<T>& payload)
    {
        const std::size_t n = payload.size() * sizeof(T);
        require(n <= bytes, "fused-opt NPU validation: payload exceeds its allocation");
        snapshot.assign(allocation.bytes, kGuardByte);
        std::memcpy(snapshot.data() + kGuard, payload.data(), n);
        allocation.upload(snapshot);
    }
    // No payload: the whole allocation is canary. Outputs and workspaces start here.
    void FillGuard()
    {
        snapshot.assign(allocation.bytes, kGuardByte);
        allocation.upload(snapshot);
    }
    std::vector<std::uint8_t> Download() const
    {
        std::vector<std::uint8_t> image(allocation.bytes);
        allocation.download(image);
        return image;
    }
    // The guard bands, on both sides, after a launch.
    void CheckGuards(const std::vector<std::uint8_t>& image, const char* what) const
    {
        for (std::size_t i = 0; i < kGuard; ++i) {
            if (image[i] != kGuardByte || image[kGuard + bytes + i] != kGuardByte) {
                std::fprintf(stderr, "guard band of %s clobbered at %zu: %02x / %02x\n", what, i,
                             image[i], image[kGuard + bytes + i]);
                require(false, "fused-opt NPU validation: 0x5A guard band clobbered");
            }
        }
    }
    void CheckUnchanged(const char* what) const
    {
        const auto image = Download();
        CheckGuards(image, what);
        if (image != snapshot) {
            std::fprintf(stderr, "input allocation %s changed under the kernel\n", what);
            require(false, "fused-opt NPU validation: an input allocation changed");
        }
    }
};

// The interior canary plan: which physical KV pages the block table does not map, and which
// scale lanes of a mapped page nothing may touch.
struct CanaryPlan {
    std::vector<std::uint8_t> unmapped_page;  // per physical page
    int unmapped_pages = 0;
    std::size_t scale_pad_floats = 0;
};

CanaryPlan plan_canaries(const Fixture& f)
{
    CanaryPlan plan;
    plan.unmapped_page.assign(std::size_t(f.physical_pages), 1);
    for (int request = 0; request < f.requests; ++request)
        for (int page = 0; page < f.pages; ++page) {
            const std::int32_t physical = f.table[std::size_t(request) * f.pages + page];
            require(physical >= 0 && physical < f.physical_pages,
                    "fused-opt NPU validation: fixture block table out of range");
            plan.unmapped_page[std::size_t(physical)] = 0;
        }
    for (std::uint8_t unmapped : plan.unmapped_page) plan.unmapped_pages += unmapped ? 1 : 0;
    require(plan.unmapped_pages > 0, "fused-opt NPU validation: fixture maps every physical page, "
                                     "so there is no unmapped-slot canary to check");
    return plan;
}

// 0x5A over every unmapped physical page of a byte-addressed cache plane.
void paint_unmapped(std::vector<std::uint8_t>& plane, const CanaryPlan& plan, std::size_t page_bytes)
{
    require(plane.size() == plan.unmapped_page.size() * page_bytes,
            "fused-opt NPU validation: cache plane is not a whole number of pages");
    for (std::size_t page = 0; page < plan.unmapped_page.size(); ++page)
        if (plan.unmapped_page[page])
            std::fill(plane.begin() + page * page_bytes, plane.begin() + (page + 1) * page_bytes,
                      kGuardByte);
}

// 0x5A over the unmapped pages of the scale plane AND over the inter-head padding lanes of
// every mapped page. The codec reads lanes [head] and [kv_heads + head] of each row's eight, so
// lanes [2 * kv_heads, 8) are padding at every row of every page -- six of eight at one KV
// head, and NONE at four, where all eight lanes are live. At four KV heads this paints the
// unmapped pages only and leaves plan.scale_pad_floats at zero for the mapped ones, which the
// run reports: a canary with nothing to guard is worth saying out loud rather than asserting.
void paint_scales(std::vector<float>& scales, const Fixture& f, CanaryPlan& plan)
{
    float canary;
    std::memcpy(&canary, &kGuardWord, 4);
    const std::size_t rows_per_page = std::size_t(f.page_tokens);
    const std::size_t used_lanes = std::size_t(2 * f.kv_heads);
    require(scales.size() == std::size_t(f.physical_pages) * rows_per_page * kScaleLanes,
            "fused-opt NPU validation: scale plane geometry drifted from the fixture");
    require(used_lanes <= kScaleLanes,
            "fused-opt NPU validation: the scale slot cannot hold this KV head count");
    for (std::size_t page = 0; page < plan.unmapped_page.size(); ++page)
        for (std::size_t row = 0; row < rows_per_page; ++row) {
            const std::size_t base = (page * rows_per_page + row) * kScaleLanes;
            const std::size_t from = plan.unmapped_page[page] ? 0 : used_lanes;
            for (std::size_t lane = from; lane < kScaleLanes; ++lane) scales[base + lane] = canary;
            if (!plan.unmapped_page[page]) plan.scale_pad_floats += kScaleLanes - from;
        }
}

struct OracleResult {
    int worst_ulp = 0;
    double worst_abs = 0;
};

// Every live output channel against the fixture's FP64 oracle, plus the output tail canary.
OracleResult check_output(const std::vector<std::uint8_t>& image, const Fixture& f,
                          std::size_t payload_bytes)
{
    OracleResult result;
    const std::size_t live = f.expected.size();
    for (std::size_t i = 0; i < live; ++i) {
        std::uint16_t bits;
        std::memcpy(&bits, image.data() + kGuard + 2 * i, 2);
        const std::uint16_t reference = fused_decode_opt_test::reference_half(f.expected[i]);
        // The fixture's results are positive finite normals, so the fp16 bit distance IS the
        // ULP distance -- no sign or subnormal case to fold in.
        const int ulp = std::abs(int(bits) - int(reference));
        const double got = ascend_micro::sim::half_to_float(bits);
        const double absolute = std::fabs(got - double(f.expected[i]));
        if (ulp > result.worst_ulp) result.worst_ulp = ulp;
        if (absolute > result.worst_abs) result.worst_abs = absolute;
        if (ulp > kUlpBudget) {
            std::fprintf(stderr,
                         "oracle mismatch at channel %zu: got %g (%04x) expected %g (%04x), "
                         "%d fp16 ULP over a budget of %d\n",
                         i, got, bits, double(f.expected[i]), reference, ulp, kUlpBudget);
            require(false, "fused-opt NPU validation: output disagrees with the FP64 oracle");
        }
    }
    for (std::size_t i = 2 * live; i < payload_bytes; ++i)
        if (image[kGuard + i] != kGuardByte) {
            std::fprintf(stderr, "output tail byte %zu past the last live channel is %02x\n", i,
                         image[kGuard + i]);
            require(false, "fused-opt NPU validation: output tail canary clobbered");
        }
    return result;
}

struct Rung {
    int context;
    int requests;
    int kv_heads;
    int group_heads;  // query heads per KV head
    std::vector<int> splits;
    const char* note;
};

int run(bool camodel_smoke)
{
    tq_npu::Session session;

    const std::vector<Rung> full_ladder = {
        // aligned, one KV head, 16 query heads
        {kPageTokens, 1, 1, 16, {1, 2, 4, 8}, "one page: S>1 leaves unmapped (empty) splits"},
        {8 * kPageTokens, 1, 1, 16, {1, 2, 4, 8}, "eight pages: bounded short path at S=8"},
        {16 * kPageTokens, 1, 1, 16, {1, 2, 3, 4, 5, 7, 8},
         "sixteen pages: deep path at every S, and the non-power-of-2 split sweep"},
        {16 * kPageTokens, 2, 1, 16, {1, 2, 4, 8},
         "sixteen pages, two tokens: the reduce phase's token dimension"},
        // ragged: context divides neither the page nor the 64-row tile
        {79, 1, 1, 16, {1, 2, 3}, "one page, 79 rows: one full tile and a 15-row tail"},
        {257, 1, 1, 16, {1, 2, 3, 5},
         "three pages, the last holding ONE token: a single-row tile across a page boundary"},
        {1055, 1, 1, 16, {1, 3, 5, 7},
         "nine pages, a 31-row tail: the tail lands on a different worker at each S"},
        // multi-KV-head: 4 x 16 = 64 query heads
        {8 * kPageTokens, 1, 4, 16, {1, 2, 4, 8},
         "4 KV heads, 64 query heads: grid reaches exactly the 32-block bound at S=8"},
        {1055, 1, 4, 16, {3, 7},
         "4 KV heads, ragged tail and a non-power-of-2 split count at once"},
    };
    const std::vector<Rung> smoke_ladder = {
        {kPageTokens, 1, 1, 16, {1, 2}, "aligned one-page CAModel smoke"},
        {79, 1, 1, 16, {1, 2}, "ragged 15-row CAModel smoke"},
        {257, 1, 1, 16, {1, 3}, "single-row tail CAModel smoke"},
    };
    const auto& ladder = camodel_smoke ? smoke_ladder : full_ladder;

    int launches = 0, worst_ulp = 0, ragged_rungs = 0, narrowest_tail = 64;
    double worst_abs = 0;
    long long canary_pages = 0, canary_pad_floats = 0;
    std::set<int> splits_seen, kv_heads_seen;

    for (const Rung& rung : ladder) {
        // One fixture per rung: the oracle is the expensive part and the split count changes
        // only how the work is partitioned, never the answer.
        const Fixture f(rung.context, rung.requests, rung.kv_heads, rung.group_heads, kPageTokens);
        CanaryPlan plan = plan_canaries(f);
        const int tail = f.context % 64;
        if (tail != 0) {
            ++ragged_rungs;
            if (tail < narrowest_tail) narrowest_tail = tail;
            // The rung has to be ragged in the way its note claims, or it is testing the
            // aligned path under a ragged name.
            require(f.valid_rows(f.tiles() - 1) == tail,
                    "fused-opt NPU validation: a ragged rung's last tile is not short");
        }
        kv_heads_seen.insert(rung.kv_heads);

        std::vector<std::uint8_t> key = f.key, value = f.value;
        std::vector<float> scales = f.scales;
        const std::size_t page_bytes = std::size_t(f.page_tokens) * f.kv_heads * 128;
        paint_unmapped(key, plan, page_bytes);
        paint_unmapped(value, plan, page_bytes);
        paint_scales(scales, f, plan);

        for (const int k : rung.splits) {
            splits_seen.insert(k);
            const fd::Launch config = f.launch(k);
            require(config.grid <= turboquant_splitk::kMaxCoResidentBlocks,
                    "fused-opt NPU validation: grid exceeds the one-wave bound the reduction's "
                    "SyncAll needs");

            Guarded query(f.query.size() * 4), keys(key.size()), values(value.size()),
                scale(scales.size() * 4), table(f.table.size() * 4), lengths(f.lengths.size() * 4),
                modes(32),
                workspace(std::size_t(f.requests) * config.heads * std::size_t(k) * (256 + 16) * 4 +
                          4096),
                out(f.query.size() * 2 + 32),
                tiling(ascend_micro::fused_decode::decode_tiling_bytes_for(config));
            query.Upload(f.query);
            keys.Upload(key);
            values.Upload(value);
            scale.Upload(scales);
            table.Upload(f.table);
            lengths.Upload(f.lengths);
            modes.Upload(std::vector<std::uint32_t>(8, 0));
            // The host tiling, from the same context lengths and block table the device gets.
            // Every tile count, live-tail length and page span this launch acts on is decided
            // here; the device derives none of it.
            {
                namespace fd_tiling = ascend_micro::fused_decode;
                std::vector<std::uint32_t> image(
                    fd_tiling::decode_tiling_words(config.tokens, config.splits, config.grid), 0);
                const auto status = fd_tiling::build_decode_tiling(
                    fd_tiling::decode_tiling_request(config), f.lengths.data(), f.table.data(),
                    image.data());
                require(status == fd_tiling::DecodeTilingStatus::kOk,
                        fd_tiling::decode_tiling_status_text(status));
                tiling.Upload(image);
            }
            workspace.FillGuard();
            out.FillGuard();

            const fd::Buffers buffers{query.ptr(), keys.ptr(), values.ptr(), scale.ptr(),
                                      table.ptr(), lengths.ptr(), modes.ptr(), workspace.ptr(),
                                      out.ptr(), tiling.ptr()};
            check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
            launch_fused_decode_opt(session.stream, &config, &buffers);
            check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");
            ++launches;

            // 1 and 2d: the oracle and the output's tail canary.
            const auto image = out.Download();
            out.CheckGuards(image, "output");
            const OracleResult oracle = check_output(image, f, out.bytes);
            worst_ulp = std::max(worst_ulp, oracle.worst_ulp);
            worst_abs = std::max(worst_abs, oracle.worst_abs);

            // 4, which also covers 2a for the inputs and 2b/2c: the unmapped pages and the
            // scale padding are part of the uploaded image, so a byte comparison of the whole
            // allocation is what checks them. They are counted separately below so the run
            // reports how much canary it actually held rather than only that nothing changed.
            query.CheckUnchanged("query");
            keys.CheckUnchanged("key cache");
            values.CheckUnchanged("value cache");
            scale.CheckUnchanged("scale cache");
            table.CheckUnchanged("block table");
            lengths.CheckUnchanged("context lengths");
            modes.CheckUnchanged("mode tables");
            canary_pages += 2 * plan.unmapped_pages;  // key and value planes
            canary_pad_floats += static_cast<long long>(plan.scale_pad_floats);

            // 3: the workspace.
            const auto partials = workspace.Download();
            workspace.CheckGuards(partials, "workspace");
            if (k > 1) {
                const std::size_t floats =
                    std::size_t(f.requests) * config.heads * std::size_t(k) * (256 + 16);
                for (std::size_t i = 0; i < floats; ++i) {
                    float v;
                    std::memcpy(&v, partials.data() + kGuard + i * 4, 4);
                    if (!std::isfinite(v)) {
                        std::fprintf(stderr, "partial float %zu is %g: left unwritten or spoiled\n",
                                     i, double(v));
                        require(false, "fused-opt NPU validation: a Split-K partial came back "
                                       "non-finite");
                    }
                }
                for (std::size_t i = floats * 4; i < workspace.bytes; ++i)
                    if (partials[kGuard + i] != kGuardByte)
                        require(false, "fused-opt NPU validation: the workspace past the live "
                                       "partial extent was written");
            } else if (partials != workspace.snapshot) {
                require(false, "fused-opt NPU validation: the workspace was written at S = 1, "
                               "where the operator takes its fused path and NeedsReduction() is "
                               "false");
            }

            std::printf("  ok  N %5d  tok %d  kvh %d  qh %2d  S %d  grid %2d  tiles %3d "
                        "tail %2d  worst %d ULP  worst abs %.3e  %d unmapped pages, %zu scale "
                        "pad floats intact\n",
                        rung.context, rung.requests, rung.kv_heads,
                        rung.kv_heads * rung.group_heads, k, config.grid, f.tiles(), tail,
                        oracle.worst_ulp, oracle.worst_abs, plan.unmapped_pages,
                        plan.scale_pad_floats);
        }
        std::printf("  rung N %5d tok %d kvh %d: %s\n", rung.context, rung.requests,
                    rung.kv_heads, rung.note);
    }

    require(ragged_rungs >= 3, "fused-opt NPU validation: the ragged rungs went missing");
    require(narrowest_tail == 1, "fused-opt NPU validation: no single-row tail was exercised");
    if (!camodel_smoke) {
        require(kv_heads_seen.count(4) == 1,
                "fused-opt NPU validation: no multi-KV-head rung ran");
        for (const int k : {3, 5, 7})
            require(splits_seen.count(k) == 1,
                    "fused-opt NPU validation: a non-power-of-2 split count went missing");
    }
    require(canary_pad_floats > 0,
            "fused-opt NPU validation: no rung had an inter-head scale padding lane to guard");

    std::string split_list;
    for (const int k : splits_seen) split_list += (split_list.empty() ? "" : ",") + std::to_string(k);
    std::printf("PASS fused decode opt NPU validation: %d launches over %zu rungs, S in {%s}, "
                "1 and 4 KV heads, 16 query heads per KV head, D=%d, %d-token pages. %d rungs "
                "are RAGGED (context divides neither the page nor the 64-row tile), the "
                "narrowest live tail being %d row. Worst fp16 ULP %d against a budget of %d; "
                "worst absolute residual %.3e (the fp16 quantisation of the oracle). 0x%08X "
                "canaries held: %lld unmapped physical KV pages, %lld inter-head scale padding "
                "floats, both guard bands of every allocation, the output tail and the "
                "workspace outside the live partial extent. Every input allocation "
                "byte-identical afterwards.\n",
                launches, ladder.size(), split_list.c_str(), int(fd::kHeadSize), kPageTokens,
                ragged_rungs, narrowest_tail, worst_ulp, kUlpBudget, worst_abs, kGuardWord,
                canary_pages, canary_pad_floats);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        bool camodel_smoke = false;
        if (argc == 2 && std::string(argv[1]) == "--camodel-smoke") {
            camodel_smoke = true;
        } else if (argc != 1) {
            throw std::runtime_error("usage: test_integration_decode [--camodel-smoke]");
        }
        return run(camodel_smoke);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL fused decode opt NPU validation: %s\n", error.what());
        return 1;
    }
}
