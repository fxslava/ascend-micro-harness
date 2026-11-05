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
// GQA at the geometry the operator ships: 16 query heads over one KV head, D = 256, 128-token
// physical pages, splits S in {1, 2, 4, 8}. 128-token pages are two 64-row Cube tiles, so the
// per-worker tile count -- which is what selects the schedule's path -- is 2 * (pages per
// worker), and the four contexts are chosen to cover every path the kernel has:
//
//   * 128 (one page): at S = 2, 4 and 8 only the first worker has a block range at all. The
//     others are UNMAPPED SPLITS, blockStart >= blockEnd, which is the early exit that now
//     leaves through ComputeTask's join like every other path.
//   * 1024 (eight pages): S = 8 gives each worker one page, two tiles -- the bounded short
//     pipeline, below the three-slot ring's depth. S <= 4 reaches the steady loop.
//   * 2048 (sixteen pages): every worker of every S is on the deep three-zone path, and at
//     S = 1 one worker walks all 32 tiles, which wraps the three-slot ring ten times.
//   * 2048 with two tokens: the token dimension of the reduce phase, whose per-token
//     NeedsReduction() loop and whose task ids both have to stay in step across the roles.
//
// WHAT THIS LADDER DOES NOT ESTABLISH: it is one launch per rung, so it says nothing about
// inter-wave behaviour -- every grid here is inside the 32-block co-resident bound
// (turboquant_splitk_contract.hpp), which is also what the reduction's SyncAll<false>()
// requires, so a second execution wave is out of scope by construction. It takes no timing and
// makes no overlap claim; scripts/profile_single_tile_msprof.sh is for that.

#include "../turboquant_npu_common.hpp"
#include "../../common/fused_decode_opt_fixture.hpp"
#include "../../../include/turboquant_splitk_contract.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
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

// Real operator geometry.
constexpr int kKvHeads = 1;
constexpr int kQueryHeads = 16;  // group heads per KV head; 16 query heads in total
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
// every mapped page. The codec reads lanes [head] and [kv_heads + head] of each row's eight,
// so lanes [2 * kv_heads, 8) are padding at every row of every page.
void paint_scales(std::vector<float>& scales, const Fixture& f, CanaryPlan& plan)
{
    float canary;
    std::memcpy(&canary, &kGuardWord, 4);
    const std::size_t rows_per_page = std::size_t(f.page_tokens);
    const std::size_t used_lanes = std::size_t(2 * f.kv_heads);
    require(scales.size() == std::size_t(f.physical_pages) * rows_per_page * kScaleLanes,
            "fused-opt NPU validation: scale plane geometry drifted from the fixture");
    require(used_lanes < kScaleLanes,
            "fused-opt NPU validation: no scale padding lanes at this KV head count");
    for (std::size_t page = 0; page < plan.unmapped_page.size(); ++page)
        for (std::size_t row = 0; row < rows_per_page; ++row) {
            const std::size_t base = (page * rows_per_page + row) * kScaleLanes;
            const std::size_t from = plan.unmapped_page[page] ? 0 : used_lanes;
            for (std::size_t lane = from; lane < kScaleLanes; ++lane) scales[base + lane] = canary;
            plan.scale_pad_floats += kScaleLanes - from;
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
    const char* note;
};

int run()
{
    tq_npu::Session session;

    const std::vector<Rung> ladder = {
        {kPageTokens, 1, "one page: S>1 leaves unmapped (empty) splits"},
        {8 * kPageTokens, 1, "eight pages: bounded short path at S=8"},
        {16 * kPageTokens, 1, "sixteen pages: deep three-zone path at every S"},
        {16 * kPageTokens, 2, "sixteen pages, two tokens: the reduce phase's token dimension"},
    };
    const std::vector<int> splits = {1, 2, 4, 8};

    int launches = 0, worst_ulp = 0;
    double worst_abs = 0;
    long long canary_pages = 0, canary_pad_floats = 0;

    for (const Rung& rung : ladder) {
        // One fixture per (context, tokens): the oracle is the expensive part and the split
        // count changes only how the work is partitioned, never the answer.
        const Fixture f(rung.context, rung.requests, kKvHeads, kQueryHeads, kPageTokens);
        CanaryPlan plan = plan_canaries(f);

        std::vector<std::uint8_t> key = f.key, value = f.value;
        std::vector<float> scales = f.scales;
        const std::size_t page_bytes = std::size_t(f.page_tokens) * f.kv_heads * 128;
        paint_unmapped(key, plan, page_bytes);
        paint_unmapped(value, plan, page_bytes);
        paint_scales(scales, f, plan);

        for (const int k : splits) {
            const fd::Launch config = f.launch(k);
            require(config.grid <= turboquant_splitk::kMaxCoResidentBlocks,
                    "fused-opt NPU validation: grid exceeds the one-wave bound the reduction's "
                    "SyncAll needs");

            Guarded query(f.query.size() * 4), keys(key.size()), values(value.size()),
                scale(scales.size() * 4), table(f.table.size() * 4), lengths(f.lengths.size() * 4),
                modes(32),
                workspace(std::size_t(f.requests) * config.heads * std::size_t(k) * (256 + 16) * 4 +
                          4096),
                out(f.query.size() * 2 + 32);
            query.Upload(f.query);
            keys.Upload(key);
            values.Upload(value);
            scale.Upload(scales);
            table.Upload(f.table);
            lengths.Upload(f.lengths);
            modes.Upload(std::vector<std::uint32_t>(8, 0));
            workspace.FillGuard();
            out.FillGuard();

            const fd::Buffers buffers{query.ptr(), keys.ptr(), values.ptr(), scale.ptr(),
                                      table.ptr(), lengths.ptr(), modes.ptr(), workspace.ptr(),
                                      out.ptr()};
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

            std::printf("  ok  context %5d  tokens %d  S %d  grid %2d  worst %d ULP  "
                        "worst abs %.3e  %d unmapped pages and %zu scale pad floats intact\n",
                        rung.context, rung.requests, k, config.grid, oracle.worst_ulp,
                        oracle.worst_abs, plan.unmapped_pages, plan.scale_pad_floats);
        }
        std::printf("  rung context %5d tokens %d: %s\n", rung.context, rung.requests, rung.note);
    }

    std::printf("PASS fused decode opt NPU validation: %d launches over %zu contexts x "
                "S in {1,2,4,8} at %d query heads over %d KV head, D=%d, %d-token pages. "
                "Worst fp16 ULP %d against a budget of %d; worst absolute residual %.3e "
                "(the fp16 quantisation of the oracle). 0x%08X canaries held: %lld unmapped "
                "physical KV pages, %lld inter-head scale padding floats, both guard bands of "
                "every allocation, the output tail and the workspace outside the live partial "
                "extent. Every input allocation byte-identical afterwards.\n",
                launches, ladder.size(), kQueryHeads, kKvHeads, int(fd::kHeadSize), kPageTokens,
                worst_ulp, kUlpBudget, worst_abs, kGuardWord, canary_pages, canary_pad_floats);
    return 0;
}

}  // namespace

int main()
{
    try {
        return run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL fused decode opt NPU validation: %s\n", error.what());
        return 1;
    }
}
