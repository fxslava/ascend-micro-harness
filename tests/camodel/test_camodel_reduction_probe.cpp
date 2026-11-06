// Synthetic Split-K CAModel probe: one real attention tile and seven neutral partials
// per KV head, followed by the production transposed Cube GEMM reducer.
#include "camodel_common.hpp"
#include "../../include/turboquant_fused_decode_opt.hpp"
#include "../../include/turboquant_fused_decode.hpp"
#include "../../kernels/ops/turboquant/fused_decode/common/turboquant_layout.h"
#include "../../kernels/ops/turboquant/fused_decode/common/turboquant_mode.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

extern "C" void launch_fused_decode_opt_sim(
    void*, const ascend_micro::fused_decode_opt::Launch*,
    const ascend_micro::fused_decode_opt::Buffers*);

namespace fd = ascend_micro::fused_decode_opt;
namespace tq = vllm_ascend::turboquant;

namespace {
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kRowsPerKv = 16;
constexpr std::uint32_t kHeads = kKvHeads * kRowsPerKv;
constexpr std::uint32_t kSplits = 8;
constexpr std::uint32_t kGrid = kKvHeads * kSplits;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kPageRows = 128;
constexpr std::uint32_t kPages = 128;
constexpr std::uint32_t kRecordFloats = kDim + 16;
constexpr std::uint32_t kWorkspaceFloats = kHeads * kSplits * kRecordFloats;
constexpr std::uint32_t kOutputElems = kHeads * kDim;
constexpr std::uint32_t kCodecTableWords = 9 * kDim + 16;
constexpr std::uint32_t kProbeSentinel = UINT32_MAX - 1;

std::uint16_t float_to_half(float value)
{
    const std::uint32_t bits = [&] { std::uint32_t out; std::memcpy(&out, &value, 4); return out; }();
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    if (exponent <= 0) return static_cast<std::uint16_t>(sign);
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    std::uint32_t rounded = mantissa + 0xfffu + ((mantissa >> 13) & 1u);
    if (rounded & 0x800000u) { rounded = 0; ++exponent; }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | (rounded >> 13));
}

void require(bool ok, const char* why)
{
    if (!ok) throw std::runtime_error(why);
}
} // namespace

int main()
{
    try {
        camodel::Session session;
        constexpr std::size_t keyBytes = std::size_t(kPages) * kPageRows * kDim / 2;
        constexpr std::size_t scaleFloats = std::size_t(kPages) * kPageRows * 8;
        camodel::GuardedDeviceBuffer query(kHeads * kDim * sizeof(float));
        camodel::GuardedDeviceBuffer key(keyBytes), value(keyBytes);
        camodel::GuardedDeviceBuffer scales(scaleFloats * sizeof(float));
        camodel::GuardedDeviceBuffer table(kPages * sizeof(std::int32_t));
        camodel::GuardedDeviceBuffer lengths(32), modes(kCodecTableWords * sizeof(std::int32_t));
        camodel::GuardedDeviceBuffer workspace(std::size_t(kWorkspaceFloats) * sizeof(float));
        camodel::GuardedDeviceBuffer output(kOutputElems * sizeof(std::uint16_t));

        auto flood = [](camodel::GuardedDeviceBuffer& b, std::uint32_t word = camodel::kCanary) {
            std::fill(b.host.begin(), b.host.end(), word);
            const auto guardWords = camodel::kGuardBytes / 4;
            std::fill(b.host.begin(), b.host.begin() + guardWords, camodel::kCanary);
            std::fill(b.host.end() - guardWords, b.host.end(), camodel::kCanary);
        };
        for (auto* b : {&query, &key, &value, &scales, &table, &lengths, &modes, &workspace, &output}) flood(*b);
        auto* q = static_cast<float*>(query.host_payload());
        std::fill(q, q + kHeads * kDim, 0.0f); // zero logits, independent of key dequantization.
        auto* codecTables = static_cast<std::int32_t*>(modes.host_payload());
        std::fill(codecTables, codecTables + kCodecTableWords, 0);
        // TurboQuantCodec<4> pool: three FWHT sign/index stages, even/odd/expand byte
        // offsets, odd-lane selectors, then centroids. The probe's neutral value mapping
        // uses the KV4 affine centroids; codes 8+i reconstruct to i+0.5, and the scale
        // cache's gain*2 then yields 2*i+1.
        std::size_t tableOffset = 0;
        for (std::uint32_t stage = 0; stage < 3; ++stage) {
            auto* signs = reinterpret_cast<float*>(codecTables + tableOffset);
            for (std::uint32_t i = 0; i < kDim; ++i) signs[i] = (i & (1u << stage)) ? -1.0f : 1.0f;
            tableOffset += kDim;
            for (std::uint32_t i = 0; i < kDim; ++i)
                codecTables[tableOffset + i] = static_cast<std::int32_t>((i ^ (1u << stage)) * sizeof(float));
            tableOffset += kDim;
        }
        constexpr std::uint32_t packedBytes = kDim / 2;
        for (std::uint32_t i = 0; i < packedBytes; ++i) {
            codecTables[tableOffset + i] = static_cast<std::int32_t>(2 * i * sizeof(float));
            codecTables[tableOffset + packedBytes + i] = static_cast<std::int32_t>((2 * i + 1) * sizeof(float));
        }
        tableOffset += 2 * packedBytes;
        for (std::uint32_t i = 0; i < kDim; ++i)
            codecTables[tableOffset + i] = static_cast<std::int32_t>((i / 2) * sizeof(float));
        tableOffset += kDim;
        auto* oddSelectors = reinterpret_cast<float*>(codecTables + tableOffset);
        for (std::uint32_t i = 0; i < kDim; ++i) oddSelectors[i] = (i & 1u) ? 1.0f : 0.0f;
        tableOffset += kDim;
        auto* centroids = reinterpret_cast<float*>(codecTables + tableOffset);
        for (std::uint32_t i = 0; i < 16; ++i) centroids[i] = static_cast<float>(i) - 7.5f;
        require(tableOffset + 16 == kCodecTableWords, "codec table layout drifted");
        auto* blocks = static_cast<std::int32_t*>(table.host_payload());
        std::fill(blocks, blocks + kPages, -1);
        blocks[0] = 1; // logical Page 0 maps to physical page 1; every other split is empty.
        *static_cast<std::int32_t*>(lengths.host_payload()) = 16384;
        auto* kp = static_cast<std::uint8_t*>(key.host_payload());
        auto* vp = static_cast<std::uint8_t*>(value.host_payload());
        auto* sc = static_cast<float*>(scales.host_payload());
        const float gain = tq::TurboQuantModeTraits<tq::TurboQuantMode::KV4_FP8>::kGain;
        for (std::uint32_t row = 0; row < 64; ++row) {
            for (std::uint32_t kv = 0; kv < kKvHeads; ++kv) {
                sc[(1 * kPageRows + row) * 8 + kv] = 2.0f * gain;
                sc[(1 * kPageRows + row) * 8 + kKvHeads + kv] = 2.0f * gain;
                for (std::uint32_t col = 0; col < kDim / 2; ++col) {
                    const auto off = std::size_t(1) * kPageRows * kDim / 2 +
                        tq::NzTiledPackedByte(row, kv, col, kKvHeads, kDim / 2);
                    const std::uint8_t level = static_cast<std::uint8_t>(col % 8);
                    const std::uint8_t lo = static_cast<std::uint8_t>(8 + level);
                    const std::uint8_t hi = static_cast<std::uint8_t>(level << 4);
                    kp[off] = 0; // zero scores: uniform softmax for the honest tile.
                    vp[off] = static_cast<std::uint8_t>(lo | hi);
                }
            }
        }

        fd::Launch config;
        config.grid = kGrid; config.tokens = 1; config.heads = kHeads; config.kv_heads = kKvHeads;
        config.head_size = kDim; config.block_size = kPageRows; config.max_blocks = kPages;
        config.splits = kSplits; config.heads_per_task = kRowsPerKv;
        config.tasks_per_block = 1; config.reduce_tasks_per_block = 2;
        config.fused_context_limit = 0; config.probe_page = kProbeSentinel;

        // THE PROBE'S TILING. kProbeSentinel is this runner's dispatch key into
        // launch_fused_decode_opt_sim and not a page index, so the descriptor request names the
        // page and tile the probe actually runs -- the first 64-row tile of logical page zero --
        // while the global context length stays full-sized so every block still takes the
        // reduction. The device used to get this as a context-length override argument that
        // ComputeTask clipped a page against; it is a host descriptor now, like every other
        // tile bound in this operator.
        camodel::GuardedDeviceBuffer tiling(
            (ascend_micro::fused_decode::decode_tiling_bytes_for(config) + 31) / 32 * 32);
        {
            namespace fdt = ascend_micro::fused_decode;
            auto request = fdt::decode_tiling_request(config);
            request.probe_page = 0;
            request.probe_tile = 0;
            std::vector<std::uint32_t> image(
                fdt::decode_tiling_words(config.tokens, config.splits, config.grid), 0);
            const auto status = fdt::build_decode_tiling(
                request, static_cast<const std::int32_t*>(lengths.host_payload()), nullptr,
                image.data());
            require(status == fdt::DecodeTilingStatus::kOk,
                    fdt::decode_tiling_status_text(status));
            std::memcpy(tiling.host_payload(), image.data(), image.size() * 4);
        }

        std::vector<std::vector<std::uint32_t>> snapshots;
        for (auto* b : {&query, &key, &value, &scales, &table, &lengths, &modes, &tiling}) {
            b->upload(); snapshots.push_back(b->host);
        }
        fd::Buffers buffers{query.payload(), key.payload(), value.payload(), scales.payload(),
            table.payload(), lengths.payload(), modes.payload(), workspace.payload(),
            output.payload(), tiling.payload()};
        launch_fused_decode_opt_sim(session.stream, &config, &buffers);
        camodel::check(aclrtSynchronizeStream(session.stream), "reduction probe synchronize");
        output.download(); output.assert_guards(); workspace.download(); workspace.assert_guards();

        const auto* partials = static_cast<const float*>(workspace.host_payload());
        std::uint32_t neutralChecks = 0;
        for (std::uint32_t head = 0; head < kHeads; ++head) {
            for (std::uint32_t split = 0; split < kSplits; ++split) {
                const std::size_t base = (std::size_t(head) * kSplits + split) * kRecordFloats;
                if (split != 0) {
                    for (std::uint32_t c = 0; c < kDim; ++c)
                        require(partials[base + c] == 0.0f, "neutral accumulator is not zero");
                    require(partials[base + kDim] == -1.0e9f,
                            "neutral partial maximum is not -1e9");
                    require(partials[base + kDim + 8] == 0.0f,
                            "neutral partial sum is not zero");
                    for (std::uint32_t lane = 1; lane < 8; ++lane) {
                        require(partials[base + kDim + lane] == 0.0f,
                                "neutral maximum tail is not zero");
                        require(partials[base + kDim + 8 + lane] == 0.0f,
                                "neutral sum tail is not zero");
                    }
                    ++neutralChecks;
                } else {
                    bool nonzero = false;
                    for (std::uint32_t c = 0; c < kDim; ++c) {
                        require(std::isfinite(partials[base + c]), "worker zero accumulator is not finite");
                        nonzero |= partials[base + c] != 0.0f;
                    }
                    require(nonzero, "worker zero did not publish a real one-tile accumulator");
                    require(partials[base + kDim] == 0.0f,
                            "worker zero's zero-logit maximum is unexpected");
                    require(partials[base + kDim + 8] == 64.0f,
                            "worker zero's one-tile softmax mass is unexpected");
                    for (std::uint32_t lane = 1; lane < 8; ++lane) {
                        require(partials[base + kDim + lane] == 0.0f,
                                "worker zero maximum tail is not zero");
                        require(partials[base + kDim + 8 + lane] == 0.0f,
                                "worker zero sum tail is not zero");
                    }
                }
            }
        }

        const auto* out = static_cast<const std::uint16_t*>(output.host_payload());
        std::uint32_t maxUlp = 0;
        for (std::uint32_t head = 0; head < kHeads; ++head) {
            const auto kv = head / kRowsPerKv;
            for (std::uint32_t d = 0; d < kDim; ++d) {
                const std::size_t base = (std::size_t(head) * kSplits) * kRecordFloats;
                const float expected = partials[base + d] / (partials[base + kDim + 8] + 1.0e-20f);
                const auto want = float_to_half(expected);
                const auto got = out[head * kDim + d];
                const auto ulp = static_cast<std::uint32_t>(std::abs(int(got) - int(want)));
                maxUlp = std::max(maxUlp, ulp);
                if (ulp != 0) {
                    std::fprintf(stderr, "mismatch kv=%u head=%u d=%u got=%04x want=%04x ulp=%u\n",
                                 kv, head, d, got, want, ulp);
                    throw std::runtime_error("synthetic Split-K reduction was not 0 ULP");
                }
            }
        }
        for (std::size_t i = 0; i < snapshots.size(); ++i) {
            auto* b = std::array<camodel::GuardedDeviceBuffer*, 8>{
                &query, &key, &value, &scales, &table, &lengths, &modes, &tiling}[i];
            b->download();
            require(b->host == snapshots[i], "an input payload changed");
        }
        std::printf("PASS synthetic reduction probe: context=16384 pages=128 splits=8 kv_heads=4 "
                    "query_heads=64 grid=32 real_tiles=4 neutral_partials=%u max_ulp=%u "
                    "guards=0xDEADBEEF-intact\n", neutralChecks, maxUlp);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL synthetic reduction probe: %s\n", e.what());
        return 1;
    }
}
