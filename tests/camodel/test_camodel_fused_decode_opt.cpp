#include "camodel_common.hpp"
#include "../../include/turboquant_fused_decode_opt.hpp"
#include "../../include/sim_micro_testbed.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace fd = ascend_micro::fused_decode_opt;
namespace tq = vllm_ascend::turboquant;

int main(int argc, char** argv)
{
    try {
        fd::Launch c;
        unsigned tiles = 1, repeats = 1;
        bool empty = false, varied = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("missing CLI value");
            const auto n = static_cast<unsigned>(std::stoul(argv[++i]));
            if (arg == "--tiles") tiles = n;
            else if (arg == "--canary") c.probe_page = n;
            else if (arg == "--tile") c.probe_tile = n;
            else if (arg == "--heads") c.heads = n;
            else if (arg == "--splits") c.splits = n;
            else if (arg == "--repeats") repeats = n;
            else if (arg == "--empty") empty = n != 0;
            else if (arg == "--varied") varied = n != 0;
            else throw std::runtime_error("unknown CLI option");
        }
        if (!tiles || tiles > 16 || !repeats || (c.probe_page >= 8 && c.probe_page != UINT32_MAX) ||
            c.probe_tile >= 2 || !c.heads || c.heads > 32)
            throw std::runtime_error("unsupported fixture geometry");
        c.grid = c.splits;
        c.tasks_per_block = (c.heads + 15) / 16;
        c.reduce_tasks_per_block = (c.heads + c.grid - 1) / c.grid;
        c.fused_context_limit = c.splits == 1 ? 4096 : 0;
        c.scale = varied ? 0.0625f : 1.0f;
        constexpr unsigned pages = 18, page_rows = 128, dim = 256;
        constexpr unsigned scale_slot = 8;
        const auto round32 = [](std::size_t n) { return (n + 31) / 32 * 32; };
        camodel::Session session;
        camodel::GuardedDeviceBuffer query(round32(c.heads * dim * 4));
        camodel::GuardedDeviceBuffer key(pages * page_rows * dim / 2);
        camodel::GuardedDeviceBuffer value(pages * page_rows * dim / 2);
        camodel::GuardedDeviceBuffer scales(pages * page_rows * scale_slot * 4);
        camodel::GuardedDeviceBuffer table(32), lengths(32), modes(32);
        camodel::GuardedDeviceBuffer workspace(round32(c.heads * c.splits * (dim + 16) * 4 + 4096));
        // Extra inactive rows are an output-canary gate.
        camodel::GuardedDeviceBuffer output(round32((c.heads + 2) * dim * 2));
        std::vector<camodel::GuardedDeviceBuffer*> inputs{&query, &key, &value, &scales, &table, &lengths, &modes};
        for (auto* b : inputs) std::fill(b->host.begin() + 16, b->host.end() - 16, 0x7FC0BEEFu);
        auto* q = static_cast<float*>(query.host_payload());
        std::fill(q, q + c.heads * dim, 1.0f / 256);
        auto* bt = static_cast<std::int32_t*>(table.host_payload());
        for (unsigned p = 0; p < 8; ++p) bt[p] = 2 * p + 1;
        const bool isolated = c.probe_page != UINT32_MAX;
        *static_cast<std::int32_t*>(lengths.host_payload()) = empty ? 0 : isolated ? 1024 : tiles * 64;
        auto* kp = static_cast<std::uint8_t*>(key.host_payload());
        auto* vp = static_cast<std::uint8_t*>(value.host_payload());
        auto* sp = static_cast<float*>(scales.host_payload());
        const float gain = tq::TurboQuantModeTraits<tq::TurboQuantMode::KV4_FP8>::kGain;
        const unsigned first_tile = isolated ? c.probe_page * 2 + c.probe_tile : 0;
        const unsigned last_tile = isolated ? first_tile + 1 : tiles;
        for (unsigned tile = first_tile; tile < last_tile; ++tile) {
            const unsigned physical = bt[tile / 2];
            for (unsigned row = tile % 2 * 64; row < tile % 2 * 64 + 64; ++row) {
                sp[(physical * page_rows + row) * scale_slot] = gain * 2;
                sp[(physical * page_rows + row) * scale_slot + 1] = gain * 2;
                for (unsigned col = 0; col < dim / 2; ++col) {
                    const auto off = physical * page_rows * dim / 2 +
                        tq::NzTiledPackedByte(row, 0, col, 1, dim / 2);
                    const unsigned kn = varied ? tile % 8 : 0;
                    kp[off] = static_cast<std::uint8_t>(kn | (kn << 4));
                    const unsigned lo = (tile + col % 128) % 8;
                    const unsigned hi = (tile + (col + 3) % 128) % 8;
                    vp[off] = static_cast<std::uint8_t>(lo | (hi << 4));
                }
            }
        }
        std::vector<std::vector<std::uint32_t>> snapshots;
        for (auto* b : inputs) { b->upload(); snapshots.push_back(b->host); }
        fd::Buffers b{query.payload(), key.payload(), value.payload(), scales.payload(),
            table.payload(), lengths.payload(), modes.payload(), workspace.payload(), output.payload()};
        for (unsigned repeat = 0; repeat < repeats; ++repeat) {
            launch_fused_decode_opt(session.stream, &c, &b);
            camodel::check(aclrtSynchronizeStream(session.stream), "fused decode synchronize");
            output.download(); output.assert_guards(); workspace.download(); workspace.assert_guards();
            const auto* actual = static_cast<const std::uint16_t*>(output.host_payload());
            for (unsigned h = 0; h < c.heads; ++h) for (unsigned d = 0; d < dim; ++d) {
                double numerator = 0, denominator = 0;
                if (!empty) {
                    for (unsigned tile = first_tile; tile < last_tile; ++tile) {
                        const unsigned col = d % 128;
                        const unsigned n = (tile + (d < 128 ? col : (col + 3) % 128)) % 8;
                        const double weight = varied ? std::exp(c.scale * (2.0 * (tile % 8) + 1)) : 1;
                        numerator += weight * (2.0 * n + 1);
                        denominator += weight;
                    }
                }
                const float expected = denominator ? static_cast<float>(numerator / denominator) : 0;
                // Compare the independently rounded FP16 oracle, including
                // odd tile counts whose rational average is not representable.
                const auto bits = actual[h * dim + d];
                const float got = ascend_micro::sim::half_to_float(bits);
                const int ulp = std::abs(int(bits) - int(ascend_micro::sim::float_to_half_rne(expected)));
                if (ulp > (varied ? 1 : 0)) {
                    std::fprintf(stderr, "mismatch h=%u d=%u got=%g expected=%g bits=%04x\n", h, d, got, expected, bits);
                    throw std::runtime_error("exact context oracle failed");
                }
            }
            const auto* raw = static_cast<const std::uint32_t*>(output.host_payload());
            for (unsigned i = c.heads * dim / 2; i < output.payload_bytes / 4; ++i)
                if (raw[i] != camodel::kCanary) throw std::runtime_error("inactive output changed");
            for (unsigned i = 0; i < inputs.size(); ++i) {
                inputs[i]->download();
                if (inputs[i]->host != snapshots[i]) throw std::runtime_error("input bytes changed");
            }
        }
        std::printf("PASS fused_decode_opt tiles=%u page=%u tile=%u heads=%u splits=%u grid=%u repeats=%u elements=%u comparison=%s inputs=unchanged guards=intact\n",
            tiles, c.probe_page, c.probe_tile, c.heads, c.splits, c.grid, repeats, c.heads * dim,
            varied ? "fp16_1ulp" : "exact");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL fused_decode_opt: %s\n", e.what()); return 1;
    }
}
