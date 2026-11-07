#include "fused_decode_opt_sim_common.hpp"
#include "../../../include/turboquant_fused_decode_opt.hpp"
#include "../../../include/turboquant_fused_decode.hpp"
#include "../../../include/sim_micro_testbed.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

extern "C" void launch_fused_decode_opt_sim(void*,
    const ascend_micro::fused_decode_opt::Launch*, const ascend_micro::fused_decode_opt::Buffers*);

namespace fd = ascend_micro::fused_decode_opt;
namespace tq = vllm_ascend::turboquant;
namespace tq_tiling = ascend_micro::fused_decode;

// The output element format, as the pair of functions the oracle comparison needs: the
// correctly-rounded 16-bit word of an fp32 expectation, and the widening used ONLY to print what
// the device produced.
//
// REFINEMENT_OF_COMBAT_KERNELS.md section 3: the bf16 check is on the 16-bit word AT ITS ORIGINAL
// POSITION. That is what the comparison below does for both formats -- `actual[h * dim + d]`
// against `encode(expected)` -- and it is why neither branch widens the device's word first. A
// check that compared floats after widening, or that accepted either half of an fp32 word, would
// pass on a halfword-swapped output; comparing the stored word cannot.
//
// For both formats the fixture's results are positive finite normals, so the unsigned bit distance
// between two words IS their ULP distance: no sign wrap and no subnormal case to fold in.
struct OutputFormat {
    const char* name;
    std::uint16_t (*encode)(float);
    float (*decode)(std::uint16_t);
};
constexpr OutputFormat kFp16Output{"fp16", &ascend_micro::sim::float_to_half_rne,
                                   &ascend_micro::sim::half_to_float};
constexpr OutputFormat kBf16Output{"bf16", &ascend_micro::sim::float_to_bfloat16_rne,
                                   &ascend_micro::sim::bfloat16_to_float};

int main(int argc, char** argv)
{
    try {
        fd::Launch c;
        unsigned tiles = 1, repeats = 1, tail_rows = 64, hole = UINT32_MAX;
        bool empty = false, varied = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("missing CLI value");
            // --dtype takes a NAME, not a number, so a rung cannot select the wrong element type
            // by an off-by-one in the ladder script.
            if (arg == "--dtype") {
                const std::string value = argv[++i];
                if (value == "fp16") c.element_type = 0;
                else if (value == "bf16") c.element_type = 1;
                else throw std::runtime_error("unknown --dtype (want fp16 or bf16)");
                continue;
            }
            const auto n = static_cast<unsigned>(std::stoul(argv[++i]));
            if (arg == "--tiles") tiles = n;
            else if (arg == "--tail-rows") tail_rows = n;
            else if (arg == "--hole") hole = n;
            else if (arg == "--canary") c.probe_page = n;
            else if (arg == "--tile") c.probe_tile = n;
            else if (arg == "--heads") c.heads = n;
            else if (arg == "--splits") c.splits = n;
            else if (arg == "--repeats") repeats = n;
            else if (arg == "--empty") empty = n != 0;
            else if (arg == "--varied") varied = n != 0;
            else throw std::runtime_error("unknown CLI option");
        }
        const OutputFormat& out_format = c.element_type == 1 ? kBf16Output : kFp16Output;
        if (!tail_rows || tail_rows > 64 || (hole != UINT32_MAX && hole >= (tiles + 1) / 2) ||
            !tiles || tiles > 16 || !repeats || (c.probe_page >= 8 && c.probe_page != UINT32_MAX) ||
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
        // Extra inactive rows are an output-canary gate. Both element types are 16-bit
        // (fd::element_bytes), so one size serves both -- stated through the header's own
        // accessor rather than a literal 2, so a third type would have to move both together.
        camodel::GuardedDeviceBuffer output(
            round32((c.heads + 2) * dim * fd::element_bytes(c.element_type)));
        // The host tiling descriptors. An input like any other: guard-banded, canary-flooded
        // outside its payload, snapshotted before upload and compared bitwise after the run.
        camodel::GuardedDeviceBuffer tiling(round32(tq_tiling::decode_tiling_bytes_for(c)));
        std::vector<camodel::GuardedDeviceBuffer*> inputs{&query, &key, &value, &scales, &table,
            &lengths, &modes, &tiling};
        for (auto* b : inputs) std::fill(b->host.begin() + 16, b->host.end() - 16, camodel::kCanary);
        auto* q = static_cast<float*>(query.host_payload());
        std::fill(q, q + c.heads * dim, 1.0f / 256);
        auto* bt = static_cast<std::int32_t*>(table.host_payload());
        for (unsigned p = 0; p < 8; ++p) bt[p] = 2 * p + 1;
        if (hole != UINT32_MAX) bt[hole] = -1;
        const bool isolated = c.probe_page != UINT32_MAX;
        *static_cast<std::int32_t*>(lengths.host_payload()) = empty ? 0 : isolated ? 1024 : (tiles - 1) * 64 + tail_rows;
        auto* kp = static_cast<std::uint8_t*>(key.host_payload());
        auto* vp = static_cast<std::uint8_t*>(value.host_payload());
        auto* sp = static_cast<float*>(scales.host_payload());
        const float gain = tq::TurboQuantModeTraits<tq::TurboQuantMode::KV4_FP8>::kGain;
        const unsigned first_tile = isolated ? c.probe_page * 2 + c.probe_tile : 0;
        const unsigned last_tile = isolated ? first_tile + 1 : tiles;
        for (unsigned tile = first_tile; tile < last_tile; ++tile) {
            if (bt[tile / 2] < 0) continue;
            const unsigned physical = bt[tile / 2];
            for (unsigned row = tile % 2 * 64; row < tile % 2 * 64 + 64; ++row) {
                sp[(physical * page_rows + row) * scale_slot] = gain * 2;
                sp[(physical * page_rows + row) * scale_slot + 1] = gain * 2;
                for (unsigned col = 0; col < dim / 2; ++col) {
                    const auto off = physical * page_rows * dim / 2 +
                        tq::NzTiledPackedByte(row, 0, col, 1, dim / 2);
                    // REFINEMENT_OF_COMBAT_KERNELS.md section 3's exact fixture, as the deleted
                    // AIV boundary runners carried it: query = 1/256, scales = 1 * gain and a
                    // UNIFORM key nibble of 3 over every live token, so every score is the same
                    // number and exp(score - m) = exp(0) = 1 exactly. Nibble 3 rather than 0 so
                    // the fixture is the documented one and the packed key plane's CONTENT reaches
                    // the score GEMM; the weights are 1 either way, so the oracle below is
                    // unchanged by the choice. --varied replaces it with a per-tile nibble and
                    // takes the documented Exp/reduction tolerance instead.
                    const unsigned kn = varied ? tile % 8 : 3;
                    kp[off] = static_cast<std::uint8_t>(kn | (kn << 4));
                    const unsigned lo = (tile + col % 128) % 8;
                    const unsigned hi = (tile + (col + 3) % 128) % 8;
                    vp[off] = static_cast<std::uint8_t>(lo | (hi << 4));
                }
            }
        }
        // THE HOST TILING, BUILT HERE AND NOWHERE ELSE.
        //
        // Every tile count, live-tail length and logical page span the kernel will act on comes
        // out of this one call, from the same context length and block table the device is given.
        // `--hole` is the exception and it is deliberate: the table is planted with an unmapped
        // interior page and the builder is NOT shown it, so the descriptor claims a dense span the
        // table does not have. That is the only way to reach the kernel's own density guard, and
        // what this rung asserts is that reaching it skips the worker -- a defined zero result,
        // canaries intact, no deadlock -- rather than indexing through an unmapped page. The
        // host-side refusal of the same geometry is a Tier-1 gate
        // (tests/cpu/test_fused_decode_opt_contract.cpp), not a CAModel case.
        const bool planted_hole = hole != UINT32_MAX;
        auto request = tq_tiling::decode_tiling_request(c);
        const auto status = tq_tiling::build_decode_tiling(request,
            static_cast<const std::int32_t*>(lengths.host_payload()),
            planted_hole ? nullptr : bt, tiling.host_payload());
        if (status != tq_tiling::DecodeTilingStatus::kOk)
            throw std::runtime_error(tq_tiling::decode_tiling_status_text(status));

        std::vector<std::vector<std::uint32_t>> snapshots;
        for (auto* b : inputs) { b->upload(); snapshots.push_back(b->host); }
        fd::Buffers b{query.payload(), key.payload(), value.payload(), scales.payload(),
            table.payload(), lengths.payload(), modes.payload(), workspace.payload(),
            output.payload(), tiling.payload()};
        // The unassigned-byte census this run holds to the canary, printed in the PASS line.
        // Section 3 asks for every unassigned output slot and both exterior guards; counting them
        // is what makes a rung that quietly stopped checking visible in its own log, and it is the
        // property the deleted Split-K boundary runner's "ASSERT_CANARY_INTACT <n> bytes" carried.
        std::size_t intact = 0;
        for (unsigned repeat = 0; repeat < repeats; ++repeat) {
            intact = 0;
            launch_fused_decode_opt_sim(session.stream, &c, &b);
            camodel::check(aclrtSynchronizeStream(session.stream), "fused decode synchronize");
            output.download(); output.assert_guards(); workspace.download(); workspace.assert_guards();
            intact += 4 * camodel::kGuardBytes;  // the output's and the workspace's exterior guards
            if (c.splits == 1) {
                const auto* words = static_cast<const std::uint32_t*>(workspace.host_payload());
                for (unsigned i = 0; i < workspace.payload_bytes / 4; ++i)
                    if (words[i] != camodel::kCanary) throw std::runtime_error("inactive workspace changed");
                intact += workspace.payload_bytes;
            }
            const auto* actual = static_cast<const std::uint16_t*>(output.host_payload());
            for (unsigned h = 0; h < c.heads; ++h) for (unsigned d = 0; d < dim; ++d) {
                double numerator = 0, denominator = 0;
                // A planted hole makes the whole worker undescribable, so the kernel skips it and
                // the token's context is the neutral element -- not the hole-free subset of its
                // tiles, which is what the walking cursor used to produce.
                if (!empty && !planted_hole) {
                    for (unsigned tile = first_tile; tile < last_tile; ++tile) {
                        if (bt[tile / 2] < 0) continue;
                        const unsigned valid = !isolated && tile + 1 == tiles ? tail_rows : 64;
                        const unsigned col = d % 128;
                        const unsigned n = (tile + (d < 128 ? col : (col + 3) % 128)) % 8;
                        const double weight = varied ? std::exp(c.scale * (2.0 * (tile % 8) + 1)) : 1;
                        numerator += valid * weight * (2.0 * n + 1);
                        denominator += valid * weight;
                    }
                }
                const float expected = denominator ? static_cast<float>(numerator / denominator) : 0;
                // Compare the independently rounded oracle in the OUTPUT's own format, including
                // odd tile counts whose rational average is not representable.
                //
                // THE BUDGET IS THE SAME FOR BOTH FORMATS, and it is not relaxed for bf16. The
                // exact rungs demand 0 ULP of the output format; only --varied, whose Exp and
                // reductions carry their own documented tolerance, allows 1.
                //
                // bf16's 7 stored significand bits against fp16's 10 make the grid coarser, so
                // the question is whether these rungs' exact rationals still land where the
                // device's fp32 result rounds to the same word. They do, and with MORE room than
                // fp16 has, not less: tiles=1 and every isolated page canary are the integers
                // 1..15, exact in both formats, and the worst ragged rung sits 2074 fp32 ULPs
                // from the nearest bf16 rounding midpoint against 80 fp32 ULPs from the nearest
                // fp16 one -- a coarser grid puts the midpoints further apart, which is the right
                // way round. The fp16 leg of this same comparison already passes at 0 ULP on
                // CAModel, so bf16 is the easier of the two and a failure here would be a real
                // defect rather than a quantisation artefact.
                //
                // scripts/check_fused_decode_bf16_margins.py derives those margins per rung and
                // is the thing to re-run if either fixture's arithmetic changes.
                const auto bits = actual[h * dim + d];
                const std::uint16_t reference = out_format.encode(expected);
                const int ulp = std::abs(int(bits) - int(reference));
                if (ulp > (varied ? 1 : 0)) {
                    std::fprintf(stderr,
                        "mismatch %s h=%u d=%u got=%g expected=%g bits=%04x ref=%04x (%d ULP)\n",
                        out_format.name, h, d, out_format.decode(bits), expected, bits, reference, ulp);
                    throw std::runtime_error("exact context oracle failed");
                }
            }
            const auto* raw = static_cast<const std::uint32_t*>(output.host_payload());
            for (unsigned i = c.heads * dim / 2; i < output.payload_bytes / 4; ++i)
                if (raw[i] != camodel::kCanary) throw std::runtime_error("inactive output changed");
            intact += output.payload_bytes -
                static_cast<std::size_t>(c.heads) * dim * fd::element_bytes(c.element_type);
            for (unsigned i = 0; i < inputs.size(); ++i) {
                inputs[i]->download();
                // assert_guards() first, so a burst that ran off either end of an INPUT is named
                // as a guard-band hit rather than as an anonymous "input bytes changed".
                inputs[i]->assert_guards();
                if (inputs[i]->host != snapshots[i]) throw std::runtime_error("input bytes changed");
                intact += inputs[i]->host.size() * 4;
            }
        }
        std::printf("PASS fused_decode_opt dtype=%s tiles=%u page=%u tile=%u heads=%u splits=%u grid=%u repeats=%u elements=%u tail_rows=%u hole=%u comparison=%s inputs=unchanged guards=0x7FC0BEEF-intact ASSERT_CANARY_INTACT %zu bytes\n",
            out_format.name, tiles, c.probe_page, c.probe_tile, c.heads, c.splits, c.grid, repeats,
            c.heads * dim, tail_rows, hole,
            varied ? "1ulp_of_output_format" : "exact_0ulp_at_stored_word", intact);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL fused_decode_opt: %s\n", e.what()); return 1;
    }
}
