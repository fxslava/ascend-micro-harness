#pragma once
#include "turboquant_npu_common.hpp"
#include "../common/fused_decode_opt_fixture.hpp"
#include "turboquant_splitk_rigs.hpp"

namespace tq_splitk_rig {
struct FusedOptBuffer {
    static constexpr std::size_t guard = 64;
    tq_npu::Buffer allocation;
    std::size_t bytes;
    std::vector<std::uint8_t> snapshot;
    explicit FusedOptBuffer(std::size_t n)
        : allocation(fused_decode_opt_test::aligned(n) + 2 * guard),
          bytes(fused_decode_opt_test::aligned(n)) {}
    void* ptr() const { return static_cast<std::uint8_t*>(allocation.ptr) + guard; }
    template<class T> void Upload(const std::vector<T>& data)
    {
        tq_npu::require(data.size() * sizeof(T) <= bytes, "fused-opt upload overflow");
        snapshot.assign(allocation.bytes, 0xAA);
        std::memcpy(snapshot.data() + guard, data.data(), data.size() * sizeof(T));
        allocation.upload(snapshot);
    }
    void Reset() { allocation.fill_canary(); }
    void Poison()
    {
        std::vector<std::uint32_t> image(allocation.bytes / 4, 0xAAAAAAAAu);
        std::fill(image.begin() + guard / 4, image.begin() + (guard + bytes) / 4, 0x7FC0BEEFu);
        allocation.upload(image);
    }
    std::vector<std::uint8_t> Download() const
    {
        std::vector<std::uint8_t> image(allocation.bytes);
        allocation.download(image);
        for (std::size_t i = 0; i < guard; ++i)
            tq_npu::require(image[i] == 0xAA && image[guard + bytes + i] == 0xAA,
                            "fused-opt exterior guard changed");
        return image;
    }
    void CheckInput() const { tq_npu::require(Download() == snapshot, "fused-opt input changed"); }
    // Download() already requires both exterior guards; this names the buffer when it fails
    // for a buffer the kernel is expected to WRITE, where CheckInput would be wrong.
    void CheckGuardsOnly(const std::vector<std::uint8_t>& image, const char* what) const
    {
        for (std::size_t i = 0; i < guard; ++i)
            if (image[i] != 0xAA || image[guard + bytes + i] != 0xAA) {
                std::fprintf(stderr, "fused-opt guard band of %s clobbered at %zu\n", what, i);
                tq_npu::require(false, "fused-opt exterior guard changed");
            }
    }
};

// WHICH ENTRY POINT, AND WHAT THE QUERY BUFFER MEANS IN EACH.
//
// pre_rotate = true is the pre-rotated entry: the host uploads the rotated query into the
// query buffer and the kernel reads it. There is no output stage -- the context comes out in
// the rotated basis, and post_rotate must be 0.
//
// pre_rotate = false is the RAW-QUERY entry. The query buffer becomes an OUTPUT of the
// prologue: the kernel rotates the original-basis query into it and the decode then reads it
// from there. So the rig uploads a canary instead of the query, and Validate compares what
// came back against the host's own Pi image -- which is a check on the prologue alone,
// independent of the attention. post_rotate then selects the output stage:
//
//   0 kRotatedBasis  the context as the pre-rotated entry emits it
//   1 kUnrotated     Pi applied once more on egress
//   2 kGated         that, divided by 1 + exp(-gate)
//
// The fixture's raw query is built so that Pi(raw) IS the pre-rotated query, bit for bit
// (RawQueryBasis is symmetric and Pi^2 = I exactly), which is why stage 0 is held to the same
// attention oracle as the pre-rotated entry rather than to a looser one.
struct FusedOptMode {
    bool pre_rotate = true;
    unsigned post_rotate = 0;  // TurboQuantOutputStage; raw-query only
    unsigned cube_chunk = 0;   // prologue Cube chunk vectors, 0 or 16; raw-query only
};

struct FusedOptRig {
    // Stage 0 is exact arithmetic against the same oracle the pre-rotated entry answers to, so
    // it keeps the 1-ULP budget. Stages 1 and 2 put a 256-term fp32 rotation (and, at 2, a
    // division) between the oracle and the output, so their budget is wider -- and this is a
    // FIRST-RUN budget, not a measurement: the rig prints the worst ULP it actually saw, and
    // that number is what should replace this once the path has run on a model or a device.
    static constexpr int kUlpBudgetRotatedBasis = 1;
    static constexpr int kUlpBudgetRestaged = 8;

    const fused_decode_opt_test::Fixture& f;
    FusedOptMode mode;
    fused_decode_opt_test::RawQueryBasis basis;
    ascend_micro::fused_decode_opt::Launch config;
    FusedOptBuffer q, key, value, scales, table, lengths, modes, workspace, out;
    // Raw-query inputs. Always allocated (about 10 KB more) so one rig covers both entries;
    // uploaded and checked only when the raw-query entry is the one being launched.
    FusedOptBuffer raw, signs, rot_tables, h16, gate;
    // The host tiling descriptors. Declared after the raw-query inputs so the existing members
    // keep their construction order, and sized from `config`, which is already built.
    FusedOptBuffer tiling;
    int worst_ulp = 0;

    FusedOptRig(const fused_decode_opt_test::Fixture& fixture, int splits, FusedOptMode m = {})
        : f(fixture), mode(m), basis(),
          config(m.pre_rotate ? f.launch(splits) : f.launch_raw(splits, m.post_rotate, m.cube_chunk)),
          q(f.query.size() * 4),
          key(f.key.size()), value(f.value.size()), scales(f.scales.size() * 4),
          table(f.table.size() * 4), lengths(f.lengths.size() * 4), modes(32),
          workspace(std::size_t(f.requests) * config.heads * std::size_t(splits) * (256 + 16) * 4 + 4096),
          out(f.query.size() * 2 + 32),
          raw(f.query.size() * 2), signs(basis.signs.size() * 4),
          rot_tables(basis.tables().size() * 4), h16(basis.h16().size() * 2),
          gate(f.query.size() * 2),
          tiling(ascend_micro::fused_decode::decode_tiling_bytes_for(config))
    {
        tq_npu::require(mode.pre_rotate || mode.post_rotate <= 2,
                        "fused-opt: output stage must be 0, 1 or 2");
        tq_npu::require(mode.pre_rotate ? (mode.post_rotate == 0 && mode.cube_chunk == 0) : true,
                        "fused-opt: the pre-rotated entry has no output stage and no prologue");
        key.Upload(f.key); value.Upload(f.value); scales.Upload(f.scales);
        table.Upload(f.table); lengths.Upload(f.lengths); modes.Upload(std::vector<std::uint32_t>(8, 0));
        // THE TILING, BUILT ON THE HOST FROM THE SAME TWO ARRAYS THE DEVICE IS GIVEN. The block
        // table goes in as well, so the builder refuses a fixture whose live span holds an
        // unmapped page instead of handing the closed-form cursor a span it cannot walk.
        {
            namespace fd_tiling = ascend_micro::fused_decode;
            std::vector<std::uint32_t> image(
                fd_tiling::decode_tiling_words(config.tokens, config.splits, config.grid), 0);
            const auto status = fd_tiling::build_decode_tiling(
                fd_tiling::decode_tiling_request(config), f.lengths.data(), f.table.data(),
                image.data());
            tq_npu::require(status == fd_tiling::DecodeTilingStatus::kOk,
                            fd_tiling::decode_tiling_status_text(status));
            tiling.Upload(image);
        }
        if (mode.pre_rotate) {
            q.Upload(f.query);
        } else {
            raw.Upload(f.raw_query(basis));
            signs.Upload(basis.signs);
            rot_tables.Upload(basis.tables());
            h16.Upload(basis.h16());
            gate.Upload(f.gate_logits());
        }
    }

    void Reset()
    {
        workspace.Poison();
        out.Reset();
        // In raw-query mode the query buffer is written by the prologue, so it is reset too:
        // a prologue that skipped a vector then leaves a canary rather than last run's value.
        if (!mode.pre_rotate) q.Reset();
    }

    void Launch(aclrtStream stream, int)
    {
        const ascend_micro::fused_decode_opt::Buffers b{q.ptr(), key.ptr(), value.ptr(),
            scales.ptr(), table.ptr(), lengths.ptr(), modes.ptr(), workspace.ptr(), out.ptr(),
            tiling.ptr()};
        if (mode.pre_rotate) {
            launch_fused_decode_opt(stream, &config, &b);
            return;
        }
        const ascend_micro::fused_decode_opt::RawQueryBuffers r{raw.ptr(), signs.ptr(),
            rot_tables.ptr(), h16.ptr(), gate.ptr(), b};
        launch_fused_decode_opt_raw_query(stream, &config, &r);
    }

    AivRig::Census Validate(int)
    {
        // 1. The prologue, on its own: the rotated query the kernel wrote against the host's
        // own Pi image of the same raw query. Bit-exact is the claim -- every value in this
        // chain is an integer multiple of a power of two inside fp32's exact range.
        if (!mode.pre_rotate) {
            const auto rotated = q.Download();
            q.CheckGuardsOnly(rotated, "rotated query");
            for (std::size_t i = 0; i < f.query.size(); ++i) {
                float got;
                std::memcpy(&got, rotated.data() + FusedOptBuffer::guard + 4 * i, 4);
                if (got != f.query[i]) {
                    std::fprintf(stderr,
                        "fused-opt prologue mismatch element=%zu got=%.9g expected=%.9g\n",
                        i, double(got), double(f.query[i]));
                    throw std::runtime_error("fused-opt query prologue disagrees with Pi(raw query)");
                }
            }
        }

        // 2. The context, against the oracle for this output stage.
        const auto image = out.Download();
        const int budget = mode.post_rotate == 0 ? kUlpBudgetRotatedBasis : kUlpBudgetRestaged;
        const std::size_t rows = f.query.size() / 256;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::vector<float> want = f.expected_row(basis, row, mode.post_rotate);
            for (std::size_t c = 0; c < 256; ++c) {
                const std::size_t i = row * 256 + c;
                std::uint16_t bits;
                std::memcpy(&bits, image.data() + FusedOptBuffer::guard + 2 * i, 2);
                const auto expected = ascend_micro::sim::float_to_half_rne(want[c]);
                // Both are positive finite normals at stage 0; at stages 1 and 2 the sign can
                // differ, so compare the rounded images rather than the raw bit distance.
                const int ulp = std::abs(int(bits) - int(expected));
                if (ulp > worst_ulp) worst_ulp = ulp;
                if (ulp > budget) {
                    std::fprintf(stderr,
                        "fused-opt mismatch element=%zu stage=%u got=%g expected=%g bits=%04x "
                        "ref=%04x (%d ULP over %d)\n", i, mode.post_rotate,
                        ascend_micro::sim::half_to_float(bits), double(want[c]), bits, expected,
                        ulp, budget);
                    throw std::runtime_error("fused-opt CPU attention reference failed");
                }
            }
        }
        for (std::size_t i = f.query.size() * 2; i < out.bytes; ++i)
            tq_npu::require(image[FusedOptBuffer::guard + i] == 0xAA,
                            "fused-opt inactive output changed");

        // The measured worst ULP, said out loud on every raw-query launch. kUlpBudgetRestaged
        // is a placeholder until a run reports what the number actually is, and a budget nobody
        // can see the margin on is a budget nobody will ever tighten. stderr, so it cannot land
        // in the middle of the benchmark's table on stdout.
        if (!mode.pre_rotate)
            std::fprintf(stderr,
                         "fused-opt basis: raw-query entry, prologue %s, output stage %u, "
                         "worst fp16 ULP %d against a budget of %d\n",
                         mode.cube_chunk ? "Cube chunk" : "vector cores", mode.post_rotate,
                         worst_ulp, budget);

        const auto partials = workspace.Download();
        if (config.splits > 1) {
            const auto floats = std::size_t(f.requests) * config.heads * config.splits * (256 + 16);
            for (std::size_t i = 0; i < floats; ++i) {
                float v;
                std::memcpy(&v, partials.data() + FusedOptBuffer::guard + i * 4, 4);
                tq_npu::require(std::isfinite(v), "fused-opt partial left unwritten or nonfinite");
            }
        }
        key.CheckInput(); value.CheckInput(); scales.CheckInput();
        table.CheckInput(); lengths.CheckInput(); modes.CheckInput(); tiling.CheckInput();
        if (mode.pre_rotate) {
            q.CheckInput();
        } else {
            // The prologue's five inputs must come back untouched; the query buffer is an
            // output here and was checked against Pi(raw) above.
            raw.CheckInput(); signs.CheckInput(); rot_tables.CheckInput();
            h16.CheckInput(); gate.CheckInput();
        }
        return {}; // This source ABI exports no worker timestamps.
    }
};
} // namespace tq_splitk_rig
