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
};

struct FusedOptRig {
    const fused_decode_opt_test::Fixture& f;
    ascend_micro::fused_decode_opt::Launch config;
    FusedOptBuffer q, key, value, scales, table, lengths, modes, workspace, out;
    FusedOptRig(const fused_decode_opt_test::Fixture& fixture, int splits)
        : f(fixture), config(f.launch(splits)), q(f.query.size() * 4),
          key(f.key.size()), value(f.value.size()), scales(f.scales.size() * 4),
          table(f.table.size() * 4), lengths(f.lengths.size() * 4), modes(32),
          workspace(std::size_t(f.requests) * config.heads * splits * (256 + 16) * 4 + 4096),
          out(f.query.size() * 2 + 32)
    {
        q.Upload(f.query); key.Upload(f.key); value.Upload(f.value); scales.Upload(f.scales);
        table.Upload(f.table); lengths.Upload(f.lengths); modes.Upload(std::vector<std::uint32_t>(8, 0));
    }
    void Reset() { workspace.Poison(); out.Reset(); }
    void Launch(aclrtStream stream, int)
    {
        const ascend_micro::fused_decode_opt::Buffers b{q.ptr(), key.ptr(), value.ptr(),
            scales.ptr(), table.ptr(), lengths.ptr(), modes.ptr(), workspace.ptr(), out.ptr()};
        launch_fused_decode_opt(stream, &config, &b);
    }
    AivRig::Census Validate(int)
    {
        const auto image = out.Download();
        for (std::size_t i = 0; i < f.expected.size(); ++i) {
            std::uint16_t bits;
            std::memcpy(&bits, image.data() + FusedOptBuffer::guard + 2 * i, 2);
            const auto expected = fused_decode_opt_test::reference_half(f.expected[i]);
            // Positive finite fixture: FP16 bit distance is ULP distance.
            if (std::abs(int(bits) - int(expected)) > 1) {
                std::fprintf(stderr, "fused-opt mismatch element=%zu got=%g expected=%g bits=%04x ref=%04x\n",
                    i, ascend_micro::sim::half_to_float(bits), f.expected[i], bits, expected);
                throw std::runtime_error("fused-opt CPU attention reference failed");
            }
        }
        for (std::size_t i = f.query.size() * 2; i < out.bytes; ++i)
            tq_npu::require(image[FusedOptBuffer::guard + i] == 0xAA,
                            "fused-opt inactive output changed");
        const auto partials = workspace.Download();
        if (config.splits > 1) {
            const auto floats = std::size_t(f.requests) * config.heads * config.splits * (256 + 16);
            for (std::size_t i = 0; i < floats; ++i) {
                float v;
                std::memcpy(&v, partials.data() + FusedOptBuffer::guard + i * 4, 4);
                tq_npu::require(std::isfinite(v), "fused-opt partial left unwritten or nonfinite");
            }
        }
        q.CheckInput(); key.CheckInput(); value.CheckInput(); scales.CheckInput();
        table.CheckInput(); lengths.CheckInput(); modes.CheckInput();
        return {}; // This source ABI exports no worker timestamps.
    }
};
} // namespace tq_splitk_rig
