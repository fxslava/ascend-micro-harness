#pragma once
#include "../../include/turboquant_fused_decode_opt.hpp"
#include "../../include/sim_micro_testbed.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace fused_decode_opt_test {
namespace fd = ascend_micro::fused_decode_opt;
namespace tq = vllm_ascend::turboquant;
inline std::size_t aligned(std::size_t n) { return (n + 31) / 32 * 32; }
// Positive normal results are in [1,15]. Round the FP32 significand before
// rebiasing, so a mantissa carry correctly increments the FP16 exponent.
inline std::uint16_t reference_half(float value)
{
    if (!(value >= 1 && value <= 15)) throw std::runtime_error("oracle result outside fixture range");
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    return static_cast<std::uint16_t>(((bits + 0xfffu + ((bits >> 13) & 1u)) >> 13) - (112u << 10));
}

// FP32 pre-rotated GQA queries, source-format KV4 NZ cache, FP16 context.
// Tile scores vary by request, KV head and query head. Values vary by tile
// and channel. A scalar FP64 attention sum supplies an independent oracle.
struct Fixture {
    int context, requests, kv_heads, group_heads, page_tokens, pages, physical_pages;
    std::vector<float> query, scales, expected;
    std::vector<std::uint8_t> key, value;
    std::vector<std::int32_t> table, lengths;
    Fixture(int n, int b, int h, int g, int block)
        : context(n), requests(b), kv_heads(h), group_heads(g), page_tokens(block),
          pages(block > 0 ? n / block : 0), physical_pages(2 * b * pages + 2)
    {
        if (n <= 0 || n > 16384 || (block != 64 && block != 128) || n % block || b < 1 || b > 16 ||
            h < 1 || h > 4 || g < 1 || g > 16)
            throw std::invalid_argument("invalid fused-opt fixture geometry");
        query.resize(std::size_t(b) * h * g * 256);
        expected.resize(query.size());
        key.resize(std::size_t(physical_pages) * block * h * 128);
        value.resize(key.size());
        for (std::size_t i = 0; i < key.size(); ++i)
            key[i] = value[i] = static_cast<std::uint8_t>(0x7FC0BEEFu >> ((i % 4) * 8));
        float poison; const std::uint32_t word = 0x7FC0BEEF;
        std::memcpy(&poison, &word, 4);
        scales.assign(std::size_t(physical_pages) * block * 8, poison);
        table.assign(aligned(std::size_t(b) * pages * 4) / 4, -1);
        lengths.assign(aligned(b * 4) / 4, 0);
        const float scale = tq::TurboQuantModeTraits<tq::TurboQuantMode::KV4_FP8>::kGain * 2;
        for (int request = 0; request < b; ++request) {
            lengths[request] = n;
            for (int page = 0; page < pages; ++page) {
                const int physical = 2 * (request * pages + page) + 1;
                table[request * pages + page] = physical;
                for (int row = 0; row < block; ++row) for (int head = 0; head < h; ++head) {
                    const int tile = (page * block + row) / 64;
                    const unsigned kn = key_level(request, head, tile);
                    const auto si = (std::size_t(physical) * block + row) * 8;
                    scales[si + head] = scales[si + h + head] = scale;
                    for (int col = 0; col < 128; ++col) {
                        const auto off = std::size_t(physical) * block * h * 128 +
                            tq::NzTiledPackedByte(row, head, col, h, 128);
                        key[off] = static_cast<std::uint8_t>(kn | (kn << 4));
                        value[off] = static_cast<std::uint8_t>(value_level(request, head, tile, col) |
                            (value_level(request, head, tile, col + 128) << 4));
                    }
                }
            }
            for (int qh = 0; qh < h * g; ++qh) {
                const int head = qh / g, coefficient = 1 + qh % 3;
                const auto base = (std::size_t(request) * h * g + qh) * 256;
                std::fill(query.begin() + base, query.begin() + base + 256, coefficient / 256.0f);
                for (int col = 0; col < 256; ++col) {
                    double numerator = 0, denominator = 0;
                    for (int tile = 0; tile < n / 64; ++tile) {
                        const double score = (2 * key_level(request, head, tile) + 1) * coefficient / 16.0;
                        const double weight = std::exp(score);
                        numerator += weight * (2 * value_level(request, head, tile, col) + 1);
                        denominator += weight;
                    }
                    expected[base + col] = static_cast<float>(numerator / denominator);
                }
            }
        }
    }
    static unsigned key_level(int request, int head, int tile)
    { return (tile + 2 * request + 3 * head) % 8; }
    static unsigned value_level(int request, int head, int tile, int channel)
    {
        const int col = channel < 128 ? channel : (channel - 128 + 3) % 128;
        return (tile + 3 * request + 5 * head + col) % 8;
    }
    fd::Launch launch(int splits) const
    {
        if (splits < 1 || splits > 8) throw std::invalid_argument("fused-opt splits must be 1..8");
        fd::Launch c;
        c.tokens = requests; c.kv_heads = kv_heads; c.heads = kv_heads * group_heads;
        c.block_size = page_tokens; c.max_blocks = pages; c.splits = splits;
        c.heads_per_task = group_heads; c.fused_context_limit = 0;
        const unsigned tasks = requests * kv_heads * splits;
        c.tasks_per_block = (tasks + 31) / 32;
        c.grid = (tasks + c.tasks_per_block - 1) / c.tasks_per_block;
        c.reduce_tasks_per_block = (requests * c.heads + c.grid - 1) / c.grid;
        if (!fd::valid_launch(c.grid, c.tokens, c.heads, c.kv_heads, c.head_size,
                c.block_size, c.max_blocks, c.splits, c.heads_per_task,
                c.tasks_per_block, c.reduce_tasks_per_block))
            throw std::invalid_argument("unsupported fused-opt launch");
        return c;
    }
    double cache_bytes() const
    { return double(requests) * context * (kv_heads * 256 + 8 * 4); }
};
} // namespace fused_decode_opt_test
