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

// ---------------------------------------------------------------- the raw-query Pi basis
//
// The pre-rotated entry takes `queryRot` already in the rotated basis. The RAW-query entry
// takes the query in the original basis plus the basis tables, rotates it once in a prologue
// (TurboQuantQueryBasis::Rotate) and can rotate the context back on egress
// (FinishHeads). This builds every input that path needs, and the host reference for it.
//
// THE BASIS. TurboQuantCodec4::ApplyPi is `x *= piSigns; FWHT(x); x *= piSigns`, and its FWHT
// is the unnormalised Walsh-Hadamard butterfly followed by one `Muls(invSqrtLen)`. So
//
//     Pi = (1 / sqrt(D)) . diag(s) . H . diag(s),   H[i][j] = (-1)^popcount(i & j)
//
// with H symmetric and H^2 = D.I, which makes Pi symmetric and Pi^2 = I EXACTLY -- not
// approximately. That is what lets the raw-query fixture reuse the pre-rotated oracle: the raw
// query whose rotation is a given target IS Pi(target).
//
// THE TABLE POOL. `rotTables` is the codec's whole constant pool, which TurboQuantCodec<4>
// ::Init copies from GM and carves up in this order (vecLen = 256, batchRows = 1, so
// ConstTableWords = 7*256 + 2*256 + 16 = 2320 words):
//
//     stage 0 sign_ (256 float) | stage 0 xorOffset_ (256 uint32) | stage 1 ... | stage 2 ...
//     evenOffset_ (128 uint32) | oddOffset_ (128) | expandOffset_ (256) | oddSelect_ (256
//     float) | centroid_ (16 float)
//
// ApplyPi reads ONLY the three early stages: `Gather(swap_, x, xorOffset_[stage], 0, n)` then
// `Mul(tmp, x, sign_[stage], n)` then `Add(x, swap_, tmp, n)`, which is
// x'[i] = x[i ^ stride] + sign[i] * x[i] -- the butterfly, with sign[i] = +1 when
// (i & stride) == 0 and -1 otherwise, and the gather index a BYTE offset (kGatherSrcBase is 0,
// and the vector service's own gather indices are byte offsets too). The rest of the pool
// belongs to Quantize4Bit/Dequantize4Bit, which the basis never calls, so it is left zero --
// zero is also an in-bounds byte offset, so a future caller that did gather with it would read
// element 0 rather than out of bounds.
struct RawQueryBasis {
    static constexpr int kEarlyStages = 3;  // TurboQuantCodec<4>::kEarlyStages
    static constexpr int kLevels = 16;      // 1 << BITS
    static constexpr int kPackFactor = 2;   // 8 / BITS
    static constexpr int kH16 = 16;

    int dim;
    std::vector<float> signs;

    explicit RawQueryBasis(int d = static_cast<int>(fd::kHeadSize))
        : dim(d), signs(static_cast<std::size_t>(d), 0.0f)
    {
        if (d <= 0 || (d & (d - 1)) != 0) throw std::invalid_argument("Pi needs a power-of-two D");
        for (int i = 0; i < d; ++i) signs[std::size_t(i)] = sign_at(i);
    }

    // Deterministic, so the fixture, the oracle and the uploaded table are one function.
    static float sign_at(int channel)
    {
        const unsigned x = static_cast<unsigned>(channel);
        return ((x * 2654435761u) >> 13) & 1u ? -1.0f : 1.0f;
    }
    static float inv_sqrt_len(int d)
    {
        float r = 1.0f;
        for (int n = d; n > 1; n >>= 2) r *= 0.5f;  // exact: D is a power of four here
        return r;
    }

    // Pi applied to one vector, in double, stage for stage as ApplyPi would.
    std::vector<float> apply(const float* x) const
    {
        std::vector<double> v(static_cast<std::size_t>(dim), 0.0);
        for (int i = 0; i < dim; ++i) v[std::size_t(i)] = double(x[i]) * signs[std::size_t(i)];
        for (int stride = 1; stride < dim; stride <<= 1)
            for (int i = 0; i < dim; ++i)
                if ((i & stride) == 0) {
                    const double a = v[std::size_t(i)], b = v[std::size_t(i ^ stride)];
                    v[std::size_t(i)] = a + b;
                    v[std::size_t(i ^ stride)] = a - b;
                }
        const double scale = double(inv_sqrt_len(dim));
        std::vector<float> out(static_cast<std::size_t>(dim), 0.0f);
        for (int i = 0; i < dim; ++i)
            out[std::size_t(i)] = static_cast<float>(v[std::size_t(i)] * signs[std::size_t(i)] * scale);
        return out;
    }

    // The codec's GM constant pool, exactly as Init unpacks it.
    std::vector<std::int32_t> tables() const
    {
        const std::size_t words = 9u * static_cast<std::size_t>(dim) + kLevels;
        std::vector<std::int32_t> pool(words, 0);
        std::size_t offset = 0;
        for (int stage = 0; stage < kEarlyStages; ++stage) {
            const int stride = 1 << stage;
            for (int i = 0; i < dim; ++i) {
                const float sign = (i & stride) == 0 ? 1.0f : -1.0f;
                std::int32_t bits;
                std::memcpy(&bits, &sign, 4);
                pool[offset + std::size_t(i)] = bits;
            }
            offset += std::size_t(dim);
            for (int i = 0; i < dim; ++i)
                pool[offset + std::size_t(i)] =
                    static_cast<std::int32_t>(std::uint32_t(i ^ stride) * sizeof(float));
            offset += std::size_t(dim);
        }
        if (offset != 2u * static_cast<std::size_t>(kEarlyStages) * static_cast<std::size_t>(dim))
            throw std::runtime_error("the early-stage pool layout drifted");
        return pool;
    }

    // The 16x16 Hadamard in fp16, which only the Cube prologue chunk reads.
    std::vector<std::uint16_t> h16() const
    {
        std::vector<std::uint16_t> out(static_cast<std::size_t>(kH16) * kH16, 0);
        for (int i = 0; i < kH16; ++i)
            for (int j = 0; j < kH16; ++j) {
                unsigned bits = static_cast<unsigned>(i & j), parity = 0;
                while (bits) { parity ^= bits & 1u; bits >>= 1; }
                out[std::size_t(i) * kH16 + j] = parity ? 0xbc00 : 0x3c00;  // -1.0h, +1.0h
            }
        return out;
    }
};

// FP32 pre-rotated GQA queries, source-format KV4 NZ cache, FP16 context.
// Tile scores vary by request, KV head and query head. Values vary by tile
// and channel. A scalar FP64 attention sum supplies an independent oracle.
//
// RAGGED CONTEXTS. `context` need not be a multiple of `page_tokens` or of the 64-row Cube
// tile: `pages` is the ceiling, the last page holds `context - (pages - 1) * page_tokens`
// live tokens, and the last live tile holds `context % 64` of them when that is nonzero. The
// cache is still filled over EVERY row of every mapped page, tail rows included, which is what
// makes the kernel's masking observable -- a tail row it fails to mask carries the same key
// level and the same value levels as the live rows of its own tile, so including it weights
// that tile by 64 rows instead of by its true count and the oracle below disagrees.
//
// The oracle weights each tile by its live row count, which is the same thing the CAModel
// runner's own oracle does (`numerator += valid * weight * ...` in
// tests/sim/kernel/test_camodel_fused_decode_opt.cpp). All 64 rows of a tile share one key
// level, so a tile of v live rows contributes v * exp(score) to both sums; at v = 64 that is
// an exact power-of-two scaling of both, so an ALIGNED context's expected values are
// bit-identical to the unweighted form this replaced. tests/cpu/test_fused_decode_opt_fixture
// pins that.
struct Fixture {
    int context, requests, kv_heads, group_heads, page_tokens, pages, physical_pages;
    std::vector<float> query, scales, expected;
    std::vector<std::uint8_t> key, value;
    std::vector<std::int32_t> table, lengths;
    Fixture(int n, int b, int h, int g, int block)
        : context(n), requests(b), kv_heads(h), group_heads(g), page_tokens(block),
          pages(block > 0 && n > 0 ? (n + block - 1) / block : 0), physical_pages(2 * b * pages + 2)
    {
        // n is no longer required to divide the page: see RAGGED CONTEXTS above. The page size
        // still has to be 64 or 128, because that is what valid_launch admits and what the
        // NZ-tiled ingest is built on.
        if (n <= 0 || n > 16384 || (block != 64 && block != 128) || b < 1 || b > 16 ||
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
                    for (int tile = 0; tile < tiles(); ++tile) {
                        const int live = valid_rows(tile);
                        const double score = (2 * key_level(request, head, tile) + 1) * coefficient / 16.0;
                        const double weight = live * std::exp(score);
                        numerator += weight * (2 * value_level(request, head, tile, col) + 1);
                        denominator += weight;
                    }
                    expected[base + col] = static_cast<float>(numerator / denominator);
                }
            }
        }
    }
    // 64-row Cube tiles over the live context, and the live rows of one of them. The last tile
    // is short exactly when context % 64 != 0; every other tile is full, including the ones in
    // a page that is itself partially live.
    int tiles() const { return (context + 63) / 64; }
    int valid_rows(int tile) const
    {
        const int rest = context - 64 * tile;
        return rest < 64 ? rest : 64;
    }
    // The live tokens of one logical page, which is page_tokens for every page but the last.
    int page_rows(int page) const
    {
        const int rest = context - page_tokens * page;
        return rest < page_tokens ? rest : page_tokens;
    }
    static unsigned key_level(int request, int head, int tile)
    { return (tile + 2 * request + 3 * head) % 8; }
    static unsigned value_level(int request, int head, int tile, int channel)
    {
        const int col = channel < 128 ? channel : (channel - 128 + 3) % 128;
        return (tile + 3 * request + 5 * head + col) % 8;
    }
    // The raw-query inputs. `raw_query` is the ORIGINAL-basis query whose rotation is the
    // pre-rotated `query` above: Pi is an involution, so that is simply Pi(query). It is
    // fp16-exact and this throws rather than rounding, because the whole point of building it
    // this way is that the device's prologue reproduces `query` bit for bit and the attention
    // oracle therefore still applies at output stage 0.
    std::vector<std::uint16_t> raw_query(const RawQueryBasis& basis) const
    {
        if (basis.dim != static_cast<int>(fd::kHeadSize))
            throw std::invalid_argument("the raw-query basis must match the head size");
        std::vector<std::uint16_t> out(query.size(), 0);
        const std::size_t rows = query.size() / fd::kHeadSize;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::vector<float> image = basis.apply(query.data() + row * fd::kHeadSize);
            for (std::size_t c = 0; c < fd::kHeadSize; ++c) {
                const std::uint16_t bits = ascend_micro::sim::float_to_half_rne(image[c]);
                if (ascend_micro::sim::half_to_float(bits) != image[c])
                    throw std::runtime_error("the raw query is not fp16-exact");
                out[row * fd::kHeadSize + c] = bits;
            }
        }
        return out;
    }

    // Gate logits for TurboQuantOutputStage::kGated, fp16 and exactly representable. Spread
    // over [-2, 2] in quarters so the sigmoid is neither saturated nor constant, and nowhere
    // near kGateLogitFloor, which is a clamp this fixture is not trying to exercise.
    static float gate_value(std::size_t index)
    { return static_cast<float>(static_cast<int>(index % 17) - 8) * 0.25f; }
    std::vector<std::uint16_t> gate_logits() const
    {
        std::vector<std::uint16_t> out(query.size(), 0);
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = ascend_micro::sim::float_to_half_rne(gate_value(i));
        return out;
    }

    // The expected output for one (token, query head) row under a given output stage.
    //   0 kRotatedBasis: the context as the pre-rotated entry emits it
    //   1 kUnrotated:    Pi applied once more
    //   2 kGated:        that, divided by 1 + exp(-gate)
    std::vector<float> expected_row(const RawQueryBasis& basis, std::size_t row,
                                    unsigned output_stage) const
    {
        const float* source = expected.data() + row * fd::kHeadSize;
        if (output_stage == 0) return std::vector<float>(source, source + fd::kHeadSize);
        std::vector<float> out = basis.apply(source);
        if (output_stage == 1) return out;
        for (std::size_t c = 0; c < fd::kHeadSize; ++c) {
            const float g = gate_value(row * fd::kHeadSize + c);
            out[c] = static_cast<float>(static_cast<double>(out[c]) /
                                        (1.0 + std::exp(-static_cast<double>(g))));
        }
        return out;
    }

    // The raw-query launch. Same geometry as launch(), plus the prologue share and the stage.
    fd::Launch launch_raw(int splits, unsigned output_stage, unsigned cube_chunk_vectors) const
    {
        if (output_stage > 2) throw std::invalid_argument("output stage must be 0, 1 or 2");
        if (cube_chunk_vectors != 0 && cube_chunk_vectors != 16)
            throw std::invalid_argument("the prologue Cube chunk is 0 or 16 vectors");
        fd::Launch c = launch(splits);
        const unsigned vectors = static_cast<unsigned>(requests) * c.heads;
        c.prologue_vectors_per_block = (vectors + c.grid - 1) / c.grid;
        c.prologue_cube_chunk_vectors = cube_chunk_vectors;
        c.output_stage = output_stage;
        return c;
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
