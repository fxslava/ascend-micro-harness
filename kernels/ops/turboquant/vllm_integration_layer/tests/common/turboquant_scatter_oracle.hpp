#pragma once
// The golden oracle for turboquant_reshape_scatter_aiv (roadmap kernel No.1):
// sub-strip BF16 ingestion with a read-modify-write value plane.
//
// Why this is a separate header from tests/common/turboquant_oracle.hpp
// --------------------------------------------------------------------
// That header is the oracle for the ALIGNED prefill writer: INT8 activations,
// a Cube rotation against resident 32x32 tables, whole 16-token strips, and a
// cache whose pages are written in full. This operator shares none of those:
// BF16 activations, a Vector butterfly that reads `pi_signs` directly, 1..16
// tokens, and a value plane that must preserve every lane it does not own.
// Extending the A8 header in place would have put two contracts behind one set
// of names, in the one file whose correctness the A8 and BF16 device
// validations both rest on.
//
// What is NOT duplicated: the quantiser. `rint_even_shift`, the nibble range
// and the low-nibble-first packing come from tq_oracle, and the rotation comes
// from rot_oracle -- so the only thing defined twice anywhere in this family
// is nothing.
//
// Why bit equality is demandable
// ------------------------------
// The fixture is integer-valued in BF16. The rotation is adds and subtracts of
// integers bounded by 256 * 128 * 2 = 2^16, exact in FP32; absmax is an
// integer; the scale is the power of two 2^(e-2) extracted by integer ops on
// the FP32 bit pattern, so y * 2^(2-e) is exact and the single
// round-to-nearest-even is the only rounding in the path. Nothing depends on a
// hardware divide, reciprocal or exponential, so nibbles and scales compare
// bit for bit.
//
// The layout contract (RFC section 3.1, normative; R = 16 is the strip height
// and the only compile-time shape constant inside the writer):
//
//   key   nibble = ((block*H + head)*S + token)*d + channel
//   value nibble = ((block*H + head)*S)*d
//                  + (token / R)*(R*d) + channel*R + (token % R)
//   scale index  = ((kv*P + block)*H + head)*S + token        kv 0 key, 1 value
//   nibbles       two per byte, low nibble first, signed INT4 in [-8, 7]

#include "rotate_butterfly_oracle.hpp"
#include "turboquant_oracle.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace tq_scatter {

constexpr int kStripHeight = 16;  // R
constexpr int kPlanes = 2;        // 0 = key, 1 = value
constexpr int kNibbleMin = -8;
constexpr int kNibbleMax = 7;

// ---------------------------------------------------------------- bfloat16
inline float bf16_to_float(std::uint16_t raw)
{
    const std::uint32_t bits = static_cast<std::uint32_t>(raw) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline std::uint16_t float_to_bf16_rne(float value)
{
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x7fffffu))
        return static_cast<std::uint16_t>((bits >> 16) | 0x40u);
    return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// A BF16 holds 8 mantissa bits, so every integer of magnitude <= 256 is exact
// and the fixture below stays inside that.
inline bool bf16_is_exact(float value)
{
    return bf16_to_float(float_to_bf16_rne(value)) == value;
}

// ---------------------------------------------------------------- geometry
struct Shape {
    int num_requests = 1;
    int num_kv_heads = 4;
    int head_size = 256;
    int block_size = 16;  // S, a multiple of kStripHeight
    int num_blocks = 8;   // P

    int strips_per_block() const { return block_size / kStripHeight; }
    float inv_sqrt_len() const { return 1.0f / std::sqrt(static_cast<float>(head_size)); }
    std::size_t cache_bytes() const
    {
        return static_cast<std::size_t>(num_blocks) * num_kv_heads * block_size * head_size / 2;
    }
    std::size_t scale_floats() const
    {
        return static_cast<std::size_t>(kPlanes) * num_blocks * num_kv_heads * block_size;
    }
    std::size_t strip_bytes() const
    {
        return static_cast<std::size_t>(kStripHeight) * head_size / 2;  // 2048
    }
    bool valid() const
    {
        return head_size == rot_oracle::kChannels && block_size % kStripHeight == 0 &&
               num_blocks > 0 && num_kv_heads > 0 && num_requests > 0;
    }
};

// ------------------------------------------------------------ index maps
inline std::size_t key_nibble(const Shape& s, int block, int head, int token, int channel)
{
    return (static_cast<std::size_t>(block * s.num_kv_heads + head) * s.block_size + token) *
               s.head_size +
           channel;
}

inline std::size_t value_nibble(const Shape& s, int block, int head, int token, int channel)
{
    const std::size_t page =
        static_cast<std::size_t>(block * s.num_kv_heads + head) * s.block_size * s.head_size;
    return page +
           static_cast<std::size_t>(token / kStripHeight) *
               (static_cast<std::size_t>(kStripHeight) * s.head_size) +
           static_cast<std::size_t>(channel) * kStripHeight + (token % kStripHeight);
}

inline std::size_t scale_index(const Shape& s, int plane, int block, int head, int token)
{
    return ((static_cast<std::size_t>(plane) * s.num_blocks + block) * s.num_kv_heads + head) *
               s.block_size +
           token;
}

// The byte offset of the (block, head, strip) RMW window: 2048 contiguous bytes.
inline std::size_t strip_byte_offset(const Shape& s, int block, int head, int strip)
{
    return (static_cast<std::size_t>(block * s.num_kv_heads + head) * s.block_size *
                s.head_size +
            static_cast<std::size_t>(strip) * kStripHeight * s.head_size) /
           2;
}

// -------------------------------------------------------------- the fixture
// Integer-valued BF16, distinct per (plane, head, token, channel) and per
// request, so a work item that read another item's rows cannot produce the
// right answer. `zero_token` makes one token's K and V rows identically zero,
// which is the only way to reach the absmax == 0 branch -- the case that never
// arises with an integer fixture and arises immediately in production.
inline float activation(int request, int plane, int head, int token, int channel,
                        int zero_token)
{
    if (token == zero_token) return 0.0f;
    const int rank = plane * 8 + head;
    const std::uint32_t h = tq_oracle::mix(
        0x51ED2701u ^ (static_cast<std::uint32_t>(request) * 0x9E3779B9u) ^
        (static_cast<std::uint32_t>(rank) * 0x85EBCA6Bu) ^
        (static_cast<std::uint32_t>(token) * 0xC2B2AE35u) ^
        (static_cast<std::uint32_t>(channel) * 0x27D4EB2Du));
    const int n = static_cast<int>(h % 25u) - 12;
    return static_cast<float>(n * (1 + rank % 3));
}

// ------------------------------------------------------- the per-row reference
struct RowRef {
    std::vector<float> rotated;  // head_size, Pi . x with no 1/16
    std::vector<int> q;          // head_size, the INT4 codes
    float absmax = 0.0f;
    int exponent = 0;
    float scale = 0.0f;
    bool zero_row = false;
    bool clamped = false;
};

// One (token, head, plane) row: the rotation and the quantiser.
//
// The quantiser, restated so the device's bit arithmetic has something to be
// compared against rather than described by. For 2^e <= absmax < 2^(e+1):
//   exp_bits = bits(absmax) & 0x7F800000  -> 2^e
//   pow      = 2^(e-2)                    -> the stored scale, times invSqrtLen
//   inv      = 2^(2-e)                    -> the exact reciprocal
//   q        = min(rint_even(y * inv), 7)      clamped from above only
// and absmax == 0 takes the explicit branch scale = 1.0f, q = 0 (RFC I15):
// the exponent field of FP32 zero is 0, so 2^(e-2) would be built from a
// borrowed exponent and the stored scale would be garbage.
inline RowRef quantise_row(const Shape& shape, const std::vector<float>& x,
                           const std::vector<float>& signs)
{
    RowRef out;
    out.rotated = rot_oracle::staged(rot_oracle::Direction::Ingress, x.data(), signs.data());
    out.q.assign(static_cast<std::size_t>(shape.head_size), 0);

    float absmax = 0.0f;
    for (float value : out.rotated) absmax = std::fmax(absmax, std::fabs(value));
    out.absmax = absmax;
    if (absmax == 0.0f) {
        out.zero_row = true;
        out.scale = 1.0f;
        out.exponent = 0;
        return out;
    }
    const int e = std::ilogb(absmax);
    out.exponent = e;
    out.scale = std::ldexp(1.0f, e - 2) * shape.inv_sqrt_len();
    for (int c = 0; c < shape.head_size; ++c) {
        // y * 2^(2-e) is exact: y is an integer of magnitude below 2^17 and
        // the factor is a power of two, so nearbyint (FE_TONEAREST, ties to
        // even) is the single rounding the whole path performs.
        const double scaled = std::ldexp(static_cast<double>(out.rotated[c]), 2 - e);
        long long v = static_cast<long long>(std::nearbyint(scaled));
        if (v > kNibbleMax) {
            v = kNibbleMax;
            out.clamped = true;
        }
        out.q[static_cast<std::size_t>(c)] = static_cast<int>(v);
    }
    return out;
}

// ---------------------------------------------------------- packing helpers
inline void write_nibble(std::vector<std::int8_t>& cache, std::size_t nibble, int value)
{
    const auto nib = static_cast<std::uint8_t>(value & 0xF);
    auto& byte = reinterpret_cast<std::uint8_t&>(cache[nibble / 2]);
    if (nibble % 2 == 0)
        byte = static_cast<std::uint8_t>((byte & 0xF0u) | nib);
    else
        byte = static_cast<std::uint8_t>((byte & 0x0Fu) | static_cast<std::uint8_t>(nib << 4));
}

inline int read_nibble(const std::int8_t* cache, std::size_t nibble)
{
    const auto byte = static_cast<std::uint8_t>(cache[nibble / 2]);
    const int raw = (nibble % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
    return raw >= 8 ? raw - 16 : raw;
}

// -------------------------------------------------------------- the launch
// One token of one request: where it goes and what it carries.
struct TokenPlan {
    int request = 0;
    int token_index = 0;  // index into the flat token list / slotMapping
    int slot = 0;         // slotMapping value; < 0 means a padded token
    int block = 0;
    int token_in_block = 0;
    int strip = 0;
};

inline TokenPlan plan_token(const Shape& shape, int request, int token_index, int slot)
{
    TokenPlan plan;
    plan.request = request;
    plan.token_index = token_index;
    plan.slot = slot;
    if (slot < 0) return plan;
    plan.block = slot / shape.block_size;
    plan.token_in_block = slot % shape.block_size;
    plan.strip = plan.token_in_block / kStripHeight;
    return plan;
}

// The whole reference. `prior_key`/`prior_value`/`prior_scale` are the cache
// images BEFORE the launch, so the value plane's read-modify-write can be
// checked for what it PRESERVED as well as for what it wrote -- which is the
// property a zero-fill reference could not express. They are copied, mutated
// where the launch owns a lane, and returned.
struct Reference {
    Shape shape;
    std::vector<std::int8_t> key_cache;
    std::vector<std::int8_t> value_cache;
    std::vector<float> scale_cache;
    // One entry per (token_index, head, plane), in that order.
    std::vector<RowRef> rows;
    // true for every nibble index the launch owns, per plane.
    std::vector<char> key_owned;
    std::vector<char> value_owned;
    std::vector<char> scale_owned;
    int zero_rows = 0;
    int clamped_rows = 0;

    std::size_t row_index(int token_index, int head, int plane) const
    {
        return (static_cast<std::size_t>(token_index) * shape.num_kv_heads + head) * kPlanes +
               plane;
    }
};

// `head_filter` is the one KV head the launch owns, or -1 for all of them. A
// head the launch did not run must come back looking exactly like a head
// nobody wrote, so the single-head canary protocol needs the reference to
// agree about that rather than to be corrected afterwards.
inline Reference compute_reference(const Shape& shape, const std::vector<TokenPlan>& tokens,
                                   const std::vector<float>& signs, int zero_token,
                                   const std::vector<std::int8_t>& prior_key,
                                   const std::vector<std::int8_t>& prior_value,
                                   const std::vector<float>& prior_scale,
                                   int head_filter = -1)
{
    Reference ref;
    ref.shape = shape;
    ref.key_cache = prior_key;
    ref.value_cache = prior_value;
    ref.scale_cache = prior_scale;
    ref.key_owned.assign(shape.cache_bytes() * 2, 0);
    ref.value_owned.assign(shape.cache_bytes() * 2, 0);
    ref.scale_owned.assign(shape.scale_floats(), 0);
    ref.rows.resize(tokens.size() * static_cast<std::size_t>(shape.num_kv_heads) * kPlanes);

    std::vector<float> x(static_cast<std::size_t>(shape.head_size));
    for (const TokenPlan& plan : tokens) {
        if (plan.slot < 0) continue;
        for (int head = 0; head < shape.num_kv_heads; ++head) {
            if (head_filter >= 0 && head != head_filter) continue;
            for (int plane = 0; plane < kPlanes; ++plane) {
                for (int c = 0; c < shape.head_size; ++c) {
                    const float raw = activation(plan.request, plane, head, plan.token_index, c,
                                                 zero_token);
                    // Through BF16 and back, because that is the operand the
                    // device reads.
                    x[static_cast<std::size_t>(c)] = bf16_to_float(float_to_bf16_rne(raw));
                }
                RowRef row = quantise_row(shape, x, signs);
                if (row.zero_row) ++ref.zero_rows;
                if (row.clamped) ++ref.clamped_rows;
                for (int c = 0; c < shape.head_size; ++c) {
                    const std::size_t nibble =
                        plane == 0 ? key_nibble(shape, plan.block, head, plan.token_in_block, c)
                                   : value_nibble(shape, plan.block, head, plan.token_in_block, c);
                    auto& cache = plane == 0 ? ref.key_cache : ref.value_cache;
                    auto& owned = plane == 0 ? ref.key_owned : ref.value_owned;
                    write_nibble(cache, nibble, row.q[static_cast<std::size_t>(c)]);
                    owned[nibble] = 1;
                }
                const std::size_t si =
                    scale_index(shape, plane, plan.block, head, plan.token_in_block);
                ref.scale_cache[si] = row.scale;
                ref.scale_owned[si] = 1;
                ref.rows[ref.row_index(plan.token_index, head, plane)] = std::move(row);
            }
        }
    }
    return ref;
}

}  // namespace tq_scatter
