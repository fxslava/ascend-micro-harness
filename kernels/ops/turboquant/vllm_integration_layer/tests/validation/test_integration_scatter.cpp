// CAModel runner and single-head canary protocol for the AIV-only scatter
// ingress operator. Ascend950PR_9599, CANN 9.2.0-beta.2,
// KERNEL_TYPE_AIV_ONLY.
//
//   --mode canary      one head, one request, `--tokens` tokens (default)
//   --mode sweep       every token count 1..16 on one head, one launch each
//   --mode sequential   the read-modify-write ladder: three launches into ONE
//                       strip, writing token 0, then token 1, then tokens 2..4
//   --mode zero         the absmax == 0 branch: one token's K and V rows are
//                       identically zero
//   --head H           the KV head to isolate, 0..3 (default 0)
//   --tokens N         tokens in the request, 1..16 (default 1)
//
// Every mode runs a grid of exactly one block, so one head of d = 256 is owned
// by one physical AIV and nothing else in the launch can touch it.
//
// The protocol, and what each part of it can see
// ----------------------------------------------
// 1. POISON. Every Global Memory payload -- both cache planes, the scale
//    plane and the launch probe -- is flooded with 0x7FC0BEEF before the
//    launch. It is a quiet NaN read as FP32, so a stray scale read propagates
//    rather than perturbs, and it is not 0x00, which a half-issued burst also
//    leaves behind. `camodel::GuardedDeviceBuffer` additionally surrounds each
//    payload with 64 bytes of 0xDEADBEEF: 0xDEADBEEF catches a burst that runs
//    off the end of a buffer, 0x7FC0BEEF catches one that lands in the wrong
//    place inside one.
//
// 2. THE EXPECTED IMAGE IS THE WHOLE CACHE. The oracle is handed the pre-
//    launch cache images and returns them with the owned lanes replaced, so
//    the comparison is byte for byte over every byte of every plane. That one
//    comparison is simultaneously the correctness gate, the
//    read-modify-write preservation gate, the unassigned-head gate and the
//    unassigned-page gate -- a lane the kernel should not have touched is
//    expected to still read 0x7FC0BEEF, and does not if it was touched.
//
// 3. THE NIBBLE CENSUS. A byte comparison alone reports a wrong page and a
//    page packed the wrong way round identically, so the mismatches are also
//    classified: owned nibbles that disagree with the reference, unowned
//    nibbles inside a written strip that moved (the read-modify-write failed
//    to preserve), and nibbles outside any written strip that moved (a burst
//    landed in the wrong place).
//
// 4. BIT EQUALITY, NOT A TOLERANCE. The fixture is integer-valued in BF16, so
//    every intermediate of the rotation is an exact integer in FP32, the
//    scale is a power of two extracted by integer ops on the FP32 bit
//    pattern, and the single round-to-nearest-even is the only rounding in
//    the path. Nibbles and scales are compared bit for bit.
//
// What this runner cannot see: a contended ring (there is none -- the kernel
// is straight-line by design), silicon timing, or a grid wider than one block.
// Zero cross-core flags, zero wait_intra_block and the HardEvent balance are
// the trace audit's job: scripts/analyze_scatter_trace.py.

#include "camodel_common.hpp"
#include "../common/turboquant_scatter_oracle.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <vector>

extern "C" void launch_turboquant_reshape_scatter_aiv(
    void*, std::uint32_t, std::uint16_t*, std::uint16_t*, std::int8_t*, std::int8_t*, float*,
    std::int32_t*, float*, std::int32_t*, std::int32_t*, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, float);

namespace {

using namespace tq_scatter;

constexpr std::uint32_t kPoison = 0x7FC0BEEFu;
constexpr int kProbeLanes = 32;

int failures = 0;

void expect(bool condition, const char* what)
{
    if (condition) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

std::uint32_t bits_of(float value)
{
    std::uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
}

void fill_payload(camodel::GuardedDeviceBuffer& buffer, std::uint32_t word)
{
    auto* payload = static_cast<std::uint32_t*>(buffer.host_payload());
    for (std::size_t i = 0; i < buffer.payload_bytes / 4; ++i) payload[i] = word;
}

int parse_u32(std::string_view text, std::uint32_t& out)
{
    const auto result = std::from_chars(text.data(), text.data() + text.size(), out);
    return result.ec == std::errc() ? 0 : -1;
}

// The fixture geometry. block_size == kStripHeight puts one strip in a page,
// which is the shape the layout contract calls byte-identical to plain
// channel-major and the one the 2048-byte read-modify-write is defined on.
Shape make_shape()
{
    Shape shape;
    shape.num_requests = 1;
    shape.num_kv_heads = 4;
    shape.head_size = rot_oracle::kChannels;
    shape.block_size = kStripHeight;
    shape.num_blocks = 8;
    return shape;
}

// Request 0 lands on page 3: an ODD page in the middle of the pool, so every
// other page must still hold its poison afterwards and a kernel that derived
// its page from the block index rather than from slotMapping fails.
constexpr int kActivePage = 3;

struct Outcome {
    std::size_t key_bytes_wrong = 0;
    std::size_t value_bytes_wrong = 0;
    std::size_t scale_words_wrong = 0;
    std::size_t owned_nibbles_wrong = 0;
    std::size_t preserved_nibbles_moved = 0;  // unowned, inside a written strip
    std::size_t foreign_nibbles_moved = 0;    // unowned, outside every written strip
    std::size_t owned_still_poison = 0;
};

// Which (block, head, strip) windows this launch wrote, so an unowned nibble
// can be classified as "inside the strip the kernel legitimately rewrote" or
// "somewhere it had no business touching".
struct StripKey {
    int block;
    int head;
    int strip;
    bool operator==(const StripKey& other) const
    {
        return block == other.block && head == other.head && strip == other.strip;
    }
};

bool in_written_strip(const Shape& shape, const std::vector<StripKey>& strips,
                      std::size_t nibble)
{
    for (const StripKey& key : strips) {
        const std::size_t begin = strip_byte_offset(shape, key.block, key.head, key.strip) * 2;
        if (nibble >= begin && nibble < begin + static_cast<std::size_t>(kStripHeight) *
                                                    shape.head_size)
            return true;
    }
    return false;
}

// One launch, start to finish: build the fixture, poison the caches, run the
// kernel, and compare the whole of Global Memory against the oracle's image.
Outcome run_launch(camodel::Session& session, const Shape& shape, std::uint32_t head,
                   const std::vector<int>& token_slots, int zero_token,
                   std::vector<std::int8_t>& key_image, std::vector<std::int8_t>& value_image,
                   std::vector<float>& scale_image, bool first_launch, const char* tag)
{
    const int tokens = static_cast<int>(token_slots.size());
    const std::vector<float> signs = rot_oracle::canonical_signs();

    // ---- host side: the activations and the plan ------------------------
    std::vector<TokenPlan> plan;
    plan.reserve(token_slots.size());
    for (int t = 0; t < tokens; ++t)
        plan.push_back(plan_token(shape, 0, t, token_slots[static_cast<std::size_t>(t)]));

    const std::size_t channels =
        static_cast<std::size_t>(shape.num_kv_heads) * shape.head_size;
    std::vector<std::uint16_t> key_rows(static_cast<std::size_t>(tokens) * channels, 0);
    std::vector<std::uint16_t> value_rows(key_rows.size(), 0);
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < shape.num_kv_heads; ++h)
            for (int c = 0; c < shape.head_size; ++c) {
                const std::size_t off = static_cast<std::size_t>(t) * channels +
                                        static_cast<std::size_t>(h) * shape.head_size + c;
                key_rows[off] = float_to_bf16_rne(activation(0, 0, h, t, c, zero_token));
                value_rows[off] = float_to_bf16_rne(activation(0, 1, h, t, c, zero_token));
            }

    // The launch owns exactly one head, so the reference is taken for that
    // head alone: a head the launch did not run must come back looking
    // exactly like a head nobody wrote, and that is part of what the byte
    // comparison below checks rather than something it is corrected for.
    const Reference ref =
        compute_reference(shape, plan, signs, zero_token, key_image, value_image, scale_image,
                          static_cast<int>(head));

    // ---- device side ----------------------------------------------------
    camodel::GuardedDeviceBuffer d_key(key_rows.size() * 2);
    camodel::GuardedDeviceBuffer d_value(value_rows.size() * 2);
    camodel::GuardedDeviceBuffer d_key_cache(shape.cache_bytes());
    camodel::GuardedDeviceBuffer d_value_cache(shape.cache_bytes());
    camodel::GuardedDeviceBuffer d_scale(shape.scale_floats() * 4);
    camodel::GuardedDeviceBuffer d_slots(((token_slots.size() * 4 + 31) / 32) * 32);
    camodel::GuardedDeviceBuffer d_signs(signs.size() * 4);
    camodel::GuardedDeviceBuffer d_cu(64);
    camodel::GuardedDeviceBuffer d_probe(kProbeLanes * 4);

    std::memcpy(d_key.host_payload(), key_rows.data(), key_rows.size() * 2);
    std::memcpy(d_value.host_payload(), value_rows.data(), value_rows.size() * 2);
    std::memcpy(d_signs.host_payload(), signs.data(), signs.size() * 4);
    std::memcpy(d_key_cache.host_payload(), key_image.data(), key_image.size());
    std::memcpy(d_value_cache.host_payload(), value_image.data(), value_image.size());
    std::memcpy(d_scale.host_payload(), scale_image.data(), scale_image.size() * 4);
    {
        auto* slots = static_cast<std::int32_t*>(d_slots.host_payload());
        for (std::size_t i = 0; i < d_slots.payload_bytes / 4; ++i) slots[i] = -1;
        for (std::size_t i = 0; i < token_slots.size(); ++i)
            slots[i] = token_slots[i];
        auto* cu = static_cast<std::int32_t*>(d_cu.host_payload());
        cu[0] = 0;
        cu[1] = tokens;
    }
    fill_payload(d_probe, kPoison);
    d_key.upload();
    d_value.upload();
    d_key_cache.upload();
    d_value_cache.upload();
    d_scale.upload();
    d_slots.upload();
    d_signs.upload();
    d_cu.upload();
    d_probe.upload();

    // One work item, one block: head `head` of request 0 and nothing else.
    const std::uint32_t item = 0u * static_cast<std::uint32_t>(shape.num_kv_heads) + head;
    std::printf("  [%s] launch: head %u, %d token(s), page %d, %u work item, grid 1\n", tag,
                head, tokens, kActivePage, 1u);
    launch_turboquant_reshape_scatter_aiv(
        session.stream, 1u, static_cast<std::uint16_t*>(d_key.payload()),
        static_cast<std::uint16_t*>(d_value.payload()),
        static_cast<std::int8_t*>(d_key_cache.payload()),
        static_cast<std::int8_t*>(d_value_cache.payload()),
        static_cast<float*>(d_scale.payload()), static_cast<std::int32_t*>(d_slots.payload()),
        static_cast<float*>(d_signs.payload()), static_cast<std::int32_t*>(d_cu.payload()),
        static_cast<std::int32_t*>(d_probe.payload()), 1u,
        static_cast<std::uint32_t>(shape.num_kv_heads),
        static_cast<std::uint32_t>(shape.head_size),
        static_cast<std::uint32_t>(shape.block_size),
        static_cast<std::uint32_t>(shape.num_blocks), item, 1u, shape.inv_sqrt_len());
    camodel::check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");

    d_key_cache.download();
    d_value_cache.download();
    d_scale.download();
    d_key.download();
    d_value.download();
    d_signs.download();
    d_probe.download();
    d_key.assert_guards();
    d_value.assert_guards();
    d_key_cache.assert_guards();
    d_value_cache.assert_guards();
    d_scale.assert_guards();
    d_slots.download();
    d_slots.assert_guards();
    d_signs.assert_guards();
    d_probe.assert_guards();

    if (first_launch) {
        const auto* probe = static_cast<const std::int32_t*>(d_probe.host_payload());
        std::printf("  [%s] launch probe: GetBlockIdx %d, GetBlockNum %d, "
                    "GetSubBlockNum %d, itemCount %d\n",
                    tag, probe[0], probe[8], probe[16], probe[24]);
    }

    // ---- the comparison --------------------------------------------------
    const auto* dev_key = static_cast<const std::int8_t*>(d_key_cache.host_payload());
    const auto* dev_value = static_cast<const std::int8_t*>(d_value_cache.host_payload());
    const auto* dev_scale = static_cast<const float*>(d_scale.host_payload());

    std::vector<StripKey> strips;
    for (const TokenPlan& p : plan) {
        if (p.slot < 0) continue;
        StripKey key{p.block, static_cast<int>(head), p.strip};
        if (std::find(strips.begin(), strips.end(), key) == strips.end()) strips.push_back(key);
    }

    Outcome out;
    for (std::size_t i = 0; i < ref.key_cache.size(); ++i)
        if (dev_key[i] != ref.key_cache[i]) ++out.key_bytes_wrong;
    for (std::size_t i = 0; i < ref.value_cache.size(); ++i)
        if (dev_value[i] != ref.value_cache[i]) ++out.value_bytes_wrong;
    for (std::size_t i = 0; i < ref.scale_cache.size(); ++i)
        if (bits_of(dev_scale[i]) != bits_of(ref.scale_cache[i])) ++out.scale_words_wrong;

    const std::size_t nibbles = shape.cache_bytes() * 2;
    bool said = false;
    for (std::size_t n = 0; n < nibbles; ++n) {
        for (int pl = 0; pl < kPlanes; ++pl) {
            const std::int8_t* dev = pl == 0 ? dev_key : dev_value;
            const auto& want = pl == 0 ? ref.key_cache : ref.value_cache;
            const auto& own = pl == 0 ? ref.key_owned : ref.value_owned;
            const int got = read_nibble(dev, n);
            const int exp = read_nibble(want.data(), n);
            if (got == exp) continue;
            if (own[n]) {
                ++out.owned_nibbles_wrong;
                if (!said) {
                    std::fprintf(stderr,
                                 "  [%s] first owned divergence: plane %d nibble %zu, "
                                 "device %d oracle %d\n",
                                 tag, pl, n, got, exp);
                    said = true;
                }
            } else if (in_written_strip(shape, strips, n)) {
                ++out.preserved_nibbles_moved;
            } else {
                ++out.foreign_nibbles_moved;
            }
        }
    }
    // Owned lanes that still read as poison were never written at all; the
    // poison pattern 0x7FC0BEEF is 0xEF 0xBE 0xC0 0x7F, whose nibbles are a
    // fixed sequence the reference almost never reproduces, so this is a
    // separate count rather than a separate comparison.
    for (std::size_t n = 0; n < nibbles; ++n) {
        if (ref.key_owned[n] && read_nibble(dev_key, n) == read_nibble(key_image.data(), n) &&
            read_nibble(ref.key_cache.data(), n) != read_nibble(key_image.data(), n))
            ++out.owned_still_poison;
        if (ref.value_owned[n] &&
            read_nibble(dev_value, n) == read_nibble(value_image.data(), n) &&
            read_nibble(ref.value_cache.data(), n) != read_nibble(value_image.data(), n))
            ++out.owned_still_poison;
    }

    std::printf("  [%s] zero rows %d, clamped rows %d; key bytes wrong %zu, value bytes "
                "wrong %zu, scale words wrong %zu\n",
                tag, ref.zero_rows, ref.clamped_rows, out.key_bytes_wrong,
                out.value_bytes_wrong, out.scale_words_wrong);
    std::printf("  [%s] nibble census: owned wrong %zu, preserved-in-strip moved %zu, "
                "foreign moved %zu, owned unwritten %zu\n",
                tag, out.owned_nibbles_wrong, out.preserved_nibbles_moved,
                out.foreign_nibbles_moved, out.owned_still_poison);

    // Carry the device images forward so a following launch sees what the
    // previous one actually left, not what it was supposed to leave. That is
    // what makes the sequential ladder a test of the device's own
    // read-modify-write rather than of the oracle's.
    key_image.assign(dev_key, dev_key + ref.key_cache.size());
    value_image.assign(dev_value, dev_value + ref.value_cache.size());
    scale_image.assign(dev_scale, dev_scale + ref.scale_cache.size());
    return out;
}

void check(const Outcome& out, const char* tag)
{
    char what[192];
    std::snprintf(what, sizeof(what), "%s: owned nibbles bit-exact", tag);
    expect(out.owned_nibbles_wrong == 0, what);
    std::snprintf(what, sizeof(what), "%s: the RMW preserved every unowned lane of the strip",
                  tag);
    expect(out.preserved_nibbles_moved == 0, what);
    std::snprintf(what, sizeof(what), "%s: no byte outside a written strip moved", tag);
    expect(out.foreign_nibbles_moved == 0, what);
    std::snprintf(what, sizeof(what), "%s: every owned lane was written", tag);
    expect(out.owned_still_poison == 0, what);
    std::snprintf(what, sizeof(what), "%s: the scale plane is bit-exact", tag);
    expect(out.scale_words_wrong == 0, what);
    std::snprintf(what, sizeof(what), "%s: both cache planes are byte-identical", tag);
    expect(out.key_bytes_wrong == 0 && out.value_bytes_wrong == 0, what);
}

void poison_caches(const Shape& shape, std::vector<std::int8_t>& key_image,
                   std::vector<std::int8_t>& value_image, std::vector<float>& scale_image)
{
    key_image.assign(shape.cache_bytes(), 0);
    value_image.assign(shape.cache_bytes(), 0);
    scale_image.assign(shape.scale_floats(), 0.0f);
    for (std::size_t i = 0; i < key_image.size(); i += 4) {
        std::memcpy(key_image.data() + i, &kPoison, 4);
        std::memcpy(value_image.data() + i, &kPoison, 4);
    }
    for (float& value : scale_image) std::memcpy(&value, &kPoison, 4);
}

}  // namespace

int main(int argc, char** argv)
{
    std::string_view mode = "canary";
    std::uint32_t head = 0;
    std::uint32_t tokens = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--head" && i + 1 < argc) {
            if (parse_u32(argv[++i], head) != 0 || head > 3) {
                std::fprintf(stderr, "--head must be 0..3\n");
                return 2;
            }
        } else if (arg == "--tokens" && i + 1 < argc) {
            if (parse_u32(argv[++i], tokens) != 0 || tokens == 0 ||
                tokens > static_cast<std::uint32_t>(kStripHeight)) {
                std::fprintf(stderr, "--tokens must be 1..%d\n", kStripHeight);
                return 2;
            }
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    const Shape shape = make_shape();
    if (!shape.valid()) {
        std::fprintf(stderr, "fixture shape is not valid\n");
        return 2;
    }
    std::printf("turboquant_reshape_scatter_aiv CAModel: mode %.*s, head %u, "
                "d %d, S %d, P %d, strip %zu B\n",
                static_cast<int>(mode.size()), mode.data(), head, shape.head_size,
                shape.block_size, shape.num_blocks, shape.strip_bytes());
    std::printf("  head %u isolated\n", head);

    // The fixture's own precondition: the rotated panel must stay exactly
    // representable, or none of the bit-equality gates below means anything.
    {
        const std::vector<float> signs = rot_oracle::canonical_signs();
        std::vector<float> x(static_cast<std::size_t>(shape.head_size));
        double peak = 0.0;
        for (int c = 0; c < shape.head_size; ++c) {
            const float raw = activation(0, 0, static_cast<int>(head), 0, c, -1);
            expect(bf16_is_exact(raw), "the fixture activation is exact in BF16");
            x[static_cast<std::size_t>(c)] = raw;
        }
        const RowRef row = quantise_row(shape, x, signs);
        for (float value : row.rotated) peak = std::fmax(peak, std::fabs(value));
        expect(peak < 16777216.0, "the rotated panel is exactly representable in FP32");
        std::printf("  fixture: peak |Pi . x| %.0f, absmax %.0f, exponent %d, scale %.9g\n",
                    peak, static_cast<double>(row.absmax), row.exponent,
                    static_cast<double>(row.scale));
    }

    camodel::Session session;
    std::vector<std::int8_t> key_image, value_image;
    std::vector<float> scale_image;

    if (mode == "canary" || mode == "zero") {
        poison_caches(shape, key_image, value_image, scale_image);
        std::vector<int> slots;
        for (std::uint32_t t = 0; t < tokens; ++t)
            slots.push_back(kActivePage * shape.block_size + static_cast<int>(t));
        const int zero_token = mode == "zero" ? 0 : -1;
        const Outcome out = run_launch(session, shape, head, slots, zero_token, key_image,
                                       value_image, scale_image, true, "run");
        check(out, "run");
        if (mode == "zero") {
            // The branch under test: an all-zero row must store scale 1.0f and
            // all-zero codes, and the oracle must have produced that case at
            // all -- a gate that passes because the fixture never reached the
            // branch is worse than no gate.
            const std::vector<float> signs = rot_oracle::canonical_signs();
            std::vector<float> zeros(static_cast<std::size_t>(shape.head_size), 0.0f);
            const RowRef zero_row = quantise_row(shape, zeros, signs);
            expect(zero_row.zero_row, "the zero fixture reaches the absmax == 0 branch");
            expect(zero_row.scale == 1.0f, "the absmax == 0 branch stores scale 1.0f");
            bool all_zero = true;
            for (int code : zero_row.q)
                if (code != 0) all_zero = false;
            expect(all_zero, "the absmax == 0 branch stores all-zero codes");
        }
    } else if (mode == "sweep") {
        for (std::uint32_t n = 1; n <= static_cast<std::uint32_t>(kStripHeight); ++n) {
            poison_caches(shape, key_image, value_image, scale_image);
            std::vector<int> slots;
            for (std::uint32_t t = 0; t < n; ++t)
                slots.push_back(kActivePage * shape.block_size + static_cast<int>(t));
            char tag[32];
            std::snprintf(tag, sizeof(tag), "n=%u", n);
            const Outcome out = run_launch(session, shape, head, slots, -1, key_image,
                                           value_image, scale_image, n == 1, tag);
            check(out, tag);
        }
    } else if (mode == "sequential") {
        // The read-modify-write ladder. Three launches into ONE strip: token 0,
        // then token 1, then tokens 2..4. Each launch's reference starts from
        // what the PREVIOUS launch actually left in Global Memory, so a kernel
        // that zeroed the lanes it does not own fails at step two and a kernel
        // that wrote the right nibbles into the wrong token slot fails at
        // step two as well.
        poison_caches(shape, key_image, value_image, scale_image);
        const int base = kActivePage * shape.block_size;
        const std::vector<std::vector<int>> steps = {
            {base + 0}, {base + 1}, {base + 2, base + 3, base + 4}};
        int step = 0;
        for (const std::vector<int>& slots : steps) {
            char tag[32];
            std::snprintf(tag, sizeof(tag), "step%d", step);
            const Outcome out = run_launch(session, shape, head, slots, -1, key_image,
                                           value_image, scale_image, step == 0, tag);
            check(out, tag);
            ++step;
        }
    } else {
        std::fprintf(stderr, "--mode must be canary, sweep, sequential or zero\n");
        return 2;
    }

    if (failures == 0) {
        std::printf("turboquant_reshape_scatter_aiv CAModel (%.*s, head %u): PASS\n",
                    static_cast<int>(mode.size()), mode.data(), head);
        return 0;
    }
    std::fprintf(stderr, "turboquant_reshape_scatter_aiv CAModel (%.*s, head %u): %d FAILURES\n",
                 static_cast<int>(mode.size()), mode.data(), head, failures);
    return 1;
}
