// Tier-1 host probe for turboquant_reshape_scatter_aiv.
//
// A CAModel run compares the device against this oracle, so a wrong oracle and
// a wrong kernel are indistinguishable there, and a run costs minutes. Five
// things are settled here instead, in well under a second:
//
//   1. THE LAYOUT MAPS ARE BIJECTIONS. Both plane index functions cover
//      [0, P*H*S*d) exactly once. A transposed or mis-strided map is an
//      orthogonal-looking cache the device comparison would report as a
//      mismatch without saying which map was wrong.
//
//   2. THE VALUE STRIP IS A WINDOW. Every nibble of (block, head, strip) lies
//      inside the 2048 bytes at `strip_byte_offset`, and no nibble outside
//      that (block, head, strip) lies inside it. The kernel reads and writes
//      exactly that window, so if this were false the read-modify-write would
//      be reading someone else's strip.
//
//   3. THE MERGE ARITHMETIC. The kernel splices a token by uint32 word and bit
//      position: word 2c + t/8, bits [4*(t%8) + 3 : 4*(t%8)]. That derivation
//      is checked against the nibble map, and then the whole masked merge is
//      SIMULATED at word granularity and compared against the nibble-level
//      reference -- which is the only check in this file that could have
//      caught an endianness or half-selection error before the device did.
//
//   4. THE QUANTISER AND ITS GUARD. The power-of-two scale extracted from the
//      FP32 bit pattern reproduces ldexp exactly; scaling y and absmax
//      together leaves every code unchanged (which is what lets invSqrtLen
//      live in the stored scale); and absmax == 0 takes the explicit branch.
//
//   5. THE FIXTURE CAN DETECT A FAULT AT ALL. All sixteen codes occur, the
//      upper clamp is reached, tokens and heads are pairwise distinct, and the
//      per-row exponents are not all equal. A fixture whose rows are
//      interchangeable cannot tell a wrong-token or wrong-head read from a
//      correct one, which is the entire point of the canary protocol.

#include "../common/turboquant_scatter_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

namespace {

using namespace tq_scatter;

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

// ------------------------------------------------------------------- (1)
void check_bijections(const Shape& shape)
{
    const std::size_t nibbles = shape.cache_bytes() * 2;
    std::vector<char> key_seen(nibbles, 0);
    std::vector<char> value_seen(nibbles, 0);
    bool key_ok = true;
    bool value_ok = true;
    for (int b = 0; b < shape.num_blocks; ++b)
        for (int h = 0; h < shape.num_kv_heads; ++h)
            for (int t = 0; t < shape.block_size; ++t)
                for (int c = 0; c < shape.head_size; ++c) {
                    const std::size_t kn = key_nibble(shape, b, h, t, c);
                    const std::size_t vn = value_nibble(shape, b, h, t, c);
                    if (kn >= nibbles || key_seen[kn]) key_ok = false;
                    if (vn >= nibbles || value_seen[vn]) value_ok = false;
                    if (kn < nibbles) key_seen[kn] = 1;
                    if (vn < nibbles) value_seen[vn] = 1;
                }
    for (std::size_t i = 0; i < nibbles; ++i) {
        if (!key_seen[i]) key_ok = false;
        if (!value_seen[i]) value_ok = false;
    }
    expect(key_ok, "the key nibble map is a bijection onto the whole plane");
    expect(value_ok, "the value nibble map is a bijection onto the whole plane");

    // At S == R the value plane is byte-identical to plain channel-major,
    // which is the form the earlier CAModel probes verified. Say so rather
    // than assume it.
    bool channel_major = true;
    for (int b = 0; b < shape.num_blocks && channel_major; ++b)
        for (int h = 0; h < shape.num_kv_heads; ++h)
            for (int t = 0; t < shape.block_size; ++t)
                for (int c = 0; c < shape.head_size; ++c) {
                    const std::size_t page = static_cast<std::size_t>(b * shape.num_kv_heads + h) *
                                             shape.block_size * shape.head_size;
                    const std::size_t plain =
                        page + static_cast<std::size_t>(c) * shape.block_size + t;
                    if (value_nibble(shape, b, h, t, c) != plain) channel_major = false;
                }
    expect(channel_major, "at S == R the value plane is plain channel-major");
}

// ------------------------------------------------------------------- (2)
void check_strip_window(const Shape& shape)
{
    const int strips = shape.strips_per_block();
    bool inside = true;
    bool exclusive = true;
    for (int b = 0; b < shape.num_blocks; ++b)
        for (int h = 0; h < shape.num_kv_heads; ++h)
            for (int s = 0; s < strips; ++s) {
                const std::size_t begin = strip_byte_offset(shape, b, h, s) * 2;
                const std::size_t end =
                    begin + static_cast<std::size_t>(kStripHeight) * shape.head_size;
                for (int t = 0; t < shape.block_size; ++t)
                    for (int c = 0; c < shape.head_size; ++c) {
                        const std::size_t n = value_nibble(shape, b, h, t, c);
                        const bool own = (t / kStripHeight) == s;
                        const bool in = n >= begin && n < end;
                        if (own && !in) inside = false;
                        if (!own && in) exclusive = false;
                    }
            }
    expect(inside, "every nibble of a strip lies inside its 2048-byte window");
    expect(exclusive, "no nibble outside a strip lies inside its window");
    expect(strip_byte_offset(shape, 0, 0, 0) == 0, "the first strip starts the plane");
    expect(shape.strip_bytes() == 2048u, "a strip is 2048 bytes");
}

// ------------------------------------------------------------------- (3)
// The kernel's own addressing, restated: the strip as 512 uint32 words, with
// token t of channel c at word 2c + t/8, bit 4*(t%8). Little-endian bytes
// within a word, which is what DaVinci is and what the nibble map below has to
// agree with.
void check_merge_arithmetic(const Shape& shape)
{
    bool agrees = true;
    for (int t = 0; t < kStripHeight && agrees; ++t)
        for (int c = 0; c < shape.head_size; ++c) {
            const std::size_t nibble_in_strip =
                static_cast<std::size_t>(c) * kStripHeight + static_cast<std::size_t>(t);
            // Where the nibble map puts it, as (word, bit).
            const std::size_t byte = nibble_in_strip / 2;
            const std::size_t word = byte / 4;
            const int bit = static_cast<int>(8 * (byte % 4) + 4 * (nibble_in_strip % 2));
            // Where the kernel puts it.
            const std::size_t kernel_word =
                2 * static_cast<std::size_t>(c) + static_cast<std::size_t>(t) / 8;
            const int kernel_bit = 4 * (t % 8);
            if (word != kernel_word || bit != kernel_bit) {
                std::fprintf(stderr,
                             "  merge arithmetic: token %d channel %d -> map (word %zu, "
                             "bit %d), kernel (word %zu, bit %d)\n",
                             t, c, word, bit, kernel_word, kernel_bit);
                agrees = false;
                break;
            }
        }
    expect(agrees, "the kernel's (word, bit) derivation matches the value nibble map");

    // Now simulate the whole masked merge the way the device does it, and
    // compare against the nibble-level reference. This is the check that would
    // have caught a half-selection or endianness error on the host.
    const std::size_t words = shape.strip_bytes() / 4;
    std::vector<std::uint32_t> prior(words);
    for (std::size_t w = 0; w < words; ++w) prior[w] = 0x7FC0BEEFu;

    // Three tokens, one from each half of the strip plus a boundary one.
    const int tokens[] = {0, 7, 8, 15, 3};
    std::vector<std::uint32_t> device = prior;
    std::vector<std::uint8_t> reference(shape.strip_bytes());
    std::memcpy(reference.data(), prior.data(), reference.size());

    for (int t : tokens) {
        // Codes distinct per channel so a wrong channel cannot pass.
        std::vector<int> codes(static_cast<std::size_t>(shape.head_size));
        for (int c = 0; c < shape.head_size; ++c)
            codes[static_cast<std::size_t>(c)] = ((c * 7 + t * 3) % 16) - 8;

        // --- the device's algorithm, word by word ---
        const std::uint32_t shift = static_cast<std::uint32_t>(4 * (t % 8));
        const std::uint32_t half = static_cast<std::uint32_t>(t / 8);
        std::uint32_t keep_pattern[2];  // the Interleave of (0xF << shift) and 0
        keep_pattern[half] = 0xFu << shift;
        keep_pattern[1u - half] = 0u;
        for (std::size_t w = 0; w < words; ++w) {
            const std::size_t channel = w / 2;
            const std::uint32_t keep = keep_pattern[w % 2];
            const std::uint32_t code =
                static_cast<std::uint32_t>(codes[channel] & 0xF) << shift;
            device[w] = (device[w] & ~keep) | (code & keep);
        }

        // --- the nibble-level reference ---
        for (int c = 0; c < shape.head_size; ++c) {
            const std::size_t n = static_cast<std::size_t>(c) * kStripHeight +
                                  static_cast<std::size_t>(t);
            const auto nib = static_cast<std::uint8_t>(codes[static_cast<std::size_t>(c)] & 0xF);
            std::uint8_t& byte = reference[n / 2];
            if (n % 2 == 0)
                byte = static_cast<std::uint8_t>((byte & 0xF0u) | nib);
            else
                byte = static_cast<std::uint8_t>((byte & 0x0Fu) |
                                                 static_cast<std::uint8_t>(nib << 4));
        }
    }
    expect(std::memcmp(device.data(), reference.data(), reference.size()) == 0,
           "the word-masked merge reproduces the nibble-level reference exactly");

    // And it preserved: tokens 1, 2, 4, 5, 6, 9..14 were never written, so
    // their nibbles must still carry the poison pattern.
    const std::set<int> written = {0, 7, 8, 15, 3};
    std::size_t moved = 0;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(device.data());
    const auto* prior_bytes = reinterpret_cast<const std::uint8_t*>(prior.data());
    for (int t = 0; t < kStripHeight; ++t) {
        if (written.count(t)) continue;
        for (int c = 0; c < shape.head_size; ++c) {
            const std::size_t n = static_cast<std::size_t>(c) * kStripHeight +
                                  static_cast<std::size_t>(t);
            const int got = (n % 2 == 0) ? (bytes[n / 2] & 0xF) : ((bytes[n / 2] >> 4) & 0xF);
            const int was = (n % 2 == 0) ? (prior_bytes[n / 2] & 0xF)
                                         : ((prior_bytes[n / 2] >> 4) & 0xF);
            if (got != was) ++moved;
        }
    }
    expect(moved == 0, "the merge preserved every nibble of every unwritten token");
}

// ------------------------------------------------------------------- (4)
void check_quantiser(const Shape& shape)
{
    const std::vector<float> signs = rot_oracle::canonical_signs();
    std::vector<float> x(static_cast<std::size_t>(shape.head_size));

    // The power-of-two scale extracted the way the device extracts it.
    bool bits_agree = true;
    for (int e = -20; e <= 20; ++e) {
        const float absmax = std::ldexp(1.3f, e);  // in [2^e, 2^(e+1))
        const std::uint32_t exp_bits = bits_of(absmax) & 0x7F800000u;
        const std::uint32_t pow_bits = exp_bits - 0x01000000u;
        const std::uint32_t inv_bits = 0x7F000000u - pow_bits;
        float pow_two, inv;
        std::memcpy(&pow_two, &pow_bits, 4);
        std::memcpy(&inv, &inv_bits, 4);
        if (pow_two != std::ldexp(1.0f, e - 2) || inv != std::ldexp(1.0f, 2 - e)) {
            std::fprintf(stderr, "  exponent %d: pow %g want %g, inv %g want %g\n", e,
                         static_cast<double>(pow_two),
                         static_cast<double>(std::ldexp(1.0f, e - 2)),
                         static_cast<double>(inv),
                         static_cast<double>(std::ldexp(1.0f, 2 - e)));
            bits_agree = false;
        }
    }
    expect(bits_agree, "the FP32 bit arithmetic gives 2^(e-2) and its exact reciprocal");

    // The zero-row branch, and what it would have produced without one.
    std::vector<float> zeros(static_cast<std::size_t>(shape.head_size), 0.0f);
    const RowRef zero_row = quantise_row(shape, zeros, signs);
    expect(zero_row.zero_row, "an all-zero row takes the absmax == 0 branch");
    expect(zero_row.scale == 1.0f, "the absmax == 0 branch stores scale 1.0f");
    bool all_zero = true;
    for (int code : zero_row.q)
        if (code != 0) all_zero = false;
    expect(all_zero, "the absmax == 0 branch stores all-zero codes");
    {
        // Without the branch: exp_bits is 0, so pow_bits borrows an exponent
        // and the stored scale is a large negative number rather than a
        // denormal. The codes survive because every y is zero -- which is
        // exactly why the fault is silent.
        const std::uint32_t pow_bits = 0u - 0x01000000u;
        float pow_two;
        std::memcpy(&pow_two, &pow_bits, 4);
        expect(!(pow_two > 0.0f),
               "without the branch the unguarded scale is not even positive");
        std::printf("  unguarded absmax == 0 scale would be %g\n",
                    static_cast<double>(pow_two));
    }

    for (int c = 0; c < shape.head_size; ++c)
        x[static_cast<std::size_t>(c)] = activation(0, 0, 1, 0, c, -1);
    const RowRef row = quantise_row(shape, x, signs);

    // SCALE INVARIANCE, which is the whole reason invSqrtLen may live in the
    // stored scale alone: the codes depend only on y / absmax, so multiplying
    // the input by a power of two must leave every code identical and move the
    // stored scale by exactly that factor. Checked over 2^-8 .. 2^8, which is
    // the property, not a restatement of it.
    bool invariant = true;
    for (int k = -8; k <= 8; ++k) {
        std::vector<float> scaled_x(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            scaled_x[i] = std::ldexp(x[i], k);
        const RowRef again = quantise_row(shape, scaled_x, signs);
        if (again.q != row.q || again.scale != std::ldexp(row.scale, k)) {
            std::fprintf(stderr,
                         "  scale invariance broke at 2^%d: scale %g want %g, codes %s\n", k,
                         static_cast<double>(again.scale),
                         static_cast<double>(std::ldexp(row.scale, k)),
                         again.q == row.q ? "equal" : "DIFFERENT");
            invariant = false;
        }
    }
    expect(invariant,
           "scaling the input by 2^k leaves every code identical and moves only the scale");

    // Reconstruction: q * scale must be y / sqrt(d) to within half a code step,
    // on every channel the upper clamp did not touch. A clamped channel is
    // allowed to be further out -- that is what clamping means -- so including
    // it would turn this into a check of whether the fixture clamps.
    double worst = 0.0;
    int clamped_channels = 0;
    for (int c = 0; c < shape.head_size; ++c) {
        if (row.q[static_cast<std::size_t>(c)] == kNibbleMax) {
            ++clamped_channels;
            continue;
        }
        const double want = static_cast<double>(row.rotated[static_cast<std::size_t>(c)]) *
                            static_cast<double>(shape.inv_sqrt_len());
        const double got = static_cast<double>(row.q[static_cast<std::size_t>(c)]) *
                           static_cast<double>(row.scale);
        worst = std::fmax(worst, std::fabs(got - want));
    }
    expect(worst <= static_cast<double>(row.scale) * 0.5 + 1e-30,
           "q * scale reconstructs y / sqrt(d) to within half a code step");
    std::printf("  reconstruction: worst |q.scale - y/16| = %.9g, half a step = %.9g, "
                "%d channel(s) at the clamp\n",
                worst, static_cast<double>(row.scale) * 0.5, clamped_channels);
}

// ------------------------------------------------------------------- (5)
void check_fixture(const Shape& shape)
{
    const std::vector<float> signs = rot_oracle::canonical_signs();
    std::set<int> codes_seen;
    std::set<int> exponents;
    int clamped = 0;
    std::vector<std::vector<int>> rows;
    double peak = 0.0;

    for (int plane = 0; plane < kPlanes; ++plane)
        for (int head = 0; head < shape.num_kv_heads; ++head)
            for (int token = 0; token < kStripHeight; ++token) {
                std::vector<float> x(static_cast<std::size_t>(shape.head_size));
                for (int c = 0; c < shape.head_size; ++c) {
                    const float raw = activation(0, plane, head, token, c, -1);
                    expect(bf16_is_exact(raw), "every fixture activation is exact in BF16");
                    x[static_cast<std::size_t>(c)] = raw;
                }
                const RowRef row = quantise_row(shape, x, signs);
                for (float value : row.rotated) peak = std::fmax(peak, std::fabs(value));
                for (int code : row.q) codes_seen.insert(code);
                exponents.insert(row.exponent);
                if (row.clamped) ++clamped;
                rows.push_back(row.q);
            }

    expect(codes_seen.size() == 16, "all sixteen INT4 codes occur in the fixture");
    expect(codes_seen.count(kNibbleMax) == 1 && codes_seen.count(kNibbleMin) == 1,
           "both INT4 extremes occur");
    expect(clamped > 0, "the upper clamp is reached at least once");
    expect(exponents.size() > 1, "the per-row exponents are not all equal");
    expect(peak < 16777216.0, "the rotated panel is exactly representable in FP32");

    bool distinct = true;
    for (std::size_t i = 0; i < rows.size() && distinct; ++i)
        for (std::size_t j = i + 1; j < rows.size(); ++j)
            if (rows[i] == rows[j]) {
                std::fprintf(stderr, "  fixture rows %zu and %zu are identical\n", i, j);
                distinct = false;
                break;
            }
    expect(distinct, "every (plane, head, token) row quantises to a distinct page");
    std::printf("  fixture: %zu distinct codes, %zu distinct exponents, %d clamped rows, "
                "peak |Pi . x| %.0f\n",
                codes_seen.size(), exponents.size(), clamped, peak);
}

// The slot grouping the kernel performs, mirrored on the host: a run is the
// maximal set of tokens with consecutive slots inside one strip.
void check_run_grouping(const Shape& shape)
{
    struct Case {
        std::vector<int> slots;
        std::vector<int> expected_runs;  // run lengths
    };
    const int S = shape.block_size;
    const std::vector<Case> cases = {
        {{3 * S + 0}, {1}},                              // one decode token
        {{3 * S + 0, 3 * S + 1, 3 * S + 2}, {3}},        // MTP, consecutive
        {{3 * S + 0, 5 * S + 0}, {1, 1}},                // two requests' pages
        {{3 * S + 0, 3 * S + 2}, {1, 1}},                // a gap breaks the run
        {{3 * S + 0, -1, 3 * S + 1}, {1, 1}},            // a padded token
    };
    bool ok = true;
    for (const Case& c : cases) {
        std::vector<int> runs;
        std::size_t i = 0;
        while (i < c.slots.size()) {
            if (c.slots[i] < 0) {
                ++i;
                continue;
            }
            const int slot0 = c.slots[i];
            const int block = slot0 / S;
            const int strip = (slot0 % S) / kStripHeight;
            std::size_t run = 1;
            while (i + run < c.slots.size() && run < static_cast<std::size_t>(kStripHeight)) {
                const int next = c.slots[i + run];
                if (next != slot0 + static_cast<int>(run)) break;
                if (next / S != block) break;
                if ((next % S) / kStripHeight != strip) break;
                ++run;
            }
            runs.push_back(static_cast<int>(run));
            i += run;
        }
        if (runs != c.expected_runs) {
            std::fprintf(stderr, "  run grouping disagreed for a %zu-slot case\n",
                         c.slots.size());
            ok = false;
        }
    }
    expect(ok, "the run grouping splits exactly where the kernel splits");
}

}  // namespace

int main()
{
    const Shape shape = make_shape();
    expect(shape.valid(), "the fixture shape is valid");
    std::printf("reshape_scatter Tier-1 probe: d %d, H %d, S %d, P %d, strip %zu B\n",
                shape.head_size, shape.num_kv_heads, shape.block_size, shape.num_blocks,
                shape.strip_bytes());

    check_bijections(shape);
    check_strip_window(shape);
    check_merge_arithmetic(shape);
    check_quantiser(shape);
    check_fixture(shape);
    check_run_grouping(shape);

    if (failures == 0) {
        std::printf("reshape_scatter probe: PASS\n");
        return 0;
    }
    std::fprintf(stderr, "reshape_scatter probe: %d FAILURES\n", failures);
    return 1;
}
