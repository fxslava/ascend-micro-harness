#include "../common/fused_decode_opt_fixture.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>

// Reports WHICH check failed. A gate that says only that it failed costs a bisect every time.
static void require_at(bool b, int line, const char* text)
{
    if (b) return;
    std::fprintf(stderr, "fused-opt fixture test failed at line %d: %s\n", line, text);
    throw std::runtime_error("fused-opt fixture test");
}
#define require(expr) require_at((expr), __LINE__, #expr)

// Byte-identical, not operator==. Two reasons, and both of them bite here: `scales` is flooded
// with 0x7FC0BEEF, which is a quiet NaN, so == is false between two identical copies; and where
// the claim is that an fp32 value did not move at all, "compares equal" is the weaker statement
// (it would also accept +0 against -0).
static bool same_bits(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::uint32_t x, y;
        std::memcpy(&x, &a[i], 4);
        std::memcpy(&y, &b[i], 4);
        if (x != y) return false;
    }
    return true;
}
int main()
{
    using fused_decode_opt_test::Element;
    using fused_decode_opt_test::Fixture;
    require(fused_decode_opt_test::reference_half(1.9999f) == 0x4000);
    require(fused_decode_opt_test::reference_half(7.9999f) == 0x4800);
    require(fused_decode_opt_test::reference_half(1.00048828125f) == 0x3c00); // halfway, even
    require(fused_decode_opt_test::reference_half(1.00146484375f) == 0x3c02); // halfway, odd

    // ------------------------------------------------- the bf16 gold rounder, pinned two ways
    //
    // reference_bfloat16 splits the fp32 word and decides the round half-to-even explicitly;
    // ascend_micro::sim::float_to_bfloat16_rne does it by adding a bias and shifting. They are
    // different code, so agreeing is evidence. Checked on the tie cases first -- where the two
    // implementations would diverge if either had the half-to-even rule wrong -- and then over
    // every oracle value the fixtures below produce.
    //
    // bf16 keeps 7 stored significand bits, so above 1.0 the quantum is 2^-7 = 0.0078125 and the
    // tie sits at 2^-8 = 0.00390625.
    require(fused_decode_opt_test::reference_bfloat16(1.0f) == 0x3f80);
    require(fused_decode_opt_test::reference_bfloat16(2.0f) == 0x4000);
    require(fused_decode_opt_test::reference_bfloat16(15.0f) == 0x4170);
    require(fused_decode_opt_test::reference_bfloat16(1.00390625f) == 0x3f80);  // halfway, even
    require(fused_decode_opt_test::reference_bfloat16(1.01171875f) == 0x3f82);  // halfway, odd
    require(fused_decode_opt_test::reference_bfloat16(1.9999f) == 0x4000);      // carries into exp
    for (int i = 0; i <= 1792; ++i) {
        const float v = 1.0f + float(i) * (1.0f / 128.0f);  // 1 .. 15 on the bf16 grid and between
        require(fused_decode_opt_test::reference_bfloat16(v) ==
                ascend_micro::sim::float_to_bfloat16_rne(v));
        // Widening a stored word is exact in both directions on the grid itself.
        const auto bits = fused_decode_opt_test::reference_bfloat16(v);
        require(fused_decode_opt_test::widen_bits(Element::kBf16, bits) ==
                ascend_micro::sim::bfloat16_to_float(bits));
    }
    // ...and on the halfway points, where a wrong rule shows and a right one does not.
    for (int i = 0; i < 1792; ++i) {
        const float lo = 1.0f + float(i) * (1.0f / 128.0f);
        const float mid = lo + 1.0f / 256.0f;
        require(fused_decode_opt_test::reference_bfloat16(mid) ==
                ascend_micro::sim::float_to_bfloat16_rne(mid));
    }
    // reference_bits selects between them and nothing else.
    require(fused_decode_opt_test::reference_bits(Element::kFp16, 3.5f) ==
            fused_decode_opt_test::reference_half(3.5f));
    require(fused_decode_opt_test::reference_bits(Element::kBf16, 3.5f) ==
            fused_decode_opt_test::reference_bfloat16(3.5f));
    require(fused_decode_opt_test::element_mantissa_bits(Element::kBf16) == 7);
    require(fused_decode_opt_test::element_mantissa_bits(Element::kFp16) == 10);
    Fixture f(192, 2, 2, 3, 64);
    std::set<int> physical;
    for (int i = 0; i < f.requests * f.pages; ++i)
        require(f.table[i] != i && physical.insert(f.table[i]).second);
    for (int b = 0; b < f.requests; ++b) for (int h = 0; h < f.kv_heads; ++h)
        for (int t = 0; t < f.context; ++t) for (int d = 0; d < 256; ++d) {
            const auto page = f.table[b * f.pages + t / f.page_tokens];
            const auto row = t % f.page_tokens, col = d % 128;
            const auto offset = std::size_t(page) * f.page_tokens * f.kv_heads * 128 +
                (row / 64) * 64 * f.kv_heads * 128 + h * 64 * 128 +
                (col / 32 * 64 + row % 64) * 32 + col % 32;
            const unsigned nibble = (f.value[offset] >> (d < 128 ? 0 : 4)) & 15;
            require(nibble == Fixture::value_level(b, h, t / 64, d));
        }
    for (int k : {1, 2, 4, 8}) {
        const auto c = f.launch(k);
        require(c.grid == unsigned(2 * 2 * k) && c.tasks_per_block == 1);
        require(c.fused_context_limit == 0);
    }
    Fixture large(128, 16, 4, 16, 128);
    const auto c = large.launch(8);
    require(c.grid == 32 && c.tasks_per_block == 16);
    require(c.heads == 64 && c.heads_per_task == 16);
    for (float v : f.expected) require(std::isfinite(v) && v >= 1 && v <= 15);
    require(f.expected[0] != f.expected[256]); // different GQA query weights
    require(f.expected[0] != f.expected[3 * 256]); // different KV head
    require(f.expected[0] != f.expected[6 * 256]); // different request
    Fixture one(64, 1, 1, 1, 64);
    for (int d = 0; d < 256; ++d)
        require(one.expected[d] == 2 * Fixture::value_level(0, 0, 0, d) + 1);

    // ------------------------------------------------------------------ ragged contexts
    // The tile and page arithmetic, first. 79 tokens on 128-token pages is one page with 79
    // live rows, two Cube tiles, the second holding 15; 257 on 128-token pages is three pages
    // whose last holds a single token, which is also a single live row in its only tile; 1055
    // is nine pages with a 31-row tail.
    {
        Fixture r(79, 1, 1, 1, 128);
        require(r.pages == 1 && r.tiles() == 2);
        require(r.valid_rows(0) == 64 && r.valid_rows(1) == 15);
        require(r.page_rows(0) == 79);
        Fixture s257(257, 1, 1, 1, 128);
        require(s257.pages == 3 && s257.tiles() == 5);
        require(s257.valid_rows(3) == 64 && s257.valid_rows(4) == 1);
        require(s257.page_rows(0) == 128 && s257.page_rows(2) == 1);
        Fixture s1055(1055, 1, 1, 1, 128);
        require(s1055.pages == 9 && s1055.tiles() == 17);
        require(s1055.valid_rows(16) == 31 && s1055.page_rows(8) == 31);
        // An aligned context keeps full tiles everywhere, including the last.
        Fixture a(1024, 1, 1, 1, 128);
        require(a.pages == 8 && a.tiles() == 16 && a.valid_rows(15) == 64);
    }

    // THE ALIGNED ORACLE IS BIT-IDENTICAL to the unweighted form that preceded the ragged
    // support. Weighting every tile by its live row count multiplies both sums by exactly 64 at
    // an aligned context, and 64 is a power of two, so the quotient does not move a single bit.
    // Recomputed here with the OLD formula and compared bitwise, because "should be identical"
    // and "is identical" are different claims about floating point.
    {
        Fixture f2(192, 2, 2, 3, 64);
        std::size_t checked = 0;
        for (int request = 0; request < f2.requests; ++request)
            for (int qh = 0; qh < f2.kv_heads * f2.group_heads; ++qh) {
                const int head = qh / f2.group_heads, coefficient = 1 + qh % 3;
                const auto base = (std::size_t(request) * f2.kv_heads * f2.group_heads + qh) * 256;
                for (int col = 0; col < 256; ++col) {
                    double numerator = 0, denominator = 0;
                    for (int tile = 0; tile < f2.context / 64; ++tile) {
                        const double score =
                            (2 * Fixture::key_level(request, head, tile) + 1) * coefficient / 16.0;
                        const double weight = std::exp(score);
                        numerator += weight * (2 * Fixture::value_level(request, head, tile, col) + 1);
                        denominator += weight;
                    }
                    const float unweighted = static_cast<float>(numerator / denominator);
                    std::uint32_t a, b;
                    std::memcpy(&a, &unweighted, 4);
                    std::memcpy(&b, &f2.expected[base + col], 4);
                    require(a == b);
                    ++checked;
                }
            }
        require(checked == f2.expected.size());
    }

    // The ragged oracle, recomputed along a different rounding path: normalise the weights
    // first and accumulate a convex combination, rather than dividing two sums at the end. A
    // restatement of the fixture's own expression would check nothing; this agrees with it only
    // if the row weighting is right.
    {
        Fixture r(79, 1, 1, 1, 128);
        for (int col = 0; col < 256; ++col) {
            double total = 0;
            for (int tile = 0; tile < r.tiles(); ++tile)
                total += r.valid_rows(tile) *
                    std::exp((2 * Fixture::key_level(0, 0, tile) + 1) * 1 / 16.0);
            double average = 0;
            for (int tile = 0; tile < r.tiles(); ++tile) {
                const double share = r.valid_rows(tile) *
                    std::exp((2 * Fixture::key_level(0, 0, tile) + 1) * 1 / 16.0) / total;
                average += share * (2 * Fixture::value_level(0, 0, tile, col) + 1);
            }
            require(std::fabs(average - r.expected[col]) < 1e-5);
        }
        // The 15-row tail must actually count for less than a full tile would: the same
        // geometry with a full second tile is a different answer at the channels where the two
        // tiles disagree. Without the weighting these two fixtures would be identical.
        Fixture full(128, 1, 1, 1, 128);
        int differ = 0;
        for (int col = 0; col < 256; ++col) {
            require(Fixture::value_level(0, 0, 0, col) != Fixture::value_level(0, 0, 1, col) ||
                    r.expected[col] == full.expected[col]);
            if (r.expected[col] != full.expected[col]) ++differ;
        }
        require(differ > 200);
    }

    // A ragged context still maps every page, so the ladder's unmapped-page canaries survive,
    // and every oracle value stays inside reference_half's range.
    {
        Fixture r(1055, 1, 4, 16, 128);
        require(r.pages == 9 && int(r.table.size()) >= r.pages);
        for (int i = 0; i < r.pages; ++i) require(r.table[i] >= 0 && r.table[i] < r.physical_pages);
        for (float v : r.expected) require(std::isfinite(v) && v >= 1 && v <= 15);
        const auto c = r.launch(7);
        require(c.heads == 64 && c.max_blocks == 9 && c.splits == 7);
        for (float v : r.expected) (void)fused_decode_opt_test::reference_half(v);
    }

    // ------------------------------------------------------- the raw-query Pi basis
    // Pi = diag(s) . H . diag(s) / sqrt(D) with H the unnormalised Walsh-Hadamard. H is
    // symmetric and H^2 = D.I, so Pi is symmetric and Pi^2 = I EXACTLY -- which is the whole
    // reason the raw-query fixture can reuse the pre-rotated oracle. Checked, not assumed.
    {
        using fused_decode_opt_test::RawQueryBasis;
        const RawQueryBasis pi;
        require(pi.dim == 256);
        require(RawQueryBasis::inv_sqrt_len(256) == 1.0f / 16.0f);
        // The codec's pool is exactly ConstTableWords(256, 1) = 7*256 + 2*256 + 16 words.
        require(pi.tables().size() == 7u * 256 + 2u * 256 + 16);
        require(pi.h16().size() == 16u * 16);
        for (int c = 0; c < 256; c += 17) {
            std::vector<float> e(256, 0.0f);
            e[std::size_t(c)] = 1.0f;
            const auto once = pi.apply(e.data());
            const auto twice = pi.apply(once.data());
            for (int i = 0; i < 256; ++i) {
                require(twice[std::size_t(i)] == e[std::size_t(i)]);  // bitwise
                require(std::fabs(std::fabs(once[std::size_t(i)]) - 1.0f / 16.0f) < 1e-12);
            }
        }
        for (int a = 0; a < 8; ++a)
            for (int b = 0; b < 8; ++b) {
                std::vector<float> ea(256, 0.0f), eb(256, 0.0f);
                ea[std::size_t(a)] = 1.0f;
                eb[std::size_t(b)] = 1.0f;
                const auto ca = pi.apply(ea.data()), cb = pi.apply(eb.data());
                double dot = 0;
                for (int i = 0; i < 256; ++i) dot += double(ca[std::size_t(i)]) * cb[std::size_t(i)];
                require(std::fabs(dot - (a == b ? 1.0 : 0.0)) < 1e-9);
            }

        // The raw query the device's prologue is handed, and what it must produce from it: the
        // pre-rotated query, BIT FOR BIT. If this ever stops holding, the raw-query entry's
        // output can no longer be held to the pre-rotated entry's oracle at output stage 0.
        Fixture r(1024, 1, 1, 16, 128);
        const auto raw = r.raw_query(pi);
        require(raw.size() == r.query.size());
        for (std::size_t row = 0; row < r.query.size() / 256; ++row) {
            std::vector<float> back(256);
            for (int c = 0; c < 256; ++c)
                back[std::size_t(c)] =
                    ascend_micro::sim::half_to_float(raw[row * 256 + std::size_t(c)]);
            const auto rotated = pi.apply(back.data());
            for (int c = 0; c < 256; ++c)
                require(rotated[std::size_t(c)] == r.query[row * 256 + std::size_t(c)]);
        }
        // The three output stages, and the gate.
        const auto s0 = r.expected_row(pi, 0, 0), s1 = r.expected_row(pi, 0, 1),
                   s2 = r.expected_row(pi, 0, 2);
        for (int c = 0; c < 256; ++c) require(s0[std::size_t(c)] == r.expected[std::size_t(c)]);
        const auto again = pi.apply(s1.data());
        for (int c = 0; c < 256; ++c)
            require(std::fabs(again[std::size_t(c)] - s0[std::size_t(c)]) < 1e-4);
        int gated = 0;
        for (int c = 0; c < 256; ++c)
            if (s2[std::size_t(c)] != s1[std::size_t(c)]) ++gated;
        require(gated > 200);
        const auto gate = r.gate_logits();
        require(gate.size() == r.query.size());
        for (std::size_t i = 0; i < 64; ++i)
            require(ascend_micro::sim::half_to_float(gate[i]) == Fixture::gate_value(i));
        // The raw-query launch covers every query vector with its prologue share.
        const auto rc = r.launch_raw(2, 1, 0);
        require(rc.output_stage == 1 && rc.prologue_cube_chunk_vectors == 0);
        require(std::uint64_t(rc.grid) * rc.prologue_vectors_per_block >=
                std::uint64_t(r.requests) * rc.heads);
    }

    // ================================================= the bf16 instance of the gold model
    //
    // The element type must change the 16-bit ENCODINGS and nothing else: every fp32 value in the
    // fixture -- the query, the cache, the scales and `expected` -- is the same number for both
    // instances, because the device's arithmetic is identical up to the final output cast. Pinned
    // bitwise, so a future change that made `element` leak into the arithmetic is a failure here
    // rather than a quiet divergence between the two ladders.
    {
        Fixture a(192, 2, 2, 3, 64, Element::kFp16);
        Fixture b(192, 2, 2, 3, 64, Element::kBf16);
        require(a.element == Element::kFp16 && b.element == Element::kBf16);
        require(same_bits(a.query, b.query));
        require(same_bits(a.scales, b.scales));
        require(same_bits(a.expected, b.expected));  // bitwise, not "close"
        require(a.key == b.key && a.value == b.value);
        require(a.table == b.table && a.lengths == b.lengths);
        // The launch descriptor carries the instance, so a fixture built for one cannot be
        // launched at the other by accident.
        require(a.launch(2).element_type == 0 && b.launch(2).element_type == 1);
        require(a.launch_raw(2, 1, 0).element_type == 0);
        require(b.launch_raw(2, 1, 0).element_type == 1);
        // Both instances emit 16-bit elements, which is what holds the UB map still.
        require(ascend_micro::fused_decode_opt::element_bytes(0) == 2);
        require(ascend_micro::fused_decode_opt::element_bytes(1) == 2);
        // Every oracle value is in reference_bfloat16's range and rounds without throwing.
        for (float v : b.expected) (void)fused_decode_opt_test::reference_bfloat16(v);
    }

    // THE 16-BIT INPUTS THE DEVICE READS, in bf16. The raw query and the gate are the only two,
    // and the gold model refuses to ROUND either: Pi(raw) has to reproduce the pre-rotated query
    // bit for bit or the stage-0 oracle does not apply, and the kGated oracle has to divide by
    // the logit the device actually read. raw_query/gate_logits therefore throw on an inexact
    // value, and what this checks is that they DON'T have to -- that the fixture's own numbers
    // land on the bf16 grid, which is a measured property of this basis and not a bound.
    {
        using fused_decode_opt_test::RawQueryBasis;
        const RawQueryBasis pi;
        Fixture r(1024, 1, 1, 16, 128, Element::kBf16);
        const auto raw = r.raw_query(pi);          // throws if any entry is not bf16-exact
        const auto gate = r.gate_logits();         // likewise
        require(raw.size() == r.query.size() && gate.size() == r.query.size());

        // The widest significand the raw query actually uses, reported rather than assumed: the
        // margin against bf16's 7 stored bits is the whole reason this fixture works at bf16.
        unsigned widest = 0;
        for (std::size_t row = 0; row < r.query.size() / 256; ++row) {
            const auto image = pi.apply(r.query.data() + row * 256);
            for (int c = 0; c < 256; ++c) {
                std::uint32_t bits;
                std::memcpy(&bits, &image[std::size_t(c)], 4);
                unsigned used = 23, m = bits & 0x7fffffu;
                if (m == 0) used = 0;
                else while (!(m & 1u)) { m >>= 1; --used; }
                if (used > widest) widest = used;
            }
        }
        std::printf("fused-opt bf16 gold model: raw query needs %u significand bits, bf16 stores "
                    "%u\n", widest, fused_decode_opt_test::element_mantissa_bits(Element::kBf16));
        require(widest <= fused_decode_opt_test::element_mantissa_bits(Element::kBf16));

        // And the round trip the device's prologue has to reproduce: widen the bf16 raw query,
        // rotate it, and land on the pre-rotated query BITWISE. This is the claim the stage-0
        // oracle rests on, restated for bf16 exactly as the fp16 block above states it.
        for (std::size_t row = 0; row < r.query.size() / 256; ++row) {
            std::vector<float> back(256);
            for (int c = 0; c < 256; ++c)
                back[std::size_t(c)] = fused_decode_opt_test::widen_bits(
                    Element::kBf16, raw[row * 256 + std::size_t(c)]);
            const auto rotated = pi.apply(back.data());
            for (int c = 0; c < 256; ++c)
                require(rotated[std::size_t(c)] == r.query[row * 256 + std::size_t(c)]);
        }
        // The gate the oracle divides by IS the gate the device read.
        for (std::size_t i = 0; i < 256; ++i)
            require(r.gate_widened(i) == Fixture::gate_value(i));
        // The gated stage still moves most channels, so stage 2 is not silently stage 1.
        const auto s1 = r.expected_row(pi, 0, 1), s2 = r.expected_row(pi, 0, 2);
        int gated = 0;
        for (int c = 0; c < 256; ++c) if (s1[std::size_t(c)] != s2[std::size_t(c)]) ++gated;
        require(gated > 200);
    }

    std::puts("PASS fused-opt fixture: fragmented pages, independent NZ inverse, GQA/request "
              "variation, exact one-tile oracle, capped grid, ragged tile/page arithmetic, "
              "bit-identical aligned oracle, an independently recomputed ragged oracle, and a "
              "Pi basis that is an exact involution whose raw query rotates back to the "
              "pre-rotated query bitwise; and the bf16 instance of the same gold model -- two "
              "independent bf16 rounders agreeing on the grid and on every halfway point, fp32 "
              "values bit-identical across instances, and a bf16 raw query that is exact and "
              "rotates back to the pre-rotated query bitwise");
}
