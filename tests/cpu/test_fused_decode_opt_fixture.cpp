#include "../common/fused_decode_opt_fixture.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>

static void require(bool b) { if (!b) throw std::runtime_error("fused-opt fixture test"); }
int main()
{
    using fused_decode_opt_test::Fixture;
    require(fused_decode_opt_test::reference_half(1.9999f) == 0x4000);
    require(fused_decode_opt_test::reference_half(7.9999f) == 0x4800);
    require(fused_decode_opt_test::reference_half(1.00048828125f) == 0x3c00); // halfway, even
    require(fused_decode_opt_test::reference_half(1.00146484375f) == 0x3c02); // halfway, odd
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

    std::puts("PASS fused-opt fixture: fragmented pages, independent NZ inverse, GQA/request "
              "variation, exact one-tile oracle, capped grid, ragged tile/page arithmetic, "
              "bit-identical aligned oracle, an independently recomputed ragged oracle, and a "
              "Pi basis that is an exact involution whose raw query rotates back to the "
              "pre-rotated query bitwise");
}
