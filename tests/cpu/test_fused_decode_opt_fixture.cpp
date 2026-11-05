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

    std::puts("PASS fused-opt fixture: fragmented pages, independent NZ inverse, GQA/request "
              "variation, exact one-tile oracle, capped grid, ragged tile/page arithmetic, "
              "bit-identical aligned oracle and an independently recomputed ragged oracle");
}
