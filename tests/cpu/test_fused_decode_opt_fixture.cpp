#include "../common/fused_decode_opt_fixture.hpp"
#include <cstdio>
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
    std::puts("PASS fused-opt fixture: fragmented pages, independent NZ inverse, GQA/request variation, exact one-tile oracle, capped grid");
}
