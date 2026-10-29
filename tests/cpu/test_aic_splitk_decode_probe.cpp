#include "../common/aic_splitk_decode_fixture.hpp"
#include <cstdio>

using ascend_micro::sim::require;
namespace test = splitk_decode_test;

void check_merge(const test::Fixture& f, int k)
{
    for (int request = 0; request < f.requests; ++request)
        for (int head = 0; head < f.heads; ++head) {
            auto full = test::ReferenceAicPartial(f, request, head, 0, f.pages);
            std::vector<test::AicPartial> parts;
            for (int split = 0; split < k; ++split) {
                const auto r = turboquant_splitk::Partition(f.pages, k, split);
                parts.push_back(test::ReferenceAicPartial(f, request, head, r.begin, r.count));
            }
            for (int row = 0; row < 16; ++row) {
                double m = -3e38f, l = 0;
                std::vector<double> acc(256, 0);
                for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
                    const double next = std::max(m, it->m[row]);
                    const double a = std::exp(m - next), b = std::exp(it->m[row] - next);
                    l = l * a + it->l[row] * b;
                    for (int c = 0; c < 256; ++c) acc[c] = acc[c] * a + it->acc[row * 256 + c] * b;
                    m = next;
                }
                require(std::abs(l - full.l[row]) <= 1e-9 * full.l[row], "partial denominator combine failed");
                std::vector<float> normalized(256);
                for (int c = 0; c < 256; ++c) normalized[c] = static_cast<float>(acc[c] / l);
                const auto rotated = single_token_decode::DenseRotate(normalized.data());
                const auto base = ((request * f.heads + head) * 16 + row) * 256;
                for (int c = 0; c < 256; ++c) {
                    const float result = rotated[c] / (1.0f + std::exp(-f.gate[base + c]));
                    require(std::abs(result - f.expected[base + c]) < 2e-6f,
                            "partial combine differs from independent full-context reference");
                }
            }
        }
}

int main()
{
    try {
        test::Fixture small(48, 2, 2, 16, 16);
        const auto q = test::AicRawQuery(small);
        // All 16 rows are populated with distinct valid queries.
        for (int row = 0; row < 16; ++row)
            for (int other = 0; other < row; ++other) {
                bool differs = false;
                for (int c = 0; c < 256; ++c) differs |= q[row * 256 + c] != q[other * 256 + c];
                require(differs, "packed query rows are not distinct");
            }
        for (int k : {1, 2, 4, 12, 16}) check_merge(small, k);
        // A nonzero page window must exclude the preceding page numerically.
        const auto window = test::ReferenceAicPartial(small, 1, 1, 1, 2);
        const auto all = test::ReferenceAicPartial(small, 1, 1, 0, 3);
        require(window.l[0] != all.l[0], "page window is not observable");
        for (int n : {1024, 4096, 16384}) {
            test::Fixture f(n, 1, 1, 16, 128);
            for (int k : {1, 4, 16}) check_merge(f, k);
        }
        std::puts("PASS: 16 distinct M rows; uneven/empty temporal splits; multiple requests/heads; "
                  "nonzero page window; N=1024,4096,16384 K=1,4,16 full-context combines.");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
