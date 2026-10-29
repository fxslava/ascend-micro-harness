#include "../common/splitk_decode_fixture.hpp"

#include <cstdio>
#include <numeric>
#include <random>

using ascend_micro::sim::require;
namespace c = turboquant_splitk;

int main()
{
    try {
        // Exhaust all supported page counts, including K>P and nondivisible
        // ranges. A page is owned exactly once; empty ranges never underflow.
        for (unsigned pages = 1; pages <= 128; ++pages)
            for (unsigned k = 1; k <= 16; ++k) {
                std::vector<int> visited(pages, 0);
                for (unsigned split = 0; split < k; ++split) {
                    const auto r = c::Partition(pages, k, split);
                    require(r.count <= (pages + k - 1) / k, "oversized split");
                    for (unsigned p = r.begin; p < r.begin + r.count; ++p) {
                        require(p < pages, "partition beyond context");
                        ++visited[p];
                    }
                }
                for (int n : visited) require(n == 1, "missing/duplicate page");
            }

        // Real partial decodes over an uneven three-page sequence, merged
        // in shuffled arrival orders, against the independent dense FP64
        // full-context oracle. Covers empty first/middle/last arrivals.
        splitk_decode_test::Fixture f(384, 1);
        std::mt19937 rng(3510);
        for (int k : {1, 2, 4, 8, 12, 16}) {
            std::vector<single_token_decode::GoldenState> partial;
            for (int split = 0; split < k; ++split) {
                const auto r = c::Partition(f.pages, k, split);
                std::vector<int> strips;
                for (unsigned p = r.begin; p < r.begin + r.count; ++p)
                    for (int strip = 0; strip < 8; ++strip) strips.push_back(p * 8 + strip);
                partial.push_back(single_token_decode::GoldenDecode(0, strips));
            }
            for (int trial = 0; trial < 8; ++trial) {
                std::vector<int> order(k);
                std::iota(order.begin(), order.end(), 0);
                std::shuffle(order.begin(), order.end(), rng);
                for (int row = 0; row < 6; ++row) {
                    float m = -3e38f, l = 0;
                    std::vector<float> acc(256, 0);
                    for (int split : order) {
                        const auto& tile = partial[split];
                        // Empty GoldenDecode uses -inf; the device uses the
                        // finite neutral sentinel to avoid inf-inf on two empties.
                        const float tm = std::isfinite(tile.m[row]) ? tile.m[row] : -3e38f;
                        const float next = std::max(m, tm);
                        const float a = std::exp(m - next), b = std::exp(tm - next);
                        l = l * a + tile.l[row] * b;
                        for (int ch = 0; ch < 256; ++ch)
                            acc[ch] = acc[ch] * a + tile.acc[row * 256 + ch] * b;
                        m = next;
                    }
                    for (float& value : acc) value /= l;
                    const auto rotated = single_token_decode::DenseRotate(acc.data());
                    for (int ch = 0; ch < 256; ++ch) {
                        const auto at = row * 256 + ch;
                        const float got = rotated[ch] / (1 + std::exp(-f.gate[at]));
                        require(std::abs(got - f.expected[at]) < 2e-5f,
                                "split merge differs from full-context oracle");
                    }
                }
            }
        }

        // Exercise genuine long-context allocation and per-request table
        // stride. Every physical page is distinct across both requests.
        splitk_decode_test::Fixture large(4096, 2);
        auto pages = large.table;
        std::sort(pages.begin(), pages.end());
        for (unsigned p = 0; p < pages.size(); ++p)
            require(pages[p] == p, "physical pages alias across requests");
        const auto expected = large.expected;
        // A cyclic-wave implementation must be distinguishable numerically.
        for (int r = 0; r < large.requests; ++r)
            for (int p = 8; p < large.pages; ++p)
                large.table[r * large.pages + p] = large.table[r * large.pages + p % 8];
        large.Reference();
        double difference = 0;
        for (unsigned i = 0; i < expected.size(); ++i)
            difference = std::max(difference, static_cast<double>(std::abs(expected[i] - large.expected[i])));
        require(difference > 0.001, "fixture cannot detect cyclic page emulation");
        std::printf("PASS: 2048 page partitions; real partial merges K=1..16; "
                    "two-request 4096-token oracle; cyclic-page error %.6g\n", difference);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
