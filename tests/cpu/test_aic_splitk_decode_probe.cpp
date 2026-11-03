#include "../common/aic_splitk_decode_fixture.hpp"
#include <cmath>
#include <cstdio>
#include <limits>

using ascend_micro::sim::require;
namespace test = splitk_decode_test;

// FP32 spacing at |value|, as tq_splitk_rig::UlpOf computes it on the device
// side. Duplicated rather than shared because that header pulls in acl.
double ulp_of(double value)
{
    const float magnitude = std::abs(static_cast<float>(value));
    if (magnitude == 0.0f) return std::nextafter(0.0f, 1.0f);
    return std::nextafter(magnitude, std::numeric_limits<float>::infinity()) -
           magnitude;
}

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

        // ------------------------------------------- the page-map modes
        // `PageMap` moves WHERE a logical token is stored without changing
        // WHAT it holds: the table entry and the data offsets both come from
        // `physical`, while the content is keyed by the LOGICAL page. So the
        // oracle's output must be IDENTICAL under both mappings -- and that
        // equality is the self-consistency check, because a permutation that
        // broke the table/data correspondence would move the numbers.
        //
        // This is also the ONLY property of the scattered mapping that can be
        // established without a device. What the mapping is for -- making a
        // kernel that ignores the block table read unrelated data rather than
        // the neighbouring page -- needs the kernel.
        int page_maps = 0;
        for (int pages : {4, 8}) {
            const int page_tokens = 16;
            test::Fixture reverse(pages * page_tokens, 1, 1, 16, page_tokens, -1,
                                  test::TailFill::kDirty, test::PageMap::kReverse);
            test::Fixture scattered(pages * page_tokens, 1, 1, 16, page_tokens, -1,
                                    test::TailFill::kDirty, test::PageMap::kScattered);
            require(reverse.pages == pages && scattered.pages == pages,
                    "page-map fixture geometry");
            bool identity = true, is_reverse = true, differs = false;
            std::vector<int> seen(static_cast<std::size_t>(pages), 0);
            for (int page = 0; page < pages; ++page) {
                const auto physical = scattered.table[static_cast<std::size_t>(page)];
                identity &= physical == static_cast<std::uint32_t>(page);
                is_reverse &= physical == static_cast<std::uint32_t>(pages - 1 - page);
                differs |= physical != reverse.table[static_cast<std::size_t>(page)];
                require(reverse.table[static_cast<std::size_t>(page)] ==
                            static_cast<std::uint32_t>(pages - 1 - page),
                        "the DEFAULT page map must still be the reverse permutation");
                require(physical < static_cast<std::uint32_t>(pages),
                        "a scattered physical page is out of range");
                seen[physical]++;
            }
            for (int page = 0; page < pages; ++page)
                require(seen[static_cast<std::size_t>(page)] == 1,
                        "the scattered page map is not a permutation");
            require(differs && !identity && !is_reverse,
                    "the scattered page map must be neither the identity nor the "
                    "reverse permutation");

            const auto a = test::ReferenceAicPartial(reverse, 0, 0, 0, pages);
            const auto b = test::ReferenceAicPartial(scattered, 0, 0, 0, pages);
            for (int row = 0; row < 16; ++row) {
                require(a.m[row] == b.m[row] && a.l[row] == b.l[row],
                        "the page map changed the oracle's (m, l)");
                for (int c = 0; c < 256; ++c)
                    require(a.acc[row * 256 + c] == b.acc[row * 256 + c],
                            "the page map changed the oracle's accumulator");
            }
            ++page_maps;
        }

        // ------------------------------- why the accumulator ULP is scaled
        // The device rigs report the accumulator deviation in ULP twice: at
        // the channel's own value, and at the row's accumulation scale. Only
        // the second carries a budget, and this section pins the reason.
        //
        // A row is a 256-channel sum with cancellation, so some channel always
        // lands far below the row's scale. For the SAME absolute deviation,
        // expressing it in ULP of that channel inflates it by
        // ulp(scale)/ulp(min |acc|) -- a pure magnitude artifact, carrying no
        // information about the kernel. Measured here, so the claim travels
        // with the code rather than living only in a report:
        //
        //   block=16  pages=4     scale 1.1496    min 0.0586743     32x
        //   block=16  pages=8     scale 1.20752   min 0.0286948     64x
        //   block=128 pages=4     scale 1.03564   min 0.00397417   256x
        //   block=128 pages=8     scale 3.03663   min 0.000642073  4096x
        //
        // Those amplifications are what the CAModel run actually produced as
        // worst per-channel ULP (48, 98, 247 for the first three rungs, on
        // absolute deviations that did NOT grow); the 0.00397417 channel above
        // is the very one the device reported. The fourth rung amplifies 4096x,
        // so a bit-correct kernel there lands near 1700 ULP -- which is why a
        // per-channel budget of 256 could not be kept.
        //
        // The assertion is that the amplification stays LARGE. If a future
        // fixture change made these accumulators well conditioned, the raw
        // metric would become usable and the rigs' comment would be stale.
        int amplifications = 0;
        for (int page_tokens : {16, 128}) {
            for (int pages : {4, 8}) {
                test::Fixture f(pages * page_tokens, 1, 1, 16, page_tokens, -1,
                                test::TailFill::kDirty, test::PageMap::kScattered);
                const auto ref = test::ReferenceAicPartial(f, 0, 0, 0, f.pages);
                double worst = 0, worst_scale = 0, worst_min = 0;
                for (int row = 0; row < 16; ++row) {
                    double scale = 0;
                    double smallest = std::numeric_limits<double>::infinity();
                    for (int c = 0; c < 256; ++c) {
                        const double a = std::abs(ref.acc[row * 256 + c]);
                        if (a > scale) scale = a;
                        if (a < smallest) smallest = a;
                    }
                    const double amp = ulp_of(scale) / ulp_of(smallest);
                    if (amp > worst) {
                        worst = amp;
                        worst_scale = scale;
                        worst_min = smallest;
                    }
                }
                require(worst >= 16.0,
                        "the accumulator ULP amplification collapsed: a raw "
                        "per-channel ULP budget may now be meaningful, so the "
                        "scaled-ULP rationale in the device rigs needs revisiting");
                require(worst_scale > worst_min,
                        "a row whose accumulator scale equals its smallest "
                        "channel is not the cancelling sum this reasons about");
                ++amplifications;
            }
        }

        std::puts("PASS: 16 distinct M rows; uneven/empty temporal splits; multiple requests/heads; "
                  "nonzero page window; N=1024,4096,16384 K=1,4,16 full-context combines.");
        std::printf("PASS page maps: %d geometries where the scattered block table is a "
                    "permutation, is neither the identity nor the reverse, and leaves the "
                    "FP64 oracle bit-identical to the default mapping\n", page_maps);
        std::printf("PASS acc ULP amplification: %d L1-reuse geometries where the worst "
                    "row amplifies a per-channel ULP distance at least 16x over a "
                    "scale-referenced one, so only the scaled metric can carry a budget\n",
                    amplifications);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
