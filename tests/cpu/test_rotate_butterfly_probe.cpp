// Tier-1 host probe for routines/routine_vector_rotate_butterfly.hpp.
//
// A CAModel run compares the device against an oracle, so a wrong oracle and a
// wrong kernel are indistinguishable there. Everything that can be settled
// without a simulator is settled here, in well under a second:
//
//   1. Pi is orthogonal:  Pi . Pi^T = 256 . I, entry for entry.
//   2. The two STAGE TABLES are right. The eight-stage cascade in each
//      direction's own order, with the sign diagonal at that direction's own
//      position, reproduces the dense matrix product bit for bit.
//   3. The two directions are genuinely different. Running the EGRESS table in
//      place of the ingress one must disagree -- otherwise the gate above
//      proves nothing about the ordering, which is the error this family is
//      most exposed to.
//   4. The three DEVICE ALGORITHMS agree with the cascade bit for bit:
//        - RegGather's lane-gather / register-pair split,
//        - RegShuffle's (P . B_0)^6 recursion with its P^5 sign frame,
//        - Approach A's UB offset and sign tables, applied the way
//          NativeRotateBlockedEngine indexes them.
//      Nothing here is taken on faith from the derivation in the header.
//   5. The round trip: (Pi . x) . Pi / 256 == x to 0 ULP.
//
// The fixture is dyadic, so every intermediate of every form is exact in FP32
// and bit equality is the right gate rather than a tolerance.

#include "../common/rotate_butterfly_oracle.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using namespace rot_oracle;

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

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b, int* first = nullptr)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (bits_of(a[i]) != bits_of(b[i])) {
            if (first) *first = static_cast<int>(i);
            return false;
        }
    return true;
}

std::uint32_t mix(std::uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

// Small signed integers scaled by a power of two: dyadic, so the 256-term
// signed sums the rotation forms are exact in FP32, and distinct per channel
// so no stage can be dropped without changing the answer.
std::vector<float> fixture(std::uint32_t seed)
{
    std::vector<float> x(kChannels);
    for (int c = 0; c < kChannels; ++c) {
        const std::uint32_t h = mix(seed ^ (static_cast<std::uint32_t>(c) * 0x9E3779B9u));
        const int n = static_cast<int>(h % 253u) - 126;
        x[c] = std::ldexp(static_cast<float>(n), -3);
    }
    return x;
}

// Approach A applied exactly as the device engine indexes its tables: the
// level-2 Gather reads BYTE offsets, the sign vector is a separate multiply,
// and the block stages are the plain strided butterfly.
std::vector<float> algo_blocked(Direction dir, const float* x, const float* signs,
                                float scale)
{
    const std::vector<std::uint32_t> offsets = blocked_gather_offsets();
    const std::vector<float> gsigns = blocked_gather_signs();
    std::vector<float> v(x, x + kChannels);

    auto gather_stage = [&](int index) {
        std::vector<float> permuted(kChannels), signed_tmp(kChannels), out(kChannels);
        const std::uint32_t* off = offsets.data() + static_cast<std::size_t>(index) * kChannels;
        const float* sg = gsigns.data() + static_cast<std::size_t>(index) * kChannels;
        for (int i = 0; i < kChannels; ++i) permuted[i] = v[off[i] / 4];
        for (int i = 0; i < kChannels; ++i) signed_tmp[i] = v[i] * sg[i];
        for (int i = 0; i < kChannels; ++i) out[i] = permuted[i] + signed_tmp[i];
        v.swap(out);
    };
    auto block_stage = [&](int stride) { butterfly_stage(v, stride); };
    auto sign_stage = [&]() {
        for (int i = 0; i < kChannels; ++i) v[i] *= signs[i];
    };

    if (dir == Direction::Egress) {
        block_stage(32);
        block_stage(64);
        block_stage(128);
        sign_stage();
        gather_stage(0);
        gather_stage(1);
        gather_stage(2);
        block_stage(8);
        block_stage(16);
    } else {
        gather_stage(0);
        gather_stage(1);
        gather_stage(2);
        block_stage(8);
        block_stage(16);
        sign_stage();
        block_stage(32);
        block_stage(64);
        block_stage(128);
    }
    for (int i = 0; i < kChannels; ++i) v[i] *= scale;
    return v;
}

void check_orthogonality(const std::vector<float>& signs)
{
    long long worst = 0;
    bool ok = true;
    for (int r = 0; r < kChannels && ok; ++r)
        for (int c = 0; c < kChannels; ++c) {
            long long acc = 0;
            for (int k = 0; k < kChannels; ++k)
                acc += static_cast<long long>(dense_entry(signs.data(), r, k)) *
                       static_cast<long long>(dense_entry(signs.data(), c, k));
            const long long want = (r == c) ? kChannels : 0;
            if (acc != want) {
                std::fprintf(stderr, "  Pi Pi^T [%d][%d] = %lld, want %lld\n", r, c, acc, want);
                ok = false;
                break;
            }
            if (acc > worst) worst = acc;
        }
    expect(ok, "Pi . Pi^T == 256 . I");
    std::printf("  Pi . Pi^T diagonal %lld over %d x %d entries\n", worst, kChannels,
                kChannels);
}

}  // namespace

int main()
{
    const std::vector<float> signs = canonical_signs();
    std::printf("rotate butterfly Tier-1 probe: d = %d, %d stages, %d registers x %d lanes\n",
                kChannels, kStages, kRegisters, kVecLanes);

    // The sign diagonal must not be degenerate: an all-+1 or near-all-+1 vector
    // would let a dropped sign stage pass every check below.
    int negatives = 0;
    for (int c = 0; c < kChannels; ++c)
        if (signs[c] < 0.0f) ++negatives;
    expect(negatives > kChannels / 4 && negatives < 3 * kChannels / 4,
           "the sign diagonal is balanced enough to detect a dropped stage");
    std::printf("  sign diagonal: %d of %d negative\n", negatives, kChannels);

    check_orthogonality(signs);

    for (std::uint32_t seed = 1; seed <= 8; ++seed) {
        const std::vector<float> x = fixture(seed * 0x51ED2701u);

        // ---- 2. the stage tables against the dense definition --------------
        const std::vector<float> want_left = dense_left(x.data(), signs.data());
        const std::vector<float> want_right = dense_right(x.data(), signs.data());
        const std::vector<float> got_left = staged(Direction::Ingress, x.data(), signs.data());
        const std::vector<float> got_right = staged(Direction::Egress, x.data(), signs.data());
        int where = -1;
        expect(bit_equal(want_left, got_left, &where),
               "ingress stage table reproduces Pi . x bit-exactly");
        expect(bit_equal(want_right, got_right, &where),
               "egress stage table reproduces x . Pi bit-exactly");

        // ---- 3. the two directions are not interchangeable ------------------
        if (seed == 1) {
            const std::vector<float> wrong_order =
                staged(Direction::Egress, x.data(), signs.data());
            expect(!bit_equal(want_left, wrong_order),
                   "the egress table does NOT compute Pi . x (the orders differ)");
            // and the one-line version of the same trap: the right table with
            // the wrong sign position.
            std::vector<float> value(x.begin(), x.end());
            for (int stage = 0; stage < kStages; ++stage) {
                if (stage == kSignAfterStage[0])  // egress position, ingress table
                    for (int i = 0; i < kChannels; ++i) value[i] *= signs[i];
                butterfly_stage(value, kStrideOf[1][stage]);
            }
            expect(!bit_equal(want_left, value),
                   "the ingress table with the egress sign position is NOT Pi . x");
        }

        // ---- 4. the device algorithms against the cascade -------------------
        expect(bit_equal(got_left,
                         algo_reg_gather(Direction::Ingress, x.data(), signs.data()), &where),
               "RegGather ingress == the stage cascade");
        expect(bit_equal(got_right,
                         algo_reg_gather(Direction::Egress, x.data(), signs.data()), &where),
               "RegGather egress == the stage cascade");
        expect(bit_equal(got_left, algo_reg_shuffle(x.data(), signs.data()), &where),
               "RegShuffle ingress == the stage cascade ((P.B_0)^6 == B_5...B_0)");
        expect(bit_equal(got_left,
                         algo_blocked(Direction::Ingress, x.data(), signs.data(), 1.0f),
                         &where),
               "Blocked ingress == the stage cascade");
        expect(bit_equal(got_right,
                         algo_blocked(Direction::Egress, x.data(), signs.data(), 1.0f),
                         &where),
               "Blocked egress == the stage cascade");

        // ---- 5. the round trip ----------------------------------------------
        const std::vector<float> back =
            staged(Direction::Egress, got_left.data(), signs.data(), 1.0f / 256.0f);
        expect(bit_equal(x, back, &where), "(Pi . x) . Pi / 256 == x to 0 ULP");

        // The rotated panel must stay inside 2^24, or none of the bit-equality
        // gates above would be meaningful on the device either.
        float peak = 0.0f;
        for (float value : got_left) peak = std::fmax(peak, std::fabs(value));
        expect(peak < 16777216.0f, "the rotated panel is exactly representable in FP32");
        if (seed == 1)
            std::printf("  peak |Pi . x| over the fixture: %.1f (FP32 exact below 2^24)\n",
                        static_cast<double>(peak));
    }

    // The scale is applied last and is a power of two, so it cannot perturb any
    // of the above -- state it rather than assume it.
    {
        const std::vector<float> x = fixture(0xABCDEF01u);
        const std::vector<float> plain = staged(Direction::Ingress, x.data(), signs.data());
        const std::vector<float> scaled =
            staged(Direction::Ingress, x.data(), signs.data(), 1.0f / 16.0f);
        bool exact = true;
        for (int i = 0; i < kChannels; ++i)
            if (bits_of(scaled[i]) != bits_of(plain[i] / 16.0f)) exact = false;
        expect(exact, "the 1/16 orthonormalisation is an exact trailing scale");
    }

    if (failures == 0) {
        std::printf("rotate butterfly probe: PASS\n");
        return 0;
    }
    std::fprintf(stderr, "rotate butterfly probe: %d FAILURES\n", failures);
    return 1;
}
