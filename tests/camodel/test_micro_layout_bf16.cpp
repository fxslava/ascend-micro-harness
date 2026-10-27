// Single-tile CAModel probe for the BF16 Cube operand layouts, Ascend 950PR /
// arch35 / dav-3510.
//
// The question
// ------------
// turboquant_reshape_and_cache_bf16.cpp is the INT8 prefill operator with one
// element widened from 1 byte to 2. The INT8 version rests on two "the same
// bytes read two ways" identities, both of which the widening puts in doubt
// because the arch-3510 fractal is 16 x (32B / sizeof(T)) and C0 drops from 32
// to 16, so a 32-wide axis stops being one fractal and becomes two:
//
//   (P) the 16 x 256 activation page, staged Nz, read as Nz(16, 32) for the
//       A role of a key chunk and as Zn(32, 16) for the B role of a value
//       chunk;
//   (B) the 32 x 32 resident rotation block, staged as a flat row-major blob,
//       read as Nz(32, 32) from L0A and Zn(32, 32) from L0B.
//
// This probe answers both in one AIC-only kernel of sixteen Mmads, with no
// Vector core, no Unified Buffer, no cross-core flag and no pipeline ring, so
// a mismatch cannot be a synchronisation artefact. It computes one
// (chunk, output-group) cell of the production GEMM in both orientations and
// compares every FP32 result bit-exactly against the shared oracle in
// tests/common/turboquant_oracle.hpp.
//
// The four cases, and what each one isolates
// ------------------------------------------
//   1 flat / row-major     the production prologue and the production blob.
//                          EXPECTED TO FAIL: this is the hypothesis.
//   2 nd2nz / row-major    the same blob through `stage_blocks_nz`, the
//                          ingress fix. EXPECTED TO PASS.
//   3 flat / Nz-imaged     the production prologue over a blob the host has
//                          already put in Nz order. EXPECTED TO PASS, and it
//                          attributes case 1 from the other side: if the
//                          ingress is the only thing wrong, fixing it on
//                          either side must be enough.
//   4 flat / Nz identity   block 0 is the identity in Nz order and blocks
//                          1..7 are zero, so C_v[c0][t] must equal
//                          x[t][c0] with no dependence on the block content
//                          at all. This isolates (P): it is the page's Nz/Zn
//                          duality on its own.
//
// Case 1 does not merely have to fail. The runner predicts, from the Nz
// index map alone, which entries the Cube will read out of a row-major blob,
// recomputes the oracle with that permuted operand, and requires the device
// to match THAT bit-exactly. A diagnosis that predicts the wrong answer
// exactly is a mechanism; one that only predicts "wrong" is a guess.

#include "camodel_common.hpp"

#include "../common/turboquant_oracle.hpp"
#include "../../routines/routine_stage_page_zn_bf16.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

extern "C" void launch_micro_layout_bf16(void* stream, const std::uint16_t* page,
                                         const std::uint16_t* table, float* out_value,
                                         float* out_key, std::uint32_t row_pitch,
                                         std::uint32_t table_nd2nz);

namespace {

constexpr int kTokens = 16;
constexpr int kHeadSize = 256;
constexpr int kGroup = 32;
constexpr int kGroups = kHeadSize / kGroup;  // 8
constexpr int kHeads = 4;
constexpr int kChannels = kHeads * kHeadSize;  // the activation row pitch
constexpr int kBlockElems = kGroup * kGroup;

void require(bool ok, const char* what)
{
    if (!ok) {
        std::fprintf(stderr, "CHECK FAILED: %s\n", what);
        throw std::runtime_error(what);
    }
}

// An integer as a BF16 bit pattern, exact or not at all.
std::uint16_t to_bf16(int value)
{
    const std::uint32_t bits = tq_oracle::float_bits(static_cast<float>(value));
    if ((bits & 0xFFFFu) != 0u) throw std::runtime_error("operand is not exact in BF16");
    return static_cast<std::uint16_t>(bits >> 16);
}

std::uint32_t nz_index(int r, int c)
{
    return ascend_micro::Bf16NzBlockIndex(static_cast<std::uint32_t>(kGroup),
                                          static_cast<std::uint32_t>(r),
                                          static_cast<std::uint32_t>(c));
}

// Y[c0][t] = sum_g sum_i blocks[g][c0][i] * page[t][g*32 + i], the cell the
// Cube computes. Exact in FP32: |blocks| <= 1, |page| <= 128, 256 terms, so
// every partial sum is an integer below 2^24 whatever order the Cube reduces
// in. Returned as int, and compared after one exact widening to float.
std::vector<int> oracle_cell(const std::vector<int>& blocks, const std::vector<int>& page)
{
    std::vector<int> y(static_cast<std::size_t>(kGroup) * kTokens, 0);
    for (int c0 = 0; c0 < kGroup; ++c0)
        for (int t = 0; t < kTokens; ++t) {
            int acc = 0;
            for (int g = 0; g < kGroups; ++g)
                for (int i = 0; i < kGroup; ++i)
                    acc += blocks[(static_cast<std::size_t>(g) * kGroup + c0) * kGroup + i] *
                           page[static_cast<std::size_t>(t) * kHeadSize + g * kGroup + i];
            y[static_cast<std::size_t>(c0) * kTokens + t] = acc;
        }
    return y;
}

struct Outcome {
    bool value_exact = false;
    bool key_exact = false;
    int first_c0 = -1;
    int first_t = -1;
    float got = 0.0f;
    int want = 0;
    std::size_t matched = 0;
};

// out_value is ND(32, 16): C_v[c0][t]. out_key is ND(16, 32): C_k[t][c0].
// They are the same numbers transposed, so one expectation checks both.
Outcome compare(const float* out_value, const float* out_key, const std::vector<int>& want)
{
    Outcome r;
    r.value_exact = true;
    r.key_exact = true;
    for (int c0 = 0; c0 < kGroup; ++c0)
        for (int t = 0; t < kTokens; ++t) {
            const int expect = want[static_cast<std::size_t>(c0) * kTokens + t];
            const float v = out_value[static_cast<std::size_t>(c0) * kTokens + t];
            const float k = out_key[static_cast<std::size_t>(t) * kGroup + c0];
            const bool ok_v = v == static_cast<float>(expect);
            const bool ok_k = k == static_cast<float>(expect);
            if (ok_v && ok_k) ++r.matched;
            if (!ok_v && r.value_exact) {
                r.first_c0 = c0;
                r.first_t = t;
                r.got = v;
                r.want = expect;
            }
            r.value_exact = r.value_exact && ok_v;
            r.key_exact = r.key_exact && ok_k;
        }
    return r;
}

struct Case {
    const char* name;
    std::uint32_t nd2nz;
    const std::vector<int>* blob;      // what the host uploads, row-major order
    const std::vector<int>* expected;  // the oracle this case must reproduce
    bool must_pass;
};

int run()
{
    // ---- operands, from the shared fixture -------------------------------
    tq_oracle::Shape shape;
    require(shape.valid(), "unsupported oracle shape");
    require(shape.groups() == kGroups, "the oracle's group count moved");

    std::vector<std::int8_t> key_in, value_in;
    tq_oracle::build_activations(shape, key_in, value_in);
    const std::vector<std::int8_t> table_blob = tq_oracle::build_tables(shape);
    require(tq_oracle::audit_tables(shape, table_blob.data()) ==
                static_cast<long long>(kHeadSize) * kHeadSize,
            "the resident blocks do not compose to the oracle's rotation matrix");

    // The value plane of head 0, strip 0: production chunk 0. The page the
    // Cube sees is the headSize-wide window of a 16 x 1024 token-major block.
    std::vector<int> page(static_cast<std::size_t>(kTokens) * kHeadSize, 0);
    for (int t = 0; t < kTokens; ++t)
        for (int c = 0; c < kHeadSize; ++c)
            page[static_cast<std::size_t>(t) * kHeadSize + c] =
                value_in[static_cast<std::size_t>(t) * kChannels + c];

    // Output group c1 = 0 takes blocks parity(0 & g) * 8 + g = g.
    std::vector<int> blocks_row(static_cast<std::size_t>(kGroups) * kBlockElems, 0);
    for (int g = 0; g < kGroups; ++g)
        for (int c0 = 0; c0 < kGroup; ++c0)
            for (int i = 0; i < kGroup; ++i)
                blocks_row[(static_cast<std::size_t>(g) * kGroup + c0) * kGroup + i] =
                    table_blob[(static_cast<std::size_t>(g) * kGroup + c0) * kGroup + i];

    // Fixture strength. A blob whose blocks were all the same, or a page whose
    // tokens were, could not distinguish a wrong-group or wrong-token read
    // from a correct one. Assert both rather than describe them.
    //
    // Not pairwise distinctness of the blocks: T[0][g] depends on g only
    // through piSigns[g*32 + c0], and pi_sign has period 7 in the channel
    // while 32 == 4 (mod 7), so blocks 0 and 7 are legitimately equal. What
    // the probe needs is that the operand varies with g at all -- the stronger
    // property, that a misread of a single block changes the answer, is the
    // `want_misread != want_true` assertion below.
    {
        bool blocks_vary = false, tokens_distinct = true;
        for (int g = 1; g < kGroups && !blocks_vary; ++g)
            for (int e = 0; e < kBlockElems && !blocks_vary; ++e)
                blocks_vary = blocks_row[static_cast<std::size_t>(g) * kBlockElems + e] !=
                              blocks_row[static_cast<std::size_t>(e)];
        for (int t = 1; t < kTokens && tokens_distinct; ++t) {
            bool same = true;
            for (int c = 0; c < kHeadSize && same; ++c)
                same = page[static_cast<std::size_t>(t) * kHeadSize + c] == page[c];
            tokens_distinct = !same;
        }
        require(blocks_vary, "every resident block is the same, so group order is untested");
        require(tokens_distinct, "two page tokens are identical");
    }

    // ---- the three operand images the four cases upload ------------------
    // Nz image of the real blocks: entry (c0, i) moves to nz_index(c0, i).
    std::vector<int> blocks_nz(blocks_row.size(), 0);
    for (int g = 0; g < kGroups; ++g)
        for (int c0 = 0; c0 < kGroup; ++c0)
            for (int i = 0; i < kGroup; ++i)
                blocks_nz[static_cast<std::size_t>(g) * kBlockElems + nz_index(c0, i)] =
                    blocks_row[(static_cast<std::size_t>(g) * kGroup + c0) * kGroup + i];

    // Nz image of "block 0 is the identity, blocks 1..7 are zero".
    std::vector<int> blocks_eye(blocks_row.size(), 0);
    for (int c0 = 0; c0 < kGroup; ++c0) blocks_eye[nz_index(c0, c0)] = 1;

    // ---- the three oracles -----------------------------------------------
    const std::vector<int> want_true = oracle_cell(blocks_row, page);

    // What the Cube reads out of a row-major blob: the entry the Nz map puts
    // at (c0, i) is whatever row-major left at that offset.
    std::vector<int> blocks_misread(blocks_row.size(), 0);
    for (int g = 0; g < kGroups; ++g)
        for (int c0 = 0; c0 < kGroup; ++c0)
            for (int i = 0; i < kGroup; ++i) {
                const std::uint32_t off = nz_index(c0, i);
                blocks_misread[(static_cast<std::size_t>(g) * kGroup + c0) * kGroup + i] =
                    blocks_row[static_cast<std::size_t>(g) * kBlockElems + off];
            }
    const std::vector<int> want_misread = oracle_cell(blocks_misread, page);
    std::size_t separated = 0;
    for (std::size_t i = 0; i < want_true.size(); ++i)
        if (want_true[i] != want_misread[i]) ++separated;
    require(separated != 0,
            "the row-major and Nz readings of the blob agree, so this probe proves nothing");
    std::printf("fixture: the two operand images disagree on %zu of %zu cell entries\n",
                separated, want_true.size());

    std::vector<int> eye_row(blocks_row.size(), 0);
    for (int c0 = 0; c0 < kGroup; ++c0)
        eye_row[static_cast<std::size_t>(c0) * kGroup + c0] = 1;
    const std::vector<int> want_eye = oracle_cell(eye_row, page);

    // ---- device buffers ---------------------------------------------------
    camodel::Session session;
    camodel::GuardedDeviceBuffer page_buf(key_in.size() * sizeof(std::uint16_t));
    camodel::GuardedDeviceBuffer table_buf(blocks_row.size() * sizeof(std::uint16_t));
    camodel::GuardedDeviceBuffer out_value(static_cast<std::size_t>(kGroup) * kTokens *
                                           sizeof(float));
    camodel::GuardedDeviceBuffer out_key(static_cast<std::size_t>(kTokens) * kGroup *
                                         sizeof(float));

    auto* pb = static_cast<std::uint16_t*>(page_buf.host_payload());
    for (std::size_t i = 0; i < value_in.size(); ++i) pb[i] = to_bf16(value_in[i]);
    page_buf.upload();

    const Case cases[] = {
        {"flat ingress, row-major blob", 0u, &blocks_row, &want_misread, true},
        {"nd2nz ingress, row-major blob", 1u, &blocks_row, &want_true, true},
        {"flat ingress, host Nz blob", 0u, &blocks_nz, &want_true, true},
        {"flat ingress, Nz identity blob", 0u, &blocks_eye, &want_eye, true},
    };

    int failures = 0;
    for (const Case& c : cases) {
        auto* tb = static_cast<std::uint16_t*>(table_buf.host_payload());
        for (std::size_t i = 0; i < c.blob->size(); ++i) tb[i] = to_bf16((*c.blob)[i]);
        table_buf.upload();

        launch_micro_layout_bf16(session.stream,
                                 static_cast<const std::uint16_t*>(page_buf.payload()),
                                 static_cast<const std::uint16_t*>(table_buf.payload()),
                                 static_cast<float*>(out_value.payload()),
                                 static_cast<float*>(out_key.payload()),
                                 static_cast<std::uint32_t>(kChannels), c.nd2nz);
        camodel::check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");

        page_buf.download();
        table_buf.download();
        out_value.download();
        out_key.download();
        page_buf.assert_guards();
        table_buf.assert_guards();
        out_value.assert_guards();
        out_key.assert_guards();

        const Outcome r = compare(static_cast<const float*>(out_value.host_payload()),
                                  static_cast<const float*>(out_key.host_payload()),
                                  *c.expected);
        const std::size_t total = static_cast<std::size_t>(kGroup) * kTokens;
        if (r.value_exact && r.key_exact) {
            std::printf("PASS %-32s %zu/%zu exact in both orientations\n", c.name, total,
                        total);
        } else {
            ++failures;
            std::printf("FAIL %-32s %zu/%zu exact (value %s, key %s)\n", c.name, r.matched,
                        total, r.value_exact ? "ok" : "wrong", r.key_exact ? "ok" : "wrong");
            if (r.first_c0 >= 0)
                std::printf("     first divergence c0=%d t=%d got %.1f want %d\n", r.first_c0,
                            r.first_t, static_cast<double>(r.got), r.want);
        }
    }

    if (failures != 0) {
        std::fprintf(stderr,
                     "FAIL micro layout bf16: %d of 4 cases did not match their "
                     "predicted operand image\n",
                     failures);
        return 1;
    }
    std::printf(
        "PASS micro layout bf16: the page's Nz/Zn duality holds at 16 bits; the 32x32 "
        "resident block needs an Nz ingress and the row-major misread is reproduced "
        "exactly\n");
    return 0;
}

}  // namespace

int main()
{
    try {
        return run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL micro layout bf16: %s\n", error.what());
        return 1;
    }
}
