// Physical-device WRITE -> READ round trip for the TurboQuant KV cache.
// =====================================================================
// One process, two real operators, one cache:
//
//   WRITE  turboquant_reshape_and_cache_bf16, built with TURBOQUANT_INGRESS_NZ, quantises
//          BF16 activations and scatters INT4 nibbles plus power-of-two scales into physical
//          pages through a fragmented slot mapping.
//   READ   turboquant_fused_decode (turboquant_mm_fused_decode_opt_kv4fp8_*) walks the same
//          pages through a block table and computes paged attention over the state the
//          prefill just left -- nothing between the two launches but a stream synchronise.
//
// WHY THIS TARGET EXISTS. The strip contract the prefill ships (RFC 3.1) and the NZ-tiled
// contract the fused decode reads name different bytes, with co-byte matchings that share not
// one pair -- tests/cpu/test_turboquant_kv_layout_bridge.cpp check 3 proves it, and that is
// why no round trip was expressible before. TURBOQUANT_INGRESS_NZ is the instance that closes
// it: check 1 of the same test is the gate on the writer's address arithmetic naming THE SAME
// BYTE AND THE SAME NIBBLE the decode reads. This file is the hardware end of that: the two
// operators actually handing each other a cache.
//
// HARDWARE ONLY. The development host for this harness has no NPU (`npu-smi` and
// `/dev/davinci*` are both absent), so what CI can establish for this file is a clean strict
// build and nothing about either kernel. See "WHAT THIS DOES NOT ESTABLISH" at the end of this
// header for what a passing hardware run does and does not buy, under
// REFINEMENT_OF_COMBAT_KERNELS.md.
//
// THE TWO BLOCK SIZES, WHICH ARE DIFFERENT NUMBERS
// ------------------------------------------------
// The cache geometry is 64-token physical pages: that is the fused decode's `block_size`, and
// Launch::block_size below carries exactly 64. The prefill quantises in 16-token STRIPS --
// kTokens, a compile-time constant of that kernel -- and four strips share one page.
//
// The prefill's `block_size` LAUNCH SCALAR is therefore 16, not 64, and that is deliberate
// rather than a transcription slip. Under the strip contract a strip filled a whole page, so
// the kernel's trip-count arithmetic uses that one scalar for both meanings:
//
//     share = tokensPerCore / blockSize;   live = (numTokens - strip) / blockSize
//
// which is a count of STRIPS. Passing 64 there would make `share` 1 and each grid block would
// write 16 of its 64 tokens, leaving three quarters of every page canary -- a short cache with
// every other byte bit-exact. The 64-token page reaches the prefill the only way the NZ
// instance needs it to: through the slot mapping, whose entries are GLOBAL slots
// (physical_page * 64 + row). The kernel's own comment says so -- "this instance needs no
// num_blocks and no blockSize in its addressing beyond the page it is given". So:
//
//     tokens_per_core = 64  one whole physical page per grid block
//     block_size      = 16  the strip width, which is what the trip count means by it
//     grid            = total tokens / 64
//
// Gate A below is what makes that reading checkable rather than asserted: it compares the
// whole cache image byte for byte, so a wrong strip count shows up as canary bytes where
// nibbles belong.
//
// WHAT EACH LAUNCH PAIR CHECKS
//
// GATE A -- THE WRITE, BIT-EXACT, 0 ULP. Every byte of both packed planes and every bit of
// every scale float against tests/common/turboquant_nz_ingress_oracle.hpp's golden NZ image,
// which places tests/common/turboquant_oracle.hpp's layout-independent INT4 codes and
// power-of-two scales through the PRODUCTION address functions
// (NzTiledChannelByte / NzTiledChannelNibble / NzTiledScaleIndex). This is an equality, not a
// tolerance: the quantiser is integer arithmetic over integer-valued BF16 activations, exactly
// as tests/npu/validation/test_turboquant_bf16_validation.cpp relies on.
//
// Which bytes the golden claims were WRITTEN is derived rather than assumed: the image is built
// twice, once flooded 0x00 and once 0xFF (scales 0.0f and 1.0f), and a byte that agrees between
// the two builds is one a strip wrote. Every other byte must still carry the 0x7FC0BEEF
// pre-fill, which covers in one comparison the unmapped physical pages, the rows of a mapped
// page no strip owns, and any over-wide MTE3 burst.
//
// GATE B -- THE READ, AGAINST AN FP64 ORACLE OVER THE BYTES THE DEVICE ACTUALLY WROTE. The
// decode's reconstruction is the one tests/cpu/test_turboquant_e2e_reference.cpp pins against
// the fused-decode fixture's own expectation:
//
//     value = (signed_nibble + 0.5) * stored_scale / kGain,  score scaled by inv_sqrt_len
//
// The oracle re-reads the DOWNLOADED cache, not the golden image, so Gate B is a statement
// about the decode alone even if Gate A were to fail; and it is scalar FP64, so it is an
// independent model rather than a transcription.
//
// IT IS A TOLERANCE, AND HERE IS WHY. The fused decode's own hardware ladder
// (test_turboquant_fused_decode_opt_npu.cpp) runs at ZERO fp16 ULP, because its fixture is
// built so every value is a small integer and the whole accumulation is exact. This one cannot
// be: kGain is 2.9832882881, so a dequantised element is not representable in binary, and the
// device sums 256 such products in FP32 where this model sums them in FP64. The gate is a ULP
// budget per element format, reported next to the achieved worst, and the budgets below are
// DERIVED from that arithmetic (a 256-term FP32 dot is ~sqrt(256) * 2^-24 relative, carried
// through a softmax of O(1) spread, against ~2^-11 per fp16 ULP) -- they are not measured on
// silicon. A hardware run that reports a worse figure is data about the budget, not
// necessarily about the kernel; raise it with --ulp-fp16 / --ulp-bf16 and say so in the report.
//
// GATE C -- DISCRIMINATION, measured rather than argued. A constant query vector cannot tell a
// correct key channel order from any permutation of it, because every permutation gives the
// same dot product -- and the NZ contract's distinguishing feature is precisely that one byte
// carries channels (c, c + 128) of one token. So the query here VARIES per channel, and Gate C
// proves that variation is load bearing: it recomputes the oracle twice more over the same
// cache, once with the query's two halves swapped (the NZ nibble pairing) and once with the
// channel order rotated by one (general channel order), and requires each to differ from the
// true oracle by far more than the Gate B budget. If either control came out inside the
// budget, Gate B would be passing vacuously and this run says so instead of passing.
//
// CANARIES, all required intact after both launches:
//   * 0x5A5A5A5A guard bands, 64 bytes on each side of every one of the device allocations.
//     64 keeps every payload 32-byte aligned (tq_npu::Buffer requires it) and is wider than one
//     UB block, so a single over-length burst cannot step past a band unseen.
//   * 0x7FC0BEEF over every cache byte and scale float no strip writes -- see Gate A.
//   * the output allocation's tail past the last live channel.
//   * the Split-K workspace: at S > 1 every float of the live partial extent must come back
//     finite, and at S == 1 the WHOLE workspace must still be canary, because the operator
//     takes its fused path there and NeedsReduction() is false.
//   * every INPUT allocation byte-identical afterwards, which also means the decode did not
//     write the cache the prefill filled.
//
// THE LADDER. 4 KV heads throughout -- the prefill's head count is a compile-time constant
// (kHeads), so it is not a free parameter of this round trip -- D = 256, 64-token pages, and a
// fragmented block table (logical page p of request r lives at physical 2*(r*pages+p)+1, so the
// physical page index is never the logical one and half the physical pages stay canary).
//
//   1.  1 request,   64 tokens, 2 query heads/KV head, S=1          one page, one tile; the
//                                                                   fused path, workspace
//                                                                   untouched
//   2.  1 request,  256 tokens, 2 query heads/KV head, S=1,2,4      four pages, split sweep
//   3.  2 requests, 128 tokens, 2 query heads/KV head, S=1,2        two requests, two block
//                                                                   table rows, one cache
//   4.  1 request, 1024 tokens, 2 query heads/KV head, S=1,2,4,8    16 pages: 16 prefill grid
//                                                                   blocks, and at S=1 one
//                                                                   decode worker walks 16
//                                                                   tiles, wrapping the
//                                                                   three-slot ring five times
//   5.  1 request,  128 tokens, 16 query heads/KV head, S=1,2       64 query heads over 4 KV
//                                                                   heads: full GQA width
//
// Both element instances of the decode (fp16 and bf16) read the SAME cache on every rung: the
// prefill is element-independent, so the write runs once per rung and the read runs twice.
//
// WHAT THIS DOES NOT ESTABLISH
//   * No second PREFILL execution wave. Every rung's prefill grid is at most 16 blocks, inside
//     the 32-block physical wave, so inter-wave behaviour of the writer is out of scope here;
//     tests/npu/validation/test_turboquant_bf16_validation.cpp is the ladder for that and
//     reaches 33, 48, 64 and 128 blocks.
//   * No ragged context. Every context is a whole number of 64-token pages, because the
//     prefill has no partial-page path at all (its launcher refuses the geometry). The decode's
//     tail masking is exercised by test_turboquant_fused_decode_opt_npu.cpp, which owns the
//     ragged rungs.
//   * No inter-head scale padding canary. A scale slot is 8 floats and the codec reads lanes
//     [head] and [4 + head]; at 4 KV heads all eight are live, so that canary is vacuous here
//     by construction. This reports it rather than skipping it.
//   * It takes no timing and makes no overlap or throughput claim.
//   * It does not disambiguate which Launch field carries the score multiplier: `scale` and
//     `inv_sqrt_len` are both set to the same value, exactly as
//     tests/cpu/test_turboquant_e2e_reference.cpp left it.
//
// Usage:
//   test_turboquant_e2e_npu [--rung N]... [--element fp16|bf16|both]
//                           [--ulp-fp16 N] [--ulp-bf16 N]

#include "../turboquant_npu_common.hpp"
#include "../../common/turboquant_nz_ingress_oracle.hpp"
#include "../../common/fused_decode_opt_fixture.hpp"
#include "../../../include/turboquant_fused_decode_opt.hpp"
#include "../../../include/turboquant_splitk_contract.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fd = ascend_micro::fused_decode_opt;
namespace fdt = ascend_micro::fused_decode;
namespace tq = vllm_ascend::turboquant;

using fused_decode_opt_test::Element;
using tq_npu::check;
using tq_npu::require;

using Mode = tq::TurboQuantModeTraits<tq::TurboQuantMode::KV4_FP8>;

// ------------------------------------------------------------------- constants
// The guard band and its byte, as the fused decode's own hardware ladder spells them.
constexpr std::uint8_t kGuardByte = 0x5A;
constexpr std::uint32_t kGuardWord = 0x5A5A5A5Au;
constexpr std::size_t kGuard = 64;
// The cache pre-fill, as the fused-decode fixture's own poison word: a quiet NaN payload, so a
// scale slot left unwritten would propagate rather than quietly scaling by something plausible.
constexpr std::uint32_t kCacheCanaryWord = 0x7FC0BEEFu;

// Real operator geometry. None of these three is a free parameter:
//   kKvHeads     the prefill's compile-time kHeads
//   kStripTokens the prefill's compile-time kTokens
//   kPageTokens  the decode's block_size, admitted at 64 or 128 by valid_launch
constexpr int kKvHeads = 4;
constexpr int kHeadSize = 256;
constexpr int kStripTokens = 16;
constexpr int kPageTokens = 64;
constexpr int kPackedRowBytes = kHeadSize / 2;
// NzScaleSlotFloats(4). Stated as a constant and checked against the shared contract below.
constexpr int kScaleLanes = 8;

// The score multiplier the operator applies to q . k, and the prefill's own invSqrtLen. Both
// are 1 / sqrt(256) here; see "does not establish" above.
constexpr float kScoreScale = 0.0625f;

// kSignedLevelOffset of common/turboquant_codec_mx.h, restated from the traits this file can
// see exactly as tests/cpu/test_turboquant_e2e_reference.cpp restates it: kNibbleSignShift is
// half the radix and kAffineBias is 7.5, so the offset is 0.5 -- a MIDRISE reconstruction.
constexpr double kNibbleSignShift = 8.0;
constexpr double kSignedLevelOffset = kNibbleSignShift - Mode::kAffineBias;

static_assert(kPageTokens % static_cast<int>(tq::kCubeTileRows) == 0,
              "a physical page must be a whole number of Cube tiles");
static_assert(static_cast<int>(tq::kCubeTileRows) % kStripTokens == 0,
              "a Cube tile must be a whole number of strips, or a strip straddles two tiles");
static_assert(kScaleLanes == static_cast<int>(tq::NzScaleSlotFloats(kKvHeads)),
              "the scale slot width drifted from the shared NZ contract");

// The target's TURBOQUANT_INGRESS_NZ is what selects the ONE prefill instance whose cache this
// decode can read. A build that linked the strip instance instead would compile, launch, pass
// no gate and be very hard to read, so the pairing is a compile error rather than a comment.
#if !defined(TURBOQUANT_INGRESS_NZ)
#error "test_turboquant_e2e_npu must be built with TURBOQUANT_INGRESS_NZ: the strip instance of the prefill writes a layout the fused decode cannot read. See cmake/NpuTargets.cmake."
#endif

// --------------------------------------------------------------- the canary fill
inline std::uint8_t canary_byte(std::size_t index)
{
    return static_cast<std::uint8_t>(kCacheCanaryWord >> ((index % 4) * 8));
}

inline float canary_float()
{
    float value;
    std::memcpy(&value, &kCacheCanaryWord, 4);
    return value;
}

inline std::uint32_t float_bits(float value)
{
    std::uint32_t raw;
    std::memcpy(&raw, &value, 4);
    return raw;
}

// ------------------------------------------------------- a guarded device allocation
// A device allocation with 0x5A guard bands on both sides and a host snapshot of what went
// into it, so an input can be byte-compared afterwards and an output's tail can be checked.
struct Guarded {
    tq_npu::Buffer allocation;
    std::size_t bytes;                   // the payload, past the leading guard
    std::vector<std::uint8_t> snapshot;  // the whole allocation, as uploaded

    explicit Guarded(std::size_t payload)
        : allocation(fused_decode_opt_test::aligned(payload) + 2 * kGuard),
          bytes(fused_decode_opt_test::aligned(payload))
    {
    }

    void* ptr() const { return static_cast<std::uint8_t*>(allocation.ptr) + kGuard; }

    // Upload `payload` into the middle and 0x5A everywhere else -- the guards AND the slack
    // between the payload's end and the aligned length.
    template <class T>
    void Upload(const std::vector<T>& payload)
    {
        const std::size_t n = payload.size() * sizeof(T);
        require(n <= bytes, "e2e NPU round trip: payload exceeds its allocation");
        snapshot.assign(allocation.bytes, kGuardByte);
        std::memcpy(snapshot.data() + kGuard, payload.data(), n);
        allocation.upload(snapshot);
    }

    // The 0x7FC0BEEF cache pre-fill over the whole payload, 0x5A on the guards.
    void FillCacheCanary()
    {
        snapshot.assign(allocation.bytes, kGuardByte);
        for (std::size_t i = 0; i < bytes; ++i) snapshot[kGuard + i] = canary_byte(i);
        allocation.upload(snapshot);
    }

    // No payload: the whole allocation is 0x5A. Outputs and workspaces start here.
    void FillGuard()
    {
        snapshot.assign(allocation.bytes, kGuardByte);
        allocation.upload(snapshot);
    }

    std::vector<std::uint8_t> Download() const
    {
        std::vector<std::uint8_t> image(allocation.bytes);
        allocation.download(image);
        return image;
    }

    void CheckGuards(const std::vector<std::uint8_t>& image, const char* what) const
    {
        for (std::size_t i = 0; i < kGuard; ++i) {
            if (image[i] != kGuardByte || image[kGuard + bytes + i] != kGuardByte) {
                std::fprintf(stderr, "guard band of %s clobbered at %zu: %02x / %02x\n", what, i,
                             image[i], image[kGuard + bytes + i]);
                require(false, "e2e NPU round trip: 0x5A guard band clobbered");
            }
        }
    }

    void CheckUnchanged(const char* what) const
    {
        const auto image = Download();
        CheckGuards(image, what);
        if (image != snapshot) {
            std::fprintf(stderr, "input allocation %s changed under a kernel\n", what);
            require(false, "e2e NPU round trip: an input allocation changed");
        }
    }
};

// ------------------------------------------------------------------- the geometry
// One rung's shapes, shared by the prefill launch, the decode launch and the oracle.
struct Geo {
    int requests = 1;
    int context = kPageTokens;   // tokens per request, a whole number of pages
    int group_heads = 2;         // query heads per KV head
    int pages = 1;               // logical pages per request
    int physical_pages = 4;
    int heads = kKvHeads * 2;    // query heads in total
    int ingress_tokens = kPageTokens;
    int ingress_strips = kPageTokens / kStripTokens;
    tq_nz::Geometry nz;
    std::vector<std::int32_t> table;    // [requests][pages], row stride = pages
    std::vector<std::int32_t> lengths;  // per request
    std::vector<std::int32_t> slots;    // the prefill's slot mapping, one global slot per token
    std::vector<tq_nz::StripSlot> strip_slots;

    std::size_t cache_bytes() const { return nz.cache_bytes(); }
    std::size_t scale_floats() const { return nz.scale_floats(); }
};

Geo build_geometry(int requests, int context, int group_heads)
{
    Geo g;
    require(requests >= 1 && requests <= 16, "e2e NPU round trip: request count out of range");
    require(context > 0 && context % kPageTokens == 0,
            "e2e NPU round trip: the context must be a whole number of 64-token pages -- the "
            "prefill has no partial-page path");
    require(group_heads >= 1 && group_heads <= 16,
            "e2e NPU round trip: query heads per KV head out of range");
    g.requests = requests;
    g.context = context;
    g.group_heads = group_heads;
    g.pages = context / kPageTokens;
    g.physical_pages = 2 * requests * g.pages + 2;
    g.heads = kKvHeads * group_heads;
    g.ingress_tokens = requests * context;
    g.ingress_strips = g.ingress_tokens / kStripTokens;

    g.nz.num_kv_heads = kKvHeads;
    g.nz.head_size = kHeadSize;
    g.nz.page_tokens = kPageTokens;
    g.nz.num_pages = g.physical_pages;
    g.nz.strip_tokens = kStripTokens;
    require(g.nz.valid(), "e2e NPU round trip: the NZ cache geometry is not one this contract "
                          "admits");

    // The FRAGMENTED block table: logical page p of request r lives at physical
    // 2 * (r * pages + p) + 1, so the physical page index is never the logical one and the even
    // physical pages stay canary for the whole run. Padded to a 32-byte allocation with -1, which
    // build_decode_tiling rejects if a live span ever reaches one.
    g.table.assign(fused_decode_opt_test::aligned(
                       static_cast<std::size_t>(requests) * g.pages * 4) / 4, -1);
    for (int r = 0; r < requests; ++r)
        for (int p = 0; p < g.pages; ++p)
            g.table[static_cast<std::size_t>(r) * g.pages + p] = 2 * (r * g.pages + p) + 1;

    g.lengths.assign(fused_decode_opt_test::aligned(static_cast<std::size_t>(requests) * 4) / 4, 0);
    for (int r = 0; r < requests; ++r) g.lengths[static_cast<std::size_t>(r)] = context;

    // The prefill's slot mapping, one GLOBAL slot per token of the flat prefill context. The
    // kernel reads it at token index strip * kStripTokens only, but every entry is filled: an
    // entry it does not read costs nothing, and a partially filled mapping would be the kind of
    // fixture that makes a wrong read look right.
    g.slots.assign(static_cast<std::size_t>(g.ingress_tokens), -1);
    g.strip_slots.assign(static_cast<std::size_t>(g.ingress_strips), tq_nz::StripSlot{});
    for (int i = 0; i < g.ingress_tokens; ++i) {
        const int request = i / context;
        const int local = i % context;
        const int page = local / kPageTokens;
        const int row = local % kPageTokens;
        const std::int32_t physical = g.table[static_cast<std::size_t>(request) * g.pages + page];
        require(physical >= 0 && physical < g.physical_pages,
                "e2e NPU round trip: the block table left a slot unmapped");
        g.slots[static_cast<std::size_t>(i)] = physical * kPageTokens + row;
        if (i % kStripTokens == 0) {
            tq_nz::StripSlot& slot = g.strip_slots[static_cast<std::size_t>(i / kStripTokens)];
            slot.page = physical;
            slot.token_in_page = row;
            // A strip must sit inside ONE Cube tile, or its rows would straddle two NZ tiles and
            // the single contiguous burst per column group would be wrong.
            require(row % kStripTokens == 0 &&
                        row / static_cast<int>(tq::kCubeTileRows) ==
                            (row + kStripTokens - 1) / static_cast<int>(tq::kCubeTileRows),
                    "e2e NPU round trip: a strip straddles two Cube tiles");
        }
    }
    return g;
}

// ------------------------------------------------------- Gate A: the golden NZ image
// The expected final cache image, plus which bytes a strip actually wrote. The written mask is
// DERIVED, not assumed: build the golden twice under two different floods and take agreement as
// evidence of a write. Everything else must still carry the 0x7FC0BEEF pre-fill.
struct Golden {
    std::vector<std::uint8_t> key, value;
    std::vector<float> scale;
    std::size_t written_key = 0, written_value = 0, written_scale = 0;
    int canary_pages = 0;  // physical pages no strip touches
};

Golden build_golden(const Geo& g, const tq_oracle::Reference& ref)
{
    const tq_nz::Image lo =
        tq_nz::build_image(g.nz, ref, g.strip_slots, static_cast<std::uint8_t>(0x00), 0.0f);
    const tq_nz::Image hi =
        tq_nz::build_image(g.nz, ref, g.strip_slots, static_cast<std::uint8_t>(0xFF), 1.0f);
    require(lo.key.size() == g.cache_bytes() && lo.scale.size() == g.scale_floats(),
            "e2e NPU round trip: the golden image geometry drifted from the launch");

    Golden out;
    out.key.resize(lo.key.size());
    out.value.resize(lo.value.size());
    out.scale.resize(lo.scale.size());
    for (std::size_t i = 0; i < out.key.size(); ++i) {
        const bool key_written = lo.key[i] == hi.key[i];
        const bool value_written = lo.value[i] == hi.value[i];
        out.key[i] = key_written ? lo.key[i] : canary_byte(i);
        out.value[i] = value_written ? lo.value[i] : canary_byte(i);
        out.written_key += key_written ? 1u : 0u;
        out.written_value += value_written ? 1u : 0u;
    }
    for (std::size_t i = 0; i < out.scale.size(); ++i) {
        const bool written = float_bits(lo.scale[i]) == float_bits(hi.scale[i]);
        out.scale[i] = written ? lo.scale[i] : canary_float();
        out.written_scale += written ? 1u : 0u;
    }

    // Every strip writes kStripTokens rows of kKvHeads heads, kPackedRowBytes each, in both
    // planes -- so the written byte count is a closed form, and a prefill that ran the wrong
    // number of strips fails HERE rather than somewhere inside the attention sum.
    const std::size_t per_strip =
        static_cast<std::size_t>(kStripTokens) * kKvHeads * kPackedRowBytes;
    const std::size_t expect = static_cast<std::size_t>(g.ingress_strips) * per_strip;
    require(out.written_key == expect && out.written_value == expect,
            "e2e NPU round trip: the golden image does not cover one whole page per strip");
    // Two scale lanes per KV head per row: a key lane and a value lane.
    require(out.written_scale ==
                static_cast<std::size_t>(g.ingress_strips) * kStripTokens * 2 * kKvHeads,
            "e2e NPU round trip: the golden scale image is not two lanes per head per row");

    std::vector<std::uint8_t> touched(static_cast<std::size_t>(g.physical_pages), 0);
    for (const tq_nz::StripSlot& slot : g.strip_slots)
        touched[static_cast<std::size_t>(slot.page)] = 1;
    for (const std::uint8_t t : touched) out.canary_pages += t ? 0 : 1;
    require(out.canary_pages > 0,
            "e2e NPU round trip: every physical page is written, so there is no unmapped-page "
            "canary to hold");
    return out;
}

// One byte-exact comparison of a downloaded plane against the golden, reporting the first few
// disagreements in a form that distinguishes "never written" from "written wrong".
std::size_t compare_plane(const std::vector<std::uint8_t>& image, const Guarded& buffer,
                          const std::vector<std::uint8_t>& golden, const char* what)
{
    std::size_t bad = 0, still_canary = 0;
    for (std::size_t i = 0; i < golden.size(); ++i) {
        const std::uint8_t got = image[kGuard + i];
        if (got == golden[i]) continue;
        ++bad;
        if (got == canary_byte(i)) ++still_canary;
        if (bad <= 8)
            std::fprintf(stderr, "  %s byte %zu: got %02x expected %02x%s\n", what, i, got,
                         golden[i],
                         got == canary_byte(i) ? "  (still pre-fill: never written)" : "");
    }
    // The slack between the golden's end and the 32-byte aligned allocation length is never a
    // cache byte, so it must still carry the pre-fill: FillCacheCanary painted it over the whole
    // payload, which includes that slack.
    for (std::size_t i = golden.size(); i < buffer.bytes; ++i)
        if (image[kGuard + i] != canary_byte(i)) {
            std::fprintf(stderr, "  %s padding byte %zu past the cache is %02x\n", what, i,
                         image[kGuard + i]);
            ++bad;
        }
    if (bad)
        std::fprintf(stderr, "  %s: %zu bytes disagree, %zu of them still the 0x%08X pre-fill\n",
                     what, bad, still_canary, kCacheCanaryWord);
    return bad;
}

// ------------------------------------------------------- Gate B: the decode's own arithmetic
// The reconstruction tests/cpu/test_turboquant_e2e_reference.cpp pins, in FP64.
inline double dequantise(int signed_nibble, float stored_scale)
{
    return (static_cast<double>(signed_nibble) + kSignedLevelOffset) *
           static_cast<double>(stored_scale) / static_cast<double>(Mode::kGain);
}

// How Gate C permutes the channel axis. Applied to the QUERY index while the key is read in its
// true order, which is the same thing as permuting the key and keeps the cache untouched.
enum class ChannelOrder { kTrue, kSwapHalves, kRotateOne };

inline int permute_channel(ChannelOrder order, int channel)
{
    switch (order) {
    case ChannelOrder::kSwapHalves:
        return channel < kHeadSize / 2 ? channel + kHeadSize / 2 : channel - kHeadSize / 2;
    case ChannelOrder::kRotateOne: return (channel + 1) % kHeadSize;
    case ChannelOrder::kTrue: break;
    }
    return channel;
}

// The whole decode, in FP64, from the cache AS DOWNLOADED and one query vector.
std::vector<double> attention(const Geo& g, const std::vector<std::uint8_t>& key,
                              const std::vector<std::uint8_t>& value,
                              const std::vector<float>& scale, const std::vector<float>& query,
                              int request, int query_head, ChannelOrder order)
{
    const int head = query_head / g.group_heads;
    const float* q =
        query.data() + (static_cast<std::size_t>(request) * g.heads + query_head) * kHeadSize;

    std::vector<double> score(static_cast<std::size_t>(g.context), 0.0);
    std::vector<int> page_of(static_cast<std::size_t>(g.context), 0);
    std::vector<int> row_of(static_cast<std::size_t>(g.context), 0);
    for (int token = 0; token < g.context; ++token) {
        const int logical = token / kPageTokens;
        const int row = token % kPageTokens;
        const int page = g.table[static_cast<std::size_t>(request) * g.pages + logical];
        page_of[static_cast<std::size_t>(token)] = page;
        row_of[static_cast<std::size_t>(token)] = row;
        const float ks = tq_nz::read_scale(g.nz, scale.data(), 0, page, head, row);
        double dot = 0.0;
        for (int c = 0; c < kHeadSize; ++c) {
            const int n = tq_nz::read_code(g.nz, key.data(), page, head, row, c);
            dot += static_cast<double>(q[permute_channel(order, c)]) * dequantise(n, ks);
        }
        score[static_cast<std::size_t>(token)] = dot * static_cast<double>(kScoreScale);
    }

    double m = score[0];
    for (const double s : score) m = s > m ? s : m;
    double z = 0.0;
    std::vector<double> p(static_cast<std::size_t>(g.context), 0.0);
    for (int i = 0; i < g.context; ++i) {
        p[static_cast<std::size_t>(i)] = std::exp(score[static_cast<std::size_t>(i)] - m);
        z += p[static_cast<std::size_t>(i)];
    }

    std::vector<double> out(static_cast<std::size_t>(kHeadSize), 0.0);
    for (int i = 0; i < g.context; ++i) {
        const int page = page_of[static_cast<std::size_t>(i)];
        const int row = row_of[static_cast<std::size_t>(i)];
        const float vs = tq_nz::read_scale(g.nz, scale.data(), 1, page, head, row);
        const double weight = p[static_cast<std::size_t>(i)];
        for (int c = 0; c < kHeadSize; ++c) {
            const int n = tq_nz::read_code(g.nz, value.data(), page, head, row, c);
            out[static_cast<std::size_t>(c)] += weight * dequantise(n, vs);
        }
    }
    for (double& v : out) v /= z;
    return out;
}

// ------------------------------------------------------------------- the query
// VARIABLE per channel, in exact binary fractions (multiples of a power of two), so the query
// itself introduces no rounding and a channel permutation genuinely changes every dot product.
// A constant query would make Gate B blind to the whole NZ nibble pairing.
std::vector<float> build_query(const Geo& g, int shift)
{
    std::vector<float> query(static_cast<std::size_t>(g.requests) * g.heads * kHeadSize, 0.0f);
    for (int r = 0; r < g.requests; ++r)
        for (int qh = 0; qh < g.heads; ++qh) {
            const std::size_t base = (static_cast<std::size_t>(r) * g.heads + qh) * kHeadSize;
            for (int c = 0; c < kHeadSize; ++c) {
                const int code = (c * 7 + qh * 13 + r * 5) % 31 - 15;
                query[base + static_cast<std::size_t>(c)] =
                    std::ldexp(static_cast<float>(code), -4 - shift);
            }
        }
    return query;
}

// The power of two that puts the worst |score| in [1.5, 3]. A degenerate softmax -- every score
// equal, or one score so far ahead that the result is a hard argmax -- would weaken Gate B
// without failing it, so the query magnitude is CHOSEN rather than hoped for, from the golden
// image (which Gate A proves is what the device wrote) before any decode launch.
int choose_query_shift(const Geo& g, const Golden& golden)
{
    const std::vector<float> probe = build_query(g, 0);
    double worst = 0.0;
    for (int r = 0; r < g.requests; ++r)
        for (int qh = 0; qh < g.heads; ++qh) {
            const int head = qh / g.group_heads;
            const float* q =
                probe.data() + (static_cast<std::size_t>(r) * g.heads + qh) * kHeadSize;
            for (int token = 0; token < g.context; ++token) {
                const int logical = token / kPageTokens;
                const int row = token % kPageTokens;
                const int page = g.table[static_cast<std::size_t>(r) * g.pages + logical];
                const float ks = tq_nz::read_scale(g.nz, golden.scale.data(), 0, page, head, row);
                double dot = 0.0;
                for (int c = 0; c < kHeadSize; ++c)
                    dot += static_cast<double>(q[c]) *
                           dequantise(
                               tq_nz::read_code(g.nz, golden.key.data(), page, head, row, c), ks);
                const double magnitude = std::fabs(dot * static_cast<double>(kScoreScale));
                if (magnitude > worst) worst = magnitude;
            }
        }
    require(worst > 0.0, "e2e NPU round trip: the probe query produces an identically zero "
                         "score, so the fixture cannot discriminate anything");
    int shift = 0;
    while (worst > 3.0 && shift < 64) {
        worst *= 0.5;
        ++shift;
    }
    while (worst < 1.5 && shift > -64) {
        worst *= 2.0;
        --shift;
    }
    return shift;
}

// ------------------------------------------------------------- the ULP comparison
// A monotonic ordinal over a 16-bit IEEE format, so the distance is meaningful across zero and
// across the sign. Both instances carry a 1-bit sign at bit 15, which is what makes one
// function serve fp16 and bf16.
inline int ordinal(std::uint16_t bits)
{
    const int magnitude = static_cast<int>(bits & 0x7FFFu);
    return (bits & 0x8000u) ? -magnitude : magnitude;
}

// The correctly-rounded 16-bit word of an oracle value, in the instance's own format.
//
// NOT fused_decode_opt_test::reference_bits, and that is a real difference rather than a
// preference: that function THROWS outside [1, 15], because the fused-decode fixture's results
// are positive normals in that band by construction. This round trip's outputs are weighted
// means of signed dequantised values over a cache a real quantiser produced, so they carry
// either sign and any magnitude. The general round-half-to-even converters are the right ones
// here, and `ordinal` above is what keeps the ULP distance meaningful across the sign and
// across zero.
inline std::uint16_t element_bits(Element element, float value)
{
    return element == Element::kBf16 ? ascend_micro::sim::float_to_bfloat16_rne(value)
                                     : ascend_micro::sim::float_to_half_rne(value);
}

inline bool finite_bits(Element element, std::uint16_t bits)
{
    return std::isfinite(fused_decode_opt_test::widen_bits(element, bits));
}

struct OracleResult {
    int worst_ulp = 0;
    double worst_abs = 0.0;
    double worst_rel = 0.0;
};

// ------------------------------------------------------------------- the ladder
struct Rung {
    int requests;
    int context;
    int group_heads;
    std::vector<int> splits;
    const char* note;
};

const std::vector<Rung>& ladder()
{
    static const std::vector<Rung> rungs = {
        {1, 1 * kPageTokens, 2, {1},
         "one page, one tile: the fused path, and the workspace must stay untouched"},
        {1, 4 * kPageTokens, 2, {1, 2, 4}, "four pages: the split sweep"},
        {2, 2 * kPageTokens, 2, {1, 2},
         "two requests, two block-table rows, one cache: the token dimension of both operators"},
        {1, 16 * kPageTokens, 2, {1, 2, 4, 8},
         "sixteen pages: 16 prefill grid blocks, and at S=1 one decode worker wraps the "
         "three-slot ring five times"},
        {1, 2 * kPageTokens, 16, {1, 2},
         "64 query heads over 4 KV heads: the full GQA width, where all 8 scale lanes are live"},
    };
    return rungs;
}

// ------------------------------------------------------------------- options
struct Options {
    std::vector<int> rungs;  // empty means all
    bool fp16 = true, bf16 = true;
    int ulp_fp16 = 4;
    int ulp_bf16 = 2;
};

void print_usage(std::FILE* out, const char* argv0)
{
    std::fprintf(out,
                 "Usage: %s [--rung N]... [--element fp16|bf16|both] [--ulp-fp16 N] "
                 "[--ulp-bf16 N]\n"
                 "  --rung N      run ladder rung N only (1..%zu, repeatable)\n"
                 "  --element E   fp16, bf16 or both (default both)\n"
                 "  --ulp-fp16 N  Gate B budget for the fp16 instance (default 4)\n"
                 "  --ulp-bf16 N  Gate B budget for the bf16 instance (default 2)\n"
                 "The budgets are DERIVED from the FP32-against-FP64 arithmetic, not measured on\n"
                 "silicon; the run reports the worst achieved figure whatever the budget is.\n",
                 argv0, ladder().size());
}

enum class ParseResult { kRun, kExitOk, kExitUsage };

ParseResult parse_options(int argc, char** argv, Options& opts)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage(stdout, argv[0]);
            return ParseResult::kExitOk;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "%s needs a value\n", arg.c_str());
            print_usage(stderr, argv[0]);
            return ParseResult::kExitUsage;
        }
        const std::string text(argv[++i]);
        if (arg == "--element") {
            opts.fp16 = text == "fp16" || text == "both";
            opts.bf16 = text == "bf16" || text == "both";
            if (!opts.fp16 && !opts.bf16) {
                std::fprintf(stderr, "--element must be fp16, bf16 or both, got %s\n",
                             text.c_str());
                return ParseResult::kExitUsage;
            }
            continue;
        }
        int value = 0;
        try {
            std::size_t consumed = 0;
            value = std::stoi(text, &consumed);
            if (consumed != text.size()) throw std::invalid_argument(text);
        } catch (const std::exception&) {
            std::fprintf(stderr, "%s: not an integer: %s\n", arg.c_str(), text.c_str());
            return ParseResult::kExitUsage;
        }
        if (arg == "--rung") {
            if (value < 1 || value > static_cast<int>(ladder().size())) {
                std::fprintf(stderr, "--rung out of range: %d\n", value);
                return ParseResult::kExitUsage;
            }
            opts.rungs.push_back(value);
        } else if (arg == "--ulp-fp16") {
            opts.ulp_fp16 = value;
        } else if (arg == "--ulp-bf16") {
            opts.ulp_bf16 = value;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            print_usage(stderr, argv[0]);
            return ParseResult::kExitUsage;
        }
    }
    return ParseResult::kRun;
}

// ------------------------------------------------------------- the cache on the device
// The three cache allocations plus the prefill's inputs: built once per rung, written by the
// prefill, then read by every decode launch of that rung without being re-uploaded. That reuse
// is itself a check -- the decode must leave all three byte-identical.
struct RoundTripCache {
    Guarded key_act, value_act, tables, signs, slots;
    Guarded key_cache, value_cache, scale_cache;

    RoundTripCache(const Geo& g, const tq_npu::Fixture& f)
        : key_act(f.key.size() * sizeof(std::uint16_t)),
          value_act(f.value.size() * sizeof(std::uint16_t)),
          tables(f.tables.size() * sizeof(std::uint16_t)),
          signs(f.signs.size() * sizeof(float)),
          slots(g.slots.size() * sizeof(std::int32_t)), key_cache(g.cache_bytes()),
          value_cache(g.cache_bytes()), scale_cache(g.scale_floats() * sizeof(float))
    {
        key_act.Upload(tq_npu::to_bfloat16(f.key));
        value_act.Upload(tq_npu::to_bfloat16(f.value));
        // Row-major 32x32 blocks, as the GM contract publishes them, widened to BF16 exactly as
        // tq_npu::Bf16Device uploads them.
        tables.Upload(tq_npu::to_bfloat16(f.tables));
        signs.Upload(f.signs);
        slots.Upload(g.slots);
        key_cache.FillCacheCanary();
        value_cache.FillCacheCanary();
        scale_cache.FillCacheCanary();
    }

    void LaunchPrefill(aclrtStream stream, const Geo& g, const tq_npu::Fixture& f) const
    {
        // tokens_per_core 64 and block_size kStripTokens: see "THE TWO BLOCK SIZES" above.
        // reuse_mode4_flags true is the mode-2 rendezvous, the only entry point that can stream
        // more than one strip per block -- and four strips per block is what fills a page.
        launch_turboquant_reshape_and_cache_bf16(
            stream, static_cast<const std::uint16_t*>(key_act.ptr()),
            static_cast<const std::uint16_t*>(value_act.ptr()),
            static_cast<std::int8_t*>(key_cache.ptr()),
            static_cast<std::int8_t*>(value_cache.ptr()),
            static_cast<float*>(scale_cache.ptr()),
            static_cast<const std::int32_t*>(slots.ptr()),
            static_cast<const float*>(signs.ptr()),
            static_cast<const std::int32_t*>(tables.ptr()),
            static_cast<std::uint32_t>(g.ingress_tokens), static_cast<std::uint32_t>(kKvHeads),
            static_cast<std::uint32_t>(kHeadSize), static_cast<std::uint32_t>(kStripTokens),
            static_cast<std::uint32_t>(g.physical_pages),
            static_cast<std::uint32_t>(kPageTokens), f.shape.inv_sqrt_len(), true);
    }

    void CheckInputsUnchanged() const
    {
        key_act.CheckUnchanged("prefill key activations");
        value_act.CheckUnchanged("prefill value activations");
        tables.CheckUnchanged("rotation tables");
        signs.CheckUnchanged("pi signs");
        slots.CheckUnchanged("slot mapping");
    }
};

// ------------------------------------------------------------------- the run
struct Totals {
    int prefill_launches = 0, decode_launches = 0;
    std::size_t cache_bytes_exact = 0, scale_floats_exact = 0, canary_bytes = 0;
    long long canary_pages = 0;
    int worst_ulp[2] = {0, 0};
    double worst_abs = 0.0, worst_rel = 0.0;
    double weakest_control = -1.0;  // the smallest Gate C separation seen, in fp16 ULP
    int ring_wraps = 0;
    int widest_grid = 0;
};

int run(const Options& opts)
{
    tq_npu::Session session;

    std::vector<Element> elements;
    if (opts.fp16) elements.push_back(Element::kFp16);
    if (opts.bf16) elements.push_back(Element::kBf16);
    const int budget[2] = {opts.ulp_fp16, opts.ulp_bf16};

    std::vector<int> chosen = opts.rungs;
    if (chosen.empty())
        for (std::size_t i = 0; i < ladder().size(); ++i) chosen.push_back(static_cast<int>(i) + 1);

    Totals totals;

    std::printf("TurboQuant KV cache WRITE -> READ round trip on the physical device\n"
                "  write: turboquant_reshape_and_cache_bf16 (TURBOQUANT_INGRESS_NZ), "
                "%d-token strips\n"
                "  read:  turboquant_fused_decode, %d-token pages, D=%d, %d KV heads\n"
                "  score multiplier %g, kGain %.10f, reconstruction "
                "(n + %.1f) * scale / kGain\n",
                kStripTokens, kPageTokens, kHeadSize, kKvHeads,
                static_cast<double>(kScoreScale), static_cast<double>(Mode::kGain),
                kSignedLevelOffset);

    for (const int index : chosen) {
        const Rung& rung = ladder()[static_cast<std::size_t>(index) - 1];
        const Geo g = build_geometry(rung.requests, rung.context, rung.group_heads);

        // ---------------------------------------------------------- the write
        // The shared fixture at the PREFILL's own strip width, so its audits (all sixteen
        // nibbles reached, the INT4 upper clamp taken, chunks and strips packing distinctly)
        // certify that this activation set can fail honestly. Its own fragmented strip-contract
        // cache images are not used here; what this file takes from it is the layout-independent
        // INT4 codes and power-of-two scales, which the NZ oracle then PLACES.
        const tq_npu::Fixture f(g.ingress_tokens);
        require(f.strips() == g.ingress_strips,
                "e2e NPU round trip: the fixture's strip count disagrees with the geometry");
        const Golden golden = build_golden(g, f.ref);

        RoundTripCache cache(g, f);
        const int prefill_grid = g.ingress_tokens / kPageTokens;
        check(aclrtSynchronizeStream(session.stream), "pre-prefill synchronize");
        cache.LaunchPrefill(session.stream, g, f);
        check(aclrtSynchronizeStream(session.stream), "post-prefill synchronize");
        ++totals.prefill_launches;
        cache.CheckInputsUnchanged();

        // GATE A. Byte for byte, bit for bit.
        const auto key_image = cache.key_cache.Download();
        const auto value_image = cache.value_cache.Download();
        const auto scale_image = cache.scale_cache.Download();
        cache.key_cache.CheckGuards(key_image, "key cache");
        cache.value_cache.CheckGuards(value_image, "value cache");
        cache.scale_cache.CheckGuards(scale_image, "scale cache");

        std::size_t bad = compare_plane(key_image, cache.key_cache, golden.key, "key cache");
        bad += compare_plane(value_image, cache.value_cache, golden.value, "value cache");
        std::size_t scale_bad = 0;
        for (std::size_t i = 0; i < golden.scale.size(); ++i) {
            float got;
            std::memcpy(&got, scale_image.data() + kGuard + i * 4, 4);
            if (float_bits(got) == float_bits(golden.scale[i])) continue;
            ++scale_bad;
            if (scale_bad <= 8)
                std::fprintf(stderr, "  scale %zu: got %g (%08x) expected %g (%08x)\n", i,
                             static_cast<double>(got), float_bits(got),
                             static_cast<double>(golden.scale[i]), float_bits(golden.scale[i]));
        }
        for (std::size_t i = golden.scale.size() * 4; i < cache.scale_cache.bytes; ++i)
            if (scale_image[kGuard + i] != canary_byte(i)) ++scale_bad;
        if (bad || scale_bad) {
            std::fprintf(stderr,
                         "rung %d (%d x %d tokens): GATE A FAILED, %zu cache bytes and %zu "
                         "scale floats disagree with the golden NZ image\n",
                         index, rung.requests, rung.context, bad, scale_bad);
            require(false, "e2e NPU round trip: the ingress write is not bit-exact");
        }

        // The cache the decode is about to read is now the DEVICE's, and it equals the golden.
        // The oracle below reads THIS, so Gate B is a statement about the decode alone.
        std::vector<std::uint8_t> key(golden.key.size()), value(golden.value.size());
        std::vector<float> scale(golden.scale.size());
        std::memcpy(key.data(), key_image.data() + kGuard, key.size());
        std::memcpy(value.data(), value_image.data() + kGuard, value.size());
        std::memcpy(scale.data(), scale_image.data() + kGuard, scale.size() * 4);

        totals.cache_bytes_exact += golden.written_key + golden.written_value;
        totals.scale_floats_exact += golden.written_scale;
        totals.canary_bytes += 2 * (golden.key.size() - golden.written_key);
        totals.canary_pages += 2 * golden.canary_pages;

        std::printf("\nrung %d: %d request(s) x %d tokens, %d query heads over %d KV heads, "
                    "%d logical / %d physical pages\n"
                    "  write: 1 launch, grid %d (%d strips of %d tokens, %d per page). GATE A "
                    "bit-exact: %zu cache bytes, %zu scale floats, %d physical pages still "
                    "0x%08X in both planes\n",
                    index, rung.requests, rung.context, g.heads, kKvHeads, g.pages,
                    g.physical_pages, prefill_grid, g.ingress_strips, kStripTokens,
                    kPageTokens / kStripTokens, golden.written_key + golden.written_value,
                    golden.written_scale, golden.canary_pages, kCacheCanaryWord);

        // The query, chosen against the cache so the softmax is neither flat nor an argmax.
        const int shift = choose_query_shift(g, golden);
        const std::vector<float> query = build_query(g, shift);

        // GATE C, once per rung: does the variable query actually discriminate channel order?
        // Computed on query head 0 of request 0, which is enough to establish that the oracle is
        // sensitive to the permutation at all -- and it is the same oracle every rung uses.
        {
            const std::vector<double> truth =
                attention(g, key, value, scale, query, 0, 0, ChannelOrder::kTrue);
            const ChannelOrder controls[] = {ChannelOrder::kSwapHalves, ChannelOrder::kRotateOne};
            const char* names[] = {"query halves swapped (the NZ nibble pairing)",
                                   "channel order rotated by one"};
            for (int c = 0; c < 2; ++c) {
                const std::vector<double> other =
                    attention(g, key, value, scale, query, 0, 0, controls[c]);
                double separation = 0.0;
                for (std::size_t i = 0; i < truth.size(); ++i) {
                    const std::uint16_t a =
                        element_bits(Element::kFp16, static_cast<float>(truth[i]));
                    const std::uint16_t b =
                        element_bits(Element::kFp16, static_cast<float>(other[i]));
                    const double d = std::fabs(static_cast<double>(ordinal(a) - ordinal(b)));
                    if (d > separation) separation = d;
                }
                std::printf("  GATE C control, %-44s %.0f fp16 ULP from the true oracle\n",
                            names[c], separation);
                require(separation >= 64.0,
                        "e2e NPU round trip: a channel permutation moves the oracle by less than "
                        "64 fp16 ULP, so Gate B would pass vacuously on a permuted cache");
                if (totals.weakest_control < 0.0 || separation < totals.weakest_control)
                    totals.weakest_control = separation;
            }
        }

        // ---------------------------------------------------------- the read
        for (const Element element : elements) {
            const int leg = element == Element::kBf16 ? 1 : 0;
            for (const int splits : rung.splits) {
                fd::Launch config;
                config.element_type = static_cast<std::uint32_t>(element);
                config.tokens = static_cast<std::uint32_t>(g.requests);
                config.kv_heads = static_cast<std::uint32_t>(kKvHeads);
                config.heads = static_cast<std::uint32_t>(g.heads);
                config.head_size = static_cast<std::uint32_t>(kHeadSize);
                config.block_size = static_cast<std::uint32_t>(kPageTokens);
                config.max_blocks = static_cast<std::uint32_t>(g.pages);
                config.splits = static_cast<std::uint32_t>(splits);
                config.heads_per_task = static_cast<std::uint32_t>(g.group_heads);
                config.fused_context_limit = 0;
                config.scale = kScoreScale;
                config.inv_sqrt_len = kScoreScale;
                const std::uint32_t tasks = config.tokens * config.kv_heads * config.splits;
                config.tasks_per_block = (tasks + 31) / 32;
                config.grid = (tasks + config.tasks_per_block - 1) / config.tasks_per_block;
                config.reduce_tasks_per_block =
                    (config.tokens * config.heads + config.grid - 1) / config.grid;
                require(fd::valid_launch(config.grid, config.tokens, config.heads, config.kv_heads,
                                         config.head_size, config.block_size, config.max_blocks,
                                         config.splits, config.heads_per_task,
                                         config.tasks_per_block, config.reduce_tasks_per_block),
                        "e2e NPU round trip: the decode launch geometry is not one the operator "
                        "admits");
                require(config.grid <= turboquant_splitk::kMaxCoResidentBlocks,
                        "e2e NPU round trip: the decode grid exceeds the one-wave bound the "
                        "reduction's SyncAll needs");
                if (static_cast<int>(config.grid) > totals.widest_grid)
                    totals.widest_grid = static_cast<int>(config.grid);

                const std::size_t partial_floats =
                    static_cast<std::size_t>(g.requests) * config.heads *
                    static_cast<std::size_t>(splits) * (kHeadSize + 16);
                const std::size_t output_bytes =
                    query.size() * fd::element_bytes(config.element_type);
                Guarded query_buffer(query.size() * 4), modes(32),
                    workspace(partial_floats * 4 + 4096), out(output_bytes + 32),
                    table(g.table.size() * 4), lengths(g.lengths.size() * 4),
                    tiling(fdt::decode_tiling_bytes_for(config));
                query_buffer.Upload(query);
                modes.Upload(std::vector<std::uint32_t>(8, 0));
                table.Upload(g.table);
                lengths.Upload(g.lengths);
                {
                    // The host tiling, from the same context lengths and block table the device
                    // gets. Every tile count, live-tail length and page span this launch acts on
                    // is decided here; the device derives none of it.
                    std::vector<std::uint32_t> image(
                        fdt::decode_tiling_words(config.tokens, config.splits, config.grid), 0);
                    const auto status = fdt::build_decode_tiling(
                        fdt::decode_tiling_request(config), g.lengths.data(), g.table.data(),
                        image.data());
                    require(status == fdt::DecodeTilingStatus::kOk,
                            fdt::decode_tiling_status_text(status));
                    tiling.Upload(image);
                }
                workspace.FillGuard();
                out.FillGuard();

                const fd::Buffers buffers{query_buffer.ptr(),
                                          cache.key_cache.ptr(),
                                          cache.value_cache.ptr(),
                                          cache.scale_cache.ptr(),
                                          table.ptr(),
                                          lengths.ptr(),
                                          modes.ptr(),
                                          workspace.ptr(),
                                          out.ptr(),
                                          tiling.ptr()};
                check(aclrtSynchronizeStream(session.stream), "pre-decode synchronize");
                launch_fused_decode_opt(session.stream, &config, &buffers);
                check(aclrtSynchronizeStream(session.stream), "post-decode synchronize");
                ++totals.decode_launches;

                // GATE B: every live output channel against the FP64 oracle over the bytes the
                // device wrote, plus the output tail canary.
                const auto image = out.Download();
                out.CheckGuards(image, "output");
                OracleResult result;
                for (int r = 0; r < g.requests; ++r)
                    for (int qh = 0; qh < g.heads; ++qh) {
                        const std::vector<double> want =
                            attention(g, key, value, scale, query, r, qh, ChannelOrder::kTrue);
                        const std::size_t base =
                            (static_cast<std::size_t>(r) * g.heads + qh) * kHeadSize;
                        for (int c = 0; c < kHeadSize; ++c) {
                            std::uint16_t bits;
                            std::memcpy(&bits,
                                        image.data() + kGuard +
                                            2 * (base + static_cast<std::size_t>(c)),
                                        2);
                            require(finite_bits(element, bits),
                                    "e2e NPU round trip: the decode emitted a non-finite channel");
                            const double expected = want[static_cast<std::size_t>(c)];
                            const std::uint16_t reference =
                                element_bits(element, static_cast<float>(expected));
                            const int ulp = std::abs(ordinal(bits) - ordinal(reference));
                            const double got = static_cast<double>(
                                fused_decode_opt_test::widen_bits(element, bits));
                            const double absolute = std::fabs(got - expected);
                            const double relative =
                                absolute /
                                (std::fabs(expected) > 1e-12 ? std::fabs(expected) : 1.0);
                            if (ulp > result.worst_ulp) result.worst_ulp = ulp;
                            if (absolute > result.worst_abs) result.worst_abs = absolute;
                            if (relative > result.worst_rel) result.worst_rel = relative;
                            if (ulp > budget[leg]) {
                                std::fprintf(stderr,
                                             "GATE B mismatch, request %d query head %d channel "
                                             "%d: got %g (%04x) expected %g (%04x), %d %s ULP "
                                             "over a budget of %d\n",
                                             r, qh, c, got, bits, expected, reference, ulp,
                                             fused_decode_opt_test::element_name(element),
                                             budget[leg]);
                                require(false,
                                        "e2e NPU round trip: the decode's output disagrees with "
                                        "the FP64 oracle over the cache the prefill wrote");
                            }
                        }
                    }
                // The output tail past the last live channel.
                for (std::size_t i = output_bytes; i < out.bytes; ++i)
                    if (image[kGuard + i] != kGuardByte) {
                        std::fprintf(stderr, "output tail byte %zu is %02x\n", i,
                                     image[kGuard + i]);
                        require(false, "e2e NPU round trip: output tail canary clobbered");
                    }
                if (result.worst_ulp > totals.worst_ulp[leg])
                    totals.worst_ulp[leg] = result.worst_ulp;
                totals.worst_abs = std::max(totals.worst_abs, result.worst_abs);
                totals.worst_rel = std::max(totals.worst_rel, result.worst_rel);

                // The workspace, and the inputs the decode must not have touched -- including
                // the whole cache the prefill filled.
                const auto partials = workspace.Download();
                workspace.CheckGuards(partials, "workspace");
                if (splits > 1) {
                    for (std::size_t i = 0; i < partial_floats; ++i) {
                        float v;
                        std::memcpy(&v, partials.data() + kGuard + i * 4, 4);
                        require(std::isfinite(v),
                                "e2e NPU round trip: a Split-K partial came back non-finite, so "
                                "the reduction left it unwritten");
                    }
                    for (std::size_t i = partial_floats * 4; i < workspace.bytes; ++i)
                        require(partials[kGuard + i] == kGuardByte,
                                "e2e NPU round trip: the workspace past the live partial extent "
                                "was written");
                } else {
                    require(partials == workspace.snapshot,
                            "e2e NPU round trip: the workspace was written at S = 1, where the "
                            "operator takes its fused path and NeedsReduction() is false");
                }
                query_buffer.CheckUnchanged("decode query");
                table.CheckUnchanged("block table");
                lengths.CheckUnchanged("context lengths");
                modes.CheckUnchanged("mode tables");
                tiling.CheckUnchanged("host tiling");
                // The cache: the decode is a reader, so every byte the prefill wrote and every
                // canary byte it did not must still be there.
                require(cache.key_cache.Download() == key_image &&
                            cache.value_cache.Download() == value_image &&
                            cache.scale_cache.Download() == scale_image,
                        "e2e NPU round trip: the decode modified the cache it was reading");

                const int tiles_per_worker =
                    (g.pages + splits - 1) / splits * (kPageTokens / static_cast<int>(fd::kTileRows));
                if (tiles_per_worker > static_cast<int>(fd::kSlots)) ++totals.ring_wraps;
                std::printf("  read:  %s S %d  grid %2d  tiles/worker %2d  worst %d ULP  "
                            "worst abs %.3e  worst rel %.3e  (budget %d)\n",
                            fused_decode_opt_test::element_name(element), splits, config.grid,
                            tiles_per_worker, result.worst_ulp, result.worst_abs,
                            result.worst_rel, budget[leg]);
            }
        }
        std::printf("  rung %d: %s\n", index, rung.note);
    }

    require(totals.prefill_launches > 0 && totals.decode_launches > 0,
            "e2e NPU round trip: nothing ran");
    require(totals.ring_wraps > 0,
            "e2e NPU round trip: no decode worker owned more tiles than the pipeline ring's "
            "depth, so nothing in this run is ring-wrap evidence");
    for (const Element element : elements) {
        const int leg = element == Element::kBf16 ? 1 : 0;
        require(totals.worst_ulp[leg] <= budget[leg],
                "e2e NPU round trip: worst ULP over the budget");
    }

    std::printf("\nPASS TurboQuant WRITE -> READ round trip on the physical device: "
                "%d prefill launch(es) and %d decode launch(es) over %zu rung(s).\n"
                "  GATE A (the write, bit-exact, 0 ULP): %zu cache bytes and %zu scale floats "
                "equal to the golden NZ-tiled image, %zu bytes and %lld physical pages still the "
                "0x%08X pre-fill.\n"
                "  GATE B (the read, against an FP64 oracle over the bytes the device wrote): "
                "worst %d fp16 ULP (budget %d), worst %d bf16 ULP (budget %d), worst absolute "
                "residual %.3e, worst relative %.3e. The budgets are DERIVED from the "
                "FP32-against-FP64 arithmetic, NOT measured on silicon.\n"
                "  GATE C (discrimination): the weakest channel permutation separates the oracle "
                "by %.0f fp16 ULP, so the variable query is load bearing.\n"
                "  Widest decode grid %d of %u co-resident blocks. 0x%08X guard bands held on "
                "every allocation, the output tail stayed canary, the workspace was untouched at "
                "S = 1, and every input -- the whole cache included -- was byte-identical after "
                "every decode launch.\n"
                "  NOT established here: a second PREFILL execution wave, any ragged context, "
                "any inter-head scale padding canary (all 8 lanes are live at 4 KV heads), and no "
                "timing or overlap claim. See this file's header.\n",
                totals.prefill_launches, totals.decode_launches, chosen.size(),
                totals.cache_bytes_exact, totals.scale_floats_exact, totals.canary_bytes,
                totals.canary_pages, kCacheCanaryWord, totals.worst_ulp[0], opts.ulp_fp16,
                totals.worst_ulp[1], opts.ulp_bf16, totals.worst_abs, totals.worst_rel,
                totals.weakest_control, totals.widest_grid,
                turboquant_splitk::kMaxCoResidentBlocks, kGuardWord);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    Options opts;
    switch (parse_options(argc, argv, opts)) {
    case ParseResult::kExitOk: return 0;
    case ParseResult::kExitUsage: return 2;
    case ParseResult::kRun: break;
    }
    try {
        return run(opts);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL TurboQuant e2e NPU round trip: %s\n", error.what());
        return 1;
    }
}
