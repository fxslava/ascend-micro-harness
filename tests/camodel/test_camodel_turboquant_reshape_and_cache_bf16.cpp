// CAModel runner for turboquant_reshape_and_cache_bf16: the BF16 input
// instance of the prefill reshape-and-cache, checked against the same shared
// oracle as the INT8 instance.
//
// Why this can be bit-exact, and what that buys
// ---------------------------------------------
// The fixture's activations are integers. A BF16 significand is 8 bits, so
// integers up to 256 are exactly representable; a product of two is at most
// 2^16 and exact in FP32; and a sum of 256 such products is an integer below
// 2^24, hence exact in FP32 *whatever order the Cube accumulates in*. The
// quantiser downstream is the same power-of-two path as the INT8 instance, so
// every nibble and every scale is reproducible by an integer host oracle.
//
// That gives a stronger check than "BF16 looks about right": for the same
// integer activations the BF16 kernel must produce **byte-identical** output to
// the INT8 kernel, because it is the same transform over the same values. So
// tests/common/turboquant_oracle.hpp is reused with no BF16-specific path at
// all -- only the input *encoding* differs, 2-byte BF16 instead of 1-byte INT8.
//
// With general (non-integer) BF16 activations this would not hold: the Cube's
// FP32 accumulation order is unspecified and no integer oracle can reproduce
// it, so such a test would have to degrade to cosine plus a relative-L2 bound,
// which no longer detects a wrong-strip read the way nibble comparison does.
// That test, if it is ever wanted, belongs in a separate, clearly labelled
// runner; it must not replace this one.
//
// Checks, in order:
//   0.  the fixture self-audit from the shared oracle;
//   0b. the 16 resident rotation blocks compose to the oracle's 256x256 matrix;
//   1.  head and tail Global Memory canaries on all eight buffers;
//   2.  every cache byte and scale float on a page no block wrote still
//       0xDEADBEEF;
//   3.  keyCache, key-major, bit-exact, per strip and chunk;
//   4.  valueCache, channel-major, bit-exact;
//   5.  scaleCache, bit-exact FP32.

#include "camodel_common.hpp"

#include "../common/turboquant_oracle.hpp"
// Pure C++17, no CCE dependency: the single source of truth for which
// mode-4 channel policy ships.
#include "../../kernels/ops/turboquant/turboquant_wave_protocol.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

extern "C" void launch_turboquant_reshape_and_cache_bf16(
    void* stream, const std::uint16_t* key, const std::uint16_t* value,
    std::int8_t* key_cache, std::int8_t* value_cache, float* scale_cache,
    const std::int32_t* slot_mapping, const float* pi_signs,
    const std::int32_t* tables, std::uint32_t num_tokens,
    std::uint32_t num_kv_heads, std::uint32_t head_size, std::uint32_t block_size,
    std::uint32_t num_blocks, std::uint32_t tokens_per_core,
    float inv_sqrt_len, bool reuse_mode4_flags);

namespace {

constexpr int kHeads = 4;
constexpr int kHeadSize = 256;
constexpr int kBlockSize = 16;
constexpr int kMaxSlices = 64;

using tq_oracle::ChunkRef;
using tq_oracle::Shape;

void require(bool ok, const char* what)
{
    if (!ok) {
        std::fprintf(stderr, "FIXTURE/CHECK FAILED: %s\n", what);
        throw std::runtime_error(what);
    }
}

// An integer in [-256, 256] as a BF16 bit pattern. BF16 is the top 16 bits of
// the FP32 pattern, and for these values the discarded low 16 bits are all
// zero, so the narrowing is exact rather than rounded.
std::uint16_t to_bf16(int value)
{
    const float f = static_cast<float>(value);
    const std::uint32_t bits = tq_oracle::float_bits(f);
    if ((bits & 0xFFFFu) != 0u) throw std::runtime_error("activation is not exact in BF16");
    return static_cast<std::uint16_t>(bits >> 16);
}

int chunk_plane(int chunk) { return chunk >= kHeads ? 0 : 1; }
int chunk_head(int chunk) { return chunk % kHeads; }

void usage(std::FILE* output, const char* argv0)
{
    std::fprintf(output,
                 "Usage: %s [--slices N] [--strips-per-block S]\n"
                 "          [--mode2 | --mode4-single-shot]\n"
                 "  N aligned 16-token strips (default 1, 1..%d), dispatched\n"
                 "  as ONE monolithic grid of N / S blocks of S strips each.\n"
                 "  There is no launch window: a grid wider than the AI Core\n"
                 "  count spans several physical waves inside the single\n"
                 "  dispatch, which is the case the inter-wave protocol exists\n"
                 "  for.\n"
                 "  S defaults to 1. S > 1 exercises the in-block strip\n"
                 "  stream -- one prime, one drain, and the cross-core\n"
                 "  counters carried across strip boundaries. N must be a\n"
                 "  multiple of S, and S > 1 needs the mode-2 entry point.\n"
                 "  --mode2 is the default: one readiness quorum and one\n"
                 "  release broadcast, on channels that count.\n"
                 "  --mode4-single-shot dispatches the mode-4 fallback that\n"
                 "  gives every event its own id; it runs one strip per block.\n"
                 "  --mode4-reuse is still accepted and now selects --mode2:\n"
                 "  the recycled mode-4 ring it used to name is refuted and no\n"
                 "  longer reachable from this kernel\n"
                 "  (reports/bf16_prefill_key_plane_fix.md).\n"
                 "  Audit the trace either way, do not read the PASS line:\n"
                 "  two runs of the recycled policy produced bit-exact output\n"
                 "  while their traces showed a second wait passing before its\n"
                 "  set, and a third produced the wrong answer.\n",
                 argv0, kMaxSlices);
}

int run(int slices, bool mode2, int strips_per_block)
{
    Shape shape;
    shape.num_tokens = slices * kBlockSize;
    shape.num_kv_heads = kHeads;
    shape.head_size = kHeadSize;
    shape.block_size = kBlockSize;
    shape.num_blocks = tq_oracle::fragmented_num_blocks(slices);
    require(shape.valid(), "unsupported shape");
    // tokensPerCore is the block's share of the context in TOKENS. At S = 1 it
    // is one strip per block, which is every rung this probe used to run; at
    // S > 1 one block streams S strips over the mode-2 rendezvous, which is the
    // case the in-block loop exists for and the only case in which a strip
    // boundary is crossed without a prime or a drain.
    require(strips_per_block >= 1, "strips per block must be at least 1");
    require(slices % strips_per_block == 0,
            "the strip count must be a whole number of blocks");
    require(mode2 || strips_per_block == 1,
            "the mode-4 fallback runs one strip per block: its eleven ids per "
            "sub block are exactly the eight-chunk schedule");
    const int tokens_per_core = strips_per_block * kBlockSize;
    const int head_bytes = shape.head_bytes();
    const int channels = kHeads * kHeadSize;

    // ---- fixture and oracle, shared with the INT8 instance ----------------
    std::vector<std::int8_t> key_in, value_in;
    tq_oracle::build_activations(shape, key_in, value_in);
    const std::vector<std::int32_t> slot_map = tq_oracle::build_slot_mapping(shape);
    const std::vector<std::int8_t> table_blob = tq_oracle::build_tables(shape);
    const tq_oracle::Reference ref = tq_oracle::compute_oracle_reference(
        shape, key_in.data(), value_in.data(), slot_map.data());
    require(ref.bad_slot_strip < 0,
            "the fixture's slot mapping breaks the one-strip-one-page contract");

    const tq_oracle::Audit audit = tq_oracle::audit_reference(ref);
    require(audit.nibbles_in_range, "the oracle produced a value outside INT4");
    require(audit.nibbles_covered == 16,
            "the fixture does not exercise all sixteen nibbles");
    require(audit.clamped_chunks > 0, "the INT4 upper clamp is never taken");
    require(audit.mixed_exponent_chunks > 0,
            "no chunk carries two different scale exponents");
    require(audit.distinct_exponents >= 3, "the scale exponent barely varies");
    require(audit.pages_distinct, "two chunks of one strip pack to identical pages");
    require(audit.tokens_distinct, "two tokens of one chunk quantise identically");
    require(audit.strips_distinct, "two strips pack to identical pages");

    const long long composed = tq_oracle::audit_tables(shape, table_blob.data());
    require(composed == static_cast<long long>(kHeadSize) * kHeadSize,
            "the resident rotation blocks do not compose to the oracle's matrix");

    // ---- device buffers ---------------------------------------------------
    camodel::Session session;
    camodel::GuardedDeviceBuffer key(key_in.size() * sizeof(std::uint16_t));
    camodel::GuardedDeviceBuffer value(value_in.size() * sizeof(std::uint16_t));
    camodel::GuardedDeviceBuffer key_cache(shape.cache_bytes());
    camodel::GuardedDeviceBuffer value_cache(shape.cache_bytes());
    camodel::GuardedDeviceBuffer scale_cache(shape.scale_floats() * sizeof(float));
    camodel::GuardedDeviceBuffer slot_mapping(slot_map.size() * sizeof(std::int32_t));
    camodel::GuardedDeviceBuffer pi_signs(kHeadSize * sizeof(float));
    camodel::GuardedDeviceBuffer tables(table_blob.size() * sizeof(std::uint16_t));

    // The activations and the resident blocks go up as BF16; everything else is
    // byte for byte what the INT8 instance uploads.
    auto* kb = static_cast<std::uint16_t*>(key.host_payload());
    auto* vb = static_cast<std::uint16_t*>(value.host_payload());
    for (std::size_t i = 0; i < key_in.size(); ++i) {
        kb[i] = to_bf16(key_in[i]);
        vb[i] = to_bf16(value_in[i]);
    }
    auto* tb = static_cast<std::uint16_t*>(tables.host_payload());
    for (std::size_t i = 0; i < table_blob.size(); ++i) tb[i] = to_bf16(table_blob[i]);
    std::memcpy(slot_mapping.host_payload(), slot_map.data(),
                slot_map.size() * sizeof(std::int32_t));
    auto* ps = static_cast<float*>(pi_signs.host_payload());
    for (int ch = 0; ch < kHeadSize; ++ch) ps[ch] = static_cast<float>(tq_oracle::pi_sign(ch));

    key.upload();
    value.upload();
    slot_mapping.upload();
    pi_signs.upload();
    tables.upload();

    // ONE launch of shape.strips() grid blocks, however many physical waves
    // that spans. This used to be a loop over capped windows, because the
    // mode-4 intra-block flag registers are per physical AI Core block and the
    // scheduler does not clear them between waves, so re-entering the kernel
    // was the only way to re-arm them. The kernel now carries the inter-wave
    // protocol itself -- the boundary guard before InitSocState, the bounded
    // cross-core channels, and the epilogue drain of every queue of every
    // role (turboquant_wave_protocol.hpp, guidelines section 6.5) -- so a
    // multi-wave grid is a single dispatch and that is what has to be tested.
    // A capped window would re-arm everything at each boundary and hide it.
    launch_turboquant_reshape_and_cache_bf16(
        session.stream,
        static_cast<const std::uint16_t*>(key.payload()),
        static_cast<const std::uint16_t*>(value.payload()),
        static_cast<std::int8_t*>(key_cache.payload()),
        static_cast<std::int8_t*>(value_cache.payload()),
        static_cast<float*>(scale_cache.payload()),
        static_cast<const std::int32_t*>(slot_mapping.payload()),
        static_cast<const float*>(pi_signs.payload()),
        static_cast<const std::int32_t*>(tables.payload()),
        static_cast<std::uint32_t>(shape.num_tokens),
        static_cast<std::uint32_t>(shape.num_kv_heads),
        static_cast<std::uint32_t>(shape.head_size),
        static_cast<std::uint32_t>(shape.block_size),
        static_cast<std::uint32_t>(shape.num_blocks),
        static_cast<std::uint32_t>(tokens_per_core), shape.inv_sqrt_len(), mode2);
    camodel::check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");

    key_cache.download();
    value_cache.download();
    scale_cache.download();
    key.download();
    value.download();
    slot_mapping.download();
    pi_signs.download();
    tables.download();

    key.assert_guards();
    value.assert_guards();
    key_cache.assert_guards();
    value_cache.assert_guards();
    scale_cache.assert_guards();
    slot_mapping.assert_guards();
    pi_signs.assert_guards();
    tables.assert_guards();

    const auto* kc = static_cast<const std::uint8_t*>(key_cache.host_payload());
    const auto* vc = static_cast<const std::uint8_t*>(value_cache.host_payload());
    const auto* sc = static_cast<const float*>(scale_cache.host_payload());

    // ---- check 2: nothing outside a written page moved --------------------
    std::size_t untouched = 0;
    for (int page = 0; page < shape.num_blocks; ++page) {
        if (ref.page_written[static_cast<std::size_t>(page)] != 0) continue;
        const std::size_t base = ref.page_offset(page);
        for (int i = 0; i < shape.page_bytes(); ++i) {
            const std::size_t at = base + static_cast<std::size_t>(i);
            const auto lane =
                static_cast<std::uint8_t>((camodel::kCanary >> (8 * (at % 4))) & 0xFFu);
            require(kc[at] == lane, "keyCache byte on an unwritten page was overwritten");
            require(vc[at] == lane, "valueCache byte on an unwritten page was overwritten");
            ++untouched;
        }
        for (int plane = 0; plane < tq_oracle::kPlanes; ++plane)
            for (int head = 0; head < kHeads; ++head) {
                const std::size_t at = ref.scale_offset(plane, page, head);
                for (int t = 0; t < shape.block_size; ++t)
                    require(tq_oracle::float_bits(sc[at + static_cast<std::size_t>(t)]) ==
                                camodel::kCanary,
                            "scaleCache float on an unwritten page moved");
            }
    }

    // ---- checks 3-5: bit-exact payloads ----------------------------------
    std::size_t key_bytes = 0, value_bytes = 0, scales = 0;
    int failing = 0;
    // Every mismatching chunk is counted; the first kReportChunks of them are
    // printed. An earlier revision broke out of both loops at the third
    // failure, which hid the *shape* of a failure set: the four value chunks
    // come first in chunk order, so a defect confined to the key plane was
    // reported as "three chunks differ" with no way to see that the fourth and
    // fifth did too. The count below is over the whole grid however many lines
    // it prints, so "how many chunks are wrong" and "which ones were shown"
    // are separate facts now.
    constexpr int kReportChunks = 64;
    for (int strip = 0; strip < shape.strips(); ++strip)
        for (int chunk = 0; chunk < 2 * kHeads; ++chunk) {
            const int plane = chunk_plane(chunk);
            const int head = chunk_head(chunk);
            const ChunkRef& want = ref.chunk(strip, plane, head);
            const std::uint8_t* got =
                (plane == 0 ? kc : vc) + ref.head_offset(want.page, head);
            const float* got_scales = sc + ref.scale_offset(plane, want.page, head);

            int bad_bytes = 0, first_bad = -1;
            for (int i = 0; i < head_bytes; ++i)
                if (got[i] != want.packed[static_cast<std::size_t>(i)]) {
                    if (first_bad < 0) first_bad = i;
                    ++bad_bytes;
                }
            int bad_scales = 0;
            for (int t = 0; t < shape.block_size; ++t)
                if (tq_oracle::float_bits(got_scales[t]) !=
                    tq_oracle::float_bits(want.scale[static_cast<std::size_t>(t)]))
                    ++bad_scales;

            if (bad_bytes == 0 && bad_scales == 0) {
                if (plane == 0)
                    key_bytes += static_cast<std::size_t>(head_bytes);
                else
                    value_bytes += static_cast<std::size_t>(head_bytes);
                scales += static_cast<std::size_t>(shape.block_size);
                continue;
            }
            ++failing;
            if (failing > kReportChunks) continue;
            std::fprintf(stderr,
                         "\nstrip %d page %d chunk %d (%s head %d): %d/%d bytes differ "
                         "(first byte %d: device 0x%02x oracle 0x%02x), %d/%d scales "
                         "differ\n",
                         strip, want.page, chunk, plane == 0 ? "key" : "value", head,
                         bad_bytes, head_bytes, first_bad,
                         first_bad < 0 ? 0u : static_cast<unsigned>(got[first_bad]),
                         first_bad < 0
                             ? 0u
                             : static_cast<unsigned>(
                                   want.packed[static_cast<std::size_t>(first_bad)]),
                         bad_scales, shape.block_size);
        }
    if (failing != 0) {
        std::fprintf(stderr, "\n%d of %d chunks mismatch (%d printed)\n", failing,
                     shape.strips() * 2 * kHeads,
                     failing < kReportChunks ? failing : kReportChunks);
        throw std::runtime_error("packed cache page mismatch");
    }

    const int chunks = shape.strips() * 2 * kHeads;
    std::printf("PASS turboquant_reshape_and_cache_bf16: %d strips x 16 tokens x %d "
                "channels, %d chunks, BF16 activations\n",
                shape.strips(), channels, chunks);
    const int blocks = shape.strips() / strips_per_block;
    std::printf("  1 launch of %d grid blocks of %d strip(s) spanning %d physical "
                "wave(s) of 32, %s, strip s -> physical page 2s+1 of %d\n",
                blocks, strips_per_block, (blocks + 31) / 32,
                mode2 ? "mode-2 readiness quorum and release broadcast"
                      : "mode-4 single-shot channels",
                shape.num_blocks);
    std::printf("  keyCache   %zu/%d bytes bit-exact, key-major\n", key_bytes,
                shape.strips() * kHeads * head_bytes);
    std::printf("  valueCache %zu/%d bytes bit-exact, channel-major\n", value_bytes,
                shape.strips() * kHeads * head_bytes);
    std::printf("  scaleCache %zu/%d FP32 bit-exact\n", scales, chunks * shape.block_size);
    std::printf("  %zu cache bytes on the %d unwritten pages untouched, %d/%d guard "
                "bytes intact\n",
                untouched, shape.num_blocks - shape.strips(),
                8 * 2 * static_cast<int>(camodel::kGuardBytes),
                8 * 2 * static_cast<int>(camodel::kGuardBytes));
    std::printf("  identical to what the INT8 instance produces for the same integer "
                "activations: same oracle, no BF16-specific reference\n");
    std::printf("  K0 = %d (BF16 fractal 16x16), Hadamard group 32 = 2 fractals, "
                "%d resident blocks of 32x32 BF16 (%zu B), %lld composed entries\n",
                32 / 2, 2 * shape.groups(), table_blob.size() * sizeof(std::uint16_t),
                composed);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    int slices = 1;
    int strips_per_block = 1;
    // The shared header's constant, so this runner cannot disagree with the
    // kernel or the NPU harnesses about which discipline ships.
    bool mode2 = turboquant_wave::kReshapeBf16DefaultMode2;
    bool saw_slices = false;
    bool saw_strips = false;
    // One parser for both integer options, so --slices and --strips-per-block
    // cannot drift in how they reject a value.
    const auto parse = [](std::string_view text, int low, int high, int& out) {
        int value = 0;
        const auto got =
            std::from_chars(text.data(), text.data() + text.size(), value);
        if (got.ec != std::errc() || got.ptr != text.data() + text.size() ||
            value < low || value > high)
            return false;
        out = value;
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argc == 2 ? stdout : stderr, argv[0]);
            return argc == 2 ? 0 : 2;
        }
        // --mode4-reuse used to name the recycled two-slot ring. That ring is
        // refuted and unreachable, so the old spelling selects the discipline
        // that replaced it rather than silently dispatching the fallback.
        if (arg == "--mode2" || arg == "--mode4-reuse") {
            mode2 = true;
            continue;
        }
        if (arg == "--mode4-single-shot") {
            mode2 = false;
            continue;
        }
        if (arg == "--slices" && !saw_slices && i + 1 < argc &&
            parse(argv[i + 1], 1, kMaxSlices, slices)) {
            ++i;
            saw_slices = true;
            continue;
        }
        if (arg == "--strips-per-block" && !saw_strips && i + 1 < argc &&
            parse(argv[i + 1], 1, kMaxSlices, strips_per_block)) {
            ++i;
            saw_strips = true;
            continue;
        }
        usage(stderr, argv[0]);
        return 2;
    }
    const char* const policy = mode2 ? "mode-2 quorum and broadcast"
                                     : "mode-4 single-shot channels";
    try {
        return run(slices, mode2, strips_per_block);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s (%d slices, %d strip(s) per block, %s): %s\n",
                     "turboquant_reshape_and_cache_bf16", slices, strips_per_block,
                     policy, error.what());
        return 1;
    }
}
