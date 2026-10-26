#pragma once
// Physical-device harness for turboquant_reshape_and_cache_a8.
//
// The golden model lives in tests/common/turboquant_oracle.hpp and is shared
// with the CAModel runner. This header is device plumbing only: ACL session,
// 32-byte aligned buffers, a canary pre-fill, the launch, and a verifier that
// reports per physical page rather than per byte.
//
// Three properties of the fixture are what make a multi-page run able to fail
// honestly, and all three come from the shared oracle:
//
//   * numBlocks is 2 * strips + 2, not strips, so the cache has pages no strip
//     owns. A page index that leaves its strip's slot -- the multi-core bug
//     this harness was written to catch -- lands on a canary page or outside
//     the buffer instead of silently overwriting a neighbour's correct page;
//   * strip s maps to page 2s + 1, so the page index is never the grid block
//     index. A kernel that took its page from GetBlockIdx() instead of from
//     slotMapping[0] passes an identity mapping and fails this one;
//   * every (strip, plane, head) panel is distinct, so a wrong-strip,
//     wrong-core or wrong-token read cannot produce the right page.

#include "../common/turboquant_oracle.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void launch_turboquant_reshape_and_cache_a8(
    void*, const std::int8_t*, const std::int8_t*, std::int8_t*, std::int8_t*,
    float*, const std::int32_t*, const float*, const std::int32_t*,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, float);

namespace tq_npu {

using tq_oracle::ChunkRef;
using tq_oracle::Shape;

constexpr int kHeads = 4;
constexpr int kHeadSize = 256;
constexpr int kBlockSize = 16;

// Output buffers are filled with this before the launch, so an untouched page
// is distinguishable from a page written with wrong data, and a partially
// written page from a wholly missing one.
constexpr std::uint8_t kCanaryByte = 0xAA;
constexpr std::uint32_t kCanaryWord = 0xAAAAAAAAu;

inline void check(aclError err, const char* op)
{
    if (err != ACL_SUCCESS)
        throw std::runtime_error(std::string(op) + " failed: " + std::to_string(err));
}

inline void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

inline std::uint32_t bits(float value) { return tq_oracle::float_bits(value); }

// --------------------------------------------------------------- the fixture
struct Fixture {
    Shape shape;
    std::vector<std::int8_t> key, value, tables;
    std::vector<std::int32_t> slots;
    std::vector<float> signs;
    tq_oracle::Reference ref;
    tq_oracle::Audit audit;

    explicit Fixture(int tokens)
    {
        require(tokens > 0 && tokens % kBlockSize == 0,
                "the context must be a whole number of 16-token strips");
        shape.num_tokens = tokens;
        shape.num_kv_heads = kHeads;
        shape.head_size = kHeadSize;
        shape.block_size = kBlockSize;
        shape.num_blocks = tq_oracle::fragmented_num_blocks(tokens / kBlockSize);
        require(shape.valid(), "unsupported shape");

        tq_oracle::build_activations(shape, key, value);
        slots = tq_oracle::build_slot_mapping(shape);
        tables = tq_oracle::build_tables(shape);
        signs.resize(static_cast<std::size_t>(kHeadSize));
        for (int c = 0; c < kHeadSize; ++c)
            signs[static_cast<std::size_t>(c)] = static_cast<float>(tq_oracle::pi_sign(c));

        // Phase one of the two-phase oracle: the resident blocks the device
        // multiplies by must compose to the matrix the oracle applies, and the
        // fixture must be able to detect a wrong read at all.
        require(tq_oracle::audit_tables(shape, tables.data()) ==
                    static_cast<long long>(kHeadSize) * kHeadSize,
                "the resident rotation blocks do not compose to the oracle's matrix");
        ref = tq_oracle::compute_oracle_reference(shape, key.data(), value.data(),
                                                 slots.data());
        require(ref.bad_slot_strip < 0, "the fixture's own slot mapping breaks the "
                                        "kernel's one-strip-one-page contract");
        audit = tq_oracle::audit_reference(ref);
        require(audit.nibbles_in_range, "the oracle produced a value outside INT4");
        require(audit.nibbles_covered == 16,
                "the fixture does not exercise all sixteen nibbles");
        require(audit.clamped_chunks > 0, "the INT4 upper clamp is never taken");
        require(audit.mixed_exponent_chunks > 0,
                "no chunk carries two different scale exponents");
        require(audit.distinct_exponents >= 3, "the scale exponent barely varies");
        // Distinct packed pages, not distinct maxima: two chunks may share a
        // max |y| without being interchangeable, because the nibbles are scale
        // invariant. See Audit::chunk_max_collisions.
        require(audit.pages_distinct, "two chunks of one strip pack to identical pages");
        require(audit.chunk_pairs > 0 && audit.chunk_max_collisions < audit.chunk_pairs,
                "every pair of chunks in a strip shares a maximum: degenerate fixture");
        require(audit.tokens_distinct, "two tokens of one chunk quantise identically");
        require(audit.strips_distinct, "two strips pack to identical pages");
        require(audit.sign_negatives >= kHeadSize / 4 &&
                    audit.sign_negatives <= 3 * kHeadSize / 4,
                "piSigns is degenerate, a dropped sign multiply would not show");
    }

    int strips() const { return shape.strips(); }
    int tokens() const { return shape.num_tokens; }
};

// ------------------------------------------------------------------- device
struct Session {
    aclrtStream stream = nullptr;
    Session()
    {
        check(aclInit(nullptr), "aclInit");
        check(aclrtSetDevice(0), "aclrtSetDevice");
        check(aclrtCreateStream(&stream), "aclrtCreateStream");
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session()
    {
        if (stream) aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }
};

struct Buffer {
    void* ptr = nullptr;
    std::size_t bytes;
    explicit Buffer(std::size_t n) : bytes(n)
    {
        require(n != 0 && n % 32 == 0, "NPU buffer size must be 32-byte aligned");
        check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
        require(reinterpret_cast<std::uintptr_t>(ptr) % 32 == 0, "NPU allocation is unaligned");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (ptr) aclrtFree(ptr); }
    template <typename T> void upload(const std::vector<T>& src)
    {
        require(src.size() * sizeof(T) == bytes, "upload size mismatch");
        check(aclrtMemcpy(ptr, bytes, src.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
              "aclrtMemcpy H2D");
    }
    template <typename T> void download(std::vector<T>& dst) const
    {
        require(dst.size() * sizeof(T) == bytes, "download size mismatch");
        check(aclrtMemcpy(dst.data(), bytes, ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy D2H");
    }
    void fill_canary()
    {
        check(aclrtMemset(ptr, bytes, kCanaryByte, bytes), "aclrtMemset canary");
    }
};

struct A8Device {
    Buffer key, value, key_cache, value_cache, scale_cache, slots, signs, tables;
    explicit A8Device(const Fixture& f)
        : key(f.key.size()), value(f.value.size()),
          key_cache(f.shape.cache_bytes()), value_cache(f.shape.cache_bytes()),
          scale_cache(f.shape.scale_floats() * sizeof(float)),
          slots(f.slots.size() * sizeof(std::int32_t)),
          signs(f.signs.size() * sizeof(float)), tables(f.tables.size())
    {
        key.upload(f.key);
        value.upload(f.value);
        slots.upload(f.slots);
        signs.upload(f.signs);
        tables.upload(f.tables);
        // Every output byte is 0xAA going in: an untouched page stays 0xAA, so
        // "never written" and "written wrong" are different diagnoses.
        key_cache.fill_canary();
        value_cache.fill_canary();
        scale_cache.fill_canary();
    }

    void launch(aclrtStream stream, const Fixture& f)
    {
        launch_turboquant_reshape_and_cache_a8(
            stream, static_cast<const std::int8_t*>(key.ptr),
            static_cast<const std::int8_t*>(value.ptr),
            static_cast<std::int8_t*>(key_cache.ptr),
            static_cast<std::int8_t*>(value_cache.ptr),
            static_cast<float*>(scale_cache.ptr),
            static_cast<const std::int32_t*>(slots.ptr),
            static_cast<const float*>(signs.ptr),
            static_cast<const std::int32_t*>(tables.ptr),
            static_cast<std::uint32_t>(f.shape.num_tokens),
            static_cast<std::uint32_t>(f.shape.num_kv_heads),
            static_cast<std::uint32_t>(f.shape.head_size),
            static_cast<std::uint32_t>(f.shape.block_size),
            static_cast<std::uint32_t>(f.shape.num_blocks),
            static_cast<std::uint32_t>(f.shape.block_size), f.shape.inv_sqrt_len());
    }
};

// ----------------------------------------------------------------- verifier
// Per page, because that is the unit the bug moves in. A byte-by-byte report
// over 32 pages says "512342 mismatches" and nothing else; a per-page report
// says which pages landed, which are untouched, and -- when a page holds the
// wrong strip's data -- which strip it actually came from, which is the
// signature of a block-index error and is invisible in an aggregate count.
struct PageVerdict {
    int page = 0;
    int owner_strip = -1;  // -1 when no strip maps here
    std::size_t key_bad = 0;
    std::size_t value_bad = 0;
    std::size_t scale_bad = 0;
    std::size_t canary_bytes = 0;  // still 0xAA inside a page a strip owns
    bool ok() const { return key_bad == 0 && value_bad == 0 && scale_bad == 0; }
};

inline void verify(const Fixture& f, const A8Device& d)
{
    const Shape& shape = f.shape;
    const tq_oracle::Reference& ref = f.ref;

    std::vector<std::int8_t> key(shape.cache_bytes()), value(shape.cache_bytes());
    std::vector<float> scales(shape.scale_floats());
    d.key_cache.download(key);
    d.value_cache.download(value);
    d.scale_cache.download(scales);

    std::vector<int> owner(static_cast<std::size_t>(shape.num_blocks), -1);
    for (int strip = 0; strip < shape.strips(); ++strip)
        owner[static_cast<std::size_t>(tq_oracle::page_of_strip(strip))] = strip;

    std::vector<PageVerdict> verdicts;
    std::size_t stray = 0;  // canary bytes clobbered on a page no strip owns
    std::size_t exact_key = 0, exact_value = 0, exact_scales = 0;

    for (int page = 0; page < shape.num_blocks; ++page) {
        const std::size_t base = ref.page_offset(page);
        if (ref.page_written[static_cast<std::size_t>(page)] == 0) {
            for (int i = 0; i < shape.page_bytes(); ++i) {
                const std::size_t at = base + static_cast<std::size_t>(i);
                if (static_cast<std::uint8_t>(key[at]) != kCanaryByte) ++stray;
                if (static_cast<std::uint8_t>(value[at]) != kCanaryByte) ++stray;
            }
            for (int plane = 0; plane < tq_oracle::kPlanes; ++plane)
                for (int head = 0; head < shape.num_kv_heads; ++head) {
                    const std::size_t at = ref.scale_offset(plane, page, head);
                    for (int t = 0; t < shape.block_size; ++t)
                        if (bits(scales[at + static_cast<std::size_t>(t)]) != kCanaryWord)
                            ++stray;
                }
            continue;
        }

        PageVerdict v;
        v.page = page;
        v.owner_strip = owner[static_cast<std::size_t>(page)];
        for (int i = 0; i < shape.page_bytes(); ++i) {
            const std::size_t at = base + static_cast<std::size_t>(i);
            if (key[at] != ref.key_cache[at]) ++v.key_bad;
            if (value[at] != ref.value_cache[at]) ++v.value_bad;
            if (static_cast<std::uint8_t>(key[at]) == kCanaryByte &&
                static_cast<std::uint8_t>(ref.key_cache[at]) != kCanaryByte)
                ++v.canary_bytes;
            if (static_cast<std::uint8_t>(value[at]) == kCanaryByte &&
                static_cast<std::uint8_t>(ref.value_cache[at]) != kCanaryByte)
                ++v.canary_bytes;
        }
        for (int plane = 0; plane < tq_oracle::kPlanes; ++plane)
            for (int head = 0; head < shape.num_kv_heads; ++head) {
                const std::size_t at = ref.scale_offset(plane, page, head);
                for (int t = 0; t < shape.block_size; ++t) {
                    const std::size_t i = at + static_cast<std::size_t>(t);
                    if (bits(scales[i]) != bits(ref.scale_cache[i])) ++v.scale_bad;
                }
            }
        if (v.ok()) {
            exact_key += static_cast<std::size_t>(shape.page_bytes());
            exact_value += static_cast<std::size_t>(shape.page_bytes());
            exact_scales += static_cast<std::size_t>(tq_oracle::kPlanes) *
                            shape.num_kv_heads * shape.block_size;
        }
        verdicts.push_back(v);
    }

    std::size_t failed = 0;
    for (const PageVerdict& v : verdicts) failed += !v.ok();

    if (failed != 0 || stray != 0) {
        std::fprintf(stderr, "\n%d tokens, %d strips over %d physical pages:\n",
                     shape.num_tokens, shape.strips(), shape.num_blocks);
        int detailed = 0;
        for (const PageVerdict& v : verdicts) {
            if (v.ok()) continue;
            std::fprintf(stderr,
                         "  page %3d (strip %3d): FAIL  key %zu/%d value %zu/%d "
                         "scales %zu/%d bytes still canary %zu\n",
                         v.page, v.owner_strip, v.key_bad, shape.page_bytes(),
                         v.value_bad, shape.page_bytes(), v.scale_bad,
                         tq_oracle::kPlanes * shape.num_kv_heads * shape.block_size,
                         v.canary_bytes);
            // Whose data is on this page? A block-index error puts strip s's
            // panel on another strip's page, and this is where that shows.
            if (++detailed > 4) continue;
            for (int plane = 0; plane < tq_oracle::kPlanes; ++plane)
                for (int head = 0; head < shape.num_kv_heads; ++head) {
                    const auto* got = reinterpret_cast<const std::uint8_t*>(
                        (plane == 0 ? key : value).data() + ref.head_offset(v.page, head));
                    int best_strip = -1;
                    std::size_t best_same = 0;
                    for (int s = 0; s < shape.strips(); ++s) {
                        const ChunkRef& c = ref.chunk(s, plane, head);
                        if (c.packed.empty()) continue;
                        std::size_t same = 0;
                        for (std::size_t i = 0; i < c.packed.size(); ++i)
                            same += (got[i] == c.packed[i]);
                        if (same > best_same) {
                            best_same = same;
                            best_strip = s;
                        }
                    }
                    if (best_strip >= 0 && best_strip != v.owner_strip &&
                        best_same * 4 > ref.chunk(0, plane, head).packed.size())
                        std::fprintf(stderr,
                                     "      %s head %d looks like strip %d: %zu/%zu bytes\n",
                                     plane == 0 ? "key  " : "value", head, best_strip,
                                     best_same, ref.chunk(0, plane, head).packed.size());
                }
        }
        if (stray != 0)
            std::fprintf(stderr,
                         "  %zu canary bytes clobbered on pages no strip owns\n", stray);
        throw std::runtime_error(std::to_string(failed) + " of " +
                                 std::to_string(verdicts.size()) +
                                 " pages mismatched, " + std::to_string(stray) +
                                 " stray canary bytes");
    }

    std::printf("PASS %d tokens: %d strips -> pages", shape.num_tokens, shape.strips());
    for (int strip = 0; strip < shape.strips() && strip < 8; ++strip)
        std::printf(" %d", tq_oracle::page_of_strip(strip));
    if (shape.strips() > 8) std::printf(" ...");
    std::printf(" of %d\n", shape.num_blocks);
    std::printf("  %zu key bytes, %zu value bytes, %zu FP32 scales bit-exact "
                "(%zu key nibbles, %zu value nibbles)\n",
                exact_key, exact_value, exact_scales, exact_key * 2, exact_value * 2);
    std::printf("  %d pages no strip owns still hold the 0x%02X canary, in both "
                "caches and the scale cache\n",
                shape.num_blocks - shape.strips(), kCanaryByte);
}

}  // namespace tq_npu
