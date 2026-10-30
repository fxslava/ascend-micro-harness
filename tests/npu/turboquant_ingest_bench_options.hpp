#pragma once
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tq_ingest_bench {
constexpr std::uint64_t kStripTokens = 16, kMaxTokens = 4096 * kStripTokens;
struct Options {
    std::vector<std::string> models;
    std::vector<std::uint64_t> contexts, batches, chunks;
    int warmups = 50, samples = 500;
    bool single = false, recycled = true, skip_native = false, help = false;
};
struct CacheLayout { int blocks; std::vector<std::int32_t> slots; };
inline CacheLayout cache_layout(std::uint64_t context, std::uint64_t batch,
                                std::uint64_t chunk) {
    if (batch == 0 || chunk == 0 || chunk % kStripTokens || chunk > kMaxTokens ||
        batch > kMaxTokens / chunk || context < chunk || context % kStripTokens ||
        context > (static_cast<std::uint64_t>(std::numeric_limits<int>::max()) - 32) / batch)
        throw std::invalid_argument("invalid ingest cache geometry");
    CacheLayout layout{static_cast<int>(batch * context / kStripTokens + 2), {}};
    layout.slots.reserve(static_cast<std::size_t>(batch * chunk));
    for (std::uint64_t b = 0; b < batch; ++b)
        for (std::uint64_t t = 0; t < chunk; ++t)
            layout.slots.push_back(static_cast<std::int32_t>(b * context + context - chunk + t));
    return layout;
}
inline std::uint64_t integer(std::string_view text) {
    std::uint64_t n = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), n);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::invalid_argument("expected unsigned integer: " + std::string(text));
    return n;
}
inline std::vector<std::uint64_t> list(std::string_view text) {
    std::vector<std::uint64_t> out;
    do {
        const auto comma = text.find(',');
        out.push_back(integer(text.substr(0, comma)));
        if (comma == std::string_view::npos) return out;
        text.remove_prefix(comma + 1);
    } while (!text.empty());
    throw std::invalid_argument("trailing comma in integer list");
}
inline Options parse(int argc, char** argv, bool default_recycled) {
    Options o;
    o.recycled = default_recycled;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") { o.help = true; return o; }
        if (arg == "--single") { o.single = true; continue; }
        if (arg == "--mode4-reuse") { o.recycled = true; continue; }
        if (arg == "--mode4-single-shot") { o.recycled = false; continue; }
        if (arg == "--skip-native") { o.skip_native = true; continue; }
        if (i + 1 >= argc) throw std::invalid_argument("option needs a value: " + std::string(arg));
        const std::string value(argv[++i]);
        if (arg == "--model") {
            if (value.empty()) throw std::invalid_argument("model label must not be empty");
            o.models.push_back(value);
            continue;
        }
        if (arg == "--warmups" || arg == "--samples") {
            const auto n = integer(value);
            if (n > 1000000 || (arg == "--samples" && n == 0))
                throw std::invalid_argument("invalid warmup/sample count");
            (arg == "--warmups" ? o.warmups : o.samples) = static_cast<int>(n);
            continue;
        }
        std::vector<std::uint64_t>* destination = nullptr;
        if (arg == "--context" || arg == "--contexts") destination = &o.contexts;
        else if (arg == "--batch" || arg == "--batches") destination = &o.batches;
        else if (arg == "--tokens" || arg == "--chunk-tokens" || arg == "--strips")
            destination = &o.chunks;
        else throw std::invalid_argument("unknown option: " + std::string(arg));
        auto values = list(value);
        for (auto n : values) {
            if (arg == "--strips") {
                if (n > kMaxTokens / kStripTokens) throw std::invalid_argument("too many strips");
                n *= kStripTokens;
            }
            destination->push_back(n);
        }
    }
    if (o.models.empty()) o.models.emplace_back("fixture-4H256");
    if (o.batches.empty()) o.batches.push_back(1);
    // Preserve the old benchmark's effective default: the shared wave ladder
    // ended at 128 strips and the benchmark selected that last rung.
    if (o.chunks.empty()) o.chunks.push_back(2048);
    for (auto chunk : o.chunks) {
        if (chunk == 0 || chunk % kStripTokens || chunk > kMaxTokens)
            throw std::invalid_argument("chunk tokens must be a multiple of 16 in 16..65536");
        for (auto batch : o.batches) {
            if (batch == 0 || batch > kMaxTokens / chunk)
                throw std::invalid_argument("B * chunk_tokens must not exceed 65536");
            for (auto context : o.contexts) {
                // Slots are INT32 and one request's ingest must fit its horizon.
                if (context < chunk || context % kStripTokens ||
                    context > (static_cast<std::uint64_t>(std::numeric_limits<int>::max()) - 32) / batch)
                    throw std::invalid_argument("context must fit its chunk, be 16-aligned, and B*S must fit INT32 slots");
            }
        }
    }
    if (o.single) { o.warmups = 0; o.samples = 1; }
    return o;
}
inline void usage(const char* program) {
    std::printf("Usage: %s [--model TAG]... [--context S[,S...]] [--batch B[,B...]]\n"
                "  [--chunk-tokens N[,N...]] [--tokens N] [--strips N]\n"
                "  [--single] [--warmups N] [--samples N] [--skip-native]\n"
                "  [--mode4-reuse | --mode4-single-shot]\n"
                "All supplied models, horizons, batches and chunks are swept.\n"
                "Model tags are labels for the fixed 4-KV-head x 256-channel fixture.\n"
                "S is the per-request cache horizon; N_chunk = B * chunk_tokens.\n"
                "Default: fixture-4H256, S=chunk_tokens=2048, B=1, 50 warmups/500 samples.\n"
                "Both warm and L2-pressure modes run. --single uses one sample per leg/mode.\n"
                "ASCEND_BENCH_TQ_AUDIT_PREFILL_CSV=PATH appends the audit metrics to CSV.\n",
                program);
}
}  // namespace tq_ingest_bench
