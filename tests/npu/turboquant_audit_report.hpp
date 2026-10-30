#pragma once
// Host-only ingestion reporting. No ACL dependency: rendering can be checked
// without a device, using the same code as the physical benchmark.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tq_ingest_audit {
struct Stats { double min = 0, median = 0, p99 = 0; };
struct Row {
    std::string model, mode, native_note;
    std::string measurement_source = "device_event";
    std::uint64_t context = 0, batch = 1, chunk_tokens = 0;
    std::uint64_t bytes_written = 0, bytes_moved = 0;
    bool recycled = true;
    Stats a8, cube, vector;
    std::optional<Stats> native;
};

inline std::uint64_t tokens(const Row& r) {
    if (r.batch == 0 || r.chunk_tokens == 0 ||
        r.batch > std::numeric_limits<std::uint64_t>::max() / r.chunk_tokens)
        throw std::invalid_argument("invalid or overflowing B * chunk_tokens");
    return r.batch * r.chunk_tokens;
}
inline const char* tier(std::uint64_t s) {
    if (s <= 8192) return "Short";
    if (s <= 65536) return "Mid";
    if (s <= 262144) return "Long";
    return "Extreme";
}
inline std::string commas(std::uint64_t n) {
    std::string s = std::to_string(n);
    for (std::size_t i = s.size(); i > 3; i -= 3) s.insert(i - 3, ",");
    return s;
}
inline bool positive(double n) { return std::isfinite(n) && n > 0; }
inline std::optional<double> ratio(double numerator, double denominator) {
    if (!positive(numerator) || !positive(denominator)) return std::nullopt;
    const double value = numerator / denominator;
    return positive(value) ? std::optional<double>(value) : std::nullopt;
}
inline std::string number(std::optional<double> n, int precision = 3) {
    if (!n || !std::isfinite(*n)) return "N/A";
    char buffer[384];
    std::snprintf(buffer, sizeof(buffer), "%.*f", precision, *n);
    return buffer;
}
inline std::string latency(double n) {
    return number(positive(n) ? std::optional<double>(n) : std::nullopt);
}
inline std::string speed(std::optional<double> n) {
    return n ? number(n) + "x" : "N/A";
}
inline std::optional<double> write_bandwidth(const Row& r) {
    if (!positive(r.cube.median)) return std::nullopt;
    const double n = static_cast<double>(r.bytes_written) / r.cube.median / 1000.0;
    return std::isfinite(n) ? std::optional<double>(n) : std::nullopt;
}
inline std::optional<double> native_speed(const Row& r, double denominator) {
    return r.native ? ratio(r.native->median, denominator) : std::nullopt;
}
inline std::string note(const Row& r) {
    std::string s = number(write_bandwidth(r), 1) + " GB/s effective cache write";
    s += "; " + r.mode + (r.recycled ? "; recycled" : "; single-shot");
    if (const auto gain = ratio(r.vector.median, r.cube.median)) {
        const double percent = (*gain - 1.0) * 100.0;
        if (std::isfinite(percent))
            s += "; " + std::string(percent >= 0 ? "+" : "") +
                 number(percent, 1) + "% throughput gain vs BF16-vec";
    }
    if (const auto vs_native = native_speed(r, r.cube.median)) {
        if (*vs_native > 1) s += "; crosses native V5 speed";
    } else s += "; " + (r.native_note.empty() ? "native unavailable" : r.native_note);
    return s;
}
using Cells = std::array<std::string, 9>;
inline Cells headings() {
    return {"Tier", "Model", "Context (S)", "Batch (B)", "Tokens Ingested",
            "TQ Ingest BF16-Cube (μs)", "Native V5 Ingest (μs)",
            "Speedup vs V5", "Throughput / Note"};
}
inline Cells cells(const Row& r) {
    const auto primary = native_speed(r, r.cube.median);
    const std::string speedup = r.native && positive(r.native->median)
        ? speed(primary) + " (" + speed(native_speed(r, r.vector.median)) + ")"
        : "N/A";
    return {tier(r.context), r.model, commas(r.context), commas(r.batch),
            commas(tokens(r)), latency(r.cube.median) + " (" +
            latency(r.vector.median) + " / " + latency(r.cube.min) + ")",
            r.native ? latency(r.native->median) : "N/A", speedup, note(r)};
}
inline std::string markdown(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '\n' || c == '\r') out += ' ';
        else {
            if (c == '|' || c == '\\') out += '\\';
            out += c;
        }
    }
    return out;
}
inline std::size_t display_width(const std::string& s) {
    // Count UTF-8 codepoints so the micro symbol does not shift a column.
    std::size_t width = 0;
    for (unsigned char c : s) if ((c & 0xc0) != 0x80) ++width;
    return width;
}
inline void PrintTableA(std::FILE* out, const std::vector<Row>& rows) {
    std::vector<Cells> lines{headings()};
    for (const Row& r : rows) lines.push_back(cells(r));
    std::array<std::size_t, 9> widths{};
    for (auto& line : lines) for (std::size_t i = 0; i < line.size(); ++i) {
        line[i] = markdown(line[i]);
        widths[i] = std::max(widths[i], display_width(line[i]));
    }
    const auto print = [&](const Cells& line) {
        std::fprintf(out, "|");
        for (std::size_t i = 0; i < line.size(); ++i) {
            std::fprintf(out, " %s", line[i].c_str());
            for (std::size_t p = display_width(line[i]); p < widths[i]; ++p)
                std::fputc(' ', out);
            std::fprintf(out, " |");
        }
        std::fputc('\n', out);
    };
    print(lines.front());
    Cells separators;
    for (std::size_t i = 0; i < separators.size(); ++i) {
        const bool centered = i >= 2 && i <= 7;
        separators[i] = ":" + std::string(widths[i] - (centered ? 2 : 1), '-') +
                        (centered ? ":" : "");
    }
    print(separators);
    for (std::size_t i = 1; i < lines.size(); ++i) print(lines[i]);
}

inline std::string csv_field(const std::string& s) {
    if (s.find_first_of(",\"\r\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '"'; out += c; }
    return out + '"';
}
inline std::string csv_line(const std::vector<std::string>& fields) {
    std::string out;
    bool first = true;
    for (const auto& f : fields) {
        if (!first) out += ',';
        first = false;
        out += csv_field(f);
    }
    return out;
}
inline std::vector<std::string> csv_headings() {
    const auto h = headings();
    std::vector<std::string> out(h.begin(), h.end());
    for (const char* s : {"mode", "measurement_source", "chunk_tokens", "tokens_ingested", "mode4_recycled",
                         "a8_median_us", "cube_median_us", "vector_median_us",
                         "cube_min_us", "native_v5_median_us", "speedup_cube_vs_v5",
                         "speedup_vector_vs_v5", "bytes_written", "bytes_moved",
                         "write_gbps", "throughput_gain_vs_vector_pct"}) out.emplace_back(s);
    return out;
}
inline void AppendPrefillCsv(const std::string& path, const std::vector<Row>& rows) {
    const std::string header = csv_line(csv_headings());
    std::ifstream existing(path, std::ios::binary);
    const bool empty = !existing || existing.peek() == std::ifstream::traits_type::eof();
    if (!empty) {
        std::string first;
        std::getline(existing, first);
        if (!first.empty() && first.back() == '\r') first.pop_back();
        if (first != header) throw std::runtime_error("incompatible prefill CSV header: " + path);
    }
    bool needs_newline = false;
    if (!empty) {
        existing.clear();
        existing.seekg(-1, std::ios::end);
        needs_newline = existing.get() != '\n';
    }
    existing.close();
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out) throw std::runtime_error("cannot append prefill CSV: " + path);
    if (empty) out << header << '\n';
    else if (needs_newline) out << '\n';
    for (const Row& r : rows) {
        const auto c = cells(r);
        std::vector<std::string> fields(c.begin(), c.end());
        const auto gain = ratio(r.vector.median, r.cube.median);
        const auto percent = gain ? std::optional<double>((*gain - 1) * 100) : std::nullopt;
        for (const auto& value : std::vector<std::string>{
                 r.mode, r.measurement_source, std::to_string(r.chunk_tokens), std::to_string(tokens(r)),
                 r.recycled ? "1" : "0", latency(r.a8.median), latency(r.cube.median),
                 latency(r.vector.median), latency(r.cube.min),
                 r.native ? latency(r.native->median) : "N/A",
                 number(native_speed(r, r.cube.median)),
                 number(native_speed(r, r.vector.median)), std::to_string(r.bytes_written),
                 std::to_string(r.bytes_moved), number(write_bandwidth(r)), number(percent)})
            fields.push_back(value);
        out << csv_line(fields) << '\n';
    }
    out.flush();
    if (!out) throw std::runtime_error("failed writing prefill CSV: " + path);
}
inline void AppendPrefillCsvFromEnv(const std::vector<Row>& rows) {
    const char* path = std::getenv("ASCEND_BENCH_TQ_AUDIT_PREFILL_CSV");
    if (path && *path) AppendPrefillCsv(path, rows);
}
}  // namespace tq_ingest_audit
