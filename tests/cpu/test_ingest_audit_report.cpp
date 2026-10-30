#include "../npu/turboquant_audit_report.hpp"
#include "../npu/turboquant_ingest_bench_options.hpp"
#include <chrono>
#include <filesystem>
#include <sstream>

namespace audit = tq_ingest_audit;
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::string render(const std::vector<audit::Row>& rows) {
    std::FILE* f = std::tmpfile();
    check(f != nullptr, "tmpfile");
    audit::PrintTableA(f, rows);
    std::rewind(f);
    std::string text;
    char buffer[4096];
    while (const auto count = std::fread(buffer, 1, sizeof(buffer), f)) text.append(buffer, count);
    std::fclose(f);
    return text;
}
tq_ingest_bench::Options parse(std::vector<std::string> args) {
    std::vector<char*> pointers;
    for (auto& s : args) pointers.push_back(s.data());
    return tq_ingest_bench::parse(static_cast<int>(pointers.size()), pointers.data(), true);
}
void invalid_cli(std::vector<std::string> args) {
    bool rejected = false;
    try { parse(std::move(args)); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid CLI was accepted");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        for (const auto& pair : std::vector<std::pair<std::uint64_t, std::string>>{
                 {8192, "Short"}, {8193, "Mid"}, {65536, "Mid"}, {65537, "Long"},
                 {262144, "Long"}, {262145, "Extreme"}, {1048576, "Extreme"}})
            check(audit::tier(pair.first) == pair.second, "tier boundary");
        check(audit::commas(2048) == "2,048" && audit::commas(8192) == "8,192" &&
              audit::commas(16384) == "16,384", "thousands separators");
        audit::Row row;
        row.model = "Qwen3.5-9B"; row.mode = "warm"; row.context = 8192;
        row.measurement_source = "synthetic_fixture";
        row.batch = 2; row.chunk_tokens = 1024;
        row.cube = {8, 10, 13}; row.vector = {15, 20, 25}; row.a8 = {5, 6, 7};
        row.native = audit::Stats{30, 40, 45};
        row.bytes_written = 204800; row.bytes_moved = 400000;
        const auto c = audit::cells(row);
        check(c[4] == "2,048", "tokens must be B * chunk, not S * B");
        check(c[5] == "10.000 (20.000 / 8.000)", "latency triplet order");
        check(c[6] == "40.000" && c[7] == "4.000x (2.000x)", "speedup orientation");
        check(c[8].find("20.5 GB/s") != std::string::npos, "write bandwidth units/bytes");
        check(c[8].find("+100.0%") != std::string::npos, "throughput gain");
        auto missing = row;
        missing.native.reset(); missing.native_note = "native skipped";
        check(audit::cells(missing)[6] == "N/A" && audit::cells(missing)[7] == "N/A",
              "missing native must not produce a speedup");
        for (const double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN()}) {
            auto broken = row;
            broken.native->median = invalid;
            check(audit::cells(broken)[7] == "N/A", "invalid native median");
            broken = row; broken.cube.median = invalid;
            check(!audit::write_bandwidth(broken) && !audit::native_speed(broken, invalid),
                  "invalid Cube denominator");
        }
        auto overflow = row;
        overflow.batch = std::numeric_limits<std::uint64_t>::max();
        bool rejected = false;
        try { audit::tokens(overflow); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "token multiplication overflow");

        auto options = parse({"bench", "--model", "Qwen3.5-9B", "--model", "GLM-5.2-744B",
                              "--context", "8192,65536,262144,1048576", "--batch", "1,2",
                              "--tokens", "1024,2048", "--single", "--skip-native"});
        check(options.models.size() == 2 && options.contexts.size() == 4 &&
              options.batches.size() == 2 && options.chunks.size() == 2 &&
              options.warmups == 0 && options.samples == 1 && options.skip_native,
              "sweep CLI or single-shot semantics");
        check(parse({"bench", "--strips", "128"}).chunks.front() == 2048, "legacy strips CLI");
        check(parse({"bench"}).chunks.front() == 2048, "legacy effective default token count");
        const auto layout = tq_ingest_bench::cache_layout(1048576, 2, 1024);
        check(layout.slots.size() == 2048 && layout.blocks == 131074 &&
              layout.slots.front() == 1047552 && layout.slots[1023] == 1048575 &&
              layout.slots[1024] == 2096128 && layout.slots.back() == 2097151,
              "chunk slots must reach each request's final horizon pages");
        for (auto args : std::vector<std::vector<std::string>>{
                 {"bench", "--tokens", "0"}, {"bench", "--tokens", "17"},
                 {"bench", "--batch", "0"}, {"bench", "--batch", "18446744073709551615"},
                 {"bench", "--context", "512"}, {"bench", "--context", "1048577"},
                 {"bench", "--context", "18446744073709551615"},
                 {"bench", "--tokens", "65536", "--batch", "2"},
                 {"bench", "--tokens", "1024,"}, {"bench", "--samples", "0"},
                 {"bench", "--model"}, {"bench", "--unknown", "1"}})
            invalid_cli(std::move(args));

        std::vector<audit::Row> rows;
        for (const auto context : {8192u, 65536u, 262144u, 1048576u}) {
            auto next = row; next.context = context;
            if (context == 65536) { next.model = "Model, \"A\""; next.native.reset(); }
            if (context == 262144) next.model = "DeepSeek-V4-Flash";
            if (context == 1048576) next.model = "GLM-5.2-744B";
            rows.push_back(next);
        }
        const std::string table = render(rows);
        check(table.find("10.000 (20.000 / 8.000)") != std::string::npos, "rendered triplet");
        check(table.find("Extreme") != std::string::npos &&
              table.find("1,048,576") != std::string::npos, "extreme rendering");
        auto escaped = row; escaped.model = "model|name\nnext";
        check(render({escaped}).find("model\\|name next") != std::string::npos, "Markdown escaping");

        const std::string path = argc > 1 ? argv[1] :
            (std::filesystem::temp_directory_path() / ("tq-ingest-audit-" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".csv")).string();
        check(!std::filesystem::exists(path), "test CSV must be a new file");
        audit::AppendPrefillCsv(path, rows);
        // Appending must also work if another producer omitted the final LF.
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
        audit::AppendPrefillCsv(path, {missing});
        std::ifstream file(path);
        std::stringstream contents; contents << file.rdbuf(); file.close();
        const auto csv = contents.str();
        const auto header = audit::csv_line(audit::csv_headings());
        check(csv.find(header) == 0 && csv.find(header, 1) == std::string::npos,
              "CSV append must write exactly one header");
        check(csv.find("\"Model, \"\"A\"\"\"") != std::string::npos, "CSV quote escaping");
        check(csv.find("\"2,048\"") != std::string::npos, "CSV comma escaping");
        {
            std::ofstream bad(path + ".bad"); bad << "old,schema\n";
        }
        rejected = false;
        try { audit::AppendPrefillCsv(path + ".bad", {row}); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "incompatible CSV header must not be silently appended");
        std::filesystem::remove(path + ".bad");
        if (argc == 1) std::filesystem::remove(path);
        if (argc > 2 && std::string(argv[2]) == "--check-env")
            audit::AppendPrefillCsvFromEnv(rows);
        std::puts("Synthetic reporting fixture only; these are not hardware timings.\n");
        audit::PrintTableA(stdout, rows);
        std::puts("PASS ingest audit: tiers, formulas, missing/invalid timings, sweep CLI, "
                  "Markdown and CSV append/escaping");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL ingest audit: %s\n", e.what());
        return 1;
    }
}
