#include "camodel_common.hpp"
#include "../common/decode_boundary_fixture.hpp"
#include "../../include/turboquant_splitk_contract.hpp"

#include <algorithm>
#include <charconv>
#include <string_view>

extern "C" void launch_turboquant_splitk_boundary_aiv(
    void*, std::uint16_t*, std::int8_t*, std::int8_t*, float*, std::uint32_t*,
    float*, float*, float*, turboquant_splitk::WorkerMetrics*,
    std::uint16_t*, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t);

namespace {
void check_boundary(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
unsigned number(const char* text)
{
    unsigned value = 0;
    const std::string_view str{text};
    const auto result = std::from_chars(str.data(), str.data() + str.size(), value);
    check_boundary(result.ec == std::errc{} && result.ptr == str.data() + str.size(), "invalid integer");
    return value;
}
}

int main(int argc, char** argv)
{
    try {
        using namespace single_token_decode;
        namespace contract = turboquant_splitk;
        unsigned begin = 0, strips = 1, strip_begin = 0, head = 2, rows = 1;
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag{argv[i]};
            check_boundary(i + 1 < argc, "missing option value");
            const auto value = number(argv[++i]);
            if (flag == "--page-begin") begin = value;
            else if (flag == "--strips") strips = value;
            else if (flag == "--strip-begin") strip_begin = value;
            else if (flag == "--head") head = value;
            else if (flag == "--rows") rows = value;
            else throw std::runtime_error("unknown option");
        }
        check_boundary(begin < 8 && strips >= 1 && strips <= 8 && strip_begin <= 8 - strips &&
                head < 4 && rows >= 1 && rows <= 6, "window out of range");
        camodel::Session session;
        camodel::GuardedDeviceBuffer q(4 * rows * 256 * 2),
            key(paged_decode::kKeyCacheBytes), value(paged_decode::kValueCacheBytes),
            scale(paged_decode::kScaleCacheFloats * 4), table(32),
            gate(4 * rows * 256 * 4), signs(256 * 4),
            // The boundary entry launches one block at splits = 1, so it takes
            // the fused path and must not touch the workspace at all. The
            // buffer is allocated at the production stride anyway: an
            // all-canary workspace afterwards is the evidence that the
            // one-split bypass really does bypass it.
            workspace(contract::WorkspaceFloats(1, 4, 1) * 4),
            metrics(4 * sizeof(contract::WorkerMetrics)), out(4 * rows * 256 * 2);
        const std::vector<camodel::GuardedDeviceBuffer*> buffers = {
            &q, &key, &value, &scale, &table, &gate, &signs, &workspace, &metrics, &out};
        for (auto* buffer : buffers) {
            // Include exterior guards in the qNaN flood for this protocol.
            std::fill(buffer->host.begin(), buffer->host.end(), decode_boundary::kCanary);
        }
        auto* qs = static_cast<std::uint16_t*>(q.host_payload());
        auto* gates = static_cast<float*>(gate.host_payload());
        for (unsigned lane = head * rows * 256; lane < (head + 1) * rows * 256; ++lane) {
            qs[lane] = float_to_bfloat16_rne(1.0f / 256.0f);
            gates[lane] = 0.0f;
        }
        auto* sign = static_cast<float*>(signs.host_payload());
        for (int c = 0; c < 256; ++c) sign[c] = static_cast<float>(PiSign(c));
        static_cast<std::uint32_t*>(table.host_payload())[begin] = paged_decode::BlockTableEntry(begin);
        std::vector<int> window;
        for (unsigned i = 0; i < strips; ++i) {
            const int strip = static_cast<int>(begin * 8 + strip_begin + i);
            window.push_back(strip);
            const int page = PhysicalPageOfStrip(strip), in_page = StripInPage(strip);
            auto* keys = static_cast<std::uint8_t*>(key.host_payload()) +
                paged_decode::KeyStripOffset(page, head, in_page);
            auto* values = static_cast<std::uint8_t*>(value.host_payload()) +
                paged_decode::ValueStripOffset(page, head, in_page);
            for (int t = 0; t < 16; ++t) {
                for (int c = 0; c < 256; ++c) {
                    paged_decode::PackNibble(keys, t * 256 + c, 3);
                    paged_decode::PackNibble(values, c * 16 + t, decode_boundary::Value(strip, c));
                }
                for (int plane = 0; plane < 2; ++plane)
                    static_cast<float*>(scale.host_payload())[
                        paged_decode::ScaleOffset(plane, page, head, in_page * 16 + t)] = 1.0f;
            }
        }
        std::vector<std::vector<std::uint32_t>> snapshots;
        for (auto* buffer : buffers) { snapshots.push_back(buffer->host); buffer->upload(); }
        launch_turboquant_splitk_boundary_aiv(session.stream,
            static_cast<std::uint16_t*>(q.payload()), static_cast<std::int8_t*>(key.payload()),
            static_cast<std::int8_t*>(value.payload()), static_cast<float*>(scale.payload()),
            static_cast<std::uint32_t*>(table.payload()), static_cast<float*>(gate.payload()),
            static_cast<float*>(signs.payload()), static_cast<float*>(workspace.payload()),
            static_cast<contract::WorkerMetrics*>(metrics.payload()),
            static_cast<std::uint16_t*>(out.payload()), head, rows, begin, 1, strips, strip_begin);
        camodel::check(aclrtSynchronizeStream(session.stream), "boundary synchronize");
        for (auto* buffer : buffers) buffer->download();
        const auto expected = decode_boundary::Oracle(window);
        const auto* context = static_cast<const std::uint16_t*>(out.host_payload());
        for (unsigned lane = 0; lane < rows * 256; ++lane)
            check_boundary(context[head * rows * 256 + lane] == float_to_bfloat16_rne(expected.context[lane]),
                    "byte-exact Split-K context mismatch");
        const auto* metric = static_cast<contract::WorkerMetrics*>(metrics.host_payload()) + head;
        for (unsigned lane = 0; lane < 8; ++lane)
            check_boundary(metric->exported[lane] == 1 && metric->tail[lane] == 1 &&
                    metric->pages[lane] == 1 && metric->reduced[lane] == 0 &&
                    metric->partition[lane] == contract::kNoPartition,
                    "invalid Split-K census replica");
        // Allow only the precise protocol/egress byte ranges. Compare all
        // other bytes, including input payloads, unused rows and both guards.
        std::size_t intact = 0;
        for (std::size_t b = 0; b < buffers.size(); ++b) {
            const auto* now = reinterpret_cast<const std::uint8_t*>(buffers[b]->host.data());
            const auto* before = reinterpret_cast<const std::uint8_t*>(snapshots[b].data());
            for (std::size_t byte = 0; byte < buffers[b]->host.size() * 4; ++byte) {
                const std::size_t payload = byte - camodel::kGuardBytes;
                bool assigned = false;
                // b == 7 is the workspace, and nothing claims a byte of it:
                // the fused path writes no partition, so every byte of it is
                // held to the canary like any other unassigned region.
                if (b == 8) {
                    const auto local = payload - head * sizeof(contract::WorkerMetrics);
                    assigned = local < sizeof(contract::WorkerMetrics);
                }
                if (b == 9) assigned = payload >= head * rows * 256 * 2 &&
                    payload < (head + 1) * rows * 256 * 2;
                if (!assigned) { check_boundary(now[byte] == before[byte], "ASSERT_CANARY_INTACT"); ++intact; }
            }
        }
        std::printf("PASS Split-K boundary page=%u strip=%u count=%u rows=%u: byte-exact context; "
                    "ASSERT_CANARY_INTACT %zu bytes\n", begin, strip_begin, strips, rows, intact);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
