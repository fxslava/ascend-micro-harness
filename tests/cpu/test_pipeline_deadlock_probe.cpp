#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../../include/sim_micro_testbed.hpp"
#include "../../kernels/skeleton/turboquant_reshape_and_cache.cpp"

namespace probe = ascend_micro::modulo3;
using ascend_micro::sim::GuardedBuffer;
using ascend_micro::sim::require;

constexpr unsigned totalTiles = probe::kStageDepth;
static_assert(totalTiles == 3);

enum class Pipe : unsigned { Mte2, Vector, Mte3 };
enum class Kind : unsigned { Set, Wait, Ingress, Vector, Egress };

struct Operation {
    Kind kind;
    probe::Route route;
    unsigned slot;
    unsigned tile;
    unsigned remaining = 0;
};

static constexpr Pipe source(probe::Route route)
{
    switch (route) {
    case probe::Route::Mte2V: return Pipe::Mte2;
    case probe::Route::VMte2: return Pipe::Vector;
    case probe::Route::VMte3: return Pipe::Vector;
    case probe::Route::Mte3V: return Pipe::Mte3;
    }
    return Pipe::Mte2;
}

static constexpr Pipe destination(probe::Route route)
{
    switch (route) {
    case probe::Route::Mte2V: return Pipe::Vector;
    case probe::Route::VMte2: return Pipe::Mte2;
    case probe::Route::VMte3: return Pipe::Mte3;
    case probe::Route::Mte3V: return Pipe::Vector;
    }
    return Pipe::Mte2;
}

class QueueModel {
public:
    QueueModel() : gm_input(totalTiles * probe::kPayloadWords),
                   gm_output(totalTiles * probe::kPayloadWords)
    {
        for (unsigned i = 0; i < gm_input.size(); ++i)
            gm_input[i] = probe::kInputCanary;
        for (unsigned i = 0; i < gm_output.size(); ++i)
            gm_output[i] = 0;
        std::memset(ub.data(), 0xA5, ub.size());
        require(probe::kSlotStrideBytes % 256u != 0u, "UB bank skew lost");
        for (unsigned slot = 0; slot < totalTiles; ++slot) {
            require((input_offset(slot) % 32u) == 0u &&
                    (output_offset(slot) % 32u) == 0u,
                    "UB payload alignment lost");
            require((input_offset(slot) / 32u) % 16u !=
                    (output_offset(slot) / 32u) % 16u,
                    "input/output UB banks collide");
        }
    }

    template <probe::Route Route, unsigned Slot>
    void set()
    {
        static_assert(Slot < totalTiles);
        enqueue(source(Route), {Kind::Set, Route, Slot, 0});
        ++submitted_sets;
    }

    template <probe::Route Route, unsigned Slot>
    void wait()
    {
        static_assert(Slot < totalTiles);
        enqueue(destination(Route), {Kind::Wait, Route, Slot, 0});
        ++submitted_waits;
    }

    template <unsigned Slot>
    void ingress(unsigned tile)
    {
        enqueue(Pipe::Mte2, {Kind::Ingress, probe::Route::Mte2V, Slot, tile});
    }

    template <unsigned Slot>
    void vector_compute(unsigned tile)
    {
        enqueue(Pipe::Vector, {Kind::Vector, probe::Route::Mte2V, Slot, tile});
    }

    template <unsigned Slot>
    void egress(unsigned tile)
    {
        enqueue(Pipe::Mte3, {Kind::Egress, probe::Route::VMte3, Slot, tile});
    }

    void execute(const std::string& trace_path)
    {
        require(submitted_sets == 18 && submitted_waits == 18,
                "wrong route/ID pairing count at dispatch");
        std::ofstream trace(trace_path);
        require(trace.good(), "could not create queue trace");
        trace << "tick,MTE2,Vector,MTE3,active_work,queued_MTE2,queued_Vector,queued_MTE3\n";
        constexpr unsigned kMaxTicks = 256;
        for (unsigned tick = 0; tick < kMaxTicks; ++tick) {
            unsigned active_work = 0;
            bool progress = false;
            std::array<std::string, 3> labels;
            for (unsigned p = 0; p < 3; ++p) {
                if (queues[p].empty()) { labels[p] = "idle"; continue; }
                auto& op = queues[p].front();
                labels[p] = label(op);
                if (is_work(op.kind)) ++active_work;
                if (advance(op)) {
                    queues[p].pop_front();
                    progress = true;
                } else if (is_work(op.kind)) {
                    progress = true;  // one unit of asynchronous work completed
                }
            }
            if (active_work == 3) ++three_way_overlap_ticks;
            trace << tick << ',' << labels[0] << ',' << labels[1] << ','
                  << labels[2] << ',' << active_work << ',' << queues[0].size()
                  << ',' << queues[1].size() << ',' << queues[2].size() << '\n';
            if (all_empty()) break;
            require(progress, "deadlock: all three execution queues stalled");
            require(tick + 1 < kMaxTicks, "timeout: execution queues did not drain");
        }
        require(all_empty(), "queue drain failed");
        require(three_way_overlap_ticks > 0, "no three-pipe overlap in queue model");
        require(executed_sets == submitted_sets &&
                executed_waits == submitted_waits,
                "event pair count differs from submitted program");
        for (unsigned r = 0; r < 4; ++r)
            for (unsigned slot = 0; slot < totalTiles; ++slot) {
                require(!tokens[r][slot], "event token leaked after epilogue");
                require(sets[r][slot] == waits[r][slot],
                        "route-scoped event count is unbalanced");
            }
        for (unsigned tile = 0; tile < totalTiles; ++tile)
            for (unsigned word = 0; word < probe::kPayloadWords; ++word)
                require(gm_output[tile * probe::kPayloadWords + word] ==
                        probe::kOutputCanary, "wrong output canary");
        ASSERT_CANARY_INTACT(gm_input);
        ASSERT_CANARY_INTACT(gm_output);
        require(canary_corruptions == 0, "UB or GM canary corruption");
    }

    unsigned three_way_overlap_ticks = 0;
    unsigned canary_corruptions = 0;
    unsigned submitted_sets = 0;
    unsigned submitted_waits = 0;
    unsigned executed_sets = 0;
    unsigned executed_waits = 0;
    GuardedBuffer<std::uint32_t> gm_input;
    GuardedBuffer<std::uint32_t> gm_output;

private:
    static constexpr unsigned input_offset(unsigned slot)
    { return probe::kInputBaseBytes + slot * probe::kSlotStrideBytes + probe::kGuardBytes; }
    static constexpr unsigned output_offset(unsigned slot)
    { return probe::kOutputBaseBytes + slot * probe::kSlotStrideBytes + probe::kGuardBytes; }
    static constexpr bool is_work(Kind kind)
    { return kind == Kind::Ingress || kind == Kind::Vector || kind == Kind::Egress; }
    static constexpr unsigned route_index(probe::Route route)
    { return static_cast<unsigned>(route); }
    static constexpr unsigned pipe_index(Pipe pipe)
    { return static_cast<unsigned>(pipe); }

    void enqueue(Pipe pipe, Operation operation)
    {
        queues[pipe_index(pipe)].push_back(operation);
    }

    bool all_empty() const
    { return queues[0].empty() && queues[1].empty() && queues[2].empty(); }

    static std::string label(const Operation& op)
    {
        switch (op.kind) {
        case Kind::Set: return "set_r" + std::to_string(route_index(op.route)) +
                               "_s" + std::to_string(op.slot);
        case Kind::Wait: return "wait_r" + std::to_string(route_index(op.route)) +
                                "_s" + std::to_string(op.slot);
        case Kind::Ingress: return "ingress_t" + std::to_string(op.tile);
        case Kind::Vector: return "vector_t" + std::to_string(op.tile);
        case Kind::Egress: return "egress_t" + std::to_string(op.tile);
        }
        return "?";
    }

    void check_guards(unsigned slot, bool input_region)
    {
        const unsigned base = (input_region ? probe::kInputBaseBytes :
            probe::kOutputBaseBytes) + slot * probe::kSlotStrideBytes;
        const unsigned tail = base + probe::kGuardBytes + probe::kPayloadBytes;
        for (unsigned i = 0; i < probe::kGuardBytes; ++i)
            if (ub[base + i] != std::byte{0xA5} ||
                ub[tail + i] != std::byte{0xA5}) ++canary_corruptions;
    }

    void finish_work(const Operation& op)
    {
        const unsigned tile_base = op.tile * probe::kPayloadWords;
        if (op.kind == Kind::Ingress) {
            std::memcpy(ub.data() + input_offset(op.slot),
                        gm_input.data() + tile_base, probe::kPayloadBytes);
            check_guards(op.slot, true);
        } else if (op.kind == Kind::Vector) {
            const auto* in = reinterpret_cast<const std::uint32_t*>(
                ub.data() + input_offset(op.slot));
            auto* out = reinterpret_cast<std::uint32_t*>(
                ub.data() + output_offset(op.slot));
            for (unsigned i = 0; i < probe::kPayloadWords; ++i) {
                if (in[i] != probe::kInputCanary) ++canary_corruptions;
                out[i] = in[i] + probe::kCanaryDelta;
            }
            check_guards(op.slot, true);
            check_guards(op.slot, false);
        } else {
            std::memcpy(gm_output.data() + tile_base,
                        ub.data() + output_offset(op.slot),
                        probe::kPayloadBytes);
            check_guards(op.slot, false);
        }
    }

    bool advance(Operation& op)
    {
        const unsigned route = route_index(op.route);
        if (op.kind == Kind::Set) {
            require(!tokens[route][op.slot], "double SetFlag on live event bit");
            tokens[route][op.slot] = true;
            ++sets[route][op.slot];
            ++executed_sets;
            return true;
        }
        if (op.kind == Kind::Wait) {
            if (!tokens[route][op.slot]) return false;
            tokens[route][op.slot] = false;
            ++waits[route][op.slot];
            ++executed_waits;
            return true;
        }
        if (op.remaining == 0) op.remaining = 3;
        if (--op.remaining != 0) return false;
        finish_work(op);
        return true;
    }

    alignas(64) std::array<std::byte, probe::kUbBytes> ub{};
    std::array<std::deque<Operation>, 3> queues{};
    bool tokens[4][totalTiles]{};
    unsigned sets[4][totalTiles]{};
    unsigned waits[4][totalTiles]{};
};

int main(int argc, char** argv)
{
    try {
        const std::string trace = argc > 1 ? argv[1] : "bin/pipeline_queue_trace.csv";
        QueueModel model;
        probe::RunThreeTiles(model);
        model.execute(trace);
        std::cout << "PASS totalTiles=3; 0 deadlocks; 0 canary corruptions; "
                     "18/18 event pairs; 0 residual tokens; three-way queue "
                     "overlap=" << model.three_way_overlap_ticks << " ticks\n";
        std::cout << "queue-model trace: " << trace << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
