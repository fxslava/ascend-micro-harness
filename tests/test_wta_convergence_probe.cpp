#include <deque>
#include <functional>
#include <iostream>
#include "wta_reference.hpp"
#include "../routines/routine_cube_wta_fixpipe.hpp"

using namespace wta_test;

// Fixed probe scale: 128.0 (0x43000000). No corpus-calibrated scale table;
// the same static scale is used for every T, sign and margin.
constexpr std::uint32_t kProbeScale = 0x43000000u;

// Three asynchronous queues. Scalar dispatch only appends instructions; random
// scheduling runs whole queue operations. Rejects a repeated set, an empty
// terminal queue with live tokens, and any read/write ownership violation.
struct Queues {
    enum Pipe { Mte1, Cube, Fix };
    enum Route { Feedback, Operands, Result };
    struct Op { int wait = -1; std::function<void()> run; };
    std::array<std::deque<Op>, 3> pipes;
    std::array<bool, 3> flags{};
    std::array<unsigned, 3> sets{}, waits{};
    void emit(Pipe p, std::function<void()> f) { pipes[p].push_back({-1, f}); }
    void set(Pipe p, Route r)
    {
        emit(p, [this, r] {
            require(!flags[r], "double set loses a one-bit token");
            flags[r] = true; ++sets[r];
        });
    }
    void wait(Pipe p, Route r)
    {
        pipes[p].push_back({r, [this, r] { flags[r] = false; ++waits[r]; }});
    }
    void execute(unsigned seed)
    {
        std::mt19937 rng(seed);
        for (;;) {
            std::array<int, 3> ready{};
            int count = 0; bool pending = false;
            for (int p = 0; p < 3; ++p) {
                if (pipes[p].empty()) continue;
                pending = true;
                const int w = pipes[p].front().wait;
                if (w < 0 || flags[w]) ready[count++] = p;
            }
            if (!pending) break;
            require(count != 0, "WTA queue deadlock");
            auto& q = pipes[ready[rng() % count]];
            auto op = q.front(); q.pop_front(); op.run();
        }
        for (int r = 0; r < 3; ++r)
            require(!flags[r] && sets[r] == waits[r], "undrained event token");
    }
};

struct Sync {
    Queues& q;
    void publish_feedback() { q.set(Queues::Fix, Queues::Feedback); }
    void wait_feedback() { q.wait(Queues::Mte1, Queues::Feedback); }
    void operands_ready()
    { q.set(Queues::Mte1, Queues::Operands); q.wait(Queues::Cube, Queues::Operands); }
    void result_ready()
    { q.set(Queues::Cube, Queues::Result); q.wait(Queues::Fix, Queues::Result); }
    void finish() {}  // execute() drains all queues and checks residual tokens.
};

template <class Config>
struct MockEngine {
    Queues& q;
    const Reference& oracle;
    GuardedBuffer<std::uint16_t, 512> state{256}, a{256}, b{256};
    GuardedBuffer<float, 1024> c{256};
    GuardedBuffer<typename Config::output_type> out{Config::output_bytes};
    unsigned loaded = 0, computed = 0, stored = 0;
    bool a_live = false, c_live = false;
    MockEngine(Queues& queues, const Reference& ref, const Tile& x) : q(queues), oracle(ref)
    {
        std::copy(x.begin(), x.end(), state.data());
        const auto w = weights();
        for (int col = 0; col < 16; ++col)
            for (int k = 0; k < 16; ++k) b[col * 16 + k] = w[k * 16 + col];
        std::fill(out.data(), out.data() + Config::output_bytes, 42);
    }
    void load_a()
    {
        q.emit(Queues::Mte1, [this] {
            require(!a_live && loaded == stored, "C1 RAW or L0A WAR");
            std::copy(state.data(), state.data() + 256, a.data());
            a_live = true; ++loaded;
        });
    }
    void mma_dispatch()
    {
        q.emit(Queues::Cube, [this] {
            require(a_live && !c_live, "L0A RAW or L0C WAR");
            for (int r = 0; r < 16; ++r)
                for (int col = 0; col < 16; ++col) {
                    // Exact product-sum over the FP16 W entries, rounded once
                    // to FP32 - the value Mmad leaves in L0C. Double keeps
                    // the accumulation exact for FP16 inputs; the FP32
                    // accumulation order itself is measured separately by
                    // reference_seq32 and by CAModel.
                    double sum = 0;
                    for (int k = 0; k < 16; ++k)
                        sum += half_to_float(a[r * 16 + k]) * half_to_float(b[col * 16 + k]);
                    c[r * 16 + col] = static_cast<float>(sum);
                }
            a_live = false; c_live = true; ++computed;
        });
    }
    template <bool Final> void fixpipe()
    {
        q.emit(Queues::Fix, [this] {
            require(c_live && computed == stored + 1, "L0C RAW");
            for (int r = 0; r < 16; ++r)
                for (int col = 0; col < 16; ++col) {
                    const int i = r * 16 + col;
                    if constexpr (Final) {
                        out[i] = static_cast<typename Config::output_type>(
                            quantize(c[i], Config::scale_bits,
                                     std::is_same<typename Config::output_type, std::int8_t>::value));
                        require(out[i] == oracle.quantized[stored][i],
                                "B8 differs from independent inhibition oracle");
                    } else {
                        state[i] = half_rne(std::max(0.0f, c[i]));
                        require(state[i] == oracle.feedback[stored][i], "FP16 feedback mismatch");
                    }
                }
            c_live = false; ++stored;
        });
    }
    void check()
    {
        require(loaded == Config::steps && computed == loaded && stored == loaded,
                "wrong instruction counts");
        ASSERT_CANARY_INTACT(state); ASSERT_CANARY_INTACT(a); ASSERT_CANARY_INTACT(b);
        ASSERT_CANARY_INTACT(c); ASSERT_CANARY_INTACT(out);
    }
};

template <unsigned T, class Out, std::uint32_t Scale, bool Recurse = true>
void run(const Tile& x, unsigned seed)
{
    using Config = ascend_micro::WtaConfig<T, Scale, Out>;
    const auto ref = reference(x, Scale, std::is_same<Out, std::int8_t>::value);
    Queues q; Sync sync{q}; MockEngine<Config> engine(q, ref, x);
    ascend_micro::cube_wta_run<Config>(engine, sync);
    q.execute(seed); engine.check();
    require(q.sets[Queues::Feedback] == T + 1 && q.sets[Queues::Operands] == T &&
            q.sets[Queues::Result] == T, "event count mismatch");
    if constexpr (Recurse && T < 8) run<T + 1, Out, Scale>(x, seed + 71);
}

int main(int argc, char** argv)
{
    try {
        const bool strict = argc == 2 && std::string(argv[1]) == "--require-one-hot";
        require(argc == 1 || strict, "usage: test_wta_convergence_probe [--require-one-hot]");
        static_assert(ascend_micro::WtaConfig<1>::deq_scalar == (1ull << 46 | 0x43000000u));
        static_assert(ascend_micro::WtaConfig<1, 0x3f801fffu, std::uint8_t>::deq_scalar == 0x3f800000u);
        require(quantize(0.5f, 0x3f800000, true) == 0 &&
                quantize(1.5f, 0x3f800000, true) == 2 &&
                quantize(2.5f, 0x3f800000, true) == 2, "round-to-even boundary");
        // The FP16 W entries the device multiplies by, and nothing else.
        require(wta_diag() == 1.224609375 && wta_off() == -0.024993896484375,
                "W is not the specified alpha/beta matrix in FP16");

        std::array<Convergence, 8> stats{};
        Stability stability;
        unsigned seq_exact = 0, seq_total = 0, seq_flips = 0, seq_max = 0;
        for (unsigned margin = 1; margin <= 100; ++margin) {
            for (unsigned seed = 0; seed < 4; ++seed) {
                std::array<unsigned, 16> winners{};
                const auto x = fixture(1000 * margin + seed, float(margin) / 100, &winners);
                run<1, std::int8_t, kProbeScale>(x, seed);
                run<1, std::uint8_t, kProbeScale>(x, seed + 1);
                const auto ref = reference(x, kProbeScale, true);
                for (unsigned t = 0; t < 8; ++t)
                    stats[t].add(x, ref.quantized[t], &winners);
                stability.add(x, winners);
                const auto seq = reference_seq32(x, kProbeScale, true);
                for (unsigned r = 0; r < 16; ++r) {
                    bool support_differs = false;
                    for (unsigned i = r * 16; i < r * 16 + 16; ++i) {
                        const int a = ref.quantized[7][i], b = seq.quantized[7][i];
                        const int diff = std::abs(a - b);
                        if (static_cast<unsigned>(diff) > seq_max) seq_max = static_cast<unsigned>(diff);
                        if (diff == 0) ++seq_exact;
                        if ((a != 0) != (b != 0)) support_differs = true;
                        ++seq_total;
                    }
                    if (support_differs) ++seq_flips;
                }
            }
        }
        // Scale truncation, saturation, small margin, ties, zero, negatives.
        for (const auto& x : {fixture(13, .01f), fixture(14, 1.f), Tile{}}) {
            run<1, std::int8_t, 0x3f801fffu>(x, 17);
            run<1, std::uint8_t, 0x43800000u>(x, 19);
        }
        Tile tied{}, negative{}, sparse{};
        tied.fill(half_rne(1)); negative.fill(half_rne(-1));
        for (unsigned r = 0; r < 16; ++r) sparse[r * 16 + r] = half_rne(1);
        run<1, std::int8_t, 0x3f800000u>(tied, 21);
        run<1, std::int8_t, 0x3f800000u>(negative, 23);
        run<1, std::int8_t, 0x3f800000u>(sparse, 25);
        const auto positive = reference(sparse, 0x3f800000u, true);
        for (const auto& q : positive.quantized)
            for (unsigned i = 0; i < 256; ++i)
                require((q[i] > 0) == (i / 16 == i % 16), "known unique-support input failed");
        const auto counter = reference(fixture(13, .01f), kProbeScale, true);
        Convergence counter_stats; counter_stats.add(fixture(13, .01f), counter.quantized[7]);
        require(counter_stats.multi > 0, "missing near-tie convergence counterexample");
        std::cout << "PASS Tier-1: T=1..8, 400 random direct-input tiles x signed/unsigned, "
                     "per-step FP16, bit-exact B8, randomized queues, balanced tokens, canaries\n";
        std::cout << "Census at scale 128, signed; T,rows,unique_survivor,rate,all_zero,"
                     "multiple,wrong_single,input_ties\n";
        for (unsigned t = 0; t < 8; ++t) {
            const auto& s = stats[t];
            require(!s.wrong_winner, "unique survivor disagrees with input argmax");
            std::cout << t + 1 << ',' << s.rows << ',' << s.support_one << ','
                      << double(s.support_one) / s.rows << ',' << s.zero << ',' << s.multi << ','
                      << s.wrong_winner << ',' << s.input_ties << '\n';
        }
        unsigned by4 = 0, by8 = 0, by64 = 0;
        for (unsigned t = 1; t <= 64; ++t) { by64 += stability.histogram[t]; if (t <= 4) by4 += stability.histogram[t]; if (t <= 8) by8 += stability.histogram[t]; }
        std::cout << "Recurrence stability rows=" << stability.rows
                  << " unique_by4=" << by4 << " unique_by8=" << by8
                  << " unique_by64=" << by64
                  << " not_unique_by64=" << stability.histogram[0] << " first_unique(t:rows)=";
        for (unsigned t = 1; t <= 64; ++t)
            if (stability.histogram[t]) std::cout << t << ':' << stability.histogram[t] << ' ';
        std::cout << '\n';
        std::cout << "FP32-sequential accumulation sensitivity vs exact oracle at T=8: exact "
                  << seq_exact << '/' << seq_total << " bytes, max |B8 diff| " << seq_max
                  << ", rows with support flips " << seq_flips << '\n';
        std::cout << "Universal full-scale one-hot convergence in <=4 (or <=8) steps: REFUTED.\n";
        require(!strict, "requested universal one-hot convergence is false for the tested recurrence");
    } catch (const std::exception& e) {
        std::cerr << "FAIL WTA: " << e.what() << '\n'; return 1;
    }
}
