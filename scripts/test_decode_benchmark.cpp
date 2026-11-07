// Host-only regression of the real benchmark control path with an ACL test
// double. This checks ordering/lifetime, and is not CAModel timing evidence.
#define main benchmark_program_main
#include "../tests/npu/bench/bench_turboquant_decode_splitk.cpp"
#undef main

namespace {
bool pending = false, fail_stream = false;
float elapsed_ms = 0.125f;
int event_creates = 0, event_destroys = 0, fail_event = -1;
int allocations = 0, frees = 0, finalized = 0, device_resets = 0;

struct FakeRig {
    int expected, launches = 0, validations = 0, resets = 0;
    void Reset() { require(!pending, "reset before completion"); ++resets; }
    void Launch(aclrtStream, int) {
        require(validations == 0 && !pending, "validation interleaved with sampling");
        pending = true;
        ++launches;
    }
    tq_splitk_rig::Census Validate(int) {
        require(!pending && launches == expected && validations == 0,
                "validation must run once after all synchronized launches");
        ++validations;
        return {{2, 6}, {4, 8}};
    }
};
}  // namespace

extern "C" {
aclError aclrtCreateEvent(aclrtEvent* event) {
    if (++event_creates == fail_event) return 1;
    *event = reinterpret_cast<void*>(static_cast<std::uintptr_t>(event_creates));
    return ACL_SUCCESS;
}
aclError aclrtDestroyEvent(aclrtEvent) { ++event_destroys; return ACL_SUCCESS; }
aclError aclrtSynchronizeStream(aclrtStream) { pending = false; return ACL_SUCCESS; }
aclError aclrtSynchronizeEvent(aclrtEvent) { pending = false; return ACL_SUCCESS; }
aclError aclrtRecordEvent(aclrtEvent, aclrtStream) { return ACL_SUCCESS; }
aclError aclrtEventElapsedTime(float* ms, aclrtEvent, aclrtEvent) {
    require(!pending, "elapsed time before completion"); *ms = elapsed_ms; return ACL_SUCCESS;
}
aclError aclInit(const char*) { return ACL_SUCCESS; }
aclError aclFinalize() { ++finalized; return ACL_SUCCESS; }
aclError aclrtSetDevice(int32_t) { return ACL_SUCCESS; }
aclError aclrtResetDevice(int32_t) { ++device_resets; return ACL_SUCCESS; }
aclError aclrtCreateStream(aclrtStream* stream) {
    if (fail_stream) return 1;
    *stream = reinterpret_cast<void*>(1); return ACL_SUCCESS;
}
aclError aclrtDestroyStream(aclrtStream) { return ACL_SUCCESS; }
aclError aclrtMalloc(void** ptr, size_t bytes, aclrtMemMallocPolicy) {
    *ptr = std::aligned_alloc(32, bytes); if (!*ptr) return 1;
    ++allocations; return ACL_SUCCESS;
}
aclError aclrtFree(void* ptr) { std::free(ptr); ++frees; return ACL_SUCCESS; }
}

int main(int argc, char** argv)
{
    try {
        require(argc == 2, "supply output CSV path");
        Options options;
        options.warmup = 3; options.iters = 7;
        {
            Events events;
            FakeRig rig{10};
            const auto result = measure(rig, options, 2, nullptr, events);
            require(rig.validations == 1 && rig.resets == 10 && result.launches == 10,
                    "warmup/sample launch accounting");
            require(result.tail == 4 && result.export_region == 6, "final-launch census");
            options.single = true; options.warmup = 0; options.iters = 1;
            FakeRig single{1};
            const auto one = measure(single, options, 2, nullptr, events);
            require(single.validations == 1 && one.launches == 1 && one.tail == 6 && one.export_region == 8,
                    "single-launch census or validation accounting");
            elapsed_ms = 0;
            FakeRig zero{1};
            require(!measure(zero, options, 2, nullptr, events).timed, "zero event duration must be unavailable");
        }
#if !defined(ASCEND_MICRO_BENCH_SIM)
        require(event_creates == event_destroys, "event leak");
        fail_event = event_creates + 2;
        bool failed = false;
        try { Events events; } catch (const std::exception&) { failed = true; }
        require(failed && event_destroys == event_creates - 1, "partial event construction leak");
#else
        require(event_creates == 0, "CAModel must not create timing events");
#endif
        // Scoped allocations across context sizes and unwinding on an error.
        for (int context : {128, 256, 512}) {
            try {
                tq_npu::Buffer first(context * 32), second(context * 64);
                if (context == 256) throw std::runtime_error("injected failure");
            } catch (const std::runtime_error&) {}
            require(allocations == frees, "device buffers survived context scope");
        }
        { tq_npu::Session session; }
        fail_stream = true;
        bool failed_session = false;
        try { tq_npu::Session session; } catch (const std::exception&) { failed_session = true; }
        require(failed_session && finalized == 2 && device_resets == 2, "session cleanup on constructor failure");

        Measurement av, ac;
        ac.present = ac.timed = true; ac.us = 100;
        std::ofstream csv(argv[1]);
        row(128, 2, 2, av, ac, 0, 200, 0, 1000000, csv);
        csv.close();
        std::ifstream input(argv[1]);
        std::string line; std::getline(input, line);
        require(line.find("128,2,2,NA,NA,NA,100.000,6.250,160.000,10.000,") == 0 &&
                line.substr(line.rfind(',') + 1) == "2.000", "AIC-only baseline or CSV alignment");
        input.close();
        ac.us = 0;
        csv.open(argv[1], std::ios::app);
        row(128, 2, 2, av, ac, 0, 200, 0, 1000000, csv);
        csv.close();
        input.open(argv[1]);
        std::getline(input, line); std::getline(input, line);
        require(line.find("128,2,2,NA,NA,NA,NA,NA,NA,NA,") == 0 &&
                line.substr(line.rfind(',') + 1) == "NA", "zero latency must never divide or report speedup");
        const char* args[] = {"bench", "--single", "--warmup", "9", "--iters", "7", "--context", "128", "--splitk", "2", "--impl", "aic"};
        const auto parsed = parse(12, const_cast<char**>(args));
        require(parsed.warmup == 0 && parsed.iters == 1 && parsed.splits == std::vector<int>{2},
                "single override must not add a baseline launch");
        std::puts("PASS: validation ordering/count, synchronized single-shot, zero timing, AIC baseline, context buffer lifetime, constructor cleanup.");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL host regression: %s\n", error.what());
        return 1;
    }
}
