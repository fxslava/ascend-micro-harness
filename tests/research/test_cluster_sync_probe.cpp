#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <acl/acl.h>

extern "C" void launch_kernel_cluster_sync_probe(void* stream, uint32_t iters);

// Which leg this binary is, so a pasted log cannot be mistaken for the other.
// SimTargets.cmake defines the macro for the CAModel twin; NpuTargets.cmake
// does not, matching bench_decode_splitk_sim's convention.
#if defined(ASCEND_MICRO_PROBE_SIM)
#define PROBE_LEG "CAModel"
#else
#define PROBE_LEG "physical device"
#endif

#define CHECK_ACL(expr, msg)                                            \
    do {                                                                \
        aclError err = (expr);                                          \
        if (err != ACL_SUCCESS) {                                       \
            std::fprintf(stderr, "FATAL: %s failed (code %d)\n", msg, err); \
            std::exit(1);                                               \
        }                                                               \
    } while (0)

int main(int argc, char** argv) {
    uint32_t iters = 2; // По умолчанию проверяем 2 итерации на выявление раннего снятия защелки
    if (argc > 1) {
        // Reject a count that would exit cleanly having tested nothing: one
        // iteration reuses no id, and zero runs no handshake at all.
        const long requested = std::atol(argv[1]);
        if (requested < 1 || requested > 4096) {
            std::fprintf(stderr,
                         "FATAL: iterations must be 1..4096 (got %ld)\n",
                         requested);
            return 2;
        }
        iters = static_cast<uint32_t>(requested);
    }

    std::printf("=== Cluster Sync Probe (1 AIC + 2 AIV), " PROBE_LEG " ===\n");
    std::printf("Target iterations: %u\n", iters);
    std::printf("Recycled mode-4 ids 0, 1, 16, 17; one x_S drain in front of "
                "every announcement.\n");

    CHECK_ACL(aclInit(nullptr), "aclInit");
    CHECK_ACL(aclrtSetDevice(0), "aclrtSetDevice");

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::printf("Launching probe kernel on block 0...\n");
    launch_kernel_cluster_sync_probe(stream, iters);

    std::printf("Synchronizing stream...\n");
    CHECK_ACL(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    std::printf("Stream synchronized successfully. No hang, no deadlock.\n");
    // What a clean exit is and is not evidence for. The probe moves no data,
    // so finishing proves the handshake does not deadlock -- it does NOT
    // distinguish a consuming latch from a sticky one, because a sticky flag
    // would retire every wait early and the run would still finish and still
    // print this line. Only the tick order of SET_INTRA_BLOCK against
    // WAIT_INTRA_BLOCK on one recycled id separates them, and only the CAModel
    // trace shows it. MICROROUTINE_DESIGN_GUIDELINES.md section 6.2c.
    std::printf("NOTE: a clean exit alone does not distinguish a consuming "
                "latch from a sticky one -- this probe carries no data, so a "
                "sticky flag would retire every wait early and still finish. "
                "Read the tick order of SET_INTRA_BLOCK against "
                "WAIT_INTRA_BLOCK per id in the " PROBE_LEG " trace.\n");

    CHECK_ACL(aclrtDestroyStream(stream), "aclrtDestroyStream");
    CHECK_ACL(aclrtResetDevice(0), "aclrtResetDevice");
    CHECK_ACL(aclFinalize(), "aclFinalize");

    return 0;
}