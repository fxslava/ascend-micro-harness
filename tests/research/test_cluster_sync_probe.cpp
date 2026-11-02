#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <acl/acl.h>

extern "C" void launch_kernel_cluster_sync_probe(void* stream, uint32_t iters, uint32_t sync_mode);

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
    // Which inter-core synchronization mode the probe exercises. 4 is the
    // existing leg (one channel per AIV, ids rebased by 16 for AIV1); 2 is
    // the fused AIC-and-all-AIVs leg, which is a different hardware
    // mechanism entirely -- NotifyEventImpl lowers mode 4 to
    // set_intra_block and every other mode to ffts_cross_core_sync, so a
    // mode-4 measurement says nothing about mode 2.
    uint32_t sync_mode = 4;
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
    if (argc > 2) {
        const long requested = std::atol(argv[2]);
        if (requested != 2 && requested != 4) {
            std::fprintf(stderr,
                         "FATAL: sync mode must be 2 or 4 (got %ld)\n",
                         requested);
            return 2;
        }
        sync_mode = static_cast<uint32_t>(requested);
    }

    std::printf("=== Cluster Sync Probe (1 AIC + 2 AIV), " PROBE_LEG " ===\n");
    std::printf("Target iterations: %u\n", iters);
    std::printf("Synchronization mode: %u\n", sync_mode);
    if (sync_mode == 2) {
        std::printf("Recycled mode-2 ids 8 (both AIVs -> AIC quorum) and 9 "
                    "(AIC -> both AIVs broadcast); one x_S drain in front "
                    "of every announcement and immediately behind every "
                    "wait. AIV1 lags AIV0 deliberately so the quorum has a "
                    "later arrival to be released by.\n");
    } else {
        std::printf("Recycled mode-4 ids 0, 1, 16, 17; one x_S drain in "
                    "front of every announcement.\n");
    }

    CHECK_ACL(aclInit(nullptr), "aclInit");
    CHECK_ACL(aclrtSetDevice(0), "aclrtSetDevice");

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::printf("Launching probe kernel on block 0...\n");
    launch_kernel_cluster_sync_probe(stream, iters, sync_mode);

    std::printf("Synchronizing stream...\n");
    CHECK_ACL(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    std::printf("Stream synchronized successfully. No hang, no deadlock.\n");
    // What a clean exit is and is not evidence for. The probe moves no data,
    // so finishing proves the handshake does not deadlock -- it does NOT
    // distinguish a consuming latch from a sticky one, because a sticky flag
    // would retire every wait early and the run would still finish and still
    // print this line. Only the stall each wait actually imposed separates
    // them, and only the CAModel trace shows it.
    // MICROROUTINE_DESIGN_GUIDELINES.md section 6.2c. Mode 4 logs its
    // handshake as SET_INTRA_BLOCK / WAIT_INTRA_BLOCK; mode 2 goes out over
    // FFTS instead, so read the stall census rather than grepping for one
    // instruction name.
    std::printf("NOTE: a clean exit alone does not distinguish a consuming "
                "latch from a sticky one -- this probe carries no data, so a "
                "sticky flag would retire every wait early and still finish. "
                "Read the per-wait stall census over the " PROBE_LEG " trace: "
                "python3 scripts/audit_mode4_tick_order.py EVIDENCE_DIR "
                "--block 0 --cross.\n");

    CHECK_ACL(aclrtDestroyStream(stream), "aclrtDestroyStream");
    CHECK_ACL(aclrtResetDevice(0), "aclrtResetDevice");
    CHECK_ACL(aclFinalize(), "aclFinalize");

    return 0;
}