#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <acl/acl.h>

extern "C" void launch_kernel_cluster_sync_probe(void* stream, uint32_t iters);

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
        iters = static_cast<uint32_t>(std::atoi(argv[1]));
    }

    std::printf("=== Cluster Sync Probe (1 AIC + 2 AIV) ===\n");
    std::printf("Target iterations: %u\n", iters);

    CHECK_ACL(aclInit(nullptr), "aclInit");
    CHECK_ACL(aclrtSetDevice(0), "aclrtSetDevice");

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::printf("Launching probe kernel on block 0...\n");
    launch_kernel_cluster_sync_probe(stream, iters);

    std::printf("Synchronizing stream...\n");
    CHECK_ACL(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    std::printf("Stream synchronized successfully. No hardware hang detected.\n");

    CHECK_ACL(aclrtDestroyStream(stream), "aclrtDestroyStream");
    CHECK_ACL(aclrtResetDevice(0), "aclrtResetDevice");
    CHECK_ACL(aclFinalize(), "aclFinalize");

    return 0;
}