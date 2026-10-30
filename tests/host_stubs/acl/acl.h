// Minimal ACL declaration stub, host-compile gate only.
//
// It exists so the reshape-and-cache host harnesses can be compiled under
// -Wall -Wextra -Werror on a machine with no CANN toolkit. It declares the
// surface those harnesses use and nothing else; it links to nothing and runs
// nothing, so a pass here means "the host code compiles clean", never "the
// harness works".
#pragma once
#include <cstddef>
#include <cstdint>

using aclError = int;
#define ACL_SUCCESS 0

using aclrtStream = void*;
using aclrtEvent = void*;
using aclrtContext = void*;

enum aclrtMemMallocPolicy { ACL_MEM_MALLOC_HUGE_FIRST, ACL_MEM_MALLOC_NORMAL_ONLY };
enum aclrtMemcpyKind {
    ACL_MEMCPY_HOST_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_DEVICE_TO_DEVICE
};

extern "C" {
aclError aclInit(const char* config);
aclError aclFinalize();
aclError aclrtSetDevice(int device);
aclError aclrtResetDevice(int device);
aclError aclrtCreateStream(aclrtStream* stream);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtSynchronizeStream(aclrtStream stream);
aclError aclrtMalloc(void** ptr, size_t size, aclrtMemMallocPolicy policy);
aclError aclrtFree(void* ptr);
aclError aclrtMemcpy(void* dst, size_t dst_max, const void* src, size_t count,
                     aclrtMemcpyKind kind);
aclError aclrtMemset(void* ptr, size_t max_count, int32_t value, size_t count);
aclError aclrtCreateEvent(aclrtEvent* event);
aclError aclrtDestroyEvent(aclrtEvent event);
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream);
aclError aclrtSynchronizeEvent(aclrtEvent event);
aclError aclrtEventElapsedTime(float* ms, aclrtEvent start, aclrtEvent end);
}
