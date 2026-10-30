#pragma once
// Native BF16 scatter comparator. Load optional operator libraries only for
// the native leg, after ACL device initialization; keep them for the process.
// Plan/tensor/workspace preparation stays outside the measured device events.
#if defined(ASCEND_MICRO_HAVE_NATIVE_INGEST)
#include "turboquant_npu_common.hpp"
#include <aclnn_scatter_pa_kv_cache.h>
#include <dlfcn.h>
#include <limits>
#include <memory>

namespace tq_ingest_bench {
struct LibraryDeleter { void operator()(void* p) const { if (p) dlclose(p); } };
using Library = std::unique_ptr<void, LibraryDeleter>;
inline Library open_library(const char* path) {
    Library handle(dlopen(path, RTLD_NOW | RTLD_LOCAL));
    if (!handle) {
        const char* error = dlerror();
        throw std::runtime_error(std::string("native ingest library: ") +
                                 (error ? error : path));
    }
    return handle;
}
template <typename Function> Function symbol(void* library, const char* name) {
    void* address = dlsym(library, name);
    if (!address) throw std::runtime_error(std::string("native ingest symbol unavailable: ") + name);
    return reinterpret_cast<Function>(address);
}
struct NativeApi {
    Library base{open_library(TQ_INGEST_NNOPBASE_PATH)};
    Library opapi{open_library(TQ_INGEST_OPAPI_PATH)};
    decltype(&aclCreateTensor) create_tensor = symbol<decltype(create_tensor)>(base.get(), "aclCreateTensor");
    decltype(&aclDestroyTensor) destroy_tensor = symbol<decltype(destroy_tensor)>(base.get(), "aclDestroyTensor");
    decltype(&aclDestroyAclOpExecutor) destroy_executor =
        symbol<decltype(destroy_executor)>(base.get(), "aclDestroyAclOpExecutor");
    decltype(&aclSetAclOpExecutorRepeatable) repeatable =
        symbol<decltype(repeatable)>(base.get(), "aclSetAclOpExecutorRepeatable");
    decltype(&aclnnScatterPaKvCacheGetWorkspaceSize) prepare =
        symbol<decltype(prepare)>(opapi.get(), "aclnnScatterPaKvCacheGetWorkspaceSize");
    decltype(&aclnnScatterPaKvCache) launch =
        symbol<decltype(launch)>(opapi.get(), "aclnnScatterPaKvCache");
};
inline NativeApi& native_api() {
    static NativeApi api;
    return api;
}
struct TensorDeleter {
    decltype(&aclDestroyTensor) destroy;
    void operator()(aclTensor* p) const { if (p) destroy(p); }
};
struct ExecutorDeleter {
    decltype(&aclDestroyAclOpExecutor) destroy;
    void operator()(aclOpExecutor* p) const { if (p) destroy(p); }
};
using Tensor = std::unique_ptr<aclTensor, TensorDeleter>;
inline Tensor tensor(const NativeApi& api, void* data, aclDataType type,
                     std::vector<std::int64_t> shape) {
    std::vector<std::int64_t> strides(shape.size(), 1);
    for (std::size_t i = shape.size(); i > 1; --i)
        strides[i - 2] = strides[i - 1] * shape[i - 1];
    Tensor t(api.create_tensor(shape.data(), shape.size(), type, strides.data(), 0,
                              ACL_FORMAT_ND, shape.data(), shape.size(), data),
             TensorDeleter{api.destroy_tensor});
    tq_npu::require(t != nullptr, "aclCreateTensor native ingest");
    return t;
}
struct NativeV5 {
    // API libraries outlive every tensor/executor using their function pointers.
    NativeApi& api;
    tq_npu::Buffer key_cache, value_cache;
    Tensor key, value, slots, key_out, value_out;
    std::unique_ptr<tq_npu::Buffer> workspace;
    std::unique_ptr<aclOpExecutor, ExecutorDeleter> executor;
    std::uint64_t workspace_size = 0;

    NativeV5(const tq_npu::Fixture& f, const tq_npu::Bf16Device& input)
        : api(native_api()), key_cache(f.shape.cache_bytes() * 4), value_cache(f.shape.cache_bytes() * 4),
          key(tensor(api, input.key.ptr, ACL_BF16, {f.tokens(), tq_npu::kHeads, tq_npu::kHeadSize})),
          value(tensor(api, input.value.ptr, ACL_BF16, {f.tokens(), tq_npu::kHeads, tq_npu::kHeadSize})),
          slots(tensor(api, input.slots.ptr, ACL_INT32, {f.tokens()})),
          key_out(tensor(api, key_cache.ptr, ACL_BF16,
                  {f.shape.num_blocks, tq_npu::kBlockSize, tq_npu::kHeads, tq_npu::kHeadSize})),
          value_out(tensor(api, value_cache.ptr, ACL_BF16,
                  {f.shape.num_blocks, tq_npu::kBlockSize, tq_npu::kHeads, tq_npu::kHeadSize})),
          executor(nullptr, ExecutorDeleter{api.destroy_executor}) {
        char cache_mode[] = "Norm", scatter_mode[] = "None";
        aclOpExecutor* raw = nullptr;
        const auto status = api.prepare(
            key.get(), key_out.get(), slots.get(), value.get(), value_out.get(),
            nullptr, nullptr, nullptr, cache_mode, scatter_mode, nullptr, nullptr,
            &workspace_size, &raw);
        executor.reset(raw);
        tq_npu::check(status, "aclnnScatterPaKvCacheGetWorkspaceSize");
        tq_npu::require(executor != nullptr, "native ingest returned null executor");
        tq_npu::check(api.repeatable(executor.get()), "native repeatable executor");
        tq_npu::require(workspace_size <= std::numeric_limits<std::size_t>::max() - 31,
                        "native workspace size overflow");
        if (workspace_size) workspace = std::make_unique<tq_npu::Buffer>(
            static_cast<std::size_t>((workspace_size + 31) / 32 * 32));
    }
    void launch(aclrtStream stream) {
        tq_npu::check(api.launch(workspace ? workspace->ptr : nullptr,
                         workspace_size, executor.get(), stream), "aclnnScatterPaKvCache");
    }
};
}  // namespace tq_ingest_bench
#endif
