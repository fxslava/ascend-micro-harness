#pragma once

#include <cstddef>
#include <cstdint>

namespace turboquant_splitk {
constexpr std::uint32_t kChannels = 256;
constexpr std::uint32_t kStateLanes = 8;
constexpr std::uint32_t kMaxRows = 8;
constexpr std::uint32_t kMaxSplits = 16;
constexpr std::uint32_t kRecordFloats = 272;

struct RowState {
    float m_global[kStateLanes];
    float l_global[kStateLanes];
    float acc_global[kChannels];
};

// One lock and arrival counter per (request, KV head), with one independent
// softmax state per GQA query row. Padding keeps every MTE payload aligned
// and separates adjacent heads' atomic sectors. Reset before every launch.
struct alignas(128) AtomicHeadSlot {
    std::uint32_t lock;
    std::uint32_t arrived_count;
    std::uint32_t reserved[30];
    RowState row[kMaxRows];
};

// Control is SIMT-written with cache-bypassing stores; timestamp fields are
// SIMD-written, each as eight replicated lanes. finish is sampled after the
// unlock (intermediate) or final out_context DMA (tail).
struct alignas(32) WorkerMetrics {
    std::uint32_t acquired, arrival, failed_exchanges, timeout;
    std::uint32_t reserved[4];
    std::uint32_t finish_hi[8], finish_lo[8], tail[8], pages[8];
    // Conservative lock-hold upper bound: dispatch acquire through completed
    // release, including contention and the single control-register handoff.
    std::uint32_t lock_region_hi[8], lock_region_lo[8];
};

struct PageRange { std::uint32_t begin, count; };
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr PageRange Partition(std::uint32_t pages, std::uint32_t splits,
                              std::uint32_t split)
{
    const std::uint32_t width = (pages + splits - 1u) / splits;
    const std::uint32_t begin = split * width;
    const std::uint32_t remaining = begin < pages ? pages - begin : 0u;
    return {begin, remaining < width ? remaining : width};
}

static_assert(sizeof(RowState) == kRecordFloats * sizeof(float), "row ABI");
static_assert(offsetof(AtomicHeadSlot, row) == 128, "aligned state DMA");
static_assert(sizeof(AtomicHeadSlot) % 128 == 0, "independent atomic sectors");
static_assert(sizeof(WorkerMetrics) == 224, "metric ABI");
}  // namespace turboquant_splitk
