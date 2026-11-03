#pragma once
#include <cstdint>
#include "../kernels/ops/turboquant/fused_decode/types.h"
#include "../kernels/ops/turboquant/fused_decode/common/turboquant_layout.h"
#include "../kernels/ops/turboquant/fused_decode/common/turboquant_mode.h"

namespace ascend_micro::fused_decode_opt {
constexpr std::uint32_t kHeadSize = 256;
constexpr std::uint32_t kTileRows = vllm_ascend::turboquant::kCubeTileRows;
constexpr std::uint32_t kSlots = vllm_ascend::turboquant::kCubePipelineSlots;
constexpr std::uint32_t kWindows = vllm_ascend::turboquant::kCubeScoreWindows;
constexpr std::uint32_t kTaskJoinFlag = 10;
static_assert(kSlots == 3 && kWindows == 2 && kTileRows == 64);
static_assert(vllm_ascend::turboquant::kOptFlagProbsReady < kTaskJoinFlag);
static_assert(kTaskJoinFlag < vllm_ascend::turboquant::kOptFlagReservedFirst);

// UB allocation sizes, in production InitBuffers order, for D=256 and <=4 KV
// heads (scaleSlot=8). Includes codec workspace; all payloads are disjoint.
constexpr std::uint32_t kUbSizes[] = {
    2048, 2048, 8192, 8192, 32, 8192, 32, 8192, 32,
    2048, 32, 2048, 32, 2048, 32, 16384, 8192, 8192, 8192,
    2048, 512, 3072, 2048, 512, 1024, 4096, 512,
    (256 * 32 + 3 * 256 + 64 + 8 + 8 * 256) * 4};
constexpr std::uint32_t ub_end()
{
    std::uint32_t bytes = 0;
    for (auto size : kUbSizes) bytes += size;
    return bytes;
}
constexpr bool aligned_allocations()
{
    for (auto size : kUbSizes) if (size % 32) return false;
    return true;
}
static_assert(aligned_allocations());
static_assert((8192 + 32) / 32 % 16 != 0, "packed slots must use different banks");
static_assert((2048 + 32) / 32 % 16 != 0, "scale slots must use different banks");
constexpr std::uint32_t kUbEnd = ub_end();
constexpr std::uint32_t kBasisUbBytes = 8 * 256 * (2 + 4 + 2 + 4) + 256 * 4 +
    (7 * 256 + 2 * 256 + 16) * 4 + (2 * 256 + 256 + 64 + 8 + 8 * 256) * 4;
constexpr std::uint32_t kReducerUbBytes = (2 * 256 + 16) * 4 + 5 * 8 * 4 +
    (2 * 64 + 2 * 8) * 4 + 256 * 2;
static_assert(kUbEnd + kBasisUbBytes + kReducerUbBytes + 256 <= 192 * 1024,
    "reserve a final full-register read pad even with raw basis and reduction");
static_assert(16 * 256 + 16 * 64 + 2 * kSlots * 64 * 256 <= 512 * 1024);

constexpr bool valid_launch(std::uint32_t grid, std::uint32_t tokens,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim,
    std::uint32_t block_size, std::uint32_t max_blocks, std::uint32_t splits,
    std::uint32_t heads_per_task, std::uint32_t tasks_per_block,
    std::uint32_t reduce_tasks_per_block)
{
    if (!grid || grid > 32 || !tokens || !kv_heads || kv_heads > 4 || !heads ||
        heads % kv_heads || dim != kHeadSize || !block_size || block_size % kTileRows ||
        !max_blocks || !splits || splits > 8 || !heads_per_task || heads_per_task > 16 ||
        !tasks_per_block || !reduce_tasks_per_block) return false;
    const auto group = heads / kv_heads;
    const auto chunk = heads_per_task < group ? heads_per_task : group;
    const std::uint64_t tasks = std::uint64_t{tokens} * kv_heads * splits *
        ((std::uint64_t{group} + chunk - 1) / chunk);
    return tasks <= UINT32_MAX && std::uint64_t{grid} * tasks_per_block >= tasks &&
        std::uint64_t{tokens} * heads <= UINT32_MAX &&
        std::uint64_t{grid} * tasks_per_block <= UINT32_MAX &&
        std::uint64_t{grid} * reduce_tasks_per_block <= UINT32_MAX &&
        std::uint64_t{grid} * reduce_tasks_per_block >= std::uint64_t{tokens} * heads;
}

struct Launch {
    std::uint32_t grid = 1, tokens = 1, heads = 16, kv_heads = 1;
    std::uint32_t head_size = kHeadSize, block_size = 128, max_blocks = 8;
    std::uint32_t splits = 1, heads_per_task = 16, tasks_per_block = 1;
    std::uint32_t reduce_tasks_per_block = 16, fused_context_limit = 4096;
    float scale = 0.0625f, inv_sqrt_len = 0.0625f;
    // Diagnostic ABI only: production launcher never receives these fields.
    std::uint32_t probe_page = UINT32_MAX, probe_tile = 0;
};
struct Buffers {
    void *query, *key, *value, *scales, *block_table, *context_lengths;
    void *mode_tables, *workspace, *output;
};
} // namespace ascend_micro::fused_decode_opt

extern "C" void launch_fused_decode_opt(void* stream,
    const ascend_micro::fused_decode_opt::Launch* launch,
    const ascend_micro::fused_decode_opt::Buffers* buffers);

// Original launch APIs retained for callers that already have upstream tiling.
namespace vllm_ascend {
void turboquant_mm_fused_decode_opt_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim,
                                         void *queryRot, void *keyCache, void *valueCache, void *scaleCache,
                                         void *blockTables, void *contextLens, void *modeTables, void *workspace,
                                         void *output, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                         uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                         uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock,
                                         uint32_t reduceTasksPerBlock, uint32_t fusedContextLimit, float scale,
                                         float invSqrtLen);
void turboquant_mm_fused_decode_opt_raw_query_impl(
    int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *query, void *piSigns, void *rotTables,
    void *h16, void *gate, void *queryRot, void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
    void *contextLens, void *modeTables, void *workspace, void *output, uint32_t numTokens, uint32_t numHeads,
    uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
    uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock, uint32_t prologueVectorsPerBlock,
    uint32_t prologueCubeChunkVectors, uint32_t outputStage, uint32_t fusedContextLimit, float scale,
    float invSqrtLen);
} // namespace vllm_ascend
