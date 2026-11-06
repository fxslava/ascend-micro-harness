// Simulator-only probe and launcher. All processing comes from production.
#include "../../../kernels/ops/turboquant/vllm_integration_layer/op_kernel/turboquant_fused_decode.cpp"
#if defined(__CCE_AICORE__)
// Isolation changes only the cursor's tile selection. All arithmetic and the
// ring body are the same ComputeTask/Run*Pipeline methods as production.
// The isolated single-tile canary. It is the production Process() over the production
// descriptor layout; which page and which 64-row tile of it are live is `block_start_idx` and
// `tile_origin` in the host tiling the runner uploads, not a probe branch inside the device's
// cursor. This entry point survives only to pin the probe's GEOMETRY (one token, one KV head,
// one split, the full 8-page table) in a translation unit under tests/.
//
// ONE BODY, TWO ELEMENT TYPES. The probes are a macro over scalar_t rather than two hand-written
// pairs, so the bf16 canary cannot drift from the fp16 one in geometry, in tiling or in the
// methods it calls -- which is the only reason the two ladders are comparable at all. Production's
// own instances come from ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT at the bottom of the included
// kernel source and are not touched here.
#define ASCEND_TQ_DECLARE_FUSED_DECODE_OPT_PROBES(SUFFIX, TYPE)                                 \
    extern "C" __global__ __aicore__ void turboquant_fused_decode_opt_probe##SUFFIX(            \
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR scales, GM_ADDR table,               \
        GM_ADDR lengths, GM_ADDR modes, GM_ADDR workspace, GM_ADDR output, GM_ADDR tiling,      \
        uint32_t heads)                                                                         \
    {                                                                                           \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                      \
        AscendC::InitSocState();                                                                \
        AscendC::TPipe pipe;                                                                    \
        TurboQuantFusedDecodeOpt<TurboQuantMode::KV4_FP8, TYPE> op(&pipe);                      \
        op.Init(query, key, value, scales, table, lengths, modes, workspace, output,            \
            tiling, 1, heads, 1, 256, 128, 8, 1, 16, 4096, 1.0f, 0.0625f);                      \
        op.Process((heads + 15) / 16);                                                          \
        DrainFusedDecodeTask();                                                                 \
    }                                                                                           \
                                                                                                \
    extern "C" __global__ __aicore__ void turboquant_fused_decode_opt_reduction_probe##SUFFIX(  \
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR scales, GM_ADDR table,               \
        GM_ADDR lengths, GM_ADDR modes, GM_ADDR workspace, GM_ADDR output, GM_ADDR tiling)      \
    {                                                                                           \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                      \
        AscendC::InitSocState();                                                                \
        AscendC::TPipe pipe;                                                                    \
        TurboQuantFusedDecodeOpt<TurboQuantMode::KV4_FP8, TYPE> op(&pipe);                      \
        constexpr uint32_t numHeads = 64, headSize = 256, numSplits = 8;                        \
        constexpr uint32_t reduceTasksPerBlock = 2;                                             \
        op.Init(query, key, value, scales, table, lengths, modes, workspace, output,            \
            tiling, 1, numHeads, 4, headSize, 128, 128, numSplits, 16, 0, 1.0f, 0.0625f);       \
        op.ProcessReductionProbe(1);                                                            \
        if (op.NeedsReduction()) {                                                              \
            AscendC::SyncAll<false>();                                                           \
            ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE(TYPE);                                           \
        }                                                                                        \
        DrainFusedDecodeTask();                                                                  \
    }

ASCEND_TQ_DECLARE_FUSED_DECODE_OPT_PROBES(, half)
ASCEND_TQ_DECLARE_FUSED_DECODE_OPT_PROBES(_bf16, bfloat16_t)

#elif defined(ASCEND_MICRO_CANN_LAUNCH)
extern "C" __global__ __aicore__ void turboquant_fused_decode_opt_probe(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    uint32_t);
extern "C" __global__ __aicore__ void turboquant_fused_decode_opt_reduction_probe(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
extern "C" __global__ __aicore__ void turboquant_fused_decode_opt_probe_bf16(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    uint32_t);
extern "C" __global__ __aicore__ void turboquant_fused_decode_opt_reduction_probe_bf16(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
extern "C" void launch_fused_decode_opt_sim(void* stream,
    const ascend_micro::fused_decode_opt::Launch* c,
    const ascend_micro::fused_decode_opt::Buffers* b)
{
    if (!c || !b) throw std::invalid_argument("null simulator descriptor");
    if (!b->tiling) throw std::invalid_argument("simulator launch without host tiling");
    // Refuse an element type with no probe rather than quietly running the fp16 one: the
    // operator's whole audit trail depends on a launch either computing what was asked for or
    // failing loudly. (ascend_type_of applies the same rule to the production entries.)
    const bool bf16 = c->element_type == 1;
    if (c->element_type > 1)
        throw std::invalid_argument("simulator launch with no probe for its element type");
    if (c->probe_page == UINT32_MAX - 1) {
        if (c->grid != 32 || c->tokens != 1 || c->heads != 64 || c->kv_heads != 4 ||
            c->head_size != 256 || c->block_size != 128 || c->max_blocks != 128 || c->splits != 8)
            throw std::invalid_argument("unsupported synthetic reduction probe geometry");
        if (bf16)
            turboquant_fused_decode_opt_reduction_probe_bf16<<<32, nullptr, stream>>>(b->query,
                b->key, b->value, b->scales, b->block_table, b->context_lengths, b->mode_tables,
                b->workspace, b->output, b->tiling);
        else
            turboquant_fused_decode_opt_reduction_probe<<<32, nullptr, stream>>>(b->query, b->key,
                b->value, b->scales, b->block_table, b->context_lengths, b->mode_tables,
                b->workspace, b->output, b->tiling);
        return;
    }
    if (c->probe_page != UINT32_MAX) {
        if (c->probe_page >= 8 || c->probe_tile >= 2 || c->grid != 1 || c->splits != 1 ||
            c->tokens != 1 || c->kv_heads != 1 || c->head_size != 256 || c->block_size != 128 ||
            c->max_blocks != 8 || !c->heads || c->heads > 32)
            throw std::invalid_argument("unsupported fused decode probe geometry");
        if (bf16)
            turboquant_fused_decode_opt_probe_bf16<<<1, nullptr, stream>>>(b->query, b->key,
                b->value, b->scales, b->block_table, b->context_lengths, b->mode_tables,
                b->workspace, b->output, b->tiling, c->heads);
        else
            turboquant_fused_decode_opt_probe<<<1, nullptr, stream>>>(b->query, b->key,
                b->value, b->scales, b->block_table, b->context_lengths, b->mode_tables,
                b->workspace, b->output, b->tiling, c->heads);
        return;
    }
    launch_fused_decode_opt(stream, c, b);
}
#endif
