// Simulator-only probe and launcher. All processing comes from production.
#include "../../../kernels/ops/turboquant/turboquant_fused_decode_opt.cpp"
#if defined(__CCE_AICORE__)
// Isolation changes only the cursor's tile selection. All arithmetic and the
// ring body are the same ComputeTask/Run*Pipeline methods as production.
extern "C" __global__ __aicore__ void turboquant_fused_decode_opt_probe(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR scales, GM_ADDR table,
    GM_ADDR lengths, GM_ADDR modes, GM_ADDR workspace, GM_ADDR output,
    uint32_t heads, uint32_t page, uint32_t tile)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    TurboQuantFusedDecodeOpt<TurboQuantMode::KV4_FP8, half> op(&pipe);
    op.Init(query, key, value, scales, table, lengths, modes, workspace, output,
        1, heads, 1, 256, 128, 8, 1, 16, 4096, 1.0f, 0.0625f);
    op.InitProbe(page, tile);
    op.Process((heads + 15) / 16);
    DrainFusedDecodeTask();
}

extern "C" __global__ __aicore__ void turboquant_fused_decode_opt_reduction_probe(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR scales, GM_ADDR table,
    GM_ADDR lengths, GM_ADDR modes, GM_ADDR workspace, GM_ADDR output)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    TurboQuantFusedDecodeOpt<TurboQuantMode::KV4_FP8, half> op(&pipe);
    constexpr uint32_t numHeads = 64, headSize = 256, numSplits = 8;
    constexpr uint32_t reduceTasksPerBlock = 2;
    op.Init(query, key, value, scales, table, lengths, modes, workspace, output,
        1, numHeads, 4, headSize, 128, 128, numSplits, 16, 0, 1.0f, 0.0625f);
    op.ProcessReductionProbe(1);
    if (op.NeedsReduction()) {
        AscendC::SyncAll<false>();
        ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE(half);
    }
    DrainFusedDecodeTask();
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)
extern "C" __global__ __aicore__ void turboquant_fused_decode_opt_probe(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    uint32_t, uint32_t, uint32_t);
extern "C" __global__ __aicore__ void turboquant_fused_decode_opt_reduction_probe(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
extern "C" void launch_fused_decode_opt_sim(void* stream,
    const ascend_micro::fused_decode_opt::Launch* c,
    const ascend_micro::fused_decode_opt::Buffers* b)
{
    if (!c || !b) throw std::invalid_argument("null simulator descriptor");
    if (c->probe_page == UINT32_MAX - 1) {
        if (c->grid != 32 || c->tokens != 1 || c->heads != 64 || c->kv_heads != 4 ||
            c->head_size != 256 || c->block_size != 128 || c->max_blocks != 128 || c->splits != 8)
            throw std::invalid_argument("unsupported synthetic reduction probe geometry");
        turboquant_fused_decode_opt_reduction_probe<<<32, nullptr, stream>>>(b->query, b->key,
            b->value, b->scales, b->block_table, b->context_lengths, b->mode_tables,
            b->workspace, b->output);
        return;
    }
    if (c->probe_page != UINT32_MAX) {
        if (c->probe_page >= 8 || c->probe_tile >= 2 || c->grid != 1 || c->splits != 1 ||
            c->tokens != 1 || c->kv_heads != 1 || c->head_size != 256 || c->block_size != 128 ||
            c->max_blocks != 8 || !c->heads || c->heads > 32)
            throw std::invalid_argument("unsupported fused decode probe geometry");
        turboquant_fused_decode_opt_probe<<<1, nullptr, stream>>>(b->query, b->key,
            b->value, b->scales, b->block_table, b->context_lengths, b->mode_tables,
            b->workspace, b->output, c->heads, c->probe_page, c->probe_tile);
        return;
    }
    launch_fused_decode_opt(stream, c, b);
}
#endif
