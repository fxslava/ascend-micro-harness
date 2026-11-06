#include "../../op_adapter/turboquant_torch_adpt.h"
#include "../../op_adapter/turboquant_torch_ops.h"

static_assert(sizeof(vllm_integration_layer::kExistingDecodeSymbol) > 1);
static_assert(sizeof(vllm_integration_layer::kExistingScatterSymbol) > 1);

int main()
{
    const ascend_micro::fused_decode_opt::Launch config{};
    return config.head_size == ascend_micro::fused_decode_opt::kHeadSize ? 0 : 1;
}
