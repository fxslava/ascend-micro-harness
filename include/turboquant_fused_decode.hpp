#pragma once
// The fused decode's host tiling contract. One definition, in the integration subtree that is
// exported to vllm-ascend; this is the repository-root spelling of the same header, so a host
// translation unit under tests/ or scripts/ includes it by the short path and still sees the
// single declaration the device kernel compiles against.
#include "../kernels/ops/turboquant/vllm_integration_layer/include/turboquant_fused_decode.hpp"
