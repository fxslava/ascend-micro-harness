// Minimal stand-in for the vendored baseline's types.h, host-compile gate only.
#pragma once

namespace vllm_ascend {
enum class AscendType { FP16, BF16, FP32, INT8 };
}  // namespace vllm_ascend
