# TurboQuant integration staging, Phase 1

Date: 2026-11-05

## Staged sources

The subtree at `kernels/ops/turboquant/vllm_integration_layer/` contains the
three requested kernel sources and their service/helper dependencies. The
existing kernel entry signatures and implementations were retained. Only
include paths, stale cross-tree comment references, and the unused scatter
constant `kPlanes` were changed; the unused declaration removal was required
for the staged `-Wall -Werror` build and has no behavior effect.

The integration layout keeps the source ABIs separate: the optimized Cube
decode remains KV4/FP8 with its existing FP16 symbols, while the batch and
scatter writers remain BF16 signed-int4 strip-cache operators. Phase 1 performs
no cross-format synthesis and no Torch registration changes.

## Build and host validation

Configured and built all standalone CANN targets with BiSheng for
`Ascend950PR_9599` in both the supplied Docker image (CANN 9.1.0) and a WSL
CANN 9.2.0-beta.2 environment. Both builds produced the three kernel
libraries, device validation executables, and benchmarks. The adapter-header
compile target passed under CANN 9.2 without linking or requiring Torch. The
Docker build used `/tmp/tq-stage1-docker`; the WSL build used
`/tmp/tq-stage1-verified`.

The existing CPU `test_reshape_scatter_probe` passed. The staged adapter header
smoke executable passed.

## Hardware and binding status

No physical NPU is exposed in WSL (`npu-smi` and `/dev/davinci*` are absent),
and the running Docker containers have no device requests. The staged device
validation executables and benchmarks therefore compiled but were not run on
hardware. No PyTorch binding build was attempted; the standalone CMake project
does not depend on Torch or `torch_npu`.
