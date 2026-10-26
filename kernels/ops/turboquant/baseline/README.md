# Vendored TurboQuant BF16 baseline

This directory contains the upstream `turboquant_reshape_and_cache_bfloat16_t`
implementation and its direct header dependencies, copied from
`vllm-ascend` commit `4d8637e396354240f4591d19fbd084a4c06c7990`:

- `csrc/attention/turboquant/op_kernel/turboquant_paged_attention.cpp`
- `csrc/attention/turboquant/op_kernel/common/{turboquant_codec_950,turboquant_common,turboquant_layout,turboquant_mode}.h`
- `csrc/attention/turboquant/op_kernel/vector/turboquant_vector_service.h`
- `csrc/kernels/types.h`

The only source change is the local `types.h` include path. The full upstream
Apache 2.0 license is in [LICENSE](LICENSE); each source file retains its
copyright and license header. The benchmark compiles this vendored source
without an external checkout.
