# Fused decode service snapshot

Imported from `C:/Ascend/vllm-ascend/csrc/attention/turboquant/op_kernel`
at checkout revision `4d8637e396354240f4591d19fbd084a4c06c7990`.
`upstream_sha256.json` records the actual imported bytes, including any local
upstream edits; the Git revision alone is not the snapshot identity.
Apache-2.0 notices remain in each imported file. Builds use this directory and
do not need the other checkout, PyTorch, or the vLLM extension.

The integration is `../vllm_integration_layer/op_kernel/turboquant_fused_decode.cpp`,
which is the single copy of this operator and the one every build leg compiles; the
service headers in this directory are the pre-staging originals and the decode no
longer reads them. Its original FP16
pre-rotated and raw-query kernel symbols and namespace launch functions are
retained. The harness API is `include/turboquant_fused_decode_opt.hpp`.

Local changes to the snapshot:

* Batched softmax uses `routine_vector_online_softmax.hpp` through the native
  64-key engine in `routine_vector_cube_softmax.hpp`; decay and context
  accumulation use the same new leaf file.
* Affine nibble widening uses `routine_turboquant_dequant4.hpp`'s signed cast
  engine. Deinterleave, the +0.5 midpoint, FP8 rounding and source scale/gain
  conventions are preserved.
* UB allocations use the public compile-time budget list, including the
  production codec workspace. The launch contract currently admits D=256,
  1..4 KV heads, 1..16 heads per task, block sizes divisible by 64,
  up to eight splits and grids up to 32 blocks. Wider/multi-wave dispatch is
  deliberately outside the admitted contract until its gates are implemented.

This cache is **not compatible** with the harness's signed-int4/BF16 16-token
strip cache. It is the source's 64-token NZ-tiled signed-nibble cache, split
into low/high channel halves, with interleaved per-token K/V scales padded
to eight FP32 lanes. Unpacking reconstructs signed nibble + 0.5, followed by
the source's gain convention. Output is FP16; rotated queries are FP32.

See `reports/fused_decode_opt.md` for verification status and limitations.
