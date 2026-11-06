# TurboQuant integration staging, Phase 1

This subtree stages the existing harness implementations for isolated build
and validation. Phase 1 preserves every kernel entry-point signature and its
existing cache ABI:

* `turboquant_fused_decode.cpp` **is** the fused decode -- the only copy. It began
  as a staging copy of `kernels/ops/turboquant/turboquant_fused_decode_opt.cpp`;
  that file is deleted, and the simulator leg, the physical leg and this
  subtree's own legs all compile this one source
  (`reports/fused_decode_host_tiling_audit_20261106.md`). Its launch ABI gained
  one GM pointer in that change: the host tiling descriptors of
  `include/turboquant_fused_decode.hpp`, which every caller must build and
  upload, and without which the launcher refuses the launch.
* `turboquant_reshape_batch.cpp` comes from `turboquant_reshape_and_cache_bf16.cpp`.
* `turboquant_reshape_scatter.cpp` comes from `turboquant_reshape_scatter_aiv.cpp`.

The Cube decode service uses its existing KV4/FP8 cache and FP16 entry points.
The batch writer and scatter writer use the harness's BF16 signed-int4 strip
cache. Phase 1 does not merge these formats or alter their interfaces.

The service classes and required micro-routines are staged in `op_kernel/`.
The decode validation and benchmark are copied from the harness NPU suite.
The scatter validation is the existing ACL-based canary runner; despite its
original CAModel filename it can also run against a physical ACL device.

## Verification and benchmark targets

| Target | Needs a device? | What it covers |
| --- | --- | --- |
| `test_integration_host_gate` | no | The oracles and shared contract headers, Tier-1 |
| `test_integration_adapter_headers` | no | The Phase 1 adapter declaration surface |
| `test_integration_decode` | yes | The fused decode ladder, nine geometries x split sweep |
| `test_integration_batch` | yes | The prefill batch writer's strip ladder, bit-exact per page |
| `test_integration_scatter` | yes | The scatter ingress canary, sweep, RMW and zero-row modes |
| `bench_integration_pipeline` | yes | Decode latency over the context/split sweep |
| `bench_integration_batch` | yes | Prefill batch latency and traffic over the strip ladder |
| `bench_integration_scatter` | yes | Scatter launch latency |

`test_integration_host_gate` is the only one of these that runs on a host with
neither an NPU nor the simulator libraries: every other executable opens an ACL
device session in `main()`. It is not kernel evidence -- no kernel is launched --
and it must not be reported as such. What it establishes is that the oracles the
device runners compare against are themselves sound, which is a precondition for
their verdicts meaning anything, plus the cross-header UB restatement between
`include/turboquant_gemm_reduce.hpp` and `include/turboquant_fused_decode_opt.hpp`
that no production translation unit can check because none includes both. Its own
header lists the six gates.

`bench_integration_batch` verifies every rung against the shared oracle before it
times it, so a kernel that is fast because it skipped work cannot produce a
figure. Built for the simulator (`-DTURBOQUANT_RUN_MODE=sim`, or through the
harness's sim leg) the benchmarks carry `ASCEND_MICRO_BENCH_SIM`: the CAModel's
stub driver rejects ACL timing events, so the latency columns read NA and the run
is a functional pass over the ladder.

## Build through the harness's own CMake

The subtree is also registered in the harness's top-level `CMakeLists.txt`
through `cmake/IntegrationTargets.cmake`. The device-free targets
(`test_stage_host_gate`, `test_stage_adapter_headers`, `turboquant_stage_tiling`)
are always built. The device legs are behind a switch, because the three kernel
sources are staging copies and building them here compiles the same translation
units a second time per leg, on every build:

```bash
cmake -B build -DTQ_VLLM_INTEGRATION_LAYER=ON
cmake --build build --target turboquant_stage_sim   # CAModel leg
cmake --build build --target turboquant_stage_npu   # physical leg
```

Those targets are named `{test,bench}_stage_*_{sim,npu}` to keep both legs in one
build tree. `TQ_GEMM_REDUCE` selects the decode operator's reduction phase and
defaults to ON in the harness build and in `standalone/` alike, so a staged build
compiles the same operator the harness verifies under that file name.

## Build with BiSheng and CANN

The standalone CMake project has no Torch or `torch_npu` dependency. The
Phase 1 build was verified in the supplied Docker image with its CANN 9.1
toolkit; this command writes build artifacts to a dedicated folder under the
harness checkout:

```bash
docker run --rm \
  -v /mnt/c/Ascend/ascend-micro-harness:/workspace/ascend-micro-harness \
  -w /workspace/ascend-micro-harness \
  --entrypoint bash bbd643260f6c -lc '
    source /usr/local/Ascend/cann-9.1.0/set_env.sh
    cmake -S kernels/ops/turboquant/vllm_integration_layer/standalone \
          -B build/turboquant-integration -DCMAKE_BUILD_TYPE=Release \
          -DSOC_VERSION=Ascend950PR_9599 \
          -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
    cmake --build build/turboquant-integration -j 4
  '
```

Use a fresh build directory for each clean build. The CANN Ascend C build
generates intermediate objects in place and may reject a second build in the
same directory. Executables are under `build/turboquant-integration/bin/`;
device tests and benchmarks require an Ascend NPU runtime.
