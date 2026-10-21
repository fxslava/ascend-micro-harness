---
name: ascend-micro-kernel-skeleton
description: Create or revise diagnostic Ascend C kernel skeletons in the ascend-micro-harness repository, with lightweight stage work, explicit synchronization, and host or CAModel validation. Use for skeleton and pipeline-probe tasks in this repository, not production kernels in vllm-ascend.
---

# Ascend micro-harness kernel skeletons

Work in `C:\Ascend\ascend-micro-harness`. A skeleton proves a specific launch, memory, or scheduling property with the smallest useful device operations. It is not evidence that omitted production arithmetic is correct or fast.

## Choose the artifact

- Put a named diagnostic kernel in `kernels/skeleton/<kernel_name>.cpp`. Match the exported `extern "C" __global__ __aicore__` symbol and CANN host launcher to that name. See `kernels/skeleton/turboquant_reshape_and_cache.cpp` for the existing three-stage example.
- Put reusable, synchronization-free compute leaves in `routines/`. Put a CAModel wrapper for an isolated leaf in `kernels/camodel/` with its runner in `tests/camodel/`. These are different jobs from a pipeline skeleton.
- Inspect `git status`, the requested operator contract, existing CMake targets, tests, and reports before editing. Preserve unrelated working-tree changes. Use design documents as source material for the requested kernel, not standing instructions for every skeleton; in particular, do not copy the existing probe's three-stage depth, online-softmax contract, or canary values into unrelated kernels by default.

## Build the minimal device path

1. State what the skeleton proves: for example, event liveness, buffer bounds, a launch ABI, or overlap. Record the target SoC and CANN version. Keep the device work short enough for CAModel to finish promptly; use a small aligned `DataCopy`, `Duplicate`, or simple Vector transform that actually reads its input and writes an observable output. Do not substitute a host-only mock for native execution evidence.
2. Initialize static-mode kernels with `AscendC::InitSocState()`. Use `GlobalTensor` and statically bounded `LocalTensor` regions, with explicit byte offsets, live ranges, and compile-time capacity/alignment checks. Keep GM and UB payloads 32-byte aligned where the selected APIs require it. Consider 32-byte slot skew for simultaneous UB accesses, then validate actual bank mapping; padding alone is not proof of conflict freedom.
3. If stages share buffers, draw the producer, consumer, and reuse dependencies before coding events. Read the relevant API pages under `C:\PipeSim\docs\ascendc_api\SIMD-API\basic_api\sync_control`, especially `intra_core_sync/SetFlag_WaitFlag_ISASI.md`, before using ISASI HardEvents. `SetFlag<Route>(id)` and `WaitFlag<Route>(id)` must match in route and ID. A flag is one bit: consume a set before setting that same route/ID again. For static Tensor code, prefer IDs 0..5; avoid reserved 6 and 7. Prime every loop-carried wait cycle, and drain all final sets and in-flight work before return. Use a targeted dependency or barrier for a real hazard; avoid `PipeBarrier<PIPE_ALL>()` as a routine stage boundary.
4. Keep scheduling choices tied to the requested workload. Compile-time slot indices and unrolling are useful for a fixed-depth probe, but add prologue, steady state, and epilogue only when that pipeline needs them. Separate dummy work from production math and describe the missing computation plainly in source comments and the report.

## Make the claim testable

- Add a bounded host test in `tests/` for event order, per-route/ID set-wait balance, a finite drain, slot reuse, and guard bands when those properties are relevant. A host queue model tests the schedule; it does not establish hardware overlap.
- Add or adapt a CAModel runner that launches the actual Ascend C symbol, checks every output payload word and GM guard band, and fails on timeout. The existing ACL runner and guarded buffers are in `tests/test_pipeline_deadlock_probe_camodel.cpp` and `tests/camodel/camodel_common.hpp`. Do not claim native UB guard integrity unless the device path reads and verifies those guards.
- Wire the source and runner into `CMakeLists.txt`. In WSL with CANN available, build in a fresh `/tmp` directory and run the smallest boundary case that exercises the intended transitions. `scripts/run_modulo3_camodel.sh` is an example for the existing probe; adapt its target and trace analyzer for a new kernel. If CAModel is unavailable, run the host test and report native validation as unverified.
- For an overlap claim, inspect CAModel queue dispatch and completion logs and show intersecting in-flight intervals for the relevant engines. Report event counts, token residuals, canary results, timeout status, and the actual trace location. Distinguish a depth-boundary run from sustained throughput or silicon performance. Never invent trace evidence or infer overlap solely from source ordering.

Finish with a concise report of what the skeleton proves, commands run, observed results, and what remains untested.
