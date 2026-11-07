# Deprecating the AIV single-token decode kernels

Date: 2026-11-07. Branch: `claude/micro-camodel-validation`.
Target: CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599` (CAModel; this host has no NPU).

The two AIV-only single-token decode kernels are deleted and the harness is consolidated
onto the fused decode operator,
`kernels/ops/turboquant/vllm_integration_layer/op_kernel/turboquant_fused_decode.cpp`.
Every gate the deleted runners carried is accounted for below: ported, already present on
the fused operator, or recorded as not applicable with the reason.

This report follows `REFINEMENT_OF_COMBAT_KERNELS.md`. Section 6.7 of that document is the
rule this report tries to obey literally: every gate is listed as run, failed or
unavailable, with the reason.

---

## 1. What was deleted

### Kernels

| File | What it was |
| --- | --- |
| `kernels/ops/turboquant/turboquant_single_token_decode_aiv.cpp` | `KERNEL_TYPE_AIV_ONLY` single-token decode, two-slot ring with an MTE3 archive edge, 6 query rows, grid of (request, kv_head) |
| `kernels/ops/turboquant/turboquant_single_token_decode_splitk_aiv.cpp` | the same operator with page-partitioned Split-K and an in-launch record-ring reduction |

### Test drivers and fixtures that launched them

| File | Gate it carried | Where that gate is now |
| --- | --- | --- |
| `tests/camodel/test_camodel_single_token_decode_aiv.cpp` | single-head canary sweep, multi-page rung, row sweep, bit-exact score panels | the fused ladder's isolated page canaries and sequence sweep |
| `tests/camodel/test_camodel_decode_splitk_boundary.cpp` | byte-exact Split-K context, `ASSERT_CANARY_INTACT` over every unassigned byte | the fused runner, which now prints the same census |
| `tests/npu/test_npu_decode_single_token_unaligned.cpp` | S = 1, 7, 15, 17, 31, 33, 127, 129, 255, 257, 1023 unaligned ladder | the fused runner's `--tail-rows` rungs, including over the whole block table |
| `tests/npu/validation/test_turboquant_decode_splitk_aiv_validation.cpp` | the AIV Split-K correctness ladder | `test_npu_fused_decode_opt` (hardware) and the `splitk-workers` rung (CAModel) |
| `tests/common/decode_boundary_fixture.hpp` | the exact arithmetic fixture (query 1/256, key nibbles 3, scales 1, gate 0) | inlined in the fused runner's exact mode |

### Scripts that drove only those kernels

`scripts/run_single_token_decode_camodel.sh`, `scripts/run_decode_pipeline_camodel.sh`,
`scripts/run_decode_pipeline_regressions.sh`, `scripts/verify_decode_pipeline_evidence.py`,
`scripts/run_decode_tail_container.sh`.

### CMake targets removed

Simulator leg (`cmake/SimTargets.cmake`): `camodel_single_token_decode_aiv_kernel`,
`test_camodel_single_token_decode_aiv`, `camodel_single_token_decode_splitk_kernel`,
`test_camodel_decode_splitk_boundary`, `test_decode_splitk_aiv_sim`,
`test_decode_single_token_unaligned_sim`.

Physical leg (`cmake/NpuTargets.cmake`): `turboquant_single_token_decode_npu`,
`turboquant_single_token_decode_splitk_npu`, `test_npu_decode_splitk_aiv`,
`test_npu_decode_single_token_unaligned` (also dropped from the RPATH loop and from
`turboquant_npu_all`).

---

## 2. What was deliberately KEPT, and why

These are the judgement calls. Each one is a file the deleted kernels were associated with
that something surviving depends on.

* **`tests/common/single_token_decode_oracle.hpp`** and **`tests/cpu/test_single_token_decode_probe.cpp`**.
  The oracle header is the shared decode geometry and FP64 reference that
  `tests/common/splitk_decode_fixture.hpp` is built on, and through it
  `tests/cpu/test_aic_splitk_decode_probe.cpp`, `tests/cpu/test_aic_splitk_acc_error_model.cpp`,
  `tests/cpu/test_decode_tail_probe.cpp` and the decode benchmark. The probe is the only
  test of that header. Deleting either would have removed coverage of code the AIC
  operator still uses, so the probe's CMake comment now says what it is about rather than
  which kernel it came from.
* **`scripts/analyze_single_token_decode_trace.py`**. Not a test target: it is the
  CAModel dump parser (`parse`, `flag_key`) that
  `scripts/analyze_fused_decode_opt_trace.py`, `scripts/analyze_decode_ring_trace.py` and
  `scripts/export_fused_decode_gantt.py` all import. Its name is legacy; its content is
  not. Renaming it was not done here because it would touch three analyzers and every
  archived command line in `reports/`.
* **`scripts/analyze_decode_ring_trace.py`** and **`tests/cpu/test_decode_ring_trace_audit.py`**.
  The route/ID-keyed capacity-one flag audit with no operator-specific expectations, plus
  its negative controls. After this change no script drives it (the fused operator has its
  own analyzer); it is kept as the generic auditor for the next structure that needs one,
  and `REFINEMENT_OF_COMBAT_KERNELS.md` §5 now says so instead of presenting it as the
  decode audit.
* **`routines/routine_vector_decode_gemv.hpp`**. A microroutine-catalogue header. No
  device kernel includes it any more; the host oracle and the README still cite its
  repeat-stride derivation, which is the lesson the catalogue exists for.
* **`tests/cpu/test_decode_page_loop_model.py`**. It models
  `turboquant_paged_decode_splitk_aic`, which this change does not touch. Only its
  docstring's cross-reference was updated.
* **The AIC paged-decode decoder itself**, its validation ladder
  (`test_npu_decode_splitk_aic`, `test_decode_splitk_aic_sim`) and its CAModel probe. The
  task was the AIV kernels; the AIC operator is a separate product with its own evidence
  trail.

---

## 3. Gates ported to `turboquant_fused_decode`

### 3.1 Boundary canaries, `0x7FC0BEEF` and `ASSERT_CANARY_INTACT`

`tests/sim/kernel/test_camodel_fused_decode_opt.cpp` already flooded every GM payload with
`0x7FC0BEEF` (`tests/sim/kernel/fused_decode_opt_sim_common.hpp`), asserted both exterior
guard bands on the output and the workspace, held every unassigned output word and the
inactive workspace to the canary, and compared every input bitwise after synchronisation.
Two things were added so it carries the deleted Split-K boundary runner's gate in full:

* `assert_guards()` is now called on every INPUT buffer as well, before the bitwise
  comparison, so a burst that ran off either end of an input is named as a guard-band hit
  instead of as an anonymous "input bytes changed".
* the run counts the bytes it held to the canary and prints
  `ASSERT_CANARY_INTACT <n> bytes` in its PASS line, which is what made the deleted
  runner's log self-describing: a rung that quietly stopped checking shows up as a changed
  census rather than as a silent pass.

### 3.2 The exact arithmetic fixture

`REFINEMENT_OF_COMBAT_KERNELS.md` §3's fixture is query = 1/256, key nibbles = 3,
scales = 1 and gate = 0, so that every score is the same number and `exp(score - m) = 1`
exactly. The fused runner's exact mode had query = 1/256 and scales = `gain * 2` but a key
nibble of **0**; it is now **3**, which is the documented fixture and which makes the
packed key plane's content reach the score GEMM. The weights are 1 either way, so the
runner's oracle is unchanged by the switch — the point is that the fixture is the one the
protocol names.

`gate = 0` does not port: the fused operator has no gate tensor. Its query basis is
selected by the `PRE_ROTATED` template parameter and the probe instantiates the
pre-rotated entry, so there is no gate to zero. Recorded here rather than silently
dropped.

### 3.3 Bounded sequence sweeps and the multi-page block-table loop

`scripts/run_fused_decode_canary_ladder.sh` had rungs at 1, 2, 5, 3 and 6 tiles. Two rungs
were added so the sweep is N = 1, 2, 3, 4, 16, and a third for the unaligned case:

| Rung | Geometry | What it is |
| --- | --- | --- |
| `rung6-depth-plus-one` | `--tiles 4` | N = 4: the first refill after the prologue, one tile past `kSlots` |
| `rung7-full-block-table` | `--tiles 16` | N = 16: 1024 tokens over all eight logical pages of the block table |
| `rung8-full-table-ragged` | `--tiles 16 --tail-rows 17` | the whole table with the context ending inside the final tile |

The fixture's table is non-identity (logical page `p` holds physical page `2p + 1`) and the
full 18-page cache allocation is kept, so a cursor that walks the table wrongly lands on a
canary page rather than on another live one. Rung 7 is the multi-page block-table loop the
deleted `--pages 4` rung provided, at twice its depth.

Rungs 6 and 7 run at fp16 and at bf16. Rung 8 runs at fp16 only, for the reason the
script's header already gives for the cursor and flag rungs: the ragged-tail arithmetic is
run at both element types by rungs 2 and 3, and what rung 8 adds over rung 7 is block-table
depth against a partial final tile, which is descriptor words and ring indices with no
`scalar_t` in them.

### 3.4 The token/queue models

`tests/cpu/test_decode_ring_model.py` modelled the deleted AIV strip ring and the deleted
AIV Split-K record ring. It is **retargeted**, not duplicated: the fused operator's
attention ring is already modelled by `tests/cpu/test_fused_decode_opt_model.py` and its
default Cube GEMM reduction by `tests/cpu/test_gemm_reduce_model.py`, so a second model of
either would only be a drift hazard. What had no model was the **vector partial reducer**
(`TurboQuantPartialReducer::Reduce` in
`op_kernel/vector/turboquant_vector_service.h`), the reduction a `-DTQ_GEMM_REDUCE=OFF`
build selects. That is what the file now explores, over the real HardEvents:

* `V_MTE2` (`SyncVectorToMte2`) — the WAR fence in front of each split's `DataCopy`;
* `MTE2_V` (`SyncMte2ToVector`) — the RAW fence in front of each merge;
* `V_MTE3` / `MTE3_V` (`WriteHeads`) — the output row's hand-off and its return credit.

Three queues (V, MTE2, MTE3), one workspace landing and one output row — the reducer is
single-buffered, so every one of those four fences is load-bearing. §1a's outer loop here is
the **reduce-task loop**: `outBuf_`'s WAR edge exists only at the task boundary, because its
reader is MTE3 and its next writer is the next task's `Cast` on V, so every run carries the
marking and both owners across consecutive tasks. Negative controls: `drop_v_mte2`,
`drop_mte2_v`, `drop_v_mte3`, `drop_mte3_v` (the §1a boundary one) and `duplicate_v_mte2`.

`tests/cpu/test_decode_page_loop_model.py` was **not** retargeted. It models
`turboquant_paged_decode_splitk_aic`'s page loop, which is a surviving kernel; retargeting
it would have deleted a live gate. Its docstring's reference to the ring model was corrected.

---

## 4. The benchmark targets

`bench_turboquant_decode_splitk` and `bench_decode_splitk_sim` compile one driver,
`tests/npu/bench/bench_turboquant_decode_splitk.cpp`.

* The **AIV dispatch is removed outright**, not switched off: `AivRig` is gone from
  `tests/npu/turboquant_splitk_rigs.hpp`, `--impl aiv` and `--impl both` no longer exist,
  and the modelled AIV-against-AIC crossover is gone with the baseline it was relative to.
  `AivRig::Census`, which both rigs reported, is now `tq_splitk_rig::Census` at namespace
  scope.
* The `aiv_*` CSV columns are **kept in the schema and permanently NA**. `reports/` holds
  runs in that schema and a reader has to be able to line an old CSV up with a new one;
  `scripts/check_decode_splitk_bench_cli.sh` now asserts that direction explicitly.
* **Both targets run the fused operator exclusively by default.** The physical target
  already did. The simulator target did not: `cmake/SimTargets.cmake` hard-coded
  `TQ_BENCH_LEGACY_DECODERS=1`, because the option was declared inside
  `add_ascend_micro_npu_targets()`, which runs after the simulator targets are created and
  so was invisible to them. The option now lives in the top-level `CMakeLists.txt` beside
  `TQ_GEMM_REDUCE` and `ASCEND_MICRO_CANN_SYSTEM_OPTIONS`, for the reason those two are
  there: both legs compile the identical source.
* `TQ_BENCH_LEGACY_DECODERS=ON` now means **the AIC comparison columns**, and the four
  scripts that drive `--impl aic` pass it: `scripts/profile_camodel_trace.sh`,
  `scripts/run_decode_splitk_aic_page_depth.sh`,
  `scripts/run_decode_splitk_aic_page_race.sh` and
  `scripts/run_decode_splitk_strip_ladder.sh` (the last two through
  `SPLITK_LEGACY_DECODERS=ON` on `scripts/run_decode_splitk_bench.sh`).

**A concern worth stating.** "Exclusively run the fused implementation" read literally would
also delete the AIC comparison path. That is not done here: the AIC paged-decode operator
is not part of this deprecation, it has its own CAModel evidence trail through those four
scripts, and removing its columns would have retired that trail as collateral. The
compromise is that neither benchmark target contains it unless asked — which is what
"exclusively" buys — while the ladders that exist for it keep working. If the intent was to
retire the AIC benchmark too, that is a one-line change to the option's default plus the
four scripts, and it should be its own commit.

---

## 5. Format and cache conventions

Nothing in the surviving callers needed a format change. The fused decode's harness
callers already go through the host tiling builder
(`ascend_micro::fused_decode::build_decode_tiling`) for every tile count, live-tail length
and logical page span, take their GM pointers through `GuardedDeviceBuffer::payload()`,
which is a `static_cast<std::uint8_t*>` plus the guard offset, and derive the strip offset
from `kCubeTileRows` through `fd::kDecodeTileRows` — the kernel carries a `static_assert`
that the two agree. The arbitrary `block_size` layout is the runner's `--page-tokens`
dimension on the benchmark and the fixture's `page_rows` on the CAModel runner; both are
built from the same descriptor the device reads. The runner's output-slot arithmetic was
changed only to use `fd::element_bytes(c.element_type)` where it previously used a literal
`2`, in the new byte census.

---

## 6. Build time

The deprecation made the build shorter by removing four `ascendc_library` instances
(two CAModel, two physical). Driving the gates after it exposed how much of the wall
clock is structural, so this section records what was measured and what was changed.

### 6.1 What one `ascendc_library` costs

| Measurement | Value |
| --- | --- |
| clean build of `test_camodel_fused_decode_opt` | **2479 s** |
| compiles of the kernel in that build | **7** -- `precompile_obj`, `preprocess_obj` (`-E`), `aiv_obj`, `aic_obj`, `host_bisheng_obj`, plus the aic and aiv `device_obj` over the generated TU that includes it |
| the next target (`bench_decode_splitk_sim`), which only adds one small TU | **2753 s** |
| compiles in a no-op rebuild of the same target | **2** -- `aiv_obj` and `aic_obj` |

CANN's `device_preprocess_project` builds three libraries out of one kernel source, and
the separate `aic_device`, `aiv_device` and `host` projects compile it again. Every one
of those ExternalProjects carries `BUILD_ALWAYS TRUE`.

The two recompiles in a *no-op* rebuild are `cmake/CannRebuild.cmake`'s delete step
doing its job. That step is load-bearing, which this probe settled rather than assumed:

```
$ ld.lld -m aicorelinux -Ttext=0 aic_obj.o -o aic_obj.o     # the object from last build
ld.lld: error: aic_obj.o: unknown file type
```

CANN links those objects in place, turning `ET_REL` into `ET_EXEC`, and its always-run
merge step cannot link an `ET_EXEC` a second time. So they have to be deleted and
recompiled before every build. That is CANN's design; it is not something this
repository can switch off, and nothing here tries to.

### 6.2 What changed

* **A compiler-cache hook** (`ASCEND_MICRO_COMPILER_CACHE`, ON when a cache is found, in
  the top-level `CMakeLists.txt`). CANN forwards `CMAKE_C_COMPILER_LAUNCHER` and
  `CMAKE_CXX_COMPILER_LAUNCHER` onto the configure line of every nested project, which is
  the only hook that reaches the compiles above. Verified by configuring with a marker
  launcher and finding it in 373 generated files -- 27 per nested project kind, i.e. all
  27 `ascendc_library` instances across `precompile`, `preprocess`, `aic_device`,
  `aiv_device` and `host`. **The speedup itself is UNMEASURED here: ccache is not
  installed on this host and installing it needs root.** The mechanism is proved, the
  effect is not.
* **One `cmake --build` invocation instead of one per target.** Each invocation pays the
  `BUILD_ALWAYS` recompile of every `ascendc_library` in the chain, so the per-target
  loop in `scripts/run_splitk_workspace_build.sh` (six targets over shared kernel
  libraries), the two invocations in `scripts/run_l2_atomic_probe_camodel.sh` and the
  two in `scripts/run_fused_decode_opt_camodel.sh` paid it two to six times over. The
  workspace gate keeps per-target reporting by retrying target-by-target only after a
  combined failure, so a green run costs one pass.
* **`-j` from the host's core count** instead of a hard-coded `-j4`, across twelve
  scripts. The `aic_device` and `aiv_device` legs are independent ExternalProjects and
  overlap, as do separate libraries; this host has 12 cores.
* **The physical-leg compile gate is skippable while iterating.**
  `scripts/run_fused_decode_opt_camodel.sh` builds `turboquant_fused_decode_opt_npu`
  beside the CAModel runner -- the same operator at `RUN_MODE=npu`, a second full pass
  through BiSheng and about half the script's build time. It stays on by default because
  it is the section 2 gate; `FUSED_DECODE_OPT_SKIP_NPU_GATE=1` drops it for a CAModel
  iteration.
* **The Cube GEMM reduction model stays out of the canary ladder's host leg.** Adding it
  there (it is one of section 1's three models) overran a 1800 s cap: its deepest batched
  rung alone explores 15 388 101 states. It has its own driver,
  `scripts/run_gemm_reduce_camodel.sh`.

### 6.3 Durability: the work roots are no longer in `/tmp`

Every one of these scripts rooted its build and evidence tree in `/tmp`, for a good
reason -- the checkout is on `/mnt/c`, a DrvFs mount that CANN's build and the simulator
should not be run on. But `/tmp` is volatile as well as local, and a WSL restart during
this work deleted a finished 41-minute build and nine passed ladder rungs at once, with
nothing in the scripts able to tell that from a run that had never happened.

`scripts/lib/work_root.sh` resolves a root that is local **and** persistent:
`$ASCEND_MICRO_WORK_ROOT`, else `$XDG_CACHE_HOME`/`$HOME/.cache/ascend-micro`, else
`/tmp` with a warning that says what it costs. A `$HOME` that is really the Windows
drive is refused, because resolving onto DrvFs would undo the reason these scripts build
outside the checkout at all. `scripts/test_work_root.sh` is its behaviour gate: nine
assertions, including the two silent cases (the announced `/tmp` fallback and the DrvFs
refusal). The canary ladder, its iteration driver, the split-K benchmark, the merge
compile gate, the GEMM reduce ladder and the AIC page-depth ladder use it; the other
`/tmp`-rooted scripts are unchanged and still volatile.

`scripts/run_fused_decode_opt_case.sh` is now **resumable** to match: a case whose
directory already holds its `PASS` line is skipped, and one without is treated as an
interrupted launch and redone, so re-running the ladder against the same root continues
instead of restarting. `FUSED_DECODE_OPT_FORCE=1` overrides.

### 6.4 One unrelated defect found on the way

`scripts/run_l2_atomic_probe_camodel.sh` had CRLF line terminators in the working tree
while `.gitattributes` declares `*.sh text eol=lf` and the committed blob is LF. A CRLF
`#!/usr/bin/env bash` does not execute on Linux, and git hides the difference because it
normalises before comparing, so the file looked clean. Converted to LF; it is now the
only `.sh` in the tree that ever was.

---

## 7. Verification

<!-- RESULTS -->

---

## 8. Known pre-existing failure, not caused by this change

`scripts/test_decode_benchmark.cpp` (driven by
`scripts/run_decode_benchmark_host_checks.sh`) does not compile, and did not compile
before this change. Compiled from a clean `git archive HEAD` tree:

```
scripts/test_decode_benchmark.cpp:107:32: error: invalid initialization of reference of type
  'const Measurement&' from expression of type 'int'
  107 |         row(128, 2, 2, av, ac, 0, 200, 0, 1000000, csv);
```

It calls `row()` with ten arguments against a thirteen-parameter signature and asserts a
CSV layout several schema revisions old (its expected last field is `2.000`, which was
`export_ticks` before the `opt_*` columns were appended). Repairing it means deciding what
its assertions should be against the current table, which is its own task. The only change
made to it here is the mandatory `AivRig::Census` to `tq_splitk_rig::Census` rename, so it
fails for exactly the reason it failed before.
