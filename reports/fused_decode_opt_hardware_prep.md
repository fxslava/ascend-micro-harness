# The fused decode operator, prepared for physical NPU validation and profiling

Target: CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599`, `KERNEL_TYPE_MIX_AIC_1_2`.

**No NPU on this host.** `npu-smi` and `/dev/davinci*` are both absent, so **nothing below is a
hardware claim**, including everything in the hardware validation suite this change adds. What
that suite establishes today is that it compiles at `-Wall -Wextra -Werror`; it has never run.
The CAModel evidence this change builds on is
[`gemm_batched_reduction_camodel_20261104.md`](gemm_batched_reduction_camodel_20261104.md) and
[`fused_decode_opt.md`](fused_decode_opt.md).

**The CAModel ladders were excluded from this change by instruction**, in favour of moving
directly to hardware verification and profiling. `AGENTS.md` asks for the
`REFINEMENT_OF_COMBAT_KERNELS.md` baseline before a pipeline refactor is tagged for physical
hardware, so §5 records, gate by gate, exactly which rungs did and did not run. The short
version: both legs compile strictly at the new default, the host gates pass, **one** CAModel
launch of the modified kernel ran and matched its FP64 oracle (§5.1 — it came out of the new
profiling script rather than out of a ladder), and the ladder's depth, wrap, ragged-batch,
canary-page and trace-audit rungs were **not run**. Read §5 before treating this tree as
hardware-ready.

## 1. What changed

| # | Change | Where |
| --- | --- | --- |
| 1 | The Cube GEMM reduction is the **default** reduction phase, on both the simulator and the physical leg | `CMakeLists.txt`, `cmake/SimTargets.cmake`, `cmake/NpuTargets.cmake` |
| 2 | The block table is prefetched once per task into a scalar cache, out of the pipeline's steady loop | `kernels/ops/turboquant/turboquant_fused_decode_opt.cpp` |
| 3 | Every exit from `ComputeTask` leaves through the task's one cross-core join | same |
| 4 | The overlay's compile-time checks, and the role contract of the default reduction phase | same |
| 5 | A hardware validation ladder, an isolated benchmark, and a single-tile `msprof` capture | `tests/npu/validation/`, `tests/npu/bench/`, `scripts/` |

## 2. The four hazards, one at a time

### 2.1 Symmetric task drain — **NOT REPRODUCED**

The brief asked for a fix to a cross-core join deadlock in the `#else` branch of
`ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE`, where the shipping vector reducer runs behind
`if ASCEND_IS_AIV` while the AIC does nothing. **That deadlock was looked for and not found.**
Three findings, and all three are checkable without a device:

1. **`DrainFusedDecodeTask()` is already a balanced two-phase rendezvous on one flag id.** Per
   `CrossCoreSetFlag(ISASI)` for Ascend 950PR (CANN 9.2, `modeId` 2): *"After the AIC core
   executes CrossCoreSetFlag, the instructions following CrossCoreWaitFlag on the two AIVs
   continue to execute. After both AIVs execute CrossCoreSetFlag, the instructions following
   CrossCoreWaitFlag on the AIC can execute."* The drain posts exactly one AIC set and one set
   from each subcore, in both directions. One set releasing two waits is the documented
   behaviour, not a lost token.
2. **The AIC announcing its join while the AIVs still reduce is the intended order.** The AIC
   sets, then blocks in its own wait until both AIVs have finished reducing *and* drained MTE3
   — which the AIV leg's `SyncEvent<HardEvent::MTE3_S>()` in front of its set is what
   guarantees.
3. **Nothing under `op.Reduce()` takes a cross-core primitive.** `TurboQuantPartialReducer`'s
   only barrier is a `PipeBarrier` inside its own `Init`; it has no `CrossCoreSetFlag`,
   `CrossCoreWaitFlag` or `SyncAll`. So the AIC's absence cannot block the AIVs either.

Recorded as unreproduced rather than quietly "fixed": a change to a sync protocol that is not
answering a real fault is how a working protocol acquires a redundant barrier that nobody
dares remove later. What was done instead is to write that contract down at the `#else` branch,
including the one property a future change must preserve — bullet 3, because a cross-core
primitive introduced anywhere under `op.Reduce()` turns the AIV-only guard into a genuine hang,
and the gate that would show it is a multi-split CAModel rung and its trace audit, not a build.

The determinism the brief asked for is real in one place, and §2.3 is that place.

### 2.2 BlockTable prefetch — **CLOSED**

`CountTiles` and `NextTile` both read `blockTableGm_` with `GetValue`, a scalar load from
Global Memory, and `NextTile` runs **inside** the steady loop, once per tile on each of the two
cursors. One logical page's entry was therefore fetched three times per task — the count on
both roles, the read cursor, the consume cursor — and two of those three fetches sat between a
softmax and the next tile's stage, on the scalar queue, inside the dependence chain the
three-zone schedule exists to keep full.

The task prologue now sweeps `[blockStart, blockEnd)` once into a 64-entry scalar cache and
both walkers read local memory: **one GM read per block instead of three, and none inside the
steady loop.** The sweep is still `GetValue` rather than a `DataCopy` into UB, deliberately —
the reduction's overlay already spends the only transient UB this operator has, and a UB
landing would make the AIC walk a `TPipe` allocation the AIV does not, which is the fault that
made the Cube GEMM reduction's first Cube run target an unallocated tensor
(`su_ccu_mpu_err_t0`). Both roles prefetch identically over the same GM words, because
`CountTiles` is the one predicate both roles branch on to decide whether to enter the pipeline.

It is a **cache, not a requirement**: a split longer than 64 blocks falls back to the per-block
direct read, exactly as the old code always did. 64 covers the validation ladder and every
benchmark sweep row except context 16384 at one split (128 blocks in one worker). The cost is
256 bytes of scalar local memory — see §5's note about what has not been checked at runtime.

Placement differs from the brief in one way worth stating: the prefetch is in the task prologue
**on both roles**, not inside `vector_.BeginTask`. `BeginTask` is vector-only and runs before
the split's block range exists, and the AIC needs the same entries.

### 2.3 Universal early-exit join — **CLOSED**

`ComputeTask` had four exits and three of them reached the join. The fourth — the duplicate
splits of a *fused* token — left by a bare `return` in front of it. That exit was symmetric
across the roles, because both cores read the same `contextLen` word, so it was not a hang; it
was the one task exit whose flag protocol differed from every other.

`ComputeTask` is now the join and nothing else; `RunTask` holds the work and has no join of its
own. Every path — the duplicate split, an unmapped split whose block range is empty, a split
whose every block-table entry is negative so `numTiles == 0`, and the ordinary pipelined task —
converges on one `DrainFusedDecodeTask()`. The cost is one mode-2 rendezvous per skipped split
of a fused token; what it buys is that the trace auditor's `dispatch == completion` count holds
over the whole task stream without a case analysis.

### 2.4 Overlay safety — **DOCUMENTED AND TIGHTENED, within what a compiler can check**

The brief asked to keep compile-time assertions verifying that `transientPool_` is *quiescent*
before `ReduceGemm()`. **A compiler cannot check quiescence**, and saying otherwise in a
`static_assert` message would be worse than saying nothing. What is checked, at the use site:

| Check | What it catches |
| --- | --- |
| `kTransientPoolBytes == kGemmSharedPoolBytes` (pre-existing) | the two headers that restate the pool's length drifting apart |
| `kGemmReducerUbBytes <= kTransientPoolBytes` (new) | the reducer outgrowing the pool the **operator** allocates, rather than its own restatement of it — so it holds even while that restatement is being changed |
| `valid_launch(..., kMaxSplits, ...)` is true, and `valid_launch(..., kMaxSplits + 1, ...)` is false (new) | the launch validator admitting a split count the reduction's partial fractal has no lane for. `valid_launch` is `constexpr`, so the bound is checked by calling it rather than by restating it |

And what is **not** checked is now written down where the aliasing happens: that
`nzOperandBuf_`, `accBuf_` and `queryWorkBuf_` are dead when the reduction starts rests on
`AscendC::SyncAll<false>()` plus the fact that no method reachable from `ReduceGemm` reads them.
A future change that reads any of the three after the attention phase breaks this **silently**,
with correct-looking output at every geometry that does not reduce, and the gate that would
catch it is a multi-split CAModel rung — §9 of the batched-reduction report — not a build.

## 3. The default flip, and what stands behind it

`TQ_GEMM_REDUCE` now defaults **ON**, and the option moved to the top-level `CMakeLists.txt`
so the simulator leg and the physical leg cannot disagree about which reduction the operator
ships. Before this change the option lived in `SimTargets.cmake` only: `NpuTargets.cmake`
compiled the same kernel source **without** the define, so the physical library had the vector
reducer no matter how the simulator leg was configured. A flip that only reached one leg would
have shipped one operator and validated another.

What the default now rests on, unchanged by this change:

* the Cube GEMM reduction's own CAModel evidence — `S ∈ {1,4,5,8}` at zero fp16 ULP, canaries
  intact, workspace byte-identical, clean trace audits (§8 of the batched-reduction report);
* the overlay running inside the operator, five rungs, `comparison=exact inputs=unchanged
  guards=0x5A-intact` (§9 there);
* the default path's own ten-rung ladder, re-run when the pool change landed (§10 there);
* a vector critical path of 4384 instructions against the shipping reducer's 13450, a factor
  of 3.07 on the busier subcore (§6 there).

What it does **not** rest on, and these are the reasons that report declined to flip it:

* **two split rungs against the default path's ten.** That has not changed.
* **no inter-wave evidence.** Every grid in the probe, in the new validation ladder and in the
  overlay rungs is inside the 32-block co-resident bound, which is also what the reduction's
  `SyncAll<false>()` requires. A production-shaped grid brings §6's inter-wave gates into scope
  and they have never been exercised.
* **no hardware.** None of it.

The fallback is one flag away and now has a gate of its own:
`scripts/gemm_reduce_switch_gate.sh` gained a third leg requiring `-DTQ_GEMM_REDUCE=OFF` to
still build, because the vector reducer is no longer on any path that is compiled by accident
and nothing else would notice it rotting.

## 4. What the new harness does

### `tests/npu/validation/test_turboquant_fused_decode_opt_npu.cpp`

36 launches over nine rungs, each a (context, tokens, KV heads, query heads per KV head)
geometry with its own split sweep. `D = 256` and 128-token pages throughout; `S` ranges over
`{1, 2, 3, 4, 5, 7, 8}` across the ladder, 1 and 4 KV heads, 16 query heads per KV head. Four
of the nine rungs are **ragged** — the context divides neither the page nor the 64-row Cube
tile. Per launch:

* **the oracle**, every emitted FP16 channel against the fixture's own scalar FP64 attention
  sum, at a budget of **zero fp16 ULP** (§8 of the batched-reduction report measured 0 on
  CAModel, which is why the budget is 0 and not 1). The residual absolute error is reported,
  not asserted — it is the fp16 quantisation of the oracle itself;
* **`0x5A5A5A5A` canaries**, required 100% intact: the 64-byte guard bands on both sides of all
  nine allocations; every physical KV page the block table does not map (the fixture
  interleaves mapped and unmapped pages, so an off-by-one in the paged addressing lands on a
  canary page rather than on another token's data); the inter-head scale padding of every
  mapped page — six of eight lanes per row at one KV head; the output tail past the last live
  channel;
* **the workspace**: at `S > 1` every live partial finite, and at `S == 1` the **whole**
  workspace still canary, which is the control for a reduction phase that runs when
  `NeedsReduction()` says it should not;
* **input immutability**, byte for byte, on all seven input allocations.

The rungs cover every path the schedule has. **Aligned, one KV head:** one page (so `S > 1`
leaves unmapped, empty workers — §2.3's exit); eight pages (the bounded short pipeline at
`S = 8`); sixteen pages (the deep three-zone path at every `S`, 32 tiles in one worker at
`S = 1`, which wraps the three-slot ring ten times, and the non-power-of-2 sweep
`S ∈ {3, 5, 7}`); sixteen pages over two tokens. **Ragged:** 79 (one page, 79 live rows — one
full tile and a 15-row tail), 257 (three pages, the last holding ONE token, so its only tile
has a single live row), and 1055 (nine pages, a 31-row tail that lands on a different worker as
`S` changes). **Multi-KV-head:** 4 KV heads × 16 query heads = 64 query heads, at 1024 aligned
(where `S = 8` dispatches a grid of exactly 32 blocks, the co-resident bound hit exactly) and at
1055 ragged with `S ∈ {3, 7}`.

Three things about the expanded set are measured rather than assumed, and two of them corrected
what the rung comments first claimed:

* **A non-power-of-2 split count is not simply an uneven partition.** `blocksPerSplit` is
  `ceil(seqBlocks / S)`, so 16 pages over 5 workers gives four workers 8 tiles each and leaves
  the **fifth empty**; over 7 it gives five workers 6 tiles, one worker 2 and one nothing. An
  empty worker still publishes a partial — `BeginTask` zeroes the accumulator and sets `runMax`
  to `-inf`, and `StageTaskOutput` writes that neutral element — so these rungs make the
  reduction combine `S` partials of which some carry `l = 0`, with its row stride equal to
  `numSplits_` rather than to a power of two.
* **The 1055 tail lands on worker 2 at `S = 3` and on worker 4 at `S = 5` and `S = 7`**, so the
  short tile belongs to a worker that is neither the first nor the last.
* **At 4 KV heads the inter-head scale padding canary is vacuous by construction.** The codec
  reads lanes `[head]` and `[4 + head]` of eight, so all eight are live and there is no padding
  lane left to guard. The ladder reports that rather than asserting it, and requires at least
  one rung elsewhere to have had padding to guard.

What the ladder does **not** canary is the TAIL ROWS of a mapped page. A masked lane's scale
still flows through the `Gather` and the `Muls` in front of the mask, so a `0x5A5A5A5A` scale
there (~3.8e15) would be testing the masking phase's numerical headroom rather than its logic,
and a failure could not be told apart from one. Those rows carry ordinary fixture data — which
the oracle already rejects if they are counted, because a tail row shares its tile's key level
and value levels, so including it weights that tile by 64 rows instead of by its true count.

### The shared fixture, and why the aligned numbers did not move

The hardware ladder, the benchmark and the benchmark's simulator twin all build their geometry
from `tests/common/fused_decode_opt_fixture.hpp`, which required `context % page_tokens == 0`
and whose oracle summed one term per whole 64-row tile. Ragged contexts needed both relaxed:
`pages` is now the ceiling, and each tile contributes `live_rows * exp(score)` to both sums
instead of `exp(score)`. All 64 rows of a tile share one key level, so that is the correct
weighting, and it is what the CAModel runner's own independent oracle already did
(`numerator += valid * weight * ...`).

The risk in that change is to the ALIGNED rungs, whose numbers are the baseline for everything
already measured. At `live_rows = 64` both sums are scaled by exactly 64, a power of two, so
the quotient cannot move a single bit — but "cannot move" and "did not move" are different
claims about floating point, so `tests/cpu/test_fused_decode_opt_fixture.cpp` now recomputes the
whole aligned oracle with the OLD expression and compares it **bitwise**, over every channel of
every query head of every token of a 192-token, 2-token, 2-KV-head, 3-group fixture. It also
recomputes the ragged oracle along a different rounding path — normalising the weights first and
accumulating a convex combination — and requires the 15-row tail to actually change the answer
at more than 200 of 256 channels, which it would not if the weighting were dropped.

The benchmark's `--context` accepts a ragged value on the fused-opt path only, since the legacy
decoders' fixture still requires alignment, and its zone decomposition counts
`ceil(live rows / 64)` per page rather than assuming whole pages.

### The query basis: pre-rotate on/off, and the post-rotate stage

The operator has two entry points. The pre-rotated one takes `queryRot` already in the rotated
basis; the RAW-QUERY one takes the query in the original basis plus the basis tables, rotates it
once in a prologue, and picks an output stage for the egress. **Neither the raw-query entry nor
the output stage had ever been called anywhere in this repository** — no test, no probe, no
benchmark, and no launcher in the convenience ABI, because `launch_fused_decode_opt` only ever
called the pre-rotated kernel. So `--pre-rotate off` was not a flag over an existing path; it
needed that path brought online first.

**The basis, read off the kernel rather than assumed.** `TurboQuantCodec4::ApplyPi` is
`x *= piSigns; FWHT(x); x *= piSigns`, and the FWHT applies `Muls(invSqrtLen)` once at the end,
so `Pi = diag(s) . H . diag(s) / sqrt(D)` with `H` the unnormalised Walsh-Hadamard. `H` is
symmetric and `H^2 = D.I`, which makes **Pi symmetric and Pi^2 = I exactly**.

**The table pool.** `rotTables` is the codec's whole constant pool, and nothing here had ever
built one. `Init` carves it as three early-stage `sign_`/`xorOffset_` pairs followed by the
quantiser's tables, which the word count pins exactly: `7*256 + 2*256 + 16 = 2320`. `ApplyPi`
reads only the early stages, and `Gather(swap_, x, xorOffset_[stage], 0, n)` followed by
`Add(x, swap_, tmp)` forces the contents — `sign[i] = +1` iff `(i & stride) == 0`, and the
gather index is a BYTE offset to `i ^ stride`, `kGatherSrcBase` being 0 and the vector service's
own gather indices being byte offsets too. The quantiser's part of the pool is left zero, which
is also an in-bounds offset.

**Why the oracle survives the mode switch.** Because Pi is an exact involution, the raw query
whose rotation is the fixture's pre-rotated query is simply `Pi(query)` — and every entry of it
is fp16-exact. So output stage 0 is held to the SAME attention oracle as the pre-rotated entry
rather than to a looser one, and `tests/cpu/test_fused_decode_opt_fixture.cpp` pins that:
`Pi(raw) == query` **bitwise**, channel for channel, plus Pi's involution and orthogonality and
the 2320-word pool length. Stages 1 and 2 compare against `Pi(context)` and
`Pi(context) / (1 + exp(-gate))` computed on the host.

The rig also validates the **prologue by itself**: in raw-query mode the query buffer is an
OUTPUT, so `Validate` compares what the kernel wrote against the host's own Pi image before it
looks at the attention at all, which separates a prologue fault from a decode fault.

**Not established.** The raw-query path had not been executed when this was written — a CAModel
sweep of the control, the three stages, the Cube-chunk prologue and the CLI guards was in flight.
And the ULP budget for stages 1 and 2 (`kUlpBudgetRestaged = 8`) is a **first-run placeholder,
not a measurement**: the rig prints the worst ULP it actually saw, and that number is what
should replace it.

### The benchmark, isolated

`bench_turboquant_decode_splitk` now contains **one kernel**: the fused operator. The legacy
AIV and AIC decoders are compiled out and their libraries are not linked, which is what makes
an `msprof` capture of this binary a capture of the operator rather than of a sweep that
happens to contain it. Defaults became the operator's own sweep: contexts 1024…16384,
`K ∈ {1,2,4,8}`, `--heads 1 --rows 16` (16 query heads), 128-token pages.
`-DTQ_BENCH_LEGACY_DECODERS=ON` restores the three-kernel comparison and the modeled crossover
exactly as they were.

**The simulator twin keeps all three kernels.** `bench_decode_splitk_sim` is what
`scripts/profile_camodel_trace.sh`, `scripts/capture_fused_decode_opt_camodel_json.sh` and
`scripts/check_decode_splitk_bench_cli.sh` drive with `--impl aiv`, `aic` and `all`, and those
are the existing CAModel evidence paths for the *other* two kernels. Isolating that leg too
would have made an `msprof` capture of the AIC decoder unreproducible, which is not what
isolating the hardware benchmark was for.

### `scripts/profile_single_tile_msprof.sh`

One launch under `msprof op simulator`, exported as the toolkit's own Chrome tracing JSON
(`msprof_trace.json`, validated by `scripts/validate_camodel_trace.py` and loadable in Perfetto
or `chrome://tracing`). Three presets, because one tile cannot show everything:

| preset | geometry | what it shows |
| --- | --- | --- |
| `tile` (default) | context 64, 64-token pages, `S = 1` | exactly **one** tile: one score GEMM, one context GEMM, 16 query heads. The bounded short pipeline — one tile's dependence chain end to end, and **no** steady-state overlap, because at one tile there is no steady state |
| `steady` | context 256, `S = 1` | 4 tiles, the minimum deep path: three-tile prologue, exactly one unrolled steady phase, bounded three-step drain. This is the preset for the overlap question |
| `reduce` | context 128, `S = 2` | two Split-K workers of one tile each, so the **reduction phase** runs — with the new default, the Cube GEMM reduction out of the aliased UB pool, which the other two presets never reach. This is the preset that was actually executed for this change: §5.1 |

It makes no overlap or speedup claim: simulator ticks are not hardware time, and a measured
overlap claim needs intersecting per-engine intervals (`scripts/export_fused_decode_gantt.py`).

### Scripts cleanup

* `probe_bufpool_api.sh`, `probe_cube_api.sh`, `probe_tpipe_reset.sh` → consolidated into
  `scripts/probe_ascendc_headers.sh {bufpool|cube|tpipe-reset|all}`, one topic each, greps
  verbatim. They were single-question header probes from the overlay work; the questions and
  the exact greps that answered them are worth keeping, three throwaway files are not.
* `wire_gemm_reduce_cmake.py` → **deleted**. It was a one-shot generator that inserted the
  `TQ_GEMM_REDUCE` option into `SimTargets.cmake`; its output is committed, has since been
  edited, and the option has moved, so re-running it now would be wrong.
* `gemm_reduce_switch_gate.sh` → **retained** as the automated sanity check, plus the new OFF
  leg (§3).
* `gemm_reduce_container.sh` → its hardcoded `/mnt/d/Projects/ascend-micro-harness` default
  silently synced the wrong tree (this checkout is elsewhere); it now derives the checkout from
  the script's own location. It also notes that CANN 9.2 is reachable directly in the
  Ubuntu-22.04 WSL instance on this host, so the device gates no longer need the container.

## 5. Gates: run, not run, unavailable

This is the section `AGENTS.md` asks for, and the honest summary is at the bottom of it.

### Run on this tree

Environment: CANN 9.2.0-beta.2 in the Ubuntu-22.04 WSL instance
(`/usr/local/Ascend/cann-9.2.0-beta.2`), `SOC_VERSION=Ascend950PR_9599`, no NPU driver.

| Gate | Result |
| --- | --- |
| Host contract gate, `-Wall -Wextra -Werror` (`tests/cpu/test_fused_decode_opt_contract.cpp`) | **PASS** — launch guards, NZ bijection, shared recurrence phases, `UB=142304+256 bytes`. `kUbEnd` is **unchanged** by the block-table cache, which is the independent check that the cache is scalar local memory and not UB |
| Host geometry gate (`tests/cpu/test_gemm_reduce_geometry.cpp`), `-Wall -Wextra -Werror` | **PASS** — workspace addressing, Zn operand, id budget, task slicing, reference combine, block-diagonal containment, subcore split, batch formation, batched ingest strides, batched budgets |
| Host fixture gate (`tests/cpu/test_fused_decode_opt_fixture.cpp`) | **PASS** — fragmented pages, independent NZ inverse, GQA/request variation, exact one-tile oracle, capped grid |
| New validation ladder, host compile, `-Wall -Wextra -Werror` | **PASS** |
| New validation ladder's **geometry**, on the host: all 36 launches of all 9 rungs | **PASS** — every launch builds a fixture and a `valid_launch`, every grid is 1..32 blocks and so inside the 32-block one-wave bound the reduction's `SyncAll<false>()` needs, and every rung has unmapped physical KV pages to guard (3 at one page, 34 at sixteen pages over two tokens). It also reconstructs the kernel's own partition arithmetic per worker and requires the per-worker tile counts to **sum to the fixture's own tile count**, so a ragged partition neither drops a live tile nor counts one twice, and no worker is handed a page with no live rows |
| Extended host fixture gate: ragged arithmetic, **bit-identical aligned oracle**, independently recomputed ragged oracle | **PASS** — the aligned expected values are bitwise unchanged by the ragged support, channel for channel; the 79-token oracle agrees with a convex-combination recomputation to within 1e-5; and the 15-row tail changes more than 200 of 256 channels against a full second tile, which it would not if the row weighting were dropped |
| Isolated benchmark, host compile, `-Wall -Wextra -Werror` | **PASS** |
| Benchmark with `-DTQ_BENCH_LEGACY_DECODERS=1`, host compile, `-Wall -Wextra -Werror` | **PASS** — the switch restores the three-kernel driver without touching the source |
| Device compile, **physical** leg, `-Wall -Werror`, at the new default (`turboquant_fused_decode_opt_npu`) | **PASS** — AIC, AIV and preprocess legs, 21m45s. `-DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1` is present in that target's own `flags.make` for both `aic_obj` and `aiv_obj`, which is the check that the define actually reached the physical leg and not only the simulator one |
| Device compile **and link and run**, simulator leg (`camodel_fused_decode_opt_kernel` through `bench_decode_splitk_sim`) | **PASS** — §5.1 |
| One CAModel launch of the modified kernel, Cube GEMM reduction active, against the FP64 attention oracle | **PASS** — §5.1 |
| `scripts/gemm_reduce_switch_gate.sh`, three legs | **PASS, all three** — §5.2 |

### 5.1 The one run: `scripts/profile_single_tile_msprof.sh reduce`

Evidence: [`reports/single_tile_msprof_reduce/`](single_tile_msprof_reduce/).

`context 128, 64-token pages, S = 2, 16 query heads over one KV head, D = 256, grid 2`, one
launch under `msprof op simulator`, `TQ_GEMM_REDUCE=ON`. It is not a ladder rung and does not
substitute for one, but it is the only execution of the modified kernel anywhere in this change,
so what it does and does not show matters:

* **the operator's own FP64 attention oracle matched** — `PASS: CPU references matched. Kernel
  launches: AIV=0 AIC=0 OPT=1.` The benchmark validates every launch before printing that, and
  `AIV=0 AIC=0` is the isolation of §4 observed rather than asserted;
* **the Cube GEMM reduction ran**, out of the aliased UB pool, by default, with no compile flag;
* **both behavioural changes executed**: the block-table prefetch on both roles, and the
  restructured `ComputeTask`/`RunTask` on 2 real Split-K workers. The operator object carries
  its new 256-byte scalar member on a real model run without a stack fault;
* `[INFO] Total tick: 40420`, 68516 trace events over a 22.4 µs model span, exported and
  schema-validated (`scripts/validate_camodel_trace.py`: `status PASS`).

The per-core pipe census, from the trace's own tid lanes:

| core | SCALAR | CUBE | FIXP | MTE1 | MTE2 | MTE3 | RVECEX | RVECLD | RVECST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `core0.cubecore0` | 5304 | **18** | 36 | 24 | 66 | — | — | — | — |
| `core0.veccore0` | 7136 | — | — | — | 6 | 60 | 2528 | 1654 | 1610 |
| `core0.veccore1` | 7136 | — | — | — | 6 | 60 | 2528 | 1654 | 1610 |
| `core1.cubecore0` | 5305 | **18** | 36 | 24 | 66 | — | — | — | — |
| `core1.veccore0` | 7137 | — | — | — | 6 | 60 | 2528 | 1654 | 1610 |
| `core1.veccore1` | 7137 | — | — | — | 6 | 60 | 2528 | 1654 | 1610 |

Two things in that table are worth naming, and both are *consistent with* the design rather than
proof of it:

1. **The two vector subcores are byte-for-byte symmetric** on every lane. That is the signature
   the batched reduction was supposed to produce — §6 of the batched-reduction report measured
   `veccore1` going from 1186 instructions and **zero** vector ops under the shipping reducer to
   a byte-for-byte match with `veccore0` under the dense batched form. Here it is symmetric at
   both blocks.
2. **18 Cube operations per block**, where the attention phase of a one-tile worker accounts for
   two (one score GEMM, one context GEMM). The remaining 16 is the slice count of the batched
   reduction — `D / 16 = 16` dense `Mmad`s over one batch of rows. The arithmetic matches; the
   trace does not label which instruction belongs to which phase, so this is an inference from
   counts, not a per-instruction attribution.

**No timing or overlap claim.** Model ticks are not hardware time, the `OPT` columns of the
benchmark row are `NA` because the simulator reports no device-event timing, and this preset's
workers are one tile deep, which is below the steady loop entirely.

### 5.2a The expanded ladder's geometries, on CAModel

Nine launches, all **PASS**. Six went through the benchmark's simulator twin, which builds from
the SHARED fixture — so these are the new row-weighted ragged oracle checked against the kernel,
not against itself — and three through the CAModel probe, whose oracle is written independently
of the fixture and which reports canaries and input immutability as well:

| rung | geometry | result |
| --- | --- | --- |
| `ragged-79-s1` | `N = 79`, one page, 2 tiles, a 15-row tail, `S = 1` | **PASS** — oracle matched, `AIV=0 AIC=0 OPT=1` |
| `ragged-257-s2` | `N = 257`, three pages, the last holding ONE token, `S = 2` | **PASS** — the worker owning page 2 has a single live row in its only tile |
| `ragged-79-s3` | `N = 79`, `S = 3`: a ragged worker and two empty ones | **PASS** — ragged tail, empty workers and a non-power-of-2 split count together |
| `npo2-384-s3` | `N = 384`, three pages, `S = 3` | **PASS** — the aligned non-power-of-2 partition |
| `kvh4-128-s1` | 4 KV heads, 64 query heads, `S = 1` | **PASS** |
| `aligned-128-s1` | the aligned control | **PASS** |
| `probe-tail-15` | probe, `--tiles 2 --tail-rows 15` | **PASS** — `comparison=exact inputs=unchanged guards=0x5A-intact` |
| `probe-tail-1` | probe, `--tiles 5 --tail-rows 1 --splits 2` | **PASS** — same, at a one-row tail with Split-K |
| `probe-aligned` | probe, `--tiles 2 --tail-rows 64` | **PASS** — the probe's own control |

The two sets matter for different reasons. The benchmark rungs are the only check that the
fixture's new oracle agrees with the kernel; the probe rungs are the only ones here that also
assert canaries and byte-for-byte input immutability, and their oracle shares no code with the
fixture's, so `comparison=exact` on a 15-row and a 1-row tail is independent corroboration that
the masking is right rather than that two copies of one formula agree.

What this still does not cover: these are CAModel launches at small geometries, not the hardware
ladder, and they do not touch the deep three-zone path with a ragged tail (every rung above is
1–6 tiles per worker). §5's summary stands.

### 5.2 The switch gate, three legs

Run against this tree, in WSL rather than in the container — which is itself part of what this
change fixed, since the gate used to `source` a container-only `set_env.sh` and default its
build root to `/work`:

| leg | result |
| --- | --- |
| 1. `-DTQ_GEMM_REDUCE=ON` must build at `-Wall -Werror` | **PASS** — the operator builds with the Cube GEMM reduction driving its reduction phase |
| 2. negative control: the pool cut to 16384 must be rejected | **PASS** — `static assertion failed due to requirement 'kGemmReducerUbBytes <= kGemmSharedPoolBytes'`, which is the reducer's own fit assert and not another |
| 3. **new:** `-DTQ_GEMM_REDUCE=OFF` must still build | **PASS** — the operator still builds with the shipping vector reducer |

Leg 2 is the one that needed care. The control cuts `kGemmSharedPoolBytes` and relaxes the
restatement pin, and §2.4's new use-site assert is pinned against `kTransientPoolBytes` rather
than against that restatement precisely so that it does **not** fire in the control and bury
the assert the control exists to observe. Its output above is the evidence that it does not:
the message that came back is the reducer's.

Leg 3 is the reason the default flip is reversible rather than nominally reversible. Leg 1 is
also established independently — §5.1 built *and ran* the `ON` tree.

### Not run, by instruction

The CAModel ladder suites were excluded from this change in favour of moving to hardware. So:

| Gate the refinement protocol asks for | Status |
| --- | --- |
| §1 capacity-one token/queue model | **NOT RE-RUN.** Its last full pass predates the per-worker ring fix, as the batched-reduction report already recorded |
| §3 isolated logical-page head/middle/tail canaries on CAModel | **NOT RUN on this tree.** The new hardware ladder implements exactly these checks and has never executed |
| §4 depth and wrap, split-count sweep, real and empty Split-K workers | **NOT RUN on this tree** |
| §5 trace audit and instruction census | **NOT RUN on this tree** |
| The operator's own default-path ladder (`scripts/run_fused_decode_opt_camodel.sh ladder`) | **NOT RUN on this tree** |

### Unavailable

| Gate | Why |
| --- | --- |
| Every rung of `tests/npu/validation/test_turboquant_fused_decode_opt_npu.cpp` | **no NPU on this host**, and on this host it does not even LINK: the kernel library is linked against `libascend_dump.so`, which needs `libascend_hal.so` from the NPU driver, and that library is absent from this CANN install. The ladder's own translation unit compiles clean at `-Wall -Wextra -Werror`; the failure is in CANN's libraries, below it. **This is pre-existing and not a property of the new ladder**: `bench_turboquant_decode_splitk`, which predates this change, fails to link on this host with the identical undefined `drvHdc*`/`hal*` references. So for the physical leg, "a clean build" means the device compile of §5's table, not a linked executable |
| `scripts/profile_single_tile_msprof.sh` on hardware | same. It is written for `msprof op simulator`, which needs no device; a hardware capture is `msprof op` against a real 950PR and has not been attempted |
| §6 inter-wave gates | out of scope by construction: the reduction's `SyncAll<false>()` requires a single wave, and `valid_launch` caps the grid at 32 blocks |

### The honest summary

**Two of the five changes are behavioural, and they have exactly one run between them.** §5.1
executed both — the block-table prefetch on both roles and the restructured task exit, on two
real Split-K workers, with the Cube GEMM reduction active — and the operator matched its FP64
oracle. That is one geometry. What it does **not** cover:

* **the deep three-zone path.** Its workers are one tile each, so the steady loop, the ragged
  residues and the drain never ran with the new block-table cache. `NextTile` reading local
  memory inside the steady loop is the whole point of §2.2 and it has not been executed there.
  The `steady` preset of the profiling script is the four-tile version and is one command.
* **the new join path.** §2.3's added rendezvous fires only for a duplicate split of a *fused*
  token, and the fixture sets `fused_context_limit = 0`, so no token is ever fused. The join
  change ran in its restructured form; its new edge did not. `run_fused_decode_opt_camodel.sh`'s
  `split-empty` rung and the new ladder's one-page rung are what reach it.
* **the cache's fallback**, at a split longer than 64 blocks.
* **canaries, depth, wrap, ragged batches, two-worker contention, and the trace audit** — all of
  §5's "not run by instruction" list. A benchmark comparing its result against an oracle is not
  a canary check: nothing in §5.1 would have noticed a write outside a buffer that the oracle
  does not read.
* **the ladder's new geometries on HARDWARE.** Ragged contexts, non-power-of-2 splits and 4 KV
  heads now have CAModel evidence at small geometries — §5.2a, nine launches, all passing,
  including a probe whose oracle is independent of the fixture's — but none of it is a hardware
  run, and none of it reaches the deep three-zone path with a ragged tail.

So: the tree compiles strictly on both legs at the new default, the switch is reversible and
gated in both directions, and the one run that exists says the operator is still correct at one
small geometry. It is **not** a completed refinement baseline, and the ladder is one command.

## 6. Commands

```bash
# The host gates, which are the whole of the evidence on a machine with no toolkit.
g++ -std=c++17 -O2 -Wall -Wextra -Werror tests/cpu/test_fused_decode_opt_contract.cpp -o /tmp/c && /tmp/c
```

```bash
# Both legs of the operator, at the new default, strict.
cmake -S . -B build-hw -DCMAKE_BUILD_TYPE=Release
cmake --build build-hw --target turboquant_fused_decode_opt_npu test_npu_fused_decode_opt bench_turboquant_decode_splitk -j8
```

```bash
# The retained sanity check: ON builds, a short pool is rejected, OFF still builds.
bash scripts/gemm_reduce_switch_gate.sh
```

```bash
# The single-tile Chrome trace. tile | steady | reduce.
bash scripts/profile_single_tile_msprof.sh tile
```

```bash
# The hardware ladder. Needs a 950PR; it has never run.
./build-hw/lib/test_npu_fused_decode_opt
```

```bash
# The isolated benchmark's own sweep, on hardware.
./build-hw/lib/bench_turboquant_decode_splitk --zone-report
```

```bash
# What this change did NOT run, and what to run first if it is going to silicon.
bash scripts/run_fused_decode_opt_camodel.sh ladder
```

```bash
# The four-tile preset, which is the one that exercises the steady loop with the new cache.
bash scripts/profile_single_tile_msprof.sh steady
```

```bash
# A ragged context on CAModel, through the shared fixture and its new row-weighted oracle.
./build-sim/lib/bench_decode_splitk_sim --impl fused-opt --single --context 79 --page-tokens 128 --heads 1 --rows 16 --requests 1 --splitk 1
```

```bash
# The same tail through the CAModel probe, whose oracle is independent of the fixture's.
./build-sim/lib/test_camodel_fused_decode_opt --tiles 2 --tail-rows 15
```

## 7. Limitations

* **CAModel at best.** No NPU on this host, and no NPU-leg executable can even be linked here
  (§5's unavailable table). Exactly one CAModel launch was run for this change, at one small
  geometry; §5.1 is what it shows and §5's summary is what it does not.
* **The deadlock in the brief was not reproduced** (§2.1). If one is ever observed, the thing to
  change is the `if ASCEND_IS_AIV` guard in the `#else` branch, and the evidence has to be a
  trace, not a build.
* **The overlay's safety still rests on a liveness argument, not a checker** (§2.4). The new
  assertions bound extents; nothing verifies that the three aliased buffers are dead.
* **The 64-entry block-table cache is a cache.** Context 16384 at one split falls back to the
  per-block GM read on blocks 64..127 of its single worker, which is the old behaviour and not a
  regression — but it means the benchmark's largest row does not exercise the new path fully.
* **The default flip inherits the batched reduction's evidence, not more of it.** Two split
  rungs, one wave, no hardware (§3).
* **No overlap or timing claim anywhere**, here or in the profiling script.
