# The Split-K final reduction as a Cube GEMV: prototype and CAModel evidence

Date: 2026-11-04. Branch `claude/gemm-final-reduction-0f1afc` (exploratory, not merged, not
pushed). Target CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599`, BiSheng.

**There is no NPU on this host.** Every device result below is the `Ascend950PR_9599` CAModel
simulator. Nothing here is a hardware claim, and simulator ticks are not silicon throughput.

## 1. What this is

`turboquant_fused_decode_opt.cpp` runs the FlashDecoding combine over `S` splits as a separate
serialized pass on the Vector engine:

```
op.Process(tasksPerBlock);
if (op.NeedsReduction()) { AscendC::SyncAll<false>(); if ASCEND_IS_AIV { ... op.Reduce(...) } }
```

That `if ASCEND_IS_AIV` is the whole motivation: the Cube is **idle for the entire reduction
phase** while the busier engine does every rescale. This prototype makes the combine an
`M=1, K=S, N=headSize` matmul on the Cube — one `Mmad` — pipelined over a two-slot ring, with
only the statistics merge and the final normalise left on Vector.

The arithmetic. Split `s` wrote an unnormalised accumulator `o_s` plus its own `m_s` and `l_s`
into `workspace[((token*numHeads + head)*numSplits + s) * 272]`. The merged head is

```
m*     = max_s m_s
w_s    = exp(m_s - m*)
out[d] = (sum_s w_s o_s[d]) / (sum_s w_s l_s)
```

so the numerator is exactly the GEMV. Note `w_s = exp(m_s - m*)` and **not**
`exp(m_s - m*) * l_s`: these partials are unnormalised, so `l_s` belongs in the denominator
only. The host gate pins that choice with a fixture on which the two conventions differ.

## 2. The headline result

**The GEMM form is the right shape for this geometry.** The `K = S` operand was the stated
design risk and it turns out to be clean rather than marginal:

* A Cube C0 block is 32 bytes, so a **b32 operand carries 8 elements along K**, and the
  decode's split count is capped at `kMaxSequenceSplits = 8`. So `K = S` lands inside **one C0
  block for every legal split count** — one `Mmad`, no K tiling — and at `S = 8` the L1 cube is
  exactly the 8 KB of partials with nothing padded away.
* **No transposing load is needed.** `DataCopy` GM→L1 with `Dn2NzParams` performs an inline
  DN→NZ conversion, which the API reference states is "equivalent to the ND-to-ZN format
  conversion". Feeding the workspace's `[S, D]` rows therefore lands the **ZN image of that
  same K×N matrix** in L1 with the split axis as C0, and a non-transposed `Load2DV2` keeps it
  (Zn2Zn). The operand takes the same load the score GEMM's N×K key tile already takes.
* That matters because the transposed b32 path is the one that would **not** work: it merges
  two consecutive 16×8 fractals into a 16×16 square and so needs a source Zn cube with
  `Col == 16`, which a `K = S ≤ 8` operand cannot provide, and it additionally constrains
  `kStep` to a multiple of 2. Avoiding the transpose avoids all of it.
* `float × float → float` Mmad is supported on Ascend 950PR, so the fp32 partials need no
  down-cast.
* `M = 1` is `Mmad`'s own GEMV case, auto-enabled, and 950PR is the one product that can switch
  it off (`MmadParams::disableGemv`). Both legs are shipped; see §6.

Measured outcome at 4 reduction tasks, `splits = 8`, both runs audited clean:

| | vector-engine instructions | Cube | Fixpipe | MTE1 | simulator ticks |
| --- | ---: | ---: | ---: | ---: | ---: |
| shipping `TurboQuantPartialReducer` | 2861 + 2861 = **5722** | 6 | 1 | 6 | 19825 |
| this GEMV stage | 851 + 1 = **852** | 18 | 17 | 24 | 11583 |

A **6.7× reduction in vector-engine instructions** for the combine, paid for with 18 Cube ops,
17 Fixpipes and 24 MTE1 loads on an engine that was doing nothing. Ticks are reported because
the runs are there; they are a simulator observation, not a throughput claim, and the tick
column is not an overlap measurement — no intersecting per-engine interval export was made for
this stage.

Caveat on the instruction comparison: the vector reducer splits its tasks between the two
subcores (2 each here), while this prototype puts all four on subcore 0. Per task the
vector-engine work drops from ~1430 to ~213; on the critical path it drops ~3.4×, not 6.7×.
That asymmetry is also the subject of the one failing gate in §5.

## 3. Protocol gates

`REFINEMENT_OF_COMBAT_KERNELS.md`, each gate recorded as met, failed or unavailable.

| Gate | Status | Evidence |
| --- | --- | --- |
| §1 capacity-one token/queue model | **MET** | `tests/cpu/test_gemm_reduce_model.py`: R = 0..10, 6 queues, every reachable completion interleaving, 9 negative controls each rejected with its own diagnosis |
| §1a outer loop modelled | **MET** | two consecutive Split-K workers with the marking carried across, plus `(4,0)`, `(0,4)`, `(1,0,5)`, `(7,1)` |
| §2 compile-time memory checks | **MET** | 20 `static_assert`s in `include/turboquant_gemm_reduce.hpp` + 8 in the service: L1/L0A/L0B/L0C/UB budgets, 32-byte bases, AKA3003 bank skew, Brcb destination extents, flag-id budget |
| §2 strict build | **MET, with one exception recorded** | device legs and runner at `-Wall -Werror` for `Ascend950PR_9599`; host contract gate at `-Wall -Wextra -Werror`. `-Wextra` on the **device** leg is blocked by a pre-existing unused parameter in production's `TurboQuantModeName` (`kernels/ops/turboquant/fused_decode/common/turboquant_mode.h:297`), which is not this prototype's to fix |
| §3 isolated boundary canaries | **MET** | every rung floods all GM with `0x7FC0BEEF`, assigns only the slices it means to, and checks both exterior guards, every unassigned output slot at its own halfword position, and that the workspace is **byte-identical** afterwards |
| §3 head / middle / tail isolation | **MET** | `gemm-mixed`: tokens 0, 2, 4 combined and 1, 3 left alone, so the stage runs at the head, the middle and the tail of the workspace in one launch and each skipped token's output slice is a canary for a task-slicing error |
| §3 exact bitwise fixture | **MET** | the exact fixture's budget is **0 fp16 ULP** and is never relaxed; the varied fixture is a separate regression on its own budget |
| §4 depth and wrap | **MET** | `kReduceSlots = 2`; rungs at 1, 2, 3, 4, 5, 6, 8 and 10 live tasks |
| §4 split-count sweep | **MET** | `splits` = 1, 4, 5, 8 — 1, 4 and 5 are the K-padding cases |
| §4 real and empty Split-K workers | **MET** | `gemm-workers-2`, `gemm-empty-worker`, `gemm-mixed-workers` |
| §5 trace audit, GEMV-off (default) leg | **MET at 13 of 14 rungs, FAILED at 1** | see §5 |
| §5 trace audit, GEMV-on leg | **FAILED at all 3 rungs** | see §5 |
| §5 instruction census | **MET** | §2 table, from each run's own `instr_popped_log.dump` |
| §5 intersecting-interval overlap claim | **NOT MADE** | no Gantt export was run for this stage, so no overlap is claimed |
| Numerical comparison vs `TurboQuantPartialReducer` | **MET** | §4 |
| §6 inter-wave gates | **NOT APPLICABLE / UNAVAILABLE** | the probe runs grids of 1–2 blocks, far below the physical block count, so no second wave exists to test. A production-shaped grid would bring §6 into scope and it has **not** been exercised |
| Physical NPU | **UNAVAILABLE** | no NPU on this host |

## 4. Numerical comparison against the shipping reducer

Both legs are compiled into the **same kernel** from the production header, so they cannot drift
apart between probe and operator: `turboquant_vector_reduce_probe` instantiates
`TurboQuantPartialReducer<half>` and applies the same subcore halving production's `Reduce()`
applies. Both legs run over the same workspace and are scored against a double-precision
reference that includes the same `1/(l* + eps)` the shared `NormalizeHeads` applies.

Result: on **every** rung measured, both legs export the correctly-rounded fp16 of the
double-precision combine at **every** channel — worst fp16 ULP 0 over 1024–2048 channels per
rung, on both the exact and the varied fixture.

| rung | leg | worst abs | worst rel | worst fp16 ULP |
| --- | --- | ---: | ---: | ---: |
| 4 tasks, S=8, exact | vector | 4.438920e-04 | 4.245924e-04 | 0 |
| 4 tasks, S=8, exact | gemm | 4.438920e-04 | 4.245924e-04 | 0 |
| 4 tasks, S=8, varied | vector | 4.879390e-04 | 4.735659e-04 | 0 |
| 4 tasks, S=8, varied | gemm | 4.879390e-04 | 4.735659e-04 | 0 |
| 8 tasks, S=8, varied | gemm | 4.879667e-04 | 4.735659e-04 | 0 |
| 4 tasks, S=1, exact | gemm | 0 | 0 | 0 |
| 4 tasks, S=5, exact | gemm | 0 | 0 | 0 |

Since ULP 0 means every channel equals the correctly-rounded oracle exactly, the two forms are
**bit-identical at the operator's output precision**. The residual `worst abs` is the fp16
quantisation of the oracle itself, not a disagreement.

What this does **not** establish: it bounds the two forms' fp32 accumulator difference to below
half an fp16 ULP without measuring it. The GEMV sums `S` products inside one Cube accumulator
while the recurrence rescales a running sum `S-1` times, so the fp32 intermediates almost
certainly differ; at `S ≤ 8` and this dynamic range the difference does not survive the fp16
export. A larger `S`, a wider output type, or an fp32 export path would be needed to measure it,
and none of those is in scope here.

## 5. The one failing gate, and what it is

`scripts/analyze_fused_decode_opt_trace.py` — the shared auditor; it is not specific to the
attention schedule's ids — passes on 13 of the 14 default-leg rungs and fails on
`gemm-mixed-workers`, and on **all three** GEMV-on rungs, with the same violation:

```
core0.veccore1 waits id m2:2 on SCALAR: stalls [3618, 51] -- the first wait blocked for 3618
ticks and the rest for at most 51, while this id's announcements are ~3567 ticks apart. The
later waits were not released by an announcement; this channel does not order its consumer.
```

This is the §6.6 class of fault, and it behaves exactly as §6.6 warns: **the arithmetic is
bit-exact, every canary is intact, and every per-id count balances.** Only the tick order shows
it.

The cause is a design decision in this prototype, not the GEMM form. `M = 1` is a single Fixpipe
row, so there is nothing for a dual-destination split to divide, and subcore 0 owns all the
combines. But `CrossCoreSetFlag<2>` is a **block-wide** notification: the Cube's one
announcement posts a token to *both* vector subcores, so subcore 1 must consume one per task
even though it carries no data. The token model already forced two concessions here — an AIV1
that never waits leaves a flag set, and one that drains passively does not pace the Cube — so
AIV1 mirrors subcore 0's flag schedule and the Cube waits once per subcore. That keeps
capacity-one intact by construction: the Cube's next announcement on slot `s` requires AIV1's
`weights_ready` for that slot, which AIV1 posts only after consuming the previous announcement,
so no token can be lost.

What it does **not** keep is the channel actually ordering AIV1. With no work of its own, AIV1
reaches its next wait while an announcement is already outstanding, and the audit's stall
pattern records that the wait was not released by an announcement. Whether that is a genuine
early release or a legitimately prefetched token is **not** something this evidence
distinguishes, and per §6.6 the correct treatment is to record it as a failing gate rather than
argue it away or relax the check. Three observations that constrain it:

* It is **timing-sensitive**: it fires at `gemm-mixed-workers` (3 live tasks per block over
  2 blocks) and at every GEMV-on rung, and not at `gemm-depth-4`, `gemm-wrap-8`, `gemm-odd-5`
  or `gemm-varied`. A fault that appears and disappears with geometry is not a fixture artefact.
* The **reference leg passes at the geometry where the GEMM leg fails**
  (`vector-mixed-workers`). Production's `Reduce()` gives both subcores real work, so both are
  genuinely paced. That isolates the fault to the idle-subcore design and rules out the harness
  and the geometry.
* The instruction census confirms the premise: 706 instructions on veccore1 against 3802 on
  veccore0, with **one** vector op.

The fix is a design change, with its own verification burden, and is deliberately **not**
attempted here:

1. Give subcore 1 half the combines, so both are paced by their own work. With `M = 1` this
   needs the Cube to direct each product at one sub-block (`FixpipeParamsC310::subBlockId`) and
   per-subcore channels, which a mode-2 broadcast cannot express; or
2. Announce per sub-block with a mode-4 channel (ids 0..10 per sub block, the two sub blocks in
   disjoint halves). That pulls in the whole of §6.2/§6.6 — the consuming-versus-sticky latch
   question, the fence audit, and the isolated cluster-sync probe — which is a task of its own.

Until one of those lands, **this prototype is not clean for any grid whose reduction share lands
3 live tasks on a block, and the GEMV-on leg is not clean at all.** The GEMV-on leg's failures
are at the same id and the same pattern, so they are most likely the same defect made easier to
hit by its different timing rather than a separate fault of `disableGemv`; that attribution is
inference, not evidence.

## 6. The GEMV knob

`kGemmGemvMode` and `kGemm` share one body and differ in **nothing** but
`MmadParams::disableGemv`, so a behavioural difference isolates that field. `disableGemv` exists
only under `__NPU_ARCH__ == 3510`, which is the target.

The default is **off** (GEMV disabled) because the two legs land different cubes in L0C and the
Fixpipe has to match: with GEMV on, C is `1 × N` split into `N/16` sub-vectors each occupying
1024 bytes of L0C of which only the low 64 bytes are valid; with it off, C is a conventional
M-unaligned cube with M padded to 16, which is the shape this operator's audited Fixpipe path
already emits. Both legs produce ULP-0 output. Only the GEMV-off leg audits clean (§5).

## 7. Five device faults, and why the printed result did not find any of them

All five produced **numerically correct or structurally plausible output** while faulting. This
is the single most transferable thing in this report: a run is not evidence until its exception
dumps are empty and its own traces audit. The first CAModel run of this probe printed
`PASS ... ULP within budget` with 34 `[error]` lines in its log, because the runner was scoring
the arithmetic and not the dumps.

1. **A vector store must be 32-byte aligned.** The statistics gather packed lane 0 of each
   split's landing with one single-element `Adds` per split; `maxes[1]` is four bytes in, and
   the core raised `vec_err_instr_misalign_t0`. Vector repeat strides are per-repeat and in
   whole blocks, so no repeat pattern expresses a 4-byte pick either. It is an
   `AscendC::Gather` over a byte-offset table now — the primitive the KV codec's expand already
   uses — with the table written by the scalar unit at `Init` and handed to the vector pipe
   through `S_V`.
2. **The Fixpipe's UB destination must be allocated by the AIC too.** The product slots were
   behind `if ASCEND_IS_AIV`, so the Cube targeted an unallocated tensor:
   `su_ccu_mpu_err_t0`, a B32 scalar access at a null base on `cubecore0`. The vector core still
   read the right values through L0C, so **the output was correct**. Both roles now walk the
   same `TPipe` allocation, exactly as the L1 buffers already did.
3. **`Mmad`'s `k` is the logical depth, not the aligned one.** `kAlignedSplits` describes the
   operand *layout*; passing it to `MmadParams` told the Cube to accumulate eight products,
   which at `splits < 8` multiplied L1 rows DN2NZ had never written and returned `inf`. With
   `k = numSplits_` the padding rows are not reached at all — so DN2NZ's documented zero fill
   was not what was carrying that case, and the zeroed weight lanes remain the second guard.
4. **A `Brcb` destination is eight DataBlocks, not one.** The API reference is explicit that
   eight elements are read per iteration and filled into eight contiguous DataBlocks, so one
   repeat writes 64 floats. The broadcast of `m*` had a single block and silently overwrote the
   seven behind it, including both per-slot `l*` values. At **one** reduction task that is
   invisible, because the clobber lands after the only read; at **two** it zeroed task 0's `l*`,
   so `1/(l* + eps)` became 1e20 and every channel of task 0 exported `inf`. Four
   `static_assert`s now pin the arena so the next allocation cannot drift into a broadcast
   destination.
5. **A null GM address is avoidable, so it should not exist.** The probe passed `nullptr` for
   the length vector in the combine-everything case. It now always passes a real allocation and
   selects the leg with `fused_limit < 0`.

Faults 3 and 4 are both invisible to a single-task probe, which is precisely the depth gate's
reason to exist. Fault 4 is also the one the token model **could not** have caught — it is a
buffer-sizing fault, and it now lives in §2 as `static_assert`s.

### What the token model missed, and the fix to the model

The missing `SyncVectorToMte2` before each task's MTE2 burst into the shared statistics landing
(a WAR across two pipes) was found on CAModel, not in the model, because the first version of
the model emitted the burst and the vector reduction on **one in-order queue** — which makes the
hazard unreachable and its negative control unrejectable. AIV0's MTE2 is now its own queue, with
`landing_ready` / `landing_free` places between it and the vector stream, and
`no_landing_fence` is a negative control that fails with *statistics landing overwritten before
its reduction*. Recorded because the model is supposed to be the gate that runs first.

## 8. Isolation

The variant cannot link into production. Its kernel (`tests/sim/kernel/gemm_reduce_kernel.cpp`)
and its service header (`tests/sim/kernel/turboquant_gemm_reduce_service.h`) live under
`tests/`; nothing in `kernels/` includes either. The only shared declaration is
`include/turboquant_gemm_reduce.hpp`, which the kernel and the host gate both compile — §6.1's
rule, so the host compile is evidence about the kernel's arithmetic and not about a
transcription of it. The combine's close-out calls the production `NormalizeHeads`,
`WriteHeads` and output-stage finisher; nothing is reimplemented but the combine itself.

## 9. Commands

```bash
# Host gates, strict build, and the full CAModel ladder.
bash scripts/run_gemm_reduce_camodel.sh

# Host gates alone, which is the whole of the evidence on a machine with no toolkit.
python3 tests/cpu/test_gemm_reduce_model.py
g++ -std=c++17 -O2 -Wall -Wextra -Werror tests/cpu/test_gemm_reduce_geometry.cpp -o /tmp/gate && /tmp/gate

# One rung.
bash scripts/run_gemm_reduce_camodel.sh reuse one TAG --leg gemm --splits 5 --tokens 1 --heads 4
```

Driven from Windows as `wsl -d Ubuntu-22.04 -- bash <script>`; the toolkit is native in WSL at
`/usr/local/Ascend/cann-9.2.0-beta.2`. No Docker.

## 10. Limitations

* CAModel only. No NPU on this host. No hardware claim is made anywhere above.
* The one failing gate in §5 is open, and the GEMV-on leg does not audit clean at any rung.
* No overlap claim. The pipelining is designed and token-modelled, but no intersecting
  per-engine interval measurement was taken, so "the Cube's GEMV overlaps the Vector stat merge"
  is an intent, not a result.
* The reduction is **not** overlapped with the attention phase, and this prototype does not
  attempt it. Token `t`'s combine needs every split of token `t` complete, which is the
  grid-wide dependency `SyncAll<false>()` discharges; removing it needs a per-token completion
  counter in L2, which is a different probe.
* §6's inter-wave gates are untested: the probe's grids are 1–2 blocks, far below the physical
  block count, so no second wave exists. A production-shaped grid brings §6 into scope.
* `kMaxTasksPerBlock` is 64 in the prototype; a larger per-block reduction share would need the
  dense task list replaced by a sliding window.
* Exhaustive interleaving exploration in the token model is for bounded instances (R ≤ 10, two
  or three workers). It is not a claim about every sequence length.
* This branch also carries a pure CRLF→LF normalisation of
  `scripts/run_l2_atomic_probe_camodel.sh` that was already in the working tree when the task
  started. Zero content change; noted because it is not this task's.
