# The dense batched Split-K reduction: M = 16 packed query rows on the Cube

Date: 2026-11-04. Branch `claude/gemm-final-reduction-0f1afc` (exploratory, not merged, not
pushed). Target CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599`, BiSheng.

**There is no NPU on this host.** Every device result below is the `Ascend950PR_9599` CAModel
simulator, run inside the `ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12` container —
this WSL instance's own CANN is 8.5.0 and knows neither `dav-3510` nor `Ascend950PR_9599`.
Nothing here is a hardware claim, and simulator ticks are not silicon throughput.

This continues `reports/gemm_final_reduction_camodel_20261104.md`, whose M = 1 Cube GEMV form
left one gate failing: the `m2:2` ordering defect on `core0.veccore1`. That gate is the reason
this form exists, and it now passes.

## 1. The requested shape, and the one place the algebra does not allow it

The task asked for a dense `Out[16, 16] = Weights[16, S] x PartialsSlice[S, 16]`: 16 packed
query rows against 16-channel slices of `D = 256`, A resident in L0A, 16 MMADs sweeping
`d ∈ [0, 16, …, 240]`, batched `Dn2NzParams`/`dnNum` ingest.

**A GEMM with one shared B operand does not compute this operator's reduction.** Writing `q`
for a packed query row,

```
out_q[c] = sum_s w_q[s] * P_q[s, c]
```

and the partials `P_q` are indexed by `q`: split `s` of `(token, head)` lands at
`workspace[((token*numHeads + head)*numSplits + s) * 272]`, so every `(token, head)` owns its
own `[S, D]` block. Query heads inside one GQA group share the **KV cache**, not the Split-K
partials — each head has its own attention probabilities and therefore its own `o_s`. A GEMM
`Out = A x B` shares `B` across all `M` rows, so a dense `[16, S] x [S, 16]` computes
`sum_s w_q[s] * P[s, c]` for a `P` that does not exist. It is a batched GEMV with 16
independent `(A, B)` pairs, and no 2-D GEMM expresses that without either batching or padding.

So **K carries the batch** and A is the block-diagonal weight matrix:

```
A[q, q'*8 + s] = (q == q') ? w_q[s] : 0        A is [16, 128], one C0 block of K per row
B[q'*8 + s, c] = P_q'[s, c0 + c]               B stacks the 16 rows' own channel slices
Out[q, c] = sum_{q',s} A[q, q'*8+s] * B[q'*8+s, c] = sum_s w_q[s] * P_q[s, c0+c]   ✓
```

Everything the requested shape asks for survives. `M = 16` and `N = 16` are both full, so the
Cube's 16x16 output line is 100% occupied; A is **one cube loaded into L0A once per batch** and
reused across all 16 slice iterations; B is a 16-fractal slice streamed into L0B with a
ping-pong; the sweep is `D / 16 = 16` Mmads per batch; and ragged groups carry zero weight
rows. The deviation is that `K = 16 * 8 = 128` rather than `K = S`, because the 16 rows do not
share a partials matrix. `K = S` would have been one C0 block; `K = 128` is sixteen, which is
still one `Load2DV2` and one `Mmad` with no K tiling.

### Two properties of the block-diagonal A that are gates, not remarks

* **A dead row cannot contaminate a live one.** `Out[q, c]` reads K block `q` and nothing else,
  so garbage in B's block `q` — a ragged batch's padding fractal, still holding §3's
  `0x7FC0BEEF` flood — reaches output row `q` and no other. The M = 1 form had no such
  guarantee: there, one NaN lane in K poisoned every channel of the product. This is checked on
  the host (`check_block_diagonal_containment`) rather than argued.
* **The split-count guards still apply within a row.** For `splits < 8` the weight row's lanes
  `s..7` are exactly zero (`maxes` is pre-filled with a finite `-FLT_MAX`, so `exp(m_s - m*)`
  underflows to 0) and DN2NZ zero-fills the destination to the 32-byte boundary. Both halves of
  that pair are kept; relying on either alone would be relying on an undocumented half.

## 2. What the form actually buys, and what it does not

Stated plainly, because two of the three things the task expected from it are not what it
delivers:

* **It does not reduce MTE2 bytes.** The combine must read every partial, and the M = 1 form
  already fetched a whole task's `[S, 256]` block in ONE burst. Neither form moves less.
* **It does not reduce MTE2 instruction count either — it raises it**, from one burst per task
  to two per slice (two because the subcore halves are static; §4). The bursts are 4 KB each
  and the bytes are unchanged, so this is a wash in practice, but it is the opposite of the
  expectation and is recorded as such.
* **It does not reduce MAC count.** The zeros in the block-diagonal A cost exactly what M = 1's
  row padding cost. "Dense 100% Cube line utilization" is true of the Cube's output line and
  not of the useful fraction of the multiplies.

What it does deliver, measured in §6:

* **Both vector subcores carry half the combines**, where M = 1 left all of them on subcore 0.
  This roughly thirds the vector critical path and is the whole of the `m2:2` fix.
* **Two cross-core announcements per BATCH rather than per task** — 16x fewer mode-2
  handshakes at a full batch.
* **One `NormalizeHeads`/`WriteHeads` per half-batch** instead of one per task, so the
  close-out's per-task overhead is amortised over eight rows and the output leaves in one burst.
* **Roughly half the MTE1 traffic**: one A load per batch plus one slice load per slice, against
  two loads per task.

## 3. The schedule

Per batch of up to 16 **consecutive** live tasks (consecutive because `dnNum` walks one uniform
source stride, so a fused token's task-id gap has to end the batch — and because it makes the
batch's outputs contiguous in GM):

| role | step | work |
| --- | --- | --- |
| AIV_c | `Stats(b)` | ONE strided `DataCopy` for every `(row, split)` statistics tail of subcore c's half; per row `m* = max m_s`, `w_s = exp(m_s - m*)`, `l* = sum w_s l_s`; a 32-byte MTE3 store of `w` into the cube's row `q`; announce `weights_ready` |
| AIC | `Gemm(b)` | the A cube into L0A **once**; then 16 x {batched DN2NZ ingest into L1, `Load2DV2` into L0B, `Mmad(16x16x128)`, dual-destination `Fixpipe`}; announce `product_ready` |
| AIV_c | `Finish(b)` | its own 8-row panel: `NormalizeHeads`, the shared finisher per token run, one `WriteHeads` burst |

The two vector subcores are symmetric: both run the same program over different data, both
stage half the block-diagonal cube, and the Cube's single `CrossCoreWaitFlag` is therefore a
genuine two-producer mode-2 quorum rather than one real producer and one mirror.

The inner slice ring carries real split credits — `slice_free`/`slice_ready` (L1),
`l0b_free`/`l0b_ready`, `l0c_free`/`l0c_ready` — so the DN2NZ of slice `j+1` overlaps the Mmad
of slice `j`. All four AIC feedback cycles (those three plus `l0a_free`) are primed and drained
**once per worker**; §7 records why that is not a detail.

## 4. The subcore split, and why the halves are static

A dual-destination Fixpipe (`dualDstCtl = 1`, production's own
`TurboQuantCubeMm::ComputeProductToUb` idiom) divides `mSize` rows in half and lands each half
at the matching subcore's own UB base. With `mSize = 16` the split point is fixed at 8: rows
0..7 reach subcore 0 and rows 8..15 subcore 1, each as rows 0..7 of its own `[8][256]` panel.
The live rows are distributed ceil/floor — subcore 0 takes rows `0..a-1` with `a = ceil(live/2)`
and subcore 1 rows `8..8+b-1` with `b = floor(live/2)` — so **both have real work for every
`live >= 2`**. `live = 1` is the one idle-subcore geometry and is a ladder rung of its own.

The halves are **static**: fractals 0..7 belong to subcore 0 and 8..15 to subcore 1 whatever
the live count. That is a correctness requirement. A fractal's off-diagonal rows must be zero
for its own row's output to be right, they are written once at Init, and they are never
rewritten; if the half boundary moved with the live count, the core that zero-fills fractal `q`
at Init and the core that writes `q`'s diagonal block in a later batch could be different
cores — and **nothing orders a block's two vector subcores against each other at Init**, so the
zero-fill could land after the weight it is meant to precede and erase it. With static halves
each core only ever touches its own eight fractals and the race cannot be constructed.

The cost is that `mSize` is always 16 and `k` always 128, so a ragged batch's dead fractals are
read by the Mmad, and that the slice ingest needs two bursts instead of one. The first is safe
only because of §1's containment; the second is the MTE2 instruction count in §2.

## 5. Protocol gates

`REFINEMENT_OF_COMBAT_KERNELS.md`, each gate recorded as met, failed, partial or unavailable.
**Read §10 first: the full ladder was not re-run on the final tree**, so several gates below are
partial by scope rather than by outcome.

| Gate | Status | Evidence |
| --- | --- | --- |
| §1 capacity-one token/queue model | **MET, at a stated bound** | `tests/cpu/test_gemm_reduce_model.py::batched_main`: 10 queues, every reachable completion interleaving, 11 negative controls. Bound is B = 0..1 two-worker and B = 2..4 one-worker — the two-worker state count is 4.1e4 at B=0 and 1.1e6 at B=1, ×26 per batch. **Its last full run is against the schedule before the per-worker ring fix; not re-run** |
| §1a outer loop modelled | **MET** | the batch loop is the outer loop and the slice sweep the inner; both are in the model, and `prime_ring_per_batch` is the control for the boundary fault it found (§7.4) |
| §2 compile-time memory checks | **MET** | the contract header's `static_assert`s plus the service's own: L1/L0A/L0B/L0C/UB budgets, 32-byte bases, AKA3003 bank skew, Brcb destination extents, `sums` sized `(rows-1)*8+64`, flag-id budget, and the UB claim pinned against what `Init()` actually allocates |
| §2 strict build | **MET** | probe, production operator and host gate at `-Wall -Werror` for `Ascend950PR_9599`; host contract gate at `-Wall -Wextra -Werror`. `-Wextra` on the device leg is still blocked by the pre-existing unused parameter in `TurboQuantModeName` (`fused_decode/common/turboquant_mode.h:297`), not this stage's to fix |
| §3 flood, canaries, workspace immutability | **MET** | every rung floods all GM with `0x7FC0BEEF`, guards with `0x5A5A5A5A`, and checks both exterior guards, every unassigned output halfword at its own word position, and that the workspace is byte-identical afterwards |
| §3 block-diagonal containment | **MET** | `check_block_diagonal_containment` on the host: `Out[q,c]` depends on K block `q` alone, so a dead fractal's flood cannot reach a live row |
| §3 head / middle / tail isolation | **PARTIAL** | the `--mixed` rungs do this and passed on the pre-final build; **not re-run on the final tree** |
| §4 split-count sweep | **MET on the final tree** | S = 1, 4, 5, 8 — §8 |
| §4 depth and wrap | **PARTIAL** | batch-ring depth/wrap rungs (16/32/44/64 tasks) and the `live = 1..15` ragged rungs passed on the pre-final build; **not re-run** |
| §4 real and empty Split-K workers | **PARTIAL** | same: passed pre-final, not re-run |
| §5 trace audit | **MET on every rung actually run** | the shared `analyze_fused_decode_opt_trace.py` on each run's own traces: capacity-one per route, dispatch == completion, empty final marking, ids 0..5, legal mode-2 channels, cross-core tick order per id, non-vacuous census |
| §5 `m2:2` ordering defect | **FIXED, and measured** | §6. This is the gate the M = 1 form failed |
| §5 instruction census | **MET** | §6, from each run's own `instr_popped_log.dump` |
| §5 intersecting-interval overlap claim | **NOT MADE** | no Gantt export was run for this stage, so no overlap is claimed |
| Integration into the operator | **WIRED, via a UB overlay** | `-DTQ_GEMM_REDUCE=ON` builds and runs; the reducer's 27264 UB bytes come from a `TBufPool` aliased onto the attention phase's dead 32768-byte transient pool, so the 19648-byte deficit is paid without spending headroom — §7.5 |
| Integration regression, default path | **MET** | the pool change touches the DEFAULT path too (three attention buffers now come from a pool), so the operator's own CAModel ladder was re-run — §10 |
| §6 inter-wave gates | **NOT APPLICABLE / UNAVAILABLE** | the probe runs grids of 1–2 blocks, far below the physical block count, so no second wave exists. A production-shaped grid brings §6 into scope and it has **not** been exercised |
| Physical NPU | **UNAVAILABLE** | no NPU on this host |

## 6. The instruction census, and the `m2:2` fix

Three legs, 16 tasks (4 tokens × 4 heads), `splits = 8`, one block, each from its own traces:

| leg | veccore0 | veccore1 | AIV total | AIV worst | vec ops 0 | vec ops 1 | CUBE | FIXP | MTE1 | MTE2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shipping vector reducer | 13450 | 1186 | 14636 | 13450 | 3223 | 0 | 0 | 0 | 0 | 0 |
| M = 1 Cube GEMV | 13450 | 1186 | 14636 | 13450 | 3223 | 0 | 54 | 65 | 84 | 65 |
| **dense batched GEMM** | **4384** | **4384** | **8768** | **4384** | 1451 | 1451 | 93 | 68 | 94 | 102 |

The vector critical path — the busier subcore, which is what the phase actually costs — drops
**13450 → 4384, a factor of 3.07**. Total AIV instructions drop 1.67×. The Cube pays 93 ops
against 54 and the Fixpipe 68 against 65, on engines that were otherwise idle.

The `m2:2` channel, which is the gate the M = 1 form failed:

| leg | `veccore1` waits on `m2:2` | verdict |
| --- | --- | --- |
| M = 1 Cube GEMV | `[28061, 24596, 20839, 17101, 13238, 9418, 5658, 1896]` | monotonically collapsing — the signature; it fires the auditor's `stall collapsed` check at the 3-live-task geometry |
| dense batched GEMM | `[20]` | measured, released after its announcement, `ok` |

Two things changed, and both were needed:

1. **`veccore1` has real work.** 1186 instructions and **zero** vector ops became 4384 and 1451
   — byte-for-byte symmetric with `veccore0`. A subcore that only consumes tokens is not paced
   by the channel that releases it; one that reduces eight query rows is.
2. **Every cross-core wait has a scalar drain behind it.** `CrossCoreWaitFlag` blocks only its
   own pipe's issue, so the `x_S` pair behind it is what makes the release tick appear in the
   trace at all. Without it the auditor reads a floor or a collapse and cannot distinguish a
   legitimately prefetched token from a latch that stopped blocking. With it the threshold-free
   early-retirement check does the real work.

Ticks are not reported as a speedup: they are a simulator observation, and no intersecting
per-engine interval export was made, so no overlap is claimed.

## 7. Faults, and which gate found each

Five, and the order matters: **the first four produced numerically correct output while
faulting.** A run is not evidence until its exception dumps are empty and its own traces audit.

1. **`M_MTE1` ids 0..2 belong to the runtime.** `kernel_tpipe_impl.h` says so in its own words
   — *"for matmul macro, set flag M_MTE1 at the begining of operator, and also wait flag at the
   end. matmul macro only use M_MTE1 event id 0 1 currently"* — and this toolkit primes three.
   A trace of the first batched run shows `SET_FLAG CUBE->MTE1` for ids 0, 1 and 2 at ticks
   1421, 1426 and 1430, before one instruction of the stage has issued. Priming my own credits
   on those ids is a double SET on a one-bit flag: `pem_ccu.cc:2270 execute_set_flag already has
   same set_flag`. **Found by the trace audit; the same run printed `worst_fp16_ulp = 0` with
   every canary intact.** What it costs is not data but an ordering guarantee — a dropped
   MTE2→MTE1 credit lets the fetch of slice `j+2` overwrite the L1 slot the Mmad of slice `j`
   is still loading. Fixed by basing every explicit id at 3, which also clears the ids
   `FetchEventID` hands out (it returns the first free bit, and a bare `SetFlag` never marks
   one occupied, so every `SyncEvent<>` in the drain path picks id 0).
2. **The statistics landing's row stride is `numSplits_`, not `kMaxSplits`.** One `DataCopy`
   with `dstStride = 0` fills the landing contiguously at the *logical* split count; indexing
   it by `kMaxSplits` agrees only at `S = 8`. At `S < 8`, row `numSplits_` and beyond read off
   the end of the filled region, which is zeroed, so `m*` came out 0, every weight `exp(0) = 1`,
   `l* = 0`, and `1/(l* + eps) = 1e20` exported `inf` for exactly row `S` of every batch — row
   1 at `S=1`, row 4 at `S=4`, row 5 at `S=5`. **Found by §4's split-count sweep**, which is the
   gate that exists for this; every `S = 8` rung passes with the bug in place.
3. **A WAR hazard on the shared weight block, across two pipes.** The loop was {`ReduceRow`
   writes the block on V, `SyncVectorToMte3`, MTE3 reads it}, and the next row's `ReduceRow` —
   on V, a different in-order queue — can overwrite the block while MTE3's 32-byte copy of the
   previous row is still outstanding. **Found by reading, not by a rung**, and no rung would
   have: the exact fixture gives every row the same statistics, so the wrong weights are the
   right ones, and the varied fixture's window is a handful of ticks. Per-row weight blocks
   remove it by construction, which is cheaper than arguing about the window.
4. **A feedback cycle primed on one queue and drained on another, per batch.** The inner slice
   ring was primed and drained inside each batch, on the theory that a credit which never
   crosses a batch boundary cannot leak across one. `l0c_free` is primed on FIX and drained on
   M, so nothing orders batch `b+1`'s prime against batch `b`'s drain: FIX runs ahead and sets
   a flag M has not consumed. **Found by the token model** — §1a's second rule verbatim — and
   kept as the `prime_ring_per_batch` control. Priming per worker removes the boundary: the
   slice ring continues across batches, 1-safe by the same induction, and `kGemmSlices` is even
   so the slot sequence stays continuous.
5. **A harness bug, not a kernel one.** The length-vector overrun check reads the word just
   past the live tokens and requires it to still hold the flood, but the buffer was sized
   `ceil(tokens/8)*8` — so at a token count that is an exact multiple of 8 that word lands in
   the trailing guard, where it holds the canary instead. A spurious *"the length vector was
   read or written past its live tokens"* at `tokens = 8` and `16` and nowhere else, which is
   why the M = 1 ladder's rungs (1, 2, 5) never showed it. Sized `tokens + 1` now.

### 7.5 The integration: a `TBufPool` overlay, because the operator has 19648 bytes too little

The reducer's panels do not fit what the operator has left, and the arithmetic is kept as a
compile-time object rather than deleted, because it is what makes the overlay a decision instead
of a habit. The batched reducer claims 27264 UB bytes (two 8224-byte panel slots, a 6208-byte
statistics arena, 4096 of fp16 staging, a 512-byte zero stage) and replaces the vector reducer's
3360, a net 23904. The operator's own assert leaves
`196608 − 142304 − 46432 − 3360 − 256 = 4256`. **Deficit 19648.**

So the reducer does not spend leftover UB. It allocates from a `TBufPool` aliased onto the
attention phase's transient pool:

```cpp
// TurboQuantVectorDecodeService::InitBuffers -- exactly kUbSizes[15..17] long
pipe->InitBufPool(transientPool_, kTransientPoolBytes);   // 16384 + 8192 + 8192 = 32768
transientPool_.InitBuffer(nzOperandBuf_, sizes[15]);
transientPool_.InitBuffer(accBuf_,       sizes[16]);
transientPool_.InitBuffer(queryWorkBuf_, sizes[17]);

// TurboQuantBatchedGemmReducer::Init -- costs zero bytes
pipe_->InitBufPool(ubPool_, kGemmReducerUbBytes, sharedPool);   // 27264 <= 32768
```

`InitBufPool`'s shared overload sets `bufPool.startAddr_ = shareBuf.startAddr_` and **does not
advance the hardware pool** — read off `kernel_tpipe_impl.h`, not the API reference. So the
reduction's 27264 bytes land on top of 27264 of the attention phase's dead 32768, leaving 5504
spare in the pool and the operator's 4256 of headroom untouched.

**Why those three buffers.** `nzOperandBuf_` is the tile ingest's NZ staging operand,
`queryWorkBuf_` the query staging, and `accBuf_` the per-task online-softmax accumulator whose
result `StageTaskOutput` has already written to the GM workspace. `SyncAll<false>()` stands
between `Process()` and the reduction, so nothing reads any of them afterwards.

**Why not a bare `TPipe::Reset()`.** It is a real allocator rewind — `ResetPool()` sets
`maxAddr = 0` for every hardware pool — but it rewinds *everything*, and
`TurboQuantQueryBasis::FinishHeads` runs *during* the reduction and reads the Pi signs in
`signBuf_`, which are content and not scratch. A rewind invalidates them. The pool names the
bytes being reused instead of reusing all of them.

**Why the pool's length is exactly the sum of what it holds.** `kUbEnd` and every buffer
allocated after it keep their offsets. That matters because the Cube's dual-destination Fixpipe
windows sit at UB offsets 0 and `kScoreBytes` and are addressed by offset on three cores with
nothing in the toolchain checking they agree — and the reduction's *own* dual-destination panels
have the same requirement, which is why `InitBuffers` and the reducer's `Init` both run on every
role. The operator's host contract gate independently reports `UB=142304+256` after the change,
unchanged.

Two checks, because a build that succeeds does not say the overlay is why:

| `scripts/gemm_reduce_switch_gate.sh` | result |
| --- | --- |
| `-DTQ_GEMM_REDUCE=ON` must build at `-Wall -Werror` | **PASS** |
| negative control: pool length cut to 16384 must be rejected | **PASS** — `static assertion failed … 'kGemmReducerUbBytes <= kGemmSharedPoolBytes'` |

The two ways out that were **not** taken, kept because they are what to reach for if the reducer
ever outgrows the pool:

* **Band-wise normalise.** The panel only has to hold the channels being normalised, so a
  64-channel band drops it from 8 KB to 2 KB a slot; `NormalizeHeads` is elementwise per row and
  `WriteHeads` becomes a strided GM burst. One panel slot then fits in 2880 bytes net, inside
  even the leftover headroom — but a single slot removes the Cube/AIV panel overlap, which is
  most of what the form buys.
* **A smaller `kGemmBatchRows`.** This does **not** work as a way to fit the leftover headroom:
  at 8 rows the net cost is still 11616 bytes and at 4 rows 5472, both over 4256, and `M = 4`
  gives up the dense line. It would shrink the pool requirement, though.

## 8. Numerical result, on the final tree

The split-count sweep, `tokens = 4, heads = 4` (16 tasks, one full batch), exact fixture, all
four run on the final build:

| rung | worst abs | worst rel | worst fp16 ULP | budget |
| --- | ---: | ---: | ---: | ---: |
| `S = 1` | 0 | 0 | **0** | 0 |
| `S = 4` | 8.370536e-04 | 4.040948e-04 | **0** | 0 |
| `S = 5` | 0 | 0 | **0** | 0 |
| `S = 8` | 4.438920e-04 | 4.245924e-04 | **0** | 0 |

ULP 0 means every channel equals the correctly-rounded fp16 of the double-precision combine
exactly; the residual `worst abs` is the fp16 quantisation of the oracle itself, not a
disagreement. `S = 1` and `S = 5` come out exact in fp32 on this fixture, hence 0. Every rung
also reports canaries intact (`0x5A5A5A5A` guards on both sides of every buffer), the workspace
byte-identical, empty exception dumps and a clean trace audit.

What this does **not** establish: it bounds the batched form's fp32 accumulator difference from
the vector recurrence to below half an fp16 ULP without measuring it. The GEMM sums `S` products
inside one Cube accumulator while the recurrence rescales a running sum `S-1` times.

## 9. The overlay, running

`scripts/gemm_reduce_overlay_camodel.sh` builds the operator with `-DTQ_GEMM_REDUCE=ON` and runs
its own CAModel probe, so the Cube GEMM reduction reads and writes inside the attention phase's
aliased UB. The split rungs are the ones that reach the reduction at all — `NeedsReduction()` is
false at one split — and the single-split rungs are the control for whether moving three
attention buffers into a `TBufPool` disturbed the attention phase itself.

| rung | geometry | result |
| --- | --- | --- |
| `overlay-splits-4` | tiles=6, splits=4, grid=4, heads=16 | **PASS** — `comparison=exact inputs=unchanged guards=0x5A-intact` |
| `overlay-splits-2` | tiles=6, splits=2, grid=2, heads=16 | **PASS** — same |
| `overlay-canary-0` | page 0, splits=1 | **PASS** (control) |
| `overlay-canary-7` | page 7, splits=1 | **PASS** (control) |
| `overlay-tiles-7` | tiles=7, splits=1 | **PASS** (control) |

Each rung's own exception dumps are empty and its own traces audit clean. `comparison=exact` is
the operator's bitwise gate against its oracle, `inputs=unchanged` that it wrote nothing it
should not have, and `guards=0x5A-intact` the exterior canaries on every buffer.

This is the gate the switch gate cannot be. A pool aliased onto the *wrong* bytes compiles
exactly as well as one aliased onto the right ones; what distinguishes them is a run in which
the reduction's panels overlap live attention data and the context comes out wrong. It does not.

## 10. What was and was not run on the final tree

This matters more than usual here, because the tree changed after the wider ladder ran.

**Run on the final tree:** the strict device build of the probe, the host gate and the
production operator's default path; the host geometry gate; the switch gate (both legs); the
overlay CAModel rungs of §9; the `S ∈ {1,4,5,8}` sweep with its canary,
workspace-immutability and trace-audit checks; and the operator's own default-path CAModel
ladder, which the pool change touches whether or not the switch is on:

| `scripts/run_fused_decode_opt_camodel.sh ladder`, switch OFF | result |
| --- | --- |
| `canary-0`, `canary-4`, `canary-7`, `canary-tail` | **PASS** — logical page 0, K/2, K-1 and the final strip in the final page |
| `tiles-3`, `tiles-6`, `tiles-7` | **PASS** |
| `repeat` (2 launches, 17 heads), `empty`, `split-empty` (6 tiles, 4 splits) | **PASS** |

All ten rungs report `comparison=exact inputs=unchanged guards=0x5A-intact` with clean trace
audits, and the host contract gate reports `UB=142304+256` — `kUbEnd` unchanged, which is the
independent check that the transient pool is exactly as long as the three buffers it replaced
and no later UB offset moved.

**Not flipped:** `TQ_GEMM_REDUCE` still defaults OFF. It builds and runs, but the evidence
behind it is two split rungs against the default path's ten plus years of use, and making the
shipping operator's reduction a Cube GEMM is a decision with its own §4 and §5 burden at
production shapes — §6's inter-wave gates among them. Turning it on is one CMake flag.

**Run on an earlier build and NOT re-run:** the rest of the ladder — `live = 1,2,3,5,7,9,15`
ragged batches, batch-ring depth and wrap at 16/32/44/64 tasks, two real Split-K workers, an
empty worker, the `--mixed` fused-token rungs, the varied-fixture regressions and the GEMV-knob
rungs. Every one of those passed, but on the build *before* the per-row weight-block fix
(§7.3), so they are evidence about a schedule one change away from this one. The full ladder is
one command (§12) and re-running it is the obvious next step.

**Not run at all:** the Python token model's last full pass is against the schedule before the
per-worker ring fix (§7.4); physical hardware; any overlap measurement; §6's inter-wave gates.

## 11. Isolation

The variant cannot link into production. Its kernel
(`tests/sim/kernel/gemm_reduce_kernel.cpp`) and its service header
(`tests/sim/kernel/turboquant_gemm_reduce_service.h`) live under `tests/`; nothing in
`kernels/` includes either. The only shared declaration is
`include/turboquant_gemm_reduce.hpp`, which the kernel and the host gate both compile — §6.1's
rule, so the host compile is evidence about the kernel's arithmetic and not about a
transcription of it. The combine's close-out calls the production `NormalizeHeads`,
`WriteHeads` and output-stage finisher, and the dual-destination Fixpipe uses production's own
`dualDstCtl`/`subBlockId` idiom; nothing is reimplemented but the packing and the schedule.

## 12. Commands

```bash
# Container plumbing (the toolkit is in Docker on this host, not in WSL).
bash scripts/gemm_reduce_container.sh sync
bash scripts/gemm_reduce_container.sh up
```

```bash
# Host gates alone, which is the whole of the evidence on a machine with no toolkit.
python3 tests/cpu/test_gemm_reduce_model.py
g++ -std=c++17 -O2 -Wall -Wextra -Werror tests/cpu/test_gemm_reduce_geometry.cpp -o /tmp/gate && /tmp/gate
```

```bash
# Strict device build, then the full ladder, then the three-leg census.
wsl -d Ubuntu-22.04 -- bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_build.sh
```

```bash
wsl -d Ubuntu-22.04 -- bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_run.sh ladder
```

```bash
wsl -d Ubuntu-22.04 -- bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_compare.sh 16 4 cmp64
```

```bash
# The overlay: TQ_GEMM_REDUCE=ON must build, and a pool too short must be rejected.
wsl -d Ubuntu-22.04 -- bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_switch_gate.sh
```

```bash
# The operator itself, running the Cube GEMM reduction out of the aliased pool.
wsl -d Ubuntu-22.04 -- bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_overlay_camodel.sh
```

```bash
# The default path's own regression, which the pool change also touches.
wsl -d Ubuntu-22.04 -- bash scripts/gemm_reduce_container.sh run scripts/run_fused_decode_opt_camodel.sh ladder
```

```bash
# One rung, and the diagnosis path when it fails.
wsl -d Ubuntu-22.04 -- bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_run.sh TAG -- --leg batched --splits 8 --tokens 4 --heads 4
```

```bash
wsl -d Ubuntu-22.04 -- bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_diagnose.sh TAG
```

## 13. Limitations

* CAModel only. No NPU on this host. No hardware claim is made anywhere above.
* **The integration is wired and runs, but is not the default.** §7.5's 19648-byte UB deficit is
  paid by a `TBufPool` overlay on the attention phase's transient pool, verified at runtime in
  §9; `TQ_GEMM_REDUCE` still defaults OFF for the reason in §10.
* **The overlay's safety rests on a liveness argument, not a checker.** Nothing in the toolchain
  verifies that `nzOperandBuf_`, `accBuf_` and `queryWorkBuf_` are dead when the reduction
  reuses their bytes; `SyncAll<false>()` and reading the code are what establish it, and the
  split rungs of §9 are what would have caught it being wrong at those geometries. A future
  change that reads any of those three after the attention phase would break this silently.
* **The full ladder was not re-run on the final tree** — §10 lists exactly what was and was not.
  Treat the depth, wrap, ragged, worker and mixed rungs as evidence about the previous build.
* The token model's coverage bound is B = 0..1 two-worker and B = 2..4 one-worker, and its last
  full pass predates the per-worker ring fix. Exhaustive exploration is for bounded instances.
* No overlap claim. The pipelining is designed, token-modelled and reflected in the engine
  counts, but no intersecting per-engine interval measurement was taken, so "the Cube's GEMM
  overlaps the vector stat merge" is an intent, not a result.
* The reduction is **not** overlapped with the attention phase, and this form does not attempt
  it. Token `t`'s combine needs every split of token `t` complete, which is the grid-wide
  dependency `SyncAll<false>()` discharges; removing it needs a per-token completion counter in
  L2, which is a different probe.
* §6's inter-wave gates are untested: the probe's grids are 1–2 blocks, far below the physical
  block count, so no second wave exists.
* `kMaxGemmTasksPerBlock` is 64, so a block whose reduction share exceeds 64 live tasks would
  need the dense batch list replaced by a sliding window.
* A batch is a run of **consecutive** live tasks. At a fused-token density that isolates every
  live task, every batch is one row wide and the form degenerates to the M = 1 case plus
  overhead. The `--mixed` rungs exercise that shape but no cost model is offered for it.
* The dense form does **not** reduce MTE2 bytes or MAC count, and it raises MTE2 instruction
  count — §2. Only the vector critical path, the handshake count and MTE1 improve.
