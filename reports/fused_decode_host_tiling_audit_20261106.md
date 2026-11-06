# Fused decode: SPU/scalar audit, host-side metadata hoisting, kernel unification

Date: 2026-11-06
Operator: `kernels/ops/turboquant/vllm_integration_layer/op_kernel/turboquant_fused_decode.cpp`
Target: Ascend 950PR / `dav-3510` / CANN 9.2.0-beta.2 / `Ascend950PR_9599`, `KERNEL_TYPE_MIX_AIC_1_2`
Protocol: `REFINEMENT_OF_COMBAT_KERNELS.md` §1–§5
Environment: WSL2 Ubuntu-22.04, BiSheng, CAModel. **No physical NPU on this host** (`npu-smi` and
`/dev/davinci*` absent), so every device claim below is CAModel and the hardware gates are
recorded as pending, not passed.

---

## 1. What changed, in one paragraph

Every quantity the decode used to derive on the DaVinci scalar unit — a Split-K worker's tile
count, its logical page span, the live row count of its final tile, and the
`(token, kv head, split, chunk)` a grid block's task range starts at — is now computed on the
CPU and handed to the device as plain words in a read-only GM buffer
(`ascend_micro::fused_decode::TurboQuantDecodeTilingData`, one 64-byte descriptor per
`(token, split)`, plus one 32-byte dispatch record per grid block). The device's tile cursor is a
counter plus a shift and a mask. There is **no integer division anywhere on the device side of
this operator**, no loop whose trip count depends on device data, and no diagnostic-ABI
comparison inside the cursor. The two copies of the kernel became one, in the subtree that is
exported to vllm-ascend.

---

## 2. Part 1 — the scalar audit, item by item

Counts are scalar operations issued on the SPU, per the named unit of work. `D` = 256,
`page = 128` rows, `kSlots = 3`, `kWindows = 2`, `kPhases = 6` throughout.

### 2.1 `CountTiles()` — eliminated entirely

| | before | after |
|---|---|---|
| trip count | `blocks_in_split`, data-dependent | — |
| per iteration | 1 probe compare, 1 block-table read (scalar cache or GM), 1 sign branch, 1 multiply `block * blockSize_`, 1 compare + conditional subtract, 1 `CeilDiv(rows, 64)` shift-add, 1 loop compare + increment | — |
| per task | ≈ 8 × `blocks_in_split`, **on both roles** | 1 scalar GM load (`num_tiles`) |

At a 16384-token context with `splits == 1` that was ≈ 1000 scalar operations per task on the AIC
and the same again on each AIV, all of it in front of the first MTE2 burst. It is now the word
`num_tiles` out of the worker's descriptor. The host computes it in closed form
(`build_decode_tiling`), and the Tier-1 gate checks that closed form against the identical page
walk `CountTiles` performed, over thirteen geometries including ragged contexts, 4 KV heads,
chunked GQA groups and non-power-of-two split counts.

`task.numTiles` is still the single predicate both roles branch on to skip an empty worker — and
it is now strictly stronger than before: it is one word both roles **read from the same place**,
rather than the return value of a loop each of them ran separately over GM.

### 2.2 `NextTile()` — closed form, no loop, no branch

Before, per cursor advance:

```
if (cursor.active) { base += 64; if (base < rows && probePage_ == UINT32_MAX) { ... return; }
                     active = false; ++block; }
while (cursor.block < blockEnd) {
    if (probePage_ != UINT32_MAX && cursor.block != probePage_) { ++block; continue; }
    physical = BlockTableEntry(token, block);          // 2 compares + subtract + load, or a GM load
    if (physical >= 0) { rows = blockSize_;
                         consumed = block * blockSize_;            // multiply
                         if (consumed + rows > contextLen) rows = contextLen - consumed;
                         if (rows > 0) { 4 stores + 1 compare/select; return; } }
    ++block; }
```

≈ 15–18 scalar operations, two to five of them branches, **twice per tile** — once for the read
cursor and once for the consume cursor — and the second of those sits between a tile's softmax
and the next tile's L1 stage, inside the dependence chain the three-zone schedule exists to keep
full. After:

```
absolute = tile_origin + tile;                        // add
page     = absolute >> pageShift;                     // shift
base     = (absolute & pageMask) << 6;                // and, shift
physical = blockCache_[page];                         // bounds compare + scalar-memory load
valid    = tile == lastTile ? lastValid : 64;         // compare + select, no branch
```

7 operations on the read cursor, 3 on the consume cursor, no loop, no mutable cursor state beyond
the counter, no sign test, and no GM load while the span is inside the scalar cache. The
`TileCursor` struct (`block`, `rows`, `base`, `physical`, `valid`, `active`) is gone; the two
cursors are `readTile` and `consumeTile`.

**The sign test is gone because it moved, not because it was dropped.** An unmapped logical page
inside a live span is a geometry an affine tile→page map cannot express. See §5.

### 2.3 Dispatch arithmetic — six runtime div/mods per task → zero

`Process()` decoded every task id with

```
chunk       = task % chunksPerGroup_;        //  mod, runtime, non-power-of-two
splitTask   = task / chunksPerGroup_;        //  div, runtime
split       = splitTask % numSplits_;        //  mod, runtime (splits of 3, 5, 6, 7 are admitted)
tokenKvHead = splitTask / numSplits_;        //  div, runtime
kvHead      = tokenKvHead % numKvHeads_;     //  mod, runtime
token       = tokenKvHead / numKvHeads_;     //  div, runtime
```

— six per task, and the DaVinci SPU has no integer-divide unit, so each is an expansion that
stalls in-order issue. A 16-task block paid 96 of them, the first six before anything could be in
flight. `ProcessReductionProbe()` carried the same six.

Now: the host writes one `TurboQuantDecodeDispatch` per grid block, the device reads it once in
`Init` (eight scalar loads from one 32-byte GM line), and `AdvanceTask()` walks an odometer —
fastest-varying first, chunk, split, kv head, token, which is the order those six divisions
imposed — at ≤ 3 compares and ≤ 3 adds per task, amortising to ~1.1 for a typical block.

`Reduce()`'s per-task `task / numHeads_` and `task % numHeads_` went the same way; both vector
subcores' starting `(token, head)` are in the dispatch record, and the ceil-halving that splits
the share is `CeilDiv(end - start, 2)` — a compile-time constant, so a shift.

**This equivalence is gated, not asserted.** `tiling_gate::check_dispatch` walks the odometer from
the host record for every task id of every grid block, over thirteen geometries × eight
`tasks_per_block` values, and requires it to equal the six divisions it replaced — and the same
for both reduction halves.

### 2.4 `RunTask()` bounds arbitration — two runtime divides → zero

```
seqBlocks      = CeilDiv(contextLen, blockSize_);                 // div, runtime
blocksPerSplit = CeilDiv(seqBlocks, fused ? 1u : numSplits_);     // div, runtime
blockStart     = split * blocksPerSplit;
blockEnd       = min(blockStart + blocksPerSplit, seqBlocks);      // compare + select
tail           = contextLen % kCubeTileRows;                       // power-of-two mod
if (blockEnd == seqBlocks && tail != 0) ComputeTailMask(tail);     // compare
```

replaced by six scalar loads out of **one 64-byte GM line** (`num_tiles`, `tail_rows`,
`block_start_idx`, `blocks_in_split`, `tile_origin`, `tiles_per_page_shift`). The tail-mask
predicate is now `tail_rows != 0`, which the host sets on exactly the worker that carries the
sequence end — the same condition `blockEnd == seqBlocks && contextLen % 64` established, minus
the arithmetic.

### 2.5 `Init()` — the last two runtime divides

`groupHeads_ = numHeads / numKvHeads` and `chunksPerGroup_ = CeilDiv(groupHeads_, headsPerTask_)`
were once per launch per block, not per task, but they are still metadata geometry on the SPU.
Both are launch-uniform, so every descriptor carries them (`group_heads`, `chunks_per_group`) and
`Init` reads the first descriptor. The `headsPerTask_` clamps that remain are compares and
selects, and they agree with the builder's.

### 2.6 `PrefetchBlockTable()` — the host's span, not a discovered one

The prologue sweep was already there and is the right shape: one pass over the split's page
entries into a 64-entry scalar cache, so the cursor reads local memory rather than GM. Three
things changed.

* The span is `blocks_in_split` from the descriptor. Nothing is read to *discover* where the live
  span ends, and nothing past it is touched — which is the "eliminate unmapped page scanning"
  item: the scan that used to establish `blockEnd` by clamping a derived `blocksPerSplit` against
  a derived `seqBlocks` is gone, and the sweep covers exactly the pages the host declared live.
* The sweep ORs the entries' sign bits into `pagesMapped_`. That costs one OR per page on a loop
  that was already loading every one of them, and it is what lets the steady-state cursor drop
  its per-tile sign test.
* `BlockTableEntry(token, block)` — two compares, a subtract and a base rebase per call — is gone.
  `TilePhysical` indexes the cache directly by page.

### 2.7 Scalar `GetValue()` from GM inside execution loops

| site | before | after |
|---|---|---|
| `NextTile` → `BlockTableEntry` | 1 per tile per cursor when the span exceeded the 64-entry cache | only when `blocks_in_split > 64` (see §6) |
| `CountTiles` | 1 per page, both roles | gone |
| `ComputeTask` | 1 (`contextLenGm_`) per task | 1 per task — **retained deliberately**, see §6 |
| `NeedsReduction` | `numTokens` per launch per block | unchanged, see §6 |
| `ComputeQueryOperandHeads` | `qScale.GetValue(...)` per head, a Vector→Scalar readback | not on this operator's path: the file `static_assert`s `Vector::kBatched`, so `ComputeQueryOperandRows` is the instantiated form and it keeps the reciprocals in UB with no readback |
| `Init` (new) | — | 8 + 2 per launch per block |
| `RunTask` (new) | — | 6 per task, one cache line |

### 2.8 Diagnostic-ABI branches in the steady state — eliminated

`probePage_ != UINT32_MAX` was compared **four times** in `CountTiles`/`NextTile`, two of them
inside `NextTile`, so an isolated-canary comparison was executed twice per tile on **every
production launch** — in the dependence chain, for a diagnostic the production launcher cannot
even reach. `InitProbe`, `probePage_` and `probeTile_` are deleted. The isolated single-tile
canary is now the descriptor fields `block_start_idx` and `tile_origin`, which also makes it
*more* general: a worker may begin mid-page, which the walking cursor could not express.

### 2.9 Cross-core fence overhead — audited, one redundancy found, deliberately NOT taken

`SyncEvent<HardEvent::FIX_S>()` is a `SetFlag`/`WaitFlag` pair that stalls scalar **issue** until
the Fixpipe queue retires: the most expensive thing this operator can put on the AIC's
instruction stream. `RunCubeSteadyPhase` issues three per phase — one in `CubeScores`, one in
`CubeContext`, one in `ReleaseCubeSlot` — and the third looks redundant at both of its call sites:

* in `RunCubeSteadyPhase` it follows `CubeContext<PHASE>` immediately, which has already drained
  FIX before setting `kOptFlagContextReady`, with no Fixpipe instruction issued in between, so it
  drains an already-retired queue;
* in `RunCubePipeline`'s prologue, `ReleaseCubeSlot<2>()` primes slot 2's credit before any
  Fixpipe work exists, so it drains an empty one.

**The size of the finding, measured.** The wrap rung's AIC trace carries 18 `FIXP -> SCALAR`
event pairs (§7), and they account exactly: one prologue prime, one `CubeScores<0>`, three per
steady phase × three phases, five in the three-step drain, one per-task join and one launch join.
Four of the eighteen are `ReleaseCubeSlot`'s. Removing them would take the AIC's FIX_S census from
18 to 14 at six tiles, and N−3+1 stalls off every longer task.

**It was removed, and then put back.** The removal argument rests entirely on "the scalar stream
is in order and `CubeContext`'s drain is still in force", because `CrossCoreSetFlag` issues on the
FLOWCTRL/scalar unit in both modes — mode 2 lowers to `ffts_cross_core_sync` — and the `pipe`
template argument orders it behind nothing (`MICROROUTINE_DESIGN_GUIDELINES.md` §6.2a). §6.2c
records a *measured* case in this operator family of an announcement overtaking its own un-drained
pipeline by 1032 and 2179 cycles, and says the hazard is invisible to every gate this change can
run: set/wait counts balance, the id census passes, the capacity-one model works on the schedule
rather than the fence, and the output stays bit-exact while a panel is read one generation early.

Worse, the prescribed source-level gate cannot settle it. With the drain removed,
`audit_cross_core_fences.py` accepts the announcement as a **structural priming set** — nothing in
the enclosing body issues work on the announced pipe — which is true of the text and false of the
steady loop. The audit would pass for the wrong reason. The gate that would settle it is an A/B
`analyze_announce_overtake.py` comparison at one geometry and one announcement ordinal, which
needs a second full build of this operator; that script has deliberately no pass/fail and the
comparison is the caller's job.

So the fence stays, the reasoning is in the kernel at `ReleaseCubeSlot`, and the finding is listed
as open in §6. No fence was touched in this pass. The per-slot `MTE2_V` events, the six cross-core
channels, the `DrainFusedDecodeTask` rendezvous and the epilogue drains are byte-for-byte as they
were; the `kOptFlag*` id budget is unchanged, so §6.2's id-space `static_assert`s hold without
edit, and `test_fused_decode_opt_trace.py::test_source_fences` still finds exactly seven
announcements with none failing.

### 2.10 Per-tile scalar cost, before and after

| per tile, steady state | before | after |
|---|---|---|
| `NextTile` (read cursor) | ≈ 15–18 ops, 2–5 branches | 7 ops, 0 branches, 1 bounds compare |
| `NextTile` (consume cursor) | ≈ 15–18 ops, 2–5 branches | 3 ops, 0 branches |
| probe-ABI compares | 2 | 0 |
| `FIX_S` scalar-issue stalls (AIC) | 3 | 3 (see §2.9) |
| GM scalar loads | 0–2 (0 while cached) | 0 while cached |

---

## 3. Part 2 — the host tiling architecture

`kernels/ops/turboquant/vllm_integration_layer/include/turboquant_fused_decode.hpp`
(repository-root spelling: `include/turboquant_fused_decode.hpp`, a one-line forwarder, so there
is exactly one definition). `<cstdint>`/`<cstddef>` only — no CANN, no CCE — because the device
kernel, the host launcher, the Tier-1 gate and the Torch adapter all include it and the device
reads these words **by offset**.

```
struct alignas(64) TurboQuantDecodeTilingData {   // one per (token, split)
    uint32_t num_tiles;             // exact number of 64-token tiles for this split task
    uint32_t regular_tiles;         // full 64-row tiles (unmasked)
    uint32_t tail_rows;             // live tail in the final tile, 0 if aligned
    uint32_t block_start_idx;       // logical page offset in blockTable for this split
    uint32_t blocks_in_split;       // pages in this split
    uint32_t head_dim;              // D = 256
    uint32_t group_heads;           // GQA ratio
    uint32_t split_k;               // split count S
    uint32_t tile_origin;           // first tile INSIDE block_start_idx (0 in production)
    uint32_t tiles_per_page_shift;  // log2(block_size / 64)
    uint32_t flags;                 // live, carries-tail
    uint32_t chunks_per_group;      // ceil(group_heads / heads_per_task)
    uint32_t reserved[4];
};
struct alignas(32) TurboQuantDecodeDispatch {     // one per grid block
    uint32_t token, kv_head, split, chunk;                 // the attention odometer's start
    uint32_t reduce_token, reduce_head;                    // vector subcore 0's reduce start
    uint32_t reduce_token_hi, reduce_head_hi;              // vector subcore 1's
};
```

The first eight words are the mandated contract, verbatim and in order. The next four are what the
closed-form cursor needs on top of it, inside the same 64 bytes rather than in a second array. The
padding is **named and zeroed**: this structure is uploaded to GM, so implicit padding would be
uninitialised bytes crossing into device memory.

Buffer layout: `descriptors[tokens * splits]` at offset 0 (so the array the device indexes most
often starts 64-byte aligned), then `dispatch[grid]`. Read-only, read by **both** roles —
`num_tiles` is the predicate the Cube and the Vector cores have to agree on.

The host launcher side: `Buffers` gains `void *tiling`, the kernel entry gains one `GM_ADDR`, and
both `turboquant_mm_fused_decode_opt*_impl` **refuse a launch with a null tiling pointer**. There
is no device fallback on purpose — silently deriving the tiling again on the SPU is the one
outcome that would look correct in a test and cost the Cube its prefetch on hardware.

`build_decode_tiling(request, context_lengths, block_table, out)` is the one place `CeilDiv`,
`%` and `log2` are taken. `block_table` is optional: given, every live worker's page span is
checked for unmapped entries; omitted, the span is assumed dense and the kernel's own
`pagesMapped_` guard is what catches a disagreement.

---

## 4. Part 3 — unification and cleanup

* `kernels/ops/turboquant/turboquant_fused_decode_opt.cpp` — **deleted** (`git rm`, 1129 lines).
  It and `vllm_integration_layer/op_kernel/turboquant_fused_decode.cpp` were the same kernel,
  differing only in include paths and cross-tree comment references; the surviving copy is the one
  in the subtree exported to vllm-ascend.
* Retained, unmodified: the 3-stage modulo ring (`kSlots=3, kWindows=2, kPhases=6`), the Cube GEMM
  reduction (`[16, S] × [S, 16]` through a dual-destination Fixpipe) and the raw-query FWHT
  involution prologue/epilogue (`ApplyPi`, the 2320-word pool, the three output stages). The
  transient-pool overlay and its four extent `static_assert`s are untouched.
* `cmake/NpuTargets.cmake`, `cmake/SimTargets.cmake` and
  `vllm_integration_layer/standalone/CMakeLists.txt` now all name that one source; the simulator
  leg, the physical leg and the staged legs get the identical three include roots, so they cannot
  compile two different operators behind one flag.
* Repointed: `tests/sim/kernel/fused_decode_opt_kernel.cpp` (the `#include` of the kernel),
  `tests/sim/kernel/test_three_zone_schedule.py` (`SOURCE`),
  `scripts/archive_fused_decode_opt_evidence.sh`, `scripts/gemm_reduce_switch_gate.sh`,
  `scripts/profile_single_tile_msprof.sh`, `scripts/import_fused_decode_opt.py`, and the comment
  references in both copies of `turboquant_gemm_reduce.hpp` and in
  `kernels/ops/turboquant/fused_decode/README.md`.
* `turboquant_fused_decode_opt_probe` survives in `tests/sim/` for one reason only: to pin the
  isolated canary's *geometry* (one token, one KV head, one split, the full 8-page table) outside
  production. Which page and which tile are live is host tiling now, so its `page`/`tile`
  arguments are gone.

---

## 5. The one contract narrowing, stated plainly

**An unmapped (negative) logical page inside a live split's span is no longer processed by
skipping it.** A closed-form tile→page map is affine; it cannot skip pages. Two things follow, and
both are deliberate:

1. `build_decode_tiling` **refuses** that geometry (`DecodeTilingStatus::kUnmappedPage`) when it
   is given the block table. Negative entries *past* the live span — which is what vLLM padding
   actually is, and the only place real block tables carry them — remain admissible and are
   asserted so in the Tier-1 gate.
2. The kernel keeps a guard for a device table that disagrees with what the host validated: the
   prologue's sign-OR sets `pagesMapped_`, and a worker whose span holds a negative entry is
   **skipped whole** — a defined neutral result, canaries intact, both roles lock-step, no flag
   touched, no deadlock. It is never indexed through.

The previously passing `unmapped` rung tested the walking cursor's page-skipping and therefore
tested behaviour that no longer exists. It is **repurposed, not removed**: the rung plants an
interior hole behind the builder's back and asserts the degradation path above, which is the only
way to reach that guard; the host-side refusal of the same geometry is asserted in the Tier-1
contract gate. Production `block_tables` are dense over `[0, ceil(seq_len / block_size))`, so no
production shape is affected.

`valid_launch` and `build_decode_tiling` additionally now refuse a `block_size` that is not a
**power-of-two** multiple of 64 (a 192-row page was admissible before and would need an SPU
integer divide per tile). No fixture, ladder rung or benchmark row used one; the real block sizes
are 64, 128 and 256.

---

## 6. Identified and NOT fixed in this pass

Recorded explicitly rather than left for a reader to find.

1. **`TurboQuantBatchedGemmReducer::Reduce` still divides per task.**
   `op_kernel/cube/turboquant_gemm_reduce_service.h` has `task / numHeads_` and `task % numHeads_`
   at six sites (lines ≈ 150, 284, 527, 707, 851, 1257). This is the **default** reduction phase
   (`TQ_GEMM_REDUCE=ON`). It is not in the attention pipeline — it runs after
   `AscendC::SyncAll<false>()`, with no Cube prefetch to starve — and converting it is a separate
   change, because the batched form walks *runs of consecutive live tasks* against one uniform
   `dnNum` source stride, so a cursor conversion has to carry the fractal-stride reasoning with
   it. The vector reducer's copy of the same arithmetic *was* converted (§2.3) even though the
   default build does not instantiate it.
2. **`contextLenGm_.GetValue(token)` per task, retained by choice.** `fused` decides whether
   `StageTaskOutput` writes the token's output or leaves a partial, and the reducers' own predicate
   reads the same word. A second copy in the descriptor would be exactly the two-sources-of-truth
   divergence the task join exists to make impossible. It is one GM load in the task prologue, not
   arithmetic and not a loop.
3. **`NeedsReduction()` still loops over `numTokens` with a GM load each**, once per launch per
   block. It is grid-uniform by design (the `SyncAll` it guards must be entered by all blocks or
   none), and a host-computed flag would be the same hoist with the same drift risk as (2).
4. **`TilePhysical`'s cache-bound compare, and its GM fallback.** `kBlockCacheEntries` is 64 (256
   bytes of scalar local memory). `blocks_in_split > 64` — one geometry in the whole harness: a
   16384-token context at `splits == 1` — falls back to a GM scalar load per tile. Removing the
   compare needs either a larger scalar cache (512 bytes at 128 entries; the existing comment says
   256 bytes was the budget-driven choice, and I did not have a figure to overrule it with) or a
   host-side cap on the span. Open.
5. **`tests/camodel/camodel_common.hpp`'s flood word is still `0xDEADBEEF`**, which is a normal
   negative float rather than a NaN. It is shared by eight unrelated probes whose comments name it,
   so changing it is not this operator's change. The fused decode's own ladder flood word is now
   `0x7FC0BEEF` (§7).
6. **No hardware.** §6.1–§6.6 of the protocol (the inter-wave gates for a `MIX_AIC_1_2` grid that
   exceeds the physical block count) and every silicon figure are **pending**. The ladder below
   runs `grid ≤ 4`; it says nothing about wave reuse.
7. **`ReleaseCubeSlot`'s `FIX_S` drain is redundant and was left in place.** §2.9 has the
   reasoning and the measurement: 4 of the wrap rung's 18 AIC `FIXP -> SCALAR` pairs are this
   drain, so taking it would be 18 -> 14 at six tiles and N−2 fewer scalar-issue stalls per task.
   The edit is one line. What it needs first is a gate that can actually see the hazard §6.2c
   measured: an A/B `scripts/analyze_announce_overtake.py` comparison of two builds at one
   geometry and one announcement ordinal. `audit_cross_core_fences.py` would accept the removal as
   a structural priming set, which is the wrong reason, so it is not that gate.
8. **`audit_cross_core_fences.py` cannot see through this operator's fences.** It is lexical, so
   `vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::X_S>()` and the symbolic
   `kSubBlockSyncMode` both defeat it; `test_fused_decode_opt_trace.py` works around that by
   textually expanding them before calling `audit_text`. Teaching the auditor the helper directly
   would make it usable on this operator's source as it stands, and would remove two standing
   false `FAIL`s. Out of scope here — the findings predate this change and the project's own gate
   is unaffected.

---

## 7. Part 4 — bounded canary verification on CAModel

Driver: `scripts/run_fused_decode_canary_ladder.sh` (new). Case runner and trace auditor:
`scripts/run_fused_decode_opt_case.sh`, `scripts/analyze_fused_decode_opt_trace.py` (unchanged).
Fast front-end gate for the edit loop: `scripts/check_fused_decode_device_syntax.sh` (new).

Protocol §3 conformance, in the runner (`tests/sim/kernel/test_camodel_fused_decode_opt.cpp`) and
its guarded-buffer helper:

* every GM payload is flooded with **`0x7FC0BEEF`** (FP32 quiet NaN) before the live slices are
  assigned — changed from `0x5A5A5A5A`, which a float view reads as an ordinary `1.5e+16` and so
  could propagate through a softmax and still land inside a tolerance;
* 64-byte guard bands either side of every payload, asserted after every launch
  (`assert_guards`);
* non-identity block table (logical page `p` → physical `2p+1`), the full 18-page cache
  allocation kept, nonzero logical page indices, no rebasing to zero;
* every input snapshotted before upload and compared **bitwise** after synchronisation — the
  tiling buffer included, since it is an input;
* every unassigned output row and the whole inactive workspace required to still read the canary;
* active rows compared against an FP64 (`double`) host oracle at **0 ULP**.

### Results

Evidence root `/tmp/tq-refactor` (ext4, not `/mnt/c`: BiSheng's CANN preprocessing can observe
stale source pages on a DrvFs checkout). 13 cases, **13 PASS, 0 failures**. Every line below
carried `comparison=exact inputs=unchanged guards=0x7FC0BEEF-intact` and a `PASS trace audit`,
and no case left a nonempty `*excp*.dump`.

| case | geometry | model ticks | verdict |
| --- | --- | ---: | --- |
| `isolated-first` | 1 tile, logical page 0 | 25 343 | PASS |
| `isolated-middle` | 1 tile, logical page 4 (= K/2) | 25 327 | PASS |
| `isolated-last` | 1 tile, logical page 7 (= K−1) | 25 336 | PASS |
| `isolated-last-tile` | final 64-row tile of page 7 | 25 328 | PASS |
| **rung 1** | tiles 1, tail 0 | 26 045 | PASS |
| **rung 2** | tiles 2, tail 15 | 36 211 | PASS |
| **rung 3** | tiles 5, tail 1, across page boundaries | 64 170 | PASS |
| **rung 4** | tiles 3 = `kSlots` — ring priming | 46 490 | PASS |
| **rung 5** | tiles 6 = 2·`kSlots` — slot reuse and drain | 77 745 | PASS |
| `unmapped-degrades` | tiles 12, interior hole at page 2 | 10 809 | PASS (worker skipped, §5) |
| `empty-worker` | context 0 | 10 027 | PASS |
| `splitk-workers` | tiles 6, 4 splits, grid 4, one empty worker | 52 040 | PASS |
| `repeat-tasks` | tiles 4, 17 heads (2 chunks), 2 launches | 198 900 | PASS |

Wall time, for scheduling a rerun rather than as a performance figure: 79 s for a single-tile
case up to 621 s for the two-launch case; the whole ladder is about 25 minutes of model time after
a ~17-minute build. **Simulator ticks are not silicon throughput** (protocol §5) and nothing here
is a latency claim.

Pass criteria, against the mandate:

* **0 ULP against the FP64 CPU oracle across all active rows** — met. The runner accumulates the
  expected context in `double` and compares against an independently RNE-rounded FP16 image, with
  the tolerance set to 0 for every case in this ladder (`comparison=exact`; the 1-ULP budget
  exists only for the `--varied` arithmetic regression, which is not a rung here).
* **Canary buffers intact** — met. Both 64-byte exterior guard bands of all ten GM buffers, every
  unassigned output row, and the whole inactive workspace still read `0x7FC0BEEF` after every
  launch; every input, the tiling descriptors included, came back bitwise identical to its
  pre-upload snapshot.
* **Zero deadlocks on cross-core synchronization flags** — met. Every case completed inside the
  1800 s timeout and its trace audit reported `capacity-one, balanced dispatch/completion, empty
  drain`.

#### The wrap rung's trace audit, in detail

`rung5-ring-wrap/audit.log`, which is the case that exercises slot reuse and the drain:

* **19 cross-core channels**, all on mode 2 and all within ids 0–10 — the legal intra-block space.
  The AIC waits on `m2:0,1,2` (the three slot-ready channels), `m2:9` (probs-ready) and `m2:10`
  (the task join); each AIV waits on `m2:3,4,5` (slot-free), `m2:6,7` (scores-ready), `m2:8`
  (context-ready) and `m2:10`. Set and wait censuses match per channel and per consumer, with the
  two sub-blocks in disjoint halves.
* **Dispatch equals completion on every HardEvent route**, capacity-one holds in completion order,
  and the final marking is empty — so no credit was left outstanding at the drain.
* **Engine census.** AIC: 3 896 instructions (SCALAR 3 567, FLOWCTRL 179, FIXP 54, MTE1 49,
  CUBE 45, MTE2 2). Each AIV: 44 164 instructions, byte-identical between the two sub-blocks
  (SCALAR 12 243, RVECEX 12 089, RVECST 8 974, RVECLD 7 844, RVECSU 1 302, FLOWCTRL 710,
  PUSHQ 552, RVECLP 274, MTE3 106, MTE2 28, VEC 42). Real Vector arithmetic on both, so the audit
  is not vacuous.
* **AIC HardEvent routes**: `FIXP->SCALAR` 18, `MTE1->CUBE` 14, `CUBE->FIXP` 12,
  `CUBE->SCALAR` 4, `CUBE->MTE1` 1 per slot. The 18 is the number §2.9 is about.

What the ladder does **not** establish: anything above `grid = 4`, so nothing about multi-wave
reuse (§6 of the protocol); any long context — deliberately, see the script's own header; and
anything at all about hardware.

---

## 8. Gates run, and their verdicts

| gate | protocol | verdict |
| --- | --- | --- |
| `tests/cpu/test_fused_decode_opt_model.py` — capacity-one token/queue model, every interleaving at N = 0,1,2,3,4,5,8,16,64 | §1 | PASS |
| `tests/sim/kernel/test_three_zone_schedule.py` — the actual `Run*` methods compiled with recording services, N = 1…130, all six residues, no steady-phase boundary branch | §1 | PASS |
| `tests/cpu/test_fused_decode_opt_trace.py` — trace-auditor negative controls **and** the producer-fence source gate (7 announcements, none failing) | §1, §6.2a | PASS, 9/9 |
| `tests/cpu/test_fused_decode_opt_contract.cpp` — UB extents, launch guards, NZ bijection, **plus the new host-tiling gate**: the dispatch odometer against the six divisions it replaced over 13 geometries × 8 `tasks_per_block`, the descriptors against the page walk, and five refusals | §2 | PASS |
| `test_integration_host_gate` — the staged subtree's Tier-1 gate, now including the same tiling gate | §2 | PASS, 211 checks |
| `scripts/check_fused_decode_device_syntax.sh` — BiSheng front end, `-Wall -Werror`, `dav-c310-cube` and `dav-c310-vec` | §2 (partial) | PASS |
| Full strict device compile of the simulator leg (`camodel_fused_decode_opt_kernel`) | §2 | PASS |
| Full strict device compile of the **physical** leg (`turboquant_fused_decode_opt_npu`, `-Wall -Werror`, `Ascend950PR_9599`) from the unified source | §2 | PASS — `lib/libturboquant_fused_decode_opt_npu.so`, built from `.../vllm_integration_layer/op_kernel/turboquant_fused_decode.cpp.o` |
| Isolated logical-page canaries: page 0, K/2, K−1, final tile of final page | §3 | PASS ×4 |
| The five bounded rungs | §3–§4 | PASS ×5 |
| Depth (N = `kSlots`) and wrap (N = 2·`kSlots`) | §4 | PASS |
| Real Split-K workers including an empty one; repeated launches over a two-chunk task | §4 | PASS |
| Per-case trace audit and instruction census | §5 | PASS ×13 |
| Inter-wave gates for a `MIX_AIC_1_2` grid above the physical block count | §6 | **PENDING** — the ladder runs `grid ≤ 4`; nothing here reaches a second wave |
| Any physical-device execution or latency figure | — | **UNAVAILABLE** — no NPU on this host |
| A/B `analyze_announce_overtake.py` comparison for the §2.9 fence finding | §6.6 | **NOT RUN** — needs a second full build; the finding was not taken |
| Full-sequence varied-score regressions and the context/split benchmark sweep | §4, §5 | **NOT RUN** in this pass — they compile (host syntax gate) but belong on silicon |

Two notes on how these were invoked, because both can mislead a reader of the raw logs.

1. `scripts/audit_cross_core_fences.py` run **directly** on the kernel source reports two `FAIL`s
   at `DrainFusedDecodeTask`'s two announcements. That is an artifact of the invocation, not a
   finding: the auditor is lexical and cannot see through the `SyncEvent<>` helper or the symbolic
   `kSubBlockSyncMode`. The project drives it through
   `test_fused_decode_opt_trace.py::test_source_fences`, which expands both first and then
   requires exactly seven findings with none failing — and that passes. Both `FAIL`s are
   reproducible on the pre-refactor file at its lines 106 and 118, so they are unchanged by this
   work.
2. The driver logs `fence audit exit=1` for the same reason, and does not gate on it.

---

## 9. Evidence, and reproducing

Archived under `reports/fused_decode_host_tiling_20261106/`: the four host-gate logs, the fence
audit, the whole ladder log, the driver log, build-log tails, the build artifact listing, and per
case its `command.txt`, `run.log` and `audit.log`. The per-case instruction and completion dumps
(tens of MB each) were left in the ext4 scratch root `/tmp/tq-refactor`, which each case's
`DUMPS.txt` names with its size; they are regenerated by a rerun.

```bash
bash scripts/check_fused_decode_device_syntax.sh      # ~1 min, BiSheng front end only
bash scripts/run_fused_decode_canary_ladder.sh        # host gates, build, the 13-case ladder
```

The ladder builds into a fresh `mktemp` root under `/tmp` and prints it. Set
`FUSED_DECODE_OPT_ROOT` to pin it, and pass `reuse` to skip the build and drive an existing
runner through `FUSED_DECODE_OPT_RUNNER`. Budget about 17 minutes for the simulator build and 25
for the ladder on 12 cores; the physical-device library is another ~25 and is not built by that
script (`cmake --build <dir> --target turboquant_fused_decode_opt_npu`).

Not reproducible on this host, and not claimed anywhere above: anything requiring a 950PR.
