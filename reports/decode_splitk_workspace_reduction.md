# Split-K decode: non-blocking workspace partition reduction

Replaces the Global Memory test-and-set reduction in the TurboQuant decode
Split-K path with a disjoint partition export plus an in-launch barrier and a
one-worker-per-head reduction, and brings the Cube path's partial export onto
the same workspace layout.

Target: CANN 9.2.0-beta.2, `dav-3510`, Ascend 950PR, `Ascend950PR_9599`.

**Status: the section 2 compile-time gate and the section 1 token/queue model
are complete. Sections 3, 4 and 5 — the CAModel canary, depth/wrap and trace
gates — have NOT been run for this change and are recorded as pending in
section 6 below. Nothing here is evidence that the refactored kernels execute
correctly; it is evidence that they are well-formed and that their schedule is
1-safe.**

## 1. What changed

### 1.1 The atomic protocol is gone

Removed from `kernels/ops/turboquant/turboquant_single_token_decode_splitk_aiv.cpp`:

| Removed | What it was |
| --- | --- |
| `turboquant_splitk::AtomicHeadSlot` | the per-(request, KV head) lock word, arrival counter and shared `RowState[8]` record |
| `Acquire` / `Release` | the two `__simt_vf__` entry points, dispatched with `asc_vf_call` |
| `asc_atomic_exch`, `asc_atomic_add`, `asc_stcg`, `asc_threadfence` | the SIMT test-and-set and its cache-bypassing control stores |
| `MergeHeadState` **inside the lock** | the merge is retained, but it now runs in the reduction pass, not in a critical section |
| `WorkerMetrics::failed_exchanges`, `::timeout`, `::acquired`, `::arrival` | the spin/watchdog census |
| the `max_attempts` operand | the watchdog bound, on both entry points and both host launchers |
| `simt_api/asc_simt.h`, `simt_api/common_functions.h` | no SIMT code remains in the translation unit |

`grep -n "asc_atomic\|AtomicHeadSlot\|asc_vf_call\|failed_exchanges\|asc_stcg\|threadIdx\|max_attempts\|simt_api"`
over the kernel returns nothing.

### 1.2 The standardized partition workspace

`include/turboquant_splitk_contract.hpp` now defines the layout both decode
paths export into and the reduction reads:

```
[requests][heads][splits][kMaxStateRows] of RowState
RowState = { float m[8]; float l[8]; float acc[256]; }   // kRecordFloats = 272
kMaxStateRows   = 16        // the Cube path's M; the Vector path writes rows <= 8
kPartitionFloats = 4352     // 17408 B, 32-byte aligned
```

A task owns one partition and nothing else:
`workspace + item * kPartitionFloats`, with
`item = (request * heads + head) * splits + split` — the split decode grid's
own block index, which is why the partition axis is minor. No two tasks address
the same bytes, so no write contends with any other.

The row bound is the Cube path's M rather than the Vector path's. The two
operators have different bounds (`kMaxRows = 8` for the Vector kernel, set by
its Unified Buffer arena and the dispatch rule that sends `rows >= 12` to the
Cube operator), and a shared stride has to hold the larger. A Vector task
writes `rows * 272` floats at the base of its partition and leaves the rest
untouched; the reduction reads exactly as many.

`WorkspaceFloats(requests, heads, splits)` sizes the allocation; every rig uses
it rather than open-coding the product.

### 1.3 The Vector path: export, barrier, reduce

`ExportPartial` is what used to be the critical section:

* `splits == 1` — bypasses the workspace entirely, runs `Egress` and writes
  `out_context` from the decode pass. The same `Egress`, over the same
  buffers, that the reduction calls, so the one-split and K-split answers come
  out of identical code.
* `splits > 1` — `PackHeadState` into a retired value page, then **one
  contiguous `rows * 272`-float MTE3 burst** into this task's partition. No
  acquire, no ticket, no merge, no release.

Then, in the entry point:

```
if (splits > 1) {
    AscendC::SyncAll<true>();
    if (item < requests * heads) ReduceHead(...);   // one worker per head
}
FinishMetric(...);                                  // once per block
```

`ReduceHead` re-reads the gate for the head it is reducing (a reducing block
decoded some other head's split), re-seeds `(m, l, Acc)` to the decode's own
seed, merges the head's K records with the **unmodified `MergeHeadState`** —
the same log-sum-exp recurrence the lock used to run — and runs `Egress` and
the context burst.

Seeding to `m = -3e38, l = 0, Acc = 0` makes the seed the neutral element of
the recurrence, so split 0 goes through the merge like every other split and
the loop has no special case.

**`SyncAll<true>`, not `<false>`.** The template parameter is `isAIVOnly`, not
a strictness knob. In CANN 9.2
(`asc/impl/basic_api/dav_3510/kernel_operator_sync_impl.h`), `false` raises
`SYNC_AIC_AIV_FLAG` and waits for the Cube cores to answer it, which is right
for the reference operator because it is `MIX_AIC_1_2`. This kernel is
`KERNEL_TYPE_AIV_ONLY`: its launch has no Cube core to answer, so `false`
would wait on a participant that does not exist. `true` raises
`SYNC_AIV_ONLY_ALL`. This is a deliberate departure from
`turboquant_fused_decode_opt.cpp`, which uses `<false>` correctly for its own
core type.

The default `SyncAllConfig` trigger pipe is `PIPE_ALL`, so the barrier opens
with its own `PipeBarrier<PIPE_ALL>()`: the exporting block's MTE3 burst is
drained before its flag is raised, and no extra drain is written on either
side.

### 1.4 The grid bound the barrier imposes

`SyncAll` completes only if every launched block is resident at the same time.
A grid that runs in two hardware waves parks wave 0 on the barrier while wave 1
has not been scheduled, and the launch hangs rather than returning a wrong
answer. So:

```cpp
constexpr std::uint32_t kMaxCoResidentBlocks = 32;
constexpr bool GridFitsOneWave(requests, heads, splits)
{ return splits <= 1u || requests * heads * splits <= kMaxCoResidentBlocks; }
```

enforced twice: the device returns uniformly before forming any address, and
the host launcher throws `std::invalid_argument`. The device guard alone would
be silent across a grid, which is why the host refuses as well.

A single-split launch never reaches the barrier and is not bounded.

**This bound changed a test matrix.** The AIV validation ladder ran
`requests = 2, heads = 4, K in {2, 4, 8}`, whose K = 8 rung is a grid of 64.
It now takes the largest request count each K admits — two at K = 2 and K = 4,
one at K = 8 — so every rung lands on or under 32. Pinning requests at two and
letting K = 8 overflow would not test a wider shape; the host refuses that
launch.

### 1.5 The Cube path's export, on the same layout

`turboquant_paged_decode_splitk_aic` took two separate output tensors,
`state[item][2][16][8]` and `acc[item][16][256]`. It now takes one `workspace`
and writes the standardized AoS record into
`workspace + item * kPartitionFloats`.

Three strided MTE3 bursts, not one contiguous burst, and deliberately: the
state lives in Unified Buffer as three separate planes — `m_prev`, `l_prev`
and `acc` are what the recurrence reads and writes — and interleaving them
into one buffer first would need a 16 × 272-float staging area this arena does
not have room for (section 2.2). A strided burst reaches the same layout with
no staging: one DataBlock per row at a 272-float pitch for each state plane,
32 blocks per row at the same pitch for the accumulator. All three land in
lanes of a partition no other task addresses.

The geometry is asserted rather than assumed — `kTokens <= kMaxStateRows`,
`kHeadSize == kChannels`, `kStateLanes == kStateLanes`,
`kRecordFloats == 2 * kStateLanes + kHeadSize`, every burst length and pitch a
whole DataBlock, and every `DataCopyParams` field inside `uint16`.

The verified 5-stage, 12-column modulo schedule across `MIX_AIC_1_2` is
untouched: the change is confined to the `if (partial && finalize)` export
block and the entry point's operand list.

### 1.6 The census

`WorkerMetrics` is now nine eight-lane replicated DataBlocks — 288 bytes, one
contiguous MTE3 burst written entirely by the Vector pipe from immediate
scalars. No SIMT store and no cache-bypassing store remain, because no worker
reads another worker's control. Replication is what makes a torn publication
detectable: a reader that finds two lanes of one field disagreeing saw a
partial burst.

| Field | Meaning |
| --- | --- |
| `exported` | 1 once the worker published a partition or a fused context |
| `partition` | the partition index written, or `kNoPartition` on the fused path |
| `reduced` | 1 when this block also ran a reduction task after the barrier |
| `pages` | logical pages this split covered |
| `tail` | 1 on the worker that wrote this head's `out_context` |
| `finish_hi/lo` | absolute completion cycle |
| `export_hi/lo` | **the export region: state pack through retired MTE3 burst** |

`export_*` is the direct successor of `lock_region_*` and the figure the
43,000-cycle lock region should be compared against. It is reported by the
benchmark as `export_ticks`, replacing the `lock_ticks`, `failed_spins` and
`timeouts` CSV columns.

## 2. Section 2 — compile-time memory checks: PASS

Both kernels compiled for `dav-3510` with `-Wall -Werror`, CANN headers
classified as system headers, inside
`ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12`.

```bash
docker run --rm -v "$(pwd)":/workspace -w /workspace ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12 bash scripts/run_splitk_workspace_build.sh
```

All eight targets build clean, zero warnings, zero errors:

| Target | Status |
| --- | --- |
| `test_splitk_decode_probe` | built, **PASS** at run |
| `test_aic_splitk_decode_probe` | built, **PASS** at run |
| `camodel_single_token_decode_splitk_kernel` | built |
| `camodel_paged_decode_attention_kernel` | built |
| `test_camodel_decode_splitk_boundary` | built |
| `bench_decode_splitk_sim` | built |
| `test_decode_splitk_aiv_sim` | built |
| `test_decode_splitk_aic_sim` | built |

Host probe output:

```
PASS: 2048 page partitions; real partial merges K=1..16; two-request 4096-token oracle; cyclic-page error 0.00832015
PASS: 16 distinct M rows; uneven/empty temporal splits; multiple requests/heads; nonzero page window; N=1024,4096,16384 K=1,4,16 full-context combines.
```

### 2.1 The Unified Buffer budget

Measured by instantiating an undefined `template <unsigned> struct ArenaProbe`
at `kArenaEnd` and reading the value out of the compiler's diagnostic, then
removing the probe. Both figures are for the 192 KiB window the harness
budgets (`kUbWindow = 196608`), not the 256 KiB physical partition.

| Kernel | `kArenaEnd` | + 256 B read pad | Window | Slack |
| --- | ---: | ---: | ---: | ---: |
| `turboquant_single_token_decode_splitk_aiv` | 190 752 B (186.3 KiB) | 191 008 B | 196 608 B | 5 600 B |
| `turboquant_paged_decode_attention` | 155 424 B (151.8 KiB) | 155 680 B | 196 608 B | 40 928 B |

The refactor adds **no new Unified Buffer operand to either kernel**. The
reduction reuses buffers the strip loop has retired by the time it runs: the
two dequantised key pages as the record ring's slots (`rows * 272` floats =
8 704 B at most, against a 16 384 B page), a retired value page as the pack
destination, and `kUbProducts` as the merge's 128-float alpha scratch. So
`kArenaEnd` is unchanged from before the refactor on both kernels.

### 2.2 Why the Vector path's reduction is capped at 8 rows

The 5 600 B of slack above is the reason the Vector reduction cannot serve the
Cube path's 16 rows. The egress panels are sized `kMaxRows * kHeadSize * 4` =
8 KiB each, and there are seven of them (`kUbQuery`, `kUbAcc`, `kUbNorm`,
`kUbRotated`, `kUbGate`, `kUbGateA`, `kUbGateB`) plus a BF16 context panel and
the raw query panel. Doubling `kMaxRows` to 16 adds roughly 65 KiB, against
5.5 KiB of headroom. The Cube path's own reduction is therefore not in this
change; see section 6.

### 2.3 Static assertions added

In the contract header: the record is a whole number of 32-byte bursts, the
partition base is 32-byte aligned, `kMaxRows <= kMaxStateRows`, the census is
nine replicated DataBlocks with no padding and a multiple of 32 bytes.

In the Cube kernel: the partition export's geometry and every
`DataCopyParams` field, as listed in section 1.5.

The existing AIV assertions — every operand base 32-byte aligned, ring-slot
bank skew, `kArenaEnd + 256 <= kUbWindow`, the state/fold register over-read
windows, `kMaxRows * kRecordFloats <= kPanelElems` — are unchanged and still
hold; the build evaluates all of them.

## 3. Section 1 — token flow and engine queues: PASS

`tests/cpu/test_decode_ring_model.py` was extended with the reduction's record
ring, a second bounded marked graph in the same block entered only after the
barrier. It has the strip ring's shape — two slots, capacity-one credits on
both routes, primed before the loop and drained after it including the slot a
`K` below the ring depth never fills — over one packed partial-state record
per split instead of a packed strip, and with no archive edge.

```bash
python3 tests/cpu/test_decode_ring_model.py
```

```
PASS reduction K=1:  32 reachable states, empty final marking
PASS reduction K=2:  50 reachable states, empty final marking
PASS reduction K=3:  62 reachable states, empty final marking
PASS reduction K=4:  80 reachable states, empty final marking
PASS reduction K=5:  92 reachable states, empty final marking
PASS reduction K=8:  140 reachable states, empty final marking
PASS reduction K=12: 200 reachable states, empty final marking
PASS reduction K=16: 260 reachable states, empty final marking
PASS rejected negative control: reduction unprime: deadlock / unprimed cycle
PASS rejected negative control: reduction no_refill_credit: unbalanced token census
PASS rejected negative control: reduction leak: unbalanced token census
PASS rejected negative control: reduction merge_without_ready: merge read the wrong record
PASS 22654 states: 1-safe, ordered ownership, liveness, finite drain; 7 negative controls
```

Every K the kernel accepts is explored exhaustively over all engine
interleavings, plus the two below the ring depth. `merge_without_ready` is the
control that matters most: it drops the forward edge **and** its SET, so the
token census still balances and the fault has to be caught by ownership — it
is the shape a reduction that trusted program order across two queues would
have.

The strip ring's own exploration is unchanged and still passes at
N = 0,1,2,3,4,5,8,16,64 for both kernels, with its three original controls.

### 3.1 Event id allocation

The reduction's ring takes ids of its own rather than reusing the strip
ring's. Both rings run in the same block — a reducing worker is a worker that
already decoded a split — and the strip ring's credits are fully drained
before the barrier, so reuse would be legal. It would also make the section 5
census unreadable: that audit keys events by `(route, id)` and expects a fixed
SET/WAIT count per strip-ring id, and two phases sharing an id would be
indistinguishable from a double SET.

| Route | Strip ring | Reduction |
| --- | --- | --- |
| `V_MTE2` | 0, 1 (packed), 2, 3 (scale) | 4, 5 (record free) |
| `MTE2_V` | 0, 1 (ingress ready) | 2, 3 (record ready), 4 (gate) |
| `V_MTE3` | 1 (egress), 2 (census), 3 (partition) | — |
| `MTE3_S` | 0 (egress drain), 2 (census), 3 (partition) | — |

All inside 0..5; 6 and 7 stay reserved.

## 4. Host gate: PASS

```bash
bash scripts/run_reshape_wave_host_gate.sh
```

Every harness that consumes the changed ABI compiles with
`-Wall -Wextra -Werror`, and the untouched legs still pass:

```
tests/npu/bench/bench_turboquant_decode_splitk.cpp                      OK
tests/npu/validation/test_turboquant_decode_splitk_aic_validation.cpp   OK
tests/npu/validation/test_turboquant_decode_splitk_aiv_validation.cpp   OK
tests/npu/test_npu_decode_single_token_unaligned.cpp                    OK
tests/cpu/test_decode_tail_geometry.cpp                                 OK
tests/cpu/test_decode_tail_probe.cpp                                    OK
```

## 5. What the census now checks

The rig's `Validate` was rewritten around the new protocol. Under the lock the
questions were "did every worker acquire" and "are the arrival tickets a
permutation". Neither exists. What replaces them:

* every worker published (`exported == 1`);
* every worker wrote the partition its own task index names
  (`partition == item`), which is what disjointness means here;
* every live lane of every record a worker wrote came back finite, and the
  exported `m`/`l` blocks are replicated across their eight lanes — the
  workspace is canary-flooded before the launch, so an unwritten or
  misaddressed record is a NaN rather than a plausible number;
* `reduced` is set on exactly the first `requests * heads` blocks of the grid;
* exactly `requests * heads` workers emitted a context. **This is a property
  of the grid, not of a head's K workers**: the emitter is block `h`, which
  reduces head `h` after the barrier whatever split it decoded, so the old
  per-head "exactly one tail" loop would have counted four tails under head 0
  and none under the others. The check was restructured accordingly;
* the per-worker census is not torn across any of its nine replicated fields;
* the gated BF16 context still matches `Fixture::expected`, the independent
  FP64 full-context oracle that applies `DenseRotate` rather than the kernel's
  factorization.

The CAModel boundary probe additionally holds the **entire workspace** to the
canary: it launches one block at `splits = 1`, so the fused path must not touch
a byte of it. That is the evidence that the one-split bypass really does
bypass.

The workspace is deliberately **not** zeroed before a launch. Under the atomic
protocol the shared head slot had to start clean, because a worker merged into
whatever it found there. A partition is written before it is read, and by one
task only, so a run that needs it pre-cleared is a run with a hole in it.

## 6. Gates NOT run, and what is not claimed

Recorded explicitly, per `AGENTS.md` and `REFINEMENT_OF_COMBAT_KERNELS.md`.

| Gate | Status |
| --- | --- |
| §1 token/queue model | **PASS** (section 3) |
| §2 compile-time memory checks | **PASS** (section 2) |
| §3 isolated single-tile boundary canaries on CAModel | **NOT RUN** |
| §4 depth and wrap tests, real Split-K workers | **NOT RUN** |
| §5 trace audit and instruction census | **NOT RUN** |
| §6 inter-wave gates for the MIX_AIC_1_2 grid | **NOT RUN** |

Nothing in this change has executed on CAModel or on silicon. In particular:

* **The barrier has never been taken.** `SyncAll<true>` is argued from the
  CANN 9.2 header and from the kernel's core type, not measured. A grid that
  hangs on it would hang, not fail, and only a CAModel or hardware run can
  tell the difference.
* **The reduction has never produced a number.** `MergeHeadState` is carried
  over unchanged and the host oracle in `test_splitk_decode_probe` exercises
  the same recurrence in FP64 over K = 1..16, but the device-side loop over
  partitions has not run.
* **The Cube path's strided partition burst has never been read back.** Its
  `DataCopyParams` arithmetic is asserted at compile time and its layout is
  checked by `AicRig::Validate`, which has not been executed for this change.
* **No performance claim is made.** The 43,000-cycle figure is the previous
  protocol's, from `reports/l2_atomic_contention_split_k.md`. `export_ticks`
  is the field that would replace it; it has not been sampled. That the
  critical section is structurally gone is a property of the source, not a
  measurement.
* **The Cube path has no reduction.** This change brings its export onto the
  shared layout, which is what the scope asked for. A reducer for its 16 rows
  does not fit the Vector kernel's arena (section 2.2) and would have to be
  either a phase of the Cube kernel itself — reusing its `EgressQueries`, which
  is row-at-a-time and UB-cheap — or a separate launch. Neither is in this
  change.

The reproducible baseline for the gates above is
`bash scripts/run_decode_pipeline_camodel.sh`, then
`bash scripts/run_decode_pipeline_regressions.sh BASELINE_DIR BUILD_DIR` and
`python3 scripts/verify_decode_pipeline_evidence.py BASELINE_DIR`, per
`REFINEMENT_OF_COMBAT_KERNELS.md` §5. The AIV validation ladder's K = 8 rung
now runs at one request rather than two (section 1.4), which is a narrower
shape than the pre-refactor ladder covered.

## 7. Commands

```bash
docker run --rm -v "$(pwd)":/workspace -w /workspace ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12 bash scripts/run_splitk_workspace_build.sh
```

```bash
python3 tests/cpu/test_decode_ring_model.py
```

```bash
bash scripts/run_reshape_wave_host_gate.sh
```
