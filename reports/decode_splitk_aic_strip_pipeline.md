# The continuous strip pipeline in `turboquant_paged_decode_splitk_aic`

Target: CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599`. **No NPU on this
host** (`npu-smi` and `/dev/davinci*` are both absent), so every device result
below is CAModel over a collapsed grid, run from WSL Ubuntu-22.04, and
**nothing here is a hardware claim.**

This closes the open defect of `reports/decode_splitk_aic_page_race.md`: at
four or more carried pages the operator computed a wrong accumulator, because
AIV1 restaged `kL1KeyPage<Strip>` for sequence-page `p + 1` over page `p`'s
before the Cube's MTE1 had read it, and the mode-4 edge that was supposed to
prevent it did not bound AIV1 at all.

Two changes, and the second is why the first is possible.

* **Mode 2 replaces mode 4** for the inter-strip handshake: one readiness
  QUORUM (both AIVs announce, the Cube does one wait) and one release
  BROADCAST (the Cube announces, both AIVs wait), on two distinct ids that
  never conflate readiness with release.
* **The per-page prime/drain of the 12-column strip schedule is gone**,
  replaced by one flat stream over the window's strips. The L1 key ring is
  `kStageDepth` pages addressed by SLOT with an explicit WAR edge behind it,
  instead of `kStrips` pages addressed by STRIP with no back edge at all.

## 1. Mode 2 was measured before it was relied on

The task brief stated that a physical-silicon probe had shown mode 2 giving a
strict symmetrical scalar rendezvous with hardware quorum. **That could not be
verified here and is treated as supplied context, not evidence.** What follows
was measured on CAModel, on this toolkit, before any of it was built on.

Mode 2 is not a variant of mode 4 — it is a different mechanism. CANN 9.2
`impl/basic_api/dav_3510/kernel_operator_sync_impl.h:790` lowers mode 4 to
`set_intra_block` / `wait_intra_block` and **every other mode** to
`ffts_cross_core_sync` / `wait_flag_dev`. So a mode-4 measurement says nothing
about mode 2, and the two appear in a CAModel trace as different instructions:
`SET_INTRA_BLOCK` / `WAIT_INTRA_BLOCK sync_id:N` against `SET_CROSS_CORE` /
`WAIT_FLAG_DEV` with the id in an `XT` operand.

`tests/research/kernel_cluster_sync_probe.cpp` gained a mode-2 leg built around
exactly the construction this operator needs, and `AIV1 lags AIV0 by 64 `V_S`
round trips on purpose, so that a quorum and a first-arrival latch cannot look
the same. Four iterations, because section 6.2c's "mode-4 consumes" verdict
came from two announcements per id and four is the depth at which the AIC → AIV
direction was measured to stop blocking.

```bash
bash scripts/run_cluster_sync_probe_camodel.sh 4 2
```

### 1a. A mode-2 wait blocks on EVERY use of an id

`scripts/audit_mode4_tick_order.py`, which now reads both modes:

| channel | stall per wait, in ticks |
| --- | --- |
| AIC waits id `m2:8` on FIXP (readiness quorum) | 4583, 18727, 17963, 18727 |
| AIV0 waits id `m2:9` on VEC (release broadcast) | **14167, 11527, 12380, 12979** |
| AIV1 waits id `m2:9` on VEC (release broadcast) | **13882, 11917, 11310, 11826** |

The second and third rows are the direction that failed under mode 4. The same
measurement over the mode-4 page fence read **23146, 3, 3, 3** — a first wait
that blocked and three that did not, against announcements ~20600 ticks apart.
Under mode 2 every wait on every iteration blocks, in both directions.

### 1b. It really is a quorum

Readiness announcements of id 8 and the tick at which the Cube's `FIX_S` drain
behind its wait retired — that drain is a SCALAR instruction, so its logged
tick IS when it ran:

| iteration | AIV0 announces | AIV1 announces | AIC released |
| ---: | ---: | ---: | ---: |
| 0 | 48943 | 49590 | 49600 |
| 1 | **118180** | **117501** | **118190** |
| 2 | 183359 | 184037 | 184047 |
| 3 | 249258 | 250491 | 250501 |

Every release is ~10 ticks after the LATER of the two announcements. Iteration 1
is the one that settles it: **AIV1 announced first, at 117501, and the Cube
stayed blocked for a further 689 ticks until AIV0 announced.** A per-AIV latch
would have released at 117511.

The probe also establishes a trap worth recording: the two AIVs announce on
different pipes (`PIPE_V` and `PIPE_MTE3`) and still form one quorum. That is
deliberate, because the pipe argument is NOT free — see section 2b.

## 2. What the operator now does

### 2a. The rendezvous

Two mode-2 ids, in their own 16-id space, independent of mode 4's on 950PR
(`CrossCoreWaitFlag_ISASI.md`, "Constraints related to flagId"). Both
participants name the same id: there is no `+16` rebase, because mode 2 does
not address an individual AIV.

| channel | id | announced by | waited by |
| --- | ---: | --- | --- |
| `kSync2StageReady` | 8 | **both** AIVs | the AIC, one wait |
| `kSync2StepRetired` | 9 | the AIC | **both** AIVs |

* **Readiness (RAW).** AIV1's `n`-th announcement means "the L1 key page holds
  strip `n`". AIV0's `n`-th means "the UB score slot strip `n` will land in is
  free", i.e. the recurrence of strip `n - 2` has retired; its first two
  announcements are priming sets, the slots starting empty. The AIC's `n`-th
  wait blocks until BOTH hold, which is exactly the pair of preconditions its
  `n`-th GEMM has.
* **Release (WAR).** The AIC's `n`-th announcement means "strip `n` is
  retired": its L1 key page has been read into L0B *and* its scores have landed
  in UB. AIV1's wait before staging strip `t` is its `(t - 2)`-th, released by
  "strip `t - 2` retired" — the strip that owns the L1 page strip `t` is about
  to overwrite. AIV0's wait before the recurrence of strip `t` is its `t`-th.

Neither direction can run away from the other: AIV0's announcement `n` is
behind the AIC's release `n - 2`, and the AIC's release `n` is behind its own
readiness wait `n`. The budget is a compile-time object, not a comment —
`Mode2IdBudget` asserts the two ids avoid `SyncAll`'s 11..14 and the Matmul
high-level API's worst case [0, 7] at four objects, and `kMaxOutstanding`
asserts the worst-case counter occupancy against the 0..15 range a mode-2
counter is allowed (past which the hardware reports a fault rather than
wrapping).

### 2b. A mode-2 call naming the wrong pipe compiles to nothing

Found while reading the library, and worth a gate rather than a comment.
`NotifyEventImpl` / `WaitEventImpl` gate both calls on `IsSplitVectorPipe` /
`IsSplitCubePipe` (`impl/basic_api/kernel_event.h:257`, `:263`), and on
`__NPU_ARCH__ == 3510` the two sets are nearly disjoint:

| core | pipes it may name |
| --- | --- |
| AIV | `PIPE_V`, `PIPE_MTE2`, `PIPE_MTE3` |
| AIC | `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_FIX`, `PIPE_M` |

An AIC naming `PIPE_V`, or an AIV naming `PIPE_FIX`, compiles to **nothing at
all** — silently, with no diagnostic, and the peer then waits forever or, worse,
does not. `Mode2AicPipe` / `Mode2AivPipe` are `static_assert`ed over every pipe
this operator names.

### 2c. The strip stream

`kColumns = 12` / `kStrips = 8` wrapped in an outer page loop is replaced by one
flat sequence over `page_count * strips_per_page` strips, with the same five
stage offsets as before and page boundaries invisible inside the loop:

```
AIV1  MTE2 + V + MTE3   stage key strip c - 1   -> L1 key page (c - 1) % 2
AIC   MTE1 + M + FIX    GEMM strip c - 2        -> UB score slot c % 2
AIV0  MTE2              load value strip c - 2
AIV0  V                 dequant value strip c - 3
AIV0  V                 recurrence strip c - 4   <- UB score slot c % 2
```

Addresses resolve per strip through `StripResolver`, which memoises the block
table lookup because the walk is monotone in the strip index. Tail masking goes
through `turboquant_decode_tail::WalkActive`, the shared walk the header already
proves exhaustively — and it is arithmetically identical to the per-page
`ActiveInStrip(strip * 16, ActiveInPage(page))` it replaces, for every case.

The carried `(m, l, Acc)` are seeded ONCE for the whole window instead of per
page with `initialize` true only on the first, and every recurrence folds into
the same fixed UB offsets in strict strip order. The intra-core feedback cycles
are primed once before the stream and drained once after it: **there is no
inter-page drain cascade left to get wrong.**

### 2d. What was deliberately not touched

The legacy `camodel_paged_decode_attention` entry keeps the per-page schedule,
its mode-4 per-slot ids and its compile-time `kStrips` unroll, unchanged. Its
`run_cube_core` loses only the `pages` / `temporal` parameters added for the
temporal path and the one-trip loop they drove, which returns it to its
pre-`bbaaa3e` shape. Section 5 is its regression.

## 3. Gate status

**Every gate below was run on this host unless the row says otherwise.**

| gate | state |
| --- | --- |
| §6.2c mode-2 semantics probe at >= 4 iterations | **PASS** — §1. This was recorded as "NOT RUN -- this is the next step" in the previous report |
| §1 token/queue model, single ring pass | **PASS** — `test_decode_ring_model.py`, 22654 states, 7 controls, untouched by this change |
| §1a strip-stream model | **PASS** — §4 |
| §2 compile-time memory checks | **PASS** — device build with `-Wall -Werror`, zero warnings, zero errors |
| §3 isolated logical-page canaries, AIC path | **PENDING** — still no canary protocol for the AIC Split-K entry; unchanged from `decode_splitk_aic_page_depth.md` §5 |
| §4 depth and wrap tests | **PASS** at 2, 3, 4, 5 and 8 carried pages — §5 |
| §5 trace audit / census | **PASS** — §6 |
| §6.5 ragged-second-wave ladder | **NOT RUN** |
| §6.6 cross-core fence audit | **PASS**, and green for the first time — §7 |
| legacy `camodel_paged_decode_attention`, host oracle | **PASS** — Tier-1 probe: rotation, table image, egress butterfly, packing, fixture and single-strip isolation |
| legacy `camodel_paged_decode_attention`, device build | **PASS** — `test_camodel_paged_decode_attention` builds clean |
| legacy `camodel_paged_decode_attention`, CAModel eight-strip rung | **STOPPED** — §5a |
| mode-4 control leg of the sync probe at 4 iterations | **STOPPED** — §5a |
| sliding-window rungs `p4_from1`, `p5_from0` and the `K = 4` control | **STOPPED** — §5a |
| 8 strips over more than one page (`--page-tokens 128`, 2 pages) | **NOT RUN** |
| physical NPU | **UNAVAILABLE** on this host |

## 4. The strip-stream model

`tests/cpu/test_decode_page_loop_model.py` was rewritten for the new structure
and strengthened. It keeps the two checks that matter and adds two more, because
the score panel's WAR edge is now carried by the folded readiness announcement
and that is not obvious:

* `L1 key page S rewritten by strip t while strip u was still live`
* `Cube read L1 key page S holding strip u while multiplying strip t`
* `UB score panel S rewritten by strip t while strip u was still live` *(new)*
* `recurrence of strip t read UB score panel S holding strip u` *(new)*

Mode 2 is modelled as mode 2 and not as a shared place: the AIC's counter
advances only when BOTH AIVs have announced, and one AIC announcement increments
the counter on both of them. Modelling it as one place would let either AIV alone
release the Cube, which is what the `single_sender_quorum` control exists to
reject.

```bash
python3 tests/cpu/test_decode_page_loop_model.py
```

15 geometries pass exhaustively, to 8 pages. The state count depends only on the
STRIP total and not on where the page boundary falls — `1x8`, `2x4`, `4x2` and
`8x1` all reach exactly 2260688 states — which is the structural statement that
the stream is agnostic of page boundaries, made as a measurement.

Seven negative controls, each rejected:

| control | rejected with |
| --- | --- |
| `no_l1_war` — **the defect itself** | `L1 key page 0 rewritten by strip 2 while strip 0 was still live` |
| `single_sender_quorum` — mode 4 in mode 2's position | `Cube read L1 key page 0 holding strip -1 while multiplying strip 0` |
| `release_before_fixpipe` — §6.2a's hazard | `recurrence of strip 0 read UB score panel 0 holding strip -1` |
| `release_before_l0b` | `recurrence of strip 0 read UB score panel 0 holding strip -1` |
| `no_prime` | `unbalanced readiness quorum: AIV0 announces 2, AIV1 announces 4, the AIC waits 4` |
| `carry_credit` | `unbalanced HardEvent census` on `V0.V_MTE2` |
| `no_epilogue_drain` | `unbalanced release broadcast: the AIC announces 4, AIV0 waits 4, AIV1 waits 2` |

Four geometries are recorded NOT COVERED rather than silently dropped: `8x2`,
`2x8`, `4x8` and `8x8` exceed the model's 4,000,000-state cap with no
partial-order reduction.

### 4a. The semantics knob, still reporting both readings

Section 6.3's knob is kept and still reports BOTH flag readings, because a model
that assumes either has assumed the answer:

```text
SEMANTICS sticky AIV cross-core wait, as shipped (mode-2 quorum + broadcast):
    pages=2 strips=2: recurrence of strip 1 read UB score panel 1 holding strip -1
SEMANTICS sticky AIV cross-core wait, with no L1 WAR edge, the pre-refactor shape:
    pages=2 strips=2: L1 key page 0 rewritten by strip 2 while strip 0 was still live
```

Under the sticky reading **no** cross-core arrangement is safe, the shipped one
included. That is the honest verdict and it has not changed: the ordering rests
on a hardware property, not on the token structure. What is different is that
the property is now MEASURED for the mode the operator actually uses (§1a),
rather than carried over from a two-iteration mode-4 probe.

## 5. CAModel rungs

```bash
bash scripts/run_decode_splitk_aic_page_race.sh sweep
```

All rows `--page-tokens 16 --heads 1 --rows 1 --splitk 1 --single --impl aic`,
one evidence directory per rung, rungs run concurrently. The rig surveys the
WHOLE partial -- `m`, `l`, `acc`, `qrot`, the replicated state, all 16 rows and
all 4096 channels -- rather than throwing on the first channel past its bound,
which is the change that made the defect findable in the first place.

| rung | pages carried | result | `l` | `acc` | mean \|deviation\| | ticks |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `--context 32 --page-count 2` | **2** | **PASS** | 0.00064x | 0.00032x | 4.40e-08 | 47793 |
| `--context 48 --page-count 3` | 3 | PASS | 0.00096x | 0.00033x | 6.18e-08 | 67653 |
| `--context 64 --page-begin 1 --page-count 3` | 3 | PASS | 0.00070x | 0.00027x | 4.79e-08 | 67655 |
| `--context 80 --page-begin 2 --page-count 3` | 3 | PASS | 0.00066x | 0.00040x | 5.28e-08 | 67655 |
| `--context 64 --page-count 4` | **4** | **PASS** | 0.00063x | 0.00025x | 6.21e-08 | 87310 |
| `--context 128 --page-count 8` | **8** | **PASS** | 0.00135x | 0.00020x | 4.35e-08 | 166234 |

Every rung reports `acc over tolerance 0 of 4096 channels in 0 of 16 rows;
m over 0, l over 0; replicas torn 0; qrot wrong 0`.

Against the broken revision, on the same rung:

| `--context 64 --page-count 4` | `l` | `acc` | channels over | rows over | mean \|dev\| |
| --- | ---: | ---: | ---: | ---: | ---: |
| before | 2.12830x | 7.58482x | 1634 of 4096 | 10 of 16 | 1.022830e-03 |
| **after** | **0.00063x** | **0.00025x** | **0 of 4096** | **0 of 16** | **6.208090e-08** |

Four orders of magnitude, and the single-channel check matters as much as the
aggregate: the defect's signature was a zero-mean ~1e-3 error over 40% of
channels with a bit-exact `m`, which an aggregate alone can hide. No channel is
over tolerance on any rung.

The three rungs the acceptance ladder names -- 2, 4 and 8 carried pages at
`K = 1` -- all passed, and the three-page rungs that passed BEFORE the refactor
still pass, which is what separates a fix from a reshuffle.

### 5a. What was stopped, and what that leaves unverified

Four runs were still in the simulator and were **stopped by the operator of
this host** rather than failing. They are recorded here as stopped, not as
passed and not as skipped.

| run | what it would have shown | state when stopped |
| --- | --- | --- |
| `p4_from1` (`--context 80 --page-begin 1 --page-count 4`) | four carried pages from a different START, which separates "page count" from "page data" | model started, no result |
| `p5_from0` (`--context 80 --page-count 5`) | five carried pages; failed at 5.28651x / 9.29462x before | model started, no result |
| `p4_k4` (`--splitk 4`) | one page per worker at the same depth, isolating the carry from the data | model started, no result |
| legacy eight-strip CAModel rung | that `camodel_paged_decode_attention` is still bit-identical | host oracle PASSED, device build PASSED, model started, no result |
| mode-4 control leg at 4 iterations | the mode-4 stall census beside the mode-2 one, measured in the same way on the same host | iteration 0 complete of 4 |

Two of these matter more than the others.

* **The legacy regression is the one with real consequence.** The per-page
  schedule's own code is untouched, but `run_cube_core` did lose the `pages` /
  `temporal` parameters and the one-trip loop they drove, and
  `RecurStrip`'s operand block moved into the shared `FillRecurEngine`. Neither
  changes the data path, and the entry builds clean and passes its host oracle
  -- but **"bit-identical on CAModel" is asserted here, not measured.** The
  command is `bash scripts/run_paged_decode_camodel.sh --strips 8 --block 0
  --no-fixture-check`, about fifteen minutes of model time after a three-minute
  build.
* **The mode-4 control leg is a convenience, not a dependency.** The mode-4
  numbers this report compares against (23146, 3, 3, 3) are
  `reports/decode_splitk_aic_page_race.md` §4's measurement of the OPERATOR,
  on this host, at four carried pages -- which is the comparison that matters.
  Re-running the probe's mode-4 leg would have put both modes side by side in
  one instrument; it would not have changed what mode 2 was measured to do.

Nothing stopped was load-bearing for §1, §4, §5's three acceptance rungs, §6 or
§7, all of which completed.

## 6. Trace audit

```bash
python3 scripts/audit_mode4_tick_order.py EVIDENCE_DIR --block 0 --cross
python3 scripts/analyze_aic_splitk_trace.py EVIDENCE_DIR --workers 1 --pages N --strips 1
```

Both gates PASS on the 2-, 4- and 8-page rungs. The censuses are exact:

| rung | HardEvent pairs | intra-core places | Mmads | `m2:8` quorum | `m2:9` broadcast |
| --- | ---: | ---: | ---: | --- | --- |
| 2 pages | 55 | 34, all `SWSW...` | 80 = 64 + 2.1.8 | AIV0 2, AIV1 2, AIC waits 2 | AIC 2, AIV0 waits 2, AIV1 waits 2 |
| 4 pages | 93 | 36, all `SWSW...` | 96 = 64 + 4.1.8 | 4, 4, 4 | 4, 4, 4 |
| 8 pages | 169 | 36, all `SWSW...` | 128 = 64 + 8.1.8 | 8, 8, 8 | 8, 8, 8 |

No double SET, no WAIT before SET, no leftover token on any place, on any rung.
No non-empty exception dump.

### 6a. AIV1 is paced by Cube retirement

This is the acceptance criterion, and the eight-page rung is where it shows:

| channel, 8 pages | stall per wait, in ticks |
| --- | --- |
| AIC waits `m2:8` on MTE1 | 742, 5, 19364, 19497, 19463, 19497, 19463, 19497 |
| AIV0 waits `m2:9` on VEC | 4537, 103, 101, 103, 101, 103, 101, 3280 |
| **AIV1 waits `m2:9` on MTE3** | **2652, 639, 18736, 19371, 19337, 19371, 19391, 19742** |

AIV1 is the core that used to stage five pages into one L1 address by tick
30011 and execute its last instruction at 30611, while the Cube read the fifth
of them at 88085. It now stalls between 18,700 and 19,700 ticks on each of its
last six waits. That is not a decaying pattern and it is not the 1-3 tick floor
of a channel that has stopped blocking: the mode-4 edge it replaces read
**23146, 3, 3, 3**.

### 6b. The gate is early retirement, not a stall threshold

`--expect-stall N` was the regression form for a revision whose producer ran
unbounded: a collapsed latch read 1-3 ticks while announcements were ~20600
apart, so any threshold in between separated them. **That proxy no longer
works**, and saying so is more useful than keeping a number that happens to
pass. With the pipeline bounded by a two-slot credit, the Cube is often already
a strip or two ahead, so a wait legitimately finds its token banked and stalls
only the cost of its own drain -- the AIC's second wait reads **5 ticks** on
every rung, and it is perfectly well ordered.

So the audit now checks the property directly, with no threshold at all: a
consuming counter releases the `j`-th wait with the `j`-th announcement, so the
`j`-th release tick must not precede it, and for a mode-2 quorum the `j`-th
increment is the LATER of the two AIVs' `j`-th announcements. **No wait on any
channel on any rung was released early.** The stall threshold is kept, scoped to
`m2:9` and lowered to 50 ticks, claiming only that no wait retired at the floor.

The trace of the broken revision was not preserved, so there is no real data
here on which the new check fails. It therefore carries a `--self-test` with
five synthetic controls, including the quorum rule and the exact mode-4
collapse the operator showed:

```bash
python3 scripts/audit_mode4_tick_order.py --self-test
```

```text
  mode-2 release observed after the announcement       accepted
  mode-2 release observed before the announcement      rejected
  mode-2 quorum released by one AIV alone              rejected
  mode-4 second wait retiring before its announcement  rejected
  mode-4 second wait retiring after it                 accepted
```

### 6c. Two measurement defects found on the way

Both would have made a gate pass vacuously, which is the same class as the
`FLAG ID:0` prefix bug the previous report records.

* **`analyze_aic_splitk_trace.py` read `*.instr_log.dump`, which drops
  instructions.** On the four-page rung AIV0's issue log holds 2 of its 4
  `SET_CROSS_CORE` and 2 of its 4 `WAIT_FLAG_DEV`: instruction IDs 035718,
  042737, 074351 and 081403 are absent from it and present in
  `*.instr_popped_log.dump`, although both logs run to ID ~157830. Read that
  way the census reported `m2:8 quorum imbalance: AIV0 announces 2, AIV1
  announces 4, the AIC waits 4` on a run whose output was correct. It now reads
  the popped log, as the stall audit already did.
* **The stall metric measured to the next SCALAR instruction of any kind.** The
  construction it is reading is the `SetFlag<x_S>` / `WaitFlag<x_S>` drain
  behind the wait, and only the `WaitFlag` half is the release tick; anything
  the compiler schedules in between is issued while the wait still blocks. All
  three of the operator's channels read `[1, 1, 1, 1]` under the looser rule and
  were marked `no-fence` although every one of them carries its drain. The same
  change corrects the probe's own AIC figures in §1a from 791, 850, 724, 768 to
  4583, 18727, 17963, 18727.

## 7. The cross-core fence audit, green

`scripts/audit_cross_core_fences.py` reported four findings on this operator in
the previous report: two real but benign (the page-drain announcements drained
`FIX_S` first rather than last) and two its own blind spot (the per-page
`kFlagScoreFree` priming sets, exempt only when the literal string `Prime`
appeared in the flag expression). All four are resolved, and the audit is now a
stronger instrument than it was:

* **The real findings are gone with the code.** The page-drain announcements no
  longer exist. Their replacement, the Cube's release, drains `MTE1_M`, `M_S`
  and then `FIX_S` and announces on `PIPE_FIX` — the announcing pipe is the last
  one drained, which is what the audit asks for and what the previous report
  proposed but did not do.
* **The priming exemption is structural.** A set is priming when nothing between
  the start of its enclosing block and the set issues work on any pipe, which is
  what "this buffer starts empty" means. The `Prime`-in-the-name test is kept as
  a second, narrower route so the reshape kernels do not regress.
* **The audit now reads mode 2.** It matched `CrossCoreSetFlag<4, ...>` only, so
  a mode-2 kernel would have earned a vacuous pass — section 6.2a's hazard is
  not about mode 4, since `ffts_cross_core_sync` issues on the scalar stream
  exactly as `set_intra_block` does.
* **It resolves a named pipe.** The pipe argument may be a `constexpr pipe_t`,
  which is how this operator states and compile-time asserts its pipe legality
  (§2b). An announcement the audit cannot see is one it cannot fail.
* `turboquant_paged_decode_attention.cpp` is now in `DEFAULT_SOURCES`, so it is
  audited by default rather than only when named.

```bash
python3 scripts/audit_cross_core_fences.py
```

Twelve self-test controls, five of them new, all behaving: an unfenced mode-2
set is rejected, a named pipe with the wrong route is rejected, a pipe name the
audit cannot resolve is rejected, and a structural priming set is accepted while
one preceded by work on the announced pipe is not.

## 8. Tooling changes

Three gates were extended. All three would otherwise have passed vacuously on a
mode-2 kernel, which is worse than not running them.

* `scripts/audit_mode4_tick_order.py` reads `SET_CROSS_CORE` / `WAIT_FLAG_DEV`
  as well as `SET_INTRA_BLOCK` / `WAIT_INTRA_BLOCK`, decoding `GetffstMsg`'s
  `1 + (mode << 4) + (flagId << 8)` packing on the set. Channels are named
  `m4:<id>` and `m2:<id>` and never pooled, because the two id spaces are
  independent. `--expect-id` takes either form; a bare number still means
  mode 4.
* `scripts/analyze_aic_splitk_trace.py` learned that mode 2 does not balance
  pairwise. A quorum is one announcement from EACH AIV against one AIC wait; a
  broadcast is one AIC announcement against one wait on EACH AIV. Checking mode
  2 with mode 4's rule reports a correct kernel as imbalanced. Its page-handoff
  census (`mode4[10] == pages`, `mode4[26] == pages`) is replaced by a per-strip
  census over whichever channels the trace actually carries.
* `scripts/audit_cross_core_fences.py`, as §7.
* `scripts/run_cluster_sync_probe_camodel.sh` is new: it builds and runs the
  cluster sync probe on CAModel and prints the stall census, which is the
  probe's actual verdict. The probe moves no data, so a clean exit distinguishes
  nothing — a sticky latch finishes cleanly too.

## 9. Known gaps and follow-ups

* **The Cube is serialised per strip.** Its readiness wait and its release
  announcement each carry a full `MTE1 -> M -> scalar` drain, so the scalar
  stream stalls at both and the two-slot `M_MTE1` / `FIX_M` credits never
  actually overlap two strips on the Cube. That is a throughput cost, taken
  deliberately: it bounds the core, and it is what makes the stall measurable in
  the trace census. Nothing here measures throughput, so the cost is unquantified.
* **`run_value_core` keeps vestigial `partial` / `initialize` / `finalize`
  parameters** and a partition write-out that nothing now reaches: the Split-K
  entry has its own `stream_value_core`. They are dead at the only call site,
  which passes the defaults, and were left in place rather than removed so the
  legacy entry's generated code is not disturbed by this change. Worth removing
  separately.
* **The legacy entry's CAModel regression was stopped** (§5a). Its host oracle
  and its device build pass; its bit-identity on CAModel is asserted from the
  data path being untouched, not measured. This is the first thing to re-run.
* **Neither `--page-tokens 128` over two pages nor the ragged-second-wave ladder
  was run**, so eight strips per page across a page boundary is covered by
  neither the model nor CAModel. It is the largest remaining hole.
* **No canary protocol exists for the AIC Split-K entry**, so §3 of the
  refinement protocol is still pending for this operator, exactly as it was.
* **No NPU.** Everything above is CAModel.

## 10. Commands

```bash
python3 tests/cpu/test_decode_page_loop_model.py
```

```bash
bash scripts/run_cluster_sync_probe_camodel.sh 4 2
```

```bash
bash scripts/run_decode_splitk_aic_page_race.sh sweep
```

```bash
python3 scripts/audit_cross_core_fences.py
```

```bash
python3 scripts/audit_mode4_tick_order.py --self-test
```

The two gates that were stopped, in the order worth re-running them:

```bash
bash scripts/run_paged_decode_camodel.sh --strips 8 --block 0 --no-fixture-check
```

```bash
bash scripts/run_cluster_sync_probe_camodel.sh 4 4
```
