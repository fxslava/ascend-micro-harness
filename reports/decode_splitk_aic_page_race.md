# The four-page defect in `turboquant_paged_decode_splitk_aic`

Target: CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599`. No NPU on this host
(`npu-smi` and `/dev/davinci*` both absent), so every device result below is
CAModel over a collapsed grid and **nothing here is a hardware claim**.

This closes the open finding of `reports/decode_splitk_aic_page_depth.md` §3.
That report attributed the four-page deviation to "a discrete event at the
third page boundary" affecting the value side, on the strength of `l` staying
inside its bound while `acc` did not. `l` does not stay inside its bound; the
rig threw on the first `acc` channel of row 0 before it ever checked rows 1-15,
so the premise of that narrowing was an artefact of the check order.

## 1. What was asked, and what the source says about it

The task supplied a diagnosis to implement: that `kFlagPageDrainedV0` is a
one-way drain signal the Cube can overrun, overwriting an unconsumed mode-4
event and corrupting the score ring; that `run_value_core` returns without
draining `PIPE_V` on a non-finalised page; and that the remedy is a reciprocal
`kFlagPageAckV0`/`kFlagPageAckV1` acknowledgement.

**Both hazards are refuted, and the acknowledgement edge already exists in a
different form.** Neither was implemented.

* **The score ring cannot be corrupted by the Cube outrunning AIV0.** The
  Cube's second `CrossCoreSetFlag(kFlagPageDrainedV0)` is transitively ordered
  after AIV0's first wait on it. Page `p + 1`'s `StepColumnsCube` cannot write
  a score window without `kFlagScoreFree<Slot>`, and those credits are issued
  by AIV0 only after it has passed the page-`p` drain wait. So the Cube's
  per-page announcement is already acknowledged -- by the score-free credit
  ring, which is what a back edge would duplicate.
* **`PIPE_V` is drained at that boundary.** `run_value_core` returns, and the
  caller then issues `CrossCoreWaitFlag<4, PIPE_V>(kFlagPageDrainedV0)`,
  `SetFlag`/`WaitFlag<V_S>` and `SetFlag`/`WaitFlag<MTE2_S>`. Every queue AIV0
  primes on is drained to the scalar stream before the next page primes it.
* The trace agrees with the source. At `pages=4, K=1`, all **33** intra-core
  HardEvent places alternate `SET, WAIT, SET, WAIT` with no double SET, no
  early wait and no leftover token, and the mode-4 census balances per id with
  a Mmad count of exactly `64 + 4*1*8 = 96` (§5 below). The page-boundary
  drain added by `bbaaa3e` is confirmed working at four pages.

The reported margins in the task (`l` 2.13x, `Acc` 7.58x) are real and are
reproduced here. Their cause is none of the above.

## 2. The measurement that localised it

`AicRig::Validate` was changed to survey the whole partial instead of throwing
on the first channel past its bound, and to report where the failure lives.
That one change is what made the defect findable:

```text
AIC partial margins (fraction of each tolerance): m=0.00000x l=2.12830x acc=7.58482x
    at item=0 row=10 c=4 got=-0.0555238128 ref=-0.0491819838
AIC partial profile: acc over tolerance 1634 of 4096 channels in 10 of 16 rows;
    m over 0, l over 2; replicas torn 0; qrot wrong 0;
    mean signed deviation -1.492294e-07, mean |deviation| 1.022830e-03
```

Four facts from that line did the work:

| fact | what it excludes |
| --- | --- |
| `m` over 0, at margin `0.00000x` | every row's score MAXIMUM is bit-exact |
| `qrot wrong 0` | the ingress rotation is bit-exact |
| `replicas torn 0` | the replicated-state invariant holds |
| mean signed `-1.5e-07` vs mean \|dev\| `1.0e-03` | the error is symmetric, so not a scale, a rescale factor or a biased `Exp` |

A zero-mean error of fixed size across 40% of channels, with `m` exact, is the
shape of a wrong contribution from ONE page -- not drift, and not `alpha`.

## 3. The page sweep: it is the page COUNT, and only the count

All rows `--page-tokens 16 --heads 1 --splitk 1 --impl aic`, CAModel:

| rung | pages carried | result | `l` | `acc` | mean \|deviation\| |
| --- | ---: | --- | ---: | ---: | ---: |
| `--context 48 --page-begin 0 --page-count 3` | 3 | PASS | 0.00096x | 0.00033x | 6.18e-08 |
| `--context 64 --page-begin 1 --page-count 3` | 3 | PASS | 0.00070x | 0.00027x | 4.79e-08 |
| `--context 80 --page-begin 2 --page-count 3` | 3 | PASS | 0.00066x | 0.00040x | 5.28e-08 |
| `--context 64 --page-begin 0 --page-count 4` | 4 | **FAIL** | **2.12830x** | **7.58482x** | 1.02e-03 |
| `--context 80 --page-begin 1 --page-count 4` | 4 | **FAIL** | **5.82813x** | **8.13241x** | 8.54e-04 |
| `--context 80 --page-begin 0 --page-count 5` | 5 | **FAIL** | **5.28651x** | **9.29462x** | 1.33e-03 |

Three pages pass wherever they start; four and five fail wherever they start.
`m` is `0.00000x` and `qrot wrong 0` in every failing rung. So this is neither
the data of any particular page nor accumulation over depth -- it is a
threshold at four carried pages. `--page-tokens` accepts only 16 or 128, so the
"same 64 tokens as one page of four strips" control is not reachable at this
geometry and was not run.

## 4. The trace, on a common clock

`*.instr_popped_log.dump` logs a `WAIT_INTRA_BLOCK` at the tick it was
**issued**, not the tick it was released, which is why a naive tick-order
comparison reports artefacts: AIV0's wait on `kFlagScoreReady` is logged before
the Cube's set on every page, and the panel it then reads is bit-exact. What
the release tick can be read from is the scalar drain behind the wait, because
a mode-4 wait blocks only its own pipe's issuance and the `x_S` pair behind it
stalls instruction issue until that pipe retires past it.

Read that way, on the four-page `K=1` run (`scripts/audit_mode4_tick_order.py`):

| channel | stall per wait, in ticks |
| --- | --- |
| Cube waits `kFlagScoreFree<0>` on FIXP | 941, **19397**, 1229, **19397**, 1083, **19396**, 1090, **19396** |
| Cube waits `kFlagScoreFree<1>` on FIXP | **19396, 19396, 19395, 19395** |
| **AIV1 waits `kFlagPageDrainedV1` on MTE3** | **23146, 3, 3, 3** |

The Cube's four announcements of that id are at ticks 26337, 47008, 67532 and
88068 -- about 20600 apart. **AIV1's first wait blocked for 23146 ticks and its
next three for three ticks each.** AIV1 therefore executed its last instruction
at tick 29611, while the Cube had not yet announced pages 1, 2 or 3 as drained.

The same run shows the construction that does work, on the same hardware, in
the same block: AIV0's `CrossCoreWaitFlag<4, PIPE_V>` on `kFlagScoreReady` was
released by the Cube's set on all four pages, and its four recurrence spans are
19386, 19386, 19385 and 19385 ticks, each beginning exactly at the Cube's set
(6929, 27600, 48125, 68661). A wait that had stopped blocking would have
started earlier and produced longer spans.

## 5. Root cause

`kL1KeyPage<Strip>` gives every strip its own L1 key page, and the `kL1End`
comment removes the per-strip L1 WAR back edge on the grounds that "every
strip writes its own page and every page is read once, so there is no reuse for
a back edge to order."

**That premise holds inside one sequence page and fails across the temporal
page loop.** Page `p + 1`'s strip `s` restages `kL1KeyPage<s>` over page `p`'s.
The reuse is back, and the only edge standing where the removed back edge used
to be is `kFlagPageDrainedV1` -- the one §4 measures as not ordering its
consumer after the first page.

So AIV1 staged all four pages into the one L1 page that strip 0 owns, at ticks
3199, 27338, 28343 and 29313, finishing at 29611. `kFlagKeyReady` is a counter
(0-15) and legitimately banks unconsumed announcements, so the Cube's third and
fourth waits on it were satisfied immediately from the queue, and its L1 -> L0B
loads for sequence-pages 2 and 3 both read whatever had been written last --
page 3's keys.

### The arithmetic confirms the timeline, as a prediction

The substitution above was derived from the trace alone. Modelling exactly it
on the host -- sequence-page 2 reads page 3's key nibbles, while its key scale,
value page and value scales stay its own, because those reach AIV0 straight
from Global Memory and never pass through L1 -- gives:

| model | `acc[row 0][c 0]` | distance from the device |
| --- | ---: | ---: |
| FP64 oracle | -1.14959846 | 8.967e-04 |
| FP32, no fault | -1.1495986 | 8.967e-04 |
| **page 2 reads page 3's keys** | **-1.15049529** | **1.21e-07** |
| pages 1 and 2 read page 3's keys | -1.1505549 | 5.97e-05 |
| pages 0, 1 and 2 read page 3's | -1.15059018 | 9.50e-05 |
| page 1 reads page 2's keys | -1.14971828 | 7.77e-04 |
| `alpha` skipped at page 3 | -1.16276979 | 1.23e-02 |
| `alpha` applied twice at page 3 | -1.13656569 | 1.39e-02 |
| page 3 reads a stale value page | -1.20444441 | 5.40e-02 |
| page 3 reads a stale value scale | -1.05659282 | 9.39e-02 |

The device measured `-1.15049517`. The predicted model agrees to **1.21e-07**,
which is the FP32-versus-FP64 noise floor the clean model already carries
(1.43e-07). No other candidate is within three orders of magnitude. This is a
confirmed prediction, not a fit: the key-page substitution came from the tick
timeline before any of these numbers were computed.

It also explains each surviving fact. `m` is the maximum over all 64 tokens and
the fixture ramps the key-scale exponent one per page, so `m` comes from the
last page -- the one page that is always read correctly, being the last
written. `l` and `acc` both take a wrong contribution from page 2 only, which
is why `l` is over in 2 rows of 16 rather than all of them. And
`acc[c] = sum_t p_t . vs_t . V[c][t]` with `V` in [-7, 7] projects that
perturbation onto the value columns with no systematic sign, which is the
zero-mean signature of §2.

Why three pages pass: AIV1 blocks on its first page-drain wait and then runs
free, so it gets about three pages ahead. With three pages it runs out of work
before it can overtake the Cube's read of any page; with four it does not.

## 6. The attempted fix, and why it does NOT work

**This section describes a change that was made, measured, and found not to fix
the defect. It is in the tree. The defect is still open.**

One line of behaviour, in the AIV1 page loop of
`turboquant_paged_decode_splitk_aic`: the page-boundary wait moves from
`PIPE_MTE3` to `PIPE_V`, and the `V_S` scalar drain moves to sit immediately
behind it.

```cpp
AscendC::CrossCoreWaitFlag<4, PIPE_V>(kFlagPageDrainedV1);
AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtPageDrain);
AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtPageDrain);
AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtPageDrain);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtPageDrain);
AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(kEvtPageDrain);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(kEvtPageDrain);
```

Two reasons, and the second matters independently of the first. A mode-4 wait
blocks only the issuance of its own pipe
(`CrossCoreWaitFlag_ISASI.md`: "CrossCoreWaitFlag blocks the subsequent
instructions of the **specified pipeline**" on Ascend 950PR), so the drain that
converts it into a stall on instruction issue has to be on the same pipe as the
wait. And `PIPE_V` is the construction measured to block on every page in §4,
on the same core type, in the same run. Draining V first also holds page
`p + 1`'s **MTE2 ingress** behind the announcement, which gating MTE3 never did
even when the wait did block.

This is the same construction AIV0 already uses at the same boundary, so the
two Vector cores now close the boundary the same way.

## 7. After the change: no effect, measured

The kernel binary changed (`md5` 91244d0b -> 542a54db) and the trace confirms
the wait moved to VEC. The numbers did not move at all:

| rung | `l` | `acc` | `acc[0][0]` | mean \|dev\| |
| --- | ---: | ---: | ---: | ---: |
| `--context 64 --page-count 4`, before | 2.12830x | 7.58482x | -1.15049517 | 1.022830e-03 |
| `--context 64 --page-count 4`, after | 2.12830x | 7.58482x | -1.15049517 | 1.022830e-03 |
| `--context 80 --page-count 5`, after | 5.28651x | 9.29462x | — | 1.333085e-03 |
| `--context 80 --page-begin 1 --page-count 4`, after | 5.82813x | 8.13241x | — | 8.541666e-04 |

Bit-identical, to the tick (89581 -> 89577). The stall census says why:

```text
core0.veccore1 waits id 26 on VEC   stall collapsed after the first wait  [23146, 2, 2, 2]
```

**The pipe was not the cause.** On this toolkit an AIC -> AIV mode-4 wait
blocks on its FIRST use of an id and then stops blocking, on `PIPE_MTE3` and on
`PIPE_V` alike. The change is still the better construction -- draining V
immediately behind the wait holds page `p + 1`'s MTE2 ingress too, which gating
MTE3 never did -- but it fixes nothing and is not presented as a fix.

### Why no arrangement of these flags fixes it

AIV1's run-ahead is unbounded. Its L1 write bursts for five pages are at ticks
2892, 27035, 28040, 29010 and 30011 -- it stages every page about 1000 ticks
apart and finishes at 30611, while the Cube reads the fifth at 88085. The one
wait that blocked is its first, and that is an accident of the first use of the
id.

So a deeper L1 key-page ring cannot help either: there is no bound to size it
against. Nor can cycling several ids, since each id's first use is the only one
that blocks. The page-loop model now says the same thing from the token side:

```text
SEMANTICS sticky AIV mode-4 wait, as shipped (page fence on PIPE_V):
    pages=2: SET into a full place ('cross', 'ScoreFree', 0) (bound 1)
SEMANTICS sticky AIV mode-4 wait, pre-fix (page fence on PIPE_MTE3):
    pages=2: SET into a full place ('cross', 'ScoreFree', 0) (bound 1)
```

Under that reading neither construction is safe, which is the honest verdict:
the ordering rests on a hardware property, not on the token structure.

### What is open, and what settles it

`MICROROUTINE_DESIGN_GUIDELINES.md` §6.2c retracted an earlier "mode-4 is not a
counting semaphore" finding and recorded the consuming reading as measured, on
the strength of `tests/research/kernel_cluster_sync_probe.cpp` over **two**
announcements per id. This operator needs four and five, and at that depth the
AIC -> AIV direction does not consume. **Re-running that probe at four or more
iterations is the next step and was not run here.** Its two targets
(`test_cluster_sync_probe` on silicon, `test_cluster_sync_probe_sim` on
CAModel) already exist; `build/lib/test_cluster_sync_probe_sim` is built.

Candidate remedies, none implemented or measured:

* a mode-2 rendezvous for the page boundary instead of mode 4 -- a separate
  16-id space whose AIC -> AIV semantics may differ, and the construction the
  task's "reciprocal acknowledgement" would actually need;
* the Cube staging the key page itself, removing the AIV1 -> L1 -> Cube hop and
  the WAR edge with it;
* single-shot ids per page boundary with a hard guard at 11 pages, which is
  §6.2c's documented fallback and does not generalise.

## 8. Gate status

**The operator is not fixed. Do not tag it for physical hardware.**

| gate | state |
| --- | --- |
| §1 token/queue model, single ring pass | not re-run; untouched by this change |
| §1a page-loop model | **PASS** under the consuming reading: 13 geometries (to pages=6), empty final marking, ordered carried state, 6 negative controls rejected, 2 geometries recorded NOT COVERED. **FAIL under the sticky reading**, for both constructions |
| §2 compile-time memory checks | device build with `-Wall -Werror`, no new warnings |
| §3 isolated logical-page canaries, AIC path | **PENDING** -- still no canary protocol for the AIC Split-K entry; unchanged from `reports/decode_splitk_aic_page_depth.md` §5 |
| §4 depth and wrap tests | **FAIL** at 4 and 5 carried pages, before and after the change (§3, §7) |
| §5 trace audit / census | RUN; HardEvent and mode-4 balance PASS, Mmad census exact, intra-core capacity-one order PASS on all 33 places, cross-core stall census **FAIL** (§4) |
| §6.5 ragged-second-wave ladder | **NOT RUN** |
| §6.6 cross-core fence audit | RUN for the first time on this operator; 4 findings, assessed below |
| §6.2c mode-4 semantics probe at >= 4 iterations | **NOT RUN** -- this is the next step |
| 8 strips over more than one page (`--page-tokens 128`, 2 pages) | **NOT RUN** |
| physical NPU | UNAVAILABLE on this host |

### §6.6, run for the first time on this operator

`scripts/audit_cross_core_fences.py` had `DEFAULT_SOURCES` covering only the
two reshape-and-cache kernels, so it had never seen this one. It reports four
findings; two are the audit's own blind spot and two are real but benign:

* lines 1063-1064, `kFlagScoreFree<0>`/`<1>` with "no V_S drain": these are the
  per-page **priming** sets, which §6.2a exempts. The audit recognises an
  exemption only by the literal `Prime` in the flag expression, which these do
  not carry. For pages after the first the drain does exist, in the caller's
  page-boundary fence, just not lexically adjacent. **False positive**; the
  audit's exemption test is a name match and should be a structural one.
* lines 1354-1355, `kFlagPageDrainedV0`/`V1`: "drained M_S but announces on
  PIPE_FIX; expected FIX_S". `FIX_S` **is** present -- it is the first of the
  three drains -- but the audit requires the announcing pipe's drain to be the
  last one before the set. Nothing issues into FIX between that drain and the
  announcement, so the announcement is fenced in substance. Reordering the
  three to `MTE1_M`, `M_S`, `FIX_S` would make the gate green and weaken
  nothing; it was **not** done here because it changes the instruction stream
  that `bbaaa3e` verified and would need its own CAModel run.

## 9. Tooling defects found on the way

Two of the gates were not measuring what they reported.

**`FLAG ID:0` prints without the `0x` prefix.** `analyze_aic_splitk_trace.py`
matched `FLAG ID:0x([0-9a-f]+)`, so every HardEvent place on event id 0 was
invisible to it -- 56 events on the Cube alone, and at one strip per page slot 0
is the ONLY slot doing real work. Its reported `hardevent_pairs` went from 61 to
**137** once the prefix was made optional, so the §5 census had been auditing
under half the places, and the balance check passed vacuously on the rest. Fixed.

**Nothing checked cross-core stall.** `scripts/audit_mode4_tick_order.py` is new
and is what found §4. It audits intra-core places for capacity-one order and
cross-core channels for the stall each wait actually imposed, and it fails a
channel whose first wait blocks and whose later waits on the same id do not
while the producer's announcements are far apart. `--expect-stall N` is the
regression form. A channel with no scalar drain behind its wait reports a floor
rather than a stall and is marked `no-fence` instead of being scored, because
the metric cannot see it -- the Cube's `kFlagKeyReady` wait reads [2, 2, 2, 2]
on this run and is perfectly well ordered.

## 10. Commands

```bash
python3 tests/cpu/test_decode_page_loop_model.py
```

```bash
bash scripts/run_decode_splitk_aic_page_race.sh
```

```bash
python3 scripts/audit_mode4_tick_order.py EVIDENCE_DIR --block 0 --cross
```
