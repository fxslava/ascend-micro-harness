# Scalar issue in the decode strip pipeline: what it costs and what it does not

Target: CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599`. **No NPU on this
host** (`npu-smi` and `/dev/davinci*` are both absent), so every device number
below is CAModel over a collapsed grid and **nothing here is a hardware
claim.**

This follows `reports/decode_splitk_aic_strip_pipeline.md`, which replaced the
per-page handshake in `turboquant_paged_decode_splitk_aic` with a continuous
strip stream over a mode-2 rendezvous. The task behind this report was to
remove scalar serialization on AIV0, on the diagnosis that
`core0.veccore0 SCALAR 54897` was starving `CUBE`, `MTE2` and `FIXP`.

**The starvation is real and measured. Its cause is not scalar serialization.**
Both halves of that sentence are load-bearing, so section 1 is the measurement
before any of the three changes, and section 4 is the measurement after.

## 1. Where AIV0's time actually goes

`scripts/analyze_strip_pipeline_overlap.py` over the four-page `K = 1` rung of
the previous report (`--context 64 --page-tokens 16 --page-count 4 --splitk 1`,
so four strips), at commit 719a390:

```bash
python3 scripts/analyze_strip_pipeline_overlap.py EVIDENCE_DIR --strips 4
```

| core | instructions, by engine |
| --- | --- |
| `core0.cubecore0` | CUBE 136, FIXP 26, MTE1 53, MTE2 39, SCALAR 451, FLOWCTRL 27 |
| `core0.veccore0` | **RVECLD 29477, RVECST 26392, RVECEX 24513, RVECSU 2471, RVECLP 982**, SCALAR 11600, FLOWCTRL 2325, PUSHQ 1764, MTE2 23, VEC 17, MTE3 6 |
| `core0.veccore1` | SCALAR 674, RVECEX 256, RVECST 140, RVECLD 132, MTE3 83, VEC 20, MTE2 15 |

Three facts, and they change what is worth doing:

* **AIV0 is 84% RISC-V vector work, not scalar.** 83,835 of its 99,571
  instructions retire on the `RVEC*` units (`RV_VLDI` 21376, `RV_VSTUI` 16512,
  `RV_VCADD` 16448, `RV_VLD` 6416, `RV_VST` 6048, `RV_VMUL` 4544 ...). SCALAR
  is 11,600, or 12%. The classic DaVinci SIMD `VEC` column holds 17
  instructions: the recurrence runs on the SIMT path.
* **The Cube is idle 85% of the span** -- `cube_idle_fraction 0.850` over 40
  bins of a 86,289-tick run. So the starvation the task describes is real.
* **It is demand-limited, not issue-limited.** The Cube's whole contribution is
  136 Mmad-class instructions against AIV0's 83,835 vector instructions. At one
  strip per page the stream is only three strips deep, so the Cube finishes a
  GEMM in about a thousand ticks and then has nothing to do until AIV0's
  recurrence frees the next score slot. No amount of scalar trimming changes
  that ratio.

**The 54,897 figure could not be reproduced** at any rung run here. AIV0's
scalar count is 11,600 at four strips and scales with the strip count, so a
geometry around 20 strips would reach it; the number may come from the
128-token-page ladder or from a profiler view that groups the `RV_S*` scalar
forms of the SIMT path with the scalar unit. Recorded as unreproduced rather
than quietly adopted.

### 1a. Which scalar instructions are per strip

The useful split is by how often each PC is hit, because code run once per
strip and code run once per query row are different problems:

| bucket | AIV0 instructions | distinct PCs | what it is |
| --- | ---: | ---: | --- |
| hit once | 429 | 429 | the launch prologue |
| hit `strips` times | **972** | **243** | the per-strip prologue -- the engine build, the cursor arithmetic, the flags |
| hit `16 * strips` times | **10048** | **157** | the 16-query-row loop inside the recurrence |
| other | 151 | 27 | — |

So the per-strip prologue the task names is **972 of 11,600 scalar instructions
(8.4%), and 1.0% of AIV0's total**. The dominant scalar cost, 10,048, is
`select_query`'s address arithmetic inside `vector_decode_recur_strip` --
shared production code whose own comment records why it is a runtime loop
rather than an unroll: "Unrolling the 16 queries at every one of the 8 strip
sites instantiated 128 copies of the body below, and the AIV translation unit
did not finish compiling in 25 minutes." That is not touched here.

243 scalar instructions per strip is nonetheless close to the task's estimate
of 80-100, it is pure recomputation of constants, and it scales with the strip
count -- at the 128-token-page geometry's 64 strips it is ~15,500
instructions. So all three changes are worth making. What they are not is a fix
for an 85% idle Cube.

## 2. The three changes

All on the streaming path. The legacy per-page schedule and its mode-4 flags
are untouched.

### 2a. The engines were hoisted, measured, and put back

This was the headline change and **it is reverted, because it is 2.1x
slower.** The numbers are in section 4a. In short: hoisting did cut the
per-strip scalar bucket by a third, exactly as intended, and it also turned
every per-query operand access inside the recurrence into a memory load. Built
inline in the strip body, a `LocalTensor`'s base is a constant the compiler
materialises as an immediate; held in an array that outlives the body, it is
memory, and the descriptor is re-read sixteen times per strip.

What IS kept from the change is the tail mask, and that one was a defect rather
than an inefficiency. `routine_vector_decode_recurrence.hpp` says of
`prime_tail_mask`: *"Primed once per kernel, before the first strip. Reads
nothing that changes, so it is hoisted out of the strip loop entirely."*
Calling it per strip was against the routine's own documented contract. It is
now primed once per core, through an engine built solely for that purpose --
the four mask operands sit at fixed UB addresses every engine shares, so
priming through any one primes all.

### 2b. One scalar drain removed -- and one deliberately kept

There were two `V_S` pairs in the per-strip recurrence path and they do
different jobs. Conflating them is how this change could have reintroduced the
hazard the previous report closed.

* **Removed: the drain BEHIND the release wait.** The wait names `PIPE_V` and
  the recurrence is V work, so the V queue already orders them. The drain only
  converted the wait into a stall on instruction ISSUE, which held AIV0's next
  MTE2 ingress behind a dependency it does not have.
* **Kept: the drain IN FRONT OF the readiness announcement.** This is not a
  pipeline fence; it is what section 6.2a requires of every announcement,
  because a cross-core set issues on the scalar unit and the `pipe` template
  argument does not order it. Removing it would let the announcement overtake
  the recurrence it reports -- section 6.2c's measured hazard -- and
  `scripts/audit_cross_core_fences.py` fails on exactly that.
* **Kept: AIV1's drain behind ITS release wait.** There the conversion is
  load-bearing twice over: it bounds the core whose unbounded run-ahead was the
  original defect, and it is the only thing `audit_mode4_tick_order.py` can
  read a stall from. AIV1 executes 674 scalar instructions in the whole run, so
  it is not where the cost is.

The removal was proved safe in the token model **before** the kernel changed.
`tests/cpu/test_decode_page_loop_model.py` gained an `aiv0_issue_fence` knob --
the edge put back -- and it must be ACCEPTED, not rejected: removing a fence
cannot be justified by a control that fails. The model passes 15 geometries to
8 pages with all 7 negative controls still rejected without that fence.

### 2c. The cursor has no division

`StripResolver` becomes `StripCursor`: a monotone walk that keeps
`strip_in_page`, `page_idx` and a running `token_base`, rolls the first over at
`strips_per_page`, and reads the block table **once per page** instead of once
per strip. `strip / strips_per_page` and `strip % strips_per_page` become an
add and a compare. `SlotOfStrip` and `ArchiveOfStrip` become masks, which is
exact because `kStageDepth` and `kStrips` are both powers of two -- asserted
rather than assumed.

The tail mask rides the same `token_base` instead of
`turboquant_decode_tail::WalkActive`, which divides internally. That is sound
because `strips_per_page * kTokens == block_size`, so the base rises by exactly
`kTokens` per strip across a page boundary as much as inside a page.

AIV0 keeps **two** cursors' worth of state: its ingress and its recurrence walk
the same sequence `kPipelineLag` columns apart, so one cursor cannot serve
both. The ingress gets a `StripCursor`; the recurrence gets a bare running
token base, which is all it needs.

`page_base * (kHeadSize / 2u)` also replaces `page_base * kHeadSize / 2u`,
folding the constant first. Same value, and it keeps the product inside 32 bits
for a cache size the other order would overflow.

## 3. A harness gap closed on the way

The ragged-context rungs the task asks for -- S = 1, 113, 127 -- were **not
reachable**. `bench_decode_splitk_sim` requires `--context` to be a multiple of
`--page-tokens`, correctly, because the context is allocated CAPACITY and the
cache has no notion of a part-page. The LIVE token count is a separate number
that the fixture has always taken and the kernels take as `seqLen`, and the
benchmark had no flag for it, so every rung it could run was aligned.

`--seq-len S` now sets it, validated against a single `--context` so that one S
cannot mean different things across a sweep. Omitting it is exactly the old
behaviour. This matters for this change specifically: the tail mask now rides a
running counter instead of a division, so an unaligned context is the rung that
exercises it.

## 4. Verification

Every rung `--impl aic --single --requests 1 --heads 1 --rows 1`, 16-token
pages so that every strip crosses a page boundary, on CAModel:

```bash
bash scripts/run_decode_splitk_strip_ladder.sh cheap
```

| rung | result | `l` | `acc` | channels over | ticks |
| --- | --- | ---: | ---: | ---: | ---: |
| pages 1, K 1 | PASS | 0.00064x | 0.00032x | 0 of 4096 | 51172 |
| pages 4, K 1 | PASS | 0.00063x | 0.00025x | 0 of 4096 | 180465 |
| pages 4, K 2 | PASS | 0.00072x | 0.00032x | 0 of 4096 | 94573 |
| pages 4, K 4 | PASS | 0.00064x | 0.00032x | 0 of 4096 | 51410 |
| pages 8, K 1 | PASS | — | — | 0 of 4096 | 352672 |

`m` is bit-exact and `qrot wrong 0` on every rung.

The ragged-context rungs, on the shipped revision, at 128-token pages so that
the cursor's WITHIN-page increment runs too (the rungs above are one strip per
page, so every strip is a page transition and `strip_in_page` never advances):

| rung | result | `l` | `acc` | channels over | mean \|deviation\| | ticks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S = 1 of 128 | PASS | 0.00000x | 0.00000x | 0 of 4096 | 0.0 | 33795 |
| S = 113 of 128 | PASS | 0.00099x | 0.00022x | 0 of 4096 | 6.12e-08 | 166590 |
| S = 127 of 128 | PASS | 0.00138x | 0.00019x | 0 of 4096 | 4.48e-08 | 166592 |

These are what exercise the tail mask riding the cursor's running token base
instead of `WalkActive`'s division. S = 1 comes out bit-exact against the
oracle, which is the expected shape rather than a suspicious one: with a single
live token `exp(0)` is exactly 1 and each channel is a single product, so there
is nothing for FP32 to round differently -- and it only holds if the mask
zeroed the other fifteen lanes exactly.

### 4a. The hoisting is 2.1x slower, measured

Same geometry throughout -- `pages=4, K=1` at 16-token pages, four strips on
one worker -- and `scripts/analyze_strip_pipeline_overlap.py --strips 4`:

| build | ticks | AIV0 per-strip scalar | AIV0 per-row scalar | RVECLD | RVECST | cube idle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| inline, before any change | **86,289** | 972 / 243 PCs | **10,048** / 157 PCs | 29,477 | 26,392 | 0.850 |
| recurrence engines hoisted | 180,465 | **640** / 160 PCs | 13,824 / 216 PCs | 33,887 | 30,718 | — |
| every engine hoisted | 181,221 | 1000 / 250 PCs | 13,760 / 215 PCs | 33,887 | 30,718 | 0.900 |
| **hoisting reverted, shipped** | **86,348** | **768** / 192 PCs | 10,112 / 158 PCs | 29,471 | 26,365 | — |

Read the first two rows together and the mechanism is plain. The per-strip
bucket -- the one the hoisting targets -- fell from 972 to 640, a third, which
is what it was supposed to do. The per-row bucket rose from 10,048 to 13,824,
and the vector unit took on 4,410 more loads and 4,326 more stores. Net: 332
scalar instructions saved per four strips, about 8,700 vector memory
operations and 3,776 scalar instructions paid, and **the run takes 2.1 times
as long.**

The third row is the rest of the engines hoisted the same way, and it makes
even the per-strip bucket worse (640 -> 1000; AIV1's goes 376 -> 1388) for the
same reason, while adding 0.4% to the tick count.

The fourth row is what ships. With the hoisting out and the cursor, the
prime-once and the drain removal kept, the per-strip bucket is **768 against
972**, a fifth lower than before any of this, and the per-row bucket and the
vector counts are back where they started: 10,112 against 10,048, RVECLD
29,471 against 29,477, RVECST 26,365 against 26,392. The tick count is 86,348
against 86,289 -- the same run to within 0.07%, which is to say the scalar
saving is real and too small to see at this geometry. It scales with the strip
count, so the 64-strip production geometry would see ~3,300 fewer scalar
instructions; that is not measured here.

Output is bit-identical to the unmodified kernel: `got=0.630742371`,
`mean |deviation| 6.208090e-08`, 0 of 4096 channels over tolerance.

So inline construction stays. The diagnosis that per-strip engine rebuilding
emits a few hundred scalar instructions was **correct** -- 243 PCs per strip,
close to the 80-100 estimated -- and hoisting is still the wrong remedy,
because on this target those instructions are how the operands become
immediates in the first place.

### 4b. A bug the revert caught

The hoisting patch had replaced `ArchiveSlotOf(strip % kStrips)` with
`ArchiveSlotOf(strip)`, dropping the ring wrap on the score archive. The
archive arena is `kStrips` slots, so any worker walking more than 8 strips
would have written past it into `kUbLBroadcast` and the egress buffers. It
survived every rung run here because all of them walk 8 strips or fewer; the
128-token-page geometry at 4 pages walks 32 and would have corrupted UB.
Restored as `ArchiveSlotOf(ArchiveOfStrip(strip))`, composing the slot index
with the slot address instead of conflating them.

### 4c. The mode-2 announcement does NOT wait for its pipe

The question from section 2b, settled by measurement rather than by either
side's reading of the documentation. Two builds differing **only** in whether
AIV0's readiness announcement carries its `V_S` drain, same geometry, and
`scripts/analyze_announce_overtake.py` reading how much of PIPE_V's work had
retired when each announcement was issued:

| AIV0 announcement, in order | drain present | drain removed | difference |
| --- | ---: | ---: | ---: |
| 1st (priming set) | 146 | 146 | 0 |
| 2nd (priming set) | 149 | 149 | 0 |
| **3rd (strip 0)** | **23,334** | **19,111** | **-4,223** |
| **4th (strip 1)** | **46,491** | **42,266** | **-4,225** |

**Without the drain the announcement is issued 4,223 vector instructions
before it is with the drain, at the same ordinal, on both announcements.** The
deltas agree to two instructions, so this is a systematic lead and not noise.
That is a direct refutation of the claim that the mode-2 quorum defers the
announcement until the announcing core's PIPE_V work has retired: on this
toolkit it does not, and the drain is not redundant.

A second build pair corroborates it. The shipped revision, drain still
removed, reads `[2, 2, 17274, 38174]` against the pre-change kernel's drained
`[2, 2, 21075, 42001]` -- a lead of 3,801 and 3,827. That pair is not perfectly
matched (it also differs by the cursor and the prime-once), which is why the
4,223 figure above is the one quoted; but two independent pairs putting the
lead at 3,800-4,200 vector instructions is the same answer twice.

The exposure: that announcement means "the score panel of strip t is free, I
have read it", and the Cube's Fixpipe for strip t + kStageDepth is gated on
it. So the Cube is cleared to overwrite the panel while AIV0 still has about
4,200 vector instructions of reads outstanding against it -- roughly a fifth
of a strip's recurrence.

**It did not produce a wrong answer at this geometry**, and that is worth
stating precisely because it is weak evidence rather than reassurance: the
undrained `pages=4, K=1` rung is numerically identical to the drained one,
down to `got=0.630742371` and `mean |deviation| 6.208090e-08`. With
`cube_idle_fraction` at 0.875 the Cube never comes close to using the licence
the early announcement hands it. The four-page defect this operator was just
fixed for was exactly such a licence going unused until the timing changed.

The drain was left removed for one revision while this was measured, with
`scripts/audit_cross_core_fences.py` reporting it as unfenced rather than being
disabled -- disabling it would also have stopped it protecting the other three
announcements here and both reshape-and-cache kernels. It is **restored** in
the shipped revision, as the deferred wait of section 4c-bis.

### 4c-bis. The fence is back, as a DEFERRED wait, at no ordering cost

The drain is restored in the form that keeps the guarantee and moves the
stall. `SetFlag<HardEvent::V_S>` is issued into the V queue and fires when V
reaches it; `WaitFlag` is the scalar instruction that blocks. So the SET stays
behind the recurrence -- and behind the archive `Muls`, which reads the same
panel -- while the WAIT and the announcement move to the end of the column,
after this column's dequant and load have been issued. The scalar unit
enqueues the next strip's work before it blocks.

Whether that still fences anything is the same measurement as section 4c, and
the answer is yes, exactly:

| AIV0 announcement, in order | 1st | 2nd | **3rd** | **4th** |
| --- | ---: | ---: | ---: | ---: |
| drain adjacent, pre-change kernel | 2 | 2 | **21,075** | **42,001** |
| no drain at all | 2 | 2 | **17,274** | **38,174** |
| **deferred wait, shipped** | 2 | 2 | **21,082** | **41,983** |

The deferred form sits within 7 and 18 instructions of the adjacent form and
about 3,800 above the unfenced one. **Deferring the wait costs nothing in
ordering.** Output is bit-identical to every other revision measured here:
`got=0.630742371`, `mean |deviation| 6.208090e-08`, 0 of 4096 channels over
tolerance, and the eight-page rung passes too.

It is also the best revision on every scalar axis measured:

| build | ticks | AIV0 SCALAR | per-strip | per-row | cube idle |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre-change, drain adjacent | 86,289 | 11,600 | 972 / 243 | 10,048 / 157 | 0.850 |
| no drain at all | 86,348 | 11,558 | 768 / 192 | 10,112 / 158 | — |
| **deferred wait** | 86,390 | **11,336** | **744 / 186** | **9,792 / 153** | **0.850** |

**The Cube idle fraction did not drop.** It is 0.850, which is where it was
before any of this work started. That was the stated goal of the change and it
is not met, for the reason section 1 gives and section 4e repeats: the Cube
retires 136 Mmad-class instructions against AIV0's ~84,000 vector
instructions, so it is waiting on AIV0's arithmetic and not on AIV0's
instruction issue. Moving the stall lets AIV0's MTE2 ingress start earlier,
and AIV0's MTE2 is 23 instructions. The tick count is 86,390 against 86,289 --
the same run to within 0.12%.

What the change IS worth: AIV0's scalar count is 11,336 against 11,600, the
lowest of any revision measured, with the fence intact. At the 64-strip
production geometry the per-strip saving scales to roughly 3,600 instructions.
That is a real improvement and it is not a fix for pipeline starvation.

### 4d. The audit was quietly exempting the line under test

Found while doing the above, and the most serious of the tooling defects in
this report. With the drain removed the fence audit did not fail -- it reported
the announcement as `priming set, structurally: no work on any pipe since the
start of this block`. The structural priming test added earlier counts braces
backwards and stops at the first enclosing `{`, so an announcement alone
inside a one-line guard --

```cpp
if (strip + kStageDepth < total) {
    AscendC::CrossCoreSetFlag<2, kSync2Aiv0Announce>(kSync2StageReady);
}
```

-- has nothing between the brace and the set and was declared priming. The
gate had stopped protecting exactly the line this report is about.

The scope is now the enclosing ITERATION, not the enclosing brace pair: `if`
and `else` braces are transparent and the scan carries on outward to a loop or
a function body.

Widening it that way then broke two things, both fixed, and the audit now
carries **sixteen** controls:

* A lone `WaitFlag<x_S>` with no matching SET anywhere fell through to the
  priming route and was exempted. An author who wrote the wait meant a fence;
  a missing set is a bug. It now FAILs.
* The legacy per-page `kFlagScoreFree<0>` / `<1>` primes lost their structural
  exemption, because the widened scope reaches the function body, which does
  contain earlier Vector work -- the m/l/Acc `Duplicate`s -- that has nothing
  to do with what those sets report. Rather than loosen the heuristic again
  they are now named `kFlagScoreFreePrime`, the same convention
  `turboquant_reshape_and_cache_a8.cpp` already uses. Same value, so the
  frozen legacy instruction stream is untouched; the exemption is
  author-declared instead of inferred.

And the deferred fence needed its own handling. Splitting the SET from the
WAIT puts the two halves in different functions, which a lexical audit cannot
pair -- it was exempting the announcement as a priming set, a vacuous pass on
the one line the pattern is about. A lone WAIT in front of an announcement,
with a matching SET on the same route and event somewhere in the translation
unit, is now accepted and reported as `defer` rather than `ok`, with the limit
stated in the output: **the audit cannot see what the set is behind.** For the
adjacent form it can at least pin the SET to the work next to it. For a
deferred one the ordering is a trace property, which is what
`scripts/analyze_announce_overtake.py` measures and what section 4c-bis
reports.

### 4e. What the trace says about overlap

`cube_idle_fraction` is 0.850 before any of this, 0.875 undrained and 0.900
with everything hoisted. The Cube's own contribution is 136 Mmad-class
instructions against AIV0's ~84,000 vector instructions, so none of these
changes moves it and none could: the Cube is not waiting on AIV0's scalar
issue, it is waiting on AIV0's arithmetic. The occupancy chart shows the same
shape in every build -- the Cube's engines busy in the first few bins and then
a single column of activity per strip.

The one thing that would move it is the value contraction. Three PCs in AIV0's
trace are hit 16,384 times each at four strips, which is exactly 256 per query
row: `ReduceRepeat`'s per-channel 16-lane reduction in `contract_values`,
49,152 of AIV0's 83,835 vector instructions. Replacing it with a pairwise tree
would be four vector operations over 2048, 1024, 512 and 256 elements instead
of 256 cross-adds -- but it changes the summation ORDER within each channel,
so it changes FP32 results, so it breaks the frozen legacy entry's bit-identity
and every oracle margin quoted in this report family. That is a separate piece
of work with its own numerical-equivalence gate, and it is not started here.

