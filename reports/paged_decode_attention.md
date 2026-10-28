# Fused paged decode attention — Ascend 950PR, 16 queries × d=256, int4 paged KV

Ascend 950PR / arch35 / `dav-3510` / CANN 9.2.0-beta.2, `Ascend950PR_9599`
CAModel. `KERNEL_TYPE_MIX_AIC_1_2`: one Cube core and **both** Vector cores.

Operator: [`kernels/ops/turboquant/turboquant_paged_decode_attention.cpp`](../kernels/ops/turboquant/turboquant_paged_decode_attention.cpp)
Leaves: [`routine_cube_decode_score.hpp`](../routines/routine_cube_decode_score.hpp),
[`routine_vector_decode_recurrence.hpp`](../routines/routine_vector_decode_recurrence.hpp),
[`routine_vector_egress_rotate.hpp`](../routines/routine_vector_egress_rotate.hpp)
Oracle and fixture: [`tests/common/paged_decode_oracle.hpp`](../tests/common/paged_decode_oracle.hpp)
Runners: [`tests/cpu/test_paged_decode_probe.cpp`](../tests/cpu/test_paged_decode_probe.cpp),
[`tests/camodel/test_camodel_paged_decode_attention.cpp`](../tests/camodel/test_camodel_paged_decode_attention.cpp)

---

## 0. The claim

| Field | Value |
| --- | --- |
| What it proves | the 64-Mmad ingress rotation, the per-strip score GEMM, the 16-query × 256-channel online softmax, and the egress rotation + gate all compute the specified arithmetic on CAModel; and that one tile reads and writes only its own Global Memory |
| What it deliberately omits | the cross-block log-sum-exp combine (see §7), any hardware measurement, and a full 32-block grid run |
| Target | `Ascend950PR_9599`, `__NPU_ARCH__ == 3510`, CANN 9.2.0-beta.2 |
| Kernel type | `MIX_AIC_1_2` — required for mode-4 flags, for a Fixpipe that picks a Vector core, and for the direct UB → L1 channel |
| Grid | `gridDim = 32` = 4 KV heads × 8 temporal blocks, one task per physical AIC |
| Ring depth | 2 on the UB landings and the score windows (the "double window"), so 8 strips wrap each of them four times. The L1 key staging is **not** a ring: one page per strip, after §4.6 |

Geometry: 16 query tokens (M), head dimension 256, a 128-token vLLM page as
eight 16-token strips, 4 KV heads, int4 symmetric KV cache with a planar FP32
scale cache.

---

## 1. The schedule

Strip `s` occupies five pipeline stages across five consecutive columns, so
eight strips span twelve columns:

```text
 T ->  [T0 ][T1 ][T2 ][T3 ][T4 ][T5 ][T6 ][T7 ][T8 ][T9 ][T10][T11]
 MTE2  LdK0 LdK1 LdK2 LdK3 LdK4 LdK5 LdK6 LdK7  --   --   --   --   AIV1
 AIV1   --  DqK0 DqK1 DqK2 DqK3 DqK4 DqK5 DqK6 DqK7  --   --   --   dequant + CV
 MTE2   --   --  LdV0 LdV1 LdV2 LdV3 LdV4 LdV5 LdV6 LdV7  --   --   AIV0
 AIC    --   --  QK0^ QK1^ QK2^ QK3^ QK4^ QK5^ QK6^ QK7^  --   --   Mmad + Fixpipe
 AIV0   --   --   --  DqV0 DqV1 DqV2 DqV3 DqV4 DqV5 DqV6 DqV7  --   dequant
 AIV0   --   --   --   --  S0V0 S1V1 S2V2 S3V3 S4V4 S5V5 S6V6 S7V7  recurrence
```

Column `j` runs, all mutually independent: AIV1's MTE2 ingress of strip `j` and
its dequant + CV staging of strip `j-1`; the AIC's Mmad and Fixpipe of strip
`j-2`; AIV0's MTE2 ingress of strip `j-2`, dequant of `j-3` and recurrence of
`j-4`. The fold is over a compile-time index sequence with runtime guards, so
every ring slot index is a constant expression.

### Two deviations from the specified Gantt, and why

**Q is staged in the Cube prologue, not spread over T0..T7.** The specified
chart draws eight `LdQ` cells on the AIC's GM → L1 row, but Q is strip
invariant and the first `Q·K0ᵀ` lands in T2, so a Q load in T3 or later could
not be an input to it. What is genuinely eight-wide about the ingress rotation
is its *Mmad nest* — eight source groups per output group — and that runs in the
prologue, before T0. The AIC's MTE2 row is therefore idle during the strip
sequence, which is what frees that queue for the two Vector cores.

**Mmad and Fixpipe of one strip are issued together.** They are attributed to
T(s+2) and T(s+3) in the chart; in the kernel they are one column body with an
explicit `M_FIX` token between them. The overlap the chart shows is what the
hardware gives for free as the Fixpipe drains each fractal, not a second
software stage.

### Mmad count

The specified chart annotates the AIC row "64 Mmads/strip". As implemented the
64-Mmad nest is the **ingress rotation**, run once in the prologue: the
256 × 256 randomised Hadamard is applied as 8 output groups × 8 source groups
against 16 resident 32 × 32 blocks. Each strip's score GEMM is **8** accumulating
Mmads of k = 32. Rotating per strip instead would be both eight times the Cube
work and wrong — the KV cache already holds rotated values, so a second rotation
would double-rotate.

---

## 2. The marked graph

Every directed cycle holds exactly one token against a two-slot ring, so the
graph is live (≥ 1 token per cycle) and bounded (≤ slot count).

Forward edges: `MTE2_V` on each Vector core (packed page readable); `V_MTE3` on
AIV1 (bf16 key page readable by the CV burst); `kFlagKeyReady` AIV1 → AIC (L1
key page staged); `kFlagValueReady` AIV0 → AIC (value page dequantised);
`kFlagScoreReady` AIC → AIV0 (score window written).

Back edges, each primed once and drained once: `V_MTE2` on each Vector core
(packed slot free — on AIV0 set by the **recurrence**, which is the slot's last
reader, not by the dequant; §4.5 is what assuming otherwise cost); `MTE3_V` on
AIV1 (bf16 page free); `kFlagScoreFree` AIV0 → AIC; `M_MTE1` (L0B slot free)
and `FIX_M` (L0C window free) on the AIC.

There is **no L1 back edge**. The L1 key staging is not a ring: strip `s` owns
L1 page `s`, written once and read once, so that cycle does not exist. §4.6 is
why, and it is a deviation from the task's §3.

A back edge is only as good as its endpoint. Both ladder failures (§4.5, §4.6)
were edges that existed, balanced, and pointed at the wrong instant — which is
why the boundedness argument above is no longer the evidence for boundedness.
The trace audit now checks the *buffers*: that each L1 key page is written in
16 bursts and read once, after its last write.

**Two rings need no edge at all**, and that is a property of the engine
assignment rather than an omission. The dequantised FP32 value ring has a
Vector producer and a Vector consumer on the same core, and the Vector queue
retires in issue order, so both the RAW and the WAR are already expressed by the
order the stages are emitted in. The dequant scratch is single buffered for the
same reason. A HardEvent on either would be a no-op that still had to be primed
and drained.

### Flag ids

| flag | ids | direction |
| --- | --- | --- |
| `kFlagValueReady` | 0..1 | AIV0 → AIC |
| `kFlagScoreFree` | 4..5 | AIV0 → AIC |
| `kFlagScoreReady` | 8..9 | AIC → AIV0 |
| `kFlagKeyReady` | 16..17 | AIV1 → AIC |

Four channels in the widest direction (AIV0 → AIC), two from AIC to AIV0, two
from AIV1 to AIC and **none** from AIC to AIV1 — within the contract's limit of
six. No raw id reaches 11, so the library's reserved 13 and 14 and their +16
images are clear. AIV1's ids are its own half's 0..1 after the
`subBlockId * INTRA_NUM` rebasing.

`kFlagL1Free` (AIC → AIV1, ids 20..21) was the fifth channel and is **gone**,
along with the two-slot L1 key ring it ordered. §4.6 is why.

### The risk this kernel carried, and how it came true

Guidelines §6.2c records a **measured** case where a mode-4 id reused across
ring slots released a wait 2179 cycles before the announcement it was waiting
for. The fix that holds under any reading of the mechanism is one id per
announcement, and this operator cannot have that: the contract caps channels at
six per direction with ids below 11, and with two-slot rings over eight strips
each id carries **four** announcements. One id per announcement would need eight
for the key edge alone.

What the first version of this operator did instead was argue the reuse was
*ordered*: every id is half of a two-edge cycle, and the set for strip `s+2` is
issued only after a wait on the paired opposite-direction flag for strip `s`.
The argument is sound as stated. **It is also not what happened.** On the
8-strip rung AIV1's third and fourth waits on `kFlagL1Free<0>` did not block,
and it overwrote L1 key pages the Cube had not yet read — §4.6, with the ticks.
The id had the right number, the right pipe, the right count, and the Cube's
announcement was correctly drained behind its Fixpipe. It still did not bound.

So the operator no longer depends on that argument where it can avoid it. The
L1 key ring is gone: each strip owns one of eight L1 pages, 112 KiB of a 512
KiB L1, and a buffer that is written once and read once needs no back edge to
order. The three surviving cross-core channels still carry four announcements
per id, and what makes them sound is **not** the id count — it is that each
one's consumer is measurably slower than its producer, so no producer ever gets
two announcements ahead. That is a property of the measured schedule, not a
guarantee, and §8 records it as such.

The check that was supposed to catch this could not have. See §4.7: a
`CrossCoreWaitFlag` retires as soon as it is reached and arms a block on its
pipe's *subsequent* instructions, so no dump tick for a wait is the moment its
flag released it, and the old check 3 — which ordered flag instructions by tick
— reported a fault on every rung including the correct ones. Check 3 now tests
the property the flags exist to provide, directly and from the data
instructions: every L1 key page is staged in 16 bursts and loaded exactly once,
after its last write.

---

## 3. Arithmetic, and where exactness stops

The rotation is the randomised Hadamard
`Π = (H8 ⊗ I32) · diag(piSigns) · (I8 ⊗ H32) / 16`, with `Π Πᵀ = I` exactly
because the `1/16` is folded into the table so every entry is ±1/16.

* **Ingress** (AIC, 64 Mmads): `Q_rot = Q · Πᵀ`, with the sign diagonal and the
  outer factor folded into 16 resident 32 × 32 bf16 blocks, so the dependence on
  the output group is a single sign and 16 blocks cover all 64 pairs. This is
  the decode-side reuse of the prefill operator's measured result
  ([`reshape_and_cache_a8.md`](reshape_and_cache_a8.md)): folding a factor
  diagonal in 32 channels into the GEMM costs eight times the multiply-adds and
  buys back four full-panel Vector passes.
* **Scores** (AIC, 8 Mmads/strip): `R = Q_rot · K_stripᵀ`, one accumulating
  Mmad per 32-channel group. The key's per-token scale is applied afterwards to
  the 16 score *columns*, not to the 4096-entry page: 16 exact FP32 multiplies
  against 4096 bf16 ones, and it keeps the Cube's key operand an exact small
  integer, which is what makes the panel bit-checkable.
* **Recurrence** (AIV0): the streaming `(m, l, Acc)` online softmax, carried as
  one replicated 32-byte DataBlock per query so every per-query operand is a
  whole DataBlock from a 32-byte aligned base. The value's per-token scale folds
  into the 16 probabilities; the denominator still sums the unscaled ones.
* **Egress** (AIV0): `ctx = (Acc / l) · Π`, as eight radix-2 butterfly stages
  plus the sign diagonal, then `sigmoid(gate)`. Five stages are strided
  `Add`/`Sub` pairs; the three sub-DataBlock strides (1, 2, 4) need a `Gather`,
  because a 4-, 8- or 16-byte tensor offset is not 32-byte aligned and a mask
  cannot reach inside a DataBlock either.

Bit-exact, and checked as such: `Q_rot` (bf16 Mmads into an FP32 accumulator,
with a fixture whose rotated entries are exact in bf16) and every raw score
panel of every strip (exact small integers against `Q_rot`, then a power-of-two
column scale).

Bounded in ULP, not bit-compared: `m`, `l`, `Acc`, the egress rotation and the
gated context. Every one passes through `Exp`, documented to permit 1 ULP per
call, and with symmetric int4 the accumulator is a cancelling sum, so budgets
are stated against the largest value in the row — guidelines §5's rule for
exactly this shape.

---

## 4. Nine faults the verification found

Recorded because each was invisible to the stage before it, and six of them
were silent. §4.5 and §4.6 came out of the strip ladder — the rungs past the
ring depth — and would not have been found by the single-tile protocol, which
passed all nine coordinates while they were live. §4.7 is a fault in the
verification itself and §4.8 was introduced by the fix for §4.5; both are
recorded on the same footing as the rest.

### 4.1 `Cast(bfloat16_t ← int16_t)` compiles and emits nothing

The 950PR `Cast` table lists, for a `bfloat16_t` destination, exactly six
sources: `int4b_t`, the two fp4 types, `half`, `float` and `double`.
**`int16_t` is not among them**, and on `dav-3510` that pair produces no
diagnostic and **no instructions**.

The symptom was a score panel 256/256 **zero** while every address, every flag
and every other instruction in the trace was correct. What located it was a
full opcode census of the staging core's dump: three vector stores and one
conversion where the four-stage unpack should have left four and two. The
missing store was the key page itself.

The fix is `turboquant_dequant4_signed`, which uses only listed pairs:
`int4b_t → bfloat16_t` directly (one instruction), and
`int4b_t → int16_t → float` for the FP32 value path because `int4b_t → float`
is not listed either. That also moved the cache to **symmetric** int4 (−8..+7),
which is what `int4b_t` means, and removed the mask stage: three fewer
4096-element passes per strip per core and 24 KiB of Unified Buffer.

### 4.2 One Mmad of k = 256 does not compute what eight of k = 32 do

The first revision issued the score GEMM as a single Mmad with k = 256, which
`MmadParams` expresses happily. In the run that established the problem it
produced a panel with 251 of 256 entries wrong, while the ingress rotation in
the *same* run — 64 accumulating Mmads of k = 32 through the identical operand
path, Fixpipe and fixture — came out bit-exact. The contraction depth was the
only difference. The depth now stays at the 32-channel group width the rest of
this operator family uses. The mechanism was not isolated.

### 4.3 The egress gate was missing its WAR back edge

The gate row is single buffered and refilled per query by MTE2. The first
revision had the `MTE2_V` edge ("this query's row has arrived") and no
`V_MTE2` edge ("the row may be refilled"), and the MTE2 queue is the one with
nothing else gating it.

The canary run measured the consequence exactly: the factor the device applied
to query 0 was `sigmoid(gate[query 2])` to within bf16 rounding, so MTE2 was
running exactly two iterations ahead of its consumer. This is the same hazard
class, on the same queue, as the missing packed-page WAR edge that guidelines
§0 records for the fused attention operator.

It was findable only because the rotation and the gate were separated: the run
also reported the egress rotation correct to 8.9 × 10⁻⁸, which left the gate as
the only suspect and made measuring the applied factor worth doing. The
`out_rotated` output exists for that reason alone.

### 4.4 Two fixture degeneracies, caught on the host

The Tier-1 probe rejected two fixtures before any device run:

* the key scale's per-strip doubling drove the late strips into a **hard max** —
  every exponential but the largest underflowed, every query's probability mass
  collapsed to 1.0, and the per-strip sums stopped being distinguishable. Fixed
  by measuring the raw score range and choosing the base exponent from it;
* `ValueNibble` had period 16 in the channel index, so the accumulator held 16
  distinct channel values repeated 16 times, the egress Hadamard of a period-16
  vector is almost entirely zero, and a permuted channel would have been
  invisible. Fixed by mixing in the high half of the channel index.

A third property had to be *constructed* rather than hoped for: the per-strip
maximum must rise for **every** query, which needs `max_t R[q][t] > 0`, and `R`
is a signed sum against a Hadamard-rotated query whose entries cancel to a mean
of zero. Token `t` therefore carries an anchor — its nibbles are biased up on
exactly the channels where `Q_rot[t]` is positive — so `max_t R[q][t] ≥ R[q][q] > 0`.
The validator checks the result rather than trusting the argument.

### 4.5 A ring slot was released by something that was not its last reader

The 4- and 8-strip rungs failed while 2 strips and all nine canary coordinates
passed. The slot index is `Column % 2`, so **2 strips is the last grid that
reuses no slot** — the ladder's rungs were placed at exactly that boundary and
the split falls on it. Two independent faults were behind the split; this is
the first.

The 4-strip rung's signature named it almost by itself: `m` wrong by 2097152
ULP and `Acc` by 4010429, while **every raw score panel stayed bit-exact**.
`m` and `l` have exactly two inputs — the scores, proven good, and the key
scale — so the key scale was corrupt.

`LoadValueStrip` lands *three* things in a slot: the packed value page, the key
scale slice and the value scale slice. `DequantValueStrip` released the slot
with `SetFlag<V_MTE2>` as soon as it had read the **packed page**, but the two
scale slices and the dequantized value page are read a column later, by
`RecurStrip`, which released only `kFlagScoreFree` and never the slot. So the
MTE2 refill for strip `s+2` was free to overwrite strip `s`'s scales while the
recurrence still needed them.

Within column `s+4` the emission order is `LoadValueStrip(s+2)`,
`DequantValueStrip(s+1)`, `RecurStrip(s)` — the load is issued *first*, and
because it is on the MTE2 queue while the recurrence is on the Vector queue,
program order inside the column orders nothing at all. Only the flag did, and
the flag said the slot was free.

Fixed by moving the release to the end of `RecurStrip`, which *is* the slot's
last reader. The count is unchanged — 2 primes + one per strip against one per
strip + 2 drains — and there is no deadlock, because the stalled MTE2 queue
does not hold back the Vector queue that releases it.

This is the same class of error as §4.3 and the third instance in this
operator: an edge whose endpoint was placed at the stage that *looked* like the
consumer rather than the one that reads the bytes last.

### 4.6 The L1 key ring's back edge did not bound the staging core

The second fault behind the same split, and the one that corrupted the scores.

At 8 strips, 480 of 2048 raw score entries were wrong. The cause is visible in
the data instructions alone, with no appeal to flag semantics. L1 key slot 0 is
`0xc060`:

| | strip 0 | strip 2 | strip 4 | strip 6 |
| --- | --- | --- | --- | --- |
| AIV1 stages `0xc060` | 6754 | **8703** | **10653** | 12608 |
| AIC loads `0xc060` | 7088 | **26994** | **66481** | 106697 |

The load for strip 2 runs at tick 26994, by which time strips 4 and 6 have both
restaged the page; it returns strip 6's keys. Same on slot 1 (`0xe080`) for
strips 3 and 5. AIV1 finished staging all eight strips by tick 13851 while the
Cube had GEMM'd one.

`kFlagL1Free<0>` was supposed to prevent this, and everything checkable about
it is correct: AIV1's wait is `PIPE:MTE3` and the burst it gates is `MTE3`; the
id is right (AIV1's local 4 is the AIC's 20, per the mode-4 table in
`CrossCoreSetFlag`'s reference); the counts balance at 5 sets and 5 waits; and
the Cube's announcement is drained behind its Fixpipe by `FIX_S`, which with the
explicit `M_FIX` token does prove the L1 → L0B load retired. The documentation
is explicit that the flag is a counter that "increases by 1 … decreases by 1",
and by tick 10653 only two increments had landed against three decrements.
**The wait did not block.** This is guidelines §6.2c, at ~16500 cycles instead
of 2179.

Fixed by removing the ring rather than the edge: each strip owns one of eight
L1 pages. 8 × 8224 B takes L1 to 112.3 KiB of 512 KiB, and a page written once
and read once has no WAR edge to get wrong. This is a **deviation from the
task's §3**, which asks for an L1 reuse back edge; the edge it asks for is the
one that was measured not to bound. The UB landing rings and the score windows
keep theirs — those are intra-core events, and they held at every rung.

### 4.7 The check that should have caught §4.6 could not have

`analyze_paged_decode_trace.py` check 3 asserted that each flag id's merged
event sequence alternates `SET, WAIT`. It fired on **every** rung, including
2 strips and the nine canary coordinates that were numerically correct — so as
written it was a false-positive generator, not a check.

The reason is in the API reference: `CrossCoreWaitFlag` blocks "the issuance of
subsequent instructions of the current core", so the wait instruction itself
retires as soon as the queue reaches it and arms a block on the *next*
instruction of its pipe. Its tick is therefore not the moment the flag released
it, in either dump. Concretely, on the 2-strip rung the Cube's drain wait on id
4 was *dispatched* at tick 8274 and *completed* at 26699, against the AIV0 set
that released it at 26688: ordered by dispatch tick the wait precedes its own
set. Moving the check to the completion log fixed those but not the underlying
unsoundness — the ordering simply is not recoverable from flag ticks.

Check 3 now tests the property the flags exist to provide, from the data
instructions whose ticks do mean what they say: every L1 key page is staged in
16 `MOV_UB_TO_L1` bursts and loaded by exactly one `LOAD_2Dv2`, after its last
write. Re-run against the archived dumps of the failing runs it **passes 2
strips and fails 4 and 8**, naming the page and the tick of each extra read.
That discrimination is the only evidence that the new check works, and it is
the reason it is stated here rather than merely fixed.

### 4.8 The fix for §4.5 broke the program order the flag needs

Moving the slot release into `RecurStrip` put the `SetFlag<V_MTE2>` for slot
`s % 2` *after* the `WaitFlag<V_MTE2>` for the same slot in `LoadValueStrip`,
because the column body emitted the load first. The model said so, twice per
run, on AIV0:

```text
[error] 3580: has no matched set_flag, pc:0x10d0f2a8, pipe:1, trigger_pipe:4, flag_id:0
[ERROR] pem_ccu.cc:2270 execute_set_flag already has same set_flag! pc:0x10d133fc
```

`0x10d0f2a8` is instruction 14966, `WAIT_FLAG PIPE:VEC, TRIGGER PIPE:MTE2,
FLAG ID:0` — the wait that is the whole point of §4.5. An intra-core
`SetFlag`/`WaitFlag` pair is **not** a semaphore that waits for a set still to
come: the scalar stream drops the set into the producing pipe's queue and the
wait into the consuming pipe's, so the set has to be *reached first in program
order*. It was not, the model let the wait through, and the producer's later
set then collided with a flag still standing.

**The run's numbers were correct anyway** — 0/2048 score entries wrong, `m`
and `l` within 2 ULP — which is the point worth keeping. The output comparison
could not see this; only the model's own diagnostic and the trace could. Fixed
by emitting the column body as recurrence, dequant, load, so every set
precedes its wait. It costs nothing: the scalar stream does not wait for the
Vector queue, so the column's `DataCopy` is issued a few scalar instructions
later onto an MTE2 queue that is idle either way.

Trace audit **check 1b** now walks each intra-core `(pipe, trigger, id)` route
in instruction-id order — which within one core *is* program order — and
requires strict `SET` before `WAIT`, never two sets standing, and nothing left
standing at the end. Counting could not catch this: the counts balance either
way. Run against the dumps of the run above it names instruction 14966, the
same one the model named.

### 4.9 Six of twelve rungs were audited against a core that did nothing

The twelve-rung ladder came back with every rung `PASS` and an overall exit
status of 1. The three strip rungs and the three `H=0` canary rungs audited
clean; the six canary rungs at `H=2` and `H=3` failed the audit with

```text
no L1 address was both staged by AIV1 and loaded by the Cube:
the key path did not run, or the dump format changed
```

Both halves of that message were wrong about the kernel and right about the
trace. `--canary T H` isolates block `8H`, so `H=2` and `H=3` are blocks 16 and
24 — and the model dumps **all 32 cores** of the grid, `core0` through
`core31`, with block `N` landing on core `N`. The runner archived
`core0.*.dump` and the audit read `core0`, so for those six rungs it read a
core that had taken the early return. Measured on the live dumps of the last
rung, which isolated block 24:

| core | `veccore0.instr_log` | instructions | vector | Mmad | UB → L1 |
| --- | --- | --- | --- | --- | --- |
| `core24` (ran block 24) | 8,455,500 B | 61,534 | 52,273 | 84 | 16 |
| `core0` (idle) | 18,573 B | 133 | 0 | 0 | 0 |

The 1:1 block → core mapping §2 claims is therefore not only designed but
observed: the work for block 24 is on core 24 and nowhere else.

What makes this worth a section is which checks let it through. An idle core is
not an empty one — it runs a scalar prologue and an early return, 133
instructions with one `MTE2` and one `MTE3` among them — and:

- checks **1, 1b and 2** are balance properties. On a core with no flag
  instructions they are all vacuously true, and they print
  `balanced ... on all 0 route(s): 0 set/wait pairs` while doing it.
- check **8**, "both Vector cores executed", would have passed as well. Its
  test is `any(p in census["AIV1"] for p in VECTOR_PIPES | {"MTE2", "MTE3"})`,
  and the prologue's single `MTE2` satisfies it. It never ran here only because
  it sits downstream of check 3.
- check **3** caught it, for the same reason it replaced the old check in §4.7:
  it asks a question about data instructions rather than about a balance.

So eleven of the twelve ladder lines said `PASS`, the summary printed no audit
column at all — the ladder script set the variable and left it out of the
`printf` — and the only signal that anything was wrong was the exit status.

Three fixes, all in the harness:

1. **New check 0**, run before everything else: each audited core must show
   the work that core is defined by — AIV1 at least 16 `MOV_UB_TO_L1` bursts,
   the Cube at least the 64 rotation Mmads, AIV0 at least one vector
   instruction. On the archived `core0` dumps it now fails first and says
   `this core did not run the kernel body ... block N runs on core N, so audit
   core<block> rather than core0`.
2. The runner archives every core's instruction logs and takes the core to
   audit from **the runner's own report** of which block it isolated, rather
   than re-deriving it from the arguments. That keeps the block → core mapping
   under test: if the work ever lands elsewhere, check 0 fails and names the
   reason.
3. The exception-dump census in the same function was dead. It globbed
   `excp_log*.dump`; the model writes `core<N>.excp_log.dump`, 32 of them, so
   the loop matched nothing and silently never fired. It now globs both names
   under `nullglob`.

The six rungs were re-run against the right core afterwards; §6 has the
results. No kernel change was involved and none of the numeric or isolation
results moved — these runs had always been correct, and the audit of them had
always been of the wrong core.

---

## 5. Two toolchain traps, for the next person

Neither is about this kernel, and both cost a full cycle.

**WSL drvfs served BiSheng a stale source file.** Building out of `/mnt/c`, the
compiler reported template errors quoting text that had already been deleted
from the file on disk; the stale text was present only inside CANN's own
preprocess object files, and both `sed` and `grep` through WSL showed the
current content. Reading every source through WSL first did not help, and
neither did a fresh build directory. The fix is to build from a copy inside the
ext4 filesystem, which `scripts/run_paged_decode_camodel.sh` does; it is also
several times faster.

**A reused build directory breaks CANN's preprocess stage.** It links each
device object over *itself* (`ld.lld … -o <the same .o>`), so a second pass
hands `ld.lld` a file it has already rewritten and the build dies with "unknown
file type". `SKELETON_KERNEL_WORKFLOW.md` says to use a fresh build directory
per full build; this is what going against it looks like.

---

## 6. Results

Commands, from WSL with the CANN environment available:

```sh
bash scripts/run_paged_decode_camodel.sh --canary 0 0   # one tile
bash scripts/run_paged_decode_camodel.sh --strips 8     # steady state
bash scripts/run_paged_decode_camodel.sh ladder         # every rung
```

`ladder` runs twelve rungs: the strip ladder at 2, 4 and 8 on block 0, then the
nine specified canary coordinates. Each rung also gets the V4 trace audit on its
own dumps, which costs no simulator time and is what caught §4.6 and §4.8.

The audit reads the core the runner says it isolated, not `core0`: the model
dumps all 32 cores of the grid and block `N` runs on core `N`, so auditing
`core0` on a run that isolated block 16 reads a core that returned early
(§4.9).

### What the trace audit checks

| # | check |
| --- | --- |
| 0 | **the audited core ran the kernel body** — 16 staging bursts, 64 Mmads, one vector instruction at minimum (§4.9) |
| 1 | intra-core `SetFlag`/`WaitFlag` balance per `(pipe, trigger, id)` route, and ids 6 and 7 never used |
| 1b | **set before wait** in program order on every route, never two sets standing, nothing standing at exit (§4.8) |
| 2 | cross-core flags paired on absolute ids, with AIV1's rebased by `INTRA_NUM`, none in the reserved band |
| 3 | **every L1 key page staged in 16 bursts and read exactly once, after its last write** (§4.6, §4.7) |
| 4 | the double-handshake quorum: one staging flag from *each* Vector core before every GEMM |
| 5 | instruction budget — 64 rotation Mmads, 8 per strip, 16 CV bursts per strip, 1 rotation Fixpipe |
| 6 | the staging core issues no Global Memory store |
| 7 | one outflow, entirely after the last Fixpipe |
| 8 | both Vector cores executed, with a per-engine instruction census |

It then reports the per-strip cadence and the measured engine overlap, neither of
which is a pass/fail condition.

### Measured

Every rung below ran on the same binary: the kernel after the §4.5, §4.6 and
§4.8 fixes. `Audit` is the nine-check trace audit of §4.9, on the core the
runner reported isolating.

| rung | block | core audited | panel entries wrong | `m`, `l` | `Acc` | egress rot. err / budget | gated ctx err / budget | wall | audit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `--strips 2` | 0 | `core0` | **0 / 512** | ≤ 2 ULP | ≤ 32 ULP | 2.98e-08 / 1.51e-04 | 1.53e-05 / 1.76e-03 | 506 s | 9/9 |
| `--strips 4` | 0 | `core0` | **0 / 1024** | ≤ 2 ULP | ≤ 80 ULP | 1.49e-08 / 6.52e-05 | 9.54e-07 / 7.55e-04 | 613 s | 9/9 |
| `--strips 8` | 0 | `core0` | **0 / 2048** | ≤ 2 ULP | ≤ 192 ULP | 1.40e-08 / 4.72e-05 | 5.96e-08 / 5.11e-04 | 897 s | 9/9 |
| `--canary 0 0` | 0 | `core0` | **0 / 256** | ≤ 2 ULP | ≤ 9 ULP | 8.94e-08 / 3.44e-04 | 1.53e-05 / 4.03e-03 | 411 s | 9/9 |
| `--canary 0 2` | 16 | `core16` | **0 / 256** | ≤ 3 ULP | ≤ 5 ULP | 5.96e-08 / 3.17e-04 | 7.63e-06 / 3.71e-03 | 420 s | 9/9 |
| `--canary 0 3` | 24 | `core24` | **0 / 256** | ≤ 1 ULP | ≤ 7 ULP | 4.47e-08 / 1.71e-04 | 1.53e-05 / 2.00e-03 | 398 s | 9/9 |
| `--canary 4 0` | 0 | `core0` | **0 / 256** | ≤ 2 ULP | ≤ 8 ULP | 5.96e-08 / 3.49e-04 | 1.91e-06 / 4.00e-03 | 415 s | 9/9 |
| `--canary 4 2` | 16 | `core16` | **0 / 256** | ≤ 2 ULP | ≤ 10 ULP | 8.94e-08 / 3.46e-04 | 9.54e-07 / 3.91e-03 | 392 s | 9/9 |
| `--canary 4 3` | 24 | `core24` | **0 / 256** | ≤ 2 ULP | ≤ 12 ULP | 4.47e-08 / 1.60e-04 | 7.63e-06 / 1.85e-03 | 400 s | 9/9 |
| `--canary 7 0` | 0 | `core0` | **0 / 256** | ≤ 1 ULP | ≤ 28 ULP | 2.98e-08 / 3.04e-04 | 3.05e-05 / 2.88e-03 | 420 s | 9/9 |
| `--canary 7 2` | 16 | `core16` | **0 / 256** | ≤ 2 ULP | ≤ 176 ULP | 4.47e-08 / 1.86e-04 | 0.00e+00 / 2.00e-03 | 406 s | 9/9 |
| `--canary 7 3` | 24 | `core24` | **0 / 256** | ≤ 2 ULP | ≤ 88 ULP | 1.49e-08 / 9.38e-05 | 9.54e-07 / 9.31e-04 | 405 s | 9/9 |

Identical on all twelve, so held out of the table: the ingress rotation at
**0 / 4096** entries wrong; all 15 guard-banded buffers intact head and tail;
the key, value and scale inputs **0 words changed**; **325,376 / 325,376**
inactive output words still `0x7FC0BEEF`, plus every unprocessed archive word;
**zero** CAModel `[error]`/`[fatal]` lines; and **zero** non-empty exception
dumps — which for the six re-run rungs means zero across all 32 cores, the
census of §4.9 having been dead before that.

Total simulator time 5,683 s, about 1 h 35 m on this host.

The score panels are compared **bit-exactly** against the host oracle, not to a
tolerance; `m`, `l` and `Acc` are ULP distances, and the two error columns are
against the per-fixture budgets §3 derives. `Acc` at `--canary 7 2` is the
loosest observed at 176 ULP, which is inside its budget and consistent with a
strip whose scores are the least separated — the ULP distance on an
accumulator is scaled by its own magnitude, and `|Acc|` there is 0.45 against
1.21 at `--canary 0 0`.

Two caveats on the twelve rungs as a set. They were **not** run concurrently:
each isolates one block, the other 31 take the early return, and §8 records
that a genuinely concurrent 32-block grid is still untested. And the six
`H=2`/`H=3` rungs are a second execution — their first pass had the same
numeric results but was audited against `core0` (§4.9), so the trace evidence
for them comes from the re-run.

### The measured steady state, 8 strips

```text
AIV1 UB->L1 staging bursts : deltas 1027 980 979 1495 963 984 963
                             columns 2..7: min 963, median 980, max 1495
AIC score Mmad groups      : deltas 19874 19905 19874 19905 19875 21360 20899
                             columns 2..7: min 19874, median 19905, max 21360
```

The staging core finishes a 16-burst key page in about 980 ticks against the
Cube's ~19,900-tick score interval, so AIV1 is idle most of the schedule and
the Cube is not what the pipeline waits on either: `AIV0 Vector` is in flight
234,863 ticks of the run against the Cube's 1,777 and AIV1's 2,457. The
recurrence is the critical path by two orders of magnitude, which is why §7
treats the Vector epilogue rather than the GEMM as the thing to attack. The
one measurement to distrust here is the 1495-tick staging outlier: it is a
single interval on a single run, and nothing isolates its cause.

---

## 7. What a 1024-token answer still needs

Each of the 32 tasks owns one 128-token page, so with a temporal split the
eight blocks of one KV head hold eight **partial** softmaxes of one 1024-token
context. The kernel writes, per block: `out_qrot`, all eight raw score panels,
`m` and `l`, the raw accumulator `out_acc` (still in the cache's rotated basis),
the pre-gate rotated context `out_rotated`, and the fused `out_context`.

The rotation is linear and commutes with a cross-block log-sum-exp combine.
**The sigmoid gate does not.** So for a genuine 1024-token result the combine
must happen before the gate, and `out_context` is the exact answer only when a
block's page is the whole context for its head — which is precisely the regime
the single-tile canary protocol runs in. `out_acc` and `out_state` are what a
combine would consume. **That combine is a separate reduction and is not in
this kernel.**

---

## 8. What remains untested

* **Silicon.** Nothing here ran on a 950PR. This host has no NPU; every number
  is cycle-accurate CAModel. No throughput, latency or power claim is made.
* **Concurrent execution of all 32 blocks.** Every run above launches the full
  `gridDim = 32` grid, and every block computes
  `GetBlockIdx() / GetSubBlockNum()` and its own `(head, temporal) → page`
  mapping in order to decide whether it is the selected one — so the block-index
  convention is exercised 32 times per run, and the canary sweep proves the
  other 31 wrote nothing. Nine different selected blocks each produced the
  correct head-specific output.

  What is *not* covered is 32 blocks doing their work at the same time. The
  reason that gap is narrow rather than alarming: mode-4 flags are
  `set_intra_block` / `wait_intra_block`, synchronising the three cores
  **inside one AI Core block**, so two grid blocks share no synchronisation
  state at all, and each has its own UB, L1 and L0. Their only shared resource
  is Global Memory, and the isolation sweep is exactly the check that their
  writes are disjoint. Still, no run here had two blocks computing
  concurrently. On this host one strip of one block is about seven minutes of
  cycle-accurate model time, so 32 blocks × 8 strips was not attempted.
* **Flag-id reuse on the three surviving cross-core channels.** §4.6 measured
  guidelines §6.2c coming true on the fourth, and the fix there was to delete
  the ring rather than to trust the edge. `kFlagValueReady`, `kFlagScoreFree`
  and `kFlagScoreReady` still carry four announcements per id over eight
  strips. They are correct on every rung run here, and the reason is **not**
  that their ids are safe: it is that in this schedule each of their consumers
  is slower than its producer, so no producer gets two announcements ahead of
  a wait. Rebalance the pipeline — a faster recurrence, a slower Cube, a
  different strip count — and that margin is gone. A design that needed to be
  robust rather than merely correct here would give those buffers one slot per
  strip too, as the key path now has.

  Removing `kFlagL1Free` also left AIV1 with **no cross-core wait at all**, so
  it is now unthrottled: it stages all eight strips within about 7000 ticks
  while the Cube is still on its first. That is safe for the L1 pages, which
  are disjoint, but it moves the exposure onto the *forward* edge
  `kFlagKeyReady` — with two ids over eight strips the Cube's wait for strip 4
  is the third announcement on id 16, and if that released early the Cube would
  read a page AIV1 had not written. What rules it out on these runs is not the
  flag: it is trace check 3, which requires each page's single `LOAD_2Dv2` to
  follow its last `MOV_UB_TO_L1`. That is a per-run observation, not a proof,
  and it is the check to keep if this kernel is changed.

  The hardware question itself is also still open. The reference says the
  mode-4 flag is a counter that increments and decrements, and §4.6 is a
  measured case of a wait proceeding when the arithmetic says the counter was
  at zero. Which of the two is wrong was not isolated.
* **Whether §4.5 and §4.6 were the only two.** They were found because the
  ladder ran rungs *past* the ring depth and the two failures had distinct
  signatures. A third fault with the same signature as either would be hidden
  behind it until the first was fixed — which is exactly how these two came
  out, one per run.
* **The mechanisms behind §4.1, §4.2 and §4.6.** All three faults are
  reproducible and all three fixes are verified. Why `CopyL0C2L1` delivered
  nothing to L0A, why a k = 256 Mmad disagreed with eight of k = 32, and why a
  mode-4 counter released a wait it should have held, were not isolated.
* **Other geometries.** 16 queries, d = 256, 8 strips, 4 heads, the two-slot UB
  rings and the eight L1 key pages are compile-time constants. Nothing here
  says anything about other shapes. In particular the L1 fix is sized by
  `kStrips`: at 112.3 KiB of 512 KiB it has room to grow, but a strip count
  above about 56 would not fit and would need the ring back — and so would need
  the §4.6 question answered first.
* **The egress on the Cube.** §3 puts the egress rotation on AIV0 as specified.
  Whether the Cube would be cheaper for it was not measured.

## 9. Retractions

None. This is the first revision of this operator, and no claim in an earlier
report is contradicted here. The `CopyL0C2L1` note in
[`reshape_and_cache_a8.md`](reshape_and_cache_a8.md) — "a `CopyL0C2L1` round
trip, which this part does support" — is **not** retracted: that note rejected a
two-pass INT8 cascade because the intermediate did not fit an INT8 operand, not
because of the copy path, and §4 here reports only that this kernel could not
make a bf16 Nz destination in L1 work and says the mechanism is unisolated.
