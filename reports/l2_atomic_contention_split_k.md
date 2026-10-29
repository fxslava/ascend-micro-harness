# L2 atomic contention and the Split-K crossover for the Cube decode pipeline

Ascend 950PR / arch35 / `dav-3510` / CANN 9.2.0-beta.2, simulator
`Ascend950PR_9599`. The probe is `KERNEL_TYPE_AIV_ONLY`; the baseline links the
frozen `MIX_AIC_1_2` decode operator unmodified.

**No NPU on this host.** Every number below is a cycle-accurate CAModel tick or
a host-oracle comparison. Nothing here is a latency, a throughput, a bandwidth
or a power figure for silicon, and §7 says which of the conclusions would
survive on hardware and which are statements about the model.

What was built:

| Path | Role |
| --- | --- |
| [`routines/routine_probe_l2_atomic_sync.hpp`](../routines/routine_probe_l2_atomic_sync.hpp) | `L2AtomicProbeEngine` and the three SIMT synchronisation VFs |
| [`kernels/camodel/l2_atomic_synthetic.cpp`](../kernels/camodel/l2_atomic_synthetic.cpp) | the probe kernel, three entry points (Modes A, B, C) |
| [`tests/common/l2_atomic_oracle.hpp`](../tests/common/l2_atomic_oracle.hpp) | the log-sum-exp combine oracle, both fixtures, the egress reference |
| [`tests/cpu/test_l2_atomic_probe.cpp`](../tests/cpu/test_l2_atomic_probe.cpp) | the Tier-1 host probe |
| [`tests/camodel/test_camodel_l2_atomic_synthetic.cpp`](../tests/camodel/test_camodel_l2_atomic_synthetic.cpp) | the CAModel runner and the three modes' gates |
| [`tests/camodel/test_camodel_cube_single_tile_baseline.cpp`](../tests/camodel/test_camodel_cube_single_tile_baseline.cpp) | the single-tile Cube baseline over the frozen operator |
| [`scripts/run_l2_atomic_probe_camodel.sh`](../scripts/run_l2_atomic_probe_camodel.sh) | build, run, archive, audit |
| [`scripts/analyze_l2_atomic_trace.py`](../scripts/analyze_l2_atomic_trace.py) | the probe trace audit, seven checks |
| [`scripts/analyze_cube_tile_baseline.py`](../scripts/analyze_cube_tile_baseline.py) | the Cube span and interval extraction |
| [`scripts/l2_atomic_crossover.py`](../scripts/l2_atomic_crossover.py) | the crossover table |

**Nothing under `kernels/ops/` was touched.** The baseline is a second runner
against the same frozen translation unit, registered in
[`cmake/SimTargets.cmake`](../cmake/SimTargets.cmake) as a separate
`ascendc_library` over
`kernels/ops/turboquant/turboquant_paged_decode_attention.cpp`. The diff to
existing files is `cmake/SimTargets.cmake` and nothing else.

---

## 0. The question, and why it cannot be answered by simulating it

The decode operator accumulates `(m, l, Acc)` in AIV0's Unified Buffer
(`kUbMPrev`, `kUbLPrev`, `kUbAcc`) across the eight-strip schedule of one
128-token page, and emits a **partial** softmax per page. Its own report says
the cross-page combine is not in it. A Qwen3.5 decode step over a
multi-thousand-token context therefore needs a temporal (Split-K) reduction,
and the design question is where that reduction lives:

* **Mode A** — every split writes its state to its own slice of a Global
  Memory workspace, and a separate pass combines them. No two cores ever
  address the same line.
* **Mode B** — the splits combine **in place** into one shared head slot under
  a Global Memory spinlock. One line, K writers, a read-modify-write each.
* **Mode C** — the tail split normalises, rotates out of the cache's channel
  basis and gates. This is the part of the single-page egress that does **not**
  commute with a cross-block combine, so it has to happen once, after it.

Simulating a 1024- or 4096-token sequence on CAModel is cycle-prohibitive: the
eight-strip single-page run is already about fifteen minutes on this host, and
a 32-split sequence would be hours per data point. So the question is answered
by **composition** rather than by simulation: this probe measures `T_sync(K)`
in isolation, the single-tile harness measures `T_compute_strip` on the real
frozen operator, and the crossover follows arithmetically. §7 states what that
composition assumes.

---

## 1. There is no SIMD atomic on this part, and that decides the kernel's shape

`asc_atomic_exch`, `asc_atomic_cas`, `asc_atomic_max` and `asc_atomic_add` are
**SIMT** API interfaces. In CANN 9.2.0-beta.2's
`asc/include/simt_api/device_atomic_functions.h` every one of them is declared

```cpp
#if (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline uint32_t asc_atomic_exch(__gm__ uint32_t* address, uint32_t val);
__SIMT_DEVICE_FUNCTIONS_DECL__ inline float    asc_atomic_max (__gm__ float*    address, float    val);
__SIMT_DEVICE_FUNCTIONS_DECL__ inline uint32_t asc_atomic_cas (__gm__ uint32_t* address, uint32_t compare, uint32_t val);
```

and `__SIMT_DEVICE_FUNCTIONS_DECL__` expands to `__simt_callee__`
(`simt_api/device_types.h`). A `__simt_callee__` is reachable only from a
`__simt_vf__` body, which only `asc_vf_call` dispatches. Two consequences are
load-bearing, and the second one would have failed silently.

**The lock cannot live on the Cube side, or anywhere the AIC leg compiles.**
`simt_api/common_functions.h` reads:

```cpp
#ifndef SPLIT_CORE_VEC
#if (defined(__DAV_VEC__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510))
#define SPLIT_CORE_VEC
#endif
#endif

template <auto funcPtr, typename... Args>
__aicore__ inline void asc_vf_call(dim3 threadNums, Args&&... args)
{
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 5102) || defined(SPLIT_CORE_VEC) || defined(ASCENDC_CPU_DEBUG)
    cce::async_invoke<funcPtr>(threadNums, args...);
#endif
}
```

At `__NPU_ARCH__ == 3510` — which is this part — `SPLIT_CORE_VEC` is defined
**only** when `__DAV_VEC__` is, i.e. in the AIV compilation leg. In the AIC leg
the body is empty: `asc_vf_call` compiles to nothing, with no diagnostic. A
Split-K combine written into the Cube's code path would therefore have had its
lock silently removed and would have raced. That is why this probe is
`KERNEL_TYPE_AIV_ONLY` and why the trace audit asserts the VF dispatch count
rather than trusting it.

**Every pointer the VFs dereference is `__gm__`, deliberately.** A `__ubuf__`
argument to a SIMT VF must live in the declared dynamic UB region or the first
SIMT UB access faults with AIV error 341 on every core, which forces
`dyn_ub_size` into the launch chevron and a single UB allocation mechanism onto
the whole kernel. The lock, the ticket counter, the spin census and the
atomically reduced maximum are all Global Memory words, so this probe needs
none of that and keeps the repository's hand-placed `LocalTensor` idiom with
the chevron's second argument left as `nullptr`.

### What the three VFs lower to

Read off the instruction dump, not assumed:

| VF | lowering |
| --- | --- |
| atomic maximum | `SIMT_LDG` (the `asc_ldcg` of this split's `m`), `SIMT_RED` (the `asc_atomic_max` — a *reduction*, because the return value is discarded), `SIMT_END` |
| acquire | `SIMT_START_DVG` / `SIMT_BRANCH` / `SIMT_END_DVG` for the `threadIdx.x != 0` guard, then one `SIMT_ATOM` per exchange attempt with `SIMT_ISETP_I` + `SIMT_BRANCH` closing the spin, then one `SIMT_ATOM` for the ticket and two `SIMT_STG` for the logs |
| release | `SIMT_MEMBAR` (the `asc_threadfence`), `SIMT_ATOM`, `SIMT_END` |

The exchange is `SIMT_ATOM` and the maximum is `SIMT_RED`: the ISA
distinguishes an atomic whose result is consumed from one whose is not, which
is why the spin census can be read straight out of the atomic count.

That lowering is also why the audit matches mnemonics for **equality** and not
as substrings. `SIMT_END` is a prefix of `SIMT_END_DVG` and `SIMT_ST` of
`SIMT_START_DVG`, and the divergent guard emits several of each: counted by
substring, a three-VF repetition reports six VF ends and eight stores. That
was the first version's bug and the trace said so immediately — six dispatches
against twelve ends.

### Why `asc_atomic_max` and not a CAS loop

The scalar `m` reduction needs no lock at all: `asc_atomic_max` on a
`__gm__ float*` is one instruction. `asc_atomic_cas` is available with the same
memory contract and would permit the usual compare-and-swap loop, but using it
where a hardware float maximum exists would measure the cost of a loop this
part does not need. The probe therefore uses `asc_atomic_max` for `m`,
`asc_atomic_exch` for the test-and-set lock the task specifies, and
`asc_atomic_add` for the ticket. The lock is a plain test-and-set and not a
test-and-**test**-and-set: a TTAS that spins on `asc_ldcg` and only then
attempts the exchange is a cheaper protocol, and substituting it would have
answered a different question than the one asked. §6 says what it would likely
change.

**`asc_atomic_max` orders negative floats correctly on this model.** The
published contract says the interface takes a `__gm__ float*` and returns the
initial value; it does not say how the comparison is implemented, and a
sign-magnitude integer comparison would order negatives backwards. Decode `m`
is negative in general, so both fixtures here use negative maxima deliberately
and the runner checks the reduced value **bit-exactly** against the host
maximum. It matches at every rung run.

---

## 2. What makes the device check a lost-update detector

The combine is the standard log-sum-exp merge

```
m'   = max(m, m_t)
a    = exp(m  - m'),   a_t = exp(m_t - m')
l'   = l . a + l_t . a_t
Acc' = Acc . a + Acc_t . a_t
```

which is associative and commutative in exact arithmetic but **not** in FP32:
the rescale factors depend on the order the splits arrive in, and a spinlock
fixes no order. A probe that compared the combined accumulator against one
canonical order could therefore not tell a lost update from a different — and
equally valid — interleaving.

The **uniform** fixture removes that ambiguity instead of tolerating it. Give
every split the same `m`, and both factors are `exp(0) = 1.0f` exactly, in
every order, so the combine degenerates to `l' = l + l_t`, `Acc' = Acc + Acc_t`
— a plain sum. Make every addend a small integer and the sum is exact in FP32
whatever the order. The addends are then chosen so the sum **names** its
contributors: split `k` contributes `2^k . (c + 1)` to channel `c`, so the
combined accumulator is `(2^K - 1)(c + 1)` and a missing split `k` leaves
`(2^K - 1 - 2^k)(c + 1)`. The widest value, at `K = 8` and eight repetitions,
is `255 . 8 . 256 = 522240` — inside FP32's exact integer range, which the
Tier-1 probe checks rather than assumes.

So the Mode B accumulator gate is `==`, not a tolerance, and a broken critical
section produces a bit-level mismatch whose shortfall identifies the split
that lost its update.

The first combine step still exercises the identity branch: the workspace
starts at `m = -3.0e38`, `l = 0`, `Acc = 0`, so `a_curr = exp(-3.0e38) = 0`
exactly and the first contribution passes through unscaled. That `Exp`
underflows to a true zero rather than a denormal is confirmed by the gate being
bit-exact at every rung.

The **staggered** fixture is the only one that exercises the rescale at all:
distinct negative maxima, not monotone in the split index, so a reduction that
happened to be ordered by block index cannot pass by accident. Its accumulator
is checked against the oracle replaying **the device's own ticket log**, so the
arrival order is the one that happened rather than an assumed one.

### What the Tier-1 probe settled before any simulator run

`bin/test_l2_atomic_probe`, milliseconds:

```text
uniform fixture: order independence and the closed form
  acc[255] at K=8 reps=8 is 522240, inside 2^24
staggered fixture: the rescale path
  two interleavings differ by at most 2.84e-07 relative
Mode C reference: staged egress against the dense matrix
  worst relative difference 0 over 256 channels
  the gate moved 112 of 112 non-zero channels
diagnostics: the uniform shortfall note
  channel 0: measured 223.0, expected 255.0, shortfall 32.0; consistent with
  one lost update from split(s) 5
```

Three of those lines are load-bearing. The uniform combine is bit-identical
across sixteen randomised interleavings at every `(K, reps)` the sweep uses —
which is what licenses `==` on the device. The staggered one is **not**: two
interleavings differ by 2.8e-07 relative, which is why that fixture is checked
against the replayed log and to a tolerance. And the staged RegGather egress
reproduces the dense 256×256 right-multiplication by Π to **zero** relative
difference, so the Mode C reference is a reference and not a second copy of the
device's algorithm.

The gate check needed correcting once and the correction is the interesting
part: the first version asserted that the sigmoid gate moves *every* channel,
and it failed. A right-multiplication by Π of a linear ramp is mostly zeros —
144 of 256 channels — and `sigmoid(gate) . 0 = 0` for every gate. The property
is that every **non-zero** channel moves, and that there are enough of them for
the check to mean anything.

---

## 3. The probe's shape, and the four edges that make Mode B correct

The state record mirrors AIV0's resident state exactly, so the bursts the probe
issues are the bursts a real combine would issue:

| field | form | bytes | DataBlocks |
| --- | --- | --- | --- |
| `m` | one lane-replicated 32-byte block | 32 | 1 |
| `l` | one lane-replicated 32-byte block | 32 | 1 |
| `Acc` | 256 FP32 | 1024 | **32** |

272 FP32 = 1088 B in all. `m` and `l` are adjacent in one 64-byte slot with no
bank skew between them, so the pair moves as a single 2-DataBlock burst and the
accumulator as the task's 32-burst transfer: two MTE instructions in each
direction, which is what makes the Mode A / Mode B comparison like for like —
same record, same bursts, and the only difference is the synchronisation and
the read-modify-write.

`m` and `l` are carried as replicated DataBlocks and not as floats because a
logically scalar value the Vector pipe produced must never be read back by the
Scalar unit: that needs a `V_S` fence, which costs a full vector drain and
needs a `TPipe` this kernel does not have
(`MICROROUTINE_DESIGN_GUIDELINES.md` §1.2).

Mode B's critical section is bracketed by four intra-core edges, and none of
them is a barrier:

| edge | what it prevents |
| --- | --- |
| `V_MTE2` | the acquire VF issues on the Vector pipe and `asc_vf_call` dispatches it **asynchronously**, so without this the MTE2 fetch — on a queue nothing else gates — reads the shared record before the lock is held. This is the edge that makes the critical section critical. |
| `MTE2_V` | the fetched record is readable by the combine |
| `V_MTE3` | the combined record is final in UB |
| `MTE3_V` | the write-back has **retired**, so the release VF may clear the lock word. The `asc_threadfence` inside the release orders accesses; it does not wait for another pipe's outstanding burst, which is why the retire has to be proved here instead. |

### One scalar Global Memory read cost a reserved event id

The first version resolved this split's `m` with `GlobalTensor::GetValue` on
the SIMD side — the idiom the decode operator uses for its block table — and
passed the value to the VF. The trace audit rejected the run:

```text
1. intra-core HardEvent balance
core 0: event ids 6 and 7 are reserved, saw [('MTE2', 'SCALAR', 7)]
```

The library implements that scalar Global Memory read with an `MTE2 -> S`
synchronisation on reserved event id 7, on a route the kernel never writes. The
fix is that the VF reads the value itself with `asc_ldcg`, which is also the
right primitive on its own terms: `asc_ldcg` loads through L2/Global Memory
without populating the per-core SIMT DataCache, and that DataCache is not
coherent with Global Memory. The audit is green afterwards, and the probe gained
a `SIMT_LDG` the census now expects.

---

## 4. Two things the trace said that the runner could not

### 4.1 Under `KERNEL_TYPE_AIV_ONLY`, split k is not core k

The audit's first version read `core<k>.veccore0` for split `k` — the mapping
the decode operator's audit uses, where the grid is `MIX_AIC_1_2` and block N
runs on core N. At `K = 2` it reported that core 1 had done nothing:

```text
core 1: 0 GM read(s) and 0 GM write(s), expected 6 and 5
```

while the runner's own gates had already passed, including split 1's
contribution landing bit-exactly. A census over every dump settled it:

```text
core0.veccore0.instr_popped_log.dump   GMread 6  GMwrite 5  VF 6
core0.veccore1.instr_popped_log.dump   GMread 6  GMwrite 5  VF 6
```

An AIV is not an AI core. Each 950PR AI core carries **two** Vector cores and
the model dumps them separately, so the mapping is

```
split k  ->  core<k / 2>.veccore<k % 2>
```

The audit now asserts this with a census over every dump instead of trusting
it, which turns a future mapping change into a diagnostic that names the
mapping rather than a core that "did not run the probe body".

**This matters for the numbers and not only for the audit.** At `K = 8` the
eight splits are four AI cores with two Vector cores each, so pairs of splits
share whatever their AI core shares on the way to L2. Every `K >= 2` figure
below is therefore a mixture of inter-core and intra-core sharing, not eight
independent cores, and §7 keeps that caveat attached.

### 4.2 Mutual exclusion, measured from the data instructions

The runner proves mutual exclusion from the **ticket log**: each acquisition
takes a ticket from a Global Memory counter while holding the lock, and across
the launch the logged tickets must be a permutation of `0 .. K.reps-1`, so a
duplicate means two cores were inside at once.

The audit proves it independently, from the **data** instructions: each split's
critical section runs from its Global Memory read of the shared record to the
retirement of its write-back, and across cores those windows must be disjoint.
The two fail independently — a lock that never excludes still produces a clean
ticket log if its races happen not to overlap on one run, and a broken ticket
counter would not perturb these windows — and the second uses only ticks whose
meaning is not in question.

A third, weaker witness agrees with both: the spin count. The kernel counts
failed exchanges in a thread register and publishes it; the audit derives the
same number from the atomic instruction count, as `SIMT_ATOM` minus the
uncontended floor of three per repetition. **The two agree exactly at every
rung.**

---

## 5. Results

### 5.1 The Cube baseline, and the premise it corrects

Three rungs over the frozen operator, block 0 isolated, every gate bit-exact
(`qrot 0 bad, scores 0 bad / 0 canary, beyond-window 0, other blocks 0`):

```text
rotation prologue, 64 Mmads, once per launch      801 ticks
per-strip SPAN (first score Mmad -> Fixpipe retired)
  strip 0, isolated                               180
  strip 7, isolated                               180
  all eight strips of the 8-strip rung            180 every one
per-strip INTERVAL (between score Mmad groups, 8-strip rung)
  19874  19905  19874  19905  19875  21360  20899
  steady state over 6 intervals: min 19874, median 19905, max 21360
whole Cube launch, 8 strips                   168878 ticks
AIV0 vector pipes in flight, 8 strips         234863 ticks over 2564 spans
```

The interval figures reproduce `reports/paged_decode_attention.md` §6 digit for
digit, from an independent runner and an independent analyser, which is the
cross-check that makes the rest of this usable.

**The task's premise that `T_compute_strip ≈ 40,000` ticks does not hold, and
it is wrong in two different directions at once.** The Cube's own cost for one
16-token strip is **180 ticks** — eight `Mmad`s and a `Fixpipe` — which is 220
times smaller. The rate at which the pipeline *retires* a strip is **19,905
ticks**, which is about half the premise. The gap between 180 and 19,905 is
not Cube work at all: over the same eight-strip run AIV0's vector pipes are in
flight 234,863 ticks, against 1,440 ticks of Cube score-GEMM span (eight strips
at 180 each). The strip interval is set by the online-softmax recurrence and
the Cube waits on it. Both of those figures also match
`reports/paged_decode_attention.md` §6, which measures AIV0 at the same 234,863
ticks.

That matters for the question being asked, not just for tidiness. "The Cube
compute interval" a Split-K synchronisation has to hide behind is the
**recurrence** interval. Using 40,000 would have halved every overhead figure
below; using 180 would have inflated them by two orders of magnitude.

### 5.2 Mode A — no contention at any K, and none appears

Per split, per reduction (total cycles / 4 repetitions):

| K | per-split publication cost (ticks) |
| --- | --- |
| 1 | 555 |
| 2 | 561, 549 |
| 4 | 556, 560, 562, 570 |
| 8 | 553, 561, 551, 574, 549, 579, 554, 581 |

Eight cores writing eight disjoint 1088-byte records cost each of them what one
core writing one record costs: **549 to 581 ticks at every K**, with no trend.
Two MTE3 bursts and a drain. The variation *across* K (555 at K = 1 against a
mean of 563 at K = 8) is smaller than the variation *within* K = 8 alone (549
to 581), so there is no contention signal here to separate from the noise — and
the trace agrees, putting the write bursts at a 532-tick median with nine of
them in flight per split.

### 5.3 Mode B — the whole K-dependence is in the spin, and nowhere else

Per split, per reduction, split into the four phases:

| K | atomic max | acquire (spin) | critical section | release | total (min–**worst**) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1228 | 1926 | 1532 | 628 | **5314** |
| 2 | 1148–1183 | 3528–4246 | 1516–1524 | 624–660 | 6824–**7604** |
| 4 | 1157–1215 | 4876–11071 | 1496–1528 | 635–708 | 8203–**14379** |
| 8 | 1110–1239 | 11408–24026 | 1489–1532 | 587–692 | 14644–**27392** |

Read the columns, not the totals. Three of the four phases are **flat in K**:

* the atomic maximum, 1,110 to 1,239 ticks at every K — one `asc_atomic_max`
  on a Global Memory word that all K cores hit, and it does not get slower;
* the critical section, 1,489 to 1,532 ticks at every K — the MTE2 fetch, the
  combine, the MTE3 write-back on the **shared** line, and it does not get
  slower either;
* the release, 587 to 708 ticks at every K.

Only the acquire scales, from 1,926 ticks uncontended to a mean of 17,987 at
K = 8. So on this model the cost of sharing one line is **serialisation and
nothing else**: a split waits for the splits ahead of it, and when its turn
comes its own work costs exactly what it costs alone. §7 is explicit that this
is a statement about the model.

A closed form for that claim, which the measurement then tests. If the only
cost is serialisation then the **worst** split waits behind all `K-1` others,
each holding the line for one critical section plus one handover, and then
runs its own:

```
T_sync(K) ≈ (K-1).(critical + handover) + critical  ≈  K.(critical + handover)
```

With the trace's own critical section and handover medians, the prediction
against the measured worst split is:

| K | predicted `K.(cs+gap)` | measured worst | error |
| --- | --- | --- | --- |
| 2 | 7292 | 7604 | −4.1% |
| 4 | 13720 | 14379 | −4.6% |
| 8 | 27176 | 27392 | −0.8% |

Three points inside 5% with no fitted parameter. `scripts/l2_atomic_crossover.py`
prints that column beside the measurement.

The trace agrees too, and from instructions rather than counters. On the
K = 8 rung:

```text
3. mutual exclusion, from the data instructions
   32 critical section(s) across 8 split(s), none overlapping
  critical section length            n= 32  min 1382  median 1508  max 1546
  handover gap (different splits)    n= 31  min 1621  median 1889  max 2134
   the shared line was held 48069 of 105934 tick(s) (45.4%)

4. Global Memory burst in-flight time (dispatch to completion)
   the shared record only, pooled over every split
     shared-line read   n=64  min 378  median 525  max 567
     shared-line write  n=64  min 395  median 541  max 581
```

Thirty-two critical sections across eight splits, **none overlapping in
time** — mutual exclusion measured from the data instructions, not from the
lock's own bookkeeping. And the pooled shared-line bursts at K = 8 (medians
525 and 541) are indistinguishable from the private ones in Mode A at the same
K (median 532), and from the K = 2 shared ones (medians 540 and 548). Eight
writers on one line do not slow a burst down.

Two independent spin censuses agree **exactly** at every rung: at K = 8 the
kernel's thread-register counter reports 874 failed exchanges over 32
acquisitions and the audit derives 874 from `SIMT_ATOM` minus the uncontended
floor of three per repetition.

**The uncontended reduction is dominated by VF dispatch, not by atomics.** At
K = 1 the trace puts the three SIMT VF windows at 609, 1,373 and 680 ticks
(median dispatch-to-`SIMT_END`-retire) — 2,662 of the 5,314-tick reduction,
**50%**, for entering and leaving SIMT mode three times. The phase counters
attribute more, 3,782 ticks or 71%, because each phase boundary carries a
`PipeBarrier<PIPE_ALL>` drain that the counter charges to the phase; the
trace's 50% is the tighter figure and the one to believe. Either way this term
is independent of K, and at K = 1 it is the whole cost.

### 5.3a The issue-stall census, and what it does not contain

Per split, from the model's CCU log, one line per cycle in which the core could
not issue:

| mode | K | stall cycles per split | of which with a GM burst outstanding | inside the VF windows |
| --- | --- | --- | --- | --- |
| A | 8 | 3448–3590 | 2695–2805 | — |
| B | 2 | 1926–1981 | 635–707 | **0** |
| B | 8 | 1720–2046 | 652–729 | **0** |
| C | 8 | 4764 | 3344 | — |

Two readings, and the second is the useful one.

The stall totals are **flat in K** like everything else — 1,926–1,981 at K = 2
against 1,720–2,046 at K = 8 — which is the same conclusion a third time.

And **zero** stall cycles fall inside the acquire VF's window, at any K, even
though that window is where a split spends up to 56,675 ticks waiting. That is
not a contradiction: while the VF runs, the core is *issuing SIMT
instructions* — the spin is a loop of `SIMT_ATOM`, `SIMT_ISETP_I`,
`SIMT_BRANCH` — so a spinning core is busy, not stalled. The practical
consequence is that **this model's stall counter cannot see lock contention at
all**, and any attempt to measure spin cost through it would report zero. The
spin is visible only in the VF window length and in the atomic count.

### 5.4 Mode C — the tail egress, bit-exact, with a model diagnostic

The tail split's flush, per reduction:

| normalise (`Acc/l`) | rotate (butterfly) | gate (sigmoid) | store | total |
| --- | --- | --- | --- | --- |
| 1714 | 354 | 1737 | 570 | **4373** |

The rotation is the cheapest part by a factor of nearly five. The
register-resident RegGather butterfly does all eight stages of `x . Π` on one
256-channel row in **354 ticks**, while each of the two 0-ULP divides around it
costs about 1,725. A whole tail egress costs 4,373 ticks, which is less than
one uncontended Mode B reduction and about a fifth of the K = 8 worst case —
the terminal stage is not where a Split-K design's cost lives. The output is **bit-exact** against the host reference —
worst relative difference 0 across all 256 channels, after bf16 rounding.

**The model raises a diagnostic on this mode that the production operator's
identical egress does not raise, and it is not resolved.** Four times per
repetition:

```text
[error] [core3.veccore1] [vec_err_idata_inf_nan_t0] (PC: 0x10d0f550)
        RVECEX : (ID: ...) RV_VABS_FP Dtype: F32 Vd[14], Vn[14], Pg[1],
```

What was established about it, in order:

1. **It is not a wrong answer.** The gated context matches the host reference
   bit-for-bit on the same runs that raise it.
2. **It is not the uninitialised bank-skew gap behind the `Duplicate` source.**
   That was the first hypothesis. The probe now writes a finite value across
   every `Duplicate` source's whole 256-byte read window before anything reads
   it — which is a correct precaution on its own terms and is kept — and the
   flag survived the change unaltered, same count, same position.
3. **It is not a non-finite operand.** The probe publishes the two divides'
   operands to Global Memory after the measured region and the runner scans
   them. All finite, in every channel, on the run that flagged:

   ```text
   l_broadcast (divide 1 denominator)  non-finite 0, range [1020, 1020]
   rotated     (divide 2 numerator)    non-finite 0, range [-911, 911]
   gate_tmp    (divide 2 denominator)  non-finite 0, range [1.0523, 20.1216]
   ```

   The fourth operand is the host-seeded accumulator, which the runner already
   verifies unchanged. `l_broadcast` reading exactly 1020 in all 256 channels
   also settles, in passing, that `Duplicate`'s tensor form broadcasts lane 0
   and not the source register.
4. **It originates inside the divide's own expansion.** The 0-ULP `Div`
   materialises FP32 classification constants in vector registers — the
   instruction dump shows `RV_VDUPS Vd[3], Sn[18]=0x0, Sn2[19]=0x7f80`, which
   is `0x7f800000`, **+Infinity**, alongside the denormal boundary patterns
   `0x007fffff` and `0x807fffff` — and the flagged `RV_VABS_FP` sits in that
   classification chain.

What is **not** explained is why the production decode operator, which issues
the same two `Div<float, PRECISION_0ULP_FTZ_FALSE>` calls over the same
256-lane shapes, does not raise it on its own runs. Register allocation and
operand exponent range are both plausible and neither was isolated. The
runner's log scan and exception-dump check therefore mark this rung **FAIL**,
which is the correct conservative verdict and is left in place: a model
`[error]` line is not something to suppress because the numbers happen to be
right.

### 5.5 The staggered fixture, and timing that does not depend on the data

The staggered rung is the only one that exercises the rescale: distinct
negative maxima, not monotone in the split index. Two results.

**The combine is exact, not merely within tolerance.** Against the oracle
replaying the device's own ticket log:

```text
K = 2:  Mode B staggered: worst relative difference 0 at channel 0 over 8 acquisition(s)
        Mode B maximum: -2, atomically reduced and carried
K = 8:  Mode B staggered: worst relative difference 0 at channel 0 over 32 acquisition(s)
        Mode B maximum: -1, atomically reduced and carried
```

Zero, over 32 acquisitions with the rescale active, against a 1e-5 gate. The
host oracle and the device agree bit-for-bit once the oracle is given the real
arrival order — which is also the strongest available evidence that the ticket
log *is* the real arrival order. And the reduced maximum is correct and
bit-exact over negative floats at both K: −2 is the maximum of the two-split
fixture and −1 of the eight-split one, so **`asc_atomic_max` orders negative
FP32 correctly on this part**, which the published contract does not say.

**The cycle counts are identical to the uniform rung's, to the digit.** Every
split, every phase, both K. That is not a copying accident — it is what the
runs produced — and it says something worth keeping: the model's timing here is
independent of the operand values. The two fixtures issue the same
instructions over the same addresses and differ only in what the floats are,
so the cycle figures in §5.3 characterise the protocol rather than the
fixture. It also means the determinism is per build and not per run: Mode A
reproduced to the digit across two separate sweeps, and Mode B's figures moved
by 1–4% between builds (5,288 to 5,314 at K = 1) when an unrelated instruction
was added to its prologue, because shifting the code shifts the arrival
pattern that the spin outcome depends on.

---

## 6. The crossover

```
T_compute(T_tile) = (T_tile / 16) . T_compute_strip
overhead %        = T_sync(K) / T_compute(T_tile) . 100
```

with `T_compute_strip = 19905` (the measured steady-state strip interval, §5.1)
and `T_sync(K)` the **worst** split's per-reduction cost, because the makespan
is governed by the slowest split. `scripts/l2_atomic_crossover.py` recomputes
all of this, and takes `--strip-interval` so a reader can substitute a
different denominator.

| T_tile (tokens/split) | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strips per split | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| T_compute (ticks) | 19905 | 39810 | 79620 | 159240 | 318480 | 636960 | 1273920 |
| **Mode A** (581) | **2.9** | 1.5 | 0.7 | 0.4 | 0.2 | 0.1 | 0.0 |
| Mode B, K=1 (5314) | 26.7 | 13.4 | 6.7 | 3.3 | **1.7** | 0.8 | 0.4 |
| Mode B, K=2 (7604) | 38.2 | 19.1 | 9.6 | 4.8 | **2.4** | 1.2 | 0.6 |
| Mode B, K=4 (14379) | 72.2 | 36.1 | 18.1 | 9.0 | 4.5 | **2.3** | 1.1 |
| Mode B, K=8 (27392) | 137.6 | 68.8 | 34.4 | 17.2 | 8.6 | 4.3 | **2.2** |

Overhead in per cent; the bold entry in each row is where it first drops below
the 3% target.

Every `T_sync` here is the **worst** split's, Mode A's included: 581 is the
slowest of the eight publications at K = 8, not their mean of 563.

One caveat on the Mode A row, because it is the most favourable number here and
the easiest to overstate: 581 ticks is the cost of a split **publishing** its
own state to its own slice. A complete Mode A strategy also needs a reduction
pass over the K records afterwards, and **that pass is not measured anywhere in
this report.** The row is therefore the cost of the part Mode A replaces Mode B
with, not the cost of the whole strategy; what it establishes is that the
*per-split* term stops mattering immediately, leaving the reduction pass as the
only thing left to measure.

As a chart, with `*` marking below-target:

```text
overhead % of the measured strip interval (19905 ticks)
                16     32     64    128    256    512   1024   tokens/split
  Mode A       2.9*   1.5*   0.7*   0.4*   0.2*   0.1*   0.0*
  B  K=1      26.7   13.4    6.7    3.3    1.7*   0.8*   0.4*
  B  K=2      38.2   19.1    9.6    4.8    2.4*   1.2*   0.6*
  B  K=4      72.2   36.1   18.1    9.0    4.5    2.3*   1.1*
  B  K=8     137.6   68.8   34.4   17.2    8.6    4.3    2.2*
             |======== 3% target crossed left-to-right ========>
```

**The exact crossover tile sizes**, as the smallest whole number of 16-token
strips whose overhead is under 3%:

| strategy | T_sync (ticks) | crossover T_tile | strips | total context at that K |
| --- | --- | --- | --- | --- |
| Mode A, any K | 581 | **16 tokens** | 1 | — |
| Mode B, K=1 | 5314 | **144 tokens** | 9 | 144 |
| Mode B, K=2 | 7604 | **208 tokens** | 13 | 416 |
| Mode B, K=4 | 14379 | **400 tokens** | 25 | 1600 |
| Mode B, K=8 | 27392 | **736 tokens** | 46 | 5888 |

### What this says about the design

**Mode A is under the target at the smallest tile the operator can process,
but only just.** One 16-token strip per split puts the collision-free
publication at **2.9%** — inside the 3% target with about 3% of margin, which
is close enough that it should be read as "one strip is the break-even point"
rather than as comfortable headroom. At two strips it is 1.5% and from there it
falls away. There is no crossover to search for: the per-split publication
stops mattering immediately, and what is left to measure is the reduction pass
Mode A implies, which this report does not measure.

**Mode B's crossover grows with K very nearly linearly**, which follows
directly from §5.3: the acquire is the only phase that scales, and it scales
because the worst split waits behind all the others. The closed form there,
`T_sync(K) ≈ K.(critical + handover)`, lands within 5% at every K with no
fitted parameter, so the crossover column is a consequence of the
serialisation and not an empirical curve.

**At K = 8 an in-place locked combine needs 736 tokens per split**, i.e. a
5,888-token context before it costs under 3%. Qwen3.5 decode at the context
lengths this pipeline is built for would clear that, but not comfortably, and
the margin buys nothing: Mode A reaches the same target at 16 tokens and needs
no lock, no atomic and no SIMT VF at all.

**The thing to attack is not the contention.** At every K the contended
acquire is the largest term, but the *uncontended* floor of 5,314 ticks is
already 26.7% of a one-strip split, and half of that floor is three SIMT VF
dispatches. A protocol that did the whole reduction in one VF entry — acquire,
combine and release inside a single `asc_vf_call`, which would also need the
accumulator in dynamic UB and `dyn_ub_size` in the chevron — would remove two
of the three dispatches before touching the lock at all. That is measurable
with this harness and was not built; §7 lists it.

The three cheaper designs the measurements point at, in order of how much they
remove:

1. **Mode A plus a separate reduction pass.** The combine is then a one-core
   read of K records, and nothing synchronises. Its cost is not measured here.
2. **Lock-free accumulation.** `asc_atomic_add` on `__gm__ float*` exists on
   this part, so a two-pass scheme — reduce `m` with `asc_atomic_max`, barrier,
   then accumulate `Acc . exp(m_tile - m)` with 256 atomic adds — needs no
   critical section at all. It trades the 1,526-tick critical section for 256
   atomics per split and changes the FP32 summation order, so the bit-exact
   gate in §2 would have to become a tolerance.
3. **Test-and-test-and-set.** The spin here is a bare `asc_atomic_exch`, so
   every waiting split issues an atomic round trip per attempt. Spinning on
   `asc_ldcg` and attempting the exchange only when the lock reads free would
   cut the atomic traffic by the mean spin count (3.5 per acquisition at K = 2)
   without changing the protocol. On this model that would not change
   `T_sync`, because the critical section is not slowed by the traffic; on
   silicon it might. The traffic it would remove is large: 874 failed
   exchanges over 32 acquisitions at K = 8, a mean of 27 per acquisition and a
   worst case of 131.

---

## 7. What these runs do not establish

* **Silicon.** Nothing here ran on a 950PR. This host has no NPU; every figure
  is a cycle-accurate CAModel tick. No latency, throughput, bandwidth or power
  claim is made.

* **Bus or L2 stall cycles, in the sense the task asks for.** These dumps
  contain no memory-system record at all: the CCU log models the core's own
  instruction-issue pipeline, the dcache log is the per-core SIMT DataCache tag
  array, and there is no per-transaction L2 log, no bank-conflict record and no
  queue-occupancy record. So cycles lost *inside* the memory system cannot be
  measured from these runs. What was measured instead, and reported as such:
  the outstanding time of each Global Memory burst (dispatch to completion) and
  the per-core issue-stall cycles attributed to the spin and the critical
  section. Those are different quantities and the report does not relabel one
  as the other.

* **That a hot shared line is free on hardware.** §5.3 shows no contention
  effect of any kind: the critical section, the atomic maximum and the Global
  Memory bursts take the same time whether one split or eight address the same
  record. Two explanations fit equally well — the 950PR L2 genuinely does not
  degrade for one hot line at K ≤ 8, or this model does not represent L2
  arbitration at that granularity — and **these runs cannot distinguish them.**
  Every "contention" conclusion here is therefore a statement about
  serialisation, which is observable, and not about bus behaviour, which is
  not. If the model under-represents L2 arbitration, the real Mode B crossovers
  are larger than the table, never smaller, so the recommendation in §6 is on
  the safe side of this uncertainty.

* **That the flag edges alone provide mutual exclusion.** Every measured region
  is bracketed by `PipeBarrier<PIPE_ALL>` for the cycle counter, and that
  barrier *also* orders the acquire VF ahead of the MTE2 fetch — the job the
  `V_MTE2` edge exists to do. So the exclusion these runs demonstrate holds for
  an instrumented probe; a production combine without the barriers would rest
  on the `V_MTE2` edge by itself, and **that configuration was not run.** This
  is the gap most worth closing before the protocol ships, and closing it means
  a second, uninstrumented entry point whose only evidence is the ticket log
  and the trace windows.

* **Throughput.** The barriers drain the pipeline at every phase boundary, so
  every cycle figure is a region *latency*. A Split-K reduction is a serial
  interval between two compute intervals, which is why latency is the right
  instrument, but these numbers must not be compared against a pipelined
  per-strip interval without saying which is which. §6 uses the measured strip
  interval as the denominator precisely because that one is a rate.

* **That the sync cannot be overlapped.** The crossover table adds a
  separately-measured `T_sync` to a separately-measured `T_compute`, i.e. it
  assumes the reduction overlaps with nothing. A scheme that reduced split `s`
  while split `s+1` computed would do better, and the table would then be an
  upper bound rather than an estimate. Nothing here measures such a scheme.

* **K beyond 8, and K = 8 on eight separate AI cores.** `kProbeMaxSplits` is 8,
  and under `KERNEL_TYPE_AIV_ONLY` split `k` lands on
  `core<k/2>.veccore<k%2>`, so the K = 8 rung is four AI cores with two Vector
  cores each. Pairs of splits therefore share whatever their AI core shares on
  the way to L2, and every K ≥ 2 figure is a mixture of inter-core and
  intra-core sharing. A launch that put eight splits on eight distinct AI cores
  — 16 blocks with only the even ones working — would separate the two and was
  not run.

* **The real operator's state was never reduced.** The probe's tile states are
  host-seeded into exactly the shape AIV0 leaves them in, deliberately, so that
  the measurement is of the synchronisation and not of the state's production.
  No run drives the frozen decode operator's `out_state`/`out_acc` through this
  combine, and the end-to-end multi-page answer the §0 question is ultimately
  about is still not computed anywhere.

* **A Split-K reduction is not bit-reproducible, even though these runs were.**
  The staggered rung matched its oracle to zero relative difference, but only
  because the oracle replayed *that run's* ticket log: the Tier-1 probe
  measures two interleavings of the same fixture differing by 2.84e-07
  relative, so the answer genuinely depends on the arrival order and the order
  is not fixed by anything. These runs agreed exactly because the model is
  deterministic per build (§5.5), which hardware will not be. Any downstream
  comparison against a Split-K result has to tolerate an order-dependent
  difference of a few ULP; the bit-exact uniform gate exists to catch lost
  updates, not to promise reproducibility.

* **Mode C's model diagnostic.** §5.4 establishes what it is not — not a wrong
  answer, not the uninitialised skew gap, not a non-finite operand — and
  locates it inside the 0-ULP divide's own Inf/NaN classification expansion.
  Why the production operator's identical divides do not raise it is
  **unexplained.** Register allocation and operand exponent range were both
  plausible; neither was isolated.

* **The three cheaper designs §6 points at.** None is built or measured: the
  separate reduction pass that Mode A implies, the lock-free
  `asc_atomic_add` accumulation, and the single-VF-entry protocol that would
  remove two of the three dispatches dominating the uncontended floor. The
  last of those needs the accumulator in dynamic UB and a `dyn_ub_size` in the
  launch chevron, which this probe deliberately avoids and which brings the
  error-341 class of failure into scope.
