# The AIV-only single-token decode operator on CAModel

Roadmap kernel №3, `turboquant_single_token_decode_aiv`. Ascend 950PR / arch35 /
`dav-3510` / CANN 9.2.0-beta.2, `KERNEL_TYPE_AIV_ONLY`, simulator
`Ascend950PR_9599`.

**No NPU on this host.** Every number below is a cycle-accurate CAModel count or a
host-oracle comparison. Nothing here is a latency, a throughput or a power figure, and
the trace spans are dispatch spans over a single launch — fill plus drain — not
steady-state intervals.

What was built:

| Path | Role |
| --- | --- |
| [`routines/routine_vector_decode_gemv.hpp`](../routines/routine_vector_decode_gemv.hpp) | the score GEMV leaf |
| [`kernels/ops/turboquant/turboquant_single_token_decode_aiv.cpp`](../kernels/ops/turboquant/turboquant_single_token_decode_aiv.cpp) | the operator |
| [`tests/common/single_token_decode_oracle.hpp`](../tests/common/single_token_decode_oracle.hpp) | the 6-row golden model and fixture |
| [`tests/cpu/test_single_token_decode_probe.cpp`](../tests/cpu/test_single_token_decode_probe.cpp) | the Tier-1 host probe |
| [`tests/camodel/test_camodel_single_token_decode_aiv.cpp`](../tests/camodel/test_camodel_single_token_decode_aiv.cpp) | the CAModel runner and canary protocol |
| [`scripts/run_single_token_decode_camodel.sh`](../scripts/run_single_token_decode_camodel.sh) | build, run, archive, audit |
| [`scripts/analyze_single_token_decode_trace.py`](../scripts/analyze_single_token_decode_trace.py) | the trace audit |

`routines/routine_vector_decode_recurrence.hpp` gained one new entry point,
`vector_decode_recur_rows`, and nothing else: the diff has **zero deleted lines**, and
`vector_decode_recur_strip` with its compile-time 16-row bound is exactly as the frozen
Cube decode operator calls it.

That the frozen operator is therefore unaffected is **measured, not argued**. Its device
objects were built at fixed paths with the addition present, the shared header was then
replaced by its `HEAD` revision, and the kernel was rebuilt at the same paths:

```
a1868bf3…c1b67a  camodel_paged_decode_attention_kernel_merge_obj_dir/device.o   (AIV)
f3984a7b…3c45c0  camodel_paged_decode_attention_kernel_aic_device_dir/device.o  (AIC)
```

Both hashes are identical before and after. A function template the translation unit
never instantiates emits nothing, and the objects say so.
`turboquant_paged_decode_attention.cpp`, `turboquant_reshape_and_cache_bf16.cpp` and
`routine_vector_egress_rotate.hpp` are byte-identical to `HEAD`. The egress here uses the
register-resident butterfly from `routine_vector_rotate_butterfly.hpp` rather than that
frozen leaf, which is what roadmap §3.1 specifies.

### Interface, against RFC §4

The operand list follows RFC §4's `turboquant_single_token_decode_impl` where this
harness's verification kernels follow it at all, and departs where the frozen №4
departs. Three points worth stating, because each is a call-site concern:

* **`query_rot` is bf16**, as RFC §4 types it, and is converted once per task to the
  FP32 panel the GEMV reads — one `Cast` over 6 × 256 elements, a listed pair, exact.
  An earlier revision took it as FP32, which would have broken the RFC's explicit
  requirement that №3 and №4 be interchangeable at the call site.
* **`gate` is FP32**, which matches the frozen №4 rather than RFC §4's `uint16_t`. The
  two decode kernels must agree with each other first; whichever way that operand
  settles, it settles for both.
* **The strip window is explicit** (`seq_pages`, `strips`, `strip_begin`) rather than
  derived from `seq_lens`. That is the verification parameterisation the frozen №4's
  runner also uses, and it is why the ragged tail is listed as not done in §6 — a
  partially filled last page needs the `-inf` masking roadmap D3 specifies, and whole
  strip extents cannot express it.

The probe outputs (`out_scores`, `out_state`, `out_acc`, `out_rotated`, `out_context`)
are the verification surface, exactly as in the frozen №4, which publishes the same five
plus `out_qrot`. The production `out` + `workspace` + `output_stage` form belongs to the
Phase 5 adapter.

---

## 1. The roadmap's two-instruction score row does not exist on this part

Roadmap §3.1 sketches the per-row GEMV as two instructions:

```
Mul(prod, key_page_fp32, q_bcast, 64 lanes, 64 repeats, {1,1,1,8,8,0})
ReduceRepeat over 256 contiguous lanes x 16 repeats -> S[q][0..15]
```

Neither line is expressible, and both fail on the same bound. From
`SIMD_compute/mask.md` ("Mask Overview"), a repeat is one 256-byte vector register, so
for a 32-bit operand the continuous-mode mask is `mask ∈ [0, 64]` — and, critically:

> in continuous mode, if mask exceeds the range, it is calculated as the maximum mask
> value of the corresponding data type

So `ReduceRepeat` over 256 lanes is **not** a compile error and **not** a runtime
fault. It silently reduces the first 64 channels of each token and returns a plausible,
wrong panel. That is the same silent-truncation class as the unlisted
`Cast(bfloat16_t ← int16_t)` that cost the Cube decode operator a full debugging cycle,
and it would have been considerably harder to find here, because a partial dot product
over 64 of 256 channels is numerically unremarkable.

The `Mul` fails differently. With 64 repeats over a token-major key page, repeat `r`
covers token `r/4`, channels `(r%4)·64 … +63`, so the broadcast operand must supply
query channel quarter `r mod 4` — a period-4 wrap. `src1RepStride = 0` does not wrap,
it stands still, so every repeat would multiply by query channels 0…63. The
`src1RepStride = 0` trick that serves the **value** contraction (`p_x4`, in the
recurrence leaf) works there because the value page is channel-major with the 16-key
axis contiguous, which makes the broadcast operand's period exactly 64 lanes. The key
page is token-major, so its period is 256 and the trick does not transfer. It is the
same layout asymmetry that makes the key plane cheap for the Cube and the value plane
cheap for the Vector reduction — just read in the other direction.

### What replaces it: seven instructions, and why that shape

```
4 x Mul           quarter j, 16 repeats, src1 standing still on q[64j … 64j+63];
                  products land QUARTER-MAJOR at prod[j*1024 + t*64 + c']
1 x ReduceRepeat  mask 64, 64 repeats -> partial[j*16 + t], 64 contiguous lanes
2 x Add           fold the quarters: mask 32 at +128 bytes, then mask 16 at +64
```

The quarter-major product layout is the whole trick, and it is chosen for the two
instructions that follow it rather than for itself. It makes the 64 first-level
reductions **one** instruction, because the partials come out contiguous and in repeat
order; and it makes the second level two 32-byte-aligned `Add`s instead of a strided
reduction the API cannot express. Summing four lanes at stride 4 — which is what a
token-major product layout would need — would require a source repeat stride of half a
DataBlock, and there is no such thing.

Seven instructions per row per strip, 42 per strip at six rows, against the sketch's 12.
The cost is real and it is reported here rather than smoothed over; what the sketch
promised was not available.

### The bit-exact panel gate rests on a fixture property, not on the arithmetic

Three different summation orders are in play — the host oracle's left-to-right loop, the
device's binary-tree accumulation inside a repeat (documented for `ReduceRepeat<SUM>`),
and the quarter fold's `(j₀+j₂) + (j₁+j₃)` across repeats. They agree to the bit only
because every partial sum on this fixture is exactly representable: `Q_rot` entries are
integer multiples of 1/16 bounded by 1, the nibbles are integers in [−7, 7], so every
partial sum is a multiple of 1/16 with magnitude under 256·7 = 1792, needing at most 15
mantissa bits.

Measured over all four heads and 32 strips, the largest sum of absolute terms is
**235.1**, against the bound of 2²⁴/16 = 1,048,576 — four orders of magnitude of margin.
`ValidateFixture` asserts the bound rather than the margin, so a future fixture that
narrowed it would fail the probe instead of quietly making the gate wrong. On production
data the orders do **not** agree, and the panel is then accurate rather than
reproducible; the bit-exact gate is a property of the fixture and is labelled as such in
both the leaf and the oracle.

---

## 2. What the Tier-1 probe caught before any simulator run

`bin/test_single_token_decode_probe`, under a second, no device and no toolchain.

**The fixture was wrong, in the way that matters most.** The first version of the oracle
took `paged_decode`'s nibble generators and handed them a *global* strip index, on the
stated assumption that two pages would then carry different payloads. They do not.
`paged_decode::KeyNibble`'s strip term is `strip * 11` taken mod 8, and its head term is
`head * 13` mod 8, so the whole pattern depends only on `(3·strip + 5·head) mod 8`:
**eight distinct key payloads exist in that generator, in total.** Global strip 8 is
therefore byte-identical to global strip 0, and a kernel that ignored the block table
and read page 0 four times would have produced exactly the right answer on a four-page
run. That is the "wrong operand yields the right answer" failure mode which let the
online-softmax WAR hazard pass its original validation.

The probe reported it as `two strips share a row maximum`. The fix mixes the page index
into the pattern in a way that vanishes on page 0 — `page * (channel + 1)` — so the
one-page fixture stays byte-identical to the Cube operator's, which
`ValidateSharedFixture` asserts rather than assumes.

**The check that caught it was itself wrong, and is now a different check.** Requiring
per-row score *maxima* to be pairwise distinct is not a structural property: a raw score
is an integer multiple of 1/16 inside a range of some 4,700 values, so over the ~2,900
strip pairs this fixture has, two strips sharing a scalar maximum is an expected
coincidence — and it says nothing about whether the strips can be told apart. Two
strips whose 96-entry *panels* differ are distinguishable whatever their maxima do. The
gate is now panel inequality, and it reports the closest pair:

| head | closest panels differ in (4 pages) | (1 page) | rescale strips | carried-max strips | min *l* |
| --- | --- | --- | --- | --- | --- |
| 0 | 16/96 | 40/96 | 11/32 | 24/32 | 107.2 |
| 1 | 8/96 | 40/96 | 11/32 | 24/32 | 116.9 |
| 2 | 8/96 | 40/96 | 10/32 | 24/32 | 118.1 |
| 3 | 12/96 | 36/96 | 10/32 | 24/32 | 131.9 |

Both recurrence branches run, which is the other thing the multi-page fixture buys.
Within a page the key scale's per-strip doubling makes every strip maximum rise, so
`alpha < 1` and the accumulator is rescaled; at a page boundary the ramp restarts and
the **carried** maximum wins, so `alpha == 1` and the tile contributes a small
`exp(S − m_prev)`. A single-page run never reaches the second branch. The ramp is taken
modulo the page deliberately: carried over 32 strips it would reach 2¹⁵, every `exp` but
the largest would underflow, each query's probability mass would collapse to 1.0 and the
strips would stop being distinguishable — the degenerate softmax the Cube operator's
probe rejected a fixture for at a base exponent of −8.

**The deviation in §1 is measured, not argued.** The probe computes the clamped-mask
panel the roadmap's sketch would have produced and requires it to disagree: it coincides
with the correct panel on **13 of 3072** entries (0.42%), the handful being tokens whose
upper 192 channels happen to cancel. A check that the sketch is wrong is only worth
making if the sketch gives a different answer.

The probe also settles, with no device: that the four-`Mul`/one-`Reduce`/two-`Add` index
arithmetic replayed from the leaf's own repeat strides is bit-exact against the direct
dot product; that all 4096 product slots are written exactly once (a map that looks
plausible can still alias, and aliasing inside one instruction is invisible in the
numbers); that the eight-stage egress butterfly agrees with the dense 256×256
right-multiplication by Π to 4.66 × 10⁻⁹ against a budget of 2.83 × 10⁻⁷; and that a
single-strip golden state is reproducible and the eight single-strip states of a head
are pairwise distinguishable.

---

## 3. The census budget, registered before the runs

Roadmap §0.2 asks for an instruction census and a per-engine breakdown against a budget
written down in advance, explicitly **not** a utilisation target: the measured Vector
profile of this operator family is register-traffic shaped and does not reach SIMD ALU
saturation. These are derived from the leaf shapes alone, per strip at six rows, in
register-level (`RV_*`) instructions:

| term | derivation | per strip |
| --- | --- | --- |
| GEMV multiplies | 4 `Mul` × 16 repeats × 6 rows | 384 `RV_VMUL` |
| GEMV reductions | 1 `ReduceRepeat` × 64 repeats × 6 rows | 384 `RV_VCADD` |
| recurrence value contraction | `Mul(weighted)` 64 repeats × 6 rows | 384 `RV_VMUL` |
| recurrence per-channel reduction | 2 × 128 repeats × 6 rows | 1536 `RV_VCADD` |
| recurrence probability sums | 1 repeat × 6 rows | 6 `RV_VCADD` |
| int4 unpack | 2 `Cast` × 4096 elements × 2 planes | — |

and once per task, in the epilogue:

| term | derivation | per task |
| --- | --- | --- |
| egress butterfly | 6 lane stages × 4 registers × 6 rows | 144 `RV_VSELR`, 144 `RV_VMULA` |
| the two 0-ULP divides | 3 FMA × 4 registers × 2 calls × 6 rows | 144 `RV_VMULA` |
| GM loads | 1 Π-sign + 2 panels + 4 per strip | 35 at eight strips |
| GM stores | 1 probe + 1 archive per strip + 5 epilogue | 14 at eight strips |

The reduction terms are the prediction worth stating, because they are exact and they
dominate: 384 + 1536 + 6 = **1926** `RV_VCADD` per strip, so 15,408 over eight strips.
All three eight-strip runs measured **15,408**, to the instruction (§5.2).
The GEMV is 20% of the reduction traffic and the value contraction is 80%,
which is the honest shape of this kernel — the roadmap's warning that `RV_VLD`/`RV_VST`
would dominate is right in spirit and slightly wrong in detail: the dominant single
term is the per-channel reduction the *value* contraction needs, and it is unchanged
from the Cube operator, because that contraction was already on the Vector core there.

### The audit's own prediction was wrong, and the trace said why

The first version of check 5 asserted `RV_VMULA == RV_VSELR`, on the reasoning that each
butterfly lane stage is one `Gather` plus one `MulAddDst`. `RV_VSELR` came back at 144 as
predicted; `RV_VMULA` came back at **288** — on the first run, and on every run since.

The extra 144 are not in the butterfly. Grouping `RV_VMULA` by program counter separates
two populations cleanly: 24 PCs retiring 6 times each — the fully unrolled butterfly
body, once per row — and 6 PCs retiring 24 times each, in two clusters of three,
surrounded by an `RV_VMULS` against `0xbf800000` and an `RV_VABS_FP`. That is the
0-ULP FP32 `Div`: **three fused multiply-adds per 64-lane register.** The kernel divides
twice in its epilogue — `acc / l`, one call per row over 4 registers, and the sigmoid,
one call over 4 × 6 registers — which is 48 register-iterations and so 144 multiply-adds.
Exactly the surplus.

The audit now asserts both terms separately, and the row sweep makes the derivation
falsifiable rather than fitted: both terms are linear in the row count, so `--rows 1` and
`--rows 8` must land on 48 and 384 and nothing else.

One thing the audit deliberately does **not** gate: the butterfly's zero-UB-spill
invariant as an exact register load and store count. In this operator the level-2
continuous APIs lower to VF code too and dominate register traffic, so a dump total
cannot be attributed to the leaf. That invariant was measured where the butterfly *was*
the whole kernel — four loads and four stores per row and not one more,
[`reports/rotate_butterfly.md`](rotate_butterfly.md) — and the totals here are reported
for the census rather than asserted.

---

## 4. The canary protocol

Every Global Memory payload — **inputs as well as outputs** — is flooded with
`0x7FC0BEEF`, and the fixture is then written into only the slices the isolated
`(request, kv_head)` item is entitled to read: the key, value and scale slices of
`(physical page, head, strip)` for the strips in the window, with the physical page
coming from the block table, which is deliberately not the identity; and that head's six
query rows and six gate rows, and no others. `0x7FC0BEEF` is a quiet NaN, so a read
outside those slices propagates into everything downstream rather than perturbing it —
a wrong page, a wrong head, a wrong strip or a wrong query row is a NaN panel, not a
close one.

Uploading only the active head's **query** rows is stronger than the Cube operator's
protocol, which uploads Q whole. There, head separation comes only from the cache slice
and the gate, so a task that read another head's query rows would be invisible. Here the
query panel is per-head by construction, because GQA's own indexing (`q head = kv_head ×
6 + r`) makes a task's rows six consecutive rows of the `[requests][H_q][d]` panel.

Writes are checked the other way round. Every output buffer is canary before the launch,
so one sweep carries three claims at once: the active item's slots must all be written,
every other item's slots must still be canary, and inside the active item the archive
slots past the processed strips and the rows past `rows` must still be canary — the
output guard band. `camodel::GuardedDeviceBuffer` adds a second, different pattern
(`0xDEADBEEF`) in 64 bytes either side of every payload, which catches a burst that runs
off the end of a buffer rather than landing in the wrong place inside one. Neither
pattern is `0x00`, which a half-issued burst also leaves behind.

### What is checked bit-exactly, and what is not

| quantity | gate | why |
| --- | --- | --- |
| raw score panels | **bit-exact**, every panel of every strip | §1: all three summation orders agree on this fixture |
| `m` | **bit-exact** | a maximum of exactly representable score columns is exact |
| `l` | ULP | passes through `Exp`, documented at ≤ 1 ULP per call |
| `acc` | ULP against the largest intermediate | a cancelling reduction of exp-weighted terms |
| egress rotation, pre-gate | relative, against the largest value in the row | eight butterfly stages over a cancelling sum |
| gated bf16 context | relative | one more `Exp` and one more `Div` |

`m` bit-exact is stronger than the roadmap's gate 3.1.1, which asks for `(m, l)` within
2 ULP; it is free, because `m` never passes through a transcendental.

The task statement asks for a **bit-exact normalised context**, and the distinction
between guaranteeing that and observing it matters here.

It cannot be *guaranteed*: the context's value passes through two `Exp` calls and two
`Div` calls, `Exp` is documented at up to 1 ULP per call, and no fixture removes that.
Asserting bit-exactness would be asserting something the arithmetic forbids, so the
gate the runner enforces is the decode report's relative budget.

It was nevertheless *met*, at all three coordinates: the measured context error is
**0.000e+00** — the device's bf16 words equal the oracle's bf16 words exactly. The
reason is bf16's 8 mantissa bits, which absorb an FP32 error of a few ULP whenever the
rounding does not fall on a tie, and `l` was measured at 1–2 ULP. That is an
observation about this fixture and this rounding, not a property to rely on: a fixture
whose values landed nearer bf16 tie points would differ in the last bit while being
equally correct, which is exactly why the enforced gate is the relative one.

The pre-gate rotation and the gated output are published and checked separately so that
a failure in either is actionable on its own. The gate is elementwise and the rotation
is eight strided stages; their errors are indistinguishable in the gated output alone.

---

## 5. Results

### 5.1 The spatial sweep — boundary, middle, boundary

`bash scripts/run_single_token_decode_camodel.sh`, six rows, one page, all eight strips,
one isolated `(request, kv_head)` item per launch on a one-block grid.

| | H = 0 | H = 2 | H = 3 |
| --- | --- | --- | --- |
| raw score panels wrong | **0 / 768** | **0 / 768** | **0 / 768** |
| `m` | bit-exact | bit-exact | bit-exact |
| `l` | 2 ULP | 2 ULP | 1 ULP |
| `acc` (largest \|acc\|) | 48 ULP (1.2075) | 72 ULP (1.1255) | 48 ULP (0.5613) |
| egress rotation err / budget | 9.31e−09 / 4.50e−05 | 1.49e−08 / 7.24e−05 | 5.59e−09 / 3.52e−05 |
| gated bf16 context err / budget | **0.00e+00** / 4.79e−04 | **0.00e+00** / 8.16e−04 | **0.00e+00** / 4.05e−04 |
| input words changed | 0 | 0 | 0 |
| canary words intact | 52384 / 52384 | 52384 / 52384 | 52384 / 52384 |
| trace audit | PASS (17 checks) | PASS (17 checks) | PASS (17 checks) |
| total tick | 181,302 | 181,279 | 181,294 |
| wall | 547 s | 591 s | 555 s |

Bit-exact on the first execution of the kernel, at every coordinate, with the gated bf16
context **exactly** equal to the oracle's — the relative budget was not needed. The
egress rotation sits three to four orders of magnitude inside its budget.

The launch observation is the roadmap's open item, re-published rather than inherited:
under `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` with a one-block grid the kernel
observes `GetBlockIdx() == 0`, `GetBlockNum() == 1` and `GetSubBlockNum() == 1`, so every
AIV is its own block and the MIX idiom `GetBlockIdx() / GetSubBlockNum()` is not needed.

### 5.2 The census, against the budget registered in §3

The three runs produce a **byte-identical instruction census and an identical dispatch
span**, which is the first thing to report because it is a claim about the design rather
than about any one head: the kernel is a straight line whose trip count depends on the
strip window and on nothing in the data, so three different heads differ only in
addresses and in the values moved.

| term | predicted (§3) | measured | basis |
| --- | --- | --- | --- |
| `RV_VCADD` | 1926 / strip → **15,408** | **15,408** | exact: 384 GEMV + 1536 value contraction + 6 probability sums |
| `RV_VCVT_I2F` | 128 / strip → **1,024** | **1,024** | exact: 4096 elements ÷ 64 lanes × 2 planes |
| `RV_VSELR` | **144** / task | **144** | exact: 6 lane stages × 4 registers × 6 rows |
| `RV_VMULA` | **288** / task | **288** | exact: 144 butterfly + 3 FMA × 48 divide registers |
| GM loads | **35** | **35** | exact: 1 Π-sign + 2 panels + 4 × 8 strip operands |
| GM stores | **14** | **14** | exact: 1 probe + 8 archived panels + 5 epilogue bursts |
| `RV_VMUL` | ~810 / strip | 837 / strip | reported, not predicted exactly |
| `RV_VLDI` / `RV_VSTUI` | — | 22,186 / 15,456 | reported |

Six of the eight terms are exact equalities derived before the run. `RV_VSELR` and
`RV_VMULA` being *unchanged* between the one-strip and eight-strip rungs is itself the
check that the egress runs once per task rather than once per strip.

`RV_VMUL` is 27 per strip above the derivation, and the report says so rather than
rounding it away: the derivable terms account for 810 and the surplus is in the lowering
of `Exp` and `Brcb`, which this census does not separate.

**Per-engine breakdown, and the one conclusion it supports.** Over the eight-strip run
the load and store engines issue `RVECLD` 37,987 + `RVECST` 33,057 = 71,044 instructions
against `RVECEX`'s 34,345 — **register traffic is 2.07× the arithmetic**. That is the
roadmap's §0.2 prediction confirmed as a measurement, and it is why §0.2 sets the
acceptance criterion as a census rather than a SIMD ALU utilisation target: this kernel
cannot reach one, and an int4-unpack-and-reduce body is load/store shaped by nature.

**Timing, stated as what it is.** The whole-dump dispatch span is 180,187 ticks and the
per-strip figure the audit prints (22,523 ticks) includes the prologue and the epilogue,
so it is not a marginal cost. The marginal cost follows from the two rungs: 181,302 ticks
at eight strips against 44,564 at one gives **19,534 ticks per additional strip** and
**~25,030 ticks fixed** — the task prologue, the egress and the launch. The fixed cost is
28% of an eight-strip task and would be 56% of a one-page-per-request decode at four
strips, which is the first number to attack if this operator ever needs to be faster;
the egress is a per-task cost that the Cube operator amortises over 16 rows and this one
over 6.

No comparison with №4's quoted ~19,900-tick per-strip interval is drawn here. That figure
comes from a different runner over a twelve-column modulo schedule with the Cube doing
the score GEMM and both Vector cores live; the measurement bases are not the same, and
putting the two numbers side by side would imply a comparison neither run supports.

### 5.3 The intra-core route census

The seven routes the eight-strip runs used, from the instructions' own
`(PIPE, TRIGGER PIPE, FLAG ID)` fields — identical at all three heads, and every count
derivable from the trip count:

| route | id | sets | = |
| --- | --- | --- | --- |
| `MTE2 → VEC` | 0 | 10 | 1 Π-sign + 1 task prologue + 8 strips |
| `VEC → MTE2` | 0 | 9 | 1 priming + 8 strips (packed pages unpacked) |
| `VEC → MTE2` | 1 | 9 | 1 priming + 8 strips (scale slices **read**) |
| `VEC → MTE3` | 0 | 8 | one raw panel archived per strip |
| `VEC → MTE3` | 1 | 2 | the launch probe, and the epilogue outputs |
| `MTE3 → VEC` | 0 | 9 | 1 priming + 8 strips (archive drained) |
| `MTE3 → SCALAR` | 0 | 2 | the probe drain and the launch drain |

All balanced, none on a reserved id, every set before its wait in program order, and
**zero** cross-core instructions, zero Cube instructions, zero MTE1 and no data mover
touching L1 — roadmap gate 3.1.4, as the positive assertion it asks for.

`VEC → MTE2` being two routes rather than one is the hazard this kernel was most likely
to get wrong, and it is wrong in a specific, documented way. One MTE2 burst group lands
three kinds of operand with two different last readers: the packed pages are last read by
the unpack, the scale slices by the **recurrence**, which runs after the GEMV. The frozen
Cube operator paid for collapsing them — releasing the slot at the dequant cost its
4-strip rung, with `m` and `l` wrong by 2,097,152 ULP and `acc` by 4,010,429 **while
every raw score panel stayed bit-exact**, because the two are on different queues and
program order inside a column orders nothing. Splitting the route also buys the only
overlap this kernel has: the next strip's packed burst issues while this strip's GEMV and
recurrence are still running.

### 5.4 Archived runs, and what was not run

Each run archives its own dumps and audit before the next overwrites them, under
`dumps_stdecode_<tag>/` and `audit_stdecode_<tag>.txt` in the build directory, with
`STDECODE_SUMMARY.txt` holding the verdict and wall time per rung:

```
canary_H0                PASS      547s
canary_H2                PASS      591s
canary_H3                PASS      555s
```

Which core the audit reads is not assumed. The isolated runs launch a one-block grid, so
the work is on `core0.veccore0`, and check 0 asserts that the core it reads issued the
vector, MTE2 and MTE3 work that core is defined by. An idle core in a CAModel dump is
not an empty one — it issues a scalar prologue and an early return, and every balance
check is vacuously true on it, including one named "both Vector cores executed", whose
test a single prologue MTE2 satisfies. That trap cost the Cube decode operator six
audited canary rungs before it was found.

The `ladder` mode defines three further rungs that were **not** run here, and they are
the ones to run next:

* `pages4_H1` — four pages through the non-identity block table. This is the only rung
  where reading the wrong page is observable at all, and the only one that exercises the
  branch where the carried maximum wins. The oracle and the fixture are validated for it
  (§2, and the probe checks `--pages 4` on every head), and the runner takes `--pages 4`
  today; at ~19,534 ticks per strip it is 32 strips and roughly 35 minutes of wall time.
* `grid_all` — four items in one launch, one per physical AIV, which is the whole grid at
  `B = 1`. Check 4 is skipped there by design, because the companion Vector core is then
  expected to work.
* `rows1_H0` / `rows8_H0` — the ends of the runtime row bound. These are what make §3's
  `RV_VMULA` derivation falsifiable rather than fitted: both of its terms are linear in
  the row count, so they must land on 48 and 384 and nothing else.

Nothing in §5.1 depends on those three; they extend the claim rather than support it.

### 5.5 The pass gates, and where each one is answered

Roadmap gate 3.1 has five items and the task statement's protocol has four; they overlap
and are mapped together here.

| gate | verdict | evidence |
| --- | --- | --- |
| Bit-exact score panels, 6 rows, 1 page | **pass** | 0/768 at H = 0, 2, 3 (§5.1) |
| Bit-exact score panels, 4 pages | **pending** | the `pages4_H1` rung, §5.4; oracle and fixture validated for it (§2) |
| `(m, l)` within 2 ULP | **pass, exceeded** | `m` bit-exact; `l` at 1–2 ULP (§5.1) |
| `acc` within its per-fixture budget | **pass** | 48–72 ULP against a 256 ULP budget (§5.1) |
| Egress rotation and gated context within budget | **pass** | rotation 3–4 orders inside budget; context error 0.000e+00 (§5.1, §4) |
| GM buffers flooded with `0x7FC0BEEF` | **pass** | inputs and outputs both, and only the entitled slices written (§4) |
| One tile reads and writes only its own GM | **pass** | 0 input words changed; 52384/52384 canary words intact (§5.1) |
| Unassigned slots and output guard bands unmutated | **pass** | same sweep; guard band covers rows past `rows` and archive slots past the window |
| Trace audit: 0 cross-core flags, 0 `wait_intra_block` | **pass** | check 1, as a positive assertion; also 0 Cube, 0 MTE1, 0 L1 (§5.3) |
| Balanced intra-core `HardEvent` tokens | **pass** | all 7 routes balanced, derivable from the trip count, set-before-wait (§5.3) |
| Instruction census within a pre-registered budget | **pass** | 6 of 8 terms exact equalities; the two reported ones named (§3, §5.2) |

---

## 6. What these runs do not establish

* **No silicon.** CAModel is cycle-accurate for the pipeline it models; it is not a
  950PR. No timing claim here survives contact with hardware.
* **No wide grid.** The canary runs launch one block. The `--all` rung launches four,
  one per `(request, kv_head)` item at `B = 1`, which is the whole grid at that batch
  size — but a production decode step at `B = 64` is 256 items over a core count this
  host cannot exercise, and nothing here says what happens when the grid exceeds the
  physical AIV count.
* **No MTP.** `rows` is a runtime bound and the row sweep covers 1 and 8, but the
  `6 × (1 + k)` family at `k ≥ 1` is 12 rows or more, which the roadmap's own dispatch
  rule routes to the Cube operator. The kernel returns early above 8 rows rather than
  pretending to serve them.
* **No ragged tail.** Every strip in these runs is a full 16 tokens. A sequence whose
  last page is partially filled needs the `-inf` masking roadmap D3 specifies for №4,
  and this kernel does not implement it: `seq_pages` and `strips` are whole-strip
  extents. That is the first thing to add before the operator is dispatched on real
  sequence lengths.
* **No `temporal_splits`.** The roadmap keeps `workspace` and `temporal_splits` in №3's
  operand list for the small-`B` occupancy case; this kernel is the
  `temporal_splits == 1` path, where the online-softmax recurrence carries `(m, l, Acc)`
  across every page and no cross-page combine is needed at all.
* **Not a comparison with №4.** The one-page fixture is byte-identical to the Cube
  operator's, which makes a first-six-rows comparison possible, but it was not run: №4
  is frozen and its runner drives 16 rows.
