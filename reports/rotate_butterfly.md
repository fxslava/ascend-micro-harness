# The randomised-Hadamard butterfly leaf, and which of three implementations wins

Ascend 950PR / arch35 / `dav-3510` / CANN 9.2.0-beta.2, `Ascend950PR_9599`
CAModel. `KERNEL_TYPE_AIV_ONLY`, one AIV, `gridDim = 1`.

**No NPU on this host.** Every number below is a cycle-accurate model tick for
one Vector core with no contention from any other engine. None of it is a
latency, throughput or power figure for the part.

| Field | Value |
| --- | --- |
| What it proves | the eight-stage `Pi` cascade is correct in both directions, and which of three permutation strategies the AIV-only ingress operator should use |
| What it deliberately omits | the quantiser, the nibble packing, the cache read-modify-write, any grid wider than one core, and concurrency with MTE2/MTE3 |
| Leaf | [`routines/routine_vector_rotate_butterfly.hpp`](../routines/routine_vector_rotate_butterfly.hpp) |
| Benchmark kernel | [`kernels/camodel/rotate_butterfly.cpp`](../kernels/camodel/rotate_butterfly.cpp) |
| Runner | [`tests/camodel/test_camodel_rotate_butterfly.cpp`](../tests/camodel/test_camodel_rotate_butterfly.cpp) |
| Tier-1 probe | [`tests/cpu/test_rotate_butterfly_probe.cpp`](../tests/cpu/test_rotate_butterfly_probe.cpp) |

```sh
bash scripts/run_rotate_butterfly_camodel.sh                   # all three, 16 rows
bash scripts/run_rotate_butterfly_camodel.sh --rows 1          # the decode shape
bash scripts/run_rotate_butterfly_camodel.sh --variant gather   # one of them
```

Run **both** shapes. `--rows 16` is the ragged-prefill ceiling and `--rows 1` is
the decode shape; the second is where the one-off table construction is
proportionally largest, and it is also the rung that caught the benchmark
kernel publishing uninitialised Unified Buffer past its row window (§4.5).

---

## 1. What is being compared

`Pi = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32) / 16 = A . D . B` with `A` the
channel-group factor (strides 32, 64, 128), `D` the sign diagonal and `B` the
in-group factor (strides 1, 2, 4, 8, 16). A radix-2 stage at stride `s` is

```
out[i] = in[i ^ s] + sgn_s(i) . in[i],      sgn_s(i) = (i & s) ? -1 : +1
```

The two directions are **not** the same stage order, and that is the most
error-prone line in this family:

| Direction | Used by | Expansion | Stage order | Sign after |
| --- | --- | --- | --- | --- |
| `x . Pi` (right) | decode egress | `((x.A).D).B` | 32, 64, 128, 1, 2, 4, 8, 16 | stage 3 |
| `Pi . x` (left) | scatter ingress | `A.(D.(B.x))` | 1, 2, 4, 8, 16, 32, 64, 128 | stage 5 |

Three implementations, bit-identical by construction — same stage order, same
associativity, FP32 throughout:

* **Blocked** (Approach A). Level-2 continuous APIs over **Unified Buffer**
  operands. Stages with `s >= 8` are two repeat-form instructions with
  `BinaryRepeatParams` strides; stages with `s` in `{1, 2, 4}` have the partner
  lane inside the 8-lane FP32 DataBlock, where a 4/8/16-byte tensor offset is
  not 32-byte aligned and the masked form cannot reach it either, so they pay a
  level-2 `Gather` against a 256-entry UB offset table plus a sign multiply.
  Three offset and three sign vectors, 6 KiB. **Every stage round-trips through
  Unified Buffer.** This is the shape
  [`routine_vector_egress_rotate.hpp`](../routines/routine_vector_egress_rotate.hpp)
  ships today, generalised to both directions.

  The coarse stages are not written differently here, and that is the faithful
  reading rather than a shortcut: at level 2 there are no *named* registers to
  pair, and the repeat form already pairs whole ones. `s = 128` compiles to
  `mask = 64, repeats = 2, repStride = 8 DataBlocks` — one 64-lane vector
  register per repeat, `V0 ± V2` then `V1 ± V3` — which is exactly "full
  64-element register pairing" expressed through the only API this approach has.
  What distinguishes Approach A is where the operands live, not which stages get
  which instruction.
* **RegGather** (Approach B, register-indexed shuffle). One `__simd_vf__` body;
  the 256 FP32 channels are four 64-lane vector registers and stay there for all
  eight stages. Strides 1..32 flip a **lane** bit, so each is one `Reg::Gather`
  plus one `Reg::MulAddDst`. Strides 64 and 128 flip a **register index** bit,
  so they are straight-line inter-register `Add`/`Sub` with no permutation at
  all. Index and sign registers are built in-register from `Arange`, `Xor` and
  a shift-or, so **no UB offset table exists**.
* **RegShuffle** (Approach B, isomorphic shuffle). Same residency; the lane-bit
  stages are `DeInterleave` + `Add` + `Sub` + `Select` and carry **no index
  register whatsoever**.

### 1.1 Why RegShuffle works at all

Inside one 64-lane register, let `P` be the permutation `(P x)[k] = x[rotl6(k)]`
— a cyclic left-rotation of the six lane-index bits — and `B_t` the butterfly on
lane bit `t`. Then

```
DeInterleave(E, O, V, V);  Add(S, E, O);  Sub(D, E, O);  Select(V, S, D, maskH)
    ==  P . B_0
```

because `DeInterleave` puts the even lanes in `E` and the odd lanes in `O`, each
duplicated across both halves of its destination, so `S` and `D` hold the two
butterfly wings duplicated, and `maskH` (`MaskPattern::H`) splices the sums into
lanes 0..31 and the differences into lanes 32..63 — the decimated order. Six such
steps give

```
(P B_0)^6 = P^6 . B_5 B_4 B_3 B_2 B_1 B_0 = B_5 ... B_0
```

since `P^6 = I` and conjugating `B_0` by `P^t` yields `B_t`. So six uniform
four-instruction steps compute the whole 64-point Hadamard with one static mask
and no table.

The cost is that the sign diagonal does not land at a step boundary: after `t`
steps the data is in `P^t` order and the ingress needs `D` at `t = 5`. The leaf
carries the **sign registers** into the `P^5` frame once per call (five
applications of the permutation-only form, two instructions each) rather than
rotating the data on every row.

**RegShuffle implements the ingress only.** The egress order puts bit 5 before
the sign and bits 0..4 after it, which an increasing-order recursion cannot
express without pre-rotating the *data* by `P^5` on every row — ten instructions
per register per row, not competitive. `Egress` with `RegShuffle` is a compile
error rather than a silently wrong answer.

---

## 2. Tier-1, and why it is not optional here

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    tests/cpu/test_rotate_butterfly_probe.cpp -o bin/test_rotate_butterfly_probe
bin/test_rotate_butterfly_probe
```

```
rotate butterfly Tier-1 probe: d = 256, 8 stages, 4 registers x 64 lanes
  sign diagonal: 73 of 256 negative
  Pi . Pi^T diagonal 256 over 256 x 256 entries
  peak |Pi . x| over the fixture: 433.4 (FP32 exact below 2^24)
rotate butterfly probe: PASS
```

Eight dyadic fixtures, and for each of them:

1. `Pi . Pi^T = 256 . I`, entry for entry, in integers — all 65536 entries.
2. The ingress stage table reproduces the dense `Pi . x` **bit-exactly**; the
   egress table reproduces `x . Pi` bit-exactly.
3. The egress table does **not** compute `Pi . x`, and neither does the ingress
   table with the egress sign position. Without this the gate above would prove
   the cascade correct and nothing about the ordering, which is the error this
   family is most exposed to.
4. All three device algorithms reproduce the cascade bit-exactly — including
   RegShuffle's `(P B_0)^6` recursion with its `P^5` sign frame, and Blocked's
   byte-offset gather tables applied the way `NativeRotateBlockedEngine` indexes
   them.
5. `(Pi . x) . Pi / 256 == x` to 0 ULP.

Check 4 is the one that earns its keep. The `P^6 = I` identity is the whole basis
of the RegShuffle leaf, and a fifteen-second host run settles it; a CAModel run
that disagreed with the oracle would only have said "the egress is wrong".

---

## 3. CAModel: all three are bit-exact

16 rows of 256 FP32 per launch, one repetition, grid 1. Every launch's output
panel is compared **bit-exactly** against `rot_oracle::staged`, and three further
properties are checked that the value comparison cannot see: the output panel is
poisoned with `0x7FC0BEEF` beforehand, so an unwritten word reads back as a quiet
NaN; rows beyond the launch's row count must still hold the poison; and the
source panel and both tables must come back unmodified.

```
rotate butterfly CAModel (blocked): PASS
rotate butterfly CAModel (gather):  PASS
rotate butterfly CAModel (shuffle): PASS
  rows checked 16, mismatching words 0, unwritten 0, beyond-range 0
```

Model wall time about 20 s per variant; 5,925 model ticks for the shuffle run.

The 16-row benchmark was run **three times** against three source states: as
first written; after the leaf's per-row stage sequences were split out of the VF
bodies into `__simd_callee__` helpers, so the scatter operator could share them
with its own quantiser epilogue; and after the §4.5 epilogue fix. **Every
vector-engine number in §4 is byte-identical across all three** — the compiler
inlines the helper exactly as it inlined the inline body, and the epilogue fix
touches an MTE3 length and not the vector pipe. So the census below is the final
source's, not an earlier measurement quoted next to it.

Two figures moved, both of them whole-kernel totals that include the SCALAR
queue and neither inside the measured region: `total_instructions` by 8 (15,359
→ 15,367 at 16 rows, 1,244 → 1,252 at 1 row) and `dispatch_span` by up to 12
ticks (gather 4,682 → 4,680, shuffle 4,828 → 4,840, blocked unchanged at
32,936). The epilogue's burst length is now computed from a kernel argument
rather than a constant, which is a handful of scalar instructions. The table
below is the final run's.

---

## 4. The engine census, which is the whole point

```
variant  total_instructions  vector_instructions  rvecex  rv_vld  rv_vst  vec_queue  vector_span  dispatch_span
blocked               15367                11250    2512    3072    1728          2        30215          32936
gather                 1415                 1203    1057      64      64          2         1850           4680
shuffle                2183                 1981    1834      64      64          2         1965           4840
```

`vector_span` is the completion-log tick span over the Reg-vector queues plus
the level-2 `VEC` queue — the figure the three kernels make comparable by sharing
their fixture, grid, GM operands, MTE2/MTE3 bursts and event ids, and differing
only between the `MTE2_V` wait and the `V_MTE3` set.

Per 256-channel row:

| variant | vector instructions / row | vector ticks / row | `RVECEX` share |
| --- | --- | --- | --- |
| Blocked | 703.1 | 1888.4 | 22.3% |
| RegShuffle | 123.8 | 122.8 | **92.6%** |
| RegGather | **75.2** | **115.6** | 87.9% |

**RegGather is the production leaf.** It is 16.3x shorter than Approach A on the
vector engine and 6% shorter than RegShuffle with 39% fewer vector instructions.

### 4.0 The two criteria disagree, and which one decides

By **`RVECEX` share** — active execution cycles as a fraction of vector-engine
cycles — RegShuffle wins: 92.6% against RegGather's 87.9%. By **total
vector-engine span** RegGather wins by 6%. They disagree because RegShuffle's
four-instruction stage is four `RVECEX` ops while RegGather's two-instruction
stage is two, so the *same* arithmetic over the *same* register residency raises
one variant's share by doing more work per row.

The span decides, and the roadmap's §0.2 says why: the acceptance criterion for
this operator family is "an instruction census and a per-engine cycle breakdown
against a stated budget, not a utilisation percentage that cannot be met". A
utilisation share is a ratio, and a ratio improves when the denominator shrinks
*or* when the numerator grows — only one of those is faster. Both figures are in
the table because the share is what rules out the third variant (22.3% is
register traffic dominating arithmetic, which is the thing to avoid), and the
span is what ranks the two that survive.

### 4.1 The zero-spill invariant, as a measured number

Both register-resident variants issue exactly **64 `RV_VLD` and 64 `RV_VST`** —
four loads and four stores per row for 16 rows, which is the row itself arriving
and leaving and not one register more. The eight butterfly stages touch Unified
Buffer zero times.

Approach A issues **3,072 loads and 1,728 stores** over the same work. That is
not a tuning gap, it is the method: nine of its writes per row are stage results
going back to UB, and every stage then reads two 256-byte operands out again.

### 4.2 Where Approach A's time actually goes

```
engine        instr      first       last       span
RVECLD         4400       2368      32525      30157
RVECST         3440       2397      32541      30144
SCALAR         2740        724      33657      32933
RVECEX         2512       2368      32530      30162
FLOWCTRL        781       1137      33660      32523
RVECSU          592       2365      32524      30159
PUSHQ           578       1123      32456      31333
RVECLP          304       2364      32475      30111
```

`RVECEX` — the arithmetic — is 22.3% of its vector instructions, while
`RVECLD + RVECST` is 69.7%. Register traffic, not arithmetic, is the dominant
term, which is what the roadmap's §0.2 qualification predicted for this operator
family and is why the acceptance criterion is a census rather than a utilisation
percentage.

Two further costs are visible and neither is arithmetic: **288 `VF`** entries
(one per level-2 API call, each opening and closing its own vector function) and
**1,696 `RV_SMEM_BAR`** plus **1,440 `RV_SEND`** of intra-VF bookkeeping. And
`RVECEX` itself is 2.4x RegGather's for identical arithmetic, because a
repeat-form instruction covers 64 lanes and a 256-lane stage therefore needs four
repeats driven by `RV_VLOOP`, where the register form has four *named* registers
and no loop at all.

### 4.3 RegGather: the instruction budget is met exactly

```
  RV_VSELR            384     Reg::Gather, the lane permutation
  RV_VMULA            384     Reg::MulAddDst, partner + sign . self
  RV_VADD              64     strides 64 and 128, the sum wings
  RV_VSUB              64     strides 64 and 128, the difference wings
  RV_VMUL              64     the sign diagonal
  RV_VMULS             64     the trailing scale
  RV_VLD               64     four register loads per row
  RV_VST               64     four register stores per row
  RV_VBR     7, RV_VCI 1, RV_VAND 6, RV_VXOR 6, RV_VSHLI 6, RV_VOR 6
                              the index and sign tables, built once per call
```

`384 = 6 lane stages x 4 registers x 16 rows`, twice over — **two instructions
per stage per register**, which is what the design claimed. `Reg::Gather` lowers
to a single `RV_VSELR`: a register-to-register lane permutation, the
`shfl_xor`-equivalent on this ISA, and the reason the UB offset tables are not
needed. The whole table construction is 32 instructions, 2.7% of the run.

### 4.4 RegShuffle: four instructions where RegGather needs two

```
  RV_VADD             448     384 butterfly wings + 64 register stages
  RV_VSUB             448     the same
  RV_VDINTLV          404     384 recursion steps + 20 sign-frame permutes
  RV_VSEL             404     the same
  RV_VLD/RV_VST        64     identical residency to RegGather
```

`404 = 6 x 4 x 16 + 5 x 4`: the 20 extra are the `P^5` sign permute, hoisted out
of the row loop exactly as designed, 2% of the run. The recursion needs no index
register, no sign vector and no `Arange`, and it pays four instructions per stage
per register instead of two.

That buys only a 6% longer vector span, not 2x, because the four are
`DeInterleave`, `Add`, `Sub`, `Select` — all single-issue Reg-vector ops with no
operand hazard between them — against RegGather's `Gather` and a fused
multiply-add. **If a future part drops `Reg::Gather` or makes it multi-cycle,
RegShuffle is a 6% fallback and not a rewrite**, which is why it stays in the
header.

### 4.5 The decode shape, and the fault running it exposed

`--rows 16` is the ragged-prefill ceiling. The decode shape is **one** row, where
the one-off table construction is proportionally largest, so that rung decides
whether the ranking is an artefact of amortisation:

| variant | vector instr | `RVECEX` | `RV_VLD` | `RV_VST` | vector span |
| --- | --- | --- | --- | --- | --- |
| Blocked | 705 | 157 | 192 | 108 | 1910 |
| RegShuffle | 181 | 154 | 4 | 4 | 268 |
| **RegGather** | **123** | 97 | 4 | 4 | **230** |

The ranking does not just hold, it **widens**: RegGather is 14% shorter than
RegShuffle at one row against 6% at sixteen, because its per-stage cost is two
instructions against four and the setup difference (32 instructions against 40)
is far smaller than that gap. Four register loads and four stores — the row
arriving and leaving — at one row as well as sixteen.

The two shapes also pin the cost model down, which is worth having because it is
what makes a third shape predictable rather than a new experiment:

```
RegGather   ~36 + 72 x rows      (123 at 1 row, 1203 at 16)
RegShuffle  ~44 + 120 x rows      (181 at 1 row, 1981 at 16)
```

**The 1-row rung also found a real fault — in the benchmark kernel, not the
leaf.** Its epilogue copied all `kMaxRows` Unified Buffer rows to Global Memory
regardless of `rows`, publishing 3,840 words of **uninitialised Unified Buffer**
past the window. All three variants failed the runner's "no row beyond the
launch's row count was written" check with `beyond-range 3840` while reporting
`mismatching words 0` — row 0 was bit-exact in every variant, so the arithmetic
was never in question. At `--rows 16` the two lengths coincide and the fault is
invisible, which is exactly why the smaller shape had to be run; it is also the
same fault the decode operator's score archive had to fix, for the same reason
([`paged_decode_attention.md`](paged_decode_attention.md)). The epilogue now
copies `rows * 256` elements and both shapes pass.

---

## 5. What this does and does not establish

**Established**, on this host, by the runs above:

* The two stage tables are correct, bit-exactly, against the dense `256 x 256`
  matrix — on the host, independent of the device.
* All three device implementations agree with that cascade bit-exactly on
  CAModel, over 16 distinguishable rows.
* RegGather is the fastest of the three on the Vector engine, by 16.3x over
  Approach A and 6% over RegShuffle.
* Neither register-resident variant spills: 4 loads and 4 stores per row.

**Not established**, and not claimed:

* **Silicon.** These are model ticks for one AIV. The model's issue rules are
  not a timing guarantee for the part, and there is no NPU on this host.
* **Concurrency.** The measured region is bracketed by a drained `MTE2_V` wait
  and a `V_MTE3` set, so no DMA overlaps it. In the production kernel it does.
* **Amortisation beyond 16 rows.** `--reps` exists to separate the one-off table
  construction from the per-row cost and was run at 1 throughout. At 16 rows the
  construction is under 3% for both register variants, and §4.6 shows the
  ranking holds at the decode shape too, so nothing here depends on it. Higher
  repetition counts have not been run.
* **The egress direction on the device.** Only the ingress was launched. The
  egress stage table is verified on the host and the `Blocked` egress is what
  `routine_vector_egress_rotate.hpp` already ships and the decode operator
  already validated; `RegGather`'s egress path is compiled and host-verified but
  has not been run on CAModel.
* **Anything about Approach A's correctness as shipped.** Approach A is in this
  header as the measured baseline. The production scatter kernel must not select
  it, and nothing here retracts the decode egress's use of the same shape
  outside a hot loop — that trade was measured separately
  ([`paged_decode_attention.md`](paged_decode_attention.md)) and the rotation
  there runs once per sequence, not once per cached token.
