# The value contraction: a DataBlock tree, and two architectural proposals

Target: CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599`. **No NPU on this
host**, so every device number is CAModel over a collapsed grid and **nothing
here is a hardware claim.**

`reports/decode_splitk_aic_scalar_issue.md` ended on a single finding: AIV0's
85% Cube-idle is demand-limited, and 49,152 of its 83,835 vector instructions
at four strips are three PCs hit 16,384 times each -- exactly 256 per query row
-- which is `ReduceRepeat` reducing the value contraction one channel at a
time. This report replaces that, and answers two architectural proposals.

## 1. The numerical gate was already a ULP gate

The premise that the contraction sat behind a frozen bit-identity check does
not hold, and it is worth correcting before anything is authorised against it.
`tests/camodel/test_camodel_paged_decode_attention.cpp` checks:

| quantity | gate | touched by the contraction? |
| --- | --- | --- |
| `out_qrot` | **bit-exact** against the host rotation | no -- the Cube's ingress rotation |
| `out_scores` | **bit-exact**, every panel of every strip | no -- straight from the Cube's Fixpipe |
| `m`, `l` | `worst_state_ulp <= 64` | no -- `m` from the score reduction, `l` from `update_denominator` |
| `acc` | `worst_acc_ulp <= 256` | **yes, and only this** |

So the only quantity the contraction can move is `acc`, and `acc` has been
under a 256-ULP budget all along. No bit-identity needed breaking; what was
needed was staying inside a budget that already existed. The authorisation to
break bit-identity is therefore not exercised by this change -- the stronger
statement is available and is the one made here.

`ulp_distance` in that test is the monotone-across-zero integer form, which is
the right one: it does not degenerate near zero the way a relative bound does.

## 2. The tree

`weighted` is channel-major, `weighted[c * 16 + t]`, because the value cache is
packed that way and the nibble dequant preserves it. 16 lanes per channel is
two 32-byte DataBlocks.

What was there: `ReduceRepeat<SUM>` over 16 contiguous lanes, 128 repeats twice
(256 exceeds the repeat bound). Each repeat reduces ONE channel to one value,
occupying one cross-lane add of a unit that reduces eight DataBlocks per
instruction -- seven eighths idle every time.

What replaces it, both documented as supported on Ascend 950PR for `float`:

```cpp
// 4096 -> 512: every DataBlock to one partial, eight blocks per iteration.
// Channel-major means pair_partials[2c] and [2c + 1] are the low and high
// halves of channel c, contiguous. 64 iterations.
ReduceDataBlock<SUM, float, float>(pair_partials, weighted, kVecLanes,
                                   kDecodeChannels * kDecodeKeys / kDecodeVecLanes,
                                   1, 1, kDecodeVecLanes / kDecodeLanesPerBlock);
// 512 -> 256: adjacent odd-even pairs, which is exactly
// pair_partials[2c] + pair_partials[2c + 1]. 8 iterations.
ReducePairElem<SUM, float, float>(weighted_sum, pair_partials, kVecLanes,
                                  2 * kDecodeChannels / kDecodeVecLanes,
                                  1, 1, kDecodeVecLanes / kDecodeLanesPerBlock);
```

72 operations against 256, **with unit strides throughout**. That last point is
what decided the shape. The obvious first level -- a block-strided `Add`
halving 16 lanes to 8 within each channel -- would have been 32 operations
instead of 64, but it needs `srcBlkStride = 2` to step over alternate
DataBlocks, and on this SIMT path a strided repeat is the thing that lowers to
one instruction per block rather than one per repeat. The two-stage form keeps
every access contiguous and pays 64 + 8 for it.

It costs a 512-float intermediate per core, appended at each arena's tail so
that no existing operand's offset moves. Three kernels share the recurrence
engine -- the paged decode operator and both single-token decode kernels -- so
all five construction sites set it; the `kArenaEnd + pad <= kUbWindow` assert
in each kernel is what would have caught a UB overflow.

### 2a. What changes numerically, and what does not

The old form summed 16 lanes in `ReduceRepeat`'s internal order. The new one
sums two groups of eight and then adds the two. Per channel that is a different
rounding of the same sum, so `acc` moves by a few ULP and nothing else moves at
all. The summation order **within** each group of eight is whatever
`ReduceDataBlock` does, and the order across channels is irrelevant because
each channel is its own accumulator -- which is why this reordering is
per-channel local rather than a global reassociation.

## 3. Measurements

All CAModel, `--impl aic --single --requests 1 --heads 1 --rows 1`, against the
deferred-wait revision that preceded this change.

### 3a. Ticks

| rung | before the tree | with the tree | change |
| --- | ---: | ---: | ---: |
| pages 4, K 1, 16-token pages | 86,390 | **73,728** | **-14.7%** |
| pages 8, K 1, 16-token pages | 164,683 | **139,110** | **-15.5%** |
| S = 127 of 128, 128-token page | 166,592 | **140,621** | **-15.6%** |

### 3b. AIV0's vector work, per query row

| mnemonic | before | tree | change |
| --- | ---: | ---: | ---: |
| `RV_VCADD` | **257.0** | **0** | gone |
| `RV_VCGADD` | 0 | 64.0 | the DataBlock reduce, one per iteration |
| `RV_VLDI` | 334.0 | **150.0** | -184 |
| `RV_VSTUI` | 258.0 | **66.0** | -192 |
| `RV_VLD` | 100.2 | 100.2 | — |
| `RV_VMUL` | 71.0 | 71.0 | — |
| `RV_VST` | 94.5 | 102.4 | +8 |

By engine, over the whole four-strip run:

| engine | before | tree | change |
| --- | ---: | ---: | ---: |
| RVECEX | 24,492 | **12,780** | -48% |
| RVECLD | 29,471 | **17,695** | -40% |
| RVECST | 26,365 | **14,525** | -45% |
| **all RVEC\*** | **~83,700** | **~48,400** | **-42%** |

The 257-per-row `RV_VCADD` is **gone**, replaced by 64 `RV_VCGADD` -- exactly
the 64 `ReduceDataBlock` iterations, one group-add each. The three PCs hit
16,384 times apiece are gone with it; the hottest PCs now take 4,352 hits,
68 per row.

AIV0's SCALAR count barely moves (11,458 against 11,336): the per-row scalar
bucket is instruction ISSUE for the recurrence body, and the body still issues
the same phases, just with far less vector work behind each one.

### 3c. What did not move, and why that is the honest reading

**`cube_idle_fraction` is 0.875, against 0.850 before.** It does not improve
and it was never going to: it is a RATIO, the Cube's 136 instructions are tiny
either way, and the span shrank by about the same proportion as the Cube's
share of it. Anyone optimising against that number will keep finding it
immovable. The tick count is the measure that responds, and it responded by
15%.

The tick saving (-15%) is also much smaller than the vector-instruction saving
(-42%), which says AIV0's critical path is not purely instruction-count bound
-- dependency latency, `RV_SEND` and `RV_SMEM_BAR` traffic, and the unchanged
scalar issue are carrying the remainder. Removing the next 42% of vector work
would not buy another 15%.

### 3d. Numerics: nothing moved at all

Every rung reports margins identical to the pre-tree revision, to every digit
printed:

```text
pages 4: acc=0.00025x  got=0.630742371 ref=0.63074216  mean |deviation| 6.208090e-08
pages 8: acc=0.00020x  mean |deviation| 4.350898e-08
S = 127: acc=0.00019x  mean |deviation| 4.480050e-08
```

`acc over tolerance 0 of 4096 channels in 0 of 16 rows` on all three, `m` and
`l` bit-exact, `qrot wrong 0`, replicas intact. So at these geometries the
reassociation changed no FP32 result -- either the two orders coincide for
this data, or `ReduceRepeat` was already reducing by DataBlock pairs
internally.

### 3e. The 192 ULP predates the tree

The legacy gate reports `acc <= 192 ULP` of its 256 budget, which is 75% of
it, and a figure that close to a limit is worth attributing rather than
accepting. So the same rung was run against commit `a86197b` -- the revision
immediately before the tree, with only the four files the tree touched
restored, and a guard that fails the run if `ReduceDataBlock` is still in the
restored source, because a silent no-op restore would have produced an
"identical" answer that proved nothing:

| revision | scores | m, l | acc | context err |
| --- | --- | --- | --- | --- |
| `a86197b`, pre-tree | 0/2048 wrong | <= 2 ULP | **<= 192 ULP** | 5.960e-08 |
| with the tree | 0/2048 wrong | <= 2 ULP | **<= 192 ULP** | 5.960e-08 |

Identical. Those 192 ULP are **pre-existing FP32-versus-FP64 drift** over an
eight-strip accumulation of 256-channel dot products, not anything the
reassociation introduced, and the margin to the budget is as wide as it ever
was. The authorisation to spend numerical accuracy here went unused.

One thing worth recording about getting that measurement: `git -C` run inside
WSL cannot read this worktree, because its `.git` file holds a Windows path.
The pre-tree files had to be extracted Windows-side and copied in.

## 4. Proposal: offload the contraction to the Cube as a GEMM

The contraction is `weighted_sum[c] = sum_t p_scaled[t] * V[c][t]`, a
matrix-vector product, and batched over the 16 query rows it is a genuine
`[256 x 16] . [16 x 16]` GEMM -- 65,536 MACs per strip, about one more Mmad
alongside the score GEMM already there. The Cube retires 136 instructions in a
run where AIV0 retires 84,000, so it has the headroom by a wide margin.

**It is feasible and I do not recommend it, for three reasons in increasing
order of seriousness.**

**Precision is the blocking one.** The Cube's Mmad takes bf16, fp8 or int8
operands with FP32 accumulation; there is no FP32 Mmad on this part. `V` after
dequant is exact in bf16 -- int4 values are -7..7, inside bf16's 8 mantissa
bits -- so the value side is free. `p_scaled` is not: it is
`exp(score - m) * value_scale`, a probability in [0, 1] carrying full FP32
significance, and bf16 rounds it to a relative 2^-8 = 3.9e-3. Against an `acc`
tolerance of `2e-4 * scale + 1e-5 * l + 1e-6`, which at `scale ~ 1` and
`l ~ 90` is about 1.1e-3, a single bf16 product is already 3.5x over budget
before any summation. The standard repair is to split `p` into a high and a low
bf16 part and issue two Mmads, recovering ~16 mantissa bits; that works
numerically and costs the Cube nothing it cannot afford, but it is now two
staged operands and two Mmads per strip to replace work that the tree above
already reduced to 72 vector operations.

**It adds a second cross-core round trip per strip.** `p_scaled` is derived
from the score panel, which the Cube produced. So the chain becomes
Cube -> AIV0 -> Cube -> AIV0 per strip, where today it is Cube -> AIV0. That
doubles the rendezvous count on the critical path, and this operator has just
spent two reports establishing how expensive it is to get one rendezvous right:
a mode-2 quorum, a release broadcast, the §6.2a drain in front of every
announcement, and a token model to prove the credits balance. A second channel
pair is not a small addition to that surface.

**It moves the value page across the CV path.** `V` is produced by AIV0's
nibble dequant into UB and would have to reach L1 for the Cube to load it --
an 8 KB bf16 MTE3 burst per strip on AIV0, plus an MTE1 load on the Cube. In
instruction count that is cheap. In structure it means the value page joins the
key page in needing an L1 WAR edge against Cube reuse, which is precisely the
hazard class that produced the four-page defect in
`reports/decode_splitk_aic_page_race.md`.

**What it would actually buy**, now measured rather than assumed. After the
tree, AIV0 retires ~756 vector instructions per query row. The `Mul` that forms
`weighted` is 71 of them and the contraction is 64 `RV_VCGADD` plus its pair
stage -- so offloading the whole product-and-contract step removes on the order
of 140, about **18% of AIV0's remaining vector work**, and by section 3a's
ratio perhaps 6-7% of wall time. That is bought with a second cross-core
rendezvous per strip and a second L1 WAR edge. The contraction is no longer
where AIV0's time goes, which is the main argument against spending the
structural budget here.

**The cheaper version of the same idea**, if the Cube is to be used at all: the
Cube already holds the score panel in L0C before Fixpipe. Folding the `exp` and
the contraction into the Cube is not possible -- there is no transcendental on
the cube -- but the `alpha` rescale of `acc` (`Mul` over 256 channels, 4
operations) and the final `Add` are the only other channel-wide operations, and
both are already negligible. There is no large, cube-shaped piece of AIV0's
work left once the contraction is a tree.

## 5. Proposal: BF16 intermediate accumulators

**Ruled out by the tolerance that is already in force, by two to three orders
of magnitude.** The numbers:

| buffer | what it holds | bf16 viable? |
| --- | --- | --- |
| `acc_cur` | the carried accumulator, summed over up to 1024 tokens | **no.** 3.9e-3 relative per add, accumulating over 64 strips. The whole point of the online-softmax carry is that it is exact enough to merge across Split-K workers |
| `weighted` | the 4096 products `p * V`, per strip | **no.** One bf16 rounding is 3.9e-3 relative against an `acc` tolerance of ~1.1e-3 |
| `pair_partials` | the 512 DataBlock partials | **no**, and for a mechanical reason as well: `ReduceDataBlock` and `ReducePairElem` both require `T` and `U` to be the same type, so a bf16 source cannot produce an FP32 partial. A `Cast` of 4096 elements to bridge it costs more than the 72 operations the tree spends in total |
| `p`, `p_scaled`, `scaled_row` | 16 lanes each | no benefit. These are a quarter of a DataBlock; halving their width changes no instruction count, because the op is already one repeat |

The throughput argument is also weaker than it looks here. bf16 doubles the
lanes per repeat, so it halves the op count for buffers big enough to span many
repeats -- `weighted` at 4096 elements is 64 repeats and would become 32. That
is a saving of 32 operations per row against a per-row budget of ~1300, bought
at 3.9e-3 relative error on the value path. The memory-pressure argument is
real but not binding: the UB arena has headroom, which is why a 512-float
intermediate could simply be appended to it.

**Where bf16 already pays, and why that case is different:** the key path.
Keys go int4 -> bf16 -> L1 -> Cube, and bf16 is *exact* there because int4
values are -7..7. The lesson is that bf16 is free wherever the value being
carried has eight or fewer significant bits, and ruinous wherever it carries
FP32 significance. `V` qualifies; `p` does not.

## 6. The pipeline overlap audit

`scripts/analyze_strip_pipeline_overlap.py --overlap` reconstructs each
engine's busy INTERVALS from its retirement ticks -- two consecutive
retirements join when the gap is at most `--gap` ticks -- and intersects them
exactly. That replaces the binned occupancy picture, whose answer moved with
the bin width, with a number. It does depend on `--gap`, so the threshold is
printed with every figure and `--gap-sweep` runs several.

`VECTOR` on an AIV means the union of the RISC-V vector units and the classic
`VEC` column, because the recurrence runs on the SIMT path: in these traces
`VEC` holds 15 instructions against `RVECEX`'s 12,780. Reading `VEC` alone
would have said the Vector pipe was idle.

### 6a. Pairwise overlap, four strips, gap 64

| pair | overlap | share of A | share of B |
| --- | ---: | ---: | ---: |
| AIV0 MTE2 vs AIV0 VECTOR | 31 ticks | **52.5%** | 0.0% |
| AIV0 VECTOR vs AIV0 MTE3 | 0 | 0.0% | 0.0% |
| Cube CUBE vs AIV0 VECTOR | 380 | **28.6%** | 0.6% |
| Cube FIXP vs AIV0 VECTOR | 7 | 33.3% | 0.0% |
| Cube MTE1 vs AIV1 MTE3 | 18 | 5.9% | 1.8% |

Given as a share of each side's own busy time, because the two sides differ by
three orders of magnitude and one percentage would hide that.

**Read §7b before quoting the occupancy figures in this section.** Everything
here is measured at a 64-tick gap threshold, which is coarser than AIV0's actual
stalls: at 16 ticks the same pipe shows 842 gaps and 27.3% stall. Both numbers
are of the same trace and both are right -- this section is about whether
*pipes* overlap, §7b is about whether one pipe issues back to back.

`VECTOR` against `MTE3` is 0.0% at every threshold, and it is correct: AIV0's
MTE3 is six instructions, the partition write-out, which by construction runs
after all vector work.

### 6b. Bubbles: one, and it is the prologue

AIV0's VECTOR busy intervals at four strips are

```text
[1809,1810] [2332,2487] [2574,2574] [3113,3279] [6854,72159]
```

-- after tick 6854 it is **one unbroken interval to the end of the stream**.
93.3% occupancy over its own span, and every gap sits before the first score
panel exists, which no schedule can remove. **There is no scalar-issue
starvation and no lingering barrier in steady state.**

The gap sweep is what makes that a finding rather than an artifact:

| gap | busy | idle | bubbles |
| ---: | ---: | ---: | ---: |
| 16 | 51,160 | 19,190 | 842 |
| 64 | 65,627 | 4,723 | 4 |
| 256 | 65,714 | 4,636 | 3 |
| 1024 | 66,775 | 3,575 | 1 |

The 842 bubbles at gap 16 are sub-16-tick spacings between individual
instructions, not structural stalls; from 64 up the count is 1 to 4 and
occupancy is stable at 93-95%.

### 6c. The Deferred Wait is measurable, and it amortises

The steady-state MTE2 ingress bursts at ticks 20337, 23023, 36713 and 39405
all fall inside that unbroken VECTOR interval, which is the intent of deferring
the wait. The stronger evidence is the trend with strip count, since a fixed
prologue cost should amortise:

| | 4 strips | 8 strips |
| --- | ---: | ---: |
| AIV0 MTE2 hidden behind VECTOR | 52.5% | **76.5%** |
| Cube CUBE hidden behind VECTOR | 28.6% | **50.5%** |

### 6d. And there is nothing left to overlap

AIV0's VECTOR holds 48,386 of the block's roughly 50,000 pipe instructions at
four strips. The Cube holds 136 and AIV0's MTE2 holds 23 -- 59 ticks of busy
time out of 72,617. The overlap percentages are capped by that distribution,
not by scheduling, and 59 ticks cannot be hidden into significance however
well they are scheduled.

That is the end of this line of optimisation. The remaining work on AIV0 is
the recurrence itself: `select_query`'s per-row issue, `Exp`, the reductions
in `update_denominator`, and the ~150 `RV_VLDI` per row that survive the tree.
None of it is a *cross-pipe* scheduling problem -- but §7b finds 19,190 ticks
of within-pipe issue stall, ten fixed 17-tick gaps per query row, which is the
same per-row issue cost named here, now measured.

## 7. Proposal: a fused streaming FMA over the 16 strip tokens

The proposal is to delete `weighted` and the `ReduceDataBlock` + `ReducePairElem`
pair, and instead loop over the 16 tokens of the strip, broadcasting `P[t]` and
accumulating `P[t] * V[t, 0..255]` into `weighted_sum` with an `Mla`. The stated
motivation is the 16 serial vector bursts visible on AIV0, read as register RAW
stalls in the contraction.

Two separate things are wrong with it, and the second is the interesting one.

### 7a. The layout premise does not hold: the value page is channel-major

The rewrite needs `V[t, 0..255]` to be a contiguous 256-element slice. It is not.
From `tests/common/splitk_decode_fixture.hpp`, which is the definition of the
packing every one of these kernels reads:

```cpp
const auto at_key = static_cast<std::size_t>(t) * kChannels + c;   // key:   token-major
const auto at_val = static_cast<std::size_t>(c) * 16 + t;          // value: CHANNEL-major
```

The key cache is token-major, the value cache is **channel-major**. For a fixed
token `t`, the 256 channels sit at a stride of 16 floats. Block strides step
whole 32-byte DataBlocks; they cannot express one element every sixteen, so
there is no repeat form that reads `V[t, ·]` as a vector.

This is not an accident of the fixture. The contraction the proposal would
remove is built on that layout, and says so in its operands:

```cpp
AscendC::Adds(p_x4, p_scaled, 0.0f, kKeyLanes, 4,
              AscendC::UnaryRepeatParams(1, 1, kKeyBlocks, 0));
AscendC::Mul(weighted, value_page, p_x4, kVecLanes,
             static_cast<std::uint8_t>(kDecodeChannels * kDecodeKeys / kDecodeVecLanes),
             AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 0));
```

`p_x4` is `p` replicated four times and `src1RepStride = 0` holds it still while
`value_page` sweeps: that only multiplies the right weight into the right slot
because consecutive lanes are consecutive *tokens* of one channel. The layout
was chosen for exactly the horizontal contraction being proposed for removal.

Getting token-major values needs either a transpose per strip -- `Transpose` on
this product handles 16x16 blocks or NCHW/NHWC, so 16x256 is sixteen calls plus
a `sharedTmpBuffer` and 4096 elements of movement -- or a change to the value
cache packing, which would ripple through the reshape-and-cache kernels, the CPU
oracle and the NPU drivers. Neither is free, and §7b says neither would pay.

(`Axpy(dst, src, scalarValue, count)` would be a better fit than `Brcb` + `Mla`,
since it takes the multiplier as a scalar -- but reading `P[t]` into a scalar
means `GetValue` on a UB value, which §1.2's zero-scalar-compute invariant
forbids. The broadcast formulation was the right instinct.)

### 7b. The stalls are real, and the contraction is not where they are

The 16 serial bursts are really there, and the earlier "93.3% occupied, one
unbroken interval" figure in §6a hid them: that was measured at a 64-tick gap
threshold, which is coarser than the stalls. Re-run with
`scripts/analyze_vector_stalls.py`, which attributes each gap to the instruction
the pipe was waiting to issue:

```
$ python3 scripts/analyze_vector_stalls.py /tmp/tree-run/ev_p4 16
vector retirements: 48386
span 70350 ticks, stalled 19190 ticks in 842 gaps > 16 ticks (27.3%)
```

So 27.3% of AIV0's span is fine-grained stall. The user's observation stands and
my §6a emphasis needed this correction.

But the gaps are not distributed the way data-dependent stalls are:

| gap size | count |
| --- | --- |
| exactly 17 ticks | 769 |
| exactly 18 ticks | 4 |
| exactly 20 ticks | 64 |
| > 32 ticks | 5 |

A RAW stall varies with the producer's latency. A quantum of *exactly* 17 ticks,
769 times, is a fixed issue cost. Sweeping the threshold shows the same cliff:
842 gaps at 16 ticks collapse to 5 at 32.

| threshold | gaps | stalled ticks | share of span |
| --- | --- | --- | --- |
| 4 | 2824 | 33987 | 48.3% |
| 8 | 1165 | 23909 | 34.0% |
| 16 | 842 | 19190 | 27.3% |
| 32 | 5 | 4765 | 6.8% |
| 64 | 4 | 4723 | 6.7% |

And every instruction the rewrite would delete retires with **zero** stall in
front of it:

| mnemonic | retired | stalls | stall ticks | what it is |
| --- | --- | --- | --- | --- |
| `RV_VLDI` | 9600 | **0** | 0 | |
| `RV_VST` | 6551 | 64 | 1092 | |
| `RV_VLD` | 6416 | **0** | 0 | |
| `RV_VMUL` | 4544 | **0** | 0 | the `Mul(weighted, ...)` |
| `RV_VSTUI` | 4224 | 64 | 1280 | |
| `RV_VCGADD` | 4096 | **0** | 0 | `ReduceDataBlock` |
| `RV_SMOVI` | 1168 | **517** | **8789** | scalar/immediate into a vector register |
| `RV_VLOOPv2` | 976 | 67 | 5724 | hardware loop setup |
| `RV_VCPADD` | 512 | **0** | 0 | `ReducePairElem` |
| `RV_SMOV` | 448 | **128** | **2176** | scalar into a vector register |

`RV_VMUL`, `RV_VCGADD` and `RV_VCPADD` -- the materialize-and-reduce sequence,
9152 retirements -- stall exactly zero times. All 19,190 stalled ticks sit on
operand setup (`RV_SMOVI` 44% of its retirements, `RV_SMOV` 29%), hardware loop
setup, and stores. Cross-core flags account for 129 ticks of the 19,190.

The 645 `SMOVI`/`SMOV` gaps over 64 row-instances (4 strips x 16 rows) are about
ten per query row: a fixed per-row cost of configuring vector operands, paid
across the whole recurrence body, not in the contraction.

It scales with the row count, which is what a structural per-row cost should do
and a one-off should not:

| run | strips | gaps > 16 | stalled | share |
| --- | ---: | ---: | ---: | ---: |
| `ev_p4`, 4 pages K=1 | 4 | 842 | 19190 | 27.3% |
| `ev_p8`, 8 pages K=1 | 8 | 1678 | 33598 | 24.7% |
| `ev_s127`, ragged S=127 | 8 | 1678 | 34070 | 24.8% |

Doubling the strips doubles the gap count (842 -> 1678, against 2 x 842 = 1684).

### 7c. Why the rewrite would make the measured number worse

A per-token broadcast of `P[t]` into a vector register *is* an `RV_SMOVI`-class
operation -- the one class that carries the 17-tick penalty, 44% of the time.
Sixteen of them per row, 64 row-instances, is 1024 added instructions of exactly
the kind that stalls; at the observed rate that is on the order of 7,600 extra
stalled ticks. What it buys back is 9,152 retirements that currently stall zero
times -- issue slots, not stall. The proposal trades instructions that cost
nothing in stall for instructions that cost 17 ticks each, to fix a stall.

So: blocked on the layout, and against the evidence even if the layout were
fixed.

### 7d. What the measurement points at instead

If the cost is ~10 operand-setup instructions per query row, paid 16 times per
strip, then the lever is the per-row structure, not the contraction. The 16
query rows are independent -- separate `m`, `l` and `acc` -- and their state is
already laid out contiguously per row (`panel` [16][16], `m`/`l` [16][8], `acc`
[16][256]). A phase-major recurrence, where each phase issues once with a repeat
form covering all 16 rows instead of 16 separate row bodies, would pay the
operand setup once per phase per strip rather than once per phase per row: the
same amortisation the DataBlock tree in §2 already used, applied to the loop
structure instead of the reduction.

That is a rewrite of `routine_vector_decode_recurrence.hpp`, which three kernels
share, and the reductions stay within a row so the arithmetic ordering per row
need not change. It is not attempted here. Note the routine's existing comment
that unrolling the 16 rows at the strip sites instantiated 128 bodies and did
not compile in 25 minutes -- phase-major is the opposite move (widen each phase,
do not replicate the body), but the compile cost is worth checking early.

## 8. Gate status

| gate | state |
| --- | --- |
| device build, `-Wall -Werror` | **PASS**, zero errors, zero warnings |
| Split-K CAModel, pages 4 and 8 at K = 1 | **PASS**, 0 of 4096 channels over tolerance |
| Split-K CAModel, ragged S = 127 | **PASS**, 0 of 4096 channels over tolerance |
| §1a strip-stream token model | **PASS**, untouched by this change -- it models the schedule, not the arithmetic |
| §6.6 cross-core fence audit | **PASS**, untouched |
| legacy `camodel_paged_decode_attention` ULP gate | see below |
| `--page-tokens 128` over more than one page | **NOT RUN** |
| §7 fused streaming FMA | **NOT IMPLEMENTED** -- blocked on the channel-major value page (§7a), and §7b/§7c measure it as a regression |
| §7d phase-major recurrence | **NOT ATTEMPTED** |
| physical NPU | **UNAVAILABLE** on this host |

The legacy gate is the one that matters for this change, because `acc` is the
only quantity the contraction can move and that test is where `acc` is held to
256 ULP. It also re-checks `out_qrot` and `out_scores` bit-exactly, which
should be unaffected and are the control for "did something else move".

