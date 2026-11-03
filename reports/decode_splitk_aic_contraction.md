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
internally. That is a stronger outcome than the ULP budget required, and it is
reported as an observation at these rungs rather than as a guarantee: a
different fixture could separate the two orderings by a few ULP, which is
exactly what the 256-ULP budget is sized for.

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

## 6. Gate status

| gate | state |
| --- | --- |
| device build, `-Wall -Werror` | **PASS**, zero errors, zero warnings |
| Split-K CAModel, pages 4 and 8 at K = 1 | **PASS**, 0 of 4096 channels over tolerance |
| Split-K CAModel, ragged S = 127 | **PASS**, 0 of 4096 channels over tolerance |
| §1a strip-stream token model | **PASS**, untouched by this change -- it models the schedule, not the arithmetic |
| §6.6 cross-core fence audit | **PASS**, untouched |
| legacy `camodel_paged_decode_attention` ULP gate | see below |
| `--page-tokens 128` over more than one page | **NOT RUN** |
| physical NPU | **UNAVAILABLE** on this host |

The legacy gate is the one that matters for this change, because `acc` is the
only quantity the contraction can move and that test is where `acc` is held to
256 ULP. It also re-checks `out_qrot` and `out_scores` bit-exactly, which
should be unaffected and are the control for "did something else move".

