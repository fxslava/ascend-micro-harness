# `turboquant_reshape_and_cache_a8` on the 950PR CAModel

Four-stage depth-4 `MIX_AIC_1_2` prefill reshape-and-cache: one Cube core and
**both** Vector cores over the eight chunks of a single 16-token strip, with a
bit-exact INT4 paged KV cache.

**Second revision.** The whole randomised Hadamard rotation now runs on the
Cube; the Vector cores only reduce, scale and pack. That rebalancing also
exposed a synchronisation fault that the first revision had been passing with —
see [the bug the rebalancing exposed](#the-bug-the-rebalancing-exposed-a-reused-mode-4-flag-id-is-not-a-token).

| Field | Value |
| --- | --- |
| What it proves | a four-stage ring with MTE2, Cube/Fixpipe, both Vector cores and MTE3 in flight together; the *complete* orthogonal rotation as INT8×INT8→INT32 Cube work fed straight from Global Memory into L1 with no Unified Buffer hop; a KV cache whose every nibble and every scale is reproduced bit for bit by an integer host oracle |
| What it omits | ragged shapes, a non-page-aligned slot mapping, silicon. Multi-core tiling is **no longer** omitted — see [the third revision](#third-revision-the-multi-block-block-index-and-one-golden-oracle) |
| Target | `Ascend950PR_9599` CAModel, `__NPU_ARCH__ == 3510`, CANN 9.2.0-beta.2, `dav-3510` |
| Kernel type | `KERNEL_TYPE_MIX_AIC_1_2`, `#error` on `__MIX_CORE_AIC_RATION__ == 1` |
| Grid | 8 chunks (4 value heads then 4 key heads) over a depth-4 L1 page ring and depth-2 per-core UB rings, so **every ring wraps twice in one run** |

Sources:
[`kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp`](../kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp),
[`tests/camodel/test_camodel_turboquant_reshape_and_cache_a8.cpp`](../tests/camodel/test_camodel_turboquant_reshape_and_cache_a8.cpp),
[`tests/cpu/test_reshape_and_cache_a8_probe.cpp`](../tests/cpu/test_reshape_and_cache_a8_probe.cpp),
[`scripts/analyze_reshape_trace.py`](../scripts/analyze_reshape_trace.py).

## Commands

```sh
bash scripts/run_modulo3_camodel.sh micro reshape_and_cache_a8
```

with `MODULO3_TIMEOUT=300s`. `MODULO3_BUILD_DIR` was a fresh ext4 directory,
`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2`,
`SOC_VERSION=Ascend950PR_9599`, `RUN_MODE=sim`, `CMAKE_BUILD_TYPE=Release`.

The Tier-1 host probe:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    tests/cpu/test_reshape_and_cache_a8_probe.cpp \
    -o bin/test_reshape_and_cache_a8_probe
bin/test_reshape_and_cache_a8_probe
```

The V0 iteration loop that preceded both, ~14 s per cycle instead of ~6 min:

```sh
bisheng -I$ASCEND_HOME_PATH/x86_64-linux/asc/include -std=c++17 \
    --npu-arch=dav-3510 -O2 -fsyntax-only --asc-aicore-lang <kernel>.cpp
```

## What changed in this revision, and why

The rotation is a randomised Hadamard. Per head of 256 channels, with
`ch = c1 * 32 + c0`:

```
Y = M . x,   M = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32)
M[c1*32 + c0][g*32 + i] = H8[c1][g] . piSigns[g*32 + c0] . H32[c0][i]      (1)
```

The sign diagonal sits *between* the two Hadamard factors, so it is indexed by
the intermediate channel `(g, c0)`. That is why `M` has one signed entry per
(output channel, input channel) pair rather than a factorisation that commutes.

**The first revision split (1) across the engine classes:** the Cube applied
the inner factor `I8 (x) H32` as one Mmad per channel group, and the Vector
cores applied `diag(piSigns)` and then `H8 (x) I32` as three radix-2 butterfly
stages over the group index. That is the cheap factorisation in multiply-adds
and the expensive one in wall clock. Its own profile said so:

| | first revision |
| --- | ---: |
| Cube `WAIT_INTRA_BLOCK`, share of attributed cycles | 33.6% (21767 of 64826) |
| Cube `MMAD`, share | 2.8% (1792) |
| Cube `FIX_L0C_TO_DST` + its `MOV_SPR_XN` writes, share | 43.7% (28328) |
| Vector `RV_VLD` + `RV_VST`, share of 170437 | 48.6% (82703) |
| Vector `RVECEX`, share | 24.4% |
| four-engine overlap | 581 cycles of a 19331-cycle dispatch span |

The Cube waited twelve times longer for a Vector core to hand back a landing
panel than it spent multiplying; its largest single cost was *eight* Fixpipe
transfers per chunk rather than any arithmetic; and the Vector cost was
register traffic, of which the sign multiply and three butterfly stages are
four full-panel passes.

**This revision folds all of (1) onto the Cube.** The contraction splits over
the eight source groups `g`, and the outer factor's sign `H8[c1][g]` goes into
the *resident operand* instead of being applied afterwards. For output group
`c1` and source group `g` the Cube multiplies the page's group-`g` fractal by
the 32×32 block

```
T[s][g][c0][i] = (s ? -1 : +1) . piSigns[g*32 + c0] . H32[c0][i],
s = parity(c1 & g) = (H8[c1][g] < 0)                                       (2)
```

and accumulates the eight products for one `c1` into one L0C region
(`cmatrixInitVal` true on `g == 0`, false on the rest). Because the dependence
on `c1` is a single sign there are only **16 distinct blocks, 16 KiB of
INT8** — `tables` carries them and they stay resident in both L0A and L0B for
the whole kernel. 64 Mmads per chunk, one per `(c1, g)` pair.

Three consequences, all of them the point:

* the Vector stage loses the sign multiply and three butterfly stages, four of
  its thirteen full-panel passes, and with them the two ping-pong panels and
  two replicated sign panels they needed;
* the Cube emits **one Fixpipe per chunk** instead of one per channel group,
  because the whole 16×256 panel is finished in L0C before anything leaves it;
* the landing-panel back edge is waited *after* the Mmads rather than before
  them, since only the Fixpipe touches Unified Buffer. The Cube's 64
  multiplies for chunk *j* now overlap the Vector core's quantisation of chunk
  *j − 1* instead of queueing behind it.

### The cost of this, stated plainly

Folding a factor that is **diagonal in 32 channels** into a GEMM costs eight
times the multiply-adds: 1024 hardware MMADs over the run against 128. There
is no cheaper arrangement. `H8 (x) I32` is a batch of 32 independent order-8
transforms, one per `c0`; expressing it as a single GEMM means padding `K` and
`N` from 8 up to the 32-wide K0 and 16-wide N0 fractals, which is the same
factor of eight, and expressing it as 32 small GEMMs is worse. The trade is
worth making only because the Cube was idle for two thirds of its attributed
cycles, which is a property of *this* pipeline and not a general rule.

### One L1 image, two operand roles — now for the blocks too

For INT8 the NZ image of a 16×256 page and the ZN image of its 256×16
transpose are the same bytes, which is what lets one GM→L1 copy per chunk serve
the key chunks as an `A` operand and the value chunks as a `B` operand. The
same symmetry covers the resident blocks: a row-major 32×32 INT8 block reads as
its own `NZ(32,32)` and as the `ZN(32,32)` image of its transpose, so with
`W = T[s][g]`,

* value chunks, `A` = NZ load of `W` → `A[c0][i] = W[c0][i]`
* key chunks, `B` = ZN load of `W` → `B[i][c0] = W[c0][i]`

and (1) needs exactly `W[c0][i]` in both roles. The 16 blocks are therefore a
flat 16 KiB blob in GM, in L1, and in each of L0A and L0B: one GM→L1 copy and
one L1→L0 copy per operand role, all three byte-for-byte identity, because
`NZ(512,32)` and `ZN(32,512)` both have a single fractal along the 32-wide
axis.

`piSigns` is consumed by whoever builds `tables` and **is no longer read by the
device**. It stays in the published signature. The runner builds (2) from the
same sign function it uploads, and both the runner and the Tier-1 probe check
that the 16 blocks, composed the way the kernel composes them, reproduce all
65536 entries of `M`.

## Where this departs from the brief, and why

The brief asked for the rotation to be factored as **two cascaded order-32
GEMMs** with a permutation between them,

```
Y = (H32 (x) I8) . P . (I8 (x) H32) . x
```

stage 1 contracting over channel blocks of 32, stage 2 a stride swap in L1 or
L0, stage 3 a second `H32` pass on the Cube, and the Fixpipe landing the fully
rotated tensor in Vector UB. This revision delivers the brief's *objective* —
the entire orthogonal transform on the Cube, the Vector cores reduced to
`absmax`, power-of-two scale and INT4 pack — by a different factorisation. Two
reasons, both load bearing.

**1. Two order-32 stages cannot compose to the order-256 rotation, with any
permutation.** Write the channel index in bits. `I8 (x) H32` transforms bits
0..4 and `H32 (x) I8` transforms bits 3..7, so bits 3 and 4 are transformed
*twice*. Applying `H2` twice to one bit gives `2 I`, not a transform: the
product is `4 . (H8 (x) I4 (x) H8)` up to relabelling — an orthogonal map, but
not `H256`, and not the map the cache contract and the host oracle specify. A
permutation between the stages can relabel which bits each stage sees, but once
an order-32 stage has consumed five of the eight bits only three remain, so the
second stage can be of order 8 at most. The exact two-stage cascade is `H32`
then `H8`, which is what the first revision already implemented — with the
order-8 stage on the Vector core, because that is where an order-8 contraction
is cheap.

**2. A second Cube pass cannot carry the intermediate exactly.** Stage 1
produces `|z| <= 32 * 127 = 4064`, which needs 13 bits. The Cube's operands are
INT8, so the intermediate cannot be re-fed as INT8; FP16 represents integers
exactly only to 2048, so `z` is not exact in FP16 either; and a `CopyL0C2L1`
round trip, which this part does support, does not change that. A bit-exact
two-pass cascade would need the intermediate split into two INT8 limbs — two
more GEMMs plus Vector-side limb formation, which puts work back on the engine
the refactor is clearing — or an FP32 Mmad path. **Bit-exactness against an
integer oracle is this probe's headline property**, and it is not worth trading
for a factorisation that is in any case not the specified rotation.

What replaces it: a single accumulating pass with the sign diagonal and the
order-8 outer factor folded into 16 resident 32×32 INT8 blocks, so the whole of
`M` is applied as 64 Mmads per chunk into one L0C panel and one Fixpipe. The
brief's pipeline table is unchanged by this — MTE2 at column *j*, Cube at
*j − 1*, Vector at *j − 2*, MTE3 at *j − 3* — because only the content of the
Cube stage changed.

### What the brief predicted, and what was measured

| Brief | Measured | |
| --- | --- | --- |
| Clean compilation, `cann-9.2.0-beta.2` / `Ascend950PR_9599` | clean; no warning this revision introduced | met |
| 100% bit-exact `keyCache` and `valueCache` | 8192/8192 and 8192/8192 bytes, plus 128/128 scales | met |
| `WAIT_INTRA_BLOCK` on `cubecore0` from 33.6% to < 10% | **0.6%** (213 of 34351 cycles) | met |
| `RV_VLD` + `RV_VST` on `veccore0/1` from 48.6% to < 20% | **38.4%** (41059 of 106896); the absolute cost halved, 82703 → 41059 | **not met** |
| Four-engine steady-state overlap from 581 to > 2000 cycles | **1133** cycles, 89% of the 1269-cycle ceiling the MTE2 ∩ MTE3 overlap imposes | **not met** |
| Cube ~3500–4000 ticks, AIV ~3500 ticks | Cube busy 8885 cycles, Vector busy 11106, over a 12590-cycle span | balanced, not at those absolute numbers |
| Net UB savings > 60 KiB | **53440 bytes** freed | **not met**, 52.2 KiB |
| Remove `kUbWorkA/B`, `kUbFold128/64/32/16/8` | `kUbWorkB`, `kUbFold32`, `kUbFold16` and all three sign buffers removed; `kUbWork`, `kUbFold128/64/8` retained | partly |

The three misses are discussed where they are measured. Two of them have the
same cause — the key layout's `absmax` reduces along the fast axis, so it still
needs a fold ladder and a scratch panel for `|Y|` — and the third is a ceiling
rather than a shortfall.

## V0 — compile time

Clean `bisheng --npu-arch=dav-3510 -fsyntax-only`, zero errors, and a clean
host compile of the runner with `-Wall -Wextra -Werror`. The `static_assert`
set grew with the arena change and now also pins the two new invariants: every
resident rotation block 512-byte aligned so it can serve as an L0 operand, and
no mode-4 flag id reaching the library's reserved 11.

A device-side `-Wall -Wextra` pass over the kernel emits four
`-Wunused-const-variable` warnings — `kPackedBytes`, `kScaleFloats`,
`kStageDepth`, `kPackDepth`, from the translation of the file in which the
device sections are preprocessed out, where those four are the geometry
constants no `static_assert` references. All four predate this revision, which
adds none; the device legs are not built with those flags.

Measured from the compiled constants, against the first revision:

| Arena | this revision | first revision |
| --- | ---: | ---: |
| L1 resident rotation table | 16384 B at offset 0 (16 blocks) | 1024 B at offset 0 (H32 only) |
| L1 page ring | 4 × 4128 B from 16416, end 32928 B of 512 KiB | 4 × 4128 B from 1056, end 17568 B |
| L0A / L0B, each | 20480 B: 16384 B table + 4096 B page fractals | 5120 B: 1024 B H32 + 4096 B page |
| L0C | 16384 B, one 16×256 or 256×16 INT32 panel | 16384 B |
| UB Fixpipe landing ring | depth 2 × 16416 B = 32832 B | 32832 B |
| UB working panels | 2 × 16384 B (`cast`, `work`) | 4 × 16384 B, plus 1 KiB of raw signs |
| UB fold ladder | 8192 + 4096 + 512 = 12800 B | 8192 + 4096 + 2048 + 1024 + 512 = 15872 B |
| UB packed egress ring | depth 2 × 2144 B = 4288 B | 4288 B |
| **UB arena end** | **92608 B** of the 196608 B window | 146048 B |

**53440 bytes of Unified Buffer freed**, 103744 B now spare behind the 256-byte
read pad. The brief expected "> 60 KiB"; 52.2 KiB is what the key-layout
`absmax` leaves, because its reduction runs along the fast axis and still needs
three fold buffers plus a scratch panel for `|Y|` alongside the signed FP32
image. Aliasing the FP16 pack staging onto `work` would reach 60 KiB and was
not done: the two are live in overlapping parts of the stage, so the alias would
have to be argued rather than read off the arena map.

L0 is where the cost landed: 20480 B of each 64 KiB L0A and L0B, against
5120 B. It is the arrangement that fits. One Mmad per chunk against the dense
256×256 matrix would need 64 KiB in L0B for the key chunks and 64 KiB in L0A
for the value chunks, leaving no room for the page; materialising the eight
per-source-group 32×256 slabs costs the same 64 KiB. 16 blocks is the only form
in which both operand roles stay resident at once.

## V1 — host model

`tests/cpu/test_reshape_and_cache_a8_probe.cpp`, 12.5 ms:

```
PASS reshape_and_cache_a8 host probe in 12459.627 us
  orthogonality: M M^T == 256 * I over 256 x 256 signed entries
  factorisation: 24576 values over 6 input patterns identical to a dense apply,
                 largest |Y| 24576 (bound 32768)
  rotation blocks: 16 x 32x32 INT8 (16384 B) compose to 65536/65536 entries of M
  quantiser: 4096 nibbles inside [-8, 7], 4096 round-tripped through the
             dequant4 packing, worst relative error 0.5000
```

The third line is new, and it is the one this revision needs. The rotation is
entirely the Cube's resident operand now, so the device's whole arithmetic
contract is "these 16 blocks, selected by `parity(c1 & g)`, compose to `M`".
That is checked here over all 65536 entries, and again in the CAModel runner as
check 0b, against an `M` built independently from `dense_entry()`. A transposed
block or a wrong sign index still yields *some* orthogonal map; it does not
yield `M`, and without this check the only symptom would be wrong nibbles with
nothing to point at.

The orthogonality and factorisation checks are unchanged and still earn their
place: they are properties of the oracle, which the CAModel comparison cannot
see because it compares *against* that oracle.

### The quantiser is power-of-two, and that is a verification decision

Unchanged from the first revision, and the reason the comparison downstream can
demand bit equality. With `2^e <= absmax < 2^(e+1)` the scale is `2^(e-2)` and
the stored value is `min(rint(Y * 2^(2-e)), 7)`, clamped only from above because
`|Y * 2^(2-e)| < 8` while round-to-nearest-even can reach +8. The exponent is
extracted with integer ops on the FP32 bit pattern — mask the mantissa, then one
integer subtract for `2^(e-2)` and one more for its exact reciprocal, since the
two exponent fields sum to 254 — so no division, reciprocal or logarithm runs
anywhere.

`Y` is an integer: each row of `M` has 256 signed entries and `|x| <= 128`, so
`|Y| <= 32768`, exact in both INT32 and FP32. `2^(2-e)` is a power of two, so
the product is exact and the single round-to-nearest-even is the only rounding
in the entire path. `invSqrtLen` folds into the stored scale alone, because
scaling `Y` and `absmax` together leaves every nibble unchanged.
**Consequence: this probe checks bit equality for an arbitrary INT8 input, not
only for a crafted fixture.**

## The bug the rebalancing exposed: a reused mode-4 flag id is not a token

Balancing the pipeline made the first clean build produce **wrong pages on
AIV1's two key chunks**, chunks 5 and 7, while chunks 0–4 and 6 were bit-exact.
The runner's diagnosis identified it rather than a guess, and it is in the
runner permanently now, because stopping at the first differing byte says
nothing about *how* a page is wrong:

```
chunk 5 (key head 1): 1973/2048 bytes differ, 16/16 scales differ
   looks like oracle chunk 1 own layout: 1214/2048 bytes
   looks like oracle chunk 5 own layout:   75/2048 bytes
chunk 7 (key head 3): 2016/2048 bytes differ, 15/16 scales differ
   looks like oracle chunk 3 own layout: 1780/2048 bytes
   looks like oracle chunk 7 own layout:   32/2048 bytes
```

Chunk 5 uses AIV1's landing slot 0, last written for chunk 1; chunk 7 uses slot
1, last written for chunk 3. Both pages were **mostly the previous occupant of
their landing slot**, with some of the right data mixed in — a read racing a
write, not a wrong address. Every device scale came out half the oracle's, which
is the same story: the stale panel held a lower-amplitude chunk, so `absmax`
landed one exponent down.

The trace put cycles on it. For chunk 7 on AIV1, with
`kFlagProductReady = Owner * INTRA_NUM + Slot` — two landing slots, so each id
carries *two* announcements across the eight chunks:

| event | CAModel cycle |
| --- | ---: |
| Cube's chunk-3 `SET_INTRA_BLOCK sync_id:17` (panel ready, slot 1) | 6840 |
| AIV1 consumes it and reads slot 1 for chunk 3 | 7282 |
| AIV1 reaches its chunk-7 wait on the same id | 8988 |
| AIV1 starts reading slot 1 for chunk 7 | 10502 |
| Cube's chunk-7 Fixpipe **completes** | 11165 |
| Cube's chunk-7 `SET_INTRA_BLOCK sync_id:17` | 11167 |

The chunk-7 wait was released 2179 cycles before the announcement it was
waiting for, by the set from 6840 that the chunk-3 wait had supposedly
consumed. The conclusion drawn at the time -- that a mode-4 flag is not a
counting token you can cycle an id through -- was the conservative reading, and
the mechanism was not isolated.

> **Superseded, 2026-10-29.** It was isolated later.
> `tests/research/kernel_cluster_sync_probe.cpp` ran a cyclic handshake over
> recycled mode-4 ids on one physical block, with an `x_S` drain before every
> announcement, and completed cleanly on physical silicon and on CAModel with
> no early wait retirement. The stale release above was
> `MICROROUTINE_DESIGN_GUIDELINES.md` §6.2a's hazard: the announcement issued
> on the scalar unit before the Fixpipe it reported had committed, and
> recycling turned that reordering into a stale release instead of a harmless
> duplicate. Mode-4 waits do consume. The fix recorded below is still correct
> and still ships as the fallback entry point, but the operator's default is
> now the recycled two-slot ring; see §6.2c, §6.5 rule 2 and
> `reports/reshape_and_cache_inter_wave.md`.

**Fix: one flag id per cross-core event, never reused.** Per Vector core, over
its four chunks `j = Chunk / 2`:

```
productReady  Owner*16 + j            4 ids, one per chunk
panelFree     Owner*16 + 4 + j        the id the Cube waits for at chunk j
panelFree     Owner*16 + 4 + j + 2    the id the AIV sets after chunk j
```

so ids 4 and 5 are the two priming tokens, 6 and 7 the releases the Cube
consumes for its last two chunks, and 8 and 9 the releases the epilogue drains.
Ten ids per core, **every one set exactly once and waited exactly once in the
whole kernel**, all clear of the library's reserved 11..14. The audit now
asserts exactly that, because set and wait counts balance under reuse: counting
alone passes vacuously.

### Why it had never shown, and what that says about the first result

A pipeline in which one engine runs far ahead never consumes a stale token. The
first revision's Cube finished all eight chunks by cycle 6121 of a 19331-cycle
run, so a stale "panel ready" always pointed at a panel that had in fact been
written long before. **That kernel passed twice, reproducibly, with every gate
green, and its id scheme was unsound the whole time.** Rebalancing the pipeline
is what made the fault reachable — an argument for this refactor beyond the
cycle counts.

Two variants were built and run side by side before the cause was understood:
the landing-panel wait placed after the Mmads, as this revision designs it, and
placed before them, where the first revision had it. The early-wait variant
**passed, bit-exact** — with a margin of 85 cycles between the chunk-7 Fixpipe
completing at 10424 and AIV1 reading that panel at 10509. It passed because
moving the wait earlier delays the Cube's Fixpipe by one Mmad block, not because
it was correct. Shipping it would have carried the fault forward.

### A wrong diagnosis, recorded

The first explanation of the same trace was that the two Vector cores share one
four-bit flag pool, so that `Owner * 16 + Slot` aliases onto `Slot` and the
Cube's announcement to AIV0 released AIV1. It had real support:
`NotifyEventImpl` / `WaitEventImpl`
(`impl/basic_api/dav_3510/kernel_operator_sync_impl.h:789`, `:816`) pass a
mode-4 id straight to `set_intra_block` / `wait_intra_block` with no sub-block
offset of their own; `GetffstMsg` masks `flagId & 0xf` on the sibling ffts path;
and the AIV side of the trace prints an id of 16 as `sync_id:0`.

Renumbering both cores into ids 0..7 to "fix" it **deadlocked the kernel**: AIV1
stalled on `WAIT_INTRA_BLOCK sync_id:2`, its "panel ready", and the Cube on
`WAIT_INTRA_BLOCK sync_id:6`, its "panel free" from AIV1 — each waiting on a
flag the peer had addressed into the other sub block. So the id space **is** per
sub block, in both directions, and §6.2 of the guidelines was right: the
`+ INTRA_NUM` offset is how a participant names the other sub block's channel,
and the AIV-side trace value is that id with the sub-block bits resolved. The
wrong reading survived a library read and a trace read; only a deadlock settled
it. §6.2c and §6.2d of
[`MICROROUTINE_DESIGN_GUIDELINES.md`](../MICROROUTINE_DESIGN_GUIDELINES.md)
record both the finding and the misdiagnosis.

### A latent fault elsewhere in this repository

`kernels/ops/turboquant/turboquant_paged_attention_opt.cpp` cycles three ring
slots through each of five cross-core ids over eight pages, so each of its ids
carries up to three announcements. It passes today with its Cube ahead of its
Vector cores — the position this operator was in. **Not investigated and not
fixed here**; it is a different operator with its own probe.

## V2/V3 — CAModel, in one run

Eight chunks over a depth-4 L1 ring and depth-2 UB rings means slot `s` is
refilled for chunk `s + 4` (L1) and for the owning core's next-but-one chunk
(UB), so **every buffer-reuse back edge is contended, not merely primed and
drained**.

**Gate 1 — exit code and scalar faults.** Exit code 0. Zero
`invalid ldst addr`, zero `unrecognize ldst addr`, zero `su_ccu_mpu_err`, zero
`[error]`, zero `[ERROR]`, zero `[fatal]` in the run log.

**Gate 2 — canaries.** 1024/1024 guard bytes intact, head and tail, across all
eight Global Memory buffers. Beyond the guard bands, **24576 cache bytes
outside the written page still hold the exact `0xDEADBEEF` lane for their
offset**, and so does every `scaleCache` float outside the eight written runs.

**Gate 3 — numerics, per chunk and per token.**

| Output | Checked | Result |
| --- | --- | --- |
| `keyCache`, key-major | 8192/8192 bytes | bit-exact |
| `valueCache`, channel-major | 8192/8192 bytes | bit-exact |
| `scaleCache` | 128/128 FP32 | bit-exact, compared as bit patterns |
| `tables` composition | 65536/65536 entries of `M` | exact |

Bit-exact, not within a tolerance. Fixture coverage in the same run is
unchanged from the first revision, which is the point — the oracle did not
move: per-chunk max `|Y|` 427, 681, 824, 1126, 1284, 1492, 2048, 1870 (pairwise
distinct); 4 distinct scale exponents with 5 of 8 chunks carrying two or more;
16/16 nibble values exercised; the INT4 upper clamp taken on 3 of 8 chunks.

**Gate 4 — event balance.** 92 intra-core `SET_FLAG` / 92 `WAIT_FLAG` across 27
`(route, ID)` pairs on three cores, zero residual, event ids 6 and 7 untouched.
20 cross-core mode-4 sets and 20 waits over **20 distinct ids** — `[0..9]` on
AIV0's channel and `[16..25]` on AIV1's — with exactly one set and one wait
each, so no id carries a second announcement.

## V4 — trace audit

`scripts/analyze_reshape_trace.py` exits 0 on seven fatal checks: intra-core
event balance; cross-core one-set-per-id and per-core id ownership; chunk
ownership both ways; both Vector cores executing and egressing; a Cube that
never touches Global Memory; the Cube census; and four-engine concurrency.

```
Total tick 13599; model wall clock 45.1 s
dispatch span   cycles   671..13261  (12590 cycles, 21973 instructions)
completion span cycles   675..13281  (12606 cycles)
AIV0 10225 instructions (RVECEX 4311, RVECLD 2066, RVECST 2048, MTE3 19, MTE2 1)
AIV1 10229 instructions (RVECEX 4311, RVECLD 2066, RVECST 2048, MTE3 19, MTE2 1)
AIC   1519 instructions (SCALAR 763, CUBE 547, MTE1 105, FIXP 41, MTE2 39)
```

against the first revision's 20444 ticks, 19331-cycle span, 34806 instructions,
16600/16604 on the Vector cores and 807 on the Cube. The work moved rather than
disappearing: **the Vector cores issue 38% fewer instructions and the Cube
88% more.**

**The Cube census is asserted now, not reported.** `MMAD x512` is exactly
8 chunks × 8 output groups × 8 source groups, and there are exactly 8
`FIX_L0C_TO_DST` transfers, one per chunk. A Vector core that still applied part
of the rotation, or a Fixpipe per channel group, fails the audit rather than
showing up as a slower run.

**Zero Global Memory workspace, and no UB hop for the page.** 0 Global Memory
stores anywhere on the AIC, 0 `MOV_UB_TO_L1` moves, and on the MTE2 pipe exactly
8 `MOV_OUT_TO_L1_MULTI_ND2NZ` page moves plus one `ALIGNv2` for the 16 KiB
rotation table: 9 data moves in all. The table is one burst, not sixteen,
because the 16 blocks are a flat blob in every memory they touch.

**Engine occupancy, and what balanced actually looks like.**

| engine, busy cycles | this revision | first revision |
| --- | ---: | ---: |
| MTE2 | 5274 | 4063 |
| Cube (CUBE + FIXP + MTE1) | 8885 | 5139 |
| Vector (both cores) | 11106 | 16007 |
| MTE3 | 4362 | 3843 |
| dispatch span | 12590 | 19331 |
| Cube / Vector | **0.80** | 0.32 |

The first revision's report said: "calling this a balanced four-stage pipeline
would be wrong — three of the four engines spend most of the run idle behind
the Vector stage." That is no longer true of the Cube, which is busy 71% of the
span against the Vector cores' 88%. Those two are not quite like for like — the
Vector figure is the *union* of both cores' busy intervals, so a cycle in which
either core works counts once — but the ratio moved from 0.32 to 0.80 on the
same definition, and the cross-core waits say which side now stalls. MTE2 and
MTE3 remain lightly loaded, at 42% and 35%, which is what a 4 KiB ingress and a
2 KiB egress per chunk cost.

**Per-chunk cadence.** The Cube now produces a panel every 957–1184 cycles
(median 1073) and each Vector core consumes one every 1636–2033 (median 1927),
so the two cores together consume one per ~963 cycles. Producer and consumer are
within 10% of each other, with the **producer now the slower of the two**: the
Cube's cross-core wait is 191 accumulated cycles against the Vector cores'
6495 and 7609. The pipeline is Cube-bound by about 10%, having been Vector-bound
by 3:1. In the first revision the Cube finished all eight chunks by cycle 6121
of a 19331-cycle run.

**Four-engine concurrency, and the ceiling on it.** 1133 cycles, widest
7760..8242, against the first revision's 581. **The brief asked for more than
2000, and that was not reached.** The trace says why rather than leaving it as a
shortfall:

| | this revision | first revision |
| --- | ---: | ---: |
| MTE2 ∩ MTE3 overlap | 1269 | 607 |
| four-engine overlap | 1133 | 581 |
| four-engine as a share of that ceiling | 89% | 96% |

A four-way intersection cannot exceed the overlap of any pair inside it, and the
binding pair is ingress ∩ egress: MTE2 and MTE3 are each busy about a third of
the span, at opposite ends of a chunk's life. **Both revisions take almost the
whole window that pair leaves** — the window itself doubled, 607 → 1269 cycles,
which is the measurable effect of balancing the pipeline. Getting past 2000
would need more ingress or egress work per chunk, not a different schedule: no
rearrangement of four engines widens the overlap of two of them. Issuing the
MTE2 bursts further ahead, which the depth-4 L1 ring allows, moves the wrong
way — it makes MTE2 finish earlier and shrinks the window.

## Profiling — `msprof op simulator`

```sh
LD_LIBRARY_PATH=$BUILD/lib:$SIM_LIB_DIR:$ASCEND_HOME_PATH/lib64:... \
/usr/local/Ascend/cann-9.2.0-beta.2/tools/profiler/bin/msprof op simulator \
    --application=$BUILD/test_camodel_turboquant_reshape_and_cache_a8 \
    --output=/tmp/a8-prof-v2 --soc-version=Ascend950PR_9599 \
    --aic-metrics=PipeUtilization \
    --kernel-name=turboquant_reshape_and_cache_a8 \
    --launch-count=1 --timeout=180
```

`LD_LIBRARY_PATH` is load bearing and is not in the first revision's recipe:
without it msprof's child exits 127 and the run reports
`No profiling data matched --kernel-name=...`, which reads like a name mismatch
rather than a link failure. Exit 0, `All task success`, 4723 relations from
`turboquant_reshape_and_cache_a8_0_mix_aic`, and the probe printed its own PASS
underneath.

| core | duration (us) | running (us) | stalled | first revision |
| --- | ---: | ---: | ---: | --- |
| core0.cubecore0 | 5.95 | 5.71 | 0.24 | 7.10 / 7.05 / 0.05 |
| core0.veccore0 | 6.35 | 6.35 | 0.00 | 10.75 / 8.93 / 1.82 |
| core0.veccore1 | 7.00 | 7.00 | 0.00 | 8.17 / 8.17 / 0.00 |

The critical path falls from 10.75 us to 7.00 us, and the three cores now sit
within 18% of each other rather than 80%.

`cycles` below is issue-to-complete latency accumulated per instruction site,
so the per-core totals exceed the kernel span: the column ranks hot spots, it
does not measure occupancy.

```
core0.cubecore0   1392 issues, 34351 accumulated cycles   (was 696, 64826)
  CUBE     50.8%   MMAD 512 x 17458                       (was  2.8%, 64 x 1792)
  MTE2     18.7%   MOV_OUT_TO_L1_MULTI_ND2NZ 8 x 4752     (was 10.2%)
  SCALAR   17.8%                                          (was  6.2%)
  MTE1      7.1%   LOAD_2Dv2 66 x 2440                    (was  3.5%)
  FIXP      4.9%   FIX_L0C_TO_DST 8 x 1668                (was 43.7%, 64 x 15790
                                                           + MOV_SPR_XN 64 x 12538)
  FLOWCTRL  0.6%   WAIT_INTRA_BLOCK 213 cycles            (was 33.6%, 21767)

core0.veccore0  10273 issues, 106896 accumulated cycles   (was 16689, 170437)
  RVECEX   28.1%   RV_PLT 1329 x 7974, RV_VMULS 778 x 6224 (was 24.4%)
  RVECST   20.2%   RV_VST 1923 x 21270                     (was 23.4%, 3235 x 38803)
  RVECLD   20.0%   RV_VLD 2028 x 19789                     (was 28.1%, 4264 x 43900)
  SCALAR    9.9%
  PUSHQ     8.8%   VF 39 x 9134                            (was  9.8%, 61 x 16407)
  FLOWCTRL  7.3%   WAIT_INTRA_BLOCK 4 x 6495               (was ~0)
  MTE3      4.2%
```

`core0.veccore1` is the same profile to within 1%: 10277 issues and 108025
cycles, identical call counts on every vector mnemonic, and 8605 cycles of
`FLOWCTRL` against 7761.

**The two acceptance criteria stated as msprof shares.**

* **`WAIT_INTRA_BLOCK` on `cubecore0`: 33.6% → 0.6%.** Target < 10%, **met**.
  The same 12 calls that cost 21767 cycles now cost 191 — one of 168 cycles
  during pipeline fill and eleven of 2 to 4 cycles each. **The Cube does not
  wait at all any more, and the stall crossed over to the Vector cores**, where
  `FLOWCTRL` is 7.3–8.0% of attributed cycles: four panel waits costing 6495 on
  `veccore0` and 7609 on `veccore1`, the largest 2900. Read with the cadence —
  the Cube produces a panel every 1073 cycles (median) and the two Vector cores
  together consume one every ~963 — that makes the Cube's 64-Mmad block the
  pacing item by about 10%. The pipeline went from Vector-bound by 3:1 to
  **marginally Cube-bound**, which is as balanced as this schedule gets without
  changing the arithmetic on one side or the other.
* **`RV_VLD` + `RV_VST` on `veccore0/1`: 48.6% → 38.4%.** Target < 20%,
  **not met**. The absolute cost halved — 82703 → 41059 cycles, and 7499 → 3951
  instructions — but the denominator fell by the same kind of factor, so the
  *share* barely moved. It cannot move much further by rebalancing engines: at
  2066 loads and 2048 stores per core over four chunks, each remaining Vector
  op is a full-panel level-2 op costing 64 register loads and 64 stores, and
  there are eight of them left (cast, `|Y|`, two fold stages, the broadcast
  multiply, the clamp, and two narrowing casts). There is no arithmetic left to
  hide the traffic behind. Getting under 20% needs *fewer passes* — fusing the
  clamp into a saturating INT4 cast, and landing FP32 from the Fixpipe instead
  of casting INT32 on the Vector side, are the two candidates and neither was
  attempted here.

`code_exe.csv` is still empty, with
`Failed to parse pc to code, maybe kernel missed debug_line information`:
source-line attribution needs the kernel rebuilt with `-g`, which is one more
build cycle and has not been done.

## Reproducibility

Two runs from separately created build directories, because CANN's nested
ExternalProject mutates its intermediates.

| | run 1 | run 2 |
| --- | --- | --- |
| cache and scale bytes bit-exact | 8192 / 8192 / 128 | 8192 / 8192 / 128 |
| guard bytes intact | 1024/1024 | 1024/1024 |
| untouched cache bytes | 24576 | 24576 |
| composed table entries matching `M` | 65536 | 65536 |
| intra-core events | 92 sets / 92 waits, 27 pairs | 92 sets / 92 waits, 27 pairs |
| cross-core flags | 20 / 20 over 20 ids | 20 / 20 over 20 ids |
| dispatch span | 12590 cycles | 12590 cycles |
| four-engine overlap | 1133 cycles, widest 7760..8242 | 1133 cycles, widest 7755..8237 |
| total tick | 13599 | 13590 |

The cycle counts are identical and the absolute tick origins differ by 5 to 9
cycles, which is model start-up rather than kernel behaviour. A third execution,
the one `msprof` drove from run 1's binary, printed the same PASS at total tick
13628 — 29 ticks of profiler overhead on the same build.

## What the numbers do *not* say

* **Nothing here is silicon.** No NPU ran. CAModel ticks are a model, not device
  time, and the 200 / 200 / 105 AIC / Scheduler / SoC periods the model prints at
  start-up are its own configuration, not a measurement.
* **The flag mechanism is not characterised.** What was measured is that a
  reused mode-4 id released its second wait early. Nothing here distinguishes a
  level bit a wait does not clear from a lost set, an ordering effect inside
  `wait_intra_block`, or a CAModel artefact. And the unique-id discipline that
  fixes it does not scale: twelve ids per sub block bounds it to about a dozen
  cross-core events per core, so a production-length loop must reuse ids and
  nothing here says what makes that safe.
* **The eight-times multiply-add cost is paid, not amortised.** This revision is
  faster because the Vector stage shrank and the Fixpipe count fell by eight,
  not because the Cube got cheaper: 1024 hardware MMADs against 128. It also
  spent the headroom it was exploiting — the Cube is now the pacing engine by
  about 10%, so a further step in this direction makes the pipeline slower, and
  on a part or a shape where the Cube were already the critical path this
  refactor would be the wrong way round from the start.
* **`RV_VLD` + `RV_VST` is still the largest Vector cost**, at 38.4%. The
  quantiser's remaining work is elementwise over a 16 KiB panel and is memory
  bound by construction.
* **The two Vector cores are still not symmetric in time** — 6.35 us against
  7.00 us on identical instruction counts — and this run explains it no better
  than the last one did.
* **One strip, one core block.** `blockDim` is 1 and the 64-core saturation
  claim in the brief is untested: no run here advances the strip base by
  `GetBlockIdx() * tokensPerCore`. *Lifted in the third revision below, which
  also found that the untested expression was wrong.*
* **One shape.** `numTokens == blockSize == 16`, `headSize == 256`,
  `numKvHeads == 4`, one page-aligned slot mapping. `numTokens` and
  `tokensPerCore` reach the device unused, and `piSigns` now does too — a
  caller that changes `piSigns` without rebuilding `tables` gets silently wrong
  results, which is a contract the runner enforces and the kernel cannot.
* **`msprof` shares and the audit's cycle shares are different measures.**
  msprof's `cycles` is accumulated issue-to-complete latency per instruction
  site; the audit's shares come from merged busy intervals, which is occupancy.
  Comparable across revisions of this kernel, not against each other. The audit
  also prints a load+store share of 69.0%, on the busy-interval denominator;
  that is not msprof's 38.4% measured differently, it is a different quantity.

## Third revision: the multi-block block index, and one golden oracle

The second revision's own limitation list said the quiet part out loud — *"no
run here advances the strip base by `GetBlockIdx() * tokensPerCore`"* — and
that untested expression was wrong. On a physical device at 512 tokens the
kernel produced one partially-correct page out of 32.

### Root cause

`AscendC::GetBlockIdx()` is not the grid block index on a Vector core of a MIX
launch. CANN 9.2's `dav_3510` implementation
(`asc/impl/basic_api/dav_3510/kernel_operator_sys_var_impl.h`,
`GetBlockIdxImpl`) returns

```text
AIC:  get_block_idx()
AIV:  get_block_idx() * get_subblockdim() + get_subblockid()
```

and the API reference agrees — `GetBlockIdx.md`, Return Value: at a 1:2
AIC:AIV ratio the AIC range is `[0, numBlocks)` and the AIV range is
`[0, 2 * numBlocks)`. So the two Vector cores of block *b* addressed strips
*2b* and *2b+1* while their Cube read strip *b*:

- only AIV0 of block 0 agreed with its own Cube, and it owns just the even
  chunks — value heads 0 and 2, key heads 0 and 2, `8192` cache bytes and
  `64` scales out of a 512-token total of `528384` comparisons;
- every other Vector core wrote a page belonging to another strip;
- every core with an AIV index at or above `numBlocks` computed a page past
  the end of the cache and wrote there, and read `slotMapping` past its end
  to do it. The old harness sized the cache to exactly the written pages and
  filled nothing with a canary, so those writes landed in other allocations
  silently.

At `blockDim == 1` the quotient is 0 either way, which is exactly why the
second revision's eight-chunk single-strip runs passed, twice, on CAModel and
on silicon. The repo's own vendored baseline had the fix all along, in
`baseline/common/turboquant_common.h`: `MixBlockIdx()` divides by
`GetSubBlockNum()`, which is 1 on the AIC and the sub-block count on an AIV.
The kernel now does the same.

### One golden oracle

There were three implementations of the rotation and the quantiser: one in the
CAModel runner, one in the device harness, one in the Tier-1 host probe. They
agreed by inspection, which is not a property anything checks. All three now
consume `tests/common/turboquant_oracle.hpp`, which owns the primitives, the
activation fixture, the slot mapping, the resident-table construction, the
reference caches and a fixture self-audit. Two deliberate exceptions:

- the Tier-1 probe keeps its own `factored()` — a radix-2 butterfly cascade —
  because its check 2 is only meaningful if two structurally different
  applications of *M* agree in value while differing in form;
- each harness maps the kernel's chunk numbering (value heads first, then key
  heads) to `(plane, head)` itself. That numbering belongs to the kernel's
  schedule, not to the golden model.

The probe now also composes **the very table blob the runners upload** against
the *M* it has just proved orthogonal, rather than against a local rebuild of
it: 65536/65536 entries.

### The harness changes that make a multi-page run able to fail

The old device fixture used `slots[t] = t` with `numBlocks = tokens /
blockSize`. That is degenerate three ways, and each is now fixed in the shared
oracle:

| old | new | what it buys |
| --- | --- | --- |
| page index == grid block index | strip *s* owns page *2s+1* | a kernel that takes its page from `GetBlockIdx()` instead of `slotMapping[0]` now fails |
| `numBlocks == strips` | `numBlocks == 2 * strips + 2` | a stray page offset lands on a canary page instead of a neighbour's correct one |
| outputs uninitialised | `0xAA` (device) / `0xDEADBEEF` (CAModel) pre-fill | "never written" and "written wrong" are different diagnoses |
| one aggregate mismatch count | per-page verdicts, plus a scan for *whose* data is on a wrong page | a block-index error is named, not just counted |

### What was actually run

No NPU on this host, so the CAModel is the gate. `--slices N` collapses the
grid: the obligation is that strip *t*'s addressing is a function of *t* alone,
uniform in *t*, with state carried only through the per-core L1 and UB rings.

```text
scripts/run_reshape_a8_camodel.sh          # strips 1, 2, 3
```

| strips | tokens | verdict | total tick | model wall |
| --- | --- | --- | --- | --- |
| 1 | 16 | PASS | 13093 | 45.7 s |
| 2 | 32 | PASS | 13205 | 53.9 s |
| 3 | 48 | PASS | 13321 | 61.3 s |
| **32** | **512** | **PASS** | 13353 | 245 s |

32 strips is the shape the device run failed on. All 32 blocks executed and
every byte matched: `262144/262144` key bytes, `262144/262144` value bytes and
`4096/4096` FP32 scales — the same 528384 comparisons that previously reported
512342 mismatches, now exact — with the 34 pages no block wrote still holding
their canary and all 1024 guard bytes intact.

Bit-exact in every case: all key bytes, all value bytes, all FP32 scales, and
every byte and scale float on the pages no strip owns still carrying its
canary. The fixture audit holds at each grid — 16/16 nibbles exercised, range
exactly `[-8, 7]`, the INT4 upper clamp taken, four distinct scale exponents,
and per-strip chunk maxima and packed pages pairwise distinct.

The tick count is flat in the strip count (+112 per strip) because the model
has 32 AI Cores, each with 2 Vector cores, so the strips dispatch
concurrently; the ~8 s of wall clock per strip is simulator cost, not
simulated time. **Concurrent multi-block dispatch is therefore what these runs
verify.**

### The control: does the gate catch the bug?

A pass means nothing if the gate cannot fail. A copy of the tree with the
pre-fix expression restored, built and run at **2 strips**:

```text
control kernel line:
  (void)block; const uint32_t strip = static_cast<uint32_t>(AscendC::GetBlockIdx()) * tokensPerCore;
control exit code: 1
FAIL turboquant_reshape_and_cache_a8 (2 slices): keyCache byte on a page no strip owns was overwritten
```

Two strips is enough, and the failure is reported as a canary violation on an
unowned page — the diagnostic the fragmented mapping and the spare pages were
added to produce. The working tree was not modified: the control builds from a
copy under `/tmp`.

### Build gates

Both target sets build clean with `-std=c++17 -Wall -Wextra -Werror`:
`scripts/build_turboquant_npu.sh` produces `test_turboquant_a8_validation` and
`bench_turboquant_a8_vs_bf16` (compile-only — there is no device here), and the
Tier-1 probe builds and passes with the documented one-liner. A host
`g++ -fsyntax-only` pass over the four test translation units is the fast
pre-gate; it catches harness breakage in seconds instead of after a CANN build.

### Compile-time sparse boundary sampling

A full 64-strip run is ~550 s of model wall clock and the strips are
independent, so most of it re-proves one thing. With
`ASCEND_MICRO_SPARSE_SAMPLING` — set on the CAModel probe's kernel library and
its runner, and on nothing else, so the physical-device library contains no
sampling predicate at all — only part of each wave executes:

| per 32-block wave | indices | what it is for |
| --- | --- | --- |
| first `kSampleEdge` = 4 | 0-3 | the wave's cold start |
| `kSampleCenter` = 4 at the midpoint | 14-17 | the steady-state interior step |
| last `kSampleEdge` = 4 | 28-31 | the wave's tail, and the highest block indices |

24 of 64 blocks, 12 of 32. Every other block returns from the kernel entry
before it reads, writes, primes or announces anything, so it leaves no
HardEvent token, no mode-4 flag and no Global Memory write; the runner then
*requires* its page to still hold the canary, which is what makes a skipped
block's silence checkable rather than assumed. The predicate
(`tests/common/turboquant_sampling.hpp`) is a function of the grid block index
and of `numTokens / tokensPerCore`, so a block's Cube and both its Vector
cores always reach the same decision — an inconsistent predicate would
deadlock rather than misreport — and the sample set is pinned by
`static_assert` so a change to the constants that silently drops an end or the
interior fails the build.

The interior cluster is the half of this that earns its place. The boundary
blocks are contiguous, so only the unowned even pages separate the pages they
write; a center block is surrounded by *skipped* blocks, so the odd pages on
either side of its page belong to strips that never ran. A burst that overran
its 2 KiB — a mis-scaled offset, an interconnect or L2 write-combining effect —
therefore has a canary to land on. The runner asserts that every written page
has an unwritten page on both sides and reports the count.

Measured on the same shape and the same source: 32 strips in **118 s** sampled
(12 blocks) against **245 s** full (32 blocks), both PASS and bit-exact over
the blocks that ran.

### Open: a grid wider than the AI Core count hangs

**This is unresolved, and it is not caused by the fix or by sampling.**

At 64 strips the model spins: 25 min of CPU at 143% with no output after
`Model Start Time`, against ~200-240 s expected. A `wait_intra_block` that
never resolves looks exactly like this, because the model spins the flag wait
rather than parking. 33 strips, where exactly one core takes a second block,
showed the same signature before the run was stopped.

What is established:

| core reuse | skipped blocks | verdict |
| --- | --- | --- |
| no (32 blocks, 32 cores) | no | PASS |
| no (32 blocks, 32 cores) | yes, 20 of 32 | PASS |
| yes (64 blocks) | yes, 40 of 64 | spins |
| yes (33 blocks) | yes, 20 of 33 | no verdict before it was stopped |

So skipping alone is harmless and 32 concurrent blocks are fine — which also
establishes that mode-4 scope is per core group rather than per device. The new
variable is a physical core executing a second block inside one launch, which
had never happened before: this kernel had only ever run at `blockDim` 1, and
32 blocks on 32 cores is one block each.

Flag residue does not explain it. Per block every mode-4 id is set exactly once
and waited exactly once, so a counting semaphore and a test-and-clear level bit
both leave zero, and a level bit that a wait does *not* clear would leave every
id set — which makes the second block's waits pass immediately and produces
wrong data, not a spin. That points at the dispatch of a MIX grid wider than
the core count rather than at the kernel's flag accounting, but the isolating
run (33 strips with sampling compiled out, which has core reuse and no skipped
blocks) was stopped before it finished, so the cause is **not** isolated and
nothing here should be read as naming one.

The architectural answer, independent of the cause, is a persistent grid-stride
kernel: launch `min(strips, cores)` blocks and have each core loop
`for (s = block; s < total; s += GetBlockNum())` with the resident tables
hoisted out of the loop and an explicit drain at each iteration boundary. Then
the grid never exceeds the core count, and successive strips on one core are
sequential iterations whose flag accounting the kernel controls. Note for
whoever writes it: `GetBlockIdx()` must be divided by `GetSubBlockNum()` and
`GetBlockNum()` must **not** be — `GetBlockNum.md` says its return value is
`numBlocks` on both roles, so dividing it would give a stride of 32 on the Cube
and 16 on its Vector cores, and the two roles would disagree about the
iteration count. That is a deadlock, not a wrong answer.

Until that lands, **the 1024-token shape is unvalidated here**, and on silicon
it is suspect for the same reason if the part has 32 or fewer AI Cores.

### What this revision still does not claim

- **Nothing about silicon.** Every result above is "executes on the
  cycle-accurate model and matches the audited golden". The device run that
  started this is not repeated here, because the device is not here.
- **Nothing about performance at scale.** The strips are independent and the
  tick count is flat, so there is no new timing claim; the 64-core saturation
  question in the brief remains open and belongs to hardware.
- **Nothing above a grid of 32.** See the open item above: 64 strips spins, and
  the cause is not isolated.
- **Shapes are unchanged.** `blockSize == tokensPerCore == 16`,
  `headSize == 256`, `numKvHeads == 4`, one strip per page, no ragged tail.
  `numTokens` still reaches the device unused.

## Retractions

**One in this revision, mine.** The chunk 5 and 7 failure was first diagnosed as
the two Vector cores sharing a four-bit mode-4 flag pool, and the guidelines'
§6.2 was edited to retract its "ids 0..15 are AIV0's channel and 16..31 are
AIV1's". That was wrong: renumbering both cores into 0..7 deadlocked the kernel,
§6.2 was restored, and §6.2d records the misdiagnosis. The real fault — a reused
flag id — was found by the same trace read once the deadlock had ruled the first
explanation out.

**One carried forward from the first revision**, which claimed 18 MTE2 moves for
the page staging, "two bursts each", because the trace analyzer counted
`MOV_SPR_XN` special-register writes as data movement. That was corrected to 9
data moves and the analyzer now asserts exactly one ND→NZ page move per chunk.
The check still passes here.

Two claims this revision relies on, both of which held: §6.1's retraction (the
UB→L1 path exists) is not used, because the page never enters the Unified
Buffer; and §6.2a's retraction (a mode-4 set runs on the scalar stream and needs
an `x_S` drain) was followed throughout — every cross-core set that means "my
pipe work is done" is preceded by a `FIX_S` or `V_S` drain.
