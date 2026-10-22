# `turboquant_reshape_and_cache_a8` on the 950PR CAModel

Four-stage depth-4 `MIX_AIC_1_2` prefill reshape-and-cache: one Cube core and
**both** Vector cores over the eight chunks of a single 16-token strip, with a
bit-exact INT4 paged KV cache.

| Field | Value |
| --- | --- |
| What it proves | a four-stage ring with MTE2, Cube/Fixpipe, both Vector cores and MTE3 in flight together; an INT8×INT8→INT32 Cube Hadamard factor fed straight from GM into L1 with no Unified Buffer hop; a KV cache whose every nibble and every scale is reproduced bit for bit by an integer host oracle |
| What it omits | multi-core tiling (`blockDim` is 1), ragged shapes, a non-page-aligned slot mapping, silicon |
| Target | `Ascend950PR_9599` CAModel, `__NPU_ARCH__ == 3510`, CANN 9.2.0-beta.2, `dav-3510` |
| Kernel type | `KERNEL_TYPE_MIX_AIC_1_2`, `#error` on `__MIX_CORE_AIC_RATION__ == 1` |
| Grid | 8 chunks (4 value heads then 4 key heads) over a depth-4 L1 page ring and depth-2 per-core UB rings, so **every ring wraps twice in one run** |

Sources:
[`kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp`](../kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp),
[`tests/camodel/test_camodel_turboquant_reshape_and_cache_a8.cpp`](../tests/camodel/test_camodel_turboquant_reshape_and_cache_a8.cpp),
[`scripts/analyze_reshape_trace.py`](../scripts/analyze_reshape_trace.py).

## Commands

```sh
bash scripts/run_modulo3_camodel.sh micro reshape_and_cache_a8
```

with `MODULO3_TIMEOUT=900s` (the probe needs 70 s of wall clock in the
cycle-accurate model, against the fused attention probe's few seconds, so the
script's 120 s default is now overridable). `MODULO3_BUILD_DIR` was a fresh
ext4 directory, `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2`,
`SOC_VERSION=Ascend950PR_9599`, `RUN_MODE=sim`, `CMAKE_BUILD_TYPE=Release`.

The Tier-1 host probe:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    tests/test_reshape_and_cache_a8_probe.cpp \
    -o bin/test_reshape_and_cache_a8_probe
bin/test_reshape_and_cache_a8_probe
```

The V0 iteration loop that preceded both, ~14 s per cycle instead of ~5 min:

```sh
bisheng -I$ASCEND_HOME_PATH/x86_64-linux/asc/include -std=c++17 \
    --npu-arch=dav-3510 -O2 -fsyntax-only --asc-aicore-lang <kernel>.cpp
```

A `-fsyntax-only` pass over the device source answers every "does this API
exist on dav-3510 with these operand types" question in seconds, and it is how
the INT8 Mmad shape, the Fixpipe conversion table and the INT4 pack were
settled before a line of the pipeline was written. The Fixpipe answer came out
of a `static_assert` in `arch/utils/check_data_type.h`: `CopyL0C2UB` supports
`(FP32, FP32)` and `(INT32, INT32)` under `NoQuant`, and **not**
`(FP32, INT32)`, so the INT32 panel is cast on the Vector core rather than by
the Fixpipe.

## The design, and the three places it departs from the brief

The rotation is a randomised Hadamard split across the engine classes. Per head
of 256 channels, with `ch = c1 * 32 + c0`:

```
Y = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32) . x,   normalised by invSqrtLen
```

the Cube applying the inner factor as one Mmad per channel group and the Vector
cores applying the sign diagonal and then the outer factor as three radix-2
butterfly stages. `tables` carries the 32×32 INT8 Sylvester matrix, resident in
L1 and in both L0A and L0B; `piSigns` carries the 256-entry sign diagonal.

**1. `kGroup` is 32, not 16.** The arch-3510 INT8 fractal is 16×32, so K0 is
32: a 16-point inner factor would be a half-filled K fractal. A 32-point inner
factor is exactly one fractal and needs no padding, which leaves the outer
factor at order 8 and three butterfly stages on the Vector core rather than
two.

**2. The Hadamard is the Cube's *B* operand for key chunks and its *A* operand
for value chunks, and there are eight Mmads per chunk rather than one.** This
is what lets one L1 image serve both cache layouts. For INT8 the NZ image of a
16×256 page and the ZN image of its 256×16 transpose are the same bytes — both
store fractal `c1` as 16 rows of 32 bytes at offset `c1 * 512` — so a single
GM→L1 ND→NZ copy produces a page that reads as

* `A`, `NZ(16, 32)` → `C[t][c0out]`, the token-major panel the key cache needs,
* `B`, `ZN(32, 16)` → `C[c0out][t]`, the channel-major panel the value cache
  needs,

with no second staging path and no element permutation anywhere. H32 is
symmetric and square, so its row-major 1 KiB image is byte for byte both its
own `NZ(32,32)` and its own `ZN(32,32)`.

**3. The work is shared one chunk per Vector core, not one token half per
core.** The brief's token split is not expressible against the value cache
contract. Channel-major means a channel's `blockSize` nibbles are adjacent, so
one channel of a 16-key page is 8 bytes; a core holding 8 of the 16 keys owns 4
bytes of every such group and can never assemble a 32-byte MTE3 burst, which is
the egress granularity. Splitting by chunk — chunk `c` to Vector core `c & 1` —
gives each core whole 16-token chunks, so both layouts come out as one
2048-byte burst, and adjacent chunks still sit on different cores, so the two
Vector cores overlap. The UB budget is unchanged: depth 2 × 16 tokens is the
same 32 KiB landing ring as depth 4 × 8 tokens, and depth 2 × 16 tokens of
packed output is the same 4 KiB.

### The quantiser is power-of-two, and that is a verification decision

With `2^e <= absmax < 2^(e+1)` the scale is `2^(e-2)` and the stored value is
`min(rint(Y * 2^(2-e)), 7)`, clamped only from above because
`|Y * 2^(2-e)| < 8` while round-to-nearest-even can reach +8. The exponent is
extracted with integer ops on the FP32 bit pattern — mask the mantissa, then
one integer subtract for `2^(e-2)` and one more for its exact reciprocal, since
the two exponent fields sum to 254 — so no division, reciprocal or logarithm
runs anywhere.

That matters because `Y` is an integer (`|x| <= 128`, inner factor ×32, outer
×8, so `|Y| <= 32768`, exact in both INT32 and FP32) and `2^(2-e)` is a power
of two, so the product is exact and the single round-to-nearest-even is the only
rounding in the entire path. A `7/absmax` scale would put the rounding of a
hardware `Div` inside the verification contract, where a 1-ULP reciprocal can
flip a tie and change a nibble. `invSqrtLen` folds into the stored scale alone,
because scaling `Y` and `absmax` together leaves every nibble unchanged.

**Consequence: this probe checks bit equality for an arbitrary INT8 input, not
only for a crafted fixture.** That is a stronger guarantee than the fused
attention probe's "keep every score a multiple of 1/128".

## V0 — compile time

Clean `bisheng --npu-arch=dav-3510 -fsyntax-only`, zero errors and zero
warnings, and clean host compile of the runner with `-Wall -Wextra -Werror`.
18 `static_assert`s make the layout contract a build failure rather than a
runtime surprise: every UB and L1 operand base 32-byte aligned, every ring slot
stride a whole number of DataBlocks and not a multiple of 256 bytes,
`kArenaEnd + 256 <= kUbWindow`, both MTE3 burst lengths whole 32-byte bursts,
the page a whole number of INT8 fractals, the L0 page region 512-byte aligned,
and the L1 ring inside the L1 buffer.

Measured from the compiled constants, against the budget the brief sets:

| Arena | This kernel | Brief |
| --- | --- | --- |
| L1 resident Hadamard | 1024 B at offset 0 | 256 B at offset 0 (order 16) |
| L1 page ring | 4 slots × 4128 B from 1056, end 17568 B of 512 KiB | 4 × 4096 B, 32-byte aligned plus 32 B skew |
| UB Fixpipe landing ring | depth 2 × 16416 B = 32832 B | depth 4 × 8 tokens × 256 ch × 4 B = 32 KiB |
| UB packed egress ring | depth 2 × 2144 B = 4288 B (2048 packed + 64 scale + skew) | depth 4 × 8 tokens × 128 B = 4 KiB |
| UB arena end | 146048 B of the 196608 B window, 50304 B spare behind the 256 B read pad | read pad 256 B reserved |
| L0A / L0B | 5120 B each: H32 1024 B resident + 4096 B of page fractals | H in L0A, page fractals in L0B |
| L0C | 16384 B, one 16×32 INT32 block per channel group | — |

The landing and egress rings are the brief's budgets exactly; the depth and the
token count are swapped because a chunk, not a token half, is the unit of work
per Vector core. The Hadamard is four times larger because `kGroup` is 32.

Three things cost a build cycle each and are recorded because they are
toolchain properties, not kernel bugs:

* `Cast<int4b_t, int16_t>` compiles but selects a **deprecated**
  `vcvt_s162s4` intrinsic. The INT32 → FP16 → INT4 chain is warning-free and
  equally exact, because the staged values are integers in `[-8, 7]` and both
  narrowing steps are lossless.
* a scalar parameter spelled `std::uint32_t` in the `__global__` signature
  **breaks the host leg**: CANN copies the parameter list verbatim into
  `aclrtlaunch_triple_chevrons_func.h`, which has no `using` for `std`. Use
  unqualified `uint32_t`. The device legs compile either way, so the failure
  arrives late.
* a `__gm__ const T*` **parameter of a `__global__` function** also breaks the
  host leg: the generated `host_stub.cpp` takes it as `const void*` and assigns
  it to a `void*` field of the argument block. The published argument order and
  types are kept and const is dropped on that one boundary, then restored one
  line into the body.

## V1 — host model

`tests/test_reshape_and_cache_a8_probe.cpp`, 13.9 ms:

```
PASS reshape_and_cache_a8 host probe in 13894.806 us
  orthogonality: M M^T == 256 * I over 256 x 256 signed entries
  factorisation: 24576 values over 6 input patterns identical to a dense apply,
                 largest |Y| 24576 (bound 32768)
  quantiser: 4096 nibbles inside [-8, 7], 4096 round-tripped through the
             dequant4 packing, worst relative error 0.5000
```

Re-running the bit-exact comparison the CAModel runner already does would prove
nothing, so this probe checks the three things that comparison *cannot* see,
because they are properties of the oracle rather than of the device:

* **the transform is genuinely orthogonal.** The explicit 256×256 matrix
  `M = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32)` satisfies `M M^T = 256 I`
  exactly, over every one of the 65536 entries. That is what makes
  `invSqrtLen = 1/sqrt(headSize)` the correct normalisation and "randomised
  Hadamard" a measured claim rather than a label;
* **the factorisation is the one the kernel implements.** The factored oracle —
  inner Hadamard per group, sign diagonal, then three radix-2 stages over the
  group index — agrees with a dense apply of `M` in exact integers for 24576
  values across six input patterns, including both INT8 extremes and the
  alternating ±127/−128 worst case. A butterfly cascade in the wrong order, or
  a stage transforming the wrong bit of the group index, still composes to
  *some* orthogonal map and would agree with a device that made the same
  mistake; it does not agree with `M`. The same run measures the largest `|Y|`
  as 24576 against the header's bound of 32768, so the INT32/FP32 exactness
  argument holds with margin;
* **the nibble contract round-trips.** All 4096 quantised values land inside
  `[-8, 7]` and come back unchanged through the unpacking the dequantisation
  leaf implements (two per byte, low nibble first, sign-extended INT4). The
  power-of-two reconstruction error is at most half a step everywhere the upper
  clamp does not bite.

Guard bands on all three mock buffers intact.

**Not done:** the discrete-event queue model in
`tests/test_pipeline_deadlock_probe.cpp` was not extended from the three-stage
ring to this four-stage one. Its unique value — finding a deadlock or a token
leak in sub-millisecond time rather than after a five-minute build — is real,
and this kernel simply did not need it: the flag graph was hand-checked for
token balance and set/wait alternation per `(route, id)` before the first
build, and V2 ran and V4 confirmed zero residual tokens on the first clean
build. That is evidence after the fact, not instead of the stage.

### The fixture, and why the first one was discarded

Section 6.4 asks for strictly increasing per-tile maxima so an online softmax's
rescale path runs on every tile. **This kernel has no running maximum** — each
chunk's `absmax` is independent — so the property is not applicable, and the
runner asserts what actually detects a wrong read instead: the eight packed
pages, the eight chunk maxima and the sixteen token nibble runs inside every
chunk are pairwise distinct, at least one chunk carries two different scale
exponents, all sixteen nibble values occur, and the INT4 clamp is taken. Every
one of those is an assertion over the oracle the runner already computes, not a
comment.

The first fixture was a low-order polynomial in `(t, c)`. Measured, it failed
three of those properties: the rotated panel came out sparse, so the per-token
maxima repeated with period three, the nibbles clustered on 0 and ±1 with
14/16 values ever occurring, and the upper clamp was **never** taken. A hashed
value in `[-12, 12]` with a per-chunk amplitude ramp gives a roughly Gaussian
rotated panel and passes all of them. The chunk index enters both the hash and
the amplitude, because an amplitude ramp alone would be invisible — the
quantised nibbles are scale invariant by construction, so proportional chunks
pack identically.

## V2/V3 — CAModel, in one run

Eight chunks over a depth-4 L1 ring and depth-2 UB rings means slot `s` is
refilled for chunk `s + 4` (L1) and chunk `s + 4` of the owning core's two
(UB), so **every buffer-reuse back edge is contended, not merely primed and
drained**. The depth-boundary run and the past-the-depth run coincide; there is
no second build parameter.

**Gate 1 — exit code and scalar faults.** Exit code 0. Zero
`invalid ldst addr`, zero `unrecognize ldst addr`, zero `su_ccu_mpu_err`, zero
`[error]`, zero `[ERROR]`, zero `[fatal]` in the run log.

**Gate 2 — canaries.** 1024/1024 guard bytes intact, head and tail, across all
eight Global Memory buffers. Beyond the guard bands, **24576 cache bytes
outside the written page still hold the exact `0xDEADBEEF` lane for their
offset**, and so does every `scaleCache` float outside the eight written runs —
so a mis-scaled page offset or a stray burst would be caught, not ignored.

**Gate 3 — numerics, per chunk and per token.**

| Output | Checked | Result |
| --- | --- | --- |
| `keyCache`, key-major | 8192/8192 bytes | bit-exact |
| `valueCache`, channel-major | 8192/8192 bytes | bit-exact |
| `scaleCache` | 128/128 FP32 | bit-exact, compared as bit patterns |

Bit-exact, not within a tolerance: there is no ULP budget to state, because the
only rounding in the path is a round-to-nearest-even of an exactly
representable product. Fixture coverage in the same run: per-chunk max `|Y|`
427, 681, 824, 1126, 1284, 1492, 2048, 1870 (pairwise distinct); 4 distinct
scale exponents with 5 of 8 chunks carrying two or more; 16/16 nibble values
exercised; the INT4 upper clamp taken on 3 of 8 chunks.

**Gate 4 — event balance.** 86 intra-core `SET_FLAG` / 86 `WAIT_FLAG` across 28
`(route, ID)` pairs on three cores, zero residual, event ids 6 and 7 untouched.
20 cross-core mode-4 sets and 20 waits, every absolute id paired AIV↔AIC, on
exactly the designed id set `[0, 1, 4, 5, 16, 17, 20, 21]` — AIV0's channel in
0..15 and AIV1's in 16..31, with AIV1 rebased by `INTRA_NUM` before pairing.

## V4 — trace audit

`scripts/analyze_reshape_trace.py`, wired into `run_modulo3_camodel.sh` under
its own name branch, exits 0. Six fatal checks: intra-core event balance,
cross-core pairing on absolute ids, chunk ownership both ways, both Vector
cores executing and egressing, a Cube that never touches Global Memory, and
four-engine concurrency.

Measured, all from `core0.{veccore0,veccore1,cubecore0}.instr_popped_log.dump`
and the matching completion dumps on one CAModel clock:

```
Total tick 20444; model wall clock 69.5 s
dispatch span   cycles   702..20033  (19331 cycles, 34806 instructions)
completion span cycles   706..20053  (19347 cycles)
AIV0 16600 instructions (RVECEX 6039, RVECLD 4270, RVECST 3522, MTE3 19, MTE2 3)
AIV1 16604 instructions (RVECEX 6039, RVECLD 4270, RVECST 3522, MTE3 19, MTE2 3)
AIC    807 instructions (SCALAR 403, FIXP 145, MTE1 105, CUBE 91, MTE2 39)
```

**Chunk ownership.** The Cube announced 4 panels to AIV0 on ids 0..15 and 4 to
AIV1 on ids 16..31; each core released 6 (2 primed plus one per owned chunk).
`MMAD x64` on the Cube pipe is exactly 8 chunks × 8 channel groups.

**Zero Global Memory workspace, and no UB hop for the page.** 0 Global Memory
stores anywhere on the AIC, 0 `MOV_UB_TO_L1` moves, and 18 MTE2 moves in all
for the rotation table and the eight pages, which is two bursts each — so every
page reached L1 straight from Global Memory. Egress is 8 Global Memory stores
per Vector core, one packed page and one scale slice for each of its four
chunks.

**Four engines in flight together.** Dispatch ticks place the window:

| Stream | CAModel cycles |
| --- | --- |
| MTE2 page staging | 1269 … 5378 |
| Mmad, first of each chunk's group of eight | 2532, 2898, 3484, 4017, 4594, 5063, 5508, 6121 |
| Fixpipe, first of each chunk's group | 2658, 2861, 3094, 3979, 4550, 4892, 5499, 6082 |
| AIV0 egress (packed, scale) | 5353/5354, 7176/7177, 13059/13060, 19205/19206 |
| AIV1 egress | 5589/5590, 7412/7413, 11183/11184, 14559/14560 |

Four-engine overlap is **581 cycles, widest 5379..5960**. Reading the four
streams across that window: chunk 7's page burst is the last MTE2 dispatch at
5378 and is still in flight, the Cube is finishing chunk 5's Mmads and starting
chunk 6's at 5508, AIV0 is rotating chunk 2 and AIV1 chunk 3 (their next
egresses are at 7176 and 7412), and MTE3 is carrying chunk 0 off AIV0 at 5353
and chunk 1 off AIV1 at 5589. That is the four-stage steady state the schedule
claims, located in cycles rather than inferred from source order. AIV0 and
AIV1 are
concurrently in flight for 12859 cycles, widest 703..8168. Engine busy cycles:
MTE2 4063, Cube 5139, Vector 16007, MTE3 3843. Pairwise overlaps, cycles:
Cube+Vector 4714, Cube+MTE2 3911, Vector+MTE2 3632, Vector+MTE3 3615,
Cube+MTE3 739, MTE2+MTE3 607.

## Reproducibility

Two independent runs, each from a freshly created build directory because
CANN's nested ExternalProject mutates its intermediates:

| | run 3 | run 4 |
| --- | --- | --- |
| cache and scale bytes bit-exact | 8192 / 8192 / 128 | 8192 / 8192 / 128 |
| guard bytes intact | 1024/1024 | 1024/1024 |
| untouched cache bytes | 24576 | 24576 |
| intra-core events | 86 sets / 86 waits, 28 pairs | 86 sets / 86 waits, 28 pairs |
| cross-core flags | 20 / 20, ids 0,1,4,5,16,17,20,21 | 20 / 20, same ids |
| dispatch span | 19331 cycles | 19331 cycles |
| four-engine overlap | 581 cycles at 5379..5960 | 581 cycles at 5374..5955 |
| total tick | 20444 | 20436 |

The cycle counts are identical and the absolute tick origins differ by 5 to 8
cycles, which is model start-up, not kernel behaviour.

## What the numbers do *not* say

* **The pipeline is Vector-bound and the four-engine window is narrow because
  of it.** Vector is busy 16007 of the 19331-cycle dispatch span while MTE2,
  Cube and MTE3 together account for 13045 and finish all eight chunks by cycle
  6121. The four stages genuinely overlap, but calling this a balanced
  four-stage pipeline would be wrong: three of the four engines spend most of
  the run idle behind the Vector stage.
* **19331 / 8 is not a throughput figure.** The per-chunk egress intervals
  measured in the steady state are 5883 and 6146 cycles on AIV0 and 3771 and
  3376 on AIV1; the whole-run span is fill plus a long Vector tail.
* **The two Vector cores are not symmetric in time, and this run does not
  explain why.** They execute identical instruction counts (16600 vs 16604) and
  identical vector mnemonic histograms over the same mix of two value and two
  key chunks, yet AIV0's last store is at cycle 19205 and AIV1's at 14559. It is
  a timing effect, not a work imbalance; the cause is untested.
* **The cost is in the Vector register traffic.** Per core, `RVECLD` 4270 and
  `RVECST` 3522 against `RVECEX` 6039: a level-2 op over a 4096-element panel is
  64 register repeats, and the key-chunk ladder's narrow stages (`mask` 32, 16,
  8) waste most of a 64-lane register per repeat. Nothing here measures what a
  wider fold or a fused stage would save.
* **Nothing here is silicon.** No NPU ran. CAModel ticks are a model, not
  device time, and the 200 / 200 / 105 AIC / Scheduler / SoC periods the model
  prints at start-up are its own configuration, not a measurement.
* **One strip, one core block.** `blockDim` is 1 and the 64-core saturation
  claim in the brief is untested: no run here advances the strip base by
  `GetBlockIdx() * tokensPerCore`.
* **One shape.** `numTokens == blockSize == 16`, `headSize == 256`,
  `numKvHeads == 4`, one page-aligned slot mapping. `numTokens` and
  `tokensPerCore` reach the device unused.

## Retractions

None. This is the first revision of this operator, and it contradicts no
earlier claim in this repository. Two claims elsewhere that it *relies* on and
that held up: section 6.1's retraction (the UB→L1 path exists) is not used here
because the page never enters the Unified Buffer, and section 6.2a's retraction
(a mode-4 set runs on the scalar stream and needs an `x_S` drain) was followed
from the start — every cross-core set in this kernel that means "my pipe work
is done" is preceded by a `FIX_S` or `V_S` drain.
