# One reshape-and-cache operator, INT8 and BF16

Design note, not an implementation report. Nothing here has been built or run;
where a claim needs the model or hardware to settle, it says so.

What is being parameterised is the **input** activation type — `DataType::INT8`
(the current A8 kernel) and `DataType::BFLOAT16` (the vendored baseline's input
precision). The output contract does not move: the INT4 nibble layouts, the
FP32 scale cache and the slot mapping are the same for both.

## 1. Where the 16-bit footprint actually lands

| surface | INT8 | BF16 | moves? |
| --- | --- | --- | --- |
| GM activations | 1 B/elem | 2 B/elem | 2x |
| L1 page, per chunk | 4096 B | 8192 B | 2x |
| L1 ring, 4 slots | 16 KiB | 32 KiB | 2x |
| fractal shape | 16x32 (K0=32) | 16x16 (K0=16) | yes |
| resident rotation tables | 16 KiB | 32 KiB | 2x |
| L0A / L0B occupancy | 20 KiB of 64 | 40 KiB of 64 | fits |
| Mmad accumulator | INT32 | FP32 | same width |
| L0C | 16 KiB INT32 | 16 KiB FP32 | no |
| Fixpipe L0C -> UB | INT32 -> INT32 | FP32 -> FP32 | no |
| **UB arena, whole Vector stage** | — | — | **no** |
| INT4 cache + FP32 scales | — | — | **no** |

The load-bearing observation: **the dtype touches only the ingress**, GM -> L1
-> L0 and the resident operand. The Fixpipe already hands the Vector core a
4-byte panel in both cases, so the absmax ladder, the power-of-two scale
extraction, the nibble pack, the egress bursts and the entire UB map are
dtype-invariant. This is a retyping of two stages, not a rewrite of the
quantiser.

## 2. kGroup stays 32, and stops meaning "K0"

The fractal is `16 x (32B / sizeof(T))`, 512 bytes
(`SIMD-API/basic_api/cube_compute_ISASI/cube_compute_fractal_intro/key_fractal_format_detailed_explanation.md`:
"8-bit (int8): K0 = 32 (fractal 16x32)", "the fractal is 16x16 for the half
type"). So a BF16 fractal spans 16 channels where an INT8 fractal spans 32, and
the current `kGroup = 32` — justified in the kernel header *because* that is the
INT8 K0 — has to be re-derived rather than inherited.

Two ways out:

**(a) follow the hardware, kGroup = K0 = 16.** The randomised Hadamard
refactors as `(H16 (x) I16) . diag(piSigns) . (I16 (x) H16)`: 16 source groups,
`2 * 16 = 32` resident blocks of 16x16, and `16 * 16 = 256` Mmads per chunk
against today's 64. Different algebra, different tables, a second oracle.
**Rejected**: it quadruples the Mmad count and forks the golden model, which is
the thing that was just unforked.

**(b) follow the algorithm, kGroup = 32 for every dtype.** A group is then one
K fractal at INT8 and two at BF16. The algebra, the tables' logical content and
the oracle are *identical* — `MmadParams`'s `k` is in elements, so it stays 32,
and the per-group L1 -> L0 load becomes `kGroup / kFractalK` fractal loads
instead of one. 64 Mmads per chunk, unchanged.

Take (b). The code consequence is that one constant splits in two:

```cpp
constexpr std::uint32_t kGroup = 32;                      // algorithm: channels per Hadamard group
constexpr std::uint32_t kFractalK = 32u / sizeof(T);      // hardware: K0 for this dtype
constexpr std::uint32_t kFractalsPerGroup = kGroup / kFractalK;   // 1 at INT8, 2 at BF16
```

and every `static_assert` that currently conflates them must be separated.
`kHeadSize == kGroups * kGroup` survives as an algorithm invariant;
`kFractalBytes = kTokens * kGroup` and `kL1PageBytes = kTokens * kHeadSize` are
byte counts that must pick up `sizeof(T)`.

## 3. The "one L1 image, two operand roles" symmetry

The kernel's cheapest trick is that one GM -> L1 ND -> NZ copy per chunk serves
as the A operand of the key chunks and the B operand of the value chunks,
because for INT8 the NZ image of a 16x256 page and the ZN image of its 256x16
transpose are the same bytes. The underlying reason is per fractal: NZ stores
`(M1, K1, M0=16, K0)` as 16 rows of 32 bytes and ZN stores `(K1, N1, N0=16, K0)`
as 16 rows of 32 bytes as well, so `A[t][i]` and `B[i][t]` read one byte image
with transposed meaning.

At 16 bits the fractal is 16x16 and a row is still 32 bytes, and with `M1 == 1`
and `N1 == 1` (16 tokens) both formats degenerate to the same linear sequence
over `K1`, so the symmetry is *expected* to survive with two fractals per group
instead of one. It is not assumed: it is exactly what check 0b and the
bit-exact nibble comparison exist to prove, and the BF16 instantiation is not
claimed to work until the CAModel gate passes with it.

Open question to settle from the docs before writing the code (one page read,
not an experiment): whether `MakeFrameLayout<NZLayoutPtn, T>` /
`CopyL12L0A` accept `bfloat16_t` on dav-3510 with the same spelling, or whether
the 16-bit path wants `half`-typed layouts with a reinterpret.

## 4. Exactness: BF16 can keep the bit-exact contract, conditionally

This is the most important consequence of the templatization, and like the
power-of-two scale it is a *verification* decision rather than a performance
one.

The INT8 path compares bit for bit because every intermediate is an integer:
`|y| <= 2^15`, products and sums below `2^24`, exact in FP32 **whatever order
the Cube accumulates in**, and the single `CAST_RINT` is the only rounding
anywhere.

BF16 keeps that property **iff the activations are integer-valued**. A BF16
significand is 8 bits, so integers up to 256 are exact; a product of two is at
most `2^16` and exact in FP32; and a sum of 256 such products is an integer
below `2^24`, hence exact regardless of summation order. So:

- **integer-valued BF16 fixture** -> the existing oracle applies *unchanged*
  (it already works in `long long`), and the comparison stays bit-exact. This
  is the BF16 validation path.
- **general BF16 fixture** -> the Cube's FP32 accumulation order is
  unspecified, the oracle cannot reproduce it, and the check must degrade to
  cosine plus a format-justified relative-L2 bound. Much weaker, and it would
  no longer detect a wrong-slot read the way a nibble comparison does.

So: validate BF16 with integer activations and the shared oracle; if a
general-input test is ever needed, it is a separate, explicitly-labelled
tolerance test that never replaces the bit-exact gate.

## 5. Shape of the change in code

- `template <typename T>` over the geometry block and the two ingress stages:
  `CubeEngine<T>`, `StagePage<T, Chunk>`, `GemmGroupLoad<T, Chunk, G>` (now
  emitting `kFractalsPerGroup` loads), `GemmBlock<T, Chunk, C1, G>`.
- `FixpipeChunk` takes the accumulator type (`std::int32_t` / `float`); the UB
  landing base and ring arithmetic are unchanged because both are 4 bytes.
- `run_vector_core` is **not** templated. It reads a 4-byte landing panel and
  writes INT4 + FP32. Leaving it dtype-free is what keeps the diff small.
- Entry points: two `extern "C" __global__` kernels over one template —
  `turboquant_reshape_and_cache_a8` and `..._bf16` — mirroring the baseline's
  `ASCEND_TQ_DECLARE_RESHAPE_AND_CACHE(T)` macro, plus a host switch shaped
  like `vllm_ascend::turboquant_reshape_and_cache_impl(AscendType, ...)`. The
  published argument order and types stay exactly as they are, including the
  unqualified `uint32_t` scalars that CANN copies verbatim into the generated
  launch header.
- L1 arena: `kL1Ring` moves behind a 32 KiB table blob; `kL1End` lands near
  64 KiB of 512 KiB. L0A/L0B go to 40 KiB of 64 KiB, still resident for the
  whole kernel.

## 6. What does not change but must still be re-proved

The slot mapping contract, the two cache layouts, the scale indexing, the
four-stage schedule, the ring depths and every flag id are dtype-independent —
but the **pipeline balance is not**. At INT8 the Cube is idle for about two
thirds of its attributed cycles and MTE2 moves 4 KiB per chunk; at BF16 the
ingress doubles while the Mmad count is unchanged, so MTE2 is the candidate for
the new binding stage. That is a measurement. There is no NPU on this host, so
it is a CAModel/PipeSim characterisation question at best, and per the project's
own rule it must not be used to rank anything without hardware latency.

## 7. Order of work

Each step ends at the same gate — `scripts/run_reshape_a8_camodel.sh`, strips
1, 2 and 3 for iteration and `A8_SLICES="1 2 3 32 64"` before a claim — so a
regression is attributed to one step. At BF16 the ingress doubles, so budget
more than the measured INT8 cost of ~45 s fixed plus ~8 s per strip.

1. Split `kGroup` from `kFractalK` and retype the L1/L0 byte constants, INT8
   only, no behaviour change. Gate must be **inert**.
2. Template the ingress on `T`; instantiate INT8 only. Gate must be inert.
3. Add the BF16 instantiation and an integer-valued BF16 activation generator
   to `tests/common/turboquant_oracle.hpp`. `compute_oracle_reference` needs no
   change: it is already dtype-agnostic.
4. Gate the BF16 instantiation at 1, 2 and 3 strips.
5. Only then widen the shape space: ragged tails (`numTokens % blockSize`),
   `headSize != 256`, `blockSize != tokensPerCore`, and a slot mapping that is
   not one strip per page. Each of those breaks a documented contract of the
   current kernel and deserves its own design note.
