# Fused TurboQuant paged attention — N = 3 boundary integration

Kernel: [`kernels/camodel/turboquant_paged_attention_opt.cpp`](../kernels/camodel/turboquant_paged_attention_opt.cpp).
Runner: [`tests/camodel/test_camodel_fused_attention_3tiles.cpp`](../tests/camodel/test_camodel_fused_attention_3tiles.cpp).

```sh
bash scripts/run_modulo3_camodel.sh micro fused_attention_3tiles
```

Target: CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599` CAModel, MIX mode
(one AIC plus AIV0), `CMAKE_BUILD_TYPE=Release`.

## What is integrated

| Stage | Leaf | Engine path |
| --- | --- | --- |
| Ingress | `routine_turboquant_dequant4.hpp` | MTE2 → UB, int4 → FP16 (K) and FP32 (V) |
| GEMM | `routine_cube_matmul.hpp` | GM → L1(ZN) → L0A/L0B → L0C → UB |
| Recurrence | `routine_vector_online_softmax.hpp` | pure Vector, replicated 32-byte state, `Brcb` broadcast |
| Outflow | `routine_vector_normalize_store.hpp` | invoked once after the final page |

All four leaves are used unmodified, through their published entry points.

### Geometry and the one scoping decision

Q carries `kDim = 16` query rows so the Cube operand geometry matches the
verified 16×16 engine exactly. The online-softmax and outflow leaves are
per-query routines, so they are instantiated for **query row 0**. The Cube
still produces all 16 rows, and the runner checks every one of the
3 × 256 = 768 score entries, which is what verifies the ingress and GEMM path.
Extending the recurrence to all 16 rows would need 16 copies of the carried
state and 16× the vector work; it is not what an N = 3 boundary probe is for.

### Why the key page round-trips through GM

The Te copy surface has no UB → L1 transfer. The available tags are
`CopyGM2L1`, `CopyGM2UB`, `CopyL12L0A/B`, `CopyL12UB`, `CopyL12BT`,
`CopyL12FB`, `CopyL0C2UB/L1/GM` and `CopyUB2GM` — there is no `CopyUB2L1`.
Dequantisation is Vector work so it lands in UB, while the Cube B operand must
reach L1 in ZN. The only path the verified API surface supports is therefore

```
int4 GM → UB → dequant (Vector) → GM workspace → L1(ZN) → L0B → Mmad
```

which is also how a production paged-attention kernel stages a dequantised
page when the fractal conversion has to happen in the GM→L1 converter. The
workspace holds one slot per page, so there is no cross-core WAR on it.

### Host packing contract

Both layout choices live in the caller, which keeps the dequant leaf a pure
1-D vector routine:

- `k_packed` nibble index `(d * kKeys + key)` = quantised `K[key][d]`, so the
  staged FP16 page is ND `[d][key]`, i.e. `K^T`, and `S = Q · K^T`.
- `v_packed` nibble index `(ch * kKeys + key)` = quantised `V[key][ch]`, the
  channel-major order `update_accumulator` expects for `v_by_channel`.
- Two nibbles per byte, low nibble first, matching the dequant4 contract.
- Scales are compile-time (`kKeyScale = 0.5`, `kValueScale = 0.25`) so no
  scalar argument is read on device; the runner mirrors them exactly.

## Pipeline structure

Ingress runs **one page ahead** of the recurrence, so the Cube never waits on
the dequantisation of the page it is about to multiply:

```
ingress(page 0)                      # prologue
for page in 0..2:
    if page + 1 < 3: ingress(page + 1)
    wait scores(page); online_softmax(page); dump panel; release score panel
normalize_store()                    # once
```

The dequantised value page is double buffered (`kUbVFloat0` / `kUbVFloat1`)
because ingress of page t+1 writes it while the recurrence on page t is still
reading it. The key staging buffer stays single because each ingress drains it
through MTE3 before the next one runs, behind an `MTE3_V` edge.

### Event graph

Cross-core uses mode 4 (AIC ↔ AIV0), which 950PR supports in both directions
with flagId 0..10:

| Flag | Direction | Meaning | Sets | Waits |
| --- | --- | --- | ---: | ---: |
| 0 | AIV → AIC | key page staged in GM workspace | 3 | 3 |
| 1 | AIC → AIV | Fixpipe landed in the shared UB panel | 3 | 3 |
| 2 | AIV → AIC | UB score panel consumed, free to overwrite | 1 primed + 3 | 3 + 1 drained |

Intra-AIV HardEvents, 7 `(route, ID)` pairs over event IDs 0..2 (6 and 7 are
reserved):

| Route | ID | Sets | Waits | Purpose |
| --- | ---: | ---: | ---: | --- |
| `MTE2_V` | 0 | 4 | 4 | packed pages readable |
| `V_MTE2` | 0 | 1 primed + 3 | 3 + 1 drained | packed buffers free (WAR) |
| `V_MTE3` | 1 | 3 | 3 | FP16 key page ready to stage |
| `MTE3_V` | 1 | 1 primed + 3 | 3 + 1 drained | staging buffer drained (WAR) |
| `V_MTE3` | 2 | 4 | 4 | score panel / context ready to store |
| `MTE3_V` | 2 | 3 | 3 | diagnostic store drained (WAR) |
| `MTE3_S` | 2 | 1 | 1 | final store visible |

Every primed token is drained after the loop, so nothing leaks into the next
launch. `scripts/analyze_fused_trace.py` asserts all of this from the trace
rather than from the source.

## Measured results

```
PASS fused attention: 3 pages, 768/768 score entries bit-exact,
  max 2.000 ULP (worst operand context), 16/16 FP16 context within 1 half ULP,
  1152/1152 GM guard bytes intact
  final m=0.125 l=34.0713806  context[0]=1.85358536 context[15]=1.82630539
```

| Metric | Value |
| --- | --- |
| Device exit code | 0 |
| `invalid ldst addr` / `unrecognize ldst addr` / `su_ccu_mpu_err` | 0 |
| `[error]` / `[fatal]` simulator lines | 0 |
| Score panel | 768/768 bit-exact FP32 |
| Max error, `m` / `l` / FP32 context | 2.000 ULP (budget 8.0) |
| FP16 context | 16/16 within one half ULP |
| GM guard bands (`0xDEADBEEF`) | 1152/1152 bytes intact, 9 buffers |
| Replicated-state invariant | all 8 lanes of the `m` and `l` blocks identical |
| Model ticks | 9203 |
| Dispatch span | cycles 707..8876 (8169 cycles, 2463 instructions) |
| Completion span | cycles 711..8897 (8186 cycles, 2343 instructions) |

### Event audit and engine overlap

```
intra-core HardEvents: 23 sets / 23 waits across 7 (route, ID) pairs; IDs 0..5; zero residual
cross-core mode-4 flags: 10 sets / 10 waits, every sync id paired AIV<->AIC
engine busy cycles: MTE2 2073, Cube 2362, Vector 6793, MTE3 3891
  overlap Cube   + Vector:  2250 cycles (widest 5600..6111)
  overlap Cube   + MTE2  :   895 cycles (widest 1243..1678)
  overlap Cube   + MTE3  :   934 cycles (widest 5676..6111)
  overlap Vector + MTE2  :  2072 cycles (widest 1244..1797)
  overlap Vector + MTE3  :  3326 cycles (widest 6993..7573)
  overlap MTE2   + MTE3  :  1026 cycles (widest 2475..2999)
four-engine overlap: 438 cycles, widest CAModel cycles 3176..3527
```

MTE2, Cube, Vector and MTE3 are concurrently in flight for 438 cycles, with
all six pairwise overlaps non-empty. Vector is the critical engine at 6793
busy cycles of an 8169-cycle span.

### What the run-ahead bought

The same kernel was first built as a strict ping-pong (ingress → GEMM →
recurrence, no look-ahead, no double buffering) to establish numerics before
any concurrency was introduced:

| Variant | Model ticks | Dispatch span | Cube MMAD dispatch ticks |
| --- | ---: | ---: | --- |
| Ping-pong | 11509 | 10424 cycles | 3583, 6004, 8601 |
| One page look-ahead | 9203 | 8169 cycles | overlapped, see above |

A 20% tick reduction and a 21.6% shorter dispatch span, with **bit-identical
results** — the same 768/768 exact score entries and the same
`m`, `l` and context values. Numerics were established first and then
re-verified after the concurrency change, which is the sequencing the
[guidelines](../MICROROUTINE_DESIGN_GUIDELINES.md) §0 argue for: a barrier
removal can expose a latent hazard, so results must be re-checked, not just
the error count.

## Fixture design, and why it is built this way

The runner's fixture is chosen so the checks can be strict:

- Q is FP16 with values that are multiples of 1/64 in [-0.0625, 0.0625], and
  the dequantised key page is a multiple of 0.5 in [0, 7.5]. Every product and
  partial sum is then a multiple of 1/128 well inside FP32, so the reference
  score panel is **exact** and step 1 can demand bit equality rather than a
  tolerance. This is what makes the ingress and GEMM path verifiable at all.
- The three pages are numerically distinct, with **strictly increasing page
  maxima** (`0.0`, `0.0625`, `0.125`), so `alpha < 1` on every page after the
  first and the rescaling path actually runs.
- Per-page `p_sum` values are distinct (12.79, 12.14, 11.39) and all 16 keys
  contribute non-negligibly, so the softmax is not degenerate.
- All 16 context channels are distinct, so a channel or key permutation
  cannot pass.

The distinctness properties are deliberate. The earlier online-softmax WAR
hazard survived its original validation precisely because every tile in that
fixture shared the same maximum, which made a wrong-tile read invisible in
`m`. A fused fixture whose pages are interchangeable would be unable to detect
a run-ahead or double-buffering fault, which is the main thing this probe
exists to check.

## Precision note

Max 2.000 ULP on the FP32 context, against a budget of 8.0. This is better
than the 8 ULP seen in the isolated online-softmax probe for a structural
reason: the dequant4 leaf masks nibbles to unsigned 0..15, so V is
non-negative and the accumulator is a non-cancelling sum. The isolated probe
used signed V with up to 10.2× cancellation. Per
[guidelines](../MICROROUTINE_DESIGN_GUIDELINES.md) §5, a ULP budget belongs
against the largest intermediate for a cancelling reduction; here there is no
cancellation, so the final value is the right reference.

## What remains untested

- Silicon. CAModel is cycle-accurate but is not hardware; it does not validate
  timing closure, power, or multi-core scale-out.
- `N = 3` is a depth boundary, not sustained throughput. It exercises
  prologue, steady state and the single-page look-ahead, but the overlap
  figures are for this depth, not a steady-state interval.
- One AIC and one AIV (`blockDim = 1`). AIV1 is idle and no inter-AI-Core
  (mode 0) synchronization is exercised.
- The recurrence and outflow cover query row 0 only, as described above.
- Dequantisation has no zero point, so the quantised range is unsigned 0..15
  times a scale. A signed or per-group quantiser is a different leaf contract.
