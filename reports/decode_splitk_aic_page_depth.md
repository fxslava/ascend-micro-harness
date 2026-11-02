# Inter-page state carry in `turboquant_paged_decode_splitk_aic`

Target: CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599`. No NPU on this host
(`npu-smi` and `/dev/davinci*` both absent), so every device result here is
CAModel over a collapsed grid and nothing below is a hardware claim.

## 1. The reported failure, as measured

The task described a softmax-normalizer (`l`) defect in the intra-worker page
loop, said to reproduce at `pages=2, K=1`. Neither half holds.

| rung | geometry | result |
| --- | --- | ---: |
| `--context 32 --page-tokens 16 --splitk 1` | 2 pages, one worker | **PASS** (48507 ticks) |
| `--context 48 --page-tokens 16 --splitk 1` | 3 pages, one worker | **PASS** (69005 ticks) |
| `--context 64 --page-tokens 16 --splitk 1` | 4 pages, one worker | **FAIL**, on `acc` |
| `--context 64 --page-tokens 16 --splitk 4` | 4 pages, one per worker | **PASS** (27931 ticks) |
| `--context 64 --page-tokens 16 --splitk 2` | 4 pages, two per worker | **PASS** |

The four-page failure is on the accumulator, not the normalizer:

```text
AIC partial mismatch item=0 row=0 c=0 got=-1.15049517 ref=-1.14959846
```

`m` (bound `2e-6` relative) and `l` (bound `1e-4` relative) are both checked
before `acc` in `AicRig::Validate` and both passed. So the quantity the task
named as the failure is the one quantity that was still inside its bound.

The task's three audit hypotheses are refuted by the source, independently of
the runs:

* **State retention.** `run_value_core`'s `initialize` argument is `p == 0`, so
  `m_prev`, `l_prev` and `acc` are seeded on the first page only and carried
  afterwards. All three live at fixed arena offsets (`kUbMPrev`, `kUbLPrev`,
  `kUbAcc`); nothing in the page loop writes them.
* **The rescaling invariant.** `reduce_max_and_alpha` issues
  `Max(m_new, m_cur, tile_max_bc)`, `Sub(alpha_delta, m_cur, m_new)`,
  `Exp(alpha, alpha_delta)`, and `update_denominator` issues
  `Mul(l_scaled, l_cur, alpha)` then `Add(l_new, l_scaled, p_sum_bc)`. The
  subtraction is in the order that makes the exponent non-positive, and `m` is
  seeded at `-3.0e38` rather than `-inf` specifically so the first strip's
  `Exp` yields `0` instead of an `inf - inf` NaN.
* **Addressing.** The physical block-table lookup and the planar scale offsets
  agree with the FP64 oracle term for term at every page count; both were
  checked against `ReferenceAicPartial`'s own index arithmetic.

## 2. Gate 1 was incomplete, and completing it found a real defect

`AGENTS.md` routes pipeline work through `REFINEMENT_OF_COMBAT_KERNELS.md`,
whose first gate is the capacity-one token model. The existing model,
`tests/cpu/test_decode_ring_model.py`, covers **one pass** of the two-slot ring
on the fused AIV kernel and the Split-K reduction. It does not model this
operator at all: three asynchronous cores, with the ring primed and drained
once per page, and a full-page completion handshake between pages. The places
that exist only at a page boundary had never been modelled.

`tests/cpu/test_decode_page_loop_model.py` models them, and reports at every
`pages >= 2`:

```text
SET into a full place ('hard', 'AIC.M_MTE1', 0) (bound 1) -- hardware loses this token
```

### The defect

The Cube primes and drains both of its feedback cycles once per page, and for
each one the prime and the drain sit on **different queues**:

| place | `SetFlag` issued on | `WaitFlag` issued on |
| --- | --- | --- |
| `M_MTE1` | M | MTE1 |
| `FIX_M` | FIX | M |

At the page boundary `run_cube_core` drained only the Fixpipe (`FIX_S`). A
scalar drain stalls instruction *issue*, so that fence does hold back
everything after it — but it only proves FIX retired. Page `p + 1`'s
`SetFlag<M_MTE1>` executes on M, and nothing makes it wait for page `p`'s
`WaitFlag<M_MTE1>`, which is sitting on MTE1. Likewise page `p + 1`'s
`SetFlag<FIX_M>` executes on FIX while page `p`'s `WaitFlag<FIX_M>` is still
pending on M. A HardEvent is one bit, so the second SET is dropped, the
matching WAIT later finds nothing, and that block's Cube stops.

Aggregate SET/WAIT counts balance exactly — two primes and two drains per page
per cycle — which is why `scripts/analyze_aic_splitk_trace.py`'s census never
saw it. This is the failure mode `REFINEMENT_OF_COMBAT_KERNELS.md` §1 states in
one line ("A second SET before consumption loses a token even if aggregate
SET/WAIT counts match") and that only a per-place capacity-one model finds.

AIV0 and AIV1 already close the identical hole at the same boundary: AIV0
drains V and MTE2, AIV1 drains MTE3, V and MTE2 — in each case every queue that
core primes on. The Cube was the only one of the three that did not.

### The fix

`HardEventAic` (CANN 9.2.0-beta.2,
`x86_64-linux/ascendc/include/basic_api/impl/kernel_event.h`) has `FIX_S` and
`M_S` but **no `MTE1_S`**, so MTE1 cannot reach the scalar stream directly. It
is routed through `MTE1_M` into M, whose wait sits behind page `p`'s own
`WaitFlag<FIX_M>` pair, and M then drains to scalar:

```cpp
SetFlag<FIX_S>(kEvtPageDrain);   WaitFlag<FIX_S>(kEvtPageDrain);
SetFlag<MTE1_M>(kEvtPageDrain);  WaitFlag<MTE1_M>(kEvtPageDrain);
SetFlag<M_S>(kEvtPageDrain);     WaitFlag<M_S>(kEvtPageDrain);
CrossCoreSetFlag<4, PIPE_FIX>(kFlagPageDrainedV0);
CrossCoreSetFlag<4, PIPE_FIX>(kFlagPageDrainedV1);
```

`kEvtPageDrain` is 5, inside the 0..5 range the protocol allows, and distinct
from the ids `MTE1_M` already uses on this core (0, 1 for ring slots, 3 in the
prologue). The whole addition is inside `if (temporal)`, so the legacy
`camodel_paged_decode_attention` entry — which calls `run_cube_core` with
`pages = 1` and `temporal = false` — emits a bit-identical instruction stream
and its verified eight-strip ladder is untouched.

### Model coverage

With the fix encoded, the model is 1-safe across page boundaries on 11
geometries, carries the recurrence in order, and drains to an empty marking:

```text
pages=1 strips=1,2,4,8     pages=2 strips=1,2,4
pages=3 strips=1,2         pages=4 strips=1,2
```

Six negative controls are each rejected, including `aic_fix_only`, which is the
pre-fix kernel and reproduces exactly the double SET above — so this is a
regression test, not just a proof.

**Not covered by this gate:** eight strips over more than one page. Three
asynchronous cores at that size is tens of millions of states with no
partial-order reduction in this model; the ladder's `--context 256
--page-tokens 128` rung carries that geometry on CAModel instead. The model
prints these as `NOT COVERED` rather than dropping them silently.

## 3. The four-page accumulator deviation is a separate, numerical matter

The token defect above is a credit-loss hazard whose symptom is a stalled Cube,
not a 7.8e-4 relative error in one channel. The `pages=4` deviation needed its
own attribution, and the first thing to establish was whether FP32 arithmetic
explains it.

A host model (`reports/decode_splitk_aic_page_depth/acc_error_model.txt`)
re-executes the kernel's own recurrence in FP32, phase by phase and strip by
strip in the order `routine_vector_decode_recurrence.hpp` issues it, and
compares it against the same FP64 oracle the rig uses:

| geometry | worst `acc` deviation, FP32 recurrence vs FP64 oracle | fraction of the rig's tolerance |
| --- | ---: | ---: |
| 1 page (16 tokens) | 1.87e-07 | 0.000x |
| 4 pages (64 tokens) | 2.62e-07 | 0.000x |

So the arithmetic the kernel performs, done exactly in FP32, lands four orders
of magnitude inside the bound. FP32 accumulation over four pages does **not**
explain the device's 8.97e-4, and the deviation is not a tolerance that merely
grew too tight with depth.

Two further explanations had to be eliminated before the device could be
blamed, and both were:

* **The oracle.** The FP32 emulation reads the caches through the same index
  arithmetic as `ReferenceAicPartial`, so agreement between those two says
  nothing about whether either addresses the cache correctly — and four
  16-token pages is a geometry `test_aic_splitk_decode_probe` never covered
  (it covers three 16-token pages, and 8/32/128 128-token pages). The probe
  now also normalizes, rotates and gates the oracle's whole-window partial and
  compares it against `Fixture::expected`, which the fixture builds from its
  own independent loop over the same bytes. The two agree **exactly** — worst
  difference `0.000e+00` — at one, two, three and four 16-token pages and at
  one and two 128-token pages.
* **The device's `Exp`.** A hardware approximation with relative error around
  `2^-12` would enter `acc` amplified by the cancellation in a signed-nibble
  contraction and enter `l`, a sum of positive terms, unamplified, which fits
  the shape of the failure. It is ruled out by the device's own margins at
  shallow depth: at one page the device reaches `acc=0.00032x` of its bound,
  which is as accurate as the exact-FP32 host emulation. Its `Exp` is not
  contributing anything near `1e-4`.

The split bisection narrows where it comes from. At `pages=4`:

* `K=4`, one page per worker and no carry at all — **PASS**.
* `K=1`, one worker carrying all four pages — **FAIL**.

Each page is therefore computed correctly on its own; the deviation appears
only in the carried state.

### The §2 fix does not change it

The page-depth ladder was re-run on the fixed kernel. The four-page rung fails
with a **bit-identical** `got` and `ref`:

```text
AIC partial mismatch item=0 row=0 c=0 got=-1.15049517 ref=-1.14959846
    deviation=8.967165e-04 tolerance=8.580223e-04 (1.045x)
```

So the credit-loss defect of §2 and this deviation are **two independent
findings**, not one cause with two symptoms. §2 is closed; this is open.

### What the shape of it rules out

The measured margins (§4) are flat at `0.00032x`, `0.00032x`, `0.00033x` for
one, two and three pages and then `1.045x` at four. In absolute terms the worst
deviation goes from about `2.4e-07` to `8.97e-04` — a factor of roughly 3700
between three pages and four. That is a cliff, not accumulation: no drift
mechanism, FP32 or `Exp`, produces three flat rungs and then a 3700-fold jump.

What is left is therefore a discrete event at the third page boundary, and the
fact that `m` and `l` stay inside their bounds while `acc` does not narrows it
to the inputs `acc` has and `l` does not: the dequantized value page, the
per-token value scale, and the `alpha` rescaling of the 256-channel
accumulator. A single wrong value-side term of order `1e-3` fits every
observation — it would move `acc` by its own size and `l` not at all, since
`l` sums probabilities and never reads the value page.

**This is an open finding.** It is not diagnosed, and nothing in this report
should be read as fixing it. The next step is a verification-only export of the
per-page score archive on the Split-K path, which separates a wrong score
(key/Cube side) from a wrong value page or scale in one run: `RecurStrip`
already writes the archive into Unified Buffer on this path and simply has no
`out_scores` to publish it to.

## 4. Depth ladder, with margins

Margins are each quantity's worst deviation as a fraction of its own tolerance,
reported on success as well as failure. They were not previously reported at
all: `AicRig::Validate` threw on the first channel that exceeded its bound and
printed nothing otherwise, so a rung passing at 0.9x looked exactly like one
passing at 0.0003x, and a depth ladder could not show a trend. Adding them is
what makes the cliff below visible as a cliff.

All rows are the **fixed** kernel, `--page-tokens 16 --heads 1 --impl aic`,
CAModel:

| rung | pages per worker | result | `m` | `l` | `acc` |
| --- | ---: | --- | ---: | ---: | ---: |
| `--context 16 --splitk 1` | 1 | PASS | 0.00000x | 0.00064x | 0.00032x |
| `--context 32 --splitk 1` | 2 | PASS | 0.00000x | 0.00064x | 0.00032x |
| `--context 48 --splitk 1` | 3 | PASS | 0.00000x | 0.00096x | 0.00033x |
| `--context 64 --splitk 1` | 4 | **FAIL** | — | — | **1.045x** |
| `--context 64 --splitk 4` | 1 | PASS | 0.00000x | 0.00064x | 0.00032x |

The failing rung throws at the first channel past its bound, so its `m` and `l`
margins are not comparable to the others' and are left blank rather than
reported as zero.

Two readings worth keeping:

* `l`'s bound is `1e-4` relative and the device uses `0.00064x` of it — the
  check is roughly 1500x looser than the quantity it measures. It would not
  catch a `1e-5`-level corruption of the normalizer. That is worth retuning,
  but only with a device-measured floor, and the floor is what the margin
  column now provides.
* `acc`'s bound is written against `max|acc|`, which does not carry the
  cancellation in a signed-nibble contraction. The host model measures that
  cancellation reaching 2687x at `--context 256 --page-tokens 128`, so the
  bound is not expressed in the right quantity even though FP32 error is small
  enough that it has never mattered.

Neither of those is the cause of the four-page failure, and neither was changed
here; retuning a bound while a 3700-fold jump is unexplained would hide it.

## 5. Gate status

| gate | state |
| --- | --- |
| §1 token/queue model, single ring pass | PASS (`test_decode_ring_model.py`, 22654 states, 7 controls) |
| §1a page-loop model (new) | PASS, 11 geometries, 6 controls; 8-strip multi-page NOT COVERED |
| §2 compile-time memory checks | PASS (device build, `-Wall -Werror`, no warnings) |
| §3 isolated logical-page canaries, AIC path | **PENDING** — see below |
| §4 depth and wrap tests | PASS to 3 pages at 1 strip; **FAIL at 4** (§3, open) |
| §5 trace audit / census | **NOT RUN** — see below |
| physical NPU | UNAVAILABLE on this host |

**§5 was not run and is not claimed.** The ladder used here overwrote each
rung's `*.dump` set before the next rung, so no completed four-page dump set
survived for `scripts/analyze_aic_splitk_trace.py`. The committed runner
(`scripts/run_decode_splitk_aic_page_depth.sh`) preserves the deepest `K=1`
rung's dumps into `trace_n64_k1/` and has an `audit` mode for exactly this, but
the census in this report's evidence directory is absent, not passing. The
audit would also be the natural place to confirm the two new HardEvent routes
balance per place, which so far only the host model asserts.

**§4's wrap rung was not reached.** `--context 256 --page-tokens 128`, the only
geometry that exercises ring wrap and a page boundary together, is in the
runner's ladder but was not executed here; the page-loop model explicitly does
not cover eight strips over several pages either, so that combination is
currently covered by neither gate.

**§3 is not met and is not claimed.** The isolated head/middle/tail canary
protocol exists for the AIV path only
(`tests/camodel/test_camodel_decode_splitk_boundary.cpp`). The AIC Split-K entry has
no equivalent: the bench rig floods only `qrot` and `workspace` with a canary
and checks numerics against FP64, so a stray write by the AIC path into any
other buffer would not be detected. That is an open finding, not a conclusion.

## 6. Commands

```bash
# Host gates only, no CANN needed.
bash scripts/run_decode_splitk_aic_page_depth.sh models
```

```bash
# Build the simulator twin and run the whole page-depth ladder and trace audit.
bash scripts/run_decode_splitk_aic_page_depth.sh all
```

```bash
# The minimal rung the task asked for, directly.
bash scripts/run_decode_splitk_bench.sh sim --context 32 --page-tokens 16 --heads 1 --rows 1 --splitk 1 --single --impl aic
```

Each CAModel rung writes roughly 37 MB across 416 `*.dump` files and the
four-page rung takes about five minutes of model time; the ladder is slow for
that reason, not because anything hangs.
