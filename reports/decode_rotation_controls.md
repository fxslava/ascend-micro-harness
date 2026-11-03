# Ingress/egress rotation controls, and the AIV grid ceiling at 64 blocks

Two changes to the TurboQuant decode operators, and the verification baseline
`REFINEMENT_OF_COMBAT_KERNELS.md` requires of a pipeline refactor of them.

1. **Compile-time rotation controls.** `INGRESS_ROTATE` on the Cube path
   eliminates the 64-Mmad query rotation prologue when the query arrives
   rotated upstream; `EGRESS_UNROTATE` on both paths eliminates the inverse
   butterfly ($\Pi^T$) so a caller can absorb it downstream
   ($W_o' = \Pi^T W_o$, or a SwiGLU gate fused on the rotated vector).
2. **The AIV single-wave grid bound, 32 → 64.** `dav-3510` has 32 Cube cores
   and **64** vector subcores, and the only operator that reaches the in-launch
   barrier is `KERNEL_TYPE_AIV_ONLY`, where a block IS a vector subcore. The
   32-block bound was the Cube core count applied to the wrong engine. It cost
   half the machine and put $K=16$ at $H{=}4, B{=}1$ and $K=8$ at $B{=}2$ out
   of reach.

Target: CANN 9.2.0-beta.2, `dav-3510` / `Ascend950PR_9599`, BiSheng,
`-Wall -Werror`, inside `ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12`.

---

## 1. Read this first: what ran and what could not

**This host has no 950PR.** `/dev/davinci*` is absent and `npu-smi` is not
installed. Every one of the mission's five acceptance commands names a
**physical-device** binary (`./lib/test_npu_decode_*`,
`./bin/bench_turboquant_decode_splitk`). Those targets are **compile-only**
here, which is the standing position of this harness recorded in
`cmake/NpuTargets.cmake` and `reports/decode_splitk_validation.md` §5.

What executed is their **CAModel twins** on the `Ascend950PR_9599` simulator,
which this host does have. CAModel is the strongest evidence tier reachable
here; it is not silicon, and no number below should be read as throughput.

| gate | status | evidence |
| --- | --- | --- |
| §1 capacity-one token/queue model, both kernels | **PASS** | `scripts/run_reshape_wave_host_gate.sh` stage 8: 22,654 states — 1-safe, ordered ownership, liveness, finite drain; 7 negative controls rejected |
| §2 compile-time memory checks, `-Wall -Werror` | **PASS** | zero compiler warnings on 6 device targets, before and after |
| §2 UB/L1 arena, AKA3003 | **PASS** | no offset expression or `static_assert` changed; see §6 |
| §3 isolated head/middle/tail canaries, AIV, **both egress legs** | **PASS** | 6 cases, byte-exact, 1,297,024 canary bytes intact each |
| §3 isolated one-tile canaries, Cube, **all three legs** | **PASS** | 6 cases, block 16 isolated, each with its own 9-check trace audit |
| §4 depth and wrap, AIV, both egress legs | **PASS** | 4 cases (depth=2, wrap=4) |
| §5 trace audit + instruction census | **PASS** | ring audit per run; 9-check Cube audit per leg; 64 Mmads measured absent |
| 64-block wave, both egress legs | **PASS** | 6 launches at grid 64 on CAModel; 32-block control |
| host grid-bound and egress-oracle gate | **PASS** | `tests/cpu/test_splitk_grid_bound.cpp` |
| AIC prerotated ladder on CAModel | **FAIL at 4 pages** | both one-page rungs `ok`; `pages=4 K=1` fails on `l` — see §7a |
| …same rung on the **contract** leg (control) | **FAIL, identically** | the failure is pre-existing, not from this change; §7a |
| …with diagnostics, 3 pages | `m` **bit-exact**, `l` 2.23e-4 relative high vs a 1e-4 budget | a 2.2× tolerance miss, not wrong data; §7a |
| full AIV matrix (5 M × 4 K × 2 bases) | **NOT RUN** | 40 CAModel launches; narrowed to M=1 at K=4/8/16 |
| full AIC ladder and tail ladder on CAModel | **NOT RUN** | pre-existing status, unchanged by this work |
| any gate on silicon | **NOT RUN** | no 950PR on this host |
| `scripts/run_decode_pipeline_camodel.sh` | **CANNOT COMPLETE** | pre-existing; see §7 |

---

## 2. The grid bound

`include/turboquant_splitk_contract.hpp`:

```cpp
constexpr std::uint32_t kAivSubcoresPerCluster = 2;
constexpr std::uint32_t kAiCoreClusters = 32;
constexpr std::uint32_t kMaxAivBlocks = kAiCoreClusters * kAivSubcoresPerCluster;
```

`kMaxCoResidentBlocks` is renamed rather than retuned, because the old name
invited a MIX operator to borrow a number that is only right for an AIV-only
one. There is still exactly **one** definition of `GridFitsOneWave`, shared by
the device's early return and the host launcher's refusal — the §6.1 rule, and
the reason the host gate below is worth anything.

### Why 64 is the right number here

The barrier is `AscendC::SyncAll<true>()` — `isAIVOnly`, raising
`SYNC_AIV_ONLY_ALL`. Its participant set is the AIV subcores, and on an
AIV-only launch `GetBlockIdx()` indexes those subcores. The simulator for this
part enumerates 32 clusters of `cubecore0`, `veccore0`, `veccore1` — 96 cores,
**64 of them AIV** — which is what `reports/decode_ring_pipeline/summary.json`
means by `"idle_cores": 95` beside a single-block probe, and what every ring
audit in this change re-reports.

### The bound as a compile-time object

Four `static_assert`s in the contract header, so a future renumbering breaks a
build rather than a launch, plus an exhaustive host sweep
(`tests/cpu/test_splitk_grid_bound.cpp`):

```
ok  grid bound: 874 triples admitted, 7318 refused, 15 land exactly on the 64-block wave
```

The sweep covers every operand triple the ABI admits, checks each against the
predicate's definition restated independently, and requires that a grid **one
block past the wave** is refused for every split count — a bound applied to the
wrong axis would let one of those through.

### What is NOT claimed

That a 64-block `SyncAll<true>` completes **on silicon**. It completes on
CAModel, six times, at two different factorizations of 64 (§5). The simulator
models the subcore count correctly, which is why this is evidence rather than
nothing, but a barrier is exactly the kind of thing a model and a chip can
disagree about. The silicon rung is owed.

---

## 3. The rotation controls

### Cube, `kernels/ops/turboquant/turboquant_paged_decode_attention.cpp`

Three `__global__` symbols over one body, forming a **chain** in which each
neighbour differs in exactly one control:

| symbol | `INGRESS_ROTATE` | `EGRESS_UNROTATE` |
| --- | --- | --- |
| `camodel_paged_decode_attention` | 1 | 1 |
| `camodel_paged_decode_attention_prerotated` | 0 | 1 |
| `camodel_paged_decode_attention_rotated_basis` | 0 | 0 |

and two over the production Split-K body, which has no egress:

| symbol | `INGRESS_ROTATE` |
| --- | --- |
| `turboquant_paged_decode_splitk_aic` | 1 |
| `turboquant_paged_decode_splitk_prerotated_aic` | 0 |

`<true, false>` is deliberately not emitted: a pipeline that has not rotated
its query upstream is not one that has absorbed $\Pi^T$ downstream, and a
symbol nothing launches is a symbol nothing checks. The chain still isolates
both variables — compare rows 1 and 2 for the ingress, rows 2 and 3 for the
egress.

On `INGRESS_ROTATE = 0` the kernel stages `gm_q` straight into `kL1QRot` with
the same ND → Nz page conversion that staged `kL1Q`, over one `MTE2_MTE1` pair,
and loads L0A from it. No `pi_table` read, no `rotate_query_stage_l0`, no
64-Mmad `rotate_query_compute`, no Fixpipe, no `out_qrot` write, no GM round
trip. No edge is dropped that the rotating leg needed for a hazard this leg
still has: there is no Mmad to wait for and no Fixpipe to drain.

### Vector, `turboquant_single_token_decode_splitk_aiv.cpp`

| symbol | `EGRESS_UNROTATE` |
| --- | --- |
| `turboquant_single_token_decode_splitk_aiv` | 1 |
| `turboquant_single_token_decode_splitk_rotated_aiv` | 0 |
| `turboquant_splitk_boundary_aiv` | 1 |
| `turboquant_splitk_boundary_rotated_aiv` | 0 |

The boundary probe is templated too, because the §3 canaries are the gate the
bypass has to pass and a probe that can only run the rotating leg cannot be it.

All nine device symbols verified present in the built libraries with `nm`.

### Two things about the bypass that are easy to get wrong

**The 1/16.** $\Pi = (H_8 \otimes I_{32})\,\mathrm{diag}(\pi)\,(I_8 \otimes
H_{32})/16$, and the cache holds $\Pi V$ with that $1/16$ already inside the
cached values. So $\mathrm{Acc}/l$ **is** $\Pi \cdot \mathrm{ctx}$, and scaling
it again would emit $\Pi\,\mathrm{ctx}/16$ — a plausible number 24 dB low. The
Cube leaf folds $\Pi$'s normalisation into `normalise()`'s input, so the bypass
leg must not apply it; the Vector leaf passes `kRotOrthoScale` to the butterfly,
which simply is not called.

**The ping-pong parity.** Eight butterfly stages starting in `buffer_a` leave
the result in `buffer_b`, and `gate_and_cast` reads `buffer_b`. With no stages
to run, the normalisation lands in `buffer_b` **directly**, so the gate chain is
the same four instructions reading the same buffer on both legs and no
operation becomes in-place. That shape is load-bearing: the first revision of
that routine computed the gate with three in-place vector operations and
matched none of nine candidate egress formulations on the device.

### Why these are symbols and not a runtime flag

Both bypasses accept the wrong operand without faulting and return a plausible
wrong answer — the same failure class as reusing the ingress stride order in
the egress, which this operator's own egress comment records. A caller has to
name the basis at the call site.

### `routine_vector_egress_rotate.hpp` was extended additively

That leaf is described elsewhere in this harness as frozen. The change adds a
defaulted template parameter and one `if constexpr`; `vector_egress_query<true>`
compiles to the instruction stream it always did, and every pre-existing caller
is textually unchanged.

---

## 3a. Three deliberate deviations from the specification

**1. The template defaults are `INGRESS_ROTATE = true` on `run_cube_core` and
`EGRESS_UNROTATE = true` everywhere, and the bypasses are named symbols.** The
specification asks for `template <bool INGRESS_ROTATE = false, bool
EGRESS_UNROTATE = true>`. Taking `false` as the ingress default would
re-specify the existing operator: `camodel_paged_decode_attention` and
`turboquant_paged_decode_splitk_aic` would stop rotating, every caller passing
a raw query would get a plausible wrong answer, `out_qrot` would stop being a
published output, and the whole verified ladder behind those two symbols
(including the bit-exact `Q_rot` gate the AIC rig performs) would be measuring
something else. The default therefore preserves the contract, and the new
behaviour is reached by naming `..._prerotated` / `..._rotated_basis` /
`..._rotated_aiv`. Everything the specification asks for is dispatchable; what
changed is which spelling is the quiet one.

**2. `GridFitsOneWave` is not duplicated into the kernel.** The specification
shows `kMaxAivBlocks` and an `__aicore__ GridFitsOneWave` inside
`turboquant_single_token_decode_splitk_aiv.cpp`. That translation unit already
contains the HOST launcher as well as the device entry, so a second definition
there would put the device guard and the host refusal on two different
functions in one file — the precise failure
`REFINEMENT_OF_COMBAT_KERNELS.md` §6.1 exists to forbid ("it is only worth
anything if the kernel and the host gate share the declaration rather than each
carrying a copy"). The constant is renamed to `kMaxAivBlocks` and raised to 64
in the shared contract header, where both already read it, and
`tests/cpu/test_splitk_grid_bound.cpp` compiles that same declaration from a
host translation unit.

**3. The arenas were not re-laid-out to reclaim the bypassed buffers.** See
§6: moving a slot would make a difference between two legs mean something
other than the rotation.

---

## 4. §3 and §4 — isolated boundaries, then depth and wrap

`bash scripts/run_decode_rotation_camodel.sh boundary`, 10/10 cases:

```
PASS Split-K boundary page=0 strip=0 count=1 rows=1 basis=model:    byte-exact context (256/256 lanes distinguish the two bases); ASSERT_CANARY_INTACT 1297024 bytes
PASS Split-K boundary page=4 strip=4 count=1 rows=1 basis=model:    ...
PASS Split-K boundary page=7 strip=7 count=1 rows=1 basis=model:    ...
PASS Split-K boundary page=0 strip=0 count=2 rows=1 basis=model:    ...
PASS Split-K boundary page=4 strip=4 count=4 rows=1 basis=model:    ...
PASS Split-K boundary page=0 strip=0 count=1 rows=1 basis=rotated:  ...
PASS Split-K boundary page=4 strip=4 count=1 rows=1 basis=rotated:  ...
PASS Split-K boundary page=7 strip=7 count=1 rows=1 basis=rotated:  ...
PASS Split-K boundary page=0 strip=0 count=2 rows=1 basis=rotated:  ...
PASS Split-K boundary page=4 strip=4 count=4 rows=1 basis=rotated:  ...
```

One tile of the operator's real 256-channel × 16-token geometry, nonzero
logical pages throughout and a non-identity block table. Every GM payload
flooded with `0x7FC0BEEF` first; the context compared **byte-exact** against
the leg's own oracle line, not against a tolerance — the fixture's scores are
all exactly 3, $l$ is 16 per strip and the values are small integers, so the
quotient is dyadic. Every unassigned byte of every buffer, and both exterior
guards, come back canary.

`256/256 lanes distinguish the two bases` is a gate of its own: if the two
oracle lines coincided, a kernel that ignored `EGRESS_UNROTATE` entirely would
pass whichever probe it ran.

Depth = 2 and wrap = 4 traverse both ring slots twice, so the buffer-reuse back
edges carry real contention rather than being merely primed and drained.

### The Cube legs, 6/6 cases

`bash scripts/run_decode_rotation_camodel.sh cube`, 6/6, one tile of one head
with block 16 isolated, at strip 0 and strip 7. Each case also runs
`scripts/analyze_paged_decode_trace.py` over its own traces, and the Mmad and
Fixpipe budgets come out exactly as the legs predict:

| leg | budget line from that case's `audit.txt` |
| --- | --- |
| contract | `72 Mmads, 2 Fixpipes, 16 UB->L1 bursts` |
| prerotated | `8 Mmads, 1 Fixpipes, 16 UB->L1 bursts` |
| rotated-basis | `8 Mmads, 1 Fixpipes, 16 UB->L1 bursts` |


| leg | scores | m, l | acc | pre-gate row | context |
| --- | --- | --- | --- | --- | --- |
| contract | 0/256 wrong | ≤ 3 ULP | ≤ 5 ULP | 5.960e-08 / budget 3.172e-04 | 7.629e-06 / budget 3.708e-03 |
| prerotated | 0/256 wrong | ≤ 3 ULP | ≤ 5 ULP | **5.960e-08** / 3.172e-04 | **7.629e-06** / 3.708e-03 |
| rotated-basis | 0/256 wrong | ≤ 3 ULP | ≤ 5 ULP | 2.235e-08 / 7.426e-05 | 3.052e-05 / 8.660e-04 |

The prerotated leg reproduces the contract leg's figures **identically**, to
every digit, including `largest |rot| 0.324663`. That is the point of the
design: the harness hands it `GoldenQRot` in bf16, which is bit for bit the A
operand the contract leg's own prologue produces, so the oracle does not
change and the comparison is the ingress and nothing else. The rotated-basis
leg's magnitudes differ (`largest |rot| 0.075944`, `largest |ctx| 0.055420`)
because it really is a different tensor.

**Both bypass legs run with `pi_table` flooded** rather than filled
(`0x7FC0BEEF` in the Cube canary probe, `0xAA` bytes in the AIC split-K rig,
which is that harness's own canary), and the rotated-basis leg also with the
three Gather tables and the sign diagonal flooded. A prologue or a butterfly that still consumed them would
build its operand out of a quiet NaN; no score panel could be bit-exact. That
is the isolation claim stated as an input rather than inferred from a trace.
`out_qrot` comes back untouched on both — 0 of 4096 words written — and the
prologue is its only writer. Isolation: 325,376/325,376 inactive output words
and 1,792/1,792 unprocessed archive words still canary, on every leg.

---

## 5. The 64-block wave

`bash scripts/run_decode_rotation_camodel.sh grid`, 3/3 cases, 6 launches:

```
ok  M=1  K=8  model    2 requests, grid  64 (full wave),  8 reducers, slowest export 10747 ticks
ok  M=1  K=8  rotated  2 requests, grid  64 (full wave),  8 reducers, slowest export 10845 ticks
ok  M=1  K=16 model    1 requests, grid  64 (full wave),  4 reducers, slowest export 10796 ticks
ok  M=1  K=16 rotated  1 requests, grid  64 (full wave),  4 reducers, slowest export 10734 ticks
ok  M=1  K=4  model    2 requests, grid  32,              8 reducers, slowest export 10842 ticks
ok  M=1  K=4  rotated  2 requests, grid  32,              8 reducers, slowest export 10780 ticks
```

Both grids the raised bound unlocks — $2{\times}4{\times}8$ and
$1{\times}4{\times}16$ — land exactly **on** the wave, on both egress legs. The
32-block rung is the control: if the full-wave rungs had failed and it passed,
the wave is the difference.

Per launch the rig requires: every worker published its own partition and no
other; every live lane of every record finite; exported $m$/$l$ replicated
across all eight lanes (a torn burst is detectable); exactly
`requests * heads` emitters, and they are the first `requests * heads` blocks;
the census drained; and the gated BF16 context against the FP64 full-context
oracle. At $K=16$ over four pages, 48 of the 64 blocks take the empty-range
neutral-element path at once — reached by geometry, not by a special case.

The barrier is what this rung is for: a `SyncAll` that did not span both
subcores of a cluster hangs rather than returning a wrong answer, so a launch
that **returns** with a correct context is the signal.

### The benchmark

`bench_decode_splitk_sim --impl aiv --splitk 16 --requests 1 --heads 4` →
grid 64, launched and `PASS: CPU references matched. Kernel launches: AIV=1`.
Before this change the host gate threw on that command. Run at
`--page-tokens 16 --context 48 --single` because CAModel cannot afford the
mission's `--context 4096`; the rejection it was meant to demonstrate is a
function of `requests * heads * splits` alone, which is 64 either way.

---

## 6. §2 — the arena, and what was deliberately not moved

**Not one UB or L1 offset expression, `NextSlot` chain entry, or
`static_assert` changed in either kernel.** Mechanically checkable:

```bash
git diff -U0 kernels/ops/turboquant/turboquant_single_token_decode_splitk_aiv.cpp \
  | grep -E '^[+-]' | grep -E 'kUb|NextSlot|static_assert|kArenaEnd'    # empty
git diff -U0 kernels/ops/turboquant/turboquant_paged_decode_attention.cpp \
  | grep -E '^[+-]' | grep -E 'k(Ub|L1)|NextSlot|static_assert'          # empty
```

So the 32-byte operand bases, the 32-byte ring bank skew and differing
`floor(offset/32) mod 16` banks (AKA3003), the private 256-byte scale landings,
the register over-read guards and `kArenaEnd + 256 <= kUbWindow <= 192 KiB` are
the **pre-existing** assertions, and they are namespace-scope, so they compile
in both instantiations of each kernel. `kUbRotated`, `kUbSign`, `kL1PiTable`
and `kL1Q` simply go unread on the legs that do not need them.

That is a choice, not an oversight. Reclaiming those 8 KiB of a 192 KiB window
on the Vector side, or those two L1 panels on the Cube side, would move
`kL1KeyRing` and every slot behind it — and a difference between the legs would
then no longer be the rotation alone. There are no dynamic allocations to
audit: every offset is a compile-time constant, which is the operators'
existing design.

Cube UB occupancy, `python3 scripts/decode_ub_budget.py --rows 16`:

```
 rows  B x (1+k)  fractals       probe   production  192KiB  256KiB
   16       2.67         1    151.12KiB     126.84KiB     yes     yes
```

---

## 7. §5 — the trace audit and the instruction census

### A bypass that agrees numerically has not been shown to BE a bypass

An `if constexpr` compiled the wrong way round, or a leaf that still runs and
happens to be the identity on the fixture, produces the same numbers. So
`scripts/audit_decode_rotation_census.py` reads each leg's own
`core*.instr_popped_log.dump` and turns the two claims into counts.

**Cube ingress**, `core16.cubecore0`, one strip:

| | contract | prerotated |
| --- | --- | --- |
| instructions | 480 | 212 |
| `MMAD` | **72** | **8** |
| `LOAD_2Dv2` | 26 | 2 |
| per-engine | SCALAR 294, CUBE 84, MTE2 39, MTE1 35, FIXP 11 | SCALAR 159, CUBE 17, MTE2 4, MTE1 8, FIXP 7 |

72 − 8 = **exactly 64 Mmads removed** — eight source groups × eight output
groups — leaving the one strip's eight accumulating score Mmads. 24 of 26
L1 → L0 fractal loads gone (16 resident rotation blocks + 8 Q group fractals).
The script gates it as an equality, `--expect-delta MMAD:prerotated:contract:64`.

**AIV egress**, `core0.veccore0`, one strip: 6,239 → **6,045** popped
instructions and 105 → 102 distinct mnemonics. Gated as
`--expect-fewer rotated:model`.

Both checks were negative-controlled: `--expect-delta …:63` and an inverted
`--expect-fewer contract:prerotated` both report `FAIL` and exit nonzero on the
same traces.

### The ring census, and the one event that changed

`scripts/analyze_decode_ring_trace.py` ran on all ten boundary cases. At wrap
(4 strips), `core0.veccore0`:

| route / id | model | rotated |
| --- | --- | --- |
| `(MTE2, VEC, 0)` | 4 | **3** |
| `(MTE2, VEC, 1)` | 2 | 2 |
| `(VEC, MTE2, 0..3)` | 3, 3, 3, 3 | 3, 3, 3, 3 |
| `(VEC, MTE3, 1)`, `(VEC, MTE3, 2)` | 1, 1 | 1, 1 |
| `residual_tokens` | 0 | 0 |
| MTE2/Vector intersections | 609 | 360 |

Every **feedback** route is identical, and matches §5's rule
$1 + \lfloor (N{+}1{-}s)/2 \rfloor$ = 3 for both slots at $N=4$. The single
difference is one forward `MTE2_V(0)` announcement: the sign-table ingress,
which the bypass leg does not fetch. It is a prologue pair consumed
immediately, not part of any reuse cycle. `residual_tokens: 0` on both — every
hardware flag drained. Both show real MTE2/Vector overlap, so
`--require-overlap` holds on the wrap rung of each leg.

Zero `SIMT_ATOM` instructions on all 64 vector cores of every run — the
lock-free property stated as a fact about the trace.

### The Cube's 9-check audit, all three legs

`scripts/analyze_paged_decode_trace.py` now takes `--no-ingress-rotation`,
which does **not relax** check 0 or check 5 but changes what they ask for: the
Cube's Mmad floor becomes the score GEMM's own count, and the Mmad and Fixpipe
budgets stay **equalities** with the prologue's terms at zero. A leg that still
ran the prologue lands exactly 64 above and fails. All three legs:
`OK paged decode trace audit: 9 checks passed`, with the prerotated and
rotated-basis legs reporting `8 Mmad(s) (score GEMM only, no prologue)`,
`budget: 8 Mmads, 1 Fixpipes, 16 UB->L1 bursts`, intra-core events balanced and
set-before-wait on all routes, cross-core flags paired on 5 absolute ids, the
L1 key page staged in 16 bursts and read exactly once after its last write, and
the quorum of one flag from each Vector core.

The rotated-basis leg's AIV0 Vector in-flight time is 78,732 ticks against
107,666 on the prerotated leg — the butterfly's eight stages over sixteen
queries. **Simulator ticks are not silicon throughput.**

### Two open findings, neither introduced here

1. **`scripts/run_decode_pipeline_camodel.sh` cannot complete on the current
   kernel.** Its `splitk` legs call `scripts/analyze_splitk_decode_trace.py
   --assert-batched`, which keys off the three atomic instruction PCs of the
   test-and-set protocol the workspace-partition rewrite removed. On a correct
   run it now raises `ValueError: no completed lock intervals`. The evidence in
   `reports/decode_ring_pipeline/` predates that rewrite — its
   `splitk_head/atomic.json` records real lock intervals. `AGENTS.md` names
   that script as the reproducible baseline, so this blocks it; the §3/§4/§5
   gates it performs are implemented in
   `scripts/run_decode_rotation_camodel.sh` instead, and the atomic auditor is
   replaced there by the claim the rewrite actually makes — zero atomics.
2. **`reports/decode_splitk_validation.md` is stale.** It describes the
   `AtomicHeadSlot` lock protocol and its §5 matrix predates both the workspace
   rewrite and this change.

---

## 7a. An open failure on the AIC split-K ladder, at four pages

`turboquant_paged_decode_splitk_prerotated_aic` is exercised by a narrowed
ladder (`test_decode_splitk_aic_sim --section prerotated --pages 1,4
--splitk 1,4`): the query operand is the rotated query, `pi_table` is flooded,
`qrot` must come back untouched, and every partial $(m, l, \mathrm{Acc})$ is
graded against the **same** FP64 oracle the rotating leg uses.

```
ok  prerotated pages=1  begin=0 count=1  K=1   pi_table flooded, qrot untouched
ok  prerotated pages=1  begin=0 count=1  K=4   pi_table flooded, qrot untouched
FAIL AIC split-K decode validation: AIC partial denominator differs from CPU reference
```

The two one-page rungs pass. The next rung — `pages=4 begin=0 count=4 K=1` —
fails on $l$, the online-softmax denominator, against
`ReferenceAicPartial`'s FP64 value at a 1e-4 relative budget.

**Attributed: it is NOT this change.** The control has since run — the same
rung on the **contract** leg, `test_decode_splitk_aic_sim --section ladder
--pages 4 --splitk 1`, which shares the per-page loop and differs only in the
prologue:

```
FAIL AIC split-K decode validation: AIC partial denominator differs from CPU reference
```

The unmodified, shipped `turboquant_paged_decode_splitk_aic` fails the same
rung, on the same quantity, against the same oracle at the same budget. The
rotation controls are clear of it. What this change did was execute that
ladder for the first time.

The supporting facts:

- the one-page rungs passing means the query basis, the prologue bypass, the
  `pi_table` flood and the `qrot`-untouched claim are all sound on this leg:
  if the prologue had run, or `f.q` were the wrong basis, $l$ would be wrong
  at one page too;
- page count is the only variable between the passing and failing rungs, and
  **nothing in this change is inside the per-page loop.** The prologue is
  before it, identically on both legs, and the `git diff` touches no event,
  no offset and no arithmetic within it;
- the AIC ladder has **never been executed on CAModel before** — the matrix in
  `reports/decode_splitk_validation.md` §5 records every rung as NOT RUN, and
  that report predates the workspace-partition rewrite as well. So this is the
  first run of the multi-page Cube split-K path on the model, on either leg.

So the finding stands as: **a pre-existing defect, or a too-tight oracle
budget, in the multi-page Cube Split-K path** — one page passes, more than one
does not, on both legs. It is open, and it is NOT closed by this change.

### The numbers

`AicRig::Validate` now prints `m`/`l`/err/budget before throwing — a bare
`require` on a tolerance says only that something is out of budget, and on a
24-strip online softmax that cannot tell accumulated rounding from a stale
panel. Rebuilt with that diagnostic, the contract leg at **three** pages (the
cheapest multi-page rung):

```
AIC partial state item=0 row=0 pages=3:
  m got 0.337402344  ref 0.337402344  (err 0.000e+00  budget 2.000e-06)
  l got 275.68634    ref 275.624844   (err 6.150e-02  budget 2.756e-02)
```

So:

- **$m$ is bit-exact.** That is the load-bearing part. The maximum is taken
  over the same scores that feed $l$, so a wrong page, a wrong window, a
  stale L1 panel or a mis-carried state across the page boundary would have to
  move $m$ as well, and it does not move at all. Combined with the score
  panels being checked **bit-exact** on every rung, the data the kernel read
  is the data it was supposed to read.
- **$l$ is high by 2.23e-4 relative** (6.150e-02 on 275.6), against a 1e-4
  relative budget — a **2.2× miss on a tolerance**, not a wrong answer, and
  biased consistently high rather than scattered.
- The miss is **superlinear in page count**: one page passes, where the same
  relative error would have been well inside the budget; three do not.

What that does *not* settle is whether the right fix is a depth-aware budget
on `AicRig`'s $l$ check or a real defect in the cross-page carry of the
denominator. A naive error model argues against pure rounding — 384 $O(1)$
terms, one documented ULP per `Exp`, a handful of rescales, lands nearer
$10^{-6}$ relative than $10^{-4}$ — but a consistent positive bias that grows
faster than the token count is also the signature of a rescale chain, and
neither story is proved by one row of one item. The `require` throws on the
first failure, so the distribution over the other 15 rows and the other items
is not even known yet.

**Recommended next step**, and it is not part of this change: collect all
rows rather than the first, at pages 1, 3, 4 and 8, and plot the relative
error against strip count. If it is flat the budget is wrong; if it grows the
carry is.

The Vector-path work and the grid bound are unaffected either way: they share
no code with the Cube Split-K entry. The three
`camodel_paged_decode_attention` legs are also unaffected — that entry calls
`run_cube_core` with `pages = 1` and never enters the loop in question, which
is consistent with all six of its cases passing in §4.

**A second defect, in this change's own harness, is fixed.** The run above
exited **0** despite printing `FAIL`: `scripts/run_decode_rotation_camodel.sh`
wrapped each case in a bare `( ... )` subshell, and `set -e` does not reliably
carry a failing subshell out of a function called at top level. Every case is
now `if ! ( ... ); then exit 1; fi`, verified against both legs of a minimal
reproduction, and the per-phase case tally (§8) is the second line of defence.
The two one-page `ok` lines above are still good; the clean exit was not.

---

## 8. Reproducing

All inside the container, with the repository copied to an ext4 path — never
built out of a drvfs mount, because the page cache can hand BiSheng a stale
copy of a source a Windows-side editor has just rewritten.

```bash
docker run -d --name tq_rot_dev -v "$HOME/work:/work" -w /work \
  ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12 tail -f /dev/null
```

```bash
docker exec tq_rot_dev bash /work/tq-src/scripts/run_decode_rotation_container.sh host
```

```bash
docker exec tq_rot_dev bash -c 'ROT_BUILD_DIR=/work/tq-b bash /work/tq-src/scripts/run_decode_rotation_container.sh strict'
```

```bash
docker exec tq_rot_dev bash -c 'ROT_BUILD_DIR=/work/tq-sim bash /work/tq-src/scripts/run_decode_rotation_container.sh sim'
```

```bash
docker exec tq_rot_dev bash -c 'ROT_BUILD_DIR=/work/tq-sim bash /work/tq-src/scripts/run_decode_rotation_camodel.sh'
```

The last command runs the boundary, cube, aic, grid and census phases in order
and tallies them; each phase can be run alone by name. A phase run truncates
`summary.txt`, so a single-phase run's tally covers that phase only.

And the whole host gate, including the new stage 7:

```bash
docker exec tq_rot_dev bash -c 'cd /work/tq-src && bash scripts/run_reshape_wave_host_gate.sh'
```

The five acceptance commands, for a host that has a 950PR — **none of them ran
here**:

```bash
./lib/test_npu_decode_single_token_unaligned
```

```bash
./lib/test_npu_decode_splitk_aiv
```

```bash
./lib/test_npu_decode_splitk_aic
```

```bash
./bin/bench_turboquant_decode_splitk --impl aiv --splitk 16 --requests 1 --heads 4 --context 4096 --single
```

---

## 9. Files

New:

| file | role |
| --- | --- |
| `scripts/run_decode_rotation_container.sh` | configure + strict/sim build + host gates, inside the CANN container |
| `scripts/run_decode_rotation_camodel.sh` | the CAModel gates: boundary, cube, aic, grid, census |
| `scripts/audit_decode_rotation_census.py` | per-leg instruction census with `--expect-fewer` / `--expect-absent` / `--expect-delta` |
| `tests/cpu/test_splitk_grid_bound.cpp` | host gate: the shared grid bound, and the two egress oracles tied together through the dense rotation |

Changed:

| file | change |
| --- | --- |
| `include/turboquant_splitk_contract.hpp` | `kMaxAivBlocks = 64`, the cluster geometry it is derived from, four `static_assert`s |
| `kernels/ops/turboquant/turboquant_paged_decode_attention.cpp` | `run_cube_core<INGRESS_ROTATE>`, `run_value_core<EGRESS_UNROTATE>`, two bodies, five device symbols, five launchers over two shared gate functions |
| `kernels/ops/turboquant/turboquant_single_token_decode_splitk_aiv.cpp` | `Egress<EGRESS_UNROTATE>` threaded through `ExportPartial`/`ReduceHead`/`ProcessItem`, two bodies, four device symbols, four launchers over one shared gate function |
| `routines/routine_vector_egress_rotate.hpp` | `vector_egress_query<UNROTATE>` and `normalise<UNROTATE>`, additively |
| `tests/common/splitk_decode_fixture.hpp` | `expected_rotated`, the FP64 rotated-basis oracle line |
| `tests/npu/turboquant_splitk_rigs.hpp` | `Basis` on `AivRig`, `Ingress` on `AicRig`; each picks the launcher and the oracle from one field |
| `tests/npu/validation/test_turboquant_decode_splitk_aiv_validation.cpp` | K=16 rung, both egress bases, `--rows/--splitk/--basis` narrowing |
| `tests/npu/validation/test_turboquant_decode_splitk_aic_validation.cpp` | prerotated ingress section, `--section/--pages/--splitk` narrowing |
| `tests/camodel/test_camodel_decode_splitk_boundary.cpp` | `--rotated`, the rotated-basis oracle line, the two-bases separation gate |
| `tests/camodel/test_camodel_paged_decode_attention.cpp` | `--leg`, the prerotated query operand, flooded `pi_table`/Gather tables, per-leg `out_qrot` and egress expectations |
| `scripts/analyze_paged_decode_trace.py` | `--no-ingress-rotation`: checks 0 and 5 retargeted, not relaxed |
| `scripts/run_reshape_wave_host_gate.sh` | stage 7, the grid bound and egress oracles |
| `cmake/SimTargets.cmake` | `test_splitk_grid_bound` |
