# MIX_AIC_1_2 inter-wave protocol for reshape-and-cache

Standardises the inter-wave behaviour of `turboquant_reshape_and_cache_a8` and
`turboquant_reshape_and_cache_bf16`, removes every host-side launch window, and
extends the wave ladder to single monolithic dispatches of 1, 32, 33, 48, 64
and 128 strips. Target: CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599`.

The canonical mode-4 channel policy is the **recycled** bilateral two-slot
ring. That is a reversal of this report's first revision, which shipped
single-shot channels because two CAModel traces appeared to refute recycling;
section 3 records the reversal and the evidence behind it.

**Read the evidence section before quoting any of this as validated.** This
workstation has no CANN 9.2 toolkit, no `Ascend950PR_9599` simulator and no
NPU. Device compilation, the CAModel boundary canaries, the wave ladder and
the trace audit are **not run here** and are listed as unavailable, per
`AGENTS.md`. The silicon and CAModel results in section 3 were produced
elsewhere and are reported, not reproduced here — section 2.5 says exactly
what that means. What *is* established here is the host-side half of
`REFINEMENT_OF_COMBAT_KERNELS.md` sections 1, 2 and the new section 6.

---

## 1. What changed

### 1.1 The protocol, in one shared header

`kernels/ops/turboquant/turboquant_wave_protocol.hpp` is new. It is pure
C++17 with no CCE dependency, both kernels include it instead of carrying
their own copies, and a host translation unit compiles it, so the arithmetic
the device build uses is the arithmetic the host gate checks.

It carries the strip boundary (`TotalStrips`, `BlockOutOfRange`), the AKA3003
arena algebra (`kBlock`, `kBankSkewBytes`, `kReadPadBytes`, `kUbWindow`,
`RoundUpBlock`, `NextSlot`, `SlotStrideSkewed`) that both kernels previously
duplicated, and both mode-4 channel policies.

### 1.2 Rule 1 — boundary and guard isolation

Both kernels now take an early return before `AscendC::InitSocState()`:

```cpp
const uint32_t block = GetBlockIdx() / GetSubBlockNum();
const uint32_t total_strips = TotalStrips(numTokens, tokensPerCore);
if (block >= total_strips) return;
AscendC::InitSocState();
```

Before SOC state, before any tensor, before a feedback cycle is primed, before
any GM address is formed. All three roles derive `block` from the same two
registers and the same two host scalars, so a guarded group returns together
and announces nothing its peers would wait for. `TotalStrips` returns 0 for a
zero `tokensPerCore`, so that guards every block instead of faulting the
scalar unit. The divide happens once and the sparse-sampling predicate reuses
it rather than recomputing it.

### 1.3 Rule 2 — two channel policies behind a non-type template parameter

```cpp
template <bool Reuse> using Flags =
    turboquant_wave::ChannelPolicy<Reuse, kChunks, kLandDepth>;

extern "C" __global__ void turboquant_reshape_and_cache_a8(...)
{ KernelBody<true>(...); }                   // recycled ring, the default
extern "C" __global__ void turboquant_reshape_and_cache_a8_mode4_single_shot(...)
{ KernelBody<false>(...); }                  // one id per event, the fallback
```

`Reuse` changes **only** the physical id each event gets. Same schedule, same
wait placement, same priming, same drain — so a behavioural difference between
the two entry points isolates id recycling and nothing else.

| policy | ids per sub block | highest local id | grows with schedule |
| --- | ---: | ---: | --- |
| recycled two-slot ring (default) | 5 | 4 of 10 | no |
| single-shot (fallback) | 11 | 10 of 10 | yes |

Both budgets are compile-time objects rather than comments: each policy's
`kMaxLocalId` is `static_assert`ed against `kMaxSafeLocalId = 10`. The
eight-chunk single-shot schedule saturates that exactly, which is the reason
the recycled ring is the default and not merely an option — a ninth chunk or a
second temporal segment does not fit single-shot allocation and must not be
made to fit by renumbering into the library's reserved 11..14.

`turboquant_wave::kReshapeDefaultReuse` is the one constant that decides this.
The kernels' NTTP default, the NPU harness default, the CAModel runner default
and the host gate's own check all read it, so the default cannot drift between
them; `tests/cpu/test_wave_protocol_header.cpp` fails if it is flipped without
the documentation being updated with it.

The brief's `-DTQ_TEST_MODE4_REUSE=1` is still honoured (and is now a no-op at
`1`); `-DTQ_TEST_MODE4_SINGLE_SHOT=1` pins a CI job to the fallback, and
`--mode4-single-shot` / `--mode4-reuse` override per run.

### 1.4 Rule 3 — the mandatory inter-wave epilogue drain

Both kernels, after draining every ring credit and every panel release:

```
Cube   MTE2_S, M_S, FIX_S  ->  CrossCoreSetFlag(kFlagPageDrained<0>),
                                CrossCoreSetFlag(kFlagPageDrained<1>)
AIV0   CrossCoreWaitFlag<4, PIPE_V>(drained)     then V_S, MTE2_S, MTE3_S
AIV1   CrossCoreWaitFlag<4, PIPE_MTE3>(drained)  then MTE3_S, V_S, MTE2_S
```

The Vector cores wait for the Cube first because the Cube is the only core
that writes their landing rings. MTE1 has no scalar route (guidelines §6.2a)
and is covered transitively by the `M_MTE1` wait behind the Mmads. MTE2 is
idle on both Vector cores in these kernels and is drained anyway: "this queue
is empty" is exactly what the wave boundary needs established.

### 1.5 Ingress lookahead 1 → 3

`kIngressLookahead = kStageDepth - 1`. Column *j* now stages chunk *j* and
multiplies chunk *j - 3*. `GemmChunk` contains a `CrossCoreWaitFlag`, which
stalls the **scalar** stream, which is what dispatches into the MTE2 queue, so
at a lookahead of 1 page ingress sat behind the Cube's wait for a landing
panel even with three L1 slots free and its own credits unconsumed. The bound
is `kIngressLookahead <= kStageDepth`, because `StagePage(c)` waits the credit
`GemmChunk(c - kStageDepth)` releases.

**1-safety and deadlock-freedom at both lookaheads are re-proved** by the
model (section 2.4). **The wall-clock claim is not measured here** and must
not be quoted: it needs a CAModel trace with intersecting MTE2/Cube intervals,
and `REFINEMENT_OF_COMBAT_KERNELS.md` §5 is explicit that source order and
balanced events do not establish overlap.

### 1.6 Host workarounds removed

| file | before | after |
| --- | --- | --- |
| `tests/npu/turboquant_npu_common.hpp` | `Bf16Device::launch_window` + `launch(stream, f, window)` | one `launch(stream, f, reuse)` |
| `tests/npu/validation/test_turboquant_bf16_validation.cpp` | `--window` (default 32), ladder 1/32/33/64 | ladder 1/32/33/48/64/128, one dispatch each |
| `tests/npu/validation/test_turboquant_a8_validation.cpp` | no CLI, fixed 16/512/1024 tokens | same ladder and CLI as BF16 |
| `tests/npu/bench/bench_turboquant_reshape_and_cache.cpp` | `kBenchWindow = 32`, BF16-cube paid 2 dispatches to the others' 1 | all three kernels one dispatch, timed on equal terms |
| `tests/camodel/test_camodel_*_a8.cpp`, `*_bf16.cpp` | `--window W` loop | single dispatch, `--mode4-reuse` |
| `scripts/run_reshape_bf16_camodel.sh` | `BF16_WINDOW` | `BF16_MODE4`, ladder `1 32 33 48 64` |

A launch window re-enters the kernel at each wave boundary, which re-arms
everything — it is not a fix, it is a way of not testing this.

New flags, the same spelling in every harness: `--strips N` / `--tokens N`
(repeatable), `--single` (no warmup, one dispatch per path; in a validation
runner, the last ladder rung only), `--mode4-reuse` / `--mode4-single-shot`,
and the build define `-DTQ_TEST_MODE4_SINGLE_SHOT=1` for a whole CI job. `--single`
is retained with the meaning `bench_turboquant_decode_splitk` already
publishes.

### 1.7 Trace analyzer

`scripts/analyze_reshape_trace.py` gains `--mode4-reuse` (the expected per-id
census differs per policy) and, more importantly, a **tick-order** audit that
counting cannot do: across all three core dumps, every `WAIT_INTRA_BLOCK` must
be released by an announcement that had already issued, and no id may ever
carry two outstanding ones. That is the check that caught the 1032- and
2179-cycle early waits; set/wait counts balance under reuse and pass
vacuously.

---

### 1.8 The fence audit, `scripts/audit_cross_core_fences.py`

New, and the gate the recycled default rests on. `CrossCoreSetFlag<4, PIPE_x>`
issues on the scalar unit and the `pipe` argument does not order it
(guidelines §6.2a), so an announcement meaning "my work on pipe x has
finished" must be preceded by its `SetFlag`/`WaitFlag<HardEvent::x_S>` pair.
The audit parses both kernels, finds every cross-core set, and requires that
pair immediately in front of it, with the route matching the announcement's
pipe. Comments and device diagnostics are transparent; any other instruction in
between is a finding. Priming sets are exempt, by name, because "this buffer
starts empty" reports no earlier work.

It is a separate gate because the property is invisible to the others: set and
wait counts balance without it, the id census passes, the token/queue model
works on the schedule rather than the fence, and the output can be bit-exact
while a landing panel is read a generation early. That is precisely what
happened twice (section 3). Four self-test controls run on every invocation —
an unfenced set, a wrong scalar route and work between the fence and the set
must all be rejected, a correctly fenced set accepted — so a clean run is
evidence rather than a no-op.

---

## 2. Evidence

### 2.1 Gates run

```bash
bash scripts/run_reshape_wave_host_gate.sh      # g++ 11.4, python 3.10, minutes
```

| gate | result |
| --- | --- |
| protocol header, `-Wall -Wextra -Werror`, every `static_assert` | PASS |
| cross-core id census, both policies | PASS |
| default policy pinned to the recycled ring | PASS |
| boundary guard on 1/32/33/48/64/128 strips and on zero strip width | PASS |
| cross-core announcement fences, both kernels **and the probe** | PASS |
| fence audit self-test (3 rejections, 2 acceptances) | PASS |
| 6 host harnesses, `-Wall -Wextra -Werror` | PASS |
| Tier-1 host probe (operator arithmetic contract unchanged) | PASS |
| inter-wave token/queue model, both policies × 4 semantics × 5 geometries | PASS |
| 3 negative controls on the wave model | PASS (rejected) |
| decode ring model (unrelated, confirms nothing regressed) | PASS, 21738 states |

Census output:

```
recycled    10 distinct ids over both sub blocks, highest local id  4 of 10, 44 ops
single-shot 22 distinct ids over both sub blocks, highest local id 10 of 10, 44 ops
```

Fence audit output — every announcement in both kernels and in the research
probe, with the drain that orders it:

```
ok     PIPE_FIX  kFlagProductReady<Reuse, Chunk>   FIX_S(kEvtL0)
ok     PIPE_FIX  kFlagPageDrained<Reuse, 0u/1u>    FIX_S(kEvtDrain)
ok     PIPE_V    kFlagPanelFreeSet<Reuse, Chunk>   V_S(pack_evt)
exempt PIPE_V    kFlagPanelFreePrime<Reuse, ...>   priming set
```

### 2.2 Gates NOT run, and why

| gate | status | reason |
| --- | --- | --- |
| Device compilation, `-Wall -Werror`, BiSheng | **unavailable** | no CANN 9.2 toolkit on this host. The installed toolkit is CANN 8.5.0 and has no `dav_3510` implementation directory. |
| CAModel boundary canaries (§3) | **unavailable** | no `Ascend950PR_9599` under `tools/simulator`. |
| CAModel depth and wrap tests (§4) | **unavailable** | same. |
| CAModel trace audit and instruction census (§5) | **unavailable** | same; nothing to audit without a run. |
| Wave ladder 1/32/33/48/64/128 on CAModel | **unavailable** | same. |
| Wave ladder on physical silicon | **unavailable** | no NPU on this host. |
| `--mode4-single-shot` A/B on silicon | **unavailable here** | the isolated probe that settled the semantics was run elsewhere (sections 2.5 and 3); the A/B on *these operators* has still not been run anywhere. |

One partial substitute was possible and was used: the `HardEvent` enumerators
the new drains need were checked against the installed CANN 8.5.0
`asc/impl/basic_api/kernel_event.h`. `V_S`, `MTE2_S`, `MTE3_S`, `FIX_S` and
`M_S` are all present and there is **no** `MTE1_S`, which confirms §6.2a's
route list. That the enumerators exist is not that they lower on `dav-3510`;
only a device build settles that.

### 2.3 The one assumption the device build must confirm first

The boundary guard returns **before** `AscendC::InitSocState()`, which the
brief requires. `AscendC::GetBlockIdx()` and `GetSubBlockNum()` are hardware
register reads and should not depend on SOC state, but that is **still not
verified on `dav-3510`**, and this change does not verify it: the directive to
check it needs a CANN 9.2 toolchain, which this host does not have.

There is now a data point on the other side. `tests/research/kernel_cluster
_sync_probe.cpp`, which does compile and run on that toolchain, calls
`InitSocState()` *first* and guards afterwards:

```cpp
AscendC::InitSocState();
const std::uint32_t block = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
if (block != 0u) return;
```

So the order the operators use is the untested one. If the first device build
faults or misreads the block index, adopt the probe's order and re-check that
the guard still precedes every GM access — it will, because nothing between
them touches memory, and `InitSocState` is core-local register setup.

### 2.4 The wave model, and what it decides

`tests/cpu/test_reshape_wave_model.py` is new. Three cores, in-order engine
queues, a scalar stream that dispatches into them in program order and stalls
on cross-core waits, capacity-one HardEvent places, mode-4 places shared
between the Cube and one Vector core, and generation ownership on every L1
slot, landing panel, packed slot, L0 page region and L0C. It runs **two** grid
blocks on the same physical block, carrying the marking across.

Two hardware properties are knobs rather than assumptions: whether
`wait_intra_block` **consumes** the flag, and whether kernel entry **clears**
the block's mode-4 registers. The first is now measured (section 3) and the
model is unchanged by that — it still reports all four, which is what makes it
useful as a regression oracle rather than a restatement of the conclusion.

| wait consumes | entry clears | single-shot | recycled |
| --- | --- | --- | --- |
| yes | yes | pass | pass |
| yes | no | pass | pass |
| no (sticky) | yes | **pass** | fail in wave 0 |
| no (sticky) | no | fail in wave 1 | fail in wave 0 |

The measured semantics put the hardware in rows 1–2, where both policies pass.
Rows 3–4 are kept as the **failure signature**, not as a prediction: both
failures surface as buffer-ownership violations rather than as assertions about
flags — a Vector core casting a landing panel the Fixpipe has not written yet —
and that is exactly the shape of both recorded CAModel traces, whose real cause
was an undrained announcement. If a revision drops a fence, this is what it
will look like.

Coverage: exhaustive at 2 chunks over a depth-2 L1 ring (57,348 reachable
states per wave; 93,678 for one case), randomised at 4 and 8 chunks, at
ingress lookaheads 1 and 3. Negative controls rejected: unprimed landing
credits deadlock, a missing epilogue drain leaks mode-4 flags `[4, 5, 20, 21]`
past the block, a doubled announcement is caught. **Exhaustive exploration is
on bounded instances; this is not a claim about every sequence length, and it
is not a CAModel result.**

### 2.5 Evidence produced elsewhere

Section 3's silicon and CAModel results for
`tests/research/kernel_cluster_sync_probe.cpp` were **not produced on this
workstation** and are not reproducible here: no NPU, no CANN 9.2 toolkit, no
`Ascend950PR_9599` simulator. They are reported as given — both iterations
clean on silicon, CAModel clean at 176,061 ticks with no early wait retirement
— and they are the reason the default policy changed.

What that is worth, and what it is not:

* the probe **source** is in the repository and builds as two targets, so the
  claim is reproducible by anyone with the hardware or the simulator; it is not
  a transcript;
* the probe's own fence discipline is checked here, mechanically, by
  `scripts/audit_cross_core_fences.py` — so the configuration described is the
  configuration committed;
* the **trace dumps are not here.** "No early wait retirement" is a reading of
  a trace this report has not seen, and it is the load-bearing half of the
  result, because a data-free probe that merely exits cleanly cannot
  distinguish a consuming latch from a sticky one (section 3.4). Treat it as a
  reported measurement, not as a gate run in this repository, until the dumps
  and a per-id event table are attached.

---

## 3. The reversal: recycling was refuted, then diagnosed

This report's first revision shipped single-shot channels as the default and
declined the brief's instruction to make the recycled ring canonical. That
decision is reversed. The reasoning on both sides is recorded here because the
measurement did not change — the interpretation did.

### 3.1 What was observed, twice

The recycled ring was synthesised, exhaustively proved 1-safe, deadlock-free
and co-reachable over 41,844,109 modulo states
(`kernels/ops/turboquant/safe_pipeline_proof.json`, semantics field: *consuming
one-bit latch*), integrated into `turboquant_reshape_and_cache_a8.cpp`, and run
on CAModel at `Ascend950PR_9599` / CANN 9.2.0-beta.2. It produced bit-exact
caches and intact guards, and its trace showed AIV1's READY id 16 set at cycle
4906, waited at 4916, **waited again at 7589**, and only set for the second
chunk at 8621 — the second wait retiring 1032 cycles before its notification.
Guidelines §6.2c records the same shape independently on an earlier revision:
2179 cycles early, with the landing panel the Vector core then read 87%
identical to that slot's previous occupant.

The conclusion drawn was that a mode-4 flag is not a token a wait consumes.
That is the conservative reading of the data, and it was wrong.

### 3.2 What isolated it

`tests/research/kernel_cluster_sync_probe.cpp` — one physical AI Core block,
`MIX_AIC_1_2`, 1 Cube + 2 Vector cores, **no data**, and a multi-iteration
cyclic bilateral handshake over **recycled** mode-4 ids 0, 1, 16 and 17, with
an explicit intra-pipe drain in front of every announcement: `V_S` on each
Vector core, `MTE1_M` then `M_S` on the Cube.

| leg | result |
| --- | --- |
| physical silicon | every iteration clean, no hang, no corruption |
| CAModel (`Ascend950PR_9599`, CANN 9.2.0-beta.2) | clean, 176,061 ticks, no early wait retirement in the trace |

So mode-4 flags **do** implement consuming-latch behaviour on this part, and
the stale releases were guidelines §6.2a's hazard: `CrossCoreSetFlag<4, PIPE_x>`
issues on the scalar unit, the `pipe` argument does not order it, and the
announcement overtook the Fixpipe it reported. Recycling an id converts that
reordering into a *stale release*; single-shot allocation converts it into a
harmless duplicate. That is why it only ever showed under recycling, and why
recycling looked like the cause.

### 3.3 Why the flip is safe in *these* kernels

Because the fence the diagnosis makes load-bearing was already present at every
cross-core announcement in both operators, and is now audited rather than
assumed:

| announcement | drain in front of it |
| --- | --- |
| Cube, `kFlagProductReady` | `FIX_S` |
| Cube, `kFlagPageDrained` ×2 | `MTE2_S`, `M_S`, `FIX_S` (epilogue) |
| AIV, `kFlagPanelFreeSet` | `V_S` |
| AIV, `kFlagPanelFreePrime` ×2 | none needed — priming |

`scripts/audit_cross_core_fences.py` produces that table from the sources and
fails on a missing or mismatched drain. It passes on both kernels and on the
probe.

### 3.4 What this does **not** establish

Stated plainly, because the probe is small and the conclusion it supports is
large:

* **The probe carries no data.** A clean exit does not by itself distinguish a
  consuming latch from a sticky one — under a sticky flag every wait would
  retire early and the run would still finish. Only the tick order of
  `SET_INTRA_BLOCK` against `WAIT_INTRA_BLOCK` on one recycled id separates
  them, and that is a *trace* reading. The runner now prints this caveat.
* **The trace was read elsewhere.** The dumps behind "no early wait retirement"
  are not in this repository, so §6.6's requirement of a `reports/` entry with
  the per-id event table is **not** satisfied. Attach the dumps and the
  `analyze_reshape_trace.py` output for the operator run when it exists.
* **One block, two iterations, three cores, no data.** The probe says nothing
  about inter-wave flag inheritance across 32 physical blocks, about these
  operators' schedules, or about their buffer lifetimes. The wave ladder is
  still the gate for those, and it has not been run.
* **These operators have not been run at all** on the recycled entry point, on
  silicon or on CAModel. The policy flip is justified by the mechanism, not by
  a run of the thing being flipped.

### 3.5 What to run first, in order

```bash
# 1. the probe, on both legs, with more iterations than the default 2
./build/lib/test_cluster_sync_probe 16
./build/lib/test_cluster_sync_probe_sim 8      # then read the trace

# 2. the operator on CAModel, default policy, and AUDIT THE TRACE
A8_SLICES=64 bash scripts/run_reshape_a8_camodel.sh
python3 scripts/analyze_reshape_trace.py \
    core0.veccore0.instr_popped_log.dump core0.veccore0.instr_completed_log.dump \
    core0.veccore1.instr_popped_log.dump core0.veccore1.instr_completed_log.dump \
    core0.cubecore0.instr_popped_log.dump core0.cubecore0.instr_completed_log.dump

# 3. the A/B that isolates recycling in the operators themselves
./build/lib/test_turboquant_a8_validation --strips 64
./build/lib/test_turboquant_a8_validation --mode4-single-shot --strips 64
```

**A green validation run promotes nothing**, in either direction. Both
revisions that were wrong were green; what caught them was the tick-order
audit, which is check 2 of `analyze_reshape_trace.py` and now defaults to the
recycled policy's census.

---

## 4. The barrier audit

The brief also asked for redundant `PipeBarrier<PIPE_V>` flushes to be removed.
All 16 in each kernel's Vector stage were audited against the buffers they
separate:

| site | separates | verdict |
| --- | --- | --- |
| after each `BlockReduceMax` (×2) | RAW on `f8`, then on `abs_max` | needed |
| after `ValueFoldEmit` | RAW on `work` | needed |
| `And` → `Adds` → `Sub` (×2) | RAW on `exp_bits`, then `pow_bits` | needed |
| before/after `Brcb` (key, ×2) | RAW on `inv`, then `inv_bcast` | needed |
| around the three `Muls` (value, ×4) | RAW on `inv`, then `inv_rep` ×3 | needed |
| before `Mins`, before `Cast` (×2) | RAW/WAW on `work`; WAR on `kUbCast` | needed |
| before and after the FP16 stage (×2) | RAW on `quant`, then `staged` | needed |
| prologue fence after the three `Duplicate`s | RAW on `one`/`exp_mask`/`inv_seed` | needed |

**None is redundant in the dependence sense**: every one sits at a genuine
same-buffer RAW, WAW or WAR boundary on the Vector pipe, so none was removed.

Guidelines §3.2 does say these are belt-and-braces — "the compiler demonstrably
does order most same-pipe UB dependencies across separate VF blocks" — and the
kernel is consistent with that: `Abs(work, cast)` → `KeyFold(f128, work)` and
`KeyFold` → `KeyFold` carry no barrier and are bit-correct. So an ablation
could probably remove many of them. §3.2 also says, in the same paragraph,
that the ablation has never been run; §0 is the standing record that removing
synchronisation changes correctness and not only speed. The only gate that
could settle it is a CAModel bit-exact run plus a trace, and that gate is
unavailable on this host, so the barriers stay and this table is the candidate
list for whoever runs it.

On the other item in the same directive: there are **no** scalar divisions in
any inner loop to hoist. `kChunkHead`, `kL1Slot`, `kLandSlot` and `kPackSlot`
are compile-time template values. The three runtime divisions are
`GetBlockIdx() / GetSubBlockNum()`, `numTokens / tokensPerCore` and
`slotMapping[0] / blockSize`, all once per kernel. The one real finding was
that `numTokens / tokensPerCore` was computed twice in sparse-sampling builds;
it is hoisted into `total_strips` now.

---

## 5. Reproducing

```bash
# Host gate, no toolkit needed (this is what was run):
bash scripts/run_reshape_wave_host_gate.sh

# On a machine with CANN 9.2 and the Ascend950PR_9599 simulator:
A8_SLICES="1 2 3 32 64" bash scripts/run_reshape_a8_camodel.sh
BF16_SLICES="1 32 33 48 64" bash scripts/run_reshape_bf16_camodel.sh

# On a machine with an NPU:
./build/lib/test_turboquant_a8_validation      # ladder 1/32/33/48/64/128
./build/lib/test_turboquant_bf16_validation
./build/lib/bench_turboquant_reshape_and_cache --single

# The policy A/B, both halves (default is the recycled ring):
A8_MODE4=single-shot A8_SLICES=64 bash scripts/run_reshape_a8_camodel.sh
./build/lib/test_turboquant_a8_validation --mode4-single-shot --strips 64

# The mode-4 semantics probe itself, on both legs:
./build/lib/test_cluster_sync_probe 16          # hardware
./build/lib/test_cluster_sync_probe_sim 8       # CAModel, then read the trace
```

## 6. Limitations

* No device compilation, no CAModel run, no silicon run. Section 2.2 lists
  every gate that is unavailable here; none of them is substituted for.
* The wave model covers synchronisation and buffer lifetime only — not
  numerics, alignment, bus burst splitting or compiler lowering.
* Exhaustive exploration is bounded (2 chunks); 4 and 8 chunks are randomised.
* The ingress-lookahead change is proved 1-safe and deadlock-free and is
  **not** measured. No throughput claim is made for it.
* Whether `wait_intra_block` consumes is settled **by a run made elsewhere**
  and reported here (section 3.2); nothing on this host reproduced it. Whether
  kernel entry clears the mode-4 registers remains open, and the shipped
  default does not depend on it.
* The recycled default is justified by the mechanism plus the audited fences,
  **not** by any run of these operators on that policy. Section 3.5 is the
  order to settle that in.
* `scripts/audit_cross_core_fences.py` reads source text, not generated code.
  It establishes that the drain is written, not that BiSheng emits it where
  expected; the trace audit is what checks the latter.
* `turboquant_paged_attention_opt.cpp` and `turboquant_paged_decode_attention.cpp`
  both recycle mode-4 ids and both carried the superseded rationale. Their
  comments now point at the corrected §6.2c, but **neither has been put through
  the fence audit**; do that before relying on their reasoning.
* `tests/host_stubs` declares the ACL surface the harnesses use and links to
  nothing. A pass there means the host code compiles clean, never that the
  harness works.
