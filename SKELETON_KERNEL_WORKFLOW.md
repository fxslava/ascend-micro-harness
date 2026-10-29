# Writing a skeleton kernel, and every stage of verifying it

Pipeline and buffer-lifetime refactors of real operators follow the primary
[combat kernel refinement baseline](REFINEMENT_OF_COMBAT_KERNELS.md): token model,
UB checks, isolated logical-page head/middle/tail canaries, ring wrap, trace audit.


Ascend 950PR / arch35 / `dav-3510` / CANN 9.2.0-beta.2, `Ascend950PR_9599`
CAModel.

This is the **procedure**. The *rules* it enforces live in
[`MICROROUTINE_DESIGN_GUIDELINES.md`](MICROROUTINE_DESIGN_GUIDELINES.md) and
are cross-referenced by section rather than repeated here. Read §1–§3 before
writing device code; read §6 before wiring leaves into a pipeline.

A skeleton kernel proves one specific launch, memory or scheduling property
with the smallest device work that still exercises it. It is **not** evidence
that omitted production arithmetic is correct or fast, and the report must say
so.

---

## 0. Write the claim down first

Before any code, record in the source header and in the report draft:

| Field | Example |
| --- | --- |
| What it proves | event liveness across a depth-3 ring; UB→L1 CV path; launch ABI |
| What it deliberately omits | FP16 normalisation, Split-K, real softmax |
| Target | `Ascend950PR_9599`, `__NPU_ARCH__ == 3510`, CANN 9.2.0-beta.2 |
| Kernel type | `MIX_AIC_1_2` (required for mode-4 and the direct CV path, §6.1) |
| Grid | pages / tiles, and why that number (see V2 and V3 below) |

A claim you cannot name is a claim no verification stage can check.

---

## 1. Pick the artifact and its place

| Thing | Directory | Notes |
| --- | --- | --- |
| Synchronisation-free compute leaf | `routines/routine_*.hpp` | duck-typed engine, no `HardEvent`, no GM |
| Isolated-leaf CAModel wrapper | `kernels/camodel/<name>.cpp` | one leaf under test |
| Pipeline skeleton | `kernels/skeleton/<name>.cpp` | diagnostic, lightweight stage work |
| Production operator | `kernels/ops/<family>/<name>.cpp` | the real fused thing |
| CAModel runner | `tests/camodel/test_camodel_<name>.cpp` | ACL host, guarded buffers, oracle |
| Tier-1 host probe | `tests/cpu/test_<name>_probe.cpp` | mock engines + scalar oracle |

Canonical examples to copy from: `kernels/skeleton/turboquant_reshape_and_cache.cpp`
(marked-graph ring), `kernels/ops/turboquant/turboquant_paged_attention_opt.cpp`
(five leaves, both Vector cores, CV path).

Inspect `git status`, the existing CMake targets, tests and reports before
editing, and preserve unrelated working-tree changes.

### Wiring the build

```cmake
# add_micro_camodel_probe(<name> [<kernel_basename>|<kernel_path>] [<define>...])
add_micro_camodel_probe(my_probe)                       # kernels/camodel/my_probe.cpp
add_micro_camodel_probe(my_ring_8tiles \
    kernels/skeleton/my_ring.cpp  MY_RING_TILES=8)      # path form + build parameter
```

Target `test_camodel_<name>` is built from `tests/camodel/test_camodel_<name>.cpp`.
A second argument containing `/` is a repo-relative source path; further
arguments are `NAME=VALUE` definitions reaching both the device kernel and the
runner. That is how one kernel source serves several grid sizes.

> **Trap, already paid for once.** `-forward-options-to-host-compiler` takes
> exactly **one** option. CANN's legacy `ascendc_compile_options` stores
> everything after that marker in `HOST_COMPILE_OPTIONS` and passes it to a
> nested ExternalProject as a single configure argument, so a two-element
> property silently loses one element — and if the lost one is
> `ASCEND_MICRO_CANN_LAUNCH`, the host leg compiles an empty translation unit
> and your `launch_*` symbol vanishes from the shared library with no error.
> Put definitions through `ascendc_compile_definitions`.

### The launch ABI

Guard the device body on `__CCE_AICORE__` and the host launcher on
`ASCEND_MICRO_CANN_LAUNCH`, exactly as the existing kernels do. The exported
symbol, the `extern "C"` launcher name and the `extern "C"` declaration in the
runner must all agree, or the link fails late and confusingly.

---

## 2. Write the kernel

Follow guidelines §1–§3. The structural skeleton:

```cpp
constexpr unsigned kStageDepth = 3;          // ring slots
constexpr unsigned kBankSkewBytes = 32u;     // §2.2
constexpr unsigned kReadPadBytes = 256u;     // §2.3, VECTOR_REG_WIDTH

constexpr std::uint32_t RoundUpBlock(std::uint32_t b) { return (b + 31u) / 32u * 32u; }
constexpr std::uint32_t NextSlot(std::uint32_t base, std::uint32_t bytes)
{ return RoundUpBlock(base + bytes) + kBankSkewBytes; }

enum class Route : unsigned { Mte2V, VMte2, VMte3, Mte3V };
```

Derive **every** operand base through `NextSlot`, never by hand. Then draw the
producer / consumer / reuse graph before writing a single `SetFlag`, and put
each back edge on the route whose consumer queue must stall (§3.1).

`AscendC::InitSocState()` first. Place `LocalTensor`s with explicit
`{TPosition, byteOffset, elementCount}`; the second field is a **byte**
address in that position's space, which is what lets two cores share a buffer
by agreeing on a constant (§6.1).

---

# The verification ladder

Five stages plus the report. **Each one catches a class the one before it
cannot see**, so none is optional and the order matters. The project's own
history is the argument: Tier 1 stayed green while the kernel computed on the
wrong tile's data, and the depth-boundary CAModel run stayed green while a
back edge was missing entirely.

| Stage | Catches | Blind to |
| --- | --- | --- |
| V0 compile-time | alignment, capacity, read pad, ratio | anything dynamic |
| V1 Tier-1 host | deadlock, token balance, arithmetic | run-ahead, WAR, scalar traps |
| V2 CAModel, N = depth | scalar traps, numerics, fill | uncontended reuse edges |
| V3 CAModel, N ≥ 2·depth | reuse WAR, steady-state cadence | silicon timing |
| V4 trace audit | leaks, quorum, CV path, outflow order | semantics |
| V5 report | overclaiming | — |

---

## V0 — Compile time

Make the layout contract a build failure, not a runtime surprise.

```cpp
static_assert(kAlpha % 32u == 0u, "every UB operand base must be 32-byte aligned");
static_assert(kArenaEnd + kReadPadBytes <= kUbWindow, "no VECTOR_REG_WIDTH read pad");
static_assert(kSlotStrideBytes % 256u != 0u, "slot stride puts every slot on one bank phase");
static_assert(kBroadcastLanes * 4u == 8u * 32u, "a Brcb repeat writes eight DataBlocks");
static_assert(kLanes * 2u % 32u == 0u, "FP16 MTE3 length must be whole 32-byte bursts");
```

If the kernel depends on the direct CV path, reject the configuration that
would silently relay through GM (§6.1):

```cpp
#if defined(__MIX_CORE_AIC_RATION__) && __MIX_CORE_AIC_RATION__ == 1
#error "needs MIX_AIC_1_2: at a 1:1 ratio DataCopy UB->L1 falls back to a KFC GM relay"
#endif
```

Runners compile with `-Wall -Wextra -Werror`. **Gate:** clean configure and
build.

---

## V1 — Tier-1 host model (sub-millisecond)

Two separate things, both on the host:

**(a) The discrete-event queue model** — `tests/cpu/test_pipeline_deadlock_probe.cpp`
schedules MTE2 / Vector / MTE3 as independent queues over one-bit tokens and
proves *structural* properties only: deadlock freedom, token-balance
liveness, no double-`SetFlag` on a live bit, mock guard bands.

**(b) A bit-exact scalar oracle** — `tests/cpu/test_<name>_probe.cpp` over
`include/sim_micro_testbed.hpp`: `GuardedBuffer<T>` with 64-byte
`0xDEADBEEF` canaries, `assert_aligned`, `require`, `float_bits`,
`float_to_half_rne`, `float_to_bfloat16_rne`, `ASSERT_CANARY_INTACT`.

```sh
bash scripts/run_modulo3_camodel.sh          # queue model + the modulo3 CAModel probe
# or just the host leg:
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    tests/cpu/test_<name>_probe.cpp -o bin/test_<name>_probe && bin/test_<name>_probe
```

**Gate:** 0 deadlocks, sets == waits per `(route, ID)`, 0 residual tokens, all
canaries intact, oracle bit-identical.

**Cannot see** (§4): scalar addressing traps, null `TPipe` dereferences, the
256-byte unmasked read footprint, engine run-ahead, any WAR hazard. Its mock
engines are sequential, so MTE2 can never outrun Vector. Do not substitute it
for native evidence.

---

## V2 — Tier-2 CAModel at the depth boundary

```sh
bash scripts/run_modulo3_camodel.sh micro <name>
```

Needs WSL with CANN; override `ASCEND_HOME_PATH`, `SOC_VERSION`,
`MODULO3_BUILD_DIR` as needed. The script sources `set_env.sh`, configures a
fresh build directory (CANN's nested ExternalProject mutates intermediates, so
reuse is unsafe), builds `test_camodel_<name>`, runs it under `timeout 120s`,
and **fails on any of** `[error]`, `[fatal]`, `invalid ldst addr`,
`unrecognize ldst addr`.

Set the grid to the smallest size that still exercises every phase — for a
ring, `N = kStageDepth`. Tens of seconds per run.

**Gate, in this order** (§4):

1. exit code 0 **and** zero `invalid ldst addr` / `su_ccu_mpu_err`. A numeric
   PASS alongside scalar faults means the faults are side-effect-free *today*,
   not that the kernel is correct — that exact situation shipped once.
2. GM canaries intact, head and tail, on every buffer.
3. numeric agreement **per tile**, not only at the end.
4. `SET_FLAG`/`WAIT_FLAG` balanced per `(route, ID)`.

**Cannot see:** a reuse back edge that is never contended. At `N = kStageDepth`
the ring never wraps, so every buffer-reuse edge is merely primed and drained.
Which is why V3 exists.

---

## V3 — Tier-2 CAModel past the ring depth

```sh
bash scripts/run_modulo3_camodel.sh micro <name>_8tiles    # depth 3 -> use 8
```

Choose `N ≥ 2 · kStageDepth`. With depth 3 the first three columns are fill and
the last two are drain, so `N = 8` leaves several steady-state columns. This
stage does two things nothing else does:

- **Contends every reuse back edge.** A missing one is invisible at
  `N = kStageDepth` and produces wrong data or a hang here. This is how AIV1's
  missing packed-page WAR edge was found.
- **Yields the per-page steady-state interval**, which is the throughput
  figure. The whole-run dispatch span of a short run is fill plus drain and is
  *not* a throughput number; do not quote it as one.

**Gate:** same as V2, plus a steady-state interval that the analyzer actually
reports (if it says the ring never wrapped, `N` is too small).

---

## V4 — Trace audit

Spans for any micro probe come from `scripts/analyze_micro_trace.py`, run
automatically. A kernel driving the Cube and both Vector cores also gets
`scripts/analyze_fused_trace.py --tiles N`, whose six checks are each **fatal**:

1. intra-core `SET_FLAG`/`WAIT_FLAG` balance per core and per
   `(pipe, trigger pipe, flag id)` — no HardEvent token leaks;
2. cross-core `SET_INTRA_BLOCK`/`WAIT_INTRA_BLOCK` pairing per **absolute**
   flag id, rebasing AIV1 by `INTRA_NUM = 16` because an AIV reports `sync_id`
   inside its own sub-block half;
3. the double-handshake quorum — the Cube waited the same number of flags from
   each Vector core, and at least one GEMM's worth;
4. both Vector cores executed, with a per-engine instruction census;
5. no GM store anywhere on the CV staging core, and exactly one `MOV_UB_TO_L1`
   per key page — this is how "zero GM workspace" is *proved* rather than
   asserted;
6. every GM store on the recurrence core happens after the last Fixpipe, which
   is what "one outflow at the end of the sequence" means in the trace rather
   than in the source.

> The audit hook in `run_modulo3_camodel.sh` is **name-gated** on
> `fused_attention_*tiles`. A new pipeline kernel either adopts that naming or
> adds its own branch — otherwise the six checks silently do not run.

**Gate:** analyzer exits 0. Overlap and cadence are then *reported*, not
assumed: show intersecting in-flight intervals from the dumps, and never infer
concurrency from source order.

---

## V5 — Report

Write to `reports/<name>.md`, and state plainly:

- what the skeleton proves, and what it deliberately omits;
- the exact commands run, and the SoC / CANN / build configuration;
- measured numbers with their units and their provenance — model ticks,
  dispatch and completion spans, per-page steady-state interval, event counts,
  token residuals, canary results, ULP against the largest intermediate (§5);
- **what remains untested**: silicon, sustained throughput, other grid shapes,
  idle cores, omitted arithmetic;
- any earlier claim this run **retracts**, labelled as a retraction rather
  than quietly edited. `reports/fused_attention_integration.md` retracting the
  "no UB→L1 transfer exists" claim is the model.

Distinguish a depth-boundary run from steady state from silicon, every time.
Never invent trace evidence.

---

## Fixture design — the part that decides whether any of this works

A fixture whose tiles are interchangeable cannot detect a run-ahead,
double-buffering or wrong-slot fault, because the wrong operand yields the
right answer (§6.4). Make the pages distinguishable:

- reduction maxima **strictly increasing** across tiles, so the rescale path
  runs every tile;
- per-tile sums distinct; every output channel distinct;
- where the arithmetic permits, choose operands whose exact result is
  representable, so the full output panel is checked **bit-exactly** instead of
  against a tolerance. Keeping every score a multiple of 1/128 is what makes
  the fused probe's 768-entry check meaningful.

This is not hypothetical. The online-softmax V→MTE2 WAR hazard survived its
original validation precisely because every tile shared the same maximum
(§0).

---

## Debugging a V2/V3 numeric divergence

Static reasoning failed on three hypotheses before a per-tile GM dump resolved
the last one in a single build cycle. At ~5 minutes per rebuild, instrument
early: widen the output buffer, store the suspect state blocks behind an
`#ifdef`, print them beside the oracle's per-tile values, and read off which
tile diverges first (§4).

---

## Definition of done

Procedural gates; the per-rule checklist is guidelines §7.

- [ ] V0 configure and build clean, layout invariants `static_assert`ed.
- [ ] V1 green: queue model and bit-exact host oracle.
- [ ] V2 green at `N = kStageDepth`, **zero** `invalid ldst addr`, per-tile
      numerics checked.
- [ ] V3 green at `N ≥ 2 · kStageDepth`, steady-state interval reported.
- [ ] V4 analyzer exits 0, and it is actually wired for this target name.
- [ ] V5 report written, with commands, numbers, retractions and untested
      scope.
- [ ] Nothing claimed that no stage measured.
