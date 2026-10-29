# Ascend 950PR 16×16 Cube Hadamard probe

Pipeline and buffer-lifetime refactors of real operators follow the primary
[combat kernel refinement baseline](REFINEMENT_OF_COMBAT_KERNELS.md): token model,
UB checks, isolated logical-page head/middle/tail canaries, ring wrap, trace audit.


## Test domains and NPU TurboQuant harness

`tests/cpu/` contains host-only oracle probes; `tests/camodel/` contains CANN
simulator runners; `tests/npu/validation/` and `tests/npu/bench/` contain physical
NPU tests. One CANN 9.2 build includes both CAModel and NPU targets for
Ascend950PR_9599, using BiSheng through `ascendc_library`.

The BF16 comparison kernel is vendored under
[`kernels/ops/turboquant/baseline/`](kernels/ops/turboquant/baseline/); building
does not require a separate `vllm-ascend` checkout.

From WSL with the CANN environment loaded, a default build compiles every
CAModel and NPU target together:

```sh
cmake -S . -B /tmp/ascend-micro-all -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ascend-micro-all -j4
```

Executables and their kernel shared libraries are placed together in
`/tmp/ascend-micro-all/lib`.

From PowerShell, compile both NPU targets:

```powershell
& .\scripts\build_turboquant_npu.ps1
```

The script creates a fresh WSL ext4 build directory and prints its path.
Compilation does not require an attached NPU. On a machine with a 950PR and
working AscendCL device access, run the printed validation binary first, then
the benchmark binary:

```powershell
wsl -- /tmp/ascend-turboquant-npu-XXXXXX/lib/test_turboquant_a8_validation
wsl -- /tmp/ascend-turboquant-npu-XXXXXX/lib/test_turboquant_bf16_validation
wsl -- /tmp/ascend-turboquant-npu-XXXXXX/lib/bench_turboquant_reshape_and_cache
wsl -- /tmp/ascend-turboquant-npu-XXXXXX/lib/bench_turboquant_decode_splitk
```

The **Split-K decode benchmark** is the second of those. It sweeps the split
count against the two decode operators and the two reduction protocols:

```sh
bash scripts/run_decode_splitk_bench.sh build   # NPU build only, no device needed
bash scripts/run_decode_splitk_bench.sh hw --splitk "2,4,8,16" --context 4096
bash scripts/run_decode_splitk_bench.sh smoke   # the functional check, on the sim twin
```

`--splitk`, `--context`, `--warmup`, `--iters`, `--impl aic|aiv|both` and
`--sync-mode atomic|workspace|both` are the sweep; `--cache warm|cold|both`
chooses whether L2 is evicted between iterations, and `--watchdog N` bounds
every spin. It reports min / mean / p50 / p95 / p99 in microseconds from ACL
device events, the host enqueue cost separately, throughput over the bytes each
phase actually moves, and — from absolute system-cycle stamps each split
publishes — the **tail delta** between the first and last split to finish.

Two things about it are worth knowing before reading any number it prints.
There is deliberately **no combine kernel**: the `atomic` mode accumulates in
place under a Global Memory spinlock with all K splits contending for one line,
because that is what makes L2 line bouncing across physical cores measurable,
and a tidy-up pass afterwards would be measured work the question does not ask
about. And every acquire is bounded by a watchdog, after which a split gives up
without the lock and publishes that it did — so a launch cannot hang, and any
run with a nonzero timeout count is reported as INVALID rather than averaged
in. The CAModel probe that chose these two protocols, and the crossover that
says when the atomic one is affordable at all, are in
[`reports/l2_atomic_contention_split_k.md`](reports/l2_atomic_contention_split_k.md).

**This host has no NPU**, so the benchmark's standing here is a clean NPU-mode
build — the same standing `reports/reshape_and_cache_a8.md` records for the
reshape-and-cache benchmark. Its CLI, statistics, tail arithmetic and watchdog
census are exercised through `bench_decode_splitk_sim`, which builds the same
source against the CAModel libraries and announces at startup that its figures
are model ticks rather than hardware latencies.

A8 validation checks one 16-token strip and 512/1024-token contexts. Each strip
uses one physical AI Core; all packed key and value nibbles and FP32 scale bits
are compared with an independent integer Hadamard oracle.

BF16 validation shares that fixture, oracle and per-page verifier — only the
input encoding differs, and with integer activations the two instances must
produce byte-identical pages — and drives its context as several launches:

```powershell
wsl -- .../test_turboquant_bf16_validation --strips 33 --window 32
wsl -- .../test_turboquant_bf16_validation --tokens 1024
```

`--strips` and `--tokens` are repeatable and build a custom ladder; with
neither, the ladder is 1, 32, 33 and 64 strips. `--window W` caps the strips per
launch at W (default 32), because the kernel allows one strip per grid block per
launch and a grid wider than the AI core count does not dispatch. The rungs are
chosen for what they can fail on: 33 is one strip in the **second** wave, a
ragged tail launch straight after a full one, where a core that has already used
its mode-4 flag ids must re-arm them; 64 is two full waves, which needs a
complete reuse cycle instead. A token leaked by the first launch shows up at 33
and at no smaller rung.

The benchmark measures three kernels — A8 Cube, BF16 Cube and the vendored BF16
Vector baseline — with 50 warmups and 500 device-event samples each, in both
warm and 128 MiB SDMA cache-pressure modes. The dummy copy does not guarantee
full L2 invalidation. Only the two BF16 figures share an input precision and a
quantizer, so only that pair isolates Cube against Vector rotation; either
against A8 also changes the input precision and is a wall-time comparison for
the same 1024-token geometry and nothing narrower.

These three binaries need an attached 950PR. On a host without one they abort
during static initialization of the NPU-mode runtime, before `main`, so they
cannot even print their usage — the CAModel probes above are the verification
gate here.

Micro-routine development rules — the zero-scalar-compute invariant, UB
alignment and bank invariants (AKA3003), RAW/WAR hazard management inside the
AIV, and the two-tier verification methodology — are in
[`MICROROUTINE_DESIGN_GUIDELINES.md`](MICROROUTINE_DESIGN_GUIDELINES.md).
Read it before adding a routine.

[`SKELETON_KERNEL_WORKFLOW.md`](SKELETON_KERNEL_WORKFLOW.md) is the procedure
that enforces those rules: where a new kernel goes, how to wire its build and
launch ABI, and the five verification stages with the gate each one must pass
— compile-time invariants, Tier-1 host model, CAModel at the ring depth,
CAModel past the ring depth for steady state, and the trace audit. Each stage
catches a class the previous one is blind to, which is why none is optional.

The three-stage modulo synchronization probe is documented in
[`reports/modulo3_deadlock_probe.md`](reports/modulo3_deadlock_probe.md). Run it
under WSL/CAModel with `bash scripts/run_modulo3_camodel.sh`.

The micro-routine CAModel probes run with
`bash scripts/run_modulo3_camodel.sh micro <dequant4|cube_matmul|normalize_store|online_softmax>`.
Latest results are in
[`reports/micro_routines_validation.md`](reports/micro_routines_validation.md).

The production fused operator in
[`kernels/ops/turboquant/turboquant_paged_attention_opt.cpp`](kernels/ops/turboquant/turboquant_paged_attention_opt.cpp)
puts five leaves in one `MIX_AIC_1_2` three-stage modulo pipeline over a ring
of key pages, driving one Cube core and **both** Vector cores, with mode-4
double-handshake flags and zero Global Memory workspace — the dequantised key
page goes straight from Vector UB into Cube L1 over the dav-3510 MTE3 channel
(`routines/routine_cv_transfer_ub_to_l1.hpp`). Its page count is a runtime
option of one probe over one kernel source:

```sh
bash scripts/run_modulo3_camodel.sh micro fused_attention 8   # steady state
bash scripts/run_modulo3_camodel.sh micro fused_attention 3   # depth boundary
```

Three pages never wrap the depth-3 ring, so that run measures a pipeline fill
and its buffer-reuse back edges are only primed and drained. Eight pages wrap
it twice, which contends every back edge and gives a per-page steady-state
interval — and which is how AIV1's missing packed-page WAR edge was found. See
[`reports/fused_attention_integration.md`](reports/fused_attention_integration.md).

That report also retracts an earlier claim, repeated in §6.1 of the
guidelines, that no UB → L1 transfer exists on this part.

The **decode** operator in
[`kernels/ops/turboquant/turboquant_paged_decode_attention.cpp`](kernels/ops/turboquant/turboquant_paged_decode_attention.cpp)
is the full-width one: 16 query tokens, head dimension 256, a symmetric int4
paged KV cache with a planar FP32 scale cache, and a `gridDim = 32` grid of
4 KV heads × 8 temporal blocks mapping one task per physical AIC. Each 128-token
page is eight 16-token strips, and a strip occupies five pipeline stages, so
eight strips span a twelve-column modulo schedule. The Unified Buffer landings
and the score windows are **two** slots deep, so eight strips wrap each of them
four times and every buffer-reuse back edge carries real contention. The L1 key
staging is deliberately *not* a ring — one page per strip — because the ladder
measured its back edge failing to bound the staging core.

```sh
bash scripts/run_paged_decode_camodel.sh --canary 0 0   # one tile, isolated
bash scripts/run_paged_decode_camodel.sh --strips 8     # steady state
bash scripts/run_paged_decode_camodel.sh ladder         # every rung
```

The 256 × 256 randomised Hadamard is applied to the query **once**, in the Cube
prologue, as 64 accumulating bf16 Mmads against 16 resident 32 × 32 blocks —
the decode-side reuse of the prefill operator's measured result. Each strip's
score GEMM is then eight accumulating Mmads of k = 32, and the egress rotates
the accumulator back out of the cache's basis on AIV0 as eight radix-2
butterfly stages before an in-place `sigmoid(gate)`. Neither quantisation scale
is ever applied to a 4096-entry page: the key's goes on the 16 score columns
after the Fixpipe and the value's into the 16 probabilities, both exactly,
because the scales are powers of two.

Its verification is a single-tile **canary protocol**. Every Global Memory
buffer is filled with `0x7FC0BEEF` and the fixture is written into one
`(page, head, strip)` slice, so a read outside that slice returns a quiet NaN
and a write outside the active block's output slices leaves a canary word
behind. `Q_rot` and every raw score panel are checked **bit-exactly**; the
softmax state, accumulator, egress rotation and gated context are bounded in
ULP against the largest value in the row, because each passes through `Exp`.

That protocol earned its keep. It located a `Cast(bfloat16_t ← int16_t)` that
is **not a listed conversion pair on this part** and silently emits no
instructions at all, an egress gate running two MTE2 iterations ahead of its
consumer for want of a WAR back edge, and a single k = 256 Mmad that disagrees
with eight accumulating k = 32 Mmads over the same operands. The host oracle
rejected two degenerate fixtures before any of that.

The **strip ladder** then earned its keep separately, and that part is the one
worth copying. All nine canary coordinates passed, and so did two strips — the
last grid that reuses no ring slot — while four and eight strips failed. Two
distinct faults sat behind that split: a slot released by the dequant when the
*recurrence* was its last reader, which corrupted the key scale while leaving
every raw score bit-exact; and an L1 key ring whose mode-4 back edge did not
bound the staging core, so AIV1 restaged a page twice more before the Cube read
it. The second was proven from the instruction trace alone — AIV1 wrote L1
`0xc060` at ticks 6754, 8703, 10653 and 12608; the Cube read it at 7088, 26994,
66481 and 106697 — and it is a measured instance of the stale-flag mechanism
the guidelines record. A third fault was in the verification itself: the trace
audit's flag-ordering check could not have caught either, because a
`CrossCoreWaitFlag` retires on arrival and arms a block on its pipe's *next*
instruction, so no dump tick for a wait marks when its flag released it. That
check now tests the L1 pages directly, and passes two strips while failing four
and eight on the archived dumps of the original runs. A fourth was introduced
by the first fix and caught the same way: releasing the slot in the recurrence
put the `SetFlag` *after* its `WaitFlag` in program order, which an intra-core
event does not tolerate, and the model said so while the output stayed
correct — so the audit now also checks set-before-wait per route.

A fifth was in the verification again, and it is the one to be wary of. The
model dumps all 32 cores of the grid and block `N` runs on core `N`, but the
runner archived and audited `core0` unconditionally, so the six canary rungs on
blocks 16 and 24 were audited against a core that had taken the early return.
An idle core is not an empty one — 133 instructions of scalar prologue — and
every balance check in the audit is vacuously true on it, including the one
named "both Vector cores executed", whose test the prologue's single `MTE2`
satisfies. Only the L1-page check noticed. The audit now begins by asserting
that the core it is reading did the work that core is defined by, and takes
which core to read from the runner's own report of the block it isolated, so
the block → core mapping is under test rather than assumed.

All nine, the two WSL and CANN toolchain traps that cost a cycle each, and
what the runs do **not** establish — no silicon, no concurrent 32-block grid,
no cross-block combine, and why a mode-4 counter released a wait it should have
held — are in
[`reports/paged_decode_attention.md`](reports/paged_decode_attention.md).

The **sub-strip ingress** operator in
[`kernels/ops/turboquant/turboquant_reshape_scatter_aiv.cpp`](kernels/ops/turboquant/turboquant_reshape_scatter_aiv.cpp)
is the one with no Cube at all: `KERNEL_TYPE_AIV_ONLY`, one KV head of `d = 256`
owned exclusively by one physical AIV, 1 to 16 tokens per work item. It serves
what the aligned writer below cannot — a decode step writes `B x (1 + k)` tokens
into `B` different pages at arbitrary non-consecutive slots, and a ragged prefill
tail writes `n < 16` — and it is straight-line: no cross-core flag, no mode-4 id,
no modulo ring, no L1.

```sh
bash scripts/run_reshape_scatter_camodel.sh            # the spatial sweep, H = 0, 2, 3
bash scripts/run_reshape_scatter_camodel.sh ladder     # every rung
```

The two planes are written completely differently and the layout is why. The
**key** plane is token-major, so one token is 128 contiguous bytes at a 32-byte
aligned offset and a run of consecutive tokens is one MTE3 burst — no
read-modify-write. The **value** plane is strip-blocked channel-major, which puts
one token's 256 nibbles at stride 16 — 128 bytes spread over 2048 at half-byte
granularity, with no aligned-burst expression at all — so the 2048-byte strip is
read in one MTE2 burst, the owned nibbles are spliced in by register bitwise
masks, and the whole strip goes back in one burst. **The read-modify-write always
preserves**: in decode a strip accumulates over 16 successive steps, so zeroing
the unowned lanes would destroy the previous fifteen tokens.

Its `Pi . x` is the Vector butterfly, and **which** butterfly was decided by
measurement. [`routines/routine_vector_rotate_butterfly.hpp`](routines/routine_vector_rotate_butterfly.hpp)
carries three bit-identical implementations of the eight-stage cascade and
[`reports/rotate_butterfly.md`](reports/rotate_butterfly.md) has the CAModel
census that picks one:

| variant | vector instr | RVECEX | RV_VLD | RV_VST | vector span |
| --- | --- | --- | --- | --- | --- |
| Blocked — Unified Buffer operands, `BinaryRepeatParams` + UB `Gather` tables | 11250 | 2512 | 3072 | 1728 | 30215 |
| RegShuffle — register-resident, `DeInterleave`/`Select`, no tables at all | 1981 | 1834 | 64 | 64 | 1965 |
| RegGather — register-resident, `Reg::Gather` + `MulAddDst` | **1203** | 1057 | 64 | 64 | **1850** |

16 rows of 256 FP32, one AIV. Both register-resident variants issue exactly four
register loads and four register stores per row and not one more, which is the
zero-UB-spill invariant as a measured number rather than an intention; Approach A
spills 1,664 extra registers over the same work and runs 16.3x longer. RegGather
is the production leaf.

```sh
bash scripts/run_rotate_butterfly_camodel.sh           # all three variants
```

The verification is a **single-head canary protocol**. Every Global Memory
payload is flooded with `0x7FC0BEEF` — a quiet NaN, so a stray scale read
propagates rather than perturbs — and the oracle is handed the pre-launch cache
images and returns them with the owned lanes replaced, so the comparison is byte
for byte over every byte of every plane. That one comparison is simultaneously
the correctness gate, the read-modify-write preservation gate and the
unassigned-head gate. The mismatches are then classified into owned nibbles that
disagree, unowned nibbles inside a written strip that moved, and nibbles outside
every written strip that moved, because a byte comparison alone reports a wrong
page and a page packed the wrong way round identically.

Both of its Tier-1 oracles run in well under a second and are what the two
scripts above run first, because a CAModel run compares the device against them
and a wrong oracle is indistinguishable from a wrong kernel there:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    tests/cpu/test_rotate_butterfly_probe.cpp -o bin/test_rotate_butterfly_probe
bin/test_rotate_butterfly_probe
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    tests/cpu/test_reshape_scatter_probe.cpp -o bin/test_reshape_scatter_probe
bin/test_reshape_scatter_probe
```

The butterfly probe checks both stage tables against the dense `256×256`
multiplication and all three register algorithms against the cascade — including
the `(P·B₀)⁶ = B₅…B₀` identity the table-free variant is built on, which is
settled there rather than by a simulator. The scatter probe checks the two
layout maps are bijections, that every nibble of a strip lies inside its
2048-byte window and no nibble outside it does, that the kernel's `(word, bit)`
merge derivation agrees with the nibble map, and then **simulates the masked
merge at word granularity** against the nibble-level reference — the one check
that could catch an endianness or half-selection error before a ten-minute
device run. It also prints what the unguarded `absmax == 0` scale would have
been: `-1.70141e+38`.

The protocol's first run also settled the roadmap's open item: under
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` with a one-block grid the kernel
observes `GetBlockIdx() == 0`, `GetBlockNum() == 1` and `GetSubBlockNum() == 1`,
so every AIV is its own block and the MIX idiom `GetBlockIdx() / GetSubBlockNum()`
is not needed. It is printed by the kernel rather than assumed. Details, the
trace audit's six checks and what the runs do **not** establish are in
[`reports/reshape_scatter_aiv.md`](reports/reshape_scatter_aiv.md).

The **single-token decode** operator in
[`kernels/ops/turboquant/turboquant_single_token_decode_aiv.cpp`](kernels/ops/turboquant/turboquant_single_token_decode_aiv.cpp)
is the other one with no Cube at all: `KERNEL_TYPE_AIV_ONLY`, a grid of
`(request, kv_head)` with one task per physical AIV, and 6 query rows — `H_q /
H_kv` at `query_tokens_per_req == 1`, which is every decode step whatever the
batch size, because `B` scales the grid and not the row count. It is the 6-row
kernel, not the `B = 1` kernel. No L1, no `Mmad`, no `Fixpipe`, no CV transfer,
no mode-4 flag, no modulo ring: its marked graph is a line, and the trace audit
asserts that as a **positive** property rather than as an absence of findings.

```sh
bash scripts/run_single_token_decode_camodel.sh         # the spatial sweep, H = 0, 2, 3
bash scripts/run_single_token_decode_camodel.sh smoke   # one strip, to time a run
bash scripts/run_single_token_decode_camodel.sh ladder  # every rung
```

Its score GEMV is **not** the two instructions the roadmap sketched, and the
reason is worth carrying here because the failure would have been silent. Both
sketched lines need a 256-lane operand, and `SIMD_compute/mask.md` bounds a
32-bit continuous-mode mask at 64 — one 256-byte vector register — then adds
that an over-range mask "is calculated as the maximum mask value of the
corresponding data type". So a `ReduceRepeat` asking for 256 lanes is neither a
compile error nor a fault: it reduces the first 64 channels of each token and
returns a plausible, wrong panel. The `Mul` fails differently, because a
token-major key page needs query channel quarter `r mod 4` on repeat `r` and
`src1RepStride = 0` stands still rather than wrapping — the `p_x4` trick works
for the *value* contraction only because the value page is channel-major, which
makes the broadcast operand's period exactly 64 lanes.
[`routines/routine_vector_decode_gemv.hpp`](routines/routine_vector_decode_gemv.hpp)
issues seven instructions instead: four quarter-major `Mul`s, one 64-repeat
`ReduceRepeat`, and two `Add` folds at 128 and 64 bytes. The quarter-major
product layout is the whole trick — it is what makes the 64 first-level
reductions one instruction and the second level two aligned `Add`s instead of a
strided reduction the API cannot express.

The panel is still compared **bit-exactly**, and that rests on the fixture
rather than on the arithmetic: three summation orders are in play — the host's
loop, the device's binary tree inside a repeat, and the quarter fold across
repeats — and they agree only because every partial sum is an exact multiple of
1/16 under 1792. The Tier-1 probe measures the margin (largest sum of absolute
terms 235.1 against the 2²⁴/16 bound) and replays the device's own repeat
strides against the direct dot product, so a wrong stride or an aliasing
product map is caught in milliseconds rather than in a two-minute simulator
run.

That probe earned its keep twice over before any device run. It rejected the
first fixture outright: `paged_decode::KeyNibble`'s strip term is `strip * 11`
mod 8 and its head term `head * 13` mod 8, so the whole pattern depends only on
`(3·strip + 5·head) mod 8` — **eight distinct key payloads exist in that
generator** — and handing it a global strip index made page 1 byte-identical to
page 0, so a kernel that ignored the block table and read page 0 four times
would have produced exactly the right answer. It also rejected the *check*: per-row
score maxima pairwise distinct is not a structural property, because a raw score
is a multiple of 1/16 inside some 4700 values and two of ~2900 strip pairs
sharing a scalar maximum is an expected coincidence. The gate is now panel
inequality, which is what "no two strips are interchangeable" actually means.

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    tests/cpu/test_single_token_decode_probe.cpp \
    -o bin/test_single_token_decode_probe
bin/test_single_token_decode_probe
```

Its canary protocol floods the inputs as well as the outputs and uploads only
the active head's **query** rows, which is stronger than the Cube operator's:
there, head separation comes only from the cache slice and the gate, so a task
reading another head's query rows would be invisible. `m` is checked
bit-exactly — a maximum of exactly representable score columns never passes
through a transcendental — while `l`, `acc` and the egress are bounded in ULP
and relative terms, because they do. The trace audit also caught its own first
prediction being wrong: `RV_VMULA` came back at 288 against 144, and grouping
by program counter separates the fully unrolled butterfly body from six other
PCs retiring once per 64-lane register, flanked by an `RV_VMULS` against
`0xbf800000` and an `RV_VABS_FP` — the 0-ULP FP32 `Div`, three fused
multiply-adds per register, twice in the epilogue. Details, the per-engine
census against a pre-registered budget, and what the runs do **not** establish
— no silicon, no grid past the physical AIV count, no ragged tail, no MTP — are
in [`reports/single_token_decode_aiv.md`](reports/single_token_decode_aiv.md).

The **Split-K synchronisation probe** in
[`kernels/camodel/l2_atomic_synthetic.cpp`](kernels/camodel/l2_atomic_synthetic.cpp)
answers where the cross-page reduction of the Cube decode operator should live.
That operator emits a *partial* softmax per 128-token page and the combine is
not in it, so a multi-thousand-token context needs one; simulating a full
sequence is cycle-prohibitive, so the question is answered by composition
instead. Three modes over the same state record — `m` and `l` as replicated
32-byte DataBlocks, `Acc` as 256 FP32 moved in one 32-burst transfer — and a
second runner over the **frozen** decode operator for the other term:

```sh
bash scripts/run_l2_atomic_probe_camodel.sh            # the K sweep, modes A/B/C
bash scripts/run_l2_atomic_probe_camodel.sh smoke      # one cheap rung, to time it
bash scripts/run_l2_atomic_probe_camodel.sh baseline    # the Cube strip baseline
bash scripts/run_l2_atomic_probe_camodel.sh all         # both, then the crossover
```

It is the only kernel here with a SIMT VF, and that is forced: there is no SIMD
atomic on this part. `asc_atomic_exch`, `asc_atomic_max` and `asc_atomic_add`
are declared `__simt_callee__`, reachable only from a `__simt_vf__` that
`asc_vf_call` dispatches — and `simt_api/common_functions.h` makes
`asc_vf_call` expand to **nothing** unless `SPLIT_CORE_VEC` is defined, which
at `__NPU_ARCH__ 3510` happens only in the AIV leg. A lock written into a
Cube-side code path would have been silently deleted and would have raced,
which is why the probe is `KERNEL_TYPE_AIV_ONLY` and why the trace audit
asserts the VF dispatch count rather than trusting it.

The results invert the premise the task started from. The Cube's own cost for
one 16-token strip is **180 ticks**, not 40,000; the rate at which the pipeline
*retires* a strip is **19,905** ticks, and that interval is set by the AIV0
recurrence, which is in flight 234,863 ticks against the Cube's 1,440 over the
same eight-strip run. On the synchronisation side every phase except the spin
is **flat in K** — the critical section is 1,487–1,530 ticks whether one split
or eight address the same line, and so are the atomic maximum and the Global
Memory bursts — so on this model the cost of sharing a line is serialisation
and nothing else. The crossover follows: a collision-free per-split workspace
is under the 3% target at **one strip**, while an in-place locked combine needs
**224 tokens** per split at K = 2 and **752** at K = 8. And 71% of the
uncontended 5,288-tick floor is three SIMT VF dispatches, not the atomics —
which is the term to attack, and it does not depend on K.

Two things the trace said that the runner could not. Under
`KERNEL_TYPE_AIV_ONLY` an AIV is **not** an AI core: each 950PR AI core carries
two Vector cores and the model dumps them separately, so split `k` runs on
`core<k/2>.veccore<k%2>` and not `core<k>.veccore0` as the MIX mapping would
have it — the audit now asserts that with a census over every dump, because the
first version reported an untouched core for every odd split. And mutual
exclusion is proved twice over, independently: the runner checks that the
acquisition tickets form a permutation, while the audit checks from the **data**
instructions that no two splits' critical-section windows overlap in time. The
measured crossover tables, the one model diagnostic that is *not* resolved, and
what these runs do not establish — no silicon, no bus-level record in the
dumps at all, and an exclusion argument that the measurement barriers partly
subsume — are in
[`reports/l2_atomic_contention_split_k.md`](reports/l2_atomic_contention_split_k.md).

The prefill counterpart in
[`kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp`](kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp)
is a **four-stage** `MIX_AIC_1_2` ring over the eight chunks of one 16-token ×
1024-channel strip: MTE2 stages each raw INT8 page from Global Memory into the
L1 ring with no Unified Buffer hop, the Cube applies the **whole** randomised
Hadamard rotation as 64 INT8×INT8→INT32 Mmads per chunk and hands the finished
panel over in one Fixpipe, both Vector cores only reduce, scale and pack to
INT4, and MTE3 writes a 2 KiB packed page plus a 64-byte scale slice per chunk.
Eight chunks over a depth-4 L1 ring and depth-2 per-core UB rings wrap every
ring twice, so one run contends every buffer-reuse back edge.

The rotation sits entirely on the Cube because the first revision split it —
Cube inner factor, Vector sign diagonal and butterfly cascade — and measured
the result: the Cube waited on a Vector core for 33.6% of its attributed cycles
against 2.8% spent multiplying, while the Vector cost was register traffic
rather than arithmetic. Folding the outer factor's signs into 16 resident
32×32 INT8 blocks costs eight times the multiply-adds and buys back four
full-panel Vector passes and 65 KiB of Unified Buffer; the trade and what it
measured are in the report.

```sh
bash scripts/run_modulo3_camodel.sh micro reshape_and_cache_a8
```

Its quantiser uses a power-of-two scale extracted by integer ops on the FP32
bit pattern, so the whole path from INT8 input to packed nibble carries exactly
one rounding step and the KV cache is checked **bit-exactly against an integer
oracle for an arbitrary input**, not against a tolerance or a crafted fixture.
Its Tier-1 probe checks what that comparison cannot — that the dense
256×256 rotation satisfies `M Mᵀ = 256 I`, that the oracle's factored form
reproduces it exactly, and that the 16 resident blocks the Cube multiplies
compose back to all 65536 entries of it:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    tests/cpu/test_reshape_and_cache_a8_probe.cpp \
    -o bin/test_reshape_and_cache_a8_probe
bin/test_reshape_and_cache_a8_probe
```

Results, including what the trace does and does not establish, are in
[`reports/reshape_and_cache_a8.md`](reports/reshape_and_cache_a8.md).

The BF16 instance of that operator raised a data-movement question the INT8
one could not: the arch-3510 fractal is 16 × (32B/sizeof(T)), so C0 drops from
32 to 16 and a 32-wide axis stops being one fractal. The single-tile probe in
[`tests/camodel/test_micro_layout_bf16.cpp`](tests/camodel/test_micro_layout_bf16.cpp)
answers it on one AIC, sixteen Mmads and no synchronization at all:

```sh
bash scripts/run_micro_layout_bf16_camodel.sh
```

The 16 × 256 activation page's Nz/Zn duality **survives** the widening, so the
Value chunks need no new load path. The 32 × 32 resident rotation block's does
**not**: a row-major blob stops being its own Nz image, and the probe
reproduces the resulting misread bit-exactly from the index map before showing
that an ND→Nz ingress removes it. MTE1 cannot repair this — its copy unit is
the 512-byte fractal and its only rearrangement is an intra-fractal transpose —
so the fix is at ingress: `stage_blocks_nz` in
[`routines/routine_stage_page_zn_bf16.hpp`](routines/routine_stage_page_zn_bf16.hpp),
with the four byte-identity L1→L0 loads in
[`routines/routine_fractal_load_zn_bf16.hpp`](routines/routine_fractal_load_zn_bf16.hpp).

That ingress is now the operator's prologue, and the operator is bit-exact on
CAModel at 1, 2 and 3 strips:

```sh
bash scripts/run_reshape_bf16_camodel.sh
```

With the flat copy put back it fails on the same runner, which is what makes
the fix measured rather than argued. The report also records why the failure
looked Value-specific — the runner caps its comparison at three chunks and the
value heads come first, so the equally broken key chunks were never printed.
Details and the gates in
[`reports/micro_layout_bf16.md`](reports/micro_layout_bf16.md).

`routines/routine_cube_hadamard16.hpp` dispatches one FP16×FP16→FP32
`Mmad` followed by one L0C→GM `Fixpipe` copy. The caller precomputes the
canonical Sylvester matrix as four Kronecker `H2` factors and stages the
resulting 16×16 FP16 tile in L0B/ZN. One dense Cube instruction computes
`X × H16`; there are no scalar elementwise operations in the AIC leaf.

## Native caller contract

- Product/API: Ascend 950PR, arch35, CANN 9.2 Tensor API.
- L0A: 16×16 FP16 NZ, 512-byte aligned; L0B: 16×16 FP16 ZN, 512-byte
  aligned; L0C: 16×16 FP32 NZ, 64-byte aligned.
- Output: 16×16 ND in GM, 32-byte aligned. Select `Output=float`, `half`, or
  `bfloat16_t` with `OutflowMode::Fp32`, `Fp16`, or `Bf16` respectively.
- The caller completes GM→L1 conversion and L1→L0 loads before invoking the
  leaf. It also waits for GM output visibility after the leaf returns. These
  lifecycle barriers belong outside the leaf.
- The recommended L1 layout has X at byte 0 and H at byte 544: 512 bytes
  for X plus a 32-byte bank skew pad. Do not alias the operands or output.
- The single `Mmad` and `Fixpipe` use matching `unitFlag=3` to enforce the
  L0C producer/consumer dependency without explicit synchronization calls.

The matrix is mathematically factorized when prepared; the runtime performs
one dense 16×16 Cube multiply. Four separate Cube butterfly stages would
need intermediate casts and inter-stage dependencies, so they are not the
operation modeled by this leaf.

## Host probe

```sh
mkdir -p bin
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    tests/cpu/test_hadamard16_probe.cpp -o bin/test_hadamard16_probe
bin/test_hadamard16_probe
```

The C++17 testbed has guarded aligned mock GM/L1/L0/UB buffers and models the
same one-Mmad/one-Fixpipe dispatch. The CPU scalar oracle independently
computes `X × H16` and checks all 256 results bit for bit in FP32, FP16 and
BF16 for five input patterns. Input values are dyadic FP16 numbers so their
signed 16-term FP32 sums are exact and insensitive to reduction ordering.
FP16 and BF16 cast checks use round-to-nearest-even.

The host timing includes staging, golden reference, execution and checks; it
is not AIC latency. The structural cycle model is 2 GM→L1 transfers,
2 L1→L0 transfers, 1 `Mmad` (4096 multiply-accumulates, 8192 FLOPs), and 1
Fixpipe drain. With stage rates in bytes/cycle for each path and FLOPs/cycle
for Cube, a serial single-tile model is
`1024/B_GM_L1 + 1024/B_L1_L0 + 8192/P_Cube + O/B_Fixpipe` cycles.
For a steady stream of independently buffered tiles, the throughput interval
is bounded below by the maximum of those four terms. `O=1024` bytes for FP32
and `O=512` bytes for FP16/BF16. Startup, drain, bank effects and caller
synchronization add to the serial model. A numeric
hardware cycle estimate requires 950PR issue/bandwidth data or a device
trace. The native branch also requires a CANN 9.2 BiSheng toolchain and
950PR hardware for compilation and execution validation.
