# Refining combat kernels: mandatory pipeline baseline

Applies to pipeline or buffer-lifetime refactors of real operators in this
repository. Target for the decode baseline: CANN 9.2.0-beta.2, `dav-3510`,
`Ascend950PR_9599`. Complete these gates before pushing a pipeline refactor
or tagging it for physical NPU execution. Section 6 adds the gates a
MIX_AIC_1_2 operator needs when its grid can exceed the physical AI Core
block count and so runs in several execution waves; it applies in
addition to sections 1-5, not instead of them. A failed or unavailable
gate is recorded as pending; compilation or a host mock cannot replace CAModel.
This procedure complements `SKELETON_KERNEL_WORKFLOW.md` and
`MICROROUTINE_DESIGN_GUIDELINES.md`.

## 1. Prove the token flow and simulate engine queues

Draw the marked graph before changing events. A HardEvent is one bit, so each
`(source, destination, ID)` place has capacity one. Every place has one producing
transition and one consuming transition. Pair routes and IDs exactly, use IDs
0..5, prime every feedback cycle, and explicitly drain every final credit,
including credits for slots unused by short or empty workers. A second SET
before consumption loses a token even if aggregate SET/WAIT counts match.

For the two-slot decode ring (`kStageDepth = 2`), for each slot `s`:

| Place | Route/ID | Initial marking | Producer → consumer |
| --- | --- | ---: | --- |
| Packed free | `V_MTE2`, `s` | 1 | dequant → packed ingress |
| Scale free | `V_MTE2`, `2+s` | 1 | recurrence → scale ingress |
| Ingress ready | `MTE2_V`, `s` | 0 | ingress join → dequant |
| Archive free, AIV only | `MTE3_V`, `s` | 1 | archive → GEMV |
| Archive ready, AIV only | `V_MTE3`, `s` | 0 | GEMV → archive |

```mermaid
flowchart LR
  PF["Packed free: 1"] --> L["MTE2 ingress join"]
  SF["Scale free: 1"] --> L
  L --> R["Ready: 0"] --> D[Dequant]
  D --> PF
  D --> G[GEMV] --> C["Online recurrence, in sequence order"] --> SF
  AF["Archive free: 1"] --> G
  G --> AR["Archive ready: 0"] --> A["MTE3 archive"] --> AF
```

The graph is instantiated twice. Vector queue order also supplies edges between
dequant, GEMV and recurrence transitions and serializes all updates of `m/l/acc`.
Dequant/GEMV and softmax share one Vector ALU; they cannot execute simultaneously.
Overlap means MTE2 and/or MTE3 execute while Vector is in flight.

Prefill `min(N, 2)` slots. At tick `i`, prepare strip `i` if it exists, retire
recurrence `i-1` if it exists, then enqueue ingress `i+1` into the retired slot.
The final tick only recurs the final strip. Each packed/scale ownership credit
is either free or owned by exactly one strip. A recurrence releases scales only
after their last Vector read; dequant releases packed bytes after both unpacks.
Each archive credit alternates free → score → DMA → free. One final WAIT moves
each remaining feedback credit to a terminal sink, leaving all hardware flags
zero. Forward ready places are consumed once for each actual strip.

An induction over slot generations proves 1-safety: a refill requires the
previous generation's credit, so it cannot overwrite that generation; the next
release requires consumption of the preceding ready token. Every directed
reuse cycle is initially marked. The finite prefill/prepare/retire schedule
contains no wait for a transition that will never be emitted. Exhaustive engine
interleaving exploration additionally checks bounded instances and ownership;
it is not a claim of exhaustive verification for every possible sequence length.

Run `python3 tests/cpu/test_decode_ring_model.py`. It explores every interleaving
for both kernels at N=0,1,2,3,4,5,8,16,64, rejects double SETs, unprimed cycles
and early scale release, and requires ordered recurrence and an empty final
marking. Keep model transitions consistent with the actual loop when editing it.

## 2. Compile-time memory checks

Compile the actual kernels with `-Wall -Werror`. Enforce 32-byte bases and DMA
sizes, non-overlapping live allocations, a 32-byte allocation skew between ring
slots, and different `floor(offset/32) mod 16` banks for paired slots (AKA3003).
Skew is checked against payload size; it is not a universal bank-conflict proof
for every instruction's repeat/stride pattern.

Packed keys/values, key/value scales, dequantized key/value pages, and raw score
panels have two slots. The nibble conversion buffer, GEMV temporaries and
recurrence temporaries are shared only in Vector FIFO order. No MTE engine
writes them during a Vector live range. Assert
`kArenaEnd + 256 <= kUbWindow <= 192*1024`; the final 256 bytes are an addressable
read pad, never payload. Give each 64-byte scale slice a private 256-byte landing before applying the
32-byte skew; an unmasked register read must not touch another DMA landing.
Check register overreads of short state/fold buffers too.
Keep scalar tensor extraction and `GetValue()` out of arithmetic; Split-K merges
all statistics together and streams accumulators in vector-register batches.
Scalar loop/address scheduling, block-table GM address loads and lock-control
branches are control flow, not scalar dequant/GEMV/softmax computation. Do not
introduce null TPipe calls or per-row scalar/MTE accumulation handshakes.

## 3. Isolated single-tile boundary canaries, before full sequences

Use one tile of the operator's real geometry. These decode kernels use
256 channels × 16 tokens, rather than changing their contract to an illustrative
128 × 128 shape. Test logical `page_idx=0`, `K/2`, and `K-1` with K=8 and a
non-identity block table. Keep the full cache/table allocations and nonzero
logical-page indices; rebasing the selected page to zero does not test offsets.
Also test the final strip within the final page (strip 7).

Flood every GM payload with `0x7FC0BEEF` before assigning only the selected
query/gate/cache/scale/table slices. This is an FP32 quiet NaN; packed int4 and
BF16 views are byte patterns, not guaranteed NaNs. Thus canaries detect writes,
and numerical checks detect wrong reads; do not claim that every int4 read
outside a slice necessarily produces a NaN. Signs are a shared real input.
Snapshot every input byte before upload, download after synchronization and
compare it bitwise. Verify every unassigned output slot/channel and both exterior
guards (`ASSERT_CANARY_INTACT`). For BF16, check the exact low/high half at its
original word position; accepting either half misses a halfword swap.

The exact boundary fixture uses query=1/256, key nibbles=3, scales=1 and gate=0.
Every score is exactly 3 and exp(0)=1. Values depend on logical strip and channel.
Values use a positive even baseline and one odd channel, keeping both division
numerators and dense rotation outputs nonzero while preserving exact arithmetic.
An independent scalar token sum and dense rotation matrix supply the oracle.
Require byte-exact scores, replicated m/l, accumulators, rotation and BF16 output
where exported. Split-K's production interface exports context and protocol
metrics only; its isolated probe checks exact context, protocol fields and all
unassigned bytes. It calls the same `ProcessItem` and reduction as production.
The regular varied-score fixture remains a separate arithmetic regression with
its documented Exp/reduction tolerances; do not relax exact boundary checks to
make that fixture pass.

## 4. Full-depth and wrap tests

Only after all three boundaries pass, test N=depth and N>=2*depth (here 2 and 4),
then odd counts, multiple pages, repeated launches and real Split-K workers,
including empty workers. Boundary isolation does not prove contention/reuse.
Preserve the same canary and oracle gates as sequence length grows. Do not infer
the tail-worker merge path from a single-worker probe.

## 5. Trace audit and instruction census

Use each run's own `core*.instr_popped_log.dump` and completion logs. Audit every
core, require real Vector work on at least one, and fail on missing traces,
timeouts, model errors or nonempty exception dumps. Key events by operand route
and ID, not by the queue column. Require equal counts, SET/WAIT alternation in
program order, capacity-one markings in completion order, and zero final tokens.
CAModel omits some register barriers from completion logs; flag completion
counts must nevertheless match dispatch counts exactly. At N strips, slot s
must have `1 + floor((N+1-s)/2)` SET/WAIT pairs on each packed/scale feedback
route; the leading one is the prologue/epilogue credit, even for unused slots.

Run `scripts/analyze_decode_ring_trace.py`, the AIV instruction/traffic census,
and the Split-K batched lock audit. Report engine counts and actual intersecting
MTE2/Vector intervals for an overlap claim; source order and balanced events
alone do not establish overlap. Do not interpret simulator ticks as silicon
throughput. Keep commands, logs, counts, canary totals, numerical results and
limitations in a report under `reports/`.

The reproducible boundary baseline is:

```bash
bash scripts/run_decode_pipeline_camodel.sh
```

It creates a fresh ext4 source/build/evidence directory under `/tmp`, prints its
path, runs host probes and strict compilation, then tests both kernels at the
three boundaries, depth=2 and wrap=4. The existing varied-score and production
Split-K runners provide the subsequent full-sequence regressions.

For the subsequent varied-score regressions, build the existing
`bench_decode_splitk_sim` target in a fresh CMake build with the same toolkit
and SoC, then run:

```bash
# Build the simulator benchmark from current sources without launching it.
decode_bench_dir=$(mktemp -d /tmp/ascend-decode-bench.XXXXXX)
SPLITK_SRC_DIR="$decode_bench_dir/src" SPLITK_BUILD_DIR="$decode_bench_dir/build" \
  bash scripts/run_decode_splitk_bench.sh sim --help
bash scripts/run_decode_pipeline_regressions.sh BASELINE_DIR "$decode_bench_dir/build"
python3 scripts/verify_decode_pipeline_evidence.py BASELINE_DIR
```

`BASELINE_DIR` is the printed `/tmp/ascend-decode-ring.*` directory. The script
requires all ten baseline passes before running the six-row, two-page, odd-tail,
K=2 wrap/merge and K=4 empty-worker cases. It uses the baseline kernel libraries
with the current benchmark ABI, checks model errors/exception dumps and audits
each case's own traces. Full varied-score arithmetic uses its existing documented
numerical budgets; exact boundary outputs remain a bitwise gate.

## 6. Inter-wave gates for a MIX_AIC_1_2 grid

Applies to any pipeline refactor of a `KERNEL_TYPE_MIX_AIC_1_2` operator whose
grid can exceed the physical AI Core block count, which is every real prefill
or paged operator at production shapes. The protocol itself is
`MICROROUTINE_DESIGN_GUIDELINES.md` §6.5; these are its gates, and they are in
addition to §1–§5, not instead of them.

Above the block count the scheduler runs the grid in waves and a physical block
executes a second grid block after its first. The mode-4 intra-block flag
registers are local to that block and are not cleared by the hardware
scheduler between waves, and nothing drains the block's engine queues either.
Every address such a kernel uses is a compile-time constant, so wave 1 reuses
wave 0's L1 ring slots, landing panels and egress slots.

**A launch window is not a fix for this; it is a way of not testing it.**
Driving a wide context as several capped launches re-enters the kernel at every
wave boundary, which re-arms everything. Remove every `--window`, cyclic slice
and staggered sub-launch before claiming anything about multi-wave behaviour,
and treat any such parameter found in a harness as an open finding.

### 6.1 Prove the boundary guard, in the same translation unit the kernel uses

`total_strips = numTokens / tokensPerCore`, computed identically by all three
roles, with the early return taken **before** `InitSocState`, before any tensor
or SOC state exists, before a feedback cycle is primed and before any GM
address is formed. A block past the end has no `slotMapping` entry; reading it
kills an AIV and deadlocks that block's Cube on its next `CrossCoreWaitFlag`,
so this is a hang gate, not an accuracy gate. A zero `tokensPerCore` must guard
every block rather than fault the scalar unit.

Put the arithmetic in a header with no CCE dependency, include it from the
kernel, and compile that header from a host translation unit with
`-Wall -Wextra -Werror`. This is the one part of §2 that survives on a machine
with no CANN toolkit, and it is only worth anything if the kernel and the host
gate share the declaration rather than each carrying a copy.

### 6.2 Make the cross-core id budget a compile-time object

Not a comment, and not inspection. The legal mode-4 space is 0..10 per sub
block (11..14 and their +16 images are library reserved), so a budget
expression plus one `static_assert` is what stops a longer schedule from
silently renumbering a channel into id 13.

Then audit the census over the actual schedule, per policy: every id legal,
sets and waits balanced per id, the two sub blocks in disjoint halves, and
**at most one outstanding notification per id** — the per-id operation
sequence alternating `SET, WAIT, SET, WAIT`. §6.2c's finding is that an
unfenced announcement on a recycled id balances every set/wait count and still
releases a wait early, so a balance check alone is not this gate; the tick-order
check and the fence audit of §6.6 are the rest of it.

### 6.3 Model two waves, not one, and leave the flag semantics as a knob

Extend §1's capacity-one token/queue model to the whole physical block — Cube
plus both Vector cores — with a scalar stream that dispatches into in-order
engine queues in program order and stalls on cross-core waits, and with
generation ownership on every L1 slot, landing panel, packed slot, L0 page
region and L0C. Then run **two** grid blocks on it, carrying the marking
across, because that is what a multi-wave grid does.

Two hardware properties decide whether channel recycling is usable at all,
and a model that assumes either has assumed the answer: whether
`wait_intra_block` consumes the flag it tested, and whether kernel entry clears
the block's mode-4 registers. Keep both as knobs and report the verdict for
each combination; they do not agree. For Ascend 950PR under CANN 9.2 the
consuming half of that table is the measured one (§6.2c's probe), but the model
keeps all four because the sticky rows are the signature to recognise if a
toolkit revision reintroduces the fault, and because the next part is not this
part.

`tests/cpu/test_reshape_wave_model.py` is the instance of this for the
reshape-and-cache operators. Exhaustive exploration is for bounded instances;
say so, and say which geometries were only sampled.

### 6.4 Audit the epilogue drain as a census, not as a comment

Per role, every asynchronous queue it owns reaches the scalar stream through
one `SetFlag<HardEvent::x_S>` / `WaitFlag<HardEvent::x_S>` pair before the
kernel body returns, and the Vector cores wait for the Cube's drain first
because the Cube is the only core writing their landing rings. There is no
`MTE1_S`; MTE1 is covered transitively by the `M_MTE1` wait behind the Mmads.
Drain an idle queue too — "this queue is empty" is the thing the wave boundary
needs established. In the trace this appears as the last `x_S` pair on each
engine of each core, with nothing issued after it but the cross-core join.

### 6.5 Run a ladder with a ragged second wave, in single dispatches

`grid = 1, 32, 33, 48, 64, 128` strips, each ONE kernel launch:

```bash
./build/lib/test_turboquant_a8_validation    --strips 1 --strips 32 --strips 33 \
                                             --strips 48 --strips 64 --strips 128
./build/lib/test_turboquant_bf16_validation  # same ladder by default
```

33 and 64 fail differently and both rungs are required: 33 is a *partial*
second wave immediately after a full one, which catches a leak the first reuse
exposes; 64 is a complete reuse cycle with no ragged tail. 48 is a ragged
second wave that is not minimal. 128 gives a leak that needs more than one
reuse somewhere to show. Verify that strip 33 — the first block of wave 1 — and
every later strip are bit-exact against the integer Hadamard oracle, not merely
that the launch returned.

Keep the pages fragmented (`numBlocks = 2 * strips + 2`, strip `s` → page
`2s + 1`) so a stray write lands on a canary page and the verifier can name the
strip whose data arrived, which is the signature of a block-index error and is
invisible in an aggregate byte count.

### 6.6 Ship both channel policies, and gate the fence the recycled one needs

Select the policy with a non-type template parameter and emit two `__global__`
symbols from one body, differing in **nothing but the id allocation** — same
schedule, same wait placement, same priming, same drain — so a behavioural
difference isolates recycling. Expose it in the harnesses
(`--mode4-single-shot`, or a build define for a whole CI job). The recycled
ring is the default for these operators; keep the single-shot fallback
dispatchable, because it is the bisection control for any later cross-core
fault and the one policy no reading of the flag mechanism can break.

**The recycled policy's precondition is a fence, so audit it.** Every
announcement must be preceded by its `SetFlag`/`WaitFlag<HardEvent::x_S>` pair
on the pipe whose work it reports; a priming set is the only exemption.
`scripts/audit_cross_core_fences.py` checks this over the kernel sources with
negative controls, and it is a gate of its own because nothing else here sees
it: counts balance without it and the numbers can be bit-exact while a panel is
read a generation early.

A green run of either variant promotes nothing. Both revisions of this operator
that recycled ids *without* the fence produced bit-exact caches and intact
guards while their traces showed a second wait passing before its set, by 1032
and 2179 cycles. What settled the semantics was an isolated probe —
`tests/research/kernel_cluster_sync_probe.cpp`, one physical block, no data,
recycled ids, fenced announcements, run on silicon and on CAModel — plus a
trace audit that checks **tick order per id**, not per-id counts. Ship the
probe as a target in both modes, not as a transcript, so a toolkit regression
is reproducible; and record in `reports/` which leg produced which trace, since
a data-free probe that merely exits cleanly does not distinguish a consuming
latch from a sticky one.

### 6.7 Record the unavailable gates explicitly

Compilation of these kernels needs BiSheng and a CANN 9.2 toolkit, and the
boundary canaries, depth/wrap tests and trace audit need the
`Ascend950PR_9599` simulator. On a host without them, the host-side gates
above are the whole of the evidence, and a report that does not say so is
making a hardware claim it has not earned. List each gate as run, failed or
unavailable, with the reason.
