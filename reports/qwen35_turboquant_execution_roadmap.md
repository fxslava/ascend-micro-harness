# TurboQuant on Qwen3.5-27B — execution roadmap for the six-kernel split

Ascend 950PR / arch35 / `dav-3510` / CANN 9.2. Model: Qwen3.5-27B, 16 full-attention
layers, `H_q = 24`, `H_kv = 4` (GQA 6:1), `d = 256`, `attn_output_gate = true`, bf16,
vLLM page 128 tokens = 8 strips of 16.

Companions: [`qwen35_turboquant_regime_audit.md`](qwen35_turboquant_regime_audit.md) is the
evidence base (findings cited as F*n* / §*n*), [`qwen35_turboquant_rfc.md`](qwen35_turboquant_rfc.md)
the interface specification. This document is the execution plan: who writes what, in what
order, against which gate.

**No NPU on this host.** Every acceptance gate below is CAModel plus host oracle. No phase
may claim a latency, throughput or power number; the measured figures quoted are
cycle-accurate model counts from the two existing reports.

---

## 0. The six kernels, and where each one comes from

Two of the six exist. That distinction drives the whole schedule, because an adaptation
inherits a verified oracle, a canary protocol and a trace audit, while a new kernel has to
earn all three.

| # | Operator | Engine | Kernel type | Provenance |
| --- | --- | --- | --- | --- |
| №1 | `turboquant_reshape_scatter_aiv` | AIV only | `KERNEL_TYPE_AIV_ONLY` | **NEW** |
| №2 | `turboquant_reshape_aligned_cube` | 1 Cube + 2 AIV | `KERNEL_TYPE_MIX_AIC_1_2` | **ADAPT** [`turboquant_reshape_and_cache_bf16.cpp`](../kernels/ops/turboquant/turboquant_reshape_and_cache_bf16.cpp) |
| №3 | `turboquant_single_token_decode_aiv` | AIV only | `KERNEL_TYPE_AIV_ONLY` | **NEW** |
| №4 | `turboquant_paged_decode_cube` | 1 Cube + 2 AIV | `KERNEL_TYPE_MIX_AIC_1_2` | **ADAPT** [`turboquant_paged_decode_attention.cpp`](../kernels/ops/turboquant/turboquant_paged_decode_attention.cpp) |
| №5 | `turboquant_rotate_q_batch_cube` | 1 Cube (+AIV idle) | `KERNEL_TYPE_MIX_AIC_1_2` | **EXTRACT** from №4's prologue leaves |
| №6 | `turboquant_rotate_vector` | AIV leaf (routine) | n/a — header | **NEW**, derived from [`routine_vector_egress_rotate.hpp`](../routines/routine_vector_egress_rotate.hpp) |

`KERNEL_TYPE_AIV_ONLY` is documented for 950PR (`set_kernel_type.md`, `KernelMetaType`) and
exists "to control operator execution so that only cores of the specified type are started,
avoiding starting cores that do not need to work and reducing the core startup overhead".
The harness already has an `KERNEL_TYPE_AIC_ONLY` precedent in
`kernels/camodel/wta_fixpipe.cpp:80`.

### 0.1 Entry points, and how this supersedes the RFC's operator inventory

The six-kernel split renames two of the RFC's operators and adds two, so
[`qwen35_turboquant_rfc.md`](qwen35_turboquant_rfc.md) §2 and §4 are superseded by this
table. The RFC's §3 layout contract, §4.1 adapter contracts and §9 Python integration all
still stand — only the inventory moved.

| # | Entry point | RFC §2 | Schema status |
| --- | --- | --- | --- |
| №1 | `npu_turboquant_reshape_scatter` | O2 `..._decode_cache_update`, widened to cover ragged prefill tails | rename + widen RFC §4 schema |
| №2 | `npu_turboquant_reshape_aligned` | O1 `..._reshape_and_cache` | rename; RFC §4 schema holds, `cu_seqlens` already in it |
| №3 | `npu_turboquant_single_token_decode` | — | **new schema needed** |
| №4 | `npu_turboquant_gqa_paged_decode` | O3 | RFC §4 schema holds, minus the rotation operands №5 now owns |
| №5 | `npu_turboquant_rotate_q_batch` | — | **new schema needed** |
| №6 | — (header leaf, no operator) | — | n/a |
| — | `npu_turboquant_decode_combine` | O4 | RFC §4 schema holds |
| — | `npu_turboquant_decode_workspace_size` | O5 | unchanged |
| — | `npu_turboquant_vector_core_num` | O6 | exists on `feat/tq-main-clean` |

Eight registered entry points, six kernels. The two schemas marked **new** are a Phase 5
prerequisite and are the one piece of interface design this roadmap does not carry: №3's
operand list is №4's minus `workspace`, `temporal_splits` and the Cube-side tables, and №5's
is `(query, pi_signs, tables, query_rot, num_heads, head_size, rows, inv_sqrt_len)` — but
both should be written into the RFC before Phase 5 starts, not invented at the adapter.

### 0.2 The two design rules, and what they are allowed to cost

**Cube-centric (№2, №4, №5): pipeline overlap over saturation.** Fractal fill is not a
target. `Mmad.md` states that when M is not a multiple of 16 "the hardware still organizes
data in 16x16 fractal blocks. Invalid data in the tail block occupies fractal block space
but does not participate in valid computation", and `MmadParams::m` is a plain `uint16_t`
runtime extent — so `m = 6` costs one fractal, exactly as `m = 16` does. There is no
`SetTail` on this path (that belongs to the high-level `Matmul` API, which none of these
kernels use), no masked accumulation, and no host-side query padding beyond a guard band.
The target is zero bubble stalls on a bounded, live marked graph.

That rule is backed by measurement, not preference. On №4's 8-strip rung the Cube is in
flight **1,777 ticks** against AIV0's **234,863** and AIV1's **2,457**; the per-strip score
interval is ~19,900 ticks while the staging core finishes its 16-burst key page in ~980. The
Cube is idle 99.2% of the schedule, so Cube work is free and Vector work is the whole
budget.

**AIV-centric (№1, №3, №6): saturation over pipelining.** No cross-core flags, no modulo
ring, no CV transfer — straight-line code, contiguous bursts, in-register shuffles.

One honest qualification on "near-100% SIMD ALU utilisation": the measured Vector profile of
this operator family does not reach it and is not expected to. On the writer's rebalanced
run `RVECEX` is 28.1% of Vector cycles while `RV_VLD + RV_VST` is 38.4% — register traffic,
not arithmetic, is the dominant term, and an int4-unpack-and-quantise kernel is load/store
shaped by nature. So the acceptance criterion for №1/№3 is **an instruction census and a
per-engine cycle breakdown against a stated budget**, not a utilisation percentage that
cannot be met. Where a pass can be removed, remove it; do not gate on a number the hardware
will not give.

### 0.3 The dependency that contradicts the phase order

№1 (Phase 1) consumes №6 (Phase 2): the scatter writer's `Pi` transform *is* the Vector
butterfly. №6 is a header leaf, not an operator, so hoisting it does not disturb the
operator phasing — **implement №6 first, as Phase 1 item 0.** Everything else follows the
requested order.

```
№6 (leaf) ──┬──► №1 ──┐
            │         ├──► Phase 5 ingress dispatch
№2 ─────────┴─────────┘

№5 ─────────┬──► №3 ──┐
            └──► №4 ──┴──► Phase 4 combine ──► Phase 5 decode dispatch
```

№2 and №5 have no dependency on each other and can proceed in parallel. №3 and №4 both
consume №5's `query_rot` and both need №6 for their egress.

---

## 1. Phase 1 — ingress engines (№6 leaf, №1, №2)

### 1.0 №6 first: the shared butterfly leaf

Files:

| Path | Change |
| --- | --- |
| `routines/routine_vector_rotate_butterfly.hpp` | **new** — the two-direction leaf |
| `routines/routine_vector_egress_rotate.hpp` | refactor onto it; keep its published entry points |
| `tests/cpu/test_rotate_butterfly_probe.cpp` | **new** — host probe against the dense `256 x 256` matrix |

`Pi = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32) / 16 = A . D . B`, with `A` the strides
`{32, 64, 128}` and `B` the strides `{1, 2, 4, 8, 16}`. **The two directions are not the
same stage order**, and this is the single most error-prone line in the phase:

| Direction | Used by | Expansion | Stage order | Sign after |
| --- | --- | --- | --- | --- |
| `x . Pi` (right) | egress of №3, №4 | `((x.A).D).B` | `32, 64, 128, 1, 2, 4, 8, 16` | stage 3 |
| `Pi . x` (left) | ingress of №1 | `A.(D.(B.x))` | `1, 2, 4, 8, 16, 32, 64, 128` | stage 5 |

The right-multiplication order is what `routine_vector_egress_rotate.hpp` already ships
(`kEgressStrideOf`, `kEgressSignBeforeStage = 3`), verified against the dense
right-multiplication by the Tier-1 probe. The left-multiplication table is new. A stage at
stride `s` is `out[i] = in[i ^ s] + sgn_s(i) . in[i]`, `sgn_s(i) = (i & s) ? -1 : +1`.

Queue allocation: Vector only, one UB ping-pong pair of `d` fp32 lanes.

Cost structure, carried over unchanged and worth restating because it sets №1's budget:
five of the eight stages (`s >= 8`) are two repeat-form instructions over contiguous runs —
`Add` on the lower half of each `2s`-lane group, `Sub` on the upper, second source offset by
`s * 4 >= 32` bytes and therefore aligned. Three stages (`s in {1, 2, 4}`) have the partner
lane inside the 8-lane fp32 DataBlock, where a 4/8/16-byte tensor offset is not 32-byte
aligned and the masked form cannot reach it either, so they pay a `Gather` plus a sign
multiply. Offset and sign vectors: three of each, 256 entries, **6 KiB**, loaded once — set
against the 16 KiB a single 4096-entry offset vector would have cost.

**Gate 1.0.** Host probe: both directions reproduce the dense `256 x 256` multiplication
entry for entry over a dyadic fixture (exact in fp32, so bit equality is demandable);
`Pi(Pi(x)) == x` to 0 ULP for both orders; the ingress table composed from the egress table
reversed, asserted equal to the independently written one.

### 1.1 №1 `turboquant_reshape_scatter_aiv` — new, AIV only

Serves **1..16 tokens**: per-step decode writes (`N = B x (1 + k)`) and every request's
ragged prefill tail (`n < 16`). This is the operator the audit's F3 says has no counterpart
today — `reshape_and_cache` is called on every step in every attention state
(`attention_v1.py:1326`) while №2's contract is `num_tokens % 16 == 0` with 16-aligned
consecutive slots, which a decode step cannot satisfy.

Files:

| Path | Change |
| --- | --- |
| `kernels/ops/turboquant/turboquant_reshape_scatter_aiv.cpp` | **new** |
| `routines/routine_vector_icdf_quantize.hpp` | extend: runtime lane count, `absmax == 0` branch |
| `tests/camodel/test_camodel_reshape_scatter_aiv.cpp` | **new** |
| `tests/common/turboquant_oracle.hpp` | extend: partial-strip and RMW reference |

**Unit of work: `(request, kv_head)`, not `(token, kv_head)`.** All `1 + k` tokens of one
request share a strip, so folding them into one work item makes the value-plane
read-modify-write a single transaction and removes any need to assert that no two work
items collide on a strip — which `(token, kv_head)` would require and which MTP
(consecutive tokens, same strip) violates.

Grid: `min(num_requests * H_kv, GetBlockNum())`, grid-stride. Straight-line body per item:

```
MTE2   K and V rows for this (request, kv_head)      (1+k) x 256 bf16, contiguous
Vector Pi . x via №6 left-direction, per row         8 butterfly stages
Vector absmax over 256 lanes, power-of-two exponent, pack to int4
MTE2   READ the 2048 B value strip                   (the RMW read)
Vector merge the owned token lanes into the strip
MTE3   WRITE the 2048 B value strip                  (the RMW write)
MTE3   key run, (1+k) x 128 B, aligned               no RMW needed
MTE3   scale, (1+k) fp32, DataCopyPad                sub-burst
```

Why the planes differ, from the layout contract (audit §2.2): the key plane is token-major,
so `token * d/2 = token * 128` bytes is 32-byte aligned and one token is one 128-byte burst
— no RMW. The value plane is strip-blocked channel-major, so one token owns 256 nibbles at
stride 16, which is 128 bytes spread over 2048 at half-byte granularity. **There is no
aligned-burst expression of that**, hence the RMW.

**RMW always preserves.** Lanes the item does not own are written back unchanged. This is
not a defensive choice: in decode a strip accumulates over 16 successive steps, so zeroing
unowned lanes would destroy the previous fifteen tokens. For a prefill tail the unowned
lanes hold whatever the page held, which is never read because `seq_len` has not advanced —
and preserving them makes the canary check *stronger*, because untouched lanes must still
read canary.

**`absmax == 0` is mandatory here, not optional** (audit I15). The exponent extraction masks
the mantissa of an fp32 bit pattern; for `0.0` the exponent field is 0 and `2^(2-e)` is a
denormal seed. A zero K or V row never occurs in the integer fixtures and occurs immediately
in production. Branch: `absmax == 0 -> scale = 1.0f, nibbles = 0`.

Queue allocation: MTE2 (activations + RMW read), Vector (butterfly, quantise, merge), MTE3
(three writes). No Cube, no L1, no cross-core flag, no mode-4 id. Intra-core HardEvents
only: `MTE2_V`, `V_MTE3`, `MTE3_V` on the RMW slot, and `V_MTE2` guarding the RMW read
against the previous item's merge.

UB: one `(1+k) x 256` fp32 working panel (`16 x 256 x 4 = 16 KiB` at the `k = 15` ceiling),
the butterfly ping-pong pair (2 KiB), the 6 KiB of №6 tables, one 2048-byte strip buffer,
scale lanes. Under 32 KiB — nowhere near the budget, which is the point of an AIV-only
kernel.

**Open item to settle on the first CAModel run:** under `KERNEL_TYPE_AIV_ONLY`, what
`GetBlockIdx()` and `GetSubBlockNum()` return. The documented MIX behaviour is
`AIV: get_block_idx() * get_subblockdim() + get_subblockid()` with `GetSubBlockNum()` the
sub-block count, and the safe MIX idiom is `GetBlockIdx() / GetSubBlockNum()`. Whether
`GetSubBlockNum() == 1` under AIV-only — making every AIV its own block, which is what this
kernel wants — must be **observed before either form is relied on**. Print both on rung 0.

**Gate 1.1.**
1. Oracle extended to partial strips; bit-exact on nibbles and scales for every
   `n in [1, 16]`, every `(1 + k) in [1, 16]`.
2. RMW preservation: write token 0, then token 1, then tokens 2..4; after each step every
   previously written lane is unchanged and every never-written lane is still canary.
3. Zero-row fixture: a K row and a V row of all zeros take the `absmax == 0` branch and
   produce `scale = 1.0`, all-zero nibbles.
4. Canary sweep: no byte outside the owned `(block, head, strip)` moves, on all three planes,
   head and tail guard bands intact.
5. Trace audit: zero cross-core flags issued, zero `wait_intra_block`, the instruction census
   matching the straight-line body above with no unexpected pass.

### 1.2 №2 `turboquant_reshape_aligned_cube` — adapt the bf16 writer

Serves `num_tokens % 16 == 0`. Four changes, in dependency order.

Files:

| Path | Change |
| --- | --- |
| `kernels/ops/turboquant/turboquant_reshape_and_cache_bf16.cpp` | the four changes below; rename to `turboquant_reshape_aligned_cube.cpp` at the end of the phase, not before |
| `tests/camodel/test_camodel_turboquant_reshape_and_cache_bf16.cpp` | `block_size` 16 -> 128; drop the `--window` loop; add the ragged and ragged-batch rungs |
| `tests/common/turboquant_oracle.hpp` | `block_size` default 16 -> 128; `cu_seqlens` reference |
| `scripts/analyze_reshape_trace.py` | extend check 1b to iteration boundaries |

**W0 — re-vendor the production copy first.** The harness copy is stale: its
`run_vector_core` derives only `block = slot0 / block_size` and adds no in-page offset, so
at `block_size = 128` every strip of a page lands on top of strip 0. The vllm-ascend copy
(`feat/tq-cube-clean`) carries `key_token_bytes` / `value_token_bytes` and the strip-blocked
value plane. Until W0 lands, every rung in this phase validates a layout the product does
not have. **Nothing else in Phase 1.2 may start before W0.**

**W1 — persistent grid-stride loop.** Replaces the `kStripsPerLaunch = 32` host loop (the
`--window 32` launcher) and the `blockDim = total_strips` grid.

```cpp
const uint32_t block  = GetBlockIdx() / GetSubBlockNum();   // divide this
const uint32_t stride = GetBlockNum();                      // do NOT divide this
for (uint32_t s = block; s < total_strips; s += stride) { run_strip(s); }
```

`blockDim = min(total_strips, core_count)` at the launcher. `GetBlockNum.md` is explicit
that the return value "is consistent with the **numBlocks** configured when `<<<>>>` is
called" on both roles — at a 1:2 ratio "the number of AICs started is numBlocks, and the
number of AIVs started is 2 * numBlocks", while `GetBlockNum()` returns `numBlocks` either
way. Dividing it would give the Cube a stride of `N` and its Vector cores `N/2`, the two
roles would disagree about the iteration count, and the result is a **deadlock, not a wrong
answer**.

Why: F5. A MIX grid wider than the AI core count *spins* — measured at 64 strips, 25 min of
CPU with a `wait_intra_block` signature and no output, cause unisolated. Grid-stride makes
that unreachable by construction, and the serialised launches go with it: a 32,768-token
prefill is 2,048 strips, today 64 serialised launches of 32 blocks each, afterwards one.

**The prologue is already outside the strip pipeline** — `run_cube_core` stages the 16
resident blocks to L1 (`stage_blocks_nz<16, 32>`), loads them into L0A and L0B, and only
then calls `StepColumnsCube`. So the table hoist is not a code motion; it is a consequence
of there being fewer blocks. Prologues per prefill drop from `total_strips` to
`min(total_strips, cores)`. Per strip the tables are 16 blocks x 32x32 x 2 B = **32,768 B**
against 8 chunks x 16 tokens x 256 ch x 2 B = **65,536 B** of activations, so the tables are
**33.3%** of GM->L1 traffic (32,768 of 98,304) and all of it is redundant after a core's
first strip. The a8 profile attributes 18.7% of Cube cycles to
`MOV_OUT_TO_L1_MULTI_ND2NZ` (8 x 4,752).

**The flag-id hazard this creates, and the required answer.** `kFlagPanelFree` is a
cross-core back edge where "each release carries its own id, so the ring's two slots cost
six ids per core rather than two". Letting those ids flow across grid-stride iterations
reuses them, which is exactly the measured §6.2c fault — a stale set releasing a later wait.
**Required: prime and drain per iteration.** The existing epilogue is four `MTE1_MTE2`
waits, `M_MTE1`, `FIX_M` and four `CrossCoreWaitFlag<4, PIPE_FIX>`; that becomes the
iteration boundary. Cost is a bubble of order a few hundred cycles against the measured
12,590-cycle strip span — call it 2-3% — and it is what makes the flag accounting provable
rather than argued. Do not optimise it away before the trace audit passes.

**W2 — `cu_seqlens` strip resolution.** Each iteration resolves its own request:

```
host     strip_begin[b] = sum_{j<b} ceil(L_j / 16)      # int32 prefix sum, uploaded once
         total_strips   = strip_begin[B]                 # NOT ceil(sum L_j / 16)
device   b = upper_bound(strip_begin, s) - 1             # scalar scan over B entries
         r = s - strip_begin[b];  t = cu_seqlens[b] + 16*r;  n = min(16, L_b - 16*r)
```

`B` is bounded by `max_num_seqs` and this is a scalar read of caller-supplied GM metadata —
the same category as the existing single `slot_mapping[0]` read, not a scalar read of
computed UB data (guidelines §1.2). The scan sits outside every pipeline stage, before the
iteration's first MTE2.

Why: audit §6.1. Today a strip spans global tokens `[16s, 16s+16)` of the flattened token
axis, so it straddles a request boundary whenever `L_b % 16 != 0`; the 16 slots then belong
to two pages of two requests, and the kernel's single scalar `slot_mapping[0]` read derives
one page and writes 16 tokens into the wrong place. `VALIDATE_SLOTS` would catch it and is
off by default, so this is a silent-corruption path.

**W3 — ragged tail, `n < 16`.** With №1 taking the sub-strip case the aligned kernel could
in principle refuse `n < 16` entirely, and the dispatch in Phase 5 does split that way. Keep
a tail path here anyway for the single-request case, where splitting a 17-token prefill into
a 16-token MIX launch plus a 1-token AIV launch costs two launches to save nothing.

Mechanism, and the one host requirement it carries:

* `stage_page_nz<Tokens, Channels>` asserts `Tokens % 16 == 0`, so a partial stage is not
  expressible through it. **Do not write a runtime-rows variant.** Instead require the host
  to present K and V with at least 15 rows of addressable, zero-filled slack past the last
  token, and stage a full 16 rows. The backend already materialises K/V through
  `.contiguous()`, so this is an allocation change in a buffer it already copies into, at
  zero runtime cost.
* Issue the Mmad with `m = n`. The rotation is row-wise — `C[t][c_out]` depends only on row
  `t` of A — so nothing from rows `[n, 16)` can reach a live row, and the hardware's
  tail-block rule covers the rest.
* The Fixpipe then writes only `n` rows to the AIV landing panel, leaving lanes `[n, 16)`
  holding the previous chunk's values. **The Vector stage zeroes them before absmax**, which
  makes the tail deterministic and canary-checkable.
* Key plane: write `n * d/2` bytes. `d/2 = 128`, so every `n` is a whole number of 32-byte
  bursts from a 32-byte aligned base.
* Value plane: write the full 2048 bytes with lanes `[n, 16)` zero. A partial channel-major
  strip is 256 runs of `n/2` bytes on an 8-byte stride — not burst-sized, not aligned — so
  padding is the only option, and the pad bytes land in the request's own unwritten page
  remainder, which `seq_len` keeps unreadable. **State this in the op docstring**: it is the
  one place the operator writes beyond the scheduled tokens, and a KV-transfer connector
  copying a page mid-prefill would see those zeros.
* Scale plane: `n` floats via `DataCopyPad`.
* `absmax == 0` guard (W4) is a **prerequisite**, not a companion: zero-padded tail lanes
  produce `absmax == 0` on the first run.

**W4 — `absmax == 0` branch and the runtime scale-plane stride.** The branch as in §1.1.
Separately, the scale plane's `kv` stride must be `num_blocks * H_kv * block_size` from a
launch scalar; the decode probe's `kNumPages = kTemporalBlocks + 1 = 9` is a fixture
constant and a mismatch moves the whole V plane silently (audit I14).

Pipeline and budget deltas:

| Resource | Before | After |
| --- | --- | --- |
| grid | `blockDim = total_strips`, host loop of 32 | `blockDim = min(total_strips, cores)`, one launch |
| Cube prologues per prefill | `total_strips` | `min(total_strips, cores)` |
| GM->L1 per strip | 98,304 B | 65,536 B after a core's first strip |
| L1 | 32 KiB tables at `kL1Table = 0` + depth-4 ring of `8192 + 32` B ≈ 64 KiB of 512 KiB | unchanged |
| L0A / L0B | `kL0Elems = 20,480` elements = **40 KiB** of 64 KiB at bf16 | unchanged |
| HardEvents | `MTE2_MTE1`, `MTE1_M`, `M_FIX`, `MTE1_MTE2` x4, `M_MTE1`, `FIX_M`; `V_MTE3`, `MTE3_V` | unchanged, but primed/drained per iteration |
| mode-4 flags | `kFlagProductReady`, `kFlagPanelFree` (6 ids/core) | unchanged ids, no reuse across iterations |

The in-file L0 comment ("16 KiB of table plus 4 KiB of page is 20 KiB of the 64 KiB") is
stale from the INT8 clone — at bf16 it is 32 KiB + 8 KiB = 40 KiB. Fix the comment in W0.

**Gate 1.2.**
1. **W0:** the existing bit-exact oracle passes at `block_size = 128`, 8 strips per page,
   head-major planes; [`scripts/kv_plane_layout_check.py`](../scripts/kv_plane_layout_check.py)
   reports 0 writer/decoder disagreements; canary sweep clean.
2. **W1:** bit-exact at `total_strips > core_count` — the shape F5 spins on today. Trace
   audit check 1b (`SET, WAIT, SET, WAIT, ...` per `(pipe, trigger, id)` with never two sets
   standing) extended to **every iteration boundary**, not just kernel exit. Measured GM->L1
   bytes per strip down ~33% after the first strip on a core.
3. **W2:** a ragged-batch fixture with `L_b % 16 != 0` for every `b`; every strip's 16 slots
   verified consecutive and 16-aligned under `VALIDATE_SLOTS=1`; bit-exact across the batch.
4. **W3/W4:** golden coverage for `n in [1, 15]` on the tail strip, including the value
   plane's UB zero-pad; canary proving the pad bytes land only in the request's own unwritten
   page remainder; zero-row fixture.
5. No byte of any plane's index function changes in this phase. A diff that moves a store
   address has failed the gate.

---

## 2. Phase 2 — independent rotations (№5, №6)

№6 landed in Phase 1.0. This phase is №5.

### 2.1 №5 `turboquant_rotate_q_batch_cube` — extract and widen

Files:

| Path | Change |
| --- | --- |
| `kernels/ops/turboquant/turboquant_rotate_q_batch_cube.cpp` | **new**, from existing leaves |
| `routines/routine_cube_decode_score.hpp` | `rotate_query_stage_l0` / `rotate_query_compute` / `rotate_query_to_gm` gain a row-tile parameter; `load_query_rotated` moves to №4 |
| `tests/camodel/test_camodel_rotate_q_batch.cpp` | **new** |
| `tests/common/paged_decode_oracle.hpp` | rotation reference widened to `B x H_q` rows |

**What it computes.** `Q_rot = Q . Pi^T` for **every** `(token, q_head)` vector of the step,
as 64 accumulating bf16 Mmads per row-tile against the 16 resident `32 x 32` blocks, with
the sign diagonal and the outer Hadamard factor folded into the resident operand:

```
T[s][g][c0][i] = (s ? -1 : +1) . piSigns[g*32 + c0] . H32[c0][i] / 16,   s = parity(c1 & g)
```

16 blocks cover all 64 `(output group, source group)` pairs because the dependence on `c1`
is a single sign.

**Why all heads at once, and what it is worth.** The rotation is row-wise: every row of the
A operand multiplies the *same* B operand (the rotation table). That is the opposite of the
score GEMM, where all `M` rows must share one key strip and the batch axis therefore cannot
enter `M` (F0). So the rotation may batch across heads, tokens and requests freely.

| | rows | fractals | fill | rotation launches per layer-step |
| --- | --- | --- | --- | --- |
| №4's prologue as built | 16 per task, 4 tasks for 24 real rows | 1 per task | **37.5%** | `B x H_kv x splits` |
| №5 at `B = 1`, `k = 0` | `B x H_q = 24` | 2 | **75%** | 1 |
| №5 at `B = 8`, `k = 0` | 192 | 12 | 100% | 1 |

Rows are `B x H_q x (1 + k)`. At `B = 1` that is 24 rows — `24 / 16 = 1.5` fractals' worth
of work occupying 2 fractals, with the tail block not participating.

**The GM round trip disappears by splitting, and not because anything was fixed.** №4's
prologue writes `Q_rot` to GM and re-ingests it (`rotate_query_to_gm` then
`stage_page_nz(gm_out_qrot, kL1QRot)`) because it needs `Q_rot` back as an L0A operand *in
the same kernel*, and the direct route failed: `CopyL0C2L1` with a bf16 Nz destination
produced a score panel of 256 zeros in a run where the same L0C reached GM bit-exactly
through `CopyL0C2GM`, mechanism unisolated. №5 has no such need — its output *is* GM — so it
is one `CopyL0C2GM` per tile and nothing more. **Do not read this as evidence that
`CopyL0C2L1` works.** It remains an open item (§6).

**Row tiling, and the one budget that is not established.** The rotation's L0C footprint is
`M_chunk x 256` fp32 = `M_chunk x 1024` B. The probe's allocation is
`__cc__ float l0_c[4096]` = 16 KiB, proven at `M_chunk = 16`. L0A holds `M_chunk x 256` bf16
and L0B the 32 KiB table.

| `M_chunk` | L0A | L0B | L0C |
| --- | --- | --- | --- |
| 16 (probe-proven) | 8 KiB | 32 KiB | 16 KiB |
| 32 | 16 KiB | 32 KiB | 32 KiB |
| 64 | 32 KiB | 32 KiB | 64 KiB |

L0A and L0B are 64 KiB each on arch35 per the writer's own in-file statement. **The L0C
capacity is not established** — the API mirror documents no numeric size and offers no
`GetL0CSize`. So: `M_chunk` is a compile-time constant with a `static_assert` against a
build-provided `kL0CBytes`, **starting at 16** and raised only after a CAModel rung at the
larger value passes. Grid: `min(ceil(rows / M_chunk), cores)`, grid-stride, tables staged
once per core exactly as in W1.

Output: bf16 ND `[T, H_q, d]`, which is what №3 and №4 both want as their A operand.

**Precision: the acceptance basis changes here, and this is the phase to say so.** The probe
asserts every rotated entry is exact in bf16, which holds only for its integer fixture — the
fixture's activations are integers in `[-256, 256]`, exact in an 8-bit significand, and the
Cube's fp32 accumulation of 256 such products stays below `2^24`. For general bf16
activations the `fp32 -> bf16` Fixpipe cast is a real rounding and no integer oracle
reproduces it. So:

* the **integer** rung stays bit-exact and is the regression gate;
* a **general bf16** rung is added, gated on cosine and relative L2 against an fp64 host
  reference, with the budget derived from `d = 256` and the bf16 significand rather than
  picked.

Do not let the second replace the first: a tolerance gate no longer detects a wrong-row read
the way bit comparison does.

**Gate 2.1.**
1. Integer fixture, bit-exact against the host rotation for every row, at
   `rows in {6, 12, 24, 48, 192}` — the last being `B = 8`.
2. General bf16 fixture within the derived cosine / rel-L2 budget.
3. `Pi` orthogonality on device: `Q_rot . Pi` recovers `Q` within the budget (this is also
   the first end-to-end check of №6's egress direction against №5's ingress).
4. Canary: `query_rot` written exactly once per row, nothing outside it touched, guard bands
   intact. Grid-stride run with `rows / M_chunk > cores`.
5. Trace audit: one table staging per core, `64 x ceil(rows/M_chunk per core)` Mmads, one
   Fixpipe per tile, no cross-core flag.

---

## 3. Phase 3 — decode engines (№3, №4)

### 3.1 №3 `turboquant_single_token_decode_aiv` — new, AIV only

**What it serves, stated more widely than its name.** `query_tokens_per_req == 1`, for **any
`B`**. Rows per `(request, kv_head)` are `6 x (1 + k) = 6`, and the grid is
`(request, kv_head)` — so `B` scales the grid, not the row count (F0). This kernel is not
"the `B = 1` kernel"; it is "the 6-row kernel", and at `k = 0` that is every decode step
regardless of batch.

Files:

| Path | Change |
| --- | --- |
| `kernels/ops/turboquant/turboquant_single_token_decode_aiv.cpp` | **new** |
| `routines/routine_vector_decode_gemv.hpp` | **new** — the score GEMV leaf |
| `routines/routine_vector_decode_recurrence.hpp` | runtime row count (shared with №4, see D1) |
| `tests/camodel/test_camodel_single_token_decode_aiv.cpp` | **new** |

**The score as a Vector GEMV, and why the key layout makes it cheap.** `K` is token-major
inside the strip (`token * 256 + channel`), so **a token's 256 channels are contiguous**.
For one query row `q`:

```
Vector  Mul(prod, key_page_fp32, q_bcast, 64 lanes, 64 repeats, {1,1,1,8,8,0})
        -> all 16 x 256 products in ONE instruction, re-reading the same 256 B of q
           on every repeat (src1RepStride = 0)
Vector  ReduceRepeat over 256 contiguous lanes x 16 repeats -> S[q][0..15]
```

Two instructions per query row per strip, so **12 per strip at 6 rows**, plus the int4
unpack of the key page. This is the second payoff of the token-major key layout, alongside
the `Nz(K) = Zn(K^T)` identity that serves the Cube: the channel axis being contiguous per
token is exactly what makes a per-token reduction a contiguous `ReduceRepeat`. The same
`src1RepStride = 0` trick already serves the value accumulation, where a 64-lane repeat
covers four channels' worth of 16 keys — see `routine_vector_decode_recurrence.hpp`, which
documents it as the reason `p_x4` exists and as what removed 32 KiB of UB and a 4096-entry
`Gather` offset vector.

**What this kernel does not have**, by construction: no Cube, no L1, no CV transfer
(`cv_transfer_nd2nz`), no mode-4 flag, no `wait_intra_block`, no ring priming or draining,
no `M_FIX` / `FIX_M` / `MTE1_M` edges. Its marked graph is a line.

Body: per `(request, kv_head)`, loop over the request's pages and the 8 strips of each:

```
MTE2   packed key strip 2048 B, packed value strip 2048 B, 2 x 16 scale floats
Vector int4 -> fp32 key page (via int16, the only listed route)
Vector score GEMV as above, x6 rows
Vector key scale on the 16 score columns  (16 multiplies, not 4096)
Vector online-softmax recurrence, 6 rows   (routine, runtime row count)
Vector value scale folded into P           (16 multiplies, denominator unscaled)
Vector weighted accumulation
--- after the last strip ---
Vector Acc / l, egress butterfly via №6 right-direction, sigmoid(gate)
MTE3   one write-out
```

Scales stay outside the 4096-entry pages exactly as in №4, and for the same reason: the key
scale multiplies the 16 score columns after the GEMV and the value scale folds into the
probabilities before the contraction, with the denominator still summing unscaled
probabilities.

UB: key page fp32 (16 KiB), value page fp32 (16 KiB), products (16 KiB), packed ingress
(2 x 2 KiB), `Acc` 6 x 256 fp32 (6 KiB), state, №6 tables (6 KiB), egress scratch. Order
40-60 KiB — an AIV-only kernel with the whole UB to itself and no ring to size.

**Acceptance criterion, stated as the census it has to be** (§0.2): an instruction census
and per-engine cycle breakdown against a budget written down *before* the run, not a
utilisation target. Expect `RV_VLD`/`RV_VST` to dominate; the int4-to-fp32 unpack is 2 passes
over 4096 elements per plane per strip and is the first thing to attack if the budget is
missed.

**Gate 3.1.**
1. Bit-exact score panels against the host oracle on the integer fixture, 6 rows, 1 page and
   4 pages; `(m, l)` within 2 ULP, `Acc` within its per-fixture budget.
2. Egress rotation and gated context within the budgets §3 of the decode report derives.
3. Canary: one tile reads and writes only its own GM; all guard bands intact; inactive
   output words still `0x7FC0BEEF`.
4. Trace audit: **zero** cross-core flags and zero `wait_intra_block` — this is a positive
   assertion about the design, not an absence of findings.
5. Instruction census within the pre-registered budget, with the per-engine breakdown
   recorded whether or not it passes.

### 3.2 №4 `turboquant_paged_decode_cube` — adapt the decode kernel

Files:

| Path | Change |
| --- | --- |
| `kernels/ops/turboquant/turboquant_paged_decode_attention.cpp` | D1-D6; rename to `turboquant_paged_decode_cube.cpp` at phase end |
| `routines/routine_vector_decode_recurrence.hpp` | D1: `kDecodeQueries` -> runtime bound |
| `routines/routine_cube_decode_score.hpp` | D2: rotation leaves move to №5; `score_strip_*` gain runtime `m` |
| `tests/camodel/test_camodel_paged_decode_attention.cpp` | rows ladder, page ladder, ring ladder |
| `tests/common/paged_decode_oracle.hpp` | runtime rows; multi-page reference |
| `scripts/analyze_paged_decode_trace.py` | ring checks at depth `D`; per-rows-bucket flag audit |

**D1 — runtime row count.** The highest-value change in the roadmap: **62.5% of the measured
critical path at 6 rows** (F6).

| Site | From | To |
| --- | --- | --- |
| `vector_decode_recur_strip` bound | `query < kDecodeQueries` (16) | `query < rows` |
| `EgressQueries` bound and its per-query gate `DataCopy` | `query < kTokens` | `query < rows` |
| `Duplicate` seeds of `m_prev`, `l_prev`, `acc` | `kTokens * ...` | `rows * ...` |
| final `DataCopy` extents | `kTokens * kHeadSize` | `rows * kHeadSize` |
| `MmadParams::m` | `kDecodeTokens` | `rows` |
| query ingress / L1->L0 / L0C window | `kTokens` | `M_pad`, three compile-time buckets |
| the `Duplicate` over-read `static_assert` | `(kTokens - 1) * ...` | `(M_MAX - 1) * ...`, `M_MAX = 48` |

The Vector side is nearly free to convert: `select_query` already takes a **runtime** query
index and forms runtime UB addresses from it, and the loop is a real loop — deliberately, because
unrolling 16 queries at 8 strip sites instantiated 128 copies of the body and the AIV
translation unit did not finish compiling in 25 minutes. Only the *bound* is a constant.

`M_pad = ceil(rows/16)*16 in {16, 32, 48}` is a compile-time bucket because the L0 fractal
counts and the Mmad nest are unrolled index sequences. All three are multiples of 16, so
`stage_page_nz`'s `static_assert(Tokens % 16 == 0)` holds and **no runtime-rows staging
variant is needed** — the query guard band (RFC §3.5: 15 addressable rows past the last used
row) covers the `[rows, M_pad)` rows the ingress reads and the Mmad discards.

**D2 — the rotation leaves move out, and that is what makes 48 rows affordable.** With №5
owning the prologue, №4's Cube does only the score GEMM. Two L0 regions are freed:

| | As built | After D2 |
| --- | --- | --- |
| L0A | `l0_a[kPanel]` bf16 = 8 KiB (Q groups, then `Q_rot`) | `M_pad x 256` bf16 = 8-24 KiB (`Q_rot` only) |
| L0B | `l0_b[kTableElems]` bf16 = **32 KiB** (16 resident blocks; key slots nested inside) | ring of `256 x 16` bf16 key slots = 8 KiB x `D` |
| L0C | `l0_c[kPanel]` fp32 = **16 KiB** (rotation panel; score windows nested inside) | `D x (M_pad x 16)` fp32 = 2-6 KiB |

The `static_assert`s that the key slots fit inside the L0B table allocation and the score
windows inside the L0C panel allocation both become unnecessary. At `M_pad = 48` the
rotation panel *would* have been `48 x 256 x 4 = 48 KiB` of L0C against 6 KiB of score
windows — so moving the rotation to №5 is the precondition for D1's upper buckets, not an
independent cleanup. Also deleted: №4's GM round trip, `kL1Q`, `kL1QRot` and the
`FIX_S` drain that stood in for a missing `FIX_MTE2` route.

**D3 — inner page loop, `temporal_splits == 1`.**

```
pages_total = ceil(seq_lens[b] / block_size)
for (page_idx = 0; page_idx < pages_total; ++page_idx)
    for (strip = 0; strip < block_size / 16; ++strip)
        ... the existing five-stage column body ...
```

`strips_in_page = block_size / 16` becomes a runtime scalar (8 at the vLLM default) in place
of `kStrips = 8`, and the last page is masked to `seq_lens[b] % block_size` tokens — a key
strip beyond the sequence must contribute `-inf` score columns, not garbage, which the
recurrence expresses by setting those columns to a large negative before the `Max`.

Grid `(request, kv_head)`, `B x H_kv` tasks. The online-softmax recurrence already carries
`(m, l, Acc)` across strips, so extending from 8 strips to `8 x ceil(S/128)` is the *same*
recurrence — and `temporal_splits == 1` therefore needs **no cross-page combine and no
partial workspace at all**, which removes the decode report's §7 gap from the critical path
rather than filling it. Phase 4 adds the split path for the small-`B` occupancy case only.

**D4 — L1 key ring.** The probe gave each strip its own L1 page — 8 pages,
`kL1KeyRing + 8 x kRingKey` with `kL1End` at 112.3 KiB of 512 KiB — specifically to escape
the §4.6 fault, where a correctly constructed mode-4 back edge did not bound the staging
core: AIV1 wrote slot 0 for strips 0, 2, 4, 6 at ticks 6754, 8703, 10653, 12608 while the
Cube read that address at 7088, 26994, 66481, 106697, and 480 of 2048 score entries were
wrong. A page loop makes the strip count `8 x ceil(S/128)`, and the report puts the
one-page-per-strip ceiling at ~56 strips, so the ring must come back.

Specified: depth `D = 4`, **one flag id per ring slot**, and §4.6 answered rather than
worked around. L1 after D2 and D4: `D x (8192 + 32)` ≈ **32.1 KiB**, down from 64.2 KiB, with
`kL1PiTable`, `kL1Q` and `kL1QRot` all gone.

**This is the riskiest item in the roadmap, and D1 makes it riskier.** The decode report is
explicit that the three surviving cross-core channels (`kFlagValueReady`, `kFlagScoreFree`,
`kFlagScoreReady`) carry four announcements per id over 8 strips and are correct on every
rung run *not* because their ids are safe but because each consumer is measurably slower than
its producer. D1 speeds the consumer up by `16/rows` — up to 2.7x at 6 rows — and D3
multiplies the announcements per id by the page count. **The flag-id audit must be re-run at
every `rows` bucket and every page count**, and the acceptance condition is trace check 3
(every L1 key page written in 16 bursts and read exactly once, after its last write) on a
rung that wraps the ring at least twice, `strips >= 2D`.

**D5 — scale-plane stride and gate dtype.** `kNumPages = kTemporalBlocks + 1 = 9` becomes the
launch scalar `num_blocks` (I14). `block_table` goes from `[temporal_blocks]` to
`[B, max_blocks_per_seq]` int32 with a per-request length. `gate` changes from `__gm__ float*`
to bf16 `[T, H_q, d]`, with the `sigmoid` in fp32 internally — removing a host cast of
`T x H_q x d` elements per layer-step. The per-query gate row keeps its two HardEvents
(`MTE2_V` **and** `V_MTE2`): the first revision had only `MTE2_V` and the canary run measured
the consequence exactly — the factor applied to query 0 was `sigmoid(gate[query 2])` to within
bf16 rounding, MTE2 running two iterations ahead of its consumer.

**D6 — probe outputs behind the verification build.** `out_qrot` goes to №5. `out_scores`,
`out_state`, `out_acc`, `out_rotated` stay only under `ASCEND_MICRO_VERIFY`, except that
`workspace` carries `(m, l, Acc)` when `output_stage == 0` for Phase 4. This frees **24.28
KiB** of UB and removes the score archive's eight per-strip `Muls`-by-1.0 copies from AIV0 —
the measured critical path.

UB budget after D1 and D6, from [`scripts/decode_ub_budget.py`](../scripts/decode_ub_budget.py):

| rows | `M_pad` | fractals | production arena | verdict |
| --- | --- | --- | --- | --- |
| 6 (`k=0`) | 16 | 1 | 109.97 KiB | fits |
| 12 (`k=1`) | 16 | 1 | 120.09 KiB | fits |
| 24 (`k=3`) | 32 | 2 | 140.34 KiB | fits |
| 48 (`k=7`) | 48 | 3 | 180.84 KiB | fits the kernel's 192 KiB window |
| 64 | 64 | 4 | 207.84 KiB | needs the whole UB |
| 96 | 96 | 6 | 261.84 KiB | **does not fit** |

So `M_MAX = 48`; `rows > 48` is tiled on the query axis, and query tiles of one request share
pages so the tiling needs no combine, only separate `(m, l, Acc)`. Size against
`GetUBSizeInBytes()`, not against the probe's `kUbWindow = 192 KiB` literal.

**Gate 3.2.** The twelve-rung canary protocol re-run, with three new axes:
1. **rows ladder** `{6, 12, 24, 48}`: score panels bit-exact on the integer fixture, `(m, l)`
   within 2 ULP, `Acc` within budget; flag-id audit at **each** bucket.
2. **page ladder** `{1, 2, 4}` pages at `splits == 1`, against a host oracle that computes the
   whole context in one pass. This is the first run in the project's history that produces a
   *correct* multi-page answer — the probe never did (decode report §7).
3. **ring ladder** `strips >= 2D` so the ring wraps twice; trace check 3 on each rung.
4. Unchanged conditions: ingress rotation 0/4096 wrong (now №5's gate), all guard bands
   intact head and tail, inputs 0 words changed, 325,376/325,376 inactive output words still
   canary, zero CAModel `[error]`/`[fatal]` lines, zero non-empty exception dumps, and the
   audit run against **the core the runner isolated** rather than `core0` (§4.9).

---

## 4. Phase 4 — workspace reduction and long-context combine

Needed only where `temporal_splits > 1`, which after D3 is an **occupancy** decision rather
than a correctness one: at `B = 1` the `(request, kv_head)` grid is 4 tasks, so a part with
32+ cores is 8x under-occupied and a temporal split buys parallelism at the price of a
reduction.

```
splits = clamp(core_count / (B * H_kv), 1, ceil(S / block_size))
tasks  = B * H_kv * splits        # and assert tasks <= core_count      (I8, F5)
```

Files:

| Path | Change |
| --- | --- |
| `kernels/ops/turboquant/turboquant_decode_combine_aiv.cpp` | **new**, AIV only |
| `routines/routine_vector_decode_combine.hpp` | **new** |
| `tests/camodel/test_camodel_decode_combine.cpp` | **new** |

Math, per `(request, kv_head, row)` over the `splits` partials:

```
m   = max_j m_j
l   = sum_j l_j * exp(m_j - m)
Acc = sum_j Acc_j * exp(m_j - m)      # Acc_j unnormalised, in the rotated basis
y   = (Acc / l) . Pi                   # №6, right direction
out = y * sigmoid(gate)
```

**Why the gate lives here and not in №4.** The egress rotation is linear and commutes with a
cross-page log-sum-exp combine; `sigmoid` does not. So with `splits > 1` the gate must be
applied after the reduction, and `output_stage == 2` with `splits > 1` must be **refused by
the adapter**, not silently produce a wrong answer. This is also why folding `Pi` into
`o_proj` stays ruled out for this model: `attn_output_gate = true` on all 16 full-attention
layers and the elementwise gate sits between attention and `o_proj`.

Workspace, which №4 writes at `output_stage == 0` and №4's own `(m, l)` representation
dictates:

```
workspace : fp32 [splits, B, H_kv, rows, d + 16]
            [..., 0:d]          = Acc
            [..., d   : d+8]    = m, replicated over the 8 lanes of a DataBlock
            [..., d+8 : d+16]   = l, likewise
floats    = splits * B * H_kv * rows * (d + 16)
```

The replicated-DataBlock form is the recurrence's existing `kDecodeStateLanes = 8` invariant
— what lets it run without a single scalar read of UB — so keeping it in the workspace means
the combine reads the shape №4 wrote with no repacking.

**Gate 4.**
1. `splits > 1` matches `splits == 1` **bit-for-bit** on the same fixture at 2 and 4 pages.
   That is the strongest available check on a reduction and it is available only because D3
   made `splits == 1` correct first.
2. `splits` ladder `{2, 4, 8}`; workspace canary showing each partial written once and read
   once.
3. Adapter refuses `output_stage == 2 && splits > 1`.

---

## 5. Phase 5 — Python integration and dispatch

Files:

| Path | Change |
| --- | --- |
| `csrc/attention/turboquant/op_adapter/turboquant_torch_ops.h` | eight schemas — RFC §4 plus the two marked new in §0.1 |
| `csrc/attention/turboquant/op_adapter/turboquant_torch_adpt.h` | validating launchers (RFC §4.1) |
| `csrc/torch_binding_meta.cpp`, `vllm_ascend/meta_registration.py` | no-op meta kernels |
| `CMakeLists.txt` | add the five new TUs; keep the `ascend950` gate; drop `turboquant_paged_attention_opt.cpp` once №4 lands |
| `vllm_ascend/attention/turboquant_v1.py` | the dispatch below |
| `vllm_ascend/attention/turboquant_layout.py` | the three plane accessors (RFC §9.1) |
| `vllm_ascend/platform.py` | validation (RFC §9.2) |
| `vllm_ascend/patch/worker/patch_qwen3_5.py` | gate hand-off (RFC §9.3) |
| `vllm_ascend/worker/model_runner_v1.py` | per-tensor `hybrid_with_attn_and_mamba` |

### 5.1 The two selection rules

**Ingress** — by alignment, per request:

```python
aligned = (L_b // 16) * 16
if aligned:           №2 on tokens [0, aligned)        # MIX, grid-stride, cu_seqlens
if L_b - aligned:     №1 on tokens [aligned, L_b)      # AIV, RMW tail
# decode step (query_tokens_per_req tokens per request): always №1
```

For a single request with a short tail, prefer №2's own W3 tail path over a second launch.
The two never contend: a tail is its own strip.

**Decode** — by row count, not by batch:

```python
rows = 6 * decode_token_per_req            # asserted == (num_heads // num_kv_heads) * ...
if rows == 6:   №3   # AIV GEMV, any B; grid (request, kv_head)
else:           №4   # MIX, M_pad bucket ceil(rows/16)*16, rows <= 48 per tile
```

The crossover is a **measured** quantity, not a derived one: at `rows == 6` №4 starts a Cube
that does 0.75% of the work and pays ring priming and three cross-core channels for it, while
at `rows >= 12` the GEMV's per-row instruction count grows linearly and the Cube's does not.
Ship the rule above, expose `VLLM_ASCEND_TURBOQUANT_DECODE_ENGINE` (`auto|aiv|cube`) to
override it, and record the measured crossover in the report rather than asserting it here.

### 5.2 Backend wiring

```python
def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
    if attn_metadata.attn_state in (DecodeOnly, SpecDecoding):
        torch.ops._C_ascend.npu_turboquant_reshape_scatter(...)        # №1
    else:
        aligned, tails = self._split_prefill(attn_metadata)
        if aligned: torch.ops._C_ascend.npu_turboquant_reshape_aligned(...)   # №2
        for t in tails: torch.ops._C_ascend.npu_turboquant_reshape_scatter(...)
    return query, key, value, output

def forward_paged_attention(self, query, attn_metadata, output, output_gate=None):
    torch.ops._C_ascend.npu_turboquant_rotate_q_batch(...)             # №5, once per layer
    splits, _ = self._decode_plan(attn_metadata)                       # I8 assert inside
    stage = 0 if splits > 1 else (2 if output_gate is not None else 1)
    if self._rows == 6: torch.ops._C_ascend.npu_turboquant_single_token_decode(...)  # №3
    else:               torch.ops._C_ascend.npu_turboquant_gqa_paged_decode(...)     # №4
    if splits > 1:      torch.ops._C_ascend.npu_turboquant_decode_combine(...)       # №4->combine
    return output
```

Keep the `feat/tq-main-clean` patterns that exist for graph-capture reasons and are easy to
lose: `_decode_workspace`, `_rotated_query` and the index staging buffers are **persistent**
because a captured graph replays only device work, so a Python-side allocation never runs
again after capture. Grow them outside capture; never allocate inside it.

Validation at `__init__`: `num_heads // num_kv_heads == 6`, `head_size == 256`,
`num_kv_heads == 4`, `sliding_window is None`, `attn_type == DECODER`,
`kv_sharing_target_layer_name is None`, `logits_soft_cap` unset.

The gate goes to the backend on every path, and the **backend** decides per step whether it
reaches №4 as `output_stage = 2` or the combine as a post-reduction multiply — so
`patch_qwen3_5.py` hands the gate over unconditionally and keeps no per-step branch of its
own. Per the repo's patching rule this stays in `patch/worker/`, never in a model file.

**Gate 5.**
1. UTs: the dispatch table above over a matrix of `(attn_state, B, k, L_b % 16)`; the
   validation messages; the plane accessors against the C++ index functions (the host form of
   [`scripts/kv_plane_layout_check.py`](../scripts/kv_plane_layout_check.py)).
2. Meta kernels: `torch.export` and FakeTensor tracing of a full layer.
3. A dry-run pass with `TURBOQUANT_CAPTURE_SIGNATURES=1` showing every op reached with the
   expected shapes, and `VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS=1` clean across a ragged batch.
4. End-to-end and accuracy gates belong to whoever has hardware. This phase does not claim
   them.

---

## 6. Risk register

| # | Risk | Where it bites | Mitigation / who can settle it |
| --- | --- | --- | --- |
| R1 | mode-4 flag-id reuse | W1 iteration boundaries, D4 ring | per-iteration prime/drain (W1); one id per ring slot (D4); re-audit at **every** `rows` bucket and page count. The §4.6 mechanism is unisolated: the reference says the flag is a counter, and a measured wait proceeded when the arithmetic said zero |
| R2 | D1 destroys the margin the surviving channels rely on | D4 | the channels are correct today only because every consumer is slower than its producer; D1 speeds consumers up to 2.7x. Trace check 3 is the gate, per bucket |
| R3 | F5 cause unisolated | W1, any grid change | grid-stride makes it unreachable; the isolating 33-strip run with sampling compiled out is still owed |
| R4 | L0C capacity unknown | №5 `M_chunk` | `static_assert` against a build constant, start at the probe-proven 16, raise only behind a rung. No `GetL0CSize` in the API mirror |
| R5 | `GetSubBlockNum()` under `KERNEL_TYPE_AIV_ONLY` | №1, №3, Phase 4 | print both forms on rung 0 before relying on either |
| R6 | `CopyL0C2L1` with a bf16 Nz destination | not on any critical path after №5 splits out | left open deliberately; re-opening it is a separate investigation with an isolated repro |
| R7 | AI core count of the target part | §4 `splits` clamp, every `blockDim` | `GetBlockNum()` / `npu_turboquant_vector_core_num()`, never a literal |
| R8 | accuracy on real activations | №5 Gate 2.1, system level | the integer fixtures prove arithmetic, not accuracy; cosine/rel-L2 rungs are added but no LongBench-style gate has been run |
| R9 | hybrid allocator | upstream of all five phases | Qwen3.5's full-attention layers share one flat buffer with the GDN layers while the three-plane carve exists only on the `not use_mamba` path and the reshape asserts `len(raw_planes) == 3`. A run dies there before any kernel executes — integration map Phase 0 |
| R10 | concurrent full-grid execution | every phase | no decode run has ever had two blocks computing at once; every rung isolates one of 32. One strip of one block is ~7 min of model time on this host |

---

## 7. Sequencing

```
Phase 1.0  №6 leaf                         ──────────────┐
Phase 1.1  №1 scatter        (needs №6)    ──────────┐   │
Phase 1.2  №2 aligned        W0 → W1 → W2 → W3/W4    │   │   (parallel with 1.1)
Phase 2.1  №5 rotate_q       (parallel with all of 1) ───┤
Phase 3.1  №3 AIV decode     (needs №5, №6) ─────────┐   │
Phase 3.2  №4 Cube decode    D2 needs №5; D1 → D3 → D4 → D5/D6
Phase 4    combine           (needs D3 correct first)
Phase 5    Python            (needs all six registered)
```

Critical path: **W0 → D2 → D1 → D3 → D4 → Phase 4 → Phase 5**, with №5 feeding D2 and №6
feeding everything with an egress. W0 blocks the whole writer line and is a half-day of
re-vendoring; it is the cheapest item on the critical path and the one most likely to be
skipped.

Two items can start immediately and in parallel with no dependency: **№6** (a leaf with a
host-only probe) and **№5** (built entirely from leaves the decode probe already verified
bit-exactly).
