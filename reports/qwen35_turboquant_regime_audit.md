# TurboQuant on Qwen3.5-27B — operational regime audit

Ascend 950PR / arch35 / `dav-3510` / CANN 9.2, `KERNEL_TYPE_MIX_AIC_1_2`.
Model: `Qwen3.5-27B`, 64 layers = 48 `linear_attention` (GDN) + 16 `full_attention`
(`full_attention_interval: 4`), GQA `H_q = 24` / `H_kv = 4` (6:1), `head_dim = 256`,
`attn_output_gate = true`, bf16 activations, vLLM page 128 tokens.

Sources read for this audit:

| Artefact | Where |
| --- | --- |
| decode kernel | [`turboquant_paged_decode_attention.cpp`](../kernels/ops/turboquant/turboquant_paged_decode_attention.cpp) (harness only) |
| writer kernel, probe copy | [`turboquant_reshape_and_cache_bf16.cpp`](../kernels/ops/turboquant/turboquant_reshape_and_cache_bf16.cpp) |
| writer kernel, production copy | `C:/vllm-ascend` `feat/tq-cube-clean` `csrc/attention/turboquant/kernels/turboquant_reshape_and_cache_bf16.cpp` |
| op schema + launcher | same branch, `csrc/attention/turboquant/op_adapter/turboquant_torch_{ops,adpt}.h` |
| Cube leaves | [`routine_cube_decode_score.hpp`](../routines/routine_cube_decode_score.hpp) |
| Vector leaves | [`routine_vector_decode_recurrence.hpp`](../routines/routine_vector_decode_recurrence.hpp), [`routine_vector_egress_rotate.hpp`](../routines/routine_vector_egress_rotate.hpp) |
| measurements | [`paged_decode_attention.md`](paged_decode_attention.md), [`reshape_and_cache_a8.md`](reshape_and_cache_a8.md) |
| vLLM dispatch | `vllm_ascend/attention/attention_v1.py`, `worker/model_runner_v1.py`, `patch/worker/patch_qwen3_5.py` (branch `feat/tq-cube-clean`); `vllm_ascend/attention/turboquant_v1.py` (branch `feat/tq-main-clean`) |
| hardware contracts | `C:/PipeSim/docs/ascendc_api` — `cube_compute_TensorAPI/mmad_compute/Mmad.md`, `mmad_compute_key_features/GEMV.md`, `.../k_direction_alignment_constraint.md`, `basic_api/tool_interface/.../GetUBSizeInBytes.md` |

No NPU on this host: every timing figure quoted is cycle-accurate CAModel, and no
latency, throughput or power claim is made about silicon.

---

## 0. Headline findings

**F0 — The brief's premise about batched decode is not realisable as stated, and the
reason is load bearing.** The brief asks what happens when `N_rows = B x 6` query
vectors are packed into the `M` axis of the Cube fractal. They cannot be: the per-strip
score GEMM contracts one `[M x 256]` query panel against **one** `[256 x 16]` key strip
(`routine_cube_decode_score.hpp`, `ScoreBlock`), so every row of `M` must attend to the
same physical page of the same KV head. The 6 Q heads of one `(request, kv_head)` do;
two different requests never do — each has its own block-table row. So the batch axis
cannot enter `M` at all. `M` is `6 x (1 + num_spec_tokens)`, and `B` scales the **grid**,
not the fractal. §4 works the numbers out both ways.

**F1 — The decode kernel named in the brief does not exist in vLLM-Ascend.**
`turboquant_paged_decode_attention.cpp` (M=16, d=256, gridDim=32, the 5-stage modulo
schedule) lives only in this harness. On `feat/tq-cube-clean` the only decode-shaped
translation unit is `turboquant_paged_attention_opt.cpp`, which `CMakeLists.txt:87-92`
builds **and deliberately does not register**: its head dimension is 16 and its dequant
scales are compile-time constants, so it cannot read the cache the writer writes. There
is no decode op schema, no adapter, no Python call site. The 4-bit cache on that branch
is write-only.

**F2 — The two branches named in the brief are different products.**
`feat/tq-main-clean` carries the full AIV+Cube generation (`turboquant_v1.py`, a backend
impl, `rotate_q` / `paged_attention` / `cube_decode` schemas, the three-plane page carve).
`feat/tq-cube-clean` is a clean rebase on v0.30.0rc2 carrying only the BF16 Cube writer —
no `turboquant_v1.py`, no attention backend, and `patch_qwen3_5.py` with no TurboQuant
hook at all (the gate is plain torch, `patch_qwen3_5.py:92-95`). The `turboquant`
package that *is* wired on cube-clean (`vllm_ascend/quantization/methods/kv_cache/turboquant/`,
`ops/triton/turboquant_finalize.py`) is the **MLA/SFA latent** path: it validates
`kv_lora_rank == 512` and `qk_rope_head_dim == 64` (`config.py:48-54`) and is irrelevant
to a GQA model. Name collision, not shared machinery.

**F3 — The writer is prefill-only by construction, and nothing says so at the schema
level.** `npu_turboquant_reshape_and_cache` asserts `num_tokens % 16 == 0` and requires
each 16-token strip's slots to be 16-aligned and consecutive. A decode step writes
exactly `B` tokens whose slots are scattered across `B` different pages. Both conditions
fail. But `AscendAttentionBackendImpl.forward` calls `reshape_and_cache` on **every**
step for every attention state (`attention_v1.py:1326`, and `:1411` on the PCP variant), so a TurboQuant layer needs
a second, scatter-shaped writer before it can decode at all. §5.4.

**F4 — The declared torch shape of the packed planes disagrees with the physical index
function, and only the sizes are checked.** The schema declares
`[num_blocks, block_size, num_kv_heads, head_size/2]` (token outside head), the adapter
checks exactly that sizes tuple, and the kernel writes
`block * (H_kv * head_bytes) + head * head_bytes + <in-page token/strip offset>` —
head **outside** token. Both readings have identical `numel`, so the `TORCH_CHECK` passes
on a tensor whose strides mean something else. Any host-side reader, any `get_kv_cache_shape`
consumer and any future decode kernel that trusts the declared axis order reads transposed
data and gets plausible wrong numbers, not an error. §2.3.

**F5 — A MIX grid wider than the AI Core count hangs; it does not merely queue.**
Measured at 64 strips: the model spins with a `wait_intra_block` signature, 25 min of CPU
with no output (`reshape_and_cache_a8.md` §"Open: a grid wider than the AI Core count
hangs"). The cause was not isolated. `kStripsPerLaunch = 32` in the adapter is the
mitigation, and it is the whole of what `--window 32` means. §6.2 is what it costs.

**F6 — The Cube is 99.2% idle on the decode path, so fractal padding is free and the
query-row count is not.** Measured over 8 strips: `AIV0 Vector` in flight 234,863 ticks
against the Cube's 1,777 and AIV1's 2,457; per-strip score interval ~19,900 ticks while
the staging core finishes its 16-burst key page in ~980. The recurrence and the egress are
both per-query loops with a **compile-time bound**: `vector_decode_recur_strip` is
`for (query = 0; query < kDecodeQueries; ++query)` with `kDecodeQueries = 16`
(`routine_vector_decode_recurrence.hpp:277-282`), and `EgressQueries` is the same loop
against `kTokens` (`turboquant_paged_decode_attention.cpp:864`). The bound is the *only*
compile-time piece — `select_query` already takes a runtime query index and nothing is
unrolled (deliberately: unrolling 16 queries at 8 strip sites did not finish compiling in
25 minutes). So ten padded rows at `B=1`
do not cost Cube throughput — they cost 62.5% of the critical path. §3.

---

## 1. Inventory, as built

| Operator | Geometry | Registered as a torch op? | Reads the 4-bit cache? |
| --- | --- | --- | --- |
| `turboquant_reshape_and_cache_bf16` (vllm copy) | strip 16 tok, `H_kv=4`, `d=256` compile time; `block_size`, `num_blocks`, `num_tokens` runtime | yes — `npu_turboquant_reshape_and_cache` | writes |
| `turboquant_reshape_and_cache_bf16` (harness copy) | same, but `block_size == 16` only (no in-page offset) | n/a | writes |
| `turboquant_reshape_and_cache_a8` | int8 activations, otherwise identical | no | writes |
| `turboquant_paged_attention_opt` | `d = 16`, compile-time dequant scales | **no, by decision** | no |
| `turboquant_paged_decode_attention` | M=16, `d=256`, 8 strips, `H_kv=4`, grid 32 | no (harness probe) | reads |

The harness writer copy is **stale**: its `run_vector_core` derives only
`block = slot0 / block_size` and adds no in-page offset, so at `block_size = 128` every
strip of a page would land on top of strip 0. The vllm copy fixes exactly that
(`key_token_bytes`, `value_token_bytes`) and generalises the value plane to strip-blocked
channel-major. Treat the vllm copy as normative and re-vendor it into the harness before
the next CAModel rung, or the probe is validating a layout the product does not have.

### 1.1 What the decode kernel actually computes

Per task: one `(kv_head, temporal_block)` pair, one 128-token page as 8 strips of 16,
`M = 16` query rows resident for the whole sequence. It emits `out_qrot`, all 8 raw score
panels, `(m, l)`, `out_acc` (still in the rotated basis), `out_rotated` (pre-gate) and
`out_context` (`Acc . Pi`, gated by `sigmoid(gate)`, bf16).

`out_context` is the exact answer **only when that one page is the whole context for its
head**. The rotation commutes with a cross-page log-sum-exp combine; the sigmoid gate does
not, so a real answer needs combine-then-gate, and the combine is not in this kernel
(`paged_decode_attention.md` §7). That is the single largest missing piece on the read side.

---

## 2. Layout contracts

### 2.1 The two kernels do agree — once the writer is read at `block_size = 128`

Writer (vllm copy), byte offset inside a plane for a strip starting at in-page token `t0`:

```
key    : block*(H_kv*block_size*d/2) + head*(block_size*d/2) + t0*d/2          [token-major]
value  : block*(H_kv*block_size*d/2) + head*(block_size*d/2) + (t0/16)*2048    [strip-blocked channel-major]
scale  : ((kv*num_blocks + block)*H_kv + head)*block_size + t0                 [planar, kv 0 = K, 1 = V]
```

Decoder:

```
strip_base_bytes = ((page*H_kv) + kv_head) * 8 * 2048
scale_plane      = num_pages*H_kv*block_size   (stride between the K and V planes)
scale_base       = ((page*H_kv) + kv_head)*block_size + strip_begin*16
```

With `block_size = 128`, `H_kv = 4`, `d = 256`: `block_size*d/2 = 16384 = 8 * 2048`, so the
writer's `head * 16384 + strip * 2048` **is** the decoder's
`(page*4 + head)*8*2048 + strip*2048`. Key plane: `t0 * 128` bytes token-major is the same
2048-byte run. Scale plane: identical index function, identical plane order. **The two
agree byte for byte.**

Two caveats:

* the decoder hardcodes `kNumPages = kTemporalBlocks + 1 = 9` for the scale-plane stride.
  In production that stride is `num_blocks * H_kv * block_size` and must become a launch
  scalar; a mismatch moves the whole V scale plane and is silent.
* the value plane's strip blocking is *only* correct because `block_size % 16 == 0`.
  A literal `ch * block_size + key` reading — which is what the harness probe verified at
  `block_size == 16` — gives 256 runs of 8 bytes on a 64-byte stride, neither burst-sized
  nor burst-aligned, and two strips of one page then share a 32-byte word and tear each
  other. The vllm copy's derivation says this; the harness copy has not been re-verified
  against it.

### 2.2 Scale handling, and why it is exact

Scales never touch the 4096-entry pages. The key scale multiplies the 16 score columns
after the Fixpipe (16 fp32 multiplies instead of 4096 bf16 ones) and the value scale folds
into the probabilities before the contraction, with the denominator still summing unscaled
probabilities. Because the quantiser's scale is the power of two `2^(e-2)` for
`2^e <= absmax < 2^(e+1)`, extracted by integer ops on the fp32 bit pattern, the Cube's key
operand stays an exact small integer in bf16 and the score panel is checkable **bit for
bit** rather than to a tolerance — which is what the 12 CAModel rungs do (0 of 2048 panel
entries wrong at 8 strips).

### 2.3 F4 in detail: the shape/stride disagreement

`turboquant_torch_ops.h` comments and `turboquant_torch_adpt.h:126-134` both say
`[num_blocks, block_size, num_kv_heads, head_size/2]`. The adapter then checks
`key_cache.size(2) == num_kv_heads && key_cache.size(3) == head_size/2`. For
`(N, 128, 4, 128)` that passes. The kernel writes head-major. A correct declaration of what
is written is

```
key_cache   : [num_blocks, num_kv_heads, block_size,            head_size/2]
value_cache : [num_blocks, num_kv_heads, block_size/16, head_size/2 * 16]   # strip-blocked
```

and the two planes do not even have the same logical rank under an honest declaration.
This is not cosmetic: `AscendTurboQuantAttentionBackend.get_kv_cache_shape` on
`feat/tq-main-clean` publishes the token-major form, `swap_blocks`/`copy_blocks` survive it
only because they move whole pages, and `turboquant_layout.turboquant_dequantize` (the
host-side reader) would decode transposed.

---

## 3. Regime 1 — single-token decode, `B = 1`

### 3.1 The GQA mapping, and where the padding actually lands

The bf16 Cube fractal is `16 x (32B / 2) = 16 x 16`, so `M = 16` is exactly one fractal
row and `M = 6` is the same one fractal row with 10 invalid rows. The hardware handles that
itself — `Mmad.md`, constraints:

> When M, K, and N are not multiples of 16, the hardware still organizes data in 16x16
> fractal blocks. Invalid data in the tail block occupies fractal block space but does not
> participate in valid computation.

`MmadParams::m` is a plain `uint16_t` runtime extent. So **passing `m = 6` is legal and
costs one fractal, exactly as `m = 16` does.** 37.5% fractal utilisation is a true
statement about occupancy and a false statement about cost.

Where the 10 rows *are* paid for:

| Stage | Owner | Row dependence | Measured share of the 8-strip run |
| --- | --- | --- | --- |
| score GEMM, 8 Mmads of k=32 | AIC | `ceil(M/16)` fractals | Cube in flight **1,777 ticks** |
| key dequant + CV staging | AIV1 | none (keys, not queries) | AIV1 in flight **2,457 ticks** |
| value dequant | AIV0 | none | — |
| online-softmax recurrence | AIV0 | **linear in M** (`for query < kTokens`) | AIV0 in flight **234,863 ticks** |
| egress rotation + gate | AIV0 | **linear in M** (`EgressQueries`, per-query gate row) | included above |

So at `B = 1` with `M` left at its compile-time 16, ~62.5% of the 234,863-tick critical
path is spent on padding. Making the loop bound a launch scalar is the whole fix, and it is
a one-line change in two places plus the `DataCopy` extents.

### 3.2 (A) pad to M=16 on the Cube vs (B) pure-AIV GEMV + 8-stage FWHT

The measurement answers this before a line is written: the Cube is in flight 1,777 of
~237,000 ticks — **0.75%**. Option (B) removes a unit that is doing 0.75% of the work and
moves its arithmetic onto the unit that is doing 99.2% of it.

Concretely, what (B) would move to the Vector cores per task:

* the ingress rotation. Today 64 accumulating Mmads against 16 resident `32x32` blocks in
  the Cube prologue, with the sign diagonal and the outer Hadamard factor folded into the
  resident operand. As a Vector butterfly it is `log2(256) = 8` radix-2 stages plus a sign
  multiply over an `M x 256` panel — 9 full-panel Vector passes. The prefill operator
  measured exactly this trade in the other direction and recorded the numbers:
  `RV_VLD + RV_VST` was 48.6% of Vector cycles against 24.4% for the whole execute pipe,
  and folding the four butterfly/sign passes into the Cube halved the absolute register
  traffic (82,703 -> 41,059 cycles) and cut Cube `WAIT_INTRA_BLOCK` from 33.6% to 0.6%
  (`reshape_and_cache_a8.md` §"What the brief predicted, and what was measured").
* the per-strip score GEMM. `M x 16 x 256` MACs per strip as Vector work, where the Cube
  does it in 8 instructions.

Against that, (B)'s stated win is "bypassing Cube and GM roundtrips entirely". The GM
round trip is **one**, in the prologue, once per launch: `rotate_query_to_gm` writes `Q_rot`
to GM and re-ingests it, because `CopyL0C2L1` with a bf16 Nz destination delivered nothing
L0A could read (all 256 score entries zero, in a run where the same L0C reached GM
bit-exactly through `CopyL0C2GM`; mechanism not isolated —
`routine_cube_decode_score.hpp`, `rotate_query_to_gm`). The strip loop itself touches GM
only to read packed pages and scales, and writes it once at the end.

There is also a hardware note against the pure-GEMV reading. `GEMV.md` says the Cube
auto-enables GEMV at `M = 1` (disabled by `MmadTrait::disableGemv`, whose Tensor-API
default is `true`), and that at `M = 1` each `1 x 16` sub-vector still occupies 1024 B of
L0C with only 32-64 B valid — a 16:1 L0C waste. So the *one-query-per-(token,head)* framing
is the worst case for the Cube, and GQA folding to `M = 6` is already 6x better on exactly
that axis.

**Verdict: (A), and it is not close.** The right move is not to choose between the two but
to stop paying for padding on the unit that matters:

1. make the recurrence and egress row counts launch scalars (`M = 6(1+k)`) — the 62.5% in
   §3.1;
2. keep the Cube prologue, and pass `m = M` to the Mmads;
3. if the prologue's GM round trip is worth removing, the lever is re-trying `CopyL0C2L1`
   with an isolated mechanism, not relocating the rotation to AIV0.

A Vector-side rotation does have one legitimate niche: a task whose `M` is so small that
the Cube prologue's fixed cost (16 resident blocks = 32 KiB of L1 staging plus an ND->Nz
conversion, per launch per block) dominates. At `M = 6` the prologue stages 32 KiB of tables
to rotate 3 KiB of query. That argues for hoisting the tables across tasks (§6.3), not for
deleting the Cube — and it is the reason the *decode-step writer* of §5.4, where `M = 1`,
should be Vector-only.

### 3.3 Layout, packing and UB budget for 6 Q heads in one AICore

Today the decode kernel's query is `[H_kv][16][256]` bf16 with
`q_base = kv_head * 16 * 256`, and the gate is fp32 with `gate_base = q_base`, read one
256-float row per query. For `B = 1` GQA folding the host must present, per
`(request, kv_head)`:

```
q_rows[r][:]    = Pi-unrotated query of q_head (kv_head*6 + r),  r in [0, 6)
gate_rows[r][:] = gate of the same q_head
```

i.e. a `[T, 24, 256]` -> `[H_kv][6][256]` slice on the head axis, plus a bf16->fp32 cast for
the gate. The slice is a pure **view** when q-heads are stored group-contiguous (head
`h*6 + g` belongs to KV head `h`), which is the standard vLLM layout — so it should be
specified as a view, not materialised. The gate cast is not free and is the one real
conversion; taking `gate` as bf16 in the kernel removes it.

UB, recomputed from the kernel's own `NextSlot` chain (32-byte bases, 32-byte bank skew,
256-byte read pad), with the verification-only buffers (`scoreArchive`, `out_rotated`)
removed as a production kernel would. Reproduce with
[`scripts/decode_ub_budget.py`](../scripts/decode_ub_budget.py), which mirrors the slot
order and must be updated alongside the kernel:

| M rows | B (GQA 6:1, no spec) | fractals | probe arena | production arena | fits kernel's 192 KiB window | fits 256 KiB UB |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 1 | 1 | 119.25 KiB | **109.97 KiB** | yes | yes |
| 12 | 2 | 1 | 138.38 KiB | 120.09 KiB | yes | yes |
| 16 | 2.67 | 1 | **151.12 KiB** (as built) | 126.84 KiB | yes | yes |
| 18 | 3 | 2 | 157.50 KiB | 130.22 KiB | yes | yes |
| 24 | 4 | 2 | 176.62 KiB | 140.34 KiB | yes | yes |
| 32 | 5.33 | 2 | 202.12 KiB | 153.84 KiB | yes | yes |
| 48 | 8 | 3 | 253.12 KiB | **180.84 KiB** | yes | yes |
| 64 | 10.67 | 4 | 304.12 KiB | 207.84 KiB | **no** | yes |
| 96 | 16 | 6 | 406.12 KiB | **261.84 KiB** | no | **no** |

M-scaled buffers are `m_prev`/`l_prev` (32 B each per row), `acc` (1 KiB), `context`
(512 B) and the two score-ring slots (128 B) — 1,728 B per row in production, plus 1,536 B
per row of verification buffers. Everything else is strip-sized and M-independent:
`packedK/V` (2 x 2080 B), `keyRing` (2 x 8224 B), `valueRing` (2 x 16416 B), `dqSigned`
(8 KiB), `weighted` (16 KiB), the egress singles and the gather tables.

Two numbers worth keeping: the arena as built is **151.12 KiB** of the kernel's
self-imposed `kUbWindow = 192 KiB`, and dropping the two verification buffers frees
**24.28 KiB**. `GetUBSizeInBytes()` is the documented constexpr query for the real figure
(`GetUBSizeInBytes.md`, 950PR: Supported); 192 KiB is the kernel's choice, not the
hardware's, and a production kernel should assert against the API rather than a literal.

The UB ceiling is what caps a tile: **48 rows fits the declared window, 64 needs the whole
UB, 96 does not fit at all.** That matters in §4.4.

At `M = 6` the arena is 109.97 KiB. There is no UB argument for a separate single-token
kernel; the existing one with a runtime `M` covers `B = 1`.

---

## 4. Regime 2 — batched decode, `B > 1`

### 4.1 What batch sizes actually arrive

Decode is dispatched when `AscendAttentionState.DecodeOnly` and
`max_num_scheduled_tokens == uniform_decode_query_len` with
`num_tokens == max_num_scheduled_tokens * num_reqs` (`model_runner_v1.py:3357-3365`).
`uniform_decode_query_len` is `decode_token_per_req = 1 + spec_token_num`
(`model_runner_v1.py:749-753`). So:

* without speculative decoding, `num_tokens = B`, one query token per request;
* with Qwen3.5 MTP at `k` draft tokens, `num_tokens = B(1+k)`, and all `1+k` tokens of a
  request attend that request's own pages.

The tensor the backend sees is padded: `num_input_tokens = num_tokens_padded`
(`model_runner_v1.py:3678`), padding slots are `-1` (`model_runner_v1.py:3583`), and
`attn_metadata.num_actual_tokens` carries the true count. Padding buckets come from vLLM's
`_set_cudagraph_sizes()` ladder — `1, 2, 4` then multiples of 8 up to the cap — reachable
through `compilation_config.cudagraph_capture_sizes`, which the e2e suites here override
freely (`[1, 2, 4, 8]`, `[6, 12, 18, 24]`, `[1, 32, 64]`). On `feat/tq-main-clean`
TurboQuant is additionally downgraded to PIECEWISE/NONE (`platform.py:556-614`), so the
attention op itself always runs eagerly between captured pieces; the padding still applies
to the tensor it receives. **`B` is therefore not free-form but it is also not guaranteed
to be a multiple of anything** — treat the bucket ladder as configuration, not invariant.

### 4.2 The invariant, and why it does not help

Per `(request, kv_head)` the number of real query rows is

```
rows = 6 * (1 + num_spec_tokens)          # exactly, always
```

6 because the GQA mapping is static: `H_q / H_kv = 24 / 4 = 6`. So yes, **`N_rows` is
always a multiple of 6** — but per request, not per batch.

If the batch axis *could* enter `M`, the alignment condition would be

```
6B = 0 (mod 16)  <=>  3B = 0 (mod 8)  <=>  B = 0 (mod 8)      since gcd(3,8) = 1
fractals = ceil(3B/8),   utilisation = 6B / (16*ceil(3B/8))
```

giving `B=1: 37.5%`, `B=2: 75%`, `B=3: 56.25%`, `B=4: 75%`, `B=5: 93.75%`, `B=7: 87.5%`,
and `B in {8,16,24,32}: 100%` — i.e. every capture bucket at or above 8 would be exactly
aligned and only `{1,2,4}` ragged. That is a tidy result and it is unreachable, because of
F0: all `M` rows share one `[256 x 16]` B operand, and requests do not share pages. Folding
two requests into one `M` tile would multiply request 1's query by request 0's keys.

What *can* fill `M` legitimately, all within one request:

| Source of rows | rows | fractals | utilisation |
| --- | --- | --- | --- |
| `B=1`, no spec | 6 | 1 | 37.5% |
| MTP `k=1` | 12 | 1 | 75% |
| MTP `k=3` (common Qwen3.5 config) | 24 | 2 | 75% |
| MTP `k=7` | 48 | 3 | 100% |
| chunked / paged prefill over the 4-bit cache, 16 query tokens | 96 | 6 | 100% |

The kernel as written — 16 query rows of one sequence against one page — is therefore
already shaped for the **MTP and paged-prefill** regimes, not for single-token decode. That
reframing is the most useful thing in this section.

### 4.3 So `SetTail`, masked accumulation or dynamic padding?

**None of the three.** `MmadParams::m` is the runtime extent and the hardware pads the tail
fractal itself with data that does not participate (`Mmad.md`). There is no `SetTail` on
this path — the ISASI/`Matmul` `SetTail` belongs to the high-level Matmul API, which this
kernel does not use. Masked accumulation would be solving a problem the hardware already
solved. Host-side dynamic padding of the query panel is harmless but pointless, and
host-side padding of the *token axis* is actively unsafe on the write side (§5.3).

What does need the row count at runtime is everything on the Vector side, because the
hardware's tail-block rule applies to the Cube only:

* `vector_decode_recur_strip`'s `query < kDecodeQueries` bound (routine `:277-282`);
* `EgressQueries`' identical loop and its per-query gate `DataCopy`;
* the `Duplicate` seeds of `m_prev`/`l_prev`/`acc`, and the final `DataCopy` extents;
* the `static_assert` that `Duplicate`'s tensor form over-reads 256 B past the **last**
  query's state block — at runtime `M` that bound must be re-derived from the maximum
  supported `M`, not from the launch's `M`, or a short `M` makes the check vacuous and a
  long one breaks it.

### 4.4 Grid and scheduling for `B > 1`

With `M` fixed at `6(1+k)` per `(request, kv_head)`, the batch and the context both scale
the grid. Two shapes, with a real trade-off:

**(i) One task per `(request, kv_head)`, inner loop over all of that request's pages.**
Tasks `= B * H_kv`. The online-softmax recurrence already carries `(m, l, Acc)` across
strips, so extending from 8 strips to `8 * ceil(S/128)` strips is the *same* recurrence —
and there is **no cross-page combine and no partial workspace at all**, which deletes the
largest missing piece in §1.1. Cost: at `B = 1` that is 4 tasks, so a part with 32+ AI cores
is 8x under-occupied. Blocker: the L1 key staging is currently one page per strip, sized by
`kStrips` — 112.3 KiB at 8 strips, and `paged_decode_attention.md` §8 states a strip count
above ~56 would not fit and "would need the ring back", which in turn needs the §4.6
flag-id question answered.

**(ii) One task per `(request, kv_head, temporal_block)`, as today.**
Tasks `= B * H_kv * ceil(S/128)`. Full occupancy, but it needs the combine, and the partial
state is `M * 256` fp32 of `Acc` plus `2 * M * 8` fp32 of `(m, l)` per task. At `S = 4096`,
`B = 8`, `k = 0`: `8 * 4 * 32 = 1024` tasks per layer and `1024 * 6 * 256 * 4 B = 6 MiB` of
`Acc` partials per layer — and 1024 exceeds the grid width F5 says hangs, so it is also
several launches.

**Recommendation:** (i) as the default, with a temporal split only when it buys occupancy:

```
splits = clamp(core_count / (B * H_kv), 1, ceil(S / 128))
tasks  = B * H_kv * splits        # and assert tasks <= core_count
```

`splits == 1` takes the no-combine path; `splits > 1` pays for the combine. That makes the
combine an optimisation for small `B` rather than a correctness prerequisite, and keeps the
grid under the core count by construction.

Note the UB ceiling from §3.3 interacting here: a tile may carry at most 48 rows inside the
declared window (64 using the whole UB), so an MTP configuration with `6(1+k) > 48` must
split the *query* axis into several tiles as well — which is free, because query tiles of
one request share the same pages and need no combine either, only separate `(m, l, Acc)`.

### 4.5 Requirements on `AscendTurboQuantAttentionBackendImpl` dispatch

1. `rows = 6 * decode_token_per_req`, derived from `num_heads / num_kv_heads` and asserted,
   never hardcoded.
2. `splits` by the clamp above, with `core_count` from `npu_turboquant_vector_core_num()`
   (already a schema on `feat/tq-main-clean`), and the result capped so
   `B * H_kv * splits <= core_count`. F5 makes this a correctness gate, not a tuning knob.
3. the query/gate **views** of §3.3, built without a copy, and refused — not silently
   reshaped — when `H_q % H_kv != 0`.
4. `seq_lens` and `block_tables` per request as int32 device tensors; the kernel's
   `block_table` is `[temporal_blocks]` today and must become `[B, max_blocks_per_seq]`
   with a per-request length.
5. padding: slice to `num_actual_tokens` before deriving `B`, and never pass a `-1` slot to
   a TurboQuant op.
6. the partial-state workspace, allocated once and persistent (the `_decode_workspace` /
   `_rotated_query` pattern on `feat/tq-main-clean` exists precisely because a graph replay
   cannot allocate), sized `B * H_kv * splits * (rows*256 + 2*rows*8)` fp32 and **zero**
   when `splits == 1`.

---

## 5. Regime 3 — ingress and reshape in short / paged prefill

### 5.1 How the reshape op is reached

On `feat/tq-cube-clean` it is not reached at all (F1/F2): no Python caller exists
(`grep npu_turboquant_reshape_and_cache --include=*.py` finds only a comment in
`envs.py:96`). On `feat/tq-main-clean` the path is
`AscendTurboQuantAttentionBackendImpl.forward -> reshape_and_cache (:784) ->
torch.ops._C_ascend.npu_turboquant_reshape_and_cache (:872-885)`, replacing
`DeviceOperator.reshape_and_cache -> torch_npu.npu_scatter_pa_kv_cache(cache_mode="Norm")`
1:1 — and it is called on **every** step, in every attention state, whenever `key` and
`value` are not `None`.

`slot_mapping` is vLLM's standard
`slot = physical_block_id * block_size + (position % block_size)`. Two consequences that
decide everything below: slots are consecutive **within one page only** (the block id is
constant inside a page and arbitrary across pages), and the token axis is the flattened
concatenation of all scheduled requests.

### 5.2 Short, page-aligned prefill

| `N_tokens` | `N % 16` | Result |
| --- | --- | --- |
| 128 (one page) | 0 | works. `blockDim = 8`, 8 AI cores, one launch. Latency ≈ one strip, ~12,590 cycles measured per strip |
| 16..112, multiple of 16 | 0 | works. `blockDim = N/16`; untouched strips of the page stay canary (verified: 325,376/325,376 inactive words intact) |
| any `N` | != 0 | **hard `TORCH_CHECK` failure** — `num_tokens % kTokensPerStrip == 0` |
| `N < 16` | != 0 | same failure. A 1-token prompt cannot be ingested |

So the kernel is fine on short *and* sub-page prefills, and the real constraint is the
16-token quantum, not the 128-token page. `block_size % 16 == 0` together with
`slot0 % 16 == 0` guarantees a strip never straddles a page boundary, so no separate
crossing check is needed — that reasoning in the adapter is correct.

### 5.3 The ragged tail, and why host padding cannot fix it

Every real prefill length is arbitrary, so `N % 16 != 0` is the common case, not the corner.
Host-side padding of the token axis needs somewhere for the pad tokens' slots to point, and
the writer requires the strip's 16 slots to be **consecutive**, so a tail strip of
`[real, real, ..., bucket, bucket]` is not expressible even with a reserved scratch page.
A stock-CANN fallback for the tail is also unavailable: `npu_scatter_pa_kv_cache` writes
unquantised K/V in a different layout.

**A masked tail strip inside the kernel is therefore required**, not optional. Its shape:
the tail strip's Vector stage quantises only `n <= 16` tokens, and its MTE3 writes `n*d/2`
bytes for the key plane. For the **value** plane that is harder — the strip is channel-major
over 16 tokens, so a partial strip is 256 runs of `n/2` bytes on an 8-byte stride, which is
neither burst-sized nor aligned. The clean answer is to zero-pad the tail strip *inside UB*
to 16 tokens and write the full 2048 bytes, having established two things: that zeros rotate
to zeros (they do — the transform is linear with `+-1` entries) and that the `absmax = 0`
case has a defined exponent (it does not today: the extraction masks the mantissa of an fp32
`0.0`, giving exponent field 0, and `2^(2-e)` for that is a denormal seed — **this needs an
explicit `absmax == 0 -> scale = 1, nibbles = 0` branch**).

The pad tokens' slots are `t0 + n .. t0 + 15`, which is exactly the unwritten remainder of
the current page — owned by this request and not yet readable, because `seq_len` has not
advanced. **This is the one place the kernel may legitimately write beyond the scheduled
tokens, and it must be stated in the contract rather than discovered.**

### 5.4 The decode-step write

`B` tokens, slots `block_id_b * 128 + (S_b % 128)` for each request `b` — different pages,
non-consecutive, and `B % 16 != 0` in general. Neither the `%16` check nor the
consecutive-slot contract can be satisfied. **The decode path needs a separate scatter
writer:** `B` independent single-token rotate-quantise-scatter operations, grid `B` or
`B * H_kv`, no strip pipeline, no Cube — at `M = 1` per `(token, head)` the Cube prologue's
32 KiB table staging dominates 512 B of payload, and `GEMV.md`'s 16:1 L0C waste applies, so
this is the one place the Vector butterfly is the right implementation. This is the single
largest missing operator after the decode kernel itself.

Note the write granularity problem it inherits: the value plane is channel-major within a
16-token strip, so a one-token write touches 256 nibbles at stride 16 — 128 bytes spread
over 2048, half-byte granular. **A single-token value write is not expressible as aligned
bursts in the current layout.** The options are a read-modify-write of the 2048-byte strip
(safe only if one core owns the strip, which it does: one request's page is written by one
request's decode step), or a layout change. §9 of the RFC takes the read-modify-write.

### 5.5 Does the window-32 launcher add sync overhead on short contexts?

No. `kStripsPerLaunch = 32` means `N <= 512` tokens is exactly one launch, so there is no
extra synchronisation and no extra launch for any short or single-page prefill. The loop is
inert below 512 tokens. §6.2 is where it starts to cost.

Memory under-utilisation on short contexts is real but small and inherent: an `N = 48`
prefill uses 3 of the available AI cores. That is a consequence of "one strip per block",
not of the window loop.

---

## 6. Regime 4 — batched prefill and long-context ingress

### 6.1 Is the kernel reused 1:1, and how are ragged boundaries handled?

Reused 1:1 — there is no alternative TurboQuant ingress and no stock fallback that writes
this format. And **ragged request boundaries are not handled.** The token axis is the
concatenation of the scheduled requests, and the writer slices it into 16-token strips at
fixed offsets. With requests of lengths `L_0, L_1, ...`, strip `s` spans global tokens
`[16s, 16s+16)`, which crosses a request boundary whenever `L_0 % 16 != 0`. Those 16 slots
then belong to two pages of two different requests and are not consecutive, so the strip's
single scalar read of `slot_mapping[0]` derives one page and one in-page offset and writes
16 tokens' worth of data into the wrong place.

Nothing catches it. `VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS` would (the per-strip
consecutiveness loop in `turboquant_torch_adpt.h`), but it is **off by default** because it
costs a device-to-host copy and a synchronisation per layer per step. The failure mode is
silently corrupted KV for both requests.

Two ways out, not equivalent:

* **per-request launches.** Call the writer once per request on its own token slice. Each
  slice is then internally consecutive and only its own tail is ragged (§5.3). Cost: `B`
  launches per layer per step instead of one, each with a grid of `ceil(L_b/16)` — and for
  short requests that is a tiny grid, which is the occupancy problem again.
* **a `cu_seqlens`-aware kernel.** Pass `query_start_loc` and have block `b` resolve its own
  `(request, strip)` from it, so one launch covers a ragged batch. This is the right answer
  and it composes with the grid-stride fix below.

### 6.2 What the window-32 launcher costs on a long context

It is a correctness mitigation for F5, and above 512 tokens a throughput ceiling:

* **occupancy caps at 32 AI cores**, regardless of how many the part has. The launcher's
  `blockDim` is `min(32, remaining_strips)`.
* **launches serialise.** They are issued on one `aclrtStream`, so a 32,768-token prefill is
  2,048 strips / 32 = **64 serialised launches** per layer, 16 full-attention layers deep,
  per prefill step.
* **the rotation tables are re-staged per block per launch.** The 16 resident `32x32` bf16
  blocks are 32,768 B, staged ND->Nz in the Cube prologue of every block. Per strip the
  activation ingress is 8 chunks x 16 tokens x 256 ch x 2 B = 65,536 B. So the tables are
  **33.3%** of the writer's GM->L1 traffic (32,768 of 98,304 B), and all of it is redundant
  after the first strip on a given core. The a8 profile attributes 18.7% of the Cube's
  cycles to `MOV_OUT_TO_L1_MULTI_ND2NZ` (8 x 4,752).

### 6.3 Per-page scatter, or a multi-core tiled batched-prefill operator?

**Refactor it — into a persistent grid-stride operator, not into a wider grid.** This is
also what `reshape_and_cache_a8.md` concludes independently, for the F5 reason:

```
blockDim = min(total_strips, core_count)
for (s = block; s < total_strips; s += GetBlockNum())   # resident tables hoisted out
```

Four things that buys, in order of size:

1. the grid never exceeds the core count, so F5 cannot be hit, and the window loop and its
   serialisation both disappear;
2. the rotation tables are staged once per core instead of once per strip — the 33.3% of
   §6.2;
3. successive strips on one core become sequential iterations whose mode-4 flag accounting
   the kernel controls, which is what `kStripsPerLaunch = 32` was buying by avoiding the
   question;
4. `cu_seqlens` resolution (§6.1) and the masked tail (§5.3) both become per-iteration
   concerns rather than per-launch ones.

Implementation note carried over verbatim because it is a deadlock rather than a wrong
answer: `GetBlockIdx()` must be divided by `GetSubBlockNum()` and `GetBlockNum()` must
**not** be — `GetBlockNum` returns `numBlocks` on both roles, so dividing it gives the Cube
a stride of 32 and its Vector cores 16, and the two roles disagree about the iteration
count.

What should *not* change: the per-page scatter character of the write itself. The three
planes' index functions are page-local and burst-aligned, `swap_blocks`/`copy_blocks` move
whole pages, and nothing in the long-context case wants a different cache layout. The
refactor is to the **work distribution**, not to the operator's contract.

---

## 7. Invariants, in one place

| # | Invariant | Where it comes from | Enforced? |
| --- | --- | --- | --- |
| I1 | `H_q / H_kv = 6` exactly; rows per `(request, kv_head)` = `6(1 + k_spec)` | model config | no — would be a new check |
| I2 | all rows of one `M` tile share one page of one KV head | `ScoreBlock`'s single B operand | implicit; violating it is silent |
| I3 | `block_size % 16 == 0`; a strip never straddles a page | adapter `TORCH_CHECK` + `slot0 % 16 == 0` | yes |
| I4 | `num_tokens % 16 == 0` | adapter `TORCH_CHECK` | yes — and this is the blocker of §5.3 |
| I5 | each strip's 16 slots consecutive and 16-aligned | the kernel's one scalar `slot_mapping[0]` read | **only under `VALIDATE_SLOTS=1`** |
| I6 | every GM operand base 32-byte aligned (`kGmBurstBytes`) | CANN burst rule; violation = error 264 | yes, `check_burst_aligned` |
| I7 | index tensors int32 | adapter | yes |
| I8 | grid <= AI core count for any MIX launch | F5, measured | by the `kStripsPerLaunch = 32` cap only |
| I9 | mode-4 flag ids: <= 6 channels per direction, ids below 11, 13/14 reserved | `MICROROUTINE_DESIGN_GUIDELINES.md` §6.2d | by construction; 4 channels used |
| I10 | every cross-core set meaning "my pipe work finished" preceded by an `x_S` drain | §6.2a, measured at 1583 cycles of run-ahead | by construction |
| I11 | rotation tables bit-identical to the host's composition of eq. (2) | `audit_tables` composes to `256*256` | yes, in the probe |
| I12 | combine before gate across pages | sigmoid does not commute with log-sum-exp | n/a — the combine does not exist |
| I13 | declared cache shape = physical index function | — | **no (F4)** |
| I14 | scale-plane stride = `num_blocks * H_kv * block_size` | writer | decoder hardcodes `kNumPages = 9` |
| I15 | `absmax == 0` has a defined scale | — | **no (§5.3)** |

---

## 8. What this audit does not establish

* **Anything about silicon.** No NPU on this host. Every cycle figure is CAModel, and the
  decode numbers come from runs that each isolate **one** of 32 grid blocks — a genuinely
  concurrent 32-block decode run has never been performed (`paged_decode_attention.md` §8).
* **The AI core count of the target part**, which §4.4's clamp and I8 both need. Use
  `npu_turboquant_vector_core_num()` / `GetBlockNum()`, never a literal.
* **The cause of F5.** 64 strips spins; 33 strips was stopped before a verdict; flag residue
  was ruled out by argument, not by isolation.
* **Whether the three surviving cross-core flag channels are safe at other strip counts.**
  They carry four announcements per id over 8 strips and are correct on every rung run, but
  the reason is that each consumer is measurably slower than its producer — rebalance the
  pipeline and that margin is gone (`paged_decode_attention.md` §8). Every §4.4 and §6.3
  change rebalances it.
* **Accuracy of the 4-bit cache on real Qwen3.5 activations.** Every bit-exactness result
  here is over *integer* fixtures, where the Cube's fp32 accumulation order does not matter.
  With general bf16 activations no integer oracle reproduces it and the check degrades to
  cosine plus a relative-L2 bound. No LongBench-style accuracy gate has been run.
* **The hybrid allocator conflict**, which is upstream of all four regimes: Qwen3.5's
  full-attention layers share one flat buffer with the GDN layers
  (`hybrid_with_attn_and_mamba`), while the TurboQuant three-plane carve exists only on the
  `not use_mamba` path and the reshape asserts `len(raw_planes) == 3`. That analysis is in
  `C:/vllm-ascend/TURBOQUANT_VLLM_ASCEND_INTEGRATION_MAP.md` §3.3-3.4 and is not re-derived
  here; it is a prerequisite for running *any* of this on Qwen3.5-27B.
