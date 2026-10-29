# RFC / PRD — TurboQuant GQA operator set for Qwen3.5-27B on Ascend 950PR

Status: draft for review. Companion to
[`qwen35_turboquant_regime_audit.md`](qwen35_turboquant_regime_audit.md), which is the
evidence base; every "F*n*" and "§*n*" reference below points there. Nothing in this
document has been built or run.

Target: Ascend 950PR / arch35 / `dav-3510`, CANN 9.2, `KERNEL_TASK_TYPE_MIX_AIC_1_2`.
Model: Qwen3.5-27B, 16 full-attention layers, `H_q = 24`, `H_kv = 4`, `d = 256`,
`attn_output_gate = true`, bf16, vLLM page 128 tokens.

---

## 1. Scope

### 1.1 Goals

G1. A TurboQuant 4-bit KV path that serves **all four** regimes — single-token decode,
batched decode, short/paged prefill, batched and long-context prefill — without the host
having to pad, split or pre-gather tensors in ways that defeat the quantisation win.

G2. One cache layout — the hardware-native one that is already in the kernels — described
honestly to Python, shared by every operator, and verified as such rather than agreed by
inspection (F4, I13). **No byte of either packed plane moves.** The `Nz(K) = Zn(K^T)`
identity and the channel-contiguous value strip are invariants, not implementation details:
§3.1a states them normatively and nothing in this RFC may introduce a transpose, gather or
repack into prefill ingestion or the decode pipeline.

G3. No operator whose grid can exceed the AI core count (F5, I8).

G4. The output gate applied where it is correct — after any cross-page combine, never
folded into `o_proj` (the gate sits between attention and `o_proj` and does not commute
with `Pi`).

### 1.2 Non-goals

N1. The hybrid (GDN + full-attention) page allocator. It is a prerequisite, not part of
this operator set; see audit §8 and
`C:/vllm-ascend/TURBOQUANT_VLLM_ASCEND_INTEGRATION_MAP.md` §3.3-3.4, Phase 0.

N2. The int8-activation (`a8`) ingress. The kernel exists in the harness; wiring it is a
separate, additive schema and does not change anything here.

N3. 910B support. Every facility this set depends on — a Fixpipe that picks a sub-block's
UB, mode-4 intra-block flags, the 16-bit fractal with `C0 = 16`, the direct UB->L1 channel —
is `dav-3510` only. `CMakeLists.txt:71-97` already scopes the library to
`SOC_VERSION MATCHES ascend950`; keep it.

N4. Sliding-window attention, cross/encoder attention, and PCP. Refuse them explicitly.

### 1.3 The decision that shapes everything else

From F0/§4.2: **the batch axis cannot enter the Cube's `M` axis.** One `M` tile is one
`(request, kv_head)` and its rows are `6 x (1 + num_spec_tokens)`. `B` scales the grid.
Every interface below follows from that.

---

## 2. Target operator set

Six kernels, eight registered entry points. Two of the six exist and are adapted; the
provenance and the phase order are in
[`qwen35_turboquant_execution_roadmap.md`](qwen35_turboquant_execution_roadmap.md) §0.

| # | Entry point | Engine | Status | Regime |
| --- | --- | --- | --- | --- |
| №1 | `npu_turboquant_reshape_scatter` | AIV | new | 1..16 tokens: per-step decode writes (F3) and ragged prefill tails |
| №2 | `npu_turboquant_reshape_aligned` | MIX | adapt the bf16 writer | prefill, `num_tokens % 16 == 0` |
| №3 | `npu_turboquant_single_token_decode` | AIV | new | `query_tokens_per_req == 1`, any `B` (6 rows) |
| №4 | `npu_turboquant_gqa_paged_decode` | MIX | adapt the decode probe | decode + MTP, rows 12..48 |
| №5 | `npu_turboquant_rotate_q_batch` | Cube | extract from №4's prologue | `Q_rot` for all `B x H_q x (1+k)` vectors |
| №6 | — (header leaf) | AIV | new | the 8-stage butterfly, both directions |
| — | `npu_turboquant_decode_combine` | AIV | new | cross-page log-sum-exp, only when `splits > 1` |
| — | `npu_turboquant_decode_workspace_size` | host | new | sizing |
| — | `npu_turboquant_vector_core_num` | host | exists on `feat/tq-main-clean` | planning, I8 |

**№3 is not subsumed by №4.** An earlier revision of this section argued that №4 with a
runtime row count covers the 6-row case and that a second kernel would only duplicate the
layout contract. The measurement says otherwise: at 6 rows №4 starts a Cube that does 0.75%
of the work and pays ring priming plus three cross-core channels for it, while №3's score is
one 4096-lane `Mul` and one contiguous `ReduceRepeat` per row with no cross-core flag at all.
The two share an operand list and are interchangeable at the call site, so the choice is a
planner decision (roadmap §5.1), not a shape constraint — and the crossover is a measured
quantity, not a derived one.

## 3. GM layout contract (normative)

This section supersedes the shape comments in `turboquant_torch_ops.h`. **The index
functions below are exactly what the writer already emits and what the decoder already
reads** (audit §2.1, verified plane by plane by
[`scripts/kv_plane_layout_check.py`](../scripts/kv_plane_layout_check.py)): no byte moves,
no stride flips, no transposes. What changes is only how the planes are *described* to
Python and to the adapter.

Both orders are hardware mandates, not choices, and §3.1a records why. Any proposal that
would make a plane's declared PyTorch strides authoritative over these index functions is
out of scope and must be refused: it would force a Vector-side permutation into a pipeline
that currently has none.

Let `P = num_blocks`, `S = block_size` (multiple of 16), `H = num_kv_heads`, `d = head_size`,
`R = 16` (the strip height, `= M0` of the DaVinci fractal and the only compile-time shape
constant inside the writer).

### 3.1 Packed planes

```
key_cache   : int8, numel = P * H * S * d/2
value_cache : int8, numel = P * H * S * d/2

key   nibble index   = ((block*H + head)*S + token) * d       + channel
value nibble index   = ((block*H + head)*S)       * d
                       + (token / R) * (R*d) + channel * R + (token % R)
```

**The planes are an opaque hardware-packed byte pool, not PyTorch-strided tensors.**
Neither plane's physical order is expressible as a contiguous PyTorch shape that a consumer
could index safely: `key_cache` carries fractal packing inside each strip that has no stride
expression at all, and `value_cache`'s in-strip order makes its honest rank differ from
`key_cache`'s. So the contract is bytes plus **plane accessors**, in both languages:

```
allocation  : int8[P * H * S * d/2]   per plane     # what get_kv_cache_shape reports
accessor    : key_byte(block, head, token, channel)  }  the index functions above,
              value_byte(block, head, token, channel) }  single-sourced, §9.1
```

For *descriptive* purposes only — never as a shape the adapter enforces or a view a
consumer indexes — the key plane's page-level nesting is that of a contiguous
`[P, H, S, d/2]`, which is what audit §2.4 verifies and what the schema's
`[P, S, H, d/2]` gets wrong. The value plane has no such equivalent above the strip.

Rationale for the value plane's strip blocking: channel-major over a whole 128-token page
would make one strip 256 runs of 8 bytes on a 64-byte stride — not burst-sized, not
burst-aligned, and two strips of a page would share a 32-byte word and tear each other.
Blocking by `R` makes every strip one aligned 2048-byte burst, and at `S == R` it is
byte-identical to plain channel-major, which is the form the CAModel probe verified.

### 3.1a Why these two orders, and not any others

Normative: an implementation may not change either order, and a reviewer should reject a
change that does.

**`key` is token-major inside the strip because `Nz(K) = Zn(K^T)`.** With
`nibble = token*d + channel`, the dequantised UB page on AIV1 *is* the ND matrix
`K[token][channel]` of shape `[R, d]`. At 16 bits `C0 = 32B/2 = 16`, and `Nz([N, D])` is
indexed `(D1, N1, N0, D0)` while `Zn([K, N])` in the Cube's B role is indexed
`(K1, N1, N0, K0)` — the same tuple, because `D` and `K` are the same channel axis. So one
L1 image serves both operand roles and the Cube reaches its B operand for
`S = Q_rot . K_strip^T` through a **byte-identity** L1->L0 load
(`load_page_group_b_zn<16, 256>`). The staging path is `cv_transfer_nd2nz`
(`routine_cv_transfer_ub_to_l1.hpp`): the dav-3510 UB->L1 channel on the AIV MTE3 queue,
`DataCopyUB2L1Impl` -> one `CopyUbufToCbuf`, with ND->NZ expressed as a **strided placement**
at 32-byte granularity — one `DataCopy` per `C0` column strip — not as an element
permutation. No TPipe, no UB scratch, no GM hop. The measured alternative is the cost of
getting this wrong: a row-major `K^T` disagrees with the fractal image in 3,840 of 4,096
offsets, so an explicit transpose is what a "shape fix" would buy.

**`value` is strip-blocked channel-major because the accumulation is a `ReduceRepeat`.**
With `nibble = channel*R + token` inside the strip, a channel's `R` keys are contiguous,
which makes the per-channel reduction over keys a contiguous `ReduceRepeat` over 16 lanes
on AIV0 and the 2048-byte strip one contiguous MTE2 read. It is also what lets the weighted
product avoid a `Gather`: a 64-lane `Mul` repeat covers four channels' worth of 16 keys, so
with `P[q]` pre-replicated into one 64-lane block and `src1RepStride = 0` a single
instruction forms all 4096 products — removing 32 KiB of UB and the 4096-entry offset
vector a `Gather` would need.

**Consequence for the whole pipeline.** Writer ND->Nz ingress, Cube rotation, packed store,
packed load, dequant, UB->L1 placement, byte-identity L1->L0 load, `Mmad`, Fixpipe to UB,
`ReduceRepeat` — every stage consumes the bytes the previous one produced, in that order.
**No transpose, gather or repack exists anywhere in prefill ingestion or in the decode
pipeline, and none is introduced by anything in this RFC.**

### 3.2 Scale plane

```
scale_cache : fp32, numel = 2 * P * H * S
index       = ((kv * P + block) * H + head) * S + token      # kv 0 = key, 1 = value
```

Planar, not token-interleaved: one chunk's `S` lanes leave as a single aligned burst, where
a `[S, 2H]` interleaving would put all `2H` chunks of a token in one 32-byte word.

**Every operator takes `P` as a launch scalar and derives the `kv` plane stride from it.**
The harness decode probe's `kNumPages = kTemporalBlocks + 1` must go (I14); a wrong stride
moves the whole V scale plane and is silent.

### 3.3 Codes and scales

4-bit signed, two nibbles per byte, low nibble = even element index, range `[-8, 7]` — the
packing `routine_turboquant_dequant4.hpp` unpacks and the hardware `Cast` to `int4b_t`
produces directly.

Per-token, per-head scale. For `2^e <= absmax < 2^(e+1)` the scale is `2^(e-2)` and the code
is `min(rint(Y * 2^(2-e)), 7)`, clamped only from above. `invSqrtLen = 1/sqrt(d)` folds into
the stored scale, never into the code.

**New requirement (I15):** `absmax == 0` must be handled explicitly as
`scale = 1.0f, codes = 0`. Today the exponent extraction masks the mantissa of fp32 `0.0`,
gives exponent field 0, and `2^(2-e)` for that is a denormal seed. The case does not arise
with the integer fixtures and arises immediately with a zero-padded tail strip (§5.2) or a
genuinely zero K/V row.

### 3.4 Rotation tables

`16` row-major `32x32` blocks of `+-1`, block `s*8 + g` holding

```
T[s][g][c0][i] = (s ? -1 : +1) * piSigns[g*32 + c0] * H32[c0][i]
s = parity(c1 & g)
```

bf16 for the bf16 writer and for the decode kernel, 16384 elements, 32768 bytes. The sign
diagonal is already folded in, so `pi_signs` is **not read by the device** by any operator
that takes `tables`. It stays in the published signatures for ABI stability; an empty tensor
is the supported way for a caller to say so and the adapter already passes `nullptr` for it.

Host and device must compose the same matrix. `tq_oracle::audit_tables` composing to
`d*d = 65536` is the existing check (I11) and is a release gate, not a test detail: a
drifted table returns plausible wrong numbers (cosine ~0.33), never an error.

### 3.5 Query, gate and output

All bf16, all in vLLM's native layout — **no host-side gather, no dtype conversion**:

```
query  : [T, H_q, d]  bf16, contiguous   (T = num_tokens_padded)
gate   : [T, H_q, d]  bf16, contiguous, optional
out    : [T, H_q, d]  bf16 (viewed from the caller's [T, H_q*d])
```

An `M` tile of `(request b, kv_head h, spec token t)` reads rows
`query[b*(1+k) + t, h*6 + g, :]` for `g in [0, 6)`. With q-heads stored group-contiguous
(head `h*6 + g` belongs to KV head `h` — the standard vLLM layout) those 6 rows are
contiguous, so the tile's A operand is a **view**, and the whole `(1+k)` x 6 row set of one
tile is contiguous only when `k == 0`; for `k > 0` the tile is `(1+k)` contiguous runs of 6
rows on a `H_q * d` stride, which `stage_page_nz` expresses as a strided ND->Nz ingress.

**Query padding requirement.** The Cube stages `M_pad = ceil(M/16)*16` rows (the ND->Nz
ingress asserts whole Nz rows) while the Mmad is issued with `m = M`, so rows `[M, M_pad)`
are read from GM and do not participate. The backend must therefore guarantee **at least
`15` rows of addressable padding past the last used row of `query`** (`15 * d * 2 = 7680 B`).
The persistent query buffer provides it; this is one line in the allocation and it is the
alternative to a per-layer per-step gather.

`gate` is read per row, exactly `M` rows, so it needs no padding.

**Gate dtype changes from the probe.** The harness kernel takes `__gm__ float* gate`; the
production op takes bf16 and does the `sigmoid` in fp32 internally. That removes a host cast
of `T * H_q * d` elements per layer per step.

---

## 4. Torch-library schemas

Added to `csrc/attention/turboquant/op_adapter/turboquant_torch_ops.h`, registered through
the existing single `TORCH_LIBRARY_FRAGMENT_EXPAND(CONCAT(_C, _ascend), ops)` from
`csrc/torch_binding.cpp`. PrivateUse1 kernel dispatch, as today.

Eight entry points for six kernels; the numbering follows the roadmap's §0.1.

```
# No.2 — aligned prefill ingestion (MIX). num_tokens % 16 == 0; cu_seqlens keeps a strip
#        from straddling a request boundary. Was O1 npu_turboquant_reshape_and_cache.
npu_turboquant_reshape_aligned(Tensor key, Tensor value,
                               Tensor! key_cache, Tensor! value_cache,
                               Tensor! scale_cache, Tensor slot_mapping,
                               Tensor pi_signs, Tensor tables,
                               Tensor? cu_seqlens=None) -> ()

# No.1 — sub-strip ingestion (AIV). 1..16 tokens: per-step decode writes and ragged
#        prefill tails. Arbitrary non-negative slots. Was O2 ..._decode_cache_update.
#        `tables` is absent: the Pi transform here is the Vector butterfly (No.6), which
#        reads `pi_signs` directly rather than a Cube table.
npu_turboquant_reshape_scatter(Tensor key, Tensor value,
                               Tensor! key_cache, Tensor! value_cache,
                               Tensor! scale_cache, Tensor slot_mapping,
                               Tensor pi_signs,
                               Tensor? cu_seqlens=None) -> ()

# No.5 — batched Q rotation (Cube). Q_rot = Q . Pi^T for every (token, q_head) vector,
#        64 accumulating Mmads per row tile against the 16 resident blocks. `num_rows`
#        is B * H_q * (1 + k); the caller may pass fewer rows than `query` holds.
#        No scale operand: the 1/sqrt(d) orthonormalisation is folded into `tables`.
npu_turboquant_rotate_q_batch(Tensor query, Tensor pi_signs, Tensor tables,
                              Tensor! query_rot, int num_rows) -> ()

# No.4 — GQA-folded paged decode (MIX). One task per (request, kv_head, temporal split).
#        Takes the ALREADY ROTATED query from No.5 — the rotation operands and the
#        `query_rot` output of the former O3 are gone with it.
#        output_stage: 0 = raw Acc + (m, l) to workspace, 1 = rotated, 2 = rotated + gated.
npu_turboquant_gqa_paged_decode(Tensor query_rot, Tensor? gate,
                                Tensor key_cache, Tensor value_cache,
                                Tensor scale_cache,
                                Tensor block_tables, Tensor seq_lens,
                                Tensor pi_signs, Tensor egress_offsets,
                                Tensor egress_signs,
                                Tensor! workspace,
                                int num_heads, int num_kv_heads,
                                int query_tokens_per_req, int temporal_splits,
                                float scale_value, int output_stage,
                                Tensor! out) -> ()

# No.3 — 6-row paged decode (AIV). Same contract as No.4 and the same operand list: it
#        serves query_tokens_per_req == 1 at ANY batch size, and it needs the temporal
#        split for exactly the case it exists for (at B = 1 the (request, kv_head) grid is
#        4 tasks). The only difference is the engine, so the two are interchangeable at the
#        call site and the choice is a planner decision, not a shape constraint.
npu_turboquant_single_token_decode(Tensor query_rot, Tensor? gate,
                                   Tensor key_cache, Tensor value_cache,
                                   Tensor scale_cache,
                                   Tensor block_tables, Tensor seq_lens,
                                   Tensor pi_signs, Tensor egress_offsets,
                                   Tensor egress_signs,
                                   Tensor! workspace,
                                   int num_heads, int num_kv_heads,
                                   int temporal_splits,
                                   float scale_value, int output_stage,
                                   Tensor! out) -> ()

# — cross-page combine (AIV). Only when temporal_splits > 1. Reads the partials No.3/No.4
#   wrote at output_stage 0, then the egress rotation, then the gate.
npu_turboquant_decode_combine(Tensor workspace, Tensor? gate,
                              Tensor seq_lens,
                              Tensor pi_signs, Tensor egress_offsets,
                              Tensor egress_signs,
                              int num_heads, int num_kv_heads,
                              int query_tokens_per_req, int temporal_splits,
                              int output_stage, Tensor! out) -> ()

# — host planning.
npu_turboquant_decode_workspace_size(int num_reqs, int num_kv_heads, int head_size,
                                     int query_tokens_per_req,
                                     int temporal_splits) -> int
npu_turboquant_vector_core_num() -> int
```

Meta/FakeTensor registrations for every mutating op in `csrc/torch_binding_meta.cpp` and
`vllm_ascend/meta_registration.py`, as no-ops, so `torch.export` and FakeTensor tracing work.

### 4.0 Three operand hazards the schemas cannot express

These are the places where two operands share a name and differ in content. Each returns
plausible wrong numbers rather than an error, which is the failure mode this family has
already been bitten by: a drifted `turboquant_codec_tables` against
`TurboQuantCodec<4>::ConstTableWords` gives cosine ≈ 0.33 and never an exception.

**H-A — `tables` is normalised differently for No.2 than for No.4/No.5.** The writer's
equation (2) is

```
T_write[s][g][c0][i] = (s ? -1 : +1) . piSigns[g*32 + c0] . H32[c0][i]
```

with `invSqrtLen = 1/sqrt(d)` folded into the *stored scale* instead — scaling `Y` and
`absmax` together leaves every nibble unchanged, and that is what keeps the quantiser
bit-exact against an integer oracle. The decode-side rotation table carries the factor:

```
T_rot[s][g][c0][i] = (s ? -1 : +1) . piSigns[g*32 + c0] . H32[c0][i] / 16
```

(`routine_cube_decode_score.hpp`, "Why 64 Mmads and not 16"). `1/16 = 2^-4` is exact in
bf16, so `T_rot` entries are exactly `±0.0625`. **These are two distinct 16384-element
blobs**, and passing one where the other belongs scales every score by 16 or by 1/16. Name
them apart in Python (`turboquant_write_tables` / `turboquant_rotate_tables`) and have the
adapter check a cheap invariant on element 0: magnitude `1.0` for No.2, `0.0625` for No.5.

**H-B — `pi_signs` is dead in No.2 and live everywhere else.** No.2 folds the sign diagonal
into `tables` and never reads the operand; the adapter already accepts an empty tensor and
passes `nullptr`. No.1, No.3, No.4 and the combine all read it, because the Vector butterfly
applies the sign diagonal as its own stage. An empty `pi_signs` reaching any of those four
is a null dereference, not a no-op, so the adapter must require `numel() == head_size` for
them and tolerate `0` only for No.2.

**H-C — `scale_value` is a new operand, not a renaming.** The probe applies **no softmax
scale at all**. The only multiply on a score row is the per-token key scale
(`Mul(scaled_row, panel_row, key_scale)`, `routine_vector_decode_recurrence.hpp:181`), and
the host oracle matches it — `p[t] = exp(scores[t] - m_new)` over
`S[q][t] = keyScale[t] . R[q][t]`. The oracle's `kInvSqrtLen = 1/16` is the
**orthonormalisation of `Pi`**, consumed by the rotation table: `Pi . Pi^T = I` is what makes
the score rotation-invariant, and it does not scale the softmax. So the probe computes
`softmax(q.k^T)`, not `softmax(q.k^T / sqrt(d))`, and production must supply
`scale_value`.

Fold it into the 16-lane key-scale vector at the scale ingress, never onto the 4096-entry
page: 16 multiplies instead of 4096, recurrence unchanged. The exactness argument survives
**provided `scale_value` is a power of two** — the key scales are `2^(e-2)` and
`1/sqrt(256) = 2^-4`, so the product stays a power of two, the score panel stays exactly
representable, and the bit-exact gate stays available. The adapter should `TORCH_CHECK` that
`scale_value` is dyadic and say in the message that a non-dyadic scale demotes the
score-panel gate from bit equality to a tolerance. For this family `d` is always 256, so the
check always passes; it is there for the day it does not.

### 4.1 Adapter contracts (`turboquant_torch_adpt.h`)

Shared by every op, failing loudly rather than silently:

* `key`/`value`/`query`/`gate` bf16, contiguous, 3-D `[*, heads, head_size]`.
  Contiguity is checked because a fused QKV projection hands over column slices.
* `head_size == 256`, `num_kv_heads == 4` — the kernels' compile-time constants, checked
  rather than passed as free parameters. **New:** `num_heads % num_kv_heads == 0` and
  `num_heads / num_kv_heads == 6` for this kernel family (I1).
* `key_cache`/`value_cache` int8, `scale_cache` fp32, each checked as a **byte budget**
  (`numel >= P*H*S*d/2`, `numel >= 2*P*H*S`) plus contiguity and burst alignment — **not**
  as a sizes tuple. A sizes-tuple check is what let F4 through: `(P, S, H, d/2)` and
  `(P, H, S, d/2)` have the same `numel` and the check cannot tell them apart, so it
  certified an axis order the kernel does not use. The adapter must not impose PyTorch
  stride semantics on a hardware-packed plane (§3.1, §3.1a); `P`, `S` and `H` arrive as
  launch scalars and are cross-checked against the byte budget.
* `block_size % 16 == 0`.
* int32 for `slot_mapping`, `block_tables`, `seq_lens`, `cu_seqlens`.
* `check_burst_aligned` on every GM operand base (I6; a violation surfaces as CANN error
  264, not an alignment report).
* `tables` bf16, `numel == 16*32*32`, plus the H-A normalisation check on element 0.
* `query` / `query_rot` guard band: `numel() >= (num_rows + tile - 1) * head_size`, with
  `tile = kRotateRowsPerTile` for No.5 and `tile = M_pad` for No.3/No.4 (§3.5). The adapter
  owns both constants, mirroring the kernels'.
* **No.3/No.4/combine:** `temporal_splits >= 1` and
  `num_reqs * num_kv_heads * temporal_splits <= npu_turboquant_vector_core_num()` — I8 is a
  correctness gate, and the adapter is where it belongs because the kernel cannot see the
  grid it was launched with.
* `VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS` keeps its current meaning for No.2 and gains a
  No.1 form (slots `>= 0`, distinct per `(block, head, strip)`, within `P*S`). Off by
  default; the cost is a D2H copy and a synchronisation per layer per step.

Per-operator, beyond the shared set:

| Op | Additional checks |
| --- | --- |
| No.2 | `num_tokens % 16 == 0`; `cu_seqlens` int32 `[B+1]` monotone with `cu_seqlens[B] == num_tokens`; `tables[0]` magnitude `1.0` (H-A); `pi_signs.numel()` is `0` or `head_size` (H-B) |
| No.1 | `1 <= num_tokens`; no `% 16` constraint; `pi_signs.numel() == head_size` (H-B); every slot `>= 0` — a `-1` padding slot must never reach this op, so slice to `num_actual_tokens` first |
| No.5 | `num_rows >= 1` and `num_rows % (num_heads) == 0` (whole tokens); `query_rot` bf16 with `numel() >= num_rows * head_size`; `tables[0]` magnitude `0.0625` (H-A); `pi_signs` may be empty — No.5's sign diagonal is in `tables` |
| No.3 | `query_tokens_per_req` is implicitly 1, so `num_actual_tokens == num_reqs`; refuse otherwise with a message naming No.4 |
| No.4 | `rows = 6 * query_tokens_per_req <= 48`; `M_pad = ceil(rows/16)*16 in {16, 32, 48}` must be a built bucket |
| No.3/No.4 | `scale_value` dyadic (H-C); `output_stage in {0, 1, 2}`; **`output_stage == 2` refused when `temporal_splits > 1`** (§7.5, I12) — the gate does not commute with the combine; `gate` present iff `output_stage == 2` |
| combine | `temporal_splits > 1`; `workspace.numel() >= npu_turboquant_decode_workspace_size(...)`; `gate` present iff `output_stage == 2` |
| all decode ops | `egress_offsets` int32 `[3, head_size]`, `egress_signs` fp32 `[3, head_size]`, `pi_signs` fp32 `[head_size]` — see §4.1a |

### 4.1a The egress tables, and the host constructor they need

No.6's three sub-DataBlock stages (strides `1, 2, 4`) cannot reach their partner lane with a
contiguous or masked operand, because a 4/8/16-byte tensor offset is not 32-byte aligned, so
they pay a `Gather` plus a sign multiply. The `Gather` footprint is per-element, which means
two tables per stage, built once on the host:

```
egress_offsets[stage][i] = (i ^ stride[stage]) * sizeof(float)    # BYTE offsets, int32
egress_signs  [stage][i] = (i & stride[stage]) ? -1.0f : +1.0f    # fp32
pi_signs      [i]        = PiSign(i)                              # fp32, +-1
stride = {1, 2, 4};  i in [0, head_size);  3 * 256 entries each = 3 KiB + 3 KiB
```

That is `6 KiB` of tables against the `16 KiB` a single 4096-entry offset vector would have
cost. The construction currently lives in the CAModel runner
(`tests/camodel/test_camodel_paged_decode_attention.cpp`); **it must move to
`vllm_ascend/attention/turboquant_layout.py` as `turboquant_egress_tables(head_size, device)`
and be single-sourced from there**, for the H-A reason — a host/device disagreement on a
`Gather` offset table reads the wrong lane and returns a plausible wrong rotation, never an
error. The existing determinism anchors apply: `turboquant_pi_signs`' LCG (seed
`0x5F3759DF`) must match the C++ host reference, and the layout twins must stay bit-identical.

Note the offsets are **byte** offsets and the kernel reads them as `uint32`. The schema
declares int32 (`TURBOQUANT_INDEX_DTYPE`) and the adapter reinterprets; the largest value is
`(head_size - 1) * 4 = 1020`, so the signedness never matters.

### 4.2 C++ launcher signatures

Mirroring the existing `*_impl(void* stream, uint32_t block_dim, ...)` convention. Scalars
are unqualified `uint32_t` because CANN copies the parameter list verbatim into
`aclrtlaunch_triple_chevrons_func.h`, which has no `using namespace std`. Activations are
`uint16_t` on the host leg: `bfloat16_t` lives behind `__CCE_AICORE__`, the two legs are
separate translation units, and the host naturally holds bf16 as bit patterns.

```cpp
// No.2 — aligned prefill ingestion, MIX, grid-stride.
extern "C" void turboquant_reshape_aligned_impl(
    void* stream, uint32_t block_dim,
    const uint16_t* key, const uint16_t* value,
    int8_t* key_cache, int8_t* value_cache, float* scale_cache,
    const int32_t* slot_mapping, const float* pi_signs, const int32_t* tables,
    const int32_t* strip_begin,           // prefix sum, nullptr => one request
    const int32_t* cu_seqlens,            // nullptr => one request
    uint32_t num_requests, uint32_t num_tokens, uint32_t num_kv_heads,
    uint32_t head_size, uint32_t block_size, uint32_t num_blocks,
    uint32_t total_strips, float inv_sqrt_len);

// No.1 — sub-strip ingestion, AIV only, grid-stride over (request, kv_head).
extern "C" void turboquant_reshape_scatter_impl(
    void* stream, uint32_t block_dim,
    const uint16_t* key, const uint16_t* value,
    int8_t* key_cache, int8_t* value_cache, float* scale_cache,
    const int32_t* slot_mapping, const float* pi_signs,
    const int32_t* cu_seqlens,            // nullptr => one token per request
    uint32_t num_requests, uint32_t num_tokens, uint32_t num_kv_heads,
    uint32_t head_size, uint32_t block_size, uint32_t num_blocks,
    float inv_sqrt_len);

// No.5 — batched Q rotation, Cube, grid-stride over row tiles.
extern "C" void turboquant_rotate_q_batch_impl(
    void* stream, uint32_t block_dim,
    const uint16_t* query, const float* pi_signs, const int32_t* tables,
    uint16_t* query_rot,
    uint32_t num_rows, uint32_t head_size);

// No.4 — GQA-folded paged decode, MIX. Takes the rotated query; no rotation operands.
extern "C" void turboquant_gqa_paged_decode_impl(
    void* stream, uint32_t block_dim,
    const uint16_t* query_rot, const uint16_t* gate,   // gate may be nullptr
    const int8_t* key_cache, const int8_t* value_cache, const float* scale_cache,
    const int32_t* block_tables, const int32_t* seq_lens,
    const float* pi_signs, const int32_t* egress_offsets, const float* egress_signs,
    float* workspace, uint16_t* out,
    uint32_t num_requests, uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t head_size, uint32_t block_size, uint32_t num_blocks,
    uint32_t max_blocks_per_seq, uint32_t query_tokens_per_req,
    uint32_t temporal_splits, uint32_t output_stage, float scale_value);

// No.3 — 6-row paged decode, AIV only. Identical operand list minus
// query_tokens_per_req, which is 1 by definition.
extern "C" void turboquant_single_token_decode_impl(
    void* stream, uint32_t block_dim,
    const uint16_t* query_rot, const uint16_t* gate,
    const int8_t* key_cache, const int8_t* value_cache, const float* scale_cache,
    const int32_t* block_tables, const int32_t* seq_lens,
    const float* pi_signs, const int32_t* egress_offsets, const float* egress_signs,
    float* workspace, uint16_t* out,
    uint32_t num_requests, uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t head_size, uint32_t block_size, uint32_t num_blocks,
    uint32_t max_blocks_per_seq, uint32_t temporal_splits,
    uint32_t output_stage, float scale_value);

// — cross-page combine, AIV only.
extern "C" void turboquant_decode_combine_impl(
    void* stream, uint32_t block_dim,
    const float* workspace, const uint16_t* gate, const int32_t* seq_lens,
    const float* pi_signs, const int32_t* egress_offsets, const float* egress_signs,
    uint16_t* out,
    uint32_t num_requests, uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t head_size, uint32_t query_tokens_per_req,
    uint32_t temporal_splits, uint32_t output_stage);
```

Five notes, each the reason a parameter is there rather than derived:

* **`total_strips` and `strip_begin` on No.2.** The adapter computes `total_strips` as
  `sum_b ceil(L_b / 16)`, which is **not** `ceil(sum_b L_b / 16)`, and uploads the prefix sum
  so the device resolves `(request, strip)` with one scalar scan (§5.3). Neither is derivable
  on device without the prefix sum.
* **`inv_sqrt_len` on No.1/No.2 and not on the decode ops.** It folds into the *stored scale*
  of the quantiser, which is a writer-side concern. The decode side's `1/sqrt(d)` is already
  in `tables` (H-A), and its softmax scale is `scale_value` (H-C). Three different factors,
  three different homes — the former O3/O4 signatures (superseded) carried an `inv_sqrt_len` that had no
  job and it is gone.
* **`num_blocks` everywhere a plane is touched.** It is the `kv` stride of the planar scale
  plane (§3.2); the decode probe's `kNumPages = 9` is a fixture constant and a mismatch moves
  the whole V plane silently (I14).
* **`output_stage` on No.3 as well as No.4.** Both write partials at `0`, and the combine is
  shared, so the AIV decode is not a split-free special case.
* **No `query_rot` output on the decode ops.** It was an input-side workspace of the former
  the former O3's in-kernel rotation; No.5 owns it now and it is a real tensor, not scratch.

---

## 5. No.2 `npu_turboquant_reshape_aligned` — modifications to `turboquant_reshape_and_cache_bf16.cpp`

Baseline: the `feat/tq-cube-clean` copy (which has the in-page offset; the harness copy is
stale, audit §1). Four changes, in dependency order.

### 5.1 Persistent grid-stride work distribution

Replaces both the `kStripsPerLaunch = 32` host loop and the `blockDim = total_strips` grid.

```cpp
const uint32_t block  = GetBlockIdx() / GetSubBlockNum();   // divide this
const uint32_t stride = GetBlockNum();                      // do NOT divide this
stage_rotation_tables_once();                               // hoisted out of the loop
for (uint32_t s = block; s < total_strips; s += stride) {
    run_strip(s);                                           // prime, pipeline, drain
}
```

`blockDim = min(total_strips, core_count)` at the launcher.

Why, and the measured size of it (audit §6.2-6.3): the grid can never exceed the core count
(F5 — 64 strips *spins*, cause unisolated), the 64 serialised launches of a 32k-token prefill
collapse to one, and the 32 KiB of rotation tables stop being re-staged per strip, which is
**33.3% of the writer's GM->L1 traffic** (32,768 of 98,304 B/strip) and 18.7% of the Cube's
attributed cycles.

`GetBlockNum()` must **not** be divided by `GetSubBlockNum()`: it returns `numBlocks` on
both roles, so dividing it gives the Cube a stride of `N` and its Vector cores `N/2`, the
two roles disagree about the iteration count, and the result is a deadlock rather than a
wrong answer.

Flag accounting across iterations is now the kernel's own problem rather than the host's.
Each iteration must prime its feedback cycles and drain them before the next, and the audit
of `paged_decode_attention.md` §4.8 applies: in program order every route must read
`SET, WAIT, SET, WAIT, ...` with nothing standing at an iteration boundary. The existing
trace-audit check 1b is exactly this check and must be extended to iteration boundaries.

### 5.2 Ragged tail strip

The `num_tokens % 16 == 0` check (I4) is deleted and the last strip of each request becomes
a masked strip of `n = L_b - 16*floor(L_b/16)` tokens, `1 <= n < 16`.

* **Vector stage:** quantise `n` tokens. The absmax reduction, the exponent extraction and
  the nibble pack all run over `n` of 16 lanes.
* **Key plane:** MTE3 writes `n * d/2` bytes. `d/2 = 128`, so every `n` gives a whole number
  of 32-byte bursts and the base `t0 * d/2` stays 32-byte aligned. No padding needed.
* **Value plane:** zero-pad rows `[n, 16)` *in UB* and write the full 2048 bytes. A partial
  channel-major strip is 256 runs of `n/2` bytes on an 8-byte stride — neither burst-sized
  nor aligned — so a masked write is not expressible here and padding is the only option.
* **Scale plane:** write `n` floats. `n*4` is not a whole burst for `n != 8, 16`, so this
  one needs `DataCopyPad`.

The pad tokens' slots are `t0 + n .. t0 + 15`, which is exactly the unwritten remainder of
the request's current page: owned by this request, and not readable because `seq_len` has
not advanced. **This is the one place the operator may write beyond the scheduled tokens,
and it is part of the contract, not an accident.** It must be stated in the op's docstring
because a KV-transfer connector that copies a page mid-prefill would see those bytes.

Zeros rotate to zeros (the transform is linear with `+-1` entries), and the padded tokens'
scale takes the §3.3 `absmax == 0` branch — which is why that branch is a prerequisite for
this change, not an independent cleanup.

### 5.3 `cu_seqlens`-aware strip resolution

Each grid-stride iteration resolves its own `(request, strip-within-request)`:

```
# host, in the adapter
strip_begin[b] = sum_{j<b} ceil(L_j / 16)      # prefix sum, int32, uploaded once
total_strips   = strip_begin[B]

# device, per iteration
b   = upper_bound(strip_begin, s) - 1          # scalar search over B entries
r   = s - strip_begin[b]                       # strip index within request b
t   = cu_seqlens[b] + 16*r                     # first token, global
n   = min(16, L_b - 16*r)                      # tail width
```

`B` is small (bounded by `max_num_seqs`) and this is a scalar read of caller-supplied GM
metadata — the same category as the existing single `slot_mapping[0]` read, not a scalar
read of computed UB data. A linear scan is fine; the branch is outside every pipeline stage.

This is what makes a strip never straddle a request boundary (audit §6.1), which is today a
silent-corruption path with the default `VALIDATE_SLOTS=0`.

### 5.4 What does not change

The rotation (64 Mmads against 16 resident `32x32` blocks, sign diagonal and outer Hadamard
factor folded into the resident operand), the power-of-two quantiser, the four-stage
MTE2 -> Cube/Fixpipe -> Vector -> MTE3 schedule, the chunk-per-Vector-core work split, and
all three plane index functions. The refactor is to **work distribution and shape coverage**,
not to the arithmetic or the cache format — so the existing bit-exact oracle stays valid
for every full strip, and only the tail needs new golden coverage.

---

## 6. No.1 `npu_turboquant_reshape_scatter` (new)

The per-step write. `num_tokens` arbitrary (`= B*(1+k)`), slots arbitrary and
non-consecutive.

**Shape:** grid `min(num_tokens * H, core_count)`, grid-stride over
`(token, kv_head)` pairs. One pair is one 256-channel K vector and one 256-channel V vector.

**Vector-only, no Cube.** This is the one place the audit's §3.2 verdict inverts: at `M = 1`
the Cube prologue stages 32 KiB of rotation tables to transform 512 B of payload, and
`GEMV.md` records that an `M = 1` Mmad leaves only 32-64 B valid in each 1024-byte L0C
sub-vector — a 16:1 waste. So `Pi . x` runs as the Vector butterfly: 8 radix-2
stages over 256 lanes with the sign diagonal between them, and one scale.

**It is not the same stage order as the egress.** `Pi = A . D . B` with `A` the strides
`{32, 64, 128}` and `B` the strides `{1, 2, 4, 8, 16}`, so the ingress `Pi . x` expands as
`A.(D.(B.x))` — strides `1, 2, 4, 8, 16` then the sign then `32, 64, 128` — while the egress
`x . Pi` expands as `((x.A).D).B` and runs the reverse, which is what
`routine_vector_egress_rotate.hpp` ships (`kEgressStrideOf`, `kEgressSignBeforeStage = 3`).
The shared leaf therefore carries **two stage tables**, and the ingress one puts its three
`Gather` stages (`s in {1, 2, 4}`, where the partner lane is inside the 8-lane DataBlock)
first rather than last. Reusing the egress table for the ingress is a silent wrong answer.

**The value-plane write is a read-modify-write of one 2048-byte strip.** A single token owns
256 nibbles at stride 16 inside the strip — 128 bytes spread over 2048, half-byte granular.
There is no aligned-burst expression of that. The RMW is safe because a request's current
page is written only by that request's own decode step, so no two cores contend for the same
strip within a launch; **the adapter must assert that `slot_mapping` contains no two tokens
in the same `(block, head, strip)`**, which uniform decode guarantees (one token per request
per step for `k = 0`) and MTP does **not** (the `1+k` tokens of one request are consecutive
and usually share a strip).

For MTP the fix is to make the unit of work `(request, kv_head)` rather than
`(token, kv_head)`: all `1+k` tokens of a request go into the strip in one RMW. Specify that
form — it is strictly more general and removes the assertion.

**Key plane** needs no RMW: `token * d/2 = token * 128` bytes is 32-byte aligned and the
write is one 128-byte burst.

**Scale plane** is one fp32 per `(kv, token, head)` — 4 bytes, sub-burst. `DataCopyPad`, or
gather the `1+k` tokens of a request into one burst in the `(request, kv_head)` form.

### 6.1 Alternative considered and rejected

Making the decode write go through №2 by padding `B` to a multiple of 16 and faking
consecutive slots. Rejected: there is no slot run to fake (the `B` tokens are in `B`
different pages), and a reserved bit-bucket page cannot produce a *consecutive* 16-slot run
mixed with real slots. Audit §5.3.

---

## 7. No.4 `npu_turboquant_gqa_paged_decode` (promoting the harness probe)

Baseline: `turboquant_paged_decode_attention.cpp`. The arithmetic, the 5-stage modulo
schedule, the marked graph, the flag-id assignment and the two measured faults it fixed
(§4.5 the ring slot released by a non-last reader, §4.6 the L1 ring's mis-pointed back edge)
all carry over unchanged. Six changes.

### 7.1 Runtime query-row count

The single highest-value change in this document: **62.5% of the measured critical path at
`B = 1`** (audit §3.1, F6).

```
rows = 6 * query_tokens_per_req          # 6 = num_heads / num_kv_heads, asserted
M_pad = ceil(rows / 16) * 16             # 16, 32 or 48
```

| What | From | To |
| --- | --- | --- |
| `vector_decode_recur_strip`'s bound | `query < kDecodeQueries` (16) | `query < rows` |
| `EgressQueries`' loop and gate `DataCopy` | `query < kTokens` | `query < rows` |
| `Duplicate` seeds of `m_prev`, `l_prev`, `acc` | `kTokens * ...` | `rows * ...` |
| final `DataCopy` extents | `kTokens * kHeadSize` | `rows * kHeadSize` |
| `MmadParams::m` | `kDecodeTokens` | `rows` |
| ND->Nz query ingress, L1->L0 load, L0C window | `kTokens` | `M_pad`, dispatched over 3 buckets |
| the `Duplicate` over-read `static_assert` | `(kTokens-1)*...` | `(M_MAX-1)*...`, `M_MAX = 48` |

The Cube needs no mask: `MmadParams::m` is a `uint16_t` runtime extent, and `Mmad.md`
states that when M is not a multiple of 16 the hardware still organises 16x16 fractals and
"invalid data in the tail block occupies fractal block space but does not participate in
valid computation". So no `SetTail`, no masked accumulation, no host-side query padding
beyond §3.5's 15-row guard band (audit §4.3).

The Vector side costs nothing to convert: `select_query` already takes a runtime query
index and the loop is a real loop, not an unrolled sequence (deliberately — unrolling 16
queries at 8 strip sites did not finish compiling in 25 minutes). Only the bound is a
constant today.

`M_pad` is a compile-time bucket because the L0 fractal counts and the Mmad nest *are*
unrolled index sequences. Three instantiations (1, 2, 3 fractals) cover `rows <= 48`, which
§7.4 shows is the UB ceiling anyway.

### 7.2 Grid, and the page loop

```
grid  = num_requests * num_kv_heads * temporal_splits
task  = (b, h, j)
pages_total = ceil(seq_lens[b] / block_size)
pages_lo    = (j     * pages_total) / temporal_splits
pages_hi    = ((j+1) * pages_total) / temporal_splits
for (page_idx = pages_lo; page_idx < pages_hi; ++page_idx)
    for (strip = 0; strip < strips_in_page; ++strip)
        ... existing 5-stage column body ...
```

with `strips_in_page = block_size / 16` (8 at the vLLM default, now a runtime scalar rather
than `kStrips = 8`) and the last page of a request masked to `seq_lens[b] % block_size`
tokens — a key strip beyond the sequence must contribute `-inf` scores, not garbage, which
the recurrence expresses by setting those score columns to a large negative before the Max.

`temporal_splits == 1` is the no-combine path: one task owns a request's whole context for
one KV head, the online-softmax recurrence carries `(m, l, Acc)` across every strip of every
page, and `out` is the final answer. **That deletes the combine from the critical path and the
partial-state workspace entirely** for any `B` where occupancy allows it. Audit §4.4.

### 7.3 L1 key staging back to a ring

The probe gave each strip its own L1 page (8 pages, 112.3 KiB) to escape the §4.6 flag
fault. A page loop makes the strip count `8 * ceil(S/128)`, so one page per strip is
impossible — `paged_decode_attention.md` §8 puts the ceiling at ~56 strips.

Specified: a ring of depth `D` (start at 4) with **one flag id per ring slot**, and the
§4.6 question answered before this lands rather than worked around. The trace audit's check
3 — every L1 key page written in 16 bursts and read exactly once, after its last write — is
the acceptance condition, and it must run on a rung that wraps the ring at least twice
(`strips >= 2D`).

This is the riskiest change in the document. `paged_decode_attention.md` §8 is explicit that
the three surviving cross-core channels are correct only because each consumer is
measurably slower than its producer, and §7.1 makes the recurrence `rows/16` times faster.
**The flag-id audit must be re-run at every `rows` bucket**, not once.

### 7.4 UB budget and the row ceiling

From audit §3.3, with the probe's two verification buffers (`scoreArchive`, `out_rotated`,
24.28 KiB) removed — reproduce with
[`scripts/decode_ub_budget.py`](../scripts/decode_ub_budget.py):

| rows | fractals | production arena | verdict |
| --- | --- | --- | --- |
| 6 (`k=0`) | 1 | 109.97 KiB | fits |
| 12 (`k=1`) | 1 | 120.09 KiB | fits |
| 24 (`k=3`) | 2 | 140.34 KiB | fits |
| 48 (`k=7`) | 3 | 180.84 KiB | fits the 192 KiB window |
| 64 | 4 | 207.84 KiB | needs the whole UB |
| 96 | 6 | 261.84 KiB | **does not fit** |

So `M_MAX = 48` and `rows > 48` must be split into several query tiles. Query tiles of one
request share pages and need no combine — only separate `(m, l, Acc)` — so the split is
free. Assert `rows <= 48` and tile above it.

The kernel must size against `GetUBSizeInBytes()` rather than the probe's
`kUbWindow = 192 KiB` literal; 192 KiB is the probe's self-imposed window, not the
hardware's.

### 7.5 `output_stage`, and where the gate goes

```
0  raw: Acc (rotated basis) + (m, l) -> workspace.   Used when temporal_splits > 1.
1  rotated: Acc/l . Pi -> out.                        Ungated layers.
2  rotated + gated: * sigmoid(gate) -> out.           Qwen3.5 full-attn layers, splits == 1.
```

`out_context` from the probe is stage 2. The constraint that makes `output_stage` necessary
rather than cosmetic: the egress rotation commutes with a cross-page log-sum-exp combine and
**the sigmoid gate does not**, so with `temporal_splits > 1` the gate must be applied by the combine,
after the combine (I12). Stage 2 with `temporal_splits > 1` must be refused by the adapter,
not silently produce a wrong answer.

`fold_pi_into_output_projection` stays ruled out for this model: `attn_output_gate = true`
on all 16 full-attention layers and the elementwise gate sits between attention and
`o_proj`, so `Pi` cannot be folded into `o_proj`.

### 7.6 Probe outputs become optional

`out_qrot`, `out_scores`, `out_state`, `out_acc`, `out_rotated` are the probe's
checkability surface. In the product, `query_rot` stays (the prologue's GM round trip needs
it as a real destination — `CopyL0C2L1` with a bf16 Nz destination delivered nothing L0A
could read, mechanism unisolated), `workspace` carries `(m, l, Acc)` when
`output_stage == 0`, and the rest go behind the verification build. That frees the 24.28 KiB
of §7.4 and removes the score archive's 8 per-strip `Muls`-by-1.0 copies from AIV0 — which
is the measured critical path.

### 7.7 The prologue round trip — removed by the No.5 split, not fixed

The former text here left the round trip in place deliberately. The six-kernel split
removes it instead, and the distinction matters: No.4 needed `Q_rot` back as an **L0A
operand in the same kernel**, which is why it went L0C -> GM -> L1 -> L0A. No.5's output
*is* GM, so it is one `CopyL0C2GM` per row tile and nothing more.

**This is not evidence that `CopyL0C2L1` works.** The one recorded attempt produced
all-256-zero score panels in a run where the same L0C reached GM bit-exactly through
`CopyL0C2GM`, mechanism unisolated. That question is simply no longer on any critical path,
and it stays open in §11.

---

## 8. `npu_turboquant_decode_combine` (new)

Only on the `temporal_splits > 1` path. Per `(b, h, row)`, over the `splits` partials:

```
m   = max_j m_j
l   = sum_j l_j * exp(m_j - m)
Acc = sum_j Acc_j * exp(m_j - m)          # Acc_j unnormalised, in the rotated basis
y   = (Acc / l) . Pi
out = y * sigmoid(gate)                    # output_stage 2
```

Grid `min(num_requests * num_kv_heads, core_count)`, grid-stride, Vector-only. Workspace
layout, which №3/№4 write and the combine reads:

```
workspace : fp32, [splits, num_requests, num_kv_heads, rows, head_size + 16]
            [..., 0:head_size]            = Acc
            [..., head_size : head_size+8]   = m, replicated over the 8 lanes of a DataBlock
            [..., head_size+8 : head_size+16] = l, likewise
floats    = splits * num_requests * num_kv_heads * rows * (head_size + 16)
```

The replicated-DataBlock representation of `m` and `l` is the probe's existing
`kDecodeStateLanes = 8` invariant and is what lets the recurrence run without a single
scalar read of UB; keeping it in the workspace means the combine reads the same shape №3/№4 wrote, with
no repacking.

`npu_turboquant_decode_workspace_size` returns exactly that `floats` count, and `0` when `splits == 1`.

---

## 9. Python integration

### 9.1 `vllm_ascend/attention/turboquant_v1.py`

Port from `feat/tq-main-clean` and change:

```python
class AscendTurboQuantAttentionBackend(AscendAttentionBackend):
    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str="") -> tuple[int, ...]:
        # PHYSICAL ALLOCATION BYTES, not an indexable logical shape. The planes
        # are hardware-packed (RFC §3.1a): the key plane carries fractal packing
        # inside each strip that has no PyTorch stride expression, and the value
        # plane's in-strip order gives it a different rank from the key plane's.
        # Nothing may index these tensors directly -- use the accessors below.
        ...
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]          # any multiple of 16 is expressible; 128 is what is verified
```

```python
# vllm_ascend/attention/turboquant_layout.py -- the single source of truth for
# the physical layout, mirroring the C++ index functions of RFC §3.1 term for
# term. Every host-side reader goes through these; none builds a view and
# indexes it.
def turboquant_key_byte(block, head, token, channel, block_size, num_kv_heads, head_size): ...
def turboquant_value_byte(block, head, token, channel, block_size, num_kv_heads, head_size): ...
def turboquant_scale_float(kv, block, head, token, num_blocks, num_kv_heads, block_size): ...

# The three table families, single-sourced here for the same reason (H-A, H-B, §4.1a):
# a drifted copy returns a plausible wrong number and never an exception.
def turboquant_write_tables(head_size, device): ...    # No.2, entries +-1
def turboquant_rotate_tables(head_size, device): ...   # No.4/No.5, entries +-1/16
def turboquant_egress_tables(head_size, device): ...   # (offsets int32, signs fp32), 3x256
def turboquant_pi_signs(head_size, device): ...        # existing, LCG seed 0x5F3759DF
```

`AscendTurboQuantAttentionSpec` (`core/kv_cache_interface.py`) needs no structural change:
it already budgets `payload_bytes`, `scale_plane_bytes` and `real_page_size_bytes` rather
than element shapes, which is exactly the right abstraction here. What must change is that
nothing downstream reconstructs a logical view from those bytes.

`turboquant_layout.turboquant_dequantize` is the one existing host reader and must be
rewritten onto the accessors; today it would read the wrong `(head, token)` pair (audit
§2.4). `swap_blocks` / `copy_blocks` are already correct because they move whole pages.

**This is the F4/I13 fix in full, and it moves no bytes.** The physical layout is right and
stays as it is; only the description changes, from a shape that contradicts it to accessors
that state it.

```python
class AscendTurboQuantAttentionBackendImpl(AscendAttentionBackendImpl):

    def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
        # The dispatch F3 says is missing today.
        if attn_metadata.attn_state in (AscendAttentionState.DecodeOnly,
                                        AscendAttentionState.SpecDecoding):
            torch.ops._C_ascend.npu_turboquant_reshape_scatter(...)       # №1
        else:
            torch.ops._C_ascend.npu_turboquant_reshape_aligned(           # №2
                ..., cu_seqlens=self._cu_seqlens(attn_metadata))
        return query, key, value, output

    def _decode_plan(self, attn_metadata) -> tuple[int, int]:
        """(temporal_splits, workspace_floats). I8 is a correctness gate."""
        cores = torch.ops._C_ascend.npu_turboquant_vector_core_num()
        b     = attn_metadata.num_actual_tokens // self.query_tokens_per_req
        pages = cdiv(int(attn_metadata.seq_lens_cpu.max()), self.block_size)
        splits = max(1, min(cores // (b * self.num_kv_heads), pages))
        assert b * self.num_kv_heads * splits <= cores
        return splits, torch.ops._C_ascend.npu_turboquant_decode_workspace_size(...)

    def forward_paged_attention(self, query, attn_metadata, output, output_gate=None):
        splits, _ = self._decode_plan(attn_metadata)
        stage = 0 if splits > 1 else (2 if output_gate is not None else 1)
        torch.ops._C_ascend.npu_turboquant_gqa_paged_decode(...)      # №4
        if splits > 1:
            torch.ops._C_ascend.npu_turboquant_decode_combine(...)        # combine
        return output
```

Keep the `feat/tq-main-clean` patterns that exist for graph-capture reasons and are easy to
lose: `_decode_workspace` and `_rotated_query` are **persistent** buffers because a captured
graph replays only device work, so a Python-side allocation never runs again after capture;
both must be monotonically grown outside capture and never allocated inside it.

New validation, raised at `__init__` rather than discovered at the first launch:

* `num_heads % num_kv_heads == 0` and `num_heads // num_kv_heads == 6` (I1);
* `head_size == 256`, `num_kv_heads == 4`;
* `sliding_window is None`, `attn_type == DECODER`, `logits_soft_cap` unset;
* `kv_sharing_target_layer_name is None`.

### 9.2 Config-time validation — `validate_turboquant`

**The hook an earlier draft of this section named does not exist on the branch these kernels
come from.** `vllm_ascend/platform.py::_validate_turboquant_config` is a `feat/tq-main-clean`
and `develop-turboquant` symbol. On `feat/tq-cube-clean` (HEAD `7e5cecf20`, the branch the
harness probes and `turboquant_torch_adpt.h` were vendored from) validation has moved to
`vllm_ascend/quantization/methods/kv_cache/turboquant/config.py::validate_turboquant`, called
from `platform.py:508-510`, and split three ways: `_validate_turboquant_common`,
`_validate_turboquant_sfa`, `_validate_turboquant_dsa`.

On that branch the Qwen3.5-on-950PR target is rejected at config time, twice over. Neither
rejection is a bug; each is a true statement about code that does not exist yet.

1. **SoC capability.** `_validate_turboquant_common` (`config.py:20-21`) requires
   `HardwareCapability.TURBOQUANT_4BIT_NC_CACHE`. That capability is granted inside
   `_STANDARD_CAPABILITIES` (`hardware_profile.py:275`), which A2 takes directly and A3 by
   union. The **A5 profile spells its 33 capabilities out explicitly and does not include
   it** (`hardware_profile.py:338-385`), and `"ascend950…" -> AscendDeviceType.A5` is the
   whole of the 950 mapping (`hardware.py:53-58`; `is_950()` is `== A5`,
   `device_config.py:61-62`). So the target raises `"only supported on Ascend A2/A3"` — and
   on this branch that message is literally true.
2. **Model family.** `validate_turboquant`'s dispatch is binary and total: SFA-sparse models
   to `_validate_turboquant_sfa`, **everything else** to `_validate_turboquant_dsa`, whose
   first check (`config.py:63`) raises `"currently requires DeepSeek V4"`. A dense-GQA model
   is not SFA-sparse, so Qwen3.5 reaches the DeepSeek validator by falling off the end of an
   `if` (`config.py:107`) rather than by being chosen.

So §9.2's ask is not "lift a rejection" but **add a third branch**:

* grant `TURBOQUANT_4BIT_NC_CACHE` to the A5 profile, gated on the §4 operators existing.
  This single line is what turns the feature on for 950PR, so it is the **last** thing
  flipped, after Phase 4 — not a bring-up convenience.
* make the dispatch three-way, `_validate_turboquant_gqa` for the dense-GQA page layout,
  selected on something narrower than "not SFA" — the spec class
  (`AscendTurboQuantAttentionSpec`, `core/kv_cache_interface.py:373`) or an explicit
  model-family predicate. Falling into a sibling family's validator by default is what makes
  today's message misleading rather than merely premature.
* in that branch: `num_heads // num_kv_heads == 6` (I1), `head_size == 256`,
  `num_kv_heads == 4`, `block_size == 128`, bf16 (already covered by the common path), and an
  explicit statement for `model_config.is_hybrid` — either support under the chosen Phase-0
  allocator option, or a clean rejection naming the three-plane carve (§9.4).
* prefix caching and chunked prefill stay rejected **until** No.2's §5.2/§5.3 changes land,
  then lift: a ragged-tail, `cu_seqlens`-aware writer is exactly what chunked prefill needs.
  Name No.2 in the message so the rejection has a visible exit.
* spec decoding is "supported with `query_tokens_per_req = 1 + k`", the Qwen3.5 MTP case and
  the regime No.4 is best shaped for (audit §4.2).
* the PIECEWISE/NONE graph downgrade is a **port, not a keep**: there is no TurboQuant
  graph-mode hook on the cube branch. `_apply_turboquant_graph_mode`
  (`feat/tq-main-clean:platform.py:556-614`) holds a TurboQuant run to a replayable ACL
  graph mode, and it must come across with the backend (audit §4.1).

The `feat/tq-main-clean` observation that "the engine passes validation and fails at cache
reshape with `len(raw_planes) == 3`" is branch-specific and does not hold here: on the cube
branch nothing reaches the allocator. The three-plane conflict is still real (§9.4); it is
simply not the first thing a run hits.

### 9.3 `vllm_ascend/patch/worker/patch_qwen3_5.py`

The current file has no TurboQuant hook: the gate is `torch.sigmoid(gate)` followed by a
multiply (`:93-95`). Add the `feat/tq-main-clean` branch, guarded:

```python
if self.attn_output_gate and turboquant_fuses_output_gate(self.attn):
    attn_output = turboquant_gated_attention_forward(self.attn, q, k, v, gate)
else:
    attn_output = self.attn(q, k, v)
    if self.attn_output_gate:
        attn_output = attn_output * torch.sigmoid(gate)
```

`turboquant_fuses_output_gate` must now also require `temporal_splits == 1`, because §7.5
forbids stage 2 with a combine. Since `splits` is a per-step quantity and
`fuses_output_gate` is a per-layer property, the honest shape is: the layer always hands the
gate to the backend, and the **backend** decides per step whether it goes to №3/№4 (stage 2) or
to the combine. That keeps the gate out of torch on both paths and removes the per-step branch from
the model patch. Per the repo's patching rule, all of this stays in `patch/worker/`, not in
a model file.

Both entry ops stay graph-splitting custom ops with `mutates_args=["output"]` and
`dispatch_key="PrivateUse1"`.

### 9.4 `vllm_ascend/worker/model_runner_v1.py`

**Correction to an earlier draft of this section**, which said
`hybrid_with_attn_and_mamba` is "a global flag set while iterating tensors" and asked for it
to be made per-tensor. Both halves are wrong. It is one expression over every spec in the
model (`:4987-4989`):

```python
self.hybrid_with_attn_and_mamba = any(
    isinstance(spec, MambaSpec) for spec in layer_kv_cache_spec.values()
) and any(isinstance(spec, AttentionSpec) for spec in layer_kv_cache_spec.values())
```

It is model-wide **by design**, and the comment above it says why: vLLM #51718 stopped
listing layers from different cache groups in one descriptor, so detecting hybridity per
descriptor is precisely what broke. There is no ordering dependence — `any` over a dict's
values does not have one. Making it per-tensor would re-introduce the bug that comment
records.

The conflict is at the **consumer**, not the definition. At `:5157-5163` the flag is read
once per layer:

```python
if ("linear_attn" in layer_name
        or self.hybrid_with_attn_and_mamba
        or layer_name in strided_attention_cache_layers
        or ...) and layer_name not in kv_cache_raw_tensors:
    # ... one private contiguous buffer per layer
```

Because the flag is model-wide, in a hybrid model **every** layer takes the single-tensor
path — the 16 full-attention layers included. That path hands each layer one contiguous
buffer, and `AscendTurboQuantAttentionSpec` needs three planes carved out of its allocation
(§3.1, §3.2). The ask is therefore narrower than the draft's, and does not touch the flag:

* the single-tensor path needs a TurboQuant-aware branch: for a layer whose spec is
  `AscendTurboQuantAttentionSpec`, carve `(key, value, scale)` per §3.1/§3.2 instead of
  returning one buffer. Equivalently, add that spec to whatever predicate selects the carve.
  The flag keeps its current model-wide meaning either way.
* that carve must become reachable on the hybrid path at all (Phase 0; out of scope here,
  N1), and the plane views must follow §3.1 — byte ranges with accessors, not indexable
  logical shapes (§9.1).
* nothing else: `slot_mapping`, `block_tables`, `seq_lens_gpu` and the padding convention are
  all already what the operators want.

### 9.5 Everything else

| File | Change |
| --- | --- |
| `csrc/torch_binding_meta.cpp`, `vllm_ascend/meta_registration.py` | no-op meta kernels for every mutating op (§4) |
| `CMakeLists.txt` | add the five new translation units (№1, №3, №5, combine, and №4's rename) to `ascendc_library(vllm_ascend_turboquant ...)`; keep the `ascend950` gate; drop `turboquant_paged_attention_opt.cpp` once №4 lands, since its only purpose was to keep a schedule compiling |
| `vllm_ascend/envs.py` | `VLLM_ASCEND_TURBOQUANT_TEMPORAL_SPLITS` (0 = auto) for bring-up; document in the `env_variables` dict |
| `vllm_ascend/attention/turboquant_layout.py` | add the three plane accessors of §9.1 as the single source of truth, and move `turboquant_dequantize` onto them; today it indexes the declared view and reads the wrong `(head, token)` pair (F4, audit §2.4) |

---

## 10. Phasing and acceptance

There is **no NPU on this host**, so every gate below is CAModel plus host-oracle, and no
phase may claim a latency, throughput or power result. The verification method is the one
this harness already uses: an inductively collapsed grid (`kSlots + 2` tiles rather than the
full grid), guard-banded GM buffers, canary sweeps over everything a tile must not touch,
and a trace audit of the dumps.

**The phase table lives in the roadmap.**
[`qwen35_turboquant_execution_roadmap.md`](qwen35_turboquant_execution_roadmap.md) carries
the five phases, their per-kernel file lists, queue and HardEvent budgets, gates and
sequencing; an earlier P0-P8 table here duplicated it against the older O-numbering and is
removed rather than kept in sync. Two conditions from it are interface requirements and so
stay here:

* **the layout phase moves no bytes.** A UT must assert that each Python plane accessor
  returns the same byte as the C++ index function for every `(block, head, token, channel)`
  of `(P, S, H, d) = (9, 128, 4, 256)` — the host-side form of
  [`scripts/kv_plane_layout_check.py`](../scripts/kv_plane_layout_check.py), whose three
  checks (writer/decoder agreement, page-level nesting, `Nz(K) = Zn(K^T)` against a
  row-major control) are the acceptance conditions. A diff that changes a kernel's store
  address has failed it.
* **`splits > 1` must match `splits == 1` bit-for-bit** on the same fixture. That is the
  strongest available check on the combine, and it is available only because the page loop
  makes `splits == 1` correct first.

### 10.1 Accuracy, stated honestly

Every bit-exactness result in this family is over **integer** fixtures, where the Cube's
unspecified fp32 accumulation order does not matter. With general bf16 activations no
integer oracle reproduces it and the check must degrade to cosine plus a relative-L2 bound,
which no longer detects a wrong-strip read the way nibble comparison does. Such a test
belongs in a separate, clearly labelled runner and **must not replace** the integer one.
No LongBench-style accuracy gate has been run on this model; that is hardware work.

---

## 11. Open questions and risks

| # | Item | Why it matters | Who can answer it |
| --- | --- | --- | --- |
| R1 | the cause of F5 (a MIX grid wider than the core count spins) | §5.1 removes the symptom by never exceeding the core count, but the mechanism is unknown, and anything that reintroduces a wide grid reintroduces the hang | CAModel, an isolating 33-strip run with sampling compiled out |
| R2 | mode-4 flag-id reuse across ring slots | §7.3 is the riskiest change here. The probe's three surviving channels are correct only because every consumer is slower than its producer, and §7.1 speeds the consumer up by `rows/16` | CAModel trace audit at every `rows` bucket; ultimately a hardware question about whether the flag is a counter or a level |
| R3 | `CopyL0C2L1` with a bf16 Nz destination | removing it is the only way to delete the prologue's GM round trip; one recorded attempt gave all-zero score panels with the mechanism unisolated | CAModel, with an isolated minimal repro |
| R4 | the AI core count of the target part | §7.2's `splits` clamp and I8 both need it; a literal would be a latent hang | runtime (`GetBlockNum()` / `npu_turboquant_vector_core_num`), never a constant |
| R5 | concurrent execution of a full grid | no decode run has ever had two blocks computing at once — every rung isolates one of 32. The isolation argument (mode-4 is per core group, each block has its own UB/L1/L0, GM writes are disjoint and canary-verified) is sound but it is an argument | CAModel, if a full-grid run is affordable; one strip of one block is ~7 min of model time on this host |
| R6 | the hybrid allocator (N1) | upstream of all of this; a TurboQuant Qwen3.5 run dies at `len(raw_planes) == 3` before any kernel executes | vLLM-Ascend, Phase 0 of the integration map |
| R7 | KV-transfer connectors vs §5.2's pad bytes | a connector that copies a page mid-prefill sees zeros in the request's unwritten remainder. Harmless if `seq_len` gates the read, which it should — but the hybrid connectors assume the documented layout and have not been checked against this | vLLM-Ascend connector review |
| R8 | the №3/№4 crossover row count | §2 ships `rows == 6 -> №3`, `rows >= 12 -> №4` with an env override, and the roadmap gates №3 on an instruction census rather than a utilisation target. Which engine wins at 12 and 24 rows is unmeasured | CAModel first, then hardware |
