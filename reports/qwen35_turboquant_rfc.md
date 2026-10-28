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

G2. One cache layout, declared honestly, shared by every operator, and verified as such
rather than agreed by inspection (F4, I13).

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

| # | Operator | Status | Regime |
| --- | --- | --- | --- |
| O1 | `npu_turboquant_reshape_and_cache` | **modify** — ragged tail, `cu_seqlens`, grid-stride | prefill, all shapes |
| O2 | `npu_turboquant_decode_cache_update` | **new** | the per-step KV write (F3, §5.4) |
| O3 | `npu_turboquant_gqa_paged_decode` | **new** — promote the harness probe | decode + MTP, all `B` |
| O4 | `npu_turboquant_decode_combine` | **new** | cross-page log-sum-exp, only when `splits > 1` |
| O5 | `npu_turboquant_decode_workspace_size` | **new** | host sizing |
| O6 | `npu_turboquant_vector_core_num` | exists on `feat/tq-main-clean` | host planning, I8 |

O3 subsumes the single-token decode kernel the brief asks about: with a runtime row count
there is no UB, layout or scheduling argument for a separate `turboquant_single_token_decode`
(§3.3 — the `M = 6` arena is 109.97 KiB), and a second kernel would be a second place for
the layout contract to drift.

---

## 3. GM layout contract (normative)

This section supersedes the shape comments in `turboquant_torch_ops.h`. The index
functions are what the writer already emits (audit §2.1); only the *declaration* changes.

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

Honest torch shapes, which is what the schema must declare and the adapter must check:

```
key_cache   : [P, H, S,     d/2]
value_cache : [P, H, S/R, R*d/2]
```

Both are `P*H*S*d/2` bytes; they differ in rank and in axis meaning, and that is the point.
A single declared shape for both planes is the root of F4 and must not be restored.

Rationale for the value plane's strip blocking: channel-major over a whole 128-token page
would make one strip 256 runs of 8 bytes on a 64-byte stride — not burst-sized, not
burst-aligned, and two strips of a page would share a 32-byte word and tear each other.
Blocking by `R` makes every strip one aligned 2048-byte burst, and at `S == R` it is
byte-identical to plain channel-major, which is the form the CAModel probe verified.

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

```
# O1 — prefill ingestion. Change: cu_seqlens added (defaulted, so existing callers are
#      unaffected), and the num_tokens % 16 restriction is lifted.
npu_turboquant_reshape_and_cache(Tensor key, Tensor value,
                                 Tensor! key_cache, Tensor! value_cache,
                                 Tensor! scale_cache, Tensor slot_mapping,
                                 Tensor pi_signs, Tensor tables,
                                 Tensor? cu_seqlens=None) -> ()

# O2 — per-step KV write. Arbitrary token count, arbitrary non-negative slots.
npu_turboquant_decode_cache_update(Tensor key, Tensor value,
                                   Tensor! key_cache, Tensor! value_cache,
                                   Tensor! scale_cache, Tensor slot_mapping,
                                   Tensor pi_signs, Tensor tables) -> ()

# O3 — GQA-folded paged decode. One task per (request, kv_head, temporal split).
#      output_stage: 0 = raw Acc + (m,l) to workspace, 1 = rotated, 2 = rotated + gated.
npu_turboquant_gqa_paged_decode(Tensor query, Tensor? gate,
                                Tensor pi_signs, Tensor tables,
                                Tensor key_cache, Tensor value_cache,
                                Tensor scale_cache,
                                Tensor block_tables, Tensor seq_lens,
                                Tensor! workspace, Tensor! query_rot,
                                int num_heads, int num_kv_heads,
                                int query_tokens_per_req, int temporal_splits,
                                float scale_value, int output_stage,
                                Tensor! out) -> ()

# O4 — cross-page combine. Only called when temporal_splits > 1. Reads the partials O3
#      wrote with output_stage 0, applies the egress rotation and then the gate.
npu_turboquant_decode_combine(Tensor workspace, Tensor? gate,
                              Tensor pi_signs, Tensor tables, Tensor seq_lens,
                              int num_heads, int num_kv_heads,
                              int query_tokens_per_req, int temporal_splits,
                              int output_stage, Tensor! out) -> ()

# O5/O6 — host planning.
npu_turboquant_decode_workspace_size(int num_reqs, int num_kv_heads, int head_size,
                                     int query_tokens_per_req,
                                     int temporal_splits) -> int
npu_turboquant_vector_core_num() -> int
```

Meta/FakeTensor registrations for all of O1-O5 in `csrc/torch_binding_meta.cpp` and
`vllm_ascend/meta_registration.py`, as no-ops, so `torch.export` and FakeTensor tracing work.

### 4.1 Adapter contracts (`turboquant_torch_adpt.h`)

Shared by every op, failing loudly rather than silently:

* `key`/`value`/`query`/`gate` bf16, contiguous, 3-D `[*, heads, head_size]`.
  Contiguity is checked because a fused QKV projection hands over column slices.
* `head_size == 256`, `num_kv_heads == 4` — the kernels' compile-time constants, checked
  rather than passed as free parameters. **New:** `num_heads % num_kv_heads == 0` and
  `num_heads / num_kv_heads == 6` for this kernel family (I1).
* `key_cache`/`value_cache` int8 with the §3.1 shapes, `scale_cache` fp32 with
  `numel >= 2*P*H*S`.
* `block_size % 16 == 0`.
* int32 for `slot_mapping`, `block_tables`, `seq_lens`, `cu_seqlens`.
* `check_burst_aligned` on every GM operand base (I6; a violation surfaces as CANN error
  264, not an alignment report).
* `tables` bf16, `numel == 16*32*32`.
* **New:** `query.numel() >= (last_used_row + 1 + 15) * head_size` (§3.5).
* **New, O3/O4:** `temporal_splits >= 1` and
  `num_reqs * num_kv_heads * temporal_splits <= npu_turboquant_vector_core_num()` — I8 is a
  correctness gate, and the adapter is where it belongs because the kernel cannot see the
  grid it was launched with.
* `VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS` keeps its current meaning for O1 and gains an O2
  form (slots `>= 0`, distinct, within `P*S`). Off by default; the cost is a D2H copy and a
  synchronisation per layer per step.

### 4.2 C++ launcher signatures

Mirroring the existing `*_impl(void* stream, uint32_t block_dim, ...)` convention. Scalars
are unqualified `uint32_t` because CANN copies the parameter list verbatim into
`aclrtlaunch_triple_chevrons_func.h`, which has no `using namespace std`.

```cpp
extern "C" void turboquant_reshape_and_cache_bf16_impl(
    void* stream, uint32_t block_dim,
    const uint16_t* key, const uint16_t* value,
    int8_t* key_cache, int8_t* value_cache, float* scale_cache,
    const int32_t* slot_mapping, const float* pi_signs, const int32_t* tables,
    const int32_t* cu_seqlens,            // nullptr => one request
    uint32_t num_requests, uint32_t num_tokens, uint32_t num_kv_heads,
    uint32_t head_size, uint32_t block_size, uint32_t num_blocks,
    uint32_t total_strips, float inv_sqrt_len);

extern "C" void turboquant_decode_cache_update_impl(
    void* stream, uint32_t block_dim,
    const uint16_t* key, const uint16_t* value,
    int8_t* key_cache, int8_t* value_cache, float* scale_cache,
    const int32_t* slot_mapping, const int32_t* tables,
    uint32_t num_tokens, uint32_t num_kv_heads, uint32_t head_size,
    uint32_t block_size, uint32_t num_blocks, float inv_sqrt_len);

extern "C" void turboquant_gqa_paged_decode_impl(
    void* stream, uint32_t block_dim,
    const uint16_t* query, const uint16_t* gate,       // gate may be nullptr
    const float* pi_signs, const int32_t* tables,
    const int8_t* key_cache, const int8_t* value_cache, const float* scale_cache,
    const int32_t* block_tables, const int32_t* seq_lens,
    float* workspace, uint16_t* query_rot, uint16_t* out,
    uint32_t num_requests, uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t head_size, uint32_t block_size, uint32_t num_blocks,
    uint32_t max_blocks_per_seq, uint32_t query_tokens_per_req,
    uint32_t temporal_splits, uint32_t output_stage,
    float scale_value, float inv_sqrt_len);

extern "C" void turboquant_decode_combine_impl(
    void* stream, uint32_t block_dim,
    const float* workspace, const uint16_t* gate,
    const float* pi_signs, const int32_t* tables, const int32_t* seq_lens,
    uint16_t* out,
    uint32_t num_requests, uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t head_size, uint32_t query_tokens_per_req,
    uint32_t temporal_splits, uint32_t output_stage, float inv_sqrt_len);
```

Note `total_strips` on O1: the adapter computes it from `cu_seqlens` (sum of
`ceil(L_b / 16)`, which is **not** `ceil(sum L_b / 16)`) so the device never has to.

---

## 5. O1 — modifications to `turboquant_reshape_and_cache_bf16.cpp`

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

## 6. O2 — `npu_turboquant_decode_cache_update` (new)

The per-step write. `num_tokens` arbitrary (`= B*(1+k)`), slots arbitrary and
non-consecutive.

**Shape:** grid `min(num_tokens * H, core_count)`, grid-stride over
`(token, kv_head)` pairs. One pair is one 256-channel K vector and one 256-channel V vector.

**Vector-only, no Cube.** This is the one place the audit's §3.2 verdict inverts: at `M = 1`
the Cube prologue stages 32 KiB of rotation tables to transform 512 B of payload, and
`GEMV.md` records that an `M = 1` Mmad leaves only 32-64 B valid in each 1024-byte L0C
sub-vector — a 16:1 waste. So `Pi = D H D` runs as the Vector butterfly: one sign multiply,
8 radix-2 stages over 256 lanes, one scale. The harness already has the leaf
(`routine_vector_egress_rotate.hpp` does the same transform on the egress side).

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

Making the decode write go through O1 by padding `B` to a multiple of 16 and faking
consecutive slots. Rejected: there is no slot run to fake (the `B` tokens are in `B`
different pages), and a reserved bit-bucket page cannot produce a *consecutive* 16-slot run
mixed with real slots. Audit §5.3.

---

## 7. O3 — `npu_turboquant_gqa_paged_decode` (new, promoting the harness probe)

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
page, and `out` is the final answer. **That deletes O4 from the critical path and the
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
**the sigmoid gate does not**, so with `temporal_splits > 1` the gate must be applied by O4,
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

### 7.7 The prologue round trip

Left as it is, deliberately. Removing it means making `CopyL0C2L1` work with a bf16 Nz
destination, and the one recorded attempt produced all-256-zero score panels in a run where
the same L0C reached GM bit-exactly. It is one GM round trip, in the prologue, once per
launch; the page loop touches GM only to read packed pages and scales. Re-opening it is a
separate investigation with an isolated mechanism, listed in §11.

---

## 8. O4 — `npu_turboquant_decode_combine` (new)

Only on the `temporal_splits > 1` path. Per `(b, h, row)`, over the `splits` partials:

```
m   = max_j m_j
l   = sum_j l_j * exp(m_j - m)
Acc = sum_j Acc_j * exp(m_j - m)          # Acc_j unnormalised, in the rotated basis
y   = (Acc / l) . Pi
out = y * sigmoid(gate)                    # output_stage 2
```

Grid `min(num_requests * num_kv_heads, core_count)`, grid-stride, Vector-only. Workspace
layout, which O3 writes and O4 reads:

```
workspace : fp32, [splits, num_requests, num_kv_heads, rows, head_size + 16]
            [..., 0:head_size]            = Acc
            [..., head_size : head_size+8]   = m, replicated over the 8 lanes of a DataBlock
            [..., head_size+8 : head_size+16] = l, likewise
floats    = splits * num_requests * num_kv_heads * rows * (head_size + 16)
```

The replicated-DataBlock representation of `m` and `l` is the probe's existing
`kDecodeStateLanes = 8` invariant and is what lets the recurrence run without a single
scalar read of UB; keeping it in the workspace means O4 reads the same shape O3 wrote, with
no repacking.

O5 returns exactly that `floats` count, and `0` when `splits == 1`.

---

## 9. Python integration

### 9.1 `vllm_ascend/attention/turboquant_v1.py`

Port from `feat/tq-main-clean` and change:

```python
class AscendTurboQuantAttentionBackend(AscendAttentionBackend):
    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str="") -> tuple[int, ...]:
        # One flat byte budget. The two packed planes have different ranks
        # (RFC §3.1), so a single 5-tuple cannot describe both honestly; the
        # per-plane views are built in _reshape_kv_cache_tensors instead.
        ...
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]          # any multiple of 16 is expressible; 128 is what is verified
```

`AscendTurboQuantAttentionSpec` (`core/kv_cache_interface.py`) keeps `payload_bytes`,
`scale_plane_bytes` and `real_page_size_bytes`; the only change is that the plane *views*
must follow §3.1's head-major order, which is the F4/I13 fix and must be asserted against
the C++ layout by a unit test rather than by comment.

```python
class AscendTurboQuantAttentionBackendImpl(AscendAttentionBackendImpl):

    def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
        # The dispatch F3 says is missing today.
        if attn_metadata.attn_state in (AscendAttentionState.DecodeOnly,
                                        AscendAttentionState.SpecDecoding):
            torch.ops._C_ascend.npu_turboquant_decode_cache_update(...)   # O2
        else:
            torch.ops._C_ascend.npu_turboquant_reshape_and_cache(         # O1
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
        torch.ops._C_ascend.npu_turboquant_gqa_paged_decode(...)          # O3
        if splits > 1:
            torch.ops._C_ascend.npu_turboquant_decode_combine(...)        # O4
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

### 9.2 `vllm_ascend/platform.py::_validate_turboquant_config`

* an explicit statement for `model_config.is_hybrid` — either support under the chosen
  Phase-0 allocator option, or a clean rejection naming the three-plane carve. Today the
  engine passes validation and fails at cache reshape with `len(raw_planes) == 3`.
* keep the existing rejections of prefix caching and chunked prefill **until** O1's §5.2/§5.3
  changes land, then lift them: a ragged-tail, `cu_seqlens`-aware writer is exactly what
  chunked prefill needs. Note in the message that the dependency is O1, so the rejection has
  a visible exit.
* spec decoding moves from "rejected" to "supported with `query_tokens_per_req = 1 + k`",
  which is the Qwen3.5 MTP case and the regime O3 is best shaped for (audit §4.2).
* keep the PIECEWISE/NONE graph downgrade.

### 9.3 `vllm_ascend/patch/worker/patch_qwen3_5.py`

The current file has no TurboQuant hook: the gate is `torch.sigmoid(gate)` followed by a
multiply (`:92-95`). Add the `feat/tq-main-clean` branch, guarded:

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
gate to the backend, and the **backend** decides per step whether it goes to O3 (stage 2) or
to O4. That keeps the gate out of torch on both paths and removes the per-step branch from
the model patch. Per the repo's patching rule, all of this stays in `patch/worker/`, not in
a model file.

Both entry ops stay graph-splitting custom ops with `mutates_args=["output"]` and
`dispatch_key="PrivateUse1"`.

### 9.4 `vllm_ascend/worker/model_runner_v1.py`

* make `hybrid_with_attn_and_mamba` **per-tensor**. It is currently a global flag set while
  iterating tensors, so once any tensor mixes mamba and attention, later pure-attention
  tensors also take the flat single-tensor path — ordering-dependently.
* the three-plane carve must become reachable on the hybrid path (Phase 0; out of scope
  here, N1) and the plane views must follow §3.1.
* nothing else: `slot_mapping`, `block_tables`, `seq_lens_gpu` and the padding convention
  are all already what the operators want.

### 9.5 Everything else

| File | Change |
| --- | --- |
| `csrc/torch_binding_meta.cpp`, `vllm_ascend/meta_registration.py` | no-op meta kernels for O1-O5 |
| `CMakeLists.txt` | add the O2/O3/O4 translation units to `ascendc_library(vllm_ascend_turboquant ...)`; keep the `ascend950` gate; drop `turboquant_paged_attention_opt.cpp` once O3 lands, since its only purpose was to keep a schedule compiling |
| `vllm_ascend/envs.py` | `VLLM_ASCEND_TURBOQUANT_TEMPORAL_SPLITS` (0 = auto) for bring-up; document in the `env_variables` dict |
| `vllm_ascend/attention/turboquant_layout.py` | `turboquant_dequantize` must read §3.1's head-major order; today it would decode transposed (F4) |

---

## 10. Phasing and acceptance

There is **no NPU on this host**, so every gate below is CAModel plus host-oracle, and no
phase may claim a latency, throughput or power result. The verification method is the one
this harness already uses: an inductively collapsed grid (`kSlots + 2` tiles rather than the
full grid), guard-banded GM buffers, canary sweeps over everything a tile must not touch,
and a trace audit of the dumps.

| Phase | Lands | Gate |
| --- | --- | --- |
| **P0** | re-vendor the vllm writer copy into the harness; raise `block_size` to 128 in the probe | existing bit-exact oracle passes at `block_size = 128`, 8 strips per page, head-major planes; canary sweep clean. This closes the audit §1 staleness and is a prerequisite for everything |
| **P1** | §3.1/§3.2 honest shapes; `absmax == 0` branch; `kNumPages` -> runtime `num_blocks` (I14, I15) | a layout unit test that asserts the Python view order equals the C++ index function, token by token, for `(P, S, H, d) = (9, 128, 4, 256)`; a zero-row fixture |
| **P2** | O1 §5.1 grid-stride + table hoist | bit-exact at `total_strips > core_count` (the shape F5 spins on today); trace audit check 1b extended to iteration boundaries; measured GM->L1 traffic down ~33% per strip |
| **P3** | O1 §5.2 ragged tail + §5.3 `cu_seqlens` | new golden coverage for `n in [1, 15]` on the tail strip, including the value plane's UB zero-pad and the canary that the pad bytes land only in the request's own unwritten page remainder; a ragged-batch fixture with `L_b % 16 != 0` for every `b` |
| **P4** | O2 decode writer | single-token and `(request, 1+k)`-token RMW both bit-exact against the oracle; canary proves no strip outside the target is touched; `VALIDATE_SLOTS` form for O2 |
| **P5** | O3 §7.1 runtime rows, §7.6 probe outputs optional | the 12-rung canary protocol re-run at `rows in {6, 12, 24, 48}`; score panels bit-exact; `(m, l)` within 2 ULP, `Acc` within its per-fixture budget; **flag-id audit re-run at every rows bucket** (§7.3's risk) |
| **P6** | O3 §7.2 page loop with `temporal_splits == 1` | a 2-page and a 4-page context, `splits = 1`, against a host oracle that does the whole context in one pass — this is the first run that produces a *correct* multi-page answer, which the probe never did |
| **P7** | O3 §7.3 L1 ring + O4 combine | `splits > 1` matches `splits = 1` bit-for-bit on the same fixture, which is the strongest available check on the combine; trace audit check 3 on a rung with `strips >= 2 * ring_depth` |
| **P8** | §9 Python integration | existing UT patterns (`test_turboquant_page_budget`, layout/codec UTs) plus a new UT for the §9.1 dispatch and the §9.2 validation messages. End-to-end and accuracy gates belong to whoever has hardware |

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
| R4 | the AI core count of the target part | §7.2's `splits` clamp and I8 both need it; a literal would be a latent hang | runtime (`GetBlockNum()` / O6), never a constant |
| R5 | concurrent execution of a full grid | no decode run has ever had two blocks computing at once — every rung isolates one of 32. The isolation argument (mode-4 is per core group, each block has its own UB/L1/L0, GM writes are disjoint and canary-verified) is sound but it is an argument | CAModel, if a full-grid run is affordable; one strip of one block is ~7 min of model time on this host |
| R6 | the hybrid allocator (N1) | upstream of all of this; a TurboQuant Qwen3.5 run dies at `len(raw_planes) == 3` before any kernel executes | vLLM-Ascend, Phase 0 of the integration map |
| R7 | KV-transfer connectors vs §5.2's pad bytes | a connector that copies a page mid-prefill sees zeros in the request's unwritten remainder. Harmless if `seq_len` gates the read, which it should — but the hybrid connectors assume the documented layout and have not been checked against this | vLLM-Ascend connector review |
| R8 | whether `rows = 6` is worth a dedicated kernel after all | §7.1 makes O3 cover it, and the UB budget says it fits comfortably. If profiling on silicon shows the strip pipeline's fixed cost dominates at `rows = 6`, a Vector-only variant on O2's pattern becomes attractive | hardware |
