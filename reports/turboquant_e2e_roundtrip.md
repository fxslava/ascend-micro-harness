# TurboQuant KV cache end-to-end round trip: NOT DEMONSTRATED, and not expressible

Date: 2026-11-07. Branch `claude/tq-e2e-roundtrip-05665a`, from
`claude/micro-camodel-validation` at `0f8120d`.
Target: `Ascend950PR_9599` / dav-3510 CAModel, CANN 9.2.0-beta.2 in
`ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12` under WSL `Ubuntu-22.04`.

## Headline

**The requested write -> read round trip was not run, because it does not
exist to be run.** The ingress writers and the fused decode use two different
4-bit cache layouts, and they do not merely disagree about addresses: they
disagree about *which two logical elements share a byte*. On a `block_size 64`,
`head_size 256` page the two co-byte matchings have **0 of 8192 pairs in
common**, on both the key and the value plane. An injective map of bytes must
carry a co-byte pair onto a co-byte pair, so **no relabelling of addresses can
bridge the layouts**; a bridge has to re-pack nibbles. There is therefore no
geometry at which an ingress store is readable by the fused decode, and nothing
honest to run end to end.

So this report takes **branch (b)** of the task: report the divergence with its
index arithmetic, run the two halves separately as the strongest available
evidence, and state what would be needed for a real round trip. No layout bridge
was invented and no round trip was faked.

Both halves pass in isolation, at one tile:

| Half | Gate | Result |
| --- | --- | --- |
| pre-flight | `test_reshape_scatter_probe` (Tier-1 ingress contract) | PASS, 0 mismatches |
| the question | `test_turboquant_kv_layout_bridge` (NEW) | PASS — the layouts **disagree**, structurally |
| write | scatter ingress CAModel, `--mode canary --head 0`, 1 tile | PASS, bit-exact, 7095 ticks |
| write | scatter trace audit, asserted at rows 1 / runs 1 / launches 1 | PASS |
| read | fused decode CAModel, `--canary 0`, 1 tile | PASS, `exact_0ulp_at_stored_word`, 25389 ticks |
| read | fused decode CAModel, `--tiles 1 --tail-rows 64` | PASS, `exact_0ulp_at_stored_word`, 26898 ticks |
| read | fused decode trace audits, both cases | `"status": "PASS"` |
| — | **the write -> read round trip itself** | **NOT RUN — not expressible** |

Exception dumps: **0 of 32 non-empty** in each of the three CAModel runs.
`invalid ldst addr`, `unrecognize ldst addr`, `su_ccu_mpu_err`: **0 matches** in
each. Details in [Simulation evidence](#simulation-evidence).

**This is not a `REFINEMENT_OF_COMBAT_KERNELS.md` baseline and supports no
hardware claim.** No ladder was run, by the task's explicit prohibition. See
[Gates not run](#gates-not-run).

## 1. The two layouts, re-derived from source

Re-derived here from the kernels rather than taken from the prior report
(`reports/ingress_v_cache_transposition.md`, which reaches the same place for
the strip layout). `R = 16` is the strip height, `d = 256`, `S` the block size.

### (A) STRIP — what the ingress writers write

Normative contract, verbatim at
[`turboquant_reshape_scatter.cpp:55`](../kernels/ops/turboquant/vllm_integration_layer/op_kernel/turboquant_reshape_scatter.cpp):

```
key   nibble = ((block*H + head)*S + token)*d + channel
value nibble = ((block*H + head)*S)*d + (token/R)*(R*d) + channel*R + (token % R)
scale index  = ((kv*P + block)*H + head)*S + token        kv 0 key, 1 value
nibbles       two per byte, low nibble first, signed INT4 in [-8, 7]
```

Host oracle: `key_nibble` / `value_nibble` / `scale_index` in
[`tests/common/turboquant_scatter_oracle.hpp`](../tests/common/turboquant_scatter_oracle.hpp).
Written by `turboquant_reshape_scatter.cpp` (`WriteValuePlane`,
`scatter_merge_token_vf`), `turboquant_reshape_batch.cpp` (`StoreChunk`) and
`turboquant_reshape_and_cache_bf16.cpp`.

### (B) NZ-TILED — what the kv4fp8 fused decode reads

Address function
[`NzTiledPackedByte`](../kernels/ops/turboquant/vllm_integration_layer/op_kernel/common/turboquant_layout.h),
with `kCubeTileRows = 64`, `kOperandC0 = 32`, `packedBytes = d/2`:

```
tileRow = slot % 64
byte    = (slot - tileRow)*numKvHeads*packedBytes
        + kvHead*64*packedBytes
        + ((column/32)*64 + tileRow)*32 + column % 32
```

with `slot = physical*S + token`. The kernel issues exactly this: `ReadPlane`
in
[`vector/turboquant_vector_service.h:919`](../kernels/ops/turboquant/vllm_integration_layer/op_kernel/vector/turboquant_vector_service.h)
reads one `tilePlaneBytes_` burst from
`(row*numKvHeads_ + kvHead*kCubeTileRows)*packedBytes_`, which is
`NzTiledPackedByte` at `tileRow = 0`. K and V are symmetric here — `ReadPlane`
branches only on which GM tensor — and V's `K x N` operand role is discharged by
MTE1's intra-fractal transpose at the L0B load (`ifTranspose = true` in
`StageRightToL0bFromKn`), not by any difference in the GM layout.

**The nibble pairing is the part that matters, and it is not the contract's.**
`UnpackAffine`
([`common/turboquant_codec_mx.h:289`](../kernels/ops/turboquant/vllm_integration_layer/op_kernel/common/turboquant_codec_mx.h))
`DeInterleave`s a chunk of `bytes` packed bytes into a low plane and a high
plane of `bytes` elements each; `StageNzTileToL1`
([`vector/turboquant_vector_service.h:1140`](../kernels/ops/turboquant/vllm_integration_layer/op_kernel/vector/turboquant_vector_service.h))
stages the low plane at L1 offset 0 and the high plane at `+ tilePlaneBytes_`.
The L1 operand is the `64 x d` NZ image, whose groups of `kOperandC0` cover 32
channels each, so the low plane lands on channels `[0, d/2)` and the high plane
on `[d/2, d)`. Hence:

```
byte holding packed column `col` carries channel `col`       in its LOW  nibble
                                      and channel `col + d/2` in its HIGH nibble
                                      -- of ONE token
scale index = (physical*S + token)*ScaleSlotFloats(H) + kv*H + head
```

The kernel's own comment states it: *"low nibbles land in place on NZ groups
`[0, groups)` and its high nibbles on `[groups, 2*groups)`"*. The fused decode's
CAModel fixture
([`tests/common/fused_decode_opt_fixture.hpp:291`](../tests/common/fused_decode_opt_fixture.hpp))
writes exactly that pairing on the host —
`value[off] = value_level(col) | (value_level(col + 128) << 4)` — which is an
independent confirmation, and also the admission that **no ingress operator in
this repo produces layout (B)**: the fixture builds its cache itself.

## 2. Where they diverge, with the arithmetic

New probe: [`tests/cpu/test_turboquant_kv_layout_bridge.cpp`](../tests/cpu/test_turboquant_kv_layout_bridge.cpp),
modelled on [`scripts/kv_plane_layout_check.py`](../scripts/kv_plane_layout_check.py)
(which does the same comparison for the strip writer/reader pair, and finds 0
disagreements there — because both of those are layout A). It maps every
`(block, kv_head, token, channel)` through both production address functions:
the oracle's `key_nibble` / `value_nibble`, and the kernels'
`NzTiledPackedByte`. Registered in `cmake/SimTargets.cmake` as
`test_turboquant_kv_layout_bridge`.

```
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    tests/cpu/test_turboquant_kv_layout_bridge.cpp -o bin/test_turboquant_kv_layout_bridge
./bin/test_turboquant_kv_layout_bridge
```

### 2.1 They agree on the page base and on nothing inside it

At `S = 64` both layouts place page `(block, head)` at
`(block*H + head) * S * d/2` = `(block*H + head) * 8192` bytes — one 64-row
Cube tile per page, so the NZ tile stride and the strip page stride coincide.
Every divergence is *inside* the page. In-page, with `c` the channel and `t` the
token:

| | ingress (STRIP) | fused decode (NZ-TILED) |
| --- | --- | --- |
| key byte | `t*128 + c/2` | `((c mod 128)/32)*2048 + 32*t + (c mod 128) mod 32` |
| key nibble | `c % 2` | `c / 128` |
| value byte | `((t/16)*4096 + 16*c + t%16) / 2` | same as its key byte |
| value nibble | `((t/16)*4096 + 16*c + t%16) % 2` | `c / 128` |

First divergence, both planes, 1 page / 1 kv head / `block_size 64`:

```
   key   plane: 16384 elements; same byte+nibble 2, same byte other nibble 0, different byte 16382
     first divergence (block 0, head 0, token 0, channel 1): ingress byte 0 nibble 1, decode byte 1 nibble 0
   value plane: 16384 elements; same byte+nibble 8, same byte other nibble 0, different byte 16376
     first divergence (block 0, head 0, token 0, channel 1): ingress byte 8 nibble 0, decode byte 1 nibble 0
```

Two of 16384 agreeing on the key plane and eight on the value plane are
coincidences at the origin of the page, and they scale per `(page, head)`: 4 and
16 at two pages, 16 and 64 at two pages and four kv heads. **Nothing agrees away
from the page origin.**

Both maps are checked to be bijections onto the same byte range first, so this
is not a capacity or an out-of-range finding:

```
   plane 0: strip out-of-cache 0, collisions 0, unreached nibbles 0
   plane 0: NZ    out-of-cache 0, collisions 0, unreached nibbles 0
   plane 1: strip out-of-cache 0, collisions 0, unreached nibbles 0
   plane 1: NZ    out-of-cache 0, collisions 0, unreached nibbles 0
```

### 2.2 The obstruction is structural, not a permutation

A byte holds two logical elements. Since both maps are bijections, "shares a
byte with" is a perfect matching on the `(token, channel)` grid of a page. Any
injective map of bytes carries each co-byte pair onto a co-byte pair, so if the
two matchings share **no** pair, no relabelling of addresses can bridge the
layouts. They share none:

```
3. co-byte matchings -- can any byte permutation bridge them?
   plane 0 (key), page (block 0, head 0):
     ingress      pairs  8192  (same token, two channels  8192 | same channel, two tokens     0)  e.g. (t 0, c 0)+(t 0, c 1)
     fused decode pairs  8192  (same token, two channels  8192 | same channel, two tokens     0)  e.g. (t 0, c 0)+(t 0, c 128)
     co-byte pairs the two layouts have in common: 0 of 8192
   plane 1 (value), page (block 0, head 0):
     ingress      pairs  8192  (same token, two channels     0 | same channel, two tokens  8192)  e.g. (t 0, c 0)+(t 1, c 0)
     fused decode pairs  8192  (same token, two channels  8192 | same channel, two tokens     0)  e.g. (t 0, c 0)+(t 0, c 128)
     co-byte pairs the two layouts have in common: 0 of 8192
```

Read off the three pairings:

```
ingress key   pairs (t, 2k)  with (t, 2k+1)      -- channel axis, distance 1
ingress value pairs (2j, c)  with (2j+1, c)      -- TOKEN axis,   distance 1
fused decode  pairs (t, c)   with (t, c + d/2)   -- channel axis, distance 128
```

The key plane's pairing runs along the same axis as the decode's but at the
wrong distance: 1 against 128. **The value plane's does not even run along the
same axis** — the strip value plane is channel-major inside a 16-token strip, so
one of its bytes holds *one channel of two tokens*, where the decode needs *two
channels of one token*. That is the sharpest statement of the gap, and it is why
a bridge is a nibble re-pack and not an address change.

### 2.3 The scale plane diverges separately

```
4. the scale plane
   scale plane: 8/1024 indices agree; slot lanes 8
     ingress floats 1024 (planar [kv][P][H][S]); fused decode floats 1024 ([P*S][slot])
     first divergence kv 0 block 0 head 0 token 1: ingress 1, decode 8
```

The ingress scale plane is planar — `((kv*P + block)*H + head)*S + token`, with
the two kv planes end to end. The fused decode's is a per-row slot of
`ScaleSlotFloats(H) = ceil(2H/8)*8 = 8` lanes:
`(physical*S + token)*8 + kv*H + head`, read as one
`kCubeTileRows * scaleSlot_` burst beside the tile. At `H = 4` the two have the
same float count and still only 8 of 1024 indices in common; at `H = 1` the
decode's plane is 4x larger (512 against 128 floats) because six of its eight
lanes are unused padding. A scale bridge is a gather, independent of the packed
planes.

## 3. What a real round trip would cost

Three options, named with their price rather than ranked:

1. **Teach an ingress writer to emit layout (B).** The key plane is the cheap
   half: it is already token-major, so it needs only the channel re-pairing
   `(c, c+128)` instead of `(2k, 2k+1)` plus the NZ column-group address, both
   expressible in the existing register splice. The value plane is the expensive
   half: the writer's 2048-byte read-modify-write strip exists *because* the
   value plane is channel-major at stride 16, and layout (B) wants it
   token-major with paired channels — which is a different kernel, not a
   different offset. It also forfeits the invariant that
   `scripts/kv_plane_layout_check.py` check 3 pins (`Nz(K[16,256]) ==
   Zn(K^T[256,16])`, 0 disagreements against 3840 for the row-major control),
   which is what lets one L1 image serve both Cube operand roles through
   byte-identity L1->L0 loads. Plus a scale-plane rewrite from planar to slotted.
   Every ingress gate re-baselines.
2. **Teach the fused decode to read layout (A).** It reads one contiguous
   `tilePlaneBytes_` burst per (tile, kv head) precisely because an NZ-tiled
   plane *is already* the NZ image of its bytes and the expand is byte-wise.
   Layout (A)'s value plane would need either a transposing ingest in UB or a
   banded load, and the repo's own README records the constraint: MTE1's copy
   unit is the 512-byte fractal and its only rearrangement is an intra-fractal
   transpose. This is the operator the whole modulo schedule and its UB budget
   were built around; changing its ingest invalidates the capacity-one model,
   the flag budget and the ladder.
3. **Insert an explicit re-pack operator between them.** Honest and cheap to
   write, but it is a third pass over the whole cache per decode step and it
   defeats the point of a 4-bit paged cache. Worth it only as a correctness
   bridge for a test, and if written it should be labelled as such.

None of these is in this task's scope, and none was attempted. Whichever is
chosen, the round trip then needs its own `REFINEMENT_OF_COMBAT_KERNELS.md`
baseline — the capacity-one model, the compile-time UB checks, the isolated
canaries, then depth and wrap — not this report.

## 4. Simulation evidence

All runs on an ext4 work root (`/home/ladislav/.cache/ascend-micro`, bound at
`/work` in container `tq_e2e`), never on `/mnt/d`. Reusable driver:
[`scripts/run_turboquant_e2e_roundtrip.sh`](../scripts/run_turboquant_e2e_roundtrip.sh)
(`host` | `ingress` | `decode` | `census` | `all`). Archived logs under
[`reports/turboquant_e2e_roundtrip/`](turboquant_e2e_roundtrip/).

### 4.1 Pre-flight host oracle — Tier 1, the ingress contract

```
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    tests/cpu/test_reshape_scatter_probe.cpp -o bin/test_reshape_scatter_probe
./bin/test_reshape_scatter_probe
```

```
reshape_scatter Tier-1 probe: d 256, H 4, S 16, P 8, strip 2048 B
  unguarded absmax == 0 scale would be -1.70141e+38
  reconstruction: worst |q.scale - y/16| = 3.875, half a step = 4, 0 channel(s) at the clamp
  fixture: 16 distinct codes, 3 distinct exponents, 6 clamped rows, peak |Pi . x| 1320
reshape_scatter probe: PASS
```

PASS, exit 0. Its assertions are the layout bijections across key, value and
scale, the 2048-byte strip window exclusivity, and a word-granular simulation of
the masked merge against the nibble map — so "0 mismatches across key, value
and scale mappings" is what a PASS here means.

### 4.2 The write half — scatter ingress CAModel, one tile

Exactly one tile: one token, one work item, grid 1, head 0 (the boundary head).
One launch. No ladder rung.

```
export SCATTER_SRC_DIR=/work/e2e-roundtrip/scatter-src
export SCATTER_BUILD_DIR=/work/e2e-roundtrip/scatter-build
bash scripts/run_reshape_scatter_camodel.sh --mode canary --head 0
```

```
turboquant_reshape_scatter_aiv CAModel: mode canary, head 0, d 256, S 16, P 8, strip 2048 B
  head 0 isolated
  fixture: peak |Pi . x| 367, absmax 367, exponent 8, scale 4
  [run] launch: head 0, 1 token(s), page 3, 1 work item, grid 1
  [run] launch probe: GetBlockIdx 0, GetBlockNum 1, GetSubBlockNum 1, itemCount 1
  [run] zero rows 0, clamped rows 0; key bytes wrong 0, value bytes wrong 0, scale words wrong 0
  [run] nibble census: owned wrong 0, preserved-in-strip moved 0, foreign moved 0, owned unwritten 0
turboquant_reshape_scatter_aiv CAModel (canary, head 0): PASS
Model RUN TIME: 4136.93 ms
[INFO] Total tick: 7095
```

- **PASS line and runtime ticks:** as above, **7095 model ticks**.
- **0 ULP / bit-exact against the golden CPU oracle:** `key bytes wrong 0,
  value bytes wrong 0, scale words wrong 0`.
- **Canary integrity:** this operator's guard protocol is the nibble census, not
  a `0x7FC0BEEF` band — `preserved-in-strip moved 0` and `foreign moved 0` say
  every lane the work item does not own read back unchanged through the
  2048-byte read-modify-write, and `owned unwritten 0` says none was missed.
  The `0x7FC0BEEF` guard bands belong to the fused decode's testbed and are
  reported there, in 4.3.
- **Exception dumps: 0 of 32 non-empty.** `find -name 'core*.excp_log.dump'`
  gives 32 files, `-size +0` gives 0; every file is 0 bytes.
- **`invalid ldst addr` 0, `unrecognize ldst addr` 0, `su_ccu_mpu_err` 0** —
  0 matches in 0 files each.

Trace audit over the same dumps. The runner's ad-hoc argument path calls the
audit with rows/runs/launches 0, which makes the instruction census *report*
rather than *assert*; `--mode canary --head 0` is 1 row, 1 run, 1 launch, so the
driver re-audits the captured dumps at those counts. No second simulator run.

```
python3 scripts/analyze_scatter_trace.py \
  dumps_scatter_run/core0.veccore0.instr_popped_log.dump \
  dumps_scatter_run/core0.veccore0.instr_log.dump \
  --aiv1 dumps_scatter_run/core0.veccore1.instr_popped_log.dump \
  --rows 1 --runs 1 --launches 1
```

```
ok   check 1: no cross-core flag instruction was issued (none)
ok   check 2: SET_FLAG == WAIT_FLAG per (pipe, trigger, id) ([])
ok   check 2: no reserved flag id 6 or 7 is used ([])
ok   check 2b: every intra-core SET precedes its WAIT in program order ([])
ok   check 3: the companion Vector core issued no Reg-vector work (0)
ok   check 3: the companion Vector core issued no outflow (0)
ok   check 4: RV_VMULA 48 == 6 stages x 4 regs x 2 planes x 1 rows = 48
ok   check 4: RV_VSELR 60 == 48 butterfly + 12 absmax = 60
ok   check 5: the core both read and wrote Global Memory
ok   check 5: every re-written Global Memory address was read first ([])
ok   check 5: 4 loads == 1 pi_signs + 3 x 1 (key rows, value rows, strip) = 4
ok   check 5: 5 stores == 1 probe + 2 x 1 (key page, strip) + 2 x 1 (one scale DataCopyPad per token per plane) = 5
scatter trace audit: PASS
```

Whole-dump dispatch span 5988 model ticks, 1359 completed-log instructions —
fill plus drain over one launch, not a throughput figure.

### 4.3 The read half — fused decode CAModel, one tile, two cases

Against the fused decode's **own host-built NZ-tiled fixture cache** — which is
the point of this report: that cache is not what any ingress writer produces.
Reused the existing build at `/work/fused-decode-canary/build`
(`reuse`; `FUSED_DECODE_OPT_SKIP_NPU_GATE=1`, no push intended), so no BiSheng
pass was repeated.

```
export FUSED_DECODE_OPT_ROOT=/work/fused-decode-canary
export FUSED_DECODE_OPT_SKIP_NPU_GATE=1
FUSED_DECODE_OPT_TAG=e2e-isolated-page0 \
  bash scripts/run_fused_decode_opt_camodel.sh reuse --canary 0
FUSED_DECODE_OPT_TAG=e2e-single-tile \
  bash scripts/run_fused_decode_opt_camodel.sh reuse --tiles 1 --tail-rows 64
```

Isolated boundary logical page 0 on the non-identity block table (logical page
`p` holds physical `2p+1`, full 18-page allocation kept):

```
PASS fused_decode_opt dtype=fp16 tiles=1 page=0 tile=0 heads=16 splits=1 grid=1 repeats=1
  elements=4096 tail_rows=64 hole=4294967295 comparison=exact_0ulp_at_stored_word
  inputs=unchanged guards=0x7FC0BEEF-intact ASSERT_CANARY_INTACT 703936 bytes
Model RUN TIME: 17038.3 ms
[INFO] Total tick: 25389
PASS trace audit: /work/fused-decode-canary/e2e-isolated-page0/audit.log
```

One full tile of the real 64-token / 256-channel geometry:

```
PASS fused_decode_opt dtype=fp16 tiles=1 page=4294967295 tile=0 heads=16 splits=1 grid=1 repeats=1
  elements=4096 tail_rows=64 hole=4294967295 comparison=exact_0ulp_at_stored_word
  inputs=unchanged guards=0x7FC0BEEF-intact ASSERT_CANARY_INTACT 703936 bytes
Model RUN TIME: 19448.8 ms
[INFO] Total tick: 26898
PASS trace audit: /work/fused-decode-canary/e2e-single-tile/audit.log
```

- **PASS line and runtime ticks:** **25389** and **26898** model ticks.
- **0 ULP / bit-exact:** `comparison=exact_0ulp_at_stored_word` against the
  golden FP64 oracle, both cases.
- **`ASSERT_CANARY_INTACT` with the `0x7FC0BEEF` guard bands:**
  `guards=0x7FC0BEEF-intact ASSERT_CANARY_INTACT 703936 bytes`, both cases.
  `inputs=unchanged` — nothing wrote back over its own operands.
- **Exception dumps: 0 of 32 non-empty**, both cases. Every
  `core*.excp_log.dump` is 0 bytes.
- **`invalid ldst addr` 0, `unrecognize ldst addr` 0, `su_ccu_mpu_err` 0** —
  0 matches in 0 files, both cases.
- Both trace audits report `"status": "PASS"` with HardEvents
  `"capacity-one, balanced dispatch/completion, empty drain"`.

### 4.4 Exception-dump census, explicit

Counts rather than an assertion, per the acceptance criteria:

```
/work/e2e-roundtrip/scatter-build/dumps_scatter_run   : 0 non-empty of 32 ; distinct sizes: [0]
/work/fused-decode-canary/e2e-isolated-page0          : 0 non-empty of 32 ; distinct sizes: [0]
/work/fused-decode-canary/e2e-single-tile             : 0 non-empty of 32 ; distinct sizes: [0]
```

96 exception dumps across 3 runs, every one 0 bytes.

### 4.5 Environment note, recorded and not chased

The simulator answers `halGetSocVersion soc_version:Ascend950PR_9589` while the
build targets `Ascend950PR_9599`, and it loads
`.../x86_64-linux/simulator/dav_3510/lib/config.json`. Recorded because it is
visible in every run log; not investigated, per the prior report's finding that
it is pre-existing.

The container image has no `rsync`, which both runner scripts want
(`apt-get install -y rsync` inside it). The Windows checkout's `.sh` files would
be CRLF, which kills every script in `scripts/` at `set -o pipefail`; this
worktree checked out LF correctly (`.gitattributes` has `*.sh text eol=lf`),
verified with `file scripts/run_fused_decode_opt_camodel.sh` before anything was
run, and the ext4 work copy is CR-stripped on sync regardless.

## Gates not run, and why

Recorded explicitly, per `AGENTS.md`.

- **The write -> read round trip itself: NOT RUN.** Not expressible — section 2.
  This is the headline, not a footnote. No ingress store was read back by the
  fused decode, in this report or anywhere in this repo.
- **No ladder, no depth rung, no wrap, no multi-worker, no multi-tile and no
  multi-wave run**, by the task's explicit prohibition (`N = 32, 64, ...`
  banned). Consequently **this report is not a baseline under
  `REFINEMENT_OF_COMBAT_KERNELS.md` and nothing in it supports tagging for
  physical hardware.** A one-tile canary is not ring-wrap or multi-worker
  evidence, and two canaries on two operators are not integration evidence.
- **The `block_size = 64` ingress path is still unexercised.** The task asked
  for `block_size 64` and the layout probe covers it on the host, but the
  ingress CAModel canary that ran is the scatter operator at its own `S = 16`
  geometry. `block_size 64` on the ingress side means the prefill writer
  `turboquant_reshape_batch.cpp`, whose strip offset `bf8bfd7` fixed and whose
  own commit message says the offset "has not been exercised on hardware". That
  gate is NOT closed here: with no round trip to run it in, running it would
  have been the prefill operator's own canary under another name, and its
  sibling `turboquant_reshape_and_cache_bf16.cpp` has a live key-plane failure
  that a separate session owns.
- **The prefill key-plane failure on chunks 5 and 7 was not touched**, per the
  task's prohibition. Another session owns it.
- **`test_decode_benchmark.cpp` was not touched**, per the task's prohibition.
- **The `RUN_MODE=npu` strict-compile leg was skipped**
  (`FUSED_DECODE_OPT_SKIP_NPU_GATE=1`), because nothing here is being pushed for
  hardware. Run the ladder without it before any push.
- **No physical hardware run.** None available in this environment.
- **`pytest` and `kv_plane_layout_check.py` were not re-run**: both passed on
  the base branch within the day (49 passed; 0 disagreements) and no source they
  cover was changed by this task.

## Files added

| File | What it is |
| --- | --- |
| [`tests/cpu/test_turboquant_kv_layout_bridge.cpp`](../tests/cpu/test_turboquant_kv_layout_bridge.cpp) | The gate that decides whether a round trip is expressible. Calls the production address functions; PASSES on them disagreeing. |
| [`scripts/run_turboquant_e2e_roundtrip.sh`](../scripts/run_turboquant_e2e_roundtrip.sh) | The driver: host gates, both halves, and the acceptance census (PASS line, ticks, exception-dump counts, the three fault patterns). |
| [`reports/turboquant_e2e_roundtrip/`](turboquant_e2e_roundtrip/) | Archived run logs and trace audits for everything quoted above. |
| `cmake/SimTargets.cmake` | One target added for the new probe. |

No kernel source was modified.
