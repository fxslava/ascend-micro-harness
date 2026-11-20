# Micro-Routine Registry

Inventory baseline `cf94dd09c4e7721f006a3eb43633a1dda5d9658c`: Ascend950PR / `dav_3510` / CANN 9.2.1. Wave 2 implementation was subsequently authorized; each CAModel run still requires its own orchestrator approval. This update vendors the frozen Wave 1 sources and goldens; it authorizes no new device run.

The original 25 headers under `C:/ascend-micro-harness/routines/` were inspected in nine slot assignments, scheduled three workers at a time. Every source hash was independently rechecked during consolidation. The JSON companion embeds full API/CANN evidence, exact declarations, resources, synchronization and audit limitations. Unknown measurements are not zero; historical reports are not current hash-bound certification.

## Frozen Wave 1 references

The authoritative `C:/Ascend/ascend-wsl-registry.json` had all 9 slots free and no simulation containers before Phase0. No slot is assigned to re-optimize Agent 05/07 SIMD dequant.

|Reference|Post-warmup ticks /8,192 elements|Ticks/element|Status|
|---|---:|---:|---|
|Agent 06 BF16 SIMT E8 dequant|9736 / 9815|1.193298340|Frozen source/binary hashes rechecked; standalone reference|
|Agent 08 INT8 SIMT E8 dequant|8011 / 7857|0.968505859|Frozen source/binary hashes rechecked; standalone reference|

Exact original paths, hashes, approval records and timing backend are in JSON `frozen_baselines`. These wrappers emit physical BF16/INT8 to UB. Their register helpers can be adapted after sign-off into new Wave 2 VFs; the wrappers cannot be chained without an intermediate UB boundary. No reference source or binary was changed.

## Portable frozen Wave 1 import

The repository now registers **27 headers**, including these two byte-identical native-slot imports:

| Header | Namespace | Selected configuration | Source SHA-256 |
|---|---|---|---|
| [routine_e8_bf16_dequant_simt.hpp](routines/routine_e8_bf16_dequant_simt.hpp) | `ascend_micro::e8` | BF16 / SIMT / Arithmetic / 512 threads / AoS / ResidualBits | `28f13039c8fec29ef91919a69a8314a60c9cb9aec3f5f3fc75b270f8e524cda2` |
| [routine_e8_int8_dequant_simt.hpp](routines/routine_e8_int8_dequant_simt.hpp) | `ascend_micro::e8_int8_simt` | INT8 / SIMT / Arithmetic / 512 threads / AoS / ResidualBits | `e366ddb4b6697ede5a16bcb41c5ab3857f1ee98bcc2ce95f1c5b8efefe9f2d45` |

Both are AIV SIMT leaves for positive multiples of256 elements; the measured workload is8192 elements/32 D256 rows/1024 E8 vectors. Caller-owned input/output views require32B base alignment and the documented256B read pad. Selected scratch/LUT/L1 footprints are zero; historical wrappers reserve32768B static UB. Peak allocated VREG remains unknown; zero emitted spills belongs to the frozen selected configuration. Caller owns MTE2/V/MTE3 staging, flags and final drains. These are E8-only leaves: composing their wrappers with inverse Hadamard creates a UB boundary; register fusion requires a separately reviewed VF.

The canonical `routines/e8_reference.py` is copied unchanged from native Agent09 and matches Agents06/08. The23 payloads in `routines/e8_wave1_goldens/` include both physical BF16/INT8 golden outputs, unrounded FP32 reconstructions, labels, scales and inputs. `routines/generate_e8_wave1_goldens.py` verifies the package or regenerates it with NumPy2.2.6 into an explicit native directory. All23 regenerated payload records match the original names, sizes and SHA-256 hashes. See [usage and exact template arguments](routines/E8_WAVE1_SIMT.md), [provenance](routines/e8_wave1_export.json) and [validation](reports/e8_wave1/export_validation.json).

The original Phase0 and Wave1 measurements remain historical. No frozen kernel source was edited, no new CAModel benchmark was run, and no full Wave2 pipeline performance is implied by this import.

## Proposed nine-slot execution plan

The Cartesian product has eight cells: `[BF16,INT8] x [Quant,Dequant] x [A,B]`. Both Family B dequant cells compare two alternatives, giving ten implementation paths before M128/M256 and serial controls.

|Native slot|Cell|Assignment after Phase0 sign-off|
|---|---|---|
|agent_01|A / BF16 / Quant|SIMT equalization, full normalized Hadamard, shared-row scale/radial policy, exact finite E8 encoder and packing|
|agent_02|A / INT8 / Quant|Same register chain with signed input conversion and emitted-scale golden|
|agent_03|A / BF16 / Dequant|Frozen06 register core, inverse SIMT Hadamard, inverse equalization, final physical output|
|agent_04|A / INT8 / Dequant|Frozen08 register core, inverse SIMT Hadamard, inverse equalization, final RNE/saturation|
|agent_05|B / BF16 / Quant|AIC full Hadamard M128/M256, FP32 dual split, parallel two-AIV exact encoder, depth2 overlap|
|agent_06|B / INT8 / Quant|Same pipeline with reviewed signed operands/conversion and precision golden|
|agent_07|B / BF16 / Dequant|Compare(i) canonical-corrected fused transition/inverse-Hadamard and(ii) SIMT dequant then AIC inverse Hadamard|
|agent_08|B / INT8 / Dequant|Same two alternatives with frozen08 register core and final physical INT8 conversion|
|agent_09|Shared verification|Independent exact oracle, layout/metadata/token model and common-clock trace audit for all eight cells|

Each slot is `/home/dmin/work/agent_XX`; new Wave 2 directories preserve Wave 1 artifacts. Wave 2 starts0/3, with Wave 1 history retained. Both B-dequant alternatives and controls share their agent's cap; it is not reset per variant.

**Batching proposal for sign-off:** M128/M256 counts total16-channel Hadamard panel rows per Cube batch, split equally across AIVs. The8,192-element workload is32 D=256 rows = 512 panels. Full H256 needs both H16 tensor-product factors and the correct signs/permutation/normalization. A monolithic M128/M256 by D=256 control pads32valid rows and must label and charge padding. E8 G-projection batches count8-coordinate vectors and are named separately. Existing H16/matmul wrappers do not already support these extensions.

## Operator and quality convention

The user confirmed Π denotes the complete normalized D256 Hadamard plus signs/permutation. Use column vectors here. E is frozen positive diagonal channel-equalization metadata.

- Forward: `z = Pi E^-1 x`.
- Reconstruction: `xhat = E Pi^T (s q)`.
- RFC G is a row basis: `G_col = G_RFC^T`. Thus `W_fused = Pi^T (I32 kron G_col)`, with E applied at the final epilogue.
- Canonical shaping is `q = G_col a - 4 N(G_col a/4)`. Raw packed coefficients times a fixed matrix omit the nonlinear term; variantB(i) needs corrected coefficients or an explicit transformed correction.

Keep LE uint16 / 8 coordinates,32 labels = 64 B per D=256 row, separateFP32 scale / 256 coordinates and32 B bases. Define the MSE scale iteration, radial radius/objective, zero/subnormal handling and rounding placement before codegen. Freeze metadata across families and train separately from heldout evaluation. Original-domain distortion and exact labels for the chosen normalized/clamped query are separate metrics. No 1D Lloyd-Max precedes E8. No benefit from radial clamp/equalization is assumed from Wave 1.

## Composition contracts

- **A_single_SIMT_VF**: Family A uses one SIMT VF with register-level rotation and E8 helpers. SIMD and SIMT VFs cannot share live VREGs. Chaining existing wrappers requires UB materialization and violates this family contract.
- **A_D256_mapping**: Propose warp32 with eight contiguous coordinates/thread: low three butterfly stages within each register array, upper five via legal warp shuffles. Prove signs/permutation, reductions, complete participation and zero spills; no tiled_partition<256>.
- **frozen_E8_core**: Agent 06/08 source/binaries stay unchanged. Their wrappers write physical dtype output to UB. Adapt Project/ShapeResidualBits register helpers in new Wave 2 code after sign-off, with explicit scale/rounding placement.
- **exact_quant**: Quant is exact finite65536-point encoding against the ACTUAL emitted FP32 row scale with smallest-uint16 ties. DecodeE8 is not an encoder; nearest-infinite-E8 plus wrap is invalid.
- **quality_metadata**: One scale per256 coordinates; freeze equalization metadata and training policy across families. Define radial radius/objective, MSE scale iterations, zero/subnormal and rounding behavior before codegen. Charge searches and metadata. Original-domain quality and clamped-query label objective remain separate.
- **H256_not_H16**: Existing native Hadamard16 takes FP16 inputs and is not full normalized H256. Both tensor-product factors, correct signs/permutation and1/16 normalization are needed. Typed M128/M256 batching and capacity are new gates.
- **W_fused**: Column convention: z=Pi E^-1 x; xhat=E Pi^T(s q); G_col=G_RFC^T. W_fused=Pi^T(I32 kron G_col), followed by E. Nonlinear canonical shaping cannot be dropped; use corrected coefficients or an explicit transformed correction.
- **dual_destination**: Installed9.2.1 dual output is FP32 with ROW_MAJOR,isToUB=true,NoQuant,noReLU. SPLIT_M divides totalM equally. Each physical AIV has full independent UB. Final BF16/INT8 conversion is counted SIMT work.
- **minimal_transport**: Only named required Cube/AIV transport boundaries may use UB/L1. E8 cosets/parity/min-residual/packing intermediates remain in VREG; input/table/metadata reads and final stores are separate. No hidden scratch, GM relay or tile shrinking. Direct UB-to-L1 depends on installed macro/launch guards.
- **depth2_protocol**: Mode2 uses identical IDs on both AIVs, AIV-to-AIC quorum and AIC-to-both broadcast. Independent coefficient-free and landing-free reverse credits protect g from g+2. Prime/drain capacity-one tokens including tails/dummies; use real FIX_S/V_S drains. No mode4 rebasing or invented MTE1_S.
- **versioned_API**: Exact installed9.2.1 guards override stale beta comments. Normal ReduceSum/ReduceMax paths inspected here use VF reduction; this is not proof every overload avoids scalar UB. Mask/read extents and selected ISA still need review.
- **layout_reconciliation**: NZ(Tokens,Group) and ZN(Group,Tokens) transpose byte identity can generalize symbolically for Tokens divisible16 with correct full-NZ pitch/group base. Historical Tokens16 tests do not certify unchanged M128/M256 wrappers, capacity, or row-major input.
- **rounding**: Agree one Wave 2 operator/golden for intermediate versus final BF16/INT8 rounding before codegen. Preserve frozen Wave 1 semantics; standalone dequant timing is not full inverse-rotation timing.
- **evidence**: Historical reported PASS, test/source presence, compiler capability and current hash-bound CAModel PASS are distinct. Unknown physical peak VREG, scalar provenance and protocol evidence remain explicit; static findings are review gates, not newly observed execution failures.

## Mandatory gates

1. User sign-off on this registry and assignment plan before device codegen.
2. Mathematical/source/API review of SIMT D256 mapping, exact finite encoder, H256 batching, fused-matrix shaping/orientation, rounding, capacity and depth2 token schedules.
3. Native WSL builds only, required image plus read-only9.2.1 overlay; audit exact generated ISA, UB/cache launch, tensor scalar accesses and spills.
4. Separate explicit orchestrator `APPROVED FOR CAMODEL` bound to source/binary/fixture/metadata/config hashes, workload and iteration. No such approval is issued here.
5. Max three optimized revisions per Wave 2 agent; preserve all failed/partial runs. Measure setup, metadata/table/basis acquisition, output and final drains. Use matched controls, exact canaries and common-clock issue/retire evidence; no cache or bank causality inferred from gaps.

Static review gates include duplicated GQA types; accumulation/barrier placement; pointer/descriptor and pitch/extent mismatches; probability plane ownership; normalization and accumulation semantics; and version-sensitive reduction/rounding comments. No source fix or new simulation was performed. Exact findings remain in JSON `audit_detail` and `inventory_batches`.

## Header index

|Header|Declarations|Execution description (abbreviated)|
|---|---:|---|
|`routine_cube_decode_score.hpp`|17|{"actual":"AIC MTE1 staging -> PIPE_M rotation and score -> PIPE_FIX GM/one selected AIV UB. 24 rotation loads,64 m16 n32 k32 rotation Mmads,8 m16 n16 k32 score|
|`routine_cube_gqa_fixpipe_dual_dst.hpp`|7|{"issuer":"AIC","pipe":"PIPE_FIX","instruction":"copy_matrix_cc_to_ub","path":"CO1/L0C NZ FP32 -> both AIV VECCALC UB ND FP32","evidence":[{"path":"/home/dmin/A|
|`routine_cube_gqa_pv_gemm.hpp`|15|{"actual":"AIC MTE1 two probability plane identity loads and V per-fractal transpose; PIPE_M two k32-step chains into same C, first-hi init only; SetMMColumnMaj|
|`routine_cube_gqa_score_gemm.hpp`|10|{"actual":"AIC PIPE_MTE1 CopyL12L0A/B then PIPE_M8xk32 Mmad at D256 Chunk64; SetMMRowMajor. No FIX in leaf.","future":"Fixpipe dual destination is separate sibl|
|`routine_cube_hadamard16.hpp`|6|{"core":"AIC","pipes":["PIPE_M Cube Mmad","Fixpipe L0C->GM"],"SIMD":false,"SIMT":false,"MIX":"not a kernel declaration; surrounding caller may be MIX, leaf itse|
|`routine_cube_matmul.hpp`|8|{"core":"AIC","pipes":["MTE1 L1->L0A/L0B","PIPE_M Cube Mmad","Fixpipe L0C->UB"],"SIMD":false,"SIMT":false,"MIX":"caller can integrate in MIX_AIC_1_2; default su|
|`routine_cube_wta_fixpipe.hpp`|16|{"issuer":"AIC only","pipes":["MTE1: LoadData C1->L0A","M: Mmad half*half->float","FIX: Fixpipe CO1->C1 with ReLU, float->half feedback or scalar int8/uint8 fin|
|`routine_cv_gqa_stage_probs.hpp`|12|{"classification":"SIMD/AIV","operations":"memory-vector Cast/Sub compatibility APIs through VF, then MTE3 UB->L1","SIMT":false,"AIC":false,"MIX":"composition n|
|`routine_cv_transfer_ub_to_l1.hpp`|8|{"issuer":"AIV","pipe":"MTE3","path":"UB VECIN/VECCALC/VECOUT -> shared L1 A1/B1; same T, block-strided placement","instruction":"copy_ubuf_to_cbuf via DataCopy|
|`routine_fractal_load_zn_bf16.hpp`|4|{"core":"AIC MTE1 leaves","pipes":["MTE1 L1->L0A or L0B"],"SIMD":false,"SIMT":false,"MIX":"no kernel task declaration; caller may use AIC branch in MIX"}|
|`routine_probe_l2_atomic_sync.hpp`|11|{"actual":"AIV hybrid SIMD vector combine/MTE2/MTE3 plus SIMT lane0 GM control","SIMD":true,"SIMT":true,"AIC":false,"MIX":false,"note":"SIMD/SIMT hybrid is not |
|`routine_stage_gqa_load_kv_chunk.hpp`|9|{"classification":"AIC/MTE2","SIMD":false,"SIMT":false,"MIX":"leaf no cross-core sync; historical wrapper is1:2 MIX for fill/readback"}|
|`routine_stage_page_zn_bf16.hpp`|6|{"classification":"AIC/MTE2 TensorAPI GM2L1","SIMD":false,"SIMT":false,"MIX":"caller composition only"}|
|`routine_turboquant_dequant4.hpp`|8|{"actual":"AIV SIMD compatibility API","SIMD":true,"SIMT":false,"AIC":false,"MIX":false}|
|`routine_vector_cube_softmax.hpp`|7|AIV Vector / compatibility APIs lowering to VF on dav_3510; caller provides UB; no Cube arithmetic in these headers|
|`routine_vector_decode_gemv.hpp`|6|{"declared_target":"Ascend950PR arch35 dav-3510 CANN9.2 comments","actual_pipeline":"AIV PIPE_V memory-vector APIs; runtime scalar loop/address arithmetic; inst|
|`routine_vector_decode_recurrence.hpp`|23|{"declared_target":"Ascend950PR arch35 dav-3510 CANN9.2 comments","actual_pipeline":"AIV PIPE_V memory-vector APIs; runtime scalar loop/address arithmetic; inst|
|`routine_vector_egress_rotate.hpp`|11|{"leaf":"AIV memory-vector SIMD PIPE_V","SIMT":false,"AIC":false,"MIX":"surrounding paged-decode operator is MIX_AIC_1_2; leaf itself no Cube"}|
|`routine_vector_gqa_accumulate.hpp`|10|{"leaf":"AIV SIMD VF decay + memory-vector Add method","SIMT":false,"AIC":false,"MIX":"caller may consume dual-destination Fixpipe context; no L0C instruction i|
|`routine_vector_gqa_normalize_store.hpp`|8|{"leaf":"AIV SIMD VF + MTE3 DataCopy","SIMT":false,"AIC":false,"MIX":"caller cross-core readiness ownership only"}|
|`routine_vector_gqa_online_softmax.hpp`|9|AIV Vector / compatibility APIs lowering to VF on dav_3510; caller provides UB; no Cube arithmetic in these headers|
|`routine_vector_icdf_quantize.hpp`|1|{"actual":"AIV SIMD VF","SIMD":true,"SIMT":false,"AIC":false,"MIX":false}|
|`routine_vector_normalize_store.hpp`|7|{"actual":"AIV SIMD compatibility arithmetic plus MTE3","SIMD":true,"SIMT":false,"AIC":false,"MIX":false}|
|`routine_vector_online_softmax.hpp`|6|AIV Vector / compatibility APIs lowering to VF on dav_3510; caller provides UB; no Cube arithmetic in these headers|
|`routine_vector_rotate_butterfly.hpp`|25|{"Blocked":"AIV SIMD Level2 UB tensor APIs; all stages write UB.","RegGather":"AIV SIMD __simd_vf__ wrappers + __simd_callee__ VREG row leaf; no SIMT entry.","R|

## Exact per-header contracts

### routine_cube_decode_score.hpp

Path: `C:/ascend-micro-harness/routines/routine_cube_decode_score.hpp`  
SHA256: `684f65611aba917669b07b5ed68d453cfcc32b0b4d92521348c9a9c8e0b805c6`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 85:
```cpp
constexpr std::uint32_t DecodeParityOf(std::uint32_t value)
```

Line 118:
```cpp
template <std::uint32_t Block>
__aicore__ inline void LoadRotationBlock(std::uint32_t l1_table_base,
                                         __cb__ bfloat16_t* l0b)
```

Line 127:
```cpp
template <std::uint32_t... Blocks>
__aicore__ inline void LoadRotationBlocks(std::uint32_t l1_table_base,
                                          __cb__ bfloat16_t* l0b,
                                          std::integer_sequence<std::uint32_t, Blocks...>)
```

Line 136:
```cpp
template <std::uint32_t G>
__aicore__ inline void LoadQueryGroup(std::uint32_t l1_q_base, __ca__ bfloat16_t* l0a)
```

Line 144:
```cpp
template <std::uint32_t... Gs>
__aicore__ inline void LoadQueryGroups(std::uint32_t l1_q_base, __ca__ bfloat16_t* l0a,
                                       std::integer_sequence<std::uint32_t, Gs...>)
```

Line 161:
```cpp
template <std::uint32_t C1, std::uint32_t G>
__aicore__ inline void RotateBlock(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                   __cc__ float* l0c)
```

Line 181:
```cpp
template <std::uint32_t C1, std::uint32_t... Gs>
__aicore__ inline void RotateBlockRow(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                      __cc__ float* l0c,
                                      std::integer_sequence<std::uint32_t, Gs...>)
```

Line 189:
```cpp
template <std::uint32_t... C1s>
__aicore__ inline void RotateBlockRows(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                       __cc__ float* l0c,
                                       std::integer_sequence<std::uint32_t, C1s...>)
```

Line 202:
```cpp
__aicore__ inline void rotate_query_stage_l0(std::uint32_t l1_table_base,
                                             std::uint32_t l1_q_base,
                                             __ca__ bfloat16_t* l0a,
                                             __cb__ bfloat16_t* l0b)
```

Line 216:
```cpp
__aicore__ inline void rotate_query_compute(__ca__ bfloat16_t* l0a,
                                            __cb__ bfloat16_t* l0b,
                                            __cc__ float* l0c)
```

Line 264:
```cpp
__aicore__ inline void rotate_query_to_gm(__gm__ bfloat16_t* gm_qrot,
                                          __cc__ float* l0c)
```

Line 280:
```cpp
__aicore__ inline void load_query_rotated(std::uint32_t l1_qrot_base,
                                          __ca__ bfloat16_t* l0a)
```

Line 321:
```cpp
__aicore__ inline void score_strip_stage_l0(std::uint32_t l1_key_base,
                                            __cb__ bfloat16_t* l0b_slot)
```

Line 329:
```cpp
template <std::uint32_t G>
__aicore__ inline void ScoreBlock(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b_slot,
                                  __cc__ float* l0c_window)
```

Line 350:
```cpp
template <std::uint32_t... Gs>
__aicore__ inline void ScoreBlocks(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b_slot,
                                   __cc__ float* l0c_window,
                                   std::integer_sequence<std::uint32_t, Gs...>)
```

Line 358:
```cpp
__aicore__ inline void score_strip_compute(__ca__ bfloat16_t* l0a,
                                           __cb__ bfloat16_t* l0b_slot,
                                           __cc__ float* l0c_window)
```

Line 374:
```cpp
__aicore__ inline void score_strip_fixpipe(__cc__ float* l0c_window,
                                           __ubuf__ float* ub_scores,
                                           bool to_sub_block_one)
```

Supported dtypes; input/output distinctions retained:
```json
{
  "BF16": "Actual fixed bfloat16_t A/B and BF16 rotated GM output; FP32 L0C. Header lines125-205,250-262,318-345,374-384.",
  "INT8": "No leaf instantiation or pointer; mention of INT8 K-fractal in comment is not support.",
  "FP32": "Actual accumulation and score output only; no FP32-input leaf."
}
```

Execution pipe:
```json
{
  "actual": "AIC MTE1 staging -> PIPE_M rotation and score -> PIPE_FIX GM/one selected AIV UB. 24 rotation loads,64 m16 n32 k32 rotation Mmads,8 m16 n16 k32 score Mmads per strip. SetMMRowMajor. No HardEvent, PipeBarrier or cross-core op in header.",
  "future": "No dual destination, no symmetric1AIC2AIV generation ring, no reverseACK/scalar drain implementation."
}
```

Geometry:
```json
{
  "M": 16,
  "D": 256,
  "group": 32,
  "groups": 8,
  "table_blocks": 16,
  "score_N": 16,
  "dynamic_shape_support": false,
  "constraints": "Correct resident L1 fractal image required; rotation table ND32x32 is not its NZ image. No descriptor runtime checks; pointers/byte bases and alignment caller owned; eight k32 accumulation steps, not one k256."
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "allocation_owner": "Caller; no allocations in leaf",
  "UB_actual_payload_bytes": 1024,
  "UB_note": "One16x16 FP32 ND score destination; full arena/ring unknown. Caller chooses AIV by bool.",
  "L1_symbolic_payload_bytes": {
    "rotation_table": 32768,
    "query": 8192,
    "rotated_query": 8192,
    "key_per_strip": 8192
  },
  "L1_actual_total": "unknown placement/lifetime overlap; bases supplied by caller",
  "L0A_actual_bytes": 8192,
  "L0B_actual_bytes": {
    "rotation_residents": 32768,
    "key_strip": 8192
  },
  "L0C_actual_bytes": {
    "rotation": 16384,
    "score_window": 1024
  },
  "VREG_actual": "No explicit vector register use; compiler register usage unknown",
  "L0_note": "L0A Q groups overwritten by Q_rot only after all rotation reads retire; L0B table no longer needed after prologue. C window/score slots caller owned."
}
```

Synchronization/lifetime contract:
```json
{
  "owned": "None, including no intra-M accumulation barrier.",
  "caller_edges": [
    "ingress->MTE1 (MTE2_MTE1 or equivalent)",
    "MTE1_M before rotation/score",
    "M_FIX after64/8 accumulation calls, unitFlag=0",
    "M_MTE1 before overwriting Q group L0A by Q_rot",
    "FIX completion before GM qrot reload and MTE2_MTE1 before L0A reload",
    "selected AIV consumer-ready + reverse-ACK before same UB slot overwrite",
    "FIX_M or equivalent before L0C window overwrite, M_MTE1 before L0 operands overwrite"
  ],
  "hazard": "m16 n16 score and m16 n32 rotation have fragment counts1 and2, both<10. Header emits no PIPE_M between same-C accumulating calls. Caller cannot place inter-call barriers inside fused functions; unresolved doc/code hazard despite historical report arithmetic success."
}
```

CAModel verification and limitations:
```json
[
  {
    "path": "C:/ascend-micro-harness/reports/paged_decode_attention.md",
    "lines": "551-578,654-697",
    "note": "Historical12-rung report claims zero score/rotation errors and audit9/9; explicitly no32-block concurrent run, reused flags remain timing-dependent; says eight L1 pages replaced ring. This does not certify proposed depth2 reverseACK/mode2 pipeline."
  },
  {
    "path": "C:/ascend-micro-harness/routines/routine_cube_decode_score.hpp",
    "lines": "214-249,281-314",
    "note": "Source comments report failed direct L0C->L1 route (all-zero scores) and single k256 Mmad251/256 wrong; mechanisms not isolated. Actual qrot export usesGM roundtrip."
  }
]
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_cube_decode_score`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_cube_gqa_fixpipe_dual_dst.hpp

Path: `C:/ascend-micro-harness/routines/routine_cube_gqa_fixpipe_dual_dst.hpp`  
SHA256: `418cda3ac19b296f0b0e98d1119db4637902f80c3e95dd82448647f2c356ceeb`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 108:
```cpp
struct GqaFixpipeDualDstGeometry {
    std::uint32_t m = 0;
    std::uint32_t n = 0;
    std::uint32_t dst_off_bytes = 0;
    std::uint32_t src_stride = 0;
    std::uint32_t dst_stride = 0;
    std::uint32_t dual_dst_ctl = 0;
    std::uint32_t sub_block_id = 0;
    std::uint32_t unit_flag = 0;
    bool to_ub = true;
};
```

Line 124:
```cpp
constexpr GqaFixpipeDualDstGeometry gqa_fixpipe_dual_dst_geometry(
    std::uint32_t m, std::uint32_t n, std::uint32_t dst_off_bytes,
    bool dual_dst, std::uint32_t unit_flag)
```

Line 143:
```cpp
template <bool DualDst, unsigned UnitFlag, class Engine>
ASCEND_MICRO_CUBE_INLINE void cube_gqa_fixpipe_dual_dst(Engine& engine,
                                                       std::uint32_t m,
                                                       std::uint32_t n,
                                                       std::uint32_t dst_off_bytes)
```

Line 156:
```cpp
constexpr AscendC::FixpipeConfig kGqaFixpipeToUb = {
    AscendC::CO2Layout::ROW_MAJOR, true};
```

Line 163:
```cpp
template <bool DualDst, unsigned M, unsigned N>
struct GqaFixpipeDualDstShape {
```

Line 180:
```cpp
struct NativeGqaFixpipeDualDstEngine {
    static constexpr std::uint32_t kL0cOffsetBytes = 0;
```

Line 183:
```cpp
    template <bool DualDst, unsigned UnitFlag>
    __aicore__ inline void fixpipe_dual_dst(std::uint32_t m, std::uint32_t n,
                                            std::uint32_t dst_off_bytes) const
```

Supported dtypes; input/output distinctions retained:
```json
{
  "BF16": {
    "supported": false,
    "evidence": {
      "path": "C:/ascend-micro-harness/routines/routine_cube_gqa_fixpipe_dual_dst.hpp",
      "line_start": 202,
      "line_end": 213,
      "constraint": "float destination and source, NoQuant; no BF16 template"
    }
  },
  "INT8": {
    "supported": false,
    "evidence": {
      "path": "C:/ascend-micro-harness/routines/routine_cube_gqa_fixpipe_dual_dst.hpp",
      "line_start": 202,
      "line_end": 213,
      "constraint": "float destination and source, NoQuant; no int8 template"
    }
  },
  "FP32": {
    "supported": true,
    "evidence": {
      "path": "C:/ascend-micro-harness/routines/routine_cube_gqa_fixpipe_dual_dst.hpp",
      "line_start": 202,
      "line_end": 213,
      "constraint": "Fixpipe<float,float,kGqaFixpipeToUb>"
    }
  }
}
```

Execution pipe:
```json
{
  "issuer": "AIC",
  "pipe": "PIPE_FIX",
  "instruction": "copy_matrix_cc_to_ub",
  "path": "CO1/L0C NZ FP32 -> both AIV VECCALC UB ND FP32",
  "evidence": [
    {
      "path": "/home/dmin/Ascend-9.2.1/cann-9.2.1/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_fixpipe_impl.h",
      "line_start": 227,
      "line_end": 281
    },
    {
      "path": "/home/dmin/Ascend-9.2.1/cann-9.2.1/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_fixpipe_impl.h",
      "line_start": 642,
      "line_end": 668
    }
  ]
}
```

Geometry:
```json
{
  "current_score": {
    "total_M": 16,
    "N_or_D": 64,
    "M_per_destination": 8,
    "srcStride_C0_units": 16,
    "dstStride_elements": 64,
    "L0C_bytes": 4096,
    "UB_bytes_per_AIV": 2048
  },
  "current_context": {
    "total_M": 16,
    "N_or_D": 256,
    "M_per_destination": 8,
    "srcStride_C0_units": 16,
    "dstStride_elements": 256,
    "L0C_bytes": 16384,
    "UB_bytes_per_AIV": 8192
  },
  "proposed_M128_D256_interpretations": [
    {
      "interpretation": "M=128 total L0C rows",
      "mSize": 128,
      "nSize": 256,
      "M_per_destination": 64,
      "srcStride_C0_units": 128,
      "dstStride_elements": 256,
      "L0C_bytes": 131072,
      "UB_bytes_per_AIV": 65536,
      "depth2_UB_landing_bytes_per_AIV": 131072
    },
    {
      "interpretation": "M=128 rows per destination",
      "mSize": 256,
      "nSize": 256,
      "M_per_destination": 128,
      "srcStride_C0_units": 256,
      "dstStride_elements": 256,
      "L0C_bytes": 262144,
      "UB_bytes_per_AIV": 131072,
      "depth2_UB_landing_bytes_per_AIV": 262144
    }
  ],
  "proposed_validated": false,
  "mapping_rule": "mSize is TOTAL M before split, not each destination M; srcStride=ceil(mSize/16)*16 C0 units; dstStride=N elements; dualDstCtl=1. A desired per-AIV M requires mSize=2*M.",
  "evidence": [
    {
      "path": "C:/ascend-micro-harness/routines/routine_cube_gqa_fixpipe_dual_dst.hpp",
      "line_start": 124,
      "line_end": 136
    },
    {
      "path": "C:/ascend-micro-harness/routines/routine_cube_gqa_fixpipe_dual_dst.hpp",
      "line_start": 163,
      "line_end": 177
    },
    {
      "path": "C:/ascend-micro-harness/routines/routine_cube_gqa_fixpipe_dual_dst.hpp",
      "line_start": 193,
      "line_end": 213
    }
  ]
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "ownership": "caller allocates; leaf creates tensor views only",
  "UB_bytes_actual_per_AIV": [
    2048,
    8192
  ],
  "UB_symbolic": "(DualDst ? M/2 : M)*N*4; each view advertises m*n elements although dual store touches half per UB",
  "L1_bytes": 0,
  "L0A_bytes": 0,
  "L0B_bytes": 0,
  "L0C_symbolic": "ceil(m/16)*16*n*4 physical source extent; when M multiple16, M*N*4",
  "L0C_offset_bytes": 0,
  "VREG_bytes": null,
  "VREG_note": "no vector arithmetic; compiler scalar register allocation not measured"
}
```

Synchronization/lifetime contract:
```json
{
  "leaf_issues_events": false,
  "caller_required": [
    "M_FIX before store unless valid matching unitFlag handles Mmad/Fixpipe contract",
    "FIX_S SetFlag/WaitFlag real completion before scalar completion/publication when using scalar publication sequencing",
    "publish after completed FIX to both AIV consumers",
    "hold L0C until FIX read completes",
    "hold each UB landing bank until that AIV consumer has completed actual reads; both must participate in mode2 reverse quorum",
    "use independent reverse credit for each depth2 bank and each independent resource lifetime"
  ],
  "evidence": {
    "path": "C:/ascend-micro-harness/routines/routine_cube_gqa_fixpipe_dual_dst.hpp",
    "line_start": 74,
    "line_end": 83
  }
}
```

CAModel verification and limitations:
```json
{
  "status": "legacy report explicitly FAIL timeout; no 9.2.1 run; no inferred PASS",
  "report": {
    "path": "C:/ascend-micro-harness/reports/gqa_fixpipe_dual_dst.md",
    "line_start": 136,
    "line_end": 179
  },
  "report_toolchain": "CANN9.2.0-beta.2",
  "reported_artifact_paths": [
    "/tmp/ascend-modulo3-camodel-fC0sk5",
    "C:/ascend-micro-harness/kernels/camodel/gqa_fixpipe_dual_dst.cpp",
    "C:/ascend-micro-harness/tests/camodel/test_camodel_gqa_fixpipe_dual_dst.cpp"
  ],
  "raw_artifact_verified": false,
  "exact_current_sha_bound_to_run": null,
  "coverage": "Report observes window0 dual halves and acknowledgments but whole two-window run times out. Shape16x256 only host evidence; no depth2 wrap or M128/D256 evidence."
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_cube_gqa_fixpipe_dual_dst`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_cube_gqa_pv_gemm.hpp

Path: `C:/ascend-micro-harness/routines/routine_cube_gqa_pv_gemm.hpp`  
SHA256: `98b4650a5e4fac386f6e37999e8380347d5cd9c49950bcbd20d3f85829968328`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 227:
```cpp
template <class Engine>
ASCEND_MICRO_GQA_PV_INLINE void gqa_pv_gemm_stage_l0(Engine& engine)
```

Line 239:
```cpp
template <class Engine>
ASCEND_MICRO_GQA_PV_INLINE void gqa_pv_gemm_compute(Engine& engine)
```

Line 246:
```cpp
template <class Engine>
ASCEND_MICRO_GQA_PV_INLINE GqaStatus gqa_pv_gemm(Engine& engine,
                                                 const GqaDesc& ctx_l0c,
                                                 const GqaDesc& p_hi_l1,
                                                 const GqaDesc& p_lo_l1,
                                                 const GqaDesc& v_l1,
                                                 const GqaPvParams* p)
```

Line 308:
```cpp
    __aicore__ inline std::uint32_t head_dim() const
```

Line 309:
```cpp
    __aicore__ inline std::uint32_t chunk() const
```

Line 311:
```cpp
    __aicore__ inline void load_p_hi()
```

Line 312:
```cpp
    __aicore__ inline void load_p_lo()
```

Line 320:
```cpp
    __aicore__ inline void load_v()
```

Line 330:
```cpp
    __aicore__ inline void mmad_chains()
```

Line 343:
```cpp
    __aicore__ inline void LoadPlane(__cbuf__ half* src_l1, __ca__ half* dst)
```

Line 356:
```cpp
    template <std::uint32_t Limb>
    __aicore__ inline void Chain(bool init)
```

Line 367:
```cpp
    template <std::uint32_t Limb>
    __aicore__ inline void BarrierAfter()
```

Line 379:
```cpp
    template <bool Enabled>
    __aicore__ inline void Barrier()
```

Line 387:
```cpp
    template <std::uint32_t Limb, std::uint32_t G>
    __aicore__ inline void Step(bool init)
```

Line 406:
```cpp
    template <std::uint32_t Limb, std::uint32_t... Gs>
    __aicore__ inline void Steps(bool init, std::integer_sequence<std::uint32_t, Gs...>)
```

Supported dtypes; input/output distinctions retained:
```json
{
  "BF16": "Explicitly unrun and not implemented lines199-200; native useshalf; descriptors requireFP16.",
  "INT8": "Absent dtype/operands; unsupported by leaf.",
  "FP32": "Output accumulator only.",
  "FP16": "Actual P_hi,P_lo,V inputs."
}
```

Execution pipe:
```json
{
  "actual": "AIC MTE1 two probability plane identity loads and V per-fractal transpose; PIPE_M two k32-step chains into same C, first-hi init only; SetMMColumnMajor. PIPE_M barriers enabled D<160 but sequencing defective for multistep narrow heads.",
  "future": "No Fixpipe in leaf,no paired UB destination,no1AIC2AIV handshake/ring/scalar drains."
}
```

Geometry:
```json
{
  "M": 16,
  "default_D": 256,
  "default_Chunk": 64,
  "native_template": "template <std::uint32_t D = kGqaPvHeadDim,std::uint32_t Chunk = kGqaPvChunk> struct NativeGqaPvGemmEngineImpl",
  "constraints": "D%16=0,Chunk%16=0,Chunk%32=0,16<=Chunk<=64 (effective32/64),16*D*2<=32768,ABytes<=65536,BBytes<=32768,CBytes<=262144,CBytes%1024=0. Runtime wrapper does not enforce all static bounds or address/stride/alignment/nonoverlap; Shape alias completeness unverified.",
  "layout": "P planes eachNZ(16,Chunk);V L1NZ(Chunk,D)->L0BZN(Chunk,D) via actual per-fractal transpose; not score K identity reinterpret. Two limbs are complete separate planes."
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "allocation_owner": "Caller; all native pointer fields independent of descriptor addresses",
  "UB_actual_bytes": 0,
  "L1_symbolic_payload_bytes": "Phi=16*Chunk*2,Plo=16*Chunk*2,V=Chunk*D*2; default2048+2048+32768",
  "L1_wrapper_actual": "Phi base0,Plo2560,V5120,arena end37888; kernels/camodel/gqa_pv_gemm.cpp:52-76",
  "L0A_symbolic_bytes": "2*16*Chunk*2",
  "L0A_actual_default_bytes": 4096,
  "L0B_actual_default_bytes": 32768,
  "L0C_actual_default_bytes": 16384,
  "VREG_actual": "None explicitly; generated register use unknown",
  "comment_mismatch": "Header lines85-90 and289-290 claim8KB each plane/16KB total L0A; formulas/native pointer offsets/wrapper yield2KB each/4KB total atChunk64. Use actual formulas, not narrative."
}
```

Synchronization/lifetime contract:
```json
{
  "owned": "SetMMColumnMajor; conditional PIPE_M inside compute, no HardEvent/cross-core flag.",
  "caller_edges": [
    "probability ready and MTE2_MTE1/V->L1 visibility before stage",
    "MTE1_M between stage and compute",
    "M_FIX after both chains",
    "CtxFree/FIX_M before C reuse",
    "M_MTE1 before A/B overwrite; L1 reverseACK after last MTE1 read"
  ],
  "hazard": "Steps lines409-412 emits comma-fold of ALL Step calls then separate comma-fold of barriers. D<160,Chunk64 needs barrier between two calls in each limb; actual barriers occur after chain. Seam barrier is placed correctly. Existing narrowD tests useChunk32(one step), so cannot detect this path. Whole wrapper calls both stages with no interposed edge."
}
```

CAModel verification and limitations:
```json
[
  {
    "path": "C:/ascend-micro-harness/reports/gqa_pv_gemm.md",
    "lines": "299-440",
    "note": "Historical CANN9.2.0-beta.2 report claims1AIC4096 bit-exact entries,256 guards,3801 dispatch cycles (budget1400),4-MMAD trace; depth/wrap NOT RUN and D64/128 not CAModel-tested. No raw source-bound artifact verified."
  },
  {
    "path": "C:/ascend-micro-harness/kernels/camodel/gqa_pv_gemm.cpp",
    "lines": "122-127",
    "note": "Native wrapper uses split stage_l0/compute plus MTE1_M/M_FIX, not combined descriptor wrapper."
  }
]
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_cube_gqa_pv_gemm`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_cube_gqa_score_gemm.hpp

Path: `C:/ascend-micro-harness/routines/routine_cube_gqa_score_gemm.hpp`  
SHA256: `cd4d6bfa9aa02bbff2d6cbfffcf19334ae84bde602a94043934a92386ed58d2d`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 203:
```cpp
template <class Engine>
ASCEND_MICRO_GQA_SCORE_INLINE void gqa_score_gemm_stage_l0(Engine& engine)
```

Line 215:
```cpp
template <class Engine>
ASCEND_MICRO_GQA_SCORE_INLINE void gqa_score_gemm_compute(Engine& engine)
```

Line 224:
```cpp
template <class Engine>
ASCEND_MICRO_GQA_SCORE_INLINE GqaStatus gqa_score_gemm(Engine& engine,
                                                      const GqaDesc& s_l0c,
                                                      const GqaDesc& q_l1,
                                                      const GqaDesc& k_l1,
                                                      const GqaScoreParams* p)
```

Line 289:
```cpp
    __aicore__ inline std::uint32_t head_dim() const
```

Line 290:
```cpp
    __aicore__ inline std::uint32_t chunk() const
```

Line 293:
```cpp
    __aicore__ inline void load_q()
```

Line 306:
```cpp
    __aicore__ inline void load_k()
```

Line 319:
```cpp
    __aicore__ inline void mmad_chain()
```

Line 330:
```cpp
    template <std::uint32_t G>
    __aicore__ inline void MmadStep()
```

Line 348:
```cpp
    template <std::uint32_t... Gs>
    __aicore__ inline void MmadSteps(std::integer_sequence<std::uint32_t, Gs...>)
```

Supported dtypes; input/output distinctions retained:
```json
{
  "BF16": "Explicitly not implemented/tested lines165-169; enum kBf16 is metadata only; wrapper requires kFp16.",
  "INT8": "Absent from GqaDType and native pointers; unsupported by leaf.",
  "FP32": "Output/accumulator only; descriptors demand FP32 L0C, half native operands.",
  "FP16": "Actual native A/B input and descriptor contract."
}
```

Execution pipe:
```json
{
  "actual": "AIC PIPE_MTE1 CopyL12L0A/B then PIPE_M8xk32 Mmad at D256 Chunk64; SetMMRowMajor. No FIX in leaf.",
  "future": "Fixpipe dual destination is separate sibling; no flags,depth2 slot/ring,scalar drain or symmetric AIV roles implemented here."
}
```

Geometry:
```json
{
  "M": 16,
  "native_D": 256,
  "native_Chunk": 64,
  "generic_host_engine": "Param matches engine.head_dim()/chunk(); D%32==0,N%16==0 and nonzero; no runtime <=64,power2,budget,address/stride validation.",
  "shape_template_constraints": "D%32=0,D%16=0,Chunk%16=0,positive power2 Chunk<=64,D*Chunk*2<=32768,D*D*4<=262144. D*D*4 is not actual score C footprint. using Shape alias alone may not force complete-type instantiation; compile not tested."
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "allocation_owner": "Caller; descriptor addr/stride metadata never bind to native pointers",
  "UB_actual_bytes": 0,
  "L1_symbolic_payload_bytes": "Q=16*D*2,K=Chunk*D*2; at256/64=8192+32768",
  "L1_wrapper_actual": "Qbase0,Kbase8704,arena end41472; kernels/camodel/gqa_score_gemm.cpp:45-66",
  "L0A_actual_bytes": 8192,
  "L0B_actual_bytes": 32768,
  "L0C_actual_bytes": 4096,
  "VREG_actual": "None explicitly; compiler scalar/register footprint unknown",
  "L0_capacity_caveat": "32KiB is declared operand budget; cannot treat one-slot budget as a depth2 capacity proof."
}
```

Synchronization/lifetime contract:
```json
{
  "owned": "None; no PIPE_M barriers.",
  "caller_edges": [
    "MTE2_MTE1 into stage_l0",
    "MTE1_M between explicit stage_l0 and compute",
    "M_FIX out",
    "ScoreFree/FIX_M before C reuse",
    "M_MTE1 before A/B overwrite",
    "L1 slot free only after MTE1 last read completes"
  ],
  "hazard": "At m16 n64 fragment count4<10, so ISASI doc asks inter-accumulation PIPE_M. Native chain omits it. Report blind-spot paragraph incorrectly says product16>=10. Whole gqa_score_gemm calls stage then compute directly without inserting MTE1_M; use split stage functions for caller scheduling."
}
```

CAModel verification and limitations:
```json
[
  {
    "path": "C:/ascend-micro-harness/reports/gqa_score_gemm.md",
    "lines": "245-306",
    "note": "Historical CANN9.2.0-beta.2 report claims1AIC1024 entries bit-exact,384 guards and3370 dispatch cycles (over1200 budget); depth/wrap NOT RUN and trace audit NOT RUN; no bound raw log verified."
  },
  {
    "path": "C:/ascend-micro-harness/kernels/camodel/gqa_score_gemm.cpp",
    "lines": "105-110",
    "note": "Wrapper uses explicit split stage functions and inserts MTE1_M/M_FIX; whole descriptor wrapper is not the tested native path."
  }
]
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_cube_gqa_score_gemm`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_cube_hadamard16.hpp

Path: `C:/ascend-micro-harness/routines/routine_cube_hadamard16.hpp`  
SHA256: `368a93559fcc78c723676f268dadd310428e0c228e994e655c7c305fbd70ffd3`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 14:
```cpp
enum class OutflowMode : std::uint8_t { Fp32 = 0, Fp16 = 1, Bf16 = 2 };
```

Line 32:
```cpp
template <OutflowMode Mode, class Engine> ASCEND_MICRO_AIC_INLINE void cube_hadamard16(Engine& engine)
```

Line 47:
```cpp
template <class Output> struct NativeCubeEngine
```

Line 54:
```cpp
ASCEND_MICRO_AIC_INLINE void NativeCubeEngine<Output>::mmad()
```

Line 65:
```cpp
template <OutflowMode Mode> ASCEND_MICRO_AIC_INLINE void NativeCubeEngine<Output>::fixpipe()
```

Line 77:
```cpp
template <OutflowMode Mode, class Output> ASCEND_MICRO_AIC_INLINE void cube_hadamard16_native(__ca__ half* l0a, __cb__ half* l0b, __cc__ float* l0c, __gm__ Output* output)
```

Supported dtypes; input/output distinctions retained:
```json
{
  "BF16": {
    "input": false,
    "output": "Output=bfloat16_t supported by float->BF16 Copy API; Mode::Bf16 names intent",
    "evidence_lines": [
      49,
      50,
      51,
      52,
      65,
      73
    ],
    "caveat": "Mode is validated by generic wrapper but unused by native fixpipe body; Output pointer dtype actually selects output conversion. No Mode/Output correspondence static_assert."
  },
  "INT8": {
    "input": false,
    "output": false,
    "evidence_lines": [
      49,
      50,
      51,
      73
    ]
  },
  "FP32": {
    "input": false,
    "accumulator": true,
    "output": true,
    "evidence_lines": [
      49,
      50,
      51,
      58,
      59,
      69,
      73
    ]
  },
  "FP16": {
    "input": true,
    "output": true
  }
}
```

Execution pipe:
```json
{
  "core": "AIC",
  "pipes": [
    "PIPE_M Cube Mmad",
    "Fixpipe L0C->GM"
  ],
  "SIMD": false,
  "SIMT": false,
  "MIX": "not a kernel declaration; surrounding caller may be MIX, leaf itself has no AIV work"
}
```

Geometry:
```json
{
  "M": 16,
  "N": 16,
  "K": 16,
  "tails": false,
  "normalization": "none; caller-supplied 16x16 sign matrix, no 1/4 scaling",
  "M128_M256_D256": "not directly supported; independent 16-row x16-column tiles can be invoked by caller but sixteen independent H16 blocks are not full H256"
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "allocation_owner": "caller; no local data buffers allocated in header",
  "required_payload_bytes": {
    "UB": 0,
    "L1": null,
    "L0A": 512,
    "L0B": 512,
    "L0C": 1024,
    "VREG": 0,
    "GM_output": "256*sizeof(Output)"
  },
  "recommended_caller_L1": "A at0, B at544 (512+32 bank skew), span1056 bytes; header constants only, no allocation",
  "M128_D256_BF16_full_operand_bytes": 65536,
  "M256_D256_BF16_full_operand_bytes": 131072,
  "large_tile_note": "hypothetical full operands only; existing fixed leaf cannot consume these shapes"
}
```

Synchronization/lifetime contract:
```json
{
  "leaf": "asynchronous; Mmad unitFlag3 + Fixpipe unitFlag3 handles their C producer/consumer ordering only",
  "caller": "complete GM->L1->L0 before entry; wait final GM visibility; preserve operand/output lifetimes and nonaliasing",
  "explicit_events_in_leaf": false,
  "evidence_lines": [
    27,
    28,
    29,
    30,
    31,
    43,
    44,
    45,
    61,
    72
  ]
}
```

CAModel verification and limitations:
```json
{
  "status": "no verified CAModel evidence found for this leaf in bounded report/README search",
  "current_CANN9_2_1_result": null,
  "evidence": [
    {
      "path": "C:/ascend-micro-harness/README.md",
      "line_start": 607,
      "line_end": 631,
      "note": "native arithmetic and lifecycle contract"
    },
    {
      "path": "C:/ascend-micro-harness/README.md",
      "line_start": 633,
      "line_end": 661,
      "note": "host mock/oracle only; native compilation/execution validation still required by report"
    }
  ],
  "raw_execution_artifact_paths": null,
  "test_presence_is_pass": false
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_cube_hadamard16`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_cube_matmul.hpp

Path: `C:/ascend-micro-harness/routines/routine_cube_matmul.hpp`  
SHA256: `c94fd44031748725393d7d685e68dcc80d3aefca91d0b08ad97e001041a4e453`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 21:
```cpp
template <class Engine> ASCEND_MICRO_CUBE_INLINE void cube_matmul_stage_l0(Engine& engine)
```

Line 28:
```cpp
template <class Engine> ASCEND_MICRO_CUBE_INLINE void cube_matmul_compute_dump(Engine& engine)
```

Line 37:
```cpp
template <int M> struct NativeCubeMatmulEngine
```

Line 49:
```cpp
__aicore__ inline void NativeCubeMatmulEngine<M>::load_a()
```

Line 56:
```cpp
__aicore__ inline void NativeCubeMatmulEngine<M>::load_b()
```

Line 63:
```cpp
__aicore__ inline void NativeCubeMatmulEngine<M>::mma_dispatch()
```

Line 64:
```cpp
__aicore__ inline void NativeCubeMatmulEngine<M>::fixpipe_to_ub()
```

Line 75:
```cpp
template <int M> __aicore__ inline void NativeCubeMatmulEngine<M>::mma_dispatch()
```

Supported dtypes; input/output distinctions retained:
```json
{
  "BF16": {
    "input": false,
    "output": false,
    "evidence_lines": [
      42,
      43,
      44,
      45,
      46,
      47
    ]
  },
  "INT8": {
    "input": false,
    "output": false,
    "evidence_lines": [
      42,
      43,
      44,
      45,
      46,
      47
    ]
  },
  "FP32": {
    "input": false,
    "accumulator": true,
    "output": true,
    "evidence_lines": [
      46,
      47,
      67,
      68,
      81
    ]
  },
  "FP16": {
    "input": true
  }
}
```

Execution pipe:
```json
{
  "core": "AIC",
  "pipes": [
    "MTE1 L1->L0A/L0B",
    "PIPE_M Cube Mmad",
    "Fixpipe L0C->UB"
  ],
  "SIMD": false,
  "SIMT": false,
  "MIX": "caller can integrate in MIX_AIC_1_2; default subBlockId false writes AIV0 UB; no MIX kernel declaration in leaf"
}
```

Geometry:
```json
{
  "M_allowed": [
    16,
    32
  ],
  "N": 16,
  "K": 16,
  "M_constraint": "static_assert M ==16 || M ==32 at39",
  "tails": false,
  "M128_M256_D256": "M128/M256 rejected; N/K256 absent; batching M16/32 is caller work, not generalized support",
  "K_accumulation": "MmadParams(...,3,true) resets C every dispatch, so repeated K16 calls do not accumulate to D256"
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "allocation_owner": "caller; no local data buffer allocation",
  "symbolic_required_payload_bytes": {
    "UB": "64*M",
    "L1": "32*M+512, before caller bank padding",
    "L0A": "32*M",
    "L0B": 512,
    "L0C": "64*M",
    "VREG": 0
  },
  "M16_required_payload_bytes": {
    "UB": 1024,
    "L1": 1024,
    "L0A": 512,
    "L0B": 512,
    "L0C": 1024
  },
  "M32_required_payload_bytes": {
    "UB": 2048,
    "L1": 1536,
    "L0A": 1024,
    "L0B": 512,
    "L0C": 2048
  },
  "bank_skew": "kCubeBankSkewBytes32 is declared but not applied by leaf"
}
```

Synchronization/lifetime contract:
```json
{
  "leaf": "stage dispatches load_a/load_b; compute dispatches Mmad+Fixpipe with matching unitFlag3; no explicit events",
  "caller": "complete L1->L0 between the phases, wait Fixpipe before UB Vector consumers, route MIX output to intended subblock and handle cross-core flags",
  "evidence_lines": [
    19,
    20,
    24,
    25,
    31,
    32,
    70,
    83
  ]
}
```

CAModel verification and limitations:
```json
{
  "status": "historical report documents PASS for M16/M32 half inputs on CANN9.2.0-beta.2, not current-target exact-header validation",
  "current_CANN9_2_1_result": null,
  "evidence": [
    {
      "path": "C:/ascend-micro-harness/reports/micro_routines_validation.md",
      "line_start": 58,
      "line_end": 75,
      "note": "M16 256/256, M32 512/512 exact,384/384guards,0 address errors,5507 ticks"
    },
    {
      "path": "C:/ascend-micro-harness/reports/micro_routines_validation.md",
      "line_start": 79,
      "line_end": 95,
      "note": "historical MIX integration and token accounting"
    },
    {
      "path": "C:/ascend-micro-harness/reports/fused_attention_integration.md",
      "line_start": 104,
      "line_end": 114,
      "note": "AIC compute and default AIV0 UB destination"
    }
  ],
  "raw_execution_artifact_paths": null,
  "test_presence_is_pass": false
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_cube_matmul`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_cube_wta_fixpipe.hpp

Path: `C:/ascend-micro-harness/routines/routine_cube_wta_fixpipe.hpp`  
SHA256: `f824cf64c5a53823b173064215d7e2d650e59eb11958ea3e264728f4da0375de`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 24:
```cpp
template <unsigned T, std::uint32_t ScaleBits = 0x43000000u,
          class Output = std::int8_t>
struct WtaConfig {
```

Line 48:
```cpp
template <class Engine>
ASCEND_WTA_INLINE void cube_wta_load(Engine& e) { e.load_a(); }
```

Line 50:
```cpp
template <class Engine>
ASCEND_WTA_INLINE void cube_wta_mmad(Engine& e) { e.mma_dispatch(); }
```

Line 52:
```cpp
template <bool Final, class Engine>
ASCEND_WTA_INLINE void cube_wta_fixpipe(Engine& e)
```

Line 59:
```cpp
template <unsigned Step, class Config, class Engine, class Sync>
ASCEND_WTA_INLINE void step(Engine& e, Sync& sync)
```

Line 79:
```cpp
template <class Config, class Engine, class Sync>
ASCEND_WTA_INLINE void cube_wta_run(Engine& e, Sync& sync)
```

Line 93:
```cpp
template <class Config>
struct NativeCubeWtaEngine {
```

Line 106:
```cpp
    ASCEND_WTA_INLINE void load_a()
```

Line 114:
```cpp
    ASCEND_WTA_INLINE void mma_dispatch()
```

Line 122:
```cpp
    template <bool Final>
    ASCEND_WTA_INLINE void fixpipe()
```

Line 143:
```cpp
template <unsigned EventId = 0>
struct NativeCubeWtaSync {
```

Line 146:
```cpp
    ASCEND_WTA_INLINE void publish_feedback()
    { AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EventId); }
```

Line 148:
```cpp
    ASCEND_WTA_INLINE void wait_feedback()
    { AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EventId); }
```

Line 150:
```cpp
    ASCEND_WTA_INLINE void operands_ready()
```

Line 155:
```cpp
    ASCEND_WTA_INLINE void result_ready()
```

Line 160:
```cpp
    ASCEND_WTA_INLINE void finish()
```

Supported dtypes; input/output distinctions retained:
```json
{
  "BF16": {
    "supported": false,
    "evidence": {
      "path": "C:/ascend-micro-harness/routines/routine_cube_wta_fixpipe.hpp",
      "line_start": 100,
      "line_end": 104,
      "constraint": "half inputs/state fixed; no BF16 configuration"
    }
  },
  "INT8": {
    "supported": true,
    "role": "final output only, int8_t or uint8_t",
    "evidence": [
      {
        "path": "C:/ascend-micro-harness/routines/routine_cube_wta_fixpipe.hpp",
        "line_start": 24,
        "line_end": 38
      },
      {
        "path": "C:/ascend-micro-harness/routines/routine_cube_wta_fixpipe.hpp",
        "line_start": 130,
        "line_end": 134
      }
    ]
  },
  "FP32": {
    "supported": true,
    "role": "L0C accumulator only; no FP32 public input/output state",
    "evidence": {
      "path": "C:/ascend-micro-harness/routines/routine_cube_wta_fixpipe.hpp",
      "line_start": 100,
      "line_end": 104
    }
  },
  "FP16": {
    "supported": true,
    "role": "operands and recurrent feedback",
    "evidence": {
      "path": "C:/ascend-micro-harness/routines/routine_cube_wta_fixpipe.hpp",
      "line_start": 100,
      "line_end": 104
    }
  }
}
```

Execution pipe:
```json
{
  "issuer": "AIC only",
  "pipes": [
    "MTE1: LoadData C1->L0A",
    "M: Mmad half*half->float",
    "FIX: Fixpipe CO1->C1 with ReLU, float->half feedback or scalar int8/uint8 final"
  ],
  "evidence": [
    {
      "path": "C:/ascend-micro-harness/routines/routine_cube_wta_fixpipe.hpp",
      "line_start": 106,
      "line_end": 138
    },
    {
      "path": "/home/dmin/Ascend-9.2.1/cann-9.2.1/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_fixpipe_impl.h",
      "line_start": 187,
      "line_end": 221
    }
  ]
}
```

Geometry:
```json
{
  "current_M": 16,
  "current_N_competitors": 16,
  "current_K": 16,
  "current_D": null,
  "current_elements": 256,
  "proposed_M128_D256": "unsupported by this fixed16x16x16 engine/config. D=256 is not this routine’s logical dimension; no D256 Hadamard or 256-competitor WTA can be inferred. Requires new geometry, load schedules, weights and resource audit.",
  "evidence": {
    "path": "C:/ascend-micro-harness/routines/routine_cube_wta_fixpipe.hpp",
    "line_start": 15,
    "line_end": 18
  }
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "caller_owned": true,
  "UB_bytes": 0,
  "L1_state_bytes": 512,
  "L1_final_output_bytes": 256,
  "L1_nonaliasing_total_bytes": 768,
  "L0A_bytes": 512,
  "L0B_bytes": 512,
  "L0C_bytes": 1024,
  "VREG_bytes": null,
  "VREG_note": "no AIV/vector register arithmetic; compiler registers unmeasured",
  "alignment": {
    "L0A": 512,
    "L0B": 512,
    "L0C": 64,
    "L1_state_and_output": 32
  },
  "evidence": {
    "path": "C:/ascend-micro-harness/routines/routine_cube_wta_fixpipe.hpp",
    "line_start": 95,
    "line_end": 104
  }
}
```

Synchronization/lifetime contract:
```json
{
  "leaves_issue_events": false,
  "driver_issues_events": true,
  "driver": "FIX_MTE1 prime; each step wait feedback, LoadData, MTE1_M pair, Mmad, M_FIX pair, Fixpipe, publish FIX_MTE1; final wait consumes token then FIX_S pair drains FIX.",
  "counts": "FIX_MTE1 T+1 sets and T+1 waits; MTE1_M T pairs; M_FIX T pairs; FIX_S 1 pair",
  "caller_reserves": "EventId 0..5 across FIX_MTE1, MTE1_M, M_FIX, FIX_S for full driver lifetime; upstream initial C1 producer completed before priming; W held resident in nonaliasing L0B",
  "lifetime": "State not overwritten until prior Mmad complete; L0A/L0C not reused until prior Fixpipe complete; final output ready when finish FIX_S completes. No V_S or cross-core operation in this routine.",
  "evidence": [
    {
      "path": "C:/ascend-micro-harness/routines/routine_cube_wta_fixpipe.hpp",
      "line_start": 59,
      "line_end": 85
    },
    {
      "path": "C:/ascend-micro-harness/routines/routine_cube_wta_fixpipe.hpp",
      "line_start": 142,
      "line_end": 164
    }
  ]
}
```

CAModel verification and limitations:
```json
{
  "status": "legacy report CLAIMED PASS only; raw artifacts not inspected, current SHA linkage null, 9.2.1 unvalidated",
  "report": {
    "path": "C:/ascend-micro-harness/reports/wta_fixpipe.md",
    "line_start": 167,
    "line_end": 208
  },
  "report_toolchain": "CANN9.2.0-beta.2,2026-10-23,Ascend950PR_9599 / driver9589",
  "reported_coverage": "64 launches T1..8 x signed/unsigned x four margins; report says64/64 passes and trace audits, exact256 bytes; not full one-hot convergence",
  "artifact_paths": [
    "C:/ascend-micro-harness/kernels/camodel/wta_fixpipe.cpp",
    "C:/ascend-micro-harness/tests/cpu/test_wta_convergence_probe.cpp",
    "C:/ascend-micro-harness/scripts/run_wta_camodel.sh",
    "C:/ascend-micro-harness/scripts/analyze_wta_trace.py"
  ],
  "raw_trace_directory": null,
  "raw_artifact_verified": false,
  "exact_current_sha_bound_to_run": null
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_cube_wta_fixpipe`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_cv_gqa_stage_probs.hpp

Path: `C:/ascend-micro-harness/routines/routine_cv_gqa_stage_probs.hpp`  
SHA256: `4b11bbb367a17ad76bf582951a8930234a2e9b5009dd450b649516e15a5e4c47`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line [178, 178]:
```cpp
{"lines": [178, 178], "exact": "constexpr std::uint32_t GqaSpPlaneElements(std::uint32_t tokens)"}
```

Line [182, 183]:
```cpp
{"lines": [182, 183], "exact": "template <std::uint32_t kTokens>\nconstexpr std::uint32_t GqaSpPlaneElementsValue = GqaSpPlaneElements(kTokens);"}
```

Line [188, 188]:
```cpp
{"lines": [188, 188], "exact": "constexpr std::uint32_t GqaSpPlaneIndex(std::uint32_t t, std::uint32_t c)"}
```

Line [210, 213]:
```cpp
{"lines": [210, 213], "exact": "template <class T>\nASCEND_MICRO_GQA_SP_INLINE std::int32_t gqa_stage_probs_extents(\n    std::uint32_t rows, std::uint32_t tokens, std::uint32_t row_offset,\n    std::uint32_t src_pitch, GqaSpExtents& out)"}
```

Line [251, 255]:
```cpp
{"lines": [251, 255], "exact": "template <class T, class Engine>\nASCEND_MICRO_GQA_SP_INLINE std::int32_t gqa_stage_probs(\n    Engine& engine, std::uint32_t p_hi_base, std::uint32_t p_lo_base,\n    std::uint32_t probs_ub, std::uint32_t hi_ub, std::uint32_t resid_ub,\n    std::uint32_t plane_ub, const GqaSpExtents& extents)"}
```

Line [282, 285]:
```cpp
{"lines": [282, 285], "exact": "template <class T>\nstatic constexpr AscendC::RoundMode kGqaSpCastModeOf = AscendC::RoundMode::CAST_NONE;\ntemplate <>\nstatic constexpr AscendC::RoundMode kGqaSpCastModeOf<half> = AscendC::RoundMode::CAST_NONE;"}
```

Line [287, 289]:
```cpp
{"lines": [287, 289], "exact": "template <>\nstatic constexpr AscendC::RoundMode kGqaSpCastModeOf<bfloat16_t> =\n    AscendC::RoundMode::CAST_ROUND;"}
```

Line [311, 312]:
```cpp
{"lines": [311, 312], "exact": "template <class T>\nstruct NativeGqaStageProbsEngine {"}
```

Line [319, 321]:
```cpp
{"lines": [319, 321], "exact": "    template <class U>\n    __aicore__ inline void cast_to_plane(std::uint32_t dst_ub, std::uint32_t src_ub,\n                                         std::uint32_t rows, std::uint32_t tokens) const"}
```

Line [330, 332]:
```cpp
{"lines": [330, 332], "exact": "    template <class U>\n    __aicore__ inline void widen_plane(std::uint32_t dst_ub, std::uint32_t src_ub,\n                                       std::uint32_t rows, std::uint32_t tokens) const"}
```

Line [340, 342]:
```cpp
{"lines": [340, 342], "exact": "    __aicore__ inline void subtract_residual(std::uint32_t dst_ub, std::uint32_t lhs_ub,\n                                             std::uint32_t rhs_ub,\n                                             std::uint32_t count) const"}
```

Line [354, 356]:
```cpp
{"lines": [354, 356], "exact": "    template <class U>\n    __aicore__ inline void stage(std::uint32_t plane_base, std::uint32_t plane_ub,\n                                 const GqaSpExtents& extents) const"}
```

Supported dtypes; input/output distinctions retained:
```json
{
  "FP32": "input probabilities, widened high limb, subtract residual",
  "BF16": "two output limbs; native specialization CAST_ROUND only under __CCE_AICORE__",
  "FP16": "two output limbs CAST_NONE",
  "INT8": "unsupported: extents static_assert sizeof(T)==2",
  "constraints": "sizeof(T)==2 alone permits other 16-bit types without matching numerical contract; caller must choose supported Cast pair"
}
```

Execution pipe:
```json
{
  "classification": "SIMD/AIV",
  "operations": "memory-vector Cast/Sub compatibility APIs through VF, then MTE3 UB->L1",
  "SIMT": false,
  "AIC": false,
  "MIX": "composition needs AIC consumer; leaf has no role branch or cross-core launch"
}
```

Geometry:
```json
{
  "actual": "8x64 FP32 dense per AIV, row_offset 0 or8 in 16x64 NZ hi+lo slot; extents rows1..16-row_offset, tokens positive multiple8, pitch multiple8 >=tokens",
  "proposed_M128_M256_D256": "Not an M128/M256 compute routine. Requires caller schedule/retiling: slot M is hardcoded16; D256 is not current probability width64. No proof for larger fusion geometry.",
  "validated_but_ignored": "src_pitch is stored but every native Cast/Sub operates dense rows*tokens; requires src_pitch==tokens in practice"
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB_bytes_actual_design": 8448,
  "UB_symbolic": "3*(rows*tokens*4)+2*(rows*tokens*2)+256 for dense buffers as header budget expression; physical nonoverlap/padding must be supplied by caller",
  "UB_comment_discrepancy": "lines156-160 claim8192; lines167-168 evaluate8448",
  "L1_bytes_actual_design": 4096,
  "L1_symbolic": "2 limbs*16*ceil(tokens/16)*16*2 bytes jointly across AIVs",
  "L0_bytes": 0,
  "VREG_bytes": null,
  "VREG_note": "256-byte architectural read pad named; register liveness/count not established"
}
```

Synchronization/lifetime contract:
```json
{
  "leaf": "two PipeBarrier<PIPE_V> in stage(), no HardEvent; high-plane copy occurs BEFORE low Cast overwrites same buffer",
  "caller": "publish probabilities; keep UB alive; free prior L1 after AIC MTE1 read; publish both AIV writes to AIC MTE1; allocate skew/pads",
  "gate": "PIPE_V is intra-pipe only. Source does not explicitly order Cast->MTE3 and MTE3->Cast shared-plane WAR. Compiler auto-sync/ISA artifact proof needed; no generated code inspected in inventory."
}
```

CAModel verification and limitations:
```json
{
  "current_pass": null,
  "historical_report": "C:/ascend-micro-harness/reports/gqa_stage_probs.md:314-315 reports compile PASS, run FAIL, Total tick3027, zero-filled planes",
  "test_sources": [
    "C:/ascend-micro-harness/kernels/camodel/gqa_stage_probs.cpp:204-210",
    "C:/ascend-micro-harness/tests/camodel/test_camodel_gqa_stage_probs.cpp:62"
  ],
  "artifact_proof": null,
  "note": "test source/report text is not independent PASS artifact; no build/run performed"
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_cv_gqa_stage_probs`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_cv_transfer_ub_to_l1.hpp

Path: `C:/ascend-micro-harness/routines/routine_cv_transfer_ub_to_l1.hpp`  
SHA256: `5c8307ef96e708f0604e8ba65f1552634cc990bc8cc03255c28c3fa1fdfd64ff`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 87:
```cpp
constexpr std::uint32_t kCvBlockBytes = 32;
```

Line 92:
```cpp
template <class T>
constexpr std::uint32_t kCvElemsPerBlock = kCvBlockBytes / sizeof(T);
```

Line 96:
```cpp
template <class T, std::uint32_t DValue>
constexpr std::uint32_t kCvStripCount =
    (DValue + kCvElemsPerBlock<T> - 1u) / kCvElemsPerBlock<T>;
```

Line 100:
```cpp
template <std::uint32_t NValue, std::uint32_t DValue, std::uint32_t SrcDValue,
          std::uint32_t DstNzC0Stride, std::uint32_t DstNzNStride, class Engine>
ASCEND_MICRO_CV_INLINE void cv_transfer_nd2nz(Engine& engine)
```

Line 110:
```cpp
template <class T>
struct NativeCvTransferEngine {
```

Line 123:
```cpp
    template <std::uint32_t NValue, std::uint32_t SrcDValue,
              std::uint32_t DstNzC0Stride, std::uint32_t DstNzNStride,
              std::uint32_t Strip>
    __aicore__ inline void emit_strip() const
```

Line 144:
```cpp
    template <std::uint32_t NValue, std::uint32_t SrcDValue,
              std::uint32_t DstNzC0Stride, std::uint32_t DstNzNStride,
              std::uint32_t... Strips>
    __aicore__ inline void emit_strips(
        std::integer_sequence<std::uint32_t, Strips...>) const
```

Line 154:
```cpp
    template <std::uint32_t NValue, std::uint32_t DValue,
              std::uint32_t SrcDValue, std::uint32_t DstNzC0Stride,
              std::uint32_t DstNzNStride>
    __aicore__ inline void place_nz() const
```

Supported dtypes; input/output distinctions retained:
```json
{
  "BF16": {
    "supported": true,
    "qualification": "generic same-type byte placement; documented hardware datatype support, no native instance validated in this review",
    "evidence": [
      {
        "path": "C:/ascend-micro-harness/routines/routine_cv_transfer_ub_to_l1.hpp",
        "line_start": 110,
        "line_end": 141
      },
      {
        "path": "C:/PipeSim/docs/ascendc_api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopy_UBToL1_highdim_split.md",
        "line_start": 81,
        "line_end": 81
      }
    ]
  },
  "INT8": {
    "supported": true,
    "qualification": "generic same-type byte placement; documented hardware datatype support, no native instance validated in this review",
    "evidence": [
      {
        "path": "C:/ascend-micro-harness/routines/routine_cv_transfer_ub_to_l1.hpp",
        "line_start": 110,
        "line_end": 141
      },
      {
        "path": "C:/PipeSim/docs/ascendc_api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopy_UBToL1_highdim_split.md",
        "line_start": 81,
        "line_end": 81
      }
    ]
  },
  "FP32": {
    "supported": true,
    "qualification": "generic same-type byte placement; documented hardware datatype support, no native instance validated in this review",
    "evidence": [
      {
        "path": "C:/ascend-micro-harness/routines/routine_cv_transfer_ub_to_l1.hpp",
        "line_start": 110,
        "line_end": 141
      },
      {
        "path": "C:/PipeSim/docs/ascendc_api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopy_UBToL1_highdim_split.md",
        "line_start": 81,
        "line_end": 81
      }
    ]
  }
}
```

Execution pipe:
```json
{
  "issuer": "AIV",
  "pipe": "MTE3",
  "path": "UB VECIN/VECCALC/VECOUT -> shared L1 A1/B1; same T, block-strided placement",
  "instruction": "copy_ubuf_to_cbuf via DataCopy(LocalTensor<T>,LocalTensor<T>,DataCopyParams)",
  "direct_path_condition": "Installed9.2.1 #if KFC_C310_SSBUF==1 || __MIX_CORE_AIC_RATION__!=1; otherwise KFC/GM relay requiring TPipe/KfcClient. Header does not itself assert ratio.",
  "evidence": [
    {
      "path": "/home/dmin/Ascend-9.2.1/cann-9.2.1/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h",
      "line_start": 184,
      "line_end": 193
    },
    {
      "path": "/home/dmin/Ascend-9.2.1/cann-9.2.1/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h",
      "line_start": 617,
      "line_end": 644
    }
  ]
}
```

Geometry:
```json
{
  "header_fixed_M_or_D": null,
  "current_documented_example": {
    "N_rows": 16,
    "D_columns": 16,
    "dtype": "FP16",
    "strip_count": 1,
    "bytes": 512
  },
  "symbolic": "C0=32/sizeof(T); strips=D/C0; Strip s source UB+s*32, destination L1+s*DstNzC0Stride*32; blockCount=N; blockLen=1; srcStride=SrcDValue/C0-1; dstStride=DstNzNStride-1",
  "proposed_M128_D256_tightly_packed": {
    "NValue": 128,
    "DValue": 256,
    "SrcDValue": 256,
    "DstNzC0Stride": 128,
    "DstNzNStride": 1,
    "BF16": {
      "strips_or_instructions": 16,
      "UB_bytes": 65536,
      "L1_bytes": 65536,
      "srcStride_blocks": 15
    },
    "INT8": {
      "strips_or_instructions": 8,
      "UB_bytes": 32768,
      "L1_bytes": 32768,
      "srcStride_blocks": 7
    },
    "FP32": {
      "strips_or_instructions": 32,
      "UB_bytes": 131072,
      "L1_bytes": 131072,
      "srcStride_blocks": 31
    },
    "validated": false
  },
  "layout_caveat": "This is a32-byte block-strided placement. The special one16x16 FP16 Nz(X)==Zn(X^T) byte identity is not a general M128/D256 layout proof; Cube operand format and dtype-specific fractal mapping must be separately derived.",
  "evidence": {
    "path": "C:/ascend-micro-harness/routines/routine_cv_transfer_ub_to_l1.hpp",
    "line_start": 53,
    "line_end": 79
  }
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "allocations": 0,
  "UB_symbolic_touched_extent_bytes": "((NValue-1)*SrcDValue + DValue)*sizeof(T), assumes DValue<=SrcDValue",
  "L1_symbolic_touched_extent_bytes": "((strips-1)*DstNzC0Stride+(NValue-1)*DstNzNStride+1)*32",
  "UB_allocation_bytes": null,
  "L1_allocation_bytes": null,
  "L0A_bytes": 0,
  "L0B_bytes": 0,
  "L0C_bytes": 0,
  "VREG_bytes": null,
  "VREG_note": "No Vector math in leaf; hardware direct path MTE3. No compiler register measurement."
}
```

Synchronization/lifetime contract:
```json
{
  "leaf_issues_events": false,
  "caller_required": [
    "V_MTE3 pair after vector production before UB read",
    "MTE3_V completion before rewriting UB source bank",
    "cross-core ready published from completed MTE3 before AIC MTE1 reads L1",
    "reverse L1 credit after AIC MTE1 reads complete before producer rewrites L1 bank",
    "both AIVs must set mode2 reverse quorum even when only one does the particular data move; symmetric scheduling needed to avoid deadlock",
    "depth2 requires independent bank/resource reverse credits and terminal drains including V_S for actual outstanding vector arithmetic and FIX_S on AIC FIX, not no-op finish callbacks"
  ],
  "evidence": {
    "path": "C:/ascend-micro-harness/routines/routine_cv_transfer_ub_to_l1.hpp",
    "line_start": 81,
    "line_end": 86
  },
  "this_leaf_depth2_or_drains": false
}
```

CAModel verification and limitations:
```json
{
  "status": "legacy composition report claims observed direct512B FP16 moves; no isolated current-sha /9.2.1 /BF16 /INT8 /FP32 /M128D256 proof",
  "report": {
    "path": "C:/ascend-micro-harness/reports/fused_attention_integration.md",
    "line_start": 12,
    "line_end": 67
  },
  "artifact_paths": [
    "C:/ascend-micro-harness/kernels/ops/turboquant/turboquant_paged_attention_opt.cpp",
    "C:/ascend-micro-harness/tests/camodel/test_camodel_fused_attention_3tiles.cpp",
    "C:/ascend-micro-harness/reports/fused_attention_cv_event_summary.txt"
  ],
  "artifact_text_not_read": "txt excluded from bounded md/json evidence search",
  "raw_artifact_verified": false,
  "exact_current_sha_bound_to_run": null,
  "report_geometry": "3 pages of16x16 FP16 staged by AIV1; AIV0 has distinct softmax role; not symmetric both-AIV role validation"
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_cv_transfer_ub_to_l1`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_fractal_load_zn_bf16.hpp

Path: `C:/ascend-micro-harness/routines/routine_fractal_load_zn_bf16.hpp`  
SHA256: `eec4cdd36459565d03da48bc2df2540654c9eadb5cfee743ec35b42fdf1f6d0c`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 72:
```cpp
template <std::uint32_t Tokens, std::uint32_t Group> __aicore__ inline void load_page_group_a_nz(std::uint32_t l1_byte_base, __ca__ bfloat16_t* dst)
```

Line 94:
```cpp
template <std::uint32_t Tokens, std::uint32_t Group> __aicore__ inline void load_page_group_b_zn(std::uint32_t l1_byte_base, __cb__ bfloat16_t* dst)
```

Line 124:
```cpp
template <std::uint32_t Dim> __aicore__ inline void load_block_a_nz(std::uint32_t l1_byte_base, __ca__ bfloat16_t* dst)
```

Line 140:
```cpp
template <std::uint32_t Dim> __aicore__ inline void load_block_b_zn(std::uint32_t l1_byte_base, __cb__ bfloat16_t* dst)
```

Supported dtypes; input/output distinctions retained:
```json
{
  "BF16": {
    "input": true,
    "output": true,
    "evidence_lines": [
      73,
      74,
      81,
      83,
      85,
      95,
      96,
      103,
      105,
      107,
      125,
      132,
      141,
      148
    ]
  },
  "INT8": {
    "input": false,
    "output": false
  },
  "FP32": {
    "input": false,
    "output": false
  }
}
```

Execution pipe:
```json
{
  "core": "AIC MTE1 leaves",
  "pipes": [
    "MTE1 L1->L0A or L0B"
  ],
  "SIMD": false,
  "SIMT": false,
  "MIX": "no kernel task declaration; caller may use AIC branch in MIX"
}
```

Geometry:
```json
{
  "page_constraints": "Tokens%16==0 and Group%16==0, lines77-79/99-101; no positivity or buffer-capacity assertion",
  "block_constraints": "Dim%16==0 and (Dim*Dim*2)%512==0, lines128-130/144-146",
  "tails": false,
  "M128_M256_D256": "templates admit positive Tokens128/256 and Group256 divisibility, but resource capacity and current-target behavior not established; no arithmetic transform performed"
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "allocation_owner": "caller; no local buffers",
  "symbolic_required_payload_bytes": {
    "UB": 0,
    "L1": "page2*Tokens*Group or block2*Dim*Dim",
    "L0A": "2*Tokens*Group (pageA) or2*Dim*Dim(blockA)",
    "L0B": "2*Tokens*Group (pageB) or2*Dim*Dim(blockB)",
    "L0C": 0,
    "VREG": 0
  },
  "page_group32_bytes": {
    "Tokens16": 1024,
    "Tokens128": 8192,
    "Tokens256": 16384
  },
  "page_group256_bytes": {
    "Tokens128": 65536,
    "Tokens256": 131072
  },
  "block_Dim32_bytes": 2048,
  "block_Dim256_bytes": 131072,
  "capacity_gate": "131072-byte monolithic BF16 operand must not be assumed legal L0A/B; installed target capacity not verified in this inventory"
}
```

Synchronization/lifetime contract:
```json
{
  "leaf": "dispatch only; no HardEvent",
  "caller": "MTE2_MTE1 before, MTE1_M after; operand overwrite/reuse lifetime and full source layout belong to caller",
  "evidence_lines": [
    51,
    52,
    53,
    54
  ]
}
```

CAModel verification and limitations:
```json
{
  "status": "historical report documents four BF16 layout cases PASS for Tokens16/Group32, Dim32; no M128/M256 result",
  "current_CANN9_2_1_result": null,
  "evidence": [
    {
      "path": "C:/ascend-micro-harness/reports/micro_layout_bf16.md",
      "line_start": 1,
      "line_end": 12,
      "note": "Ascend950PR_9599 arch3510, CANN9.2.0-beta.2; simulator only"
    },
    {
      "path": "C:/ascend-micro-harness/reports/micro_layout_bf16.md",
      "line_start": 82,
      "line_end": 107,
      "note": "one16x256 page/eight32x32 blocks; AIC-only no UB/ring; exact integer fixture"
    },
    {
      "path": "C:/ascend-micro-harness/reports/micro_layout_bf16.md",
      "line_start": 132,
      "line_end": 153,
      "note": "four PASS lines512/512 both orientations,zero error/fatal/address errors,canaries intact"
    }
  ],
  "raw_execution_artifact_paths": null,
  "test_presence_is_pass": false
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_fractal_load_zn_bf16`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_probe_l2_atomic_sync.hpp

Path: `C:/ascend-micro-harness/routines/routine_probe_l2_atomic_sync.hpp`  
SHA256: `ba92f6346ac00bc981dad29aefedecfbc42926bdf27b3b288b6d8fcba7717890`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 204:
```cpp
enum class ProbeMode:uint32_t {CollisionFree,SharedAtomic,TailEgress}
```

Line 294:
```cpp
__simt_vf__ __launch_bounds__(32) probe_lock_acquire_vf(__gm__ uint32_t* lock,*ticket,*ticket_log,*spin_census,uint32_t slot) [294-297]
```

Line 316:
```cpp
__simt_vf__ __launch_bounds__(32) probe_lock_release_vf(__gm__ uint32_t* lock)
```

Line 343:
```cpp
__simt_vf__ __launch_bounds__(32) probe_state_max_vf(__gm__ float* gm_m,__gm__ float* gm_tile_m)
```

Line 366:
```cpp
__simt_vf__ __launch_bounds__(32) splitk_lock_acquire_vf(__gm__ uint32_t* lock,*ticket,*ticket_log,*spin_census,*acquired,uint32_t slot,uint32_t max_attempts) [366-370]
```

Line 403:
```cpp
__simt_vf__ __launch_bounds__(32) splitk_lock_release_vf(__gm__ uint32_t* lock,__gm__ uint32_t* acquired,uint32_t slot) [403-405]
```

Line 424:
```cpp
struct L2AtomicProbeEngine
```

Line 467:
```cpp
void combine() const
```

Line 505:
```cpp
void fetch_record() const
```

Line 514:
```cpp
void store_record() const
```

Line 526:
```cpp
void store_tile() const
```

Supported dtypes; input/output distinctions retained:
```json
{
  "BF16": false,
  "INT8": false,
  "FP32": "SIMD state/acc combine; SIMT lane0 scalar atomic max of m.",
  "control": "uint32 lock,ticket,spin/acquired counters"
}
```

Execution pipe:
```json
{
  "actual": "AIV hybrid SIMD vector combine/MTE2/MTE3 plus SIMT lane0 GM control",
  "SIMD": true,
  "SIMT": true,
  "AIC": false,
  "MIX": false,
  "note": "SIMD/SIMT hybrid is not AIC/AIV MIX. Installed asc_vf_call gated by SPLIT_CORE_VEC."
}
```

Geometry:
```json
{
  "D": 256,
  "M": 1,
  "state_lanes": 8,
  "record_elements": 272,
  "record_bytes": 1088,
  "metric_bytes_per_split": 768,
  "max_probe_splits": 8,
  "max_hardware_splits": 16,
  "max_reps": 8,
  "SIMT_threads": 32,
  "active_SIMT_lanes": 1,
  "proposed_M128": "No128-row implementation; workspace record per head/split.",
  "proposed_M256": "No256-row implementation; fixed D256 is not M256."
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB_owned_bytes": 0,
  "UB_symbolic": "Core combine logical disjoint views3*(64+4D)+4*32+4*(4D)=320+28D=7488B atD256, excluding skew/readpad; pair aliases m/l.",
  "full_wrapper_UB_bytes": null,
  "L2_GM_record_bytes": 1088,
  "lock_bytes": 4,
  "VREG_physical_bytes": null
}
```

Synchronization/lifetime contract:
```json
{
  "owner": "Caller launches VFs, owns pipe edges; combine has internal PIPE_V barriers.",
  "edges": [
    "V->MTE2 acquire retirement before fetch",
    "MTE2->V fetch ready",
    "V->MTE3 combine ready",
    "MTE3->V store retired before unlock VF"
  ],
  "lifetimes": "Current/tile/new distinct; m/l contiguous64B; acc256FP32. Alpha Duplicate source256B addressable. Lock spans fetch/combine/store.",
  "release": "asc_threadfence does not retire MTE3. Bounded splitk release must use same acquired[slot]; max_attempts0 unbounded.",
  "cache": "All SIMT pointers GM; asc_ldcg/stcg L2 bypass per-core DataCache."
}
```

CAModel verification and limitations:
```json
{
  "status": "Historical report and runner presence; report has failing rung at586; no blanket/current hash PASS.",
  "evidence": [
    {
      "path": "C:/ascend-micro-harness/reports/l2_atomic_contention_split_k.md",
      "lines": [
        16,
        17,
        20,
        586
      ],
      "claim": "Historical report including FAIL rung"
    },
    {
      "path": "C:/ascend-micro-harness/tests/camodel/test_camodel_l2_atomic_synthetic.cpp",
      "lines": [
        64,
        69,
        74
      ],
      "claim": "Mode A/B/C runner presence"
    }
  ]
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_probe_l2_atomic_sync`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_stage_gqa_load_kv_chunk.hpp

Path: `C:/ascend-micro-harness/routines/routine_stage_gqa_load_kv_chunk.hpp`  
SHA256: `de43308273658a45b446b75c5b93b7ea1029ede97346f5b64fada8f31cec2269`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line [143, 144]:
```cpp
{"lines": [143, 144], "exact": "template <class T>\nconstexpr std::uint32_t kLoadKvC0 = 32u / sizeof(T);"}
```

Line [147, 147]:
```cpp
{"lines": [147, 147], "exact": "constexpr std::uint32_t LoadKvAlignUp(std::uint32_t value, std::uint32_t unit)"}
```

Line [153, 155]:
```cpp
{"lines": [153, 155], "exact": "constexpr std::uint32_t LoadKvSlotElements(std::uint32_t head_dim,\n                                           std::uint32_t rows_aligned,\n                                           std::uint32_t c0)"}
```

Line [163, 165]:
```cpp
{"lines": [163, 165], "exact": "constexpr std::uint32_t LoadKvSlotIndex(std::uint32_t r, std::uint32_t c,\n                                        std::uint32_t rows_aligned,\n                                        std::uint32_t c0)"}
```

Line [195, 196]:
```cpp
{"lines": [195, 196], "exact": "template <class T, std::uint32_t kCols, std::uint32_t kRowsAligned>\nconstexpr std::uint32_t kLoadKvSlotElements = kCols * kCols / kLoadKvC0<T> * kRowsAligned / 16u;"}
```

Line [213, 218]:
```cpp
{"lines": [213, 218], "exact": "template <class T>\nASCEND_MICRO_LKVC_INLINE std::int32_t gqa_load_kv_chunk_extents(\n    std::uint32_t b, std::uint32_t kh, std::uint32_t token0, std::uint32_t tokens,\n    std::uint32_t head_dim, std::uint32_t nkv_stride,\n    std::uint32_t batch_count, std::uint32_t kv_tokens, std::uint32_t kv_heads,\n    LoadKvChunkExtents& out)"}
```

Line [257, 260]:
```cpp
{"lines": [257, 260], "exact": "template <class T, class Engine>\nASCEND_MICRO_LKVC_INLINE std::int32_t load_kv_chunk(Engine& engine, std::uint64_t gm_base,\n                                                    std::uint32_t slot_base,\n                                                    const LoadKvChunkExtents& extents)"}
```

Line [297, 298]:
```cpp
{"lines": [297, 298], "exact": "template <class T>\nstruct NativeLoadKvChunkEngine {"}
```

Line [299, 301]:
```cpp
{"lines": [299, 301], "exact": "    template <class U>\n    __aicore__ inline void stage_chunk(std::uint64_t gm_base, std::uint32_t slot_base,\n                                       const LoadKvChunkExtents& extents) const"}
```

Supported dtypes; input/output distinctions retained:
```json
{
  "BF16": "supported hardware transfer, native sizeof(U)==2",
  "FP16": "supported hardware transfer, source wrapper uses half",
  "INT8": "native rejected sizeof(U)==2 although hardware API supports int8",
  "FP32": "native rejected sizeof(U)==2 although hardware API supports float",
  "constraints": "extents mapper itself has no sizeof(T)==2 guard; native T class parameter unused, stage U drives actual dtype"
}
```

Execution pipe:
```json
{
  "classification": "AIC/MTE2",
  "SIMD": false,
  "SIMT": false,
  "MIX": "leaf no cross-core sync; historical wrapper is1:2 MIX for fill/readback"
}
```

Geometry:
```json
{
  "actual": "D256, tokens64; ND GM [B,Skv,Nkv,D] pitched Nkv*D -> NZ ceil(tokens/16)*16 x ceil(D/16)*16",
  "proposed_M128_M256_D256": "Descriptor mapping admits tokens128/256,D256; respective one K or V slots64KiB/128KiB. This is a footprint calculation, not evidence of composition/feasibility or tested geometry."
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB_bytes": 0,
  "L1_bytes_actual_design": 32768,
  "L1_symbolic": "alignUp(tokens,16)*alignUp(head_dim,16)*2 per K OR V slot; rings and simultaneous K/V multiply this",
  "L0_bytes": 0,
  "VREG_bytes": 0,
  "native_tensor_view_elements_bug": "lines309-311: rows_aligned*kLoadKvSlotElements<U,256,16> = rows_aligned*4096; actual D256 slot is rows_aligned*256, so declared LocalTensor view16x actual. Descriptor transfer extent does not itself multiply by view size; do not report as observed16x write."
}
```

Synchronization/lifetime contract:
```json
{
  "leaf": "one MTE2 DataCopy; tokens0 returns0 and emits nothing; no flags/barriers/scalar data reads",
  "caller": "wait MTE1->MTE2 slot free; MTE2_MTE1 publish before L1 load; cross-core kv_full counter and lifetime; GM source live until DMA completion"
}
```

CAModel verification and limitations:
```json
{
  "current_pass": null,
  "historical_report": "C:/ascend-micro-harness/reports/gqa_load_kv_chunk.md:185-198 and header20-22 report build/link but model hang before image comparison",
  "test_sources": [
    "C:/ascend-micro-harness/kernels/camodel/gqa_load_kv_chunk.cpp:226",
    "C:/ascend-micro-harness/tests/camodel/test_camodel_gqa_load_kv_chunk.cpp:118"
  ],
  "artifact_proof": null,
  "note": "No executed PASS artifact established by bounded search; no run performed"
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_stage_gqa_load_kv_chunk`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_stage_page_zn_bf16.hpp

Path: `C:/ascend-micro-harness/routines/routine_stage_page_zn_bf16.hpp`  
SHA256: `0f00f201885b9a5d59167b1d2647baa177309ad02965460473ab10ccf2c51afb`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line [71, 72]:
```cpp
{"lines": [71, 72], "exact": "constexpr std::uint32_t Bf16NzBlockIndex(std::uint32_t dim, std::uint32_t r,\n                                         std::uint32_t c)"}
```

Line [91, 94]:
```cpp
{"lines": [91, 94], "exact": "template <std::uint32_t Tokens, std::uint32_t Channels>\n__aicore__ inline void stage_page_nz(__gm__ const bfloat16_t* src,\n                                     std::uint32_t l1_byte_base,\n                                     std::uint32_t row_pitch)"}
```

Line [124, 126]:
```cpp
{"lines": [124, 126], "exact": "template <std::uint32_t Dim, std::uint32_t B>\n__aicore__ inline void StageBlockNz(__gm__ const bfloat16_t* src,\n                                    std::uint32_t l1_byte_base)"}
```

Line [139, 142]:
```cpp
{"lines": [139, 142], "exact": "template <std::uint32_t Dim, std::uint32_t... Bs>\n__aicore__ inline void StageBlocksNzEmit(__gm__ const bfloat16_t* src,\n                                         std::uint32_t l1_byte_base,\n                                         std::integer_sequence<std::uint32_t, Bs...>)"}
```

Line [147, 149]:
```cpp
{"lines": [147, 149], "exact": "template <std::uint32_t Blocks, std::uint32_t Dim>\n__aicore__ inline void stage_blocks_nz(__gm__ const bfloat16_t* src,\n                                       std::uint32_t l1_byte_base)"}
```

Line [161, 163]:
```cpp
{"lines": [161, 163], "exact": "template <std::uint32_t Blocks, std::uint32_t Dim>\n__aicore__ inline void stage_blocks_flat(__gm__ const bfloat16_t* src,\n                                         std::uint32_t l1_byte_base)"}
```

Supported dtypes; input/output distinctions retained:
```json
{
  "BF16": "all device functions hardcoded __gm__ const bfloat16_t* and cbuf BF16",
  "INT8": "not implemented; comments explain existing8bit analogy only",
  "FP32": "not implemented",
  "constraints": "Tokens and Channels multiples16, Dim multiple16 and Dim*Dim*2 multiple512; positive sizes, bounds, L1 base alignment and row_pitch>=Channels not checked"
}
```

Execution pipe:
```json
{
  "classification": "AIC/MTE2 TensorAPI GM2L1",
  "SIMD": false,
  "SIMT": false,
  "MIX": "caller composition only"
}
```

Geometry:
```json
{
  "actual": "page16x256; resident Blocks x32x32 BF16 rotations, one ND2NZ descriptor per block; flat routine is negative-control alternative",
  "proposed_M128_M256_D256": "stage_page_nz<128,256>/<256,256> template shapes satisfy static asserts and occupy64KiB/128KiB L1. Critical page NZ-as-group-ZN byte identity proof is ONLY M1=1 (Tokens16); cannot reuse same group-G L1/L0 descriptors unchanged for128/256."
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB_bytes": 0,
  "L1_bytes_actual_page": 8192,
  "L1_page_symbolic": "Tokens*Channels*2",
  "L1_blocks_symbolic": "Blocks*Dim*Dim*2 (Dim32 =>2048/rotation block)",
  "L0_bytes": 0,
  "VREG_bytes": 0,
  "note": "No storage allocated by leaf; supplied address windows, total coexistence/ring budget unknown"
}
```

Synchronization/lifetime contract:
```json
{
  "leaf": "no events/barriers; asynchronous GM2L1 Copy, compile-time fold emits Blocks copies",
  "caller": "publish MTE2_MTE1 before use; prevent MTE2 overwrite while MTE1 reads; keep GM table/page alive; own ring/reset schedules"
}
```

CAModel verification and limitations:
```json
{
  "current_pass": null,
  "historical_report": "C:/ascend-micro-harness/reports/micro_layout_bf16.md:132-161 claims four cases512/512 both orientations, Total tick14190; lines229-230 report production stage_blocks_nz bit-exact at1/2/3 strips.",
  "test_sources": [
    "C:/ascend-micro-harness/kernels/camodel/micro_layout_bf16.cpp:207",
    "C:/ascend-micro-harness/tests/camodel/test_micro_layout_bf16.cpp:51"
  ],
  "artifact_proof": null,
  "note": "Historical report prose/stdout excerpt is not independently located log/trace with current header hash. No current execution, M128/M256 or full Pi proof established here."
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_stage_page_zn_bf16`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_turboquant_dequant4.hpp

Path: `C:/ascend-micro-harness/routines/routine_turboquant_dequant4.hpp`  
SHA256: `73de13987a321a45171207acd87d34b8b0e2e4fe04a879f02ed1f470692a8958`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 24:
```cpp
template<class Engine> void turboquant_dequant4(Engine&,uint32_t count) [24-25]
```

Line 62:
```cpp
template<class Engine> void turboquant_dequant4_signed(Engine&,uint32_t count) [62-64]
```

Line 71:
```cpp
template<class Output> struct NativeDequant4Engine [71-79]
```

Line 81:
```cpp
void cast_packed_nibbles(uint32_t count)
```

Line 88:
```cpp
void mask_unsigned_nibbles(uint32_t count)
```

Line 98:
```cpp
void cast_signed(uint32_t count)
```

Line 111:
```cpp
void cast_and_scale(uint32_t count)
```

Line 124:
```cpp
template<class Output> void turboquant_dequant4_native(NativeDequant4Engine<Output>&,uint32_t) [124-126]
```

Supported dtypes; input/output distinctions retained:
```json
{
  "input": "int8_t byte storage reinterpreted int4b_t; not numeric INT8.",
  "BF16": "Signed direct int4b_t->BF16 supported; unsigned int16->BF16 scale path unsupported (absent Cast table).",
  "INT8": "No INT8 arithmetic/output contract.",
  "FP32": "Signed/unsigned output via int16->float supported.",
  "FP16": "Signed and unsigned paths supported.",
  "template_hazard": "sizeof(Output)==4 is not an explicit FP32 type constraint."
}
```

Execution pipe:
```json
{
  "actual": "AIV SIMD compatibility API",
  "SIMD": true,
  "SIMT": false,
  "AIC": false,
  "MIX": false
}
```

Geometry:
```json
{
  "D": null,
  "M": 1,
  "count_multiple": 64,
  "current_native_count": 64,
  "proposed_M128": "Caller loops128 rows with per-row scale/lifecycle; signed no-scale may flatten.",
  "proposed_M256": "Caller loops256 rows; no batched contract in leaf."
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB_owned_bytes": 0,
  "UB_unsigned_symbolic": "N/2+6N+2N*sizeof(Output) logical bytes; skew/alignment/read pad excluded.",
  "UB_signed_symbolic": "N/2+N*sizeof(Output)+(sizeof(Output)==4 ? 2N:0).",
  "native_runner_arena_bytes": 1088,
  "VREG_physical_bytes": null
}
```

Synchronization/lifetime contract:
```json
{
  "owner": "Caller-owned completion outside asynchronous leaf (17-19); no events in leaf.",
  "lifetimes": "packed through cast; intermediate planes through their consumers; output through MTE3 retirement.",
  "wrapper_events": "MTE2_V0,V_MTE3 0,MTE3_S0"
}
```

CAModel verification and limitations:
```json
{
  "status": "Historical unsigned FP32 native report and test presence only; signed BF16 standalone proof absent.",
  "evidence": [
    {
      "path": "C:/ascend-micro-harness/reports/micro_routines_validation.md",
      "lines": [
        46,
        68
      ],
      "claim": "Historical native64 exact and128guardbytes; not current hash PASS"
    },
    {
      "path": "C:/ascend-micro-harness/tests/camodel/test_camodel_dequant4.cpp",
      "lines": [
        7,
        36
      ],
      "claim": "Runner presence"
    }
  ]
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_turboquant_dequant4`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_vector_cube_softmax.hpp

Path: `C:/ascend-micro-harness/routines/routine_vector_cube_softmax.hpp`  
SHA256: `8bb3b0e38bc5264f70bf72bfa50986a2d24f37c8590bfac7fe9d433d5827623d`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 23:
```cpp
template <AscendC::ReduceType Kind> __aicore__ inline void reduce(const AscendC::LocalTensor<float>& dst)
```

Line 35:
```cpp
__aicore__ inline void update_max_and_alpha()
```

Line 46:
```cpp
__aicore__ inline void exponentiate_scores()
```

Line 53:
```cpp
__aicore__ inline void update_denominator()
```

Line 60:
```cpp
__aicore__ inline void quantize_probabilities()
```

Line 76:
```cpp
__aicore__ inline void vector_cube_scale_rows( const AscendC::LocalTensor<float>& rows, const AscendC::LocalTensor<float>& scale, std::uint32_t heads, std::uint32_t channels, const AscendC::BinaryRepeatParams& repeat)
```

Line 85:
```cpp
__aicore__ inline void vector_cube_accumulate( const AscendC::LocalTensor<float>& acc, const AscendC::LocalTensor<float>& context, const AscendC::LocalTensor<float>& probability_scale, const AscendC::LocalTensor<float>& inverse, std::uint32_t heads, std::uint32_t channels, const AscendC::BinaryRepeatParams& repeat)
```

Supported dtypes; input/output distinctions retained:
```json
{
  "FP32": "actual LocalTensor<float> arithmetic and recurrence",
  "BF16": "not an arithmetic engine dtype; cube Operand template alone does not prove usable codec instantiation",
  "INT8": "not an arithmetic engine dtype; uint8_t GQA keep is predicate storage, not INT8 data support",
  "descriptor": null,
  "descriptor_evidence": "routines.yaml not present in assigned C:/ascend-micro-harness root; no descriptor support inferred"
}
```

Execution pipe:
```json
AIV Vector / compatibility APIs lowering to VF on dav_3510; caller provides UB; no Cube arithmetic in these headers
```

Geometry:
```json
{
  "current": "H=heads <=8 by header comment, K=64, D=channels in scale/accumulate; D loop always handles64 lanes, requires D multiple64 or caller padding; valid<64 selects supplied mask",
  "M128_D256": "requires caller subdivide M into H<=8 groups and retain64-key reduction geometry; no M128 specialization",
  "M256_D256": "requires caller subdivide M into H<=8 groups; no M256 specialization",
  "guards": "no runtime guard for heads==0/>8, valid>64, channels%64; caller contract"
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB": {
    "actual_bytes": null,
    "allocation_owner": "caller",
    "symbolic": "Caller-specified symbolic extents: scores H*64 FP32; key_scale/value_scale/mask at least64 FP32 with by_vector strides caller-defined; q_inv and seven fields run_max,run_sum,tile_max,new_max,alpha,prob_scale,part each typically H*8 FP32; row_reduce at least8 FP32; probabilities H*64*sizeof(Operand). Exact scale strides, storage sizes and aliasing require caller mapping; no exact sum asserted.",
    "minimum_operation_extents": "each Brcb dst>=64 FP32=256 bytes even H<8; row_reduce>=8 FP32 read despite reduction producing H lanes; row scales>=8*H floats"
  },
  "L1": {
    "actual_bytes": 0,
    "reason": "no L1 allocation/access in leaf"
  },
  "L0": {
    "actual_bytes": 0,
    "reason": "no L0 allocation/access in leaf"
  },
  "VREG": {
    "actual_bytes": null,
    "symbolic": "compiler/lowering owned; 256-byte vector width, no persistent RegTensor interface"
  }
}
```

Synchronization/lifetime contract:
```json
{
  "leaf": "PipeBarrier<PIPE_V> only; no explicit SetFlag/WaitFlag/TPipe or DMA in leaf",
  "caller": "input producer readiness, output consumer visibility, allocation/aliasing/lifetime and inter-core synchronization"
}
```

CAModel verification and limitations:
```json
{
  "current_hash_verified_pass": null,
  "status": "unverified_current_source; inventory-only, no runs",
  "evidence_paths": [
    "C:/ascend-micro-harness/reports/fused_decode_opt.md:55-66",
    "C:/ascend-micro-harness/reports/fused_decode_opt_20261103/compiled-source-sha256.txt:13-14"
  ],
  "evidence_note": "report claims historical passes; archived cube hash d5cd1ccb... and online hash8667b60b... differ from current assigned headers; not current PASS"
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_vector_cube_softmax`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_vector_decode_gemv.hpp

Path: `C:/ascend-micro-harness/routines/routine_vector_decode_gemv.hpp`  
SHA256: `7b1c9625ac230ddb31fd9b8c38586454b3c3dc1ded214fedfb086c7811486cfb`  
Namespace: `{'name': 'ascend_micro', 'line': 92}`

Exact public/template declarations:

Line 170:
```cpp
__aicore__ inline void select_row(std::uint32_t row)
```

Line 183:
```cpp
__aicore__ inline void multiply() const
```

Line 202:
```cpp
__aicore__ inline void reduce_quarters() const
```

Line 218:
```cpp
__aicore__ inline void fold_quarters() const
```

Line 232:
```cpp
template <class Engine> __aicore__ inline void vector_decode_gemv_row(Engine& engine)
```

Line 245:
```cpp
template <class Engine> __aicore__ inline void vector_decode_gemv_strip(Engine& engine, std::uint32_t rows)
```

Supported dtypes; input/output distinctions retained:
```json
{
  "FP32": "All math inputs/output/state/scratch LocalTensor<float>; no dtype template. Engine wrapper templates accept duck-typed Engine, do not widen native engine dtype support.",
  "BF16": {
    "native_supported": false,
    "conversion_in_leaf": false,
    "caller_requirement": "Convert query/pages to FP32 externally"
  },
  "INT8": {
    "native_supported": false,
    "conversion_in_leaf": false,
    "note": "uint8 tail_sel is predicate storage, not INT8 compute"
  }
}
```

Execution pipe:
```json
{
  "declared_target": "Ascend950PR arch35 dav-3510 CANN9.2 comments",
  "actual_pipeline": "AIV PIPE_V memory-vector APIs; runtime scalar loop/address arithmetic; installed implementation lowers memory APIs to VF Reg loads/stores",
  "L1_bytes": 0,
  "L0A_bytes": 0,
  "L0B_bytes": 0,
  "L0C_bytes": 0,
  "explicit_VREG_bytes": 0,
  "physical_VREG_peak_bytes": null,
  "physical_VREG_note": "No explicit RegTensor in leaf; internal memory-vector lowering uses transient registers; allocation/spills need generated artifact, forbidden in this phase."
}
```

Geometry:
```json
{
  "actual": {
    "D": 256,
    "strip_keys": 16,
    "rows": "runtime rows, caller commonly6",
    "query_layout": "rows x256 row-major",
    "key_layout": "16x256 token-major FP32",
    "panel_layout": "rows x16 contiguous FP32",
    "quarters": 4,
    "lanes_per_repeat": 64
  },
  "proposed": {
    "D": 256,
    "M": null,
    "note": "Header documents rejected 1Mul+ReduceRepeat(mask256) sketch; implemented4Mul+1ReduceRepeat+2Add."
  },
  "constraints": "No runtime row bounds checks; tensor capacities/base32-byte alignment caller-owned. Constants not templated."
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "allocation_bytes": null,
  "UB_logical_bytes_symbolic": "33152 + 1088*rows (includes key16384, query1024R, products16384, partials256, fold128, panel64R)",
  "UB_logical_bytes_rows6": 39680,
  "scratch_bytes": {
    "products": 16384,
    "partials": 256,
    "fold": 128
  },
  "source_claimed_full_register_read_envelope": {
    "partials_min_bytes": 384,
    "fold_min_bytes": 320,
    "derivation": "partials[32] offset128+256; fold[16] offset64+256. This is source conservative read-pad contract, not measured9.2.1 read width."
  },
  "aliases": "query_row/panel_row are views, not extra storage; fold must not alias partials."
}
```

Synchronization/lifetime contract:
```json
{
  "barrier_lines": [
    221,
    236,
    238,
    251
  ],
  "hard_events": [],
  "lifetime": "key_page reused all rows; products->partials->fold->panel; per-row scratch reused only after strip loop PIPE_V barrier; outputs remain caller-owned until consumed. Ingress ready/egress completion/refill events external."
}
```

CAModel verification and limitations:
```json
{
  "new_runs": 0,
  "installed_9_2_1_proof": null,
  "exact_current_header_hash_artifact_binding": null,
  "raw_result_or_trace_artifacts_verified": [],
  "status": "Historical report evidence only, no exact-hash9.2.1 CAModel proof established. Test existence is not execution proof.",
  "reports": [
    {
      "path": "C:/ascend-micro-harness/reports/single_token_decode_aiv.md",
      "lines": [
        337,
        352
      ],
      "claim": "CANN9.2.0-beta.2 historical six rows/all8strips/H0,H2,H3 one-block isolated launch, score0/768 wrong, trace17checks PASS."
    },
    {
      "path": "C:/ascend-micro-harness/reports/decode_tail_masking.md",
      "lines": [
        164,
        169,
        282
      ],
      "claim": "Historical S1/S17 CAModel numeric pass but residual errors/NOT CLEAN; this does not establish full instruction-trace cleanliness."
    }
  ]
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_vector_decode_gemv`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_vector_decode_recurrence.hpp

Path: `C:/ascend-micro-harness/routines/routine_vector_decode_recurrence.hpp`  
SHA256: `5e04efb080ba21c533d947fb16dab71b579466d6fbd370a998f775623415bac7`  
Namespace: `{'name': 'ascend_micro', 'line': 71}`

Exact public/template declarations:

Line 182:
```cpp
__aicore__ inline void select_query(std::uint32_t query)
```

Line 200:
```cpp
__aicore__ inline void scale_row()
```

Line 212:
```cpp
__aicore__ inline void prime_tail_mask(float masked_score)
```

Line 226:
```cpp
__aicore__ inline void select_tail(std::uint32_t active)
```

Line 252:
```cpp
__aicore__ inline void mask_tail_scales()
```

Line 281:
```cpp
__aicore__ inline void mask_tail_scores()
```

Line 301:
```cpp
__aicore__ inline void clamp_tail_probabilities()
```

Line 307:
```cpp
__aicore__ inline void reduce_max_and_alpha()
```

Line 323:
```cpp
__aicore__ inline void update_max_and_alpha()
```

Line 333:
```cpp
__aicore__ inline void update_max_and_alpha_masked()
```

Line 345:
```cpp
__aicore__ inline void exponentiate_scores()
```

Line 354:
```cpp
__aicore__ inline void update_denominator()
```

Line 373:
```cpp
__aicore__ inline void fold_value_scale()
```

Line 378:
```cpp
__aicore__ inline void contract_values()
```

Line 424:
```cpp
__aicore__ inline void update_accumulator()
```

Line 433:
```cpp
__aicore__ inline void update_accumulator_masked()
```

Line 444:
```cpp
__aicore__ inline void commit_state()
```

Line 453:
```cpp
template <class Engine> __aicore__ inline void vector_decode_recur_query(Engine& engine)
```

Line 466:
```cpp
template <class Engine> __aicore__ inline void vector_decode_recur_strip(Engine& engine)
```

Line 492:
```cpp
template <class Engine> __aicore__ inline void vector_decode_recur_rows(Engine& engine, std::uint32_t rows)
```

Line 514:
```cpp
template <class Engine> __aicore__ inline void vector_decode_recur_query_masked(Engine& engine)
```

Line 531:
```cpp
template <class Engine> __aicore__ inline void vector_decode_recur_rows_masked(Engine& engine, std::uint32_t rows, std::uint32_t active)
```

Line 554:
```cpp
template <class Engine> __aicore__ inline void vector_decode_recur_strip_masked(Engine& engine, std::uint32_t active)
```

Supported dtypes; input/output distinctions retained:
```json
{
  "FP32": "All math inputs/output/state/scratch LocalTensor<float>; no dtype template. Engine wrapper templates accept duck-typed Engine, do not widen native engine dtype support.",
  "BF16": {
    "native_supported": false,
    "conversion_in_leaf": false,
    "caller_requirement": "Convert query/pages to FP32 externally"
  },
  "INT8": {
    "native_supported": false,
    "conversion_in_leaf": false,
    "note": "uint8 tail_sel is predicate storage, not INT8 compute"
  }
}
```

Execution pipe:
```json
{
  "declared_target": "Ascend950PR arch35 dav-3510 CANN9.2 comments",
  "actual_pipeline": "AIV PIPE_V memory-vector APIs; runtime scalar loop/address arithmetic; installed implementation lowers memory APIs to VF Reg loads/stores",
  "L1_bytes": 0,
  "L0A_bytes": 0,
  "L0B_bytes": 0,
  "L0C_bytes": 0,
  "explicit_VREG_bytes": 0,
  "physical_VREG_peak_bytes": null,
  "physical_VREG_note": "No explicit RegTensor in leaf; internal memory-vector lowering uses transient registers; allocation/spills need generated artifact, forbidden in this phase."
}
```

Geometry:
```json
{
  "actual": {
    "D": 256,
    "strip_keys": 16,
    "rows": "vector_decode_recur_strip frozen16; vector_decode_recur_rows runtime; masked variants same",
    "panel_layout": "rows x16",
    "value_layout": "256channels x16keys channel-major",
    "state_layout": "m,l replicated8 FP32/query; acc256/query"
  },
  "proposed": {
    "D": 256,
    "M": null,
    "note": "No newly proposed M specialization established from source. Runtime rows supports6*(1+k) if caller provides capacity."
  },
  "constraints": "Masked active must1..15, full16 uses unmasked, active0 must bypass externally; no runtime assertion. masked_score must suppress dead Exp contribution for actual finite score range. Buffers/read pads caller-owned."
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "allocation_bytes": null,
  "UB_logical_bytes_symbolic": "38946 + 1152*rows, including logical2byte tail predicate; excludes backing alignment/pads and aliases",
  "UB_logical_bytes_rows16": 57378,
  "UB_logical_bytes_rows6": 45858,
  "components": {
    "panel_bytes": "64*rows",
    "m_l_acc_bytes": "1088*rows",
    "scale_bytes": 128,
    "value_page_bytes": 16384,
    "per_query_scratch_excluding_tail_bytes": 22240,
    "tail_logical_bytes": 194
  },
  "scratch_large": {
    "weighted_bytes": 16384,
    "pair_partials_bytes": 2048,
    "weighted_sum_bytes": 1024,
    "acc_scaled_bytes": 1024,
    "p_x4_bytes": 256
  },
  "padding": "Header requires64lane outputs for Brcb and conservative256-byte backing for tail constant vectors/state last-row reads; precise caller allocation cannot be derived from this header. uint8 predicate backing alignment not logical2byte allocation.",
  "aliases": "panel_row,m_cur,l_cur,acc_cur are views; no allocation in engine."
}
```

Synchronization/lifetime contract:
```json
{
  "barrier_lines": [
    315,
    317,
    326,
    338,
    340,
    348,
    359,
    361,
    384,
    389,
    408,
    416,
    420,
    427,
    436,
    438,
    537,
    543,
    559,
    561
  ],
  "hard_events": [],
  "lifetime": "per-strip panel/scales/value shared across query loop; scales mutated once by masked entry and must not be refilled until last query read. m/l/acc carried across strips; query scratch reused sequentially. No caller DMA/refill or cross-core proof inside leaf. Consecutive dependent memory-vector ops sometimes have no explicit PIPE_V barrier in source; installed VF-local semantics/compiler ordering must be assessed from actual emitted artifact before any refactor."
}
```

CAModel verification and limitations:
```json
{
  "new_runs": 0,
  "installed_9_2_1_proof": null,
  "exact_current_header_hash_artifact_binding": null,
  "raw_result_or_trace_artifacts_verified": [],
  "status": "Historical report evidence only, no exact-hash9.2.1 CAModel proof established. Test existence is not execution proof.",
  "reports": [
    {
      "path": "C:/ascend-micro-harness/reports/single_token_decode_aiv.md",
      "lines": [
        337,
        352
      ],
      "claim": "CANN9.2.0-beta.2 historical six rows/all8strips/H0,H2,H3 one-block isolated launch, score0/768 wrong, trace17checks PASS."
    },
    {
      "path": "C:/ascend-micro-harness/reports/decode_tail_masking.md",
      "lines": [
        164,
        169,
        282
      ],
      "claim": "Historical S1/S17 CAModel numeric pass but residual errors/NOT CLEAN; this does not establish full instruction-trace cleanliness."
    }
  ]
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_vector_decode_recurrence`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_vector_egress_rotate.hpp

Path: `C:/ascend-micro-harness/routines/routine_vector_egress_rotate.hpp`  
SHA256: `a973949328f58f0c08440ffab57f113f9a9a568a003137f76322016a9f68e4b9`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 90:
```cpp
template<uint32_t S> constexpr uint32_t kEgressLanesPerRepeat
```

Line 92:
```cpp
template<uint32_t S> constexpr uint32_t kEgressRepeats
```

Line 95:
```cpp
template<uint32_t S> constexpr uint32_t kEgressRepStrideBlocks
```

Line 107:
```cpp
struct NativeEgressRotateEngine
```

Line 158:
```cpp
__aicore__ inline void NativeEgressRotateEngine::select_query(uint32_t query)
```

Line 186:
```cpp
__aicore__ inline void NativeEgressRotateEngine::normalise()
```

Line 200:
```cpp
template<uint32_t S> __aicore__ inline void NativeEgressRotateEngine::block_stage(const AscendC::LocalTensor<float>& dst,const AscendC::LocalTensor<float>& src) const
```

Line 216:
```cpp
template<uint32_t Index> __aicore__ inline void NativeEgressRotateEngine::gather_stage(const AscendC::LocalTensor<float>& dst,const AscendC::LocalTensor<float>& src) const
```

Line 229:
```cpp
__aicore__ inline void NativeEgressRotateEngine::rotate()
```

Line 264:
```cpp
__aicore__ inline void NativeEgressRotateEngine::gate_and_cast()
```

Line 283:
```cpp
template<class Engine> __aicore__ inline void vector_egress_query(Engine& engine)
```

Supported dtypes; input/output distinctions retained:
```json
{
  "actual": {
    "acc_l_gate_sign_scratch_rotated": "FP32",
    "gather_offsets": "UINT32 byte offsets",
    "context": "BF16 CAST_RINT"
  },
  "BF16": "output only",
  "INT8": "unsupported; no INT8 path",
  "FP32": "computation/input and optional rotated output",
  "descriptor": "no separate matching generic descriptor in this header; strongly typed fields 109-135"
}
```

Execution pipe:
```json
{
  "leaf": "AIV memory-vector SIMD PIPE_V",
  "SIMT": false,
  "AIC": false,
  "MIX": "surrounding paged-decode operator is MIX_AIC_1_2; leaf itself no Cube"
}
```

Geometry:
```json
{
  "D_actual": 256,
  "M_actual": "one query per entry; caller normally loops M=16",
  "current_128_support": false,
  "current_256_support": true,
  "proposed_128_256": "D128 requires seven-stage transform and 1/sqrt(128), not truncating this fixed D256 implementation; D256 stage geometry exists but full user Pi metadata/inverse equivalence unproven"
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB": {
    "symbolic": "distinct field storage 10*M*D + 32*M + 56*D bytes, plus >=224 bytes addressability beyond l_prev logical end if header full-VL Duplicate claim is retained; D=256; alignment/skews additional",
    "logical_at_M16_D256_bytes": 55808,
    "with_l_read_padding_bytes": 56032,
    "actual_allocation_bytes": null,
    "sources": [
      109,
      135,
      160,
      165,
      168,
      173
    ]
  },
  "L1": {
    "symbolic": "0 within leaf",
    "actual_bytes": null
  },
  "L0": {
    "symbolic": "0 within leaf",
    "actual_bytes": null
  },
  "VREG": {
    "symbolic": "memory-vector wrappers manage registers; no public RegTensor lifetime hook",
    "actual_register_count": null
  }
}
```

Synchronization/lifetime contract:
```json
{
  "internal": "PIPE_V barriers between dependent normalization, butterfly, sign, gate and cast operations; block_stage src/dst must not alias (196-211)",
  "caller": "owns all LocalTensor allocations and query views, MTE2_V before gate consumption, V_MTE3 before context publication, and scratch lifetime/reuse across each query (280-282)",
  "register_composition_hook": "none: Engine stages accept LocalTensor state, not RegTensor; eight butterfly stages plus sign/gating materialize UB intermediates"
}
```

CAModel verification and limitations:
```json
{
  "status": "historical fused-operator report claims arithmetic on beta.2; not a current 9.2.1 PASS and no isolated current-header evidence",
  "report": "C:/ascend-micro-harness/reports/paged_decode_attention.md",
  "report_lines": [
    3,
    20,
    22,
    27,
    617,
    638
  ],
  "test_source": "C:/ascend-micro-harness/tests/camodel/test_camodel_paged_decode_attention.cpp",
  "oracle": "C:/ascend-micro-harness/tests/common/paged_decode_oracle.hpp:648",
  "raw_proof_path": null,
  "current_run": null
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_vector_egress_rotate`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_vector_gqa_accumulate.hpp

Path: `C:/ascend-micro-harness/routines/routine_vector_gqa_accumulate.hpp`  
SHA256: `2eeb0d8301a81b67d376930d14a7d08029deb943ee1b399b3113a4e223ccbd07`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 89:
```cpp
constexpr bool GqaAccShapeValid(int32_t rows,int32_t row_elements)
```

Line 102:
```cpp
ASCEND_MICRO_GQA_ACC_INLINE constexpr bool GqaAccDecayRow(int32_t rows,uint32_t row)
```

Line 113:
```cpp
template<uint32_t kRowElements,class MulLane,class SplatLane> void GqaAccDecayWalk(int32_t rows,MulLane&& mul_lane,SplatLane&& splat_lane)
```

Line 140:
```cpp
template<uint32_t kRowElements,uint32_t kRow,bool kDecay=true> __simd_vf__ inline void gqa_accumulate_decay_row_vf(__ubuf__ float* accAddr,__ubuf__ float* corrAddr)
```

Line 165:
```cpp
struct NativeGqaAccumulateEngine
```

Line 178:
```cpp
void NativeGqaAccumulateEngine::set_shape(int32_t tile_rows,int32_t elements_per_row)
```

Line 196:
```cpp
ASCEND_MICRO_GQA_ACC_INLINE void NativeGqaAccumulateEngine::decay()
```

Line 220:
```cpp
ASCEND_MICRO_GQA_ACC_INLINE void NativeGqaAccumulateEngine::add_context()
```

Line 225:
```cpp
ASCEND_MICRO_GQA_ACC_INLINE int32_t NativeGqaAccumulateEngine::elements() const
```

Line 259:
```cpp
template<class Engine> ASCEND_MICRO_GQA_ACC_INLINE void gqa_vector_accumulate(Engine& engine)
```

Supported dtypes; input/output distinctions retained:
```json
{
  "actual": "acc,ctx,corr and vector arithmetic FP32 only",
  "BF16": false,
  "INT8": false,
  "FP32": true,
  "descriptor": {
    "path": "C:/GQA/interface/routines/vector_accumulate.h",
    "lines": [
      6,
      11,
      25,
      27
    ],
    "signature": "template<typename Engine> status_t gqa_pipe::vector_accumulate(Engine&,desc_t acc,desc_t ctx,desc_t state,const vector_accumulate_params*)",
    "mismatch": "desc_t types generic (FP16/BF16/FP32 enum in gqa_desc.h:12), implementation fixed FP32 and actual public duck stage omits required +ctx"
  }
}
```

Execution pipe:
```json
{
  "leaf": "AIV SIMD VF decay + memory-vector Add method",
  "SIMT": false,
  "AIC": false,
  "MIX": "caller may consume dual-destination Fixpipe context; no L0C instruction in leaf"
}
```

Geometry:
```json
{
  "M_actual": "rows 1..8 accepted, one AIV half of M16 Cube tile",
  "D_actual": "shape validator accepts 64..512 step64; decay dispatcher actually only 64,256,else512",
  "current_128_support": "validator accepts but default dispatch executes 512-lane rows; unsafe",
  "current_256_support": "dispatch exists but all rows address acc base 0; not correct full tile",
  "proposed_128_256": "requires exact width dispatch and row addressing, plus required add stage before use"
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB": {
    "symbolic": "intended acc=4*M*D, ctx=4*M*D, corr logical32 bytes (header demands 256-byte landing); total 8*M*D+256 before alignment/skew",
    "at_M8_D256_bytes": 16640,
    "at_M8_D512_bytes": 33024,
    "actual_allocation_bytes": null,
    "budget_caveat": "16384-byte static_assert 84-85 counts only acc maximum, not full live acc+ctx+corr"
  },
  "L1": {
    "symbolic": "0 within leaf",
    "actual_bytes": null
  },
  "L0": {
    "symbolic": "0 within leaf, ctx arrives in UB",
    "actual_bytes": null
  },
  "VREG": {
    "symbolic": "decay per row corrReg+accReg and full mask; two FP32 VL registers source-visible, allocator count unknown",
    "actual_register_count": null
  }
}
```

Synchronization/lifetime contract:
```json
{
  "internal": "PIPE_V after all row VFs before separate Add method (211-213)",
  "caller": "owns FIX->V CtxReady, V->FIX CtxFree, corr order, operand alignment/addressability (248-254)",
  "register_composition_hook": "gqa_accumulate_decay_row_vf takes UB pointers and stores each step; no public RegTensor entry/exit; add_context separate UB sweep"
}
```

CAModel verification and limitations:
```json
{
  "status": "historical beta.2 report FAIL 2048/2048; report says Add ran, contradicting current public leaf which omits Add; no current PASS",
  "report": "C:/ascend-micro-harness/reports/gqa_accumulate.md",
  "report_lines": [
    4,
    270,
    276,
    279,
    286,
    297
  ],
  "test_source": "C:/ascend-micro-harness/tests/camodel/test_camodel_gqa_accumulate.cpp",
  "wrapper": "C:/ascend-micro-harness/kernels/camodel/gqa_accumulate.cpp",
  "raw_proof_path": null,
  "current_run": null,
  "registry": {
    "path": "C:/GQA/routines.yaml",
    "lines": [
      78,
      87
    ],
    "status": "camodel_pending; running(batch) historical metadata, not PASS"
  }
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_vector_gqa_accumulate`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_vector_gqa_normalize_store.hpp

Path: `C:/ascend-micro-harness/routines/routine_vector_gqa_normalize_store.hpp`  
SHA256: `664b7901e3bac20bee38ca3589076841a6d0570be92b600fb6ca6114c894807f`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 140:
```cpp
template<class Engine> ASCEND_MICRO_GQA_NS_INLINE void gqa_normalize_store(Engine& engine)
```

Line 151:
```cpp
template<class T> struct GqaNsType; specializations half and bfloat16_t
```

Line 198:
```cpp
template<class T> __simd_vf__ inline void gqa_normalize_store_clamp_vf(__ubuf__ float* lAddr,__ubuf__ T* denomAddr)
```

Line 222:
```cpp
template<class T,uint32_t kHead> __simd_vf__ inline void gqa_normalize_store_vf(__ubuf__ float* srcAddr,__ubuf__ float* denomAddr,__ubuf__ T* dstAddr,uint32_t outStride)
```

Line 263:
```cpp
template<class Output> struct NativeGqaNormalizeStoreEngine
```

Line 278:
```cpp
__aicore__ inline void NativeGqaNormalizeStoreEngine<Output>::copy_acc_rows()
```

Line 280:
```cpp
__aicore__ inline void NativeGqaNormalizeStoreEngine<Output>::normalize()
```

Line 291:
```cpp
__aicore__ inline void NativeGqaNormalizeStoreEngine<Output>::store()
```

Supported dtypes; input/output distinctions retained:
```json
{
  "actual": "FP32 acc/denominator; output half or BF16 only",
  "BF16": "implemented CAST_ROUND specialization, current CAModel wrapper selects half only",
  "INT8": false,
  "FP32": "compute/input only, no FP32 Output engine accepted",
  "descriptor": {
    "path": "C:/GQA/interface/routines/normalize_store.h",
    "lines": [
      6,
      13,
      28,
      30
    ],
    "signature": "template<typename Engine> status_t gqa_pipe::normalize_store(Engine&,desc_t out_gm,desc_t acc,desc_t state,const normalize_store_params*)",
    "mismatch": "descriptor has rows/tails and arbitrary head_dim; implementation fixes 8 rows, native normalize always kHead=256, copy repitch no-op"
  }
}
```

Execution pipe:
```json
{
  "leaf": "AIV SIMD VF + MTE3 DataCopy",
  "SIMT": false,
  "AIC": false,
  "MIX": "caller cross-core readiness ownership only"
}
```

Geometry:
```json
{
  "M_actual": 8,
  "D_actual": "VF template permits multiples64 <=256 (also erroneously zero); native engine normalize always256",
  "current_128_support": "template callable at128 with fixed256 source pitch; native head_dim=128 changes store only, normalize still256; no repitch",
  "current_256_support": "shape exists; denominator stride crosses declared rows and clamp not published",
  "proposed_128_256": "must expose actual D specialization and output pitch; standalone signed context requires removing acc<=0 empty heuristic"
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB": {
    "symbolic": "intended src 8*256*4=8192, denom 8*256=2048, narrow dst 8*outStride*2=4096 at outStride256; total14336 plus raw unfolded l storage/padding if clamp used; no internal allocation",
    "actual_allocation_bytes": null,
    "denominator_actual_addressed_span_bytes": 2816,
    "denominator_declared_span_bytes": 2048,
    "source": "denRow=row*64 floats plus step*64 at 237,246, step0..3 -> last read starts float640 and ends704; overread768B at D256",
    "budget_caveat": "12288 external budget is already below acc+denom+out distinct live storage14336"
  },
  "L1": {
    "symbolic": "0 within leaf",
    "actual_bytes": null
  },
  "L0": {
    "symbolic": "0 within leaf",
    "actual_bytes": null
  },
  "VREG": {
    "symbolic": "normalize source-visible zero,acc,den,narrowed registers plus full/empty masks; clamp oneReg,lReg,fan plus mask; exact register allocation unknown",
    "actual_register_count": null
  }
}
```

Synchronization/lifetime contract:
```json
{
  "internal": "none; no event/barrier in leaf",
  "caller": "MTE2_V and cross-core CtxReady before normalization; V_MTE3 before store; gqa_normalize_store wrapper calls normalize then store without a barrier hook, caller may invoke methods separately",
  "register_composition_hook": "Div feeds Cast within VF without UB quotient store, but no public RegTensor input/output hook; whole VF still loads acc/denom from UB and writes narrow UB"
}
```

CAModel verification and limitations:
```json
{
  "status": "historical beta.2 report FAIL (divide by zero), no PASS line, no measured cycle figure; no current 9.2.1 proof",
  "report": "C:/ascend-micro-harness/reports/gqa_normalize_store.md",
  "report_lines": [
    4,
    11,
    21,
    23,
    167,
    198,
    205
  ],
  "test_source": "C:/ascend-micro-harness/tests/camodel/test_camodel_gqa_normalize_store.cpp",
  "wrapper": "C:/ascend-micro-harness/kernels/camodel/gqa_normalize_store.cpp",
  "wrapper_lines": [
    39,
    53,
    62,
    122,
    131,
    133,
    152
  ],
  "wrapper_caveat": "current wrapper contains DBG probe instrumentation and half-only Engine; history report cannot certify exact current SHA",
  "raw_proof_path": null,
  "current_run": null,
  "registry": {
    "path": "C:/GQA/routines.yaml",
    "lines": [
      89,
      98
    ],
    "status": "camodel_pending; running(batch) historical metadata, not PASS"
  }
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_vector_gqa_normalize_store`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_vector_gqa_online_softmax.hpp

Path: `C:/ascend-micro-harness/routines/routine_vector_gqa_online_softmax.hpp`  
SHA256: `71713bee68753b8174d01aeed94ca9b6131aec4ad970f8a82cc8fe4257163189`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 71:
```cpp
template <class Engine> ASCEND_MICRO_GQA_SM_INLINE void gqa_online_softmax_update(Engine& engine)
```

Line 212:
```cpp
ASCEND_MICRO_GQA_SM_INLINE void duplicate(AscendC::LocalTensor<float>& dst, AscendC::LocalTensor<float>& src)
```

Line 218:
```cpp
ASCEND_MICRO_GQA_SM_INLINE void scale_scores()
```

Line 223:
```cpp
ASCEND_MICRO_GQA_SM_INLINE void mask_scores()
```

Line 257:
```cpp
ASCEND_MICRO_GQA_SM_INLINE void find_row_max()
```

Line 265:
```cpp
ASCEND_MICRO_GQA_SM_INLINE void update_running_max()
```

Line 296:
```cpp
ASCEND_MICRO_GQA_SM_INLINE void exponentiate()
```

Line 317:
```cpp
ASCEND_MICRO_GQA_SM_INLINE void accumulate()
```

Line 327:
```cpp
ASCEND_MICRO_GQA_SM_INLINE void commit()
```

Supported dtypes; input/output distinctions retained:
```json
{
  "FP32": "actual LocalTensor<float> arithmetic and recurrence",
  "BF16": "not an arithmetic engine dtype; cube Operand template alone does not prove usable codec instantiation",
  "INT8": "not an arithmetic engine dtype; uint8_t GQA keep is predicate storage, not INT8 data support",
  "descriptor": null,
  "descriptor_evidence": "routines.yaml not present in assigned C:/ascend-micro-harness root; no descriptor support inferred"
}
```

Execution pipe:
```json
AIV Vector / compatibility APIs lowering to VF on dav_3510; caller provides UB; no Cube arithmetic in these headers
```

Geometry:
```json
{
  "current": "rows=8, tokens=64, FP32 scores 8x64; runtime rows intended1..8; packed8-lane row state, broadcast64 lanes",
  "M128_D256": "16 groups of8 rows with K64; D256 not part of softmax leaf; wider M not directly supported",
  "M256_D256": "32 groups of8 rows with K64; D256 not part of leaf",
  "guards": "no rows range check; rows0 not legalized; per-row limits expected clamped by caller to tokens; noncausal leaf only clamps lower bound"
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB": {
    "actual_bytes": null,
    "allocation_owner": "caller",
    "logical_nonalias_payload_bytes": 7328,
    "symbolic": "3*(8*64*4) + 11*(8*4) + 2*(64*4) +64*4 +64 mask =7328 bytes; excludes caller bank skew/read pads",
    "camodel_wrapper_arena_bytes": 7872,
    "camodel_wrapper_arena_plus_read_pad_bytes": 8128,
    "allocation_evidence": "C:/ascend-micro-harness/kernels/camodel/gqa_online_softmax.cpp:42-85",
    "warning": "logical allocation does not prove masked-stride access safety: m_new allocated32 bytes but Max mask8 x rows8 with dstRepStride1 spans256 bytes"
  },
  "L1": {
    "actual_bytes": 0,
    "reason": "no L1 allocation/access in leaf"
  },
  "L0": {
    "actual_bytes": 0,
    "reason": "no L0 allocation/access in leaf"
  },
  "VREG": {
    "actual_bytes": null,
    "symbolic": "compiler/lowering owned; 256-byte vector width, no persistent RegTensor interface"
  }
}
```

Synchronization/lifetime contract:
```json
{
  "leaf": "PipeBarrier<PIPE_V> only; no explicit SetFlag/WaitFlag/TPipe or DMA in leaf",
  "caller": "input producer readiness, output consumer visibility, allocation/aliasing/lifetime and inter-core synchronization"
}
```

CAModel verification and limitations:
```json
{
  "current_hash_verified_pass": null,
  "status": "unverified_current_source; inventory-only, no runs",
  "evidence_paths": [
    "C:/ascend-micro-harness/reports/gqa_online_softmax.md:15-17",
    "C:/ascend-micro-harness/reports/gqa_online_softmax.md:147-171",
    "C:/ascend-micro-harness/tests/camodel/test_camodel_gqa_online_softmax.cpp",
    "C:/ascend-micro-harness/kernels/camodel/gqa_online_softmax.cpp"
  ],
  "evidence_note": "host mock PASS is not device proof; report Tier2 says see missing section, records mirror builds/concurrent rewrites, measured:null; no hash-bound raw current proof"
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_vector_gqa_online_softmax`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_vector_icdf_quantize.hpp

Path: `C:/ascend-micro-harness/routines/routine_vector_icdf_quantize.hpp`  
SHA256: `db856a6e034bb242a525dbc41f507115d090e8621fb357d47014abd84d1a2b45`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 94:
```cpp
template<class Input> __simd_vf__ inline void icdf_quantize_vf(__ubuf__ Input* inAddr,__ubuf__ int4x2_t* outAddr,uint16_t repeats) [94-96]
```

Supported dtypes; input/output distinctions retained:
```json
{
  "BF16": "Input via SAT/RINT BF16->half; compute half.",
  "INT8": "No INT8 arithmetic; output int4x2_t packed storage.",
  "FP32": "No input support; non-BF16 branch hardcodes LoadAlign<half>.",
  "FP16": true
}
```

Execution pipe:
```json
{
  "actual": "AIV SIMD VF",
  "SIMD": true,
  "SIMT": false,
  "AIC": false,
  "MIX": false
}
```

Geometry:
```json
{
  "D": null,
  "M": null,
  "lanes_per_repeat": 128,
  "in_bytes_per_repeat": 256,
  "out_bytes_per_repeat": 64,
  "default_repeats": 8,
  "proposed_M128": "Flatten MD only when divisible128, repeats=MD/128 within uint16; no rowwise contract.",
  "proposed_M256": "Same; no E8 geometry implemented."
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB_owned_bytes": 0,
  "UB_symbolic": "256R input +64R output bytes, caller supplied.",
  "VREG_symbolic": "numReg half,x half,s4 int4x2 =3x256B declarations; optional xBf adds256B; MaskReg. Physical peak null.",
  "current_wrapper_arena_bytes": 9312,
  "current_wrapper_addressable_with_pad_bytes": 9568
}
```

Synchronization/lifetime contract:
```json
{
  "owner": "Caller owns MTE2/V/MTE3 readiness, completion, and lifetimes. Leaf no events or TPipe.",
  "lifetimes": "input through VF completion; output through MTE3 drain.",
  "wrapper_events": "MTE2_V0,V_MTE2 1,V_MTE3 2,MTE3_V3,MTE3_S4; depth2 input/output slots, one FIFO token/route."
}
```

CAModel verification and limitations:
```json
{
  "status": "Tests and historical baseline report only; no current source hash run.",
  "evidence": [
    {
      "path": "C:/ascend-micro-harness/tests/camodel/test_camodel_icdf_quantize.cpp",
      "lines": [
        22,
        24
      ],
      "claim": "FP16/BF16 launchers present"
    },
    {
      "path": "C:/ascend-micro-harness/reports/lloydmax_quantize/REPORT.md",
      "lines": [
        59
      ],
      "claim": "Historical ICDF baseline, not E8."
    }
  ]
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_vector_icdf_quantize`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_vector_normalize_store.hpp

Path: `C:/ascend-micro-harness/routines/routine_vector_normalize_store.hpp`  
SHA256: `aa0a4dd8f3809a413ec9b39489a21a921f08d4c4a32852980505a721cec67a52`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 19:
```cpp
template<class Engine> void vector_normalize(Engine&,uint32_t) [19-20]
```

Line 27:
```cpp
template<class Engine> void vector_store(Engine&,uint32_t) [27-28]
```

Line 38:
```cpp
template<class Output> struct NativeNormalizeStoreEngine [38-45]
```

Line 56:
```cpp
void broadcast_denominator(uint32_t count)
```

Line 60:
```cpp
void divide(uint32_t count)
```

Line 65:
```cpp
void cast_output(uint32_t count)
```

Line 70:
```cpp
void mte3_store(uint32_t count)
```

Supported dtypes; input/output distinctions retained:
```json
{
  "input": "FP32 acc and FP32 l[0].",
  "BF16": "Output supported Float->BF16 CAST_RINT; native fixture exists.",
  "INT8": "No supported Float->INT8 pair in current Cast table.",
  "FP32": "Arithmetic FP32; output float->float Cast not verified, absent relevant table; do not claim.",
  "FP16": "Output supported."
}
```

Execution pipe:
```json
{
  "actual": "AIV SIMD compatibility arithmetic plus MTE3",
  "SIMD": true,
  "SIMT": false,
  "AIC": false,
  "MIX": false
}
```

Geometry:
```json
{
  "D": null,
  "M": 1,
  "count_multiple": 16,
  "current_runner_count": 16,
  "proposed_M128": "Per-row call/denominator required; flatten varying l rows is wrong.",
  "proposed_M256": "Same; no built-in batched rowwise leaf."
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB_owned_bytes": 0,
  "UB_symbolic": "12N+N*sizeof(Output)+denominator view; l requires256B addressable backing even for tiny count.",
  "wrapper_payload_end_bytes": 384,
  "wrapper_output_addressable_end_bytes": 608,
  "VREG_physical_bytes": null
}
```

Synchronization/lifetime contract:
```json
{
  "owner": "Caller MTE2->V and V->MTE3 with separated normalize/store leaves; no events allocated by leaf.",
  "lifetimes": "All arithmetic sources/intermediates until consumers finish; UB output until MTE3 retires.",
  "wrapper_events": "MTE2_V0,V_MTE3 0,MTE3_S0"
}
```

CAModel verification and limitations:
```json
{
  "status": "Historical native normalization infrastructure PASS is not E8 proof or current hash PASS.",
  "evidence": [
    {
      "path": "C:/ascend-micro-harness/reports/micro_routines_validation.md",
      "lines": [
        49,
        70
      ],
      "claim": "Historical FP16/BF16 16/16 each,384guards"
    },
    {
      "path": "C:/ascend-micro-harness/tests/camodel/test_camodel_normalize_store.cpp",
      "lines": [
        1
      ],
      "claim": "Native runner present"
    }
  ]
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_vector_normalize_store`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_vector_online_softmax.hpp

Path: `C:/ascend-micro-harness/routines/routine_vector_online_softmax.hpp`  
SHA256: `b54a39fe21543006fadc61c6be37b9dd924d221cbe5522382bf7f6bf9d58635a`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 31:
```cpp
template <class Engine> ASCEND_MICRO_ONLINE_INLINE void vector_online_softmax_probabilities(Engine& engine)
```

Line 41:
```cpp
template <class Engine> ASCEND_MICRO_ONLINE_INLINE void vector_online_softmax(Engine& engine)
```

Line 139:
```cpp
__aicore__ inline void update_max_and_alpha()
```

Line 151:
```cpp
__aicore__ inline void exponentiate_scores()
```

Line 159:
```cpp
__aicore__ inline void update_denominator()
```

Line 170:
```cpp
__aicore__ inline void update_accumulator()
```

Supported dtypes; input/output distinctions retained:
```json
{
  "FP32": "actual LocalTensor<float> arithmetic and recurrence",
  "BF16": "not an arithmetic engine dtype; cube Operand template alone does not prove usable codec instantiation",
  "INT8": "not an arithmetic engine dtype; uint8_t GQA keep is predicate storage, not INT8 data support",
  "descriptor": null,
  "descriptor_evidence": "routines.yaml not present in assigned C:/ascend-micro-harness root; no descriptor support inferred"
}
```

Execution pipe:
```json
AIV Vector / compatibility APIs lowering to VF on dav_3510; caller provides UB; no Cube arithmetic in these headers
```

Geometry:
```json
{
  "current": "one16-key row,16 output channels; V column-major panel16x16; gathered probability offsets0,4,...60 repeated per channel",
  "M128_D256": "fixed16x16 leaf cannot become M128/D256 by descriptor; partition rows independently and D into16-channel blocks, avoid repeating denominator updates; redesign/adapter needed",
  "M256_D256": "same requirement across256 rows; no native M256 or D256 loop",
  "guards": "fixed constants; no dynamic tails"
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB": {
    "actual_bytes": null,
    "allocation_owner": "caller",
    "logical_nonalias_payload_bytes": 5856,
    "symbolic": "7*16*4 +4*16*16*4 +9*8*4 +4*64*4 =5856 bytes",
    "camodel_wrapper_arena_bytes": 6592,
    "camodel_wrapper_arena_plus_read_pad_bytes": 6848,
    "allocation_evidence": "C:/ascend-micro-harness/kernels/camodel/online_softmax.cpp:38-101"
  },
  "L1": {
    "actual_bytes": 0,
    "reason": "no L1 allocation/access in leaf"
  },
  "L0": {
    "actual_bytes": 0,
    "reason": "no L0 allocation/access in leaf"
  },
  "VREG": {
    "actual_bytes": null,
    "symbolic": "compiler/lowering owned; 256-byte vector width, no persistent RegTensor interface"
  }
}
```

Synchronization/lifetime contract:
```json
{
  "leaf": "PipeBarrier<PIPE_V> only; no explicit SetFlag/WaitFlag/TPipe or DMA in leaf",
  "caller": "input producer readiness, output consumer visibility, allocation/aliasing/lifetime and inter-core synchronization"
}
```

CAModel verification and limitations:
```json
{
  "current_hash_verified_pass": null,
  "status": "unverified_current_source; inventory-only, no runs",
  "evidence_paths": [
    "C:/ascend-micro-harness/tests/camodel/test_camodel_online_softmax.cpp",
    "C:/ascend-micro-harness/kernels/camodel/online_softmax.cpp",
    "C:/ascend-micro-harness/reports/online_softmax_scalar_fault_trace.txt",
    "C:/ascend-micro-harness/reports/fused_decode_opt_20261103/compiled-source-sha256.txt:14"
  ],
  "evidence_note": "test/wrapper/old fault trace presence is not current PASS; archive hash mismatch"
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_vector_online_softmax`, fields `intrinsic_api_audit` and `audit_detail`.

### routine_vector_rotate_butterfly.hpp

Path: `C:/ascend-micro-harness/routines/routine_vector_rotate_butterfly.hpp`  
SHA256: `a1d57d385e4db3838a8fda0a37d5ed04b29002fdbbb901dc43e6709a429b11c8`  
Namespace: `ascend_micro`

Exact public/template declarations:

Line 167:
```cpp
constexpr unsigned RotDirIndex(RotateDirection d)
```

Line 205:
```cpp
template <std::uint32_t S> __aicore__ inline void block_stage(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src) const
```

Line 219:
```cpp
template <std::uint32_t Index> __aicore__ inline void gather_stage(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src) const
```

Line 230:
```cpp
__aicore__ inline void sign_stage(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src) const
```

Line 243:
```cpp
template <RotateDirection D> __aicore__ inline void rotate(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src)
```

Line 292:
```cpp
__aicore__ inline void scale(const AscendC::LocalTensor<float>& dst, float factor)
```

Line 317:
```cpp
__simd_callee__ inline void rot_reg_stage64(RotRegSet& v, AscendC::Reg::MaskReg& mask)
```

Line 330:
```cpp
__simd_callee__ inline void rot_reg_stage128(RotRegSet& v, AscendC::Reg::MaskReg& mask)
```

Line 370:
```cpp
__simd_callee__ inline void rot_one_table(AscendC::Reg::RegTensor<std::uint32_t>& idx, AscendC::Reg::RegTensor<std::uint32_t>& sgn_bits, AscendC::Reg::RegTensor<std::uint32_t>& lane, AscendC::Reg::RegTensor<std::uint32_t>& one_bits, std::uint32_t k, AscendC::Reg::MaskReg& mask)
```

Line 387:
```cpp
__simd_callee__ inline void rot_build_gather_tables(RotGatherTables& t, AscendC::Reg::MaskReg& mask)
```

Line 413:
```cpp
__simd_callee__ inline void rot_gather_one(AscendC::Reg::RegTensor<float>& v, AscendC::Reg::RegTensor<std::uint32_t>& idx, AscendC::Reg::RegTensor<std::uint32_t>& sgn_bits, AscendC::Reg::MaskReg& mask)
```

Line 426:
```cpp
__simd_callee__ inline void rot_gather_stage(RotRegSet& v, AscendC::Reg::RegTensor<std::uint32_t>& idx, AscendC::Reg::RegTensor<std::uint32_t>& sgn, AscendC::Reg::MaskReg& mask)
```

Line 440:
```cpp
__simd_callee__ inline void rot_shuffle_one(AscendC::Reg::RegTensor<float>& v, AscendC::Reg::MaskReg& mask, AscendC::Reg::MaskReg& half)
```

Line 451:
```cpp
__simd_callee__ inline void rot_shuffle_stage(RotRegSet& v, AscendC::Reg::MaskReg& mask, AscendC::Reg::MaskReg& half)
```

Line 463:
```cpp
__simd_callee__ inline void rot_shuffle_permute(AscendC::Reg::RegTensor<float>& v, AscendC::Reg::MaskReg& half)
```

Line 472:
```cpp
__simd_callee__ inline void rot_load_row(RotRegSet& v, __ubuf__ float* in)
```

Line 482:
```cpp
__simd_callee__ inline void rot_store_row(__ubuf__ float* out, RotRegSet& v, AscendC::Reg::MaskReg& mask)
```

Line 494:
```cpp
__simd_callee__ inline void rot_scale_row(RotRegSet& v, float factor, AscendC::Reg::MaskReg& mask)
```

Line 507:
```cpp
__simd_callee__ inline void rot_load_signs(RotSignRegs& p, __ubuf__ float* sign)
```

Line 518:
```cpp
__simd_callee__ inline void rot_apply_signs(RotRegSet& v, RotSignRegs& p, AscendC::Reg::MaskReg& mask)
```

Line 537:
```cpp
template <RotateDirection D> __simd_callee__ inline void rot_row_gather(RotRegSet& v, RotGatherTables& t, RotSignRegs& p, AscendC::Reg::MaskReg& mask)
```

Line 566:
```cpp
__simd_callee__ inline void rot_row_shuffle(RotRegSet& v, RotSignRegs& p, AscendC::Reg::MaskReg& mask, AscendC::Reg::MaskReg& half)
```

Line 585:
```cpp
__simd_callee__ inline void rot_permute_signs_p5(RotSignRegs& p, AscendC::Reg::MaskReg& half)
```

Line 602:
```cpp
template <RotateDirection D> __simd_vf__ inline void rotate_butterfly_gather_vf(__ubuf__ float* dst, __ubuf__ float* src, __ubuf__ float* sign, float scale, std::uint16_t rows)
```

Line 627:
```cpp
__simd_vf__ inline void rotate_butterfly_shuffle_vf(__ubuf__ float* dst, __ubuf__ float* src, __ubuf__ float* sign, float scale, std::uint16_t rows)
```

Supported dtypes; input/output distinctions retained:
```json
{
  "FP32": {
    "supported": true,
    "evidence": "All native arithmetic operands and boundary pointers are float; lines188-198,306-310,603-605,627-629."
  },
  "BF16": {
    "supported": false,
    "reason": "No bfloat16_t overload or conversion in this header; callers can convert to FP32 externally."
  },
  "INT8": {
    "supported": false,
    "reason": "No int8 data overload or quantization/packing in this header."
  }
}
```

Execution pipe:
```json
{
  "Blocked": "AIV SIMD Level2 UB tensor APIs; all stages write UB.",
  "RegGather": "AIV SIMD __simd_vf__ wrappers + __simd_callee__ VREG row leaf; no SIMT entry.",
  "RegShuffle": "AIV SIMD __simd_vf__ + __simd_callee__; ingress only.",
  "SIMT_existing": false,
  "native_guard": "__CCE_AICORE__ || ASCEND_MICRO_ENABLE_NATIVE (120,169); comments target Ascend950PR arch35 dav-3510."
}
```

Geometry:
```json
{
  "D_actual": 256,
  "D_evidence": "Fixed constants127-133; 4 FP32 RegTensor each64 lanes; no D template/tail mask.",
  "D_proposed": null,
  "M_actual": "VF wrappers accept uint16_t rows (0..65535), contiguous row stride256 floats; no runtime buffer-bound guard; all 256 channels valid. Blocked rotate handles one row.",
  "M_verified": [
    1,
    16
  ],
  "directions": "Gather and Blocked use Egress [32,64,128,1,2,4,8,16], signs after3; Ingress [1,2,4,8,16,32,64,128], signs after5. Shuffle implements ingress only; no Egress template exposed.",
  "scale": "Caller float; default normalization not automatic; kRotOrthoScale=1/16."
}
```

UB/L1/L0/VREG footprint; unknowns explicit:
```json
{
  "UB": {
    "allocation": "Caller-owned; leaf allocates none.",
    "Blocked": "6144B offset/sign tables +1024B pi_sign +4x1024B scratch =11264B descriptors; caller src/dst may add row/panel storage; alias constraints apply.",
    "RegGather_RegShuffle": "1024B natural sign vector +1024*M byte src panel +1024*M byte dst panel (src=dst allowed); no stage intermediates in source."
  },
  "L1_bytes": 0,
  "L0_bytes": 0,
  "VREG": {
    "bytes_per_reg": 256,
    "hardware_limit": 32,
    "doc": "C:/PipeSim/docs/ascendc_api/SIMD-API/basic_api/reg_vector_compute/register_data_types/RegTensor.md:31,50-51",
    "RegGather_symbolic": "4 data +4 pi-sign +12 table VREG persist per row =20; at register stages up to4 named float temps, plus transient table-building ramp/one_bits/stride/bit/shifted outside row loop. Exact compiler live count unknown.",
    "RegShuffle_symbolic": "4 data +4 sign persist =8; one-register shuffle helper declares4 temporary float VREG; inter-register stage declares4. Exact compiler live count unknown.",
    "actual_register_count": null,
    "spill_claim": "Report old-toolchain trace: 4 row loads+4 row stores per row; source alone cannot guarantee fused caller remains spill-free."
  }
}
```

Synchronization/lifetime contract:
```json
{
  "leaf": "Blocked gather_stage barrier at226; rotate inserts PIPE_V between stages249-282. Last rotate stage and scale have no final barrier. SIMD callee arithmetic ordered within VF; no HardEvent ownership.",
  "caller": "Provide initialized aligned UB src/sign/tables and scratch; Blocked dst must not alias src and block stage dst/src distinct (202-204,240-242). Fence producer MTE2->V before use, V->MTE3 before export, handle reuse hazards. Bench lines136-137 MTE2_V;176-177 V_MTE3; caller places PIPE_V after rotate/scale156,158. SIMD src=dst allowed because all4 data registers loaded first (597-600)."
}
```

CAModel verification and limitations:
```json
{
  "current_CANN_9_2_1_header_hash_verified": null,
  "status": "Historical reported CAModel PASS, not executed in Phase0. No standalone summary_butterfly*.json located in bounded /home/dmin/work or Windows repo scan.",
  "reports": [
    {
      "path": "C:/ascend-micro-harness/reports/rotate_butterfly.md",
      "lines": [
        3,
        4,
        159,
        168,
        169,
        170,
        171,
        241,
        327,
        384
      ],
      "scope": "CANN9.2.0-beta.2 Ascend950PR_9599 grid1; ingress all3 variants, M1/M16; source/output poison and input preservation checks,4VLD+4VST/row for register forms. Report explicitly says standalone egress not run."
    },
    {
      "path": "C:/ascend-micro-harness/reports/single_token_decode_aiv.md",
      "lines": [
        3,
        4,
        41,
        42,
        348,
        352
      ],
      "scope": "Later integration historical egress RegGather CAModel numeric rotation errors 9.31e-9,1.49e-8,5.59e-9 below budgets, trace PASS17 checks; not current-hash CANN9.2.1 proof."
    },
    {
      "path": "C:/ascend-micro-harness/reports/l2_atomic_contention_split_k.md",
      "lines": [
        527,
        536,
        540,
        543,
        548
      ],
      "scope": "ModeC egress output bit-exact after BF16 rounding but unresolved model diagnostic; NOT clean PASS."
    }
  ],
  "test_sources_not_PASS": [
    "C:/ascend-micro-harness/kernels/camodel/rotate_butterfly.cpp",
    "C:/ascend-micro-harness/tests/camodel/test_camodel_rotate_butterfly.cpp",
    "C:/ascend-micro-harness/tests/cpu/test_rotate_butterfly_probe.cpp"
  ]
}
```

Full intrinsic-by-intrinsic local API/CANN evidence is in JSON entry `routine_vector_rotate_butterfly`, fields `intrinsic_api_audit` and `audit_detail`.

## External frozen E8 entries

JSON `external_reference_entries` records exact Agent06/08 signatures, selected zero-scratch/arithmetic layouts, native source paths, caller-owned synchronization, resources and Wave1 backend/timing evidence. These are additional external references, not copies overwriting repository headers.

## Provenance

JSON `inventory_batches` embeds native per-slot inventory paths, hashes and composition findings. All25 source hashes and both frozen baselines were verified during publication. Phase0 consumes no optimized iteration. Only this file and its JSON companion are part of the registry commit; unrelated work is preserved.
