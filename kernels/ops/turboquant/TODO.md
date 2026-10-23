```cpp
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_a8(
    // ------------------- [INPUTS: Входные активации] -------------------
    __gm__ const int8_t* __restrict__ key,          // [IN] Несжатые проекции K текущего батча
    __gm__ const int8_t* __restrict__ value,        // [IN] Несжатые проекции V текущего батча
    
    // ------------------- [OUTPUTS: Физический Paged KV-кэш] -----------
    __gm__ int8_t* __restrict__ keyCache,           // [OUT] Сжатый 4-bit пул ключей (Key-Major)
    __gm__ int8_t* __restrict__ valueCache,         // [OUT] Сжатый 4-bit пул значений (Channel-Major)
    __gm__ float* __restrict__ scaleCache,          // [OUT] Масштабные коэффициенты (по головам)
    
    // ------------------- [METADATA & ROTATION TABLES] ------------------
    __gm__ const int32_t* __restrict__ slotMapping, // [IN] Физические ID слотов в кэше для токенов
    __gm__ const float* __restrict__ piSigns,       // [IN] Знаки для ортогонального вращения
    __gm__ const int32_t* __restrict__ tables,      // [IN] Таблицы квантования / codebook
    
    // ------------------- [SCALARS: Размеры и сетка] -------------------
    uint32_t numTokens,                             // [IN] Число токенов в текущем батче
    uint32_t numKvHeads,                            // [IN] Число KV-голов (GQA / MQA)
    uint32_t headSize,                              // [IN] Размерность головы (head_dim, D)
    uint32_t blockSize,                             // [IN] Число токенов в одном блоке кэша (kKeys)
    uint32_t numBlocks,                             // [IN] Общее число физических блоков в пуле
    uint32_t tokensPerCore,                         // [IN] Число токенов на один AI Core
    float invSqrtLen                                // [IN] 1.0f / sqrt(headSize) для вращения
);
```

```
T -> [<-T0->][<-T1->][<-T2->][<-T3->][<-T4->][<-T5->][<-T6->][<-T7->][<-T8->][<-T9->][<-T10>]
MTE2 [ LdV0 ][ LdV1 ][ LdV2 ][ LdV3 ][ LdK0 ][ LdK1 ][ LdK2 ][ LdK3 ][ ---- ][ ---- ][ ---- ]
CUBE [ ---- ][ MMV0 ][ MMV1 ][ MMV2 ][ MMV3 ][ MMK0 ][ MMK1 ][ MMK2 ][ MMK3 ][ ---- ][ ---- ]
AIV0 [ ---- ][ ---- ][BtflV0][BtflV1][BtflV2][BtflV3][BtflK0][BtflK1][BtflK2][BtflK3][ ---- ]
AIV0 [ ---- ][ ---- ][BtflV0][BtflV1][BtflV2][BtflV3][BtflK0][BtflK1][BtflK2][BtflK3][ ---- ]
MTE3 [ ---- ][ ---- ][ ---- ][ StV0 ][ StV1 ][ StV2 ][ StV3 ][ StK0 ][ StK1 ][ StK2 ][ StK3 ]

V0,V1,V2,V3 - 16 х [0 - 255], 16 х [256 + 511], 16 х [512 + 767], 16 х [768 + 1023] of 16 х 1024 (16 tokens of V)
K0,K1,K2,K3 - 16 х [0 - 255], 16 х [256 + 511], 16 х [512 + 767], 16 х [768 + 1023] of 16 х 1024 (16 tokens of K)
```

---
---
---

# TASK: Implement & Verify `turboquant_reshape_and_cache_a8` (MIX_AIC_1_2, 4-Stage Depth-4 Pipeline)

Target Architecture: Huawei DaVinci v3 (`dav-3510` / Ascend 950PR / CANN 9.2).
Simulator: `Ascend950PR_9599` CAModel.
Cluster Topology: `MIX_AIC_1_2` (1 Cube Core / AIC + 2 Vector Cores / AIV0 & AIV1).

## 1. Objective and Boundary
Design, implement, and verify the bare-metal Ascend C prefill reshape-and-cache kernel:
`kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp` and its host CAModel test runner:
`tests/camodel/test_camodel_turboquant_reshape_and_cache_a8.cpp`.

### Target Dimension for CAModel Verification Probe
For this standalone test probe, process exactly ONE tile strip of 16 tokens across 1024 channels:
- Input Key: 16 tokens × 1024 elements (INT8, 16 KiB total)
- Input Value: 16 tokens × 1024 elements (INT8, 16 KiB total)
The 1024 channels are chunked into 4 sequential sub-tiles of shape [16 tokens × 256 channels] (4 KiB per chunk in INT8):
- V0, V1, V2, V3: channels [0..255], [256..511], [512..767], [768..1023] of Value.
- K0, K1, K2, K3: channels [0..255], [256..511], [512..767], [768..1023] of Key.

Once verified on this single strip, the kernel trivially tiles across `blockDim = 64` (1024 tokens = 64 AI Cores × 16 tokens/core = 100% chip saturation).

---

## 2. Kernel Signature & Strict Type Contracts
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_a8(
    // ------------------- [INPUTS: Raw A8 Activations] -------------------
    __gm__ const int8_t* __restrict__ key,          // [IN] Raw K projections [16, 1024]
    __gm__ const int8_t* __restrict__ value,        // [IN] Raw V projections [16, 1024]
    
    // ------------------- [OUTPUTS: Physical Paged KV-Cache] -------------
    __gm__ int8_t* __restrict__ keyCache,           // [OUT] Compressed INT4 Key (Key-Major)
    __gm__ int8_t* __restrict__ valueCache,         // [OUT] Compressed INT4 Value (Channel-Major)
    __gm__ float* __restrict__ scaleCache,          // [OUT] Scale factors (per-channel/head)
    
    // ------------------- [METADATA & ROTATION TABLES] --------------------
    __gm__ const int32_t* __restrict__ slotMapping, // [IN] Physical slot index for 16 tokens
    __gm__ const float* __restrict__ piSigns,       // [IN] Signs for orthogonal rotation
    __gm__ const int32_t* __restrict__ tables,      // [IN] Quantization codebook tables
    
    // ------------------- [SCALARS: Grid & Dimensions] --------------------
    uint32_t numTokens,                             // = 16 (for this single-strip test)
    uint32_t numKvHeads,                            // = 4
    uint32_t headSize,                              // = 256 (4 heads * 256 = 1024 channels)
    uint32_t blockSize,                             // = 16 (tokens per physical page)
    uint32_t numBlocks,                             // Total blocks in cache
    uint32_t tokensPerCore,                         // = 16
    float invSqrtLen                                // 1.0f / sqrt(256.0f)
);

Layout Matching Contracts (CRITICAL):Key Cache (keyCache): Must be stored in Key-Major layout:$$\text{nibble\_index} = \text{key} \times \text{headSize} + d$$Matches ZN fractal transposition byte-for-byte to permit zero-permutation direct MTE3 CV transfers during Decode attention.Value Cache (valueCache): Must be stored in Channel-Major layout:$$\text{nibble\_index} = ch \times \text{blockSize} + \text{key}$$Enables sequential streaming reads for vector Online Softmax recurrence.Nibble Packing: 2 nibbles per byte, low-nibble first (even index in [3:0], odd in [7:4]), signed int4 in [-8, 7].3. The 4-Stage Modulo Schedule (Depth-4 Ring)Execute the 4 chunks of $V$ followed immediately by the 4 chunks of $K$ in one continuous schedule across 11 cycles without draining pipeline bubbles:

```Plaintext
T ->  [<-T0->][<-T1->][<-T2->][<-T3->][<-T4->][<-T5->][<-T6->][<-T7->][<-T8->][<-T9->][<-T10>]
MTE2: [ LdV0 ][ LdV1 ][ LdV2 ][ LdV3 ][ LdK0 ][ LdK1 ][ LdK2 ][ LdK3 ][ ---- ][ ---- ][ ----  ]
CUBE: [ ---- ][ MMV0 ][ MMV1 ][ MMV2 ][ MMV3 ][ MMK0 ][ MMK1 ][ MMK2 ][ MMK3 ][ ---- ][ ----  ]
AIV0: [ ---- ][ ---- ][BtflV0][BtflV1][BtflV2][BtflV3][BtflK0][BtflK1][BtflK2][BtflK3][ ----  ]
AIV1: [ ---- ][ ---- ][BtflV0][BtflV1][BtflV2][BtflV3][BtflK0][BtflK1][BtflK2][BtflK3][ ----  ]
MTE3: [ ---- ][ ---- ][ ---- ][ StV0 ][ StV1 ][ StV2 ][ StV3 ][ StK0 ][ StK1 ][ StK2 ][ StK3  ]
```

Sub-Stage Responsibilities:Ld (MTE2): Direct transfer of 4 KiB chunk ($16 \times 256$ INT8) from GM into $L_1$ ring buffer (TPosition::B1), completely bypassing UB.MM (Cube Core / AIC):Resident $H_{16}$ matrix (16×16 INT8, 256 bytes) pinned in $L_0A$.MTE1 unpacks 16 fractals of 16×16 from $L_1$ to $L_0B$.Mmad ($INT8 \times INT8 \to INT32$) runs across all 16 fractals.Fixpipe transfers the $16 \times 256$ result into AIV0 and AIV1 UB landing buffers.Btfl (AIV0 & AIV1):Work-sharing: AIV0 handles rows 0..7 (8 tokens), AIV1 handles rows 8..15 (8 tokens).Execute remaining 2 stages of fast Hadamard butterfly cascades across channel dimension.Row-wise absolute maximum reduction $\to$ compute per-channel/head scale.Quantize into signed INT4 [-8, 7] and pack pairs of nibbles into bytes.Local UB permutation into Key-Major (for K) or Channel-Major (for V).St (MTE3):Write 2 KiB packed bytes + 16-byte scale slice from UB to GM physical cache page.4. Hardware Memory Budgets & InvariantsFollow MICROROUTINE_DESIGN_GUIDELINES.md strictly:L1 Ring (Cube L1, 512 KiB capacity):$H_{16}$ resident weight: 256 B at offset 0.Depth-4 input ring: $4 \text{ slots} \times 4096 \text{ B} = 16384 \text{ B}$ (32-byte aligned + 32B bank skew).UB Ring (256 KiB isolated per AIV):Fixpipe landing ring: Depth-4 × $8 \text{ tokens} \times 256 \text{ channels} \times 4\text{ B (FP32)} = 32 \text{ KiB}$.Butterfly scratch & packing output buffer: Depth-4 × $8 \text{ tokens} \times 128 \text{ B (INT4)} = 4 \text{ KiB}$.Every base 32-byte aligned (AKA3003).Reserve 256 bytes trailing read pad (kReadPadBytes = 256u, VECTOR_REG_WIDTH).Synchronization & Hazard Invariants:Use Mode 4 intra-block flags (set_intra_block / wait_intra_block) between AIC and AIVs.Remember the discovery from fused-attention-8pages-findings.md: set_intra_block issues on FLOWCTRL (scalar stream). Always insert an explicit HardEvent::*_S drain before issuing a cross-core flag!Enforce back-edges:MTE3_V: AIV waits for MTE3 egress before overwriting packing slot.kFlagPanelFree: Cube waits for AIVs before Fixpipe overwrites landing slot.kFlagL1Free: MTE2 waits for MTE1 L0B load before overwriting L1 slot.5. Verification Ladder (V0 to V4)V0 Compile-time: Clean compilation with -Wall -Wextra -Werror inside cann92_dev container. All offsets asserted via static_assert.V1 Host Test: Scalar oracle bit-exact validation of $16 \times 1024$ rotation and INT4 packing.V2 CAModel Single Strip:
Execute via scripts/run_modulo3_camodel.sh micro reshape_and_cache_a8.Gate 1: Exit code 0, ZERO invalid ldst addr, ZERO su_ccu_mpu_err.Gate 2: Head & tail GM canaries (0xDEADBEEF) intact on all buffers.Gate 3: Bit-exact match of output keyCache, valueCache, and scaleCache against CPU golden oracle.V4 Trace Audit:
Verify steady-state pipeline cadence: MTE2, Cube Mmad, AIV Vector, and MTE3 executing concurrently in flight from T3 to T7.


T ->       [<-T0->][<-T1->][<-T2->][<-T3->][<-T4->][<-T5->][<-T6->][<-T7->][<-T8->][<-T9->][<-T10>][<-T11>]
MTE2/MTE1: [ LdV0 ][ LdV1 ][ LdV2 ][ LdV3 ][ LdK0 ][ LdK1 ][ LdK2 ][ LdK3 ][ ---- ][ ---- ][ ---- ][ ---- ] GM -> L1 -> L0A, L0B
CUBE:      [ ---- ][ MMV0 ][ MMV1 ][ MMV2 ][ MMV3 ][ MMK0 ][ MMK1 ][ MMK2 ][ MMK3 ][ ---- ][ ---- ][ ---- ] GEMM (32х32 * 32х32)
FIXP:      [ ---- ][ ---- ][ StV0 ][ StV1 ][ StV2 ][ StV3 ][ StK0 ][ StK1 ][ StK2 ][ StK3 ][ ---- ][ ---- ] L0C -> UB
MTE3/MTE1: [ ---- ][ ---- ][ ---- ][ LdV0 ][ LdV1 ][ LdV2 ][ LdV3 ][ LdK0 ][ LdK1 ][ LdK2 ][ LdK3 ][ ---- ] UB -> L1 -> L0A, L0B
CUBE:      [ ---- ][ ---- ][ ---- ][ ---- ][ MMV0 ][ MMV1 ][ MMV2 ][ MMV3 ][ MMK0 ][ MMK1 ][ MMK2 ][ MMK3 ] GEMM (32х32 * 32х32)
GEMM:      [ ---- ][ -x1- ][ -x1- ][ -x2- ][ -x2- ][ -x2- ][ -x2- ][ -x2- ][ -x2- ][ -x1- ][ -x1- ][ -x1- ]

The 1024 channels are chunked into 4 sequential sub-tiles of shape [16 tokens × 256 channels] (4 KiB per chunk in INT8):
- V0, V1, V2, V3: channels [0..255], [256..511], [512..767], [768..1023] of Value.
- K0, K1, K2, K3: channels [0..255], [256..511], [512..767], [768..1023] of Key.

