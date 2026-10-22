// TurboQuant A8 prefill reshape-and-cache, Ascend 950PR / arch35 / dav-3510 /
// CANN 9.2, MIX_AIC_1_2: one Cube core and BOTH Vector cores in a four-stage
// depth-4 modulo pipeline over the eight chunks of one 16-token strip.
//
// What this proves
// ----------------
//   * a four-stage ring (MTE2 -> Cube/Fixpipe -> Vector -> MTE3) in which all
//     four engines plus both Vector cores are in flight together;
//   * an INT8 x INT8 -> INT32 Cube Hadamard factor driven straight out of a
//     Global Memory page staged into L1 with no Unified Buffer hop;
//   * a bit-exact INT4 KV cache: the whole quantisation path is integer and
//     power-of-two, so the host oracle reproduces every nibble exactly for an
//     arbitrary INT8 input, not just for a crafted fixture.
//
// What it deliberately omits
// --------------------------
//   * multi-core tiling. blockDim is 1. The 64-core form is the same kernel
//     with the strip base advanced by GetBlockIdx() * tokensPerCore; nothing
//     here measures it.
//   * ragged shapes. numTokens == blockSize == 16, headSize == 256 and
//     numKvHeads == 4 are compile-time constants (kTokens, kHeadSize,
//     kHeads); the scalar arguments drive the physical cache addressing only.
//   * a non-page-aligned slot mapping: the strip must occupy one whole
//     physical page (see "Slot mapping contract" below).
//
// Target: Ascend950PR_9599 CAModel, __NPU_ARCH__ == 3510, CANN 9.2.0-beta.2.
// Grid:   8 chunks over a depth-4 L1 ring and depth-2 per-core UB rings, so
//         every ring wraps at least twice in one run and every buffer-reuse
//         back edge is contended rather than merely primed and drained
//         (MICROROUTINE_DESIGN_GUIDELINES.md section 3.1).
//
//   engine        owner        work
//   ------------  -----------  --------------------------------------------
//   MTE2          AIC          GM -> L1 page, 4 KiB per chunk, no UB hop
//   MTE1/M/FIX    AIC          L1 -> L0, 8 INT8 Mmads, 8 Fixpipes -> AIV UB
//   Vector        AIV0 + AIV1  sign diagonal, H8 butterfly cascade, absmax,
//                              power-of-two quantisation, INT4 nibble pack
//   MTE3          AIV0 + AIV1  2 KiB packed page + 64 B scale slice -> GM
//
// The rotation
// ------------
// Per head of kHeadSize channels, with ch = c1 * kGroup + c0:
//
//   Y = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32) . x,   normalised by
//   invSqrtLen = 1 / sqrt(headSize)
//
// a randomised Hadamard rotation split across the two engine classes:
//
//   Cube   the inner factor I8 (x) H32, one Mmad per channel group. H32 is
//          the order-32 Sylvester sign matrix, H32[i][j] = (-1)^popcount(i&j),
//          supplied in `tables` as 32x32 INT8 and resident in L0A and L0B.
//   Vector diag(piSigns) then the outer factor H8 (x) I32 as three radix-2
//          butterfly stages over c1.
//
// Both factors are exact in INT32 and in FP32: |x| <= 128, the inner factor
// multiplies the bound by 32 and the outer by 8, so |Y| <= 32768 and every
// intermediate is an integer below 2^24. The whole rotation is therefore
// bit-exact against an integer oracle for *any* INT8 input.
//
// kGroup is 32 because that is the INT8 K0: the arch-3510 INT8 fractal is
// 16 x 32, so a 32-point inner factor is exactly one K fractal and needs no
// zero padding. The outer factor is then order 8 and costs three butterfly
// stages on the Vector core.
//
// One L1 image, two operand roles
// -------------------------------
// For INT8 the NZ image of a 16 x 256 page and the ZN image of its 256 x 16
// transpose are the same bytes: NZ is (M1, K1, M0=16, K0=32) with the fractal
// stored as 16 rows of 32 bytes, and ZN is (K1, N1, N0=16, K0=32) with the
// fractal stored as 16 rows of 32 bytes as well. Channel group c1 is fractal
// c1 of both, at byte offset c1 * 512. So one GM -> L1 ND -> NZ copy per chunk
// produces a page that serves as
//
//   A operand, NZ(16, 32)  -> C[t][c0out]   for the key chunks, and
//   B operand, ZN(32, 16)  -> C[c0out][t]   for the value chunks,
//
// which is exactly the orientation each cache layout needs. The resident H32
// is symmetric, and a row-major 32 x 32 INT8 matrix is byte for byte both its
// own NZ(32, 32) and its own ZN(32, 32) image, so the same 1 KiB of L1 feeds
// L0A and L0B unchanged.
//
// Layout contracts
// ----------------
//   keyCache   key-major. Within the (block, head) page the nibble index is
//              key * headSize + d, so one chunk is 2048 contiguous bytes.
//   valueCache channel-major. Within the (block, head) page the nibble index
//              is ch * blockSize + key, so one chunk is 2048 contiguous bytes.
//   scaleCache per (kv, block, head) a contiguous run of blockSize FP32, one
//              scale per token: index
//              ((kv * numBlocks + block) * numKvHeads + head) * blockSize + key
//              with kv 0 for the key tensor and 1 for the value tensor.
//   Nibbles    two per byte, low nibble first (even element index in bits
//              [3:0], odd in [7:4]), signed INT4 in [-8, 7]. This is the
//              packing `routines/routine_turboquant_dequant4.hpp` unpacks, and
//              the hardware Cast to int4b_t produces it directly.
//
// Work sharing: one chunk per Vector core, not one token half
// ----------------------------------------------------------
// The natural reading of the four-stage schedule gives both Vector cores half
// the tokens of the *same* chunk. That is not expressible against the value
// cache contract above. Channel-major means a channel's blockSize nibbles are
// adjacent, so one channel of a 16-key page is 8 bytes; a core holding only 8
// of the 16 keys owns 4 bytes of every such group and can never assemble a
// 32-byte MTE3 burst, which is the egress granularity. Splitting instead by
// chunk - chunk c is owned by Vector core c & 1 - gives each core whole
// 16-token chunks, so both cache layouts come out as one 2048-byte burst, and
// adjacent chunks still sit on different cores, so the two Vector cores are in
// flight together. Each core then owns two value chunks and two key chunks.
// The UB budget is unchanged: depth 2 x 16 tokens is the same 32 KiB landing
// ring as depth 4 x 8 tokens.
//
// The quantiser is power-of-two, and that is a verification decision
// ------------------------------------------------------------------
// With 2^e <= absmax < 2^(e+1) the scale is 2^(e-2) and the quantised value is
// min(rint(Y * 2^(2-e)), 7), clamped only from above because |Y * 2^(2-e)| < 8
// and rint can reach +8. Y is an integer and 2^(2-e) a power of two, so the
// product is exact in FP32 and round-to-nearest-even on it is exact: the
// device and an integer host oracle agree bit for bit. A 7/absmax scale would
// put the rounding of a hardware `Div` inside the verification contract, where
// a 1-ULP reciprocal can flip a tie and change a nibble. The exponent is
// extracted with integer ops on the FP32 bit pattern - mask the mantissa, and
// form 2^(2-e) as one integer subtract - so no division or logarithm runs.
// invSqrtLen folds into the stored scale, not into the quantised value, since
// scaling Y and absmax together leaves every nibble unchanged.
//
// Slot mapping contract
// ---------------------
// A prefill strip fills one whole physical page: slotMapping[i] must equal
// slotMapping[0] + i and slotMapping[0] must be a multiple of blockSize. The
// kernel reads slotMapping[0] once, on the scalar unit, before the pipeline
// starts; that is a scalar read of caller-supplied Global Memory metadata, the
// same category as the online-softmax leaf's GM-resident index vector, and not
// a scalar read of computed Unified Buffer data (guidelines section 1.2). The
// runner asserts the contract.
//
// Synchronisation
// ---------------
// Marked graph, one token per free slot, every feedback cycle primed before
// the schedule and drained after it. Inside a core the edges are HardEvent
// tokens on the route whose *consumer queue* must stall; across cores they are
// mode-4 intra-block flags. Every cross-core set that means "my pipe work has
// finished" is preceded by a HardEvent::x_S drain of that pipe, because
// set_intra_block issues on the scalar stream and the pipe argument does not
// order it - guidelines section 6.2a, discovered by the eight-page fused
// attention run.
//
// Back edges, all contended in this run:
//   AIC  intra   MTE1_MTE2     L1 page slot free, after MTE1 read it
//   AIC  intra   M_MTE1        L0A/L0B free, after the Mmads
//   AIC  intra   FIX_M         L0C free, after the Fixpipes
//   AIVk -> AIC  kFlagPanelFree   landing panel free, after the FP32 cast
//   AIVk intra   MTE3_V        packed page slot free, after the MTE3 burst

#include <cstdint>
#include <utility>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#endif

namespace turboquant_a8 {

// ------------------------------------------------------------ strip geometry
constexpr std::uint32_t kTokens = 16;   // tokens per strip == blockSize
constexpr std::uint32_t kHeads = 4;     // numKvHeads
constexpr std::uint32_t kHeadSize = 256;
constexpr std::uint32_t kGroup = 32;    // channels per Cube group == INT8 K0
constexpr std::uint32_t kGroups = kHeadSize / kGroup;  // 8, order of the outer factor
constexpr std::uint32_t kChunks = 2u * kHeads;         // 4 value chunks then 4 key chunks
constexpr std::uint32_t kPanel = kTokens * kHeadSize;  // 4096 elements per chunk
constexpr std::uint32_t kPackedBytes = kPanel / 2u;    // 2048 bytes of nibbles
constexpr std::uint32_t kScaleFloats = kTokens;        // one scale per token

// Ring depths. The L1 ring is the pipeline depth; the per-core UB rings are
// half of it because each core owns every other chunk.
constexpr std::uint32_t kStageDepth = 4;
constexpr std::uint32_t kLandDepth = 2;
constexpr std::uint32_t kPackDepth = 2;

static_assert(kHeadSize == kGroups * kGroup);
static_assert(kChunks / 2u == kLandDepth * 2u,
              "each core must wrap its landing ring at least twice");

}  // namespace turboquant_a8

#if defined(__CCE_AICORE__)

// Mode-4 intra-block flags and a Fixpipe that picks a Vector core both need a
// non-1:1 AIC:AIV ratio. At 1:1 there is one Vector core per Cube core, the
// sub-block index is meaningless and the library reroutes cross-core traffic.
#if defined(__MIX_CORE_AIC_RATION__) && (__MIX_CORE_AIC_RATION__ == 1)
#error "MIX_AIC_1_2 is required: a 1:1 AIC:AIV ratio has no second Vector core"
#endif

namespace turboquant_a8 {

// --------------------------------------------------------------- chunk traits
// Chunks 0..3 are the value heads, 4..7 the key heads, which is the order the
// four-stage schedule runs them in. Every trait is a compile-time value, so no
// column of the schedule contains a runtime predicate or a modulo.
template <std::uint32_t C>
constexpr bool kIsKey = (C >= kHeads);
template <std::uint32_t C>
constexpr std::uint32_t kChunkHead = C % kHeads;
template <std::uint32_t C>
constexpr std::uint32_t kOwner = C & 1u;
template <std::uint32_t C>
constexpr std::uint32_t kL1Slot = C % kStageDepth;
template <std::uint32_t C>
constexpr std::uint32_t kLandSlot = (C / 2u) % kLandDepth;
template <std::uint32_t C>
constexpr std::uint32_t kPackSlot = (C / 2u) % kPackDepth;

// --------------------------------------------------------------- UB arena
// Rule AKA3003: every operand base 32-byte aligned, one DataBlock of bank skew
// between live slots, and VECTOR_REG_WIDTH of addressable UB behind the last
// operand because a level-2 vector op reads a whole vector register per source
// unmasked. One map serves both Vector cores, and the Cube forms its Fixpipe
// destination from the same constants.
constexpr std::uint32_t kBlock = 32u;
constexpr std::uint32_t kBankSkewBytes = 32u;
constexpr std::uint32_t kReadPadBytes = 256u;  // VECTOR_REG_WIDTH on dav-3510
constexpr std::uint32_t kUbWindow = 192u * 1024u;
constexpr std::uint32_t kLanesPerBlock = kBlock / sizeof(float);  // 8

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkewBytes;
}

constexpr std::uint32_t kPanelBytes = kPanel * 4u;        // 16384, FP32 or INT32
constexpr std::uint32_t kLandSlotBytes = kPanelBytes + kBankSkewBytes;
constexpr std::uint32_t kPackSlotBytes =
    RoundUpBlock(kPackedBytes + kScaleFloats * 4u) + kBankSkewBytes;

// The Fixpipe landing ring stays at UB 0 so the Cube forms a raw __ubuf__
// pointer without depending on any Vector-side allocation.
constexpr std::uint32_t kUbLandRing = 0u;
constexpr std::uint32_t kUbLandBytes = kLandDepth * kLandSlotBytes;

// piSigns as delivered, then the two replicated sign panels. A panel is built
// once, outside the pipeline, because the two cache layouts need the sign
// vector expanded along different axes:
//   key   S[t][ch]        = piSigns[ch]           (kHeadSize-periodic)
//   value S[c1][c0][t]    = piSigns[c1*32 + c0]   (each sign over kTokens lanes)
constexpr std::uint32_t kUbSigns = NextSlot(kUbLandRing, kUbLandBytes);
constexpr std::uint32_t kUbSignKey = NextSlot(kUbSigns, kHeadSize * 4u);
constexpr std::uint32_t kUbSignValue = NextSlot(kUbSignKey, kPanelBytes);

// Working panels. Single buffered: reuse across chunks is a same-pipe Vector
// dependency, which the Vector queue orders in issue order, not a cross-engine
// hazard (guidelines section 3.2).
//   kUbCast   FP32 image of the INT32 landing panel, later the INT32 quantised
//             panel, which is the same 16 KiB read as 4-byte lanes
//   kUbWorkA  butterfly stage 1 and 3 destination, then |Y|, then Y * inv
//   kUbWorkB  butterfly stage 2 destination, and the rotated result Y
constexpr std::uint32_t kUbCast = NextSlot(kUbSignValue, kPanelBytes);
constexpr std::uint32_t kUbWorkA = NextSlot(kUbCast, kPanelBytes);
constexpr std::uint32_t kUbWorkB = NextSlot(kUbWorkA, kPanelBytes);

// Row-wise max ladder for the key layout, where the reduction runs along the
// fast axis and the partials shrink by half each step. The value layout folds
// in place and needs none of these.
constexpr std::uint32_t kUbFold128 = NextSlot(kUbWorkB, kPanelBytes);
constexpr std::uint32_t kUbFold64 = NextSlot(kUbFold128, kTokens * 128u * 4u);
constexpr std::uint32_t kUbFold32 = NextSlot(kUbFold64, kTokens * 64u * 4u);
constexpr std::uint32_t kUbFold16 = NextSlot(kUbFold32, kTokens * 32u * 4u);
constexpr std::uint32_t kUbFold8 = NextSlot(kUbFold16, kTokens * 16u * 4u);

// FP16 staging for the nibble pack: the hardware Cast to int4b_t takes a
// 2-byte source. The staged values are exact integers in [-8, 7], so the
// INT32 -> FP16 -> INT4 chain rounds once, in the INT32 cast, and the two
// narrowing steps are exact. The INT16 source form compiles to a deprecated
// vcvt_s162s4 intrinsic, which is the only reason this goes through FP16.
constexpr std::uint32_t kUbHalf = NextSlot(kUbFold8, kTokens * 8u * 4u);

// Logically scalar state, one lane-replicated DataBlock each where the API
// reads whole blocks. kScaleFloats is 16 == two DataBlocks.
constexpr std::uint32_t kUbAbsMax = NextSlot(kUbHalf, kPanel * 2u);
constexpr std::uint32_t kUbExpBits = NextSlot(kUbAbsMax, kScaleFloats * 4u);
constexpr std::uint32_t kUbPow = NextSlot(kUbExpBits, kScaleFloats * 4u);
constexpr std::uint32_t kUbInv = NextSlot(kUbPow, kScaleFloats * 4u);
constexpr std::uint32_t kUbInvRep = NextSlot(kUbInv, kScaleFloats * 4u);
constexpr std::uint32_t kUbInvBcast = NextSlot(kUbInvRep, 4u * kScaleFloats * 4u);
constexpr std::uint32_t kUbOne = NextSlot(kUbInvBcast, kTokens * kLanesPerBlock * 4u);
constexpr std::uint32_t kUbExpMask = NextSlot(kUbOne, kScaleFloats * 4u);
constexpr std::uint32_t kUbInvSeed = NextSlot(kUbExpMask, kScaleFloats * 4u);

// Packed egress ring: 2 KiB of nibbles plus the 64-byte scale slice, so one
// back edge covers both MTE3 bursts of a chunk.
constexpr std::uint32_t kUbPackRing = NextSlot(kUbInvSeed, kScaleFloats * 4u);
constexpr std::uint32_t kArenaEnd = kUbPackRing + kPackDepth * kPackSlotBytes;

static_assert(kUbLandRing % kBlock == 0u && kUbSigns % kBlock == 0u &&
                  kUbSignKey % kBlock == 0u && kUbSignValue % kBlock == 0u &&
                  kUbCast % kBlock == 0u && kUbWorkA % kBlock == 0u &&
                  kUbWorkB % kBlock == 0u && kUbFold128 % kBlock == 0u &&
                  kUbFold64 % kBlock == 0u && kUbFold32 % kBlock == 0u &&
                  kUbFold16 % kBlock == 0u && kUbFold8 % kBlock == 0u &&
                  kUbHalf % kBlock == 0u && kUbAbsMax % kBlock == 0u &&
                  kUbExpBits % kBlock == 0u && kUbPow % kBlock == 0u &&
                  kUbInv % kBlock == 0u &&
                  kUbInvRep % kBlock == 0u && kUbInvBcast % kBlock == 0u &&
                  kUbOne % kBlock == 0u && kUbExpMask % kBlock == 0u &&
                  kUbInvSeed % kBlock == 0u && kUbPackRing % kBlock == 0u,
              "every UB operand base must be 32-byte aligned");
static_assert(kLandSlotBytes % kBlock == 0u && kPackSlotBytes % kBlock == 0u,
              "every ring slot stride must be a whole number of DataBlocks");
static_assert(kLandSlotBytes % 256u != 0u && kPackSlotBytes % 256u != 0u,
              "a 256-byte ring stride puts every slot on the same bank phase");
static_assert(kArenaEnd + kReadPadBytes <= kUbWindow,
              "UB arena leaves no VECTOR_REG_WIDTH read pad behind the state");
static_assert(kUbInvSeed + kReadPadBytes <= kUbWindow,
              "the last state block must have a whole vector register behind it");
static_assert(kPackedBytes % kBlock == 0u && kScaleFloats * 4u % kBlock == 0u,
              "both MTE3 bursts must be whole 32-byte bursts");
static_assert(kTokens * kLanesPerBlock * 4u == kTokens * kBlock,
              "a Brcb repeat writes eight DataBlocks per source element");

template <std::uint32_t Slot>
constexpr std::uint32_t kLandBase = kUbLandRing + Slot * kLandSlotBytes;
template <std::uint32_t Slot>
constexpr std::uint32_t kPackBase = kUbPackRing + Slot * kPackSlotBytes;
template <std::uint32_t Slot>
constexpr std::uint32_t kScaleBase = kPackBase<Slot> + kPackedBytes;

// --------------------------------------------------------------- L1 arena
// H32 is page invariant and reaches L1 once. The page ring has one slot per
// pipeline stage. L1 operands need 32-byte alignment; the 512-byte rule
// applies to L0A and L0B, and the fractal offsets inside a slot are multiples
// of 512 from a 32-byte aligned base, which is the layout the fused attention
// kernel validated.
constexpr std::uint32_t kL1H = 0u;
constexpr std::uint32_t kL1HBytes = kGroup * kGroup;  // 1024, INT8
constexpr std::uint32_t kL1PageBytes = kTokens * kHeadSize;  // 4096, INT8
constexpr std::uint32_t kL1Ring = NextSlot(kL1H, kL1HBytes);
constexpr std::uint32_t kL1SlotBytes = kL1PageBytes + kBankSkewBytes;
constexpr std::uint32_t kL1End = kL1Ring + kStageDepth * kL1SlotBytes;
constexpr std::uint32_t kFractalBytes = kTokens * kGroup;  // 512

static_assert(kL1H % kBlock == 0u && kL1Ring % kBlock == 0u &&
                  kL1SlotBytes % kBlock == 0u,
              "every L1 operand base must be 32-byte aligned");
static_assert(kL1SlotBytes % 256u != 0u, "L1 slot stride collapses the bank phase");
static_assert(kL1End <= 512u * 1024u, "L1 ring exceeds the L1 buffer");
static_assert(kFractalBytes * kGroups == kL1PageBytes,
              "the page must be a whole number of INT8 fractals");

template <std::uint32_t Slot>
constexpr std::uint32_t kL1PageBase = kL1Ring + Slot * kL1SlotBytes;

// --------------------------------------------------------------- L0 arena
// H32 stays resident in both L0A and L0B: the key chunks take it as the B
// operand and the value chunks as the A operand, and nothing writes that
// region after the prologue. The page fractals land behind it.
constexpr std::uint32_t kL0HElems = kGroup * kGroup;              // 1024 INT8
constexpr std::uint32_t kL0PageElems = kGroups * kFractalBytes;   // 4096 INT8
constexpr std::uint32_t kL0Elems = kL0HElems + kL0PageElems;
constexpr std::uint32_t kL0CElems = kGroups * kTokens * kGroup;   // 4096 INT32
static_assert(kL0HElems % 512u == 0u, "the L0 page region must stay 512-byte aligned");

// ------------------------------------------------- cross-core flag ids
// Mode 4 ids are per sub block: 0..15 belong to AIV0 and 16..31 to AIV1, the
// split GetIntraFlagId uses. 13, 14 and their +16 images are library reserved.
template <std::uint32_t Owner, std::uint32_t Slot>
constexpr std::uint16_t kFlagProductReady =
    static_cast<std::uint16_t>(Owner * 16u + Slot);          // 0..1, 16..17 AIC -> AIV
template <std::uint32_t Owner, std::uint32_t Slot>
constexpr std::uint16_t kFlagPanelFree =
    static_cast<std::uint16_t>(Owner * 16u + 4u + Slot);     // 4..5, 20..21 AIV -> AIC

// Intra-core HardEvent ids. 6 and 7 are reserved, so slot ids stay below 4 and
// the prologue and epilogue use 4.
constexpr std::int32_t kEvtL0 = 0;
constexpr std::int32_t kEvtEpilogue = 4;

// ------------------------------------------------- compile-time column list
using ChunkSeq = std::make_integer_sequence<std::uint32_t, kChunks>;

// =========================================================== AIC: Cube core
// MTE2 ingress of the raw INT8 page into the L1 ring, then per chunk eight
// L1 -> L0 fractal loads, eight INT8 Mmads and eight Fixpipes into the owning
// Vector core's landing panel.
struct CubeEngine {
    __gm__ const std::int8_t* key;
    __gm__ const std::int8_t* value;
    std::uint32_t row_pitch;   // numKvHeads * headSize, elements
    std::uint32_t head_size;   // headSize, elements
};

// MTE2: one strided ND -> NZ burst per chunk. The source is the headSize-wide
// window of head h inside the kTokens x (numKvHeads * headSize) activation
// block, so the row pitch is a runtime stride and the window is expressed with
// MakePatternLayout rather than MakeFrameLayout, whose NDExt row pitch is
// pinned to the column count.
template <std::uint32_t Chunk>
__aicore__ inline void StagePage(CubeEngine& engine)
{
    using namespace AscendC::Te;
    constexpr std::uint32_t slot = kL1Slot<Chunk>;
    constexpr std::int32_t evt = static_cast<std::int32_t>(slot);

    __gm__ const std::int8_t* base = kIsKey<Chunk> ? engine.key : engine.value;
    auto* src_ptr = const_cast<__gm__ std::int8_t*>(base) +
                    kChunkHead<Chunk> * engine.head_size;
    auto* dst_ptr = reinterpret_cast<__cbuf__ std::int8_t*>(kL1PageBase<slot>);

    auto shape = MakeShape(MakeShape(_1{}, kTokens), MakeShape(_1{}, kHeadSize));
    auto stride = MakeStride(MakeStride(_0{}, engine.row_pitch), MakeStride(_0{}, _1{}));
    auto src = MakeTensor(MakeMemPtr(src_ptr),
                          MakePatternLayout<NDExtLayoutPtn,
                                            LayoutTrait<AscendC::Std::ignore_t, _1>>(shape, stride));
    auto dst = MakeTensor(MakeMemPtr(dst_ptr),
                          MakeFrameLayout<NZLayoutPtn, std::int8_t>(kTokens, kHeadSize));

    // The L1 slot is free once MTE1 has read the page it held kStageDepth
    // chunks ago. WaitFlag<MTE1_MTE2> stalls the MTE2 queue, which is the
    // queue that must not run ahead here: nothing else gates it.
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(evt);
    Copy(MakeCopy(CopyGM2L1{}), dst, src);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(evt);
}

// MTE1 + Cube + Fixpipe for one channel group. Key and value chunks differ
// only in which operand is the page and in the Fixpipe destination geometry;
// the arithmetic is the same H32 applied to the group.
template <std::uint32_t Chunk, std::uint32_t G>
__aicore__ inline void GemmGroupLoad(__ca__ std::int8_t* l0a, __cb__ std::int8_t* l0b)
{
    using namespace AscendC::Te;
    constexpr std::uint32_t slot = kL1Slot<Chunk>;
    auto* page = reinterpret_cast<__cbuf__ std::int8_t*>(kL1PageBase<slot> + G * kFractalBytes);

    if constexpr (kIsKey<Chunk>) {
        // A = the page group, NZ(kTokens, kGroup): A[t][c0] = x[t][c1*32 + c0].
        auto src = MakeTensor(MakeMemPtr(page),
                              MakeFrameLayout<NZLayoutPtn, std::int8_t>(kTokens, kGroup));
        auto dst = MakeTensor(MakeMemPtr(l0a + kL0HElems + G * kFractalBytes),
                              MakeFrameLayout<NZLayoutPtn, std::int8_t>(kTokens, kGroup));
        Copy(MakeCopy(CopyL12L0A{}), dst, src);
    } else {
        // B = the page group, ZN(kGroup, kTokens): B[c0][t] = x[t][c1*32 + c0],
        // the same bytes read with the transposed meaning.
        auto src = MakeTensor(MakeMemPtr(page),
                              MakeFrameLayout<ZNLayoutPtn, std::int8_t>(kGroup, kTokens));
        auto dst = MakeTensor(MakeMemPtr(l0b + kL0HElems + G * kFractalBytes),
                              MakeFrameLayout<ZNLayoutPtn, std::int8_t>(kGroup, kTokens));
        Copy(MakeCopy(CopyL12L0B{}), dst, src);
    }
}

template <std::uint32_t Chunk, std::uint32_t G>
__aicore__ inline void GemmGroupCompute(__ca__ std::int8_t* l0a, __cb__ std::int8_t* l0b,
                                        __cc__ std::int32_t* l0c)
{
    using namespace AscendC::Te;
    constexpr std::uint32_t land = kLandSlot<Chunk>;
    auto* c_ptr = l0c + G * kTokens * kGroup;

    if constexpr (kIsKey<Chunk>) {
        auto a = MakeTensor(MakeMemPtr(l0a + kL0HElems + G * kFractalBytes),
                            MakeFrameLayout<NZLayoutPtn, std::int8_t>(kTokens, kGroup));
        auto b = MakeTensor(MakeMemPtr(l0b),
                            MakeFrameLayout<ZNLayoutPtn, std::int8_t>(kGroup, kGroup));
        auto c = MakeTensor(MakeMemPtr(c_ptr),
                            MakeFrameLayout<NZLayoutPtn, 16>(kTokens, kGroup));
        MmadParams mp(static_cast<int>(kTokens), static_cast<int>(kGroup),
                      static_cast<int>(kGroup), 3, true);
        Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(mp), c, a, b);

        // Key panel is token major, [t][c1][c0], so group G lands as a
        // kTokens x kGroup block with a kHeadSize element row pitch.
        auto* ub = reinterpret_cast<__ubuf__ std::int32_t*>(kLandBase<land> +
                                                            G * kGroup * 4u);
        auto shape = MakeShape(MakeShape(_1{}, kTokens), MakeShape(_1{}, kGroup));
        auto stride = MakeStride(MakeStride(_0{}, kHeadSize), MakeStride(_0{}, _1{}));
        auto y = MakeTensor(MakeMemPtr(ub),
                            MakePatternLayout<NDExtLayoutPtn,
                                              LayoutTrait<AscendC::Std::ignore_t, _1>>(shape, stride));
        FixpipeParams fp;
        fp.unitFlag = 3;
        fp.subBlockId = (kOwner<Chunk> == 1u);
        Copy(MakeCopy(CopyL0C2UB{}).with(fp), y, c);
    } else {
        auto a = MakeTensor(MakeMemPtr(l0a),
                            MakeFrameLayout<NZLayoutPtn, std::int8_t>(kGroup, kGroup));
        auto b = MakeTensor(MakeMemPtr(l0b + kL0HElems + G * kFractalBytes),
                            MakeFrameLayout<ZNLayoutPtn, std::int8_t>(kGroup, kTokens));
        auto c = MakeTensor(MakeMemPtr(c_ptr),
                            MakeFrameLayout<NZLayoutPtn, 16>(kGroup, kTokens));
        MmadParams mp(static_cast<int>(kGroup), static_cast<int>(kTokens),
                      static_cast<int>(kGroup), 3, true);
        Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(mp), c, a, b);

        // Value panel is channel major, [c1][c0][t], so group G lands as a
        // contiguous kGroup x kTokens block.
        auto* ub = reinterpret_cast<__ubuf__ std::int32_t*>(kLandBase<land> +
                                                            G * kGroup * kTokens * 4u);
        auto y = MakeTensor(MakeMemPtr(ub), MakeFrameLayout<NDExtLayoutPtn>(kGroup, kTokens));
        FixpipeParams fp;
        fp.unitFlag = 3;
        fp.subBlockId = (kOwner<Chunk> == 1u);
        Copy(MakeCopy(CopyL0C2UB{}).with(fp), y, c);
    }
}

template <std::uint32_t Chunk, std::uint32_t... Gs>
__aicore__ inline void GemmGroupLoads(__ca__ std::int8_t* l0a, __cb__ std::int8_t* l0b,
                                      std::integer_sequence<std::uint32_t, Gs...>)
{
    (GemmGroupLoad<Chunk, Gs>(l0a, l0b), ...);
}

template <std::uint32_t Chunk, std::uint32_t... Gs>
__aicore__ inline void GemmGroupComputes(__ca__ std::int8_t* l0a, __cb__ std::int8_t* l0b,
                                         __cc__ std::int32_t* l0c,
                                         std::integer_sequence<std::uint32_t, Gs...>)
{
    (GemmGroupCompute<Chunk, Gs>(l0a, l0b, l0c), ...);
}

template <std::uint32_t Chunk>
__aicore__ inline void GemmChunk(__ca__ std::int8_t* l0a, __cb__ std::int8_t* l0b,
                                 __cc__ std::int32_t* l0c)
{
    constexpr std::uint32_t slot = kL1Slot<Chunk>;
    constexpr std::uint32_t land = kLandSlot<Chunk>;
    constexpr std::uint32_t owner = kOwner<Chunk>;
    using GroupSeq = std::make_integer_sequence<std::uint32_t, kGroups>;

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<std::int32_t>(slot));
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
    GemmGroupLoads<Chunk>(l0a, l0b, GroupSeq{});
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtL0);
    // Issued on MTE1 and behind the eight fractal loads, so the page slot is
    // announced free only once they have read it.
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<std::int32_t>(slot));

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtL0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEvtL0);
    // The owning Vector core must have consumed the previous occupant of this
    // landing panel before Fixpipe overwrites it.
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFree<owner, land>);
    AscendC::SetMMRowMajor();
    GemmGroupComputes<Chunk>(l0a, l0b, l0c, GroupSeq{});
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEvtL0);

    // A mode-4 set runs on the scalar stream, so drain the Fixpipe into it
    // first; without this the announcement overtakes the eight Fixpipes it
    // reports (guidelines section 6.2a).
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtL0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtL0);
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagProductReady<owner, land>);
}

// Column j of the schedule: MTE2 stages chunk j while the Cube multiplies
// chunk j - 1. Neither depends on the other, so MTE2, MTE1, Cube and Fixpipe
// are all free to be in flight together.
template <std::uint32_t Column>
__aicore__ inline void StepColumnCube(CubeEngine& engine, __ca__ std::int8_t* l0a,
                                      __cb__ std::int8_t* l0b, __cc__ std::int32_t* l0c)
{
    if constexpr (Column < kChunks) StagePage<Column>(engine);
    if constexpr (Column >= 1u) GemmChunk<Column - 1u>(l0a, l0b, l0c);
}

template <std::uint32_t... Columns>
__aicore__ inline void StepColumnsCube(CubeEngine& engine, __ca__ std::int8_t* l0a,
                                       __cb__ std::int8_t* l0b, __cc__ std::int32_t* l0c,
                                       std::integer_sequence<std::uint32_t, Columns...>)
{
    (StepColumnCube<Columns>(engine, l0a, l0b, l0c), ...);
}

__aicore__ inline void run_cube_core(__gm__ const std::int8_t* key,
                                     __gm__ const std::int8_t* value,
                                     __gm__ const std::int32_t* tables,
                                     std::uint32_t num_kv_heads,
                                     std::uint32_t head_size)
{
    using namespace AscendC::Te;
    __ca__ std::int8_t l0_a[kL0Elems] __attribute__((aligned(512)));
    __cb__ std::int8_t l0_b[kL0Elems] __attribute__((aligned(512)));
    __cc__ std::int32_t l0_c[kL0CElems] __attribute__((aligned(1024)));

    CubeEngine engine{key, value, num_kv_heads * head_size, head_size};

    // H32 reaches L1 once as a plain 1 KiB ND copy. A row-major symmetric
    // 32x32 INT8 matrix is byte for byte both its NZ(32,32) and its ZN(32,32)
    // image, so the same bytes feed L0A for the value chunks and L0B for the
    // key chunks with no second staging path.
    auto* l1_h = reinterpret_cast<__cbuf__ std::int8_t*>(kL1H);
    auto h_src = MakeTensor(MakeMemPtr(reinterpret_cast<__gm__ std::int8_t*>(
                                const_cast<__gm__ std::int32_t*>(tables))),
                            MakeFrameLayout<NDExtLayoutPtn>(kGroup, kGroup));
    auto h_dst = MakeTensor(MakeMemPtr(l1_h), MakeFrameLayout<NDExtLayoutPtn>(kGroup, kGroup));
    Copy(MakeCopy(CopyGM2L1{}), h_dst, h_src);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtEpilogue);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtEpilogue);

    auto ha_src = MakeTensor(MakeMemPtr(l1_h),
                             MakeFrameLayout<NZLayoutPtn, std::int8_t>(kGroup, kGroup));
    auto ha_dst = MakeTensor(MakeMemPtr(l0_a),
                             MakeFrameLayout<NZLayoutPtn, std::int8_t>(kGroup, kGroup));
    Copy(MakeCopy(CopyL12L0A{}), ha_dst, ha_src);
    auto hb_src = MakeTensor(MakeMemPtr(l1_h),
                             MakeFrameLayout<ZNLayoutPtn, std::int8_t>(kGroup, kGroup));
    auto hb_dst = MakeTensor(MakeMemPtr(l0_b),
                             MakeFrameLayout<ZNLayoutPtn, std::int8_t>(kGroup, kGroup));
    Copy(MakeCopy(CopyL12L0B{}), hb_dst, hb_src);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtEpilogue);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtEpilogue);

    // Feedback cycle priming: every L1 page slot starts free, and L0A/L0B and
    // L0C are free for the first chunk.
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(3);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEvtL0);

    StepColumnsCube(engine, l0_a, l0_b, l0_c,
                    std::make_integer_sequence<std::uint32_t, kChunks + 1u>{});

    // Drain every primed token and every panel-free token the Vector cores
    // handed back, so no event bit and no intra-block flag leaks into the next
    // launch. These waits also retire the last Fixpipes before the kernel ends.
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(3);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEvtL0);
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFree<0u, 0u>);
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFree<0u, 1u>);
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFree<1u, 0u>);
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFree<1u, 1u>);
}

// ======================================================= AIV: Vector cores
// Both cores run this body; the chunk list is filtered by owner, so each core
// handles two value chunks and two key chunks and nothing else.
struct VectorEngine {
    AscendC::GlobalTensor<std::int8_t> key_cache;
    AscendC::GlobalTensor<std::int8_t> value_cache;
    AscendC::GlobalTensor<float> scale_cache;
    std::uint32_t page_bytes;    // byte base of (block, head 0) in either cache
    std::uint32_t head_bytes;    // blockSize * headSize / 2
    std::uint32_t key_scale;     // float base of (kv 0, block, head 0)
    std::uint32_t value_scale;   // float base of (kv 1, block, head 0)
    std::uint32_t head_scale;    // blockSize
    float inv_sqrt_len;
};

constexpr auto kUb = AscendC::TPosition::VECCALC;

template <class T>
__aicore__ inline AscendC::LocalTensor<T> View(std::uint32_t base, std::uint32_t count)
{
    return AscendC::LocalTensor<T>(kUb, base, count);
}

// --- butterfly: value layout -------------------------------------------------
// The panel is [c1][c0][t], so a radix-2 stage over c1 at distance d pairs two
// contiguous halves of each 2*d*kGroup*kTokens element period: sums into the
// low half, differences into the high half. Three stages with d = 4, 2, 1
// compose to H8 with no index permutation, because stage d transforms exactly
// bit log2(d) of the c1 index.
template <std::uint32_t Period, std::uint32_t... Ps>
__aicore__ inline void ValueStageEmit(const AscendC::LocalTensor<float>& dst,
                                      const AscendC::LocalTensor<float>& src,
                                      std::integer_sequence<std::uint32_t, Ps...>)
{
    constexpr std::int32_t half = static_cast<std::int32_t>(Period / 2u);
    ((AscendC::Add(dst[Ps * Period], src[Ps * Period], src[Ps * Period + Period / 2u], half),
      AscendC::Sub(dst[Ps * Period + Period / 2u], src[Ps * Period],
                   src[Ps * Period + Period / 2u], half)),
     ...);
}

template <std::uint32_t Period>
__aicore__ inline void ValueButterflyStage(const AscendC::LocalTensor<float>& dst,
                                           const AscendC::LocalTensor<float>& src)
{
    static_assert(kPanel % Period == 0u, "the butterfly period must tile the panel");
    ValueStageEmit<Period>(dst, src,
                           std::make_integer_sequence<std::uint32_t, kPanel / Period>{});
}

// --- butterfly: key layout ---------------------------------------------------
// The panel is [t][c1][c0], so a stage over c1 at distance d pairs elements
// d*kGroup apart inside each token row. One level-0 call covers every token at
// once when the pairing period tiles the panel uniformly; the widest stage,
// whose period is a whole token row, needs two calls because a repeat covers
// at most one vector register of 64 FP32 lanes.
template <std::uint32_t Period, std::uint32_t... Cs>
__aicore__ inline void KeyStageEmit(const AscendC::LocalTensor<float>& dst,
                                    const AscendC::LocalTensor<float>& src,
                                    std::integer_sequence<std::uint32_t, Cs...>)
{
    constexpr std::uint32_t half = Period / 2u;
    constexpr std::int32_t mask = static_cast<std::int32_t>(half > 64u ? 64u : half);
    constexpr std::uint8_t repeats = static_cast<std::uint8_t>(kPanel / Period);
    AscendC::BinaryRepeatParams rp;
    rp.dstBlkStride = 1;
    rp.src0BlkStride = 1;
    rp.src1BlkStride = 1;
    rp.dstRepStride = static_cast<std::uint8_t>(Period / kLanesPerBlock);
    rp.src0RepStride = rp.dstRepStride;
    rp.src1RepStride = rp.dstRepStride;
    ((AscendC::Add<float>(dst[Cs * mask], src[Cs * mask], src[half + Cs * mask],
                          mask, repeats, rp),
      AscendC::Sub<float>(dst[half + Cs * mask], src[Cs * mask], src[half + Cs * mask],
                          mask, repeats, rp)),
     ...);
}

template <std::uint32_t Period>
__aicore__ inline void KeyButterflyStage(const AscendC::LocalTensor<float>& dst,
                                         const AscendC::LocalTensor<float>& src)
{
    constexpr std::uint32_t half = Period / 2u;
    constexpr std::uint32_t mask = half > 64u ? 64u : half;
    static_assert(Period / kLanesPerBlock <= 255u, "repeat stride is a uint8 block count");
    KeyStageEmit<Period>(dst, src, std::make_integer_sequence<std::uint32_t, half / mask>{});
}

// Row-wise max ladder for the key layout: fold N partials per token down to
// N/2, from a SrcPitch-strided source into a DstPitch-strided destination.
template <std::uint32_t N, std::uint32_t SrcPitch, std::uint32_t DstPitch,
          std::uint32_t... Cs>
__aicore__ inline void KeyFoldEmit(const AscendC::LocalTensor<float>& dst,
                                   const AscendC::LocalTensor<float>& src,
                                   std::integer_sequence<std::uint32_t, Cs...>)
{
    constexpr std::uint32_t half = N / 2u;
    constexpr std::int32_t mask = static_cast<std::int32_t>(half > 64u ? 64u : half);
    AscendC::BinaryRepeatParams rp;
    rp.dstBlkStride = 1;
    rp.src0BlkStride = 1;
    rp.src1BlkStride = 1;
    rp.dstRepStride = static_cast<std::uint8_t>(DstPitch / kLanesPerBlock);
    rp.src0RepStride = static_cast<std::uint8_t>(SrcPitch / kLanesPerBlock);
    rp.src1RepStride = rp.src0RepStride;
    (AscendC::Max<float>(dst[Cs * mask], src[Cs * mask], src[half + Cs * mask], mask,
                         static_cast<std::uint8_t>(kTokens), rp),
     ...);
}

template <std::uint32_t N, std::uint32_t SrcPitch, std::uint32_t DstPitch>
__aicore__ inline void KeyFold(const AscendC::LocalTensor<float>& dst,
                               const AscendC::LocalTensor<float>& src)
{
    constexpr std::uint32_t half = N / 2u;
    constexpr std::uint32_t mask = half > 64u ? 64u : half;
    static_assert(DstPitch >= kLanesPerBlock, "a fold destination pitch is a block count");
    KeyFoldEmit<N, SrcPitch, DstPitch>(
        dst, src, std::make_integer_sequence<std::uint32_t, half / mask>{});
}

// In-place halving fold for the value layout, where the reduction axis is the
// slow one and every step is one contiguous level-2 op.
template <std::uint32_t... Steps>
__aicore__ inline void ValueFoldEmit(const AscendC::LocalTensor<float>& buf,
                                     std::integer_sequence<std::uint32_t, Steps...>)
{
    // Step s folds kPanel >> s elements onto kPanel >> (s + 1).
    (AscendC::Max(buf, buf, buf[kPanel >> (Steps + 1u)],
                  static_cast<std::int32_t>(kPanel >> (Steps + 1u))),
     ...);
}

// One level-0 op per vector register of a token row: the key panel has the
// token index outermost, so the per-token inverse block advances once per
// repeat (src1RepStride 1) and is read by every DataBlock of that repeat
// (src1BlkStride 0).
template <std::uint32_t... Qs>
__aicore__ inline void KeyQuantiseEmit(const AscendC::LocalTensor<float>& dst,
                                       const AscendC::LocalTensor<float>& src,
                                       const AscendC::LocalTensor<float>& inv_bcast,
                                       const AscendC::BinaryRepeatParams& rp,
                                       std::integer_sequence<std::uint32_t, Qs...>)
{
    (AscendC::Mul<float>(dst[Qs * 64u], src[Qs * 64u], inv_bcast, 64,
                         static_cast<std::uint8_t>(kTokens), rp),
     ...);
}

// Repeated doubling of a kHeadSize-element prefix up to the whole panel. Muls
// by 1.0f is an exact FP32 copy, including for signed zeros, and stays on the
// Vector pipe, so no cross-pipe edge is needed between the steps.
constexpr std::uint32_t kTokensLog2 = 4;  // kTokens == 1 << kTokensLog2
static_assert((1u << kTokensLog2) == kTokens, "the doubling ladder must reach kPanel");

template <std::uint32_t... Steps>
__aicore__ inline void DoubleEmit(const AscendC::LocalTensor<float>& buf,
                                  std::integer_sequence<std::uint32_t, Steps...>)
{
    ((AscendC::PipeBarrier<PIPE_V>(),
      AscendC::Muls(buf[kHeadSize << Steps], buf, 1.0f,
                    static_cast<std::int32_t>(kHeadSize << Steps))),
     ...);
}

// --- the Vector stage --------------------------------------------------------
template <std::uint32_t Chunk>
__aicore__ inline void RotateAndQuantise(VectorEngine& engine)
{
    constexpr std::uint32_t land = kLandSlot<Chunk>;
    constexpr std::uint32_t pack = kPackSlot<Chunk>;
    constexpr std::uint32_t owner = kOwner<Chunk>;
    constexpr std::int32_t pack_evt = static_cast<std::int32_t>(pack);
    constexpr std::int32_t all = static_cast<std::int32_t>(kPanel);

    auto landing = View<std::int32_t>(kLandBase<land>, kPanel);
    auto cast = View<float>(kUbCast, kPanel);
    auto work_a = View<float>(kUbWorkA, kPanel);
    auto work_b = View<float>(kUbWorkB, kPanel);
    auto signs = View<float>(kIsKey<Chunk> ? kUbSignKey : kUbSignValue, kPanel);

    AscendC::CrossCoreWaitFlag<4, PIPE_V>(kFlagProductReady<owner, land>);
    // Fixpipe lands INT32: L0C INT32 -> UB FP32 is not a supported Fixpipe
    // conversion on this part, the pairs are (FP32, FP32) and (INT32, INT32),
    // so the cast is one contiguous Vector op here. Exact: every value is an
    // integer below 2^24.
    AscendC::Cast(cast, landing, AscendC::RoundMode::CAST_NONE, kPanel);
    // The landing panel is dead from here on, so release it before the rest of
    // the stage runs and let the Cube refill it immediately. The drain makes
    // the scalar-stream set wait for the Vector queue to retire the cast.
    AscendC::SetFlag<AscendC::HardEvent::V_S>(pack_evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(pack_evt);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagPanelFree<owner, land>);

    // diag(piSigns) between the two Hadamard factors.
    AscendC::Mul(work_a, cast, signs, all);

    // The outer factor, three radix-2 stages, ping-ponging A -> B -> A -> B.
    if constexpr (kIsKey<Chunk>) {
        KeyButterflyStage<kGroup * 8u>(work_b, work_a);
        KeyButterflyStage<kGroup * 4u>(work_a, work_b);
        KeyButterflyStage<kGroup * 2u>(work_b, work_a);
    } else {
        ValueButterflyStage<kTokens * kGroup * 8u>(work_b, work_a);
        ValueButterflyStage<kTokens * kGroup * 4u>(work_a, work_b);
        ValueButterflyStage<kTokens * kGroup * 2u>(work_b, work_a);
    }

    // Row-wise absolute maximum, one value per token, left contiguous so the
    // scale slice is a single 64-byte MTE3 burst.
    auto abs_max = View<float>(kUbAbsMax, kScaleFloats);
    auto one = View<float>(kUbOne, kScaleFloats);
    AscendC::Abs(work_a, work_b, all);
    if constexpr (kIsKey<Chunk>) {
        auto f128 = View<float>(kUbFold128, kTokens * 128u);
        auto f64 = View<float>(kUbFold64, kTokens * 64u);
        auto f32 = View<float>(kUbFold32, kTokens * 32u);
        auto f16 = View<float>(kUbFold16, kTokens * 16u);
        auto f8 = View<float>(kUbFold8, kTokens * 8u);
        KeyFold<256u, 256u, 128u>(f128, work_a);
        KeyFold<128u, 128u, 64u>(f64, f128);
        KeyFold<64u, 64u, 32u>(f32, f64);
        KeyFold<32u, 32u, 16u>(f16, f32);
        KeyFold<16u, 16u, 8u>(f8, f16);
        // Eight lanes per token remain, one whole DataBlock each, and
        // BlockReduceMax writes one result per block contiguously.
        AscendC::BlockReduceMax<float>(abs_max, f8, 2, 64, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(abs_max, abs_max, one, static_cast<std::int32_t>(kScaleFloats));
    } else {
        ValueFoldEmit(work_a, std::make_integer_sequence<std::uint32_t, 8u>{});
        AscendC::PipeBarrier<PIPE_V>();
        // Floor at 1.0 and copy out of the fold buffer in one op: every value
        // is an integer, so 1.0 is the smallest non-zero magnitude and an
        // all-zero token gets scale 2^-2 rather than a division by zero.
        AscendC::Max(abs_max, work_a, one, static_cast<std::int32_t>(kScaleFloats));
    }

    // Power-of-two scale, by integer arithmetic on the FP32 bit pattern. A
    // float 2^k carries the exponent field k + 127, so with
    // 2^e <= absmax < 2^(e+1):
    //   exp_bits  = absmax & 0x7F800000       -> 2^e,       field e + 127
    //   pow_bits  = exp_bits - (2 << 23)      -> 2^(e-2),   field e + 125
    //   inv_bits  = (254 << 23) - pow_bits    -> 2^(2-e),   field 129 - e
    // The last line is the exact reciprocal of a power of two: the two fields
    // sum to 254, which is what (254 << 23) - x computes. No division, no
    // logarithm, and no hardware reciprocal rounding enters the contract.
    auto exp_bits = View<std::int32_t>(kUbExpBits, kScaleFloats);
    auto exp_mask = View<std::int32_t>(kUbExpMask, kScaleFloats);
    auto inv_seed = View<std::int32_t>(kUbInvSeed, kScaleFloats);
    auto pow_bits = View<std::int32_t>(kUbPow, kScaleFloats);
    auto pow_two = View<float>(kUbPow, kScaleFloats);
    auto inv_bits = View<std::int32_t>(kUbInv, kScaleFloats);
    auto inv = View<float>(kUbInv, kScaleFloats);
    constexpr std::int32_t lanes = static_cast<std::int32_t>(kScaleFloats);

    AscendC::And(exp_bits, abs_max.template ReinterpretCast<std::int32_t>(), exp_mask, lanes);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(pow_bits, exp_bits, static_cast<std::int32_t>(-0x01000000), lanes);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(inv_bits, inv_seed, pow_bits, lanes);

    // Broadcast inv along the panel's fast axis and quantise. The value layout
    // has the token index innermost, so one level-0 op with a repeat-invariant
    // second operand covers the whole panel; the key layout has the token index
    // outermost, so the per-token block advances once per repeat and a repeat
    // covers a quarter of a token row.
    auto quant = View<std::int32_t>(kUbCast, kPanel);
    auto staged = View<half>(kUbHalf, kPanel);
    auto packed = View<std::int8_t>(kPackBase<pack>, kPackedBytes);
    AscendC::BinaryRepeatParams bc;
    bc.dstBlkStride = 1;
    bc.src0BlkStride = 1;
    bc.dstRepStride = 8;
    bc.src0RepStride = 8;
    if constexpr (kIsKey<Chunk>) {
        auto inv_bcast = View<float>(kUbInvBcast, kTokens * kLanesPerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(inv_bcast, inv, 2, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        bc.src1BlkStride = 0;
        bc.src1RepStride = 1;
        bc.dstRepStride = static_cast<std::uint8_t>(kHeadSize / kLanesPerBlock);
        bc.src0RepStride = bc.dstRepStride;
        KeyQuantiseEmit(work_a, work_b, inv_bcast, bc,
                        std::make_integer_sequence<std::uint32_t, kHeadSize / 64u>{});
    } else {
        // Four copies of the 16-token block fill one vector register, and the
        // register repeats unchanged across the whole panel.
        auto inv_rep = View<float>(kUbInvRep, 4u * kScaleFloats);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(inv_rep, inv, 1.0f, lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(inv_rep[kScaleFloats], inv_rep, 1.0f, lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(inv_rep[2u * kScaleFloats], inv_rep, 1.0f,
                      static_cast<std::int32_t>(2u * kScaleFloats));
        AscendC::PipeBarrier<PIPE_V>();
        bc.src1BlkStride = 1;
        bc.src1RepStride = 0;
        AscendC::Mul<float>(work_a, work_b, inv_rep, 64,
                            static_cast<std::uint8_t>(kPanel / 64u), bc);
    }

    // |Y * inv| < 8 and round-to-nearest-even can reach +8, which INT4 cannot
    // hold, so clamp from above only; -8 is in range. The INT32 cast is the
    // only rounding in the whole path, and it rounds an exactly representable
    // product, so the host oracle reproduces it in integer arithmetic.
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins(work_a, work_a, 7.0f, all);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(quant, work_a, AscendC::RoundMode::CAST_RINT, kPanel);

    // The packed slot and its scale slice are one ring slot, so one back edge
    // covers both: its previous occupant is still being read by MTE3 until now.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pack_evt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(staged, quant, AscendC::RoundMode::CAST_RINT, kPanel);
    AscendC::PipeBarrier<PIPE_V>();
    auto nibbles = packed.template ReinterpretCast<AscendC::int4b_t>();
    AscendC::Cast(nibbles, staged, AscendC::RoundMode::CAST_RINT, kPanel);
    // The quantised panel is scale invariant - scaling Y and absmax together
    // leaves every nibble unchanged - so invSqrtLen folds into the stored scale
    // alone, and q * scale reconstructs the orthonormal rotation.
    auto scale = View<float>(kScaleBase<pack>, kScaleFloats);
    AscendC::Muls(scale, pow_two, engine.inv_sqrt_len, lanes);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pack_evt);
}

// --- the MTE3 stage ----------------------------------------------------------
template <std::uint32_t Chunk>
__aicore__ inline void StoreChunk(VectorEngine& engine)
{
    constexpr std::uint32_t pack = kPackSlot<Chunk>;
    constexpr std::int32_t pack_evt = static_cast<std::int32_t>(pack);
    auto packed = View<std::int8_t>(kPackBase<pack>, kPackedBytes);
    auto scale = View<float>(kScaleBase<pack>, kScaleFloats);

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pack_evt);
    const std::uint32_t page = engine.page_bytes + kChunkHead<Chunk> * engine.head_bytes;
    const std::uint32_t slice =
        (kIsKey<Chunk> ? engine.key_scale : engine.value_scale) +
        kChunkHead<Chunk> * engine.head_scale;
    if constexpr (kIsKey<Chunk>) {
        AscendC::DataCopy(engine.key_cache[page], packed, kPackedBytes);
    } else {
        AscendC::DataCopy(engine.value_cache[page], packed, kPackedBytes);
    }
    AscendC::DataCopy(engine.scale_cache[slice], scale, kScaleFloats);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pack_evt);
}

// Column j of the schedule on Vector core k: the rotation of chunk j - 2 and
// the egress of chunk j - 3, each only if core k owns that chunk. Both are
// emitted at compile time, so a core's instruction stream contains only its
// own chunks and no predicate.
template <std::uint32_t Column, std::uint32_t Core>
__aicore__ inline void StepColumnVector(VectorEngine& engine)
{
    if constexpr (Column >= 2u && Column - 2u < kChunks &&
                  kOwner<(Column >= 2u ? Column - 2u : 0u)> == Core) {
        RotateAndQuantise<Column - 2u>(engine);
    }
    if constexpr (Column >= 3u && Column - 3u < kChunks &&
                  kOwner<(Column >= 3u ? Column - 3u : 0u)> == Core) {
        StoreChunk<Column - 3u>(engine);
    }
}

template <std::uint32_t Core, std::uint32_t... Columns>
__aicore__ inline void StepColumnsVector(VectorEngine& engine,
                                         std::integer_sequence<std::uint32_t, Columns...>)
{
    (StepColumnVector<Columns, Core>(engine), ...);
}

template <std::uint32_t Core>
__aicore__ inline void run_vector_core(__gm__ std::int8_t* key_cache,
                                       __gm__ std::int8_t* value_cache,
                                       __gm__ float* scale_cache,
                                       __gm__ const std::int32_t* slot_mapping,
                                       __gm__ const float* pi_signs,
                                       std::uint32_t num_kv_heads,
                                       std::uint32_t head_size,
                                       std::uint32_t block_size,
                                       std::uint32_t num_blocks,
                                       float inv_sqrt_len)
{
    // One scalar read of caller-supplied Global Memory metadata, before the
    // pipeline. A prefill strip fills one whole page, so the page index is all
    // the schedule needs and no stage reads the mapping again.
    const std::uint32_t slot0 = static_cast<std::uint32_t>(slot_mapping[0]);
    const std::uint32_t block = slot0 / block_size;
    const std::uint32_t head_bytes = block_size * head_size / 2u;

    VectorEngine engine;
    engine.key_cache.SetGlobalBuffer(key_cache);
    engine.value_cache.SetGlobalBuffer(value_cache);
    engine.scale_cache.SetGlobalBuffer(scale_cache);
    engine.page_bytes = block * num_kv_heads * head_bytes;
    engine.head_bytes = head_bytes;
    engine.key_scale = block * num_kv_heads * block_size;
    engine.value_scale = (num_blocks + block) * num_kv_heads * block_size;
    engine.head_scale = block_size;
    engine.inv_sqrt_len = inv_sqrt_len;

    // ---- prologue: constants and the two replicated sign panels ----------
    auto raw_signs = View<float>(kUbSigns, kHeadSize);
    AscendC::GlobalTensor<float> gm_signs;
    gm_signs.SetGlobalBuffer(const_cast<__gm__ float*>(pi_signs));
    AscendC::DataCopy(raw_signs, gm_signs, kHeadSize);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtEpilogue);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtEpilogue);

    // Immediate-sourced constants never read UB and carry no tensor hazard.
    AscendC::Duplicate(View<float>(kUbOne, kScaleFloats), 1.0f,
                       static_cast<std::int32_t>(kScaleFloats));
    AscendC::Duplicate(View<std::int32_t>(kUbExpMask, kScaleFloats),
                       static_cast<std::int32_t>(0x7F800000),
                       static_cast<std::int32_t>(kScaleFloats));
    AscendC::Duplicate(View<std::int32_t>(kUbInvSeed, kScaleFloats),
                       static_cast<std::int32_t>(0x7F000000),
                       static_cast<std::int32_t>(kScaleFloats));

    // Key panel: the sign vector repeated once per token. Muls by 1.0f is an
    // exact FP32 copy, including for signed zeros, and stays on the Vector pipe.
    auto sign_key = View<float>(kUbSignKey, kPanel);
    AscendC::Muls(sign_key, raw_signs, 1.0f, static_cast<std::int32_t>(kHeadSize));
    DoubleEmit(sign_key, std::make_integer_sequence<std::uint32_t, kTokensLog2>{});

    // Value panel: each sign over kTokens adjacent lanes, which is two
    // DataBlocks. A Brcb repeat splats eight source elements into eight blocks,
    // so two interleaved passes with a two-block destination stride fill it.
    auto sign_value = View<float>(kUbSignValue, kPanel);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Brcb(sign_value, raw_signs, static_cast<std::uint8_t>(kHeadSize / 8u), {2, 16});
    AscendC::Brcb(sign_value[kLanesPerBlock], raw_signs,
                  static_cast<std::uint8_t>(kHeadSize / 8u), {2, 16});
    AscendC::PipeBarrier<PIPE_V>();

    // Feedback cycle priming: both packed page slots start free, and both
    // landing panels are free for the Cube to write.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagPanelFree<Core, 0u>);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagPanelFree<Core, 1u>);

    StepColumnsVector<Core>(
        engine, std::make_integer_sequence<std::uint32_t, kChunks + 3u>{});

    // Drain the primed tokens. These waits also retire the last MTE3 bursts
    // before the kernel returns, so Global Memory is visible to the host.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
}

}  // namespace turboquant_a8

// The input pointers lose their const at this boundary and get it back one
// line into the body. CANN 9.2 generates host_stub.cpp with a `const void*`
// parameter for a `__gm__ const T*` kernel parameter and then assigns it to a
// `void* ` field of the argument block, which does not compile. Dropping const
// on the five lines below is the only way to keep the published argument order
// and types; every device-side function still takes `__gm__ const`, so nothing
// inside the kernel can write an input.
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_a8(
    __gm__ std::int8_t* __restrict__ key,
    __gm__ std::int8_t* __restrict__ value,
    __gm__ std::int8_t* __restrict__ keyCache,
    __gm__ std::int8_t* __restrict__ valueCache,
    __gm__ float* __restrict__ scaleCache,
    __gm__ std::int32_t* __restrict__ slotMapping,
    __gm__ float* __restrict__ piSigns,
    __gm__ std::int32_t* __restrict__ tables,
    // Unqualified uint32_t, not std::uint32_t: CANN generates
    // aclrtlaunch_triple_chevrons_func.h by copying this parameter list
    // verbatim into a header that has no using-declaration for std, so a
    // qualified scalar type breaks the host leg of the build while the device
    // legs compile cleanly. Pointer parameters are rewritten to void* and are
    // unaffected.
    uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
    uint32_t blockSize, uint32_t numBlocks, uint32_t tokensPerCore,
    float invSqrtLen)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    // This revision specialises the strip shape at compile time, so numTokens
    // and tokensPerCore carry no device-side work; the runner asserts that they
    // match kTokens. The remaining scalars drive the physical cache addressing.
    (void)numTokens;
    (void)tokensPerCore;
    // The only role dispatch in the kernel, outside every pipeline stage.
    if ASCEND_IS_AIC {
        turboquant_a8::run_cube_core(key, value, tables, numKvHeads, headSize);
    } else if (AscendC::GetSubBlockIdx() == 0) {
        turboquant_a8::run_vector_core<0u>(keyCache, valueCache, scaleCache, slotMapping,
                                           piSigns, numKvHeads, headSize, blockSize,
                                           numBlocks, invSqrtLen);
    } else {
        turboquant_a8::run_vector_core<1u>(keyCache, valueCache, scaleCache, slotMapping,
                                           piSigns, numKvHeads, headSize, blockSize,
                                           numBlocks, invSqrtLen);
    }
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_a8(
    __gm__ std::int8_t*, __gm__ std::int8_t*, __gm__ std::int8_t*,
    __gm__ std::int8_t*, __gm__ float*, __gm__ std::int32_t*,
    __gm__ float*, __gm__ std::int32_t*, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, float);

// The launcher keeps the const-correct interface the caller sees and strips it
// at the triple-chevron call, which is where the generated argument block
// forces a non-const pointer.
extern "C" void launch_turboquant_reshape_and_cache_a8(
    void* stream, const std::int8_t* key, const std::int8_t* value,
    std::int8_t* key_cache, std::int8_t* value_cache, float* scale_cache,
    const std::int32_t* slot_mapping, const float* pi_signs,
    const std::int32_t* tables, std::uint32_t num_tokens, std::uint32_t num_kv_heads,
    std::uint32_t head_size, std::uint32_t block_size, std::uint32_t num_blocks,
    std::uint32_t tokens_per_core, float inv_sqrt_len)
{
    turboquant_reshape_and_cache_a8<<<1, nullptr, stream>>>(
        const_cast<std::int8_t*>(key), const_cast<std::int8_t*>(value), key_cache,
        value_cache, scale_cache, const_cast<std::int32_t*>(slot_mapping),
        const_cast<float*>(pi_signs), const_cast<std::int32_t*>(tables), num_tokens,
        num_kv_heads, head_size, block_size, num_blocks, tokens_per_core,
        inv_sqrt_len);
}

#endif
