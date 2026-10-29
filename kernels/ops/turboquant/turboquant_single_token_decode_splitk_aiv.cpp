#include <cstdint>
#include "../../../include/turboquant_splitk_contract.hpp"

// True single-launch paged Split-K decode, Ascend 950PR / dav-3510 / CANN 9.2.
// Each block computes one disjoint page range; only the last arrival emits
// the normalized, register-butterfly-rotated, sigmoid-gated BF16 context.
// Derived from single_token_decode_aiv; production kernels remain unchanged.
#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "../../../routines/routine_vector_decode_gemv.hpp"
#include "../../../routines/routine_vector_decode_recurrence.hpp"
#include "../../../routines/routine_vector_rotate_butterfly.hpp"
#include "../../../routines/routine_turboquant_dequant4.hpp"

namespace turboquant_splitk_device {

using ascend_micro::kGemvChannels;
using ascend_micro::kGemvKeys;
using ascend_micro::kGemvPartials;
using ascend_micro::kGemvProductElems;
using ascend_micro::kGemvVecLanes;
using ascend_micro::kRotOrthoScale;

// ---------------------------------------------------------------- geometry
constexpr std::uint32_t kHeadSize = kGemvChannels;  // 256, compile-time
constexpr std::uint32_t kKeys = kGemvKeys;          // 16 tokens per strip
constexpr std::uint32_t kStripHeight = kKeys;
constexpr std::uint32_t kPackedBytes = kStripHeight * kHeadSize / 2u;  // 2048
constexpr std::uint32_t kPanelElems = kStripHeight * kHeadSize;        // 4096
constexpr std::uint32_t kStateLanes = ascend_micro::kDecodeStateLanes;  // 8
constexpr std::uint32_t kVecLanes = kGemvVecLanes;                      // 64

// The compile-time row bound. Six is the contract (H_q / H_kv at k = 0) and
// `rows` is a runtime operand, so the buffers are sized for a bound rather
// than for six. Eight, not sixteen: the roadmap's dispatch rule sends
// rows >= 12 to the Cube operator, so this kernel's whole domain is rows <= 8,
// and sizing for sixteen would cost 48 KiB of Unified Buffer in the three
// per-row egress panels alone to serve a regime that never reaches here.
constexpr std::uint32_t kMaxRows = 8;
// ------------------------------------------------------- UB layout, AKA3003
constexpr std::uint32_t kBlock = 32;
constexpr std::uint32_t kBankSkew = 32;
constexpr std::uint32_t kVecReg = 256;  // VECTOR_REG_WIDTH on dav-3510
constexpr std::uint32_t kUbWindow = 192u * 1024u;

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkew;
}

// ---- the strip ingress and its dequantised pages ----
constexpr std::uint32_t kUbPackedKey = 0;                                   // int8
constexpr std::uint32_t kUbPackedVal = NextSlot(kUbPackedKey, kPackedBytes); // int8
constexpr std::uint32_t kUbNib16 = NextSlot(kUbPackedVal, kPackedBytes);     // int16
constexpr std::uint32_t kUbKeyPage = NextSlot(kUbNib16, kPanelElems * 2u);   // fp32
constexpr std::uint32_t kUbValuePage = NextSlot(kUbKeyPage, kPanelElems * 4u);
// ---- the score GEMV ----
constexpr std::uint32_t kUbProducts = NextSlot(kUbValuePage, kPanelElems * 4u);
constexpr std::uint32_t kUbPartials = NextSlot(kUbProducts, kGemvProductElems * 4u);
constexpr std::uint32_t kUbFold = NextSlot(kUbPartials, kGemvPartials * 4u);
constexpr std::uint32_t kUbPanel = NextSlot(kUbFold, kGemvPartials * 4u);
// ---- the rotated query panel, live for the whole task ----
// It arrives as bf16 -- kernel No.5 writes it, and RFC section 4 types the
// operand `uint16_t`, so No.3 and No.4 take the same tensor at the same call
// site -- and is converted once per task into the FP32 panel the GEMV reads.
// bf16 -> float is a listed Cast pair and exact, so the conversion costs one
// instruction and no accuracy.
constexpr std::uint32_t kUbQueryRaw = NextSlot(kUbPanel, kMaxRows * kKeys * 4u);
constexpr std::uint32_t kUbQuery = NextSlot(kUbQueryRaw, kMaxRows * kHeadSize * 2u);
// ---- the two scale slices ----
constexpr std::uint32_t kUbKeyScale = NextSlot(kUbQuery, kMaxRows * kHeadSize * 4u);
constexpr std::uint32_t kUbValueScale = NextSlot(kUbKeyScale, kKeys * 4u);
// ---- the carried state ----
constexpr std::uint32_t kUbMPrev = NextSlot(kUbValueScale, kKeys * 4u);
constexpr std::uint32_t kUbLPrev = NextSlot(kUbMPrev, kMaxRows * kStateLanes * 4u);
constexpr std::uint32_t kUbAcc = NextSlot(kUbLPrev, kMaxRows * kStateLanes * 4u);
// ---- the recurrence's per-row scratch ----
constexpr std::uint32_t kUbScaledRow = NextSlot(kUbAcc, kMaxRows * kHeadSize * 4u);
constexpr std::uint32_t kUbCentered = NextSlot(kUbScaledRow, kKeys * 4u);
constexpr std::uint32_t kUbP = NextSlot(kUbCentered, kKeys * 4u);
constexpr std::uint32_t kUbPScaled = NextSlot(kUbP, kKeys * 4u);
constexpr std::uint32_t kUbPx4 = NextSlot(kUbPScaled, kKeys * 4u);
constexpr std::uint32_t kUbMNew = NextSlot(kUbPx4, kVecLanes * 4u);
constexpr std::uint32_t kUbLNew = NextSlot(kUbMNew, kStateLanes * 4u);
constexpr std::uint32_t kUbAlpha = NextSlot(kUbLNew, kStateLanes * 4u);
constexpr std::uint32_t kUbAlphaDelta = NextSlot(kUbAlpha, kStateLanes * 4u);
constexpr std::uint32_t kUbLScaled = NextSlot(kUbAlphaDelta, kStateLanes * 4u);
constexpr std::uint32_t kUbTileMaxRaw = NextSlot(kUbLScaled, kStateLanes * 4u);
constexpr std::uint32_t kUbPSumRaw = NextSlot(kUbTileMaxRaw, kStateLanes * 4u);
constexpr std::uint32_t kUbTileMaxBc = NextSlot(kUbPSumRaw, kStateLanes * 4u);
constexpr std::uint32_t kUbMBc = NextSlot(kUbTileMaxBc, kVecLanes * 4u);
constexpr std::uint32_t kUbPSumBc = NextSlot(kUbMBc, kVecLanes * 4u);
constexpr std::uint32_t kUbAlphaBc = NextSlot(kUbPSumBc, kVecLanes * 4u);
constexpr std::uint32_t kUbWeighted = NextSlot(kUbAlphaBc, kVecLanes * 4u);
constexpr std::uint32_t kUbWeightedSum = NextSlot(kUbWeighted, kPanelElems * 4u);
constexpr std::uint32_t kUbAccScaled = NextSlot(kUbWeightedSum, kHeadSize * 4u);
// ---- the egress ----
constexpr std::uint32_t kUbSign = NextSlot(kUbAccScaled, kHeadSize * 4u);
constexpr std::uint32_t kUbNorm = NextSlot(kUbSign, kHeadSize * 4u);
constexpr std::uint32_t kUbRotated = NextSlot(kUbNorm, kMaxRows * kHeadSize * 4u);
constexpr std::uint32_t kUbGate = NextSlot(kUbRotated, kMaxRows * kHeadSize * 4u);
constexpr std::uint32_t kUbGateA = NextSlot(kUbGate, kMaxRows * kHeadSize * 4u);
constexpr std::uint32_t kUbGateB = NextSlot(kUbGateA, kMaxRows * kHeadSize * 4u);
constexpr std::uint32_t kUbContext = NextSlot(kUbGateB, kMaxRows * kHeadSize * 4u);
// ---- the launch probe ----
constexpr std::uint32_t kProbeLanes = 32;
constexpr std::uint32_t kUbProbe = NextSlot(kUbContext, kMaxRows * kHeadSize * 2u);
constexpr std::uint32_t kArenaEnd = kUbProbe + kProbeLanes * 4u;

static_assert(kUbPackedKey % kBlock == 0u && kUbPackedVal % kBlock == 0u &&
                  kUbNib16 % kBlock == 0u && kUbKeyPage % kBlock == 0u &&
                  kUbValuePage % kBlock == 0u && kUbProducts % kBlock == 0u &&
                  kUbPartials % kBlock == 0u && kUbFold % kBlock == 0u &&
                  kUbPanel % kBlock == 0u && kUbQueryRaw % kBlock == 0u &&
                  kUbQuery % kBlock == 0u && kUbKeyScale % kBlock == 0u &&
                  kUbValueScale % kBlock == 0u && kUbMPrev % kBlock == 0u &&
                  kUbLPrev % kBlock == 0u && kUbAcc % kBlock == 0u &&
                  kUbNorm % kBlock == 0u && kUbRotated % kBlock == 0u &&
                  kUbGate % kBlock == 0u && kUbGateA % kBlock == 0u &&
                  kUbGateB % kBlock == 0u && kUbContext % kBlock == 0u &&
                  kUbSign % kBlock == 0u && kUbProbe % kBlock == 0u,
              "every UB operand base must be 32-byte aligned");
// Rule A3: a level-2 continuous instruction reads a whole 256-byte vector
// register from each source base, unmasked, whatever the lane count says. The
// arena therefore needs that much pad behind it...
static_assert(kArenaEnd + kVecReg <= kUbWindow, "UB arena leaves no read pad");
// ...and the per-row state blocks need it behind the LAST row, because
// `Brcb`, `Duplicate` and the 8-lane `Max`/`Mul` in the recurrence all read a
// full register from `m_prev + (rows - 1) * 32` and `l_prev + ...`. The slot
// after each is far enough away that the over-read lands in the kernel's own
// arena rather than off the end of the window.
static_assert(kUbMPrev + (kMaxRows - 1u) * kStateLanes * 4u + kVecReg <= kArenaEnd &&
                  kUbLPrev + (kMaxRows - 1u) * kStateLanes * 4u + kVecReg <= kArenaEnd,
              "the state over-read must stay inside the arena");
// The two GEMV folds read a whole register from `partials + 32` and
// `fold + 16`, which is past the end of each 64-lane buffer.
static_assert(kUbPartials + kGemvPartials * 4u + kVecReg <= kArenaEnd &&
                  kUbFold + kGemvPartials * 4u + kVecReg <= kArenaEnd,
              "the GEMV fold over-read must stay inside the arena");
// The register-resident butterfly addresses the egress panels as raw __ubuf__
// pointers, so a row base must be a whole number of vector registers from the
// panel base.
static_assert(kHeadSize * 4u % kVecReg == 0u, "a 256-channel FP32 row is whole registers");
static_assert(turboquant_splitk::kMaxRows * turboquant_splitk::kRecordFloats <= kPanelElems,
              "full-head packed states must fit in each retired strip panel");
static_assert(kVecLanes * 4u == kVecReg, "64 FP32 lanes is one vector register");
// Every DMA length below is a whole number of 32-byte bursts.
static_assert(kPackedBytes % kBlock == 0u && kKeys * 4u % kBlock == 0u &&
                  kHeadSize * 4u % kBlock == 0u && kHeadSize * 2u % kBlock == 0u &&
                  kStateLanes * 4u % kBlock == 0u,
              "DMA lengths must be whole 32-byte bursts");
static_assert(kMaxRows * kKeys * 4u % kBlock == 0u,
              "an archived panel must be whole 32-byte bursts");

// Event ids, static code 0..5 (6 and 7 reserved). One token in flight per
// (route, id).
constexpr std::int32_t kEvtIngressReady = 0;  // MTE2_V
constexpr std::int32_t kEvtPackedFree = 0;    // V_MTE2
constexpr std::int32_t kEvtScaleFree = 1;     // V_MTE2
constexpr std::int32_t kEvtEgressReady = 1;   // V_MTE3
constexpr std::int32_t kEvtDrain = 0;         // MTE3_S

constexpr auto kUb = AscendC::TPosition::VECCALC;

// A 0-ULP divide. Declared here rather than taken from
// `routine_vector_egress_rotate.hpp`, which is frozen and which this kernel
// deliberately does not include: that leaf is the Blocked butterfly with its
// 6 KiB of Unified Buffer Gather tables, and this kernel uses the measured
// register-resident one instead.
constexpr AscendC::DivConfig kDecodePreciseDiv = {
    AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};

template <class T>
__aicore__ inline AscendC::LocalTensor<T> View(std::uint32_t base, std::uint32_t elems)
{
    return AscendC::LocalTensor<T>{kUb, base, elems};
}

// ============================================================ the operator
struct Geometry {
    std::uint32_t num_kv_heads;
    std::uint32_t rows;
    std::uint32_t head_size;
    std::uint32_t block_size;    // S, tokens per page
    std::uint32_t cache_pages;   // P, physical pages in the cache
    std::uint32_t strips_in_page;
    std::uint32_t seq_pages;     // logical pages this request spans
    std::uint32_t strips;        // strips processed per page
    std::uint32_t page_begin;
    std::uint32_t table_stride;
    std::uint32_t strip_begin;
};

// The packed strip's byte offset. Both planes share the layout
// [pages][heads][strips][2048], which is why one expression serves both.
__aicore__ inline std::uint64_t StripByteOffset(std::uint32_t page, std::uint32_t head,
                                                std::uint32_t strip,
                                                const Geometry& geom)
{
    return ((static_cast<std::uint64_t>(page) * geom.num_kv_heads + head) *
                geom.strips_in_page +
            strip) *
           kPackedBytes;
}

// The planar scale cache: [2][pages][heads][block_size], plane 0 key, 1 value.
// A strip's 16 tokens are 16 consecutive FP32 lanes at a 64-byte aligned
// offset, so one plain `DataCopy` of two DataBlocks reads it.
__aicore__ inline std::uint64_t ScaleOffset(std::uint32_t plane, std::uint32_t page,
                                            std::uint32_t head, std::uint32_t strip,
                                            const Geometry& geom)
{
    return ((static_cast<std::uint64_t>(plane) * geom.cache_pages + page) *
                geom.num_kv_heads +
            head) *
               geom.block_size +
           static_cast<std::uint64_t>(strip) * kStripHeight;
}

// ---------------------------------------------------------- the strip body
// MTE2: the strip's four landings. Two 2 KiB packed pages and two 64-byte
// scale slices, with the packed and the scale waits SEPARATED so the packed
// refill of strip s + 1 can issue while strip s is still in its recurrence.
__aicore__ inline void LoadStrip(const AscendC::GlobalTensor<std::int8_t>& key_cache,
                                 const AscendC::GlobalTensor<std::int8_t>& value_cache,
                                 const AscendC::GlobalTensor<float>& scale_cache,
                                 std::uint32_t page, std::uint32_t head,
                                 std::uint32_t strip, const Geometry& geom)
{
    const std::uint64_t strip_base = StripByteOffset(page, head, strip, geom);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtPackedFree);
    AscendC::DataCopy(View<std::int8_t>(kUbPackedKey, kPackedBytes),
                      key_cache[strip_base], kPackedBytes);
    AscendC::DataCopy(View<std::int8_t>(kUbPackedVal, kPackedBytes),
                      value_cache[strip_base], kPackedBytes);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtScaleFree);
    AscendC::DataCopy(View<float>(kUbKeyScale, kKeys),
                      scale_cache[ScaleOffset(0u, page, head, strip, geom)], kKeys);
    AscendC::DataCopy(View<float>(kUbValueScale, kKeys),
                      scale_cache[ScaleOffset(1u, page, head, strip, geom)], kKeys);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady);
}

// int4 -> fp32 for both planes, through the only listed conversion route
// (int4b_t -> int16_t -> float; int4b_t -> float is not a listed pair, and on
// this part an unlisted pair compiles and emits nothing at all).
//
// The key page comes out token-major -- key_page[t * 256 + c] -- which is what
// the GEMV's `Mul` strides address; the value page comes out channel-major
// with the 16-token axis contiguous, which is what the recurrence's
// `ReduceRepeat` over 16 lanes and its `p_x4` broadcast address. Neither needs
// a transpose: both are the cache's own order.
__aicore__ inline void DequantStrip()
{
    ascend_micro::NativeDequant4Engine<float> key_dq;
    key_dq.packed = View<std::int8_t>(kUbPackedKey, kPackedBytes);
    key_dq.signed_nibbles = View<std::int16_t>(kUbNib16, kPanelElems);
    key_dq.output = View<float>(kUbKeyPage, kPanelElems);
    ascend_micro::turboquant_dequant4_signed(key_dq, kPanelElems);
    AscendC::PipeBarrier<PIPE_V>();

    ascend_micro::NativeDequant4Engine<float> value_dq;
    value_dq.packed = View<std::int8_t>(kUbPackedVal, kPackedBytes);
    value_dq.signed_nibbles = View<std::int16_t>(kUbNib16, kPanelElems);
    value_dq.output = View<float>(kUbValuePage, kPanelElems);
    ascend_micro::turboquant_dequant4_signed(value_dq, kPanelElems);
}

// Vector-only layout maps: the first rows*8 lanes address the replicated
// m/l blocks in an AoS [row][m8,l8,acc256] payload. All rows' statistics are
// merged in one register, with no scalar extraction or per-row scheduling.
__simd_callee__ inline void StateIndices(
    AscendC::Reg::RegTensor<std::uint32_t>& index, AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    RegTensor<std::int32_t> ramp;
    Arange<std::int32_t>(ramp, 0);
    auto& lane = reinterpret_cast<RegTensor<std::uint32_t>&>(ramp);
    RegTensor<std::uint32_t> row, hi, lo, seven, within;
    ShiftRights<std::uint32_t>(row, lane, static_cast<std::int16_t>(3), mask);
    ShiftLefts<std::uint32_t>(hi, row, static_cast<std::int16_t>(8), mask);
    ShiftLefts<std::uint32_t>(lo, row, static_cast<std::int16_t>(4), mask);
    Add<std::uint32_t>(index, hi, lo, mask);  // row * 272
    Duplicate<std::uint32_t>(seven, 7u);
    And<std::uint32_t>(within, lane, seven, mask);
    Add<std::uint32_t>(index, index, within, mask);
}

__simd_vf__ inline void PackHeadState(__ubuf__ float* packed,
    __ubuf__ float* m, __ubuf__ float* l, __ubuf__ float* acc, std::uint16_t rows)
{
    using namespace AscendC::Reg;
    std::uint32_t count = rows * 8u;
    MaskReg stat_mask = UpdateMask<float>(count);
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<std::uint32_t> index;
    StateIndices(index, stat_mask);
    RegTensor<float> mr, lr, ar;
    LoadAlign(mr, m);
    LoadAlign(lr, l);
    Scatter<float, std::uint32_t>(packed, mr, index, stat_mask);
    Scatter<float, std::uint32_t>(packed + 8, lr, index, stat_mask);
    // Register-width stream over the full accumulator panel, outside lock.
    // No DMA, pipe barriers or scalar tensor reads occur in this VF loop.
    for (std::uint16_t v = 0; v < rows * 4u; ++v) {
        LoadAlign(ar, acc + v * 64u);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(
            packed + (v / 4u) * 272u + 16u + (v % 4u) * 64u, ar, full);
    }
}

__simd_vf__ inline void MergeHeadState(__ubuf__ float* packed,
    __ubuf__ float* m, __ubuf__ float* l, __ubuf__ float* acc,
    __ubuf__ float* alpha, std::uint16_t rows)
{
    using namespace AscendC::Reg;
    std::uint32_t count = rows * 8u;
    MaskReg stat_mask = UpdateMask<float>(count);
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<std::uint32_t> index;
    StateIndices(index, stat_mask);
    RegTensor<float> mg, mt, mn, lg, lt, ln, ag, at, d, a, b;
    Gather<float, float, std::uint32_t>(mg, packed, index, stat_mask);
    Gather<float, float, std::uint32_t>(lg, packed + 8, index, stat_mask);
    LoadAlign(mt, m);
    LoadAlign(lt, l);
    Max(mn, mg, mt, stat_mask);
    Sub(d, mg, mn, stat_mask);
    Exp(ag, d, stat_mask);
    Sub(d, mt, mn, stat_mask);
    Exp(at, d, stat_mask);
    Mul(a, lg, ag, stat_mask);
    Mul(b, lt, at, stat_mask);
    Add(ln, a, b, stat_mask);
    StoreAlign<float, StoreDist::DIST_NORM_B32>(m, mn, stat_mask);
    StoreAlign<float, StoreDist::DIST_NORM_B32>(l, ln, stat_mask);
    Scatter<float, std::uint32_t>(packed, mn, index, stat_mask);
    Scatter<float, std::uint32_t>(packed + 8, ln, index, stat_mask);
    StoreAlign<float, StoreDist::DIST_NORM_B32>(alpha, ag, stat_mask);
    StoreAlign<float, StoreDist::DIST_NORM_B32>(alpha + 64, at, stat_mask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    RegTensor<float> cg, ct, bg, bt, ca, cb, cn;
    // A vector-register stream, not a row-wise Scalar/MTE schedule: all row
    // rescale factors were computed in parallel above. Each iteration handles
    // 64 channels with ordinary register loads, mul/add and stores.
    for (std::uint16_t v = 0; v < rows * 4u; ++v) {
        const std::uint32_t row_block = (v / 4u) * 8u;
        __ubuf__ float* current = packed + (v / 4u) * 272u + 16u + (v % 4u) * 64u;
        LoadAlign(cg, current);
        LoadAlign(ct, acc + v * 64u);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(bg, alpha + row_block);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(bt, alpha + 64u + row_block);
        Mul(ca, cg, bg, full);
        Mul(cb, ct, bt, full);
        Add(cn, ca, cb, full);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(current, cn, full);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(acc + v * 64u, cn, full);
    }
}

__simd_vf__ inline void BroadcastDenominators(__ubuf__ float* dst,
    __ubuf__ float* l, std::uint16_t registers)
{
    using namespace AscendC::Reg;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> denominator;
    for (std::uint16_t v = 0; v < registers; ++v) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(denominator, l + (v / 4u) * 8u);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dst + v * 64u, denominator, full);
    }
}

// --------------------------------------------------------------- the egress
// Acc / l, the right-multiplication by Pi, then sigmoid(gate).
//
// The rotation is the RIGHT-multiplication order x . Pi -- strides
// 32, 64, 128, sign, 1, 2, 4, 8, 16 -- because the accumulator lives in the
// cache's rotated channel basis: Acc[q] = sum_t P[q][t] . (Pi . V[t]) =
// Pi . ctx[q], and Pi Pi^T = I. Reusing the ingress order here is a silent
// wrong answer, since both orders are orthogonal maps and the result is
// plausible and never faults. `RotateDirection::Egress` is what picks it, and
// the leaf's stage table is checked against the dense 256x256 matrix by the
// Tier-1 probe rather than by this kernel.
//
// `RotateImpl::RegGather`, by measurement: over 16 rows on CAModel the three
// bit-identical implementations cost 1203 / 1981 / 11250 vector instructions
// (RegGather / RegShuffle / Blocked), and only the two register-resident ones
// issue four register loads and four stores per row and not one more.
// RegShuffle is ingress-only by construction, so Egress has one measured
// winner and the kernel takes it.
__aicore__ inline void Egress(const Geometry& geom)
{
    const std::uint32_t lanes = geom.rows * kHeadSize;

    // One register broadcast pass, followed by one continuous panel divide.
    BroadcastDenominators(
        reinterpret_cast<__ubuf__ float*>(View<float>(kUbProducts, lanes).GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(View<float>(kUbLPrev, kMaxRows * kStateLanes).GetPhyAddr()),
        static_cast<std::uint16_t>(geom.rows * 4u));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div<float, kDecodePreciseDiv>(
        View<float>(kUbNorm, lanes), View<float>(kUbAcc, lanes),
        View<float>(kUbProducts, lanes), static_cast<std::int32_t>(lanes));
    AscendC::PipeBarrier<PIPE_V>();

    // All `rows` rows in one VF entry: the sign registers and the index
    // tables are built once per call rather than once per row.
    ascend_micro::rotate_butterfly_gather_vf<ascend_micro::RotateDirection::Egress>(
        reinterpret_cast<__ubuf__ float*>(
            View<float>(kUbRotated, kMaxRows * kHeadSize).GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(
            View<float>(kUbNorm, kMaxRows * kHeadSize).GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(View<float>(kUbSign, kHeadSize).GetPhyAddr()),
        kRotOrthoScale, static_cast<std::uint16_t>(geom.rows));
    AscendC::PipeBarrier<PIPE_V>();

    // out = y . sigmoid(gate) = y / (1 + exp(-gate)), over all rows at once --
    // the level-2 continuous form takes a plain element count, so the whole
    // 6 x 256 panel is five instructions rather than five per row.
    //
    // Every destination is distinct from its sources. That is not a style
    // preference: the first revision of the frozen Cube egress computed this
    // chain with three IN-PLACE vector operations and matched none of nine
    // candidate formulations on the device.
    AscendC::Muls(View<float>(kUbGateA, lanes), View<float>(kUbGate, lanes), -1.0f,
                  static_cast<std::int32_t>(lanes));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(View<float>(kUbGateB, lanes), View<float>(kUbGateA, lanes),
                 static_cast<std::int32_t>(lanes));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(View<float>(kUbGateA, lanes), View<float>(kUbGateB, lanes), 1.0f,
                  static_cast<std::int32_t>(lanes));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div<float, kDecodePreciseDiv>(
        View<float>(kUbGateB, lanes), View<float>(kUbRotated, lanes),
        View<float>(kUbGateA, lanes), static_cast<std::int32_t>(lanes));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(View<bfloat16_t>(kUbContext, lanes), View<float>(kUbGateB, lanes),
                  AscendC::RoundMode::CAST_RINT, static_cast<std::int32_t>(lanes));
}


// A single SIMT lane owns the TAS operation. All VF traffic bypasses the
// noncoherent per-core DataCache. No SIMT UB accesses / dynamic UB allocation.
__simt_vf__ __launch_bounds__(32) inline void Acquire(
    __gm__ turboquant_splitk::AtomicHeadSlot* slot,
    __gm__ turboquant_splitk::WorkerMetrics* metric, std::uint32_t max_attempts)
{
    if (threadIdx.x != 0u) return;
    std::uint32_t failed = 0u, arrival = 0u, got = 0u;
    while (failed < max_attempts) {
        if (asc_atomic_exch(&slot->lock, 1u) == 0u) {
            got = 1u;
            arrival = asc_atomic_add(&slot->arrived_count, 1u) + 1u;
            break;
        }
        ++failed;
    }
    asc_stcg(&metric->acquired, got);
    asc_stcg(&metric->arrival, arrival);
    asc_stcg(&metric->failed_exchanges, failed);
    asc_stcg(&metric->timeout, 1u - got);
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(32) inline void Release(
    __gm__ turboquant_splitk::AtomicHeadSlot* slot)
{
    if (threadIdx.x != 0u) return;
    asc_threadfence();
    (void)asc_atomic_exch(&slot->lock, 0u);
}

__aicore__ inline void FinishMetric(
    __gm__ turboquant_splitk::WorkerMetrics* metric, bool tail, std::uint32_t pages,
    std::uint64_t lock_region)
{
    // The caller has retired the unlock VF or the out_context MTE3 burst.
    const auto stamp = static_cast<std::uint64_t>(AscendC::GetSystemCycle());
    auto buf = View<std::uint32_t>(kUbPackedKey, 48);
    AscendC::Duplicate(buf, static_cast<std::uint32_t>(stamp >> 32), 8);
    AscendC::Duplicate(buf[8], static_cast<std::uint32_t>(stamp), 8);
    AscendC::Duplicate(buf[16], static_cast<std::uint32_t>(tail), 8);
    AscendC::Duplicate(buf[24], pages, 8);
    AscendC::Duplicate(buf[32], static_cast<std::uint32_t>(lock_region >> 32), 8);
    AscendC::Duplicate(buf[40], static_cast<std::uint32_t>(lock_region), 8);
    AscendC::GlobalTensor<std::uint32_t> dst;
    dst.SetGlobalBuffer(metric->finish_hi);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::DataCopy(dst, buf, 48);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(2);
}

__aicore__ inline void ReduceAndFinish(
    __gm__ turboquant_splitk::AtomicHeadSlot* slots,
    __gm__ turboquant_splitk::WorkerMetrics* metrics,
    const AscendC::GlobalTensor<bfloat16_t>& out_context, std::uint32_t item,
    std::uint32_t head_item, std::uint32_t splits, std::uint32_t max_attempts,
    const Geometry& geom)
{
    auto* slot = slots + head_item;
    auto* metric = metrics + item;
    // Pack the local state before entering the critical section. Strip
    // scratch is dead; both full-head packed buffers fit in existing slots.
    PackHeadState(
        reinterpret_cast<__ubuf__ float*>(View<float>(kUbValuePage, geom.rows * turboquant_splitk::kRecordFloats).GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(View<float>(kUbMPrev, 64).GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(View<float>(kUbLPrev, 64).GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(View<float>(kUbAcc, geom.rows * kHeadSize).GetPhyAddr()),
        static_cast<std::uint16_t>(geom.rows));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(3);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(3);
    const auto lock_begin = static_cast<std::uint64_t>(AscendC::GetSystemCycle());
    asc_vf_call<Acquire>(dim3{32u, 1u, 1u}, slot, metric, max_attempts);
    // asc_vf_call is asynchronous and returns void in CANN 9.2. Wait once
    // for protocol completion, invalidate the scalar cache line, and read the
    // SIMT-published arrival directly from GM into a scalar register. There
    // is no UB scalar load, status DMA, GetValue, or MTE2_S rendezvous.
    AscendC::SetFlag<AscendC::HardEvent::V_S>(4);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(4);
    AscendC::GlobalTensor<std::uint32_t> control;
    control.SetGlobalBuffer(&metric->arrival);
    AscendC::DataCacheCleanAndInvalid<std::uint32_t,
        AscendC::CacheLine::SINGLE_CACHE_LINE>(control);
    const std::uint32_t arrival = *reinterpret_cast<__gm__ volatile std::uint32_t*>(&metric->arrival);
    if (arrival == 0u) {
        FinishMetric(metric, false, geom.seq_pages, 0u);
        return;
    }
    const bool tail = arrival == splits;
    const std::uint32_t state_lanes = geom.rows * turboquant_splitk::kRecordFloats;
    auto packed = View<float>(kUbValuePage, state_lanes);
    if (arrival > 1u) {
        // One contiguous 6x272-float MTE2 burst for the entire head state.
        AscendC::GlobalTensor<float> state;
        state.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(slot->row));
        packed = View<float>(kUbKeyPage, state_lanes);
        AscendC::DataCopy(packed, state, state_lanes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
        MergeHeadState(
            reinterpret_cast<__ubuf__ float*>(packed.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbMPrev, 64).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbLPrev, 64).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbAcc, geom.rows * kHeadSize).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbProducts, 128).GetPhyAddr()),
            static_cast<std::uint16_t>(geom.rows));
    }
    if (!tail) {
        AscendC::GlobalTensor<float> state;
        state.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(slot->row));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
        // One contiguous MTE3 writeback; the tail retains its state in UB.
        AscendC::DataCopy(state, packed, state_lanes);
    }
    // Publication must retire on MTE3 before the release VF clears the lock.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
    asc_vf_call<Release>(dim3{32u, 1u, 1u}, slot);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(5);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(5);
    const auto lock_region = static_cast<std::uint64_t>(AscendC::GetSystemCycle()) - lock_begin;
    if (tail) {
        Egress(geom);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtEgressReady);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtEgressReady);
        AscendC::DataCopy(out_context[static_cast<std::uint64_t>(head_item) *
                                     geom.rows * kHeadSize],
                          View<bfloat16_t>(kUbContext, geom.rows * kHeadSize),
                          geom.rows * kHeadSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    }
    FinishMetric(metric, tail, geom.seq_pages, lock_region);
}
__aicore__ inline void ProcessItem(const AscendC::GlobalTensor<bfloat16_t>& q_rot,
                                   const AscendC::GlobalTensor<std::int8_t>& key_cache,
                                   const AscendC::GlobalTensor<std::int8_t>& value_cache,
                                   const AscendC::GlobalTensor<float>& scale_cache,
                                   __gm__ const std::uint32_t* block_table,
                                   const AscendC::GlobalTensor<float>& gate,
                                   __gm__ turboquant_splitk::AtomicHeadSlot* slots,
                                   __gm__ turboquant_splitk::WorkerMetrics* metrics,
                                   const AscendC::GlobalTensor<bfloat16_t>& out_context,
                                   std::uint32_t splits, std::uint32_t max_attempts,
                                   std::uint32_t item, std::uint32_t request,
                                   std::uint32_t head, const Geometry& geom)
{
    const std::uint32_t rows = geom.rows;
    const std::uint32_t lanes = rows * kHeadSize;
    // Query head j = kv_head * rows + r, which is GQA's own indexing, so the
    // task's rows are `rows` CONSECUTIVE rows of the [requests][H_q][d] panel
    // and reach Unified Buffer in one burst.
    const std::uint64_t row_base =
        (static_cast<std::uint64_t>(request) * geom.num_kv_heads + head) * rows *
        kHeadSize;

    // ---- the per-task prologue: the query panel and the gate ------------
    // Both are `rows` CONSECUTIVE rows of a [requests][H_q][d] tensor, so each
    // is one contiguous burst, and each lands in a buffer of its own that
    // stays live for the whole task. An earlier revision staged the query
    // through the egress normalisation buffer to save 8 KiB of a 192 KiB
    // window; two live ranges overlapping across a routine boundary is the
    // shape of fault this operator family has paid for repeatedly, and the
    // saving bought nothing.
    AscendC::DataCopy(View<bfloat16_t>(kUbQueryRaw, kMaxRows * kHeadSize),
                      q_rot[row_base], lanes);
    AscendC::DataCopy(View<float>(kUbGate, kMaxRows * kHeadSize), gate[row_base], lanes);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady);
    // bf16 -> float, once per task rather than once per strip. A listed Cast
    // pair, and exact: every bf16 value is an FP32 value.
    AscendC::Cast(View<float>(kUbQuery, kMaxRows * kHeadSize),
                  View<bfloat16_t>(kUbQueryRaw, kMaxRows * kHeadSize),
                  AscendC::RoundMode::CAST_NONE, static_cast<std::int32_t>(lanes));
    AscendC::PipeBarrier<PIPE_V>();

    // ---- seed the carried state -----------------------------------------
    // Seeded from immediate scalars, which never read Unified Buffer and so
    // carry none of the tensor-source hazards. m starts at a large negative
    // rather than -inf: the first strip's `Max` selects the tile maximum
    // either way, and exp(m_prev - m_new) is then 0 rather than a
    // NaN-producing inf - inf. The same constant the frozen Cube operator
    // uses, so the two seed identically.
    AscendC::Duplicate(View<float>(kUbMPrev, kMaxRows * kStateLanes), -3.0e38f,
                       static_cast<std::int32_t>(rows * kStateLanes));
    AscendC::Duplicate(View<float>(kUbLPrev, kMaxRows * kStateLanes), 0.0f,
                       static_cast<std::int32_t>(rows * kStateLanes));
    AscendC::Duplicate(View<float>(kUbAcc, kMaxRows * kHeadSize), 0.0f,
                       static_cast<std::int32_t>(lanes));
    AscendC::PipeBarrier<PIPE_V>();

    // ---- the two engines, built once and reused across every strip -------
    ascend_micro::NativeDecodeGemvEngine gemv;
    gemv.key_page = View<float>(kUbKeyPage, kPanelElems);
    gemv.query = View<float>(kUbQuery, kMaxRows * kHeadSize);
    gemv.products = View<float>(kUbProducts, kGemvProductElems);
    gemv.partials = View<float>(kUbPartials, kGemvPartials);
    gemv.fold = View<float>(kUbFold, kGemvPartials);
    gemv.panel = View<float>(kUbPanel, kMaxRows * kKeys);
    gemv.query_base = kUbQuery;
    gemv.panel_base = kUbPanel;
    gemv.position = kUb;

    ascend_micro::NativeDecodeRecurrenceEngine rec;
    rec.panel = View<float>(kUbPanel, kMaxRows * kKeys);
    rec.key_scale = View<float>(kUbKeyScale, kKeys);
    rec.value_scale = View<float>(kUbValueScale, kKeys);
    rec.value_page = View<float>(kUbValuePage, kPanelElems);
    rec.m_prev = View<float>(kUbMPrev, kMaxRows * kStateLanes);
    rec.l_prev = View<float>(kUbLPrev, kMaxRows * kStateLanes);
    rec.acc = View<float>(kUbAcc, kMaxRows * kHeadSize);
    rec.scaled_row = View<float>(kUbScaledRow, kKeys);
    rec.centered = View<float>(kUbCentered, kKeys);
    rec.p = View<float>(kUbP, kKeys);
    rec.p_scaled = View<float>(kUbPScaled, kKeys);
    rec.p_x4 = View<float>(kUbPx4, kVecLanes);
    rec.m_new = View<float>(kUbMNew, kStateLanes);
    rec.l_new = View<float>(kUbLNew, kStateLanes);
    rec.alpha = View<float>(kUbAlpha, kStateLanes);
    rec.alpha_delta = View<float>(kUbAlphaDelta, kStateLanes);
    rec.l_scaled = View<float>(kUbLScaled, kStateLanes);
    rec.tile_max_raw = View<float>(kUbTileMaxRaw, kStateLanes);
    rec.p_sum_raw = View<float>(kUbPSumRaw, kStateLanes);
    rec.tile_max_bc = View<float>(kUbTileMaxBc, kVecLanes);
    rec.m_bc = View<float>(kUbMBc, kVecLanes);
    rec.p_sum_bc = View<float>(kUbPSumBc, kVecLanes);
    rec.alpha_bc = View<float>(kUbAlphaBc, kVecLanes);
    rec.weighted = View<float>(kUbWeighted, kPanelElems);
    rec.weighted_sum = View<float>(kUbWeightedSum, kHeadSize);
    rec.acc_scaled = View<float>(kUbAccScaled, kHeadSize);
    rec.panel_base = kUbPanel;
    rec.m_base = kUbMPrev;
    rec.l_base = kUbLPrev;
    rec.acc_base = kUbAcc;
    rec.position = kUb;

    // ---- prime the reuse routes ------------------------------------------
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtPackedFree);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtScaleFree);

    // ---- the straight-line strip loop ------------------------------------
    // Pages outside, strips inside. The physical page comes from the block
    // table, which is deliberately not the identity, so a kernel that indexed
    // the cache by its own loop counter would read the wrong page -- and under
    // the canary protocol the wrong page is a quiet NaN.
    for (std::uint32_t page_idx = 0; page_idx < geom.seq_pages; ++page_idx) {
        const std::uint32_t page = block_table[request * geom.table_stride + geom.page_begin + page_idx];
        for (std::uint32_t i = 0; i < geom.strips; ++i) {
            const std::uint32_t strip = geom.strip_begin + i;

            LoadStrip(key_cache, value_cache, scale_cache, page, head, strip, geom);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady);

            DequantStrip();
            AscendC::PipeBarrier<PIPE_V>();
            // The packed pages have been read; their refill may start now,
            // and will overlap the GEMV and the recurrence below. The SCALE
            // slices have NOT been read yet -- the recurrence reads them --
            // which is why they are a separate route released at the bottom.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtPackedFree);

            ascend_micro::vector_decode_gemv_strip(gemv, rows);
            AscendC::PipeBarrier<PIPE_V>();

            ascend_micro::vector_decode_recur_rows(rec, rows);
            AscendC::PipeBarrier<PIPE_V>();
            // The recurrence was the scale slices' last reader.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtScaleFree);
        }
    }

    // ---- drain the primed tokens before the epilogue ---------------------
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtPackedFree);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtScaleFree);

    ReduceAndFinish(slots, metrics, out_context, item,
                    request * geom.num_kv_heads + head, splits, max_attempts, geom);
}


}  // namespace turboquant_splitk_device

extern "C" __global__ __aicore__ void turboquant_single_token_decode_splitk_aiv(
    __gm__ bfloat16_t* q_rot, __gm__ std::int8_t* key, __gm__ std::int8_t* value,
    __gm__ float* scales, __gm__ uint32_t* table, __gm__ float* gate,
    __gm__ float* signs, __gm__ turboquant_splitk::AtomicHeadSlot* slots,
    __gm__ turboquant_splitk::WorkerMetrics* metrics, __gm__ bfloat16_t* out,
    uint32_t requests, uint32_t heads, uint32_t rows,
    uint32_t block_size, uint32_t cache_pages, uint32_t pages,
    uint32_t splits, uint32_t max_attempts)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    using namespace turboquant_splitk_device;
    if (requests == 0u || heads == 0u || rows == 0u || rows > kMaxRows ||
        block_size == 0u || block_size % kKeys != 0u || pages == 0u ||
        pages > cache_pages || splits == 0u || splits > turboquant_splitk::kMaxSplits ||
        max_attempts == 0u) return;
    const auto item = static_cast<uint32_t>(AscendC::GetBlockIdx());
    if (item >= requests * heads * splits) return;
    const auto head_item = item / splits;
    const auto range = turboquant_splitk::Partition(pages, splits, item % splits);
    Geometry geom{heads, rows, kHeadSize, block_size, cache_pages,
                  block_size / kKeys, range.count, block_size / kKeys,
                  range.begin, pages, 0u};
    AscendC::GlobalTensor<bfloat16_t> q, context;
    AscendC::GlobalTensor<std::int8_t> key_cache, value_cache;
    AscendC::GlobalTensor<float> scale_cache, gate_tensor, pi_signs;
    AscendC::GlobalTensor<uint32_t> block_table;
    q.SetGlobalBuffer(q_rot);
    context.SetGlobalBuffer(out);
    key_cache.SetGlobalBuffer(key);
    value_cache.SetGlobalBuffer(value);
    scale_cache.SetGlobalBuffer(scales);
    gate_tensor.SetGlobalBuffer(gate);
    pi_signs.SetGlobalBuffer(signs);
    AscendC::DataCopy(View<float>(kUbSign, kHeadSize), pi_signs, kHeadSize);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady);
    ProcessItem(q, key_cache, value_cache, scale_cache, table, gate_tensor,
                slots, metrics, context, splits, max_attempts, item,
                head_item / heads, head_item % heads, geom);
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

extern "C" __global__ __aicore__ void turboquant_single_token_decode_splitk_aiv(
    __gm__ bfloat16_t*, __gm__ std::int8_t*, __gm__ std::int8_t*, __gm__ float*,
    __gm__ uint32_t*, __gm__ float*, __gm__ float*,
    __gm__ turboquant_splitk::AtomicHeadSlot*,
    __gm__ turboquant_splitk::WorkerMetrics*, __gm__ bfloat16_t*,
    uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t);

extern "C" void launch_turboquant_single_token_decode_splitk_aiv(
    void* stream, std::uint16_t* q_rot, std::int8_t* key, std::int8_t* value,
    float* scales, std::uint32_t* table, float* gate, float* signs,
    turboquant_splitk::AtomicHeadSlot* slots, turboquant_splitk::WorkerMetrics* metrics,
    std::uint16_t* out, std::uint32_t requests, std::uint32_t heads,
    std::uint32_t rows, std::uint32_t block_size, std::uint32_t cache_pages,
    std::uint32_t pages, std::uint32_t splits, std::uint32_t max_attempts)
{
    turboquant_single_token_decode_splitk_aiv<<<requests * heads * splits,
                                             nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(q_rot), key, value, scales, table, gate, signs,
        slots, metrics, reinterpret_cast<bfloat16_t*>(out), requests, heads, rows,
        block_size, cache_pages, pages, splits, max_attempts);
}
#endif
