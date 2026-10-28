#include <cstdint>

// turboquant_single_token_decode_aiv -- single-token paged decode, AIV only.
// Roadmap kernel No.3. Ascend 950PR / arch35 / dav-3510 / CANN 9.2,
// KERNEL_TYPE_AIV_ONLY.
//
// What it serves, stated more widely than its name
// -----------------------------------------------
// `query_tokens_per_req == 1`, for ANY batch size B. Rows per
// (request, kv_head) are 6 x (1 + k) = 6 at k = 0, and the grid is
// (request, kv_head) -- so B scales the GRID, not the row count. This is not
// "the B = 1 kernel"; it is "the 6-row kernel", and at k = 0 that is every
// decode step regardless of batch.
//
// One (request, kv_head) task is owned exclusively by one physical AIV. The
// head is the butterfly's operand and the butterfly does not decompose across
// cores, so there is no intra-head splitting.
//
// Why Vector and not Cube, at six rows
// ------------------------------------
// The Cube decode operator (`turboquant_paged_decode_attention.cpp`, frozen)
// measured its own answer: on its 8-strip rung the Cube is in flight 1,777
// ticks against AIV0's 234,863 -- idle 99.2% of the schedule -- so at 16 rows
// Cube work is free and Vector work is the whole budget. At six rows the Mmad
// costs exactly what it costs at sixteen, because `MmadParams::m` is a runtime
// extent over a 16x16 fractal and a tail block "occupies fractal block space
// but does not participate in valid computation". What does NOT survive the
// drop to six rows is everything around it: an L1 key ring, a CV transfer per
// strip, two mode-4 flag ids per announcement and a Fixpipe, all to contract
// six rows. The roadmap's dispatch rule puts the crossover at rows >= 12,
// where the GEMV's per-row instruction count grows linearly and the Cube's
// does not. Below it, this kernel; at and above it, No.4.
//
// What this kernel does not have, by construction
// -----------------------------------------------
// No Cube. No L1, no L0A/L0B/L0C, no `Mmad`, no `Fixpipe`, no CV transfer. No
// `CrossCoreSetFlag`, no `CrossCoreWaitFlag`, no `wait_intra_block`, no mode-4
// flag id, no modulo ring, no ring priming or draining, no `M_FIX` / `FIX_M` /
// `MTE1_M` edge. Its marked graph is a line, and the trace audit asserts that
// as a POSITIVE property -- zero cross-core instructions issued -- rather than
// as an absence of findings.
//
// The score as a Vector GEMV
// --------------------------
// `routines/routine_vector_decode_gemv.hpp`, seven instructions per row per
// strip: four quarter-major `Mul`s, one 64-repeat `ReduceRepeat`, two folds.
// That leaf's header records why it is seven and not the two the roadmap
// sketched -- a 32-bit continuous-mode mask is bounded at 64 and an over-range
// mask is CLAMPED, not diagnosed, so the sketched form silently reduces the
// first 64 channels of each token -- and the Tier-1 probe measures the
// difference rather than arguing it.
//
// Scales never touch a 4096-entry page
// ------------------------------------
// Exactly as in No.4, and for the same reason. The key scale multiplies the 16
// score columns after the GEMV (16 multiplies, not 4096) and the value scale
// folds into the 16 probabilities before the contraction, with the denominator
// still summing unscaled probabilities. Both are exact because the scales are
// powers of two, and keeping them off the page is what leaves the raw panel an
// exact small multiple of 1/16 -- the property that makes it bit-checkable.
//
// Synchronisation: intra-core HardEvents only, and the one that is easy to get
// wrong
// ------------------------------------------------------------------------
// One buffer set, straight line, with three reuse routes:
//
//   MTE2_V  kEvtIngressReady  the strip's four landings are filled
//   V_MTE2  kEvtPackedFree    the two 2 KiB packed pages have been unpacked
//   V_MTE2  kEvtScaleFree     the two 16-float scale slices have been READ
//   V_MTE3  kEvtPanelReady    the raw panel is ready to archive
//   MTE3_V  kEvtPanelFree     the archive burst has drained
//
// `kEvtPackedFree` and `kEvtScaleFree` are two routes and not one, and that
// split is not tidiness. One MTE2 burst group lands three kinds of operand
// with two different last readers: the packed pages are last read by the
// unpack, the scale slices by the RECURRENCE, which runs after the GEMV. The
// frozen Cube operator paid for collapsing them -- releasing the slot at the
// dequant cost its 4-strip rung, with m and l wrong by 2,097,152 ULP while
// every raw score panel stayed bit-exact, because the two are on different
// queues and program order inside a column orders nothing. Splitting the route
// also buys the only overlap this kernel has: the next strip's packed burst
// issues while this strip's GEMV and recurrence are still running.
//
// Deliberately NOT double buffered. The AIV-centric rule is saturation over
// pipelining (roadmap section 0.2): with one buffer set the next strip's scale
// burst waits behind this strip's recurrence, and whether that costs anything
// is a question for the instruction census, not for a ring this kernel would
// then have to prove correct. The census is in reports/single_token_decode_aiv.md.

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "../../../routines/routine_vector_decode_gemv.hpp"
#include "../../../routines/routine_vector_decode_recurrence.hpp"
#include "../../../routines/routine_vector_rotate_butterfly.hpp"
#include "../../../routines/routine_turboquant_dequant4.hpp"

namespace turboquant_single_token {

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
// The score archive's per-strip stride, fixed so the runner's indexing is
// static while `rows` is not.
constexpr std::uint32_t kArchiveStride = kMaxRows * kKeys;  // 128 floats
// The archive's depth: 8 pages x 8 strips, the most a request may span here.
constexpr std::uint32_t kMaxArchiveStrips = 64;

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
constexpr std::uint32_t kUbLBcast = NextSlot(kUbSign, kHeadSize * 4u);
constexpr std::uint32_t kUbNorm = NextSlot(kUbLBcast, kHeadSize * 4u);
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
constexpr std::int32_t kEvtPanelReady = 0;    // V_MTE3
constexpr std::int32_t kEvtEgressReady = 1;   // V_MTE3
constexpr std::int32_t kEvtPanelFree = 0;     // MTE3_V
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

    // Normalise, one row at a time: `Duplicate`'s tensor form broadcasts lane
    // 0 of its source across the destination, so the row's scalar denominator
    // reaches 256 lanes without a single scalar read of Unified Buffer.
    for (std::uint32_t row = 0; row < geom.rows; ++row) {
        auto l_cur = View<float>(kUbLPrev + row * kStateLanes * 4u, kStateLanes);
        AscendC::Duplicate(View<float>(kUbLBcast, kHeadSize), l_cur,
                           static_cast<std::int32_t>(kHeadSize));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div<float, kDecodePreciseDiv>(
            View<float>(kUbNorm + row * kHeadSize * 4u, kHeadSize),
            View<float>(kUbAcc + row * kHeadSize * 4u, kHeadSize),
            View<float>(kUbLBcast, kHeadSize), static_cast<std::int32_t>(kHeadSize));
        AscendC::PipeBarrier<PIPE_V>();
    }

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

// --------------------------------------------------------------- the body
__aicore__ inline void ProcessItem(const AscendC::GlobalTensor<bfloat16_t>& q_rot,
                                   const AscendC::GlobalTensor<std::int8_t>& key_cache,
                                   const AscendC::GlobalTensor<std::int8_t>& value_cache,
                                   const AscendC::GlobalTensor<float>& scale_cache,
                                   const AscendC::GlobalTensor<std::uint32_t>& block_table,
                                   const AscendC::GlobalTensor<float>& gate,
                                   const AscendC::GlobalTensor<float>& out_scores,
                                   const AscendC::GlobalTensor<float>& out_state,
                                   const AscendC::GlobalTensor<float>& out_acc,
                                   const AscendC::GlobalTensor<float>& out_rotated,
                                   const AscendC::GlobalTensor<bfloat16_t>& out_context,
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
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtPanelFree);

    // ---- the straight-line strip loop ------------------------------------
    // Pages outside, strips inside. The physical page comes from the block
    // table, which is deliberately not the identity, so a kernel that indexed
    // the cache by its own loop counter would read the wrong page -- and under
    // the canary protocol the wrong page is a quiet NaN.
    std::uint32_t archived = 0;
    for (std::uint32_t page_idx = 0; page_idx < geom.seq_pages; ++page_idx) {
        const std::uint32_t page = block_table.GetValue(page_idx);
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

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtPanelFree);
            ascend_micro::vector_decode_gemv_strip(gemv, rows);
            AscendC::PipeBarrier<PIPE_V>();

            // Archive the raw panel. It is read by MTE3 and by the recurrence;
            // both are readers, so the only edge needed is the one that stops
            // the NEXT strip's GEMV from overwriting a panel MTE3 has not
            // drained -- the MTE3_V wait above.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtPanelReady);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtPanelReady);
            if (archived < kMaxArchiveStrips)
                AscendC::DataCopy(
                    out_scores[(static_cast<std::uint64_t>(item) * kMaxArchiveStrips +
                                archived) *
                               kArchiveStride],
                    View<float>(kUbPanel, kMaxRows * kKeys), rows * kKeys);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtPanelFree);
            ++archived;

            ascend_micro::vector_decode_recur_rows(rec, rows);
            AscendC::PipeBarrier<PIPE_V>();
            // The recurrence was the scale slices' last reader.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtScaleFree);
        }
    }

    // ---- drain the primed tokens before the epilogue ---------------------
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtPackedFree);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtScaleFree);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtPanelFree);

    // ---- the epilogue ----------------------------------------------------
    Egress(geom);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtEgressReady);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtEgressReady);

    const std::uint64_t state_base =
        static_cast<std::uint64_t>(item) * 2u * kMaxRows * kStateLanes;
    AscendC::DataCopy(out_state[state_base], View<float>(kUbMPrev, kMaxRows * kStateLanes),
                      rows * kStateLanes);
    AscendC::DataCopy(out_state[state_base + kMaxRows * kStateLanes],
                      View<float>(kUbLPrev, kMaxRows * kStateLanes), rows * kStateLanes);
    const std::uint64_t row_out = static_cast<std::uint64_t>(item) * kMaxRows * kHeadSize;
    AscendC::DataCopy(out_acc[row_out], View<float>(kUbAcc, kMaxRows * kHeadSize), lanes);
    AscendC::DataCopy(out_rotated[row_out],
                      View<float>(kUbRotated, kMaxRows * kHeadSize), lanes);
    AscendC::DataCopy(out_context[row_out],
                      View<bfloat16_t>(kUbContext, kMaxRows * kHeadSize), lanes);
}

__aicore__ inline void Run_(__gm__ bfloat16_t* q_rot_gm, __gm__ std::int8_t* key_cache_gm,
                            __gm__ std::int8_t* value_cache_gm,
                            __gm__ float* scale_cache_gm, __gm__ std::uint32_t* block_gm,
                            __gm__ float* gate_gm, __gm__ float* pi_signs_gm,
                            __gm__ float* out_scores_gm, __gm__ float* out_state_gm,
                            __gm__ float* out_acc_gm, __gm__ float* out_rotated_gm,
                            __gm__ bfloat16_t* out_context_gm,
                            __gm__ std::int32_t* probe_gm, std::uint32_t num_requests,
                            std::uint32_t num_kv_heads, std::uint32_t rows,
                            std::uint32_t head_size, std::uint32_t block_size,
                            std::uint32_t cache_pages, std::uint32_t seq_pages,
                            std::uint32_t strips, std::uint32_t strip_begin,
                            std::uint32_t item_begin, std::uint32_t item_count)
{
    // `head_size` and `rows` are runtime operands while the butterfly's four
    // resident registers, the strip geometry and the GEMV's repeat strides are
    // 256 channels by construction. The kernel is only defined where they
    // agree, and this is the device half of a contract the host adapter checks
    // too -- a silently wrong head size would read one head's rows and write
    // another head's slice.
    if (head_size != kHeadSize) return;
    if (rows == 0u || rows > kMaxRows) return;
    if (block_size % kStripHeight != 0u) return;
    if (strips == 0u || seq_pages == 0u) return;
    if (strip_begin + strips > block_size / kStripHeight) return;

    Geometry geom;
    geom.num_kv_heads = num_kv_heads;
    geom.rows = rows;
    geom.head_size = head_size;
    geom.block_size = block_size;
    geom.cache_pages = cache_pages;
    geom.strips_in_page = block_size / kStripHeight;
    geom.seq_pages = seq_pages;
    geom.strips = strips;
    geom.strip_begin = strip_begin;

    AscendC::GlobalTensor<bfloat16_t> q_rot;
    AscendC::GlobalTensor<float> scale_cache, gate, pi_signs;
    AscendC::GlobalTensor<float> out_scores, out_state, out_acc, out_rotated;
    AscendC::GlobalTensor<std::int8_t> key_cache, value_cache;
    AscendC::GlobalTensor<std::uint32_t> block_table;
    AscendC::GlobalTensor<bfloat16_t> out_context;
    q_rot.SetGlobalBuffer(q_rot_gm);
    key_cache.SetGlobalBuffer(key_cache_gm);
    value_cache.SetGlobalBuffer(value_cache_gm);
    scale_cache.SetGlobalBuffer(scale_cache_gm);
    block_table.SetGlobalBuffer(block_gm);
    gate.SetGlobalBuffer(gate_gm);
    pi_signs.SetGlobalBuffer(pi_signs_gm);
    out_scores.SetGlobalBuffer(out_scores_gm);
    out_state.SetGlobalBuffer(out_state_gm);
    out_acc.SetGlobalBuffer(out_acc_gm);
    out_rotated.SetGlobalBuffer(out_rotated_gm);
    out_context.SetGlobalBuffer(out_context_gm);

    const std::uint32_t block_idx = static_cast<std::uint32_t>(AscendC::GetBlockIdx());
    const std::uint32_t block_num = static_cast<std::uint32_t>(AscendC::GetBlockNum());

    // What `GetBlockIdx` / `GetBlockNum` / `GetSubBlockNum` report under
    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY) is OBSERVED here rather
    // than assumed, because the safe MIX idiom (GetBlockIdx() /
    // GetSubBlockNum()) and the AIV-only one differ by exactly this. The
    // scatter ingress operator settled it for a one-block grid -- idx 0,
    // num 1, sub-block num 1, so every AIV is its own block -- and this kernel
    // re-publishes it rather than inheriting the claim.
    //
    // Four DataBlocks rather than four lanes: a scalar reaches Unified Buffer
    // through `Duplicate`, which writes a whole 32-byte block from a
    // vector-pipe broadcast, avoiding both a LocalTensor::SetValue and the V_S
    // fence a scalar UB write would need.
    if (probe_gm != nullptr && block_idx == 0u) {
        AscendC::GlobalTensor<std::int32_t> probe;
        probe.SetGlobalBuffer(probe_gm);
        auto probe_ub = View<std::int32_t>(kUbProbe, kProbeLanes);
        AscendC::Duplicate(probe_ub, static_cast<std::int32_t>(block_idx), 8);
        AscendC::Duplicate(probe_ub[8], static_cast<std::int32_t>(block_num), 8);
        AscendC::Duplicate(probe_ub[16],
                           static_cast<std::int32_t>(AscendC::GetSubBlockNum()), 8);
        AscendC::Duplicate(probe_ub[24], static_cast<std::int32_t>(item_count), 8);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtEgressReady);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtEgressReady);
        AscendC::DataCopy(probe[0], probe_ub, kProbeLanes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    }

    // Pi's sign diagonal, once per launch: 256 FP32 read by the butterfly's
    // own sign stage, which is the one part of the rotation that is not a
    // Hadamard sign the leaf can build in-register from `Arange` and `Xor`.
    AscendC::DataCopy(View<float>(kUbSign, kHeadSize), pi_signs[0], kHeadSize);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady);

    const std::uint32_t item_end = item_begin + item_count;
    for (std::uint32_t item = item_begin + block_idx; item < item_end;
         item += block_num) {
        const std::uint32_t request = item / num_kv_heads;
        const std::uint32_t head = item % num_kv_heads;
        if (request >= num_requests) continue;
        ProcessItem(q_rot, key_cache, value_cache, scale_cache, block_table, gate,
                    out_scores, out_state, out_acc, out_rotated, out_context, item,
                    request, head, geom);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
}

}  // namespace turboquant_single_token

extern "C" __global__ __aicore__ void turboquant_single_token_decode_aiv(
    __gm__ bfloat16_t* __restrict__ qRot, __gm__ std::int8_t* __restrict__ keyCache,
    __gm__ std::int8_t* __restrict__ valueCache, __gm__ float* __restrict__ scaleCache,
    __gm__ std::uint32_t* __restrict__ blockTable, __gm__ float* __restrict__ gate,
    __gm__ float* __restrict__ piSigns, __gm__ float* __restrict__ outScores,
    __gm__ float* __restrict__ outState, __gm__ float* __restrict__ outAcc,
    __gm__ float* __restrict__ outRotated, __gm__ bfloat16_t* __restrict__ outContext,
    __gm__ std::int32_t* __restrict__ launchProbe, uint32_t numRequests,
    uint32_t numKvHeads, uint32_t rows, uint32_t headSize, uint32_t blockSize,
    uint32_t cachePages, uint32_t seqPages, uint32_t strips, uint32_t stripBegin,
    uint32_t itemBegin, uint32_t itemCount)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    turboquant_single_token::Run_(qRot, keyCache, valueCache, scaleCache, blockTable,
                                  gate, piSigns, outScores, outState, outAcc, outRotated,
                                  outContext, launchProbe, numRequests, numKvHeads, rows,
                                  headSize, blockSize, cachePages, seqPages, strips,
                                  stripBegin, itemBegin, itemCount);
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

extern "C" __global__ __aicore__ void turboquant_single_token_decode_aiv(
    __gm__ bfloat16_t*, __gm__ std::int8_t*, __gm__ std::int8_t*, __gm__ float*,
    __gm__ std::uint32_t*, __gm__ float*, __gm__ float*, __gm__ float*, __gm__ float*,
    __gm__ float*, __gm__ float*, __gm__ bfloat16_t*, __gm__ std::int32_t*, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t);

extern "C" void launch_turboquant_single_token_decode_aiv(
    void* stream, std::uint32_t block_dim, std::uint16_t* q_rot, std::int8_t* key_cache,
    std::int8_t* value_cache, float* scale_cache, std::uint32_t* block_table, float* gate,
    float* pi_signs, float* out_scores, float* out_state, float* out_acc,
    float* out_rotated, std::uint16_t* out_context, std::int32_t* launch_probe,
    std::uint32_t num_requests, std::uint32_t num_kv_heads, std::uint32_t rows,
    std::uint32_t head_size, std::uint32_t block_size, std::uint32_t cache_pages,
    std::uint32_t seq_pages, std::uint32_t strips, std::uint32_t strip_begin,
    std::uint32_t item_begin, std::uint32_t item_count)
{
    turboquant_single_token_decode_aiv<<<block_dim, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(q_rot), key_cache, value_cache, scale_cache,
        block_table, gate, pi_signs,
        out_scores, out_state, out_acc, out_rotated,
        reinterpret_cast<bfloat16_t*>(out_context), launch_probe, num_requests,
        num_kv_heads, rows, head_size, block_size, cache_pages, seq_pages, strips,
        strip_begin, item_begin, item_count);
}

#endif
