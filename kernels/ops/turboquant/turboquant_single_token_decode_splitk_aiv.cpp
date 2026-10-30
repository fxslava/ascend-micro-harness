#include <cstdint>
#include "../../../include/turboquant_splitk_contract.hpp"

// True single-launch paged Split-K decode, Ascend 950PR / dav-3510 / CANN 9.2.
// Each block computes one disjoint page range and exports its partial softmax
// state to a task-indexed workspace partition of its own; a grid-wide barrier
// separates that pass from the reduction, which merges a head's K partials and
// emits the normalized, register-butterfly-rotated, sigmoid-gated BF16 context.
// Derived from single_token_decode_aiv; production kernels remain unchanged.
//
// WHY THERE IS NO LOCK HERE ANY MORE
//
// The first revision of this kernel reduced in place: a Global Memory
// test-and-set per (request, KV head), the arriving worker merging its state
// into the shared record inside the critical section, and the last arrival
// running the egress. It is correct and it does not scale. Every worker of a
// head serializes through one cache line, the merge itself -- 6 x 272 floats
// of Gather/Max/Exp/Mul/Add -- runs INSIDE the lock, and the line bounces
// between the competing cores on every handoff. Benchmarking on Ascend 950PR
// put the cost above 43,000 cycles at N <= 2048, where the decode itself is
// short enough that the contention dominates the launch.
//
// What replaces it is two passes over disjoint memory. Pass one: every worker
// packs its own (m, l, Acc) and writes it to `workspace + item *
// kPartitionFloats`, a partition no other worker addresses, as one contiguous
// MTE3 burst. No atomic, no spin, no shared line, and no worker waits for
// another. Pass two, after `AscendC::SyncAll`: one worker per (request, KV
// head) reads that head's K partitions, merges them with the same recurrence
// the lock used to run, and writes the context. The merge is now on the
// critical path of exactly one worker per head instead of all K of them, and
// it is reading its own partitions rather than contending for a line.
//
// The barrier is what the grid bound in the contract header is about: it only
// completes if every launched block is resident, so a split launch is capped
// at the physical block count. A single-split launch never reaches it -- that
// path runs the egress straight out of the decode pass and never touches the
// workspace at all.
#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
// After kernel_operator.h so TQ_TAIL_FN carries the device core-type
// attribute; see the header's note on the link error it otherwise produces.
#include "turboquant_decode_tail.hpp"
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
constexpr std::uint32_t kStageDepth = 2;
constexpr std::uint32_t kBlock = 32;
constexpr std::uint32_t kBankSkew = 32;
constexpr std::uint32_t kVecReg = 256;  // VECTOR_REG_WIDTH on dav-3510
// A scale instruction reads a full register, even for 16 active lanes.
constexpr std::uint32_t kScaleLandingBytes = kVecReg;
constexpr std::uint32_t kUbWindow = 192u * 1024u;
static_assert(kStageDepth == 2u && kUbWindow <= 192u * 1024u, "decode ring/window contract");

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkew;
}

// ---- the strip ingress and its dequantised pages ----
constexpr std::uint32_t kUbPackedKey[kStageDepth] = {
    0, NextSlot(kUbPackedKey[0], kPackedBytes)};
constexpr std::uint32_t kUbPackedVal[kStageDepth] = {
    NextSlot(kUbPackedKey[1], kPackedBytes), NextSlot(kUbPackedVal[0], kPackedBytes)};
// V-only scratch: FIFO/barriers serialize both slots; no DMA writer.
constexpr std::uint32_t kUbNib16 = NextSlot(kUbPackedVal[1], kPackedBytes);     // int16
constexpr std::uint32_t kUbKeyPage[kStageDepth] = {
    NextSlot(kUbNib16, kPanelElems * 2u), NextSlot(kUbKeyPage[0], kPanelElems * 4u)};
constexpr std::uint32_t kUbValuePage[kStageDepth] = {
    NextSlot(kUbKeyPage[1], kPanelElems * 4u), NextSlot(kUbValuePage[0], kPanelElems * 4u)};
// ---- the score GEMV ----
constexpr std::uint32_t kUbProducts = NextSlot(kUbValuePage[1], kPanelElems * 4u);
constexpr std::uint32_t kUbPartials = NextSlot(kUbProducts, kGemvProductElems * 4u);
constexpr std::uint32_t kUbFold = NextSlot(kUbPartials, kGemvPartials * 4u);
constexpr std::uint32_t kUbPanel[kStageDepth] = {
    NextSlot(kUbFold, kGemvPartials * 4u), NextSlot(kUbPanel[0], kMaxRows * kKeys * 4u)};
// ---- the rotated query panel, live for the whole task ----
// It arrives as bf16 -- kernel No.5 writes it, and RFC section 4 types the
// operand `uint16_t`, so No.3 and No.4 take the same tensor at the same call
// site -- and is converted once per task into the FP32 panel the GEMV reads.
// bf16 -> float is a listed Cast pair and exact, so the conversion costs one
// instruction and no accuracy.
constexpr std::uint32_t kUbQueryRaw = NextSlot(kUbPanel[1], kMaxRows * kKeys * 4u);
constexpr std::uint32_t kUbQuery = NextSlot(kUbQueryRaw, kMaxRows * kHeadSize * 2u);
// ---- the two scale slices ----
constexpr std::uint32_t kUbKeyScale[kStageDepth] = {
    NextSlot(kUbQuery, kMaxRows * kHeadSize * 4u), NextSlot(kUbKeyScale[0], kScaleLandingBytes)};
constexpr std::uint32_t kUbValueScale[kStageDepth] = {
    NextSlot(kUbKeyScale[1], kScaleLandingBytes), NextSlot(kUbValueScale[0], kScaleLandingBytes)};
// ---- the carried state ----
constexpr std::uint32_t kUbMPrev = NextSlot(kUbValueScale[1], kScaleLandingBytes);
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
// ---- the tail mask ----
// Four operands for a partial strip. The three float buffers are a whole
// vector register each, not 16 lanes: `CreateVecIndex` and `Duplicate` write a
// full register whatever the element count says, which is the same rule this
// arena already honours behind `Brcb`'s destinations. `kUbTailSel` holds one
// bit per key lane and needs only a DataBlock, but gets a whole slot so the
// next offset stays aligned.
constexpr std::uint32_t kUbLaneIndex = NextSlot(kUbContext, kMaxRows * kHeadSize * 2u);
constexpr std::uint32_t kUbMaskedFill = NextSlot(kUbLaneIndex, kVecReg);
constexpr std::uint32_t kUbZeroFill = NextSlot(kUbMaskedFill, kVecReg);
constexpr std::uint32_t kUbTailSel = NextSlot(kUbZeroFill, kVecReg);
// ---- the launch probe ----
constexpr std::uint32_t kProbeLanes = 32;
constexpr std::uint32_t kUbProbe = NextSlot(kUbTailSel, kBlock);
constexpr std::uint32_t kArenaEnd = kUbProbe + kProbeLanes * 4u;

static_assert(kUbPackedKey[1] % kBlock == 0u && kUbPackedVal[1] % kBlock == 0u &&
                  kUbNib16 % kBlock == 0u && kUbKeyPage[1] % kBlock == 0u &&
                  kUbValuePage[1] % kBlock == 0u && kUbProducts % kBlock == 0u &&
                  kUbPartials % kBlock == 0u && kUbFold % kBlock == 0u &&
                  kUbPanel[1] % kBlock == 0u && kUbQueryRaw % kBlock == 0u &&
                  kUbQuery % kBlock == 0u && kUbKeyScale[1] % kBlock == 0u &&
                  kUbValueScale[1] % kBlock == 0u && kUbMPrev % kBlock == 0u &&
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


static_assert(kUbPackedKey[0] % kBlock == 0u && kUbPackedKey[1] % kBlock == 0u &&
    kUbPackedKey[1] == kUbPackedKey[0] + kPackedBytes + kBankSkew &&
    ((kUbPackedKey[1] - kUbPackedKey[0]) / kBlock) % 16u != 0u,
    "AKA1002/1003/3003: aligned, disjoint, bank-skew ring slots");
static_assert(kUbPackedVal[0] % kBlock == 0u && kUbPackedVal[1] % kBlock == 0u &&
    kUbPackedVal[1] == kUbPackedVal[0] + kPackedBytes + kBankSkew &&
    ((kUbPackedVal[1] - kUbPackedVal[0]) / kBlock) % 16u != 0u,
    "AKA1002/1003/3003: aligned, disjoint, bank-skew ring slots");
static_assert(kUbKeyPage[0] % kBlock == 0u && kUbKeyPage[1] % kBlock == 0u &&
    kUbKeyPage[1] == kUbKeyPage[0] + kPanelElems * 4u + kBankSkew &&
    ((kUbKeyPage[1] - kUbKeyPage[0]) / kBlock) % 16u != 0u,
    "AKA1002/1003/3003: aligned, disjoint, bank-skew ring slots");
static_assert(kUbValuePage[0] % kBlock == 0u && kUbValuePage[1] % kBlock == 0u &&
    kUbValuePage[1] == kUbValuePage[0] + kPanelElems * 4u + kBankSkew &&
    ((kUbValuePage[1] - kUbValuePage[0]) / kBlock) % 16u != 0u,
    "AKA1002/1003/3003: aligned, disjoint, bank-skew ring slots");
static_assert(kUbPanel[0] % kBlock == 0u && kUbPanel[1] % kBlock == 0u &&
    kUbPanel[1] == kUbPanel[0] + kMaxRows * kKeys * 4u + kBankSkew &&
    ((kUbPanel[1] - kUbPanel[0]) / kBlock) % 16u != 0u,
    "AKA1002/1003/3003: aligned, disjoint, bank-skew ring slots");
static_assert(kUbKeyScale[0] % kBlock == 0u && kUbKeyScale[1] % kBlock == 0u &&
    kUbKeyScale[1] == kUbKeyScale[0] + kScaleLandingBytes + kBankSkew &&
    ((kUbKeyScale[1] - kUbKeyScale[0]) / kBlock) % 16u != 0u,
    "AKA1002/1003/3003: aligned, disjoint, bank-skew ring slots");
static_assert(kUbValueScale[0] % kBlock == 0u && kUbValueScale[1] % kBlock == 0u &&
    kUbValueScale[1] == kUbValueScale[0] + kScaleLandingBytes + kBankSkew &&
    ((kUbValueScale[1] - kUbValueScale[0]) / kBlock) % 16u != 0u,
    "AKA1002/1003/3003: aligned, disjoint, bank-skew ring slots");
static_assert(kUbKeyScale[0] + kVecReg <= kUbKeyScale[1] &&
              kUbKeyScale[1] + kVecReg <= kUbValueScale[0] &&
              kUbValueScale[0] + kVecReg <= kUbValueScale[1] &&
              kUbValueScale[1] + kVecReg <= kUbMPrev,
              "scale register overreads must not reach another DMA landing");

// Event ids, static code 0..5 (6 and 7 reserved). One token in flight per
// (route, id).
//
// The reduction's record ring gets ids of its own rather than reusing the
// strip ring's. Both rings run in the same block -- a reducing worker is a
// worker that already decoded a split -- and the strip ring's credits are
// fully drained before the barrier, so reuse would be legal. It would also
// make the trace census of section 5 unreadable: that audit keys events by
// (route, id) and expects a fixed SET/WAIT count per strip-ring id, and two
// phases sharing an id would be indistinguishable from a double SET.
constexpr std::int32_t kEvtIngressReady = 0;  // MTE2_V
constexpr std::int32_t kEvtPackedFree = 0;    // V_MTE2
constexpr std::int32_t kEvtScaleFree = 2;     // V_MTE2
constexpr std::int32_t kEvtRecordReady = 2;   // MTE2_V, the reduction's ring
constexpr std::int32_t kEvtRecordFree = 4;    // V_MTE2, the reduction's ring
constexpr std::int32_t kEvtReduceGate = 4;    // MTE2_V, the reducer's gate load
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
    // Live tokens in the whole sequence, in the request's own logical token
    // numbering -- NOT this split's share of them. That is what makes the tail
    // work under Split-K without any per-split arithmetic: `page_begin` already
    // places this worker's window in that numbering, so the one worker whose
    // range contains the last live token finds a partial strip and every other
    // finds only full ones.
    std::uint32_t seq_len;
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
// scale slices per ring slot. Packed and scale credits have different last
// readers, so separate route IDs guard their reuse.
__aicore__ inline void LoadStrip(const AscendC::GlobalTensor<std::int8_t>& key_cache,
                                 const AscendC::GlobalTensor<std::int8_t>& value_cache,
                                 const AscendC::GlobalTensor<float>& scale_cache,
                                 std::uint32_t page, std::uint32_t head,
                                 std::uint32_t strip, const Geometry& geom, std::uint32_t slot)
{
    const std::uint64_t strip_base = StripByteOffset(page, head, strip, geom);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtPackedFree + slot);
    AscendC::DataCopy(View<std::int8_t>(kUbPackedKey[slot], kPackedBytes),
                      key_cache[strip_base], kPackedBytes);
    AscendC::DataCopy(View<std::int8_t>(kUbPackedVal[slot], kPackedBytes),
                      value_cache[strip_base], kPackedBytes);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtScaleFree + slot);
    AscendC::DataCopy(View<float>(kUbKeyScale[slot], kKeys),
                      scale_cache[ScaleOffset(0u, page, head, strip, geom)], kKeys);
    AscendC::DataCopy(View<float>(kUbValueScale[slot], kKeys),
                      scale_cache[ScaleOffset(1u, page, head, strip, geom)], kKeys);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady + slot);
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
__aicore__ inline void DequantStrip(std::uint32_t slot)
{
    ascend_micro::NativeDequant4Engine<float> key_dq;
    key_dq.packed = View<std::int8_t>(kUbPackedKey[slot], kPackedBytes);
    key_dq.signed_nibbles = View<std::int16_t>(kUbNib16, kPanelElems);
    key_dq.output = View<float>(kUbKeyPage[slot], kPanelElems);
    ascend_micro::turboquant_dequant4_signed(key_dq, kPanelElems);
    AscendC::PipeBarrier<PIPE_V>();

    ascend_micro::NativeDequant4Engine<float> value_dq;
    value_dq.packed = View<std::int8_t>(kUbPackedVal[slot], kPackedBytes);
    value_dq.signed_nibbles = View<std::int16_t>(kUbNib16, kPanelElems);
    value_dq.output = View<float>(kUbValuePage[slot], kPanelElems);
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


// ------------------------------------------------------------- the census
// Nine replicated DataBlocks in one MTE3 burst. Every lane is written by the
// Vector pipe from an immediate scalar -- no SIMT store, no scalar read of
// Unified Buffer -- and every field is replicated eight ways, so a reader that
// finds two lanes of a field disagreeing saw a torn burst rather than a value.
__aicore__ inline void FinishMetric(
    __gm__ turboquant_splitk::WorkerMetrics* metric, std::uint32_t partition,
    bool reduced, bool tail, std::uint32_t pages, std::uint64_t export_region)
{
    // The caller has retired the partition burst, or the out_context burst.
    const auto stamp = static_cast<std::uint64_t>(AscendC::GetSystemCycle());
    auto buf = View<std::uint32_t>(kUbPackedKey[0], turboquant_splitk::kMetricLanes);
    AscendC::Duplicate(buf, 1u, 8);
    AscendC::Duplicate(buf[8], partition, 8);
    AscendC::Duplicate(buf[16], static_cast<std::uint32_t>(reduced), 8);
    AscendC::Duplicate(buf[24], pages, 8);
    AscendC::Duplicate(buf[32], static_cast<std::uint32_t>(tail), 8);
    AscendC::Duplicate(buf[40], static_cast<std::uint32_t>(stamp >> 32), 8);
    AscendC::Duplicate(buf[48], static_cast<std::uint32_t>(stamp), 8);
    AscendC::Duplicate(buf[56], static_cast<std::uint32_t>(export_region >> 32), 8);
    AscendC::Duplicate(buf[64], static_cast<std::uint32_t>(export_region), 8);
    AscendC::GlobalTensor<std::uint32_t> dst;
    dst.SetGlobalBuffer(metric->exported);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::DataCopy(dst, buf, turboquant_splitk::kMetricLanes);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(2);
}

// ---------------------------------------------------------- the egress DMA
// `rows` consecutive rows of the [requests][H_q][d] context panel, which is
// one contiguous burst for the same reason the query ingress is.
__aicore__ inline void StoreContext(const AscendC::GlobalTensor<bfloat16_t>& out_context,
                                    std::uint32_t head_item, const Geometry& geom)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtEgressReady);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtEgressReady);
    AscendC::DataCopy(out_context[static_cast<std::uint64_t>(head_item) * geom.rows *
                                  kHeadSize],
                      View<bfloat16_t>(kUbContext, geom.rows * kHeadSize),
                      geom.rows * kHeadSize);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
}

// ------------------------------------------------- pass one: the export
// The whole of what used to be the critical section. There is no acquire, no
// arrival ticket, no merge and no release: the worker packs its own state and
// writes it to a partition nothing else addresses.
//
// `splits == 1` has no partition to write and no second pass to wait for, so
// it bypasses the workspace entirely and runs the egress here. That is not an
// optimisation bolted on the side; it is the same `Egress` the reduction
// calls, over the same buffers, so the one-split answer and the K-split answer
// come out of identical code.
//
// It returns the region it cost rather than publishing it, because a block
// that goes on to reduce must publish its census once, at the end, with every
// field final: two `FinishMetric` calls would leave the second overwriting the
// export measurement this one exists to take.
__aicore__ inline std::uint64_t ExportPartial(
    const AscendC::GlobalTensor<float>& workspace,
    const AscendC::GlobalTensor<bfloat16_t>& out_context, std::uint32_t item,
    std::uint32_t head_item, std::uint32_t splits, const Geometry& geom)
{
    const auto export_begin = static_cast<std::uint64_t>(AscendC::GetSystemCycle());
    if (splits == 1u) {
        Egress(geom);
        StoreContext(out_context, head_item, geom);
        return static_cast<std::uint64_t>(AscendC::GetSystemCycle()) - export_begin;
    }

    // Strip scratch is dead by now, so the packed record lands in a retired
    // value page: kMaxRows * kRecordFloats floats fit one strip panel, which
    // the arena asserts.
    const std::uint32_t state_lanes = geom.rows * turboquant_splitk::kRecordFloats;
    auto packed = View<float>(kUbValuePage[0], state_lanes);
    PackHeadState(
        reinterpret_cast<__ubuf__ float*>(packed.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(View<float>(kUbMPrev, 64).GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(View<float>(kUbLPrev, 64).GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(
            View<float>(kUbAcc, geom.rows * kHeadSize).GetPhyAddr()),
        static_cast<std::uint16_t>(geom.rows));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
    // One contiguous rows x 272-float burst into this task's own partition.
    AscendC::DataCopy(workspace[static_cast<std::uint64_t>(item) *
                                turboquant_splitk::kPartitionFloats],
                      packed, state_lanes);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(3);
    return static_cast<std::uint64_t>(AscendC::GetSystemCycle()) - export_begin;
}

// ------------------------------------------------ pass two: the reduction
// One worker per (request, KV head), reading that head's K partitions and
// merging them with `MergeHeadState` -- the same log-sum-exp recurrence the
// lock used to run, unchanged, now over partitions this worker owns rather
// than a line K workers contend for.
//
// The running state is seeded exactly as the decode seeds it, so split 0 goes
// through the merge like every other split: exp(-3e38 - m0) is 0 and l starts
// at 0, which makes the seed the neutral element of the recurrence rather than
// a case the loop has to special-case.
//
// A two-slot record ring, not a lock-step fetch. A record is rows * 1088 bytes
// and there are up to kMaxSplits of them, so the MTE2 of record i + 1 is worth
// overlapping with the merge of record i. The two retired key pages are the
// slots; the ring is primed, drained and capacity-one on both routes,
// including the slot a K below the ring depth never fills.
__aicore__ inline void ReduceHead(const AscendC::GlobalTensor<float>& workspace,
                                  const AscendC::GlobalTensor<float>& gate,
                                  const AscendC::GlobalTensor<bfloat16_t>& out_context,
                                  std::uint32_t head_item, std::uint32_t splits,
                                  const Geometry& geom)
{
    const std::uint32_t rows = geom.rows;
    const std::uint32_t lanes = rows * kHeadSize;
    const std::uint32_t state_lanes = rows * turboquant_splitk::kRecordFloats;
    const std::uint64_t row_base =
        static_cast<std::uint64_t>(head_item) * rows * kHeadSize;

    // This block decoded some OTHER head's split, so every buffer the egress
    // reads has to be re-established for the head being reduced. The gate is
    // the one egress input that is not carried by the partials.
    AscendC::DataCopy(View<float>(kUbGate, kMaxRows * kHeadSize), gate[row_base], lanes);
    AscendC::Duplicate(View<float>(kUbMPrev, kMaxRows * kStateLanes), -3.0e38f,
                       static_cast<std::int32_t>(rows * kStateLanes));
    AscendC::Duplicate(View<float>(kUbLPrev, kMaxRows * kStateLanes), 0.0f,
                       static_cast<std::int32_t>(rows * kStateLanes));
    AscendC::Duplicate(View<float>(kUbAcc, kMaxRows * kHeadSize), 0.0f,
                       static_cast<std::int32_t>(lanes));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtReduceGate);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtReduceGate);
    AscendC::PipeBarrier<PIPE_V>();

    for (std::uint32_t slot = 0; slot < kStageDepth; ++slot)
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtRecordFree + slot);
    for (std::uint32_t split = 0; split < splits && split < kStageDepth; ++split) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtRecordFree + split);
        AscendC::DataCopy(View<float>(kUbKeyPage[split], state_lanes),
                          workspace[turboquant_splitk::PartitionOffset(head_item, split,
                                                                       splits)],
                          state_lanes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtRecordReady + split);
    }
    for (std::uint32_t split = 0; split < splits; ++split) {
        const std::uint32_t slot = split % kStageDepth;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtRecordReady + slot);
        MergeHeadState(
            reinterpret_cast<__ubuf__ float*>(
                View<float>(kUbKeyPage[slot], state_lanes).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbMPrev, 64).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbLPrev, 64).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbAcc, lanes).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbProducts, 128).GetPhyAddr()),
            static_cast<std::uint16_t>(rows));
        // The merge reads this slot to its last register; the refill DMA may
        // not overwrite it until that read has retired on the Vector pipe.
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtRecordFree + slot);
        const std::uint32_t next = split + kStageDepth;
        if (next < splits) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtRecordFree + slot);
            AscendC::DataCopy(View<float>(kUbKeyPage[slot], state_lanes),
                              workspace[turboquant_splitk::PartitionOffset(head_item, next,
                                                                           splits)],
                              state_lanes);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtRecordReady + slot);
        }
    }
    // Consume exactly the final credit in every reuse place, including the
    // slot a K below the ring depth never filled.
    for (std::uint32_t slot = 0; slot < kStageDepth; ++slot)
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtRecordFree + slot);

    Egress(geom);
    StoreContext(out_context, head_item, geom);
}
__aicore__ inline std::uint64_t ProcessItem(const AscendC::GlobalTensor<bfloat16_t>& q_rot,
                                   const AscendC::GlobalTensor<std::int8_t>& key_cache,
                                   const AscendC::GlobalTensor<std::int8_t>& value_cache,
                                   const AscendC::GlobalTensor<float>& scale_cache,
                                   __gm__ const std::uint32_t* block_table,
                                   const AscendC::GlobalTensor<float>& gate,
                                   const AscendC::GlobalTensor<float>& workspace,
                                   const AscendC::GlobalTensor<bfloat16_t>& out_context,
                                   std::uint32_t splits,
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
    gemv.key_page = View<float>(kUbKeyPage[0], kPanelElems);
    gemv.query = View<float>(kUbQuery, kMaxRows * kHeadSize);
    gemv.products = View<float>(kUbProducts, kGemvProductElems);
    gemv.partials = View<float>(kUbPartials, kGemvPartials);
    gemv.fold = View<float>(kUbFold, kGemvPartials);
    gemv.panel = View<float>(kUbPanel[0], kMaxRows * kKeys);
    gemv.query_base = kUbQuery;
    gemv.panel_base = kUbPanel[0];
    gemv.position = kUb;

    ascend_micro::NativeDecodeRecurrenceEngine rec;
    rec.panel = View<float>(kUbPanel[0], kMaxRows * kKeys);
    rec.key_scale = View<float>(kUbKeyScale[0], kKeys);
    rec.value_scale = View<float>(kUbValueScale[0], kKeys);
    rec.value_page = View<float>(kUbValuePage[0], kPanelElems);
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
    rec.panel_base = kUbPanel[0];
    rec.m_base = kUbMPrev;
    rec.l_base = kUbLPrev;
    rec.acc_base = kUbAcc;
    rec.lane_index = View<float>(kUbLaneIndex, kKeys);
    rec.masked_fill = View<float>(kUbMaskedFill, kKeys);
    rec.zero_fill = View<float>(kUbZeroFill, kKeys);
    rec.tail_sel = View<std::uint8_t>(kUbTailSel, kKeys);
    rec.position = kUb;

    // Capacity-one feedback places, including unused slots (N=0/1).
    for (std::uint32_t slot = 0; slot < kStageDepth; ++slot) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtPackedFree + slot);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtScaleFree + slot);
    }
    // Live strips of THIS split's page range. A split whose range lies
    // entirely past the last live token gets 0 and contributes the neutral
    // element, which is the same path an empty `Partition` range already takes.
    // As in the plain operator: built once per item, read by every strip.
    rec.prime_tail_mask(turboquant_decode_tail::kMaskedScore);
    AscendC::PipeBarrier<PIPE_V>();
    const std::uint32_t count = turboquant_decode_tail::LiveStripsInWindow(
        geom.seq_len, geom.page_begin, geom.seq_pages, geom.block_size,
        geom.strip_begin, geom.strips);
    // Prefill both landings before the first Vector wait. Refill is issued
    // only after retiring that slot's recurrence, never ahead of its credit.
    for (std::uint32_t index = 0; index < count && index < kStageDepth; ++index) {
        const auto page = block_table[request * geom.table_stride + geom.page_begin + index / geom.strips];
        LoadStrip(key_cache, value_cache, scale_cache, page, head,
                  geom.strip_begin + index % geom.strips, geom, index % kStageDepth);
    }
    // Prepare strip i before recurring i-1. The score/value/scale pages stay
    // live in separate slots; all arithmetic and recurrence state stay on V.
    for (std::uint32_t tick = 0; tick <= count; ++tick) {
        if (tick < count) {
            const std::uint32_t slot = tick % kStageDepth;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady + slot);
            DequantStrip(slot);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtPackedFree + slot);
            gemv.key_page = View<float>(kUbKeyPage[slot], kPanelElems);
            gemv.panel = View<float>(kUbPanel[slot], kMaxRows * kKeys);
            gemv.panel_base = kUbPanel[slot];
            ascend_micro::vector_decode_gemv_strip(gemv, rows);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (tick > 0u) {
            const std::uint32_t slot = (tick - 1u) % kStageDepth;
            rec.panel = View<float>(kUbPanel[slot], kMaxRows * kKeys);
            rec.panel_base = kUbPanel[slot];
            rec.key_scale = View<float>(kUbKeyScale[slot], kKeys);
            rec.value_scale = View<float>(kUbValueScale[slot], kKeys);
            rec.value_page = View<float>(kUbValuePage[slot], kPanelElems);
            // One partial strip per sequence at most, so `active` is 16
            // on every strip of every other split and the aligned path is
            // unchanged. `count` is the live prefix, so `active` is never 0.
            const std::uint32_t active = turboquant_decode_tail::WalkActive(
                tick - 1u, geom.strips, geom.page_begin, geom.strip_begin,
                geom.block_size, geom.seq_len);
            if (active == turboquant_decode_tail::kStripTokens)
                ascend_micro::vector_decode_recur_rows(rec, rows);
            else
                ascend_micro::vector_decode_recur_rows_masked(rec, rows, active);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtScaleFree + slot);
            const std::uint32_t index = tick - 1u + kStageDepth;
            if (index < count) {
                const auto page = block_table[request * geom.table_stride + geom.page_begin + index / geom.strips];
                LoadStrip(key_cache, value_cache, scale_cache, page, head,
                          geom.strip_begin + index % geom.strips, geom, slot);
            }
        }
    }
    // Consume exactly the final credit in every reuse place. Ready places
    // are empty already. This also orders scratch reuse in Split-K reduction.
    for (std::uint32_t slot = 0; slot < kStageDepth; ++slot) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtPackedFree + slot);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtScaleFree + slot);
    }

    return ExportPartial(workspace, out_context, item,
                         request * geom.num_kv_heads + head, splits, geom);
}


}  // namespace turboquant_splitk_device

extern "C" __global__ __aicore__ void turboquant_single_token_decode_splitk_aiv(
    __gm__ bfloat16_t* q_rot, __gm__ std::int8_t* key, __gm__ std::int8_t* value,
    __gm__ float* scales, __gm__ uint32_t* table, __gm__ float* gate,
    __gm__ float* signs, __gm__ float* workspace,
    __gm__ turboquant_splitk::WorkerMetrics* metrics, __gm__ bfloat16_t* out,
    uint32_t requests, uint32_t heads, uint32_t rows,
    uint32_t block_size, uint32_t cache_pages, uint32_t pages,
    uint32_t splits, uint32_t seq_len)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    using namespace turboquant_splitk_device;
    // Every guard below is a function of the launch operands alone, so all
    // blocks evaluate it identically and either all of them return or none
    // does. That is what makes the barrier further down safe to place after
    // them: a grid in which some blocks returned early and the rest entered
    // `SyncAll` is a hung launch, not a wrong answer.
    if (requests == 0u || heads == 0u || rows == 0u || rows > kMaxRows ||
        block_size == 0u || block_size % kKeys != 0u || pages == 0u ||
        pages > cache_pages || splits == 0u ||
        splits > turboquant_splitk::kMaxSplits) return;
    // The barrier only completes with every launched block resident, so a
    // split launch may not span hardware waves. See the contract header.
    if (!turboquant_splitk::GridFitsOneWave(requests, heads, splits)) return;
    // `seq_len == 0` means every token of the `pages` allocated pages is live:
    // the pre-tail contract, so existing callers keep their exact behaviour.
    const uint32_t live = seq_len == 0u ? pages * block_size : seq_len;
    if (!turboquant_decode_tail::SeqLenInCapacity(live, pages, block_size)) return;
    const auto item = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const auto tasks = requests * heads * splits;
    if (item >= tasks) return;
    const auto head_item = item / splits;
    const auto range = turboquant_splitk::Partition(pages, splits, item % splits);
    Geometry geom{heads, rows, kHeadSize, block_size, cache_pages,
                  block_size / kKeys, range.count, block_size / kKeys,
                  range.begin, pages, 0u, live};
    AscendC::GlobalTensor<bfloat16_t> q, context;
    AscendC::GlobalTensor<std::int8_t> key_cache, value_cache;
    AscendC::GlobalTensor<float> scale_cache, gate_tensor, pi_signs, partials;
    AscendC::GlobalTensor<uint32_t> block_table;
    q.SetGlobalBuffer(q_rot);
    context.SetGlobalBuffer(out);
    key_cache.SetGlobalBuffer(key);
    value_cache.SetGlobalBuffer(value);
    scale_cache.SetGlobalBuffer(scales);
    gate_tensor.SetGlobalBuffer(gate);
    pi_signs.SetGlobalBuffer(signs);
    partials.SetGlobalBuffer(workspace);
    AscendC::DataCopy(View<float>(kUbSign, kHeadSize), pi_signs, kHeadSize);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady);
    const auto export_region =
        ProcessItem(q, key_cache, value_cache, scale_cache, table, gate_tensor,
                    partials, context, splits, item,
                    head_item / heads, head_item % heads, geom);

    bool reduced = false;
    if (splits > 1u) {
        // Grid-wide: every partition is in Global Memory once this returns.
        // The condition is `splits > 1`, which every block computes from the
        // same operand, so the barrier is entered by all of them or by none.
        //
        // `isAIVOnly = true`, which is where this departs from the reference
        // operator's `SyncAll<false>()`. That template parameter is not a
        // strictness knob: `false` raises SYNC_AIC_AIV_FLAG and waits for the
        // Cube cores to answer it, which is right for the reference because it
        // is a MIX_AIC_1_2 kernel. This one is KERNEL_TYPE_AIV_ONLY, so its
        // launch has no Cube core to answer, and `false` here would wait for a
        // participant that does not exist. `true` raises SYNC_AIV_ONLY_ALL.
        //
        // The default config's trigger pipe is PIPE_ALL, so the barrier opens
        // with its own `PipeBarrier<PIPE_ALL>()` and this block's export MTE3
        // is drained before the flag is raised. Nothing here needs to repeat
        // that drain on either side.
        AscendC::SyncAll<true>();

        // One reducer per (request, KV head). The reducing blocks are the
        // first `requests * heads` of a grid `splits` times that size, so a
        // reducer is a block that already decoded some split -- usually of a
        // different head -- and has to re-establish every buffer the egress
        // reads.
        reduced = item < requests * heads;
        if (reduced) ReduceHead(partials, gate_tensor, context, item, splits, geom);
    }
    // Once per block, with every field final: the fused worker and the
    // reducer are the two that wrote this head's context.
    FinishMetric(metrics + item,
                 splits == 1u ? turboquant_splitk::kNoPartition : item, reduced,
                 splits == 1u || reduced, geom.seq_pages, export_region);
}

// Diagnostic entry uses the production ProcessItem and export path unchanged.
// Selects one logical-page/strip window without rebasing the block table.
//
// It launches one block at splits = 1, so it takes the fused path: no
// partition is written, no barrier is entered, and the workspace operand is
// present only to keep one signature between this entry and production.
extern "C" __global__ __aicore__ void turboquant_splitk_boundary_aiv(
    __gm__ bfloat16_t* q_rot, __gm__ std::int8_t* key, __gm__ std::int8_t* value,
    __gm__ float* scales, __gm__ uint32_t* table, __gm__ float* gate,
    __gm__ float* signs, __gm__ float* workspace,
    __gm__ turboquant_splitk::WorkerMetrics* metrics, __gm__ bfloat16_t* out,
    uint32_t head, uint32_t rows, uint32_t page_begin, uint32_t page_count,
    uint32_t strips, uint32_t strip_begin)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    using namespace turboquant_splitk_device;
    if (head >= 4u || rows == 0u || rows > kMaxRows || page_begin >= 8u ||
        page_count == 0u || page_count > 8u - page_begin || strips == 0u ||
        strips > 8u || strip_begin > 8u - strips) return;
    // The diagnostic entry has no seq_len operand, so it declares the whole
    // window live, which is what it meant before the tail change.
    Geometry geom{4u, rows, kHeadSize, 128u, 9u, 8u, page_count, strips,
                  page_begin, 8u, strip_begin,
                  static_cast<std::uint32_t>(turboquant_decode_tail::WholeWindowTokens(
                      page_begin, page_count, 128u, strip_begin, strips))};
    AscendC::GlobalTensor<bfloat16_t> q, context;
    AscendC::GlobalTensor<std::int8_t> keys, values;
    AscendC::GlobalTensor<float> scale_tensor, gates, sign_tensor, partials;
    q.SetGlobalBuffer(q_rot);
    context.SetGlobalBuffer(out);
    keys.SetGlobalBuffer(key);
    values.SetGlobalBuffer(value);
    scale_tensor.SetGlobalBuffer(scales);
    gates.SetGlobalBuffer(gate);
    sign_tensor.SetGlobalBuffer(signs);
    partials.SetGlobalBuffer(workspace);
    AscendC::DataCopy(View<float>(kUbSign, kHeadSize), sign_tensor, kHeadSize);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtIngressReady);
    const auto export_region =
        ProcessItem(q, keys, values, scale_tensor, table, gates, partials,
                    context, 1u, head, 0u, head, geom);
    FinishMetric(metrics + head, turboquant_splitk::kNoPartition, false, true,
                 page_count, export_region);
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

#include <stdexcept>

#include "turboquant_decode_tail.hpp"

extern "C" __global__ __aicore__ void turboquant_single_token_decode_splitk_aiv(
    __gm__ bfloat16_t*, __gm__ std::int8_t*, __gm__ std::int8_t*, __gm__ float*,
    __gm__ uint32_t*, __gm__ float*, __gm__ float*, __gm__ float*,
    __gm__ turboquant_splitk::WorkerMetrics*, __gm__ bfloat16_t*,
    uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t);

extern "C" void launch_turboquant_single_token_decode_splitk_aiv(
    void* stream, std::uint16_t* q_rot, std::int8_t* key, std::int8_t* value,
    float* scales, std::uint32_t* table, float* gate, float* signs,
    float* workspace, turboquant_splitk::WorkerMetrics* metrics,
    std::uint16_t* out, std::uint32_t requests, std::uint32_t heads,
    std::uint32_t rows, std::uint32_t block_size, std::uint32_t cache_pages,
    std::uint32_t pages, std::uint32_t splits, std::uint32_t seq_len)
{
    // 0 means "the whole allocation is live", the pre-tail contract. Anything
    // else must fit the pages the caller says it allocated; the device returns
    // on a violation, which on a grid of blocks is silent, so the host raises.
    if (seq_len != 0u &&
        !turboquant_decode_tail::SeqLenInCapacity(seq_len, pages, block_size))
        throw std::invalid_argument(
            "single_token_decode_splitk: seqLen exceeds the pages allocated to "
            "the sequence");
    // The in-launch reduction's barrier needs every block resident. The device
    // guard returns rather than hanging, but a silent empty output is a worse
    // diagnostic than a thrown argument, so the host refuses the launch.
    if (!turboquant_splitk::GridFitsOneWave(requests, heads, splits))
        throw std::invalid_argument(
            "single_token_decode_splitk: a split launch must fit one hardware "
            "wave (requests * heads * splits <= kMaxCoResidentBlocks)");
    turboquant_single_token_decode_splitk_aiv<<<requests * heads * splits,
                                             nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(q_rot), key, value, scales, table, gate, signs,
        workspace, metrics, reinterpret_cast<bfloat16_t*>(out), requests, heads, rows,
        block_size, cache_pages, pages, splits, seq_len);
}
extern "C" __global__ __aicore__ void turboquant_splitk_boundary_aiv(
    __gm__ bfloat16_t*, __gm__ std::int8_t*, __gm__ std::int8_t*, __gm__ float*,
    __gm__ uint32_t*, __gm__ float*, __gm__ float*, __gm__ float*,
    __gm__ turboquant_splitk::WorkerMetrics*, __gm__ bfloat16_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

extern "C" void launch_turboquant_splitk_boundary_aiv(
    void* stream, std::uint16_t* q, std::int8_t* key, std::int8_t* value,
    float* scales, std::uint32_t* table, float* gate, float* signs,
    float* workspace, turboquant_splitk::WorkerMetrics* metrics,
    std::uint16_t* out, std::uint32_t head, std::uint32_t rows,
    std::uint32_t begin, std::uint32_t count, std::uint32_t strips,
    std::uint32_t strip_begin)
{
    turboquant_splitk_boundary_aiv<<<1, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(q), key, value, scales, table, gate, signs,
        workspace, metrics, reinterpret_cast<bfloat16_t*>(out), head, rows, begin,
        count, strips, strip_begin);
}
#endif
