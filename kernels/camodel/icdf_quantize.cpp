#include <cstdint>

// CAModel wrapper for the AIV-only inverse-CDF int4x2 quantization leaf
// (routines/routine_vector_icdf_quantize.hpp).
//
// What this kernel proves
// -----------------------
// 1. The register-resident scaled-sigmoid chain (vneg/vexp/vadds/vdiv/vadds)
//    plus the Cast->int4x2 + Or + DIST_PACK_B16 compression store runs on one
//    Vector core with full-lane (unmasked, 256 B) instructions and produces,
//    byte for byte, the low-nibble-first packed int4x2 layout the oracle
//    predicts -- for FP16 inputs directly and for BF16 inputs after the
//    on-the-fly BF16->half widening.
// 2. A depth-2 MTE2 -> Vector -> MTE3 ring over hand-placed UB slots with one
//    event id per route (single token in flight per route) keeps the vector
//    pipe fed: back edges primed before the loop, drained after it, sets ==
//    waits per (route, id), no queue left ungated (MICROROUTINE_DESIGN_
//    GUIDELINES sections 3.1/3.4).
// 3. Zero scalar traps: the leaf never leaves the vector pipe, no TPipe is
//    instantiated, and the run log must contain no invalid-ldst/su_ccu_mpu_err
//    lines (checked by scripts/run_icdf_camodel.sh).
//
// What it deliberately omits: any reduction over the block, ablocking with a
// real Cube producer (the upstream Fixpipe export is modelled by the GM
// fixture), and multi-core distribution. Grid is one AIV by construction.
//
// Target: Ascend 950PR_9599 (dav-3510, __NPU_ARCH__ 3510), CANN 9.2.0-beta.2,
// CAModel RUN_MODE=sim. ICDF_QUANT_TILES selects the grid; 4 is the depth-2
// ring boundary, >= 2x that wraps every slot (V3).

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "../../routines/routine_vector_icdf_quantize.hpp"

namespace {

// UB layout, rule AKA3003. Every base 32-byte aligned, one DataBlock of bank
// skew between live slots, VECTOR_REG_WIDTH of read pad behind the arena.
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

constexpr std::uint32_t kTileRepeats = ascend_micro::kIcdfTileRepeats;   // 4
constexpr std::uint32_t kTileElems = kTileRepeats * 2u * ascend_micro::kIcdfLanes;  // 1024
constexpr std::uint32_t kTileBytes = kTileElems * 2u;                    // 2048 B in
constexpr std::uint32_t kTileOutBytes = kTileRepeats * ascend_micro::kIcdfPackedBytesPerRepeat;  // 512 B

// Depth-2 input ring, then depth-2 packed-output ring.
constexpr std::uint32_t kIn0 = 0;
constexpr std::uint32_t kIn1 = NextSlot(kIn0, kTileBytes);              // 2080
constexpr std::uint32_t kOut0 = NextSlot(kIn1, kTileBytes);             // 4160
constexpr std::uint32_t kOut1 = NextSlot(kOut0, kTileOutBytes);         // 4704
constexpr std::uint32_t kArenaEnd = kOut1 + kTileOutBytes;              // 5216

static_assert(kIn0 % kBlock == 0 && kIn1 % kBlock == 0 && kOut0 % kBlock == 0 &&
              kOut1 % kBlock == 0, "every UB operand base must be 32-byte aligned");
// Rule A2: a slot stride that is a multiple of 256 B puts every slot of a ring
// on the same bank phase.
static_assert((kIn1 - kIn0) % 256u != 0u && (kOut1 - kOut0) % 256u != 0u,
              "ring slot stride must not be a multiple of 256 B");
// Rule A3: source over-read pad. The DINTLV load reads exactly 512 B per
// repeat, but keep the harness-wide 256 B pad behind the arena regardless.
static_assert(kArenaEnd + kVecReg <= kUbWindow, "UB arena leaves no read pad");
// MTE2/MTE3 lengths must be whole 32-byte bursts.
static_assert(kTileBytes % kBlock == 0 && kTileOutBytes % kBlock == 0,
              "DMA lengths must be 32-byte aligned");

#ifndef ICDF_QUANT_TILES
#define ICDF_QUANT_TILES 4
#endif
constexpr std::uint32_t kTiles = ICDF_QUANT_TILES;
static_assert(kTiles >= 2, "V2 needs at least the ring depth (2)");

// Event ids, static code 0..5 (6/7 reserved). One id per route with a single
// token in flight: consumption is FIFO per route, which is sound for ordered
// ring reuse (the set produced after tile t proves reads/stores through t are
// done, which is a superset of what the next consumer of that slot needs).
constexpr std::int32_t kEvtInReady = 0;   // MTE2_V: input slot filled
constexpr std::int32_t kEvtInFree = 1;    // V_MTE2: input slot read
constexpr std::int32_t kEvtOutReady = 2;  // V_MTE3: packed bytes in UB
constexpr std::int32_t kEvtOutFree = 3;   // MTE3_V: output slot drained
constexpr std::int32_t kEvtDrain = 4;     // MTE3_S: final outflow retired

}  // namespace

template <class Input>
__aicore__ inline void run_icdf_quantize(__gm__ Input* gm_in, __gm__ std::uint8_t* gm_out)
{
    constexpr auto ub = AscendC::TPosition::VECCALC;
    AscendC::LocalTensor<Input> in0{ub, kIn0, kTileElems};
    AscendC::LocalTensor<Input> in1{ub, kIn1, kTileElems};
    AscendC::LocalTensor<std::uint8_t> out0{ub, kOut0, kTileOutBytes};
    AscendC::LocalTensor<std::uint8_t> out1{ub, kOut1, kTileOutBytes};
    AscendC::GlobalTensor<Input> in;
    AscendC::GlobalTensor<std::uint8_t> out;
    in.SetGlobalBuffer(gm_in);
    out.SetGlobalBuffer(gm_out);

    __ubuf__ Input* in_base = reinterpret_cast<__ubuf__ Input*>(in0.GetPhyAddr());
    __ubuf__ int4x2_t* out_base = reinterpret_cast<__ubuf__ int4x2_t*>(out0.GetPhyAddr());
    const std::uint32_t in_slot_stride = kIn1 - kIn0;        // bytes between ring slots
    const std::uint32_t out_slot_stride = kOut1 - kOut0;    // int4x2 bytes between slots

    // Prime the ring-free routes: both slots start empty/drained. One token
    // per route; the post-loop waits drain them so no bit leaks across
    // launches.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtInFree);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtOutFree);
    for (std::uint32_t t = 0; t < kTiles; ++t) {
        const std::uint32_t slot = t & 1u;
        // MTE2: refill this slot only once the vector pipe finished reading
        // its previous contents (WAR back edge; waits stall the MTE2 queue).
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtInFree);
        AscendC::DataCopy(slot ? in1 : in0, in[t * kTileElems], kTileElems);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtInReady);
        // Vector: wait the fill, and wait until MTE3 drained this output
        // slot's previous payload before the VF store overwrites it.
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtInReady);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtOutFree);
        __ubuf__ Input* in_slot = in_base + (slot ? in_slot_stride / 2u : 0u);
        __ubuf__ int4x2_t* out_slot = out_base + (slot ? out_slot_stride : 0u);
        ascend_micro::icdf_quantize_vf<Input>(in_slot, out_slot,
                                              static_cast<std::uint16_t>(kTileRepeats));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtInFree);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtOutReady);
        // MTE3: outflow once the packed bytes are in UB.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtOutReady);
        AscendC::DataCopy(out[t * kTileOutBytes], slot ? out1 : out0, kTileOutBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtOutFree);
    }
    // Drain: the primed tokens and the final per-route sets are consumed here.
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtInFree);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtOutFree);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
}

extern "C" __global__ __aicore__ void camodel_icdf_quantize_fp16(
    __gm__ half* in, __gm__ std::uint8_t* out)
{
    AscendC::InitSocState();
    run_icdf_quantize<half>(in, out);
}
extern "C" __global__ __aicore__ void camodel_icdf_quantize_bf16(
    __gm__ bfloat16_t* in, __gm__ std::uint8_t* out)
{
    AscendC::InitSocState();
    run_icdf_quantize<bfloat16_t>(in, out);
}
#elif defined(ASCEND_MICRO_CANN_LAUNCH)
extern "C" __global__ __aicore__ void camodel_icdf_quantize_fp16(
    __gm__ half*, __gm__ std::uint8_t*);
extern "C" __global__ __aicore__ void camodel_icdf_quantize_bf16(
    __gm__ bfloat16_t*, __gm__ std::uint8_t*);
extern "C" void launch_camodel_icdf_quantize_fp16(void* stream, std::uint16_t* in,
                                                  std::uint8_t* out)
{
    camodel_icdf_quantize_fp16<<<1, nullptr, stream>>>(
        reinterpret_cast<half*>(in), out);
}
extern "C" void launch_camodel_icdf_quantize_bf16(void* stream, std::uint16_t* in,
                                                  std::uint8_t* out)
{
    camodel_icdf_quantize_bf16<<<1, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(in), out);
}
#endif
