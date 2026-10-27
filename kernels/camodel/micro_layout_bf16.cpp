// Single-tile BF16 operand-layout probe, Ascend 950PR / arch35 / dav-3510,
// CANN 9.2. AIC only: one 16-token x 256-channel page, eight 32 x 32 resident
// blocks, sixteen accumulating Mmads and two Fixpipes straight to Global
// Memory. No Vector core, no Unified Buffer, no cross-core flag, no ring --
// the point is to isolate the data-movement question that
// turboquant_reshape_and_cache_bf16.cpp raises and nothing else.
//
// What it computes
// ----------------
// One (chunk, output-group) cell of the production GEMM, in both operand
// orientations, over the same operands:
//
//   value orientation   C_v[c0][t] = sum_G sum_i W_G[c0][i] . x[t][G*32 + i]
//                       A = block Nz(32, 32), B = page group Zn(32, 16)
//   key orientation     C_k[t][c0] = the same sum
//                       A = page group Nz(16, 32), B = block Zn(32, 32)
//
// so C_k is C_v transposed, and one host oracle checks both. With integer
// activations and +-1 blocks every product and every partial sum is an
// integer below 2^24, so FP32 accumulation is exact whatever order the Cube
// reduces in and the comparison is bit-exact rather than toleranced.
//
// What the `table_nd2nz` argument selects
// ---------------------------------------
// The resident-block ingress, and only that:
//
//   0  flat ND -> ND, the INT8 kernel's prologue copy, which assumes a
//      row-major 32 x 32 block is its own Nz image -- true at C0 = 32, false
//      at C0 = 16;
//   1  per-block ND -> Nz, `stage_blocks_nz`, which converts on the way in.
//
// Everything downstream -- the page ingress, all four MTE1 loads, all sixteen
// Mmads, both Fixpipes -- is identical between the two. So a difference in the
// output is attributable to the resident-block ingress alone, and the runner
// can additionally upload an already-Nz-imaged blob under mode 0 to confirm
// the attribution from the other side.

#include <cstdint>
#include <utility>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "../../routines/routine_stage_page_zn_bf16.hpp"
#include "../../routines/routine_fractal_load_zn_bf16.hpp"

namespace micro_layout {

constexpr std::uint32_t kTokens = 16;
constexpr std::uint32_t kHeadSize = 256;
constexpr std::uint32_t kGroup = 32;
constexpr std::uint32_t kGroups = kHeadSize / kGroup;  // 8
constexpr std::uint32_t kElemBytes = 2u;

constexpr std::uint32_t kFractalElems = kTokens * kGroup;            // 512
constexpr std::uint32_t kFractalBytes = kFractalElems * kElemBytes;  // 1024
constexpr std::uint32_t kBlockElems = kGroup * kGroup;               // 1024
constexpr std::uint32_t kBlockBytes = kBlockElems * kElemBytes;      // 2048

// ----------------------------------------------------------------- L1 arena
constexpr std::uint32_t kBankSkewBytes = 32u;
constexpr std::uint32_t kL1Page = 0u;
constexpr std::uint32_t kL1PageBytes = kTokens * kHeadSize * kElemBytes;  // 8192
constexpr std::uint32_t kL1Table = kL1Page + kL1PageBytes + kBankSkewBytes;
constexpr std::uint32_t kL1TableBytes = kGroups * kBlockBytes;  // 16384
constexpr std::uint32_t kL1End = kL1Table + kL1TableBytes;

static_assert(kL1Page % 32u == 0u && kL1Table % 32u == 0u,
              "every L1 operand base must be 32-byte aligned");
static_assert(kL1End <= 512u * 1024u, "L1 arena exceeds the L1 buffer");

// ----------------------------------------------------------------- L0 arena
// The eight blocks sit at element 0 of both L0A and L0B, the eight page
// fractals behind them, exactly as the production kernel lays them out.
constexpr std::uint32_t kL0TableElems = kGroups * kBlockElems;   // 8192
constexpr std::uint32_t kL0PageElems = kGroups * kFractalElems;  // 4096
constexpr std::uint32_t kL0Elems = kL0TableElems + kL0PageElems;
constexpr std::uint32_t kL0CValue = 0u;
constexpr std::uint32_t kL0CKey = kGroup * kTokens;  // 512 FP32
constexpr std::uint32_t kL0CElems = 2u * kGroup * kTokens;

static_assert(kBlockElems * kElemBytes % 512u == 0u,
              "every resident block must be 512-byte aligned in L0");
static_assert(kL0TableElems * kElemBytes % 512u == 0u,
              "the L0 page region must stay 512-byte aligned");
static_assert(kFractalElems * kElemBytes % 512u == 0u,
              "every page fractal must be 512-byte aligned in L0");
static_assert(kL0Elems * kElemBytes <= 64u * 1024u, "L0A/L0B hold 64 KiB on arch35");
static_assert(kL0CKey * sizeof(float) % 1024u == 0u,
              "the second L0C region must be 1024-byte aligned");

using GroupSeq = std::make_integer_sequence<std::uint32_t, kGroups>;

constexpr std::int32_t kEvt = 0;

// ---------------------------------------------------------------- MTE1 loads
template <std::uint32_t G>
__aicore__ inline void LoadGroup(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b)
{
    ascend_micro::load_block_a_nz<kGroup>(kL1Table + G * kBlockBytes,
                                          l0a + G * kBlockElems);
    ascend_micro::load_block_b_zn<kGroup>(kL1Table + G * kBlockBytes,
                                          l0b + G * kBlockElems);
    ascend_micro::load_page_group_a_nz<kTokens, kGroup>(
        kL1Page + G * kFractalBytes, l0a + kL0TableElems + G * kFractalElems);
    ascend_micro::load_page_group_b_zn<kTokens, kGroup>(
        kL1Page + G * kFractalBytes, l0b + kL0TableElems + G * kFractalElems);
}

template <std::uint32_t... Gs>
__aicore__ inline void LoadGroups(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                  std::integer_sequence<std::uint32_t, Gs...>)
{
    (LoadGroup<Gs>(l0a, l0b), ...);
}

// --------------------------------------------------------------------- Mmads
// Value orientation: the resident block is A and the page group is B, which
// is the operand pairing the channel-major value cache needs.
template <std::uint32_t G>
__aicore__ inline void MmadValue(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                 __cc__ float* l0c)
{
    using namespace AscendC::Te;
    auto a = MakeTensor(MakeMemPtr(l0a + G * kBlockElems),
                        MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kGroup, kGroup));
    auto b = MakeTensor(MakeMemPtr(l0b + kL0TableElems + G * kFractalElems),
                        MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTokens));
    auto c = MakeTensor(MakeMemPtr(l0c + kL0CValue),
                        MakeFrameLayout<NZLayoutPtn, 16>(kGroup, kTokens));
    MmadParams mp(static_cast<std::uint16_t>(kGroup), static_cast<std::uint16_t>(kTokens),
                  static_cast<std::uint16_t>(kGroup), 0, G == 0u);
    Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(mp), c, a, b);
}

// Key orientation: the page group is A and the resident block is B.
template <std::uint32_t G>
__aicore__ inline void MmadKey(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                               __cc__ float* l0c)
{
    using namespace AscendC::Te;
    auto a = MakeTensor(MakeMemPtr(l0a + kL0TableElems + G * kFractalElems),
                        MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTokens, kGroup));
    auto b = MakeTensor(MakeMemPtr(l0b + G * kBlockElems),
                        MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kGroup));
    auto c = MakeTensor(MakeMemPtr(l0c + kL0CKey),
                        MakeFrameLayout<NZLayoutPtn, 16>(kTokens, kGroup));
    MmadParams mp(static_cast<std::uint16_t>(kTokens), static_cast<std::uint16_t>(kGroup),
                  static_cast<std::uint16_t>(kGroup), 0, G == 0u);
    Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(mp), c, a, b);
}

template <std::uint32_t... Gs>
__aicore__ inline void MmadValues(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                  __cc__ float* l0c,
                                  std::integer_sequence<std::uint32_t, Gs...>)
{
    (MmadValue<Gs>(l0a, l0b, l0c), ...);
}

template <std::uint32_t... Gs>
__aicore__ inline void MmadKeys(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                __cc__ float* l0c,
                                std::integer_sequence<std::uint32_t, Gs...>)
{
    (MmadKey<Gs>(l0a, l0b, l0c), ...);
}

// ------------------------------------------------------------------ Fixpipes
// unitFlag = 0: eight accumulating Mmads feed one transfer, so the L0C
// dependency is carried by an explicit M_FIX event rather than by how the unit
// flag pairs a group of Mmads.
__aicore__ inline void DumpValue(__cc__ float* l0c, __gm__ float* out)
{
    using namespace AscendC::Te;
    auto c = MakeTensor(MakeMemPtr(l0c + kL0CValue),
                        MakeFrameLayout<NZLayoutPtn, 16>(kGroup, kTokens));
    auto y = MakeTensor(MakeMemPtr(out), MakeFrameLayout<NDExtLayoutPtn>(kGroup, kTokens));
    FixpipeParams fp;
    fp.unitFlag = 0;
    Copy(MakeCopy(CopyL0C2GM{}).with(fp), y, c);
}

__aicore__ inline void DumpKey(__cc__ float* l0c, __gm__ float* out)
{
    using namespace AscendC::Te;
    auto c = MakeTensor(MakeMemPtr(l0c + kL0CKey),
                        MakeFrameLayout<NZLayoutPtn, 16>(kTokens, kGroup));
    auto y = MakeTensor(MakeMemPtr(out), MakeFrameLayout<NDExtLayoutPtn>(kTokens, kGroup));
    FixpipeParams fp;
    fp.unitFlag = 0;
    Copy(MakeCopy(CopyL0C2GM{}).with(fp), y, c);
}

__aicore__ inline void run(__gm__ const bfloat16_t* page, __gm__ const bfloat16_t* table,
                           __gm__ float* out_value, __gm__ float* out_key,
                           std::uint32_t row_pitch, std::uint32_t table_nd2nz)
{
    __ca__ bfloat16_t l0_a[kL0Elems] __attribute__((aligned(512)));
    __cb__ bfloat16_t l0_b[kL0Elems] __attribute__((aligned(512)));
    __cc__ float l0_c[kL0CElems] __attribute__((aligned(1024)));

    // MTE2. The page ingress is production's; the block ingress is the probe's
    // single degree of freedom.
    ascend_micro::stage_page_nz<kTokens, kHeadSize>(page, kL1Page, row_pitch);
    if (table_nd2nz != 0u) {
        ascend_micro::stage_blocks_nz<kGroups, kGroup>(table, kL1Table);
    } else {
        ascend_micro::stage_blocks_flat<kGroups, kGroup>(table, kL1Table);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEvt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvt);

    LoadGroups(l0_a, l0_b, GroupSeq{});
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvt);

    AscendC::SetMMRowMajor();
    MmadValues(l0_a, l0_b, l0_c, GroupSeq{});
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEvt);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEvt);
    DumpValue(l0_c, out_value);

    // The two orientations own disjoint L0C regions, so the key Mmads are free
    // to issue while the value Fixpipe drains; only their own M_FIX edge
    // orders them against their own transfer.
    MmadKeys(l0_a, l0_b, l0_c, GroupSeq{});
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEvt + 1);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEvt + 1);
    DumpKey(l0_c, out_key);

    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace micro_layout

// Unqualified uint32_t, not std::uint32_t: CANN copies this parameter list
// verbatim into aclrtlaunch_triple_chevrons_func.h, which has no
// using-declaration for std, so a qualified scalar type breaks the host leg of
// the build while the device legs compile cleanly. Pointer parameters are
// rewritten to void* and are unaffected.
extern "C" __global__ __aicore__ void micro_layout_bf16(
    __gm__ bfloat16_t* page, __gm__ bfloat16_t* table, __gm__ float* out_value,
    __gm__ float* out_key, uint32_t row_pitch, uint32_t table_nd2nz)
{
    AscendC::InitSocState();
    if ASCEND_IS_AIC {
        micro_layout::run(page, table, out_value, out_key, row_pitch, table_nd2nz);
    }
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

// This leg has no CANN headers -- the device types live behind
// `#if defined(__CCE_AICORE__)` -- so `bfloat16_t` is not available here.
// std::uint16_t stands in for it: the two legs are separate translation units,
// every pointer becomes void* in the generated argument block, and the host
// side holds BF16 as bit patterns anyway.
extern "C" __global__ __aicore__ void micro_layout_bf16(
    __gm__ std::uint16_t* page, __gm__ std::uint16_t* table, __gm__ float* out_value,
    __gm__ float* out_key, uint32_t row_pitch, uint32_t table_nd2nz);

extern "C" void launch_micro_layout_bf16(void* stream, const std::uint16_t* page,
                                         const std::uint16_t* table, float* out_value,
                                         float* out_key, std::uint32_t row_pitch,
                                         std::uint32_t table_nd2nz)
{
    micro_layout_bf16<<<1, nullptr, stream>>>(const_cast<std::uint16_t*>(page),
                                              const_cast<std::uint16_t*>(table),
                                              out_value, out_key, row_pitch,
                                              table_nd2nz);
}

#endif
