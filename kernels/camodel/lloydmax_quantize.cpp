#include <cstdint>
#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "../../include/routines/routine_vector_lloydmax_quantize.hpp"
#include "../../include/routines/routine_vector_gaussian_cdf_quantize.hpp"
#include "../../routines/routine_vector_icdf_quantize.hpp"
namespace {
constexpr unsigned kUbWindow=192*1024, kCapacity=2048;
constexpr unsigned kIn=32, kInGuard=kIn+kCapacity*4;
constexpr unsigned kRecon=kInGuard+32, kReconGuard=kRecon+kCapacity*4;
constexpr unsigned kPacked=kReconGuard+32, kPackedGuard=kPacked+kCapacity/2;
constexpr unsigned kCentroidTableBase=kPackedGuard+32;
constexpr unsigned kHeaderGuard=kCentroidTableBase+512, kArenaEnd=kHeaderGuard+32;
static_assert(kCentroidTableBase%32u==0);
static_assert(kCentroidTableBase+256u<=kUbWindow);
static_assert(kIn%32==0 && kRecon%32==0 && kPacked%32==0);
static_assert((kRecon-kIn)%256!=0 && (kPacked-kRecon)%256!=0);
static_assert(kArenaEnd+256<=kUbWindow);
__simd_vf__ inline void initialize(__ubuf__ std::uint32_t* arena) {
    using namespace ascend_micro::lloydmax_detail;
    M all=R::CreateMask<std::uint32_t,R::MaskPattern::ALL>(); U value;
    R::Duplicate(value,0xdeadbeefu,all);
    // Entire arena initialized, including all padding and unused output lanes.
    for(std::uint16_t i=0;i<(kArenaEnd+255)/256;++i) R::StoreAlign(arena+i*64,value,all);
}
template<unsigned Bits,unsigned Mode>
__aicore__ inline void run(__gm__ float* src,__gm__ float* dst,
    __gm__ std::uint8_t* codes,__gm__ std::uint32_t* guards,unsigned n,unsigned rows) {
    constexpr auto ub=AscendC::TPosition::VECCALC;
    AscendC::LocalTensor<float> in{ub,kIn,kCapacity},recon{ub,kRecon,kCapacity};
    AscendC::LocalTensor<std::uint8_t> packed{ub,kPacked,kCapacity/2};
    AscendC::LocalTensor<float> header{ub,kCentroidTableBase,128};
    auto ip=reinterpret_cast<__ubuf__ float*>(in.GetPhyAddr());
    auto rp=reinterpret_cast<__ubuf__ float*>(recon.GetPhyAddr());
    auto pp=reinterpret_cast<__ubuf__ std::uint32_t*>(packed.GetPhyAddr());
    auto hp=reinterpret_cast<__ubuf__ float*>(header.GetPhyAddr());
    AscendC::LocalTensor<std::uint32_t> arena{ub,0,(kArenaEnd+255)/256*64};
    initialize(reinterpret_cast<__ubuf__ std::uint32_t*>(arena.GetPhyAddr()));
    // Initialization writes input and header memory: MTE2 must wait for V.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::GlobalTensor<float> gin,gout; AscendC::GlobalTensor<std::uint8_t> gp;
    AscendC::GlobalTensor<std::uint32_t> gg;
    gin.SetGlobalBuffer(src);gout.SetGlobalBuffer(dst);gp.SetGlobalBuffer(codes);gg.SetGlobalBuffer(guards);
    AscendC::DataCopy(in,gin[0],Mode==3?n*rows/2:(Mode>=4?n*rows*Bits/32:n*rows));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
    if constexpr(Mode==1 || Mode==5) {
        ascend_micro::lloydmax_init_header_vf<Bits>(hp);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if constexpr(Mode==0 || Mode==1)
        ascend_micro::lloydmax_quantize_vf<Bits,Mode==0?ascend_micro::LloydMaxTable::Immediate:ascend_micro::LloydMaxTable::UbHeader>(ip,rp,pp,hp,n*rows/128);
    if constexpr(Mode==2) ascend_micro::gaussian_cdf_quantize_vf<Bits>(ip,rp,pp,n*rows/128);
    if constexpr(Mode==3) ascend_micro::icdf_quantize_vf<half>(
        reinterpret_cast<__ubuf__ half*>(ip),reinterpret_cast<__ubuf__ int4x2_t*>(pp),n*rows/128);
    if constexpr(Mode>=4) ascend_micro::lloydmax_dequantize_vf<Bits,
        Mode==4?ascend_micro::LloydMaxTable::Immediate:ascend_micro::LloydMaxTable::UbHeader>(
            reinterpret_cast<__ubuf__ std::uint32_t*>(ip),rp,hp,n*rows/128);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::DataCopy(gout[0],recon,kCapacity);
    AscendC::DataCopy(gp[0],packed,kCapacity/2);
    // Only guard blocks leave UB, never the centroid/threshold header.
    constexpr unsigned offsets[5]={0,kInGuard,kReconGuard,kPackedGuard,kHeaderGuard};
    for(unsigned j=0;j<5;++j) {
        AscendC::LocalTensor<std::uint32_t> g{ub,offsets[j],8};
        AscendC::DataCopy(gg[j*8],g,8);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(3);
}
}
extern "C" __global__ __aicore__ void camodel_lloydmax_quantize(
    __gm__ float* src,__gm__ float* dst,__gm__ std::uint8_t* codes,
    __gm__ std::uint32_t* guards,unsigned bits,unsigned mode,unsigned n,unsigned rows) {
    AscendC::InitSocState();
    if(bits==2) {
        if(mode==0) run<2,0>(src,dst,codes,guards,n,rows);
        if(mode==1) run<2,1>(src,dst,codes,guards,n,rows);
        if(mode==2) run<2,2>(src,dst,codes,guards,n,rows);
        if(mode==4) run<2,4>(src,dst,codes,guards,n,rows);
        if(mode==5) run<2,5>(src,dst,codes,guards,n,rows);
    } else {
        if(mode==0) run<4,0>(src,dst,codes,guards,n,rows);
        if(mode==1) run<4,1>(src,dst,codes,guards,n,rows);
        if(mode==2) run<4,2>(src,dst,codes,guards,n,rows);
        if(mode==3) run<4,3>(src,dst,codes,guards,n,rows);
        if(mode==4) run<4,4>(src,dst,codes,guards,n,rows);
        if(mode==5) run<4,5>(src,dst,codes,guards,n,rows);
    }
}
#elif defined(ASCEND_MICRO_CANN_LAUNCH)
extern "C" __global__ __aicore__ void camodel_lloydmax_quantize(
    __gm__ float*,__gm__ float*,__gm__ std::uint8_t*,__gm__ std::uint32_t*,unsigned,unsigned,unsigned,unsigned);
extern "C" void launch_camodel_lloydmax_quantize(void* stream,float* src,float* dst,
    std::uint8_t* codes,std::uint32_t* guards,unsigned bits,unsigned mode,unsigned n,unsigned rows) {
    camodel_lloydmax_quantize<<<1,nullptr,stream>>>(src,dst,codes,guards,bits,mode,n,rows);
}
#endif
