#pragma once
#include "routine_vector_lloydmax_quantize.hpp"

// Explicit matched FP32 Gaussian comparator. This is an extension, not the
// FP16 signed-int4 contract of routines/routine_vector_icdf_quantize.hpp.
namespace ascend_micro {
template<unsigned Bits> struct GaussianCdfCodebook;
template<> struct GaussianCdfCodebook<2> {
    inline static constexpr float centroids[4]={-1.1503493804f,-0.318639364f,0.318639364f,1.1503493804f};
    inline static constexpr float thresholds[3]={}; // unused by table builder
};
template<> struct GaussianCdfCodebook<4> {
    inline static constexpr float centroids[16]={
        -1.8627318674f,-1.3180108973f,-1.0099901692f,-0.7764217611f,
        -0.5791321623f,-0.4022500653f,-0.2372021093f,-0.0784124127f,
         0.0784124127f,0.2372021093f,0.4022500653f,0.5791321623f,
         0.7764217611f,1.0099901692f,1.3180108973f,1.8627318674f};
    inline static constexpr float thresholds[15]={};
};
#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
namespace lloydmax_detail {
static constexpr R::CastTrait kCdfFloor = {
    R::RegLayout::ZERO, R::SatMode::SAT, R::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_FLOOR};
template<unsigned Bits>
__simd_callee__ inline void sigmoid(F& x, U& q, M& all) {
    F xx,z,num; I qi;
    R::Mul(xx,x,x,all);
    R::Muls(z,xx,0.070565f,all);
    R::Adds(z,z,1.5976f,all);
    R::Mul(z,z,x,all);
    R::Neg(z,z,all); R::Exp(z,z,all); R::Adds(z,z,1.f,all);
    R::Duplicate(num,float(1u<<Bits),all); R::Div(z,num,z,all);
    R::Mins(z,z,float((1u<<Bits)-1),all);
    R::Cast<std::int32_t,float,kCdfFloor>(qi,z,all);
    q=reinterpret_cast<U&>(qi);
}
}
template<unsigned Bits>
__simd_vf__ inline void gaussian_cdf_quantize_vf(__ubuf__ float* in,
    __ubuf__ float* recon,__ubuf__ std::uint32_t* packed,std::uint16_t pairs) {
    using namespace lloydmax_detail;
    M all=R::CreateMask<float,R::MaskPattern::ALL>(); I lane; F c;
    R::Arange(lane,0);
    table<Bits,false,0,GaussianCdfCodebook<Bits>>(c,lane,all);
    for(std::uint16_t i=0;i<pairs;++i) {
        F x0,x1,y0,y1; U q0,q1;
        R::LoadAlign(x0,in+i*128); R::LoadAlign(x1,in+i*128+64);
        sigmoid<Bits>(x0,q0,all); sigmoid<Bits>(x1,q1,all);
        R::Gather(y0,c,q0); R::Gather(y1,c,q1);
        R::StoreAlign(recon+i*128,y0,all); R::StoreAlign(recon+i*128+64,y1,all);
        pack_pair<Bits>(packed+i*(4*Bits),q0,q1,lane,all);
    }
}
#endif
}
