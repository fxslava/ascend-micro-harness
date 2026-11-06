#pragma once
#include "../../include/routines/routine_vector_lloydmax_quantize.hpp"
#include "../../include/routines/routine_vector_gaussian_cdf_quantize.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lloyd {
inline void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
inline double phi(double x) { return std::exp(-x*x/2) / std::sqrt(2*std::acos(-1.)); }
inline double cdf(double x) { return std::erfc(-x/std::sqrt(2.))/2; }
inline double invcdf(double p) {
    double a=-12, b=12;
    for (int i=0;i<80;++i) { double m=(a+b)/2; if(cdf(m)<p) a=m; else b=m; }
    return (a+b)/2;
}
template<unsigned Bits> using CdfBook=ascend_micro::GaussianCdfCodebook<Bits>;
// Matched Gaussian compander comparator, not the repository's FP16 sigmoid
// leaf: floor(2^b sigmoid(1.5976*x + .070565*x^3)), normal mid-quantile decode.
template<unsigned Bits> inline unsigned sigmoid_code(float x) {
    float xx=x*x, cubic=0.070565f*xx, slope=cubic+1.5976f;
    float z=slope*x, e=std::exp(-z), d=e+1.f;
    float v=float(1u<<Bits)/d;
    return std::min((1u<<Bits)-1, unsigned(std::max(0.f,std::floor(v))));
}
inline unsigned code(unsigned bits, float x, bool sigmoid) {
    if(bits==2) return sigmoid ? sigmoid_code<2>(x) : ascend_micro::lloydmax_code<2>(x);
    return sigmoid ? sigmoid_code<4>(x) : ascend_micro::lloydmax_code<4>(x);
}
inline float centroid(unsigned bits, unsigned q, bool sigmoid) {
    if(bits==2) return sigmoid ? CdfBook<2>::centroids[q] : ascend_micro::LloydMaxCodebook<2>::centroids[q];
    return sigmoid ? CdfBook<4>::centroids[q] : ascend_micro::LloydMaxCodebook<4>::centroids[q];
}
inline std::vector<float> fixture(unsigned n, unsigned rows, bool boundaries=false, unsigned bits=4) {
    std::vector<float> x(n*rows);
    std::uint32_t state=0x49a23;
    auto uniform=[&]() { state=1664525u*state+1013904223u; return (double(state)+.5)/4294967296.; };
    for(unsigned r=0;r<rows;++r) {
        for(unsigned j=0;j<n;j+=2) {
            double u=uniform(), a=2*std::acos(-1.)*uniform(), v=std::sqrt(-2*std::log(u));
            x[r*n+j]=float(v*std::cos(a)); x[r*n+j+1]=float(v*std::sin(a));
        }
        // Normalized Hadamard on independent standard normals preserves N(0,1).
        for(unsigned s=1;s<n;s*=2) for(unsigned j=0;j<n;j+=2*s) for(unsigned k=0;k<s;++k) {
            float a=x[r*n+j+k],b=x[r*n+j+k+s];
            x[r*n+j+k]=a+b; x[r*n+j+k+s]=a-b;
        }
        for(unsigned j=0;j<n;++j) x[r*n+j]/=std::sqrt(float(n));
    }
    if(boundaries) {
        unsigned at=0;
        auto add=[&](float t) {
            x.at(at++)=std::nextafter(t,-INFINITY); x.at(at++)=t;
            x.at(at++)=std::nextafter(t,INFINITY);
        };
        if(bits==2) for(float t: ascend_micro::LloydMaxCodebook<2>::thresholds) add(t);
        else for(float t: ascend_micro::LloydMaxCodebook<4>::thresholds) add(t);
        add(0.f); x.at(at++)=-0.f; x.at(at++)=-100.f; x.at(at++)=100.f;
    }
    return x;
}
inline std::vector<std::uint8_t> pack(const std::vector<unsigned>& q,unsigned bits) {
    std::vector<std::uint8_t> out(q.size()*bits/8,0);
    for(unsigned i=0;i<q.size();++i) out[i*bits/8]|=std::uint8_t(q[i]<<(i*bits%8));
    return out;
}
inline std::uint32_t ulp(float a,float b) {
    std::uint32_t x,y; std::memcpy(&x,&a,4);std::memcpy(&y,&b,4);
    auto order=[](std::uint32_t u) { return u&0x80000000u ? ~u : u|0x80000000u; };
    x=order(x);y=order(y);return x>y?x-y:y-x;
}
// Exact N(0,1) distortion integral for the supplied decision boundaries.
inline double distortion(const std::vector<double>& edges,const std::vector<double>& centres) {
    double mse=0;
    for(unsigned i=0;i<centres.size();++i) {
        double a=edges[i],b=edges[i+1],pa=phi(a),pb=phi(b),p=cdf(b)-cdf(a);
        double ap=std::isfinite(a)?a*pa:0.,bp=std::isfinite(b)?b*pb:0.,c=centres[i];
        mse+=p+ap-bp-2*c*(pa-pb)+c*c*p;
    }
    return mse;
}
} // namespace lloyd
