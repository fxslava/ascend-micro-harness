#include "camodel_common.hpp"
#include "../common/lloydmax_oracle.hpp"
#include "../cpu/icdf_fixture.hpp"
#include <cstdlib>
extern "C" void launch_camodel_lloydmax_quantize(void*,float*,float*,std::uint8_t*,std::uint32_t*,unsigned,unsigned,unsigned,unsigned);
int main(int argc,char** argv) {
    if(argc!=6) {std::fprintf(stderr,"usage: %s bits mode(0=reg,1=UB,2=sigmoid,3=original,4=decode,5=UBdecode) n rows boundaries\n",argv[0]);return 2;}
    unsigned bits=std::atoi(argv[1]),mode=std::atoi(argv[2]),n=std::atoi(argv[3]),rows=std::atoi(argv[4]);
    bool boundaries=std::atoi(argv[5])!=0;
    try {
        lloyd::require((bits==2||bits==4)&&mode<=5&&(mode!=3||bits==4)&&(n==128||n==256)&&rows>=1&&rows<=8,"arguments");
        auto x=lloyd::fixture(n,rows,boundaries,bits);
        camodel::Session session;
        camodel::GuardedDeviceBuffer input(n*rows*4),output(2048*4),packed(1024),guards(160);
        // Original leaf takes half. Its own oracle applies a 1/16 margin
        // because hardware half Exp/Div may move near-boundary codes.
        std::vector<std::uint8_t> original;
        if(mode==3) {
            auto fx=icdf::make_fixture(false,2); icdf::assert_properties(fx);
            std::memcpy(input.host_payload(),fx.in_bits.data(),n*rows*2);
            original.assign(fx.oracle_packed.begin(),fx.oracle_packed.begin()+n*rows/2);
        } else if(mode>=4) {
            std::vector<unsigned> q;
            for(unsigned i=0;i<x.size();++i) q.push_back(i% (1u<<bits)); // every code, every word position
            auto p=lloyd::pack(q,bits);std::memcpy(input.host_payload(),p.data(),p.size());
        } else std::memcpy(input.host_payload(),x.data(),x.size()*4);
        input.upload();
        launch_camodel_lloydmax_quantize(session.stream,static_cast<float*>(input.payload()),
            static_cast<float*>(output.payload()),static_cast<std::uint8_t*>(packed.payload()),
            static_cast<std::uint32_t*>(guards.payload()),bits,mode,n,rows);
        camodel::check(aclrtSynchronizeStream(session.stream),"Lloyd-Max sync");
        input.download();output.download();packed.download();guards.download();
        input.assert_guards();output.assert_guards();packed.assert_guards();guards.assert_guards();
        auto* g=static_cast<std::uint32_t*>(guards.host_payload());
        for(unsigned i=0;i<40;++i) lloyd::require(g[i]==camodel::kCanary,"UB boundary canary");
        auto* got=static_cast<float*>(output.host_payload());auto* p=static_cast<std::uint8_t*>(packed.host_payload());
        double err=0,power=0,maxabs=0;unsigned maxulp=0,active=n*rows;
        std::vector<unsigned> q;
        if(mode!=3) {
            for(unsigned i=0;i<active;++i) {
                unsigned c=mode>=4?i%(1u<<bits):lloyd::code(bits,x[i],mode==2);q.push_back(c);
                float golden=lloyd::centroid(bits,c,mode==2);
                maxabs=std::max(maxabs,std::abs(double(got[i])-golden));
                maxulp=std::max(maxulp,lloyd::ulp(got[i],golden));
                double d=double(x[i])-got[i];err+=d*d;power+=double(x[i])*x[i];
            }
            if(mode<4) {
                auto expected=lloyd::pack(q,bits);
                lloyd::require(std::memcmp(p,expected.data(),expected.size())==0,"packed code mismatch");
            }
            lloyd::require(maxulp==0,"FP32 decode differs from golden");
        } else {
            lloyd::require(std::memcmp(p,original.data(),original.size())==0,"original ICDF bytes"); active=0;
        }
        // Whole inactive tails, not only one word, must remain poison.
        auto* rw=static_cast<std::uint32_t*>(output.host_payload());
        for(unsigned i=active;i<2048;++i) lloyd::require(rw[i]==camodel::kCanary,"inactive reconstruction overwrite");
        unsigned used=mode>=4?0:(mode==3?n*rows/2:n*rows*bits/8);
        for(unsigned i=used;i<1024;++i) lloyd::require(p[i]==reinterpret_cast<const std::uint8_t*>(&camodel::kCanary)[i%4],"inactive packed overwrite");
        std::printf("PASS bits=%u mode=%u n=%u rows=%u boundaries=%u nmse=%.10f max_abs=%.9g max_ulp=%u UB_guards=160 GM_guards=512\n",bits,mode,n,rows,boundaries,power?err/power:0,maxabs,maxulp);
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
