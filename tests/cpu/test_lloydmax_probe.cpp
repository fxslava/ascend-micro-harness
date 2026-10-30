#include "../common/lloydmax_oracle.hpp"
#include <array>
#include <cstdio>
#include <set>

// Exhaust all queue interleavings of the one-shot wrapper. No ring is claimed:
// V initializes UB -> MTE2 fills -> V codec -> MTE3 drains -> Scalar returns.
// Every route is a capacity-one bit; data hazards are checked independently.
struct QueueState {
    std::array<unsigned,4> pc{};
    unsigned tokens=0,completed=0;
};
void queue_model() {
    struct Op { int kind, value; }; // 0 wait, 1 set, 2 work
    const std::array<std::vector<Op>,4> queues={{
        {{2,0},{1,0},{0,1},{2,2},{1,2}}, // V
        {{0,0},{2,1},{1,1}},             // MTE2
        {{0,2},{2,3},{1,3}},             // MTE3
        {{0,3}}}};                       // Scalar
    std::set<std::array<unsigned,6>> visited;
    unsigned terminals=0;
    auto visit=[&](auto&& self,QueueState s)->void {
        std::array<unsigned,6> key={s.pc[0],s.pc[1],s.pc[2],s.pc[3],s.tokens,s.completed};
        if(!visited.insert(key).second)return;
        bool progress=false,done=true;
        for(unsigned q=0;q<4;++q) {
            if(s.pc[q]==queues[q].size())continue;
            done=false; Op op=queues[q][s.pc[q]];
            if(op.kind==0 && !(s.tokens&(1u<<op.value)))continue;
            QueueState next=s;++next.pc[q];progress=true;
            if(op.kind==0) next.tokens&=~(1u<<op.value);
            if(op.kind==1) {
                lloyd::require(!(s.tokens&(1u<<op.value)),"capacity-one event overwritten");
                next.tokens|=1u<<op.value;
            }
            if(op.kind==2) {
                if(op.value>0)lloyd::require(s.completed&(1u<<(op.value-1)),"cross-pipe data hazard");
                next.completed|=1u<<op.value;
            }
            self(self,next);
        }
        lloyd::require(done||progress,"queue deadlock");
        if(done) {lloyd::require(s.tokens==0&&s.completed==15,"residual event/work");++terminals;}
    };
    visit(visit,QueueState{});
    lloyd::require(terminals==1,"missing drained state");
    std::printf("PASS capacity-one one-shot queue model: %zu states, zero residual tokens\n",visited.size());
}

template<unsigned B> void run() {
    using Book=ascend_micro::LloydMaxCodebook<B>;
    std::vector<double> c(std::begin(Book::centroids),std::end(Book::centroids));
    std::vector<double> t={-INFINITY};
    t.insert(t.end(),std::begin(Book::thresholds),std::end(Book::thresholds)); t.push_back(INFINITY);
    double stationary=0;
    for(unsigned i=0;i<c.size();++i) {
        double expected=(lloyd::phi(t[i])-lloyd::phi(t[i+1]))/(lloyd::cdf(t[i+1])-lloyd::cdf(t[i]));
        stationary=std::max(stationary,std::abs(c[i]-expected));
        if(i+1<c.size()) lloyd::require(std::abs(t[i+1]-(c[i]+c[i+1])/2)<2e-7,"midpoint mismatch");
    }
    lloyd::require(stationary<(B==2?2e-5:5e-7),"conditional mean fixed point mismatch");
    for(unsigned i=0;i<(1u<<B)-1;++i) {
        float v=Book::thresholds[i];
        lloyd::require(ascend_micro::lloydmax_code<B>(std::nextafter(v,-INFINITY))==i,"lower boundary");
        lloyd::require(ascend_micro::lloydmax_code<B>(v)==i+1,"threshold equality");
        lloyd::require(ascend_micro::lloydmax_code<B>(std::nextafter(v,INFINITY))==i+1,"upper boundary");
    }
    std::vector<double> ct,et={-INFINITY},st={-INFINITY};
    for(unsigned i=0;i<(1u<<B);++i) {
        ct.push_back(lloyd::invcdf((i+.5)/(1u<<B)));
        lloyd::require(std::abs(ct.back()-lloyd::centroid(B,i,true))<1e-7,"CDF decode constant");
        if(i+1<(1u<<B)) {
            double p=double(i+1)/(1u<<B); et.push_back(lloyd::invcdf(p));
            double a=-10,b=10,z=std::log(p/(1-p));
            for(int k=0;k<80;++k) { double m=(a+b)/2; if(1.5976*m+.070565*m*m*m<z)a=m;else b=m; }
            st.push_back((a+b)/2);
        }
    }
    et.push_back(INFINITY);st.push_back(INFINITY);
    double lm=lloyd::distortion(t,c),cdf=lloyd::distortion(et,ct),sig=lloyd::distortion(st,ct);
    lloyd::require(lm<cdf && lm<sig,"Lloyd-Max population NMSE regression");
    // Exhaustive packed-word positions, all codes, both requested vector sizes.
    for(unsigned n:{128u,256u}) for(unsigned rows:{1u,4u,8u}) {
        auto x=lloyd::fixture(n,rows,true,B);std::vector<unsigned> q;
        for(float v:x) {
            unsigned nearest=0;
            for(unsigned j=1;j<c.size();++j) if(std::abs(double(v)-c[j])<=std::abs(double(v)-c[nearest]))nearest=j;
            unsigned code=ascend_micro::lloydmax_code<B>(v);
            // Rounded threshold can differ from the exact double midpoint by
            // <1 FP32 ULP; assert distance optimality at that precision.
            lloyd::require(std::abs(double(v)-c[code])<=std::abs(double(v)-c[nearest])+3e-7,"nearest centroid");
            q.push_back(code);
        }
        for(unsigned i=0;i<q.size();++i) q[i]=i%(1u<<B);
        auto packed=lloyd::pack(q,B);
        for(unsigned i=0;i<q.size();++i) lloyd::require(((packed[i*B/8]>>(i*B%8))&((1u<<B)-1))==q[i],"packing");
        // Replay the register pack tree independently of the byte oracle.
        // Vary every code and word position; wrap lanes must never leak into
        // the selected group-leading lanes.
        for(unsigned pattern=0;pattern<64;++pattern) {
            std::array<std::uint32_t,64> reg{};
            for(unsigned i=0;i<64;++i)reg[i]=(i*i+17*i+pattern*((i%5)+1))% (1u<<B);
            auto original=reg;
            for(unsigned s=1;s<32/B;s*=2) {
                auto before=reg;
                for(unsigned i=0;i<64;++i)reg[i]=before[i]|(before[(i+s)%64]<<(s*B));
            }
            for(unsigned start=0;start<64;start+=32/B) {
                std::uint32_t word=0;
                for(unsigned k=0;k<32/B;++k)word|=original[start+k]<<(k*B);
                lloyd::require(reg[start]==word,"register reduction-tree pack");
            }
        }
    }
    std::printf("PASS bits=%u Gaussian population NMSE LM=%.10f CDF=%.10f sigmoid=%.10f fixed_point_residual=%.3g\n",B,lm,cdf,sig,stationary);
}
int main() {
    try { queue_model();run<2>();run<4>();return 0; }
    catch(const std::exception& e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
