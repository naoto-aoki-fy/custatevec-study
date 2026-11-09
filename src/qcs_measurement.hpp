#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace qcs_measurement {
inline std::vector<int> indices(const std::int16_t* list,int count,int bound) {
    if(count<0 || count>bound || (count && !list)) throw std::runtime_error("Invalid measurement list/count");
    std::vector<int> result;
    std::vector<bool> used(bound,false);
    for(int i=0;i<count;++i) {
        int q=list[i];
        if(q<0 || q>=bound || used[q]) throw std::runtime_error("Measurement indices must be distinct and in range");
        used[q]=true; result.push_back(q);
    }
    return result;
}
inline std::vector<int> order(const std::vector<int>& qubits,const std::vector<int>& l2p,int local,bool sorted) {
    std::vector<int> result(qubits.size()); std::iota(result.begin(),result.end(),0);
    if(sorted) std::sort(result.begin(),result.end(),[&](int a,int b) {
        int pa=l2p[qubits[a]],pb=l2p[qubits[b]];
        if((pa<local)!=(pb<local)) return pa<local;
        return pa>pb;
    });
    return result;
}
struct Group {
    std::array<std::int32_t,4> targets{};
    std::array<int,4> output_bits{};
    std::array<std::int32_t,63> mask_bits{}, mask_values{};
    int width=0,local_count=0,mask_count=0;
    unsigned fixed=0;
    std::uint64_t requested=0,fresh=0;
    bool active=true;
};
inline Group group(const int* qubits,int width,const std::vector<int>& l2p,int local,int rank,
                   std::uint64_t measured,std::uint64_t value) {
    if(width<1 || width>4) throw std::runtime_error("Invalid measurement group width");
    Group result; result.width=width;
    for(std::size_t q=0;q<l2p.size();++q) if((measured>>q)&1) {
        int p=l2p[q],bit=int((value>>q)&1);
        if(p<local) {result.mask_bits[result.mask_count]=p;result.mask_values[result.mask_count++]=bit;}
        else if(((rank>>(p-local))&1)!=bit) result.active=false;
    }
    for(int j=0;j<width;++j) {
        int q=qubits[j],p=l2p[q]; result.requested|=UINT64_C(1)<<q;
        if((measured>>q)&1) result.fixed|=unsigned((value>>q)&1)<<j;
        else if(p>=local) result.fixed|=unsigned((rank>>(p-local))&1)<<j;
        else { result.targets[result.local_count]=p;result.output_bits[result.local_count++]=j; }
    }
    result.fresh=result.requested&~measured;
    return result;
}
inline std::array<double,16> expand(const Group& group,const std::array<double,16>& compact) {
    std::array<double,16> result{};
    if(!group.active) return result;
    for(int bits=0;bits<(1<<group.local_count);++bits) {
        unsigned outcome=group.fixed;
        for(int j=0;j<group.local_count;++j) outcome|=unsigned((bits>>j)&1)<<group.output_bits[j];
        result[outcome]=compact[bits];
    }
    return result;
}
inline double total(const double* weights,int width) {
    double sum=0;
    for(int j=0;j<(1<<width);++j) {
        if(!std::isfinite(weights[j]) || weights[j]<0) throw std::runtime_error("Invalid measurement probability component");
        sum+=weights[j];
    }
    if(!std::isfinite(sum) || sum<=0) throw std::runtime_error("Measurement requires finite positive branch mass");
    return sum;
}
inline unsigned select(const double* weights,int width,double draw) {
    double mass=total(weights,width);
    if(!std::isfinite(draw) || draw<0 || draw>=1) throw std::runtime_error("Invalid measurement draw");
    double threshold=std::min(draw*mass,std::nextafter(mass,0.0));
    double prefix=0; unsigned last=0;
    for(int j=0;j<(1<<width);++j) {
        prefix+=weights[j];
        if(weights[j]>0) {last=unsigned(j);if(prefix>threshold) return unsigned(j);}
    }
    return last;
}
} // namespace qcs_measurement
