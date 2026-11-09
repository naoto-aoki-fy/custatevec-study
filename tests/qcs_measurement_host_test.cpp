#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/qcs_measurement.hpp"
#include <cassert>
#include <complex>
#include <iostream>

int main() {
    using namespace qcs_measurement;
    const int n=5;
    std::vector<int> p2l{0,1,2,3,4};
    const int qubits[]={4,1,3,0};
    unsigned cases=0;
    do {
        std::vector<int> l2p(n);for(int p=0;p<n;++p) l2p[p2l[p]]=p;
        for(int local=0;local<=n;++local) for(int width=1;width<=4;++width)
            for(std::uint64_t measured:{UINT64_C(0),UINT64_C(1),UINT64_C(10)}) for(auto value:{UINT64_C(0),measured}) {
            std::array<double,16> actual{},expected{};
            for(int rank=0;rank<(1<<(n-local));++rank) {
                auto descriptor=group(qubits,width,l2p,local,rank,measured,value);
                std::array<double,16> compact{};
                for(int j=0;j<(1<<local);++j) {
                    int physical=j|(rank<<local), logical=0;
                    for(int p=0;p<n;++p) logical|=((physical>>p)&1)<<p2l[p];
                    double mass=std::norm(std::complex<double>(0.13*(physical+1),0.07*(physical%3-1)));
                    if((std::uint64_t(logical)&measured)==value) {
                        int outcome=0;for(int k=0;k<width;++k) outcome|=((logical>>qubits[k])&1)<<k;
                        expected[outcome]+=mass;
                    }
                    bool included=true;
                    for(int k=0;k<descriptor.mask_count;++k)
                        if(((j>>descriptor.mask_bits[k])&1)!=descriptor.mask_values[k]) included=false;
                    if(included) {
                        int bin=0;for(int k=0;k<descriptor.local_count;++k) bin|=((j>>descriptor.targets[k])&1)<<k;
                        compact[bin]+=mass;
                    }
                }
                auto expanded=expand(descriptor,compact);
                for(int b=0;b<(1<<width);++b) actual[b]+=expanded[b];
            }
            for(int b=0;b<(1<<width);++b) assert(std::abs(actual[b]-expected[b])<1e-11);
            auto sorted=order({4,1,3,0},l2p,local,true);
            for(std::size_t j=1;j<sorted.size();++j) {
                int a=l2p[qubits[sorted[j-1]]],b=l2p[qubits[sorted[j]]];
                assert((a<local && b>=local) || ((a<local)==(b<local) && a>b));
            }
            assert((order({4,1,3,0},l2p,local,false)==std::vector<int>{0,1,2,3}));
            ++cases;
        }
    } while(std::next_permutation(p2l.begin(),p2l.end()));
    double bins[]={0,1,0,3};
    assert(total(bins,2)==4);
    assert(select(bins,2,0)==1 && select(bins,2,std::nextafter(0.25,0.0))==1);
    assert(select(bins,2,0.25)==3 && select(bins,2,std::nextafter(1.0,0.0))==3);
    auto rejects=[](auto action) {bool failed=false;try{action();}catch(const std::runtime_error&){failed=true;}assert(failed);};
    for(double bad:{-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        double invalid[]={1,bad};rejects([&]{total(invalid,1);});
    }
    double zero[]={0,0},overflow[]={std::numeric_limits<double>::max(),std::numeric_limits<double>::max()};
    rejects([&]{total(zero,1);});rejects([&]{total(overflow,1);});
    rejects([&]{select(bins,2,1);});
    std::int16_t duplicate[]={1,1},outside[]={5};
    assert(indices(nullptr,0,5).empty());
    rejects([&]{indices(nullptr,1,5);}); rejects([&]{indices(nullptr,-1,5);});
    rejects([&]{indices(duplicate,2,5);}); rejects([&]{indices(outside,1,5);});
    std::cout<<"PASS: "<<cases<<" conditional-bin/order fixtures, zero/local/global slices and sampling boundaries\n";
}
