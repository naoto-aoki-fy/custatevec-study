#ifdef NDEBUG
#undef NDEBUG
#endif
#include "qcs_gate_catalog.hpp"
#include <cassert>
#include <iostream>
#include <numeric>
template<unsigned Pairs> void check_swaps() {
    int d=1<<(2*Pairs);
    std::vector<qcs_gate::Complex> input(d),expected(d),actual(d);
    for(int j=0;j<d;++j) input[j]={0.17*j-0.3,0.11*(j%5)-0.2};
    expected=actual=input;
    std::vector<const qcs_gate::Complex*> in(d);
    std::vector<qcs_gate::Complex*> out(d);
    for(int j=0;j<d;++j) {in[j]=&input[j];out[j]=&expected[j];}
    reference_gate::gate::swap<Pairs>().apply(in.data(),out.data());
    auto m=qcs_gate::matrix(GateKind::swap);
    for(unsigned pair=0;pair<Pairs;++pair) {
        auto before=actual;
        int shift=2*pair, mask=3<<shift;
        for(int base=0;base<d;++base) if(!(base&mask))
            for(int row=0;row<4;++row) {
                actual[base|(row<<shift)]=0;
                for(int col=0;col<4;++col) actual[base|(row<<shift)]+=m[row*4+col]*before[base|(col<<shift)];
            }
    }
    for(int j=0;j<d;++j) assert(std::abs(actual[j]-expected[j])<1e-14);
}
int main() {
    check_swaps<1>(); check_swaps<2>(); check_swaps<3>(); check_swaps<4>();
    // Differential checks against the pinned reference's apply formulas, not a
    // second call to the production matrix builder. Include phase-sensitive rows.
    for (auto kind:all_gates) for (double angle:{0.0,0.71,-1.93,3.141592653589793}) {
        std::array<double,4> a{angle,-0.39,1.17,-0.23};
        auto m=qcs_gate::matrix(kind,a);
        int d=1<<qcs_gate::arity(kind);
        for (int col=0;col<d;++col) {
            auto expected=reference_column(kind,col,a);
            for (int row=0;row<d;++row) assert(std::abs(m[row*d+col]-expected[row])<2e-15);
        }
        // Independent asymmetric input catches interference and phase changes.
        std::vector<qcs_gate::Complex> direct(d),computed(d);
        for (int col=0;col<d;++col) {
            auto expected=reference_column(kind,col,a);
            qcs_gate::Complex amplitude((col+1)*0.13,0.07*(col%3-1));
            for (int row=0;row<d;++row) {
                direct[row]+=expected[row]*amplitude;
                computed[row]+=m[row*d+col]*amplitude;
            }
        }
        for (int row=0;row<d;++row) assert(std::abs(direct[row]-computed[row])<2e-14);
    }
    // Exhaust permutations and target lists; protect every original local target.
    std::vector<int> perm{0,1,2,3,4};
    unsigned plans=0;
    do {
        for (int a=0;a<5;++a) for (int b=0;b<5;++b) if(a!=b) {
            auto plan=qcs_gate::localize(perm,{a,b},3);
            assert(plan.l2p[a]<3 && plan.l2p[b]<3);
            for(int p=0;p<5;++p) assert(plan.l2p[plan.p2l[p]]==p);
            auto physical=perm;
            for(auto pair:plan.swaps) {
                assert(pair.first>=0 && pair.first<3 && pair.second>=3);
                assert(physical[pair.first]!=a && physical[pair.first]!=b);
                std::swap(physical[pair.first],physical[pair.second]);
            }
            assert(physical==plan.p2l); ++plans;
        }
    } while(std::next_permutation(perm.begin(),perm.end()));
    auto fixed=qcs_gate::localize({0,1,2,3,4},{4,2,3},3);
    assert((fixed.swaps==std::vector<std::pair<int,int>>{{1,4},{0,3}}));
    bool rejected=false;
    try { qcs_gate::localize({0,1,2},{0,1},1); } catch(const std::runtime_error&) {rejected=true;}
    assert(rejected);
    std::cout<<"PASS: "<<all_gates.size()<<" gate formulas, four SWAP widths, four angle fixtures, "<<plans<<" localization plans\n";
}
