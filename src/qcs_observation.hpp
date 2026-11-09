#pragma once
#include "qcs.h"
#include <cstdint>
#include <vector>
namespace qcs_observation {
// Raw Hermitian Pauli contraction; all nonidentity operands are local.
template<class Complex> double contract(const std::vector<Complex>& host,
        const std::vector<int>& l2p,const bit_num_t* qubits,
        const qcs_pauli_operator* operators,int count) {
    double sum=0;
    for(std::uint64_t i=0;i<host.size();++i) {
        std::uint64_t other=i; int power=0;
        for(int j=0;j<count;++j) {
            int op=operators[j];
            if(op==QCS_PAULI_I) continue;
            auto bit=UINT64_C(1)<<l2p[qubits[j]];
            if(op==QCS_PAULI_X || op==QCS_PAULI_Y) other^=bit;
            if(op==QCS_PAULI_Y) power+=1+2*bool(i&bit);
            if(op==QCS_PAULI_Z) power+=2*bool(i&bit);
        }
        // conj(psi[P i]) * phase(i) * psi[i].
        auto x=host[i],y=host[other];
        double re=y.x*x.x+y.y*x.y,im=y.x*x.y-y.y*x.x;
        switch(power%4) {case 0:sum+=re;break;case 1:sum-=im;break;case 2:sum-=re;break;case 3:sum+=im;break;}
    }
    return sum;
}
}
