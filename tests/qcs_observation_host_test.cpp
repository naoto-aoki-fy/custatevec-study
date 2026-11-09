#include "../src/qcs_observation.hpp"
#include <cassert>
#include <cmath>
#include <complex>
#include <numeric>
struct Value {double x,y;};
int main() {
    using C=std::complex<double>;
    for(int reverse=0;reverse<2;++reverse) {
        std::vector<int> map=reverse?std::vector<int>{2,1,0}:std::vector<int>{0,1,2};
        std::vector<Value> state(8);
        for(int i=0;i<8;++i) state[i]={double(i+1),double(3-i)};
        bit_num_t qs[]={0,1,2};qcs_pauli_operator ops[3];
        for(int code=0;code<64;++code) {
            int n=code;for(auto& op:ops) {op=qcs_pauli_operator(n%4);n/=4;}
            // Independent tensor-product matrix oracle.
            C expected=0;
            for(int row=0;row<8;++row) for(int col=0;col<8;++col) {
                C factor=1;
                for(int q=0;q<3;++q) {
                    int r=(row>>map[q])&1,c=(col>>map[q])&1;
                    switch(ops[q]) {
                        case QCS_PAULI_I:factor*=double(r==c);break;
                        case QCS_PAULI_X:factor*=double(r!=c);break;
                        case QCS_PAULI_Y:factor*=r==c?C(0):C(0,r?1:-1);break;
                        case QCS_PAULI_Z:factor*=r==c?double(r?-1:1):0;break;
                    }
                }
                expected+=std::conj(C(state[row].x,state[row].y))*factor*C(state[col].x,state[col].y);
            }
            assert(std::abs(qcs_observation::contract(state,map,qs,ops,3)-expected.real())<1e-10);
        }
        double norm=0;for(auto z:state) norm+=z.x*z.x+z.y*z.y;
        assert(qcs_observation::contract(state,map,nullptr,nullptr,0)==norm);
    }
}
