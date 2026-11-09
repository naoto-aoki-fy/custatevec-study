#define QCS_GATE_API_TEST
#include "qcs_gate_catalog.hpp"
#include "../src/qcs_internal.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <numeric>

using C=qcs_gate::Complex;
static std::vector<C> cpu_apply(const std::vector<C>& input,GateKind kind,const std::vector<bit_num_t>& targets,
                               const std::vector<bit_num_t>& negative,const std::vector<bit_num_t>& positive,
                               std::array<double,4> angles) {
    auto result=input;
    std::uint64_t mask=0;
    for(int q:targets) mask|=UINT64_C(1)<<q;
    for(std::size_t index=0;index<input.size();++index) {
        bool active=true;
        for(int q:negative) if((index>>q)&1) active=false;
        for(int q:positive) if(!((index>>q)&1)) active=false;
        if(!active) continue;
        if(kind==GateKind::swap) {
            auto partner=index;
            for(std::size_t j=0;j<targets.size();j+=2)
                if(((index>>targets[j])^(index>>targets[j+1]))&1)
                    partner^=(UINT64_C(1)<<targets[j])|(UINT64_C(1)<<targets[j+1]);
            result[partner]=input[index];
        } else if((index&mask)==0) {
            int d=1<<targets.size();
            std::vector<C> out(d);
            auto basis=[&](int bits) {
                auto full=index;
                for(std::size_t j=0;j<targets.size();++j) full|=std::size_t((bits>>j)&1)<<targets[j];
                return full;
            };
            for(int col=0;col<d;++col) {
                auto column=reference_column(kind,col,angles);
                for(int row=0;row<d;++row) out[row]+=column[row]*input[basis(col)];
            }
            for(int row=0;row<d;++row) result[basis(row)]=out[row];
        }
    }
    return result;
}
static std::uint64_t logical_index(qcs_simulator* s,std::uint64_t physical) {
    std::uint64_t logical=0;
    for(int p=0;p<s->num_qubits;++p) logical|=((physical>>p)&1)<<s->p2l[p];
    return logical;
}
static std::vector<C> fixture(std::size_t dim) {
    std::vector<C> state(dim);
    for(std::size_t j=0;j<dim;++j) state[j]=C(0.01*(int(j%37)-18),0.013*(int((j*7)%29)-14));
    return state;
}
static void upload(qcs_simulator* s,const std::vector<C>& logical) {
    qcs_simulator_reinitialize_mapping_cxx(s);
    qcs_simulator_set_zero_state_cxx(s);
    std::vector<cuDoubleComplex> slice(s->elements);
    for(std::size_t j=0;j<slice.size();++j) {
        C z=logical[logical_index(s,j+s->rank*s->elements)];
        slice[j]=make_cuDoubleComplex(z.real(),z.imag());
    }
    assert(cudaSetDevice(s->device)==cudaSuccess);
    assert(cudaMemcpy(s->state,slice.data(),slice.size()*sizeof(slice[0]),cudaMemcpyHostToDevice)==cudaSuccess);
}
static void check(qcs_simulator* s,const std::vector<C>& expected) {
    std::vector<cuDoubleComplex> slice(s->elements);
    assert(cudaMemcpy(slice.data(),s->state,slice.size()*sizeof(slice[0]),cudaMemcpyDeviceToHost)==cudaSuccess);
    int bad=0, any=0;
    for(std::size_t j=0;j<slice.size();++j) {
        C want=expected[logical_index(s,j+s->rank*s->elements)];
        if(std::abs(C(slice[j].x,slice[j].y)-want)>1e-12+5e-12*std::abs(want)) bad=1;
    }
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,s->comm);
    if(any) { std::cerr<<"Phase-sensitive state comparison failed\n"; MPI_Abort(s->comm,1); }
    for(int p=0;p<s->num_qubits;++p) assert(s->l2p[s->p2l[p]]==p);
}
static unsigned errors=0;
static void error(const char*,size_t) { ++errors; }
int main() {
    qcs_set_exception_callback(error);
    auto* s=qcs_simulator_create_cxx();
    const int local=10, n=local+s->global_bits;
    qcs_simulator_set_num_qubits_cxx(s,n);
    qcs_simulator_allocate_memory_cxx(s);
    auto initial=fixture(std::size_t(1)<<n);
    std::array<double,4> angles{0.71,-0.39,1.17,-0.23};
    unsigned cases=0;
    for(bool reverse_mapping:{false,true}) {
        std::vector<bit_num_t> permutation(n);
        for(int p=0;p<n;++p) permutation[p]=reverse_mapping?n-1-p:p;
        qcs_simulator_set_mapping_cxx(s,permutation.data(),n);
        for(auto kind:all_gates) for(int placement=0;placement<3;++placement) for(int control=0;control<4;++control) {
            upload(s,initial);
            int width=qcs_gate::arity(kind);
            std::vector<int> positions(n); std::iota(positions.begin(),positions.end(),0);
            // High-first places global targets first and protects later local targets.
            if(placement==1) std::reverse(positions.begin(),positions.end());
            if(placement==2) std::rotate(positions.begin(),positions.begin()+n/2,positions.end());
            std::vector<bit_num_t> t,neg,pos;
            for(int j=0;j<width;++j) t.push_back(permutation[positions[j]]);
            if(control&1) neg.push_back(permutation[positions[n-1]]);
            if(control&2) pos.push_back(permutation[positions[n-2]]);
            auto expected=cpu_apply(initial,kind,t,neg,pos,angles);
            call_gate(s,kind,angles,t,neg,pos,cases%2==0);
            check(s,expected); ++cases;
        }
        for(int pairs=1;pairs<=4;++pairs) for(bool numbered:{false,true}) {
            upload(s,initial);
            std::vector<bit_num_t> t,neg{permutation[0]},pos{permutation[1]};
            for(int j=0;j<2*pairs;++j) t.push_back(permutation[n-1-j]);
            auto expected=cpu_apply(initial,GateKind::swap,t,neg,pos,angles);
            if(!numbered) call_gate(s,GateKind::swap,angles,t,neg,pos,true);
            else {
                using Swap=int (*)(qcs_simulator*,const bit_num_t*,bit_num_t,const bit_num_t*,bit_num_t,const bit_num_t*,bit_num_t);
                Swap functions[]={qcs_simulator_gate_swap1,qcs_simulator_gate_swap2,qcs_simulator_gate_swap3,qcs_simulator_gate_swap4};
                assert(functions[pairs-1](s,t.data(),t.size(),neg.data(),1,pos.data(),1));
                assert(!functions[pairs-1](s,t.data(),0,nullptr,0,nullptr,0));
            }
            check(s,expected); ++cases;
        }
        // Changing widths on the same handle exercises workspace/worker reuse.
        upload(s,initial); auto expected=initial;
        for(auto kind:{GateKind::h,GateKind::rcccx,GateKind::global_phase,GateKind::u4}) {
            std::vector<bit_num_t> t;
            for(int j=0;j<qcs_gate::arity(kind);++j) t.push_back(permutation[n-1-j]);
            call_gate(s,kind,angles,t,{}, {},true);
            expected=cpu_apply(expected,kind,t,{}, {},angles);
        }
        check(s,expected);
    }
    // Private metadata fixtures establish the gate-side integration contract for
    // the later public measurement task (which must populate the same masks).
    upload(s,initial);
    bit_num_t measured_target=0, spectator=1;
    s->measured_mask=s->pending_mask=(UINT64_C(1)<<measured_target)|(UINT64_C(1)<<spectator);
    s->measured_value=UINT64_C(1)<<measured_target;
    auto projected=initial;
    for(std::size_t j=0;j<projected.size();++j) if(!((j>>measured_target)&1)) projected[j]=0;
    auto after=cpu_apply(projected,GateKind::h,{measured_target},{spectator},{},angles);
    qcs_simulator_gate_h_cxx(s,&measured_target,1,nullptr,0,nullptr,0);
    check(s,after);
    assert(s->measured_mask==(UINT64_C(1)<<spectator) && s->pending_mask==(UINT64_C(1)<<spectator));
    auto before_mapping=s->p2l;
    qcs_simulator_gate_x_cxx(s,&measured_target,1,nullptr,0,&spectator,1);
    check(s,after); assert(s->p2l==before_mapping); // Contradictory cached control.
    // A second handle exercises one-local-bit limits and L=0 global phase.
    auto* small=qcs_simulator_create_cxx();
    qcs_simulator_set_num_qubits_cxx(small,std::max(1,small->global_bits));
    qcs_simulator_allocate_memory_cxx(small);
    auto small_input=fixture(std::size_t(1)<<small->num_qubits);
    upload(small,small_input);
    qcs_simulator_gate_global_phase_cxx(small,angles[0],nullptr,0,nullptr,0);
    auto small_expected=cpu_apply(small_input,GateKind::global_phase,{}, {}, {},angles);
    check(small,small_expected);
    bit_num_t scalar_control=0;
    qcs_simulator_gate_global_phase_cxx(small,angles[0],&scalar_control,1,nullptr,0);
    small_expected=cpu_apply(small_expected,GateKind::global_phase,{}, {0}, {},angles);
    check(small,small_expected);
    bit_num_t pair[]={0,1};
    assert(!qcs_simulator_gate_swap1(small,pair,2,nullptr,0,nullptr,0));
    check(small,small_expected);
    qcs_simulator_destroy_cxx(small);
    // Argument errors leave state intact; even a one-rank invalid operand agrees.
    auto expected=initial; upload(s,initial);
    bit_num_t bad=s->rank==0 ? -1 : 0;
    assert(!qcs_simulator_gate_x(s,&bad,1,nullptr,0,nullptr,0)); check(s,expected);
    bit_num_t overlap=0;
    assert(!qcs_simulator_gate_x(s,&overlap,1,&overlap,1,nullptr,0));
    bool threw=false;
    try { qcs_simulator_gate_rccx_cxx(s,nullptr,0,nullptr,0,nullptr,0); } catch(const std::exception&) {threw=true;}
    assert(threw && errors>0);
    if(s->rank==0) std::cout<<"PASS: "<<cases<<" QCS gate cases plus sequence/error checks\n";
    qcs_simulator_destroy_cxx(s);
}
