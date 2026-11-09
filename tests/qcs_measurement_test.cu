#include "../src/qcs_internal.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unistd.h>
using C=std::complex<double>;
extern "C" int qcs_measurement_c_checks(qcs_simulator*);
static void need(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
static void cu(cudaError_t status) {need(status==cudaSuccess,"CUDA test operation failed");}
static std::vector<cuDoubleComplex> snapshot(qcs_simulator* s) {
    cu(cudaSetDevice(s->device));std::vector<cuDoubleComplex> v(s->elements);
    cu(cudaMemcpy(v.data(),s->state,v.size()*sizeof(v[0]),cudaMemcpyDeviceToHost));return v;
}
static void identical(const std::vector<cuDoubleComplex>& a,const std::vector<cuDoubleComplex>& b) {
    need(a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]))==0,"Initializer/storage changed unexpectedly");
}
static std::uint64_t logical(qcs_simulator* s,std::uint64_t physical) {
    std::uint64_t value=0;for(int p=0;p<s->num_qubits;++p) value|=((physical>>p)&1)<<s->p2l[p];return value;
}
static double norm(qcs_simulator* s) {
    double local=0,total=0;for(auto z:snapshot(s)) local+=z.x*z.x+z.y*z.y;
    MPI_Allreduce(&local,&total,1,MPI_DOUBLE,MPI_SUM,s->comm);return total;
}
static void basis(qcs_simulator* s,std::uint64_t index,double amplitude) {
    auto v=snapshot(s);int bad=0,any=0;
    for(std::size_t j=0;j<v.size();++j) {
        double expected=logical(s,j+s->rank*s->elements)==index?amplitude:0;
        if(std::abs(v[j].x-expected)>1e-12 || std::abs(v[j].y)>1e-12) bad=1;
    }
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,s->comm);need(!any,"Raw projected state mismatch");
}
static void h(qcs_simulator* s,bit_num_t q) {qcs_simulator_gate_h_cxx(s,&q,1,nullptr,0,nullptr,0);}
static void x(qcs_simulator* s,bit_num_t q) {qcs_simulator_gate_x_cxx(s,&q,1,nullptr,0,nullptr,0);}
static void cx(qcs_simulator* s,bit_num_t q,bit_num_t c) {qcs_simulator_gate_x_cxx(s,&q,1,nullptr,0,&c,1);}
static void zero(qcs_simulator* s) {qcs_simulator_set_zero_state_cxx(s);}
static void bell(qcs_simulator* s) {zero(s);h(s,0);cx(s,2,0);}
static unsigned errors=0;
static void report(const char*,size_t) {++errors;}
int main() {
    try {
        qcs_set_exception_callback(report);
        auto* s=qcs_simulator_create_cxx();
        need(qcs_simulator_measure_many(s,nullptr,0,nullptr),"Empty unallocated request failed");
        int n=4+s->global_bits;
        qcs_simulator_set_num_qubits_cxx(s,n);qcs_simulator_set_num_clbits_cxx(s,n+3);
        qcs_simulator_allocate_memory_cxx(s);
        for(bool reverse:{false,true}) {
            std::vector<bit_num_t> map(n);for(int p=0;p<n;++p) map[p]=reverse?n-1-p:p;
            qcs_simulator_set_mapping_cxx(s,map.data(),n);qcs_simulator_reinitialize_mapping_cxx(s);
            need(qcs_measurement_c_checks(s),"C measurement/reset callers failed");
            zero(s);h(s,0);
            auto before_mapping=s->p2l;
            need(qcs_simulator_measure_cxx(s,3)==0,"Partial deterministic outcome");
            need(s->p2l==before_mapping,"Measurement unexpectedly localized targets");
            h(s,0);need(qcs_simulator_measure_cxx(s,0)==0,"Partial measurement destroyed unmeasured coherence");
            bell(s);
            int bit=qcs_simulator_measure_cxx(s,0);auto cursor=s->measurement_cursor;
            need(qcs_simulator_measure_cxx(s,0)==bit && s->measurement_cursor==cursor,"Cached measurement drew again");
            need(qcs_simulator_measure_cxx(s,2)==bit,"Bell correlation");
            qcs_simulator_reset_measurement_state_cxx(s);
            basis(s,(std::uint64_t(bit)<<0)|(std::uint64_t(bit)<<2),std::sqrt(0.5));
            need(std::abs(norm(s)-0.5)<1e-12,"Projection was normalized");
            cursor=s->measurement_cursor;
            need(qcs_simulator_measure_cxx(s,0)==bit,"Metadata reset restored discarded amplitudes");
            need(s->measurement_cursor==cursor+1,"Metadata reset did not release cached outcomes");
            // A measured target can be rotated and sampled afresh, with its raw branch mass preserved.
            h(s,0);(void)qcs_simulator_measure_cxx(s,0);
            qcs_simulator_reset_measurement_state_cxx(s);
            need(std::abs(norm(s)-0.25)<1e-12,"Measured-target gate/collapse norm");
            bell(s);bit=qcs_simulator_measure_cxx(s,0);qcs_simulator_reset_cxx(s,0);
            need(qcs_simulator_measure_cxx(s,0)==0 && qcs_simulator_measure_cxx(s,2)==bit,"Bell per-qubit reset");
            qcs_simulator_reset_measurement_state_cxx(s);basis(s,std::uint64_t(bit)<<2,std::sqrt(0.5));
            zero(s);x(s,0);need(qcs_simulator_measure_to_clbit_cxx(s,0,1)==1,"Clbit single assignment");
            qcs_simulator_reset_cxx(s,0);
            need(qcs_simulator_measure_cxx(s,0)==0 && qcs_simulator_read_cxx(s,1)==1,"Reset changed clbit");
            auto projected=snapshot(s);qcs_simulator_reset_clbits_cxx(s);identical(projected,snapshot(s));
            // Full GHZ measured in an unsorted request spanning group boundaries.
            zero(s);h(s,0);for(bit_num_t q=1;q<n;++q) cx(s,q,0);
            std::vector<bit_num_t> qs(n),cs(n);std::vector<bit_t> values(n),again(n);
            for(int j=0;j<n;++j) {qs[j]=n-1-j;cs[j]=n-j;}
            cursor=s->measurement_cursor;
            qcs_simulator_measure_many_to_clbits_cxx(s,qs.data(),n,cs.data(),n,values.data());
            need(s->measurement_cursor-cursor==std::uint64_t((n+3)/4),"Fresh group cursor count");
            for(int j=0;j<n;++j) need(values[j]==values[0] && qcs_simulator_read_cxx(s,cs[j])==values[j],"GHZ caller/destination correlation");
            cursor=s->measurement_cursor;
            qcs_simulator_measure_many_cxx(s,qs.data(),n,again.data());
            need(values==again && s->measurement_cursor==cursor,"Cached groups changed results/RNG");
            // Check specified execution order within this backend, with fixed seed/cursor and a flat state.
            qcs_simulator_set_flat_state_cxx(s);cursor=s->measurement_cursor;
            std::vector<bit_num_t> sorted=qs;
            std::sort(sorted.begin(),sorted.end(),[&](int a,int b) {
                int pa=s->l2p[a],pb=s->l2p[b];int local=n-s->global_bits;
                if((pa<local)!=(pb<local)) return pa<local;
                return pa>pb;
            });
            qcs_simulator_measure_many_cxx(s,sorted.data(),n,again.data());
            qcs_simulator_set_flat_state_cxx(s);s->measurement_cursor=cursor;
            qcs_simulator_measure_many_to_clbits_cxx(s,qs.data(),n,cs.data(),n,values.data());
            for(int j=0;j<n;++j) {
                auto k=std::find(sorted.begin(),sorted.end(),qs[j])-sorted.begin();
                need(values[j]==again[k],"Physical execution order/caller result restoration");
            }
            // One invalid rank must not strand its peers or modify output buffers.
            zero(s);bit_t sentinel=9;bit_num_t q=0;
            need(!qcs_simulator_measure(s,q,s->rank==0?nullptr:&sentinel),"NULL result accepted");
            need(sentinel==9,"Failed single request wrote output");
            bit_num_t duplicate[]={0,0};bit_t outputs[]={9,9};
            need(!qcs_simulator_measure_many(s,duplicate,2,outputs),"Duplicate request accepted");
            bit_num_t distinct[]={0,1};bit_num_t badcs[]={1,1};
            need(!qcs_simulator_measure_many_to_clbits(s,distinct,2,badcs,2,outputs),"Duplicate destinations accepted");
            need(!qcs_simulator_measure_many_to_clbits(s,distinct,2,badcs,1,outputs),"Count mismatch accepted");
            need(!qcs_simulator_measure_many(s,nullptr,1,outputs),"NULL list accepted");
            need(!qcs_simulator_measure_many(s,distinct,-1,outputs),"Negative count accepted");
            bit_num_t invalid_clbits[]={-1,0};
            need(!qcs_simulator_measure_many_to_clbits(s,distinct,2,invalid_clbits,2,outputs),"Invalid clbit accepted");
            auto before_errors=errors;bool threw=false;
            try {qcs_simulator_measure_many_cxx(s,nullptr,1,outputs);}catch(const std::exception&){threw=true;}
            need(threw && errors==before_errors,"C++ measurement did not throw independently of C callback");
            need(outputs[0]==9 && outputs[1]==9,"Failed batch wrote results");
            need(qcs_simulator_measure_many(s,nullptr,0,nullptr),"Empty request failed");
        }
        // Restoration for every selected initializer, including RNG and file-backed state.
        char original[4096];need(getcwd(original,sizeof(original)),"getcwd");
        char directory[64]="qcs-measurement-XXXXXX";if(s->rank==0) need(mkdtemp(directory),"mkdtemp");
        MPI_Bcast(directory,sizeof(directory),MPI_CHAR,0,s->comm);need(chdir(directory)==0,"chdir shared fixture");
        if(s->rank==0) {
            FILE* file=std::fopen("statevector_input.bin","wb");need(file,"open fixture");
            for(int j=0;j<(1<<n);++j) {auto z=make_cuDoubleComplex(j+1,-0.3*j);need(std::fwrite(&z,sizeof(z),1,file)==1,"write fixture");}
            need(std::fclose(file)==0,"close fixture");
        }
        MPI_Barrier(s->comm);
        for(int selection=0;selection<=5;++selection) {
            auto* restored=qcs_simulator_create_cxx();qcs_simulator_set_num_qubits_cxx(restored,n);
            qcs_simulator_set_num_clbits_cxx(restored,1);
            qcs_simulator_allocate_memory_cxx(restored,static_cast<qcs_initial_state>(selection));zero(restored);
            auto initial=snapshot(restored);(void)qcs_simulator_measure_to_clbit_cxx(restored,0,0);h(restored,1);
            auto mapping=restored->p2l;auto cursor=restored->measurement_cursor;int clbit=qcs_simulator_read_cxx(restored,0);
            zero(restored);identical(initial,snapshot(restored));
            need(restored->p2l==mapping && !restored->measured_mask && !restored->pending_mask,"Initializer changed mapping or kept metadata");
            need(restored->measurement_cursor==cursor && qcs_simulator_read_cxx(restored,0)==clbit,"Initializer reset cursor/clbit");
            qcs_simulator_destroy_cxx(restored);
        }
        MPI_Barrier(s->comm);if(s->rank==0) need(std::remove("statevector_input.bin")==0,"remove fixture");
        need(chdir(original)==0,"restore directory");MPI_Barrier(s->comm);if(s->rank==0) need(rmdir(directory)==0,"remove directory");
        // Non-unit norm and zero-mass ranks, including one scalar per rank (L=0).
        auto* small=qcs_simulator_create_cxx();qcs_simulator_set_num_qubits_cxx(small,std::max(1,small->global_bits));
        qcs_simulator_allocate_memory_cxx(small,QCS_INITIAL_STATE_FLAT);zero(small);
        std::vector<bit_num_t> all(small->num_qubits);std::iota(all.begin(),all.end(),0);
        std::vector<bit_t> bits(all.size());qcs_simulator_measure_many_cxx(small,all.data(),all.size(),bits.data());
        qcs_simulator_reset_measurement_state_cxx(small);need(std::abs(norm(small)-1)<1e-12,"Flat projection rescaled scalar branch");
        // All-zero state is invalid; no draw, record or output may be committed.
        cu(cudaMemset(small->state,0,small->elements*sizeof(cuDoubleComplex)));
        small->measured_mask=small->measured_value=small->pending_mask=0;
        auto cursor=small->measurement_cursor;bit_t output=9;
        need(!qcs_simulator_measure(small,0,&output) && output==9 && cursor==small->measurement_cursor && !small->measured_mask,"Zero norm error consumed RNG/state");
        zero(small);small->measurement_cursor=UINT64_MAX;
        (void)qcs_simulator_measure_cxx(small,0);need(small->measurement_exhausted,"Last RNG counter wrapped");
        (void)qcs_simulator_measure_cxx(small,0);zero(small);
        need(!qcs_simulator_measure(small,0,&output),"Exhausted cursor reset by initialization");
        qcs_simulator_destroy_cxx(small);
        // Analytical 50/50 marginal bound (six standard deviations at 1024 shots).
        unsigned ones=0;for(int shot=0;shot<1024;++shot) {bell(s);ones+=qcs_simulator_measure_cxx(s,0);}
        need(ones>=416 && ones<=608,"Bell marginal outside predeclared six-sigma bound");
        if(s->rank==0) std::cout<<"PASS: QCS measurement/reset, C callers, initializer restores and Bell marginal "<<ones<<"/1024\n";
        qcs_simulator_destroy_cxx(s);
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<'\n';int started=0,finished=0;MPI_Initialized(&started);MPI_Finalized(&finished);
        if(started && !finished) MPI_Abort(MPI_COMM_WORLD,1);
        return 1;
    }
}
