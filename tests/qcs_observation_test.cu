#include "../src/qcs_internal.hpp"
#include "../src/qcs_gate.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
static void need(bool b) {if(!b) throw std::runtime_error("Observation assertion failed");}
static void near(double a,double b) {need(std::abs(a-b)<1e-9);}
int main() {
    MPI_Init(nullptr,nullptr);
    auto* s=qcs_simulator_create_cxx();
    try {
        qcs_simulator_set_num_qubits_cxx(s,4);
        qcs_simulator_set_num_clbits_cxx(s,1);
        bit_num_t perm[]={3,2,1,0};qcs_simulator_set_mapping_cxx(s,perm,4);
        qcs_simulator_allocate_memory_cxx(s,QCS_INITIAL_STATE_LOAD_STATEVECTOR);
        // Loader consumes physical slices, regardless of the mapping.
        if(s->rank==0) {
            FILE* f=std::fopen("statevector_input.bin","wb");need(f);
            for(int i=0;i<16;++i) {cuDoubleComplex z=make_cuDoubleComplex(i+1,0.25-i);need(std::fwrite(&z,sizeof(z),1,f)==1);}
            need(std::fclose(f)==0);
        }
        MPI_Barrier(s->comm);qcs_simulator_set_zero_state_cxx(s);
        std::vector<cuDoubleComplex> v(s->elements);
        need(cudaMemcpy(v.data(),s->state,v.size()*sizeof(v[0]),cudaMemcpyDeviceToHost)==cudaSuccess);
        for(std::size_t i=0;i<v.size();++i) {near(v[i].x,i+s->rank*s->elements+1);near(v[i].y,0.25-i-s->rank*s->elements);}
        qcs_simulator_save_statevector_cxx(s,"qcs_observation_output.bin");
        if(s->rank==0) {
            FILE* f=std::fopen("qcs_observation_output.bin","rb");need(f);
            for(int logical=0;logical<16;++logical) {
                int physical=0;for(int p=0;p<4;++p) physical|=((logical>>perm[p])&1)<<p;
                cuDoubleComplex z;need(std::fread(&z,sizeof(z),1,f)==1);near(z.x,physical+1);near(z.y,0.25-physical);
            }
            std::fclose(f);
        }
        qcs_simulator_set_flat_state_cxx(s);
        near(qcs_simulator_expectation_pauli_cxx(s,nullptr,nullptr,0,2),32);
        bit_num_t q=0; qcs_pauli_operator op=QCS_PAULI_Y;
        qcs_simulator_gate_s_cxx(s,&q,1,nullptr,0,nullptr,0);
        auto plan=qcs_gate::localize(s->p2l,{q},s->num_qubits-s->global_bits);
        near(qcs_simulator_expectation_pauli_cxx(s,&q,&op,1,0.5),8);
        need(s->p2l==plan.p2l);
        int outcome=qcs_simulator_measure_to_clbit_cxx(s,q,0);
        auto mask=s->measured_mask,value=s->measured_value,cursor=s->measurement_cursor;
        auto exhausted=s->measurement_exhausted;auto bits=s->clbits;
        near(qcs_simulator_expectation_pauli_cxx(s,nullptr,nullptr,0,1),8);
        near(qcs_simulator_expectation_pauli_cxx(s,&q,&op,1,1),0);
        need(mask==s->measured_mask && value==s->measured_value && cursor==s->measurement_cursor && exhausted==s->measurement_exhausted && bits==s->clbits);
        qcs_simulator_save_statevector_cxx(s,"qcs_observation_output.bin");
        if(s->rank==0) {
            FILE* f=std::fopen("qcs_observation_output.bin","rb");need(f);
            double norm=0;
            for(int i=0;i<16;++i) {
                cuDoubleComplex z;need(std::fread(&z,sizeof(z),1,f)==1);
                double mass=z.x*z.x+z.y*z.y;norm+=mass;
                if((i&1)!=outcome) near(mass,0);
            }
            near(norm,8);std::fclose(f);
        }
        need(qcs_simulator_measure_cxx(s,q)==outcome && s->measurement_cursor==cursor);
        double result=123;
        need(!qcs_simulator_expectation_pauli(s,&q,&op,-1,1,&result) && result==123);
        qcs_pauli_operator bad=qcs_pauli_operator(99);
        need(!qcs_simulator_expectation_pauli(s,&q,&bad,1,1,&result) && result==123);
        need(!qcs_simulator_expectation_pauli(s,nullptr,nullptr,0,NAN,&result) && result==123);
        need(!qcs_simulator_expectation_pauli(s,nullptr,nullptr,0,1,nullptr));
        need(!qcs_simulator_save_statevector(s,"/nonexistent-qcs-directory/output.bin"));
        int start=qcs_simulator_event_create_cxx(s),stop=qcs_simulator_event_create_cxx(s);
        qcs_simulator_event_record_cxx(s,start);
        qcs_simulator_set_flat_state_cxx(s);
        qcs_simulator_event_record_cxx(s,stop);
        double seconds=qcs_simulator_event_get_elapsed_time_cxx(s,start,stop);need(std::isfinite(seconds) && seconds>=0);
        float ms=0,max_ms=0;need(cudaEventElapsedTime(&ms,s->events[start],s->events[stop])==cudaSuccess);
        MPI_Allreduce(&ms,&max_ms,1,MPI_FLOAT,MPI_MAX,s->comm);near(seconds,max_ms*1e-3);
        need(!qcs_simulator_event_record(s,-1));
        need(!qcs_simulator_event_get_elapsed_time(s,-1,stop,&result) && result==123);
        need(!qcs_simulator_event_create(s,nullptr));
        MPI_Barrier(s->comm);
        if(s->rank==0) {std::remove("statevector_input.bin");std::remove("qcs_observation_output.bin");}
        qcs_simulator_destroy_cxx(s);MPI_Finalize();return 0;
    } catch(...) {MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
