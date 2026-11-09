#include "qcs.h"
int qcs_measurement_c_checks(qcs_simulator* sim) {
    bit_t result=9, values[2]={9,9};
    const bit_num_t qs[]={1,0},cs[]={3,2},duplicate[]={0,0};
    if(!qcs_simulator_set_zero_state(sim) || !qcs_simulator_reset_clbits(sim)) return 0;
    if(!qcs_simulator_measure(sim,0,&result) || result!=0) return 0;
    if(!qcs_simulator_measure_to_clbit(sim,1,1,&result) || result!=0) return 0;
    if(!qcs_simulator_measure_many(sim,qs,2,values) || values[0] || values[1]) return 0;
    if(!qcs_simulator_measure_many_to_clbits(sim,qs,2,cs,2,values) || values[0] || values[1]) return 0;
    if(!qcs_simulator_measure_many(sim,0,0,0) || !qcs_simulator_measure_many_to_clbits(sim,0,0,0,0,0)) return 0;
    if(qcs_simulator_measure_many(sim,duplicate,2,values)!=0 || qcs_simulator_measure(sim,0,0)!=0) return 0;
    if(!qcs_simulator_reset(sim,0) || !qcs_simulator_reset_measurement_state(sim)) return 0;
    return 1;
}
