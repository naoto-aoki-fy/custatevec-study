// Included after qcs_gates.cuh in qcs.cu's private namespace.
__global__ void measurement_uniform(double* output,std::uint64_t seed,std::uint64_t counter) {
    curandStatePhilox4_32_10_t rng;
    curand_init(static_cast<unsigned long long>(seed),static_cast<unsigned long long>(counter),0ULL,&rng);
    uint4 raw=curand4(&rng);
    std::uint64_t mantissa=(std::uint64_t(raw.x)<<21)|(std::uint64_t(raw.y)>>11);
    *output=static_cast<double>(mantissa)*0x1.0p-53;
}
void measurement_metadata(qcs_simulator& s) {
    same(s.comm,s.num_qubits);
    for(int p:s.p2l) same(s.comm,p);
    for(auto value:{s.measured_mask,s.measured_value,s.pending_mask,s.measurement_cursor}) {
        auto root=value; check_mpi(MPI_Bcast(&root,1,MPI_UINT64_T,0,s.comm));
        phase(s.comm,[&] {require(root==value,"Measurement metadata differs across ranks");});
    }
    same(s.comm,s.measurement_exhausted?1:0);
}
void reserve_measurement_draw(qcs_simulator& s) {
    double* candidate=nullptr;
    try {
        phase(s.comm,[&] {
            DeviceScope device(s.device);
            if(!s.measurement_draw) check_cuda(cudaMalloc(reinterpret_cast<void**>(&candidate),sizeof(double)));
        });
    } catch(...) {
        try {DeviceScope device(s.device);if(candidate) fatal_cuda(cudaFree(candidate));}
        catch(...) {check_mpi(MPI_ERR_OTHER);}
        throw;
    }
    if(candidate) s.measurement_draw=candidate;
}
unsigned measure_group(qcs_simulator& s,const int* qubits,int width) {
    auto group=qcs_measurement::group(qubits,width,s.l2p,s.num_qubits-s.global_bits,s.rank,s.measured_mask,s.measured_value);
    if(!group.fresh) return group.fixed;
    phase(s.comm,[&] {require(!s.measurement_exhausted,"Measurement RNG counter exhausted");});
    std::array<double,16> compact{},local{},weights{};
    try {
        phase(s.comm,[&] {
            DeviceScope device(s.device);
            try {
                if(group.active) check_sv(custatevecAbs2SumArray(s.handle,s.state,CUDA_C_64F,s.num_qubits-s.global_bits,
                    compact.data(),group.local_count?group.targets.data():nullptr,group.local_count,
                    group.mask_count?group.mask_values.data():nullptr,group.mask_count?group.mask_bits.data():nullptr,group.mask_count));
                check_cuda(cudaStreamSynchronize(s.stream));
            } catch(...) {fatal_cuda(cudaStreamSynchronize(s.stream));throw;}
            local=qcs_measurement::expand(group,compact);
        });
    } catch(...) {s.poisoned=true;throw;}
    check_mpi(MPI_Allreduce(local.data(),weights.data(),1<<width,MPI_DOUBLE,MPI_SUM,s.comm));
    // Invalid mass never draws or advances a cursor. All ranks participate in
    // the error decision, including zero-contribution and excluded ranks.
    phase(s.comm,[&] {(void)qcs_measurement::total(weights.data(),width);});
    double draw=0;
    try {
        phase(s.comm,[&] {
            if(s.rank!=0) return;
            DeviceScope device(s.device);
            try {
                measurement_uniform<<<1,1,0,s.stream>>>(s.measurement_draw,s.seed,s.measurement_cursor);
                check_cuda(cudaGetLastError());
                check_cuda(cudaMemcpyAsync(&draw,s.measurement_draw,sizeof(draw),cudaMemcpyDeviceToHost,s.stream));
                check_cuda(cudaStreamSynchronize(s.stream));
            } catch(...) {fatal_cuda(cudaStreamSynchronize(s.stream));throw;}
        });
    } catch(...) {s.poisoned=true;throw;}
    unsigned outcome=0;
    phase(s.comm,[&] {if(s.rank==0) outcome=qcs_measurement::select(weights.data(),width,draw);});
    check_mpi(MPI_Bcast(&outcome,1,MPI_UNSIGNED,0,s.comm));
    if(s.measurement_cursor==std::numeric_limits<std::uint64_t>::max()) s.measurement_exhausted=true;
    else ++s.measurement_cursor;
    for(int j=0;j<width;++j) {
        std::uint64_t bit=UINT64_C(1)<<qubits[j];
        if(!(group.fresh&bit)) continue;
        s.measured_mask|=bit;
        s.measured_value=(s.measured_value&~bit)|(std::uint64_t((outcome>>j)&1)<<qubits[j]);
        s.pending_mask|=bit;
    }
    return outcome;
}
void measure_request(qcs_simulator* sim,const bit_num_t* list,bit_num_t count,
                     const bit_num_t* destinations,bit_num_t destination_count,bit_t* results,bool to_clbits) {
    auto& s=live(sim);
    same(s.comm,count);same(s.comm,destination_count);same(s.comm,to_clbits?1:0);
    std::vector<int> qubits,clbits,order,sorted;
    std::vector<bit_t> temporary;
    phase(s.comm,[&] {
        qubits=qcs_measurement::indices(list,count,s.num_qubits);
        require(count==0 || results,"Null measurement results");
        if(to_clbits) {
            require(count==destination_count,"Qubit/clbit counts must match");
            clbits=qcs_measurement::indices(destinations,destination_count,s.num_clbits);
        }
        if(count) require(s.state && s.initialized && !s.poisoned,"Measurement requires allocated, initialized, healthy state");
        order=qcs_measurement::order(qubits,s.l2p,s.num_qubits-s.global_bits,to_clbits);
        for(int i:order) sorted.push_back(qubits[i]);
        temporary.resize(count);
    });
    if(count==0) return;
    for(int q:qubits) same(s.comm,q);
    if(to_clbits) {same(s.comm,s.num_clbits);for(int c:clbits) same(s.comm,c);}
    measurement_metadata(s);
    std::uint64_t requested=0;for(int q:qubits) requested|=UINT64_C(1)<<q;
    if(requested&~s.measured_mask) reserve_measurement_draw(s);
    for(int offset=0;offset<count;offset+=4) {
        int width=std::min(4,int(count)-offset);
        unsigned outcome=measure_group(s,sorted.data()+offset,width);
        for(int j=0;j<width;++j) temporary[order[offset+j]]=bit_t((outcome>>j)&1);
    }
    // Like the reference, expose arrays and clbits only after all groups succeed.
    if(to_clbits) for(int j=0;j<count;++j) s.clbits[clbits[j]]=temporary[j];
    std::copy(temporary.begin(),temporary.end(),results);
}
void clear_measurement(qcs_simulator* sim) {
    auto& s=live(sim);measurement_metadata(s);
    phase(s.comm,[&] {require(!s.poisoned && (!s.pending_mask || (s.state && s.initialized)),"Cannot clear measurement state on unhealthy storage");});
    try {
        phase(s.comm,[&] {
            DeviceScope device(s.device);
            project_targets(s,s.pending_mask);
            check_cuda(cudaStreamSynchronize(s.stream));
        });
        discard_measurements(s);
    } catch(...) {s.poisoned=true;throw;}
}
