// Included in qcs.cu's private namespace to share checked collective phases.
void fatal_sv(custatevecStatus_t status) {
    if (status == CUSTATEVEC_STATUS_SUCCESS) return;
    std::fprintf(stderr,"QCS fatal cuStateVec swap error: %s\n",custatevecGetErrorString(status));
    std::fflush(stderr);
    check_mpi(MPI_ERR_OTHER);
}
void fatal_cuda(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr,"QCS fatal CUDA swap/resource error: %s\n",cudaGetErrorString(status));
    std::fflush(stderr);
    check_mpi(MPI_ERR_OTHER);
}
void close_swap(qcs_simulator& s) {
    try {
        DeviceScope device(s.device);
        if (s.worker) { fatal_sv(custatevecSVSwapWorkerDestroy(s.handle,s.worker)); s.worker=nullptr; }
        if (s.scheduler) { fatal_sv(custatevecDistIndexBitSwapSchedulerDestroy(s.handle,s.scheduler)); s.scheduler=nullptr; }
        if (s.communicator) { fatal_sv(custatevecCommunicatorDestroy(s.handle,s.communicator)); s.communicator=nullptr; }
        if (s.transfer) { fatal_cuda(cudaFree(s.transfer)); s.transfer=nullptr; }
        if (s.swap_extra) { fatal_cuda(cudaFree(s.swap_extra)); s.swap_extra=nullptr; }
        if (s.swap_event) { fatal_cuda(cudaEventDestroy(s.swap_event)); s.swap_event=nullptr; }
        if (s.transfer_stream) { fatal_cuda(cudaStreamDestroy(s.transfer_stream)); s.transfer_stream=nullptr; }
        s.transfer_bytes=s.swap_extra_bytes=0;
    } catch (...) { check_mpi(MPI_ERR_OTHER); }
}
void prepare_swap(qcs_simulator& s) {
    if (s.worker) return;
    std::size_t minimum=0;
    std::string plugin;
    custatevecCommunicatorType_t type{};
    phase(s.comm,[&] {
        if (const char* path=std::getenv("QCS_CUSTATEVEC_COMMUNICATOR_LIBRARY")) {
            require(*path && std::strlen(path)<512,"Invalid communicator plugin path");
            plugin=path; type=CUSTATEVEC_COMMUNICATOR_TYPE_EXTERNAL;
        } else {
#if defined(OMPI_MAJOR_VERSION)
            type=CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI;
#elif defined(MPICH_NAME) || defined(MPICH_VERSION)
            type=CUSTATEVEC_COMMUNICATOR_TYPE_MPICH;
#else
            throw std::runtime_error("Unknown MPI ABI: set QCS_CUSTATEVEC_COMMUNICATOR_LIBRARY to a matching cuStateVec MPI plugin");
#endif
        }
    });
    char root_plugin[512]={};
    if (s.rank==0) std::snprintf(root_plugin,sizeof(root_plugin),"%s",plugin.c_str());
    check_mpi(MPI_Bcast(root_plugin,sizeof(root_plugin),MPI_CHAR,0,s.comm));
    phase(s.comm,[&] { require(plugin==root_plugin,"Communicator plugin differs across ranks"); });
    try {
        phase(s.comm,[&] {
            DeviceScope device(s.device);
            check_cuda(cudaStreamCreateWithFlags(&s.transfer_stream,cudaStreamNonBlocking));
            check_cuda(cudaEventCreateWithFlags(&s.swap_event,cudaEventDisableTiming | cudaEventBlockingSync));
        });
        phase(s.comm,[&] {
            // Do not skip a potentially collective communicator creation because
            // another local resource failed. A transport/setup error is fatal.
            try {
                DeviceScope device(s.device);
                fatal_sv(custatevecCommunicatorCreate(s.handle,&s.communicator,type,plugin.empty()?nullptr:plugin.c_str()));
            } catch (...) { check_mpi(MPI_ERR_OTHER); }
        });
        phase(s.comm,[&] {
            DeviceScope device(s.device);
            check_sv(custatevecSVSwapWorkerCreate(s.handle,&s.worker,s.communicator,s.state,s.rank,s.swap_event,
                CUDA_C_64F,s.transfer_stream,&s.swap_extra_bytes,&minimum));
            if (s.swap_extra_bytes) check_cuda(cudaMalloc(&s.swap_extra,s.swap_extra_bytes));
            check_sv(custatevecSVSwapWorkerSetExtraWorkspace(s.handle,s.worker,s.swap_extra,s.swap_extra_bytes));
            check_sv(custatevecDistIndexBitSwapSchedulerCreate(s.handle,&s.scheduler,s.global_bits,s.num_qubits-s.global_bits));
        });
        std::uint64_t budget=0, common_budget=0, common_minimum=0, local_minimum=minimum;
        phase(s.comm,[&] {
            DeviceScope device(s.device);
            std::size_t free=0,total=0; check_cuda(cudaMemGetInfo(&free,&total));
            budget=bench::transfer_policy::budget(free,0.95);
        });
        check_mpi(MPI_Allreduce(&budget,&common_budget,1,MPI_UINT64_T,MPI_MIN,s.comm));
        check_mpi(MPI_Allreduce(&local_minimum,&common_minimum,1,MPI_UINT64_T,MPI_MAX,s.comm));
        phase(s.comm,[&] {
            DeviceScope device(s.device);
            s.transfer_bytes=bench::transfer_policy::select(common_budget,common_minimum,false);
            check_cuda(cudaMalloc(&s.transfer,s.transfer_bytes));
        });
        phase(s.comm,[&] {
            DeviceScope device(s.device);
            check_sv(custatevecSVSwapWorkerSetTransferWorkspace(s.handle,s.worker,s.transfer,s.transfer_bytes));
        });
    } catch (...) { close_swap(s); throw; }
}
void exchange_bit(qcs_simulator& s,int local,int global) {
    // From the benchmark's two-stream protocol. Once communication starts, an
    // execution error is fatal rather than unwinding past a peer's worker call.
    try {
        DeviceScope device(s.device);
        fatal_cuda(cudaStreamSynchronize(s.stream));
        const int2 pair=make_int2(local,global);
        std::uint32_t batches=0;
        fatal_sv(custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(s.handle,s.scheduler,&pair,1,nullptr,nullptr,0,&batches));
        for (std::uint32_t batch=0;batch<batches;++batch) {
            custatevecSVSwapParameters_t parameters{};
            fatal_sv(custatevecDistIndexBitSwapSchedulerGetParameters(s.handle,s.scheduler,batch,s.rank,&parameters));
            fatal_sv(custatevecSVSwapWorkerSetParameters(s.handle,s.worker,&parameters,parameters.dstSubSVIndex));
            fatal_sv(custatevecSVSwapWorkerExecute(s.handle,s.worker,0,parameters.transferSize));
        }
        fatal_cuda(cudaStreamSynchronize(s.transfer_stream));
    } catch (...) { check_mpi(MPI_ERR_OTHER); }
}
__global__ void project_branch(cuDoubleComplex* state,std::uint64_t count,std::uint64_t mask,std::uint64_t value) {
    for (std::uint64_t j=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<count;j+=std::uint64_t(blockDim.x)*gridDim.x)
        if ((j&mask)!=value) state[j]=make_cuDoubleComplex(0,0);
}
void project_targets(qcs_simulator& s,std::uint64_t release) {
    auto pending=s.pending_mask&release;
    if (!pending) return;
    std::uint64_t mask=0,value=0;
    bool active=true;
    int local=s.num_qubits-s.global_bits;
    for (int q=0;q<s.num_qubits;++q) if ((pending>>q)&1) {
        int p=s.l2p[q]; int bit=int((s.measured_value>>q)&1);
        if (p<local) { mask|=UINT64_C(1)<<p; value|=std::uint64_t(bit)<<p; }
        else if (((s.rank>>(p-local))&1)!=bit) active=false;
    }
    if (!active) check_cuda(cudaMemsetAsync(s.state,0,s.elements*sizeof(cuDoubleComplex),s.stream));
    else if (mask) {
        unsigned blocks=unsigned(std::min<std::uint64_t>((s.elements+255)/256,65535));
        project_branch<<<blocks,256,0,s.stream>>>(s.state,s.elements,mask,value);
        check_cuda(cudaGetLastError());
    }
    s.pending_mask&=~pending;
}
void operate_gate(qcs_simulator* sim,qcs_gate::Kind kind,std::array<double,4> angles,
                  const bit_num_t* target,bit_num_t nt,const bit_num_t* negative,bit_num_t nn,
                  const bit_num_t* positive,bit_num_t np,int numbered_width=0) {
    qcs_gate::Timer timer;
    auto& s=live(sim);
    const int local=s.num_qubits-s.global_bits;
    same(s.comm,int(kind)); same(s.comm,nt); same(s.comm,nn); same(s.comm,np); same(s.comm,numbered_width);
    auto root_angles=angles;
    check_mpi(MPI_Bcast(root_angles.data(),sizeof(root_angles),MPI_BYTE,0,s.comm));
    std::vector<int> targets, negatives, positives;
    std::vector<cuDoubleComplex> matrix;
    qcs_gate::Localization plan;
    std::uint64_t target_mask=0;
    bool permitted=true;
    phase(s.comm,[&] {
        require(s.state && s.initialized && !s.poisoned,"Gate requires allocated, initialized, healthy state");
        require(std::memcmp(angles.data(),root_angles.data(),sizeof(angles))==0,"Gate parameters differ across ranks");
        require(nt>=0 && nn>=0 && np>=0 && int(nt)+nn+np<=s.num_qubits,"Invalid gate list counts");
        require((nt==0 || target) && (nn==0 || negative) && (np==0 || positive),"Null gate list");
        if (kind==qcs_gate::Kind::swap) {
            require(nt>=2 && nt<=8 && nt%2==0,"SWAP requires 2,4,6 or 8 targets");
            require(!numbered_width || nt==numbered_width,"Numbered SWAP target count mismatch");
        } else require(nt==qcs_gate::arity(kind),"Invalid gate target count");
        require(nt<=local,"Number of targets exceeds local qubits");
        std::vector<bool> used(s.num_qubits,false);
        auto copy=[&](const bit_num_t* input,int count,std::vector<int>& output) {
            for (int j=0;j<count;++j) {
                int q=input[j]; require(q>=0 && q<s.num_qubits && !used[q],"Gate operands must be distinct, disjoint and in range");
                used[q]=true; output.push_back(q);
            }
        };
        copy(target,nt,targets); copy(negative,nn,negatives); copy(positive,np,positives);
        for (int q:targets) target_mask|=UINT64_C(1)<<q;
        // Keep measured non-targets as controls; a contrary explicit control is
        // a true no-op (including mapping and metadata), as in the reference.
        for (int q=0;q<s.num_qubits;++q) if (((s.measured_mask&~target_mask)>>q)&1) {
            bool bit=(s.measured_value>>q)&1;
            auto& matching=bit?positives:negatives;
            auto& opposite=bit?negatives:positives;
            if (std::find(opposite.begin(),opposite.end(),q)!=opposite.end()) permitted=false;
            else if (std::find(matching.begin(),matching.end(),q)==matching.end()) matching.push_back(q);
        }
        plan=qcs_gate::localize(s.p2l,targets,local);
        auto values=qcs_gate::matrix(kind,angles);
        matrix.reserve(values.size());
        for (auto v:values) matrix.push_back(make_cuDoubleComplex(v.real(),v.imag()));
    });
    // Validate replicated metadata before any rank can skip or start a worker.
    for (int j=0;j<nt;++j) same(s.comm,target[j]);
    for (int j=0;j<nn;++j) same(s.comm,negative[j]);
    for (int j=0;j<np;++j) same(s.comm,positive[j]);
    for (int p:s.p2l) same(s.comm,p);
    // All ranks must fail together if metadata differs.
    // Broadcast outside phase; comparison itself is a coordinated local stage.
    for (auto mask:{s.measured_mask,s.measured_value,s.pending_mask}) {
        auto root=mask; check_mpi(MPI_Bcast(&root,1,MPI_UINT64_T,0,s.comm));
        phase(s.comm,[&] { require(mask==root,"Measurement metadata differs across ranks"); });
    }
    if (!permitted) {
        timer.finish(s.rank,s.gate_timing_sequence,qcs_gate::name(kind),true);
        return;
    }
    std::vector<int32_t> physical_targets, controls, values;
    bool active=true;
    phase(s.comm,[&] {
        for (int q:targets) physical_targets.push_back(plan.l2p[q]);
        for (int bit=0;bit<=1;++bit) for (int q:bit?positives:negatives) {
            int p=plan.l2p[q];
            if (p<local) { controls.push_back(p); values.push_back(bit); }
            else if (((s.rank>>(p-local))&1)!=bit) active=false;
        }
    });
    // Query this matrix/control configuration before touching amplitudes. The
    // gate scratch is independent of the worker transfer allocation and can grow
    // between synchronous calls without moving any pointer borrowed by a worker.
    std::size_t required=0;
    phase(s.comm,[&] {
        DeviceScope device(s.device);
        check_sv(custatevecApplyMatrixGetWorkspaceSize(s.handle,CUDA_C_64F,local,matrix.data(),CUDA_C_64F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW,0,qcs_gate::arity(kind),controls.size(),CUSTATEVEC_COMPUTE_64F,&required));
    });
    void* candidate=nullptr;
    try {
        phase(s.comm,[&] {
            DeviceScope device(s.device);
            if (required>s.gate_bytes) check_cuda(cudaMalloc(&candidate,required));
        });
    } catch (...) {
        try { DeviceScope device(s.device); if (candidate) fatal_cuda(cudaFree(candidate)); }
        catch (...) { check_mpi(MPI_ERR_OTHER); }
        throw;
    }
    phase(s.comm,[&] {
        try {
            DeviceScope device(s.device);
            if (candidate) { if (s.gate_workspace) fatal_cuda(cudaFree(s.gate_workspace)); s.gate_workspace=candidate; s.gate_bytes=required; }
        } catch (...) { check_mpi(MPI_ERR_OTHER); }
    });
    if (!plan.swaps.empty()) prepare_swap(s);
    try {
        for (auto pair:plan.swaps) exchange_bit(s,pair.first,pair.second);
        s.p2l.swap(plan.p2l); s.l2p.swap(plan.l2p);
        phase(s.comm,[&] {
            DeviceScope device(s.device);
            try {
                project_targets(s,s.measured_mask&target_mask);
                if (active) {
                    int width=qcs_gate::arity(kind);
                    int applications=kind==qcs_gate::Kind::swap ? nt/2 : 1;
                    for (int j=0;j<applications;++j) {
                        auto* ptr=physical_targets.empty()?nullptr:physical_targets.data()+(kind==qcs_gate::Kind::swap?2*j:0);
                        check_sv(custatevecApplyMatrix(s.handle,s.state,CUDA_C_64F,local,matrix.data(),CUDA_C_64F,
                            CUSTATEVEC_MATRIX_LAYOUT_ROW,0,ptr,width,controls.data(),values.data(),controls.size(),
                            CUSTATEVEC_COMPUTE_64F,s.gate_workspace,s.gate_bytes));
                    }
                }
            } catch (...) {
                // Preserve host matrix storage until all earlier applications
                // have stopped, even if a later SWAP pair/application failed.
                fatal_cuda(cudaStreamSynchronize(s.stream));
                throw;
            }
            check_cuda(cudaStreamSynchronize(s.stream));
        });
        s.measured_mask&=~target_mask; s.measured_value&=s.measured_mask; s.pending_mask&=s.measured_mask;
    } catch (...) { s.poisoned=true; throw; }
    timer.finish(s.rank,s.gate_timing_sequence,qcs_gate::name(kind),false);
}
