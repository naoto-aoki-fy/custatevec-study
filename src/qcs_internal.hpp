#pragma once
#include "qcs.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>
#include <cstdint>
#include <vector>

// Private ownership only; not installed as part of the public interface.
struct qcs_simulator {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0, ranks = 1, global_bits = 0, device = -1;
    int num_qubits = 0, num_clbits = 0;
    bool disposed = false, initialized = false, poisoned = false;
    bool lease = false;
    qcs_initial_state selected = QCS_INITIAL_STATE_ZERO;
    std::vector<bit_t> clbits;
    std::vector<int> initial_p2l, p2l, l2p;
    std::uint64_t seed = 0, measurement_cursor = 0;
    bool measurement_exhausted = false;
    double* measurement_draw = nullptr;
    std::uint64_t measured_mask = 0, measured_value = 0, pending_mask = 0;
    std::uint64_t elements = 0;
    cuDoubleComplex* state = nullptr;
    cudaStream_t stream = nullptr;
    custatevecHandle_t handle = nullptr;
    unsigned long long gate_timing_sequence = 0;
    // Gate scratch never aliases memory borrowed by the distributed worker.
    void* gate_workspace = nullptr;
    std::size_t gate_bytes = 0;
    custatevecCommunicatorDescriptor_t communicator = nullptr;
    custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler = nullptr;
    custatevecSVSwapWorkerDescriptor_t worker = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudaEvent_t swap_event = nullptr;
    void* swap_extra = nullptr;
    void* transfer = nullptr;
    std::size_t swap_extra_bytes = 0, transfer_bytes = 0;
    // Default workspace and timing events retain independent owners.
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;
    std::vector<cudaEvent_t> events;
};
