#pragma once

#if !__has_include(<mpi.h>)
#error "Missing mpi.h: CONFIG_MK must contain include flags for the target MPI installation"
#endif
#include <mpi.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>
#include "device_assignment.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bench {

[[noreturn]] void fail(const std::string& message);
void mpi_check(int status, const char* expression);
void cuda_check(cudaError_t status, const char* expression);
void sv_check(custatevecStatus_t status, const char* expression);
#define MPI_TRY(expr) ::bench::mpi_check((expr), #expr)
#define CUDA_TRY(expr) ::bench::cuda_check((expr), #expr)
#define SV_TRY(expr) ::bench::sv_check((expr), #expr)

struct Options {
    std::string job;
    std::string gate_timings;
    std::string initial_state = "uniform";
    int nqubits = 31;
    int repeats = 128;
    int warmup = 0;
    std::uint64_t seed = 1;
    bool smoke = false;
    bool check_hadamard = false;
    bool check_measure = false;
    bool help = false;
    bool explicit_nqubits = false;
    std::string communicator = "auto";
    std::string communicator_library;
    std::string p2p = "auto";
    std::string transfer_buffer = "auto";
    double transfer_buffer_fraction = 0.95;
};
Options parse_options(int argc, char** argv);
void print_help();

// Throws on failure; must never use the MPI-aware fail()/CUDA_TRY path.
DeviceAssignment initialize_cuda_before_mpi();

// World rank is the contiguous sub-state-vector index, independent of node layout.
struct Topology {
    int rank = 0, ranks = 1, local_rank = 0, local_ranks = 1;
    int global_bits = 0, local_bits = 0;
    MPI_Comm shared = MPI_COMM_NULL;
    explicit Topology(int nqubits);
    void close();
};

std::size_t checked_add(std::size_t a, std::size_t b);
std::size_t checked_multiply(std::uint64_t count, std::size_t bytes);

class Runtime {
public:
    Runtime(const Options& options, const Topology& topology, const DeviceAssignment& assignment);
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    // Cleanup is explicit: errors must abort the world before MPI_Finalize.
    void close();
    void synchronize();
    void initialize_zero();
    void initialize_uniform();
    void initialize_ghz();
    void reset_mapping();
    void reserve_gate_workspace(const cuDoubleComplex* matrix, int targets);
    void reserve_measurement_workspace();
    void apply_matrix(const cuDoubleComplex* matrix, const int32_t* targets, int count);
    void swap_local_global(int local_bit, int global_bit);
    void smoke_check();
    void finalize_preparation();

    const Topology& topology;
    int nqubits;
    std::uint64_t local_elements;
    std::size_t state_bytes;
    cuDoubleComplex* state = nullptr;
    custatevecHandle_t handle = nullptr;
    cudaStream_t stream = nullptr; // compute: handle operations and initialization
    std::vector<int> logical_to_physical;
private:
    Options options_;
    bool preparation_finalized_ = false;
    std::size_t minimum_transfer_ = 0;
    void allocate(void** pointer, std::size_t bytes, const char* purpose);
    void report_metadata(const Options& options, int device, const cudaDeviceProp& properties);
    void expect_basis(std::uint64_t index);
    void setup_p2p(const Options& options);
    // Retained until the worker is destroyed; these are IPC imports, not allocations.
    std::vector<void*> p2p_states_;
    std::vector<int32_t> p2p_ranks_;
    std::vector<cudaEvent_t> p2p_events_;
    bool ipc_capable_ = false;
    custatevecCommunicatorDescriptor_t communicator_ = nullptr;
    custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler_ = nullptr;
    custatevecSVSwapWorkerDescriptor_t worker_ = nullptr;
    cudaStream_t transfer_stream_ = nullptr; // dedicated to worker_; only for ranks > 1
    cudaEvent_t event_ = nullptr;
    void* swap_extra_ = nullptr;
    void* transfer_ = nullptr;
    void* gate_workspace_ = nullptr;
    void* measurement_workspace_ = nullptr;
    std::size_t measurement_bytes_ = 0;
    std::size_t swap_extra_bytes_ = 0, transfer_bytes_ = 0, gate_bytes_ = 0;
};

struct JobResult {
    std::uint64_t measurement = 0; // logical qubit q is bit q
    bool has_measurement = false;
};

class Job {
public:
    virtual ~Job() = default;
    // Allocated by the harness only for opt-in profiling, outside repetitions.
    struct GateTiming {
        cudaEvent_t start = nullptr, stop = nullptr;
        int logical_qubit = -1; // -1 denotes full-register measurement
    };
    std::vector<GateTiming> gate_timings;
    bool profile_gates = false;
    std::size_t timed_gate_count = 0;
    void begin_gate(Runtime& runtime, int logical_qubit);
    void end_gate(Runtime& runtime);
    virtual void prepare(Runtime&) = 0; // reusable descriptors/workspaces, before repetitions
    virtual void initialize(Runtime&, std::uint64_t seed, std::uint64_t repetition) = 0;
    // Complete all outstanding MPI requests before returning. GPU completion is
    // also enforced by the harness. Format the result only after the timer stops.
    virtual JobResult execute(Runtime&, std::uint64_t seed, std::uint64_t repetition) = 0;
};
std::unique_ptr<Job> make_job(const std::string& name, const std::string& initial_state = "uniform");
void check_hadamard(Runtime& runtime);
void check_measure(Runtime& runtime);
void run_benchmark(Runtime& runtime, Job& job, const Options& options);

} // namespace bench
