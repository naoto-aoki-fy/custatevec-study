#include "bench.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace bench {
[[noreturn]] void fail(const std::string& message) {
    int initialized = 0, finalized = 0, rank = -1;
    MPI_Initialized(&initialized);
    if (initialized) MPI_Finalized(&finalized);
    if (initialized && !finalized) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "[rank %d] fatal: %s\n", rank, message.c_str());
    std::fflush(stderr);
    if (initialized && !finalized) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::exit(EXIT_FAILURE);
}
void mpi_check(int status, const char* expression) {
    if (status == MPI_SUCCESS) return;
    char text[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(status, text, &length);
    fail(std::string(expression) + ": " + std::string(text, length));
}
void cuda_check(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) fail(std::string(expression) + ": " + cudaGetErrorString(status));
}
void sv_check(custatevecStatus_t status, const char* expression) {
    if (status != CUSTATEVEC_STATUS_SUCCESS)
        fail(std::string(expression) + ": " + custatevecGetErrorString(status));
}
std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) fail("Allocation-size addition overflow");
    return a + b;
}
std::size_t checked_multiply(std::uint64_t count, std::size_t bytes) {
    if (bytes == 0) return 0;
    if (count > std::numeric_limits<std::size_t>::max() / bytes)
        fail("Allocation-size multiplication overflow");
    return static_cast<std::size_t>(count) * bytes;
}

Topology::Topology(int nqubits) {
    MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (ranks < 1 || (ranks & (ranks - 1))) fail("MPI rank count must be a power of two");
    for (int count = ranks; count > 1; count >>= 1) ++global_bits;
    local_bits = nqubits - global_bits;
    if (local_bits < 1) fail("Require nqubits > log2(mpi_ranks), with at least one local qubit");
    MPI_TRY(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared));
    MPI_TRY(MPI_Comm_set_errhandler(shared, MPI_ERRORS_RETURN));
    MPI_TRY(MPI_Comm_rank(shared, &local_rank));
    MPI_TRY(MPI_Comm_size(shared, &local_ranks));
}
void Topology::close() { MPI_TRY(MPI_Comm_free(&shared)); }

namespace {
__global__ void initialize_kernel(cuDoubleComplex* state, std::uint64_t count,
                                  bool uniform, bool first_rank, double amplitude) {
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        state[i] = make_cuDoubleComplex(uniform ? amplitude : ((first_rank && i == 0) ? 1.0 : 0.0), 0.0);
    }
}
unsigned blocks(std::uint64_t count) {
    return static_cast<unsigned>(std::min<std::uint64_t>((count + 255) / 256, 65535));
}
std::size_t power_of_two(std::size_t minimum) {
    std::size_t size = 1;
    while (size < minimum) size = checked_add(size, size);
    return size;
}
custatevecCommunicatorType_t communicator_type(const Options& options) {
    if (options.communicator == "external") return CUSTATEVEC_COMMUNICATOR_TYPE_EXTERNAL;
    if (options.communicator == "openmpi") return CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI;
    if (options.communicator == "mpich") return CUSTATEVEC_COMMUNICATOR_TYPE_MPICH;
#if defined(OMPI_MAJOR_VERSION)
    return CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI;
#elif defined(MPICH_NAME) || defined(MPICH_VERSION)
    return CUSTATEVEC_COMMUNICATOR_TYPE_MPICH;
#else
    fail("Cannot identify MPI ABI from headers; select --communicator and, if needed, --communicator-library");
#endif
}
const cuDoubleComplex x_matrix[4] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
}

Runtime::Runtime(const Options& options, const Topology& t)
    : topology(t), nqubits(options.nqubits), local_elements(std::uint64_t{1} << t.local_bits),
      state_bytes(checked_multiply(local_elements, sizeof(cuDoubleComplex))),
      logical_to_physical(static_cast<std::size_t>(options.nqubits)) {
    static_assert(sizeof(std::size_t) >= 8, "A 64-bit host is required");
    static_assert(sizeof(custatevecIndex_t) >= 8, "64-bit cuStateVec indices are required");
    int visible = 0;
    CUDA_TRY(cudaGetDeviceCount(&visible));
    if (visible == 0) fail("No visible CUDA GPU");
    const int device = visible == 1 ? 0 : t.local_rank;
    if (device >= visible) fail("GPU oversubscription: node-local rank exceeds visible GPU count");
    CUDA_TRY(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CUDA_TRY(cudaGetDeviceProperties(&properties, device));
    // Device ordinals are process-local; compare actual UUIDs across the shared node.
    constexpr int uuid_bytes = sizeof(properties.uuid.bytes);
    std::vector<char> uuids(checked_multiply(t.local_ranks, uuid_bytes));
    MPI_TRY(MPI_Allgather(properties.uuid.bytes, uuid_bytes, MPI_BYTE, uuids.data(),
                          uuid_bytes, MPI_BYTE, t.shared));
    for (int a = 0; a < t.local_ranks; ++a)
        for (int b = a + 1; b < t.local_ranks; ++b)
            if (std::memcmp(uuids.data() + static_cast<std::size_t>(a) * uuid_bytes,
                            uuids.data() + static_cast<std::size_t>(b) * uuid_bytes, uuid_bytes) == 0)
                fail("Duplicate GPU UUID on this node; assign one distinct GPU per MPI process");

    CUDA_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    SV_TRY(custatevecCreate(&handle));
    SV_TRY(custatevecSetStream(handle, stream));
    allocate(reinterpret_cast<void**>(&state), state_bytes, "complex128 sub-state-vector");
    if (t.ranks > 1) {
        CUDA_TRY(cudaStreamCreateWithFlags(&transfer_stream_, cudaStreamNonBlocking));
        const auto type = communicator_type(options);
        const char* library = options.communicator_library.empty() ? nullptr : options.communicator_library.c_str();
        SV_TRY(custatevecCommunicatorCreate(handle, &communicator_, type, library));
        if (t.rank == 0)
            std::cerr << "resolved_custatevec_communicator="
                      << (type == CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI ? "openmpi" :
                          type == CUSTATEVEC_COMMUNICATOR_TYPE_MPICH ? "mpich" : "external")
                      << '\n';
        CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventInterprocess | cudaEventDisableTiming));
        std::size_t minimum_transfer = 0;
        SV_TRY(custatevecSVSwapWorkerCreate(handle, &worker_, communicator_, state, t.rank,
                                            event_, CUDA_C_64F, transfer_stream_, &swap_extra_bytes_, &minimum_transfer));
        transfer_bytes_ = power_of_two(minimum_transfer);
        // Check the combined remaining requirements before allocating either buffer.
        std::size_t free_bytes = 0, total_bytes = 0;
        CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (checked_add(swap_extra_bytes_, transfer_bytes_) > free_bytes)
            fail("Insufficient GPU memory for cuStateVec swap workspace and communication buffer");
        allocate(&swap_extra_, swap_extra_bytes_, "cuStateVec swap workspace");
        allocate(&transfer_, transfer_bytes_, "communication transfer buffer");
        SV_TRY(custatevecSVSwapWorkerSetExtraWorkspace(handle, worker_, swap_extra_, swap_extra_bytes_));
        SV_TRY(custatevecSVSwapWorkerSetTransferWorkspace(handle, worker_, transfer_, transfer_bytes_));
        SV_TRY(custatevecDistIndexBitSwapSchedulerCreate(handle, &scheduler_, t.global_bits, t.local_bits));
    }
    reset_mapping();
    // This smoke matrix is also a workspace-query example for later workload stages.
    if (options.smoke) reserve_gate_workspace(x_matrix, 1);
    report_metadata(options, device, properties);
}

void Runtime::allocate(void** pointer, std::size_t bytes, const char* purpose) {
    if (!bytes) return;
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (bytes > free_bytes)
        fail(std::string("Insufficient GPU memory for ") + purpose + ": need " +
             std::to_string(bytes) + ", free " + std::to_string(free_bytes) + " bytes");
    CUDA_TRY(cudaMalloc(pointer, bytes));
}
void Runtime::report_metadata(const Options& options, int device, const cudaDeviceProp& properties) {
    char host[MPI_MAX_PROCESSOR_NAME] = {};
    int host_length = 0;
    MPI_TRY(MPI_Get_processor_name(host, &host_length));
    char pci[32] = {};
    CUDA_TRY(cudaDeviceGetPCIBusId(pci, sizeof(pci), device));
    std::ostringstream uuid;
    for (unsigned char value : properties.uuid.bytes)
        uuid << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(value);
    const auto required = checked_add(checked_add(state_bytes, swap_extra_bytes_),
                                      checked_add(transfer_bytes_, gate_bytes_));
    // Serialize diagnostics so rank mappings remain legible even with merged stderr.
    for (int r = 0; r < topology.ranks; ++r) {
        MPI_TRY(MPI_Barrier(MPI_COMM_WORLD));
        if (topology.rank == r) {
            std::cerr << "rank=" << r << " host=" << std::string(host, host_length)
                      << " local_rank=" << topology.local_rank << " visible_device=" << device
                      << " gpu=\"" << properties.name << "\" pci=" << pci << " uuid=" << uuid.str()
                      << " local_bits=" << topology.local_bits << " local_elements=" << local_elements
                      << " state_bytes=" << state_bytes << " swap_workspace_bytes=" << swap_extra_bytes_
                      << " transfer_bytes=" << transfer_bytes_ << " gate_workspace_bytes=" << gate_bytes_
                      << " explicit_allocation_bytes=" << required << '\n' << std::flush;
        }
    }
    if (topology.rank == 0) {
        int runtime_version = 0, driver_version = 0;
        CUDA_TRY(cudaRuntimeGetVersion(&runtime_version));
        CUDA_TRY(cudaDriverGetVersion(&driver_version));
        char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING] = {};
        int length = 0;
        MPI_TRY(MPI_Get_library_version(mpi_version, &length));
        std::cerr << "CUDA_runtime=" << runtime_version << " CUDA_driver=" << driver_version
                  << " cuStateVec=" << custatevecGetVersion() << " MPI=" << std::string(mpi_version, length)
                  << "\nseed=" << options.seed << " warmup=" << options.warmup << " repeats=" << options.repeats
                  << " nqubits=" << nqubits << " precision=complex128 compute=64F fusion=disabled"
                  << " communicator=" << options.communicator << " communicator_library="
                  << (options.communicator_library.empty() ? "loaded MPI library" : options.communicator_library)
                  << "\ntiming_scope=complete operation and communication; device synchronization before stop;"
                  << " rank-local MPI_Wtime elapsed reduced with MPI_MAX after stop;"
                  << " allocation/initialization/output excluded; measurement probability preprocessing included\n";
    }
}
void Runtime::synchronize() {
    // Covers every stream, including internal library work, before timer stop.
    CUDA_TRY(cudaDeviceSynchronize());
}
void Runtime::reset_mapping() {
    std::iota(logical_to_physical.begin(), logical_to_physical.end(), 0);
}
void Runtime::initialize_zero() {
    initialize_kernel<<<blocks(local_elements), 256, 0, stream>>>(state, local_elements, false, topology.rank == 0, 0.0);
    CUDA_TRY(cudaGetLastError());
}
void Runtime::initialize_uniform() {
    const double amplitude = std::exp2(-0.5 * static_cast<double>(nqubits));
    initialize_kernel<<<blocks(local_elements), 256, 0, stream>>>(state, local_elements, true, false, amplitude);
    CUDA_TRY(cudaGetLastError());
}
void Runtime::reserve_gate_workspace(const cuDoubleComplex* matrix, int targets) {
    std::size_t required = 0;
    SV_TRY(custatevecApplyMatrixGetWorkspaceSize(handle, CUDA_C_64F, topology.local_bits, matrix,
              CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 0, CUSTATEVEC_COMPUTE_64F, &required));
    if (required <= gate_bytes_) return;
    synchronize();
    if (gate_workspace_) CUDA_TRY(cudaFree(gate_workspace_));
    gate_workspace_ = nullptr;
    gate_bytes_ = 0;
    allocate(&gate_workspace_, required, "complex128 gate workspace");
    gate_bytes_ = required;
    const auto total = checked_add(checked_add(state_bytes, swap_extra_bytes_),
        checked_add(transfer_bytes_, checked_add(gate_bytes_, measurement_bytes_)));
    std::cerr << "rank=" << topology.rank << " reserved_gate_workspace_bytes=" << gate_bytes_
              << " explicit_allocation_bytes=" << total << '\n';
}
void Runtime::reserve_measurement_workspace() {
    if (measurement_workspace_) return;
    SV_TRY(custatevecGetDefaultWorkspaceSize(handle, &measurement_bytes_));
    allocate(&measurement_workspace_, measurement_bytes_, "cuStateVec measurement workspace");
    SV_TRY(custatevecSetWorkspace(handle, measurement_workspace_, measurement_bytes_));
    const auto required = checked_add(checked_add(state_bytes, swap_extra_bytes_),
        checked_add(transfer_bytes_, checked_add(gate_bytes_, measurement_bytes_)));
    std::cerr << "rank=" << topology.rank << " measurement_workspace_bytes=" << measurement_bytes_
              << " explicit_allocation_bytes=" << required << '\n';
}
void Runtime::apply_matrix(const cuDoubleComplex* matrix, const int32_t* targets, int count) {
    SV_TRY(custatevecApplyMatrix(handle, state, CUDA_C_64F, topology.local_bits, matrix, CUDA_C_64F,
              CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, count, nullptr, nullptr, 0,
              CUSTATEVEC_COMPUTE_64F, gate_workspace_, gate_bytes_));
}
void Runtime::swap_local_global(int local_bit, int global_bit) {
    if (!worker_ || local_bit < 0 || local_bit >= topology.local_bits ||
        global_bit < topology.local_bits || global_bit >= nqubits)
        fail("swap_local_global requires valid local and global physical bit positions");
    // The worker uses a separate transfer stream. Finish compute writes before
    // the worker accesses the state. Both stream waits belong to the timed
    // operation when invoked by a workload.
    CUDA_TRY(cudaStreamSynchronize(stream));
    const int2 pair = make_int2(local_bit, global_bit);
    uint32_t batches = 0;
    SV_TRY(custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(handle, scheduler_, &pair, 1,
                                                             nullptr, nullptr, 0, &batches));
    for (uint32_t batch = 0; batch < batches; ++batch) {
        custatevecSVSwapParameters_t parameters{};
        SV_TRY(custatevecDistIndexBitSwapSchedulerGetParameters(handle, scheduler_, batch, topology.rank, &parameters));
        SV_TRY(custatevecSVSwapWorkerSetParameters(handle, worker_, &parameters, parameters.dstSubSVIndex));
        SV_TRY(custatevecSVSwapWorkerExecute(handle, worker_, 0, parameters.transferSize));
    }
    // The worker owns MPI requests and synchronizes with its transfer peer.
    // As in NVIDIA's distributed-swap example, wait on the worker's stream
    // after all batches: Execute returning alone is not GPU completion.
    // Subsequent compute gates may access the state only after this wait.
    CUDA_TRY(cudaStreamSynchronize(transfer_stream_));
    for (auto& physical : logical_to_physical) {
        if (physical == local_bit) physical = global_bit;
        else if (physical == global_bit) physical = local_bit;
    }
}
void Runtime::expect_basis(std::uint64_t index) {
    synchronize();
    std::vector<cuDoubleComplex> host(static_cast<std::size_t>(local_elements));
    CUDA_TRY(cudaMemcpy(host.data(), state, state_bytes, cudaMemcpyDeviceToHost));
    double local_norm = 0.0;
    int bad = 0;
    const auto base = static_cast<std::uint64_t>(topology.rank) * local_elements;
    for (std::uint64_t i = 0; i < local_elements; ++i) {
        const double expected = base + i == index ? 1.0 : 0.0;
        if (!std::isfinite(host[i].x) || !std::isfinite(host[i].y) ||
            std::abs(host[i].x - expected) > 1e-12 || std::abs(host[i].y) > 1e-12) bad = 1;
        local_norm += host[i].x * host[i].x + host[i].y * host[i].y;
    }
    int any_bad = 0;
    double norm = 0.0;
    MPI_TRY(MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    MPI_TRY(MPI_Allreduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    if (any_bad || !std::isfinite(norm) || std::abs(norm - 1.0) > 1e-12) fail("Runtime smoke check: distributed basis state mismatch");
}
void Runtime::smoke_check() {
    reset_mapping();
    initialize_zero();
    expect_basis(0);
    const int32_t target = 0;
    apply_matrix(x_matrix, &target, 1);
    expect_basis(1);
    if (topology.ranks > 1) {
        swap_local_global(0, topology.local_bits);
        expect_basis(std::uint64_t{1} << topology.local_bits);
        swap_local_global(0, topology.local_bits);
        expect_basis(1);
    }
    apply_matrix(x_matrix, &target, 1);
    expect_basis(0);
    if (topology.rank == 0)
        std::cerr << "Runtime smoke check passed: allocation, initialization, local complex128 gate,"
                  << (topology.ranks > 1 ? " distributed swap and inverse," : "")
                  << " double MPI norm reduction; no benchmark timings emitted\n";
}
void Runtime::close() {
    synchronize();
    MPI_TRY(MPI_Barrier(MPI_COMM_WORLD));
    if (scheduler_) SV_TRY(custatevecDistIndexBitSwapSchedulerDestroy(handle, scheduler_));
    if (worker_) SV_TRY(custatevecSVSwapWorkerDestroy(handle, worker_));
    if (communicator_) SV_TRY(custatevecCommunicatorDestroy(handle, communicator_));
    SV_TRY(custatevecDestroy(handle));
    if (measurement_workspace_) CUDA_TRY(cudaFree(measurement_workspace_));
    if (gate_workspace_) CUDA_TRY(cudaFree(gate_workspace_));
    if (transfer_) CUDA_TRY(cudaFree(transfer_));
    if (swap_extra_) CUDA_TRY(cudaFree(swap_extra_));
    CUDA_TRY(cudaFree(state));
    if (event_) CUDA_TRY(cudaEventDestroy(event_));
    // The device was synchronized before cleanup and worker_ is destroyed.
    if (transfer_stream_) CUDA_TRY(cudaStreamDestroy(transfer_stream_));
    CUDA_TRY(cudaStreamDestroy(stream));
}
} // namespace bench
