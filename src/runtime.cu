#include "bench.hpp"
#include "transfer_policy.hpp"

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
DeviceAssignment initialize_cuda_before_mpi() {
    // Unlike cuda_check(), this path cannot call even MPI_Initialized().
    const auto check = [](cudaError_t status, const char* expression) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
    };
    int visible = 0;
    check(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    const auto assignment = device_assignment::discover(visible,
        [](const char* name) { return std::getenv(name); });
    std::fprintf(stderr, "[pre-MPI] local_rank_source=%s local_rank=%d visible_devices=%d visible_device=%d\n",
                 assignment.source.c_str(), assignment.local_rank, visible, assignment.device);
    check(cudaSetDevice(assignment.device), "cudaSetDevice");
    check(cudaFree(nullptr), "cudaFree(nullptr): initialize assigned CUDA context");
    std::fprintf(stderr, "[pre-MPI] visible_device=%d context_initialization=succeeded\n", assignment.device);
    return assignment;
}

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
                                  bool uniform, bool first_rank, bool last_rank, double amplitude) {
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        const bool endpoint = (first_rank && i == 0) || (last_rank && i == count - 1);
        state[i] = make_cuDoubleComplex((uniform || endpoint) ? amplitude : 0.0, 0.0);
    }
}
unsigned blocks(std::uint64_t count) {
    return static_cast<unsigned>(std::min<std::uint64_t>((count + 255) / 256, 65535));
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

// Only capability/resource mapping failures are eligible for MPI fallback.
// Bad handles/arguments, allocation failures, and asynchronous CUDA failures
// indicate a programming or runtime failure and must abort the world.
bool ipc_attempt(cudaError_t status, const char* operation, std::string& reason) {
    if (status == cudaSuccess) return true;
    if (status != cudaErrorNotSupported && status != cudaErrorPeerAccessUnsupported &&
        status != cudaErrorTooManyPeers && status != cudaErrorMapBufferObjectFailed &&
        status != cudaErrorOperatingSystem)
        cuda_check(status, operation);
    reason = std::string(operation) + ": " + cudaGetErrorString(status);
    // Consume the handled API error so later kernel launch checks do not see it.
    const auto pending = cudaGetLastError();
    if (pending != cudaSuccess && pending != status) cuda_check(pending, operation);
    return false;
}
}

Runtime::Runtime(const Options& options, const Topology& t, const DeviceAssignment& assignment)
    : topology(t), nqubits(options.nqubits), local_elements(std::uint64_t{1} << t.local_bits),
      state_bytes(checked_multiply(local_elements, sizeof(cuDoubleComplex))),
      logical_to_physical(static_cast<std::size_t>(options.nqubits)), options_(options) {
    static_assert(sizeof(std::size_t) >= 8, "A 64-bit host is required");
    static_assert(sizeof(custatevecIndex_t) >= 8, "64-bit cuStateVec indices are required");
    device_assignment::validate(assignment, t.local_rank);
    const int device = assignment.device;
    int current_device = -1;
    CUDA_TRY(cudaGetDevice(&current_device));
    if (current_device != device) fail("Current CUDA device changed during MPI initialization");
    std::fprintf(stderr, "rank=%d local_rank=%d visible_device=%d pre_mpi_assignment=validated local_rank_source=%s\n",
                 t.rank, t.local_rank, device, assignment.source.c_str());
    cudaDeviceProp properties{};
    CUDA_TRY(cudaGetDeviceProperties(&properties, device));
    int ipc_events = 0;
    CUDA_TRY(cudaDeviceGetAttribute(&ipc_events, cudaDevAttrIpcEventSupport, device));
    ipc_capable_ = properties.unifiedAddressing && ipc_events;
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
    if (t.ranks > 1 && (options.smoke || options.check_hadamard ||
                        options.job == "hadamard" || options.job == "hadamard-reverse")) {
        CUDA_TRY(cudaStreamCreateWithFlags(&transfer_stream_, cudaStreamNonBlocking));
        const auto type = communicator_type(options);
        const char* library = options.communicator_library.empty() ? nullptr : options.communicator_library.c_str();
        SV_TRY(custatevecCommunicatorCreate(handle, &communicator_, type, library));
        if (t.rank == 0)
            std::cerr << "resolved_custatevec_communicator="
                      << (type == CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI ? "openmpi" :
                          type == CUSTATEVEC_COMMUNICATOR_TYPE_MPICH ? "mpich" : "external")
                      << '\n';
        if (ipc_capable_) {
            std::string reason;
            ipc_capable_ = ipc_attempt(cudaEventCreateWithFlags(&event_,
                cudaEventDisableTiming | cudaEventInterprocess), "cudaEventCreateWithFlags(IPC)", reason);
            if (!ipc_capable_)
                std::cerr << "rank=" << t.rank << " P2P event unavailable: " << reason << '\n';
        }
        if (!ipc_capable_)
            CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
        SV_TRY(custatevecSVSwapWorkerCreate(handle, &worker_, communicator_, state, t.rank,
                                            event_, CUDA_C_64F, transfer_stream_, &swap_extra_bytes_, &minimum_transfer_));
        allocate(&swap_extra_, swap_extra_bytes_, "cuStateVec swap workspace");
        SV_TRY(custatevecSVSwapWorkerSetExtraWorkspace(handle, worker_, swap_extra_, swap_extra_bytes_));
        SV_TRY(custatevecDistIndexBitSwapSchedulerCreate(handle, &scheduler_, t.global_bits, t.local_bits));
    }
    reset_mapping();
    // This smoke matrix is also a workspace-query example for later workload stages.
    if (options.smoke) reserve_gate_workspace(x_matrix, 1);
    report_metadata(options, device, properties);
    if (worker_) setup_p2p(options);
}

void Runtime::finalize_preparation() {
    if (preparation_finalized_) return;
    synchronize();
    const auto explicit_total = [&] {
        return checked_add(checked_add(state_bytes, swap_extra_bytes_),
            checked_add(transfer_bytes_, checked_add(gate_bytes_, measurement_bytes_)));
    };
    std::cerr << "rank=" << topology.rank << " transfer_buffer_policy=" << options_.transfer_buffer
              << " transfer_buffer_fraction=" << std::setprecision(17) << options_.transfer_buffer_fraction
              << " transfer_allocation_skipped=" << (worker_ ? "false" : "true") << '\n';
    if (!worker_) {
        std::cerr << "rank=" << topology.rank
                  << " transfer_skip_reason=no-distributed-swaps transfer_bytes=0 explicit_allocation_bytes="
                  << explicit_total() << '\n';
        preparation_finalized_ = true;
        return;
    }
    std::uint64_t common_minimum = 0;
    const std::uint64_t local_minimum = minimum_transfer_;
    MPI_TRY(MPI_Allreduce(&local_minimum, &common_minimum, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD));
    const auto refresh = [&]() -> std::uint64_t {
        // All successful allocations from a failed attempt have been released
        // on all ranks before querying this snapshot.
        MPI_TRY(MPI_Barrier(MPI_COMM_WORLD));
        std::size_t free = 0, total = 0;
        CUDA_TRY(cudaMemGetInfo(&free, &total));
        const std::uint64_t local_budget = transfer_policy::budget(free, options_.transfer_buffer_fraction);
        std::uint64_t common_budget = 0;
        MPI_TRY(MPI_Allreduce(&local_budget, &common_budget, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD));
        std::cerr << "rank=" << topology.rank << " transfer_baseline_free_bytes=" << free
                  << " transfer_local_budget=" << local_budget << " transfer_common_budget=" << common_budget
                  << " transfer_worker_minimum=" << local_minimum
                  << " transfer_common_minimum=" << common_minimum << '\n';
        return common_budget;
    };
    const auto initial = transfer_policy::select(refresh(), common_minimum, options_.transfer_buffer == "auto");
    std::cerr << "rank=" << topology.rank << " transfer_initial_bytes=" << initial << '\n';
    unsigned retries = 0;
    transfer_bytes_ = transfer_policy::allocate_common(initial, common_minimum,
        [&](std::uint64_t bytes) {
            // Only cudaErrorMemoryAllocation is recoverable here. Every other
            // CUDA failure retains the existing world-abort behavior.
            const auto status = cudaMalloc(&transfer_, static_cast<std::size_t>(bytes));
            if (status != cudaSuccess && status != cudaErrorMemoryAllocation)
                cuda_check(status, "cudaMalloc(transfer)");
            int failed = status == cudaErrorMemoryAllocation, any_failed = 0;
            if (failed) {
                transfer_ = nullptr;
                std::cerr << "rank=" << topology.rank << " transfer_attempt_bytes=" << bytes
                          << " allocation_failure=cudaErrorMemoryAllocation\n";
                const auto pending = cudaGetLastError();
                if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation)
                    cuda_check(pending, "cudaGetLastError after transfer allocation failure");
            }
            MPI_TRY(MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
            return !any_failed;
        },
        [&] {
            if (transfer_) CUDA_TRY(cudaFree(transfer_));
            transfer_ = nullptr;
        }, refresh,
        [&](std::uint64_t previous, std::uint64_t next) {
            ++retries;
            std::cerr << "rank=" << topology.rank << " transfer_retries=" << retries
                      << " previous_bytes=" << previous << " next_bytes=" << next
                      << " reduction_reason=collective-allocation-failure-and-refreshed-budget\n";
        });
    // No rank registers or starts work until every allocation has succeeded.
    SV_TRY(custatevecSVSwapWorkerSetTransferWorkspace(handle, worker_, transfer_, transfer_bytes_));
    std::size_t remaining = 0, total = 0;
    CUDA_TRY(cudaMemGetInfo(&remaining, &total));
    std::cerr << "rank=" << topology.rank << " transfer_final_bytes=" << transfer_bytes_
              << " transfer_retries=" << retries << " remaining_free_bytes=" << remaining
              << " explicit_allocation_bytes=" << explicit_total() << '\n';
    MPI_TRY(MPI_Barrier(MPI_COMM_WORLD));
    preparation_finalized_ = true;
}

void Runtime::setup_p2p(const Options& options) {
    const auto& t = topology;
    std::cerr << "rank=" << t.rank << " requested_p2p=" << options.p2p
              << " p2p_transport=explicit-CUDA-IPC (not a link-type claim)"
              << " other_peers=CUDA-aware-MPI\n";
    if (options.p2p == "off") {
        std::cerr << "rank=" << t.rank << " registered_p2p_peers=[] reason=off;"
                  << " communicator may still use CUDA IPC internally\n";
        return;
    }

    // The actual workloads swap one local/global bit at a time, hence their
    // remote endpoint is rank XOR (1 << global-rank-bit). Measurement has no
    // swaps; the smoke check only uses the lowest global bit. Never infer node
    // membership from these indices: shared is the authority for locality.
    const auto needed = [&](int peer) {
        const unsigned delta = static_cast<unsigned>(t.rank ^ peer);
        if (options.smoke) return delta == 1;
        return (options.job == "hadamard" || options.job == "hadamard-reverse" || options.check_hadamard) &&
               delta != 0 && (delta & (delta - 1)) == 0;
    };
    std::vector<int> ranks(t.local_ranks);
    MPI_TRY(MPI_Allgather(&t.rank, 1, MPI_INT, ranks.data(), 1, MPI_INT, t.shared));
    bool has_peer = false;
    for (int peer : ranks) has_peer = has_peer || needed(peer);

    cudaIpcMemHandle_t memory_handle{};
    cudaIpcEventHandle_t event_handle{};
    std::string export_reason = "no same-node swap peers needed by workload";
    int exported = 0;
    if (has_peer) {
        export_reason = "GPU lacks unified addressing or IPC event support";
        if (ipc_capable_ &&
            ipc_attempt(cudaIpcGetMemHandle(&memory_handle, state), "cudaIpcGetMemHandle", export_reason) &&
            ipc_attempt(cudaIpcGetEventHandle(&event_handle, event_), "cudaIpcGetEventHandle", export_reason))
            exported = 1;
        if (!exported)
            std::cerr << "rank=" << t.rank << " P2P export unavailable: " << export_reason << '\n';
    }
    std::vector<int> exports(t.local_ranks);
    std::vector<cudaIpcMemHandle_t> memories(t.local_ranks);
    std::vector<cudaIpcEventHandle_t> events(t.local_ranks);
    MPI_TRY(MPI_Allgather(&exported, 1, MPI_INT, exports.data(), 1, MPI_INT, t.shared));
    MPI_TRY(MPI_Allgather(&memory_handle, sizeof(memory_handle), MPI_BYTE,
                          memories.data(), sizeof(memory_handle), MPI_BYTE, t.shared));
    MPI_TRY(MPI_Allgather(&event_handle, sizeof(event_handle), MPI_BYTE,
                          events.data(), sizeof(event_handle), MPI_BYTE, t.shared));

    std::vector<void*> pointers(t.local_ranks, nullptr);
    std::vector<cudaEvent_t> imports(t.local_ranks, nullptr);
    std::vector<int> usable(t.local_ranks, 0);
    std::vector<std::string> reasons(t.local_ranks);
    for (int i = 0; i < t.local_ranks; ++i) {
        if (!needed(ranks[i])) continue;
        if (!exported || !exports[i]) {
            reasons[i] = !exported ? export_reason : "peer could not export memory/event (see peer diagnostic)";
            continue;
        }
        // IPC import resolves the exporting GPU even if scheduler isolation
        // hides it from this process. Never use the peer's device ordinal.
        if (!ipc_attempt(cudaIpcOpenMemHandle(&pointers[i], memories[i], cudaIpcMemLazyEnablePeerAccess),
                         "cudaIpcOpenMemHandle", reasons[i])) continue;
        if (!ipc_attempt(cudaIpcOpenEventHandle(&imports[i], events[i]),
                         "cudaIpcOpenEventHandle", reasons[i])) continue;
        usable[i] = 1;
    }
    // Rows are importing local ranks, columns are exporting local ranks.
    std::vector<int> outcomes(checked_multiply(t.local_ranks, static_cast<std::size_t>(t.local_ranks)));
    MPI_TRY(MPI_Allgather(usable.data(), t.local_ranks, MPI_INT, outcomes.data(),
                          t.local_ranks, MPI_INT, t.shared));
    int missing = 0;
    for (int i = 0; i < t.local_ranks; ++i) {
        if (!needed(ranks[i])) continue;
        const bool mutual = usable[i] &&
            outcomes[static_cast<std::size_t>(i) * t.local_ranks + t.local_rank];
        if (mutual) {
            p2p_states_.push_back(pointers[i]);
            p2p_ranks_.push_back(static_cast<int32_t>(ranks[i]));
            p2p_events_.push_back(imports[i]);
        } else {
            missing = 1;
            if (imports[i]) CUDA_TRY(cudaEventDestroy(imports[i]));
            if (pointers[i]) CUDA_TRY(cudaIpcCloseMemHandle(pointers[i]));
            std::cerr << "rank=" << t.rank << " peer_world_rank=" << ranks[i]
                      << " path=CUDA-aware-MPI fallback_reason="
                      << (usable[i] ? "reverse import unavailable (see peer diagnostic)" : reasons[i]) << '\n';
        }
    }
    // Every world rank participates, including single-rank nodes and nodes
    // with no required pairs. Fail before registration/timing, never by return
    // into collective cleanup while another rank is still in setup.
    int any_missing = 0;
    MPI_TRY(MPI_Allreduce(&missing, &any_missing, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    if (options.p2p == "required" && any_missing) {
        // No worker has used imports yet. Release even successful imports on
        // the failure path before coherently aborting; exporters remain alive.
        for (auto event : p2p_events_) CUDA_TRY(cudaEventDestroy(event));
        for (auto pointer : p2p_states_) CUDA_TRY(cudaIpcCloseMemHandle(pointer));
        MPI_TRY(MPI_Barrier(MPI_COMM_WORLD));
        fail("--p2p required: a needed same-node swap pair is unavailable");
    }
    if (!p2p_ranks_.empty())
        SV_TRY(custatevecSVSwapWorkerSetSubSVsP2P(handle, worker_, p2p_states_.data(),
                  p2p_ranks_.data(), p2p_events_.data(), static_cast<uint32_t>(p2p_ranks_.size())));
    MPI_TRY(MPI_Barrier(MPI_COMM_WORLD));
    std::ostringstream peers;
    peers << "rank=" << t.rank << " registered_p2p_peer_world_ranks=[";
    for (std::size_t i = 0; i < p2p_ranks_.size(); ++i) {
        if (i) peers << ',';
        peers << p2p_ranks_[i];
    }
    peers << "] scope=workload-swap-peers";
    if (!has_peer) peers << " reason=no-same-node-swap-peers-needed";
    std::cerr << peers.str() << '\n';
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
                  << "\ntiming_backend=CUDA events\ntiming_scope=complete operation and communication; device synchronization before stop;"
                  << " rank-local GPU event-marker elapsed (including intervening host/MPI waits);"
                  << " milliseconds converted to seconds and reduced with MPI_MAX after stop;"
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
    initialize_kernel<<<blocks(local_elements), 256, 0, stream>>>(state, local_elements, false, topology.rank == 0, false, 1.0);
    CUDA_TRY(cudaGetLastError());
}
void Runtime::initialize_uniform() {
    const double amplitude = std::exp2(-0.5 * static_cast<double>(nqubits));
    initialize_kernel<<<blocks(local_elements), 256, 0, stream>>>(state, local_elements, true, false, false, amplitude);
    CUDA_TRY(cudaGetLastError());
}
void Runtime::initialize_ghz() {
    const double amplitude = 1.0 / std::sqrt(2.0);
    initialize_kernel<<<blocks(local_elements), 256, 0, stream>>>(state, local_elements,
        false, topology.rank == 0, topology.rank == topology.ranks - 1, amplitude);
    CUDA_TRY(cudaGetLastError());
}
void Runtime::reserve_gate_workspace(const cuDoubleComplex* matrix, int targets) {
    std::size_t required = 0;
    SV_TRY(custatevecApplyMatrixGetWorkspaceSize(handle, CUDA_C_64F, topology.local_bits, matrix,
              CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 0, CUSTATEVEC_COMPUTE_64F, &required));
    if (required <= gate_bytes_) return;
    if (preparation_finalized_) fail("Gate workspace growth after transfer budget finalization");
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
    if (preparation_finalized_) fail("Measurement workspace reservation after budget finalization");
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
    if (!preparation_finalized_ || !transfer_ || !worker_ || local_bit < 0 || local_bit >= topology.local_bits ||
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
    finalize_preparation();
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
    // All workers must stop using exports before any process closes imports.
    MPI_TRY(MPI_Barrier(MPI_COMM_WORLD));
    for (auto event : p2p_events_) CUDA_TRY(cudaEventDestroy(event));
    for (auto pointer : p2p_states_) CUDA_TRY(cudaIpcCloseMemHandle(pointer));
    // Only now may exporting ranks release their local event/state allocation.
    MPI_TRY(MPI_Barrier(MPI_COMM_WORLD));
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
