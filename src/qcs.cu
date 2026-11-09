#include "qcs_internal.hpp"
#include "device_assignment.hpp"
#include "transfer_policy.hpp"
#include "qcs_gate.hpp"
#include "qcs_measurement.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>

namespace {
// All library calls, including independent simulators and callback changes, are
// serialized by the caller. MPI collectives must use the same order on all ranks.
int leases = 0;
bool owns_mpi = false;
qcs_exception_callback callback = nullptr;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check_mpi(int status) {
    if (status == MPI_SUCCESS) return;
    // A failed transport cannot safely participate in an error agreement.
    MPI_Abort(MPI_COMM_WORLD, status);
    std::abort();
}
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check_sv(custatevecStatus_t status) {
    if (status != CUSTATEVEC_STATUS_SUCCESS)
        throw std::runtime_error(custatevecGetErrorString(status));
}
void check_curand(curandStatus_t status) {
    require(status == CURAND_STATUS_SUCCESS, "CURAND operation failed");
}
// Every local stage completes on every rank before the next communication.
// Fixed-size diagnostics avoid allocating before the status collective.
template<class F> void phase(MPI_Comm comm, F action) {
    char message[512] = {};
    int rank = 0, source = std::numeric_limits<int>::max();
    check_mpi(MPI_Comm_rank(comm, &rank));
    try { action(); }
    catch (const std::exception& e) { std::snprintf(message, sizeof(message), "%s", e.what()); source = rank; }
    catch (...) { std::snprintf(message, sizeof(message), "Unknown C++ exception"); source = rank; }
    int first = source;
    check_mpi(MPI_Allreduce(&source, &first, 1, MPI_INT, MPI_MIN, comm));
    if (first == std::numeric_limits<int>::max()) return;
    check_mpi(MPI_Bcast(message, sizeof(message), MPI_CHAR, first, comm));
    throw std::runtime_error(message);
}
void same(MPI_Comm comm, int value) {
    int lo = value, hi = value;
    check_mpi(MPI_Allreduce(&value, &lo, 1, MPI_INT, MPI_MIN, comm));
    check_mpi(MPI_Allreduce(&value, &hi, 1, MPI_INT, MPI_MAX, comm));
    require(lo == hi, "Collective arguments differ across ranks");
}
qcs_simulator& live(qcs_simulator* sim) {
    require(sim && !sim->disposed, "Simulator is null or disposed");
    return *sim;
}
const qcs_simulator& live(const qcs_simulator* sim) {
    return live(const_cast<qcs_simulator*>(sim));
}
struct DeviceScope {
    int previous = -1;
    explicit DeviceScope(int device) {
        check_cuda(cudaGetDevice(&previous));
        if (device != previous) check_cuda(cudaSetDevice(device));
    }
    ~DeviceScope() {
        if (previous >= 0 && cudaSetDevice(previous) != cudaSuccess)
            check_mpi(MPI_ERR_OTHER);
    }
};
void release_session() {
    if (--leases == 0 && owns_mpi) { owns_mpi = false; check_mpi(MPI_Finalize()); }
}
void mapping(qcs_simulator& s) {
    std::vector<int> p2l(s.num_qubits), l2p(s.num_qubits);
    for (int p = 0; p < s.num_qubits; ++p) {
        int q = s.initial_p2l.empty() ? p : s.initial_p2l[p];
        p2l[p] = q; l2p[q] = p;
    }
    s.p2l.swap(p2l); s.l2p.swap(l2p);
}
void discard_measurements(qcs_simulator& s) {
    s.measured_mask = s.measured_value = s.pending_mask = 0;
}
__global__ void fill_state(cuDoubleComplex* state, std::uint64_t count,
                           int rank, int ranks, int method) {
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::uint64_t(blockDim.x) * gridDim.x) {
        double value = 0;
        if (method == QCS_INITIAL_STATE_FLAT) value = 1;
        else if (method == QCS_INITIAL_STATE_SEQUENTIAL) value = double(i + count * std::uint64_t(rank));
        else if (rank == 0) value = i == 0 ? 1 : 0;
        else if (method == QCS_INITIAL_STATE_ENTANGLED && rank == ranks - 1)
            value = i == count - 1 ? 1 : 0;
        state[i] = make_cuDoubleComplex(value, 0);
    }
}
void initialize(qcs_simulator* sim, qcs_initial_state method) {
    auto& s = live(sim);
    same(s.comm, int(method));
    phase(s.comm, [&] {
        require(s.state && !s.poisoned, "State is not allocated or is poisoned");
        require(method >= QCS_INITIAL_STATE_ZERO && method <= QCS_INITIAL_STATE_LOAD_STATEVECTOR,
                "Invalid initial state");
    });
    bool mutation_started = false;
    try {
        // Complete file reads on all ranks before changing device state.
        std::vector<cuDoubleComplex> input;
        phase(s.comm, [&] {
            if (method != QCS_INITIAL_STATE_LOAD_STATEVECTOR) return;
            input.resize(s.elements);
            std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen("statevector_input.bin", "rb"), &std::fclose);
            require(bool(file), "Cannot open statevector_input.bin");
            const auto bytes = s.elements * sizeof(cuDoubleComplex);
            require(std::uint64_t(s.rank) <= std::uint64_t(std::numeric_limits<off_t>::max()) / bytes,
                    "Statevector file offset overflow");
            require(fseeko(file.get(), off_t(std::uint64_t(s.rank) * bytes), SEEK_SET) == 0,
                    "Statevector seek failed");
            require(std::fread(input.data(), sizeof(cuDoubleComplex), s.elements, file.get()) == s.elements,
                    "Statevector short read");
        });
        mutation_started = true;
        phase(s.comm, [&] {
            DeviceScope device(s.device);
            if (method == QCS_INITIAL_STATE_LOAD_STATEVECTOR) {
                check_cuda(cudaMemcpyAsync(s.state, input.data(), s.elements * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, s.stream));
            } else if (method == QCS_INITIAL_STATE_USE_CURAND) {
                curandGenerator_t generator = nullptr;
                check_curand(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
                try {
                    check_curand(curandSetStream(generator, s.stream));
                    check_curand(curandSetPseudoRandomGeneratorSeed(generator, static_cast<unsigned int>(s.seed + std::uint64_t(s.rank))));
                    check_curand(curandGenerateNormalDouble(generator, reinterpret_cast<double*>(s.state), s.elements * 2, 0.0, 1.0));
                    check_cuda(cudaStreamSynchronize(s.stream));
                } catch (...) { curandDestroyGenerator(generator); throw; }
                check_curand(curandDestroyGenerator(generator));
            } else {
                unsigned blocks = unsigned(std::min<std::uint64_t>((s.elements + 255) / 256, 65535));
                fill_state<<<blocks, 256, 0, s.stream>>>(s.state, s.elements, s.rank, s.ranks, int(method));
                check_cuda(cudaGetLastError());
            }
            check_cuda(cudaStreamSynchronize(s.stream));
        });
        discard_measurements(s); s.initialized = true;
    } catch (...) { if (mutation_started) { s.poisoned = true; s.initialized = false; } throw; }
}
#include "qcs_gates.cuh"
#include "qcs_measurement.cuh"
}

qcs_simulator* qcs_simulator_create_cxx() {
    int started = 0, finished = 0;
    check_mpi(MPI_Initialized(&started)); check_mpi(MPI_Finalized(&finished));
    require(!finished, "MPI has already been finalized");
    bench::DeviceAssignment assignment;
    char early_error[512] = {};
    int previous = -1;
    // Even an early CUDA failure joins initialization and status agreement, so
    // another rank cannot be left in MPI_Init or later setup collectives.
    try {
        check_cuda(cudaGetDevice(&previous));
        int visible = 0; check_cuda(cudaGetDeviceCount(&visible));
        assignment = bench::device_assignment::discover(visible, [](const char* name) { return std::getenv(name); });
        check_cuda(cudaSetDevice(assignment.device)); check_cuda(cudaFree(nullptr));
    } catch (const std::exception& e) { std::snprintf(early_error, sizeof(early_error), "%s", e.what()); }
    if (!started) { check_mpi(MPI_Init(nullptr, nullptr)); owns_mpi = true; }
    ++leases;
    std::unique_ptr<qcs_simulator> sim;
    MPI_Comm shared = MPI_COMM_NULL;
    try {
        phase(MPI_COMM_WORLD, [&] {
            require(!*early_error, early_error);
            sim.reset(new qcs_simulator);
        });
        auto& s = *sim;
        s.lease = true; s.device = assignment.device;
        check_mpi(MPI_Comm_dup(MPI_COMM_WORLD, &s.comm));
        check_mpi(MPI_Comm_set_errhandler(s.comm, MPI_ERRORS_RETURN));
        check_mpi(MPI_Comm_rank(s.comm, &s.rank)); check_mpi(MPI_Comm_size(s.comm, &s.ranks));
        phase(s.comm, [&] { require((s.ranks & (s.ranks - 1)) == 0, "Rank count must be a power of two"); });
        for (int p = s.ranks; p > 1; p >>= 1) ++s.global_bits;
        check_mpi(MPI_Comm_split_type(s.comm, MPI_COMM_TYPE_SHARED, s.rank, MPI_INFO_NULL, &shared));
        int local_rank = 0, local_size = 0;
        check_mpi(MPI_Comm_rank(shared, &local_rank)); check_mpi(MPI_Comm_size(shared, &local_size));
        std::vector<cudaUUID_t> ids;
        cudaUUID_t id{};
        phase(s.comm, [&] {
            bench::device_assignment::validate(assignment, local_rank);
            check_cuda(cudaSetDevice(s.device));
            cudaDeviceProp props{}; check_cuda(cudaGetDeviceProperties(&props, s.device)); id = props.uuid;
            ids.resize(local_size);
        });
        check_mpi(MPI_Allgather(&id, sizeof(id), MPI_BYTE, ids.data(), sizeof(id), MPI_BYTE, shared));
        phase(s.comm, [&] {
            for (int a = 0; a < local_size; ++a)
                for (int b = a + 1; b < local_size; ++b)
                    require(std::memcmp(&ids[a], &ids[b], sizeof(id)) != 0, "Duplicate GPU UUID on node");
            check_cuda(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
            check_sv(custatevecCreate(&s.handle)); check_sv(custatevecSetStream(s.handle, s.stream));
            if (s.rank == 0) s.seed = std::random_device{}();
        });
        check_mpi(MPI_Bcast(&s.seed, 1, MPI_UINT64_T, 0, s.comm));
        check_mpi(MPI_Comm_free(&shared));
        phase(s.comm, [&] { if (previous >= 0) check_cuda(cudaSetDevice(previous)); });
        return sim.release();
    } catch (...) {
        if (sim) {
            if (sim->handle) custatevecDestroy(sim->handle);
            if (sim->stream) cudaStreamDestroy(sim->stream);
        }
        if (shared != MPI_COMM_NULL) check_mpi(MPI_Comm_free(&shared));
        if (sim && sim->comm != MPI_COMM_NULL) check_mpi(MPI_Comm_free(&sim->comm));
        release_session();
        if (previous >= 0) cudaSetDevice(previous);
        throw;
    }
}
void qcs_simulator_dispose_cxx(qcs_simulator* sim) {
    require(sim != nullptr, "Null simulator");
    if (sim->disposed) return;
    auto& s = *sim;
    std::exception_ptr error;
    try {
        phase(s.comm, [&] { DeviceScope device(s.device); check_cuda(cudaStreamSynchronize(s.stream));
            if (s.transfer_stream) check_cuda(cudaStreamSynchronize(s.transfer_stream)); });
    } catch (...) { error = std::current_exception(); }
    check_mpi(MPI_Barrier(s.comm));
    close_swap(s);
    // Destruction failures cannot unwind before the other ranks release leases.
    try {
        phase(s.comm, [&] {
            DeviceScope device(s.device);
            cudaError_t cu_status = cudaSuccess;
            for (auto event : s.events) { auto result = cudaEventDestroy(event); if (result != cudaSuccess) cu_status = result; }
            s.events.clear();
            auto sv_status = s.handle ? custatevecDestroy(s.handle) : CUSTATEVEC_STATUS_SUCCESS; s.handle = nullptr;
            auto measurement_status=s.measurement_draw ? cudaFree(s.measurement_draw) : cudaSuccess; s.measurement_draw=nullptr;
            auto gate_status = s.gate_workspace ? cudaFree(s.gate_workspace) : cudaSuccess; s.gate_workspace=nullptr;
            auto state_status = s.state ? cudaFree(s.state) : cudaSuccess; s.state = nullptr;
            auto workspace_status = s.workspace ? cudaFree(s.workspace) : cudaSuccess; s.workspace = nullptr;
            auto stream_status = s.stream ? cudaStreamDestroy(s.stream) : cudaSuccess; s.stream = nullptr;
            check_cuda(measurement_status); check_cuda(gate_status); check_cuda(cu_status); check_cuda(state_status); check_cuda(workspace_status); check_sv(sv_status); check_cuda(stream_status);
        });
    } catch (...) { if (!error) error = std::current_exception(); }
    check_mpi(MPI_Comm_free(&s.comm)); s.disposed = true; s.initialized = false;
    if (s.lease) { s.lease = false; release_session(); }
    if (error) std::rethrow_exception(error);
}
void qcs_simulator_destroy_cxx(qcs_simulator* sim) {
    if (!sim) return;
    std::unique_ptr<qcs_simulator> owner(sim);
    qcs_simulator_dispose_cxx(sim);
}
int qcs_simulator_get_num_procs_cxx(qcs_simulator* s) { return live(s).ranks; }
int qcs_simulator_get_proc_num_cxx(qcs_simulator* s) { return live(s).rank; }
int qcs_simulator_get_num_qubits_cxx(const qcs_simulator* s) { return live(s).num_qubits; }
int qcs_simulator_get_num_clbits_cxx(const qcs_simulator* s) { return live(s).num_clbits; }
void qcs_simulator_set_num_qubits_cxx(qcs_simulator* sim, bit_num_t n) {
    auto& s = live(sim); require(!s.state, "Cannot resize allocated state");
    require(n >= 1 && n <= 63 && n >= s.global_bits, "Require 1<=N<=63 and N>=log2(P)");
    require(s.initial_p2l.empty() || int(s.initial_p2l.size()) == n, "Configured mapping size differs");
    s.num_qubits = n;
}
void qcs_simulator_set_num_clbits_cxx(qcs_simulator* sim, bit_num_t n) {
    auto& s = live(sim); require(n >= 0, "Negative clbit count");
    s.clbits.resize(n, 0); s.num_clbits = n;
}
void qcs_simulator_set_mapping_cxx(qcs_simulator* sim, const bit_num_t* perm, bit_num_t n) {
    auto& s = live(sim); require(n == s.num_qubits && n > 0 && perm, "Invalid mapping size/pointer");
    std::vector<int> next(n); std::vector<bool> used(n, false);
    for (int p = 0; p < n; ++p) {
        require(perm[p] >= 0 && perm[p] < n && !used[perm[p]], "Mapping is not a permutation");
        next[p] = perm[p]; used[perm[p]] = true;
    }
    s.initial_p2l = std::move(next);
}
void qcs_simulator_reinitialize_mapping_cxx(qcs_simulator* sim) {
    auto& s = live(sim);
    same(s.comm, s.num_qubits);
    phase(s.comm, [&] { require(s.state && !s.poisoned, "State not allocated or poisoned"); });
    for (int p = 0; p < s.num_qubits; ++p)
        same(s.comm, s.initial_p2l.empty() ? p : s.initial_p2l[p]);
    phase(s.comm, [&] { mapping(s); });
}
void qcs_simulator_allocate_memory_cxx(qcs_simulator* sim) { qcs_simulator_allocate_memory_cxx(sim, QCS_INITIAL_STATE_ZERO); }
void qcs_simulator_allocate_memory_cxx(qcs_simulator* sim, qcs_initial_state selected) {
    auto& s = live(sim);
    same(s.comm, s.num_qubits); same(s.comm, int(selected));
    phase(s.comm, [&] {
        require(!s.state && !s.poisoned, "State already allocated or poisoned");
        require(s.num_qubits >= 1 && s.num_qubits <= 63 && s.num_qubits >= s.global_bits, "Invalid qubit count");
        require(selected >= QCS_INITIAL_STATE_ZERO && selected <= QCS_INITIAL_STATE_LOAD_STATEVECTOR, "Invalid initial state");
        s.elements = UINT64_C(1) << (s.num_qubits - s.global_bits);
        require(s.elements <= std::numeric_limits<std::size_t>::max() / sizeof(cuDoubleComplex), "State size overflow");
        mapping(s);
    });
    for (int p = 0; p < s.num_qubits; ++p) same(s.comm, s.p2l[p]);
    cuDoubleComplex* candidate = nullptr;
    void* scratch = nullptr;
    std::size_t scratch_bytes = 0;
    try {
        phase(s.comm, [&] {
            DeviceScope device(s.device);
            check_sv(custatevecGetDefaultWorkspaceSize(s.handle, &scratch_bytes));
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&candidate), s.elements * sizeof(cuDoubleComplex)));
            if (scratch_bytes) check_cuda(cudaMalloc(&scratch, scratch_bytes));
            check_sv(custatevecSetWorkspace(s.handle, scratch, scratch_bytes));
        });
    } catch (...) {
        try { phase(s.comm, [&] {
            DeviceScope device(s.device);
            // A rollback error makes the distributed session unsafe.
            if (custatevecSetWorkspace(s.handle, nullptr, 0) != CUSTATEVEC_STATUS_SUCCESS) check_mpi(MPI_ERR_OTHER);
            if (scratch && cudaFree(scratch) != cudaSuccess) check_mpi(MPI_ERR_OTHER);
            if (candidate && cudaFree(candidate) != cudaSuccess) check_mpi(MPI_ERR_OTHER);
        }); } catch (...) { check_mpi(MPI_ERR_OTHER); }
        throw;
    }
    s.state = candidate; s.workspace = scratch; s.workspace_bytes = scratch_bytes;
    s.selected = selected; s.initialized = false; discard_measurements(s);
}
void qcs_simulator_set_zero_state_cxx(qcs_simulator* sim) { initialize(sim, live(sim).selected); }
void qcs_simulator_set_flat_state_cxx(qcs_simulator* sim) { initialize(sim, QCS_INITIAL_STATE_FLAT); }
void qcs_simulator_set_sequential_state_cxx(qcs_simulator* sim) { initialize(sim, QCS_INITIAL_STATE_SEQUENTIAL); }
void qcs_simulator_set_entangled_state_cxx(qcs_simulator* sim) { initialize(sim, QCS_INITIAL_STATE_ENTANGLED); }
void qcs_simulator_set_random_state_cxx(qcs_simulator* sim) { initialize(sim, QCS_INITIAL_STATE_USE_CURAND); }
void qcs_simulator_get_clbits_cxx(const qcs_simulator* sim, bit_t* result) {
    auto& s = live(sim); require(result || s.num_clbits == 0, "Null clbits output");
    if (!s.clbits.empty()) std::copy(s.clbits.begin(), s.clbits.end(), result);
}
void qcs_simulator_get_clbits_string_cxx(const qcs_simulator* sim, char* result) {
    auto& s = live(sim); require(result, "Null string output");
    for (int i = 0; i < s.num_clbits; ++i) result[i] = s.clbits[s.num_clbits - 1 - i] ? '1' : '0';
    result[s.num_clbits] = '\0';
}
int qcs_simulator_read_cxx(qcs_simulator* sim, bit_num_t index) {
    auto& s = live(sim); require(index >= 0 && index < s.num_clbits, "Clbit index out of range"); return s.clbits[index];
}
void qcs_simulator_reset_clbits_cxx(qcs_simulator* sim) { auto& s = live(sim); std::fill(s.clbits.begin(), s.clbits.end(), 0); }

// Public throwing APIs share checked engines; remaining placeholders are explicit.
void qcs_simulator_reset_cxx(qcs_simulator *sim, bit_num_t qubit_num) { if (qcs_simulator_measure_cxx(sim, qubit_num)) qcs_simulator_gate_x_cxx(sim, &qubit_num, 1, nullptr, 0, nullptr, 0); }
void qcs_simulator_reset_measurement_state_cxx(qcs_simulator *sim) { clear_measurement(sim); }
void qcs_simulator_gate_global_phase_cxx(qcs_simulator *sim, double theta, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::global_phase, {theta}, nullptr, 0, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_h_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::h, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_x_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::x, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_y_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::y, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_z_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::z, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_s_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::s, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_sdg_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::sdg, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_t_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::t, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_tdg_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::tdg, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_sx_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::sx, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_rx_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::rx, {theta}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_ry_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::ry, {theta}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_rz_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::rz, {theta}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_u4_cxx(qcs_simulator *sim, double theta, double phi, double lambda, double gamma, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::u4, {theta, phi, lambda, gamma}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_u3_cxx(qcs_simulator *sim, double theta, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::u3, {theta, phi, lambda}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_u2_cxx(qcs_simulator *sim, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::u2, {phi, lambda}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_u1_cxx(qcs_simulator *sim, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::u1, {lambda}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_u_cxx(qcs_simulator *sim, double theta, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::u, {theta, phi, lambda}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_p_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::p, {theta}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_swap_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::swap, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_iswap_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::iswap, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_id_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::id, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_sxdg_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::sxdg, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_r_cxx(qcs_simulator *sim, double theta, double phi, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::r, {theta, phi}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_rxx_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::rxx, {theta}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_ryy_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::ryy, {theta}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_rzz_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::rzz, {theta}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_rzx_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::rzx, {theta}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_dcx_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::dcx, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_ecr_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::ecr, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_xx_plus_yy_cxx(qcs_simulator *sim, double theta, double beta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::xx_plus_yy, {theta, beta}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_xx_minus_yy_cxx(qcs_simulator *sim, double theta, double beta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::xx_minus_yy, {theta, beta}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_rccx_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::rccx, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
void qcs_simulator_gate_rcccx_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { operate_gate(sim, qcs_gate::Kind::rcccx, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }
double qcs_simulator_expectation_pauli_cxx(qcs_simulator *sim, const bit_num_t *qubit_num_list, const qcs_pauli_operator *operators, bit_num_t qubit_num_count, double coefficient) { throw std::runtime_error("qcs_simulator_expectation_pauli_cxx: scheduled for a later compatibility stage"); }
int qcs_simulator_measure_cxx(qcs_simulator *sim, bit_num_t qubit_num) { bit_t result=0; measure_request(sim, &qubit_num, 1, nullptr, 0, &result, false); return result; }
int qcs_simulator_measure_to_clbit_cxx(qcs_simulator *sim, bit_num_t qubit_num, bit_num_t clbit_num) { bit_t result=0; measure_request(sim, &qubit_num, 1, &clbit_num, 1, &result, true); return result; }
void qcs_simulator_measure_many_cxx(qcs_simulator *sim, const bit_num_t *qubit_num_list, bit_num_t qubit_num_count, bit_t *results) { measure_request(sim, qubit_num_list, qubit_num_count, nullptr, 0, results, false); }
void qcs_simulator_measure_many_to_clbits_cxx(qcs_simulator *sim, const bit_num_t *qubit_num_list, bit_num_t qubit_num_count, const bit_num_t *clbit_num_list, bit_num_t clbit_num_count, bit_t *results) { measure_request(sim, qubit_num_list, qubit_num_count, clbit_num_list, clbit_num_count, results, true); }
void qcs_simulator_save_statevector_cxx(qcs_simulator *sim, const char *outfn) { throw std::runtime_error("qcs_simulator_save_statevector_cxx: scheduled for a later compatibility stage"); }
int qcs_simulator_event_create_cxx(qcs_simulator *sim) { throw std::runtime_error("qcs_simulator_event_create_cxx: scheduled for a later compatibility stage"); }
void qcs_simulator_event_record_cxx(qcs_simulator *sim, int event_num) { throw std::runtime_error("qcs_simulator_event_record_cxx: scheduled for a later compatibility stage"); }
double qcs_simulator_event_get_elapsed_time_cxx(qcs_simulator *sim, event_num_t start_event_num, event_num_t stop_event_num) { throw std::runtime_error("qcs_simulator_event_get_elapsed_time_cxx: scheduled for a later compatibility stage"); }

namespace {
void report(const char* message) noexcept {
    if (!callback) return;
    try { callback(message, std::strlen(message)); } catch (...) { /* C boundary. */ }
}
template<class F> int boundary(F action) noexcept {
    try { action(); return 1; }
    catch (const std::exception& e) { report(e.what()); }
    catch (...) { report("Unknown C++ exception"); }
    return 0;
}
}
extern "C" int qcs_set_exception_callback(qcs_exception_callback value) { callback = value; return 1; }
extern "C" int qcs_simulator_create(qcs_simulator **result) { return boundary([&] { auto* created = qcs_simulator_create_cxx(); try { phase(created->comm, [&] { require(result, "Null result pointer"); }); } catch (...) { qcs_simulator_destroy_cxx(created); throw; } *result = created; }); }
extern "C" int qcs_simulator_destroy(qcs_simulator *sim) { return boundary([&] { qcs_simulator_destroy_cxx(sim); }); }
extern "C" int qcs_simulator_allocate_memory(qcs_simulator *sim) { return boundary([&] { qcs_simulator_allocate_memory_cxx(sim); }); }
extern "C" int qcs_simulator_allocate_memory_with_initial_state(qcs_simulator *sim, qcs_initial_state initial_state) { return boundary([&] { qcs_simulator_allocate_memory_cxx(sim, initial_state); }); }
extern "C" int qcs_simulator_dispose(qcs_simulator *sim) { return boundary([&] { qcs_simulator_dispose_cxx(sim); }); }
extern "C" int qcs_simulator_get_num_procs(qcs_simulator *sim, bit_t *result) { return boundary([&] { require(result, "Null result pointer"); *result = static_cast<std::remove_reference_t<decltype(*result)>>(qcs_simulator_get_num_procs_cxx(sim)); }); }
extern "C" int qcs_simulator_get_proc_num(qcs_simulator *sim, bit_t *result) { return boundary([&] { require(result, "Null result pointer"); *result = static_cast<std::remove_reference_t<decltype(*result)>>(qcs_simulator_get_proc_num_cxx(sim)); }); }
extern "C" int qcs_simulator_get_num_qubits(const qcs_simulator *sim, bit_t *result) { return boundary([&] { require(result, "Null result pointer"); *result = static_cast<std::remove_reference_t<decltype(*result)>>(qcs_simulator_get_num_qubits_cxx(sim)); }); }
extern "C" int qcs_simulator_get_num_clbits(const qcs_simulator *sim, bit_t *result) { return boundary([&] { require(result, "Null result pointer"); *result = static_cast<std::remove_reference_t<decltype(*result)>>(qcs_simulator_get_num_clbits_cxx(sim)); }); }
extern "C" int qcs_simulator_set_num_qubits(qcs_simulator *sim, bit_num_t num_qubits) { return boundary([&] { qcs_simulator_set_num_qubits_cxx(sim, num_qubits); }); }
extern "C" int qcs_simulator_set_mapping(qcs_simulator *sim, const bit_num_t *perm_p2l, bit_num_t perm_p2l_count) { return boundary([&] { qcs_simulator_set_mapping_cxx(sim, perm_p2l, perm_p2l_count); }); }
extern "C" int qcs_simulator_set_num_clbits(qcs_simulator *sim, bit_num_t num_clbits) { return boundary([&] { qcs_simulator_set_num_clbits_cxx(sim, num_clbits); }); }
extern "C" int qcs_simulator_get_clbits(const qcs_simulator *sim, bit_t *clbits) { return boundary([&] { qcs_simulator_get_clbits_cxx(sim, clbits); }); }
extern "C" int qcs_simulator_get_clbits_string(const qcs_simulator *sim, char *clbits_string) { return boundary([&] { qcs_simulator_get_clbits_string_cxx(sim, clbits_string); }); }
extern "C" int qcs_simulator_reset(qcs_simulator *sim, bit_num_t qubit_num) { return boundary([&] { qcs_simulator_reset_cxx(sim, qubit_num); }); }
extern "C" int qcs_simulator_set_zero_state(qcs_simulator *sim) { return boundary([&] { qcs_simulator_set_zero_state_cxx(sim); }); }
extern "C" int qcs_simulator_set_sequential_state(qcs_simulator *sim) { return boundary([&] { qcs_simulator_set_sequential_state_cxx(sim); }); }
extern "C" int qcs_simulator_set_flat_state(qcs_simulator *sim) { return boundary([&] { qcs_simulator_set_flat_state_cxx(sim); }); }
extern "C" int qcs_simulator_set_entangled_state(qcs_simulator *sim) { return boundary([&] { qcs_simulator_set_entangled_state_cxx(sim); }); }
extern "C" int qcs_simulator_set_random_state(qcs_simulator *sim) { return boundary([&] { qcs_simulator_set_random_state_cxx(sim); }); }
extern "C" int qcs_simulator_reset_clbits(qcs_simulator *sim) { return boundary([&] { qcs_simulator_reset_clbits_cxx(sim); }); }
extern "C" int qcs_simulator_reset_measurement_state(qcs_simulator *sim) { return boundary([&] { qcs_simulator_reset_measurement_state_cxx(sim); }); }
extern "C" int qcs_simulator_reinitialize_mapping(qcs_simulator *sim) { return boundary([&] { qcs_simulator_reinitialize_mapping_cxx(sim); }); }
extern "C" int qcs_simulator_gate_global_phase(qcs_simulator *sim, double theta, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_global_phase_cxx(sim, theta, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_h(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_h_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_x(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_x_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_y(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_y_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_z(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_z_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_s(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_s_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_sdg(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_sdg_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_t(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_t_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_tdg(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_tdg_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_sx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_sx_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_rx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_rx_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_ry(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_ry_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_rz(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_rz_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_u4(qcs_simulator *sim, double theta, double phi, double lambda, double gamma, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_u4_cxx(sim, theta, phi, lambda, gamma, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_u3(qcs_simulator *sim, double theta, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_u3_cxx(sim, theta, phi, lambda, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_u2(qcs_simulator *sim, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_u2_cxx(sim, phi, lambda, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_u1(qcs_simulator *sim, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_u1_cxx(sim, lambda, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_u(qcs_simulator *sim, double theta, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_u_cxx(sim, theta, phi, lambda, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_p(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_p_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_swap(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_swap_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_swap1(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { operate_gate(sim, qcs_gate::Kind::swap, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count, 2); }); }
extern "C" int qcs_simulator_gate_swap2(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { operate_gate(sim, qcs_gate::Kind::swap, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count, 4); }); }
extern "C" int qcs_simulator_gate_swap3(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { operate_gate(sim, qcs_gate::Kind::swap, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count, 6); }); }
extern "C" int qcs_simulator_gate_swap4(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { operate_gate(sim, qcs_gate::Kind::swap, {}, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count, 8); }); }
extern "C" int qcs_simulator_gate_iswap(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_iswap_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_id(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_id_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_sxdg(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_sxdg_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_r(qcs_simulator *sim, double theta, double phi, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_r_cxx(sim, theta, phi, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_rxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_rxx_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_ryy(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_ryy_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_rzz(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_rzz_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_rzx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_rzx_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_dcx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_dcx_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_ecr(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_ecr_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_xx_plus_yy(qcs_simulator *sim, double theta, double beta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_xx_plus_yy_cxx(sim, theta, beta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_xx_minus_yy(qcs_simulator *sim, double theta, double beta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_xx_minus_yy_cxx(sim, theta, beta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_rccx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_rccx_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_gate_rcccx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count) { return boundary([&] { qcs_simulator_gate_rcccx_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); }); }
extern "C" int qcs_simulator_expectation_pauli(qcs_simulator *sim, const bit_num_t *qubit_num_list, const qcs_pauli_operator *operators, bit_num_t qubit_num_count, double coefficient, double *result) { return boundary([&] { require(result, "Null result pointer"); *result = static_cast<std::remove_reference_t<decltype(*result)>>(qcs_simulator_expectation_pauli_cxx(sim, qubit_num_list, operators, qubit_num_count, coefficient)); }); }
extern "C" int qcs_simulator_measure(qcs_simulator *sim, bit_num_t qubit_num, bit_t *result) { return boundary([&] { measure_request(sim, &qubit_num, 1, nullptr, 0, result, false); }); }
extern "C" int qcs_simulator_measure_to_clbit(qcs_simulator *sim, bit_num_t qubit_num, bit_num_t clbit_num, bit_t *result) { return boundary([&] { measure_request(sim, &qubit_num, 1, &clbit_num, 1, result, true); }); }
extern "C" int qcs_simulator_measure_many(qcs_simulator *sim, const bit_num_t *qubit_num_list, bit_num_t qubit_num_count, bit_t *results) { return boundary([&] { qcs_simulator_measure_many_cxx(sim, qubit_num_list, qubit_num_count, results); }); }
extern "C" int qcs_simulator_measure_many_to_clbits(qcs_simulator *sim, const bit_num_t *qubit_num_list, bit_num_t qubit_num_count, const bit_num_t *clbit_num_list, bit_num_t clbit_num_count, bit_t *results) { return boundary([&] { qcs_simulator_measure_many_to_clbits_cxx(sim, qubit_num_list, qubit_num_count, clbit_num_list, clbit_num_count, results); }); }
extern "C" int qcs_simulator_read(qcs_simulator *sim, bit_num_t clbit_num, bit_t *result) { return boundary([&] { require(result, "Null result pointer"); *result = static_cast<std::remove_reference_t<decltype(*result)>>(qcs_simulator_read_cxx(sim, clbit_num)); }); }
extern "C" int qcs_simulator_save_statevector(qcs_simulator *sim, const char *outfn) { return boundary([&] { qcs_simulator_save_statevector_cxx(sim, outfn); }); }
extern "C" int qcs_simulator_event_create(qcs_simulator *sim, bit_t *result) { return boundary([&] { require(result, "Null result pointer"); *result = static_cast<std::remove_reference_t<decltype(*result)>>(qcs_simulator_event_create_cxx(sim)); }); }
extern "C" int qcs_simulator_event_record(qcs_simulator *sim, int event_num) { return boundary([&] { qcs_simulator_event_record_cxx(sim, event_num); }); }
extern "C" int qcs_simulator_event_get_elapsed_time(qcs_simulator *sim, event_num_t start_event_num, event_num_t stop_event_num, double *result) { return boundary([&] { require(result, "Null result pointer"); *result = static_cast<std::remove_reference_t<decltype(*result)>>(qcs_simulator_event_get_elapsed_time_cxx(sim, start_event_num, stop_event_num)); }); }
