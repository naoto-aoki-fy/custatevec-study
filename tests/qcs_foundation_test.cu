#include "../src/qcs_internal.hpp"
#include "device_assignment.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <algorithm>
#include <unistd.h>
#include <sys/stat.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" int qcs_c_caller_checks(qcs_simulator*);
static unsigned reports = 0;
static void on_error(const char* text, size_t size) { assert(size && size == std::strlen(text)); ++reports; }
static std::vector<cuDoubleComplex> snapshot(qcs_simulator* sim) {
    std::vector<cuDoubleComplex> values(sim->elements);
    assert(cudaSetDevice(sim->device) == cudaSuccess);
    assert(cudaMemcpy(values.data(), sim->state, values.size() * sizeof(values[0]), cudaMemcpyDeviceToHost) == cudaSuccess);
    return values;
}
int main(int argc, char** argv) {
    // Run default and --borrowed in separate launcher invocations: owned MPI
    // cannot be initialized again after the last handle is destroyed.
    bool borrowed = argc > 1 && std::string(argv[1]) == "--borrowed";
    if (borrowed) {
        int visible = 0; assert(cudaGetDeviceCount(&visible) == cudaSuccess);
        auto assignment = bench::device_assignment::discover(visible, [](const char* n) { return std::getenv(n); });
        assert(cudaSetDevice(assignment.device) == cudaSuccess);
        assert(cudaFree(nullptr) == cudaSuccess);
        assert(MPI_Init(&argc, &argv) == MPI_SUCCESS);
    }
    qcs_set_exception_callback(on_error);
    auto* sim = qcs_simulator_create_cxx();
    auto* other = qcs_simulator_create_cxx();
    qcs_simulator_set_num_qubits_cxx(other, std::max(1, other->global_bits));
    qcs_simulator_allocate_memory_cxx(other);
    qcs_simulator_set_zero_state_cxx(other);
    for (size_t i = 0; i < other->elements; ++i)
        assert(snapshot(other)[i].x == double(other->rank == 0 && i == 0));
    qcs_simulator* flat = nullptr;
    assert(qcs_simulator_create(&flat));
    qcs_simulator_set_num_qubits_cxx(flat, std::max(1, flat->global_bits));
    assert(qcs_simulator_allocate_memory(flat, QCS_INITIAL_STATE_FLAT));
    qcs_simulator_set_zero_state_cxx(flat);
    for (auto v : snapshot(flat)) assert(v.x == 1 && v.y == 0);
    qcs_simulator_destroy_cxx(flat);
    char original[4096]; assert(getcwd(original, sizeof(original)));
    char directory[64] = "qcs-foundation-XXXXXX";
    if (sim->rank == 0) assert(mkdtemp(directory));
    MPI_Bcast(directory, sizeof(directory), MPI_CHAR, 0, sim->comm);
    assert(chdir(directory) == 0);
    assert(qcs_c_caller_checks(sim));
    qcs_set_exception_callback([](const char*, size_t) { throw std::runtime_error("callback failure"); });
    assert(!qcs_simulator_set_num_clbits(sim, -1)); // No exception may cross C.
    qcs_set_exception_callback(on_error);
    // C getters intentionally narrow; C++ getter keeps the full value.
    qcs_simulator_set_num_clbits_cxx(sim, 257);
    bit_t narrow = 0; assert(qcs_simulator_get_num_clbits(sim, &narrow) && narrow == 1);
    assert(qcs_simulator_get_num_clbits_cxx(sim) == 257);
    qcs_simulator_set_num_qubits_cxx(sim, 8);
    bit_num_t reverse[8]; for (int p = 0; p < 8; ++p) reverse[p] = 7 - p;
    qcs_simulator_set_mapping_cxx(sim, reverse, 8);
    bit_num_t invalid[8] = {};
    assert(!qcs_simulator_set_mapping(sim, invalid, 8));
    auto invalid_method = static_cast<qcs_initial_state>(999);
    assert(!qcs_simulator_allocate_memory_with_initial_state(sim, invalid_method));
    // Allocation with LOAD succeeds with no file present: it must not read it.
    qcs_simulator_allocate_memory_cxx(sim, QCS_INITIAL_STATE_LOAD_STATEVECTOR);
    assert(!sim->initialized && sim->selected == QCS_INITIAL_STATE_LOAD_STATEVECTOR);
    for (int p = 0; p < 8; ++p) assert(sim->p2l[p] == 7 - p && sim->l2p[7-p] == p);
    assert(!qcs_simulator_allocate_memory(sim));
    assert(!qcs_simulator_set_num_qubits(sim, 9));
    qcs_simulator_set_flat_state_cxx(sim);
    for (auto v : snapshot(sim)) assert(v.x == 1 && v.y == 0);
    assert(sim->selected == QCS_INITIAL_STATE_LOAD_STATEVECTOR);
    qcs_simulator_set_sequential_state_cxx(sim);
    auto sequential = snapshot(sim);
    for (size_t i = 0; i < sequential.size(); ++i)
        assert(sequential[i].x == double(i + sim->elements * sim->rank) && sequential[i].y == 0);
    qcs_simulator_set_entangled_state_cxx(sim);
    auto entangled = snapshot(sim);
    for (size_t i = 0; i < entangled.size(); ++i) {
        bool endpoint = sim->rank == 0 ? i == 0 : sim->rank == sim->ranks-1 && i == entangled.size()-1;
        assert(entangled[i].x == double(endpoint) && entangled[i].y == 0);
    }
    sim->measurement_cursor = 17; sim->measured_mask = sim->pending_mask = 1;
    qcs_simulator_set_random_state_cxx(sim); auto random = snapshot(sim);
    assert(sim->measurement_cursor == 17 && !sim->measured_mask && !sim->pending_mask);
    qcs_simulator_set_random_state_cxx(sim); auto restored = snapshot(sim);
    assert(std::memcmp(random.data(), restored.data(), random.size() * sizeof(random[0])) == 0);
    // Shared test directory, world-rank physical slices, no implicit normalization.
    const char* file_name = "statevector_input.bin";
    if (sim->rank == 0) {
        auto* file = std::fopen(file_name, "wb"); assert(file);
        for (int i = 0; i < 256; ++i) { cuDoubleComplex v = make_cuDoubleComplex(i, -i); assert(std::fwrite(&v, sizeof(v), 1, file) == 1); }
        assert(std::fclose(file) == 0);
    }
    MPI_Barrier(sim->comm);
    qcs_simulator_set_zero_state_cxx(sim);
    auto loaded = snapshot(sim);
    for (size_t i = 0; i < loaded.size(); ++i) {
        double expected = double(i + sim->elements * sim->rank);
        assert(loaded[i].x == expected && loaded[i].y == -expected);
    }
    sim->clbits[2] = 1;
    std::vector<char> text(sim->num_clbits + 1);
    qcs_simulator_get_clbits_string_cxx(sim, text.data());
    assert(text[254] == '1' && text[257] == '\0');
    std::swap(sim->p2l[0], sim->p2l[1]);
    std::swap(sim->l2p[7], sim->l2p[6]);
    qcs_simulator_reinitialize_mapping_cxx(sim);
    qcs_simulator_set_zero_state_cxx(sim);
    assert(sim->clbits[2] == 1 && sim->p2l[0] == 7);
    MPI_Barrier(sim->comm);
    // Only rank zero loses access; the other ranks can still read their slices.
    if (sim->rank == 0) { assert(mkdir("missing", 0700) == 0); assert(chdir("missing") == 0); }
    assert(!qcs_simulator_set_zero_state(sim));
    if (sim->rank == 0) assert(chdir("..") == 0);
    qcs_simulator_set_zero_state_cxx(sim); // Retry after a pre-write file error.
    auto retried = snapshot(sim);
    assert(std::memcmp(loaded.data(), retried.data(), loaded.size() * sizeof(loaded[0])) == 0);
    MPI_Barrier(sim->comm);
    if (sim->rank == 0) { assert(std::remove(file_name) == 0); assert(rmdir("missing") == 0); }
    assert(chdir(original) == 0);
    MPI_Barrier(sim->comm);
    if (sim->rank == 0) assert(rmdir(directory) == 0);
    bool threw = false;
    try { qcs_simulator_event_create_cxx(sim); } catch (const std::runtime_error&) { threw = true; }
    assert(threw && reports >= 6);
    assert(qcs_simulator_dispose(sim)); assert(qcs_simulator_dispose(sim));
    assert(qcs_simulator_destroy(sim)); assert(qcs_simulator_destroy(nullptr));
    int finalized = 0; MPI_Finalized(&finalized); assert(!finalized); // other still owns a lease
    qcs_simulator_destroy_cxx(other);
    MPI_Finalized(&finalized); assert(finalized == !borrowed);
    if (borrowed) assert(MPI_Finalize() == MPI_SUCCESS);
}
