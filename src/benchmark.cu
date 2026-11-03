#include "bench.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace bench {
namespace {
// Row-major complex128 H = 1/sqrt(2) * [[1, 1], [1, -1]].
const double h_scale = 1.0 / std::sqrt(2.0);
const cuDoubleComplex h_matrix[4] = {
    {h_scale, 0.0}, {h_scale, 0.0}, {h_scale, 0.0}, {-h_scale, 0.0}
};

void apply_hadamard(Runtime& runtime, int logical) {
    int physical = runtime.logical_to_physical[logical];
    if (physical >= runtime.topology.local_bits) {
        // Deterministic victim: physical bit 0, even if its logical gate has
        // already executed. The swap updates both logical owners' positions.
        runtime.swap_local_global(0, physical);
        physical = runtime.logical_to_physical[logical];
    }
    const int32_t target = physical;
    runtime.apply_matrix(h_matrix, &target, 1);
}

class HadamardJob final : public Job {
public:
    void prepare(Runtime& runtime) override {
        runtime.reserve_gate_workspace(h_matrix, 1);
        if (runtime.topology.rank == 0)
            std::cerr << "hadamard_gate_calls_per_repetition=" << runtime.nqubits
                      << " localization=swap-global-target-with-physical-bit-0"
                      << " final_mapping=retained; no restoration\n";
    }
    void initialize(Runtime& runtime, std::uint64_t, std::uint64_t) override {
        runtime.initialize_zero();
    }
    JobResult execute(Runtime& runtime, std::uint64_t, std::uint64_t) override {
        for (int q = 0; q < runtime.nqubits; ++q) apply_hadamard(runtime, q);
        return {};
    }
};

// These host-side helpers are used only by the opt-in, untimed small-N check.
void cpu_hadamard(std::vector<cuDoubleComplex>& state, int logical) {
    const std::size_t stride = std::size_t{1} << logical;
    for (std::size_t base = 0; base < state.size(); base += 2 * stride) {
        for (std::size_t j = 0; j < stride; ++j) {
            const auto a = state[base + j];
            const auto b = state[base + j + stride];
            state[base + j] = make_cuDoubleComplex((a.x + b.x) * h_scale, (a.y + b.y) * h_scale);
            state[base + j + stride] = make_cuDoubleComplex((a.x - b.x) * h_scale, (a.y - b.y) * h_scale);
        }
    }
}

void compare_logical_state(Runtime& runtime, const std::vector<cuDoubleComplex>& reference,
                           const std::string& label) {
    runtime.synchronize();
    std::vector<cuDoubleComplex> actual(static_cast<std::size_t>(runtime.local_elements));
    CUDA_TRY(cudaMemcpy(actual.data(), runtime.state, runtime.state_bytes, cudaMemcpyDeviceToHost));
    double local_error = 0.0, local_norm = 0.0;
    int invalid = 0;
    const auto base = static_cast<std::uint64_t>(runtime.topology.rank) * runtime.local_elements;
    for (std::uint64_t i = 0; i < runtime.local_elements; ++i) {
        const auto physical = base + i;
        std::uint64_t logical = 0;
        for (int q = 0; q < runtime.nqubits; ++q)
            logical |= ((physical >> runtime.logical_to_physical[q]) & 1) << q;
        const auto a = actual[i];
        const auto b = reference[logical];
        if (!std::isfinite(a.x) || !std::isfinite(a.y)) invalid = 1;
        local_error = std::max(local_error, std::hypot(a.x - b.x, a.y - b.y));
        local_norm += a.x * a.x + a.y * a.y;
    }
    double error = 0.0, norm = 0.0;
    int any_invalid = 0;
    MPI_TRY(MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    MPI_TRY(MPI_Allreduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    MPI_TRY(MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    if (any_invalid || !std::isfinite(norm) || error > 1e-11 || std::abs(norm - 1.0) > 1e-11)
        fail("Hadamard check failed: " + label + ", max_error=" + std::to_string(error) +
             ", norm=" + std::to_string(norm));
    if (runtime.topology.rank == 0)
        std::cerr << "Hadamard check passed: " << label << " max_error=" << error << " norm=" << norm << '\n';
}
}

std::unique_ptr<Job> make_job(const std::string& name) {
    if (name == "hadamard") return std::make_unique<HadamardJob>();
    throw std::runtime_error("Job '" + name + "' is not implemented; no timings produced");
}

void check_hadamard(Runtime& runtime) {
    if (runtime.nqubits > 20) fail("Hadamard check requires at most 20 total qubits");
    HadamardJob job;
    job.prepare(runtime);
    const auto count = std::size_t{1} << runtime.nqubits;
    std::vector<cuDoubleComplex> reference(count);

    // Repeat the uniform case to exercise reinitialization after a changed map.
    for (int repetition = 0; repetition < 2; ++repetition) {
        runtime.reset_mapping();
        job.initialize(runtime, 0, repetition);
        runtime.synchronize();
        job.execute(runtime, 0, repetition);
        const double uniform = std::exp2(-0.5 * runtime.nqubits);
        std::fill(reference.begin(), reference.end(), make_cuDoubleComplex(uniform, 0.0));
        compare_logical_state(runtime, reference, "all H on |0>, repetition " + std::to_string(repetition + 1));
    }

    // Unequal, signed complex amplitudes whose values depend on the full logical
    // index expose permutations that the uniform state cannot distinguish.
    std::vector<cuDoubleComplex> input(count);
    double norm = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double index = static_cast<double>(i + 1);
        input[i] = make_cuDoubleComplex(std::sin(0.37 * index) + 0.2 * std::cos(0.11 * index),
                                       std::cos(0.23 * index) - 0.3 * std::sin(0.07 * index));
        norm += input[i].x * input[i].x + input[i].y * input[i].y;
    }
    const double scale = 1.0 / std::sqrt(norm);
    for (auto& value : input) { value.x *= scale; value.y *= scale; }
    const auto load_input = [&] {
        runtime.synchronize();
        runtime.reset_mapping();
        const auto offset = static_cast<std::size_t>(runtime.topology.rank) * runtime.local_elements;
        CUDA_TRY(cudaMemcpyAsync(runtime.state, input.data() + offset, runtime.state_bytes,
                                 cudaMemcpyHostToDevice, runtime.stream));
        runtime.synchronize();
    };
    // Each individual logical target starts in canonical layout; multi-rank
    // launches therefore exercise both local and global physical targets.
    for (int q = 0; q < runtime.nqubits; ++q) {
        load_input();
        reference = input;
        cpu_hadamard(reference, q);
        apply_hadamard(runtime, q);
        compare_logical_state(runtime, reference, "individual H on logical q=" + std::to_string(q));
    }
    load_input();
    reference = input;
    for (int q = 0; q < runtime.nqubits; ++q) cpu_hadamard(reference, q);
    job.execute(runtime, 0, 0);
    compare_logical_state(runtime, reference, "all H on asymmetric complex input");
}

void run_benchmark(Runtime& runtime, Job& job, const Options& options) {
    job.prepare(runtime);
    const auto one = [&](std::uint64_t repetition, bool measured, int iteration) {
        runtime.reset_mapping();
        job.initialize(runtime, options.seed, repetition);
        runtime.synchronize();
        MPI_TRY(MPI_Barrier(MPI_COMM_WORLD));
        const double start = MPI_Wtime();
        const auto result = job.execute(runtime, options.seed, repetition);
        runtime.synchronize();
        const double elapsed = MPI_Wtime() - start;
        if (measured) {
            double maximum = 0.0;
            MPI_TRY(MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
            if (runtime.topology.rank == 0) {
                std::string bits;
                if (result.has_measurement) {
                    bits.reserve(static_cast<std::size_t>(options.nqubits));
                    for (int q = options.nqubits - 1; q >= 0; --q)
                        bits.push_back(((result.measurement >> q) & 1) ? '1' : '0');
                }
                std::cout << options.job << ',' << iteration << ',' << options.nqubits << ','
                          << runtime.topology.ranks << ',' << std::setprecision(17) << maximum
                          << ',' << bits << '\n';
                if (!std::cout) fail("CSV output failed");
            }
        }
    };
    // Warm-ups have a separate seed-index domain and never consume measured indices.
    for (int i = 0; i < options.warmup; ++i)
        one((std::uint64_t{1} << 63) + static_cast<std::uint64_t>(i), false, 0);
    if (runtime.topology.rank == 0)
        std::cout << "job,iteration,nqubits,mpi_ranks,elapsed_seconds,measurement_bits\n";
    for (int i = 0; i < options.repeats; ++i)
        one(static_cast<std::uint64_t>(i), true, i + 1);
}
} // namespace bench
