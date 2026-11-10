#include "bench.hpp"
#include "hadamard_order.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <numeric>
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
    explicit HadamardJob(bool reverse = false, HadamardRange range = HadamardRange::all)
        : reverse_(reverse), range_(range) {}
    void prepare(Runtime& runtime) override {
        runtime.reserve_gate_workspace(h_matrix, 1);
        if (runtime.topology.rank == 0)
            std::cerr << "hadamard_gate_calls_per_repetition=" << hadamard_end(runtime.nqubits, range_) - hadamard_begin(runtime.nqubits, range_)
                      << " localization=swap-global-target-with-physical-bit-0"
                      << " final_mapping=retained; no restoration\n";
    }
    void initialize(Runtime& runtime, std::uint64_t, std::uint64_t) override {
        runtime.initialize_zero();
    }
    JobResult execute(Runtime& runtime, std::uint64_t, std::uint64_t) override {
        execute_sequence(runtime, [](int) {});
        return {};
    }
    // The untimed check observes each logical target after its gate completes.
    template<class AfterGate>
    void execute_sequence(Runtime& runtime, AfterGate after_gate) {
        for_each_hadamard_target(runtime.nqubits, reverse_, [&](int q) {
            begin_gate(runtime, q);
            apply_hadamard(runtime, q);
            end_gate(runtime);
            after_gate(q);
        }, range_);
    }
private:
    bool reverse_;
    HadamardRange range_;
};

// Counter-derived SplitMix64 draws: warm-ups use the harness's high-bit domain.
// Taking 53 bits gives a reproducible double in [0,1), including zero.
double measurement_draw(std::uint64_t seed, std::uint64_t repetition) {
    std::uint64_t z = seed + UINT64_C(0x9e3779b97f4a7c15) * (repetition + 1);
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return static_cast<double>((z ^ (z >> 31)) >> 11) * 0x1.0p-53;
}

class MeasurementJob final : public Job {
public:
    explicit MeasurementJob(const std::string& initial_state = "uniform") : initial_state_(initial_state) {}
    void prepare(Runtime& runtime) override {
        ordering_.resize(runtime.topology.local_bits);
        bits_.resize(runtime.topology.local_bits);
        std::iota(ordering_.rbegin(), ordering_.rend(), 0);
        masses_.resize(runtime.topology.ranks);
        runtime.reserve_measurement_workspace();
        if (runtime.topology.rank == 0)
            std::cerr << "measurement=joint-full-register rank_selection=measured-FP64-masses"
                      << " local_bit_order=descending initial_state=" << initial_state_
                      << " rng=SplitMix64-counter-53-bits host_mass_buffer_bytes="
                      << checked_multiply(masses_.size(), sizeof(double)) << '\n';
    }
    void initialize(Runtime& runtime, std::uint64_t, std::uint64_t) override {
        if (initial_state_ == "uniform") runtime.initialize_uniform();
        else if (initial_state_ == "zero") runtime.initialize_zero();
        else if (initial_state_ == "ghz") runtime.initialize_ghz();
        else fail("Unknown measurement initial state: " + initial_state_);
    }
    JobResult execute(Runtime& runtime, std::uint64_t seed, std::uint64_t repetition) override {
        begin_gate(runtime, -1);
        const auto result = measure(runtime, measurement_draw(seed, repetition));
        end_gate(runtime);
        return result;
    }
    // Untimed assertion used by --check-measure.
    void check_ordering() const {
        for (std::size_t i = 0; i < ordering_.size(); ++i)
            if (ordering_[i] != static_cast<int32_t>(ordering_.size() - 1 - i))
                fail("Measurement check: local bit ordering must be descending");
    }
    // Controlled draws exercise the production path without allocation or caching.
    JobResult measure(Runtime& runtime, double draw) {
        const auto& t = runtime.topology;
        double local_mass = 0.0;
        SV_TRY(custatevecAbs2SumArray(runtime.handle, runtime.state, CUDA_C_64F,
            t.local_bits, &local_mass, nullptr, 0, nullptr, nullptr, 0));
        CUDA_TRY(cudaStreamSynchronize(runtime.stream));
        MPI_TRY(MPI_Gather(&local_mass, 1, MPI_DOUBLE, masses_.data(), 1, MPI_DOUBLE,
                           0, MPI_COMM_WORLD));
        int owner = -1;
        double conditional_draw = 0.0;
        if (t.rank == 0) {
            if (!std::isfinite(draw) || draw < 0.0 || draw >= 1.0)
                fail("Measurement draw must be in [0,1)");
            double total = 0.0;
            for (const auto mass : masses_) {
                if (!std::isfinite(mass) || mass < 0.0)
                    fail("Measurement found an invalid sub-state-vector squared norm");
                total += mass;
            }
            if (!std::isfinite(total) || total <= 0.0)
                fail("Measurement requires a finite positive global squared norm");
            // Multiplication can round to total even though draw < 1.
            const double target = std::min(draw * total, std::nextafter(total, 0.0));
            double prefix = 0.0;
            for (int rank = 0; rank < t.ranks; ++rank) {
                const double end = prefix + masses_[rank];
                // Half-open intervals skip zero-mass ranks, including leading
                // and trailing ones. Equality belongs to the following rank.
                if (masses_[rank] > 0.0 && target < end) {
                    owner = rank;
                    conditional_draw = std::min(std::max((target - prefix) / masses_[rank], 0.0),
                                                std::nextafter(1.0, 0.0));
                    break;
                }
                prefix = end;
            }
            if (owner < 0) fail("Measurement could not select a positive-mass rank");
        }
        MPI_TRY(MPI_Bcast(&owner, 1, MPI_INT, 0, MPI_COMM_WORLD));
        MPI_TRY(MPI_Bcast(&conditional_draw, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));

        std::uint64_t physical = 0;
        if (t.rank == owner) {
            // Condition the single global draw on the selected rank. Offset 0
            // and the measured local mass avoid subtracting a large global
            // offset inside the device's local cumulative-probability search.
            SV_TRY(custatevecBatchMeasureWithOffset(runtime.handle, runtime.state, CUDA_C_64F,
                t.local_bits, bits_.data(), ordering_.data(), t.local_bits,
                conditional_draw, CUSTATEVEC_COLLAPSE_NONE, 0.0, local_mass));
            CUDA_TRY(cudaStreamSynchronize(runtime.stream));
            physical = static_cast<std::uint64_t>(owner) << t.local_bits;
            // bitString[i] is the value of physical qubit bitOrdering[i],
            // not physical qubit i. Keep these pairs for mask and collapse too.
            for (int i = 0; i < t.local_bits; ++i) {
                if (bits_[i] != 0 && bits_[i] != 1) fail("Invalid local measurement bit");
                physical |= static_cast<std::uint64_t>(bits_[i]) << ordering_[i];
            }
            double selected_mass = 0.0;
            SV_TRY(custatevecAbs2SumArray(runtime.handle, runtime.state, CUDA_C_64F,
                t.local_bits, &selected_mass, nullptr, 0, bits_.data(), ordering_.data(), t.local_bits));
            CUDA_TRY(cudaStreamSynchronize(runtime.stream));
            if (!std::isfinite(selected_mass) || selected_mass <= 0.0)
                fail("Measurement selected a zero or invalid probability amplitude");
            // Scale the actual complex amplitude by a positive real factor;
            // its phase is retained. All local bits are fixed by this collapse.
            SV_TRY(custatevecCollapseByBitString(runtime.handle, runtime.state, CUDA_C_64F,
                t.local_bits, bits_.data(), ordering_.data(), t.local_bits, selected_mass));
        } else {
            CUDA_TRY(cudaMemsetAsync(runtime.state, 0, runtime.state_bytes, runtime.stream));
        }
        MPI_TRY(MPI_Bcast(&physical, 1, MPI_UINT64_T, owner, MPI_COMM_WORLD));
        std::uint64_t logical = 0;
        for (int q = 0; q < runtime.nqubits; ++q)
            logical |= ((physical >> runtime.logical_to_physical[q]) & std::uint64_t{1}) << q;
        return {logical, true};
    }
private:
    std::string initial_state_;
    std::vector<int32_t> ordering_, bits_;
    std::vector<double> masses_;
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

std::unique_ptr<Job> make_job(const std::string& name, const std::string& initial_state) {
    if (name == "hadamard") return std::make_unique<HadamardJob>();
    if (name == "hadamard-reverse") return std::make_unique<HadamardJob>(true);
    if (name == "hadamard-low") return std::make_unique<HadamardJob>(false, HadamardRange::low);
    if (name == "hadamard-high") return std::make_unique<HadamardJob>(false, HadamardRange::high);
    if (name == "measure") return std::make_unique<MeasurementJob>(initial_state);
    throw std::runtime_error("Job '" + name + "' is not implemented; no timings produced");
}

void check_hadamard(Runtime& runtime) {
    if (runtime.nqubits > 20) fail("Hadamard check requires at most 20 total qubits");
    HadamardJob job;
    job.prepare(runtime);
    runtime.finalize_preparation();
    const auto count = std::size_t{1} << runtime.nqubits;
    std::vector<cuDoubleComplex> reference(count);

    // Repeat both orders to exercise reinitialization after a changed map.
    for (bool reverse : {false, true}) {
        HadamardJob ordered_job(reverse);
        for (int repetition = 0; repetition < 2; ++repetition) {
            runtime.reset_mapping();
            ordered_job.initialize(runtime, 0, repetition);
            runtime.synchronize();
            ordered_job.execute(runtime, 0, repetition);
            const double uniform = std::exp2(-0.5 * runtime.nqubits);
            std::fill(reference.begin(), reference.end(), make_cuDoubleComplex(uniform, 0.0));
            compare_logical_state(runtime, reference, std::string(reverse ? "reverse" : "ascending") +
                " all H on |0>, repetition " + std::to_string(repetition + 1));
        }
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
    for (int variant = 0; variant < 4; ++variant) {
        const bool reverse = variant == 1;
        const auto range = variant == 2 ? HadamardRange::low :
                           variant == 3 ? HadamardRange::high : HadamardRange::all;
        HadamardJob ordered_job(reverse, range);
        const std::string label = variant == 2 ? "low" : variant == 3 ? "high" :
                                  reverse ? "reverse" : "ascending";
        // Independent expected bounds also check the production range selector.
        const int begin = variant == 3 ? std::min(5, runtime.nqubits) : 0;
        const int end = variant == 2 ? std::min(5, runtime.nqubits) : runtime.nqubits;
        load_input();
        reference = input;
        int step = 0;
        ordered_job.execute_sequence(runtime, [&](int q) {
            const int expected = reverse ? end - 1 - step : begin + step;
            if (q != expected) fail("Incorrect " + label + " logical Hadamard target order");
            cpu_hadamard(reference, expected);
            compare_logical_state(runtime, reference, label + " prefix through q=" + std::to_string(q));
            ++step;
        });
        if (step != end - begin) fail("Incorrect Hadamard gate count");
        // Also exercise the production execute entry point without observers.
        load_input();
        ordered_job.execute(runtime, 0, 0);
        compare_logical_state(runtime, reference, label + " selected H on asymmetric complex input");
    }
}

void check_measure(Runtime& runtime) {
    if (runtime.nqubits > 20) fail("Measurement check requires at most 20 total qubits");
    MeasurementJob job;
    job.prepare(runtime);
    runtime.finalize_preparation();
    job.check_ordering();
    const auto count = std::size_t{1} << runtime.nqubits;
    const auto base = static_cast<std::uint64_t>(runtime.topology.rank) * runtime.local_elements;
    std::vector<cuDoubleComplex> input(count), actual(runtime.local_elements);
    const std::vector<double> draws = {0.0, 0.125, 0.25, 0.5, 0.75, std::nextafter(1.0, 0.0)};
    const auto abs2 = [](cuDoubleComplex a) { return a.x * a.x + a.y * a.y; };
    const auto check = [&](const std::string& label, const std::string& initial_state, bool permute_mapping = false) {
        MeasurementJob initializer(initial_state);
        const bool direct = !initial_state.empty();
        double total = 0.0;
        for (const auto a : input) total += abs2(a);
        for (const double controlled_draw : draws) {
            // Odd-N uniform amplitudes have rounded squared magnitudes. Keep
            // their interior probes away from CDF boundaries so CPU sequential
            // sums and GPU tree sums need not round identically. The exact
            // sparse fixture below exercises boundaries without this ambiguity.
            const double draw = initial_state == "uniform" && controlled_draw > 0.0 && controlled_draw < 0.9
                ? controlled_draw + 0.0314159265358979 : controlled_draw;
            runtime.reset_mapping();
            if (direct) initializer.initialize(runtime, 0, 0);
            else CUDA_TRY(cudaMemcpyAsync(runtime.state, input.data() + base, runtime.state_bytes,
                                          cudaMemcpyHostToDevice, runtime.stream));
            runtime.synchronize();
            if (direct) {
                CUDA_TRY(cudaMemcpy(actual.data(), runtime.state, runtime.state_bytes, cudaMemcpyDeviceToHost));
                int bad = 0, any_bad = 0;
                double norm = 0.0, global_norm = 0.0;
                for (std::size_t i = 0; i < actual.size(); ++i) {
                    if (actual[i].x != input[base + i].x || actual[i].y != 0.0) bad = 1;
                    norm += abs2(actual[i]);
                }
                MPI_TRY(MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
                if (!std::isfinite(global_norm) || std::abs(global_norm - 1.0) > 1e-11) bad = 1;
                MPI_TRY(MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
                if (any_bad) fail("Measurement check: direct initialization or normalization mismatch: " + label);
            }
            // Independent CPU cumulative reference over the full small fixture.
            const double target = std::min(draw * total, std::nextafter(total, 0.0));
            double cumulative = 0.0;
            std::uint64_t expected = 0;
            for (; expected < count; ++expected) {
                cumulative += abs2(input[expected]);
                if (target < cumulative) break;
            }
            if (expected == count) fail("Measurement CPU reference could not select an outcome");
            if (permute_mapping)
                std::reverse(runtime.logical_to_physical.begin(), runtime.logical_to_physical.end());
            std::uint64_t expected_logical = 0;
            for (int q = 0; q < runtime.nqubits; ++q)
                expected_logical |= ((expected >> runtime.logical_to_physical[q]) & 1) << q;
            const auto result = job.measure(runtime, draw);
            runtime.synchronize();
            std::uint64_t minimum = 0, maximum = 0;
            MPI_TRY(MPI_Allreduce(&result.measurement, &minimum, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD));
            MPI_TRY(MPI_Allreduce(&result.measurement, &maximum, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD));
            if (!result.has_measurement || minimum != expected_logical || maximum != expected_logical)
                fail("Measurement check: outcome/reference or rank agreement mismatch: " + label);
            CUDA_TRY(cudaMemcpy(actual.data(), runtime.state, runtime.state_bytes, cudaMemcpyDeviceToHost));
            const double scale = 1.0 / std::sqrt(abs2(input[expected]));
            const auto phase = make_cuDoubleComplex(input[expected].x * scale, input[expected].y * scale);
            int bad = 0, any_bad = 0;
            std::uint64_t survivors = 0, global_survivors = 0;
            double norm = 0.0, global_norm = 0.0;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const auto a = actual[i];
                if (!std::isfinite(a.x) || !std::isfinite(a.y)) bad = 1;
                if (a.x != 0.0 || a.y != 0.0) ++survivors;
                norm += abs2(a);
                if (base + i == expected) {
                    if (std::hypot(a.x - phase.x, a.y - phase.y) > 1e-11) bad = 1;
                } else if (a.x != 0.0 || a.y != 0.0) bad = 1;
            }
            MPI_TRY(MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
            MPI_TRY(MPI_Allreduce(&survivors, &global_survivors, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD));
            MPI_TRY(MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
            if (any_bad || global_survivors != 1 || !std::isfinite(global_norm) ||
                std::abs(global_norm - 1.0) > 1e-11)
                fail("Measurement check: collapse, phase, or normalization mismatch: " + label);
        }
        if (runtime.topology.rank == 0)
            std::cerr << "Measurement check passed: " << label << " controlled_draws=" << draws.size() << '\n';
    };
    // Basis states on low, middle, and high rank indices; negative imaginary
    // phase and a nonunit input norm exercise phase-preserving normalization.
    for (const int owner : {0, runtime.topology.ranks / 2, runtime.topology.ranks - 1}) {
        std::fill(input.begin(), input.end(), make_cuDoubleComplex(0.0, 0.0));
        input[static_cast<std::uint64_t>(owner) * runtime.local_elements + runtime.local_elements - 1] =
            make_cuDoubleComplex(0.0, -2.0);
        check("basis on rank " + std::to_string(owner), "");
    }
    // One-hot local patterns expose reversal errors on every MPI owner,
    // including ranks that the small fixed draw set might otherwise miss.
    for (int owner = 0; owner < runtime.topology.ranks; ++owner) {
        for (int q = 0; q < runtime.topology.local_bits; ++q) {
            std::fill(input.begin(), input.end(), make_cuDoubleComplex(0.0, 0.0));
            input[static_cast<std::uint64_t>(owner) * runtime.local_elements +
                  (std::uint64_t{1} << q)] = make_cuDoubleComplex(-1.0, 2.0);
            check("one-hot local q=" + std::to_string(q) + " on rank " + std::to_string(owner), "");
        }
    }
    if (runtime.topology.local_bits >= 3) {
        // NVIDIA's descending-order batch_measure sample: draw 0.5 selects
        // physical local index 6 and returns {1,1,0}, not packed index 3.
        const cuDoubleComplex sample[] = {{0,0}, {0,0.1}, {0.1,0.1}, {0.1,0.2},
                                         {0.2,0.2}, {0.3,0.3}, {0.3,0.4}, {0.4,0.5}};
        std::fill(input.begin(), input.end(), make_cuDoubleComplex(0.0, 0.0));
        std::copy(std::begin(sample), std::end(sample), input.begin());
        check("NVIDIA descending-order asymmetric sample", "");
    }
    // Dense asymmetric complex input also checks selected mass and phase when
    // reversing the local output bits cannot be hidden by symmetric endpoints.
    for (std::size_t i = 0; i < count; ++i)
        input[i] = make_cuDoubleComplex(1.0 + (i % 7), -0.25 * (1.0 + (i % 11)));
    check("asymmetric complex amplitudes", "");
    check("asymmetric complex amplitudes with reversed logical mapping", "", true);
    for (int sparse = 0; sparse < 2; ++sparse) {
        std::fill(input.begin(), input.end(), make_cuDoubleComplex(0.0, 0.0));
        for (int rank = 0; rank < runtime.topology.ranks; ++rank) {
            // In sparse mode only odd ranks survive (rank 0 for P=1).
            if (sparse && runtime.topology.ranks > 1 && rank % 2 == 0) continue;
            const auto offset = static_cast<std::uint64_t>(rank) * runtime.local_elements;
            const double weight = rank + 1.0;
            input[offset] = make_cuDoubleComplex(-weight, 0.0);
            input[offset + runtime.local_elements - 1] = make_cuDoubleComplex(0.0, 2.0 * weight);
        }
        check(sparse ? "unequal masses and zero-probability ranks" : "unequal rank and local masses", "");
    }
    // Equal exact masses give controlled draws precisely on rank and local
    // interval boundaries, with zero-probability gaps between the two entries.
    std::fill(input.begin(), input.end(), make_cuDoubleComplex(0.0, 0.0));
    for (int rank = 0; rank < runtime.topology.ranks; ++rank) {
        const auto offset = static_cast<std::uint64_t>(rank) * runtime.local_elements;
        input[offset] = make_cuDoubleComplex(1.0, 0.0);
        input[offset + runtime.local_elements - 1] = make_cuDoubleComplex(0.0, -1.0);
    }
    check("exact interval boundaries with local zero-probability gaps", "");
    const double uniform = std::exp2(-0.5 * runtime.nqubits);
    std::fill(input.begin(), input.end(), make_cuDoubleComplex(uniform, 0.0));
    check("independently GPU-initialized globally normalized uniform state", "uniform");
    std::fill(input.begin(), input.end(), make_cuDoubleComplex(0.0, 0.0));
    input.front() = make_cuDoubleComplex(1.0, 0.0);
    check("independently GPU-initialized zero state", "zero");
    input.front() = input.back() = make_cuDoubleComplex(h_scale, 0.0);
    // Draws on both sides of 0.5 exercise both GHZ endpoints at P=1 and
    // on the first/last ranks at P>1, including zero-mass intermediate ranks.
    check("independently GPU-initialized GHZ state, both outcomes", "ghz");
}

void Job::begin_gate(Runtime& runtime, int logical_qubit) {
    if (!profile_gates) return;
    if (timed_gate_count >= gate_timings.size()) fail("Per-gate timing capacity exceeded");
    auto& timing = gate_timings[timed_gate_count];
    timing.logical_qubit = logical_qubit;
    CUDA_TRY(cudaEventRecord(timing.start, runtime.stream));
}

void Job::end_gate(Runtime& runtime) {
    if (!profile_gates) return;
    CUDA_TRY(cudaEventRecord(gate_timings[timed_gate_count].stop, runtime.stream));
    ++timed_gate_count;
}

void run_benchmark(Runtime& runtime, Job& job, const Options& options) {
    std::ofstream gate_output;
    if (!options.gate_timings.empty()) {
        if (runtime.topology.rank == 0) {
            gate_output.open(options.gate_timings);
            if (!gate_output) fail("Cannot open gate timing CSV: " + options.gate_timings);
            gate_output << "job,iteration,nqubits,mpi_ranks,gate_index,gate,logical_qubit,elapsed_seconds\n";
        }
        job.gate_timings.resize(runtime.nqubits);
        for (auto& timing : job.gate_timings) {
            CUDA_TRY(cudaEventCreateWithFlags(&timing.start, cudaEventBlockingSync));
            CUDA_TRY(cudaEventCreateWithFlags(&timing.stop, cudaEventBlockingSync));
        }
    }
    job.prepare(runtime);
    runtime.finalize_preparation();
    // Separate, timing-enabled events on this rank's already assigned GPU.
    cudaEvent_t start_event = nullptr, stop_event = nullptr;
    CUDA_TRY(cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync));
    CUDA_TRY(cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync));
    const auto one = [&](std::uint64_t repetition, bool measured, int iteration) {
        job.profile_gates = measured && !options.gate_timings.empty();
        job.timed_gate_count = 0;
        runtime.reset_mapping();
        job.initialize(runtime, options.seed, repetition);
        runtime.synchronize();
        MPI_TRY(MPI_Barrier(MPI_COMM_WORLD));
        CUDA_TRY(cudaEventRecord(start_event, runtime.stream));
        const auto result = job.execute(runtime, options.seed, repetition);
        // Gate operations use runtime.stream; swaps finish transfer work before
        // returning. Queue stop before waiting to avoid a post-execution host gap.
        CUDA_TRY(cudaEventRecord(stop_event, runtime.stream));
        CUDA_TRY(cudaEventSynchronize(stop_event));
        float elapsed_ms = 0.0f;
        CUDA_TRY(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        const double elapsed = static_cast<double>(elapsed_ms) / 1000.0;
        if (measured) {
            // All event reads, rank reductions and output occur after the total timer.
            for (std::size_t g = 0; g < job.timed_gate_count; ++g) {
                const auto& timing = job.gate_timings[g];
                float gate_ms = 0.0f;
                CUDA_TRY(cudaEventElapsedTime(&gate_ms, timing.start, timing.stop));
                const double seconds = static_cast<double>(gate_ms) / 1000.0;
                double maximum_gate = 0.0;
                MPI_TRY(MPI_Reduce(&seconds, &maximum_gate, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
                if (runtime.topology.rank == 0) {
                    gate_output << options.job << ',' << iteration << ',' << options.nqubits << ','
                                << runtime.topology.ranks << ',' << g + 1 << ','
                                << (timing.logical_qubit < 0 ? "measure" : "h") << ',';
                    if (timing.logical_qubit >= 0) gate_output << timing.logical_qubit;
                    gate_output << ',' << std::setprecision(17) << maximum_gate << '\n';
                    if (!gate_output) fail("Gate timing CSV output failed");
                }
            }
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
    job.profile_gates = false;
    for (auto& timing : job.gate_timings) {
        CUDA_TRY(cudaEventDestroy(timing.stop));
        CUDA_TRY(cudaEventDestroy(timing.start));
    }
    job.gate_timings.clear();
    if (runtime.topology.rank == 0 && gate_output.is_open()) {
        gate_output.close();
        if (!gate_output) fail("Gate timing CSV close failed");
    }
    CUDA_TRY(cudaEventDestroy(stop_event));
    CUDA_TRY(cudaEventDestroy(start_event));
    // Detect buffered output errors while MPI is still active, so peers abort
    // coherently rather than reporting success with a truncated CSV file.
    if (runtime.topology.rank == 0) {
        std::cout.flush();
        if (!std::cout) fail("CSV output flush failed");
    }
}
} // namespace bench
