#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr unsigned norm_blocks = 1024;
constexpr unsigned norm_threads = 256;
constexpr std::size_t norm_workspace_bytes = (norm_blocks + 1) * sizeof(double);

// Fixed-size partial sums keep validation storage independent of state size.
__global__ void squared_norm_partials(const cuDoubleComplex* state,
                                      std::uint64_t count, double* partials) {
    __shared__ double sums[norm_threads];
    double sum = 0.0;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const auto value = state[i];
        sum += value.x * value.x + value.y * value.y;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2; offset != 0; offset /= 2) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = sums[0];
}

__global__ void squared_norm_total(const double* partials, double* result) {
    __shared__ double sums[norm_threads];
    double sum = 0.0;
    for (unsigned i = threadIdx.x; i < norm_blocks; i += blockDim.x)
        sum += partials[i];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2; offset != 0; offset /= 2) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) *result = sums[0];
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check(custatevecStatus_t status, const char* operation) {
    if (status != CUSTATEVEC_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " +
                                 custatevecGetErrorString(status));
}

// Cleanup also runs on partial initialization and asynchronous failures.
struct Resources {
    cudaStream_t stream = nullptr;
    custatevecHandle_t handle = nullptr;
    void* state = nullptr;
    void* workspace = nullptr;

    Resources() = default;
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;

    bool release() noexcept {
        bool ok = true;
        auto cleanup = [&ok](auto status, const char* operation) {
            try {
                check(status, operation);
            } catch (const std::exception& error) {
                std::cerr << "Cleanup error: " << error.what() << '\n';
                ok = false;
            }
        };
        if (stream) cleanup(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        if (handle) {
            cleanup(custatevecDestroy(handle), "custatevecDestroy");
            handle = nullptr;
        }
        if (workspace) {
            cleanup(cudaFree(workspace), "cudaFree(workspace)");
            workspace = nullptr;
        }
        if (state) {
            cleanup(cudaFree(state), "cudaFree(state)");
            state = nullptr;
        }
        if (stream) {
            cleanup(cudaStreamDestroy(stream), "cudaStreamDestroy");
            stream = nullptr;
        }
        return ok;
    }

    ~Resources() { release(); }
};

void usage(const char* executable) {
    std::cout << "Usage: " << executable << " [--nqubits N] [--validate] [--help]\n"
              << "  N: 1 through 31 (default: 31); uses CUDA device 0.\n"
              << "  --validate: GPU squared norm and exhaustive (N <= 10) or sampled amplitudes.\n";
}

bool validate_state(Resources& resources, std::uint32_t nqubits,
                    std::uint64_t count) {
    auto* scratch = static_cast<double*>(resources.workspace);
    auto* state = static_cast<const cuDoubleComplex*>(resources.state);
    double norm = 0.0;

    const bool exhaustive = nqubits <= 10;
    std::vector<std::uint64_t> indices;
    if (exhaustive) {
        for (std::uint64_t i = 0; i < count; ++i) indices.push_back(i);
    } else {
        indices.push_back(0);
        for (std::uint32_t q = 0; q < nqubits; ++q)
            indices.push_back(std::uint64_t{1} << q);
        indices.push_back(count - 1);
        // Specified unsigned arithmetic makes this sample portable and reproducible.
        std::uint64_t seed = 0x6a09e667f3bcc909ULL;
        for (unsigned i = 0; i < 64; ++i) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            indices.push_back((seed >> 32) & (count - 1));
        }
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    }
    std::vector<cuDoubleComplex> values(indices.size());
    // Drain pending copies before local host destinations die on an exception.
    struct TransferCompletion {
        cudaStream_t stream;
        ~TransferCompletion() {
            if (!stream) return;
            const auto status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess)
                std::cerr << "Validation cleanup error: " << cudaGetErrorString(status) << '\n';
        }
    } completion{resources.stream};
    squared_norm_partials<<<norm_blocks, norm_threads, 0, resources.stream>>>(state, count, scratch);
    check(cudaGetLastError(), "squared_norm_partials launch");
    squared_norm_total<<<1, norm_threads, 0, resources.stream>>>(scratch, scratch + norm_blocks);
    check(cudaGetLastError(), "squared_norm_total launch");
    check(cudaMemcpyAsync(&norm, scratch + norm_blocks, sizeof(norm), cudaMemcpyDeviceToHost,
                          resources.stream), "cudaMemcpyAsync(squared norm)");
    if (exhaustive) {
        check(cudaMemcpyAsync(values.data(), state, values.size() * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToHost, resources.stream),
              "cudaMemcpyAsync(exhaustive amplitudes)");
    } else {
        for (std::size_t i = 0; i < indices.size(); ++i)
            check(cudaMemcpyAsync(&values[i], state + indices[i], sizeof(cuDoubleComplex),
                                  cudaMemcpyDeviceToHost, resources.stream),
                  "cudaMemcpyAsync(validation amplitude)");
    }
    check(cudaStreamSynchronize(resources.stream), "cudaStreamSynchronize(validation)");
    completion.stream = nullptr;

    const double expected = std::exp2(-0.5 * static_cast<double>(nqubits));
    constexpr double relative_tolerance = 1.0e-12;
    constexpr double norm_tolerance = 1.0e-11;
    const double amplitude_tolerance = expected * relative_tolerance;
    double max_error = 0.0;
    bool amplitudes_ok = true;
    for (const auto& value : values) {
        // Complex magnitude of error checks both components and rejects NaN/Inf.
        const double error = std::hypot(value.x - expected, value.y);
        if (!std::isfinite(error)) max_error = std::numeric_limits<double>::infinity();
        else max_error = std::max(max_error, error);
        amplitudes_ok = amplitudes_ok && std::isfinite(error) && error <= amplitude_tolerance;
    }
    const bool norm_ok = std::isfinite(norm) && std::abs(norm - 1.0) <= norm_tolerance;
    const bool passed = norm_ok && amplitudes_ok;
    std::cout << std::scientific << std::setprecision(17)
              << "Amplitude validation: " << (exhaustive ? "exhaustive" : "sampled")
              << " (" << values.size() << " of " << count << " amplitudes)\n"
              << "Measured squared norm: " << norm << "\n"
              << "Squared norm tolerance: " << norm_tolerance << "\n"
              << "Squared norm check: " << (norm_ok ? "PASS" : "FAIL") << "\n"
              << "Expected amplitude: " << expected << " + 0i\n"
              << "Amplitude tolerance: " << amplitude_tolerance
              << " (relative " << relative_tolerance << ")\n"
              << "Maximum absolute amplitude error: " << max_error << "\n"
              << "Amplitude check: " << (amplitudes_ok ? "PASS" : "FAIL") << "\n"
              << "Validation status: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

std::uint32_t parse_qubits(std::string_view value) {
    unsigned int number = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        number < 1 || number > 31)
        throw std::invalid_argument("--nqubits requires an integer from 1 through 31");
    return number;
}

void report_memory(std::size_t state_bytes, std::size_t workspace_bytes,
                   std::size_t available, std::size_t total) {
    std::cerr << "GPU memory: state requires " << state_bytes << " bytes, workspace requires "
              << workspace_bytes << " bytes; combined " << state_bytes + workspace_bytes
              << " bytes; available " << available << " bytes of " << total << " bytes.\n";
}
}  // namespace

int main(int argc, char** argv) {
    std::uint32_t calls = 0;
    try {
        std::uint32_t nqubits = 31;
        bool seen_nqubits = false;
        bool validate = false;
        for (int i = 1; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            if (argument == "--help") {
                usage(argv[0]);
                return 0;
            }
            if (argument == "--validate" && !validate) {
                validate = true;
                continue;
            }
            if (argument != "--nqubits" || seen_nqubits || i + 1 == argc)
                throw std::invalid_argument("Expected --nqubits N, --validate, or --help");
            nqubits = parse_qubits(argv[++i]);
            seen_nqubits = true;
        }

        static_assert(sizeof(cuDoubleComplex) == 16, "complex128 requires 16 bytes");
        const std::uint64_t amplitudes = std::uint64_t{1} << nqubits;
        if (amplitudes > std::numeric_limits<std::size_t>::max() / sizeof(cuDoubleComplex))
            throw std::overflow_error("State-vector size does not fit size_t");
        const std::size_t state_bytes = static_cast<std::size_t>(amplitudes) * sizeof(cuDoubleComplex);
        constexpr double gib = 1024.0 * 1024.0 * 1024.0;
        std::cout << "Qubits: " << nqubits << "\nPrecision: FP64 / complex128\n"
                  << "Amplitudes: " << amplitudes << "\nState-vector memory: "
                  << std::fixed << std::setprecision(6) << state_bytes / gib << " GiB ("
                  << state_bytes << " bytes)\n" << std::flush;

        // Keep the small host matrix and initialization scalar alive until cleanup.
        const double scale = 1.0 / std::sqrt(2.0);
        const cuDoubleComplex hadamard[4] = {
            {scale, 0.0}, {scale, 0.0}, {scale, 0.0}, {-scale, 0.0}};
        const cuDoubleComplex one{1.0, 0.0};
        Resources resources;
        check(cudaSetDevice(0), "cudaSetDevice(0)");
        check(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags");
        check(custatevecCreate(&resources.handle), "custatevecCreate");
        check(custatevecSetStream(resources.handle, resources.stream), "custatevecSetStream");

        std::size_t workspace_bytes = 0;
        check(custatevecApplyMatrixGetWorkspaceSize(
                  resources.handle, CUDA_C_64F, nqubits, hadamard, CUDA_C_64F,
                  CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, 0, CUSTATEVEC_COMPUTE_64F,
                  &workspace_bytes), "custatevecApplyMatrixGetWorkspaceSize");
        std::cout << "Additional workspace: " << workspace_bytes << " bytes\n" << std::flush;
        // Validation reuses gate workspace after the gate stream has completed.
        const std::size_t allocated_workspace_bytes =
            std::max(workspace_bytes, validate ? norm_workspace_bytes : std::size_t{0});
        std::cout << "Allocated reusable workspace: " << allocated_workspace_bytes << " bytes\n";
        if (allocated_workspace_bytes > std::numeric_limits<std::size_t>::max() - state_bytes)
            throw std::overflow_error("Combined allocation size does not fit size_t");

        std::size_t available = 0, total = 0;
        check(cudaMemGetInfo(&available, &total), "cudaMemGetInfo");
        if (state_bytes + allocated_workspace_bytes > available) {
            report_memory(state_bytes, allocated_workspace_bytes, available, total);
            throw std::runtime_error("Insufficient GPU memory; requested qubit count is unchanged");
        }
        // Allocation can still fail if memory availability changes after the query.
        auto allocate = [&](void** pointer, std::size_t bytes, const char* operation) {
            const auto status = cudaMalloc(pointer, bytes);
            if (status != cudaSuccess) {
                std::size_t current_available = available, current_total = total;
                const auto info_status = cudaMemGetInfo(&current_available, &current_total);
                report_memory(state_bytes, allocated_workspace_bytes, current_available, current_total);
                if (info_status != cudaSuccess)
                    std::cerr << "Memory re-query failed: " << cudaGetErrorString(info_status) << '\n';
                check(status, operation);
            }
        };
        allocate(&resources.state, state_bytes, "cudaMalloc(state)");
        if (allocated_workspace_bytes != 0)
            allocate(&resources.workspace, allocated_workspace_bytes, "cudaMalloc(workspace)");

        check(cudaMemsetAsync(resources.state, 0, state_bytes, resources.stream),
              "cudaMemsetAsync(state)");
        check(cudaMemcpyAsync(resources.state, &one, sizeof(one), cudaMemcpyHostToDevice,
                              resources.stream), "cudaMemcpyAsync(amplitude 0)");

        // No application-level gate fusion: exactly one API call per target.
        // This does not constrain internal cuStateVec implementation optimizations.
        constexpr std::uint32_t gate_targets = 1;
        constexpr std::uint32_t gate_controls = 0;
        std::array<unsigned, 31> target_visits{};
        for (std::int32_t target = 0; target < static_cast<std::int32_t>(nqubits); ++target) {
            ++calls;
            ++target_visits[target];
            check(custatevecApplyMatrix(
                      resources.handle, resources.state, CUDA_C_64F, nqubits,
                      hadamard, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                      &target, gate_targets, nullptr, nullptr, gate_controls, CUSTATEVEC_COMPUTE_64F,
                      resources.workspace, workspace_bytes), "custatevecApplyMatrix");
        }
        check(cudaStreamSynchronize(resources.stream), "cudaStreamSynchronize(after gates)");
        const bool no_fusion_ok = calls == nqubits && gate_targets == 1 && gate_controls == 0 &&
            std::all_of(target_visits.begin(), target_visits.begin() + nqubits,
                        [](unsigned visits) { return visits == 1; });
        std::cout << "Gate pass: " << calls << " actual custatevecApplyMatrix calls; "
                  << "nTargets=" << gate_targets << ", nControls=" << gate_controls << '\n'
                  << "Application-level no-fusion check: " << (no_fusion_ok ? "PASS" : "FAIL")
                  << " (one visit per target and one API call per gate)\n";
        if (!no_fusion_ok) throw std::runtime_error("Gate pass accounting failed");
        if (validate && !validate_state(resources, nqubits, amplitudes))
            throw std::runtime_error("State validation failed");
        if (!validate) std::cout << "Validation status: NOT REQUESTED\n";
        if (!resources.release())
            throw std::runtime_error("Resource cleanup failed");
        std::cout << "custatevecApplyMatrix calls: " << calls
                  << "\nExecution status: SUCCESS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "custatevecApplyMatrix calls: " << calls
                  << "\nExecution status: FAILED\n" << error.what() << '\n';
        return 1;
    }
}
