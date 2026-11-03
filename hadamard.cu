#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
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
    std::cout << "Usage: " << executable << " [--nqubits N] [--help]\n"
              << "  N: 1 through 31 (default: 31); uses CUDA device 0.\n";
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
        for (int i = 1; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            if (argument == "--help") {
                usage(argv[0]);
                return 0;
            }
            if (argument != "--nqubits" || seen_nqubits || i + 1 == argc)
                throw std::invalid_argument("Expected --nqubits N or --help");
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
        if (workspace_bytes > std::numeric_limits<std::size_t>::max() - state_bytes)
            throw std::overflow_error("Combined allocation size does not fit size_t");

        std::size_t available = 0, total = 0;
        check(cudaMemGetInfo(&available, &total), "cudaMemGetInfo");
        if (state_bytes + workspace_bytes > available) {
            report_memory(state_bytes, workspace_bytes, available, total);
            throw std::runtime_error("Insufficient GPU memory; requested qubit count is unchanged");
        }
        // Allocation can still fail if memory availability changes after the query.
        auto allocate = [&](void** pointer, std::size_t bytes, const char* operation) {
            const auto status = cudaMalloc(pointer, bytes);
            if (status != cudaSuccess) {
                std::size_t current_available = available, current_total = total;
                const auto info_status = cudaMemGetInfo(&current_available, &current_total);
                report_memory(state_bytes, workspace_bytes, current_available, current_total);
                if (info_status != cudaSuccess)
                    std::cerr << "Memory re-query failed: " << cudaGetErrorString(info_status) << '\n';
                check(status, operation);
            }
        };
        allocate(&resources.state, state_bytes, "cudaMalloc(state)");
        if (workspace_bytes != 0)
            allocate(&resources.workspace, workspace_bytes, "cudaMalloc(workspace)");

        check(cudaMemsetAsync(resources.state, 0, state_bytes, resources.stream),
              "cudaMemsetAsync(state)");
        check(cudaMemcpyAsync(resources.state, &one, sizeof(one), cudaMemcpyHostToDevice,
                              resources.stream), "cudaMemcpyAsync(amplitude 0)");

        // No application-level gate fusion: exactly one API call per target.
        // This does not constrain internal cuStateVec implementation optimizations.
        for (std::int32_t target = 0; target < static_cast<std::int32_t>(nqubits); ++target) {
            ++calls;
            check(custatevecApplyMatrix(
                      resources.handle, resources.state, CUDA_C_64F, nqubits,
                      hadamard, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                      &target, 1, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_64F,
                      resources.workspace, workspace_bytes), "custatevecApplyMatrix");
        }
        check(cudaStreamSynchronize(resources.stream), "cudaStreamSynchronize(after gates)");
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
