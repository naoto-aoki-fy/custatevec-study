#include "bench.hpp"
#include "transfer_policy.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace bench {
namespace {
std::uint64_t number(const std::string& text, const std::string& option) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(option + " requires an unsigned decimal integer");
    std::size_t consumed = 0;
    try {
        const auto value = std::stoull(text, &consumed);
        if (consumed == text.size()) return value;
    } catch (const std::exception&) {}
    throw std::runtime_error(option + " is out of range");
}
}
Options parse_options(int argc, char** argv) {
    Options o;
    bool job_selected = false;
    bool initial_state_selected = false;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { o.help = true; continue; }
        if (key == "--smoke-check") { o.smoke = true; continue; }
        if (key == "--check-hadamard") { o.check_hadamard = true; continue; }
        if (key == "--check-measure") { o.check_measure = true; continue; }
        if (key != "--job" && key != "--nqubits" && key != "--repeats" &&
            key != "--warmup" && key != "--seed" && key != "--communicator" &&
            key != "--communicator-library" && key != "--p2p" &&
            key != "--transfer-buffer" && key != "--transfer-buffer-fraction" &&
            key != "--initial-state")
            throw std::runtime_error("Unknown option: " + key);
        if (++i == argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[i];
        if (key == "--job") {
            if (job_selected) throw std::runtime_error("Select --job exactly once");
            job_selected = true;
            o.job = value;
            continue;
        }
        if (key == "--initial-state") {
            initial_state_selected = true;
            o.initial_state = value == "flat" ? "uniform" : value;
            if (o.initial_state != "uniform" && o.initial_state != "zero" && o.initial_state != "ghz")
                throw std::runtime_error("--initial-state must be uniform (alias flat), zero, or ghz");
            continue;
        }
        if (key == "--communicator") { o.communicator = value; continue; }
        if (key == "--communicator-library") { o.communicator_library = value; continue; }
        if (key == "--p2p") { o.p2p = value; continue; }
        if (key == "--transfer-buffer") { o.transfer_buffer = value; continue; }
        if (key == "--transfer-buffer-fraction") {
            std::size_t consumed = 0;
            o.transfer_buffer_fraction = std::stod(value, &consumed);
            if (consumed != value.size() || !transfer_policy::valid_fraction(o.transfer_buffer_fraction))
                throw std::runtime_error("--transfer-buffer-fraction must be finite and in (0,0.95]");
            continue;
        }
        const auto n = number(value, key);
        if (key == "--seed") { o.seed = n; continue; }
        if (n > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error(key + " exceeds INT_MAX");
        if (key == "--nqubits") { o.nqubits = static_cast<int>(n); o.explicit_nqubits = true; }
        if (key == "--repeats") o.repeats = static_cast<int>(n);
        if (key == "--warmup") o.warmup = static_cast<int>(n);
    }
    if (o.transfer_buffer != "auto" && o.transfer_buffer != "min")
        throw std::runtime_error("--transfer-buffer must be auto or min");
    if (initial_state_selected && (o.job != "measure" || o.smoke || o.check_hadamard || o.check_measure))
        throw std::runtime_error("--initial-state is only valid with --job measure, not check modes");
    if (o.help) return o;
    if (static_cast<int>(o.smoke) + static_cast<int>(o.check_hadamard) + static_cast<int>(o.check_measure) +
        static_cast<int>(!o.job.empty()) != 1)
        throw std::runtime_error("Select exactly one of --job hadamard, --job measure, --smoke-check, --check-hadamard, or --check-measure");
    if (!o.job.empty() && o.job != "hadamard" && o.job != "measure")
        throw std::runtime_error("--job must be hadamard or measure");
    // Signed cuStateVec indices have 63 value bits. Byte arithmetic is checked separately.
    if (o.nqubits < 1 || o.nqubits > 62) throw std::runtime_error("--nqubits must be in [1,62]");
    if (o.repeats < 1) throw std::runtime_error("--repeats must be positive");
    if (o.p2p != "auto" && o.p2p != "off" && o.p2p != "required")
        throw std::runtime_error("--p2p must be auto, off, or required");
    if (o.communicator != "auto" && o.communicator != "openmpi" &&
        o.communicator != "mpich" && o.communicator != "external")
        throw std::runtime_error("--communicator must be auto, openmpi, mpich, or external");
    if (o.communicator == "external" && o.communicator_library.empty())
        throw std::runtime_error("external communicator requires --communicator-library /path/to/plugin.so");
    return o;
}
void print_help() {
    std::cerr << "Usage: custatevec_bench (--job hadamard|measure | --smoke-check | --check-hadamard | --check-measure) [options]\n"
              << "  --nqubits N       default 31; checks default max(8,log2(ranks)+1), cap 20\n"
              << "  --repeats R       measured repetitions (default 128)\n"
              << "  --warmup W        separate warm-ups (default 0)\n"
              << "  --initial-state uniform|flat|zero|ghz  --job measure only (default uniform; flat aliases uniform)\n"
              << "  --seed S          reproducible uint64 seed (default 1)\n"
              << "  --communicator auto|openmpi|mpich|external\n"
              << "  --communicator-library PATH  MPI library or external communicator plugin\n"
              << "  --p2p auto|off|required  explicit same-node CUDA IPC P2P (default auto)\n"
              << "    off uses only the communicator; MPI may still use CUDA IPC internally\n"
              << "    required aborts before timing if a needed same-node swap pair is unavailable\n"
              << "  --transfer-buffer auto|min  common power-of-two swap workspace (default auto)\n"
              << "  --transfer-buffer-fraction F  fraction of remaining free bytes, (0,0.95], default 0.95\n"
              << "    rounding and the least-memory rank can substantially reduce the allocation\n"
              << "Hadamard, full-register measurement, and separate untimed checks are implemented.\n";
}
} // namespace bench

int main(int argc, char** argv) {
    bench::DeviceAssignment assignment;
    try {
        assignment = bench::initialize_cuda_before_mpi();
    } catch (const std::exception& error) {
        std::cerr << "[pre-MPI] context_initialization=failed fatal: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    const int initialized = MPI_Init(&argc, &argv);
    if (initialized != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed\n";
        return EXIT_FAILURE;
    }
    MPI_TRY(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    try {
        auto options = bench::parse_options(argc, argv);
        int rank = 0, ranks = 0;
        MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
        if (options.help) {
            if (rank == 0) bench::print_help();
        } else {
            const bool checking = options.smoke || options.check_hadamard || options.check_measure;
            if (checking && !options.explicit_nqubits) {
                int bits = 0;
                for (int p = ranks; p > 1; p >>= 1) ++bits;
                options.nqubits = bits + 1 > 8 ? bits + 1 : 8;
            }
            if (checking && options.nqubits > 20)
                bench::fail("Untimed checks are limited to 20 total qubits; use a smaller MPI launch/state");
            bench::Topology topology(options.nqubits);
            // Reject placeholders before allocating a production-size state or emitting CSV.
            auto job = checking ? nullptr : bench::make_job(options.job, options.initial_state);
            bench::Runtime runtime(options, topology, assignment);
            if (options.smoke) runtime.smoke_check();
            else if (options.check_hadamard) bench::check_hadamard(runtime);
            else if (options.check_measure) bench::check_measure(runtime);
            else bench::run_benchmark(runtime, *job, options);
            runtime.close();
            topology.close();
        }
    } catch (const std::exception& error) {
        bench::fail(error.what());
    } catch (...) {
        bench::fail("Unknown failure");
    }
    MPI_TRY(MPI_Finalize());
    return EXIT_SUCCESS;
}
