#include "bench.hpp"

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
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { o.help = true; continue; }
        if (key == "--smoke-check") { o.smoke = true; continue; }
        if (key != "--job" && key != "--nqubits" && key != "--repeats" &&
            key != "--warmup" && key != "--seed" && key != "--communicator" &&
            key != "--communicator-library")
            throw std::runtime_error("Unknown option: " + key);
        if (++i == argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[i];
        if (key == "--job") { o.job = value; continue; }
        if (key == "--communicator") { o.communicator = value; continue; }
        if (key == "--communicator-library") { o.communicator_library = value; continue; }
        const auto n = number(value, key);
        if (key == "--seed") { o.seed = n; continue; }
        if (n > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error(key + " exceeds INT_MAX");
        if (key == "--nqubits") { o.nqubits = static_cast<int>(n); o.explicit_nqubits = true; }
        if (key == "--repeats") o.repeats = static_cast<int>(n);
        if (key == "--warmup") o.warmup = static_cast<int>(n);
    }
    if (o.help) return o;
    if (o.smoke == !o.job.empty())
        throw std::runtime_error("Select exactly one of --job hadamard, --job measure, or --smoke-check");
    if (!o.job.empty() && o.job != "hadamard" && o.job != "measure")
        throw std::runtime_error("--job must be hadamard or measure");
    // Signed cuStateVec indices have 63 value bits. Byte arithmetic is checked separately.
    if (o.nqubits < 1 || o.nqubits > 62) throw std::runtime_error("--nqubits must be in [1,62]");
    if (o.repeats < 1) throw std::runtime_error("--repeats must be positive");
    if (o.communicator != "auto" && o.communicator != "openmpi" &&
        o.communicator != "mpich" && o.communicator != "external")
        throw std::runtime_error("--communicator must be auto, openmpi, mpich, or external");
    if (o.communicator == "external" && o.communicator_library.empty())
        throw std::runtime_error("external communicator requires --communicator-library /path/to/plugin.so");
    return o;
}
void print_help() {
    std::cerr << "Usage: custatevec_bench (--job hadamard|measure | --smoke-check) [options]\n"
              << "  --nqubits N       default 31; smoke default max(8,log2(ranks)+1), cap 20\n"
              << "  --repeats R       measured repetitions (default 128)\n"
              << "  --warmup W        separate warm-ups (default 0)\n"
              << "  --seed S          reproducible uint64 seed (default 1)\n"
              << "  --communicator auto|openmpi|mpich|external\n"
              << "  --communicator-library PATH  MPI library or external communicator plugin\n"
              << "Stage 1 implements only the untimed smoke check. Both jobs fail explicitly.\n";
}
} // namespace bench

int main(int argc, char** argv) {
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
            if (options.smoke && !options.explicit_nqubits) {
                int bits = 0;
                for (int p = ranks; p > 1; p >>= 1) ++bits;
                options.nqubits = bits + 1 > 8 ? bits + 1 : 8;
            }
            if (options.smoke && options.nqubits > 20)
                bench::fail("Smoke check is limited to 20 total qubits; use a smaller MPI launch/state");
            bench::Topology topology(options.nqubits);
            // Reject placeholders before allocating a production-size state or emitting CSV.
            auto job = options.smoke ? nullptr : bench::make_job(options.job);
            bench::Runtime runtime(options, topology);
            if (options.smoke) runtime.smoke_check();
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
