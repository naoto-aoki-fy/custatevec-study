#include "bench.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace bench {
std::unique_ptr<Job> make_job(const std::string& name) {
    throw std::runtime_error("Job '" + name + "' is not implemented in Stage 1; no timings produced");
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
