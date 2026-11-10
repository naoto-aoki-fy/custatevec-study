# Distributed cuStateVec benchmark and QCS library

C++17/CUDA/MPI statevector benchmarks and a C/C++ QCS compatibility library pinned
to `rqs-svg` revision `10a4da5b29a65f3eec0f556344ce06c3ff24072a`.

Requires 64-bit Linux, nvcc, classic cuStateVec, CUDA-aware MPI, and one distinct
GPU per process. The QCS library also requires CURAND. Supply an existing
[nvccoptions](https://github.com/naoto-aoki-fy/nvccoptions) configuration; the build
never generates or overwrites `config.mk`.

```sh
make CONFIG_MK=/path/to/config.mk CUSTATEVEC_ROOT=/path/to/cuquantum
make library CONFIG_MK=/path/to/config.mk CUSTATEVEC_ROOT=/path/to/cuquantum
mpiexec -n 2 ./custatevec_bench --job hadamard --nqubits 8 --repeats 1
```

Timings use CUDA events for complete repetitions, reported in seconds with MPI_MAX
across ranks; host submission gaps and MPI waits between markers may contribute.

- [Benchmark setup, CLI, launch scripts and checks](docs/benchmark.md)
- [Benchmark contract](SPEC.md)
- [Imported example builds, execution and validation](docs/qcs-examples.md)
- [Standalone QCS CLI and circuit compilation](docs/qcs-cli.md)
- [QCS build/link and API usage](docs/qcs-library.md)
- [Complete public API inventory](docs/qcs-api-compatibility.md)
- [Final review, executed checks, gaps and reproduction commands](docs/qcs-validation.md)

All public APIs have implementations. Host tests and reference-header consumer
compilation pass; real CUDA linking, GPU/MPI behavior and benchmark runtime
regressions remain unvalidated in the current environment.
