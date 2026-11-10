# Validation status and reproduction procedures

This document separates source inspection, host-side execution, and real CUDA/MPI
acceptance. Do not interpret source observations as execution evidence. The
benchmark requirements are defined in [SPEC.md](SPEC.md); implementation
observations and historical decisions are in [DEVELOPMENT.md](DEVELOPMENT.md).

## Current evidence

The benchmark change records state that its builds, tests, and validation were
skipped at the user's request. No benchmark execution evidence is recorded in
those stage reviews. Earlier QCS library review evidence below is distinct and
does not validate benchmark runtime behavior.

The QCS compatibility review dated 2026-11-09 reported the following host-side
checks passed on GCC/G++ 11.5.0:

- `tests/check_qcs_header.py` against the pinned reference: C and C++ declaration
  address/type probes, inline overload, type/enum assertions, pinned hash, and
  object compilation of unchanged reference consumers.
- Six host C++ fixtures covering device assignment, transfer policy, Hadamard
  ordering, gate oracle/localization, measurement oracle, and complex Pauli
  contraction.
- `tests/check_launch_scripts.py`, `make help`, and `git diff --check`.
- C11 syntax-only checks for two C callers.

`make library` was attempted and stopped at the intended missing `config.mk`
validation. The review reported `nvcc`, `mpicxx`, `mpiexec`, and `nvidia-smi`
unavailable at that time, and CUDA/cuStateVec headers absent at default paths.
These are historical observations, not a fresh probe of the current environment.
Temporary nonfunctional declaration shims allowed syntax checks only; they do
not establish installed API compatibility or device behavior.

Runtime acceptance is not established. No real CUDA library/benchmark executable
was built or run, dynamic exports were not inspected, and no sanitizer result is
available. GPU numerical behavior, MPI distribution, communicator compatibility,
stream/transport completion, and timing/CSV regressions remain unverified.
Likewise, source inspection showing implementations for all public QCS APIs does
not establish runtime compatibility.

## Requirements and known limitations

- Build and link with the target site's existing generated configuration and
  matching installed classic cuStateVec headers/libraries.
- Establish the communicator ABI/plugin and CUDA-aware MPI behavior for the
  actual launcher, including multi-node transport and non-contiguous rank
  placement.
- Run smoke, Hadamard, measurement, QCS API suites, and representative consumers
  on assigned GPUs; check correctness, completion, output, and teardown.
- Verify memory-capacity/OOM coordination, duplicate assignment, and launcher
  timeout behavior under target cluster conditions.
- Verify explicit CUDA IPC P2P fallback/required modes and transfer-buffer
  adaptive sizing on hardware. Source-level logic and host policy fixtures do
  not test IPC or worker behavior.
- Historical benchmark source reviews describe CUDA event timing as a marker
  interval that may include host submission gaps and MPI waits. This is the
  intended/implemented timing scope by inspection, not a measured performance
  result.

The pinned cuStateVec API selection was initially made from NVIDIA online
references because installed headers were not available in the recorded
workspace. Matching installed-release compatibility remains an explicit gap.
The implementation's stated constraints include 64-bit Linux, C++17/CUDA,
classic cuStateVec distributed swap support, CUDA-aware MPI, power-of-two rank
counts, one distinct GPU per rank, and sufficient local qubits. These are
intended compatibility requirements, not cluster certifications.

## Reproduction on the target cluster

Use a scratch checkout on shared writable storage, the matching site CUDA,
cuStateVec, CURAND, and CUDA-aware MPI environment, and an existing nvccoptions
configuration. Set `REF` to a clean checkout of the pinned `rqs-svg` revision.
Do not generate `config.mk` as part of validation.

```sh
export REF=/path/to/rqs-svg
export CONFIG_MK=/path/to/existing/config.mk
export CUSTATEVEC_ROOT=/path/to/cuquantum
make CONFIG_MK="$CONFIG_MK" CUSTATEVEC_ROOT="$CUSTATEVEC_ROOT" all library \
  build/qcs_foundation_test build/qcs_gate_test build/qcs_measurement_test build/qcs_observation_test
export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
python3 tests/check_qcs_header.py --reference-root "$REF" --library "$PWD/libqcs.so"

for ranks in 1 2 4; do
  for suite in foundation gate measurement observation; do
    timeout 600s mpiexec -n "$ranks" "./build/qcs_${suite}_test"
  done
  timeout 600s mpiexec -n "$ranks" ./build/qcs_foundation_test --borrowed
done

for ranks in 1 2 4; do
  timeout 600s mpiexec -n "$ranks" ./custatevec_bench --smoke-check --nqubits 8
  timeout 600s mpiexec -n "$ranks" ./custatevec_bench --check-hadamard --nqubits 8
  timeout 600s mpiexec -n "$ranks" ./custatevec_bench --check-measure --nqubits 8
  for job in hadamard hadamard-reverse; do
    timeout 600s mpiexec -n "$ranks" ./custatevec_bench --job "$job" --nqubits 8 --repeats 3 --warmup 1 --seed 1
  done
  for initial in uniform flat zero ghz; do
    timeout 600s mpiexec -n "$ranks" ./custatevec_bench --job measure --initial-state "$initial" --nqubits 8 --repeats 16 --warmup 1 --seed 1
  done
done
```

Also run the pinned reference consumers listed in the former QCS review, the
multi-node configurations, P2P modes, and sequential transfer-buffer policy A/B
runs described in [benchmark.md](benchmark.md). Preserve installed versions,
rank/GPU placement, complete stdout/stderr, exit codes, and sanitizer logs.
Exercise P=4,N=3 measurement checks to include zero-mass intermediate ranks.
Compare deterministic states and distributions without requiring stochastic
bitwise trajectory matching or specific timing magnitudes. Add site-specific
failure-injection and memory-boundary checks. Record results here only after
actual execution; none of these procedures is claimed to have run in this
reorganization.
