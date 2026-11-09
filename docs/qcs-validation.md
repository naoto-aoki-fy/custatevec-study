# Final QCS compatibility review

Review date: 2026-11-09. Acceptance reference: `jhpc-quantum/rqs-svg` revision
`10a4da5b29a65f3eec0f556344ce06c3ff24072a`, available locally as
`/tmp/qcs-reference-10a4da5`. Its HEAD and the unchanged header/caller files were
verified. No reference GPU executable was available. This report supersedes
historical deferred-check statements in the earlier task reports.

## Verdict and coverage

All 139 public declarations (72 C plus 67 C++ declarations, counting both throwing
allocation overloads) have implementations. The inline two-argument C++ convenience
allocation overload is checked separately. There are no remaining intentional
unsupported placeholders. **Runtime compatibility is not yet accepted:** every
API family's real CUDA/MPI execution remains pending, independently of source and
host-test coverage. See the [exhaustive inventory](qcs-api-compatibility.md).

| Area | Evidence obtained | Remaining acceptance work |
| --- | --- | --- |
| Types, signatures, enums, linkage | Byte-identical reference header; real C11/C++17 object compilation; all function address probes; width/signedness/enum assertions | Actual shared-library link and dynamic-export inspection |
| Reference consumers | GHZ and Hadamard/measurement C examples, Pauli C helper and C++ public API test compile unchanged | Link and execute against real `libqcs.so`; compare with reference execution |
| Initializers and cleanup | Source review, existing C/C++ integration fixtures | All six initializers, raw norms, CURAND restoration, owned/borrowed MPI, teardown and failure paths on GPU |
| Gates and controls | 34 reference formulas, four SWAP widths, four angle fixtures, 2,400 localization plans pass on host | CUDA numerical checks, local/global controls, MPI swapping and capacity limits |
| Measurement/reset | 17,280 independent conditional-mass/order fixtures pass on host | Partial projection, subsequent operations, cached results, clbits, RNG, L=0 and distributed sampling |
| Expectations | 128 independent complex tensor-product comparisons and identity checks pass on host | Public CUDA path, measurement preservation, retained mapping, zero-local-capacity rejection and errors |
| Files | Save logical order and load physical order inspected independently in pinned source; test fixtures present | Native complex128 file precision/order, all rank slices, measured-state output, coordinated I/O failures |
| Events | Source checks match blocking events, compute-stream recording, waits, float MPI_MAX milliseconds converted to double seconds | Real event synchronization, timing/aggregation, ID narrowing and error checks |
| Benchmark | Six existing/new host tests and launch-script argument tests pass; source invariants reviewed | Real CLI/check modes, initial-state distributions, CSV and timing regression runs |

C wrappers return 1/0 and contain callback exceptions; throwing C++ forms retain
normal C++ linkage. Expectation results are written only on success. Narrow C
getters/event creation intentionally use `bit_t`, not `int*`. Defensive validation,
idempotent cleanup, poisoned-state tracking and externally owned MPI handling are
robustness extensions described in the library guide; reference undefined behavior
is not used as a failure oracle. No source edits were required for the four
representative reference consumers during object compilation. Any link/runtime
source-compatibility gaps remain unknown until actual execution.

Expectation uses host staging after cuStateVec distributed localization, with local
host memory/transfer cost proportional to the vector. Bitwise random samples across
the reference and cuStateVec backends are not promised. QCS currently uses the MPI
swap path without explicit CUDA IPC imports. Supported-by-design configurations are
64-bit Linux, C++17/CUDA, classic cuStateVec distributed swap APIs, CURAND,
CUDA-aware Open MPI/MPICH (or a matching external communicator plugin), power-of-two
rank count, one distinct GPU per rank, and N>=log2(P). N>=1; L=0 scalar slices are
supported for operations whose nonidentity target support fits L. These are intended
configurations, not tested cluster certifications.

Review fixes drain queued observation readbacks before host buffers can unwind on
CUDA errors. Save now poisons the handle after projection/readback device failure,
consistent with other mutating state paths. Ordinary file-open/write failures still
report through the coordinated C/C++ error path. Those device-failure paths require
GPU fault-injection validation; host tests cannot establish their safety.

## Commands actually run

GCC/G++ 11.5.0 on this host. The following commands passed:

```sh
python3 tests/check_qcs_header.py --reference-root /tmp/qcs-reference-10a4da5
# 72 C and 139 combined C++ address/type probes, inline overload, type/enum
# assertions, pinned SHA-256; four unchanged reference consumers compile to objects.
for source in tests/*_test.cpp; do
  c++ -std=c++17 -Wall -Wextra -Werror -Wno-unknown-pragmas -Iinclude "$source" -o /tmp/qcs-final-host-test && /tmp/qcs-final-host-test || exit
done
# Six executables: device assignment, transfer policy, Hadamard ordering,
# gate oracle/localization, measurement oracle, complex Pauli contraction.
python3 tests/check_launch_scripts.py
# Three launch wrappers: defaults, overrides, delimiter handling and spaced paths.
make help
git diff --check
```

`make library` was attempted and failed at the intended missing-`config.mk`
validation. `nvcc`, `mpicxx`, `mpiexec` and `nvidia-smi` are unavailable; CUDA and
cuStateVec headers are absent at their default installation paths. The reference
implementation requires the same absent GPU environment (plus its NCCL/CCCL
requirements). No real CUDA library or benchmark executable was built or run, no
real shared-library symbols were inspected, and no sanitizer results are claimed.
Temporary nonfunctional declaration shims also passed syntax checks of the
reviewed `qcs.cu` (launch syntax removed) and `qcs_observation_test.cu`; these are
syntax evidence only, not installed-API, device-failure or numerical validation.
The existing `qcs_c_caller.c` and `qcs_measurement_c.c` also passed C11
`-Wall -Wextra -Werror -fsyntax-only` checks.

The benchmark implementation files `src/main.cu`, `src/runtime.cu`,
`src/benchmark.cu`, `include/bench.hpp`, `scripts/`, and `SPEC.md` have no diff from
inventory starting revision `7ca8e923e63d39bc92d207cebfb5fbd985fc999c`:

```sh
git diff 7ca8e923e63d39bc92d207cebfb5fbd985fc999c -- src/main.cu src/runtime.cu src/benchmark.cu include/bench.hpp scripts SPEC.md
```

Source review confirms zero-state Hadamard setup, normalized benchmark measurement
initializers (separate from raw QCS initializers), complex128/FP64, descending
logical bit output, six-column CSV, 17-digit elapsed output and unchanged CLI.
The harness resets mapping/initializes/synchronizes/barriers before MPI_Wtime,
executes and synchronizes before stopping, then uses MPI_DOUBLE MAX. Public CUDA
event timing is separate. Source inspection is not numerical or timing validation.

## Exact target-cluster reproduction

Run from the repository root in a scratch checkout on a shared writable filesystem;
the observation fixture creates/removes `statevector_input.bin`. Set `REF` to a
clean checkout at the pinned revision. Load the site's matching CUDA, cuStateVec,
CURAND and CUDA-aware MPI environment and assign distinct GPUs. Replace the two
installation paths below with existing site paths; do not generate `config.mk`.

```sh
export REF=/path/to/rqs-svg
export CONFIG_MK=/path/to/existing/config.mk
export CUSTATEVEC_ROOT=/path/to/cuquantum
make CONFIG_MK="$CONFIG_MK" CUSTATEVEC_ROOT="$CUSTATEVEC_ROOT" all library \
  build/qcs_foundation_test build/qcs_gate_test build/qcs_measurement_test build/qcs_observation_test
export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
python3 tests/check_qcs_header.py --reference-root "$REF" --library "$PWD/libqcs.so"
# This performs real linking and nm -D coverage for all required C/C++ symbols.
for ranks in 1 2 4; do
  for suite in foundation gate measurement observation; do
    timeout 600s mpiexec -n "$ranks" "./build/qcs_${suite}_test"
  done
  timeout 600s mpiexec -n "$ranks" ./build/qcs_foundation_test --borrowed
done

# Unchanged representative consumers; build configuration is the only adaptation.
cc -std=c11 -I"$REF/include" "$REF/examples/sharedlibrary/ghz.c" -L. -lqcs -Wl,-rpath,"$PWD" -o /tmp/qcs-reference-ghz
cc -std=c11 -I"$REF/include" "$REF/examples/sharedlibrary/hadamard-measure.c" -L. -lqcs -Wl,-rpath,"$PWD" -o /tmp/qcs-reference-hm
cc -std=c11 -I"$REF/include" -c "$REF/tests/pauli_api_c.c" -o /tmp/qcs-reference-pauli-c.o
c++ -std=c++17 -I"$REF/include" "$REF/tests/pauli_api_test.cpp" /tmp/qcs-reference-pauli-c.o -L. -lqcs -Wl,-rpath,"$PWD" -o /tmp/qcs-reference-pauli
for ranks in 1 2 4; do
  timeout 600s mpiexec -n "$ranks" /tmp/qcs-reference-ghz --num-qubits 8 --num-samples 16
  timeout 600s mpiexec -n "$ranks" /tmp/qcs-reference-hm --num-qubits 8 --num-samples 16
  timeout 600s mpiexec -n "$ranks" /tmp/qcs-reference-pauli
done

# Benchmark correctness and CSV smoke runs, distinct from API event timing.
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
for tool in memcheck racecheck synccheck; do
  timeout 600s mpiexec -n 2 compute-sanitizer --tool "$tool" ./build/qcs_observation_test
done
```

Repeat across nodes and noncontiguous world-rank placements, with shared and
per-rank visibility and matching communicator ABIs. Also run benchmark P2P modes
and transfer-buffer A/B procedures from [benchmark documentation](benchmark.md).
Record installed versions, placement, exit codes, stdout/stderr and sanitizer logs.
Compare deterministic reference results, raw amplitudes and distributions; avoid
requiring bitwise matching stochastic trajectories or timing magnitudes. OOM,
rank-local file/device failures and transport failure injection remain manual
cluster acceptance checks. No runtime/distributed compatibility claim follows
from successful compilation alone.
