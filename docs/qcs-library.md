# QCS library foundation

`libqcs.so` implements the foundation of the public API pinned in
[qcs-api-compatibility.md](qcs-api-compatibility.md). The installed header is
`include/qcs.h`, byte-identical to reference revision
`10a4da5b29a65f3eec0f556344ce06c3ff24072a`. Full circuit simulation through this
interface is not implemented yet: gates, reset/measurement, expectations, save,
and timing events currently throw in C++ or return 0/report the callback in C.

## Build and link

Use the same external nvccoptions configuration and cuStateVec installation as the
benchmark, plus CURAND from CUDA. No MPI wrapper flags are discovered separately.
The default `make` target still builds `custatevec_bench`.

```sh
make library CONFIG_MK=/path/to/config.mk CUSTATEVEC_ROOT=/path/to/cuquantum
cc -std=c11 -Iinclude -c caller.c -o caller.o
cc caller.o -L. -lqcs -Wl,-rpath,"$PWD" -o caller
# C++ callers can use c++ instead of cc with the same include/link options.
```

Create a handle, configure qubits/clbits and optional physical-to-logical mapping,
allocate, then explicitly initialize. C returns 1 for success and 0 for failure;
register `qcs_set_exception_callback` for diagnostics. C++ `_cxx` functions throw.
The callback receives a borrowed message and its length without NUL; callback
exceptions are contained. All calls and callback registration must be serialized
process-wide, on a thread allowed by the application's MPI thread level. All ranks
must use the same live collective handles and call order. Invalid or missing handles
on only some ranks cannot be repaired by a collective API.

## Lifetimes and state

Create selects/initializes CUDA using the benchmark's launcher/visibility policy,
checks node-local rank consistency and duplicate UUIDs, creates a private duplicated
world communicator, compute stream and cuStateVec handle. It does not allocate state.
Library operations restore the previous current CUDA device. No CUDA context reset
or caller-owned resource destruction occurs. With multiple visible GPUs, provide
Open MPI/direct-Slurm local-rank metadata; otherwise expose one distinct GPU per rank.

If MPI is uninitialized the library initializes it, after attempting CUDA setup on
every rank, and finalizes after the last core is disposed. Even a rank with early
CUDA failure joins MPI initialization and error agreement, allowing a coordinated
failure instead of abandoning peers at setup. Applications borrowing preinitialized
MPI are responsible for pre-MPI CUDA setup if their transport requires it. Borrowed
MPI is never finalized. MPI cannot restart after finalization; use application-owned
MPI for repeated create/destroy sessions. Concurrent live handles hold separate
leases; operations on them remain serialized and collectively ordered.

Dispose releases the core's resources/lease and is idempotent; destroy also deletes
the wrapper, and destroy(NULL) succeeds. Destroy after dispose is safe. Do not use
a deleted wrapper or externally finalize MPI while cores still exist. Metadata
vectors remain owned by the disposed wrapper until destruction.

Allocation uses complex128 storage, leaves its contents unspecified/uninitialized,
reserves the cuStateVec default workspace, and remembers the selected initializer.
It never calls a setter or reads a file. Reallocation/resizing allocated qubits is
rejected. N is 1..63 with N>=log2(P); P is a power of two. L=0 scalar slices are
permitted. Checked state-byte sizing and actual allocation success further constrain
capacity. This foundation does not execute signed-index cuStateVec gate operations
at N=63; their limits remain future acceptance work.

All six initializers preserve raw reference amplitudes: zero, all-ones flat,
physical index sequential, unnormalized entangled endpoints (only index 0 at P=1),
CURAND normal real/imaginary amplitudes, or the physical rank slice of
`statevector_input.bin`. The file is native real/imaginary double pairs without a
header. The selected file is reread on every `set_zero_state`; direct setters do not
change the remembered selection. The initialization seed is broadcast once and
retained for random restoration, independently of the future measurement cursor.
Initialization clears measured/pending masks without changing clbits or mapping.

`set_mapping` validates and remembers a physical-to-logical permutation. Allocation
or reinitialization installs it and builds the inverse without moving amplitudes.
Clbit resize retains existing values and zeroes new entries. Arrays are ascending
clbit index, strings descending index plus NUL; the caller supplies enough storage.
The narrow C getter output types intentionally retain modulo-256 conversion.

Private owners already identify mapping, classical bits, measured/pending masks,
RNG cursor, workspace and event resources. No swap worker, IPC imports or event
handles are created in this stage. Later stages must reserve/reprepare all operation
workspace and peers independently of the first gate; the benchmark's prepared
workspace limit is unchanged. Placeholder operations never consume RNG, allocate
resources or return successful results.

## Distributed errors

Create, allocation, mapping reinitialization, initialization and disposal are
collective. Local configuration/getter/clbit calls and placeholders do not communicate.
Allocation validates matching dimensions, initializer and mapping before device work.
Each local setup/allocation/file/device stage catches exceptions, agrees on the first
failing rank, broadcasts its bounded diagnostic, and throws on all ranks before any
next dependent phase. Failed candidate allocations are released on successful ranks;
file reads all succeed before state writes begin. File-read failures preserve the
previous state and can be retried. Device-write failures poison the handle on every
rank: further state operations are rejected; metadata queries and disposal remain available. Cleanup drains work, barriers, destroys
resources and communicators, then releases the lease, including on reported cleanup
errors. Cleanup failures may leave the CUDA context unusable.

Transport errors, failed caller-device restoration and failed allocation rollback use MPI_Abort because reliable
agreement/recovery is no longer assured. GPU/transport hangs and process loss require
a launcher timeout; ordinary MPI provides no bounded failure recovery. Calling an
operation on only some ranks is outside the contract. No library path calls the
benchmark's aborting `bench::fail`, and benchmark source/timing/numerics are unchanged.

## Checks and execution status

Commands run successfully in the implementation environment:

```sh
python3 tests/check_qcs_header.py
cc -std=c11 -Wall -Wextra -Werror -Iinclude -fsyntax-only tests/qcs_c_caller.c
c++ -std=c++17 -Iinclude tests/device_assignment_test.cpp -o /tmp/qcs-device-test
/tmp/qcs-device-test
c++ -std=c++17 -Iinclude tests/transfer_policy_test.cpp -o /tmp/qcs-transfer-test
/tmp/qcs-transfer-test
c++ -std=c++17 -Iinclude tests/hadamard_order_test.cpp -o /tmp/qcs-order-test
/tmp/qcs-order-test
make help
```

The header check compiles address/type probes for all 72 C and 67 C++ declarations
and the inline overload, and verifies the pinned header SHA-256. A temporary host
syntax shim also checked C++ syntax of the implementation with the CUDA launch
syntax removed; C/C++ address probes also linked to that temporary shim and
the integration test passed shim-based syntax checking. These are not CUDA
compilation, MPI/GPU execution or installed-API validation.
`make library` was attempted and correctly rejected missing `config.mk`. Neither
`nvcc` nor `mpicxx` is on PATH here. No real library link, CUDA initializer execution,
MPI failure test or benchmark GPU regression has run.

On a configured GPU/MPI system, run:

```sh
make library build/qcs_foundation_test CONFIG_MK=/path/to/config.mk CUSTATEVEC_ROOT=/path/to/cuquantum
python3 tests/check_qcs_header.py --library ./libqcs.so
export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
timeout 120s mpiexec -n 1 ./build/qcs_foundation_test
timeout 120s mpiexec -n 1 ./build/qcs_foundation_test --borrowed
timeout 120s mpiexec -n 2 ./build/qcs_foundation_test
timeout 120s mpiexec -n 4 ./build/qcs_foundation_test --borrowed
```

Use a shared writable working directory for multi-node tests: the fixture creates
an isolated directory there and removes it on success. Use scheduler placement to
assign one GPU per rank. The test covers C/C++ error reporting, allocation variants,
all initializer patterns/restoration, physical mapping, narrow getters, clbit
preservation, multiple handles/ownership, repeated dispose and a file unavailable
on only rank zero. Run the existing benchmark checks from README separately,
including multi-node transport. OOM/transport fault injection, sanitizers, caller
device restoration failures and target installed-version validation remain required
cluster checks; no such results are claimed here.
