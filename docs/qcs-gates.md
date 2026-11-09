# QCS gate implementation

The library implements all 34 throwing gate functions, their 34 C counterparts,
and C-only SWAP1..4. The public header remains byte-identical to reference commit
`10a4da5b29a65f3eec0f556344ce06c3ff24072a`. [Measurement/reset](qcs-measurement.md) now have public implementations. Expectations,
statevector saving and timing events remain explicit failures for later tasks.

## Semantics and implementation

The specification is the pinned reference's
[gate apply functions](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/src/qcs.cu#L307),
[gate preparation](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/src/qcs.cu#L3260),
and [public declarations](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/include/qcs.h).
Reference measurement/Pauli tests supply representative H/X/S and controlled-X
callers; a separate exhaustive reference gate suite was not found. The test oracle
adapts the pinned apply functions for CPU execution, independently of the new
matrix builder, preserving sparse writes and phases.

`src/qcs_gate.hpp` constructs row-major complex-double matrices. Target list element
zero is the least significant matrix basis bit; target lists specify operands of
one gate, not a broadcast over independent gates. Angles are radians in reference
parameter order. `u=u3`, `p=u1`, and `u4` includes its explicit global phase. RCCX
and RCCCX retain the reference's relative phases. `id` still takes one target and
follows localization/measurement-metadata handling. Global phase takes zero targets
and applies its scalar matrix only where all controls match, including L=0 slices.

Application uses cuStateVec with `CUDA_C_64F`, row-major layout, no adjoint, and
`CUSTATEVEC_COMPUTE_64F`. Workspace is queried for the actual matrix, target width
and local control count, using the documented
[ApplyMatrix APIs](https://docs.nvidia.com/cuda/cuquantum/26.03.2/custatevec/api-reference/custatevec/index.html#custatevecapplymatrix).
No custom gate-computation kernel or CPU simulation path is used in production.
SWAP accepts 2/4/6/8 disjoint operands and pairs consecutive entries. All operands
are localized together, then disjoint pairs use two-target cuStateVec matrices with
the same controls. This preserves the reference's full target-width restriction
and final mapping without allocating an unnecessary eight-target dense matrix.

Gate validation rejects incorrect counts, nonempty NULL arrays, invalid indices,
duplicates and target/control overlaps before state changes. Negative controls
match zero, positive controls one. Valid controls may occupy either local or rank
bits. Explicit control contradictions/overlaps are rejected as invalid operands;
a contradiction with an already-recorded measured non-target is a reference no-op.
Reference assertions/unchecked invalid accesses are replaced with safe failure.
Numeric argument bit patterns are compared across ranks; no extra finite-angle
precondition is imposed on the reference interface.

## Mapping and distributed resource preparation

The library retains both `p2l[physical]=logical` and its inverse `l2p[logical]=physical`.
It does not confuse the public mapping direction with the benchmark's l2p vector.
For each global target in caller order, choose a descending local physical victim
that is not any original local target. Compute the complete plan before touching
state; exchange each disjoint physical local/global pair, then install both inverse
maps. Mapping changes persist after the gate. Local targets later in the operand
list are protected, so physical bit zero is not a universal victim.

Communication reuses the benchmark's classic cuStateVec scheduler/worker protocol,
with a dedicated nonblocking transfer stream. Finish compute work before exchange,
execute every scheduler batch, and finish transfer work before updating mappings
or applying a matrix. Every rank participates even when its global controls exclude
local computation. Controls are translated using the post-localization mapping.

Workers are prepared lazily on the first operation needing exchange, independently
of which API was called first. An earlier local-only gate or initializer does not
lock out later distributed operations. Preparation reserves worker extra workspace,
then chooses the common smallest power-of-two transfer buffer satisfying every
worker minimum within 95% of the least remaining free-memory budget. This reuses
`transfer_policy.hpp`'s checked budgeting/minimum selection. Allocation failure is
agreed and all partially prepared workers/buffers are released before returning.
Since the candidate is already the minimum, no smaller valid retry is possible.

The library deliberately uses the MPI communicator transport for every swap pair;
there are no explicit CUDA IPC imports or job-specific peer lists. CUDA-aware MPI
may optimize same-node communication itself. Open MPI/MPICH ABI selection uses
`mpi.h` macros. For other ABIs set `QCS_CUSTATEVEC_COMMUNICATOR_LIBRARY` identically
on every rank to a cuStateVec external communicator plugin built against that MPI
(the plugin must wrap MPI_COMM_WORLD, not be an arbitrary libmpi). Calls on multiple
handles remain serialized; no subcommunicator API is introduced.

Gate workspace is a separate persistent allocation, never borrowed by a worker.
Before each operation, query and grow it transactionally, retaining old scratch on
allocation failure. Growth occurs between synchronous operations, not behind an
active worker. The minimum transfer buffer leaves remaining memory available for
later widths/control configurations; insufficient memory returns failure without
starting the operation. Worker/state addresses remain stable. There is no inherited
benchmark `preparation_finalized` prohibition in this API path. The benchmark's
closed Job::prepare policy and adaptive transfer defaults are unchanged.

Worker execution/transfer failures abort MPI rather than unwinding past a peer
still executing a worker. Preflight, matrix preparation and memory errors use the
foundation's coordinated status/diagnostic phases. A local compute/projection failure
is agreed and poisons the handle on all ranks. Disposal completes both streams,
barriers, destroys the worker/scheduler/communicator before their buffers/events,
then destroys ordinary cuStateVec/device resources. Hangs/process loss still need
a launcher timeout. No new implementation is claimed for MPI fault tolerance.

Supported gate widths require at least as many local qubits as the entire target
list: 1 for single-target gates, 2 for two-target gates, 3/4 for RCCX/RCCCX and
2K for K SWAP pairs. Controls do not each consume a local target slot. Zero-target
global phase can operate with no local qubits. All allocations and the installed
cuStateVec release impose additional capacity/API constraints; N=63 is not a
hardware-capacity promise. No installed-version CUDA build is available here.

## Measurement integration

Measured masks, values and pending projections are logical metadata shared with
the foundation. The gate path already handles reference-style metadata:

- Keep measured non-targets as implicit controls. A contrary explicit control
  returns without localization, amplitude changes, RNG consumption or cache release.
- Localize targets, then materialize only pending projections of measured targets
  using the updated mapping. Projection zeros rejected amplitudes without rescaling.
- After successful gate completion, release target measurement records; preserve
  measured/pending non-targets, clbits and RNG cursor.

Private-mask integration fixtures cover target projection and a contradictory
cached control. The measurement task adds actual public APIs and integration test sources for
these invariants. GPU execution remains pending. The acceptance requirements include local/global cached controls, target/non-target projections, repeated
measure-gate-measure, id/global phase, multiple SWAP pairs, raw expectations and
save. Compare against the pinned deferred-projection behavior and verify no implicit
normalization or extra draw. See [measurement validation](qcs-measurement.md) for implemented checks and
remaining GPU/MPI execution requirements.

## Validation

Executed successfully:

```sh
c++ -std=c++17 -Wall -Wextra -Wno-unknown-pragmas -Iinclude tests/qcs_gate_host_test.cpp -o /tmp/qcs-gate-host-test
/tmp/qcs-gate-host-test
python3 tests/check_qcs_header.py
```

The numerical host test checks every matrix column and asymmetric complex vectors
against the pinned CPU oracle at four angle fixtures. All 34 gate families passed,
as did all four SWAP-width decompositions and 2,400 localization plans over all five-qubit permutations and ordered target
pairs. Nonfunctional temporary CUDA/MPI declaration shims also allowed host syntax
checking of the new CUDA sources and linking of every public symbol. Those checks
are not validation of installed CUDA API signatures, backend numerics or transport.
Existing host device-assignment/transfer-policy/Hadamard-order tests also passed.

The new GPU test compares full complex amplitudes, not probabilities or states
modulo global phase, against a small logical-state CPU oracle. It includes all C
and C++ gate families; unnormalized asymmetric inputs; three target placements;
positive/negative/combined/no controls; identity/reverse mappings; global targets
and controls; numbered/general SWAP widths 1..4; sequential mixed-width calls;
one-rank invalid operands; private measurement-mask integration; and a scalar
slice/one-local-bit limit check. Its default successful-case count is 832, plus
sequence, integration and rejection checks. It is intended for P=1,2,4; each rank
owns ten local qubits in the primary test.

Not run: real CUDA build/link, GPU numerical tests, MPI transport/teardown, memory
failure injection, sanitizer or performance checks. `nvcc` and `mpicxx` are absent
from PATH and no site config.mk is supplied. `make library` still reports the
missing external configuration. On a configured target system:

```sh
make library build/qcs_gate_test CONFIG_MK=/path/to/config.mk CUSTATEVEC_ROOT=/path/to/cuquantum
python3 tests/check_qcs_header.py --library ./libqcs.so
export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
timeout 180s mpiexec -n 1 ./build/qcs_gate_test
timeout 180s mpiexec -n 2 ./build/qcs_gate_test
timeout 180s mpiexec -n 4 ./build/qcs_gate_test
# Repeat two/four-rank checks on multiple nodes and noncontiguous rank placement.
# Run the existing foundation and benchmark checks from qcs-library.md/README.md.
```

Use one distinct GPU per process and target-site placement flags; test both shared
and per-process CUDA visibility. Run Compute Sanitizer memcheck/racecheck/synccheck
on single- and multiple-rank cases. Record installed CUDA/cuStateVec/MPI versions,
plugin and placement. Run allocation failure tests before/after first worker setup,
then verify retry/disposal, and force a transfer error under a timeout. No numerical
GPU agreement, throughput improvement, P2P topology or multi-node success is claimed.
