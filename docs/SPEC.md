# Distributed cuStateVec benchmark specification

## Internal requirement notes

The distributed-runtime section requires at least one local qubit, expressed as
`N > log2(P)`. The QCS compatibility section separately describes support for
`N >= log2(P)` and scalar local slices for applicable operations. These are
distinct scopes (benchmark runtime versus QCS library); do not treat the QCS
edge case as a relaxation of the benchmark constraint. Target-cluster behavior
for either boundary remains subject to validation.

This document is the authoritative intended contract. It does not assert that
the implementation meets the contract or that any build or runtime checks pass.
Implementation observations and execution evidence are tracked in
[DEVELOPMENT.md](DEVELOPMENT.md) and [VALIDATION.md](VALIDATION.md).

This is the common contract for all four sequential stages:
1. Build system, distributed runtime, and benchmark infrastructure.
2. Hadamard workload.
3. Full-register measurement workload.
4. Correctness verification and execution documentation.

Stage 1 supplies the runtime and timing harness. The workload factory must reject
unimplemented jobs without producing benchmark CSV or fabricated timings.
The separate, untimed `--smoke-check` is capped at 20 total qubits and defaults to
`max(8, log2(P)+1)` qubits. It is infrastructure checking, not a third benchmark.

## Overall benchmark requirements

The finished program must provide three independently executable jobs:

- `--job hadamard`: initialize an N-qubit state to |0>^N and apply one Hadamard gate to every logical qubit in ascending order q=0,...,N-1.
- `--job hadamard-reverse`: the same initialization and separate gate calls in descending logical order q=N-1,...,0. Share preparation and gate application with `hadamard`, including physical-bit-0 localization, retained final mapping, and identity mapping at the start of each repetition. Timing, warm-ups, precision, and CSV behavior are identical; record `hadamard-reverse` in the job column with empty measurement bits.
- `--job measure`: directly initialize the selected normalized N-qubit state (uniform by default) and perform one joint computational-basis measurement of the complete register, including state collapse.

`--check-hadamard` validates both orders with repeated zero-state runs, explicit
logical target order and gate count checks, and asymmetric complex-state prefix
and full-sequence comparisons against an independent CPU reference. Use both
single-rank and multi-rank launches, including one local bit, under all P2P policies.

Each invocation runs exactly one job. The jobs must be usable as separate MPI launches and separate scheduler jobs.

The production defaults are:
- N = 31 qubits.
- 128 independently initialized and separately timed repetitions per job.
- Zero warm-up repetitions.
- Double precision throughout.
- Quantum gate fusion disabled.

Allow smaller qubit counts and fewer repetitions for correctness tests. Provide `--nqubits`, `--repeats`, `--warmup`, and a reproducible `--seed` option.

Use complex128 state-vector storage (`CUDA_C_64F`), complex128 gate matrices, and `CUSTATEVEC_COMPUTE_64F` wherever a compute type is required. Probability calculations and their MPI reductions must use double precision.

Use the classic cuStateVec API for the baseline implementation, with explicit gate applications and distributed index-bit swaps.

## Measurement initial states

`--initial-state uniform|flat|zero|ghz` applies only to `--job measure`, with
`uniform` as the default when omitted. `flat` is an alias normalized to `uniform`,
including in the reported state. Reject invalid values and explicit use
with any other job or check mode. Report the selected state as
`initial_state=<value>` to stderr without adding CSV columns.

Initialize directly on the GPU in the canonical distributed layout before
every warm-up and measured repetition, using complex128 and no preparation gates:

- `uniform`: every global amplitude is `2^(-N/2)`.
- `zero`: only global index 0 has amplitude 1.
- `ghz`: only global indices 0 and `2^N - 1` have amplitude `1/sqrt(2)`.

All other amplitudes are zero. For P>1, the GHZ endpoints belong to ranks 0
and P-1; for P=1, both are local. Normalize the complete distributed state,
not each rank independently. Complete initialization outside timing. Preserve
measurement, collapse, RNG, warm-up seed domains, CSV, and Hadamard behavior.

Examples (target-environment commands, not execution results):

```sh
mpiexec -n 1 ./custatevec_bench --job measure --initial-state zero --nqubits 8 --repeats 4
mpiexec -n 2 ./custatevec_bench --job measure --initial-state ghz --nqubits 8 --warmup 2 --repeats 4
mpiexec -n 2 ./custatevec_bench --job measure --initial-state uniform --nqubits 8 --repeats 4
```

`--check-measure` must exercise all three direct GPU initializers independently
before each controlled draw. Check each rank's complete initial slice and the
global unit norm, compare outcomes with the CPU reference and across ranks,
and check the complete collapsed state and its global unit norm. Zero must
always yield all zeros. Controlled draws below and above 0.5 must exercise
both all-zero and all-one GHZ outcomes deterministically. Run this mode with
P=1, P=2, and P=4 (for example N=8,8,3 respectively), including zero-mass
intermediate ranks at P=4. Builds, tests, and validation for this change were
skipped at the user's request; CUDA/MPI execution remains unverified.

## Build system

Use GNU Make and nvcc for compilation and final linking.

The compiler, MPI, linker, and GPU architecture settings must come from an existing config.mk generated by:
[nvccoptions](https://github.com/naoto-aoki-fy/nvccoptions)

Read the nvccoptions README and inspect the site-supplied config.mk when it is available.
No config.mk was supplied in the Stage 1 workspace; it remains a required external input.

Support:
`make CONFIG_MK=/path/to/config.mk`

Use `CONFIG_MK ?= config.mk` and include the selected file. Consume the generated variables:
- `CFLAGS_VENDOR`
- `LDFLAGS_VENDOR`
- `GENCODE_FLAGS`

Preserve any environment-variable exports contained in the included configuration.

Treat config.mk as an external input. Do not overwrite it, silently regenerate it, or replace its MPI settings with hard-coded flags or a separate wrapper-discovery mechanism.

Keep project-specific options and cuStateVec include/library options separate from the imported settings. Provide documented overrides for the cuStateVec installation path and any additional required flags. Follow the nvccoptions guidance for forwarding host-compiler options when needed.

Provide clear errors for missing configuration or dependencies. Do not introduce CMake.

## Distributed runtime

Use MPI with exactly one GPU per process. All ranks must cooperate on one distributed state vector.

Support both multiple GPUs within one node and multiple nodes.

For P = 2^g ranks, use N-g local index bits and g global index bits. Each rank owns one contiguous device-resident sub-state-vector. Require a power-of-two rank count and at least one local qubit. Permit P=1 for testing.

Discover node-local rank using `MPI_Comm_split_type` with `MPI_COMM_TYPE_SHARED`.

Respect launcher-provided GPU visibility:
- If each rank sees one assigned GPU, use visible device 0.
- If all GPUs on a node are visible, assign GPUs using node-local rank.
- Detect duplicate GPU assignments on a node and reject oversubscription.

Use a cuStateVec communicator compatible with the MPI implementation used by the application. Document the CUDA-aware MPI requirement and any communicator-library configuration needed by the installed cuStateVec version.

Use 64-bit sizes and indices, checked allocation-size arithmetic, and explicit memory-capacity checks. Account for state storage, cuStateVec workspace, and communication buffers.

Handle MPI, CUDA, and cuStateVec failures coherently so other ranks do not remain blocked indefinitely.

Explicit P2P uses the classic `custatevecSVSwapWorkerSetSubSVsP2P` API with
CUDA IPC imports of the peer state allocation and the event passed to its swap
worker. `--p2p auto|off|required` defaults to auto. Auto registers only mutually
usable same-node workload swap peers and falls back to the CUDA-aware MPI
communicator for other pairs. Off skips explicit registration; MPI may still
use IPC internally. Required aborts coherently before timing if any same-node
pair needed by the workload lacks mutual P2P; inter-node pairs remain on MPI
and no same-node swap peers is valid. Both Hadamard orders/check-hadamard need rank XOR
each global-rank bit, smoke needs rank XOR 1, and measurement needs no swaps.

Exchange identities, handles, and directional setup outcomes within the actual
shared-node communicator, without assuming contiguous world-rank placement.
Use local GPU capabilities and IPC imports, never another process's device
ordinal or a single-visible-device heuristic. Keep only pairs with usable
memory and events in both directions; release rejected partial imports.
Expected unsupported/mapping/peer-resource/OS IPC failures diagnose fallback;
unrelated CUDA errors abort. Report requested mode, registered world ranks,
GPU identities, and fallback reasons to stderr without inferring NVLink.
Do all setup outside repetitions. Apply the adaptive transfer-workspace policy below.

## Timing contract

Allocate reusable state storage, handles, descriptors, and workspaces outside measured repetitions.

For each measured repetition:
1. Initialize the required input state and reset logical-to-physical qubit mapping.
2. Finish initialization on the GPU.
3. Execute an MPI barrier before starting the timer.
4. Record a timing-enabled start CUDA event on `Runtime::stream`.
5. Execute the complete benchmark operation, including required communication.
6. Complete all GPU work on every stream used by that rank and all required outstanding MPI operations.
7. Record a stop CUDA event on `Runtime::stream`, wait with `cudaEventSynchronize`,
   and obtain rank-local milliseconds with `cudaEventElapsedTime`; convert to seconds.
8. Reduce elapsed times with `MPI_MAX` after the timed interval.

Create one reusable pair of timing-enabled CUDA events per rank on its assigned
GPU outside the repetition loops, and destroy them during benchmark cleanup.
Keep the timing-disabled swap/P2P events dedicated to communication.
The timing backend is CUDA events: elapsed time between GPU event markers may
include host submission gaps and MPI-related waits, not just active kernels.
Record both markers on the explicit compute stream. The pre-start device wait
excludes initialization; compute-before-transfer and transfer-before-compute swap
waits plus the device-wide wait before recording stop cover all workload streams
without relying on default-stream behavior.

Report the maximum rank-local elapsed time in seconds. Do not subtract timestamps from different ranks.

Exclude allocation, input-state initialization, validation, and output formatting from the measured interval. Include all state-dependent probability computation and preprocessing required by the measurement job.

Warm-ups, when explicitly requested, must be separate from the 128 measured repetitions.

## Output

Name the executable `custatevec_bench`.

Rank 0 must write CSV to stdout with:
`job,iteration,nqubits,mpi_ranks,elapsed_seconds,measurement_bits`

Use iteration numbers 1 through 128 for a default run. Leave `measurement_bits` empty for the Hadamard job.

Send diagnostics and metadata to stderr, including software versions, GPU identities, rank-to-GPU mapping, memory requirements, seed, warm-up count, and timing scope.

## API references and implementation conventions

Use installed `custatevec.h` and documentation matching that installed release:
- [Classic API reference](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/api-reference/custatevec/index.html)
- [Distributed swaps](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/overview/distributed-index-bit-swap.html)
- [Communicators](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/overview/communicator.html)

Stage 1's workspace had no installed headers. API selection used NVIDIA's online
reference and public distributedIndexBitSwap.cpp sample; compatibility with the
target installation remains unverified.

World rank equals the sub-state-vector index. Physical bits `[0,N-g)` are local;
higher physical bits encode world rank. `logical_to_physical[q]` tracks the
physical position of logical qubit q and is reset to identity each repetition.
CSV measurement strings will contain exactly N bits, ordered q[N-1] through q[0].
A workload returns its logical outcome to rank 0 before returning from execute.

Use two distinct nonblocking CUDA streams on the rank's assigned GPU.
`Runtime::stream` is the compute stream bound to the handle with
`custatevecSetStream`, used for ordinary operations and initialization. Create
`transfer_stream_` only with distributed swap resources (P>1), and pass it to
`custatevecSVSwapWorkerCreate`. NVIDIA requires a dedicated transfer stream
because worker execution synchronizes with its transfer peer independently of
the handle's stream. Never enqueue gates or initialization on that stream.

Before each distributed swap, synchronize the compute stream, execute all
scheduler/worker batches, then synchronize the transfer stream before updating
the mapping or submitting another gate. This follows NVIDIA's
[stream contract](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/overview/index.html#using-cuda-stream-with-distributed-index-bit-swap-api)
and distributed-swap example's final worker-stream wait. Worker calls manage
their communication requests; their return alone is not treated as GPU
completion. This ordering requires the selected cuStateVec communicator and
CUDA-aware MPI to implement the worker's communication/completion contract;
a CUDA stream wait alone does not complete arbitrary MPI requests. No extra
MPI barrier is required per swap by that documented sequence.

Both swap waits stay inside the timed Hadamard execute call. Synchronize
the device before timing and before stopping; complete workload-owned MPI requests
inside execute. Keep initialization and allocation outside timing and reduce
rank-local CUDA event durations (converted to seconds) with MPI_MAX. All ranks must invoke the same
collectives and swap schedule. During close, finish all GPU work and barrier
before destroying swap resources. Destroy the worker before its transfer
stream, retain its state/workspaces/event/communicator until then, and destroy
both streams. Check errors through the existing world-aborting handlers.

With explicit P2P, imported pointers/events remain alive through worker
destruction. After the completion barrier, destroy workers on all ranks and
barrier again before closing imports with cudaIpcCloseMemHandle and destroying
imported events. Barrier after closing imports, before freeing exported state
allocations or destroying local events. Required-mode setup rejection closes
all imports before coordinated MPI_Abort. Fatal unexpected failures abort
without entering normal collective cleanup.

No installed cuStateVec headers or release documentation were found in the
project, `/usr/local/include`, or `/opt` for this fix. The online references
describe the intended completion sequence; installed-release compatibility
and GPU/MPI behavior remain unverified. Builds, tests, and validation were
skipped at the user's request.

Warm-up seed indices use a separate high-bit domain; measured indices start at 0,
independent of warm-up count. Future stochastic jobs must use the supplied seed
and repetition index deterministically and coordinate draws across ranks.

Query and reserve every workload's additional workspace in Job::prepare, outside
repetitions. Report its size and include it in capacity checks. State and swap
workspaces are capacity-checked individually and combined as information becomes
available; runtime/library/context overhead is reflected in cudaMemGetInfo.
Do not allocate buffers or create descriptors from Job::execute.

Failures from MPI, CUDA, cuStateVec, allocation, or argument handling abort the
world; do not unwind into collective cleanup while peers might be waiting.
This handles reported errors, not external process death or a transport hang;
site launcher timeouts remain necessary for those cases.


## Adaptive transfer-workspace policy

Default to `--transfer-buffer auto --transfer-buffer-fraction 0.95`; support
`min` and finite fractions in `(0,0.95]`. After mandatory state, job workspaces,
swap descriptors/extra workspace and communication/IPC setup, snapshot free
bytes per rank. Budget floor(fraction * free); agree on the minimum budget and
maximum worker minimum using MPI_UINT64_T reductions. Select the largest valid
power of two for auto, or smallest for min, without exceeding the common cap.
Use checked 64-bit sizes and exact wide integer products for budgeting. Do not
apply fixed byte or state-vector-size caps. The default fraction is exact 19/20.
The published worker API requires a power of two at least the worker minimum;
installed-library compatibility remains a target-cluster check.

Finalize after Job::prepare and before checks or benchmark warm-ups. Smoke
reserves its matrix workspace before finalization. Reject later workspace growth.
P=1 and measurement create no swap resources. Distributed swapping retains an
MPI fallback buffer with explicit P2P. Setup requires no temporary transfer buffer.

Coordinate cudaErrorMemoryAllocation across ranks before registering buffers.
Release all successful allocations, refresh budgets after a barrier, and retry
with the smaller of half the prior size and the rounded-down refreshed common
budget. At most 64 strictly decreasing attempts are possible. Abort coherently
when the minimum cannot fit; unrelated CUDA errors remain fatal. Register only
a collectively successful allocation and finish finalization on all ranks before
swapping. Report policy, fraction, skip status, baseline/free budgets, minima,
initial/final sizes, retries/reasons, remaining free memory and explicit totals
to stderr outside timing. Preserve CSV, MPI_MAX, all communication within the
timed operation, complex128/FP64, gate sequence and logical mapping. Rounding
and the least-memory rank can substantially reduce the allocated fraction.

See [benchmark.md](benchmark.md) for practical usage and
[VALIDATION.md](VALIDATION.md) for deferred host tests, cluster correctness
commands and sequential four-combination P2P/policy A/B runs. No tests or
validation were run for this change, as requested by the user.
