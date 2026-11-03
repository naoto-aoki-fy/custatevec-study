# Task 2: CUDA stream separation review

## Scope and outcome

This is a source and documentation review of `SPEC.md`, `include/bench.hpp`,
`src/runtime.cu`, and `src/benchmark.cu`, with supporting reads of the Makefile,
option parsing, launch script, README, and Stage 4 record. No stream-ordering
defect requiring a runtime patch was identified. The stale shared-stream claim
in `STAGE4.md` was corrected. Benchmark code and workload are unchanged.

The user's instruction to skip tests and validation overrides task2.md's build,
installed-header validation, correctness execution, and benchmark regression
execution requirements. No build, dry run, test, linter, dependency/hardware
probe, or GPU/MPI execution was performed. This review is not full validation.
No external configuration was modified or regenerated.

## Stream and lifetime findings from source

| Requirement | Source observation |
| --- | --- |
| Distinct streams | The constructor creates `stream` with `cudaStreamNonBlocking`, binds it through `custatevecSetStream`, and separately creates `transfer_stream_` inside the P>1 branch. The latter is passed to `custatevecSVSwapWorkerCreate`. |
| Dedicated transfer stream | Initialization kernels, matrix applications, probability computation, measurement, and asynchronous state copies/clears use the compute stream or its bound handle. Only the swap worker receives the transfer stream. |
| Gates finish before swapping | `swap_local_global` waits on the compute stream before scheduler/worker calls. |
| Swaps finish before subsequent gates | All scheduler batches execute before the transfer-stream wait; mapping updates and the next gate occur after that wait. Worker return alone is not treated as GPU completion. |
| Explicit ordering | Both streams are nonblocking; dependencies use explicit stream waits. The harness uses device synchronization before timing and before stopping. Correctness does not require legacy default-stream ordering. |
| Resource lifetime | Both streams persist across repetitions. `close` synchronizes the device and barriers before destroying descriptors. The worker is destroyed while its state, event, workspaces, communicator, handle, and transfer stream remain alive. Both streams are subsequently destroyed. API failures abort the world. |
| Single rank | P=1 creates no communicator, scheduler, worker, transfer stream, swap event, or transfer workspace. Every valid Hadamard target is local. |
| Timing and communication | Swap waits occur inside Hadamard execution. Measurement uses blocking MPI collectives and waits before consuming GPU-produced host probabilities/bits. Final device synchronization precedes the elapsed-time calculation. `MPI_Reduce` with `MPI_MAX` follows that calculation. |

Communication completion relies on the cuStateVec worker/communicator contract.
A CUDA stream wait is not a general MPI request-completion operation. There are
no application-owned nonblocking MPI requests in the inspected workloads.

For P=4,N=3, source inspection gives one local bit and two global bits.
The Hadamard sequence uses physical targets 0, then swaps (0,1), then (0,2).
Updating both logical owners after each swap yields final mapping [1,2,0].
This is a reasoning example, not an executed correctness result.

## NVIDIA references consulted

- [cuQuantum 25.03 overview, Using CUDA stream](https://archive.docs.nvidia.com/cuda/cuquantum/25.03.0/custatevec/overview.html#using-cuda-stream)
  describes handle-bound operations and the swap worker's separate transfer
  stream with peer synchronization. This is an archived reference, not a claim
  about the installed release.
- [NVIDIA distributedIndexBitSwap.cpp](https://github.com/NVIDIA/cuQuantum/blob/main/samples/custatevec/custatevec/samples_mpi/distributedIndexBitSwap.cpp)
  passes its transfer stream to worker creation and waits on it after the
  worker batches. It also retains worker dependencies until worker destruction.

The current overview link in SPEC.md was opened, but its retrieved text did not
expose the referenced stream section; the archived overview above supplied the
explicit stream description. A 25.09.1 overview lookup was inaccessible.
These online reads do not establish target header signatures, loaded-library
compatibility, or CUDA-aware MPI transport behavior. Installed headers and
matching release documentation were not inspected in this task.

## Benchmark contract findings from source

`Options` and launcher defaults retain N=31, 128 measured repetitions, and zero
warm-ups. Each repetition resets mapping and independently initializes input,
finishes initialization, and barriers before taking a rank-local start time.
Preparation and reusable allocations occur outside repetitions. Hadamard uses
N separate one-target applications with complex128 matrices/state and FP64
computation; no gate fusion is introduced.

Measurement directly initializes a global uniform state, recomputes FP64 rank
masses inside timing, coordinates one draw, measures all local bits on the
selected rank, collapses its state preserving phase, and clears other ranks.
The broadcast physical outcome is translated through logical mapping. CSV
formatting occurs after timing and retains the existing header, empty Hadamard
outcome, and N-bit logical measurement outcome. Separate job selectors and
launch entry points remain available.

Topology uses shared-node local rank for GPU assignment, selects device 0 when
only one GPU is visible, and compares UUIDs to reject duplicate assignments.
World rank remains the sub-state-vector index across nodes. These are source
observations, not evidence that any particular placement or transport works.

## Execution record and deferred evidence

Build commands actually executed: **none**. Test/benchmark/validation commands
actually executed: **none**. Pass/fail results: **not run**, with no executed
program result claimed. Repository reads used `pwd`, `rg`, `cat`, and `sed`;
online documentation reads and Git diff/status/commit bookkeeping are separate
from program validation.

| Configuration | Result |
| --- | --- |
| GNU Make/nvcc with existing config.mk and installed headers | Skipped at user request |
| P=1,N=8 Hadamard and measurement checks | Skipped at user request |
| P=2,N=8 on two GPUs, both correctness checks | Skipped at user request |
| P=4,N=3 on four GPUs, both correctness checks | Skipped at user request |
| P=2 across two nodes | Skipped at user request |
| Separate default-size jobs and CSV regression | Skipped at user request |

Hardware, compiler, installed headers, and multi-node availability were not
probed; none of these configurations is classified as available or unavailable.
README.md contains deferred build and launch commands. The existing untimed
Hadamard checks include asymmetric complex input, individual global targets,
repeated localization, and logical-state comparison; measurement checks cover
unequal masses, zero-mass ranks, interval boundaries, collapse phase, and norm.
Their presence is not evidence they pass.

Remaining risks are installed-release API compatibility, worker-owned
communication completion on the selected MPI/communicator combination, GPU
numerical correctness, and inter-node progress/completion. No additional
per-swap MPI barrier is added on the basis of source inspection. Target build,
GPU/MPI checks, and separate benchmark regression remain required before
reporting the implementation as fully validated.
