# Development history and implementation notes

This record consolidates the former Stage 4 implementation log and Task 2
stream-ordering review. It records source observations and historical decisions;
source inspection is not runtime validation. The intended contract is in
[SPEC.md](SPEC.md), and execution evidence is separated in
[VALIDATION.md](VALIDATION.md).

## Benchmark implementation review

The former Stage 4 review inspected the benchmark contract against the source.
The following are implementation observations, not numerical-correctness,
installed-API, transport-completion, or successful-execution claims:

| Contract area | Source observation |
| --- | --- |
| Defaults and jobs | `Options` provides N=31, 128 measured repetitions, and zero warm-ups; `parse_options` selects one job/check mode and rejects repeated job selectors. |
| State and precision | Repetitions initialize independently and reset logical mapping. State, matrices, and computation use complex128 / `CUDA_C_64F` / `CUSTATEVEC_COMPUTE_64F`; probability reductions use double precision. |
| Hadamard workload | `HadamardJob::execute` issues N separate one-target operations. A global logical target is localized to physical bit 0 through distributed swaps, updating logical ownership. |
| Measurement workload | `MeasurementJob::measure` computes local norm masses, selects the rank, samples its local bits jointly, collapses the selected actual state, clears other ranks, and reconstructs logical output. State-dependent work is inside execute. |
| Timing | CUDA event marker durations are converted to seconds and reduced with `MPI_MAX`. Setup, initialization, synchronization, and the start barrier precede the start marker; device completion precedes stop. Event intervals can include host submission gaps and MPI waits. |
| GPU ownership | Shared-node rank informs assignment; UUID allgather detects duplicates. One-visible-device assignments select visible ordinal zero. |
| Memory and failures | Source uses 64-bit indices/counts, checked allocation arithmetic and capacity accounting, and world-aborting error handlers rather than collective unwinding after fatal errors. |
| Build configuration | Make includes externally supplied `CONFIG_MK`, consumes vendor compile/link and architecture variables, and does not generate configuration or replace MPI settings. |

## Stream-ordering review

The Task 2 review found no source-level stream-ordering defect requiring a
runtime patch. The handle-bound compute stream is distinct from the transfer
stream passed to the cuStateVec swap worker. Initialization, gates, probabilities,
and measurement use compute; only the worker receives transfer. Each distributed
swap waits for compute before scheduler/worker calls and waits for transfer
completion before mapping changes or later gates. The harness waits for device
completion before stopping. Both streams are nonblocking and dependencies are
explicit, so correctness does not rely on default-stream ordering.

Worker, state, event, workspace, communicator, handle, and transfer-stream
lifetimes extend through worker destruction. With explicit P2P, imported
pointers/events stay live through worker destruction; barriers sequence import
closure and exported allocation teardown. P=1 creates no distributed swap
resources. These observations rely on the selected communicator and CUDA-aware
MPI honoring the worker's communication contract; a CUDA stream wait alone does
not complete arbitrary MPI requests.

The review used NVIDIA's published stream overview and distributed-swap example
as design references. These online references do not prove that the target's
installed API release, MPI ABI, plugin, launcher, or transport is compatible.

## Significant implementation changes recorded

- Repeated job selectors are rejected.
- CSV output is flushed and checked while MPI is still active.
- Runtime diagnostics report the resolved communicator family and updated
  explicit workspace totals.
- Independent Hadamard and measurement launch entry points share a configurable
  launcher implementation.
- Transfer workspace sizing was changed to adaptive common-budget selection
  after mandatory allocations and IPC setup. Collective allocation failures
  release successful allocations and retry with smaller sizes. P=1 and
  measurement do not create swap resources.
- Documentation now treats API compatibility, stream/transport assumptions,
  and benchmark execution as separate evidence categories.

These items summarize historical implementation notes; they are not newly
validated by this documentation reorganization. No source code was changed.
