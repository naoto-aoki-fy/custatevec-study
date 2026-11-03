# Stage 4 implementation record

The user's instruction to skip tests and validation overrides task4.md's build
and runtime verification requests. This record describes source inspection and
implementation only. No build, dry run, linter, shell syntax check, correctness
check, benchmark, dependency probe, or GPU/MPI execution was performed in this
stage. There are no successful or failed runtime tests to report. Toolchain
availability observations in the README are historical and were not refreshed.

## Source inspection against SPEC.md

| Contract | Source implementation |
| --- | --- |
| N=31, 128 measured repetitions, zero warm-ups | `Options` defaults in `include/bench.hpp`; check modes have separate small-N defaults |
| Exactly one selected job or check mode | `parse_options` in `src/main.cu`; repeated `--job` is now also rejected |
| Fresh input and mapping each repetition | `run_benchmark` resets mapping and calls initialize before every warm-up and measured operation |
| Double state, matrix, computation, probabilities | `cuDoubleComplex`, `CUDA_C_64F`, `CUSTATEVEC_COMPUTE_64F`, double host masses and `MPI_DOUBLE` |
| N separate Hadamards, no gate fusion | `HadamardJob::execute` loops through logical targets and calls one-target ApplyMatrix once for each |
| Global-target communication | `apply_hadamard` invokes the distributed scheduler/worker to swap the current global target with local physical bit 0; both logical owners are updated |
| Actual state and complete-register measurement | Each `MeasurementJob::measure` recomputes local norms, gathers rank masses, selects the owner, jointly samples its local bits, and combines owner bits with local bits |
| State-dependent work stays timed | Norm calculation, rank selection, conditional local measurement, selected-amplitude mass query, and collapse are all inside execute |
| Distributed collapse | Owner collapses all local bits using the selected actual mass; other ranks zero their state; the full physical outcome is broadcast and mapped to logical bits |
| Completion before stop | Gates and swap worker share one stream with explicit swap ordering; the harness synchronizes the device after execute; application collectives are blocking |
| Maximum rank-local interval | A local `MPI_Wtime` difference is taken before `MPI_Reduce(MPI_MAX)` |
| Exclusions | Preparation/allocation, mapping reset, input initialization, initialization synchronization, start barrier, and CSV formatting are outside the interval; safety/error checks required by the operation remain inside execute |
| Default CSV | Rank 0 writes one header and measured iterations 1..128; warm-ups do not write rows; final flush now detects buffered output failure before MPI cleanup |
| GPU ownership | Shared-node communicator discovers local rank; one visible device selects ordinal 0, otherwise local rank; allgather of device UUIDs detects duplicate assignment |
| Index and allocation arithmetic | N is limited to 1..62, indices and counts are uint64, state bytes use checked multiplication, buffer totals use checked addition, and allocations check current free GPU memory |
| Workspace capacity and cleanup | State, swap, transfer, and job workspaces are allocated before repetitions; CUDA frees and descriptor destruction occur in explicit synchronized close; gate preparation now also reports the updated explicit allocation total |
| Failure handling | API error wrappers and exception handlers abort MPI_COMM_WORLD; no exception unwinding enters collective resource cleanup |
| External build configuration | Makefile includes CONFIG_MK and consumes CFLAGS_VENDOR in compilation, LDFLAGS_VENDOR in linking, and GENCODE_FLAGS in both; it does not regenerate configuration or discover alternate MPI flags |

These are observations about code structure, not claims of numerical correctness,
installed API compatibility, stream/transport completion guarantees on a target
release, or successful execution. In particular, communicator compatibility
requires the target MPI, plugin, and launcher. Resolved communicator-family
metadata is now emitted after successful multi-rank communicator creation;
successful creation alone cannot establish CUDA-aware networking support.

## Changes

- `src/main.cu`: reject repeated job selectors.
- `src/benchmark.cu`: flush and check CSV output while MPI is still active.
- `src/runtime.cu`: report the resolved communicator family and updated gate
  workspace allocation total, outside timing.
- `scripts/launch-hadamard.sh` and `scripts/launch-measure.sh`: independent
  entry points; `scripts/launch-job.sh` supplies their shared configurable
  launcher implementation without evaluating shell text.
- `README.md`: complete launch, scheduler assumptions, API capabilities,
  distribution, normalization, CSV, memory, and deferred acceptance instructions.
- `STAGE4.md`: this record, including the validation boundary.

No generated config.mk was changed or supplied. The Makefile and benchmark
operations, initialization algorithms, gate order, RNG, and timing boundaries
are unchanged. The pre-existing untracked task1.md through task4.md are excluded
from the implementation commit.

## Commands actually executed and results

Repository/source inspection used `pwd`, `rg --files`, `cat`, `sed -n`,
`git status --short`, and `git log -5 --oneline`. The initial Git status showed
only the four untracked task documents. Source reads completed; one combined
file-discovery command stopped with status 1 because no AGENTS/config.mk match
was listed. No build command followed it.
`git config user.name` and `git config user.email` found an existing identity.
Edits used apply_patch; one patch failed to match README context and was
reissued with the correct context. Final diff/status review and explicitly
scoped Git staging/commit are repository bookkeeping, not program validation.
`chmod +x scripts/launch-hadamard.sh scripts/launch-measure.sh scripts/launch-job.sh`
marked the launch scripts executable and returned status 0.

## Still required on the target cluster

Follow the deferred commands in README.md to inspect the real generated
configuration and compiler/linker expansion, build with matching installed
headers, and establish MPI/communicator compatibility. Record exact software
versions and memory overhead. Run all three small-state check modes on one
rank, multiple GPUs on one node, and multiple nodes with one GPU per rank;
include P=4,N=3 when resources permit. Existing fixtures cover asymmetric
complex states, unequal probability masses, zero-probability ranks, interval
boundaries, and collapse phase/norm checks. Exercise invalid rank counts,
duplicate assignment, insufficient memory, and no-hang behavior with launcher
timeouts. Finally run the two default-size jobs in separate invocations and
check 128-row CSV output. None of these acceptance results is available here.
