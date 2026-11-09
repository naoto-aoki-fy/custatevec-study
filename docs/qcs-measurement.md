# QCS measurement and reset

All six C and six throwing C++ measurement/reset entry points are implemented in
`libqcs.so`, using the unchanged public header and reference revision
`10a4da5b29a65f3eec0f556344ce06c3ff24072a`. Their specification is the pinned
[public header](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/include/qcs.h),
[grouped measurement implementation](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/src/qcs.cu#L2617),
[public wrappers/reset](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/src/qcs.cu#L3744),
and [sampler](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/src/measurement_reduce.cuh#L260).

## API behavior

| Stem (both `qcs_simulator_STEM` and `qcs_simulator_STEM_cxx`) | Behavior |
| --- | --- |
| `measure` | Measure one logical qubit, return its selected bit; no clbit write |
| `measure_to_clbit` | Same measurement, also write the specified clbit |
| `measure_many` | Measure distinct logical qubits in caller order, groups up to four; result index matches input index |
| `measure_many_to_clbits` | Sort local physical targets descending, then global physical targets descending, group, and restore caller order for results/destinations |
| `reset` | Measure the logical qubit, apply X only if the result is 1; do not clear/write clbits |
| `reset_measurement_state` | Materialize pending projections, then forget outcome records; retain the projected amplitudes, mapping, clbits and RNG cursor |

A zero-count request is a no-op and permits NULL list/result pointers, including
on a live unallocated handle. The destination variant still requires matching
counts. Nonempty requests require initialized, healthy allocated storage. Validate
logical indices, unique targets, distinct in-range clbit destinations, counts and
pointers before changing records/results. Safe failures replace legacy unchecked
single-clbit indexing; no invalid access/crash is emulated. C returns 1/0 and
reports errors through the callback; C++ throws without invoking that callback.
C single-result pointers are checked within the same collective engine, so an
invalid output on one rank cannot make it return while peers enter reduction.

Only requested fresh qubits are sampled. Cached qubits retain their recorded
outcomes and condition every later probability calculation. Entirely cached groups
need no new probability reduction or draw. Ordinary measurement does not update
clbits. Batch clbit/result writes occur only after every group succeeds, as in the
reference's temporary-result wrappers. A later-group numerical failure is not a
transactional rollback of earlier groups or their consumed cursors.

## Conditional probabilities, projection and MPI

`qcs_measurement.hpp` contains host validation, execution ordering, group descriptors,
bin expansion and small-CDF selection. `qcs_measurement.cuh` owns the cuStateVec/MPI
path and is included in the library's checked private namespace.

For each group of one through four entries, use `custatevecAbs2SumArray` to compute
a histogram over only the fresh local physical bits. All previously measured local
bits are mask conditions, including cached entries in this group. Fresh global bits
come from the rank number; cached bits come from logical outcome metadata. Ranks
excluded by earlier measured global bits contribute exact zeros. Expand at most 16
compact bins into the original group bit order, then sum FP64 masses with MPI.
This neither measures spectator qubits nor requires localizing global targets.
Measurements leave both physical/logical mapping directions unchanged.

Reject nonfinite/negative probability components and nonfinite/nonpositive totals
before drawing or updating metadata. Raw norms are permitted: sampling is relative
to the current branch total, but storage is never normalized. All ranks participate
in validation/reduction, including zero-mass ranks. MPI errors use the foundation's
fatal transport policy; local reduction/device failures are agreed and poison the
handle. Device work is completed before host buffers leave scope. Launcher timeouts
remain required for transport/GPU hangs and process loss.

Keep the reference's **deferred unnormalized projection**. Successful measurements
record logical measured mask/value and pending projection mask; amplitudes are
unchanged until a consumer needs projection. Gates project measured targets using
their post-localization mapping and release their records. Measured non-targets
remain implicit controls. A contrary explicit control is a no-op. Metadata reset
projects every pending bit without rescaling and only then discards records, so it
cannot restore rejected amplitudes. This retains unmeasured coherence and the raw
branch norm exposed to later expectations/statevector output.

State initialization overwrites amplitudes and clears records without changing
mapping, clbits, cursor or cursor-exhaustion state. Clbit clearing only changes the
classical register. Mapping reinitialization only installs the configured mapping;
it does not initialize or project storage. These remain separate operations.
Reset inherits the reference gate constraint: an outcome of 1 requires at least one
local qubit for X/localization. With L=0, measurement and metadata reset work, and a
zero-outcome per-qubit reset can return without a gate; resetting a one outcome
reports the same target-width limitation as the reference.

## RNG and resource ownership

Use the pinned sampler's Philox4x32-10 mapping: `curand_init(seed,cursor,0)`, followed
by curand4 and its first two words to form the 53-bit uniform in [0,1). Rank zero
produces the draw on CUDA and selects from globally reduced bins, then broadcasts
the outcome. Every rank advances one cursor per fresh group, even if its outcome
is deterministic. Empty/cached groups and invalid masses do not consume a cursor.
The maximum uint64 cursor is usable once, then a persistent exhaustion flag rejects
fresh groups while still allowing cached reads. Initialization/metadata reset do
not rewind sampling or clear exhaustion.

The small host CDF uses half-open intervals, skips zero weights, and clamps endpoint
rounding below the total. Its accumulation order and cuStateVec/MPI probability
reductions differ from the reference's CUB/NCCL path. The interface promises the
conditional distribution and group/cursor rules, not identical sampled bits across
these backends/hardware/reduction orders. Tests compare analytical outcomes and
statistics, or replay identical group requests **within this backend**; they do not
assume cross-backend bitwise trajectories. Optional reference performance backend
selectors are not implemented as alternative library measurement backends.

The cuStateVec handle's default workspace is already owned by allocation. A separate
persistent eight-byte draw buffer is allocated collectively before the first fresh
request. Group histograms/descriptors use bounded host arrays and do not grow a
worker-borrowed allocation. Disposal frees draw storage after work completes.
Measurement-first, then arbitrary gates, remains valid: swap workers/gate scratch
are prepared independently by the gate path. No benchmark job-specific setup or
normalized full-register collapse is reused. Benchmark source/CLI/timing/CSV and
normalized initialization semantics remain unchanged.

## Validation and deferred execution

Executed successfully in this environment:

```sh
c++ -std=c++17 -Wall -Wextra -Werror -Iinclude tests/qcs_measurement_host_test.cpp -o /tmp/qcs-measure-host-test
/tmp/qcs-measure-host-test
cc -std=c11 -Wall -Wextra -Werror -Iinclude -fsyntax-only tests/qcs_measurement_c.c
python3 tests/check_qcs_header.py
```

The host test compares expanded conditional masses against an independent direct
enumeration over all five-qubit permutations, every local count 0..5, multiple
prior masks/values and group widths 1..4: 17,280 fixtures pass. It also checks
physical/caller ordering, invalid indices/masses, zero bins, half-open boundaries
and endpoint rounding. Existing gate-formula/localization and host runtime-policy
tests pass. Nonfunctional temporary declaration shims permitted host syntax checks
of the new CUDA sources/test and all-symbol linkage; these do not validate installed
CUDA signatures, device execution or MPI transport.

The new C/C++ GPU integration test covers identity/reverse mappings, Bell/GHZ
correlation, coherence under partial measurement followed by H, cached reads,
measured-target gates, reset, raw branch norms, clbit preservation/destinations,
physical execution order, empty/invalid/one-rank-NULL requests, every selected
initializer's restoration (including CURAND and file-backed), scalar L=0 slices,
zero-mass ranks, invalid total and cursor exhaustion. A 1,024-shot Bell marginal
uses the predeclared [416,608] acceptance interval (six binomial standard deviations
around 512). Those GPU assertions have **not run** here.

`make library` was attempted and rejected missing `config.mk`; `nvcc` and `mpicxx`
are absent from PATH. No real CUDA compilation/link, GPU numerical test, MPI run,
sanitizer or benchmark GPU regression is claimed. On a configured cluster:

```sh
make library build/qcs_measurement_test CONFIG_MK=/path/to/config.mk CUSTATEVEC_ROOT=/path/to/cuquantum
python3 tests/check_qcs_header.py --library ./libqcs.so
export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
timeout 600s mpiexec -n 1 ./build/qcs_measurement_test
timeout 600s mpiexec -n 2 ./build/qcs_measurement_test
timeout 600s mpiexec -n 4 ./build/qcs_measurement_test
```

Use one distinct GPU per process, with CUDA-aware MPI and a shared writable working
directory for the isolated file fixture. Repeat across nodes/noncontiguous world
rank placement and with both shared/per-rank GPU visibility. Run the gate and
foundation tests and existing benchmark checks as documented in qcs-gates.md,
qcs-library.md and README.md. Run Compute Sanitizer memcheck/racecheck/synccheck and
one-rank allocation/transport failure injection under timeouts. Retain exact tool,
GPU and MPI versions and logs. Public expectation/save/timing implementations and
their observation of projected state remain subsequent work.
