# QCS API compatibility plan

## Scope and pinned evidence

This document records the compatibility plan and the foundation implementation status.
All public declarations now have implementations; real CUDA/MPI validation remains incomplete.
The goal is source and link compatibility with the complete public C and C++ interface,
and matching defined observable behavior, using cuStateVec for computation.

- Implementation: `naoto-aoki-fy/custatevec-study`, starting commit
  `7ca8e923e63d39bc92d207cebfb5fbd985fc999c` (clean working tree at inspection).
- Reference: `jhpc-quantum/rqs-svg`, exclusively commit
  `10a4da5b29a65f3eec0f556344ce06c3ff24072a`.
- The original investigation changed no implementation code. The foundation task starts
  from `67208f8` and adds the public header, library foundation and checks; reference files remain unchanged.
- Existing benchmark CLI, numerical behavior, timing boundaries, CSV, and external
  `config.mk` contract are invariants, not migration targets.

Reference links below are pinned; none means the default branch:

| Evidence | Purpose |
| --- | --- |
| [include/qcs.h](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/include/qcs.h) | Authoritative signatures, explicit allocation, measurement, expectation contracts |
| [src/qcs.cu](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/src/qcs.cu) | Gate definitions, core setup/allocation, mapping, projection, wrappers, binary I/O, events |
| [docs/usage.md](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/docs/usage.md) | Selected initializer, explicit initialization and example workflows |
| [docs/measurement-reducer.md](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/docs/measurement-reducer.md), [measurement-auto.md](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/docs/measurement-auto.md) | Group order, cached outcomes, unnormalized probabilities, RNG, backend distinctions |
| [docs/pauli-expectation.md](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/docs/pauli-expectation.md) | Raw expectations, localization, serialization and error rules |
| [tests/initial_state_test.cu](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/tests/initial_state_test.cu), [pauli_api_test.cpp](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/tests/pauli_api_test.cpp), [pauli_api_c.c](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/tests/pauli_api_c.c) | Concrete initializer expectations and C/C++ validation fixtures |
| [examples/sharedlibrary/ghz.c](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/examples/sharedlibrary/ghz.c), [hadamard-measure.c](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/examples/sharedlibrary/hadamard-measure.c) | Unchanged C callers, allocation/initialization, events, repeated samples, strings |
| [Makefile](https://github.com/jhpc-quantum/rqs-svg/blob/10a4da5b29a65f3eec0f556344ce06c3ff24072a/Makefile) | `libqcs.so`, PIC shared-library link, C++17 and CURAND dependencies |

Implementation evidence is [bench.hpp](../include/bench.hpp), [runtime.cu](../src/runtime.cu),
[benchmark.cu](../src/benchmark.cu), [main.cu](../src/main.cu), [Makefile](../Makefile),
[SPEC.md](../SPEC.md), [README.md](../README.md), and
[measurement-order.md](measurement-order.md). Host policy test sources were inspected.
The public measurement integration tests (`tests/measurement_integration_test.cpp`)
also supply coherence, shuffled destinations, group remainder and cached-result fixtures.
Source/test inspection is evidence of intended behavior, not proof of GPU execution.
Reference documentation itself records substantial unverified CUDA/distributed coverage.

## Foundation implementation status

See [QCS library foundation](qcs-library.md) for build/link instructions, ownership,
distributed failure rules and exact executed/deferred checks. `include/qcs.h` is
byte-identical to the pinned reference. All declarations have definitions, but
symbol linking against CUDA is not validated in the current environment.

| Inventory entries (both C/C++ unless noted) | Current status |
| --- | --- |
| `create`, `dispose`, `destroy`, process/qubit/clbit queries and configuration | Implemented foundation; GPU/MPI execution pending |
| Both `allocate_memory_cxx` overloads, C allocation variants and inline C++ overload | Implemented uninitialized allocation and selected-state dispatch |
| `set_zero_state`, `set_flat_state`, `set_sequential_state`, `set_entangled_state`, `set_random_state`, selected LOAD | Implemented raw initializer behavior; CURAND/file restoration; device execution pending |
| `set_mapping`, `reinitialize_mapping`, clbit get/string/read/reset | Implemented foundation |
| C-only `qcs_set_exception_callback` and all C translation wrappers | Implemented; callback exceptions contained |
| All `gate_*` and numbered SWAPs | Implemented; per-function status below; GPU/MPI validation pending |
| All measurement variants, `reset`, `reset_measurement_state` | Implemented; per-function status below; GPU/MPI execution pending |
| `expectation_pauli`, `save_statevector`, all event APIs | Implemented; host contraction/ABI checks pass; CUDA/MPI validation pending |

Default workspace is allocated at state allocation. The gate task adds per-call
workspace queries and independent growable scratch, lazy persistent MPI swap workers,
reference localization and measured-target projection support. Explicit IPC imports
are not used. Grouped cuStateVec measurement, coordinated sampling and public reset
APIs and timing events are implemented. See
[gate implementation and validation](qcs-gates.md).
Host ABI and existing policy tests passed. Real library build/run and distributed
acceptance criteria remain unverified, as detailed in the linked execution report.

## Gate task: per-function status

Starting implementation commit: `01dc69e`. Every row means both
`qcs_simulator_gate_NAME` and `qcs_simulator_gate_NAME_cxx`, unless marked C-only.
All rows are implemented with complex128/FP64 cuStateVec application; host oracle
checks pass, while actual CUDA/MPI numerical validation remains pending. Basis
indices in the table use target zero as the low bit; unspecified entries are
unchanged. Positive/negative controls and mapping handling apply to every row.

| NAME | Target count | Implemented semantics |
| --- | --- | --- |
| `global_phase` | 0 | Conditional `exp(i*theta)` scalar |
| `h` | 1 | Hadamard |
| `x` | 1 | Pauli X |
| `y` | 1 | Pauli Y |
| `z` | 1 | Pauli Z |
| `s` | 1 | diag(1,i) |
| `sdg` | 1 | diag(1,-i) |
| `t` | 1 | diag(1,exp(i*pi/4)) |
| `tdg` | 1 | diag(1,exp(-i*pi/4)) |
| `sx` | 1 | Reference square-root-X including its phase |
| `rx` | 1 | exp(-i*theta*X/2) |
| `ry` | 1 | exp(-i*theta*Y/2) |
| `rz` | 1 | exp(-i*theta*Z/2) |
| `u4` | 1 | exp(i*gamma)*U3(theta,phi,lambda) |
| `u3` | 1 | U3(theta,phi,lambda) |
| `u2` | 1 | U3(pi/2,phi,lambda) |
| `u1` | 1 | diag(1,exp(i*lambda)) |
| `u` | 1 | Alias of u3 |
| `p` | 1 | u1(theta) |
| `swap` | 2/4/6/8 | Consecutive disjoint pairs; all localized before applying |
| `iswap` | 2 | Exchange basis indices 1 and 2 with factor i |
| `id` | 1 | Identity with normal gate metadata handling |
| `sxdg` | 1 | Adjoint square-root-X |
| `r` | 1 | Equatorial-axis rotation R(theta,phi) |
| `rxx` | 2 | XX rotation |
| `ryy` | 2 | YY rotation |
| `rzz` | 2 | ZZ rotation |
| `rzx` | 2 | Z on first operand, X on second; theta/2 convention |
| `dcx` | 2 | Basis cycle 1->2->3->1 |
| `ecr` | 2 | Pinned reference ECR signs/phases |
| `xx_plus_yy` | 2 | Rotate indices 1,2 with theta/2 and beta phases |
| `xx_minus_yy` | 2 | Rotate indices 0,3 with theta/2 and beta phases |
| `rccx` | 3 | 3<-(-i)*7; 7<-i*3; 5<-(-1)*5 |
| `rcccx` | 4 | 3<-i*3; 11<-(-i)*11; 7<-15; 15<-(-1)*7 |
| `swap1` (C-only) | 2 | Exact-count wrapper for 1 consecutive SWAP pairs |
| `swap2` (C-only) | 4 | Exact-count wrapper for 2 consecutive SWAP pairs |
| `swap3` (C-only) | 6 | Exact-count wrapper for 3 consecutive SWAP pairs |
| `swap4` (C-only) | 8 | Exact-count wrapper for 4 consecutive SWAP pairs |

See [QCS gates](qcs-gates.md) for distributed preparation, constraints, executed
commands and unrun checks. Public post-measurement integration test sources are now implemented; their
GPU/MPI execution remains a release gate.

## Measurement/reset task: per-function status

Starting implementation commit: `08254a8`. Each stem below denotes both the C and
throwing C++ declaration. All 12 entry points are implemented; host checks pass,
while real CUDA/MPI execution is pending. See [measurement design, tests and exact
commands](qcs-measurement.md).

| Stem | Implemented behavior |
| --- | --- |
| `measure` | One logical Z measurement, cached results, no clbit write |
| `measure_to_clbit` | One logical measurement plus specified classical destination |
| `measure_many` | Caller-order groups up to four; only requested fresh targets are sampled |
| `measure_many_to_clbits` | Local-descending then global-descending physical execution; caller-order results and requested clbits |
| `reset` | Measure then X only for outcome 1, with reference gate-width constraints |
| `reset_measurement_state` | Materialize all unnormalized projections then clear records; retain mapping, clbits and RNG |

The implementation retains deferred raw projections, caches outcomes, uses
cuStateVec conditional probability bins and FP64 MPI reductions, and broadcasts
a Philox-derived sampled outcome. Empty requests, invalid lists/pointers/counts,
selected-initializer restoration and counter exhaustion have focused checks.
Cross-backend bitwise samples are not claimed. Expectation/save/timing APIs now
observe projected states; see the remaining-API report below.

## Remaining API completion: implementation versus validation

The exhaustive 139-declaration header inventory has no remaining failure placeholders.
This includes both C++ allocation overloads, the inline C++ convenience overload,
C-only numbered SWAPs and callback registration, and both language forms of the
five previously deferred functions. The pinned header is unchanged.

| Function (C and C++) | Implementation | Validation |
| --- | --- | --- |
| `expectation_pauli` | Finite coefficient, distinct logical indices and valid enum checks; identity allows null lists; nonidentity support localized with retained mapping; raw complex128 contraction and MPI SUM | 128 complex/unnormalized host tensor-product comparisons plus identity checks passed; public GPU/MPI tests written, unrun |
| `save_statevector` | All pending projections materialized; raw native complex128 pairs in logical basis order; serialized rank writes and coordinated file errors | Mapping-sensitive and measured-state GPU tests written, unrun |
| selected LOAD initializer | Native complex128 contiguous physical slices from `statevector_input.bin`, independent of mapping | Independent physical-order fixture written, unrun |
| `event_create` | Owned CUDA timing events with blocking synchronization flag; C++ int IDs, C bit_t narrowing | ABI probes passed; GPU tests written, unrun |
| `event_record` | Record on simulator compute stream; prior distributed exchanges already join this stream | Syntax checked; GPU/MPI validation pending |
| `event_get_elapsed_time` | Wait for both events; float milliseconds, MPI_FLOAT MAX, return double seconds | Unit/conversion/aggregation and invalid-ID tests written, unrun |

Expectation currently stages each rank's local state on the host for the Pauli
contraction after existing cuStateVec localization. This preserves raw amplitudes
and phases but has host-memory and transfer costs proportional to the local vector.
Observation preserves measured outcomes, classical bits and RNG; only pending
projection work is consumed. C output pointers are assigned only after successful
completion; null expectation output is rejected before any state mutation.

Save and load ordering were inspected separately at the pinned source: save uses
`perm_p2l` to seek logical indices, while load reads rank-contiguous physical
slices without applying the mapping. Consequently a saved file loaded under a
nonidentity mapping is not generally the same logical state. Files have no header
or normalization and contain consecutive native real/imaginary doubles. The new
fixture writes independent physical input values and checks saved logical output,
rather than asserting an invalid universal round trip.

Executed checks: all six host C++ test executables passed (gate oracle requires
`-Wno-unknown-pragmas` for its CUDA pragma), pinned-header C/C++ probes passed,
and `git diff --check` passed. Temporary nonfunctional CUDA/MPI declaration shims
also passed implementation and new integration-test syntax checks; they do not
validate actual CUDA signatures or execution. `make library` fails because
`config.mk` is absent; neither `nvcc` nor `mpicxx` is in PATH.

**Remaining gap:** real CUDA compilation/linking, GPU numerical execution, MPI
localization/file ordering/timing, sanitizers and benchmark regression execution.
No public implementation is knowingly left deferred. Run the new target in an
isolated writable working directory (it uses/removes `statevector_input.bin`):

```sh
make library build/qcs_observation_test CONFIG_MK=/path/to/config.mk CUSTATEVEC_ROOT=/path/to/cuquantum
python3 tests/check_qcs_header.py --library ./libqcs.so
export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
timeout 600s mpiexec -n 1 ./build/qcs_observation_test
timeout 600s mpiexec -n 2 ./build/qcs_observation_test
timeout 600s mpiexec -n 4 ./build/qcs_observation_test
```

The benchmark sources and their timing boundaries are unchanged. Run the existing
foundation, gate, measurement and benchmark acceptance checks on the same cluster.

## Public inventory and ABI

The appendix records every function declaration verbatim, including parameter types,
constness, returns and overloads. The tables below map every declaration to work.
Only `qcs.h` is the public scope: internal templates, environment-controlled reference
performance backends, circuit-loader entry points and CLI flags are not extra API symbols.

| Public type | Exact meaning |
| --- | --- |
| `qcs_initial_state` | Enum: `QCS_INITIAL_STATE_ZERO=0`, `QCS_INITIAL_STATE_FLAT=1`, `QCS_INITIAL_STATE_ENTANGLED=2`, `QCS_INITIAL_STATE_SEQUENTIAL=3`, `QCS_INITIAL_STATE_USE_CURAND=4`, `QCS_INITIAL_STATE_LOAD_STATEVECTOR=5` |
| `qcs_pauli_operator` | Enum: `QCS_PAULI_I=0`, `QCS_PAULI_X=1`, `QCS_PAULI_Y=2`, `QCS_PAULI_Z=3` |
| `qcs_simulator` | Opaque `typedef struct qcs_simulator qcs_simulator`; do not expose `bench::Runtime` layout |
| `qcs_exception_callback` | `void (*)(char const *message, size_t message_length)`; process-wide callback, NULL disables reporting |
| `bit_num_t` | `int16_t` qubit/clbit index and list count |
| `proc_num_t` | `int32_t`; public typedef even though getters do not use it |
| `bit_t` | `uint8_t`; bits **and** C getter/event outputs |
| `event_num_t` | `int16_t`; elapsed-time event inputs, distinct from record's `int` |

Do not repair narrow C output pointers to `int*`: that breaks the interface.
The C wrappers currently cast C++ process counts, ranks, qubit/clbit counts and event
IDs to `bit_t` (modulo 256 for these nonnegative integers). Keep that conversion;
C++ getters/event creation return `int`. Callers needing larger IDs/counts must use
existing C++ forms. Report this limitation, without adding a substitute signature.
Use the pinned header unchanged for compile/link acceptance. C symbols have C linkage;
`*_cxx` functions have normal C++ linkage and throw. The final inline two-argument
`qcs_simulator_allocate_memory` is C++ source convenience forwarding to the C
`_with_initial_state` symbol, not an additional C export. There are two throwing
`allocate_memory_cxx` overloads. Numbered SWAP exports are C-only.

### Complete mapping to implementation work

For this table, a stem `foo` denotes both `qcs_simulator_foo` and
`qcs_simulator_foo_cxx`, unless explicitly marked C-only. All C wrappers are implemented in the foundation;
shared core logic should avoid duplicating semantics. “Reuse” means extract/adapt,
not call the current world-aborting runtime directly from the C boundary.

| Entry-point stems | Existing reusable code | Required work |
| --- | --- | --- |
| `create`, `destroy`, `dispose` | Topology, CUDA assignment, Runtime resource setup/close | New opaque handle, ownership manager, staged lifecycle and exception-safe teardown |
| `allocate_memory` (both C++ overloads), C-only `allocate_memory_with_initial_state` | Checked sizing, state allocation, workspace queries | Separate create/configure/allocate/initialize; remember enum, reserve full API resources |
| `get_num_procs`, `get_proc_num` | `Topology::ranks/rank` | Thin getters and exact C narrowing |
| `get_num_qubits`, `set_num_qubits` | Runtime N, checked size arithmetic | Configurable preallocation metadata; reject unsafe postallocation resize explicitly |
| `set_mapping`, `reinitialize_mapping` | `logical_to_physical`, `reset_mapping` | Physical-to-logical input validation, inverse map, saved initial permutation; do not always reset to identity |
| `get_num_clbits`, `set_num_clbits`, `get_clbits`, `get_clbits_string`, `read`, `reset_clbits` | No classical register | New host register, count handling, reverse-order NUL-terminated string |
| `set_zero_state` | `initialize_zero` | Dispatch remembered selection; clear measurement metadata, preserve mapping/clbits |
| `set_flat_state` | Uniform GPU initialization pattern | Fill amplitude 1, not normalized uniform amplitudes |
| `set_entangled_state` | GHZ endpoint fill pattern | Preserve reference amplitudes 1 and single-rank endpoint behavior; do not use normalized GHZ |
| `set_sequential_state` | Distributed physical slicing | New GPU index-valued initializer |
| `set_random_state` | No equivalent | CURAND normal complex amplitudes and retained initialization seed |
| `reset`, `reset_measurement_state` | Measurement building blocks only | Measure-then-X reset; materialize projections before clearing metadata |
| `gate_h` | H matrix, matrix application and distributed swaps | General target/control/mapping path; one target, not a broadcast list |
| `gate_x`, `gate_y`, `gate_z`, `gate_s`, `gate_sdg`, `gate_t`, `gate_tdg`, `gate_sx`, `gate_sxdg`, `gate_id` | `apply_matrix`, smoke X | New exact reference matrices, controls and cache effects; id still follows gate metadata path |
| `gate_rx`, `gate_ry`, `gate_rz`, `gate_r`, `gate_u4`, `gate_u3`, `gate_u2`, `gate_u1`, `gate_u`, `gate_p` | Matrix application infrastructure | New parameterized matrices; preserve argument order, phase and aliases |
| `gate_global_phase` | No zero-target wrapper | Conditional scalar phase, zero targets and arbitrary controls |
| `gate_swap`, C-only `gate_swap1`, `gate_swap2`, `gate_swap3`, `gate_swap4` | Distributed layout swap infrastructure | Actual logical controlled SWAP operation; pair order and arity checks, not just relabeling |
| `gate_iswap`, `gate_rxx`, `gate_ryy`, `gate_rzz`, `gate_rzx`, `gate_dcx`, `gate_ecr`, `gate_xx_plus_yy`, `gate_xx_minus_yy` | Generic multi-target cuStateVec matrix path | Exact two-target matrices, asymmetric target-order tests |
| `gate_rccx`, `gate_rcccx` | Generic matrix path | Exact three-/four-target relative-phase operations, distinct from ordinary controlled X |
| `measure`, `measure_to_clbit`, `measure_many`, `measure_many_to_clbits` | `Abs2SumArray`, probability reductions, physical/logical outcome conversion | Partial grouped conditional measurement, retained outcomes and pending unnormalized projection; classical destinations |
| `expectation_pauli` | State, handle, swaps and MPI reductions | New validation, local Pauli cuStateVec evaluation and global sum; preserve raw norm and measurement metadata |
| `save_statevector` | Mapping and distributed slice knowledge | Logical-order raw binary output, materialize projections, coordinated I/O errors |
| `event_create`, `event_record`, `event_get_elapsed_time` | CUDA streams and MPI timing infrastructure | Owned timing events, compute/transfer dependencies, CUDA milliseconds to seconds and MPI_MAX |
| C-only `qcs_set_exception_callback` | No equivalent | Process-wide registration, return 1, exception translation with message length |

The selected LOAD_STATEVECTOR initializer also needs a new loader; there is no
standalone public load function. It reads the fixed working-directory filename.

## Behavioral contracts and compatibility decisions

### Lifecycle and allocation

Reference create performs core setup (including unconditional MPI initialization)
before the user sets dimensions. The normal caller sequence is create, set dimensions
and optional mapping, allocate, explicitly initialize, operations, destroy.
`allocate_memory` defaults to ZERO selection but neither initializes nor zero-fills
amplitudes. Selecting LOAD does not access the file during allocation. Invalid
initializer enums fail. Direct state setters do not change the remembered selection.
`set_zero_state` restores that selection on every invocation, despite its name.
Initialization discards measurement metadata but leaves mapping and clbits alone.

Implement internal states `configured`, `allocated`, `initialized`, `disposed`,
plus a poisoned/fatal state. Tracking initialization makes a caller precondition
checkable; the reference does not itself track it. Do not claim reference errors
for all use-before-initialize cases. Repeated allocation is rejected by reference
core storage checks. Dimension changes after allocation and repeated dispose are
not a supported resizing/reopening protocol. Reference dispose frees core resources
and finalizes MPI; destroy invokes dispose and deletes the wrapper. Destroy(NULL)
is explicitly harmless. Reference destroy calls the nonthrowing C dispose wrapper,
which can hide disposal errors: do not use that accident as the new error design.
Make cleanup internally idempotent to handle partial construction and dispose then
destroy safely; document this as a robustness extension for otherwise unsafe usage.

### Initial amplitudes and normalization

| Selection/setter | Observed reference amplitudes in physical layout |
| --- | --- |
| ZERO | Only physical global index zero is 1 |
| FLAT | Every amplitude is `1+0i`; squared norm is `2^N` |
| SEQUENTIAL | Real amplitude equals physical global basis index |
| ENTANGLED | Rank 0's first amplitude is 1; on P>1 rank P-1's last amplitude is also 1; others zero |
| USE_CURAND | Consecutive real/imaginary normal doubles (mean 0, standard deviation 1), not normalized |
| LOAD_STATEVECTOR | Raw per-rank physical slice from `statevector_input.bin`, not normalized |

ENTANGLED uses `if (rank==0) ... else if (rank==P-1)`: P=1 has only the first
endpoint, as explicitly asserted in `initial_state_test.cu`. Preserve this tested
quirk; silently replacing it with normalized GHZ would break compatibility.
Random-state restoration regenerates the same vector with the saved initialization
seed (rank/area offsets); measurement sampling has a separate advancing counter.
The public header provides no seed setter. Do not substitute the benchmark's
SplitMix seed/repetition policy or assume bitwise reproducibility across CURAND versions.

### Mapping, gates and controls

`set_mapping` accepts a complete **physical-to-logical** permutation, validates
length/range/uniqueness, and stores the initial mapping. Allocation or explicit
`reinitialize_mapping` installs it and constructs the inverse. The latter changes
interpretation metadata, not amplitude storage. With no configured mapping it
installs identity. Do not reset mapping during an initializer or infer that
`set_mapping` immediately rearranges amplitudes.

Every API qubit index is logical. Physical bits below L=N-log2(P) index local
storage; higher bits select rank. Reference localization scans target list order
and chooses descending available local physical victims, avoiding all existing
local targets. Retain resulting mapping across calls. The benchmark instead swaps
with physical bit 0; retain that separate benchmark policy. Mapping choices affect
later measurement grouping/RNG consumption, so matching final logical amplitudes
alone is insufficient to ignore reference localization order.

Positive controls require 1, negative controls require 0, jointly. Partition controls
after localization: global controls filter rank participation in local matrix
application; local controls become cuStateVec control bits and values. Every rank
must still participate in required localization/collectives. A contradictory
already-measured control makes the operation a no-op. Measured non-target bits
continue conditioning operations; measured target bits have pending projections
materialized and their records released after the gate.

Single-target functions require exactly one target, not an arbitrary list of
independent applications. Two-target gates require two; RCCX three, RCCCX four;
SWAP takes 2/4/6/8 distinct targets paired consecutively. Numbered SWAPs require
their exact width. Global phase has zero targets. Distinct in-range operands,
disjoint targets/controls and serialized identical collective call order are the
valid domain; several reference violations are assertions or unchecked accesses,
not defined callback errors. Validate these in the new core without promising to
reproduce undefined behavior.

Construct matrices from the pinned `qcs::gate` definitions, including basis order
(first target is low basis bit), signs, and relative/global phases. In particular
`u=u3`, `p=u1`, `u4=exp(i*gamma)*u3`, and global phase multiplies by `exp(i*theta)`.
Use complete complex amplitude comparisons, including controlled operations;
probabilities alone miss phase errors. Localize all targets before evaluating a
compound SWAP; decomposition must not silently accept widths exceeding L.

### Measurement, collapse and classical bits

Reference default is grouped measurement, groups of up to four in caller order.
`measure_many_to_clbits` first sorts local physical targets descending, then global
physical targets descending, groups them, then restores original input order for
results and clbit assignments. Counts must match; qubit and destination lists must
be distinct and in range. Empty requests permit NULL arrays and are no-ops.
The first target in each internal group supplies the least significant outcome bit.
Global-only measurement can use rank probabilities without localization.

Compute raw conditional masses over the previously selected branch. Validate
finite nonnegative bins and a positive finite total; sample using probabilities
relative to that total, without normalizing stored amplitudes. Cache outcomes;
repeating a cached measurement returns the same value without a fresh draw.
Fresh default grouped draws advance the measurement cursor once per fresh group;
empty/cached groups and invalid masses do not consume it. Preserve the documented
ordering/cursor rules; backend-specific optional fast paths can consume draws
differently, so bitwise trajectories across every reference environment backend
are not implied by this header. Default grouped behavior is the baseline target.

Collapse is **projection without rescaling**: rejected amplitudes become zero,
retained amplitudes are unchanged. Reference records logical pending masks and
materializes when needed by a gate, save, expectation or metadata reset. This
contrasts with benchmark full-register measurement, which normalizes the surviving
amplitude with `custatevecCollapseByBitString`. Reuse its probability machinery,
not that normalized collapse semantics. Prefer matching reference deferred masks
and target-release rules; eager projection requires explicit sequence-equivalence
proof, including reset, mapping and raw expectations, before substitution.
`reset_measurement_state` materializes all pending projections before forgetting
records; it does not restore rejected amplitudes. `reset(q)` measures q and applies
X if 1; it does not write a clbit. State setters discard pending metadata because
they overwrite the state, without rewinding measurement RNG.

Clbits are independent host metadata: single `_to_clbit` and many `_to_clbits`
write specified destinations; ordinary measurement does not. `read` returns a bit.
`get_clbits` writes ascending clbit indices; string output reverses that order and
adds NUL (caller needs count+1 bytes). Resetting clbits does not change amplitudes
or measurement records. Newly resized register storage follows reference vector
resize semantics (retained entries survive, new entries false).

### Expectations

Return `coefficient * <psi|P|psi>` without normalization. Empty/all-identity
strings return coefficient times the current squared norm, not coefficient alone.
Validate distinct logical indices including I entries, valid operators, finite
coefficient, counts and pointers. Nonidentity support must fit L; strip I only
after validation. Materialize **all** pending projections before localization;
retain updated mapping, outcomes, clbits and RNG. C result must be non-NULL and
is written only on success. Calls are synchronous, collective and serialized.
Use cuStateVec expectation primitives on local slices then FP64 MPI sum and apply
the coefficient once. Verify installed APIs' normalization conventions with
unnormalized fixtures before choosing the primitive; never normalize implicitly.

### Binary files and timing

Save is headerless native complex-double storage (real then imaginary doubles on
the target platform), one amplitude per logical basis index; expected bytes are
`16 * 2^N`. It first materializes pending projections. Loader reads rank-contiguous
**physical** slices at `rank * 16 * 2^L` from `statevector_input.bin`. Hence saving
and reloading with a nonidentity mapping is not automatically a logical-state
round trip. Preserve that asymmetry. No portable endian/version format is defined.
Reject open/short-read/write/offset failures collectively. Avoid reference's
`fsync(fileno(fp))` after `fclose(fp)` and ensure asynchronous copies finish before
host access; these are implementation hazards, not required observable behavior.

Events are owned CUDA timing events, IDs start at zero. Record uses the compute
stream. Elapsed waits for both events, gets float milliseconds, reduces MPI_MAX
as float, and returns seconds (`*1e-3`). It is not an MPI_Wtime wall-clock interval.
Ensure transfer work precedes the stop event using the existing compute/transfer
synchronization contract. Preserve units and rank maximum; do not replace benchmark
timing with API event timing. Event range misuse is unchecked in the reference;
add safe validation without treating its crashes as a compatibility requirement.

### Errors and evidence boundaries

C entry points return 1 on success, 0 when exceptions are caught, reporting through
the registered process-wide callback with byte length excluding the terminating
NUL. Callback storage is borrowed for the call; do not retain its message pointer.
C++ forms throw, and do not invoke the C translation layer. Exact diagnostic text
and exception subclasses are generally unspecified. Callback reentrancy/throwing
is not a defined workflow; invoke after releasing internal locks and forbid it
from throwing across the C boundary. Serialize callback changes with API use.
Only expectation explicitly guarantees validation of result before side effects;
some legacy wrappers check output pointers after core execution. Do not infer
transactional rollback or a general untouched-output guarantee from their names.

Unconditional MPI init/finalize, unsafe repeated dispose, unchecked indexing,
assertion-only arities and I/O lifetime bugs are not reasons to reproduce crashes.
Record deliberate robustness extensions and test defined valid sequences against
the pinned reference. Exact random sample streams across different reduction
algorithms/hardware are unresolved; preserve documented RNG rules and test
controlled probability fixtures plus distributions, without claiming bitwise
agreement before differential evidence exists.

## Library architecture and ownership design

### MPI and CUDA setup

Introduce a process-wide session manager independent of simulator state. On first
create query MPI_Initialized/MPI_Finalized. If finalized, fail: MPI cannot restart.
If uninitialized, select and initialize CUDA first using the existing non-MPI
assignment helper, then call MPI_Init (or MPI_Init_thread with requested serialized
use and validate supplied support). Record library ownership. If the application
already initialized MPI, borrow it and never finalize it; initialize the selected
CUDA context before creating library communication resources and document that
pre-MPI CUDA ordering is then the application's responsibility.

Reference world-rank semantics stay MPI_COMM_WORLD; the header has no communicator
argument. Do not promise arbitrary subcommunicators. Use a duplicate communicator
with MPI_ERRORS_RETURN for library-owned validation/reductions where possible;
verify the installed cuStateVec communicator/plugin routes swaps through the same
rank group (current external plugin wraps world). Do not modify the caller's world
error handler. Match MPI ABI via existing config and communicator selection.

Maintain process-local reference counts for live cores, including partially
constructed cleanup. Release the MPI lease on dispose; destroy subsequently only
releases the wrapper. Finalize library-owned MPI only after the final live core's
collective disposal, all library requests and resources are released. All ranks
must create/dispose collective handles in the same order. Multiple concurrent
simulators require serialized operations and distinct communication resources;
test this before advertising support. After library-owned finalization, subsequent
create fails; applications wanting repeated sessions should initialize MPI itself.
Never finalize borrowed MPI on success or setup failure.

Respect CUDA_VISIBLE_DEVICES, one GPU per process, launcher local-rank discovery,
post-MPI shared-rank consistency and duplicate-UUID checks. Borrow the device
context, never cudaDeviceReset. Record owning device and restore caller's current
device around operations if it differs. P=1 without launcher rank can use device 0
when only one is visible; multiple-visible/no-local-rank needs an explicit documented
library selection policy before support, not a global-rank guess. Existing benchmark
pre-MPI CUDA setup remains unchanged.

### Resource lifetime and failure protocol

Give each core ownership of state, inverse mappings, clbits, RNG/cursor, measured
masks, cuStateVec handle, compute/transfer streams, events, swap scheduler/worker,
communicator, IPC imports, and mandatory operation scratch. Configuration exists
before allocation; events can be created independently of state storage. Use
partial-construction guards that never enter a collective destructor unexpectedly.

Normal collective close: finish both streams and MPI requests; barrier; destroy
workers; barrier; destroy imported events/close IPC mappings; barrier; free exported
state/events and operation scratch, descriptors, handle and streams; free shared
and duplicated communicators; release MPI lease last. Keep every buffer, local
IPC event and communicator alive until its worker is destroyed. Constructor failure
must distinguish local resources acquired from globally committed resources.

Do not reuse `bench::fail` in library operations: it calls MPI_Abort/world and exit,
so a surrounding try/catch cannot convert it to C status. Extract throwing checked
operations plus an explicit failure policy; benchmark adapters retain abort behavior.
All ranks preflight validation/allocation/I/O readiness at agreed safe boundaries
before dependent collectives. Reduce status, choose a deterministic diagnostic
source, and return/throw coherently on recoverable errors. Prepare candidate
resources then commit collectively; on coordinated OOM release successful candidates
and retry or return, without leaving peers in worker execution.

During distributed execution, place status agreement between ordered phases when
all ranks can still participate. A mid-worker CUDA/MPI/transport error may prevent
safe agreement or leave MPI unusable: mark the session poisoned, diagnose, and use
an explicit documented fatal MPI_Abort fallback if progress cannot be restored.
Do not promise recovery, bounded timeouts or ordinary exception unwinding from a
hung collective/process death. Launcher watchdogs remain required. MPI_ERRORS_RETURN
alone does not provide fault tolerance. This fatal exception to ordinary C errors
must be visible in library documentation. No extra public qcs.h error-policy
parameter is proposed.

### Arbitrary call sequences and workspace preparation

The current runtime creates swap resources only for Hadamard/smoke/check jobs;
P2P imports are filtered to workload peers, and measurement prepares no worker.
Gate workspace queries assume no controls; after `finalize_preparation`, gate and
measurement workspace growth fails. Transfer auto can consume most remaining
memory. Merely linking the existing Runtime into libqcs would fail on measure-then-
gate, first multi-target/controlled operation, or unexpected localization.

Keep the benchmark's closed preparation model. Add an API resource profile:

1. At allocation, plan for the full finite gate catalog and all valid local arities
   up to min(8,L), maximum legal disjoint controls, grouped measurement widths 1..4,
   Pauli evaluation up to L, projection and I/O staging. Query the installed
   cuStateVec APIs with correct targets/controls/layout/compute precision. Do not
   assume workspace monotonicity with arity, control count, matrix contents or
   expectation form: enumerate variants or use a documented upper bound. Cache
   capacities and validate each operation requirement before enqueue.
2. Reserve mandatory state/operation/matrix/host buffers before transfer sizing.
   For all distributed API states with L>0 prepare a worker/scheduler regardless
   of the first operation. Serialize index-bit swaps so possible peers are rank
   XOR each global bit; import all eligible same-node pairs in that graph,
   with mutual-success checks and MPI fallback. This covers arbitrary target
   localization without importing all P peers or guessing from a benchmark job.
3. Retain common-minimum free-memory budgeting, common-maximum worker minimum,
   power-of-two sizing, exact budget arithmetic and coordinated shrinking retries.
   A library may choose minimum transfer size by default to leave application
   headroom; that is a documented library policy, not a benchmark default change.
4. If installed APIs cannot provide a complete finite bound, implement explicit
   collective reprepare between calls: quiesce streams/requests, detach or destroy
   workers safely, reclaim transfer allocation, grow mandatory scratch, recreate
   bindings/worker/IPC registration as needed, then rebudget and collectively
   commit. Never grow memory behind an active worker or retain stale pointers.
   Roll back to a usable prepared state or poison the handle coherently on failure.
   Do not simply disable the finalized check. First release can use eager bounded
   preparation only if the installed-version bound is demonstrated complete.
5. No workload plan or new “prepare” API call is required from callers. Fresh gate
   after measurement, expectation after a new mapping, and repeated initialization
   all work within the allocated dimensions. Host staging needed for save is
   either reserved or collectively allocated before starting I/O.

cuStateVec remains the gate/probability/expectation/swap computational backend.
Use small support kernels for initializers, raw projection and format conversion;
do not transplant the reference NCCL/custom gate/reduction runtime as an alternative
backend. CURAND is a support dependency for the specified random initializer.

## Configurations and operation limits

| Configuration/operation | Existing implementation | Compatibility target and acceptance condition |
| --- | --- | --- |
| Platform/build | 64-bit Linux, C++17, nvcc, CUDA, classic cuStateVec, CUDA-aware MPI | Same baseline; C callers need a C compiler and installed header, shared library built with PIC and nvcc link; verify installed API signatures |
| Distributed topology | P power of two; one distinct GPU per process, contiguous slice; intra/inter-node | Preserve; Open MPI/MPICH-compatible ABI or matching external plugin; no oversubscription or rank-placement assumption |
| Qubit range | N=1..62, L>=1 | Reference Pauli validation defines N=1..63 and L>=0. Implement L=0 scalar-slice path and checked N=63 representability where feasible; otherwise explicitly partial compatibility, never label current limits complete |
| L=0 | Rejected by Topology | Initializers, measurement, raw projection, file output, classical operations, events and identity expectation can work; no nonidentity local support or target gate fits; avoid constructing invalid zero-bit swap resources |
| One-target gates | H/X infrastructure | Require L>=1; controls may remain global and do not each consume a target slot |
| Two-target gates | Generic apply mechanism only | L>=2; localize distinct targets before applying |
| RCCX/RCCCX | Missing | L>=3/4 respectively, plus valid extra disjoint controls |
| SWAP K pairs | Missing logical API | K=1..4, 2K targets, L>=2K as reference requires, even if a decomposition could mathematically work with fewer local bits |
| Global phase | Missing | Zero targets, conditional rank/local phase; test L=0 rather than assume one local target |
| Measurement | Whole register, normalized collapse | Arbitrary subsets including global-only and L=0; no support<=L restriction; preserve grouping/caching, unnormalized projection |
| Pauli | Missing | Nonidentity support<=L; identities may name global bits; empty string valid even at L=0 |
| Memory | Per-rank checked allocation and free-memory policy | `16*2^L` plus scratch/transport/context overhead must fit every rank; N bound alone never guarantees capacity. N=63 total binary size may exceed offset/filesystem limits; reject safely |
| Counts/IDs | Benchmark int/uint64 metadata | Preserve signed 16-bit API counts and C 8-bit narrowing; event record takes int but elapsed takes int16; no silent header widening |
| Threading | Serial harness | Same simulator calls serialized; MPI thread support must match actual use; no concurrent collectives on separate handles until verified |

N=63 is a specification gap with the existing signed cuStateVec index bound, not a
promise of physical allocatability. Audit each installed primitive's index limits;
use unsigned checked host bookkeeping and operation-specific failures or a proven
cuStateVec decomposition. Record any remaining backend limit in the support matrix.
Reference setup's unconditional MPI ownership does not establish valid use with
preinitialized MPI or multiple cores; proposed ownership rules are intentional
library extensions, tested independently of that unsafe reference behavior.

## Stages and acceptance criteria

1. **Freeze interface and oracle.** Install the exact pinned qcs.h in a future
   implementation change; generate a declaration/symbol manifest, both C++
   allocation overloads and inline convenience compile probes. Compile C and C++
   translation units that reference every declaration; link against libqcs without
   changing API calls. Check C exports with nm/readelf and C++ demangled signatures.
   Pin oracle build, GPU/toolchain/environment and reference test status. Confirm
   all ambiguities above with small probes, not assumptions from gate names.
2. **Library lifecycle/build.** Separate core checks and abort policy, introduce
   ownership/session guards, and add libqcs.so as a separate Make target while
   keeping the default benchmark target. Keep CONFIG_MK and its exported
   CFLAGS_VENDOR/LDFLAGS_VENDOR/GENCODE_FLAGS, nvcc compilation/final linking,
   existing overrides and missing-config diagnostics. Do not discover alternate
   MPI wrapper flags, introduce CMake, or generate/overwrite config.mk. Test
   borrowed/owned MPI, partial allocation failures, dispose/destroy, NULL destroy,
   multiple live cores and finalized-MPI rejection in separate launcher processes.
3. **State/configuration/resources.** Implement all six initializers, saved mapping,
   clbits and full preparation. Compare complete small slices to fixtures, including
   P=1 entangled, P>1 endpoints, unnormalized flat/sequential, repeat random restores,
   deferred load and mapping preservation. Verify allocation does not invoke a
   setter/loader; do not test unspecified bytes for a particular value. Exercise
   library profile and coordinated OOM/reprepare with fault injection.
4. **All gates and controls.** Implement the full catalog and SWAP aliases. Compare
   every basis vector and asymmetric complex superpositions against the pinned
   gate formulas/CPU oracle, including parameter boundaries, positive/negative
   local/global controls and phases. Use reversed/nonidentity mappings, P=1/2/4,
   local/global targets and L exactly at/one below required arity. Check retained
   mapping and measured-target release, not only probabilities.
5. **Measurement and Pauli.** Port/adapt public fixtures, keeping reference-only
   private includes out of the new library tests. Test empty/invalid/duplicate lists,
   shuffled clbit destinations, groups crossing width four, cached/fresh mixtures,
   GHZ correlations, global-only targets, inactive/zero-mass ranks, L=0, invalid
   total mass, and repeated measurement-gate-expectation sequences. Verify raw
   branch norm through save and identity expectation (e.g. H|0> then measure has
   branch squared norm 1/2). Use Y-sensitive complex states, negative/zero
   coefficients and untouched expectation sentinel on errors. Match exact
   deterministic fixtures; assess random distributions with predeclared statistical
   bounds and confirm cursor rules independently of cross-backend bitwise draws.
6. **I/O, events, integration and regression.** Compare small binary files bytewise
   on the same platform in identity/nonidentity mappings, raw loader slices and
   projected states; inject file errors on one rank. Verify event units, rank maximum,
   repeated IDs and transfer completion. Build/run unchanged shared-library GHZ and
   Hadamard-measure callers and all-symbol C/C++ callers. Exercise one node and
   multiple nodes, one-visible-GPU isolation, noncontiguous rank placement and
   P2P off/auto/required. Fault-injection runs must finish or abort coherently under
   launcher timeout; a hang is a failure, not a successful error return.

For numerical acceptance use full complex128 comparisons on small states with
initial thresholds `abs_error <= 1e-12 + 5e-12*abs(reference)` for short circuits
and Pauli fixtures (matching reference API-test scale). Predeclare a depth-scaled
bound for longer circuits and investigate failures rather than relaxing silently.
No phase-insensitive comparison for phase-sensitive API output. File fixtures use
exact known representable amplitudes; event durations need unit/order/completion
checks, not equality across runs.

Benchmark regression is a release gate at every runtime extraction stage: run
existing host policy tests and smoke/Hadamard/measurement GPU checks; compare CLI
help/rejections/defaults, ascending/reverse separate H calls and physical-bit-0
localization, normalized uniform/zero/GHZ measurement and seed/warmup behavior.
Verify `Job::prepare` and initialization remain outside timing; timed operations
include all probability work, swaps and completion; MPI_Wtime durations are reduced
with MPI_MAX after timing. Require unchanged rank-0 CSV schema
`job,iteration,nqubits,mpi_ranks,elapsed_seconds,measurement_bits`, default 128 rows,
empty Hadamard measurement fields and logical q[N-1]..q[0] measurement strings.
Metadata stays on stderr. Timing values naturally vary; compare boundaries and
numerical results, not equal elapsed seconds. Preserve benchmark's closed workspace
preparation and transport policies independently of the new API profile.

Completion means every inventoried declaration links and passes its applicable
behavior tests, supported configuration limits are explicit, no unexplained
reference conflict remains, and benchmark/build invariants pass. Until L=0/N=63
and any installed-version restrictions are resolved, advertise only a documented
subset rather than complete compatibility. The foundation stage is implemented; remaining stages require subsequent work.

## Investigation validation

Only source/document inspection and inventory/diff consistency checks are required
for this documentation change. No CUDA/MPI build, GPU execution, timing, sanitizer
or reference differential run is claimed. The staged criteria above are future
implementation acceptance work, not results obtained during this investigation.

## Appendix: exhaustive function declarations

The following declarations are extracted from the pinned header. Parameter lists
are retained to make ABI review independent of table shorthand. The inline C++
convenience definition is included separately. Types and enum values are inventoried
above; no internal qcs namespace types are public in this header.

There are 72 C declarations and 67 throwing C++ declarations
(counting allocation overloads separately), plus one inline C++ convenience overload.

### C linkage

```c
int qcs_set_exception_callback(qcs_exception_callback callback);
int qcs_simulator_create(qcs_simulator **result);
int qcs_simulator_destroy(qcs_simulator *sim);
int qcs_simulator_allocate_memory(qcs_simulator *sim);
int qcs_simulator_allocate_memory_with_initial_state(qcs_simulator *sim, qcs_initial_state initial_state);
int qcs_simulator_dispose(qcs_simulator *sim);
int qcs_simulator_get_num_procs(qcs_simulator *sim, bit_t *result);
int qcs_simulator_get_proc_num(qcs_simulator *sim, bit_t *result);
int qcs_simulator_get_num_qubits(const qcs_simulator *sim, bit_t *result);
int qcs_simulator_get_num_clbits(const qcs_simulator *sim, bit_t *result);
int qcs_simulator_set_num_qubits(qcs_simulator *sim, bit_num_t num_qubits);
int qcs_simulator_set_mapping(qcs_simulator *sim, const bit_num_t *perm_p2l, bit_num_t perm_p2l_count);
int qcs_simulator_set_num_clbits(qcs_simulator *sim, bit_num_t num_clbits);
int qcs_simulator_get_clbits(const qcs_simulator *sim, bit_t *clbits);
int qcs_simulator_get_clbits_string(const qcs_simulator *sim, char *clbits_string);
int qcs_simulator_reset(qcs_simulator *sim, bit_num_t qubit_num);
int qcs_simulator_set_zero_state(qcs_simulator *sim);
int qcs_simulator_set_sequential_state(qcs_simulator *sim);
int qcs_simulator_set_flat_state(qcs_simulator *sim);
int qcs_simulator_set_entangled_state(qcs_simulator *sim);
int qcs_simulator_set_random_state(qcs_simulator *sim);
int qcs_simulator_reset_clbits(qcs_simulator *sim);
int qcs_simulator_reset_measurement_state(qcs_simulator *sim);
int qcs_simulator_reinitialize_mapping(qcs_simulator *sim);
int qcs_simulator_gate_global_phase(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_h(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_x(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_y(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_z(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_s(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_sdg(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_t(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_tdg(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_sx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rx(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_ry(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rz(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_u4(
    qcs_simulator *sim,
    double theta,
    double phi,
    double lambda,
    double gamma,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_u3(
    qcs_simulator *sim,
    double theta,
    double phi,
    double lambda,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_u2(
    qcs_simulator *sim,
    double phi,
    double lambda,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_u1(
    qcs_simulator *sim,
    double lambda,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_u(
    qcs_simulator *sim,
    double theta,
    double phi,
    double lambda,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_p(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_swap(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_swap1(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_swap2(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_swap3(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_swap4(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_iswap(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_id(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_sxdg(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_r(
    qcs_simulator *sim,
    double theta,
    double phi,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rxx(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_ryy(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rzz(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rzx(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_dcx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_ecr(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_xx_plus_yy(
    qcs_simulator *sim,
    double theta,
    double beta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_xx_minus_yy(
    qcs_simulator *sim,
    double theta,
    double beta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rccx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rcccx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
int qcs_simulator_expectation_pauli(
    qcs_simulator *sim,
    const bit_num_t *qubit_num_list,
    const qcs_pauli_operator *operators,
    bit_num_t qubit_num_count,
    double coefficient,
    double *result);
int qcs_simulator_measure(qcs_simulator *sim, bit_num_t qubit_num, bit_t *result);
int qcs_simulator_measure_to_clbit(
    qcs_simulator *sim,
    bit_num_t qubit_num,
    bit_num_t clbit_num,
    bit_t *result);
int qcs_simulator_measure_many(
    qcs_simulator *sim,
    const bit_num_t *qubit_num_list,
    bit_num_t qubit_num_count,
    bit_t *results);
int qcs_simulator_measure_many_to_clbits(
    qcs_simulator *sim,
    const bit_num_t *qubit_num_list,
    bit_num_t qubit_num_count,
    const bit_num_t *clbit_num_list,
    bit_num_t clbit_num_count,
    bit_t *results);
int qcs_simulator_read(qcs_simulator *sim, bit_num_t clbit_num, bit_t *result);
int qcs_simulator_save_statevector(qcs_simulator *sim, const char *outfn);
int qcs_simulator_event_create(qcs_simulator *sim, bit_t *result);
int qcs_simulator_event_record(qcs_simulator *sim, int event_num);
int qcs_simulator_event_get_elapsed_time(
    qcs_simulator *sim,
    event_num_t start_event_num,
    event_num_t stop_event_num,
    double *result);
```

### C++ linkage

```cpp
qcs_simulator *qcs_simulator_create_cxx(void);
void qcs_simulator_destroy_cxx(qcs_simulator *sim);
void qcs_simulator_allocate_memory_cxx(qcs_simulator *sim);
void qcs_simulator_allocate_memory_cxx(qcs_simulator *sim, qcs_initial_state initial_state);
void qcs_simulator_dispose_cxx(qcs_simulator *sim);
int qcs_simulator_get_num_procs_cxx(qcs_simulator *sim);
int qcs_simulator_get_proc_num_cxx(qcs_simulator *sim);
int qcs_simulator_get_num_qubits_cxx(const qcs_simulator *sim);
int qcs_simulator_get_num_clbits_cxx(const qcs_simulator *sim);
void qcs_simulator_set_num_qubits_cxx(qcs_simulator *sim, bit_num_t num_qubits);
void qcs_simulator_set_mapping_cxx(qcs_simulator *sim, const bit_num_t *perm_p2l, bit_num_t perm_p2l_count);
void qcs_simulator_set_num_clbits_cxx(qcs_simulator *sim, bit_num_t num_clbits);
void qcs_simulator_get_clbits_cxx(const qcs_simulator *sim, bit_t *clbits);
void qcs_simulator_get_clbits_string_cxx(const qcs_simulator *sim, char *clbits_string);
void qcs_simulator_reset_cxx(qcs_simulator *sim, bit_num_t qubit_num);
void qcs_simulator_set_zero_state_cxx(qcs_simulator *sim);
void qcs_simulator_set_sequential_state_cxx(qcs_simulator *sim);
void qcs_simulator_set_flat_state_cxx(qcs_simulator *sim);
void qcs_simulator_set_entangled_state_cxx(qcs_simulator *sim);
void qcs_simulator_set_random_state_cxx(qcs_simulator *sim);
void qcs_simulator_reset_clbits_cxx(qcs_simulator *sim);
void qcs_simulator_reset_measurement_state_cxx(qcs_simulator *sim);
void qcs_simulator_reinitialize_mapping_cxx(qcs_simulator *sim);
void qcs_simulator_gate_global_phase_cxx(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_h_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_x_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_y_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_z_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_s_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_sdg_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_t_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_tdg_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_sx_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rx_cxx(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_ry_cxx(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rz_cxx(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_u4_cxx(
    qcs_simulator *sim,
    double theta,
    double phi,
    double lambda,
    double gamma,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_u3_cxx(
    qcs_simulator *sim,
    double theta,
    double phi,
    double lambda,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_u2_cxx(
    qcs_simulator *sim,
    double phi,
    double lambda,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_u1_cxx(
    qcs_simulator *sim,
    double lambda,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_u_cxx(
    qcs_simulator *sim,
    double theta,
    double phi,
    double lambda,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_p_cxx(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_swap_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_iswap_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_id_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_sxdg_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_r_cxx(
    qcs_simulator *sim,
    double theta,
    double phi,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rxx_cxx(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_ryy_cxx(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rzz_cxx(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rzx_cxx(
    qcs_simulator *sim,
    double theta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_dcx_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_ecr_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_xx_plus_yy_cxx(
    qcs_simulator *sim,
    double theta,
    double beta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_xx_minus_yy_cxx(
    qcs_simulator *sim,
    double theta,
    double beta,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rccx_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rcccx_cxx(
    qcs_simulator *sim,
    const bit_num_t *target_qubit_num_list,
    bit_num_t target_qubit_num_count,
    const bit_num_t *negctrl_qubit_num_list,
    bit_num_t negctrl_qubit_num_count,
    const bit_num_t *ctrl_qubit_num_list,
    bit_num_t ctrl_qubit_num_count);
double qcs_simulator_expectation_pauli_cxx(
    qcs_simulator *sim,
    const bit_num_t *qubit_num_list,
    const qcs_pauli_operator *operators,
    bit_num_t qubit_num_count,
    double coefficient);
int qcs_simulator_measure_cxx(qcs_simulator *sim, bit_num_t qubit_num);
int qcs_simulator_measure_to_clbit_cxx(qcs_simulator *sim, bit_num_t qubit_num, bit_num_t clbit_num);
void qcs_simulator_measure_many_cxx(
    qcs_simulator *sim,
    const bit_num_t *qubit_num_list,
    bit_num_t qubit_num_count,
    bit_t *results);
void qcs_simulator_measure_many_to_clbits_cxx(
    qcs_simulator *sim,
    const bit_num_t *qubit_num_list,
    bit_num_t qubit_num_count,
    const bit_num_t *clbit_num_list,
    bit_num_t clbit_num_count,
    bit_t *results);
int qcs_simulator_read_cxx(qcs_simulator *sim, bit_num_t clbit_num);
void qcs_simulator_save_statevector_cxx(qcs_simulator *sim, const char *outfn);
int qcs_simulator_event_create_cxx(qcs_simulator *sim);
void qcs_simulator_event_record_cxx(qcs_simulator *sim, int event_num);
double qcs_simulator_event_get_elapsed_time_cxx(
    qcs_simulator *sim,
    event_num_t start_event_num,
    event_num_t stop_event_num);
```

### Inline C++ convenience overload

```cpp
inline int qcs_simulator_allocate_memory(qcs_simulator *sim, qcs_initial_state initial_state)
{
    return qcs_simulator_allocate_memory_with_initial_state(sim, initial_state);
}
```
