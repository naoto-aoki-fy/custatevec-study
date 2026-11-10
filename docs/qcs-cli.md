# Standalone QCS CLI

The CLI and circuit helper are ported from [rqs-svg revision
10a4da5b29a65f3eec0f556344ce06c3ff24072a](https://github.com/jhpc-quantum/rqs-svg/tree/10a4da5b29a65f3eec0f556344ce06c3ff24072a),
under the repository's Apache-2.0 license. The generated argument parser is
included; gengetopt and atlc are not required. Local changes add checked symbols,
API errors, mapping/integer validation and resource cleanup. The public API and
numerical library are unchanged.

## Build and run

Use the [library prerequisites and site configuration](qcs-library.md):

```sh
make bin/qcs CONFIG_MK=/path/to/existing/config.mk CUSTATEVEC_ROOT=/path/to/cuquantum
./bin/qcs --help
./bin/qcs --version
```

The default `make` target still builds only `custatevec_bench`. CLI C++ compilation
and linking use nvcc with `CFLAGS_VENDOR`, `LDFLAGS_VENDOR`, `GENCODE_FLAGS` and the
existing extra flag overrides. The parser uses host `CC`, `CPPFLAGS`, `CFLAGS`
and `HOST_CFLAGS` (default `-O2 -std=c11`). Do not pass nvcc-only flags to host
flag variables. The executable finds `libqcs.so` relative to its own location
(`$ORIGIN/..`); keep `bin/qcs` and the root library in that layout. CUDA/MPI runtime
libraries must also be available through the configured runtime search paths or
site environment.

For the preserved upstream programs, see [imported examples](qcs-examples.md).

A minimal C circuit (save as `circuit.c`):

```c
#include <qcs.h>
void circuit_init(qcs_simulator *sim) {
    qcs_simulator_set_num_qubits(sim, 2);
    qcs_simulator_set_num_clbits(sim, 2);
}
void circuit_run(qcs_simulator *sim) {
    bit_num_t target = 0;
    bit_t result;
    qcs_simulator_gate_h(sim, &target, 1, NULL, 0, NULL, 0);
    qcs_simulator_measure_to_clbit(sim, 0, 0, &result);
}
```

```sh
./compile_circuit.sh circuit.c -o circuit.so
mpiexec -n 1 ./bin/qcs --num-samples 3 ./circuit.so
```

The helper accepts compiler arguments directly and honors `CC` (one executable),
`CPPFLAGS`, `CFLAGS` and `LDFLAGS` (whitespace-separated flags). Pass arguments
containing spaces directly as quoted arguments. Circuit files export
`void circuit_init(qcs_simulator *)` and `void circuit_run(qcs_simulator *)` with
C linkage. QCS references in the shared circuit resolve from the executable's
linked `libqcs.so`. Both exports and all load-time references are checked before
simulator creation. The circuit should configure dimensions in `circuit_init`;
the CLI allocates and initializes storage afterward. Circuit API failures reported
through the CLI callback cause a nonzero exit after the circuit function returns.
All ranks must execute matching collective calls; circuit code remains responsible
for coherent control flow and checking return values inside a circuit.

Options retain upstream spelling and defaults:

- `-s, --num-samples N`: positive sample count, default 1.
- `--mapping CSV`: physical-to-logical permutation; default identity.
- `-r, --reversed-mapping`: invert the supplied mapping.
- `--initial-state NAME`: `zero` (default), `flat`, `entangled`, `sequential`,
  `use_curand`, or `load_statevector`. The last reads `statevector_input.bin`.
- `--output-statevector PATH`: save the final sample's state through the library.

Each sample records events around `circuit_run`, then rank zero prints one JSON
line with `sample_num` (starting at zero), `clbits`, and `elapsed_time` in seconds.
Between samples the CLI restores mapping, the chosen initial state, and classical
bits. Output and initialization formats follow the existing library contract.

## Validation

```sh
python3 tests/check_qcs_cli.py
python3 tests/check_qcs_header.py
```

The CLI check compiles the real parser and CLI with strict host warnings and links
them to a temporary test double named `libqcs.so`. It checks help/version, invalid
options and sample counts, absent files and exports, mapping validation, all initial
state names, repeated samples, saving, runtime library lookup, resolution of C QCS
symbols from a circuit, large classical registers and rank-zero-only JSON output
(including rank 256). It does not validate GPU computations or MPI communication.

In this workspace these checks pass, as do existing host device assignment,
transfer policy, Hadamard order, gate, measurement and observation checks.
`make bin/qcs` stops at the required missing site `config.mk`; nvcc and MPI tools
are unavailable. Actual linkage to the cuStateVec implementation and GPU/MPI
execution have not been validated. On a configured machine, build and run the
minimal circuit above and the existing library/benchmark regression checks linked
from [QCS validation](qcs-validation.md).
