# Imported QCS examples

All eleven C files under `examples/` are copied byte-for-byte from
[rqs-svg at 10a4da5b29a65f3eec0f556344ce06c3ff24072a](https://github.com/jhpc-quantum/rqs-svg/tree/10a4da5b29a65f3eec0f556344ce06c3ff24072a/examples).
They are distributed under the upstream Apache-2.0 license, reproduced in the
root [LICENSE](../LICENSE). All upstream source content is retained; this revision
has no NOTICE file. No example implementation changes or upstream CUDA/NCCL
backend files are imported. Local additions are build rules, documentation and
validation outside the imported directory. SHA-256 hashes in
`tests/qcs_examples_upstream.json` record the original files.

## Build and run

Use 64-bit Linux, CUDA/nvcc (including CURAND), classic cuStateVec and a matching
CUDA-aware MPI installation, with one distinct GPU per process. Load the site's
compiler/MPI environment and supply an existing nvccoptions `config.mk` containing
`CFLAGS_VENDOR`, `LDFLAGS_VENDOR`, and `GENCODE_FLAGS`. See
[library setup](qcs-library.md) for communicator ABI and GPU assignment details.

```sh
export CONFIG_MK=/path/to/existing/config.mk
export CUSTATEVEC_ROOT=/path/to/cuquantum
make bin/qcs library examples
```

`examples` builds both example groups, separately from the default benchmark.
`examples-standalone` builds six loadable circuits with the host C compiler and
needs no site configuration or CUDA installation. `examples-sharedlibrary` builds
five ordinary C executables: host `CC` compiles C objects and the existing nvcc
link configuration links them against this project's `libqcs.so`. It does not
use the upstream library layout. Host compilation uses `CPPFLAGS`, `CFLAGS` and
`HOST_CFLAGS`; standalone linking also honors `LDFLAGS`. CUDA/MPI flags remain in
the existing vendor/extra nvcc variables.

Standalone circuits export `circuit_init` and `circuit_run`; `bin/qcs` loads them,
initializes the simulator, runs samples and resets state between samples.
Every standalone example reads `NUM_QUBITS` from the environment:

```sh
./compile_circuit.sh examples/standalone/ghz_circuit.c -o ghz_circuit.so
NUM_QUBITS=3 mpiexec -n 1 ./bin/qcs --num-samples 10 ./ghz_circuit.so
# Or use the Make-built module:
NUM_QUBITS=3 mpiexec -n 1 ./bin/qcs --num-samples 10 build/examples/standalone/ghz_circuit.so
```

Shared-library examples own their simulator lifecycle and accept command-line
qubit/sample counts rather than the standalone environment variable:

```sh
mpiexec -n 1 ./build/examples/sharedlibrary/ghz --num-qubits 3 --num-samples 10
```

Executables find the root `libqcs.so` through a relative runtime search path.
Keep the build directory layout intact. CUDA, cuStateVec, CURAND, MPI and the
cuStateVec MPI communicator plugin must be discoverable through site runtime
paths or `LD_LIBRARY_PATH` (include `$CUSTATEVEC_LIBDIR`, normally
`$CUSTATEVEC_ROOT/lib`, and your site's CUDA/MPI library directories). Configure
`QCS_CUSTATEVEC_COMMUNICATOR_LIBRARY` if your MPI ABI requires a different plugin.
MPI must forward `NUM_QUBITS` to ranks for standalone examples.

Both groups contain single-qubit/all-qubit Hadamard examples without measurement
(`hadamard-single`, `hadamard-whole`, empty classical strings) and measured
Hadamard circuits (`hadamard-measure`, `measure_circuit`). GHZ examples measure
correlated bits (`ghz_circuit` standalone, `ghz` shared). The standalone-only
`measure-only` measures the zero state. Output is one JSON record per sample on
rank zero; GHZ strings must be all zero or all one.

## Validation

```sh
make test-qcs-examples-host
python3 tests/check_qcs_cli.py
python3 tests/check_qcs_header.py
# Requires the real build above, GPUs and MPI:
python3 tests/check_qcs_examples.py --runtime
# Requires two distinct GPUs; --launcher accepts site MPI options:
python3 tests/check_qcs_examples.py --runtime --ranks 2
```

The host example check verifies all imported hashes, compiles every source with
`-Wall -Wextra -Werror`, and checks both dynamic exports of each circuit. It
compiles shared-library examples to objects only; this is not a real link test.
The optional runtime check runs all eleven built examples for sixteen samples,
checks JSON sample numbers/widths, zero measurements and GHZ correlations, and
uses a separate temporary X/measurement fixture to verify sample reinitialization.
It imposes a 180-second timeout per launch. It does not require both stochastic
GHZ outcomes to appear in a short run. A failure is reported rather than silently
skipped when `--runtime` is requested.

Executed here: all host example checks, all standalone Make builds, existing CLI
checks (test double), public-header checks and QCS gate/measurement/observation
host regressions passed. The imported files were also compared directly with the
pinned checkout. No nvcc, mpiexec, GPU tooling or site `config.mk` is available;
real shared-library linking, circuit loading against cuStateVec, GPU measurement,
sample reset and multi-rank execution were not run. For real library, MPI and
benchmark regression commands, follow [QCS validation](qcs-validation.md),
including foundation, gate, measurement and observation suites.
