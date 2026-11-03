# Unfused FP64 Hadamard layer with cuStateVec

This C++17/CUDA program uses CUDA device 0 and the low-level cuStateVec C API.
It defaults to 31 qubits and applies one Hadamard to each target in ascending
order. A successful 31-qubit run issues exactly 31 `custatevecApplyMatrix`
calls, each with one target, no controls, no adjoint, and a row-major matrix.
There is no application-level gate fusion or circuit optimizer. Internal
cuStateVec implementation optimizations are outside this guarantee.

The state and matrix use `cuDoubleComplex` / `CUDA_C_64F`, with
`CUSTATEVEC_COMPUTE_64F`. Initialization zeros the GPU state and copies just
one complex scalar to amplitude zero. All initialization and gate operations
use one CUDA stream; synchronization checks asynchronous errors before success
is reported. State-vector results are copied to the host only with `--validate`.

## Build

Prerequisites: a 64-bit host, GNU Make, a CUDA toolkit supporting C++17, the
cuQuantum SDK containing cuStateVec headers and shared libraries, and a compatible
NVIDIA GPU/driver. Set paths to your installed toolkit and SDK:

```sh
make CUDA_ROOT=/usr/local/cuda CUQUANTUM_ROOT=/opt/nvidia/cuquantum
```

If your SDK places libraries in `lib64`, use:

```sh
make CUDA_ROOT=/usr/local/cuda CUQUANTUM_ROOT=/opt/nvidia/cuquantum \
  CUQUANTUM_LIB_DIR=/opt/nvidia/cuquantum/lib64
```

The Makefile embeds the selected cuStateVec library directory as a runtime
search path. Ensure the CUDA runtime is also visible to the dynamic loader,
for example when using a toolkit outside the system loader configuration:

```sh
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

## Run

```sh
./hadamard --nqubits 10
./hadamard
# Equivalent default:
./hadamard --nqubits 31
./hadamard --nqubits 31 --validate
./hadamard --help
```

`--nqubits N` accepts integers from 1 through 31. To select a physical GPU,
expose it as logical device 0, for example `CUDA_VISIBLE_DEVICES=1 ./hadamard`.

At 31 qubits, the state has 2,147,483,648 amplitudes and requires exactly
34,359,738,368 bytes (32 GiB), plus the queried extra workspace and CUDA/library
overhead. A GPU advertised with 32 GiB will generally lack enough free memory.
After creating the stream and library handle, the program queries extra
workspace requirements and available memory before allocating the state and
reusable workspace. Insufficient memory or allocation failures report required
and available bytes and return a nonzero exit code. Qubit count and precision
are never reduced automatically. Concurrent GPU workloads can change available
memory between the query and allocation.

The summary reports qubits, precision, amplitudes, state memory in GiB and
bytes, extra workspace bytes, the number of actual gate API calls issued
(including a failing call if one returns an error), and execution status.
Resources are released on both success and failure; cleanup errors are reported.

## Optional correctness validation

`--validate` preserves initialization and the individual gate applications.
It then computes the total squared norm on the GPU using two FP64 reduction
kernels. A fixed buffer of 1,025 doubles (8,200 bytes) holds partial sums and
the final scalar. This buffer reuses the gate workspace; allocation and the
memory preflight use the larger of gate workspace and validation requirements.
These kernels only read the completed state and do not apply gates.

For 1 through 10 qubits, every complex amplitude is copied back and checked.
For larger states, checks are **sampled**, including index 0, every single-bit
index `1ULL << q`, the final index, and up to 64 additional distinct indices
from a fixed-seed generator. The source specifies the seed and recurrence;
duplicate indices are removed. For 31 qubits, no full host copy or additional
state-vector-sized device buffer is allocated. Only the norm scalar and the
selected amplitudes are copied back.

The expected positive real amplitude is `exp2(-0.5 * double(nqubits))`.
The absolute complex amplitude error (checking both real and imaginary parts)
must be at most `expected_amplitude * 1e-12`. No fixed absolute tolerance floor
is used, so a zero amplitude cannot pass. The squared norm must differ from
1 by at most `1e-11`. NaN and infinity fail either check. Output includes
the measured norm, expected amplitude, tolerances, maximum checked amplitude
error, and PASS/FAIL. A failed check returns a nonzero exit code.
The 31-qubit norm covers the entire state, while its amplitude checks provide
sampled validation rather than exhaustive verification.

On a suitable machine, build using the commands above, then run the small
cases before attempting the full state:

```sh
./hadamard --nqubits 1 --validate
./hadamard --nqubits 2 --validate
./hadamard --nqubits 5 --validate
./hadamard --nqubits 31 --validate
```

Every completed pass also reports the actual API call count and checks one
visit per target, one target per call, zero controls, and `calls == nqubits`.
The single call site passes those tracked target/control counts directly to
`custatevecApplyMatrix`. The matrix remains the same 2-by-2 Hadamard for each
call; no fused matrix, Ex State Vector Updater, or direct uniform-state
initialization is present. This application-level accounting does not imply
one internal CUDA kernel per API call or constrain library optimizations.

## Implementation references and validation status

API argument order and stream behavior were taken from NVIDIA's official
[cuStateVec API reference](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/api-reference/custatevec/index.html),
including `custatevecApplyMatrixGetWorkspaceSize`, `custatevecApplyMatrix`,
and `custatevecSetStream`.

The implementation environment had no existing project build conventions,
`nvcc`, or installed `custatevec.h` available. Installed-header comparison could
therefore not be performed. Compilation, smoke runs, tests, and correctness
validation were skipped as requested. The build and run commands above are
provided for a CUDA/cuStateVec-equipped machine; no runtime correctness or
performance results are claimed. The optional validation implementation has
been added, but none of the small cases or the full 31-qubit case was executed.
The session request to skip tests and validation applies to all these runs;
the missing compiler and cuStateVec installation also prevent building here.
