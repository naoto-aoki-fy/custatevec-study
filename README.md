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
is reported. No state-vector results are copied back to the host.

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
performance results are claimed. Detailed correctness validation is deferred
to the follow-up task.
