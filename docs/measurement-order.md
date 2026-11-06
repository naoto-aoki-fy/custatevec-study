# Descending measurement bit order

The measurement job passes local physical qubits in descending order:
`local_bits - 1, ..., 0`. This remains one joint measurement, not a sequence
of individual measurements or gates.

The [cuStateVec API reference](https://docs.nvidia.com/cuda/cuquantum/26.03.2/custatevec/api-reference/custatevec/index.html#custatevecbatchmeasurewithoffset)
specifies that `custatevecBatchMeasureWithOffset` writes an integer array
whose entries correspond to `bitOrdering`. Thus entry `i` belongs to physical
qubit `bitOrdering[i]`; it is not a packed integer with bit position `i`.
NVIDIA's [batch measurement sample](https://github.com/NVIDIA/cuQuantum/blob/main/samples/custatevec/custatevec/batch_measure.cu)
uses descending `{2,1,0}`, returns `{1,1,0}`, and collapses physical index 6
for draw 0.5. This confirms physical-index cumulative sampling rather than
sampling a bit-reversed probability array. The check includes this fixture.
The implementation reconstructs the physical outcome using that qubit position.
It passes the unchanged value/ordering pairs as the mask to
`custatevecAbs2SumArray` and as the target to `custatevecCollapseByBitString`.
The unmasked rank norm query has no bit ordering. Selected amplitude magnitude
sets normalization, retaining its complex phase.

MPI rank selection still uses measured rank masses and the same conditional
random draw. Rank bits still supply the global physical qubits, and the final
physical outcome is translated through the logical-to-physical mapping. CSV
`measurement_bits` remains logical MSB to LSB. Outcome probabilities and the
collapsed state retain their meaning; reversing the returned array alone would
incorrectly reverse local qubit identities.

Preparation still allocates the same buffers. The timed path has the same
cuStateVec calls, synchronizations, MPI operations, and asymptotic work.
Library kernel performance can depend on ordering; no performance improvement
or regression has been measured here. Compare timings on the target hardware.

## Validation

`--check-measure` explicitly checks descending ordering. In addition to the
existing initializer, rank-mass, boundary, normalization, and phase checks, it
exercises every one-hot local basis pattern on every rank, dense asymmetric
complex amplitudes, and reversed logical mappings. The CPU reference checks
outcomes against the physical-index cumulative distribution; it independently
checks the logical outcome and surviving physical amplitude.

Run configurations with at least two local bits to expose bit reversal, and
one local bit to cover the distributed minimum:

```sh
mpiexec -n 1 ./custatevec_bench --check-measure --nqubits 4
mpiexec -n 2 ./custatevec_bench --check-measure --nqubits 4
mpiexec -n 4 ./custatevec_bench --check-measure --nqubits 4
mpiexec -n 4 ./custatevec_bench --check-measure --nqubits 3
```

These GPU/MPI checks require the target CUDA/cuStateVec/MPI environment.
