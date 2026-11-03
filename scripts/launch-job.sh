#!/usr/bin/env bash
# Shared implementation; invoke launch-hadamard.sh or launch-measure.sh.
set -euo pipefail

job=$1
shift
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Arguments before -- belong to the launcher; arguments after it belong to
# the benchmark. Arrays preserve spaces without evaluating shell text.
launcher_args=()
while (($#)) && [[ $1 != -- ]]; do
    launcher_args+=("$1")
    shift
done
if (($#)); then shift; fi
communicator_args=()
if [[ -n ${COMMUNICATOR:-} ]]; then
    communicator_args+=(--communicator "$COMMUNICATOR")
fi
if [[ -n ${COMMUNICATOR_LIBRARY:-} ]]; then
    communicator_args+=(--communicator-library "$COMMUNICATOR_LIBRARY")
fi

exec "${MPIEXEC:-mpiexec}" "${NP_FLAG:--n}" "${RANKS:-2}" \
    "${launcher_args[@]}" "${BENCHMARK:-$project_dir/custatevec_bench}" \
    --job "$job" --nqubits "${NQUBITS:-31}" --repeats "${REPEATS:-128}" \
    --warmup "${WARMUP:-0}" --seed "${SEED:-1}" \
    --p2p "${P2P:-auto}" \
    "${communicator_args[@]}" "$@"
