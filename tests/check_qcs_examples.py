#!/usr/bin/env python3
"""Preservation/host compilation checks; opt-in real GPU/MPI acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime', action='store_true', help='Run built examples against the real QCS library')
parser.add_argument('--ranks', type=int, default=1, choices=(1, 2))
parser.add_argument('--launcher', default='mpiexec', help='MPI command and optional flags, shell-style quoting')
args = parser.parse_args()


def checked(command, **kwargs):
    result = subprocess.run([str(x) for x in command], cwd=ROOT, text=True,
                            capture_output=True, timeout=180, **kwargs)
    if result.returncode:
        raise RuntimeError(f'{command}: exit {result.returncode}\n{result.stdout}\n{result.stderr}')
    return result.stdout


manifest = json.loads((ROOT / 'tests/qcs_examples_upstream.json').read_text())
files = sorted((ROOT / 'examples').rglob('*.c'))
assert {str(p.relative_to(ROOT / 'examples')) for p in files} == set(manifest)
for source in files:
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest[str(source.relative_to(ROOT / 'examples'))], source
with tempfile.TemporaryDirectory(prefix='qcs-examples-') as directory:
    work = Path(directory)
    for source in files:
        output = work / (source.stem + ('.so' if source.parent.name == 'standalone' else '.o'))
        if source.parent.name == 'standalone':
            checked([ROOT / 'compile_circuit.sh', '-Wall', '-Wextra', '-Werror', source, '-o', output])
            exports = {line.split()[-1] for line in checked(['nm', '-D', '--defined-only', output]).splitlines()}
            assert {'circuit_init', 'circuit_run'} <= exports, source
        else:
            checked([os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror',
                     '-Iinclude', '-c', source, '-o', output])
print('PASS: all 11 upstream files unchanged; 6 circuit builds/exports and 5 C object builds')

if args.runtime:
    launcher = shlex.split(args.launcher) + ['-n', str(args.ranks)]
    samples = 16
    env = {**os.environ, 'NUM_QUBITS': '3'}

    def verify(command, width, allowed=None):
        rows = [json.loads(line) for line in checked(launcher + command, env=env).splitlines() if line.strip()]
        assert len(rows) == samples, rows
        assert [row['sample_num'] for row in rows] == list(range(samples)), rows
        for row in rows:
            bits = row['clbits']
            assert len(bits) == width and set(bits) <= {'0', '1'}, row
            assert row['elapsed_time'] >= 0, row
            if allowed is not None:
                assert bits in allowed, row

    for source in files:
        width = 0 if source.stem in ('hadamard-single', 'hadamard-whole') else 3
        allowed = {'000', '111'} if source.stem in ('ghz', 'ghz_circuit') else None
        if source.stem == 'measure-only':
            allowed = {'000'}
        if source.parent.name == 'standalone':
            command = [ROOT / 'bin/qcs', '--num-samples', str(samples),
                       ROOT / 'build/examples/standalone' / (source.stem + '.so')]
        else:
            command = [ROOT / 'build/examples/sharedlibrary' / source.stem,
                       '--num-qubits', '3', '--num-samples', str(samples)]
        verify(command, width, allowed)
    # X on |0> followed by measurement distinguishes reinitialization from
    # accidentally reusing the previous sample's collapsed state (1,0,1,...).
    with tempfile.TemporaryDirectory(prefix='qcs-reset-') as directory:
        circuit = Path(directory) / 'reset.c'
        circuit.write_text('#include "qcs.h"\n'
                           'void circuit_init(qcs_simulator *s) {\n'
                           'qcs_simulator_set_num_qubits(s, 3); qcs_simulator_set_num_clbits(s, 1); }\n'
                           'void circuit_run(qcs_simulator *s) { bit_num_t t=0; bit_t b;\n'
                           'qcs_simulator_gate_x(s, &t, 1, 0, 0, 0, 0);\n'
                           'qcs_simulator_measure_to_clbit(s, 0, 0, &b); }\n')
        binary = Path(directory) / 'reset.so'
        checked([ROOT / 'compile_circuit.sh', circuit, '-o', binary])
        verify([ROOT / 'bin/qcs', '--num-samples', str(samples), binary], 1, {'1'})
    print(f'PASS: real QCS execution of all 11 examples, zero/GHZ/reset checks, {args.ranks} MPI rank(s)')
else:
    print('SKIP: real library linking/loading, GPU execution and MPI checks (use --runtime after make examples bin/qcs)')
