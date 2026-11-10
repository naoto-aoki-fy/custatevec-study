#!/usr/bin/env python3
"""Host-only CLI integration checks. The test double does not validate CUDA/MPI."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(args, **kwargs):
    return subprocess.run([str(a) for a in args], text=True, capture_output=True, **kwargs)


def checked(args, **kwargs):
    result = run(args, **kwargs)
    assert result.returncode == 0, (args, result.stdout, result.stderr)
    return result


with tempfile.TemporaryDirectory(prefix='qcs-cli-') as directory:
    work = Path(directory)
    (work / 'bin').mkdir()
    # Define only API calls actually used by the CLI and the circuit fixture.
    # Exact declarations come from the public header, so signature drift fails.
    source = (ROOT / 'src/qcs_main.cpp').read_text()
    names = set(re.findall(r'\b(qcs_(?:simulator_\w+|set_exception_callback))\s*\(', source))
    names.update(['qcs_simulator_set_num_qubits', 'qcs_simulator_set_num_clbits'])
    bodies = {
        'qcs_simulator_create': '*result = new qcs_simulator; return 1;',
        'qcs_simulator_destroy': 'delete sim; return 1;',
        'qcs_simulator_get_num_qubits_cxx': 'return sim->qubits;',
        'qcs_simulator_get_num_clbits_cxx': 'return 300;',
        'qcs_simulator_get_proc_num_cxx': 'return std::getenv("QCS_TEST_RANK") ? std::atoi(std::getenv("QCS_TEST_RANK")) : 0;',
        'qcs_simulator_get_clbits_string_cxx': 'std::memset(clbits_string, sim->runs ? \'1\' : \'0\', 300); clbits_string[300] = 0;',
        'qcs_simulator_set_num_qubits': 'sim->qubits = num_qubits; return 1;',
        'qcs_simulator_set_num_clbits': 'if (sim->runs++) std::abort(); return 1;',
        'qcs_simulator_reset_clbits_cxx': 'sim->runs = 0;',
        'qcs_simulator_event_get_elapsed_time': '*result = 0.125; return 1;',
        'qcs_simulator_save_statevector_cxx': 'std::ofstream(outfn) << "saved";',
    }
    definitions = []
    header = (ROOT / 'include/qcs.h').read_text()
    for name in sorted(names):
        matches = re.findall(r'^(?:int|void|double|qcs_simulator \*)\s*' + name + r'\([^;]*\);', header, re.M)
        assert matches, name
        for declaration in matches:
            body = bodies.get(name, 'return 1;' if declaration.startswith('int ') else '')
            definitions.append(declaration[:-1] + ' { ' + body + ' }')
    stub = work / 'stub.cpp'
    stub.write_text('#include "qcs.h"\n#include <cstdlib>\n#include <cstring>\n#include <fstream>\n'
                    'struct qcs_simulator { int qubits = 2; int runs = 0; };\n' + '\n'.join(definitions))
    cc = os.environ.get('CC', 'cc')
    cxx = os.environ.get('CXX', 'c++')
    checked([cxx, '-std=c++17', '-shared', '-fPIC', '-I' + str(ROOT / 'include'), stub, '-o', work / 'libqcs.so'])
    checked([cc, '-std=c11', '-Wall', '-Wextra', '-Werror', '-c', ROOT / 'src/qcs_args.c', '-o', work / 'args.o'])
    executable = work / 'bin/qcs'
    checked([cxx, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I' + str(ROOT / 'include'),
             ROOT / 'src/qcs_main.cpp', work / 'args.o', '-L' + str(work), '-lqcs', '-ldl',
             '-Wl,-rpath,$ORIGIN/..', '-o', executable])
    assert 'Usage: qcs' in checked([executable, '--help']).stdout
    assert checked([executable, '--version']).stdout.strip() == 'qcs 1.0'
    for args in [[], ['--unknown'], ['-s', 'abc'], ['-s', '4294967297'], ['-s', ''], ['-s', '0', 'missing.so'],
                 ['--initial-state', 'bad', 'missing.so'], ['missing.so']]:
        result = run([executable, *args])
        assert result.returncode > 0 and result.stderr, (args, result)
    assert 'Circuit binary not found' in run([executable, 'missing.so']).stderr
    bad_file = work / 'bad.so'
    bad_file.write_text('not a shared object')
    result = run([executable, bad_file])
    assert result.returncode > 0 and 'Failed to load circuit binary' in result.stderr
    for symbols, missing in [('void other(void) {}', 'circuit_init'),
                             ('void circuit_init(void *s) {}', 'circuit_run')]:
        circuit = work / 'missing.c'
        circuit.write_text(symbols)
        checked([ROOT / 'compile_circuit.sh', circuit, '-o', work / 'missing.so'])
        result = run([executable, work / 'missing.so'])
        assert result.returncode > 0 and missing in result.stderr, result
    circuit = work / 'circuit.c'
    circuit.write_text('#include "qcs.h"\n'
                       'void circuit_init(qcs_simulator *s) { qcs_simulator_set_num_qubits(s, 2); }\n'
                       'void circuit_run(qcs_simulator *s) { qcs_simulator_set_num_clbits(s, 300); }\n')
    checked([ROOT / 'compile_circuit.sh', circuit, '-o', work / 'circuit.so'])
    for initial in ['zero', 'flat', 'entangled', 'sequential', 'use_curand', 'load_statevector']:
        result = checked([executable, '-s', '3', '--mapping', '1,0', '-r', '--initial-state', initial,
                          '--output-statevector', work / 'state.bin', work / 'circuit.so'])
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        assert [row['sample_num'] for row in rows] == [0, 1, 2]
        assert all(row['clbits'] == '1' * 300 and row['elapsed_time'] == 0.125 for row in rows)
    assert (work / 'state.bin').read_text() == 'saved'
    for rank in ['1', '256']:
        assert not checked([executable, work / 'circuit.so'], env={**os.environ, 'QCS_TEST_RANK': rank}).stdout
    for mapping in ['', '0,', '0,x', '0,1junk', '0,65536', '-1,0', '0,0', '0']:
        result = run([executable, '--mapping', mapping, work / 'circuit.so'])
        assert result.returncode > 0 and result.stderr, (mapping, result)
print('QCS CLI host checks passed (test double; no CUDA/MPI execution)')
