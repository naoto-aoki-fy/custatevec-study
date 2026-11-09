#!/usr/bin/env python3
"""Host-only ABI probes; optional --library verifies real linking and dynamic exports."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--library', type=Path)
parser.add_argument('--reference-root', type=Path, help='Checkout of the pinned reference; compile unchanged callers')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
include = root / 'include'
if args.reference_root:
    revision = subprocess.check_output(['git', '-C', str(args.reference_root), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == '10a4da5b29a65f3eec0f556344ce06c3ff24072a', revision
    subprocess.run(['git', '-C', str(args.reference_root), 'diff', '--exit-code', 'HEAD', '--',
                    'include/qcs.h', 'examples/sharedlibrary/ghz.c',
                    'examples/sharedlibrary/hadamard-measure.c', 'tests/pauli_api_c.c',
                    'tests/pauli_api_test.cpp'], check=True)
    include = args.reference_root.resolve() / 'include'
    assert (include / 'qcs.h').read_bytes() == (root / 'include/qcs.h').read_bytes()
header = (include / 'qcs.h').read_text()
assert hashlib.sha256(header.encode()).hexdigest() == '48930c1e4a61baf5651aa4c6ab308b7fb32a25095c61f3530778c8ece087004f'
declarations = re.findall(r'^(?:int|void|double|qcs_simulator \*)\s*\w*qcs_.*;\s*$', header, re.M)
assert len(declarations) == 139
with tempfile.TemporaryDirectory() as directory:
    temp = Path(directory)
    for cxx in (False, True):
        lines = ['#include "qcs.h"']
        assertion = 'static_assert' if cxx else '_Static_assert'
        for expression in ('sizeof(bit_num_t)==2', 'sizeof(proc_num_t)==4',
                           'sizeof(bit_t)==1', 'sizeof(event_num_t)==2',
                           '(bit_num_t)-1<0', '(proc_num_t)-1<0', '(event_num_t)-1<0', '(bit_t)-1>0',
                           'QCS_INITIAL_STATE_ZERO==0 && QCS_INITIAL_STATE_FLAT==1 && QCS_INITIAL_STATE_ENTANGLED==2',
                           'QCS_INITIAL_STATE_SEQUENTIAL==3 && QCS_INITIAL_STATE_USE_CURAND==4 && QCS_INITIAL_STATE_LOAD_STATEVECTOR==5',
                           'QCS_PAULI_I==0 && QCS_PAULI_X==1 && QCS_PAULI_Y==2 && QCS_PAULI_Z==3'):
            lines.append(f'{assertion}({expression}, "public type/enum contract");')
        count = 0
        for declaration in declarations:
            if not cxx and '_cxx(' in declaration:
                continue
            match = re.match(r'(.+?)\s*(qcs_\w+)\((.*)\);', declaration.strip())
            result, name, params = match.groups()
            types = ', '.join(re.sub(r'\b\w+$', '', p.strip()).strip() for p in params.split(','))
            # Explicit pointer types resolve both C++ allocation overloads.
            lines.append(f'{result} (*probe_{count})({types}) = &{name};')
            count += 1
        if cxx:
            lines.append('int (*inline_allocation)(qcs_simulator*, qcs_initial_state) = &qcs_simulator_allocate_memory;')
        lines.append('int main(void) { return 0; }')
        path = temp / ('probe.cpp' if cxx else 'probe.c')
        path.write_text('\n'.join(lines) + '\n')
        command = ['c++' if cxx else 'cc', '-std=c++17' if cxx else '-std=c11', '-Wall', '-Wextra', '-Werror', '-I', str(include), str(path)]
        if args.library:
            command += [str(args.library.resolve()), '-o', str(temp / ('cxx' if cxx else 'c'))]
        else:
            command += ['-c', '-o', str(temp / ('cxx.o' if cxx else 'c.o'))]
        subprocess.run(command, check=True)
        print(f'PASS: {count} {"C++" if cxx else "C"} declaration address/type probes')
    if args.reference_root:
        for relative in ('examples/sharedlibrary/ghz.c', 'examples/sharedlibrary/hadamard-measure.c',
                         'tests/pauli_api_c.c', 'tests/pauli_api_test.cpp'):
            source = args.reference_root.resolve() / relative
            cxx = source.suffix == '.cpp'
            output = temp / (source.stem + '.o')
            subprocess.run(['c++' if cxx else 'cc', '-std=c++17' if cxx else '-std=c11',
                            '-Wall', '-Wextra', '-Werror', '-I', str(include), '-c', str(source), '-o', str(output)], check=True)
            print(f'PASS: unchanged reference caller compiled: {relative}')
        if args.library:
            for name in ('ghz', 'hadamard-measure'):
                subprocess.run(['cc', str(temp / (name + '.o')), str(args.library.resolve()), '-o', str(temp / name)], check=True)
            subprocess.run(['c++', str(temp / 'pauli_api_test.o'), str(temp / 'pauli_api_c.o'),
                            str(args.library.resolve()), '-o', str(temp / 'pauli_api_test')], check=True)
            print('PASS: unchanged reference callers linked (not executed)')
    if args.library:
        # Probe objects carry the compiler's exact mangled overload names.
        exports = {line.split()[-1] for line in subprocess.check_output(
            ['nm', '-D', '--defined-only', str(args.library.resolve())], text=True).splitlines() if line.split()}
        # Recompile without linking so undefined references remain inspectable.
        required = set()
        for suffix, compiler, standard in (('c', 'cc', 'c11'), ('cpp', 'c++', 'c++17')):
            obj = temp / (suffix + '-exports.o')
            subprocess.run([compiler, '-std=' + standard, '-I', str(include), '-c',
                            str(temp / ('probe.' + suffix)), '-o', str(obj)], check=True)
            required.update(line.split()[-1] for line in subprocess.check_output(
                ['nm', '-u', str(obj)], text=True).splitlines() if 'qcs_' in line)
        assert not required - exports, sorted(required - exports)
        print(f'PASS: {len(required)} required dynamic exports, including C++ overloads')
print('PASS: public header matches pinned reference SHA-256')
