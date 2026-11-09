#!/usr/bin/env python3
"""Host-only ABI probes; optional --library also links every public symbol."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--library', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
header = (root / 'include/qcs.h').read_text()
assert hashlib.sha256(header.encode()).hexdigest() == '48930c1e4a61baf5651aa4c6ab308b7fb32a25095c61f3530778c8ece087004f'
declarations = re.findall(r'^(?:int|void|double|qcs_simulator \*)\s*\w*qcs_.*;\s*$', header, re.M)
assert len(declarations) == 139
with tempfile.TemporaryDirectory() as directory:
    temp = Path(directory)
    for cxx in (False, True):
        lines = ['#include "qcs.h"']
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
        command = ['c++' if cxx else 'cc', '-std=c++17' if cxx else '-std=c11', '-Wall', '-Wextra', '-Werror', '-I', str(root / 'include'), str(path)]
        if args.library:
            command += [str(args.library.resolve()), '-o', str(temp / ('cxx' if cxx else 'c'))]
        else:
            command += ['-fsyntax-only']
        subprocess.run(command, check=True)
        print(f'PASS: {count} {"C++" if cxx else "C"} declaration address/type probes')
print('PASS: public header matches pinned reference SHA-256')
