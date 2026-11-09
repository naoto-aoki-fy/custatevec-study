#!/usr/bin/env python3
"""Exercise launch argument forwarding without MPI or a GPU."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    capture = Path(directory) / 'launcher'
    capture.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    capture.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key not in {
        'MPIEXEC', 'NP_FLAG', 'RANKS', 'BENCHMARK', 'NQUBITS', 'REPEATS', 'WARMUP',
        'SEED', 'P2P', 'TRANSFER_BUFFER', 'TRANSFER_BUFFER_FRACTION', 'COMMUNICATOR',
        'COMMUNICATOR_LIBRARY'}}
    env['MPIEXEC'] = str(capture)
    for job in ('hadamard', 'hadamard-reverse', 'measure'):
        script = root / 'scripts' / f'launch-{job}.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        args = json.loads(subprocess.check_output(['bash', str(script)], env=env, text=True))
        assert args == ['-n', '2', str(root / 'custatevec_bench'), '--job', job,
                        '--nqubits', '31', '--repeats', '128', '--warmup', '0', '--seed', '1',
                        '--p2p', 'auto', '--transfer-buffer', 'auto', '--transfer-buffer-fraction', '0.95']
        custom = dict(env, RANKS='4', BENCHMARK='/path with spaces/bench', COMMUNICATOR='external',
                      COMMUNICATOR_LIBRARY='/path with spaces/plugin.so', TRANSFER_BUFFER='min')
        args = json.loads(subprocess.check_output(['bash', str(script), '--host', 'node one', '--',
                                                   '--nqubits', '8', '--initial-state', 'ghz'], env=custom, text=True))
        assert args[:5] == ['-n', '4', '--host', 'node one', '/path with spaces/bench']
        assert args[-8:] == ['--communicator', 'external', '--communicator-library',
                             '/path with spaces/plugin.so', '--nqubits', '8', '--initial-state', 'ghz']
        assert args[args.index('--transfer-buffer')+1] == 'min'
print('PASS: three launch scripts, defaults, overrides, argument boundaries and spaces')
