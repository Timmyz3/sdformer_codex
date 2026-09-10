"""Versioned complete-operator CPU experiment; preserves failures and exact inputs."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert len(sys.argv) == 2, 'Usage: python3.12 c1_run.py new_attempt_name'
    name = sys.argv[1]
    assert name.startswith('c1_attempt_') and Path(name).name == name
    target = BASE / name
    target.mkdir(exist_ok=False)
    snapshot = target / 'source_snapshot'
    snapshot.mkdir()
    files = ['c1_cycle_model.cpp', 'c1_plan.json', 'c1_plan_amendments.json',
             'c1_input_identity.json', 'c1_run.py']
    identities = {f: sha(BASE / f) for f in files + ['c1_input.bin']}
    for f in files:
        shutil.copyfile(BASE / f, snapshot / f)
    receipt = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'input_sha256': identities, 'commands': [], 'status': 'RUNNING'}
    commands = [
        ['g++', '-O2', '-std=c++14', '-Wall', '-Wextra', '-o', str(target / 'model'),
         str(snapshot / 'c1_cycle_model.cpp')],
        [str(target / 'model'), str(BASE / 'c1_input.bin'), str(target / 'result.json')],
    ]
    for index, command in enumerate(commands):
        with (target / f'{index}_stdout.log').open('w') as out, (target / f'{index}_stderr.log').open('w') as err:
            run = subprocess.run(command, stdout=out, stderr=err, cwd=BASE)
        receipt['commands'].append({'argv': command, 'exit_code': run.returncode})
        if run.returncode:
            receipt['status'] = 'FAILED_DO_NOT_CITE'
            break
    else:
        result = json.loads((target / 'result.json').read_text())
        assert len(result['points']) == 10
        assert all(p['diagnostic_lane_values_checked'] == 24000 for p in result['points'])
        receipt['result_sha256'] = sha(target / 'result.json')
        receipt['binary_sha256'] = sha(target / 'model')
        receipt['status'] = 'COMPLETE_CPU_MODEL_PENDING_INDEPENDENT_REVIEW'
    receipt['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (target / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    if receipt['status'].startswith('FAILED'):
        sys.exit(1)


if __name__ == '__main__':
    main()
