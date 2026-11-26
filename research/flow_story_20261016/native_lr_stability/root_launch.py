#!/usr/bin/env python3
"""Root orchestration of the reviewed, fixed native-only LR diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent
R13 = HERE.parent / 'coverage_recovery_round13'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'recover'), required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--label', required=True)
    a = parser.parse_args()
    freeze = json.loads((HERE / 'EXECUTION_FREEZE_V1.json').read_text())
    for local_path, expected in freeze['source_sha256'].items():
        assert sha(Path(local_path)) == expected, local_path
    remote = '/data/flow_a/code/research_20261016/native_lr_stability'
    r13code = '/data/flow_a/code/research_20261016/coverage_recovery_round13'
    r13runs = '/data/flow_a/runs/coverage_recovery_round13_20261016'
    runs = '/data/flow_a/runs/native_lr_stability_20261016'
    paths = {
        'runtime-manifest': (R13 / 'runtime_v1.json', r13code + '/runtime_v1.json'),
        'membership': (R13 / 'data_protocol/manifest_v1.json', r13code + '/data_protocol/manifest_v1.json'),
        'bn-buffers': (R13 / 'preparation/bn_buffers.npz', r13runs + '/bn_v1/bn_buffers.npz'),
        'bn-receipt': (R13 / 'preparation/bn_receipt.json', r13runs + '/bn_v1/bn_receipt.json'),
        'baseline-result': (R13 / 'raw/experiments_v1/diversity/recover/native/results.json',
                            r13runs + '/experiments_v1/diversity/recover/native/results.json'),
    }
    command = ['env', 'CUDA_VISIBLE_DEVICES=0', 'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2',
               '/data/flow_a/envs/v2/bin/python', '-B', remote + '/run_native_lr.py']
    for key, (local, target) in paths.items():
        command.extend(['--' + key, target, '--' + key + '-sha256', sha(local)])
    command.extend(['--mode', a.mode, '--runner-checks', remote + '/runner_cpu_results.json',
                    '--output-dir', runs + '/' + a.mode + '_v1', '--wall-minutes', '30'])
    if a.mode == 'recover':
        command.extend(['--smoke-receipt', runs + '/smoke_v1/results.json'])
    if a.execute:
        command.append('--execute')
    else:
        command.extend(['--preflight-output', runs + '/' + a.label + '.json'])
    argv = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ControlPath=/tmp/flow_a100_r13b_%C', '-p', '30449', 'root@36.213.75.115',
            ' '.join(shlex.quote(x) for x in command)]
    logs = HERE / 'execution_logs'
    logs.mkdir(exist_ok=True)
    receipt = {'remote_argv': command, 'execute': a.execute,
               'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'freeze_sha256': sha(HERE / 'EXECUTION_FREEZE_V1.json')}
    with (logs / (a.label + '.log')).open('x') as log:
        job = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True)
        for line in job.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = job.wait()
    receipt.update(exit_code=code, ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with (logs / (a.label + '.json')).open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
