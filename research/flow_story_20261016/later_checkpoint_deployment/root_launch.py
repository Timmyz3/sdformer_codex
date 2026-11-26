#!/usr/bin/env python3
"""Orchestrate one reviewed deployment diagnostic; no training or automatic next run."""
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
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    p.add_argument('--label', required=True)
    a = p.parse_args()
    if not a.label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in a.label):
        raise ValueError('label must be a simple artifact name')
    freeze_path = HERE / 'EXECUTION_FREEZE_V1.json'
    freeze = json.loads(freeze_path.read_text())
    for path, expected in freeze['source_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError('source freeze mismatch: ' + path)
    remote = '/data/flow_a/code/research_20261016/later_checkpoint_deployment'
    r13remote = '/data/flow_a/code/research_20261016/coverage_recovery_round13'
    runs = '/data/flow_a/runs/later_checkpoint_deployment_20261016'
    command = ['env', 'CUDA_VISIBLE_DEVICES=0', 'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2',
               '/data/flow_a/envs/v2/bin/python', '-B', remote + '/run_deployment.py',
               '--runtime-manifest', r13remote + '/runtime_v1.json',
               '--runtime-manifest-sha256', sha(R13 / 'runtime_v1.json'),
               '--membership', r13remote + '/data_protocol/manifest_v1.json',
               '--membership-sha256', sha(R13 / 'data_protocol/manifest_v1.json'),
               '--checkpoint', '/data/flow_a/runs/native_crop9_fresh_20261012/checkpoint_step_000271482.pt',
               '--checkpoint-sha256', '1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977',
               '--cpu-checks', remote + '/cpu_results.json',
               '--output-dir', runs + '/run_v1', '--wall-minutes', '30']
    if a.execute:
        command.append('--execute')
    else:
        command.extend(['--preflight-output', runs + '/' + a.label + '.json'])
    argv = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ControlPath=/tmp/flow_a100_r13b_%C', '-p', '30449',
            'root@36.213.75.115', ' '.join(shlex.quote(x) for x in command)]
    logs = HERE / 'execution_logs'
    logs.mkdir(exist_ok=True)
    receipt = {'remote_argv': command, 'execute': a.execute,
               'freeze_sha256': sha(freeze_path),
               'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with (logs / (a.label + '.log')).open('x') as log:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = proc.wait()
    receipt.update(exit_code=code, ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with (logs / (a.label + '.json')).open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
