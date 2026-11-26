#!/usr/bin/env python3
"""Orchestrate the reviewed fixed8 selected16 same-weight BN diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent


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
    remote = '/data/flow_a/code/research_20261016/selected16_bn_round15'
    runs = '/data/flow_a/runs/selected16_bn_round15_20261016'
    command = ['env', 'CUDA_VISIBLE_DEVICES=0', 'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2',
               '/data/flow_a/envs/v2/bin/python', '-B', remote + '/run_selected16.py',
               '--inputs-manifest', remote + '/runtime_inputs_v1.json',
               '--inputs-manifest-sha256', sha(HERE / 'runtime_inputs_v1.json'),
               '--cpu-checks', remote + '/cpu_results.json',
               '--output-dir', runs + '/run_v1', '--wall-minutes', '15']
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
