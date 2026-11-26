#!/usr/bin/env python3
"""Explicit one-shot native pilot: CPU check,32+4,control36,compare,continue500.

No validation, no compression training, no automatic long segment.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/data/flow_a')
CODE = Path(__file__).resolve().parent
PYTHON = str(ROOT / 'envs/v2/bin/python')
RUN = ROOT / 'runs/native_crop9_fresh_20261012'
CONTROL = ROOT / 'runs/native_crop9_direct36_20261012'
OPS = ROOT / 'runs/native_pilot_operations_20261012'
ACCEPTANCE_SHA = 'b1d87de8fd48b710fe5a49dd37d6d64427d2fe111df249efa73ba2299f454540'


def atomic(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def command(directory, steps, gpu=False, resume=False):
    cmd = [PYTHON, str(CODE / 'train_native_baseline.py'),
           '--source-root', str(ROOT / 'code/SDformerFlow'),
           '--data-root', str(ROOT / 'data/DSEC_native_v1'), '--run-dir', str(directory),
           '--manifest', str(ROOT / 'setup/common_targets.json'),
           '--source-manifest', str(ROOT / 'setup/native_source_sha256.json'),
           '--archive-audit', str(ROOT / 'setup/archive_data_audit.json'),
           '--data-acceptance', str(ROOT / 'setup/native_cpu_pipeline_20261012/acceptance/acceptance.json'),
           '--acceptance-sha256', ACCEPTANCE_SHA,
           '--phase', 'pilot', '--max-steps', str(steps), '--wall-hours', '0.25',
           '--checkpoint-steps', '250', '--workers', '2', '--cpu-threads', '2', '--seed', '0']
    if gpu: cmd.append('--run-gpu')
    if resume: cmd += ['--resume', str(directory / 'latest.json')]
    return cmd


def main():
    OPS.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'runs/.flow_a_gpu0.lock').open('a') as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(directory.exists() and any(directory.iterdir()) for directory in (OPS, RUN, CONTROL)):
            raise RuntimeError('One-shot pilot directories have prior records; inspect and resume explicitly')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                        '--format=csv,noheader'], text=True).strip()
        if apps:
            raise RuntimeError('GPU compute process exists: ' + apps)
        start = time.monotonic()
        state = {'status': 'STARTED', 'pid': os.getpid(), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'GPU': 'dedicated A10040 device0', 'results': [], 'automatic_long_training': False,
                 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        stages = [('preflight', command(RUN, 32)),
                  ('fresh32', command(RUN, 32, True)),
                  ('resume36', command(RUN, 36, True, True)),
                  ('direct36', command(CONTROL, 36, True)),
                  ('compare36', [PYTHON, str(CODE / 'compare_native_resume.py'), '--resumed', str(RUN),
                                  '--direct', str(CONTROL), '--output', str(OPS / 'resume_comparison.json')]),
                  ('pilot500', command(RUN, 500, True, True))]
        atomic(OPS / 'launch_contract.json', {'stages': stages, **state})
        active = None

        def interrupted(signum, frame):
            raise InterruptedError('Pilot operator interrupted by signal ' + str(signum))

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)

        def reap_group():
            if active is None:
                return
            # A process group also owns DataLoader children. Do not release the
            # GPU lock while an interrupted stage can still execute GPU work.
            for sig, timeout in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
                try:
                    os.killpg(active.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    active.wait(timeout=timeout)
                    # Direct child may exit before its remaining children.
                    if sig == signal.SIGTERM:
                        try: os.killpg(active.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                    return
                except subprocess.TimeoutExpired:
                    continue
            raise RuntimeError('Stage process group did not terminate')

        try:
            for name, argv in stages:
                state.update(status='RUNNING', stage=name, elapsed_seconds=time.monotonic()-start)
                atomic(OPS / 'status.json', state)
                with (OPS / (name + '.log')).open('w') as log:
                    t = time.monotonic()
                    active = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env={
                        **os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
                    returncode = active.wait(timeout=1200)
                    if returncode:
                        reap_group()
                    active = None
                    state['results'].append({'stage': name, 'returncode': returncode,
                                             'wall_seconds': time.monotonic()-t})
                    if returncode: raise RuntimeError(name + ' failed; see stage log')
                if name in ('fresh32', 'resume36', 'direct36', 'pilot500'):
                    directory = CONTROL if name == 'direct36' else RUN
                    result = json.loads((directory / 'segment_result.json').read_text())
                    expected = {'fresh32': 32, 'resume36': 36, 'direct36': 36, 'pilot500': 500}[name]
                    if result['status'] != 'SEGMENT_COMPLETE' or result['training']['completed_optimizer_steps'] != expected:
                        raise RuntimeError(name + ' did not reach declared cumulative step')
                    atomic(OPS / (name + '_result.json'), result)
            state.update(status='PILOT_SEQUENCE_COMPLETE_NO_LONG_TRAINING', elapsed_seconds=time.monotonic()-start)
            atomic(OPS / 'status.json', state)
        except BaseException as exc:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            reap_group()
            state.update(status='FAILED', exception=type(exc).__name__, message=str(exc), elapsed_seconds=time.monotonic()-start)
            atomic(OPS / 'status.json', state)
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
