#!/usr/bin/env python3
"""Explicitly launch one <=12h segment after reviewed real native pilot."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/data/flow_a')
CODE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/native_crop9_fresh_20261012'
OPS = ROOT / 'runs/native_pilot_operations_20261012'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-marker-sha256', required=True)
    a = parser.parse_args()
    launch_path = RUN / 'first_long_launch.json'
    marker_path = RUN / 'pilot_step_000000500.json'
    with (ROOT / 'runs/.flow_a_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if launch_path.exists() or (RUN / 'first_long.log').exists():
            raise RuntimeError('First long segment already has records; inspect explicitly')
        status = json.loads((OPS / 'status.json').read_text())
        comparison = json.loads((OPS / 'resume_comparison.json').read_text())
        if status['status'] != 'PILOT_SEQUENCE_COMPLETE_NO_LONG_TRAINING' or not comparison['passed']:
            raise RuntimeError('Actual pilot and restart comparison not complete')
        if hashlib.sha256(marker_path.read_bytes()).hexdigest() != a.pilot_marker_sha256:
            raise RuntimeError('Pilot marker does not match explicitly reviewed SHA')
        manifest = json.loads((CODE / 'deployment_sha256.json').read_text())
        for name, expected in manifest.items():
            if hashlib.sha256((CODE / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Previously deployed source differs: ' + name)
        from run_native_pilot_sequence import command
        argv = command(RUN, 480000, gpu=False, resume=True)
        argv[argv.index('--phase') + 1] = 'continue'
        argv[argv.index('--wall-hours') + 1] = '12'
        argv += ['--pilot-marker', str(marker_path), '--pilot-marker-sha256', a.pilot_marker_sha256]
        env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        with (RUN / 'first_long_preflight.log').open('x') as output:
            subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=120, env=env)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                        '--format=csv,noheader'], text=True).strip()
        if apps:
            raise RuntimeError('GPU compute process exists before launch: ' + apps)
        argv.append('--run-gpu')
        with (RUN / 'first_long.log').open('x') as output:
            # This inherited open file description keeps flock held after this
            # short launcher exits; the trainer releases it only on process exit.
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),), env=env)
        record = {'status': 'LAUNCHED_CHECK_LIVE_LOG_FOR_PROGRESS', 'pid': process.pid,
                  'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'argv': argv,
                  'pilot_marker_sha256': a.pilot_marker_sha256,
                  'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'starting_step': 500, 'wall_hours_cap': 12, 'whole_run_target_steps': 480000,
                  'automatic_next_segment': False, 'GPU_lock_inherited_fd': lock.fileno(),
                  'completion_evidence': 'new segments/*.json and segment_result.json; launch is not completion'}
        temp = launch_path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2)+'\n'); temp.replace(launch_path)
        print(json.dumps(record))


if __name__ == '__main__':
    main()
