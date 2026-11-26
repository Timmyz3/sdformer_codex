#!/usr/bin/env python3
"""Explicitly resume the unchanged native baseline for one further <=12h.

Run after breadth experiments; never called by the evaluation queue itself.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/data/flow_a')
RUN = ROOT/'runs/native_crop9_fresh_20261012'


def main():
    with (RUN.parent/'.flow_a_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        launch_path = RUN/'second_long_launch_20261013.json'
        assert not launch_path.exists() and not (RUN/'second_long_20261013.log').exists()
        previous = json.loads((RUN/'segment_result.json').read_text())
        latest = json.loads((RUN/'latest.json').read_text())
        assert previous['status'] == 'SEGMENT_COMPLETE' and previous['stop_reason'] == 'wall_budget'
        assert previous['training']['completed_optimizer_steps'] == 135930
        assert latest['completed_optimizer_steps'] == 135930
        assert latest['checkpoint_sha256'] == 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
        prior = json.loads((RUN/'first_long_launch.json').read_text())
        argv = list(prior['argv']);argv.remove('--run-gpu')
        code = Path(argv[1]).parent
        for name, expected in json.loads((code/'deployment_sha256.json').read_text()).items():
            assert hashlib.sha256((code/name).read_bytes()).hexdigest() == expected, name
        assert argv[argv.index('--resume')+1] == str(RUN/'latest.json')
        assert argv[argv.index('--wall-hours')+1] == '12'
        assert argv[argv.index('--max-steps')+1] == '480000'
        env = {**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
               'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        with (RUN/'second_long_preflight_20261013.log').open('x') as out:
            subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT,env=env,check=True,timeout=180)
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
                                        '--format=csv,noheader'],text=True).strip()
        assert not apps, apps
        argv.append('--run-gpu')
        with (RUN/'second_long_20261013.log').open('x') as out:
            process = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,
                                       start_new_session=True,pass_fds=(lock.fileno(),),env=env)
        record = {'status':'LAUNCHED_CHECK_LIVE_PROGRESS','pid':process.pid,'argv':argv,
                  'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'starting_step':135930,'whole_run_target_steps':480000,'wall_hours_cap':12,
                  'optimizer_recipe_changed':False,'compression_applied_to_training':False,
                  'automatic_next_segment':False,'GPU_lock_inherited':True,
                  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        launch_path.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record))


if __name__ == '__main__':main()
