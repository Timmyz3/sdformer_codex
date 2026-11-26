#!/usr/bin/env python3
"""Run only three explicitly prepared, bounded follow-ups after calibrated42."""
import contextlib
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess

RUN = Path('/data/flow_a/runs/breadth_20261013')


def main():
    prior = json.loads((RUN/'calibrated42_results.json').read_text())
    assert prior['status'] == 'BREADTH_CASES_FINISHED' and prior['complete_cases'] == 42
    destination = RUN/'followup_measurements_status.json'
    assert not destination.exists()
    svs = json.loads((RUN/'svs4_base_argv.json').read_text())
    svs[svs.index('--output')+1] = str(RUN/'svs4_results.json');svs.append('--run-inference')
    jobs = [('svs4',svs,900,False),
            ('actual_skip_local_probe',json.loads((RUN/'actual_probe_argv.json').read_text()),900,True),
            ('actual7',json.loads((RUN/'actual7_argv.json').read_text()),1200,False)]
    env = {**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
           'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    record = {'status':'RUNNING','pid':os.getpid(),'results':[],
              'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'automatic_next_queue':False,'optimizer_steps':0}
    destination.write_text(json.dumps(record,indent=2)+'\n')
    try:
        for name,argv,timeout,self_lock in jobs:
            manager = contextlib.nullcontext() if self_lock else (RUN.parent/'.flow_a_gpu0.lock').open('a')
            with manager as lock:
                if lock is not None:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                                    '--format=csv,noheader'],text=True).strip()
                with (RUN/(name+'.log')).open('x') as output:
                    proc = subprocess.run(argv,stdout=output,stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,env=env,timeout=timeout,check=True,
                        pass_fds=() if lock is None else (lock.fileno(),))
                record['results'].append({'name':name,'argv':argv,'returncode':proc.returncode})
                destination.write_text(json.dumps(record,indent=2)+'\n')
        record['status'] = 'COMPLETED_CHECK_ARTIFACTS'
    except BaseException as exc:
        record.update(status='FAILED_NO_NEXT_STAGE',error=type(exc).__name__+': '+str(exc));raise
    finally:
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        destination.write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':main()
