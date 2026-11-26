#!/usr/bin/env python3
"""One bounded remote queue: crop BN diagnostic then full-frame CMA176.

Run only after explicitly inspected initial36 results; takes the dedicated GPU
lock and refuses existing output. Never resumes training or queues itself again.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/data/flow_a')
CODE = ROOT/'code/research_20261013/breadth'
RUN = ROOT/'runs/breadth_20261013'


def replace(argv, key, value):
    out = list(argv);out[out.index('--'+key)+1] = str(value);return out


def main():
    with (RUN.parent/'.flow_a_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        e = json.loads((RUN/'initial36_results.json').read_text())
        assert e['status'] == 'BREADTH_CASES_FINISHED' and e['complete_cases'] == 35
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        status_path = RUN/'bn_diagnostics_status.json'
        assert not status_path.exists()
        argv = json.loads((RUN/'initial36_launch.json').read_text())['argv']
        geometry = replace(argv,'cases',CODE/'cases_geometry2.json')
        geometry = replace(geometry,'cases-sha256',hashlib.sha256((CODE/'cases_geometry2.json').read_bytes()).hexdigest())
        geometry = replace(geometry,'geometry','crop_center9')
        geometry = replace(geometry,'wall-minutes',10)
        geometry = replace(geometry,'output',RUN/'geometry_crop2.json')
        calibration = list(argv);calibration[1] = str(CODE/'collect_bn_cma.py')
        calibration = replace(calibration,'wall-minutes',15)
        calibration = replace(calibration,'output',RUN/'bn_cma176_full9.json')
        calibration += ['--schedule',str(CODE/'bn_calibration_schedule.json')]
        env = {**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
               'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        record = {'status':'RUNNING','pid':os.getpid(),
                  'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'automatic_next_queue':False,'optimizer_steps':0,'results':[],
                  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'code_sha256':{n:hashlib.sha256((CODE/n).read_bytes()).hexdigest() for n in
                                ('evaluate_breadth.py','collect_bn_cma.py','cases_geometry2.json','bn_calibration_schedule.json')}}
        status_path.write_text(json.dumps(record,indent=2)+'\n')
        try:
            for name, command, timeout in [('geometry_crop2',geometry,900),('bn_cma176_full9',calibration,1200)]:
                with (RUN/(name+'.log')).open('x') as out:
                    result = subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,env=env,
                                            stdin=subprocess.DEVNULL,timeout=timeout,check=True,
                                            pass_fds=(lock.fileno(),))
                record['results'].append({'name':name,'returncode':result.returncode,'argv':command})
                status_path.write_text(json.dumps(record,indent=2)+'\n')
            record['status'] = 'COMPLETED_CHECK_ARTIFACTS'
        except BaseException as exc:
            record.update(status='FAILED_NO_NEXT_STAGE',error=type(exc).__name__+': '+str(exc));raise
        finally:
            record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            status_path.write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':main()
