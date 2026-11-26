#!/usr/bin/env python3
"""Dedicated A100 only, three fixed ordinary recovery arms in sequence."""
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path('/data/flow_a/runs/round27_20261019')
SCRIPT='/data/flow_a/code/research_20261019/round27/implementation/run_recovery.py'
RUNNER_SHA='495f7dbfdc9d54f74ed2f40ef5cfb9c6fcc58da5a03998efca733845d0b6753a'
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0')


def main():
    commands=[]
    for arm in ('native','Q8','Q6'):
        output=ROOT/('formal_'+arm+'_v1')
        assert not output.exists()
        argv=['timeout','--signal=TERM','--kill-after=30s','3700','/data/flow_a/envs/v2/bin/python','-B',SCRIPT,
              '--r26-root','/data/flow_a/code/research_20261017/window_precision_round26',
              '--members','/data/flow_a/code/research_20261019/round27/independent_review/membership_v1.json',
              '--runner-sha256',RUNNER_SHA,'--arm',arm,'--output',str(output),'--execute','--wall-minutes','60']
        commands.append(dict(arm=arm,output=str(output),argv=argv))
    receipt=dict(schema='flow_r27_three_arm_launch_v1',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 arms=commands,common_updates=512,stop_on_failure=True,quality_selection=False,status='RUNNING')
    path=ROOT/'formal_launch_v1.json'
    with path.open('x') as f:json.dump(receipt,f,indent=2)
    for cmd in commands:
        probe=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        assert not probe.stdout.strip(),'Other GPU process found; do not compete or kill it'
        started=time.time()
        with (ROOT/('formal_'+cmd['arm']+'_v1.log')).open('x') as log:
            run=subprocess.run(cmd['argv'],env=ENV,stdout=log,stderr=subprocess.STDOUT)
        cmd.update(returncode=run.returncode,wall_seconds=time.time()-started)
        print(json.dumps({k:cmd[k] for k in ('arm','returncode','wall_seconds','output')}),flush=True)
        if run.returncode:
            receipt['status']='STOPPED_ON_FAILURE';path.write_text(json.dumps(receipt,indent=2)+'\n');raise SystemExit(run.returncode)
        path.write_text(json.dumps(receipt,indent=2)+'\n')
    receipt['status']='THREE_FIXED_ARMS_COMPLETE';receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
