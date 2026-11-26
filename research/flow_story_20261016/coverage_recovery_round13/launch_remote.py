#!/usr/bin/env python3
"""Root orchestration of one explicitly requested R13 stage, with local full logs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['smoke','recover'],required=True)
    p.add_argument('--conditions',nargs='+',choices=['reference','diversity'],default=['reference','diversity'])
    p.add_argument('--execute',action='store_true')
    p.add_argument('--label',required=True)
    a=p.parse_args()
    freeze=json.loads((HERE/'EXECUTION_FREEZE_V1.json').read_text())
    for relative,expected in freeze['source_sha256'].items():
        assert hashlib.sha256((HERE/relative).read_bytes()).hexdigest()==expected,relative
    remote='/data/flow_a/code/research_20261016/coverage_recovery_round13'
    inputs_sha=hashlib.sha256((HERE/'inputs_v1.json').read_bytes()).hexdigest()
    command=['env','CUDA_VISIBLE_DEVICES=0','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2',
        '/data/flow_a/envs/v2/bin/python','-B',remote+'/implementation/launch_stage.py',
        '--inputs',remote+'/inputs_v1.json','--inputs-sha256',inputs_sha,
        '--output-root','/data/flow_a/runs/coverage_recovery_round13_20261016/experiments_v1',
        '--stage',a.stage,'--conditions']+a.conditions
    if a.execute:command.append('--execute')
    argv=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10',
          '-o','ControlPath=/tmp/flow_a100_r13b_%C','-p','30449','root@36.213.75.115',
          ' '.join(shlex.quote(x) for x in command)]
    logs=HERE/'execution_logs';logs.mkdir(exist_ok=True)
    record={'remote_argv':command,'execute':a.execute,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    with (logs/(a.label+'.log')).open('x') as log:
        job=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
        for line in job.stdout:
            log.write(line);log.flush()
            try:
                item=json.loads(line)
                if 'argv' in item:item={k:v for k,v in item.items() if k!='argv'}
                print(json.dumps(item),flush=True)
            except ValueError:print(line,end='',flush=True)
        code=job.wait()
    record.update(exit_code=code,ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (logs/(a.label+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    raise SystemExit(code)

if __name__=='__main__':main()
