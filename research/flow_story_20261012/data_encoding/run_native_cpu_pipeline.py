#!/usr/bin/env python3
"""Wait for verified raw recovery, then bounded 4-CPU native regeneration.

Dedicated flow server only. Never invokes CUDA training or modifies old data.
Explicit --run is required. All paths and source digests are recorded at launch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/flow_a'))
    p.add_argument('--generator-sha256', required=True)
    p.add_argument('--validator-sha256', required=True)
    p.add_argument('--cross-version-evidence', type=Path, required=True)
    p.add_argument('--max-wait-hours', type=float, default=6)
    p.add_argument('--max-generation-hours', type=float, default=12)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    if not 0 < a.max_wait_hours <= 6 or not 0 < a.max_generation_hours <= 12: p.error('Unbounded time limit')
    root = a.root.resolve(); setup=root/'setup'; recovery=setup/'raw_recovery_20261012'
    generator=setup/'generate_native_inputs.py'; validator=setup/'verify_native_dataset.py'
    if sha(generator)!=a.generator_sha256 or sha(validator)!=a.validator_sha256:
        raise ValueError('CPU pipeline source SHA changed')
    cross=json.loads(a.cross_version_evidence.read_text())
    if cross.get('status')!='PASS' or cross.get('comparison')!='exact_npy_sha256' or len(cross.get('targets',[]))!=2:
        raise ValueError('Required new-host vs archive two-target encoding proof absent')
    pilot=json.loads((recovery/'pilot01_a/generation.json').read_text())
    if pilot['status']!='BOUNDED_SUBSET_COMPLETE' or pilot['script_sha256']!=a.generator_sha256:
        raise ValueError('Native pilot/generator changed')
    for r in pilot['records']:
        if sha(root/'data/DSEC_native_v1'/r['relative_path'])!=r['sha256']:
            raise ValueError('Accepted pilot array changed')
    out=setup/'native_cpu_pipeline_20261012'
    contract={'schema':'flow_native_cpu_pipeline_v1','generator_sha256':a.generator_sha256,
        'validator_sha256':a.validator_sha256,'cross_version_evidence_sha256':sha(a.cross_version_evidence),
        'source_root':str(root/'code/SDformerFlow'),'data_root':str(root/'data/DSEC_native_v1'),
        'CPU_workers':4,'torch_threads_each':1,'GPU_training':False,
        'max_wait_hours':a.max_wait_hours,'max_generation_hours':a.max_generation_hours,
        'canonical_sha256':sha(setup/'common_targets.json'),'script_sha256':sha(__file__)}
    if not a.run:print(json.dumps(contract));return
    import fcntl
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=contract:
        raise ValueError('Existing pipeline contract differs')
    write(out/'contract.json',contract)
    started=time.monotonic(); processes=[]; logs=[]
    def interrupted(signum, frame):
        raise InterruptedError('CPU pipeline signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    def status(state,**extra):
        write(out/'status.json',dict(status=state,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,
            GPU_training_started=False,**extra))
    try:
        status('WAITING_FOR_RAW_AND_GT')
        while True:
            current=json.loads((recovery/'recovery_status.json').read_text())
            if current['status']=='raw_and_GT_verified_ready_for_separate_CPU_generation':break
            if current['status']=='failed':raise RuntimeError('Raw recovery failed; no generation')
            if time.monotonic()-started>a.max_wait_hours*3600:raise TimeoutError('Raw recovery wait budget reached')
            launch=json.loads((recovery/'launch_reviewed.json').read_text())
            if not Path('/proc/'+str(launch['pid'])).exists():raise RuntimeError('Raw recovery process gone without completion')
            time.sleep(30)
        if sha(generator)!=a.generator_sha256 or sha(setup/'common_targets.json')!=contract['canonical_sha256']:
            raise ValueError('Generator/canonical changed while waiting')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        reports=[]
        for index in range(4):
            report=out/('shard'+str(index)+'.json'); reports.append(report)
            args=[str(root/'envs/v2/bin/python'),str(generator),'--source-root',str(root/'code/SDformerFlow'),
                '--raw-root',str(root/'data/DSEC_native_v1'),'--output-root',str(root/'data/DSEC_native_v1'),
                '--canonical',str(setup/'common_targets.json'),'--raw-hashes',str(recovery/'source_hashes.csv'),
                '--full-canonical','--shard-count','4','--shard-index',str(index),'--run','--report',str(report)]
            log=(out/('shard'+str(index)+'.log')).open('ab');logs.append(log)
            processes.append(subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env))
        generation_start=time.monotonic();status('GENERATING_CPU_ONLY',worker_pids=[p.pid for p in processes])
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('A native generation shard failed')
            if time.monotonic()-generation_start>a.max_generation_hours*3600:raise TimeoutError('CPU generation wall limit reached')
            time.sleep(15)
        if any(p.returncode for p in processes):raise RuntimeError('A native generation shard failed')
        status('VERIFYING_NEW_NATIVE_DATASET')
        if sha(validator)!=a.validator_sha256 or sha(setup/'common_targets.json')!=contract['canonical_sha256']:
            raise ValueError('Validator/canonical changed before validation')
        args=[str(root/'envs/v2/bin/python'),str(validator),'--root',str(root/'data/DSEC_native_v1'),
            '--canonical',str(setup/'common_targets.json'),'--recovery-dir',str(recovery),
            '--output-dir',str(out/'acceptance')]
        for report in reports:args.extend(['--shard-report',str(report)])
        with (out/'acceptance.log').open('ab') as log:
            validation=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
            processes.append(validation); validation.wait(timeout=7200)
        if validation.returncode:raise RuntimeError('Independent native dataset validation failed')
        accepted_path=out/'acceptance/acceptance.json';accepted=json.loads(accepted_path.read_text())
        if (accepted.get('status')!='VERIFIED_NATIVE_DATASET' or accepted.get('passed') is not True
                or accepted.get('root')!=str(root/'data/DSEC_native_v1') or accepted.get('validator_sha256')!=a.validator_sha256
                or accepted.get('counts',{}).get('generated_inputs')!=8152 or accepted.get('counts',{}).get('files')!=24514
                or accepted.get('training_started') is not False):
            raise RuntimeError('Validator completion marker rejected')
        status('CPU_PREPARATION_COMPLETE_NO_TRAINING',acceptance_dir=str(out/'acceptance'),acceptance_sha256=sha(accepted_path))
    except BaseException as exc:
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,signal.SIG_IGN)
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        status('FAILED_NO_TRAINING',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        for log in logs:log.close()


if __name__=='__main__':main()
