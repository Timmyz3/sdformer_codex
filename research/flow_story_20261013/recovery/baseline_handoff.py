#!/usr/bin/env python3
"""One controlled yield/resume of the already-authorized second baseline segment."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/data/flow_a')
BASELINE = ROOT/'runs/native_crop9_fresh_20261012'
SECOND_LAUNCH = BASELINE/'second_long_launch_20261013.json'
GPU_LOCK = ROOT/'runs/.flow_a_gpu0.lock'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(data, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def alive(pid):
    p = Path('/proc')/str(pid)/'status'
    if not p.exists(): return False
    try: return '\nState:\tZ' not in '\n'+p.read_text()
    except FileNotFoundError: return False


def sources_unchanged(argv):
    code = Path(argv[1]).parent
    for name, expected in json.loads((code/'deployment_sha256.json').read_text()).items():
        assert sha(code/name) == expected, 'Frozen baseline source changed: '+name


def inspect_launch():
    launch = json.loads(SECOND_LAUNCH.read_text())
    assert launch['starting_step'] == 135930 and launch['wall_hours_cap'] == 12
    assert launch['optimizer_recipe_changed'] is False and launch['compression_applied_to_training'] is False
    argv = launch['argv']; sources_unchanged(argv)
    assert argv[argv.index('--run-dir')+1] == str(BASELINE)
    assert argv[argv.index('--max-steps')+1] == '480000'
    return launch


def process_identity(launch):
    proc = Path('/proc')/str(launch['pid'])
    assert proc.joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0') == [os.fsencode(x) for x in launch['argv']]
    ticks = int(proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19])
    boot = int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
    born = boot+ticks/os.sysconf('SC_CLK_TCK')
    launched = datetime.datetime.fromisoformat(launch['started_utc']).timestamp()
    assert abs(born-launched) < 5, 'PID birth does not match recorded launch'
    return {'pid': launch['pid'], 'start_ticks': ticks,
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'birth_unix': born}


def checkpoint_state(pt, latest, run):
    import torch
    side = json.loads(pt.with_suffix('.json').read_text())
    assert side == latest and sha(pt) == latest['checkpoint_sha256']
    value = torch.load(pt, map_location='cpu', weights_only=True)
    required = {'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'rng_state', 'epoch_order', 'training'}
    assert required <= value.keys() and value['model_state_dict'] and value['optimizer_state_dict']['state']
    digest = hashlib.sha256(json.dumps(value['contract'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert value['contract'] == run['contract'] and digest == value['contract_sha256'] == run['contract_sha256'] == latest['contract_sha256']
    t = value['training']
    assert t['completed_optimizer_steps'] == latest['completed_optimizer_steps'] == t['epoch']*6000+t['next_index']
    assert value['scheduler_state_dict']['last_epoch'] == t['epoch'] and len(value['epoch_order']) == 6000
    assert {'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= value['rng_state'].keys() and len(value['rng_state']['torch_cuda']) == 1
    return {'internal_full_state_checked': True, 'training': t}


def verify_saved(directory, pause):
    saved = directory/'baseline_saved_state'
    for name, expected in pause['saved_files_sha256'].items():
        assert Path(name).name == name and sha(saved/name) == expected, 'Saved state hash differs'
    segment = json.loads((saved/'segment_result.json').read_text())
    latest = json.loads((saved/'latest.json').read_text())
    assert segment == pause['segment'] and latest == pause['latest'] == segment['checkpoint']
    assert segment['status'] == 'SEGMENT_COMPLETE' and segment['stop_reason'] == 'signal'
    assert segment['segment_id'].endswith('_'+str(pause['pid']))
    spent = float(segment['invocation_wall_seconds']); remaining = 43200-spent
    assert 0 < remaining < 43200 and math.isclose(spent, pause['spent_seconds'], abs_tol=1e-8)
    assert math.isclose(remaining, pause['remaining_seconds'], abs_tol=1e-8)
    assert json.loads((BASELINE/'latest.json').read_text()) == latest, 'Baseline was advanced by another operator'
    assert sha(BASELINE/latest['checkpoint_file']) == latest['checkpoint_sha256']
    checkpoint_state(saved/latest['checkpoint_file'], latest, json.loads((saved/'run_contract.json').read_text()))
    assert json.loads((BASELINE/'run_contract.json').read_text())['contract_sha256'] == latest['contract_sha256']
    return latest, spent, remaining


def yield_baseline(directory, execute):
    launch = inspect_launch(); pid = launch['pid']
    assert alive(pid), 'Expected baseline is no longer alive; inspect its terminal result separately'
    identity = process_identity(launch)
    log = BASELINE/'train_steps.jsonl'
    with log.open('rb') as f:
        f.seek(max(0, log.stat().st_size-8192)); data = f.read().decode()
    records = data.splitlines() if data.endswith('\n') else data.splitlines()[:-1]
    last = json.loads(records[-1])
    assert last['segment_id'].endswith('_'+str(pid)) and last['step'] >= 135930
    segment_birth = datetime.datetime.strptime(last['segment_id'].rsplit('_', 1)[0], '%Y%m%dT%H%M%S.%fZ').replace(tzinfo=datetime.timezone.utc).timestamp()
    assert 0 <= segment_birth-identity['birth_unix'] < 180
    assert time.time()-log.stat().st_mtime < 60, 'Training-loop liveness not established'
    record = {'schema': 'flow_baseline_yield_v1', 'status': 'PREFLIGHT_ONLY', 'pid': pid,
              'launch_sha256': sha(SECOND_LAUNCH), 'process_identity': identity, 'last_pre_signal_train_record': last,
              'signal_target': 'trainer PID only, never its process group',
              'second_segment_total_seconds': 43200, 'script_sha256': sha(__file__)}
    if not execute: return record
    directory.mkdir(parents=True, exist_ok=True)
    output = directory/'baseline_yield.json'
    assert not output.exists(), 'One-shot yield record already exists'
    record.update(status='SIGNAL_REQUESTED_WAITING_FOR_SAFE_CHECKPOINT', signal_utc=now())
    atomic(output, record)
    assert process_identity(launch) == identity
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic()+180
    while alive(pid) and time.monotonic() < deadline: time.sleep(.25)
    assert not alive(pid), 'Graceful baseline stop did not complete; do not force-kill workers'
    return finish_yield(directory, True)


def finish_yield(directory, execute):
    """Finish a recorded signal after inherited worker file descriptors close."""
    launch = inspect_launch(); output = directory/'baseline_yield.json'
    record = json.loads(output.read_text()); pid = record['pid']
    assert record['status'] == 'SIGNAL_REQUESTED_WAITING_FOR_SAFE_CHECKPOINT'
    assert record['launch_sha256'] == sha(SECOND_LAUNCH) and pid == launch['pid'] and not alive(pid)
    last = record['last_pre_signal_train_record']
    if not execute: return {'status': 'FINALIZE_PREFLIGHT_NO_MUTATION', 'signal_record': record}
    with GPU_LOCK.open('a') as lock:
        deadline = time.monotonic()+30
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); break
            except BlockingIOError:
                assert time.monotonic() < deadline, 'Inherited worker GPU lock did not drain'
                time.sleep(.25)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                            '--format=csv,noheader'], text=True).strip()
        segment = json.loads((BASELINE/'segment_result.json').read_text())
        latest = json.loads((BASELINE/'latest.json').read_text())
        assert segment['status'] == 'SEGMENT_COMPLETE' and segment['stop_reason'] == 'signal'
        assert segment['segment_id'].endswith('_'+str(pid))
        assert segment['checkpoint'] == latest and segment['training']['completed_optimizer_steps'] == latest['completed_optimizer_steps']
        assert latest['completed_optimizer_steps'] >= last['step']
        pt = BASELINE/latest['checkpoint_file']; side = json.loads(pt.with_suffix('.json').read_text())
        assert sha(pt) == latest['checkpoint_sha256'] == side['checkpoint_sha256']
        assert latest['contract_sha256'] == side['contract_sha256']
        spent = float(segment['invocation_wall_seconds'])
        assert 0 < spent < 43200, 'No remaining originally-authorized baseline segment budget'
        saved = directory/'baseline_saved_state'; saved.mkdir()
        for source in (pt, pt.with_suffix('.json')): os.link(source, saved/source.name)
        for name in ('run_contract.json', 'latest.json', 'segment_result.json'):
            (saved/name).write_bytes((BASELINE/name).read_bytes())
        state_check = checkpoint_state(saved/pt.name, latest, json.loads((saved/'run_contract.json').read_text()))
        record.update(status='BASELINE_SAFELY_YIELDED', finished_utc=now(), segment=segment,
                      latest=latest, checkpoint_hardlink=str(saved/pt.name),
                      committed_optimizer_steps_lost=0, spent_seconds=spent,
                      remaining_seconds=43200-spent, GPU_lock_released_by_baseline=True,
                      saved_files_sha256={p.name: sha(p) for p in saved.iterdir()},
                      finisher_sha256=sha(__file__), **state_check)
        atomic(output, record)
    return record


def resume_baseline(directory, execute):
    launch = inspect_launch()
    pause = json.loads((directory/'baseline_yield.json').read_text())
    assert pause['status'] == 'BASELINE_SAFELY_YIELDED' and pause['launch_sha256'] == sha(SECOND_LAUNCH)
    assert not alive(pause['pid'])
    latest, spent, remaining = verify_saved(directory, pause)
    argv = list(launch['argv']); argv.remove('--run-gpu')
    argv[argv.index('--wall-hours')+1] = str(remaining/3600)
    argv[argv.index('--resume')+1] = str(BASELINE/latest['checkpoint_file'])
    record = {'schema': 'flow_baseline_remainder_launch_v1', 'status': 'PREFLIGHT_ONLY',
              'remaining_wall_hours': remaining/3600, 'original_second_segment_cap_seconds': 43200,
              'prior_spent_seconds': spent, 'resume_step': latest['completed_optimizer_steps'],
              'optimizer_recipe_changed': False, 'automatic_third_segment': False,
              'yield_record_sha256': sha(directory/'baseline_yield.json'), 'argv': argv,
              'script_sha256': sha(__file__)}
    if not execute: return record
    output = directory/'baseline_remainder_launch.json'
    assert not output.exists(), 'One-shot remainder launch record already exists'
    with GPU_LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert verify_saved(directory, pause) == (latest, spent, remaining)
        env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        with (directory/'baseline_remainder_preflight.log').open('x') as out:
            subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT, env=env, check=True, timeout=180)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                            '--format=csv,noheader'], text=True).strip()
        assert verify_saved(directory, pause) == (latest, spent, remaining)
        argv.append('--run-gpu')
        with (directory/'baseline_remainder.log').open('x') as out:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                env=env, start_new_session=True, pass_fds=(lock.fileno(),))
        record.update(status='BASELINE_REMAINDER_LAUNCHED_CHECK_LIVE_PROGRESS', pid=process.pid,
                      started_utc=now(), argv=argv, GPU_lock_inherited=True)
        atomic(output, record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('yield', 'finish-yield', 'resume'))
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    record = {'yield': yield_baseline, 'finish-yield': finish_yield, 'resume': resume_baseline}[a.action](a.directory, a.run)
    print(json.dumps(record))


if __name__ == '__main__': main()
