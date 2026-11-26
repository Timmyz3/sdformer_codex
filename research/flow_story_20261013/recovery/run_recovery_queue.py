#!/usr/bin/env python3
"""Single-use, bounded four-arm GPU queue; restore original baseline remainder."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import baseline_handoff as handoff

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/flow_a')
BREADTH = ROOT/'code/research_20261013/breadth'
OLD_RUNS = ROOT/'runs/breadth_20261013'
PYTHON = ROOT/'envs/v2/bin/python'
ARMS = ('dense', 'lowrank_param25', 'svs_hidden25', 'magnitude_hidden25')
ENV = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
       'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}


def load(path):
    return json.loads(Path(path).read_text())


def flatten(options):
    return [str(v) for pair in options.items() for v in pair]


def verify_code():
    pins = load(HERE/'deployment_sha256.json')
    for name, expected in pins.items():
        assert handoff.sha(HERE/name) == expected, 'Queue source changed: '+name
    return pins


def common_options():
    argv = load(OLD_RUNS/'initial36_base_argv.json')[2:]
    options = dict(zip(argv[::2], argv[1::2]))
    keys = ('prior-root', 'source-root', 'data-root', 'source-manifest', 'canonical', 'roles',
            'archive-audit', 'acceptance', 'acceptance-sha256', 'checkpoint', 'checkpoint-sha256')
    return {'--'+key: options['--'+key] for key in keys}


def latest(directory):
    side = load(directory/'latest.json')
    pt = directory/side['checkpoint_file']
    assert handoff.sha(pt) == side['checkpoint_sha256'] and load(pt.with_suffix('.json')) == side
    return pt, side


def train_argv(common, arm, directory, svs, count, resume=False):
    options = {**common, '--breadth-root': BREADTH, '--recovery-manifest': HERE/'manifests/gatefit4573_recovery.json',
               '--run-dir': directory, '--arm': arm, '--max-targets': count,
               '--wall-minutes': 45, '--workers': 2, '--cpu-threads': 2, '--device': 0,
               '--checkpoint-targets': 250, '--log-every': 25}
    if arm == 'svs_hidden25':
        options.update({'--svs-artifact': svs, '--svs-artifact-sha256': handoff.sha(svs),
                        '--svs-fit-manifest': HERE/'manifests/svs_fit20.json',
                        '--svs-collector-sha256': handoff.sha(HERE/'collect_svs20.py'), '--svs-fit-scope': 'fit20'})
    if resume:
        pt, side = latest(directory)
        options.update({'--resume': pt, '--resume-sha256': side['checkpoint_sha256']})
    return [str(PYTHON), str(HERE/'train_recovery.py'), *flatten(options)]


def evaluate_argv(common, directory, stage, output):
    pt, side = latest(directory)
    options = {**common, '--breadth-root': BREADTH, '--candidate-checkpoint': pt,
        '--candidate-sha256': side['checkpoint_sha256'], '--model-api-sha256': handoff.sha(HERE/'model_api.py'),
        '--evaluation-core-sha256': handoff.sha(HERE/'evaluation_core.py'),
        '--recovery-manifest': HERE/'manifests/gatefit4573_recovery.json',
        '--check64-manifest': HERE/'manifests/check64.json', '--stage': stage,
        '--cases': HERE/'preflight_dense.json', '--cases-sha256': handoff.sha(HERE/'preflight_dense.json'),
        '--regions': OLD_RUNS/'regions_gatefit4573.json',
        '--regions-sha256': '8d64b18d3c69adb6c2a1bc146f55c254a2f6bbf02e50366b069f1d1d57fa8d26',
        '--geometry': 'full9', '--device': 'cuda:0', '--cpu-threads': 2,
        '--wall-minutes': 15, '--actual-pair-count': 0, '--output': output}
    return [str(PYTHON), str(HERE/'evaluate_recovery.py'), *flatten(options)]


def run_queue(directory):
    pins = verify_code(); common = common_options()
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory/'queue_status.json').exists(), 'Queue is single-use; inspect prior status'
    record = {'schema': 'flow_recovery_queue_v1', 'status': 'STARTING', 'pid': os.getpid(),
              'started_utc': handoff.now(), 'deployment_sha256': pins, 'jobs': [],
              'arms': list(ARMS), 'final_valid_used': False, 'automatic_third_baseline_segment': False}
    def publish(): handoff.atomic(directory/'queue_status.json', record)
    publish()

    def job(name, argv, timeout, own_gpu_lock=True):
        verify_code()
        event = {'name': name, 'argv': argv, 'status': 'RUNNING', 'started_utc': handoff.now()}
        record['jobs'].append(event); record['status'] = 'RUNNING'; record['active_job'] = name; publish()
        lock = handoff.GPU_LOCK.open('a') if own_gpu_lock else None
        child = None
        def stop_child():
            # Collector's KeyboardInterrupt path executes its bounded CPU-worker cleanup.
            child.send_signal(signal.SIGINT if Path(argv[1]).name == 'collect_svs20.py' else signal.SIGTERM)
        try:
            if lock:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            with (directory/(name+'.log')).open('x') as out:
                child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                    env=ENV, start_new_session=True, pass_fds=(lock.fileno(),) if lock else ())
                event['pid'] = child.pid; publish()
                try:
                    code = child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    stop_child()  # Trainer handles TERM after current update; never kill its process group.
                    try: child.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError('Child did not yield safely; inherited GPU lock prevents baseline overlap')
                    raise RuntimeError('Bounded job exceeded wall deadline: '+name)
            assert code == 0, 'Job failed; inspect '+str(directory/(name+'.log'))
            event.update(status='COMPLETE', finished_utc=handoff.now()); publish()
        except BaseException as exc:
            if child is not None and child.poll() is None:
                stop_child()
                try:
                    child.wait(timeout=180)
                    event['exception_child_cleanup'] = 'CONTROLLED_SIGNAL_TO_PID_AND_EXIT_CONFIRMED'
                except subprocess.TimeoutExpired:
                    event['exception_child_cleanup'] = 'STILL_RUNNING_INHERITED_LOCK_PREVENTS_OVERLAP'
            event.update(status='FAILED', error=type(exc).__name__+': '+str(exc)); publish(); raise
        finally:
            if lock: lock.close()

    def train(name, arm, folder, count, svs, resume=False):
        argv = train_argv(common, arm, folder, svs, count, resume)
        job(name, argv+['--run-gpu'], 3000)
        result = load(folder/'segment_result.json')
        assert result['status'] == 'SEGMENT_COMPLETE' and result['training']['visited_targets'] == count
        pt, side = latest(folder)
        assert result['checkpoint'] == side
        return pt, side

    def inspect_smoke(arm, folder):
        import torch
        import model_api as api
        pt, side = latest(folder)
        final, _ = api.read_checkpoint(pt, side['checkpoint_sha256'])
        zero = folder/'checkpoint_visit_000000.pt'
        initial, _ = api.read_checkpoint(zero, load(zero.with_suffix('.json'))['checkpoint_sha256'])
        assert final['training']['visited_targets'] == final['training']['completed_optimizer_steps'] == 8
        changed = [n for n, value in final['model_state_dict'].items() if value.is_floating_point() and
                   not any(x in n for x in ('running_mean', 'running_var')) and not torch.equal(value, initial['model_state_dict'][n])]
        assert changed and any('mlp' in name for name in changed), 'No actual MLP parameter update'
        rows = [json.loads(line) for line in (folder/'train_steps.jsonl').read_text().splitlines()]
        steps = [r for r in rows if r.get('status') == 'UPDATED']
        assert len(steps) == 8 and all(r['gradient_audit']['PSN_grad_nonzero'] > 0 for r in steps)
        return final, {'arm': arm, 'updated_parameters': changed, 'updates': 8,
                       'PSN_nonzero_gradient_all8': True, 'checkpoint_sha256': side['checkpoint_sha256']}

    failure = None
    try:
        pause = load(directory/'baseline_yield.json')
        assert pause['status'] == 'BASELINE_SAFELY_YIELDED'
        handoff.verify_saved(directory, pause)
        svs_dir = directory/'svs_fit20'
        collect = [str(PYTHON), str(HERE/'collect_svs20.py'), 'collect', *flatten(common),
                   '--breadth-root', str(BREADTH), '--bn', 'P_paper', '--geometry', 'full9',
                   '--output-dir', str(svs_dir), '--run-inference', '--device', 'cuda:0',
                   '--gpu-lock', str(handoff.GPU_LOCK)]
        job('collect_svs20', collect, 600, own_gpu_lock=False)
        svs = svs_dir/'svs_replay.json'
        assert svs.exists()
        job('compare_svs20', [str(PYTHON), str(HERE/'compare_svs_2_to_20.py'),
            '--reference2', str(OLD_RUNS/'svs_Ppaper_fit2/svs_replay.json'), '--candidate20', str(svs),
            '--candidate-sha256', handoff.sha(svs), '--output', str(directory/'svs_stability.json')], 120, False)
        smoke = []
        import torch
        import train_recovery as trainer
        for arm in ARMS:
            direct = directory/'smoke'/arm/'direct8'
            train('smoke_'+arm+'_direct8', arm, direct, 8, svs)
            payload, audit = inspect_smoke(arm, direct)
            if arm in ('dense', 'lowrank_param25'):
                split = directory/'smoke'/arm/'split4_8'
                train('smoke_'+arm+'_split4', arm, split, 4, svs)
                train('smoke_'+arm+'_resume8', arm, split, 8, svs, True)
                resumed, _ = inspect_smoke(arm, split)
                keys = ('model_state_dict', 'optimizer_state_dict', 'rng_state', 'training_order', 'architecture', 'contract')
                exact = {key: trainer.same_tree(torch, payload[key], resumed[key]) for key in keys}
                assert all(exact.values()), 'Actual GPU resume differs: '+str(exact)
                audit['direct8_vs_4_resume8_exact'] = exact
            smoke.append(audit)
            handoff.atomic(directory/'smoke_audit.json', {'status': 'PARTIAL' if len(smoke)<4 else 'ALL_FOUR_ARMS_PASS', 'arms': smoke})
        for arm in ARMS:
            folder = directory/'arms'/arm
            train(arm+'_initial', arm, folder, 0, svs)
            out = directory/(arm+'_initial_eval.json')
            job(arm+'_initial_eval', evaluate_argv(common, folder, 'initial', out)+['--run-inference'], 1050)
            assert load(out)['status'] == 'CANDIDATE_EVALUATION_COMPLETE'
            train(arm+'_recovery4573', arm, folder, 4573, svs, True)
            out = directory/(arm+'_endpoint_eval.json')
            job(arm+'_endpoint_eval', evaluate_argv(common, folder, 'endpoint', out)+['--run-inference'], 1050)
            assert load(out)['status'] == 'CANDIDATE_EVALUATION_COMPLETE'
        record['experiment_status'] = 'FOUR_ARMS_COMPLETE'
    except BaseException as exc:
        failure = exc
        record.update(experiment_status='INCOMPLETE', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    finally:
        record['active_job'] = None; record['status'] = 'RESTORING_BASELINE_REMAINDER'
        try: publish()
        except OSError as exc: print('Status write failed; still attempting baseline restore: '+str(exc), file=sys.stderr, flush=True)
        try:
            record['baseline_remainder'] = handoff.resume_baseline(directory, True)
            record['status'] = 'COMPLETE_BASELINE_REMAINDER_LAUNCHED' if failure is None else 'EXPERIMENT_INCOMPLETE_BASELINE_REMAINDER_LAUNCHED'
        except BaseException as exc:
            record.update(status='NEEDS_OPERATOR_BASELINE_RESUME_NOT_CONFIRMED', baseline_resume_error=type(exc).__name__+': '+str(exc),
                          baseline_resume_traceback=traceback.format_exc())
        record['finished_utc'] = handoff.now(); publish()
    if failure: raise failure


def main():
    p = argparse.ArgumentParser(); p.add_argument('--directory', type=Path, required=True); p.add_argument('--run', action='store_true')
    a = p.parse_args(); pins = verify_code()
    if not a.run:
        print(json.dumps({'status': 'QUEUE_PREFLIGHT_NO_GPU', 'pins': pins, 'arms': ARMS,
                          'per_arm_cumulative_wall_minutes': 45, 'common': common_options()})); return
    def request_stop(signum, frame):
        raise RuntimeError('Queue received controlled stop signal '+str(signum))
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    with (a.directory/'.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_queue(a.directory)


if __name__ == '__main__': main()
