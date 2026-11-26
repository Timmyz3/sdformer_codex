#!/usr/bin/env python3
"""Collect112 RNE4/Q8 predictions +6 dense engineering forwards; root schedules GPU."""
import argparse
import copy
import fcntl
import signal
import sys
import time
import traceback
from pathlib import Path
sys.dont_write_bytecode = True
import probe_contract as contract_tools

HERE = Path(__file__).resolve().parent
require, sha, read, write = (contract_tools.require, contract_tools.sha,
                            contract_tools.read, contract_tools.write)


def prepare(a):
    import torch
    require(not torch.cuda.is_initialized() and not a.output.exists(), 'Fresh CPU-only preflight')
    require(sha(__file__) == a.runner_sha256 and sha(a.source_freeze) == a.source_freeze_sha256,
            'Explicit frozen collector and manifest bytes')
    require(Path(contract_tools.__file__).resolve() == HERE/'probe_contract.py', 'Actual local helper')
    frozen = read(a.source_freeze)
    require(frozen['schema'] == 'flow_e3_rne_probe_source_freeze_v1', 'E3-specific frozen source manifest')
    for name, digest in frozen['files'].items():
        require(Path(name).name == name and sha(HERE/name) == digest, 'Frozen local source '+name)
    require(set(('collect_probe.py', 'probe_contract.py', 'test_cpu.py')).issubset(frozen['files']),
            'Required E3 source files bound')
    core = contract_tools.load_core(a.e0_core)
    require(sha(a.contract) == a.contract_sha256, 'Exact final endpoint-bound contract')
    c = contract_tools.validate(read(a.contract), core)
    require(sha(a.selections) == contract_tools.SELECTION_SHA
            and read(a.selections)['e2_reserved_ids'] == c['target_ids'], 'Frozen16 role order')
    endpoint, prior = contract_tools.endpoint_binding(a.endpoint_result, a.weights, c['target_ids'])
    require(endpoint == c['endpoint'], 'No checkpoint, state or endpoint-result substitution')
    base = a.r28_root/'implementation'
    for name, digest in (('run_smoke_v2.py', contract_tools.SMOKE_SHA),
                         ('run_recovery_v2.py', contract_tools.HELPER_SHA),
                         ('recovery_integer_core.py', contract_tools.R28_CORE_SHA)):
        require(sha(base/name) == digest, 'Pinned original R28 helper '+name)
    smoke = contract_tools.load(base/'run_smoke_v2.py', '_e3_r28_smoke')
    helper = contract_tools.load(base/'run_recovery_v2.py', '_e3_r28_helper')
    old = copy.copy(a); old.runner_sha256 = contract_tools.SMOKE_SHA
    support, members, _, _ = smoke.prepare(old)
    helper.validate_order(members)
    rows = {r['target_id']: r for r in members['dev_members']}
    require(all(t in rows for t in c['target_ids']), 'All selected identities in original development64')
    return core, c, smoke, helper, support, members, [rows[t] for t in c['target_ids']], prior


def validate_work(work, calls, bns, c, core, action):
    require(set(work) == set(core.TARGETS) and set(calls) == set(core.TARGETS)
            and all(v == 1 for v in calls.values()), '28 actual one-time target calls')
    require(len(bns) == 78 and all(v == 1 for v in bns.values()), '78 actual one-time BN calls')
    for name, w in work.items():
        g = c['geometries'][name]
        rows = g['T'] * g['H'] * g['W']
        complete = g['T'] * (g['H']//9*9) * (g['W']//9*9)
        restored = 151*81*g['T'] if action >= 0 and name in core.CONVS else 0
        require(w['rows'] == rows and w['Cin'] == g['K'] and w['Cout'] == g['Cout']
                and w['tail_rows'] == rows-complete
                and w['restored_complete_rows'] == restored
                and w['high8_rows'] == rows-complete+restored
                and w['low4_rows'] == complete-restored
                and w['geometric_shape'] == [10, 1, g['H'], g['W']],
                'Actual geometry/domain cancellation '+name)
        require(w['representation']['variant'] == 'rne4'
                and w['scale_variants'] == 2 and w['canonical_scatter_before_BN']
                and not w['reference'] and w['independent_domain_execution'],
                'Frozen independent RNE4/current-BN semantics '+name)
        require(w['logical_row_plane_work'] == 8*w['high8_rows']+4*w['low4_rows'],
                'Actual row-plane work '+name)
        for bit in range(8):
            expected = w['high8_rows'] + (w['low4_rows'] if bit >= 4 else 0)
            require(sum(k['rows'] for k in w['kernel_calls'] if k['bit'] == bit) == expected,
                    'All actual bit-plane domains counted')
    return dict(affine_count=28, BN_count=78,
                added_core_ticks_vs_Q4=0 if action < 0 else 82192320,
                logical_row_plane_work=sum(w['logical_row_plane_work'] for w in work.values()),
                net_cycle_claim=False)


def execute(a, core, c, smoke, helper, support, members, rows, prior, result):
    import numpy as np
    import torch
    require(not torch.cuda.is_initialized(), 'Global lock precedes CUDA')
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(20261019); torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model, functional, bn = smoke.make_model(support, 'full9', device, torch)
    cp = torch.load(a.weights, map_location='cpu', weights_only=False)
    state = c['endpoint']['state_sha256']
    require(cp['schema'] == 'flow_e0_rne4_joint_checkpoint_v1'
            and cp['completed_optimizer_steps'] == 512 and cp['final_state_sha256'] == state,
            'Actual formal512 checkpoint metadata')
    model.load_state_dict(cp['model_state_dict'], strict=True); del cp
    model.eval().requires_grad_(False)
    require(support['modules'][0].state_digest(model) == state, 'Actual endpoint model state')
    bn_signature = support['base'].bn_signature(model, torch)
    result.update(status='RUNNING', BN_scope=bn, model_forwards=0, formal_model_forwards=0,
                  engineering_model_forwards=0, evaluations={}, parity={},
                  runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                               device=torch.cuda.get_device_name(0), AMP=False, TF32=False))
    write(a.output/'results.json', result)
    for action in range(-1, 6):
        label = 'background_Q4' if action == -1 else f'upgrade_{action}'
        bound = contract_tools.BoundCore(core, c, action)
        folder = a.output/'predictions'/label; folder.mkdir(parents=True, exist_ok=False)
        frames = []; tic = time.monotonic()
        for index, row in enumerate(rows):
            tid = row['target_id']; evidence = helper.source_evidence(members, tid)
            events = helper.event_only(support, row, evidence, np)
            chunk = support['base'].prepare_input(events, device, torch); del events
            input_record = helper.tensor_record(chunk)
            if action >= 0 and index == 0:
                actual, ar = smoke.capture_full(model, functional, chunk, bound.mode, bound, torch, ste=False)
                result['model_forwards'] += 1; result['formal_model_forwards'] += 1
                dense, dr = smoke.capture_full(model, functional, chunk, bound.mode, bound, torch,
                                               ste=False, reference=True)
                result['model_forwards'] += 1; result['engineering_model_forwards'] += 1
                cmp = smoke.compare_full(actual, dense, torch)
                result['parity'][label] = dict(target_id=tid, comparison=cmp,
                                              actual_record=ar, dense_record=dr)
                require(cmp['pass'], 'Whole graph integer/PSN/four-head dense parity '+label)
                heads = {f'head{i}': helper.tensor_record(actual[f'head{i}'][0]) for i in range(4)}
                prediction = actual['head3'][0].numpy().copy()
                work, bns, calls = ar['work'], ar['actual_BN_calls'], ar['actual_target_calls']
                del actual, dense
            else:
                functional.reset_net(model)
                with bound.installed(model, bound.mode, ste=False) as wraps:
                    with helper.observed(model, bound, torch) as (bns, measured):
                        with torch.no_grad():
                            flows = model(chunk)['flow']
                    work = {n: copy.deepcopy(w.last_audit) for n, w in wraps.items()}
                result['model_forwards'] += 1; result['formal_model_forwards'] += 1
                require(len(flows) == 4, 'Four actual heads')
                heads = {f'head{i}': helper.tensor_record(v[0]) for i, v in enumerate(flows)}
                prediction = flows[3][0].detach().cpu().numpy().copy()
                calls = {n: len(v) for n, v in measured.items()}
                del flows
            require(prediction.shape == (2, 480, 640) and prediction.dtype == np.float32
                    and np.isfinite(prediction).all(), 'Full finite head3')
            checked = validate_work(work, calls, bns, c, core, action)
            if action == -1:
                require(heads == prior[tid]['heads']
                        and evidence == prior[tid]['input_evidence'], 'Endpoint Q4 four-head/input replay')
            file = folder/f'{tid}.npz'; np.savez_compressed(file, head3=prediction)
            frames.append(dict(target_id=tid, sequence=row['sequence'], action=action,
                input_evidence=evidence, actual_input=input_record,
                prediction_file=str(file.relative_to(a.output)), prediction_file_sha256=sha(file),
                prediction_sha256=heads['head3']['sha256'], heads=heads,
                actual_BN_calls=bns, actual_affine_call_counts=calls, work=work, work_checks=checked))
            del chunk, prediction
            write(a.output/'progress.json', dict(action=label, identities_completed=index+1,
                formal_model_forwards=result['formal_model_forwards'],
                engineering_model_forwards=result['engineering_model_forwards']))
            if (index+1) % 4 == 0:
                print(label, index+1, flush=True)
        require(support['modules'][0].state_digest(model) == state
                and support['base'].bn_signature(model, torch) == bn_signature, 'Unchanged W and current BN')
        result['evaluations'][label] = dict(action=action, mask_layout=bound.mask_record,
            frames=frames, state_sha256=state, model_forwards=16, GT_arrays_read=False,
            wall_seconds=time.monotonic()-tic)
        write(a.output/'results.json', result)
    require((result['model_forwards'], result['formal_model_forwards'],
             result['engineering_model_forwards']) == (118, 112, 6), 'Exact112+6 completed forward budget')
    result.update(status='E3_RNE112_AND6_DENSE_REFERENCES_COMPLETE', final_state_sha256=state,
        background_replay_bitwise_equal_all16=True, completed_optimizer_steps=0,
        wall_seconds=time.monotonic()-start, max_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('e0-core', 'r28-root', 'r26-root', 'members', 'weights', 'endpoint-result',
                 'selections', 'contract', 'source-freeze', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('runner-sha256', 'contract-sha256', 'source-freeze-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--wall-minutes', type=int, default=60)
    a = parser.parse_args()
    require(a.wall_minutes == 60, 'Fixed60min cap; no automatic continuation')
    core, c, smoke, helper, support, members, rows, prior = prepare(a)
    result = dict(schema='flow_e3_rne_probe_v1', status='CPU_PREFLIGHT_PASS',
        sources=dict(runner=sha(__file__), contract_tool=sha(HERE/'probe_contract.py'),
            core=contract_tools.CORE_SHA, source_freeze=a.source_freeze_sha256,
            action_contract=a.contract_sha256, selections=contract_tools.SELECTION_SHA,
            members=contract_tools.MEMBERS_SHA, smoke=contract_tools.SMOKE_SHA,
            helper=contract_tools.HELPER_SHA, **c['endpoint']),
        action_contract=c, selected_ids=c['target_ids'], selection_scope=c['selection_scope'],
        source_state_sha256=c['endpoint']['state_sha256'], expected_model_forwards=118,
        expected_formal_forwards=112, expected_engineering_forwards=6,
        training_performed=False, development_GT_arrays_read=False, quality_computed=False,
        quality_selection=False, predictor_implemented=False, resume_allowed=False,
        wall_cap_minutes=60, loader_source_receipt=support['binding'])
    if not a.execute:
        require(a.preflight and not a.preflight.exists(), 'Fresh CPU preflight receipt')
        write(a.preflight, result)
        return
    lock = support['modules'][3].GPU_LOCK.open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output.mkdir(parents=True, exist_ok=False)
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('E3 fixed60min cap')))
        signal.alarm(3600)
        try:
            execute(a, core, c, smoke, helper, support, members, rows, prior, result)
        except BaseException as error:
            result.update(status='E3_RNE_PARTIAL_FAILED', error=str(error), traceback=traceback.format_exc())
            write(a.output/'results.json', result)
            raise
        finally:
            signal.alarm(0)
        write(a.output/'results.json', result)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
