#!/usr/bin/env python3
"""Frozen-head, real joint k2/k4 diagnostic. Default preflight, <=293 full forwards."""
import argparse
import copy
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import signal
import time
import traceback
from source_contract import HERE, TARGET, OLD_TARGET, PREFIX_NAMES, load_source, require, sha, source_path, verify_all
from select_members import verify_membership
import frozen_policies as policies

SCHEMA = 'flow_round11_frozen_joint_budget_v1'
STATUS = 'ROUND11_FROZEN_JOINT_BUDGET_COMPLETE'
ANCHORS = ('native', 'FF', 'CC', 'fine_s2b2')
ROWS = {'native': 558000, 'FF': 558000, 'CC': 153600, 'fine_s2b2': 164200, 'k2': 155040, 'k4': 156480}
BUDGET = 293
OWN_SOURCES = ('run_joint.py', 'test_joint_cpu.py', 'source_contract.py', 'select_members.py', 'frozen_policies.py')


def expected_rows(shapes, selected=(), mode='actual', arm=None):
    rows = 0
    for name, (t, b, h, w, c) in shapes.items():
        if arm == 'native' or not any('.layers.{}.'.format(s) in name for s in (0, 2)):
            rows += t*b*h*w
            continue
        for y in range(0, h, 9):
            for x in range(0, w, 9):
                hh, ww = min(9, h-y), min(9, w-x)
                representatives = math.ceil(hh/3)*math.ceil(ww/3)
                fine = arm == 'FF' or (name == TARGET and (arm == 'fine_s2b2' or (y//9, x//9) in selected))
                points = hh*ww+representatives if mode == 'dual_reference' else (hh*ww if fine else representatives)
                rows += t*b*points
    return rows


def check_execution(execution, shapes, selected=(), mode='actual', arm=None):
    require(execution['summed_fc1_rows'] == execution['summed_fc2_rows'] == expected_rows(shapes, selected, mode, arm),
            'Independent all12 FC service differs')
    names = set() if arm == 'native' else {n for n in shapes if any('.layers.{}.'.format(s) in n for s in (0, 2))}
    require(set(execution['layers']) == names and len(names) == (0 if arm == 'native' else 8), 'Wrapper scope differs')
    for name, layer in execution['layers'].items():
        h, w = shapes[name][2:4]
        mask = [[arm == 'FF' or (name == TARGET and (arm == 'fine_s2b2' or (r, c) in selected))
                 for c in range(math.ceil(w/9))] for r in range(math.ceil(h/9))]
        require(layer['fine_mask'] == mask, 'Wrong complete joint mask or true tail: '+name)


def check_prefix(execution, baseline):
    layers, old = execution['layers'], baseline['layers']
    require(all(layers[n]['input_sha256'] == old[n]['input_sha256'] for n in PREFIX_NAMES), 'Same-CC prefix input differs')
    require(all(layers[n]['output_sha256'] == old[n]['output_sha256'] for n in PREFIX_NAMES[:-1]), 'Prior-prefix output differs')
    execution['r11_same_CC_prefix_inputs'] = list(PREFIX_NAMES)
    execution['r11_same_CC_prior_outputs'] = list(PREFIX_NAMES[:-1])


def audited_forward(raw_forward):
    baseline = {}
    def forward(model, functional, targets, shapes, chunk, selected, merge, torch, **kwargs):
        require(len(selected) in (0, 2, 4), 'Only CC/k2/k4 legal')
        require(set(selected) <= set(policies.TILES) and len(set(selected)) == len(selected), 'Illegal selected tile')
        out = raw_forward(model, functional, targets, shapes, chunk, selected, merge, torch, **kwargs)
        check_execution(out[1], shapes, selected, kwargs.get('mode', 'actual'))
        if not selected and kwargs.get('mode', 'actual') == 'actual' and kwargs.get('expected_u_sha') is None:
            baseline.clear(); baseline.update(copy.deepcopy(out[1]))
        require(bool(baseline), 'Current-frame CC required before joint action')
        check_prefix(out[1], baseline)
        return out
    return forward


def load_runtime():
    verify_all()
    collector = load_source('collector', '_r11_frozen_collector')
    probe = collector.load_probe()
    require(probe.TARGET == OLD_TARGET, 'Inherited target changed')
    probe.TARGET = TARGET
    probe.forward_probe = audited_forward(probe.forward_probe)
    parent = probe.load_scope_runner()
    r9 = load_source('r9_runner', '_r11_frozen_anchor_adapter')
    r8 = r9.load_r8()
    require(sha(r8.__file__) == sha(source_path('r8_runner')), 'R8 anchor helper differs')
    return collector, probe, parent, r9, r8


def anchor_forward(r9, r8, parent, model, functional, targets, shapes, chunk, arm, merge, torch, **kwargs):
    require(arm in ('native', 'FF', 'fine_s2b2'), 'CC goes through shared prefix path')
    out = r9.forward_arm(r8, parent, model, functional, targets, shapes, chunk, arm, merge, torch, **kwargs)
    check_execution(out[1], shapes, mode=kwargs.get('mode', 'actual'), arm=arm)
    return out


def check_pair(actual, dual, torch):
    flows, ex, branches = actual[:3]
    dflows, dex, dbranches = dual[:3]
    require(len(flows) == len(dflows) and len(flows) > 0 and all(torch.equal(x, y) for x, y in zip(flows, dflows)),
            'Actual/dual complete flow list differs')
    require(set(branches) == set(dbranches) == set(ex['layers']) == set(dex['layers']) and len(branches) == 8,
            'Actual/dual branch coverage differs')
    require(all(torch.equal(branches[n], dbranches[n]) and ex['layers'][n]['input_sha256'] == dex['layers'][n]['input_sha256']
                for n in branches), 'Actual/dual branch or input differs')
    return {'flow_list_length': len(flows), 'all_flows_exact': True, 'all8_inputs_exact': True, 'all8_branches_exact': True,
            'actual_execution': ex, 'dual_execution': dex}


def write_decisions_before_evaluation(path, payload, write):
    require(not path.exists(), 'Refuse decision overwrite')
    require(payload['GT_used'] is False and payload['fitting_performed'] is False, 'No-GT inference required')
    write(path, payload)
    require(path.is_file(), 'Decision archive must exist before evaluation')
    return sha(path)


def event_only(native, data_root, row, hashes):
    import numpy as np
    path = native.checked_path(data_root, native.target_files(row)['input'])
    payload = path.read_bytes(); digest = hashlib.sha256(payload).hexdigest()
    require(digest == hashes[native.target_files(row)['input']]['sha256'], 'Accepted event input differs')
    events = np.load(io.BytesIO(payload), allow_pickle=False)
    require(events.shape == (10, 480, 640) and str(events.dtype) == 'float32' and np.isfinite(events).all(),
            'Native event shape/type differs')
    return events, digest


def prepare(a, collector, probe, parent):
    require(0 < a.wall_minutes <= 30, 'At most30 minutes')
    member = verify_membership()
    require(sha(a.selection_manifest) == sha(HERE/'selection_manifest_v1.json'), 'Fixed compatibility selection differs')
    inherited = argparse.Namespace(**vars(a)); inherited.group = 'check'; inherited.runner_checks = a.collector_runner_checks
    prepared = collector.prepare(inherited, probe, parent)
    require([row['target_id'] for _, row in prepared[6]] == member['groups']['check'], 'Freshcheck8 order differs')
    own = parent.checked_cpu_receipt(a.runner_checks, [HERE/name for name in OWN_SOURCES])
    reference = json.loads(source_path('r10_result').read_text())
    require(reference['status'] == 'ROUND10_BLOCK2_SINGLETON_COLLECTION_COMPLETE' and reference['full_forward_count'] == 366,
            'Complete R10 numerical reference required')
    require(reference['original_model_unchanged'] and reference['BN_buffers_unchanged'] and reference['original_modules_restored'],
            'R10 state closure differs')
    models = policies.load_models()
    canonical = prepared[2][2].canonical_rows(prepared[1].canonical)
    ref = next(f for f in reference['frames'] if f['group'] == 'check')
    anchor_row = next(row for row in canonical if row['target_id'] == ref['target_id'])
    require(anchor_row['target_id'] not in member['groups']['check'], 'Archived replay overlaps new cohort')
    binding = {'runner_sha256': sha(__file__), 'runner_checks_sha256': own,
        'source_pins_sha256': sha(HERE/'source_pins.json'), 'membership_sha256': sha(HERE/'membership_v1.json'),
        'inference_models_sha256': sha(HERE/'inference_models_v1.json'), 'origin_decisions_sha256': sha(source_path('r10_decisions')),
        'reference_r10_sha256': sha(source_path('r10_result')), 'checkpoint_sha256': collector.CHECKPOINT_SHA,
        'target': TARGET, 'target_ids': member['groups']['check'], 'full_forward_maximum': BUDGET,
        'main_logical_arm_count': 288, 'audit_forward_count': 5, 'budgets': list(policies.BUDGETS),
        'policy_names': list(policies.POLICIES), 'anchors': list(ANCHORS), 'uniform_order': list(policies.UNIFORM_ORDER),
        'candidate_tiles': [list(t) for t in policies.TILES], 'inherited_data_and_numerical_contract': prepared[7],
        'GPU_lock': str(parent.GPU_LOCK), 'wall_minutes': a.wall_minutes,
        'scope': 'Fixed R10 heads, new identity-only check8 in old backbone-seen development sequences; real joint actions at both budgets, exact same-frame mask aliases only. No fitting, recovery, official test or hardware gain.'}
    return prepared, reference, ref, anchor_row, models, binding


def execute(a, prepared, result, collector, probe, parent, r9, r8):
    import numpy as np
    import torch
    inherited, reference, archived, anchor_row, models, binding = prepared
    c, b, context, base, scope, loader, chosen, oldbinding, oldhead = inherited
    require(not torch.cuda.is_initialized(), 'CUDA before shared lock')
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0'); torch.cuda.set_device(device); torch.cuda.manual_seed_all(20261014)
    started = time.monotonic()
    def deadline():
        require(time.monotonic()-started < a.wall_minutes*60, 'R11 fixed wall budget exhausted')
    native, trainer, metrics, dev32, hashes, config, saved, side, regions, plan = context
    model, functional = loader.load_model(b, context, base, torch, device)
    bn_info = scope.apply_scope(model, 'MLP_CMA176', b.bn_payload, torch)
    bn_info['execution'] = ('Native has no tiled MLP wrappers; every other arm has8stage02 wrappers. '
        'FF allfine; CC allcoarse; fullblock2 includes true fine tails; joint k2/k4 onlycomplete target tiles fine, tailscoarse.')
    require(bn_info['scope'] == 'MLP_CMA176' and bn_info['BN_count'] == 78 and bn_info['MLP_BN_count'] == 24
            and len(bn_info['frozen_BN_names']) == 24 and len(bn_info['current_stat_BN_names']) == 54,
            'Frozen24/current54 BN numerical contract differs')
    model.eval().requires_grad_(False)
    targets, shapes = parent.target_layers(model); merge = parent.load_frozen_merge()
    before, bn_before = parent.source_digest(model), base.bn_signature(model, torch)
    require(before == reference['original_state_sha256'], 'R10 model state differs')
    result.update(status='RUNNING', original_state_sha256=before, BN_scope=bn_info, frames=[],
        full_forward_count=0, main_actual_forward_count=0, alias_logical_arm_count=0, forward_records=[],
        anchor_replays={}, dual_checks={}, runtime={'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'device_name': torch.cuda.get_device_name(device), 'NumPy': np.__version__, 'selector_device': 'CPU NumPy folded FP32', 'TF32': False, 'AMP': False, 'threads': 2})
    arm_names = list(ANCHORS)+[policies.arm_name(p, k) for p in policies.POLICIES for k in policies.BUDGETS]
    meters = {arm: metrics.MetricAccumulator() for arm in arm_names}
    times = {x['target_id']: x['timestamps'] for x in oldbinding['new_member_data_preflight']}

    def forward(tid, chunk, arm, purpose='main', selected=None, **kwargs):
        if selected is not None:
            out = probe.forward_probe(model, functional, targets, shapes, chunk, selected, merge, torch, deadline=deadline, **kwargs)
        else:
            out = anchor_forward(r9, r8, parent, model, functional, targets, shapes, chunk, arm, merge, torch, deadline=deadline, **kwargs)
        result['full_forward_count'] += 1
        require(result['full_forward_count'] <= BUDGET, 'R11 full-forward cap exceeded')
        if purpose == 'main': result['main_actual_forward_count'] += 1
        expected = 609600 if kwargs.get('mode') == 'dual_reference' else (153600+720*len(selected) if selected is not None else ROWS[arm])
        require(out[1]['summed_fc1_rows'] == out[1]['summed_fc2_rows'] == expected, 'Production FC service differs')
        fid = result['full_forward_count']
        result['forward_records'].append({'forward_id': fid, 'target_id': tid, 'arm': arm, 'purpose': purpose,
            'mode': kwargs.get('mode', 'actual'), 'selected_tiles': None if selected is None else [list(t) for t in selected],
            'execution': out[1]})
        return out, fid

    with torch.inference_mode():
        events, digest = event_only(native, b.data_root, anchor_row, hashes)
        require(digest == archived['input_evidence']['events'], 'Archived event input differs')
        chunk = base.prepare_input(events, device, torch)
        replay, fid = forward(anchor_row['target_id'], chunk, 'CC', purpose='archive_replay', selected=())
        pred_sha = hashlib.sha256(replay[0][-1][0].numpy().tobytes()).hexdigest()
        old = archived['arms']['coarse_stage02']
        require(pred_sha == old['prediction_sha256'], 'Archived CC final prediction differs')
        require(set(replay[1]['layers']) == set(old['execution']['layers']) and all(
            replay[1]['layers'][n][k] == old['execution']['layers'][n][k]
            for n in replay[1]['layers'] for k in ('input_sha256', 'output_sha256')), 'Archived CC all8hashes differ')
        result['anchor_replays']['CC'] = {'target_id': anchor_row['target_id'], 'forward_id': fid,
            'prediction_sha256': pred_sha, 'exact_archived_prediction': True, 'all8_input_output_SHA_exact': True,
            'execution': replay[1]}
        del replay, chunk, events
        for index, (_, row) in enumerate(chosen):
            deadline(); tid = row['target_id']
            events, event_sha = event_only(native, b.data_root, row, hashes)
            chunk = base.prepare_input(events, device, torch)
            cc, cc_id = forward(tid, chunk, 'CC', selected=(), capture_u=True, capture_bn=index == 0)
            u = cc[3]['u']; u_sha = cc[1]['layers'][TARGET]['input_sha256']
            require(merge.tensor_sha(u) == u_sha, 'Captured CC input differs')
            ordinary, cost = probe.feature_scores(u, targets[TARGET], merge, torch)
            cells = collector.cell_descriptors(u).cpu().numpy()
            payload = policies.infer(models, cells[0], ordinary)
            payload.update(target_id=tid, sequence=row['sequence'], events_sha256=event_sha, predecision_u_sha256=u_sha,
                ordinary_scores=ordinary, ordinary_score_cost=cost, decision_source='Current-frame CC prefix only',
                joint_plan=policies.joint_plan(payload['decisions']))
            input_path = a.output_dir/('input_'+tid+'.npz')
            np.savez_compressed(input_path, predecision_u=u.cpu().numpy(), no_GT_cell_descriptor=cells,
                                candidate_tiles=np.asarray(policies.TILES))
            payload.update(predecision_input_file=input_path.name, predecision_input_file_sha256=sha(input_path))
            decision_path = a.output_dir/('decisions_'+tid+'.json')
            decision_sha = write_decisions_before_evaluation(decision_path, payload, parent.write)
            # First GT/validity arrays are loaded below, after all32 masks are committed.
            evidence, expected = {}, {}
            for key, relative in native.target_files(row).items():
                expected['events' if key == 'input' else key] = hashes[relative]['sha256']
            arrays, _ = metrics.load_arrays(b.data_root/'saved_flow_data', row, evidence=evidence, expected_hashes=expected)
            require(evidence['events'] == event_sha and np.array_equal(arrays['events'], events), 'Evaluation event differs')
            masks = metrics.partitions(arrays['gt'], arrays['mask'], events, regions)
            frame = {'target_id': tid, 'sequence': row['sequence'], 'group': 'frozen_head_newcheck',
                'input_evidence': evidence, 'timestamps': times[tid], 'arms': {}, 'decisions': payload['decisions'],
                'decision_file': decision_path.name, 'decision_file_sha256': decision_sha,
                'decision_archived_before_GT_load': True, 'predecision_u_sha256': u_sha,
                'predecision_input_file': input_path.name, 'predecision_input_file_sha256': sha(input_path),
                'ordinary_scores': ordinary, 'ordinary_score_cost': cost}
            predictions = {}
            def record(arm, out, forward_id, alias_of=None, selected=None):
                prediction = out[0][-1][0].numpy().copy()
                predictions[arm] = prediction
                frame['arms'][arm] = {'metrics': meters[arm].update(tid, row['sequence'], prediction, arrays['gt'], masks),
                    'prediction_sha256': hashlib.sha256(prediction.tobytes()).hexdigest(), 'execution': out[1],
                    'actual_forward_id': forward_id, 'alias_of': alias_of,
                    'selected_tiles': None if selected is None else [list(t) for t in selected]}
            record('CC', cc, cc_id, selected=())
            if index == 0:
                require(len(cc[1]['batch_norm_inputs']) == 78 and all(x['input_calls'] for x in cc[1]['batch_norm_inputs'].values()),
                        'All78 BN input calls required')
                captured_bn = cc[1]['batch_norm_inputs']
                require(sum(v['track_running_stats'] and not v['training'] and v['running_mean_exists'] for v in captured_bn.values()) == 24
                        and sum(not v['track_running_stats'] and not v['training'] and not v['running_mean_exists'] for v in captured_bn.values()) == 54,
                        'Executed frozen24/current54 BN split differs')
            cc_execution = cc[1]
            del cc, u, cells
            for arm in ('native', 'FF', 'fine_s2b2'):
                actual, aid = forward(tid, chunk, arm, capture=index == 0 and arm != 'native')
                if arm == 'fine_s2b2': check_prefix(actual[1], cc_execution)
                record(arm, actual, aid)
                if index == 0 and arm != 'native':
                    dual, did = forward(tid, chunk, arm, purpose='dual_audit', mode='dual_reference', capture=True)
                    pair = check_pair(actual, dual, torch)
                    pair.update(actual_forward_id=aid, dual_forward_id=did, target_id=tid)
                    result['dual_checks'][arm] = pair
                    del dual
                del actual
            primary_masks = {tuple(tuple(t) for t in payload['decisions']['flow11_task']['selected'][str(k)]):
                             policies.arm_name('flow11_task', k) for k in policies.BUDGETS}
            for action in payload['joint_plan']:
                arm, alias_of = action['arm'], action['alias_of']
                selected = tuple(tuple(t) for t in action['selected_tiles'])
                if alias_of is not None:
                    require(alias_of in frame['arms'], 'Alias must refer to earlier same-frame actual arm')
                    prior = frame['arms'][alias_of]
                    require(prior['alias_of'] is None and prior['selected_tiles'] == action['selected_tiles'], 'Alias mask/source differs')
                    frame['arms'][arm] = copy.deepcopy(prior)
                    frame['arms'][arm]['alias_of'] = alias_of
                    predictions[arm] = predictions[alias_of].copy()
                    frame['arms'][arm]['metrics'] = meters[arm].update(tid, row['sequence'], predictions[arm], arrays['gt'], masks)
                    result['alias_logical_arm_count'] += 1
                    continue
                need_pair = index == 0 and selected in primary_masks
                actual, aid = forward(tid, chunk, arm, selected=selected, expected_u_sha=u_sha, capture=need_pair)
                record(arm, actual, aid, selected=selected)
                if need_pair:
                    dual, did = forward(tid, chunk, arm, purpose='dual_audit', selected=selected,
                                        mode='dual_reference', capture=True, expected_u_sha=u_sha)
                    pair = check_pair(actual, dual, torch)
                    pair.update(actual_forward_id=aid, dual_forward_id=did, executed_actual_arm=arm, target_id=tid)
                    result['dual_checks'][primary_masks[selected]] = pair
                    del dual
                del actual
            require(set(frame['arms']) == set(arm_names), 'Every36 logical arm must be reported')
            prediction_path = a.output_dir/('predictions_'+tid+'.npz')
            np.savez_compressed(prediction_path, **predictions)
            frame.update(prediction_file=prediction_path.name, prediction_file_sha256=sha(prediction_path))
            result['frames'].append(frame)
            result['summaries'] = {arm: meter.summary() for arm, meter in meters.items()}
            parent.write(a.output_dir/'results.json', result)
            print(json.dumps({'target_id': tid, 'completed_targets': len(result['frames']),
                'full_forwards': result['full_forward_count'], 'alias_arms': result['alias_logical_arm_count']}), flush=True)
            del chunk, events, arrays, predictions
    require(parent.source_digest(model) == before and base.bn_signature(model, torch) == bn_before, 'Model/BN state changed')
    require(all(model.get_submodule(n) is m for n, m in targets.items()), 'Original modules not restored')
    require(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()), 'Audit hooks remain')
    require(len(result['frames']) == 8 and result['main_actual_forward_count']+result['alias_logical_arm_count'] == 288,
            'Exactly288 logical endpoints required')
    require(result['full_forward_count'] == result['main_actual_forward_count']+5 <= BUDGET,
            'Exactlyfive audits plus unique main forwards required')
    require(set(result['dual_checks']) == {'FF', 'fine_s2b2', 'flow11_task_k2', 'flow11_task_k4'}, 'Allfour fixed duals required')
    result.update(status=STATUS, completed_targets=8, original_model_unchanged=True, BN_buffers_unchanged=True,
        original_modules_restored=True, all_hooks_removed=True, wall_seconds=time.monotonic()-started,
        hardware_benefit_claim=False, scope=binding['scope'])


def main(argv=None):
    collector, probe, parent, r9, r8 = load_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-contract', 'identity-bank', 'prior-screen-results', 'prior-scope-results',
                 'selection-manifest', 'fit-membership', 'prior-check-membership'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    for name, default in (('module-checks', probe.BASE/'cpu_results.json'),
                          ('scope-runner-checks', probe.BASE/'scope_extension_v1/runner_cpu_results.json'),
                          ('probe-runner-checks', probe.HERE/'runner_cpu_results.json'),
                          ('collector-runner-checks', source_path('collector_receipt')),
                          ('runner-checks', HERE/'runner_cpu_results.json')):
        parser.add_argument('--'+name, type=Path, default=default)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preflight-output', type=Path)
    parser.add_argument('--wall-minutes', type=float, default=30)
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args(argv); prepared = prepare(a, collector, probe, parent)
    result = {'schema': SCHEMA, 'status': 'PREFLIGHT_PASS_NO_INFERENCE', 'binding': prepared[-1],
              'inference_executed': False, 'training_performed': False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(), 'Fresh preflight destination required')
        parent.write(a.preflight_output, result); print(result['status']); return
    require(parent.GPU_LOCK.parent.is_dir(), 'Existing shared GPU lock directory required')
    with parent.GPU_LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output_dir.mkdir(parents=True, exist_ok=False)
        previous = signal.getsignal(signal.SIGALRM)
        def timeout(signum, frame): raise TimeoutError('R11 fixed wall budget exhausted')
        signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, a.wall_minutes*60)
        try:
            result['inference_executed'] = True
            execute(a, prepared, result, collector, probe, parent, r9, r8)
        except BaseException as exc:
            result.update(status='INCOMPLETE_NO_SCIENTIFIC_CONCLUSION', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
            parent.write(a.output_dir/'results.json', result)


if __name__ == '__main__':
    main()
