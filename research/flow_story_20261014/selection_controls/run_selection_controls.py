#!/usr/bin/env python3
"""Fixed CPU controls for the existing learned-fast motion selection pilot.

The full-network run body is an explicit fork of pinned run_fast_runtime.py;
all model, scoring, grouping, budget, fast and metric implementations are reused.
No optimization, GT-based selection, new geometry, or GPU work is introduced.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True

PARENT_CONTRACT_SHA = '564d70a02b65d00c1aec40eacf069babb38f600c5a553a5ceb2a263630f3c389'
PARENT_RUNNER_SHA = 'aef1e570f2814e33438cfd04c644d3c6bdaa7d377d8fb828ba6246ea22e550ed'
PRIOR_RESULTS_SHA = '08ec5c7032dbfe99d33c159e09369c147514cfe3ae8e3889c248b48c52244ccf'
TARGETS = ('zurich_city_01_a_0083', 'zurich_city_05_a_0165',
           'zurich_city_01_a_0030', 'zurich_city_05_a_0352')
ARM_SPECS = (
    ('native_dense', None, None), ('tiled_all_keep', 'zero', None),
    ('independent_trained', 'trained', 'independent'),
    ('motion_trained', 'trained', 'motion'), ('motion_mean', 'mean', 'motion'),
    ('horizontal_local_trained', 'trained', 'horizontal_local'),
    ('rotated90_trained', 'trained', 'rotated90'))
ANCHORS = ('native_dense', 'tiled_all_keep', 'independent_trained', 'motion_trained')
CONTRASTS = (
    ('motion_trained', 'motion_mean'), ('motion_mean', 'motion_zero'),
    ('motion_trained', 'independent_trained'),
    ('motion_trained', 'horizontal_local_trained'),
    ('motion_trained', 'rotated90_trained'),
    ('horizontal_local_trained', 'independent_trained'),
    ('rotated90_trained', 'independent_trained'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def bind_prior(c, prior):
    require(c['schema'] == 'flow_fast_runtime_contract_v1' and c['target_count'] == 4, 'Unknown parent scope')
    require(prior.get('status') == 'FAST_WARMSTART_RUNTIME_COMPLETE' and
            prior.get('complete_frames') == 4 and tuple(prior.get('targets', [])) == TARGETS,
            'Prior run incomplete or targets differ')
    require(prior.get('contract') == c and prior.get('contract_sha256') == PARENT_CONTRACT_SHA,
            'Prior contract differs')
    require(tuple(f['target_id'] for f in prior['frames']) == TARGETS, 'Prior frame order differs')
    require(prior.get('BN_buffers_unchanged') is True and prior['runtime']['GPU_initialized'] is False,
            'Prior state/device evidence differs')


def verify_frame_service(frame):
    arms = frame['arms']
    local = [(name, branch, policy, arms[name]) for name, branch, policy in ARM_SPECS if policy]
    first = local[0][3]['runtime_selector']
    for name, branch, policy, rec in local:
        e, selector = rec['execution'][0], rec['runtime_selector']
        require(e['branch'] == branch and selector['policy'] == policy, 'Wrong branch/policy identity')
        require(e['slow_points'] == 900 and e['fast_region_points'] == 300, 'Real slow/fast point budget changed')
        require(e['observed']['slow_fc1_rows'] == e['observed']['slow_fc2_rows'] == 9000,
                'Actual slow FC rows changed')
        expected_fast = 3000 if branch == 'trained' else 0
        require(e['observed']['fast_fc_rows'] == e['fast_LN_rows'] == expected_fast, 'Fast FC/LN rows changed')
        require(selector['selection']['slow_service_cost'] == selector['common_budget']['target_cost'] == 900,
                'Exact common cost changed')
        for key in ('input_sha256', 'descriptor_sha256', 'flow_dxdy_pixels', 'scores', 'points_per_tile'):
            require(selector[key] == first[key], 'Selector information changed: '+key)
        require(selector['GT_or_final_flow_used'] is False and
                selector['all_four_proposals_computed_for_cost_matching'] is True,
                'Selector source/planning differs')
    require(arms['motion_mean']['execution'][0]['keep'] == arms['motion_trained']['execution'][0]['keep'],
            'Mean replacement changed motion mask')
    require(arms['motion_mean']['execution'][0]['observed']['mean_fill_elements'] == 3000*384,
            'Mean fill work differs')
    return True


def verify_prior_frame(frame, prior):
    require(frame['target_id'] == prior['target_id'] and frame['sequence'] == prior['sequence'], 'Prior target differs')
    for name in ANCHORS:
        new, old = frame['arms'][name], prior['arms'][name]
        require(new['final_flow_sha256'] == old['final_flow_sha256'], 'Prior prediction differs: '+name)
        require(new['metrics'] == old['metrics'], 'Prior metric/GT partition differs: '+name)
        if new['execution']:
            require(len(new['execution']) == len(old['execution']) == 1, 'Prior call count differs')
            for key in ('input_sha256', 'output_sha256', 'keep', 'slow_points', 'observed'):
                require(new['execution'][0][key] == old['execution'][0][key], 'Prior execution differs: '+name+'/'+key)
    require(frame['arms']['motion_mean']['execution'][0]['keep'] ==
            prior['arms']['motion_zero']['execution'][0]['keep'], 'Mean vs prior zero mask differs')
    return {'all_four_anchor_predictions_and_metrics_exact': True,
            'motion_mean_mask_equals_prior_zero': True,
            'prior_results_sha256': PRIOR_RESULTS_SHA}


def paired_contrasts(arms):
    out = {}
    for left, right in CONTRASTS:
        if right not in arms:
            continue
        ls, rs = arms[left]['summary'], arms[right]['summary']
        require(ls['frame_count'] == rs['frame_count'] == 4, 'Contrast missing frames')
        regions = {}
        for region, lm in ls['regions'].items():
            rm = rs['regions'][region]
            for key in ('valid_pixels', 'frames_with_region', 'empty_region_frames'):
                require(lm[key] == rm[key], 'Contrast population differs')
            regions[region] = {'frames_with_region': lm['frames_with_region'], 'valid_pixels': lm['valid_pixels']}
            for key in ('frame_mean_AEE', 'pixel_weighted_AEE'):
                require((lm[key] is None) == (rm[key] is None), 'Contrast null mismatch')
                regions[region]['delta_'+key] = None if lm[key] is None else lm[key]-rm[key]
        out[left+'_minus_'+right] = regions
    return out

def run(c, a, context, base, scope, loader, result, prior):
    import numpy as np
    import torch
    import coarse_head
    import intervention_core as core
    import runtime_group_adapter as policy_module
    import same_shape_tile_mlp as tile
    import fast_tile_adapter as adapter
    import fit_fast_cache
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    started = time.monotonic()
    def deadline():
        require(time.monotonic()-started < 1800, 'CPU wall budget exhausted')
    head_weights = torch.load(c['head_weights'], map_location='cpu', weights_only=True)
    head = coarse_head.RuntimeCoarseFlowHead(384, flow_contract=c['flow_contract'])
    require(head_weights['provenance']['cache_sha256'] == c['head_fit_cache_sha256'] and
            head_weights['flow_contract'] == c['flow_contract'], 'Coarse head provenance differs')
    head.load_state_dict(head_weights['stages']['2']['methods']['C16_GELU2']['state_dict'], strict=True)
    head.eval().requires_grad_(False)
    weights = torch.load(c['fast_weights'], map_location='cpu', weights_only=True)
    require(weights['schema'] == 'flow_fast_warmstart_weights_v1' and
            weights['provenance']['cache_sha256'] == c['fast_cache_sha256'] and
            weights['provenance']['checkpoint_sha256'] == a.checkpoint_sha256 and
            weights['provenance']['BN_scope'] == 'MLP_CMA176' and
            weights['recipe']['steps'] == 500 and weights['recipe']['seed'] == 20261014,
            'Fast warmstart provenance/recipe differs')
    require(weights['provenance']['fit_script_sha256'] == sha(fit_fast_cache.__file__),
            'Fast training source differs from the frozen fitter')
    fast = fit_fast_cache.load_fast_class(Path(c['adapter_dir']))(384)
    fast.load_state_dict(weights['stages']['2']['state_dict'], strict=True)
    fast.eval().requires_grad_(False)
    mean = weights['stages']['2']['mean_branch']
    model, functional = loader.load_model(a, context, base, torch, torch.device('cpu'))
    bn = scope.apply_scope(model, 'MLP_CMA176', a.bn_payload, torch)
    bn['execution'] = 'Only stage2 first MLP actual fast/slow; frozen fit20 teacher-feature fast warmstart'
    before = base.bn_signature(model, torch); original = model.get_submodule(adapter.LAYER)
    native, trainer, metrics, rows, hashes, config, saved_run, side, regions, plan = context
    targets = core.choose_targets(rows, 4)
    require([r['target_id'] for r in targets] == list(TARGETS), 'Fixed targets changed')
    prior_frames = {f['target_id']: f for f in prior['frames']}
    arm_specs = ARM_SPECS
    meters = {p: metrics.MetricAccumulator() for p, _, _ in arm_specs}
    result.update(status='RUNNING', inference_executed=True, BN_scope=bn, frames=[], input_evidence={},
        runtime={'device': 'cpu', 'threads': 2, 'GPU_initialized': False, 'torch': str(torch.__version__), 'TF32': False},
        targets=[r['target_id'] for r in targets])

    def predict(chunk, branch=None, policy=None, mode='actual'):
        deadline(); functional.reset_net(model); begin = time.monotonic()
        cm = adapter.replace_one(model, fast, mean, mode, branch, policy, tile) if branch else contextlib.nullcontext(None)
        with cm as wrapper:
            out = model(chunk)['flow'][-1]
            require(tuple(out.shape) == (1, 2, 480, 640) and bool(torch.isfinite(out).all()), 'Invalid full flow')
            pred = out[0].detach().cpu().numpy().copy()
            records = list(wrapper.records) if wrapper else []
        return pred, {'execution': records, 'wall_seconds_CPU': time.monotonic()-begin,
                      'final_flow_sha256': hashlib.sha256(pred.tobytes()).hexdigest()}

    with torch.inference_mode():
        for index, row in enumerate(targets):
            deadline(); evidence = {}; expected = {}
            for kind, rel in native.target_files(row).items():
                native.checked_path(a.data_root, rel)
                expected['events' if kind == 'input' else kind] = hashes[rel]['sha256']
            arrays, _ = metrics.load_arrays(a.data_root/'saved_flow_data', row, evidence=evidence, expected_hashes=expected)
            chunk = base.prepare_input(arrays['events'], torch.device('cpu'), torch)
            frame = {'target_id': row['target_id'], 'sequence': row['sequence'], 'arms': {}}
            result['frames'].append(frame); result['input_evidence'][row['target_id']] = evidence
            predictions = {}
            for name, branch, policy_name in arm_specs:
                if name == 'native_dense':
                    pred, rec = predict(chunk)
                else:
                    policy = policy_module.RuntimeGroupPolicy(head, policy_name) if policy_name else (
                        lambda name, u: torch.ones((4, 5), dtype=torch.bool))
                    pred, rec = predict(chunk, branch, policy)
                    require(len(rec['execution']) == 1, 'Target invocation differs')
                    if policy_name:
                        require(len(policy.records) == 1, 'Runtime selector invoked incorrectly')
                        rec['runtime_selector'] = policy.records[0]
                    if index == 0 and policy_name:
                        replay = tile.ReplayPolicy(rec['execution'])
                        ref, reference = predict(chunk, branch, replay, 'reference'); replay.complete()
                        actual_trace, ref_trace = rec['execution'][0], reference['execution'][0]
                        exact = bool(np.array_equal(pred, ref))
                        rec['same_shape_reference'] = {'final_flow_exact': exact,
                            'final_flow_max_abs': float(np.abs(pred-ref).max()), 'reference': reference}
                        require(exact and all(actual_trace[k] == ref_trace[k] for k in
                            ('input_sha256', 'output_sha256', 'keep', 'input_stride', 'output_stride')),
                            'Actual vs same-shape dual-production reference differs')
                frame['arms'][name] = rec; predictions[name] = pred
                write(a.output, result)
            frame['native_to_tiled_conversion'] = {
                'final_flow_exact': bool(np.array_equal(predictions['native_dense'], predictions['tiled_all_keep'])),
                'final_flow_max_abs': float(np.abs(predictions['native_dense']-predictions['tiled_all_keep']).max()),
                'quality_baseline': 'tiled_all_keep; native conversion reported separately'}
            selected = [frame['arms'][name] for name, _, policy_name in arm_specs if policy_name]
            require(all(r['execution'][0]['slow_points'] == 900 for r in selected), 'Slow work differs from900')
            require(len({r['execution'][0]['input_sha256'] for r in selected}) == 1, 'Different pre-MLP inputs')
            flow = selected[0]['runtime_selector']['flow_dxdy_pixels']
            require(all(r['runtime_selector']['flow_dxdy_pixels'] == flow for r in selected), 'Different runtime flow inputs')
            verify_frame_service(frame)
            masks = metrics.partitions(arrays['gt'], arrays['mask'], arrays['events'], regions)
            extra, definition = direction_regions(arrays['gt'], arrays['mask'], flow, c['flow_contract'])
            masks.update(extra); frame['direction_error_strata'] = definition
            for name in predictions:
                frame['arms'][name]['metrics'] = meters[name].update(row['target_id'], row['sequence'], predictions[name], arrays['gt'], masks)
            frame['prior_anchor_check'] = verify_prior_frame(frame, prior_frames[row['target_id']])
            output = a.output.parent/('predictions_'+row['target_id']+'.npz')
            require(not output.exists(), 'Prediction artifact already exists')
            np.savez_compressed(output, **predictions)
            frame['predictions_artifact'] = {'file': output.name, 'sha256': sha(output)}
            write(a.output, result)
            print(json.dumps({'target': row['target_id'], 'complete_frames': index+1, 'all_valid_AEE':
                {name: rec['metrics']['regions']['all_valid']['AEE'] for name, rec in frame['arms'].items()}}), flush=True)
            del chunk, predictions
    require(base.bn_signature(model, torch) == before and model.get_submodule(adapter.LAYER) is original,
            'BN or native module changed')
    require(not torch.cuda.is_initialized(), 'CUDA initialized')
    result.update(status='SELECTION_CONTROLS_COMPLETE', complete_frames=4, BN_buffers_unchanged=True,
                  wall_seconds=time.monotonic()-started,
                  arms={name: {'summary': meter.summary(), 'frames': meter.frames} for name, meter in meters.items()})
    result['paired_contrasts'] = paired_contrasts(result['arms'])
    write(a.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-contract', type=Path, required=True)
    parser.add_argument('--parent-contract-sha256', required=True)
    parser.add_argument('--prior-results', type=Path, required=True)
    parser.add_argument('--prior-results-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--preflight-output', type=Path)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU environment required')
    require(args.parent_contract_sha256 == PARENT_CONTRACT_SHA == sha(args.parent_contract), 'Parent contract binding differs')
    require(args.prior_results_sha256 == PRIOR_RESULTS_SHA == sha(args.prior_results), 'Prior results binding differs')
    require(not args.output_dir.exists(), 'New output directory required')
    c = json.loads(args.parent_contract.read_text()); prior = json.loads(args.prior_results.read_text())
    bind_prior(c, prior)
    for path, digest in c['source_pins'].items():
        require(sha(path) == digest, 'Source differs: '+path)
    for key in ('head_weights', 'fast_weights'):
        require(sha(c[key]) == c[key+'_sha256'], 'Weights differ: '+key)
    matches = [Path(p) for p, h in c['source_pins'].items() if Path(p).name == 'run_fast_runtime.py' and h == PARENT_RUNNER_SHA]
    require(len(matches) == 1, 'Pinned parent runner missing/ambiguous')
    fast_root = matches[0].parent
    required = [fast_root/name for name in ('run_fast_runtime.py', 'fast_tile_adapter.py', 'fit_fast_cache.py')]
    required += [Path(c['motion_root'])/name for name in ('runtime_group_adapter.py', 'exact_budget_policy.py',
        'group_policy.py', 'coarse_head.py', 'intervention_core.py')]
    required += [Path(c['breadth_root'])/name for name in ('same_shape_tile_mlp.py', 'evaluate_breadth.py',
        'evaluate_breadth_extended.py', 'run_bn_scope.py', 'run_same_shape_reference.py', 'probe_actual_skip.py')]
    required += [Path(c['adapter_dir'])/name for name in ('adapter.py', 'source_provenance.json')]
    require(all(str(p) in c['source_pins'] for p in required), 'Missing source pin')
    sys.path[:0] = [str(fast_root), c['motion_root'], c['breadth_root']]
    import run_fast_runtime as parent
    require(Path(parent.__file__).resolve() == matches[0].resolve() and sha(parent.__file__) == PARENT_RUNNER_SHA,
            'Imported parent runner differs')
    global direction_regions
    direction_regions = parent.direction_regions
    import evaluate_breadth as base
    import evaluate_breadth_extended as extended
    import run_bn_scope as scope
    import run_same_shape_reference as loader
    base_args = list(c['base_args'])
    require(base_args.count('--output') == 1, 'Parent output argument differs')
    output = args.output_dir/'results.json'
    base_args[base_args.index('--output')+1] = str(output)
    a = extended.parse_extended(base_args)
    require(a.device == 'cpu' and a.cpu_threads == 2 and a.wall_minutes == 30 and
            a.geometry == 'full9' and not a.run_inference and a.actual_pair_count == 0 and not a.output.exists(),
            'Invalid fixed CPU arguments or output exists')
    context = extended.extended_preflight(a)
    result = {'schema': 'flow_selection_controls_results_v1', 'status': 'PREFLIGHT_PASS_NO_INFERENCE',
        'contract': c, 'parent_contract_sha256': args.parent_contract_sha256,
        'entry_sha256': sha(__file__), 'prior_results_sha256': args.prior_results_sha256,
        'native_preflight': context[-1], 'effective_base_args': base_args,
        'arm_specs': [list(x) for x in ARM_SPECS], 'planned_full_forwards': 33,
        'inference_executed': False, 'backbone_or_gate_recovery_performed': False,
        'source_scope': 'Pinned DynamicSwin FastBranch plus unchanged runtime four-policy selector; ordinary controls, not new training',
        'data_scope': 'Same4 training-side development targets; backbone-seen, repeatedly explored; not confirmation/test',
        'cost_scope': 'Equal900 slow points; trained share3000 fast rows, mean has no fast FC; controller costs separate; no PE claim',
        'selector_scope': 'Same pre-MLP head/magnitude/eligibility/tail/exact DP; only group direction/constant replacement differs',
        'control_limit': 'Horizontal/rotated alter border eligibility and group topology/counts; fixed local not Euclidean-distance matched',
        'hardware_claim': False}
    if not args.execute:
        require(args.preflight_output and not args.preflight_output.exists(), 'New preflight output required')
        write(args.preflight_output, result); print(result['status']); return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(c, a, context, base, scope, loader, result, prior)
        # Prior zero is included ONLY after all current anchors reproduced exactly.
        result['paired_contrasts_including_verified_prior_zero'] = paired_contrasts(
            {**result['arms'], 'motion_zero': prior['arms']['motion_zero']})
        result['prior_zero_metrics'] = prior['arms']['motion_zero']
        write(a.output, result)
    except BaseException as exc:
        result.update(status='SELECTION_CONTROLS_INCOMPLETE', error=type(exc).__name__+': '+str(exc),
                      traceback=traceback.format_exc())
        write(a.output, result); raise


if __name__ == '__main__':
    main()
