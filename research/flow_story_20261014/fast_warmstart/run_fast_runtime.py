#!/usr/bin/env python3
"""CPU full-flow evaluation of fit-only teacher-feature fast-branch warmstart."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True


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
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def direction_regions(gt, valid, runtime_flow, flow_contract):
    """POSTHOC evaluation strata; never called by the runtime selector."""
    import numpy as np
    import torch
    import coarse_head
    import group_policy
    target = coarse_head.make_tile_targets(torch.from_numpy(gt), torch.from_numpy(valid),
        feature_hw=(30, 40), feature_stride_pixels=(16, 16),
        flow_contract=flow_contract, image_origin_xy=(0., 0.))
    # Explicit key bound to the frozen label helper; no alternative labels guessed.
    truth = target['flow'][0].numpy()
    labels = np.zeros((4, 5), dtype=np.int8)
    def direction(v):
        return group_policy.quantize_offset(float(v[0]), float(v[1]), tile_size_pixels=(144., 144.),
            association_mode='direction_one_hop', min_motion_pixels=1.)
    for y in range(4):
        for x in range(5):
            if not bool(target['valid'][0, y, x]):
                continue
            actual = direction(truth[y, x])
            if actual is None:
                continue
            predicted = direction(runtime_flow[y][x])
            labels[y, x] = 3 if predicted is None else (1 if predicted == actual else 2)
    expanded = np.repeat(np.repeat(labels, 144, axis=0), 144, axis=1)[:480, :640]
    masks = {'direction_correct': valid & (expanded == 1),
             'direction_wrong': valid & (expanded == 2),
             'direction_abstain': valid & (expanded == 3),
             'direction_GT_ineligible': valid & (expanded == 0)}
    require(np.array_equal(sum(m.astype(np.int8) for m in masks.values()), valid.astype(np.int8)), 'Direction strata do not partition GT')
    return masks, {'definition': 'GT tile-mean norm>=1px; 8-bin angle; predicted norm<1px=abstain; not actual correspondence/skip-risk truth',
        'head_source': 'same frozen runtime flow shared by all arms, computed before evaluation GT masks',
        'tile_labels_0ineligible_1correct_2wrong_3abstain': labels.tolist(),
        'GT_used_by_inference': False}


def run(c, a, context, base, scope, loader, result):
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
    arm_specs = [('native_dense', None, None), ('tiled_all_keep', 'zero', None),
                 ('independent_zero', 'zero', 'independent'), ('motion_zero', 'zero', 'motion'),
                 ('independent_mean', 'mean', 'independent'),
                 ('independent_trained', 'trained', 'independent'), ('motion_trained', 'trained', 'motion')]
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
                    if index == 0 and name in ('independent_zero', 'independent_mean', 'independent_trained', 'motion_trained'):
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
            for family in ('independent', 'motion'):
                require(frame['arms'][family+'_zero']['execution'][0]['keep'] ==
                        frame['arms'][family+'_trained']['execution'][0]['keep'], 'Branch change modified selector')
            masks = metrics.partitions(arrays['gt'], arrays['mask'], arrays['events'], regions)
            extra, definition = direction_regions(arrays['gt'], arrays['mask'], flow, c['flow_contract'])
            masks.update(extra); frame['direction_error_strata'] = definition
            for name in predictions:
                frame['arms'][name]['metrics'] = meters[name].update(row['target_id'], row['sequence'], predictions[name], arrays['gt'], masks)
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
    result.update(status='FAST_WARMSTART_RUNTIME_COMPLETE', complete_frames=4, BN_buffers_unchanged=True,
                  wall_seconds=time.monotonic()-started,
                  arms={name: {'summary': meter.summary(), 'frames': meter.frames} for name, meter in meters.items()})
    write(a.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True); parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--preflight-output', type=Path)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and sha(args.contract) == args.contract_sha256,
            'Explicit CPU or contract binding differs')
    c = json.loads(args.contract.read_text())
    require(c['schema'] == 'flow_fast_runtime_contract_v1' and c['target_count'] == 4, 'Unknown scope')
    for path, digest in c['source_pins'].items():
        require(sha(path) == digest, 'Source differs: '+path)
    for key in ('head_weights', 'fast_weights'):
        require(sha(c[key]) == c[key+'_sha256'], 'Weights differ: '+key)
    here = Path(__file__).resolve().parent
    required = [here/name for name in ('run_fast_runtime.py', 'fast_tile_adapter.py', 'fit_fast_cache.py')]
    required += [Path(c['motion_root'])/name for name in ('runtime_group_adapter.py', 'exact_budget_policy.py',
        'group_policy.py', 'coarse_head.py', 'intervention_core.py')]
    required += [Path(c['breadth_root'])/name for name in ('same_shape_tile_mlp.py', 'evaluate_breadth.py',
        'evaluate_breadth_extended.py', 'run_bn_scope.py', 'run_same_shape_reference.py', 'probe_actual_skip.py')]
    required += [Path(c['adapter_dir'])/name for name in ('adapter.py', 'source_provenance.json')]
    require(all(str(p) in c['source_pins'] for p in required), 'Missing source pin')
    sys.path.insert(0, c['motion_root']); sys.path.insert(0, c['breadth_root'])
    import evaluate_breadth as base
    import evaluate_breadth_extended as extended
    import run_bn_scope as scope
    import run_same_shape_reference as loader
    a = extended.parse_extended(c['base_args'])
    require(a.device == 'cpu' and a.cpu_threads == 2 and a.wall_minutes == 30 and
            a.geometry == 'full9' and not a.run_inference and a.actual_pair_count == 0 and not a.output.exists(),
            'Invalid fixed CPU arguments or output exists')
    context = extended.extended_preflight(a)
    result = {'schema': 'flow_fast_runtime_results_v1', 'status': 'PREFLIGHT_PASS_NO_INFERENCE',
        'contract': c, 'contract_sha256': args.contract_sha256, 'native_preflight': context[-1],
        'inference_executed': False, 'backbone_or_gate_recovery_performed': False,
        'source_scope': 'pinned DynamicSwin FastBranch topology; fit20 teacher-feature warmstart only; not full method reproduction',
        'data_scope': 'same4 train-side development targets; backbone-seen, fast-fit-disjoint, not formal validation',
        'cost_scope': 'same slow work; learned/constant/identity fast paths have different additional costs; no PE claim',
        'selector_scope': 'frozen pre-MLP coarse head + fixed magnitude score; no GT/final flow',
        'direction_strata_scope': 'posthoc coarse direction error, not GT skip-risk labels', 'hardware_claim': False}
    if not args.execute:
        require(args.preflight_output and not args.preflight_output.exists(), 'New preflight output required')
        write(args.preflight_output, result); print(result['status']); return
    try:
        run(c, a, context, base, scope, loader, result)
    except BaseException as exc:
        result.update(status='FAST_WARMSTART_RUNTIME_INCOMPLETE', error=type(exc).__name__+': '+str(exc),
                      traceback=traceback.format_exc())
        write(a.output, result); raise


if __name__ == '__main__':
    main()
