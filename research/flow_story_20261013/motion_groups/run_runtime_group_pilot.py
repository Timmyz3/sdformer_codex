#!/usr/bin/env python3
"""Bounded CPU runtime-head / actual MLP cancellation pilot; zero recovery."""
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
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def run(c, a, context, base, scope, loader, result):
    import numpy as np
    import torch
    import coarse_head
    import intervention_core as core
    import runtime_group_adapter as adapter
    import same_shape_tile_mlp as tile
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA for bounded CPU pilot')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    started = time.monotonic()
    def deadline():
        require(time.monotonic()-started < a.wall_minutes*60, 'CPU pilot wall budget exhausted')
    weights = torch.load(c['head_weights'], map_location='cpu', weights_only=True)
    require(weights['schema'] == 'flow_coarse_folded_head_weights_v1' and
            weights['provenance']['cache_sha256'] == c['head_fit_cache_sha256'] and
            weights['provenance']['source_sha256'] == c['head_fit_source_sha256'] and
            weights['provenance']['GPU_used'] is False and
            weights['flow_contract'] == c['flow_contract'], 'Head provenance differs')
    recipe = weights['recipe']
    require(recipe['steps_per_trainable_head'] == 1000 and recipe['seed'] == 20261013 and
            recipe['development_selection'] is False, 'Fixed head fit recipe differs')
    head = coarse_head.RuntimeCoarseFlowHead(384, flow_contract=c['flow_contract'])
    head.load_state_dict(weights['stages']['2']['methods']['C16_GELU2']['state_dict'], strict=True)
    head.eval().requires_grad_(False)
    model, functional = loader.load_model(a, context, base, torch, torch.device('cpu'))
    bn = scope.apply_scope(model, 'MLP_CMA176', a.bn_payload, torch)
    bn['execution'] = 'Only stage2 first MLP tiled: actual online whole-T10 cancellation; all other modules native'
    bn_before = base.bn_signature(model, torch)
    original = model.get_submodule(adapter.LAYER)
    fc_shapes = {k: [getattr(original, k).in_features, getattr(original, k).out_features] for k in ('fc1', 'fc2')}
    native, trainer, metrics, rows, hashes, config, saved_run, side, regions, original_plan = context
    targets = core.choose_targets(rows, 4)
    arms = ('native_dense', 'tiled_all_keep') + adapter.POLICIES
    meters = {p: metrics.MetricAccumulator() for p in arms}
    result.update(status='RUNNING', inference_executed=True, BN_scope=bn, fc_shapes=fc_shapes,
        targets=[r['target_id'] for r in targets], frames=[], input_evidence={},
        runtime={'device': 'cpu', 'threads': 2, 'torch': str(torch.__version__), 'GPU_initialized': False,
                 'autocast': False, 'TF32': False}, numerical_reference_frames=1)

    def predict(chunk, mode=None, policy=None):
        deadline(); functional.reset_net(model)
        begin = time.monotonic(); captures = {}; observed = {k+'_rows': 0 for k in ('fc1', 'fc2')}
        cm = adapter.replace_one_mlp(model, tile, mode, policy) if mode else contextlib.nullcontext(None)
        with cm as wrapper:
            current = model.get_submodule(adapter.LAYER)
            def pre(m, inputs):
                require(not captures, 'Target called more than once')
                captures.update(input_sha256=adapter.tensor_sha(inputs[0]), shape=list(inputs[0].shape))
            def post(m, inputs, out):
                captures.update(branch_sha256=adapter.tensor_sha(out), output_stride=list(out.stride()))
            handles = [current.register_forward_pre_hook(pre), current.register_forward_hook(post)]
            if wrapper is None:
                for key in ('fc1', 'fc2'):
                    def count(m, inputs, k=key):
                        observed[k+'_rows'] += inputs[0].numel()//inputs[0].shape[-1]
                    handles.append(getattr(original, key).register_forward_pre_hook(count))
            try:
                out = model(chunk)['flow'][-1]
                require(tuple(out.shape) == (1, 2, 480, 640) and bool(torch.isfinite(out).all()), 'Invalid output')
                pred = out[0].detach().cpu().numpy().copy()
            finally:
                for handle in handles:
                    handle.remove()
            records = list(wrapper.records) if wrapper else []
        require(captures.get('shape') == list(adapter.SHAPE), 'Missing/wrong target invocation')
        if wrapper is None:
            require(observed == {'fc1_rows': 12000, 'fc2_rows': 12000}, 'Native FC service differs')
        else:
            require(len(records) == 1, 'Wrong wrapper call count'); observed = records[0]['observed']
        return pred, {'capture': captures, 'execution': records, 'native_FC_observed': observed,
                      'wall_seconds_CPU': time.monotonic()-begin}

    with torch.inference_mode():
        for index, row in enumerate(targets):
            deadline(); expected = {}; evidence = {}
            for kind, rel in native.target_files(row).items():
                native.checked_path(a.data_root, rel)
                expected['events' if kind == 'input' else kind] = hashes[rel]['sha256']
            arrays, _ = metrics.load_arrays(a.data_root/'saved_flow_data', row,
                                           evidence=evidence, expected_hashes=expected)
            masks = metrics.partitions(arrays['gt'], arrays['mask'], arrays['events'], regions)
            chunk = base.prepare_input(arrays['events'], torch.device('cpu'), torch)
            frame = {'target_id': row['target_id'], 'sequence': row['sequence'], 'arms': {}}
            result['frames'].append(frame); result['input_evidence'][row['target_id']] = evidence
            predictions = {}
            for arm in arms:
                if arm == 'native_dense':
                    pred, record = predict(chunk)
                elif arm == 'tiled_all_keep':
                    pred, record = predict(chunk, 'tiled_all_then_mask',
                        lambda name, u: torch.ones((4, 5), dtype=torch.bool, device=u.device))
                else:
                    policy = adapter.RuntimeGroupPolicy(head, arm)
                    pred, record = predict(chunk, 'tiled_kept_only', policy)
                    require(len(policy.records) == 1, 'Runtime selector call count differs')
                    record['runtime_selector'] = policy.records[0]
                    require(record['execution'][0]['kept_real_pixels'] ==
                            policy.records[0]['selection']['slow_service_cost'], 'Actual work differs from decision')
                    if index == 0:
                        replay = tile.ReplayPolicy(record['execution'])
                        ref, ref_record = predict(chunk, 'tiled_all_then_mask', replay)
                        replay.complete()
                        exact = bool(np.array_equal(pred, ref))
                        record['same_shape_reference'] = {'execution': ref_record,
                            'final_flow_exact': exact, 'final_flow_max_abs': float(np.abs(pred-ref).max())}
                        require(record['capture'] == ref_record['capture'] and exact,
                                'Actual cancellation differs from same-mask same-shape full production')
                predictions[arm] = pred
                record['metrics'] = meters[arm].update(row['target_id'], row['sequence'], pred, arrays['gt'], masks)
                frame['arms'][arm] = record
                write(a.output, result)
            require(len({frame['arms'][p]['capture']['input_sha256'] for p in arms}) == 1,
                    'Policy arms do not share identical upstream input')
            work = [frame['arms'][p]['execution'][0]['kept_real_pixels'] for p in adapter.POLICIES]
            require(len(set(work)) == 1, 'Four policies consumed different actual work')
            frame['common_actual_slow_points'] = work[0]
            frame['native_vs_all_tiled'] = {'final_flow_exact': bool(np.array_equal(predictions['native_dense'], predictions['tiled_all_keep'])),
                'final_flow_max_abs': float(np.abs(predictions['native_dense']-predictions['tiled_all_keep']).max())}
            print(json.dumps({'target': row['target_id'], 'complete_frames': index+1, 'common_actual_slow_points': work[0]}), flush=True)
            del chunk, predictions
    require(base.bn_signature(model, torch) == bn_before and model.get_submodule(adapter.LAYER) is original,
            'BN buffers or native module changed')
    require(not torch.cuda.is_initialized(), 'CPU run unexpectedly initialized CUDA')
    result.update(status='RUNTIME_GROUP_CPU_COMPLETE', complete_frames=4,
                  wall_seconds=time.monotonic()-started, BN_buffers_unchanged=True,
                  arms={p: {'summary': m.summary(), 'frames': m.frames} for p, m in meters.items()})
    write(a.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--preflight-output', type=Path)
    args = parser.parse_args()
    require(sha(args.contract) == args.contract_sha256, 'Contract SHA differs')
    c = json.loads(args.contract.read_text())
    require(c['schema'] == 'flow_runtime_group_cpu_contract_v1' and c['target_count'] == 4 and c['cap'] == 900,
            'Scope differs')
    for path, digest in c['source_pins'].items():
        require(sha(path) == digest, 'Source differs: '+path)
    require(sha(c['head_weights']) == c['head_weights_sha256'], 'Head weights differ')
    here = Path(__file__).resolve().parent; breadth = Path(c['breadth_root']).resolve()
    required = [here/name for name in ('run_runtime_group_pilot.py', 'runtime_group_adapter.py',
        'exact_budget_policy.py', 'coarse_head.py', 'group_policy.py', 'intervention_core.py')]
    required += [breadth/name for name in ('same_shape_tile_mlp.py', 'evaluate_breadth.py',
        'evaluate_breadth_extended.py', 'run_bn_scope.py', 'run_same_shape_reference.py', 'probe_actual_skip.py')]
    require(all(str(p) in c['source_pins'] for p in required), 'Missing source pin')
    sys.path.insert(0, str(breadth))
    import evaluate_breadth as base
    import evaluate_breadth_extended as extended
    import run_bn_scope as scope
    import run_same_shape_reference as loader
    a = extended.parse_extended(c['base_args'])
    require(a.device == 'cpu' and a.cpu_threads == 2 and a.wall_minutes == 30 and
            not a.run_inference and a.geometry == 'full9' and a.actual_pair_count == 0,
            'Only fixed bounded CPU experiment allowed')
    require(not a.output.exists(), 'Output exists; refuse overwrite')
    context = extended.extended_preflight(a)
    result = {'schema': 'flow_runtime_group_cpu_results_v1', 'status': 'PREFLIGHT_PASS_NO_INFERENCE',
        'contract_sha256': args.contract_sha256, 'contract': c, 'native_preflight': context[-1],
        'inference_executed': False, 'training_performed': False,
        'source_scope': 'locally implemented zero-recovery identity bypass diagnostic; not full DynamicSwin reproduction',
        'data_scope': 'fixed4 train-side head-dev8 / backbone-seen targets; not independent validation',
        'producer_cancellation_scope': 'stage2 first MLP only, complete T10 spatial9x9 valid-tail tiles',
        'runtime_signal': 'frozen fit20 coarse head from current pre-MLP u; no GT/final flow in selector',
        'hardware_claim': False,
        'cost_scope': 'exact same retained real points/native FC rows; NOT identical PE cycles, packing or controller cost',
        'scheduler_scope': 'CPU research DP computes all four proposals online; all work recorded, not ASIC implementation',
        'aggregation_scope': 'group=max(member score), independent=single score; objectives not numerically comparable',
        'upstream_attention_cancelled': False}
    if not args.execute:
        require(args.preflight_output and not args.preflight_output.exists(), 'New preflight path required')
        write(args.preflight_output, result); print(result['status']); return
    try:
        run(c, a, context, base, scope, loader, result)
    except BaseException as exc:
        result.update(status='RUNTIME_GROUP_CPU_INCOMPLETE', error=type(exc).__name__+': '+str(exc),
                      traceback=traceback.format_exc())
        write(a.output, result)
        raise


if __name__ == '__main__':
    main()
