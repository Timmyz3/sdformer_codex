#!/usr/bin/env python3
"""Bounded, SHA-pinned full-network paired-output-omission diagnostic.

Uses dense teacher flow OFFLINE. All MLP producers still execute. This is a
structural probe before learning the runtime fast/slow predictor, not its result.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
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
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def run(contract, a, context, base, scope, loader, result):
    import numpy as np
    import torch
    import intervention_core as core
    import group_policy
    torch.set_num_threads(a.cpu_threads); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device(a.device)
    require(device.type in ('cpu', 'cuda'), 'Unsupported device')
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable')
        torch.cuda.set_device(device)
    started = time.monotonic()
    def deadline():
        require(time.monotonic()-started < a.wall_minutes*60, 'Bounded probe wall time exhausted')
    model, functional = loader.load_model(a, context, base, torch, device)
    bn_record = scope.apply_scope(model, 'MLP_CMA176', a.bn_payload, torch)
    bn_record['execution'] = 'Complete native producers execute; selected MLP branch outputs are zeroed before residual'
    before_bn = base.bn_signature(model, torch)
    modules = {n: m for n, m in model.named_modules() if type(m).__name__ == 'MS_Spiking_Mlp'}
    names = [f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp' for stage in contract['stages']]
    require(set(names) <= set(modules) and len(set(names)) == len(names), 'Declared layers absent/duplicate')
    native, trainer, metrics, rows, hashes, config, saved_run, side, regions, original_plan = context
    chosen = core.choose_targets(rows, contract['target_count'])
    result.update(status='RUNNING', inference_executed=True, BN_scope=bn_record,
                  targets=[r['target_id'] for r in chosen], layers=names,
                  runtime={'device': str(device), 'torch': str(torch.__version__),
                           'GPU_inference_executed': device.type == 'cuda', 'autocast': False, 'TF32': False},
                  frames=[], input_evidence={})
    def predict(chunk, layer=None, dropped=()):
        deadline(); functional.reset_net(model)
        cm = core.masked_branch(modules[layer], dropped) if layer else contextlib.nullcontext([])
        with cm as calls:
            out = model(chunk)['flow'][-1]
            require(tuple(out.shape) == (1, 2, 480, 640) and bool(torch.isfinite(out).all()), 'Invalid full flow')
            out = out[0].detach().cpu().numpy().copy()
        return out, list(calls)
    with torch.inference_mode():
        for row in chosen:
            deadline(); expected = {}; evidence = {}
            for kind, rel in native.target_files(row).items():
                native.checked_path(a.data_root, rel)
                expected['events' if kind == 'input' else kind] = hashes[rel]['sha256']
            arrays, _ = metrics.load_arrays(a.data_root/'saved_flow_data', row,
                                            evidence=evidence, expected_hashes=expected)
            masks = metrics.partitions(arrays['gt'], arrays['mask'], arrays['events'], regions)
            chunk = base.prepare_input(arrays['events'], device, torch)
            captured = {}; handles = []
            def capture(name, u):
                require(name not in captured and tuple(u.shape[:2]) == (10, 1), 'Layer count/layout differs')
                captured[name] = {'shape_TBHWC': list(u.shape), 'C': modules[name].fc1.in_features,
                                  'F': modules[name].fc1.out_features}
            for name in names:
                handles.append(modules[name].register_forward_pre_hook(lambda m, inp, n=name: capture(n, inp[0])))
            try:
                p0, _ = predict(chunk)
            finally:
                for handle in handles:
                    handle.remove()
            require(set(captured) == set(names), 'Missing dense shape capture')
            frame = {'target_id': row['target_id'], 'sequence': row['sequence'],
                     'base': metrics.MetricAccumulator().update(row['target_id'], row['sequence'], p0, arrays['gt'], masks),
                     'layers': []}
            result['frames'].append(frame); result['input_evidence'][row['target_id']] = evidence
            for name in names:
                shape = captured[name]['shape_TBHWC']; h, w = shape[2:4]
                # Area-pool teacher vectors; vectors remain in full-output pixel units.
                feature_flow = torch.nn.functional.interpolate(torch.from_numpy(p0)[None], size=(h, w), mode='area')[0]
                grid = [[feature_flow[:, gy*9:min((gy+1)*9, h), gx*9:min((gx+1)*9, w)].mean((1, 2)).tolist()
                         for gx in range(math.ceil(w/9))] for gy in range(math.ceil(h/9))]
                tile_pixels = (9*640/w, 9*480/h)
                quantize = lambda dx, dy: group_policy.quantize_offset(dx, dy, tile_size_pixels=tile_pixels,
                    association_mode='direction_one_hop', min_motion_pixels=contract['min_motion_pixels'])
                pairs, selection = core.comparable_pairs(grid, h, w, target=row['target_id'], layer=name,
                    count=contract['pairs_per_layer'], quantize=quantize)
                item = {'layer': name, **captured[name], 'selection': selection, 'pairs': [],
                        'nominal_tile_extent_output_pixels_xy': list(tile_pixels),
                        'flow_mapping': 'area pooled full-output teacher field; direction neighborhood hypothesis, not exact receptive-field correspondence'}
                frame['layers'].append(item)
                cache = {}
                def intervention(tiles):
                    key = tuple(sorted(tuple(t) for t in tiles))
                    if key not in cache:
                        pred, calls = predict(chunk, name, key)
                        cache[key] = (pred, calls)
                    return cache[key]
                # Empty intervention must not change original full-shape arithmetic.
                replay, empty_calls = intervention([])
                require(np.array_equal(p0, replay), 'Empty hook/repeated dense forward changed output')
                item['empty_intervention_exact'] = True
                for pair in pairs:
                    anchor = pair['anchor_yx']; pa, ca = intervention([anchor])
                    record = {**pair, 'conditions': {}}
                    item['pairs'].append(record)
                    for policy, endpoint in pair['endpoints_yx'].items():
                        pb, cb = intervention([endpoint]); pab, cab = intervention([anchor, endpoint])
                        statistic = core.pair_statistics(p0, pa, pb, pab, arrays['gt'], masks)
                        record['conditions'][policy] = {'regions': statistic,
                            'a_execution': ca, 'b_execution': cb, 'ab_execution': cab,
                            'same_pair_as_other_control': [other for other, end in pair['endpoints_yx'].items()
                                                         if other != policy and end == endpoint]}
                item['unique_intervention_forwards'] = len(cache)
                del cache
                write(a.output, result)
            print(json.dumps({'target': row['target_id'], 'complete_frames': len(result['frames']),
                              'planned_frames': len(chosen)}), flush=True)
            del chunk
    require(base.bn_signature(model, torch) == before_bn, 'BN state changed during fixed diagnostic')
    result.update(status='PAIR_DIAGNOSTIC_COMPLETE', wall_seconds=time.monotonic()-started,
                  BN_buffers_unchanged=True, complete_frames=len(result['frames']),
                  pair_anchors=sum(len(layer['pairs']) for frame in result['frames'] for layer in frame['layers']))
    write(a.output, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--preflight-output', type=Path)
    parser.add_argument('--wait-for-gpu', action='store_true')
    args = parser.parse_args(argv)
    require(sha(args.contract) == args.contract_sha256, 'Contract SHA differs')
    contract = json.loads(args.contract.read_text())
    require(contract['schema'] == 'flow_motion_pair_probe_contract_v1', 'Unknown contract')
    require(1 <= contract['target_count'] <= 8 and 1 <= contract['pairs_per_layer'] <= 2 and
            set(contract['stages']) <= {0, 2} and bool(contract['stages']), 'Probe scope exceeds bound')
    require(contract['min_motion_pixels'] == 1.0, 'Motion eligibility threshold changed')
    require(not args.wait_for_gpu or args.execute, 'Waiting requires an execution request')
    for path, expected in contract['source_pins'].items():
        require(sha(path) == expected, 'Source changed: '+path)
    here = Path(__file__).resolve().parent
    required = [here/'run_pair_probe.py', here/'intervention_core.py', here/'group_policy.py']
    breadth = Path(contract['breadth_root']).resolve()
    required += [breadth/name for name in ('evaluate_breadth.py', 'evaluate_breadth_extended.py',
                 'run_bn_scope.py', 'run_same_shape_reference.py', 'probe_actual_skip.py')]
    require(all(str(p) in contract['source_pins'] for p in required), 'Required source not pinned')
    sys.path.insert(0, str(breadth))
    import evaluate_breadth as base
    import evaluate_breadth_extended as extended
    import run_bn_scope as scope
    import run_same_shape_reference as loader
    a = extended.parse_extended(contract['base_args'])
    require(not a.run_inference and a.geometry == 'full9' and a.regions and a.bn_artifact and
            a.actual_pair_count == 0 and a.wall_minutes <= 20, 'Invalid fixed diagnostic arguments')
    context = extended.extended_preflight(a)
    result = {'schema': 'flow_motion_pair_probe_v1', 'status': 'PREFLIGHT_PASS_NO_INFERENCE',
              'inference_executed': False, 'contract_sha256': args.contract_sha256, 'contract': contract,
              'native_preflight': context[-1], 'training_performed': False,
              'source_scope': 'Locally implemented paired-output-omission diagnostic; not DynamicSwin/TP full reproduction',
              'flow_source': 'same-target dense final prediction, OFFLINE teacher diagnostic unavailable to runtime gate',
              'intervention': 'full-shape dense native MLP then zero selected complete-T10 branches before residual',
              'hardware_claim': False, 'producer_cancellation_executed': False,
              'network_interaction_scope': 'Includes downstream nonlinearities and full-domain current-stat external BN; not by itself evidence of motion-specific dependence',
              'data_scope': '8 or fewer samples from existing train-side development32, already seen by backbone; not independent validation',
              'rival_explanation_control': 'additive prediction counterfactual separates EPE norm nonlinearity from network response'}
    if not args.execute:
        require(args.preflight_output and not args.preflight_output.exists(), 'New preflight output required')
        write(args.preflight_output, result)
        print(json.dumps({'status': result['status']})); return
    queue_path = Path(contract['queue_status'])
    require(not queue_path.exists(), 'Queue receipt already exists')
    waiting_started = time.monotonic()
    lock = Path(contract['gpu_lock']).open('a') if a.device.startswith('cuda') else contextlib.nullcontext()
    try:
        with lock as handle:
            if a.device.startswith('cuda'):
                while True:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        require(args.wait_for_gpu, 'GPU lock busy; no concurrent inference permitted')
                        elapsed = time.monotonic()-waiting_started
                        require(elapsed < 14*3600, 'Maximum14h queue wait reached; no automatic retry')
                        write(queue_path, {'status': 'WAITING_GPU_LOCK', 'pid': os.getpid(),
                                          'wait_seconds': elapsed, 'maximum_wait_seconds': 14*3600,
                                          'GPU_initialized': False, 'contract_sha256': args.contract_sha256,
                                          'existing_training_interrupted': False})
                        time.sleep(30)
                # A queued process may wait hours: bind the actual files again
                # while owning the device, before any CUDA initialization.
                require(sha(args.contract) == args.contract_sha256, 'Queued contract changed')
                for path, expected in contract['source_pins'].items():
                    require(sha(path) == expected, 'Queued source changed: '+path)
                require(not a.output.exists(), 'Queued output already exists; refusing overwrite')
                context = extended.extended_preflight(a)
                result['native_preflight'] = context[-1]
                result['post_wait_provenance_rechecked'] = True
            write(queue_path, {'status': 'RUNNING', 'pid': os.getpid(),
                               'wait_seconds': time.monotonic()-waiting_started,
                               'contract_sha256': args.contract_sha256, 'device': a.device})
            run(contract, a, context, base, scope, loader, result)
        write(queue_path, {'status': result['status'], 'pid': os.getpid(), 'output': str(a.output),
                           'contract_sha256': args.contract_sha256,
                           'complete_frames': result.get('complete_frames'), 'pair_anchors': result.get('pair_anchors')})
    except BaseException as exc:
        result.update(status='PAIR_DIAGNOSTIC_INCOMPLETE', error=type(exc).__name__+': '+str(exc),
                      traceback=traceback.format_exc())
        write(a.output, result)
        write(queue_path, {'status': 'FAILED', 'error': result['error'], 'output': str(a.output)})
        raise


if __name__ == '__main__':
    main()
