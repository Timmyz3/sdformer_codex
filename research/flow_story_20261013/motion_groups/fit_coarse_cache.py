#!/usr/bin/env python3
"""Fixed-recipe CPU coarse-head fit; training-side exploration, not full-flow AEE.

CLI accepts only a SHA-bound cache with 20 fit / 8 dev targets and stages0/2.
Dev is never used in input normalization, optimization, model/step selection.
Only small folded head weights are saved; no backbone/checkpoint is changed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn

from coarse_head import RuntimeCoarseFlowHead
from group_policy import quantize_offset


SEED = 20261013
STEPS = 1000
METHODS = ('constant_component_median', 'linear', 'C16_GELU2')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_cache(cache, strict_population=False):
    require(isinstance(cache, dict) and cache.get('schema') == 'flow_coarse_feature_cache_v1', 'Foreign cache schema')
    contract = cache.get('flow_contract')
    require(isinstance(contract, dict) and set(contract) == {'direction', 'time_interval', 'units'} and
            all(isinstance(v, str) and v.strip() for v in contract.values()) and
            contract['units'] == 'original_image_pixels', 'Explicit pixel-displacement flow contract required')
    frames = cache.get('frames')
    require(isinstance(frames, list) and frames, 'Nonempty frames required')
    ids, roles, sequences, stage_names, channels = set(), {'fit': [], 'dev': []}, {'fit': set(), 'dev': set()}, None, {}
    for frame in frames:
        target, role, sequence = frame.get('target_id'), frame.get('role'), frame.get('sequence')
        require(isinstance(target, str) and target and target not in ids, 'Repeated/missing target_id')
        require(role in roles and isinstance(sequence, str) and sequence and target.startswith(sequence + '_'), 'Bad role/sequence')
        ids.add(target); roles[role].append(target); sequences[role].add(sequence)
        stages = frame.get('stages')
        require(isinstance(stages, dict) and stages and set(stages) <= {'0', '2'}, 'Only stage0/2 are supported')
        if stage_names is None:
            stage_names = set(stages)
        require(set(stages) == stage_names, 'Every frame must contain the same stages')
        for stage, item in stages.items():
            x, target_info = item.get('descriptor'), item.get('targets')
            require(isinstance(x, torch.Tensor) and x.device.type == 'cpu' and x.is_floating_point() and
                    x.ndim == 4 and x.shape[0] == 1 and all(v > 0 for v in x.shape) and
                    bool(torch.isfinite(x).all()), 'Finite CPU descriptor [1,ny,nx,C] required')
            channels.setdefault(stage, x.shape[-1])
            require(channels[stage] == x.shape[-1], 'Stage channel count changed')
            require(isinstance(target_info, dict) and target_info.get('schema') == 'flow_runtime_coarse_targets_v1' and
                    target_info.get('flow_contract') == contract, 'Target flow contract differs')
            y, valid, counts = (target_info.get(k) for k in ('flow', 'valid', 'valid_pixel_count'))
            require(isinstance(y, torch.Tensor) and y.device.type == 'cpu' and y.is_floating_point() and
                    tuple(y.shape) == tuple(x.shape[:-1]) + (2,), 'Target/descriptor shape mismatch')
            require(isinstance(valid, torch.Tensor) and valid.device.type == 'cpu' and valid.dtype == torch.bool and
                    tuple(valid.shape) == tuple(x.shape[:-1]), 'Invalid tile validity mask')
            require(isinstance(counts, torch.Tensor) and counts.device.type == 'cpu' and
                    counts.dtype in (torch.int32, torch.int64) and tuple(counts.shape) == tuple(valid.shape) and
                    bool((counts >= 0).all()) and bool((valid == (counts > 0)).all()), 'Invalid GT counts')
            require(bool(torch.isfinite(y[valid]).all()), 'Nonfinite valid GT label')
    require(roles['fit'] and roles['dev'], 'Both fit and development frames are required')
    require(not (sequences['fit'] & sequences['dev']), 'Fit/development sequences overlap')
    if strict_population:
        require(len(roles['fit']) == 20 and len(roles['dev']) == 8 and stage_names == {'0', '2'},
                'CLI requires fixed fit20/dev8 with both stages0/2')
    return {'target_ids': roles, 'target_ids_sha256': {k: digest(v) for k, v in roles.items()},
            'sequence_counts': {k: {s: sum(f['role'] == k and f['sequence'] == s for f in frames)
                                    for s in sorted(sequences[k])} for k in roles},
            'stages': sorted(stage_names), 'channels': channels,
            'fit_dev_target_intersection': 0, 'fit_dev_sequence_intersection': 0}


def load_verified_cache(path, expected_sha, strict_population=True):
    require(isinstance(expected_sha, str) and len(expected_sha) == 64 and
            all(c in '0123456789abcdef' for c in expected_sha), 'Full lowercase cache SHA256 required')
    require(sha_file(path) == expected_sha, 'Cache SHA256 mismatch')
    # No unsafe pickle fallback, even for older or foreign cache objects.
    cache = torch.load(path, map_location='cpu', weights_only=True)
    return cache, validate_cache(cache, strict_population=strict_population)


def fit_normalizer(frames, stage):
    """Population mean/std over ALL fit descriptors, including no-GT tiles.

    No GT value, dev descriptor, or dev selection enters these statistics.
    Near-constant channels (std<1e-6) use divisor1, with the count recorded.
    """
    x = torch.cat([f['stages'][stage]['descriptor'].detach().reshape(-1, f['stages'][stage]['descriptor'].shape[-1]).double()
                   for f in frames if f['role'] == 'fit'], dim=0)
    mean = x.mean(dim=0)
    std = ((x - mean).square().mean(dim=0)).sqrt()
    divisor = torch.where(std < 1e-6, torch.ones_like(std), std)
    return {'mean': mean.float(), 'divisor': divisor.float(), 'population_std': std.float(),
            'descriptor_count': x.shape[0], 'constant_or_tiny_std_channels': int((std < 1e-6).sum()),
            'minimum_std_rule': 'std<1e-6 uses divisor1', 'source_role': 'fit_only_all_descriptors'}


def descriptor_forward(model, method, x):
    if method == 'linear':
        return model(x)
    if method == 'C16_GELU2':
        return model.fc2(model.activation(model.fc1(x)))
    raise ValueError('Unknown trainable head')


def fold_normalizer(model, method, normalizer):
    """Fold (raw-mean)/divisor into the first FC; model copy accepts raw u means."""
    folded = copy.deepcopy(model)
    first = folded if method == 'linear' else folded.fc1
    with torch.no_grad():
        weight = first.weight.double()
        mean, divisor = normalizer['mean'].double(), normalizer['divisor'].double()
        folded_weight = weight / divisor.unsqueeze(0)
        folded_bias = first.bias.double() - (folded_weight * mean.unsqueeze(0)).sum(dim=1)
        first.weight.copy_(folded_weight.to(first.weight.dtype))
        first.bias.copy_(folded_bias.to(first.bias.dtype))
    return folded


def _trainable_head(channels, method, flow_contract, seed):
    torch.manual_seed(seed)
    if method == 'linear':
        return nn.Linear(channels, 2)
    return RuntimeCoarseFlowHead(channels, flow_contract=flow_contract)


def fit_head(frames, stage, method, flow_contract, normalizer, *, steps=STEPS, seed=SEED):
    fit = [f['stages'][stage] for f in frames if f['role'] == 'fit']
    x = torch.cat([item['descriptor'].detach()[item['targets']['valid']].float() for item in fit], dim=0)
    y = torch.cat([item['targets']['flow'].detach()[item['targets']['valid']].float() for item in fit], dim=0)
    require(x.shape[0] > 0, 'No valid fit GT tiles; refusing fabricated supervision')
    if method == 'constant_component_median':
        return y.median(dim=0).values, {'optimized_steps': 0, 'fit_valid_tiles': x.shape[0],
                'median_definition': 'per-component lower order statistic; an L1-optimal constant'}
    require(method in ('linear', 'C16_GELU2'), 'Unknown method')
    normalized = (x - normalizer['mean']) / normalizer['divisor']
    model = _trainable_head(x.shape[-1], method, flow_contract, seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    first_loss = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = descriptor_forward(model, method, normalized)
        loss = (prediction - y).abs().mean()
        require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
        if first_loss is None:
            first_loss = float(loss.detach())
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                'Missing/nonfinite head gradient')
        optimizer.step()
    model.eval()
    folded = fold_normalizer(model, method, normalizer).eval()
    # Both fit and dev validate only the algebraic fold, after all training.
    # No parameter update, best-step/model selection or normalization fitting.
    max_error, max_reference, all_close = 0., 0., True
    with torch.no_grad():
        for frame in frames:
            raw = frame['stages'][stage]['descriptor'].float()
            reference = descriptor_forward(model, method, (raw - normalizer['mean']) / normalizer['divisor'])
            actual = descriptor_forward(folded, method, raw)
            require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all()), 'Nonfinite folded head')
            max_error = max(max_error, float((actual - reference).abs().max()))
            max_reference = max(max_reference, float(reference.abs().max()))
            all_close = all_close and torch.allclose(actual, reference, atol=5e-5, rtol=5e-5)
        last_loss = float((descriptor_forward(model, method, normalized) - y).abs().mean())
    require(all_close, 'Input-standardization fold exceeded fixed FP32 tolerance')
    return folded, {'optimized_steps': steps, 'fit_valid_tiles': x.shape[0],
                    'initial_fit_component_L1': first_loss, 'final_normalized_head_fit_component_L1': last_loss,
                    'parameter_count': sum(p.numel() for p in model.parameters()),
                    'fold_max_abs_error_all_frames': max_error, 'fold_reference_max_abs': max_reference,
                    'fold_checked_roles': ['fit', 'dev'], 'fold_atol': 5e-5, 'fold_rtol': 5e-5, 'fold_pass': True}


def frame_metrics(prediction, targets):
    """All direction denominators stay on valid GT with |GT tile mean|>=1px."""
    valid = targets['valid']; y = targets['flow'][valid].float(); p = prediction[valid].float()
    require(bool(torch.isfinite(p).all()), 'Nonfinite prediction')
    n = y.shape[0]
    result = {'valid_tiles': n, 'empty_tiles': valid.numel() - n,
              'valid_GT_pixels': int(targets['valid_pixel_count'].sum()),
              'component_L1': None, 'EPE': None, 'sum_component_L1': 0., 'sum_EPE': 0.,
              'gt_motion_eligible_tiles': 0, 'pred_motion_tiles': 0,
              'direction_correct': 0, 'direction_abstain_on_GT_eligible': 0,
              'direction_wrong_nonabstain': 0, 'direction_accuracy_GT_eligible': None,
              'GT_motion_eligible_fraction_valid_tiles': None, 'pred_motion_fraction_valid_tiles': None}
    if not n:
        return result
    l1 = (p - y).abs().mean(dim=-1).double()
    epe = torch.linalg.vector_norm(p - y, dim=-1).double()
    result.update(component_L1=float(l1.mean()), EPE=float(epe.mean()),
                  sum_component_L1=float(l1.sum()), sum_EPE=float(epe.sum()))
    for truth, pred in zip(y.tolist(), p.tolist()):
        # Square footprint is sufficient for the shared eight-direction rule;
        # only direction is used, not endpoint displacement or correspondence.
        expected = quantize_offset(*truth, tile_size_pixels=(1., 1.),
                                   association_mode='direction_one_hop', min_motion_pixels=1.)
        observed = quantize_offset(*pred, tile_size_pixels=(1., 1.),
                                   association_mode='direction_one_hop', min_motion_pixels=1.)
        result['pred_motion_tiles'] += observed is not None
        if expected is not None:
            result['gt_motion_eligible_tiles'] += 1
            if observed is None:
                result['direction_abstain_on_GT_eligible'] += 1
            elif observed == expected:
                result['direction_correct'] += 1
            else:
                result['direction_wrong_nonabstain'] += 1
    eligible = result['gt_motion_eligible_tiles']
    result['direction_accuracy_GT_eligible'] = result['direction_correct'] / eligible if eligible else None
    result['GT_motion_eligible_fraction_valid_tiles'] = eligible / n
    result['pred_motion_fraction_valid_tiles'] = result['pred_motion_tiles'] / n
    return result


def aggregate_metrics(rows):
    observed = [r for r in rows if r['valid_tiles'] > 0]
    directional = [r for r in rows if r['gt_motion_eligible_tiles'] > 0]
    count_keys = ('valid_tiles', 'empty_tiles', 'valid_GT_pixels', 'gt_motion_eligible_tiles', 'pred_motion_tiles',
                  'direction_correct', 'direction_abstain_on_GT_eligible', 'direction_wrong_nonabstain')
    result = {k: sum(r[k] for r in rows) for k in count_keys}
    n, eligible = result['valid_tiles'], result['gt_motion_eligible_tiles']
    result.update(frames=len(rows), frames_with_valid_GT_tiles=len(observed),
                  frames_without_valid_GT_tiles=len(rows) - len(observed),
                  frame_equal_component_L1=sum(r['component_L1'] for r in observed) / len(observed) if observed else None,
                  frame_equal_EPE=sum(r['EPE'] for r in observed) / len(observed) if observed else None,
                  pooled_component_L1=sum(r['sum_component_L1'] for r in rows) / n if n else None,
                  pooled_EPE=sum(r['sum_EPE'] for r in rows) / n if n else None,
                  direction_accuracy_GT_eligible=result['direction_correct'] / eligible if eligible else None,
                  frame_equal_direction_accuracy_GT_eligible=sum(r['direction_accuracy_GT_eligible'] for r in directional) / len(directional) if directional else None,
                  GT_motion_eligible_fraction_valid_tiles=eligible / n if n else None,
                  pred_motion_fraction_valid_tiles=result['pred_motion_tiles'] / n if n else None)
    return result


def evaluate_head(frames, stage, method, model):
    by_role = {'fit': [], 'dev': []}
    with torch.no_grad():
        for frame in frames:
            item = frame['stages'][stage]; x = item['descriptor'].float()
            prediction = model.reshape(1, 1, 1, 2).expand(*x.shape[:-1], 2) if method == 'constant_component_median' else descriptor_forward(model, method, x)
            row = frame_metrics(prediction, item['targets'])
            row.update(target_id=frame['target_id'], sequence=frame['sequence'])
            by_role[frame['role']].append(row)
    return {role: {'summary': aggregate_metrics(rows), 'frames': rows} for role, rows in by_role.items()}


def train_and_evaluate(cache, *, steps=STEPS, seed=SEED, strict_population=False):
    require(type(steps) is int and steps > 0, 'Positive fixed training steps required')
    torch.set_num_threads(2)
    membership = validate_cache(cache, strict_population=strict_population)
    result = {'schema': 'flow_coarse_head_fit_results_v1', 'status': 'COMPLETE', 'membership': membership,
              'flow_contract': cache['flow_contract'], 'recipe': {'seed': seed, 'steps_per_trainable_head': steps,
              'optimizer': 'AdamW', 'lr': .003, 'weight_decay': .0001, 'training_batch': 'all valid fit tiles',
              'loss': 'tile-equal component L1', 'normalization': 'all fit descriptors only; population std; folded first FC',
              'development_selection': False, 'device': 'cpu', 'threads': 2},
              'scope': 'backbone-seen train-side head-fit/dev exploration; not full-model AEE, test or ASIC benefit',
              'metric_definition': {'primary': 'mean of per-frame valid-tile metric; no-GT frames excluded and counted',
              'pooled': 'equal valid tiles across frames, not dense pixel-weighted flow',
              'direction': '8-bin direction_one_hop; GT tile-mean norm>=1px denominator; pred<1px is abstain and incorrect',
              'coverage_denominator': 'all valid-GT tiles', 'empty_value': None}, 'stages': {}}
    artifact = {'schema': 'flow_coarse_folded_head_weights_v1', 'flow_contract': cache['flow_contract'],
                'scope': result['scope'], 'recipe': result['recipe'], 'stages': {}}
    for stage in membership['stages']:
        normalizer = fit_normalizer(cache['frames'], stage)
        result['stages'][stage] = {'channels': membership['channels'][stage],
            'normalizer': {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in normalizer.items()}, 'methods': {}}
        artifact['stages'][stage] = {'channels': membership['channels'][stage], 'normalizer_audit_only': normalizer, 'methods': {}}
        for method in METHODS:
            started = time.monotonic()
            head, training = fit_head(cache['frames'], stage, method, cache['flow_contract'], normalizer, steps=steps, seed=seed)
            evaluation = evaluate_head(cache['frames'], stage, method, head)
            result['stages'][stage]['methods'][method] = {'training': training, 'evaluation': evaluation,
                                                         'CPU_wall_seconds': time.monotonic() - started}
            artifact['stages'][stage]['methods'][method] = {'constant_dxdy': head.detach().clone()} if method == 'constant_component_median' else {
                'state_dict': {k: v.detach().clone() for k, v in head.state_dict().items()},
                'input_contract': 'raw descriptor; normalization already folded; do NOT normalize again',
                'architecture': 'Linear(C,2)' if method == 'linear' else 'RuntimeCoarseFlowHead fc1/activation/fc2'}
    return result, artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--cache-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'Output directory already exists; refusing overwrite')
    cache, _ = load_verified_cache(args.cache, args.cache_sha256, strict_population=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        result, artifact = train_and_evaluate(cache, strict_population=True)
        provenance = {'cache_sha256': args.cache_sha256, 'cache_path': str(args.cache.resolve()),
                      'source_sha256': {p.name: sha_file(p) for p in (Path(__file__), Path(__file__).with_name('coarse_head.py'), Path(__file__).with_name('group_policy.py'))},
                      'torch': str(torch.__version__), 'python': sys.version.split()[0], 'GPU_used': False}
        result['provenance'] = provenance; artifact['provenance'] = provenance
        result['CPU_wall_seconds'] = time.monotonic() - started
        weights_path = args.output_dir / 'weights.pt'
        torch.save(artifact, weights_path)
        result['weights_file'] = 'weights.pt'; result['weights_sha256'] = sha_file(weights_path)
        (args.output_dir / 'results.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps({'status': result['status'], 'output_dir': str(args.output_dir),
                          'CPU_wall_seconds': result['CPU_wall_seconds'], 'GPU_used': False}))
    except BaseException as exc:
        failure = {'status': 'FAILED', 'cache_sha256': args.cache_sha256,
                   'error': type(exc).__name__ + ': ' + str(exc), 'GPU_used': False}
        (args.output_dir / 'failure.json').write_text(json.dumps(failure, indent=2, allow_nan=False) + '\n')
        raise


if __name__ == '__main__':
    main()
