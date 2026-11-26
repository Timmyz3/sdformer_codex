#!/usr/bin/env python3
"""Fixed CPU warm-start of the borrowed DynamicSwin LN + Linear fast branch.

Usage: python fit_fast_cache.py --cache features.pt --cache-sha256 SHA \
  --output-dir NEW_DIR [--adapter-dir /path/to/transfer_experiments/dynamicswin]
No backbone, GT, development data or GPU is loaded. Branch fit MSE is not AEE.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


SEED = 20261014
STEPS = 500
BATCH_ROWS = 1024
SAMPLE_PREFIX = 'flow-fast-warmstart-spatial-v1-20261014:'
CHANNELS = {'0': 96, '2': 384}
SHAPES = {'0': [10, 1, 120, 160, 96], '2': [10, 1, 30, 40, 384]}
ADAPTER_SHA = '75c97bb1e35e31a4e7b0fb97adefe434e6346dfa156ee6e9c297547efdbe30d7'
PROVENANCE_SHA = '4971687125cd6df959d2314e71e3b44c8a4cbb5514694fb10f0bf126fe3a29c1'
CHECKPOINT_SHA = 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
FIT_MANIFEST_SHA = '29ec7ba9c9faca74c62c3e170c9d3c70855a4605a7eab31f9331c519f56f8915'
FIT_IDS_SHA = '99c2409ca95b8b96903a9449767435df59f6e9a845c95c9b35eb97c250be1011'
RECIPE = {'seed': SEED, 'steps': STEPS, 'batch_rows': BATCH_ROWS,
          'optimizer': 'AdamW', 'lr': 0.001, 'weight_decay': 0.0001,
          'betas': [0.9, 0.999], 'eps': 1e-8, 'precision': 'fp32',
          'cpu_threads': 2, 'minibatch_sampling': 'uniform rows with replacement',
          'loss': 'mean squared error over rows and channels',
          'extra_input_normalization': False, 'dev_model_selection': False,
          'stages_trained_independently': True}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_fast_class(adapter_dir=None):
    """Return the pinned existing FastBranch class; never create a new topology."""
    if adapter_dir is None:
        adapter_dir = Path(__file__).resolve().parents[2] / 'flow_story_20261012/transfer_experiments/dynamicswin'
    directory = Path(adapter_dir)
    require(sha_file(directory / 'adapter.py') == ADAPTER_SHA, 'Borrowed adapter SHA mismatch')
    require(sha_file(directory / 'source_provenance.json') == PROVENANCE_SHA,
            'Borrowed source provenance SHA mismatch')
    name = '_flow_fast_warmstart_pinned_adapter'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, directory / 'adapter.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].FastBranch


def expected_indices(target_id, stage):
    shape = SHAPES[stage]
    ranked = sorted(range(shape[2] * shape[3]), key=lambda index: (
        hashlib.sha256(f'{SAMPLE_PREFIX}{target_id}:stage{stage}:{index}'.encode()).hexdigest(), index))
    return sorted(ranked[:128])


def validate_cache(cache):
    require(isinstance(cache, dict) and cache.get('schema') == 'flow_fast_branch_cache_v1', 'Foreign cache schema')
    require(cache.get('status') == 'FAST_CACHE_COMPLETE', 'Incomplete cache')
    require(cache.get('checkpoint_sha256') == CHECKPOINT_SHA, 'Wrong teacher checkpoint')
    require(cache.get('BN_scope') == 'MLP_CMA176' and cache.get('backbone_seen') is True,
            'Wrong teacher BN/exposure contract')
    contract = cache.get('contract')
    # Preserve the collector's exact-file SHA separately from canonical JSON SHA.
    # The caller-provided complete-cache SHA binds that collector receipt.
    require(isinstance(contract, dict) and valid_sha(cache.get('contract_sha256')) and
            cache.get('contract_canonical_sha256') == digest(contract), 'Contract SHA mismatch')
    require(cache.get('fit_manifest_sha256') == FIT_MANIFEST_SHA and
            contract.get('fit_manifest_sha256') == FIT_MANIFEST_SHA, 'Wrong fit20 membership manifest')
    pins = cache.get('source_pins')
    require(isinstance(pins, dict) and pins and all(isinstance(k, str) and valid_sha(v) for k, v in pins.items()),
            'Source pins required')
    frames = cache.get('frames')
    require(isinstance(frames, list) and len(frames) == 20, 'Exactly fixed fit20 is required')
    ids = [frame.get('target_id') for frame in frames]
    require(ids == cache.get('fit_target_ids') and digest(ids) == FIT_IDS_SHA and
            cache.get('fit_target_ids_sha256') == FIT_IDS_SHA, 'Fixed ordered fit20 identity mismatch')
    sequences, audit = Counter(), []
    for frame in frames:
        target_id, sequence = frame['target_id'], frame.get('sequence')
        require(frame.get('role') == 'fit' and isinstance(sequence, str) and target_id.startswith(sequence + '_'),
                'Only fit targets with explicit sequence are accepted')
        sequences[sequence] += 1
        require(set(frame.get('stages', {})) == {'0', '2'}, 'Both native stages are required')
        require(valid_sha(frame.get('input_evidence', {}).get('events')), 'Event provenance missing')
        stage_audit = {}
        for stage, item in frame['stages'].items():
            expected_shape = [10, 128, CHANNELS[stage]]
            require(item.get('source_input_shape_TBHWC') == SHAPES[stage] and
                    item.get('source_branch_shape_TBHWC') == SHAPES[stage] and
                    item.get('sampled_shape_TNC') == expected_shape, 'Wrong native/sample shape')
            indices = item.get('spatial_indices')
            require(isinstance(indices, list) and all(type(v) is int for v in indices) and
                    indices == expected_indices(target_id, stage) and
                    item.get('indices_sha256') == digest(indices), 'Spatial sampling/indices SHA mismatch')
            for key in ('u', 'target'):
                value = item.get(key)
                require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
                        value.dtype == torch.float32 and list(value.shape) == expected_shape and
                        not value.requires_grad and bool(torch.isfinite(value).all()),
                        'Finite detached CPU float32 [10,128,C] required: ' + key)
                require(item.get(key + '_sha256') == tensor_sha(value), 'Tensor SHA mismatch: ' + key)
            stage_audit[stage] = {'spatial_points': len(indices), 'rows': 1280,
                                  'indices_sha256': item['indices_sha256'],
                                  'u_sha256': item['u_sha256'], 'target_sha256': item['target_sha256']}
        audit.append({'target_id': target_id, 'sequence': sequence, 'role': 'fit',
                      'input_evidence': frame['input_evidence'], 'stages': stage_audit})
    require(len(sequences) == 10 and set(sequences.values()) == {2}, 'Expected ten fit sequences with two targets each')
    return {'target_ids': ids, 'target_ids_sha256': FIT_IDS_SHA, 'sequence_counts': dict(sorted(sequences.items())),
            'frames': audit, 'rows_per_target_per_stage': 1280, 'rows_per_stage': 25600,
            'sampling_uses_GT': False, 'role': 'fit_only', 'backbone_seen': True,
            'fit_manifest_BN_policy_used_for_membership_only': True}


def load_verified_cache(path, expected_sha):
    require(valid_sha(expected_sha) and sha_file(path) == expected_sha, 'Cache SHA256 mismatch')
    cache = torch.load(path, map_location='cpu', weights_only=True)
    return cache, validate_cache(cache)


def stage_rows(cache, stage):
    # cat owns fresh storage: neither teacher nor input cache is modified by fitting.
    return tuple(torch.cat([f['stages'][stage][key].reshape(-1, CHANNELS[stage])
                            for f in cache['frames']], dim=0) for key in ('u', 'target'))


def evaluate_mse(model_or_constant, x, y, rows_per_target, target_ids):
    require(x.shape == y.shape and x.shape[0] == rows_per_target * len(target_ids), 'MSE member accounting mismatch')
    per_target = []
    with torch.no_grad():
        for i, target_id in enumerate(target_ids):
            xx, yy = x[i * rows_per_target:(i + 1) * rows_per_target], y[i * rows_per_target:(i + 1) * rows_per_target]
            prediction = model_or_constant(xx) if callable(model_or_constant) else model_or_constant.expand_as(yy)
            require(prediction.shape == yy.shape and bool(torch.isfinite(prediction).all()), 'Nonfinite/wrong shape prediction')
            mse = float((prediction.double() - yy.double()).square().mean())
            per_target.append({'target_id': target_id, 'rows': rows_per_target, 'MSE': mse})
    return {'frame_equal_MSE': sum(v['MSE'] for v in per_target) / len(per_target),
            'pooled_MSE': sum(v['MSE'] for v in per_target) / len(per_target),
            'equal_rows_per_target': True, 'per_target': per_target}


def fit_stage(x, y, target_ids, rows_per_target, fast_class, *, steps=STEPS, batch_rows=BATCH_ROWS, seed=SEED):
    """Pure CPU fitting; non-default steps/shapes are for synthetic tests only."""
    torch.set_num_threads(2)
    require(x.ndim == 2 and x.shape == y.shape and x.shape[0] == len(target_ids) * rows_per_target and
            x.shape[1] > 0 and steps > 0 and batch_rows > 0, 'Invalid row shapes/recipe')
    require(all(v.device.type == 'cpu' and v.dtype == torch.float32 and not v.requires_grad and
                bool(torch.isfinite(v).all()) for v in (x, y)), 'Finite detached FP32 CPU rows required')
    input_before = (tensor_sha(x), tensor_sha(y))
    torch.manual_seed(seed)
    model = fast_class(x.shape[1]).cpu().float()
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.eval()
    constant = y.double().mean(0).float()
    metrics = {'zero_branch_identity_residual': evaluate_mse(torch.zeros(x.shape[1]), x, y, rows_per_target, target_ids),
               'fit_mean_constant_branch': evaluate_mse(constant, x, y, rows_per_target, target_ids),
               'initialized_fast': evaluate_mse(model, x, y, rows_per_target, target_ids)}
    generator = torch.Generator(device='cpu').manual_seed(seed)
    draws = torch.zeros(x.shape[0], dtype=torch.int64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001,
                                  betas=(0.9, 0.999), eps=1e-8)
    trace, loss_sum = [], 0.0
    model.train()
    start = time.monotonic()
    for step in range(1, steps + 1):
        indices = torch.randint(x.shape[0], (batch_rows,), generator=generator)
        draws.index_add_(0, indices, torch.ones_like(indices))
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x[indices]) - y[indices]).square().mean()
        require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                'Missing/nonfinite fast gradient')
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite fast state')
        loss_sum += float(loss.detach())
        if step % 100 == 0 or step == steps:
            count = 100 if step % 100 == 0 else step % 100
            trace.append({'optimizer_steps': step, 'block_steps': count,
                          'mean_pre_update_minibatch_MSE': loss_sum / count,
                          'last_pre_update_minibatch_MSE': float(loss.detach())})
            loss_sum = 0.0
    wall_seconds = time.monotonic() - start
    model.eval()
    metrics['trained_fast'] = evaluate_mse(model, x, y, rows_per_target, target_ids)
    require(input_before == (tensor_sha(x), tensor_sha(y)), 'Input/teacher rows changed')
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {'state_dict': state, 'initial_state_dict': initial_state, 'mean_branch': constant}, {
        'channels': x.shape[1], 'optimized_steps': steps, 'fit_rows': x.shape[0],
        'parameters': sum(p.numel() for p in model.parameters()), 'metrics': metrics,
        'training_loss_every_100_steps': trace, 'optimization_wall_seconds': wall_seconds,
        'minibatch_row_draws': int(draws.sum()), 'unique_rows_drawn': int((draws > 0).sum()),
        'draws_per_target': [{'target_id': target, 'draws': int(draws[i * rows_per_target:(i + 1) * rows_per_target].sum())}
                             for i, target in enumerate(target_ids)],
        'input_and_teacher_rows_unchanged': True, 'all_states_and_gradients_finite': True}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def run(cache_path, cache_sha, output_dir, adapter_dir=None):
    destination = Path(output_dir)
    require(not destination.exists(), 'Output directory exists; refuse overwrite')
    cache, members = load_verified_cache(cache_path, cache_sha)
    fast_class = load_fast_class(adapter_dir)
    destination.mkdir(parents=True, exist_ok=False)
    provenance = {'cache_sha256': cache_sha, 'cache_contract_sha256': cache['contract_sha256'],
                  'cache_contract_canonical_sha256': cache['contract_canonical_sha256'],
                  'checkpoint_sha256': cache['checkpoint_sha256'], 'BN_scope': cache['BN_scope'],
                  'source_pins': cache['source_pins'], 'adapter_sha256': ADAPTER_SHA,
                  'adapter_source_provenance_sha256': PROVENANCE_SHA,
                  'fit_target_ids_sha256': FIT_IDS_SHA, 'fit_manifest_sha256': FIT_MANIFEST_SHA,
                  'fit_script_sha256': sha_file(__file__), 'backbone_seen': True}
    result = {'schema': 'flow_fast_warmstart_results_v1', 'status': 'RUNNING',
              'recipe': RECIPE, 'provenance': provenance, 'members': members, 'stages': {},
              'torch': str(torch.__version__), 'device': 'cpu', 'GPU_used': False,
              'GT_used': False, 'dev_used': False, 'task_AEE_evaluated': False,
              'complete_DynamicSwin_recovery': False, 'ASIC_benefit_claim': False,
              'scope': 'fit-only native branch reconstruction warm-start; no residual/backbone/gate training'}
    write_json(destination / 'results.json', result)
    weights = {'schema': 'flow_fast_warmstart_weights_v1', 'recipe': RECIPE,
               'provenance': provenance, 'stages': {}}
    try:
        for stage in ('0', '2'):
            x, y = stage_rows(cache, stage)
            weights['stages'][stage], result['stages'][stage] = fit_stage(
                x, y, members['target_ids'], 1280, fast_class)
            del x, y
            write_json(destination / 'results.json', result)
        temp = destination / 'weights.pt.tmp'
        torch.save(weights, temp)
        os.replace(temp, destination / 'weights.pt')
        result.update(status='FAST_WARMSTART_COMPLETE', weights_file='weights.pt',
                      weights_sha256=sha_file(destination / 'weights.pt'))
        write_json(destination / 'results.json', result)
    except Exception as exc:
        result.update(status='FAILED', error_type=type(exc).__name__, error=str(exc))
        write_json(destination / 'results.json', result)
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--cache-sha256', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--adapter-dir', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run(args.cache, args.cache_sha256, args.output_dir, args.adapter_dir)
    print(json.dumps({'status': result['status'], 'output_dir': str(args.output_dir),
                      'weights_sha256': result['weights_sha256'],
                      'trained_fit_MSE': {s: v['metrics']['trained_fast']['frame_equal_MSE']
                                          for s, v in result['stages'].items()}}, allow_nan=False))


if __name__ == '__main__':
    main()
