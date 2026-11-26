#!/usr/bin/env python3
"""GPU engineering evidence only: dense-masked vs actual whole-tile MLP calls.

Uses a NEW engineering-subset checkpoint and a hash-bound canonical TRAIN input.
No accuracy, deployment calibration, speedup, or hardware-cycle claim.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import torch
from tile_mlp import tiled_mlp_model, pattern_mask


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): d.update(b)
    return d.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'source-manifest', 'data-root', 'manifest', 'reference-hashes',
                 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--target-id', required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Preserve existing output')
    a.source_root = a.source_root.resolve()
    sources = json.loads(a.source_manifest.read_text())
    for name, checksum in sources.items():
        if sha(a.source_root / name) != checksum: raise ValueError('Source changed: ' + name)
    if sha(a.manifest) != '6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22':
        raise ValueError('Canonical manifest changed')
    row = next(r for r in json.loads(a.manifest.read_text())['records'] if r['target_id'] == a.target_id)
    if row['protocol_split'] != 'train': raise ValueError('Engineering TRAIN only')
    with a.reference_hashes.open() as f:
        hashes = {r['relative_path']: r['sha256'] for r in csv.DictReader(f)}
    relative = 'saved_flow_data/event_tensors/10bins/left/' + row['sequence'] + '/' + row['v2_target_filename']
    input_path = a.data_root / relative
    if sha(input_path) != hashes[relative]: raise ValueError('Input hash mismatch')
    sidecar = json.loads(a.checkpoint.with_suffix('.json').read_text())
    if sha(a.checkpoint) != sidecar['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch')
    if sidecar['created_by'] != 'flow_a_20261012_native_baseline': raise ValueError('Not new checkpoint')
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['contract_sha256'] != hashlib.sha256(json.dumps(checkpoint['contract'], sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(): raise ValueError('Contract hash')
    provenance = checkpoint['provenance']
    if (provenance['training_scope'] != 'engineering_subset' or provenance['source_sha256'] != sources
            or provenance['old_checkpoint_loaded'] or not provenance['random_init']
            or provenance['window_size'] != [2, 9, 9]
            or checkpoint['training']['completed_optimizer_steps'] <= 0):
        raise ValueError('Engineering checkpoint provenance rejected')
    config = copy.deepcopy(checkpoint['effective_config'])
    config['swin_transformer']['input_size'] = [480, 640]
    config['loader']['crop'] = None
    config['runtime'] = {'snn_backend': 'torch', 'allow_tf32': False, 'cudnn_benchmark': False}
    state, training = checkpoint['model_state_dict'], checkpoint['training']
    del checkpoint
    sys.path.insert(0, str(a.source_root))
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from utils.runtime_backend import configure_snn_backend
    from spikingjelly.activation_based import functional
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0')
    os.environ['SDFORMER_SNN_BACKEND'] = 'torch'
    model = MS_SpikingformerFlowNet_en4(copy.deepcopy(config['model']), copy.deepcopy(config['swin_transformer']))
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    functional.set_step_mode(model, 'm'); configure_snn_backend(model, device, config, PSN)
    bn_before = {n: tuple(sha_tensor(getattr(m, x)) for x in ('running_mean', 'running_var', 'num_batches_tracked'))
                 for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
    events = np.load(input_path, allow_pickle=False)
    if events.shape != (10, 480, 640) or events.dtype != np.float32 or not np.isfinite(events).all():
        raise ValueError('Unexpected input array')
    chunk = torch.from_numpy(events).unsqueeze(0).to(device)
    chunk = torch.cat((torch.relu(chunk).unsqueeze(2), torch.relu(-chunk).unsqueeze(2)), dim=2)
    nz = chunk[chunk != 0]
    if not nz.numel(): raise ValueError('Empty input')
    lo, hi = nz.min(), nz.max()
    if lo != hi: chunk[chunk != 0] = (chunk[chunk != 0]-lo)/(hi-lo)
    result = {'schema': 'flow_a_whole_model_tile_engineering_v1', 'status': 'RUNNING',
        'scope': 'One TRAIN input, new 36-step engineering weights, frozen P-code BN; no AEE/calibration claim',
        'checkpoint_sha256': sidecar['checkpoint_sha256'], 'training': training,
        'source_manifest_sha256': sha(a.source_manifest), 'input_sha256': sha(input_path),
        'target_id': a.target_id, 'geometry': [480,640], 'window': [2,9,9],
        'BN_policy': 'native tracked running frozen; not calibrated deployment D',
        'precision': 'fp32, TF32 disabled', 'torch': str(torch.__version__),
        'device': torch.cuda.get_device_name(device), 'script_sha256': sha(__file__),
        'tile_module_sha256': sha(Path(__file__).with_name('tile_mlp.py')), 'cases': []}
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            functional.reset_net(model)
            reference = model(chunk)['flow'][-1].detach().clone()
            for pattern in ('all_keep', 'all_skip', 'checkerboard', 'last_edge_only'):
                pair = {}; branches = {}; inputs = {}; record = {'pattern': pattern, 'arms': {}}
                for mode in ('dense_masked', 'actual_skip'):
                    functional.reset_net(model)
                    def policy(name, x): return pattern_mask(pattern, x.shape[2], x.shape[3], device=x.device)
                    start = time.perf_counter()
                    with tiled_mlp_model(model, mode, policy) as wrappers:
                        branches[mode] = {}
                        inputs[mode] = {}
                        handles = []
                        for wrapper in wrappers:
                            def capture(layer, args, output, name=wrapper.layer_name):
                                branches[mode][name] = output.detach().cpu().clone()
                                inputs[mode][name] = args[0].detach().cpu().clone()
                            handles.append(wrapper.register_forward_hook(capture))
                        try: prediction = model(chunk)['flow'][-1].detach().clone()
                        finally:
                            for handle in handles: handle.remove()
                        torch.cuda.synchronize()
                        if not torch.isfinite(prediction).all(): raise RuntimeError('Nonfinite flow')
                        pair[mode] = prediction
                        record['arms'][mode] = {'wall_seconds_instrumented_no_speed_claim': time.perf_counter()-start,
                            'layers': [w.records[-1] for w in wrappers]}
                diff = (pair['actual_skip']-pair['dense_masked']).abs()
                record['final_flow_pair_difference'] = {'max_abs': diff.max().item(), 'mean_abs': diff.mean().item(),
                    'different_values': int(torch.count_nonzero(diff).item()),
                    'allclose_atol1e_5_rtol1e_5': bool(torch.allclose(pair['actual_skip'], pair['dense_masked'], atol=1e-5, rtol=1e-5))}
                record['per_layer_branch_comparisons'] = {}
                for name, dense_branch in branches['dense_masked'].items():
                    actual_branch = branches['actual_skip'][name]
                    delta = (actual_branch-dense_branch).abs()
                    input_delta = (inputs['actual_skip'][name]-inputs['dense_masked'][name]).abs()
                    record['per_layer_branch_comparisons'][name] = {
                        'max_abs_difference': delta.max().item(), 'different_values': int(torch.count_nonzero(delta)),
                        'mean_abs_difference': delta.mean().item(),
                        'allclose_atol1e_5_rtol1e_5': bool(torch.allclose(actual_branch, dense_branch, atol=1e-5, rtol=1e-5)),
                        'input_u_max_abs_difference': input_delta.max().item(),
                        'input_u_different_values': int(torch.count_nonzero(input_delta)),
                        'input_u_allclose_atol1e_5_rtol1e_5': bool(torch.allclose(inputs['actual_skip'][name], inputs['dense_masked'][name], atol=1e-5, rtol=1e-5)),
                        'finite': bool(torch.isfinite(actual_branch).all() and torch.isfinite(dense_branch).all()
                                       and torch.isfinite(input_delta).all()),
                        'retained_branch_nonzero_values': int(torch.count_nonzero(actual_branch)),
                        'retained_branch_max_abs': actual_branch.abs().max().item()}
                record['masked_vs_original_final_flow_max_abs'] = (pair['actual_skip']-reference).abs().max().item()
                if pattern == 'all_keep':
                    record['dense_original_max_abs'] = (pair['actual_skip']-reference).abs().max().item()
                    if record['dense_original_max_abs'] != 0: raise RuntimeError('All-keep changed original')
                if pattern == 'all_skip':
                    for layer in record['arms']['actual_skip']['layers']:
                        if layer['observed']['fc1_rows'] or layer['observed']['fc2_rows']:
                            raise RuntimeError('All-skip executed producer FC')
                result['cases'].append(record)
                print(json.dumps({'pattern': pattern, 'difference': record['final_flow_pair_difference']}), flush=True)
                if not record['final_flow_pair_difference']['allclose_atol1e_5_rtol1e_5'] or not all(
                        r['allclose_atol1e_5_rtol1e_5'] and r['input_u_allclose_atol1e_5_rtol1e_5'] and r['finite'] for r in record['per_layer_branch_comparisons'].values()):
                    raise RuntimeError('Numerical equivalence guard failed; see saved per-layer/final differences')
                if pattern == 'all_skip' and any(r['retained_branch_nonzero_values'] for r in record['per_layer_branch_comparisons'].values()):
                    raise RuntimeError('All-skip branch not zero')
        bn_after = {n: tuple(sha_tensor(getattr(m, x)) for x in ('running_mean', 'running_var', 'num_batches_tracked'))
                    for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
        result['BN_buffers_unchanged'] = bn_after == bn_before
        if not result['BN_buffers_unchanged']: raise RuntimeError('BN buffers mutated')
        result['status'] = 'ENGINEERING_PROBE_COMPLETE'
    except BaseException as exc:
        result['status'] = 'FAILED'
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        result['wall_seconds'] = time.perf_counter()-started
        result['max_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        result['max_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


def sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


if __name__ == '__main__': main()
