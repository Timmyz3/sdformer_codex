#!/usr/bin/env python3
"""Independent full-network native v2 evaluator for NEW traced checkpoints only.

No training, source edits, old checkpoints, implicit geometry conversion, or AEE
on random weights. Default checks the plan without importing torch. Explicit
--run-inference evaluates both P_paper and P_code with dropout disabled.
"""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
sys.dont_write_bytecode = True
from flow_eval_contract import (CANONICAL_SHA, AUX_ROLES_SHA, canonical_rows, auxiliary_roles,
    select_rows, paths_for, load_arrays, validate_region_contract, partitions, crop_slices,
    MetricAccumulator, sha, ids_sha)

SCHEMA = 'flow_a_native_v2_checkpoint_v1'
CREATOR = 'flow_a_20261012_native_baseline'
SOURCE_MANIFEST_SHA = '8007d0fa80db95aaff3ff93d34c386a0d6e396df9fd5ae5ecef8a2b1e407594e'


def parse_args():
    old = Path(__file__).resolve().parents[2] / 'flow_story_20261011/preflight/data_protocol/artifacts'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, default=old.parents[2] / 'server_setup_20261011/native_source_sha256.json')
    p.add_argument('--reference-hashes', type=Path, help='Source/accepted inventory CSV; required for final valid')
    p.add_argument('--reference-hashes-sha256', help='Explicit frozen SHA of the inventory CSV')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--checkpoint-sidecar', type=Path, help='Default: checkpoint with .json suffix')
    p.add_argument('--manifest', type=Path, default=old / 'common_targets.json')
    p.add_argument('--roles', type=Path, default=old / 'train_auxiliary_roles.csv')
    p.add_argument('--regions', type=Path, required=True)
    p.add_argument('--regions-sha256', required=True)
    p.add_argument('--scope', required=True, choices=['valid_all', 'train_gate_dev', 'train_gate_fit'])
    p.add_argument('--target-id', action='append', default=None, help='Train-side scope only; final valid always all2152')
    p.add_argument('--geometry', required=True, choices=['crop_center9', 'full9', 'full15'])
    p.add_argument('--device', default='cuda:0', help='Explicit cpu or cuda:N; no automatic fallback')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--run-inference', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.run_inference and a.output is None:
        p.error('--run-inference requires --output')
    if a.output and a.output.exists():
        p.error('Output exists; preserve evidence and select a new path')
    if not 1 <= a.cpu_threads <= 16:
        p.error('cpu-threads must be 1..16')
    a.source_root, a.data_root, a.checkpoint = a.source_root.resolve(), a.data_root.resolve(), a.checkpoint.resolve()
    a.checkpoint_sidecar = a.checkpoint_sidecar or a.checkpoint.with_suffix('.json')
    return a


def load_plan(a):
    records = canonical_rows(a.manifest); roles = auxiliary_roles(a.roles, records)
    rows = select_rows(records, roles, a.scope, a.target_id)
    if sha(a.source_manifest) != SOURCE_MANIFEST_SHA:
        raise ValueError('Unexpected native source-manifest version')
    a.pinned_source_sha256 = json.loads(a.source_manifest.read_text())
    for relative, digest in a.pinned_source_sha256.items():
        if sha(a.source_root / relative) != digest:
            raise ValueError('Current model source differs from pinned snapshot: ' + relative)
    if a.scope == 'valid_all' and a.reference_hashes is None:
        raise ValueError('Final valid requires a source/accepted hash inventory, not existence checks alone')
    a.reference_inventory = None
    if a.reference_hashes is not None:
        if not a.reference_hashes_sha256 or sha(a.reference_hashes) != a.reference_hashes_sha256:
            raise ValueError('Explicit source/accepted hash inventory SHA mismatch')
        with a.reference_hashes.open(newline='') as stream:
            inventory = list(csv.DictReader(stream))
        a.reference_inventory = {row['relative_path']: row['sha256'] for row in inventory}
        if len(inventory) != len(a.reference_inventory):
            raise ValueError('Duplicate hash-inventory entries')
        for row in rows:
            for path in paths_for(a.data_root, row).values():
                relative = 'saved_flow_data/' + str(path.relative_to(a.data_root))
                if relative not in a.reference_inventory:
                    raise ValueError('Selected target missing from source hash inventory: ' + relative)
    if sha(a.checkpoint) != a.checkpoint_sha256:
        raise ValueError('Explicit checkpoint SHA mismatch')
    sidecar = json.loads(a.checkpoint_sidecar.read_text())
    if sidecar.get('checkpoint_sha256') != a.checkpoint_sha256:
        raise ValueError('Checkpoint sidecar SHA mismatch')
    if sidecar.get('schema_version', sidecar.get('schema')) != SCHEMA:
        raise ValueError('Checkpoint sidecar schema mismatch')
    if sidecar.get('created_by') != CREATOR or sidecar.get('checkpoint_file') != a.checkpoint.name:
        raise ValueError('Checkpoint sidecar creator/filename mismatch')
    a.canonical_training_ids_sha256 = hashlib.sha256(json.dumps(
        [r['target_id'] for r in records if r['protocol_split'] == 'train'],
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if sha(a.regions) != a.regions_sha256:
        raise ValueError('Explicit region-artifact SHA mismatch')
    regions = validate_region_contract(json.loads(a.regions.read_text()), records, roles,
                                       require_complete_fit=a.scope == 'valid_all')
    missing = [str(path) for row in rows for path in paths_for(a.data_root, row).values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('%d selected arrays missing; do not silently evaluate partial split. First: %s' % (len(missing), missing[0]))
    return rows, regions, sidecar, {
        'schema': 'flow_a_native_v2_evaluation_v1', 'status': 'CPU_PLAN_VALIDATED_NO_INFERENCE',
        'inference_executed': False, 'source_root': str(a.source_root),
        'checkpoint': {'path': str(a.checkpoint), 'sha256': a.checkpoint_sha256,
                       'sidecar': str(a.checkpoint_sidecar), 'sidecar_sha256': sha(a.checkpoint_sidecar)},
        'canonical_manifest_sha256': CANONICAL_SHA, 'auxiliary_roles_sha256': AUX_ROLES_SHA,
        'source_manifest_sha256': SOURCE_MANIFEST_SHA,
        'input_hash_binding': {'mode': 'source_inventory_checked_on_exact_loaded_bytes' if a.reference_inventory else 'exploratory_loaded_bytes_hash_record_only',
                               'inventory_sha256': a.reference_hashes_sha256},
        'regions': {'path': str(a.regions), 'sha256': a.regions_sha256, 'contract': regions},
        'population': a.scope, 'target_count': len(rows), 'target_ids': [r['target_id'] for r in rows],
        'target_ids_sha256': ids_sha([r['target_id'] for r in rows]),
        'population_label': ('All 2152 fixed local validation targets; not official hidden DSEC test' if a.scope == 'valid_all'
                             else 'Explicit TRAIN-side auxiliary population; not held-out backbone accuracy or final validation'),
        'geometry': a.geometry,
        'geometry_label': {'crop_center9': 'Deterministic center crop288x384/W9; separate from full-frame metrics',
                           'full9': 'Full480x640/W9 deployment geometry; not paper full15 fine-tuning',
                           'full15': 'Full480x640/W15; requires checkpoint with native W15 weights, no implicit interpolation'}[a.geometry],
        'BN_arms': ['P_paper', 'P_code'], 'evaluation_precision': 'fp32_tf32_disabled',
        'metrics': 'all valid GT, final native flow output, per-frame mean then frame average; pixel weighted / sequence / GT-region diagnostics separate',
        'dropout': 'eval disabled for both arms, unlike legacy B1 training-validation model.train()',
        'training_performed': False, 'checkpoint_saved': False, 'source_edited': False,
        'script_sha256': sha(__file__), 'metric_contract_module_sha256': sha(Path(__file__).with_name('flow_eval_contract.py'))}


def validate_checkpoint_metadata(checkpoint, sidecar, a):
    if checkpoint.get('schema_version') != SCHEMA:
        raise ValueError('Only new explicitly traced baseline checkpoint schema accepted')
    provenance = checkpoint['provenance']
    if (provenance.get('created_by') != CREATOR or provenance.get('random_init') is not True
            or provenance.get('old_checkpoint_loaded') is not False
            or provenance.get('training_protocol_split') != 'train'
            or provenance.get('canonical_manifest_sha256') != CANONICAL_SHA
            or provenance.get('BN_policy') != 'native_train_tracked'):
        raise ValueError('New-checkpoint initialization/training/member/BN provenance rejected')
    if checkpoint['training']['completed_optimizer_steps'] <= 0:
        raise ValueError('Random initialization without training is not an accuracy baseline')
    if provenance.get('training_scope') not in ('canonical6000', 'engineering_subset'):
        raise ValueError('Unknown training scope')
    if (sidecar.get('training_scope') != provenance['training_scope'] or
            sidecar.get('completed_optimizer_steps') != checkpoint['training']['completed_optimizer_steps']):
        raise ValueError('Checkpoint/sidecar scope or training-step mismatch')
    if (provenance['training_scope'] == 'canonical6000' and
            provenance.get('training_target_ids_sha256') != a.canonical_training_ids_sha256):
        raise ValueError('Canonical-training target list SHA mismatch')
    if a.scope == 'valid_all' and provenance['training_scope'] != 'canonical6000':
        raise ValueError('Engineering-subset weights are train-side diagnostics only, not formal valid baseline')
    if not checkpoint.get('contract_sha256') or checkpoint['contract_sha256'] != sidecar.get('contract_sha256'):
        raise ValueError('Checkpoint/sidecar training-contract SHA mismatch')
    # Recompute using the baseline producer's canonical JSON convention.
    contract_sha = hashlib.sha256(json.dumps(checkpoint['contract'], sort_keys=True, separators=(',', ':'),
                                             allow_nan=False).encode()).hexdigest()
    if contract_sha != checkpoint['contract_sha256']:
        raise ValueError('Embedded training contract digest mismatch')
    for key in ('source_sha256', 'canonical_manifest_sha256', 'training_target_ids_sha256'):
        if checkpoint['contract'].get(key) != provenance.get(key):
            raise ValueError('Training contract/provenance mismatch: ' + key)
    desired_window = [2, 15, 15] if a.geometry == 'full15' else [2, 9, 9]
    if provenance.get('window_size') != desired_window:
        raise ValueError('Checkpoint/evaluation window mismatch; no implicit 9-to-15 interpolation or reverse conversion')
    source_hashes = provenance.get('source_sha256', {})
    if source_hashes != a.pinned_source_sha256 or checkpoint['contract'].get('source_manifest_sha256') != SOURCE_MANIFEST_SHA:
        raise ValueError('Checkpoint source hashes differ from pinned 63-file native snapshot')
    required = {'models/STSwinNet_SNN/Spiking_STSwinNet.py',
                'models/STSwinNet_SNN/Spiking_swin_transformer3D.py',
                'models/STSwinNet_SNN/Spiking_modules.py', 'models/STSwinNet_SNN/Spiking_submodules.py',
                'configs/parser.py', 'utils/runtime_backend.py'}
    if not required <= set(source_hashes):
        raise ValueError('Checkpoint source provenance lacks required native model/parser/backend files')
    for name, expected in source_hashes.items():
        path = (a.source_root / name).resolve()
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_relative_to(a.source_root):
            raise ValueError('Unsafe source provenance path')
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Checkpoint/source mismatch: ' + name)
    config = copy.deepcopy(checkpoint['effective_config'])
    normalized = copy.deepcopy(config)
    normalized['data']['path'] = '<verified_DSEC>/saved_flow_data'
    normalized['loader']['gpu'] = '<logical-device>'
    if normalized != checkpoint['contract'].get('effective_config'):
        raise ValueError('Effective config differs from embedded training contract')
    for observed, wanted, label in (
            (config['model']['name'], 'MS_SpikingformerFlowNet_en4', 'model'),
            (config['model']['encoding'], 'voxel', 'encoding'),
            (config['model']['norm_input'], 'minmax', 'input normalization'),
            (config['data']['num_frames'], 10, 'T'), (config['data']['num_chunks'], 1, 'chunks'),
            (config['loader']['batch_size'], 1, 'batch'), (config['loader']['polarity'], True, 'signed voxel'),
            (config['metrics']['flow_scaling'], 1, 'flow scaling'),
            (config['swin_transformer']['window_size'], desired_window, 'window')):
        if observed != wanted:
            raise ValueError('Unsupported checkpoint contract for ' + label)
    return config, provenance


def apply_bn_policy(model, policy):
    import torch
    if policy not in ('P_paper', 'P_code'):
        raise ValueError('Unknown BN policy')
    model.eval()
    bn_count = 0
    for layer in model.modules():
        if layer.training:
            raise RuntimeError('Evaluation must disable all training/dropout/drop-path modules')
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            bn_count += 1
            if policy == 'P_paper':
                layer.track_running_stats = False
                layer.running_mean = layer.running_var = layer.num_batches_tracked = None
            elif (not layer.track_running_stats or layer.running_mean is None or layer.running_var is None
                  or layer.num_batches_tracked is None or layer.num_batches_tracked.item() <= 0
                  or not torch.isfinite(layer.running_mean).all().item()
                  or not torch.isfinite(layer.running_var).all().item()
                  or (layer.running_var < 0).any().item()):
                raise ValueError('P_code requires finite populated native running stats for every BN')
    if bn_count == 0:
        raise RuntimeError('Native BN module discovery unexpectedly empty')
    return bn_count


def run_inference(a, rows, regions, sidecar, result):
    import torch
    import torchvision
    # Strict safe load: never retry weights_only=False or allow arbitrary globals.
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    config, provenance = validate_checkpoint_metadata(checkpoint, sidecar, a)
    state = checkpoint['model_state_dict']
    if not state or any(not isinstance(tensor, torch.Tensor) or
                        (tensor.is_floating_point() and not torch.isfinite(tensor).all().item())
                        for tensor in state.values()):
        raise ValueError('State dict must contain finite tensors only')
    result['training_provenance'] = provenance
    result['training_progress'] = checkpoint['training']
    result['scientific_accuracy_claim_eligible'] = False
    result['claim_gate'] = 'This evaluator verifies computation/provenance, not baseline convergence, native-event-encoding equivalence, or paper protocol. These require separate evidence before using metrics as baseline/BN rankings or killing A.'
    result['baseline_maturity'] = ('Engineering-subset connectivity diagnostics; cannot rank BN quality, kill A, or support baseline accuracy/reproduction' if
                                   provenance['training_scope'] == 'engineering_subset' else
                                   'Canonical-membership self-training; adequacy/convergence and native-paper replication still require separate evidence')
    # Free optimizer/RNG payloads immediately; retain only model weights/config/provenance.
    del checkpoint
    sys.path.insert(0, str(a.source_root))
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from utils.runtime_backend import configure_snn_backend
    from spikingjelly.activation_based import functional
    for item in (MS_SpikingformerFlowNet_en4, PSN, configure_snn_backend):
        if not Path(inspect.getfile(item)).resolve().is_relative_to(a.source_root):
            raise RuntimeError('Native import outside verified source root')
    torch.set_num_threads(a.cpu_threads)
    device = torch.device(a.device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise RuntimeError('Requested device unavailable; no fallback')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    os.environ['SDFORMER_SNN_BACKEND'] = 'torch'
    config['loader']['crop'] = [288, 384] if a.geometry == 'crop_center9' else None
    config['swin_transformer']['input_size'] = [288, 384] if a.geometry == 'crop_center9' else [480, 640]
    config['runtime'] = {'snn_backend': 'torch', 'allow_tf32': False, 'cudnn_benchmark': False}
    result['effective_evaluation_config'] = config
    result['runtime'] = {'python': sys.version, 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                         'cuda_runtime': torch.version.cuda, 'device': str(device),
                         'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                         'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                         'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'spikingjelly', 'timm', 'einops']}}
    ys, xs = crop_slices(a.geometry)
    result['inference_executed'] = True
    result['arms'] = {}
    data_evidence = {}

    def bn_signature(model):
        signature = {}
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                signature[name] = {'training': layer.training, 'tracking': layer.track_running_stats,
                    'buffers': {key: None if getattr(layer, key) is None else hashlib.sha256(
                        getattr(layer, key).detach().cpu().numpy().tobytes()).hexdigest()
                        for key in ('running_mean', 'running_var', 'num_batches_tracked')}}
        return signature

    for policy in ('P_paper', 'P_code'):
        model = MS_SpikingformerFlowNet_en4(copy.deepcopy(config['model']), copy.deepcopy(config['swin_transformer']))
        model.load_state_dict(state, strict=True)
        model.to(device); model.eval()
        functional.set_step_mode(model, 'm'); configure_snn_backend(model, device, config, PSN)
        bn_count = apply_bn_policy(model, policy)
        before = bn_signature(model)
        metrics = MetricAccumulator(); started = time.perf_counter()
        entry = {'status': 'RUNNING', 'BN_layer_count': bn_count, 'frames': metrics.frames}
        result['arms'][policy] = entry
        try:
            with torch.inference_mode():
                for i, row in enumerate(rows):
                    paths = paths_for(a.data_root, row)
                    expected_hashes = None if a.reference_inventory is None else {
                        key: a.reference_inventory['saved_flow_data/' + str(path.relative_to(a.data_root))]
                        for key, path in paths.items()}
                    evidence = {}
                    arrays, paths = load_arrays(a.data_root, row, evidence=evidence, expected_hashes=expected_hashes)
                    # Compute GT region masks in full coordinates, then crop; crop borders never create B.
                    masks = {name: mask[ys, xs] for name, mask in partitions(arrays['gt'], arrays['mask'], arrays['events'], regions).items()}
                    gt = arrays['gt'][:, ys, xs]
                    events = np.ascontiguousarray(arrays['events'][:, ys, xs], dtype=np.float32)
                    chunk = torch.from_numpy(events).unsqueeze(0).to(device)
                    neg, pos = torch.nn.functional.relu(-chunk), torch.nn.functional.relu(chunk)
                    chunk = torch.cat((pos.unsqueeze(2), neg.unsqueeze(2)), dim=2)
                    nonzero = chunk[chunk != 0]
                    if not nonzero.numel():
                        raise ValueError('Native nonzero minmax undefined on empty events; no silent replacement')
                    minimum, maximum = nonzero.min(), nonzero.max()
                    if not minimum == maximum:
                        chunk[chunk != 0] = (chunk[chunk != 0] - minimum) / (maximum - minimum)
                    functional.reset_net(model); functional.set_step_mode(model, 'm')
                    outputs = model(chunk)['flow']
                    if not outputs or tuple(outputs[-1].shape) != (1, 2) + gt.shape[1:]:
                        raise ValueError('Unexpected native final-flow shape')
                    prediction = outputs[-1][0].detach().cpu().numpy()
                    metrics.update(row['target_id'], row['sequence'], prediction, gt, masks)
                    if row['target_id'] in data_evidence and data_evidence[row['target_id']] != evidence:
                        raise RuntimeError('Input file changed between BN arms')
                    data_evidence[row['target_id']] = evidence
                    if (i + 1) % 25 == 0:
                        print(json.dumps({'policy': policy, 'completed': i + 1, 'of': len(rows)}), flush=True)
            if before != bn_signature(model):
                raise RuntimeError('BN buffers/policy changed during evaluation')
            if len(metrics.frames) != len(rows):
                raise RuntimeError('Evaluation did not cover every selected target')
            entry['status'] = 'PASS'
        finally:
            entry['wall_seconds_including_data_and_metrics'] = time.perf_counter() - started
            entry['summary'] = metrics.summary()
            entry['BN_buffers_unchanged'] = before == bn_signature(model)
            result['input_file_sha256'] = data_evidence
            if entry['status'] != 'PASS':
                entry['status'] = 'FAILED_PARTIAL_NOT_COMPLETE_METRIC'
        functional.reset_net(model)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    result['status'] = 'DUAL_BN_EVALUATION_COMPLETE'
    result['paired_P_code_minus_P_paper_frame_AEE'] = (
        result['arms']['P_code']['summary']['regions']['all_valid']['frame_mean_AEE'] -
        result['arms']['P_paper']['summary']['regions']['all_valid']['frame_mean_AEE'])


def main():
    a = parse_args()
    rows, regions, sidecar, result = load_plan(a)
    if not a.run_inference:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return
    a.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_inference(a, rows, regions, sidecar, result)
    except BaseException as exc:
        result['status'] = 'FAILED_NO_COMPLETE_ACCURACY_CONCLUSION'
        result['exception'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
