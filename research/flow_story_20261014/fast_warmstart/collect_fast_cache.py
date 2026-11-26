#!/usr/bin/env python3
"""CPU fit20 native-MLP feature distillation cache; no GT, training or GPU.

All20 full native network forwards remain intact. Save full-T10 input/branch
at128 deterministic spatial points per stage0/2 firstMLP. The branch target
excludes the external residual; this cache is not a trained fast/gate method.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True

SEED = 'flow-fast-warmstart-spatial-v1-20261014:'
CHECKPOINT_SHA = 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
FIT_SHA = '29ec7ba9c9faca74c62c3e170c9d3c70855a4605a7eab31f9331c519f56f8915'
SHAPES = {0: (10, 1, 120, 160, 96), 2: (10, 1, 30, 40, 384)}
PINS = {
    'evaluate_breadth.py': '1cacbaeea4d9eaf19a304c3faaa13f5278f0212b6cdc92210c4619447d571628',
    'evaluate_breadth_extended.py': 'c49f2979c435242c641619cc4d8433e36dbb6c4e13ec5461ca8e8b78b3e2fdf0',
    'run_bn_scope.py': '2b5c61e3abedcc02ba09adadb432dfc0ff2cd2fcd2b7b3e3a2475c63d7a11840',
    'run_same_shape_reference.py': '6b6c2f93d24bac9fa1dd3a99fde0db05aa180d05d897482d13a028208340774b',
    'probe_actual_skip.py': '06672b84bd686d5cfd4a888fef7c2b6521b2d0c8f71361495447d1d59727bd57',
    'intervention_core.py': '6178520f2e841eebd39ac97b22d716944771c656d7a76ce26ed7a1c57d0d6dcb',
    'collect_coarse_cache.py': '533726df5e50f31b576a0d5972fe171fca0c55d0500b99a5a8514b6c0e143304',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4194304), b''):
            h.update(part)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); os.replace(temp, path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def sample_indices(target_id, stage):
    """Geometry/identity-only selection; sorted row-major IDs, never GT weighted."""
    require(isinstance(target_id, str) and target_id and stage in SHAPES, 'Invalid sampling identity/stage')
    _, _, h, w, _ = SHAPES[stage]
    ordered = sorted(range(h*w), key=lambda i: (hashlib.sha256(
        (SEED+target_id+':stage'+str(stage)+':'+str(i)).encode()).hexdigest(), i))
    return sorted(ordered[:128])


def sample_tensor(tensor, indices, expected_shape):
    """Gather spatial columns only; retain all T and channels in [T,N,C]."""
    import torch
    require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu'
            and tensor.dtype == torch.float32 and tuple(tensor.shape) == tuple(expected_shape), 'Native tensor shape/type differs')
    t, b, h, w, c = expected_shape
    require(t == 10 and b == 1 and len(indices) == 128 and indices == sorted(set(indices))
            and all(type(i) is int and 0 <= i < h*w for i in indices), 'Expected128 unique valid spatial indices and fullT10/B1')
    require(bool(torch.isfinite(tensor).all()), 'Nonfinite full native tensor')
    index = torch.tensor(indices, dtype=torch.long)
    out = tensor[:, 0].reshape(t, h*w, c).index_select(1, index).detach().contiguous()
    require(tuple(out.shape) == (10, 128, c) and not out.requires_grad, 'Sampled tensor shape/gradient differs')
    return out


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def fit_members(fit, canonical, roles, dev_rows):
    require(fit.get('schema') == 'flow_recovery_svs_fit20_v1' and fit.get('checkpoint_sha256') == CHECKPOINT_SHA
            and fit.get('checkpoint_optimizer_steps') == 135930, 'Fixed fit20/checkpoint manifest required')
    ids = fit['target_ids']; by_id = {r['target_id']: r for r in canonical}
    require(len(ids) == len(set(ids)) == 20 and digest(ids) == fit['target_ids_sha256'], 'Fit20 IDs/order digest differs')
    rows = [by_id[t] for t in ids]
    require(all(r['protocol_split'] == 'train' and roles.get(r['target_id']) == 'gate_fit' for r in rows), 'Only train gate-fit may enter feature cache')
    counts = {seq: sum(r['sequence'] == seq for r in rows) for seq in {r['sequence'] for r in rows}}
    require(len(counts) == 10 and set(counts.values()) == {2} and counts == fit['sequence_counts'], 'Fit20 sequence coverage differs')
    require(len(dev_rows) == 8 and len({r['target_id'] for r in dev_rows}) == 8
            and all(r['protocol_split'] == 'train' and roles.get(r['target_id']) == 'gate_development' for r in dev_rows)
            and not set(ids) & {r['target_id'] for r in dev_rows}
            and not set(counts) & {r['sequence'] for r in dev_rows}, 'Fit/development population overlaps or differs')
    return rows


def read_events(data_root, row, native, hashes):
    """Only event NPY is opened; GT/mask paths are not dereferenced."""
    import numpy as np
    paths = native.target_files(row); rel = paths['input']
    payload = native.checked_path(data_root, rel).read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    require(checksum == hashes[rel]['sha256'], 'Event NPY SHA differs from accepted native inventory')
    events = np.load(io.BytesIO(payload), allow_pickle=False)
    require(events.shape == (10, 480, 640) and events.dtype == np.float32
            and bool(np.isfinite(events).all()), 'Event shape/dtype/finite check failed')
    return events, paths, {'events': checksum}


def prepare(contract, contract_sha):
    require(contract.get('schema') == 'flow_fast_cache_contract_v1' and contract.get('CPU_only') is True
            and contract.get('stages') == [0, 2] and contract.get('fit_count') == 20
            and contract.get('samples_per_stage') == 128 and contract.get('sampling_seed') == SEED,
            'Unknown fast-cache contract')
    for path, expected in contract['source_pins'].items():
        require(sha(path) == expected, 'Source pin differs: '+path)
    require(contract['source_pins'].get(str(Path(__file__).resolve())) == sha(__file__), 'Collector is not explicitly pinned')
    breadth, motion = Path(contract['breadth_root']).resolve(), Path(contract['motion_root']).resolve()
    for name, expected in PINS.items():
        path = (motion if name in ('intervention_core.py', 'collect_coarse_cache.py') else breadth)/name
        require(contract['source_pins'].get(str(path)) == expected, 'Missing or foreign borrowed helper: '+name)
    sys.path.insert(0, str(breadth))
    import evaluate_breadth as base
    import evaluate_breadth_extended as extended
    import run_bn_scope as scope
    import run_same_shape_reference as loader
    for module in (base, extended, scope, loader):
        require(Path(module.__file__).resolve() == breadth/Path(module.__file__).name, 'Wrong helper import location')
    core = load_module('fast_cache_membership_core', motion/'intervention_core.py')
    timestamps = load_module('fast_cache_timestamp_helper', motion/'collect_coarse_cache.py')
    a = extended.parse_extended(contract['base_args'])
    require(a.device == 'cpu' and a.cpu_threads == 2 and not a.run_inference and a.geometry == 'full9'
            and a.actual_pair_count == 0 and 0 < a.wall_minutes <= 30 and a.bn_artifact is not None
            and a.svs_artifact is None and a.checkpoint_sha256 == CHECKPOINT_SHA, 'Only bounded fixed135930 CPU/MLP_CMA176 collection allowed')
    context = extended.extended_preflight(a)
    import torch
    require(not torch.cuda.is_initialized(), 'CPU preflight unexpectedly initialized CUDA')
    native, trainer, metrics, dev32, hashes, config, run, side, regions, plan = context
    require(side['completed_optimizer_steps'] == 135930, 'Unexpected teacher checkpoint step')
    require(contract['fit_manifest_sha256'] == FIT_SHA and sha(contract['fit_manifest']) == FIT_SHA, 'Fixed fit20 manifest SHA differs')
    fit = json.loads(Path(contract['fit_manifest']).read_text())
    canonical = metrics.canonical_rows(a.canonical); roles = metrics.auxiliary_roles(a.roles, canonical)
    dev_rows = core.choose_targets(dev32, 8)
    rows = fit_members(fit, canonical, roles, dev_rows)
    return a, context, rows, dev_rows, base, extended, scope, loader, timestamps


def collect(contract, contract_sha, prepared):
    import torch
    a, context, rows, dev_rows, base, extended, scope, loader, timestamps = prepared
    native, trainer, metrics, dev32, hashes, config, run, side, regions, plan = context
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    require(not torch.cuda.is_initialized(), 'CPU collection found initialized CUDA')
    output = Path(contract['output_directory']); require(not output.exists(), 'Output exists; refuse overwrite')
    output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    result = {'schema': 'flow_fast_cache_report_v1', 'status': 'RUNNING', 'contract_sha256': contract_sha,
        'contract': contract, 'native_preflight': plan, 'frames': [], 'GPU_initialized': False,
        'fit_GT_arrays_read': False, 'dev_arrays_read': False, 'final_valid_arrays_read': False,
        'training_performed': False, 'full_model_AEE_measured': False, 'hardware_claim': False}
    report_path = output/'cache_report.json'; status_path = output/'cache_status.json'
    try:
        acceptance = json.loads(a.acceptance.read_text()); inventory_path = a.acceptance.parent/'native_deployment_manifest.csv'
        require(sha(inventory_path) == acceptance['deployment_csv_sha256'], 'Native inventory CSV changed')
        with inventory_path.open(newline='') as stream: inventory = {r['relative_path']: r for r in csv.DictReader(stream)}
        model, functional = loader.load_model(a, context, base, torch, torch.device('cpu'))
        bn_record = scope.apply_scope(model, 'MLP_CMA176', a.bn_payload, torch)
        bn_record['execution'] = 'Complete native full-grid full-network forward; only input/output sampling hooks on stage0/2 first MLP'
        before_bn = base.bn_signature(model, torch); all_bn, _ = scope.bn_inventory(model, torch)
        before_non_bn = extended.non_bn_digest(model, all_bn)
        names = {s: 'sttmultires_unet.encoders.swin3d.layers.'+str(s)+'.swin_blocks.0.mlp' for s in SHAPES}
        modules = {s: model.get_submodule(name) for s, name in names.items()}
        require(all(type(m).__name__ == 'MS_Spiking_Mlp' for m in modules.values()), 'Native MLP types differ')
        ids = [r['target_id'] for r in rows]
        cache = {'schema': 'flow_fast_branch_cache_v1', 'status': 'COLLECTING', 'contract': contract,
            'contract_sha256': contract_sha, 'contract_canonical_sha256': digest(contract),
            'checkpoint_sha256': CHECKPOINT_SHA, 'checkpoint_optimizer_steps': 135930,
            'BN_scope': 'MLP_CMA176', 'BN_before': before_bn, 'source_pins': contract['source_pins'],
            'native_source_sha256': plan['source_sha256'], 'data_contract': config['data_contract'],
            'fit_manifest_sha256': FIT_SHA, 'fit_target_ids': ids, 'fit_target_ids_sha256': digest(ids),
            'dev_membership_metadata_only': [r['target_id'] for r in dev_rows], 'backbone_seen': True,
            'sampling_seed': SEED, 'sampling_rule': 'SHA256(seed+target_id+":stage"+str(stage)+":"+str(row_major_index)); first128, then sorted spatial IDs',
            'target_definition': 'Native MS_Spiking_Mlp output branch before external residual; no GT or final-flow labels',
            'fit_GT_arrays_read': False, 'dev_arrays_read': False, 'training_performed': False, 'GPU_initialized': False,
            'frames': []}
        result.update(BN_scope=bn_record, fit_target_ids=ids, fit_target_ids_sha256=digest(ids),
                      dev_membership_metadata_only=cache['dev_membership_metadata_only'])
        write(report_path, result)
        with torch.no_grad():
            for row in rows:
                require(time.monotonic()-started < a.wall_minutes*60, 'Bounded CPU collection wall budget exhausted')
                events, paths, evidence = read_events(a.data_root, row, native, hashes)
                timing = timestamps.timestamp_binding(a.data_root, row, paths, hashes, inventory, acceptance)
                chunk = base.prepare_input(events, torch.device('cpu'), torch)
                captured = {}; handles = []
                def pre(stage, value):
                    require(stage not in captured, 'Duplicate MLP input capture')
                    indices = sample_indices(row['target_id'], stage)
                    sampled = sample_tensor(value, indices, SHAPES[stage])
                    captured[stage] = {'u': sampled, 'spatial_indices': indices, 'indices_sha256': digest(indices),
                        'source_input_shape_TBHWC': list(value.shape), 'sampled_shape_TNC': list(sampled.shape),
                        'u_sha256': tensor_sha(sampled)}
                def post(stage, value):
                    require(stage in captured and 'target' not in captured[stage], 'Missing/duplicate MLP branch capture')
                    sampled = sample_tensor(value, captured[stage]['spatial_indices'], SHAPES[stage])
                    captured[stage].update(target=sampled, source_branch_shape_TBHWC=list(value.shape), target_sha256=tensor_sha(sampled))
                functional.reset_net(model)
                try:
                    for stage, module in modules.items():
                        handles.append(module.register_forward_pre_hook(lambda m, inp, s=stage: pre(s, inp[0])))
                        handles.append(module.register_forward_hook(lambda m, inp, out, s=stage: post(s, out)))
                    full = model(chunk)['flow'][-1]
                    require(tuple(full.shape) == (1, 2, 480, 640) and bool(torch.isfinite(full).all()), 'Native full forward failed')
                    require(set(captured) == {0, 2} and all('target' in x for x in captured.values()), 'Incomplete teacher captures')
                finally:
                    for handle in handles: handle.remove()
                    functional.reset_net(model)
                frame = {'target_id': row['target_id'], 'sequence': row['sequence'], 'role': 'fit',
                    'input_evidence': evidence, 'timestamps': timing,
                    'accepted_source_hashes': {k: {'relative_path': rel, **hashes[rel]} for k, rel in paths.items()},
                    'accepted_source_hashes_note': 'GT/mask hashes copied from acceptance metadata only; their arrays were not opened',
                    'stages': {str(s): captured[s] for s in SHAPES}}
                cache['frames'].append(frame)
                report_frame = {k: v for k, v in frame.items() if k != 'stages'}
                report_frame['stages'] = {str(s): {k: v for k, v in captured[s].items() if k not in ('u', 'target')} for s in SHAPES}
                result['frames'].append(report_frame)
                write(status_path, {'status': 'RUNNING', 'pid': os.getpid(), 'complete_frames': len(cache['frames']),
                    'expected_frames': 20, 'latest_target': row['target_id'], 'elapsed_seconds': time.monotonic()-started,
                    'GPU_initialized': False})
                print(json.dumps({'cached_fit_frames': len(cache['frames']), 'target': row['target_id']}), flush=True)
                del chunk, full, events, captured
        after_bn = base.bn_signature(model, torch)
        require(after_bn == before_bn and extended.non_bn_digest(model, all_bn) == before_non_bn, 'Teacher BN/weight state changed')
        require(not torch.cuda.is_initialized() and len(cache['frames']) == 20, 'Device/completion guard failed')
        cache.update(status='FAST_CACHE_COMPLETE', BN_after=after_bn, BN_unchanged=True,
                     non_bn_state_sha256=before_non_bn, non_bn_state_unchanged=True)
        path = output/'features.pt'
        with path.open('xb') as stream: torch.save(cache, stream)
        result.update(status='FAST_CACHE_COMPLETE', complete_frames=20, feature_cache_sha256=sha(path),
            feature_cache_bytes=path.stat().st_size, BN_before=before_bn, BN_after=after_bn, BN_unchanged=True,
            non_bn_state_unchanged=True, source_pins=contract['source_pins'], native_source_sha256=plan['source_sha256'],
            elapsed_seconds=time.monotonic()-started, expected_raw_tensor_bytes=98304000)
        write(report_path, result); write(status_path, {k: result[k] for k in ('status', 'complete_frames', 'feature_cache_sha256', 'BN_unchanged', 'GPU_initialized', 'elapsed_seconds')})
    except BaseException as exc:
        result.update(status='FAST_CACHE_FAILED', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic()-started)
        write(report_path, result); write(status_path, {'status': result['status'], 'error': result['error']})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--preflight-output', type=Path)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CUDA_VISIBLE_DEVICES="" required')
    require(sha(args.contract) == args.contract_sha256, 'Contract SHA differs')
    contract = json.loads(args.contract.read_text()); prepared = prepare(contract, args.contract_sha256)
    if not args.execute:
        require(args.preflight_output is not None and not args.preflight_output.exists(), 'New preflight output required')
        write(args.preflight_output, {'schema': 'flow_fast_cache_preflight_v1', 'status': 'FAST_CACHE_PREFLIGHT_PASS_NO_INFERENCE',
            'contract_sha256': args.contract_sha256, 'native_preflight': prepared[1][-1],
            'fit_target_ids': [r['target_id'] for r in prepared[2]], 'GPU_initialized': False,
            'data_array_reads_performed': False, 'note': 'Strict acceptance metadata, current file stat and sources checked; no network or event/GT arrays read.'})
        print('FAST_CACHE_PREFLIGHT_PASS_NO_INFERENCE'); return
    collect(contract, args.contract_sha256, prepared)


if __name__ == '__main__':
    main()
