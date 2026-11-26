#!/usr/bin/env python3
"""Pinned CPU-only native-prefix descriptor cache; no backbone optimization.

Ends the forward at stage2's first MLP input. This offline truncation is not
an executable compressed optical-flow network and establishes no speedup.
"""
import argparse
import csv
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
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def timestamp_binding(root, row, paths, hashes, inventory, acceptance):
    """Verify receipt and independently reread the accepted timestamp row."""
    rel = paths['input']
    source = inventory[rel]
    receipt_path = root / '.native_encoding_receipts' / (row['target_id'] + '.json')
    require(sha(receipt_path) == source['generation_receipt_sha256'], 'Receipt SHA differs')
    receipt = json.loads(receipt_path.read_text())
    require(receipt['schema_version'] == 'flow_native_input_receipt_v1' and
            receipt['target_id'] == row['target_id'] and receipt['relative_path'] == rel and
            receipt['sha256'] == hashes[rel]['sha256'] == source['sha256'] and
            receipt['contract_sha256'] == acceptance['encoding_contract_sha256'] and
            receipt['timestamp_index_1based'] == row['target_index_1based'], 'Receipt binding differs')
    ts_rel = 'train_optical_flow/' + row['sequence'] + '/flow/forward_timestamps.txt'
    ts_path = root / ts_rel
    require(inventory[ts_rel]['category'] == 'timestamps' and
            sha(ts_path) == inventory[ts_rel]['sha256'] == receipt['source_hashes'][ts_rel]['sha256'],
            'Timestamp source differs')
    lines = ts_path.read_text().splitlines()
    pair = [int(v.strip()) for v in lines[row['target_index_1based'] - 1].split(',')]
    require(len(pair) == 2 and pair == receipt['timestamp_pair_us'] and pair[1] > pair[0],
            'Timestamp values differ')
    return {'timestamp_pair_us': pair, 'interval_us': pair[1] - pair[0],
            'timestamp_index_1based': row['target_index_1based'],
            'timestamp_source_sha256': inventory[ts_rel]['sha256'],
            'generation_receipt_sha256': source['generation_receipt_sha256'],
            'per_target_interval_verified': True}


class FeaturesReady(Exception):
    """Only expected early stop, raised after both hook captures complete."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--contract-sha256', required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CUDA_VISIBLE_DEVICES="" required')
    require(sha(args.contract) == args.contract_sha256, 'Contract SHA differs')
    contract = json.loads(args.contract.read_text())
    require(contract['schema'] == 'flow_coarse_cache_contract_v1' and
            contract['CPU_only'] is True and contract['stages'] == [0, 2], 'Unknown cache contract')
    for path, expected in contract['source_pins'].items():
        require(sha(path) == expected, 'Pinned source changed: ' + path)
    here = Path(__file__).resolve().parent
    require(all(str(here / name) in contract['source_pins'] for name in
                ('collect_coarse_cache.py', 'coarse_head.py', 'intervention_core.py', 'group_policy.py')),
            'Collector/helper source not pinned')
    sys.path.insert(0, contract['breadth_root'])
    import evaluate_breadth as base
    import evaluate_breadth_extended as extended
    import run_bn_scope as scope
    import run_same_shape_reference as loader
    import intervention_core as core
    import coarse_head
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261013)
    require(not torch.cuda.is_initialized(), 'CUDA already initialized')
    a = extended.parse_extended(contract['base_args'])
    require(a.device == 'cpu' and a.cpu_threads == 2 and not a.run_inference and
            a.geometry == 'full9' and a.actual_pair_count == 0 and a.wall_minutes <= 30,
            'CPU pilot scope differs')
    output = Path(contract['output_directory'])
    require(not output.exists(), 'Refuse to overwrite output directory')
    output.mkdir(parents=True)
    result = {'schema': 'flow_coarse_cache_report_v1', 'status': 'PREFLIGHT',
              'contract_sha256': args.contract_sha256, 'contract': contract,
              'CPU_only': True, 'backbone_optimized': False, 'full_model_AEE_measured': False,
              'hardware_claim': False, 'frames': []}
    status_path = output / 'cache_status.json'
    started = time.monotonic()
    write(status_path, {'status': 'PREFLIGHT', 'pid': os.getpid(), 'GPU_initialized': False})
    model = None
    before_bn = None
    try:
        context = extended.extended_preflight(a)
        native, trainer, metrics, dev32, hashes, config, saved_run, side, regions, plan = context
        fit_path = Path(contract['fit_manifest'])
        require(sha(fit_path) == contract['fit_manifest_sha256'], 'Fit membership SHA differs')
        fit = json.loads(fit_path.read_text())
        records = metrics.canonical_rows(a.canonical)
        roles = metrics.auxiliary_roles(a.roles, records)
        by_id = {r['target_id']: r for r in records}
        fit_rows = [by_id[t] for t in fit['target_ids']]
        dev_rows = core.choose_targets(dev32, 8)
        require(len(fit_rows) == 20 and len(set(fit['target_ids'])) == 20 and
                fit['checkpoint_sha256'] == a.checkpoint_sha256 and
                all(roles[r['target_id']] == 'gate_fit' for r in fit_rows) and
                all(roles[r['target_id']] == 'gate_development' for r in dev_rows) and
                not (set(fit['target_ids']) & {r['target_id'] for r in dev_rows}), 'Auxiliary roles differ')
        chosen = [('fit', r) for r in fit_rows] + [('dev', r) for r in dev_rows]
        require(all(r['protocol_split'] == 'train' for _, r in chosen), 'Final-valid prohibited')
        acceptance = json.loads(a.acceptance.read_text())
        inventory_path = a.acceptance.parent / 'native_deployment_manifest.csv'
        require(sha(inventory_path) == acceptance['deployment_csv_sha256'], 'Native manifest CSV differs')
        with inventory_path.open(newline='') as stream:
            inventory = {r['relative_path']: r for r in csv.DictReader(stream)}
        model, functional = loader.load_model(a, context, base, torch, torch.device('cpu'))
        bn_record = scope.apply_scope(model, 'MLP_CMA176', a.bn_payload, torch)
        bn_record['execution'] = 'CPU full-shape prefix; terminate at stage2 first MLP input; no full flow result'
        before_bn = base.bn_signature(model, torch)
        modules = dict(model.named_modules())
        names = {s: 'sttmultires_unet.encoders.swin3d.layers.' + str(s) + '.swin_blocks.0.mlp'
                 for s in (0, 2)}
        shapes = {0: (10, 1, 120, 160, 96), 2: (10, 1, 30, 40, 384)}
        strides = {0: 4, 2: 16}
        heads = {s: coarse_head.RuntimeCoarseFlowHead(shapes[s][-1], flow_contract=contract['flow_contract'])
                 for s in (0, 2)}
        cache = {'schema': 'flow_coarse_feature_cache_v1', 'flow_contract': contract['flow_contract'],
                 'contract_sha256': args.contract_sha256, 'checkpoint_sha256': a.checkpoint_sha256,
                 'BN_scope': 'MLP_CMA176', 'label_geometry': 'logical_cell_support_origin0_v1',
                 'backbone_seen': True, 'frames': []}
        result.update(status='RUNNING', native_preflight=plan, BN_scope=bn_record,
                      label_geometry=cache['label_geometry'], device='cpu', torch=str(torch.__version__),
                      target_ids_by_role={'fit': [r['target_id'] for r in fit_rows],
                                          'dev': [r['target_id'] for r in dev_rows]})
        with torch.no_grad():
            for role, row in chosen:
                require(time.monotonic() - started < a.wall_minutes * 60, 'CPU cache wall bound exhausted')
                require(not torch.cuda.is_initialized(), 'CUDA initialized during CPU collection')
                evidence = {}
                paths = native.target_files(row)
                expected = {'events' if k == 'input' else k: hashes[p]['sha256'] for k, p in paths.items()}
                arrays, _ = metrics.load_arrays(a.data_root / 'saved_flow_data', row,
                                                evidence=evidence, expected_hashes=expected)
                timing = timestamp_binding(a.data_root, row, paths, hashes, inventory, acceptance)
                chunk = base.prepare_input(arrays['events'], torch.device('cpu'), torch)
                descriptors = {}
                handles = []
                def capture(stage, u):
                    require(stage not in descriptors and tuple(u.shape) == shapes[stage], 'Capture shape/count differs')
                    descriptor = heads[stage].aggregate(u).detach().cpu().contiguous()
                    require(bool(torch.isfinite(descriptor).all()), 'Nonfinite descriptor')
                    descriptors[stage] = descriptor
                    if stage == 2:
                        require(set(descriptors) == {0, 2}, 'Missing stage0 before early stop')
                        raise FeaturesReady()
                functional.reset_net(model)
                try:
                    for s, name in names.items():
                        require(type(modules[name]).__name__ == 'MS_Spiking_Mlp', 'Unexpected native module')
                        handles.append(modules[name].register_forward_pre_hook(
                            lambda module, inp, stage=s: capture(stage, inp[0])))
                    try:
                        model(chunk)
                    except FeaturesReady:
                        pass
                    else:
                        raise ValueError('Forward did not stop at declared prefix')
                finally:
                    for handle in handles:
                        handle.remove()
                    functional.reset_net(model)
                frame = {'target_id': row['target_id'], 'sequence': row['sequence'], 'role': role,
                         'timestamps': timing, 'input_evidence': evidence, 'stages': {}}
                report_frame = {k: v for k, v in frame.items() if k != 'stages'}
                report_frame['stages'] = {}
                for s in (0, 2):
                    targets = coarse_head.make_tile_targets(torch.from_numpy(arrays['gt']),
                        torch.from_numpy(arrays['mask']), feature_hw=shapes[s][2:4],
                        feature_stride_pixels=(strides[s], strides[s]),
                        flow_contract=contract['flow_contract'], image_origin_xy=(0., 0.))
                    require(int(targets['valid_pixel_count'].sum()) == int(arrays['mask'].sum()) and
                            int(targets['support_pixel_count'].sum()) == 480 * 640,
                            'Logical tile partition does not cover complete image/valid GT')
                    frame['stages'][str(s)] = {'descriptor': descriptors[s], 'targets': targets}
                    report_frame['stages'][str(s)] = {'input_shape_TBHWC': list(shapes[s]),
                        'descriptor_shape': list(descriptors[s].shape),
                        'valid_tiles': int(targets['valid'].sum()),
                        'empty_tiles': int((~targets['valid']).sum()),
                        'valid_pixels': int(targets['valid_pixel_count'].sum()),
                        'support_pixels': int(targets['support_pixel_count'].sum())}
                cache['frames'].append(frame)
                result['frames'].append(report_frame)
                write(status_path, {'status': 'RUNNING', 'pid': os.getpid(), 'complete_frames': len(cache['frames']),
                      'expected_frames': 28, 'latest_target': row['target_id'],
                      'elapsed_seconds': time.monotonic() - started, 'GPU_initialized': False})
                print(json.dumps({'cached': len(cache['frames']), 'target': row['target_id'], 'role': role}), flush=True)
        require(base.bn_signature(model, torch) == before_bn, 'BN state changed during prefix collection')
        require(not torch.cuda.is_initialized(), 'CUDA initialized')
        cache_path = output / 'features.pt'
        with cache_path.open('xb') as stream:
            torch.save(cache, stream)
        result.update(status='COARSE_CACHE_COMPLETE', complete_frames=len(cache['frames']),
                      feature_cache_sha256=sha(cache_path), feature_cache_bytes=cache_path.stat().st_size,
                      BN_unchanged=True, GPU_initialized=False, elapsed_seconds=time.monotonic()-started)
        write(output / 'cache_report.json', result)
        write(status_path, {k: result[k] for k in ('status', 'complete_frames', 'feature_cache_sha256',
                                                'BN_unchanged', 'GPU_initialized', 'elapsed_seconds')})
    except BaseException as exc:
        result.update(status='COARSE_CACHE_FAILED', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
        write(output / 'cache_report.json', result)
        write(status_path, {'status': result['status'], 'error': result['error']})
        raise


if __name__ == '__main__':
    main()
