#!/usr/bin/env python3
"""CPU-only upstream-faithful v2 voxels into a NEW versioned dataset root.

Default: inspect a bounded plan, no arrays generated. --run creates <=32 selected
members; --full-canonical explicitly allows all8152, never changes the split.
Existing inputs require a matching receipt and current SHA; never overwrite.
GT/mask are managed independently by the migration owner, not rewritten here.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import csv
import fcntl
import hashlib
import inspect
import io
import json
from pathlib import Path
import resource
import sys
import time
sys.dont_write_bytecode = True

CANONICAL_SHA = '6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22'
ENCODING_ID = 'sdformerflow_upstream_voxel_10bins_cpu_v1'
SOURCE_PINS = {
    'DSEC_dataloader/event_representations.py': 'f02493e7440e12828b487b272d0c17f96a1a2a6c785fd8a6764981b09825403a',
    'DSEC_dataloader/DSEC_dataset_preprocess.py': '63fa4eaf500910d798e2d81fc17de85a0d4b6cd52f0ae6ade4d841be16256294',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def path_under(root, relative):
    part = Path(relative)
    if part.is_absolute() or '..' in part.parts:
        raise ValueError('Unsafe relative path')
    path = root / part
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes dataset root')
    return path


def load_hashes(path):
    rows = {}
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            rel = row['relative_path']
            if rel in rows or len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256']):
                raise ValueError('Duplicate/invalid source hash inventory row')
            rows[rel] = {'bytes': int(row['bytes']), 'sha256': row['sha256']}
    return rows


def native_voxel(events, rectify_map, representations, np, torch):
    """Exact native preprocessing order, except rejecting its undefined inputs."""
    if events is None or len(events['t']) < 2 or events['t'][-1] == events['t'][0]:
        raise ValueError('Empty/degenerate event chunk: native time normalization is undefined')
    xy = representations.rectify_events(events['x'], events['y'], rectify_map)
    t = (events['t'] - events['t'][0]).astype('float32')
    t = t / t[-1]
    event_tensors = {'p': torch.from_numpy(events['p'].astype('float32')),
                     't': torch.from_numpy(t), 'x': torch.from_numpy(xy[:, 0]),
                     'y': torch.from_numpy(xy[:, 1])}
    # No whole-event in-image filter, no floor/clamp repair, no voxel z-score.
    voxel = representations.VoxelGrid((10, 480, 640)).convert_CHW(event_tensors).numpy()
    if voxel.shape != (10, 480, 640) or voxel.dtype != np.float32 or not np.isfinite(voxel).all():
        raise ValueError('Invalid generated native voxel')
    return voxel


def sequence_shards(records, count):
    """Greedy whole-sequence allocation by canonical target count; deterministic."""
    counts = {}
    for row in records:
        counts[row['sequence']] = counts.get(row['sequence'], 0) + 1
    loads, mapping = [0] * count, {}
    for sequence in sorted(counts, key=lambda s: (-counts[s], s)):
        shard = min(range(count), key=lambda i: (loads[i], i))
        mapping[sequence] = shard; loads[shard] += counts[sequence]
    return mapping, loads


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True, help='Pinned native SDformerFlow source')
    p.add_argument('--raw-root', type=Path, required=True, help='Root containing train_events/train_optical_flow')
    p.add_argument('--output-root', type=Path, required=True, help='NEW DSEC_native_v1 root; never the old DSEC root')
    p.add_argument('--canonical', type=Path, required=True)
    p.add_argument('--raw-hashes', type=Path, required=True, help='Trusted transferred source/destination SHA inventory')
    p.add_argument('--target-list', type=Path, help='Bounded probe: one predeclared canonical ID per line')
    p.add_argument('--full-canonical', action='store_true')
    p.add_argument('--shard-count', type=int, default=1, help='Full canonical only:1..4 disjoint whole-sequence workers')
    p.add_argument('--shard-index', type=int, default=0, help='Zero-based worker index, separate --report per worker')
    p.add_argument('--run', action='store_true')
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve())
    if a.full_canonical == bool(a.target_list):
        p.error('Choose exactly --target-list (<=32) or explicit --full-canonical')
    if not 1 <= a.shard_count <= 4 or not 0 <= a.shard_index < a.shard_count:
        p.error('Need1<=shard-count<=4 and0<=shard-index<shard-count')
    if not a.full_canonical and (a.shard_count != 1 or a.shard_index != 0):
        p.error('Shards apply only to the complete canonical plan, not bounded probes')
    if not a.output_root.name.startswith('DSEC_native_'):
        p.error('Output root must be a NEW versioned DSEC_native_* root, not old DSEC')
    return a


def main():
    a = parse_args(); started = time.monotonic()
    if sha(a.canonical) != CANONICAL_SHA:
        raise ValueError('Canonical manifest pin mismatch')
    canonical = json.loads(a.canonical.read_text())['records']
    if len(canonical) != 8152 or sum(r['protocol_split'] == 'train' for r in canonical) != 6000 or \
            sum(r['protocol_split'] == 'valid' for r in canonical) != 2152:
        raise ValueError('Canonical split changed')
    shard_map, shard_loads = sequence_shards(canonical, a.shard_count)
    if a.full_canonical:
        selected = [r for r in canonical if shard_map[r['sequence']] == a.shard_index]
    else:
        ids = [line.strip() for line in a.target_list.read_text().splitlines() if line.strip()]
        by_id = {r['target_id']: r for r in canonical}
        if not 1 <= len(ids) <= 32 or len(set(ids)) != len(ids) or any(t not in by_id for t in ids):
            raise ValueError('Bad bounded target list')
        selected = [by_id[t] for t in ids]
    selected = sorted(selected, key=lambda r: (r['sequence'], r['target_index_1based']))
    for rel, expected in SOURCE_PINS.items():
        if sha(path_under(a.source_root, rel)) != expected:
            raise ValueError('Native preprocessing source pin changed: ' + rel)
    hashes = load_hashes(a.raw_hashes)
    required = {}
    for seq in sorted({r['sequence'] for r in selected}):
        for rel in ('train_events/' + seq + '/events/left/events.h5',
                    'train_events/' + seq + '/events/left/rectify_map.h5',
                    'train_optical_flow/' + seq + '/flow/forward_timestamps.txt'):
            if rel not in hashes:
                raise ValueError('Missing source-file SHA: ' + rel)
            path = path_under(a.raw_root, rel)
            if not path.is_file() or path.stat().st_size != hashes[rel]['bytes']:
                raise ValueError('Missing/wrong-size raw source: ' + rel)
            required[rel] = hashes[rel]
    result = {'encoding_id': ENCODING_ID, 'status': 'PLAN_ONLY', 'GPU_initialized': False,
              'canonical_manifest_sha256': CANONICAL_SHA, 'target_count': len(selected),
              'target_ids': [r['target_id'] for r in selected], 'source_pins': SOURCE_PINS,
              'raw_files': required, 'raw_hash_inventory_sha256': sha(a.raw_hashes),
              'script_sha256': sha(__file__), 'input_NPY_expected_bytes_each': 12288128,
              'output_root': str(a.output_root), 'GT_mask_modified': False,
              'shard': {'index': a.shard_index, 'count': a.shard_count,
                        'whole_sequence_assignment': shard_map, 'target_loads': shard_loads,
                        'global_canonical_targets': 8152}}
    if not a.run:
        atomic_json(a.report, result); print(json.dumps(result, indent=2)); return
    # Bound current content to trusted full-file SHA, even if transfer receipts already verified it.
    for rel, expected in required.items():
        if sha(path_under(a.raw_root, rel)) != expected['sha256']:
            raise ValueError('Current raw content differs from transferred SHA: ' + rel)
    result['raw_fullhash_wall_seconds'] = time.monotonic() - started
    import numpy as np
    import torch
    import h5py
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    sys.path.insert(0, str(a.source_root))
    from DSEC_dataloader import event_representations as rep
    if Path(inspect.getfile(rep)).resolve() != a.source_root / 'DSEC_dataloader/event_representations.py':
        raise ValueError('Wrong imported native event representations')
    contract = {'encoding_id': ENCODING_ID, 'canonical_manifest_sha256': CANONICAL_SHA,
                'source_pins': SOURCE_PINS, 'script_sha256': sha(__file__),
                'torch': str(torch.__version__), 'numpy': np.__version__, 'h5py': h5py.__version__,
                'CPU_threads': 1, 'bins': 10, 'shape': [10, 480, 640], 'dtype': 'float32',
                'time': '(t-t[0]).astype(float32); t/t[-1]; native convert_CHW renormalization',
                'spatial': 'all sliced events; native rectify and tap-wise validity',
                'polarity': 'signed 2p-1; no post-voxel z-score', 'slice': '[t_beg,t_end)'}
    contract_sha = digest(contract)
    result.update(contract=contract, contract_sha256=contract_sha, status='RUNNING', records=[])
    receipts = a.output_root / '.native_encoding_receipts'
    receipts.mkdir(parents=True, exist_ok=True)
    # Every worker holds its whole sequence locks. Different shard topologies or
    # a concurrent probe cannot silently write the same sequence.
    locks = []
    for seq in sorted({r['sequence'] for r in selected}):
        lock = (receipts / ('.sequence_' + seq + '.lock')).open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    try:
        for seq in sorted({r['sequence'] for r in selected}):
            source_rel = 'train_events/' + seq + '/events/left/'
            timestamp_rel = 'train_optical_flow/' + seq + '/flow/forward_timestamps.txt'
            rows = [r for r in selected if r['sequence'] == seq]
            source_hashes = {rel: required[rel] for rel in (source_rel + 'events.h5', source_rel + 'rectify_map.h5', timestamp_rel)}
            stamps = np.loadtxt(path_under(a.raw_root, timestamp_rel), dtype=np.int64, delimiter=',')
            with h5py.File(path_under(a.raw_root, source_rel + 'events.h5'), 'r') as h5, \
                    h5py.File(path_under(a.raw_root, source_rel + 'rectify_map.h5'), 'r') as rh5:
                slicer = rep.EventSlicer(h5); slicer.t_offset = np.int64(slicer.t_offset)
                rectify = rh5['rectify_map'][()]
                for row in rows:
                    per_started = time.monotonic(); target = row['target_id']
                    pair = stamps[row['target_index_1based'] - 1]
                    rel = 'saved_flow_data/event_tensors/10bins/left/' + seq + '/' + row['v2_target_filename']
                    out = path_under(a.output_root, rel); receipt_path = receipts / (target + '.json')
                    identity = {'target_id': target, 'protocol_split': row['protocol_split'],
                                'relative_path': rel, 'timestamp_index_1based': row['target_index_1based'],
                                'timestamp_pair_us': pair.tolist(), 'source_hashes': source_hashes,
                                'contract_sha256': contract_sha, 'schema_version': 'flow_native_input_receipt_v1'}
                    if out.exists() or receipt_path.exists():
                        if not out.is_file() or not receipt_path.is_file():
                            raise ValueError('Orphan/unreceipted existing output; manual audit required: ' + target)
                        receipt = json.loads(receipt_path.read_text())
                        if any(receipt.get(k) != v for k, v in identity.items()) or \
                                out.stat().st_size != receipt['bytes'] or sha(out) != receipt['sha256']:
                            raise ValueError('Existing native output differs; never overwrite: ' + target)
                        result['records'].append(dict(receipt, reused=True)); continue
                    events = slicer.get_events(int(pair[0]), int(pair[1]))
                    voxel = native_voxel(events, rectify, rep, np, torch)
                    raw_slice_sha = {k: {'dtype': str(events[k].dtype), 'count': len(events[k]),
                                        'sha256_C_order_bytes': hashlib.sha256(events[k].tobytes(order='C')).hexdigest()}
                                     for k in ('p', 'x', 'y', 't')}
                    out.parent.mkdir(parents=True, exist_ok=True)
                    temp = out.with_name(out.name + '.tmp')
                    with temp.open('xb') as f:
                        np.save(f, voxel, allow_pickle=False); f.flush(); os.fsync(f.fileno())
                    file_sha = sha(temp)
                    # link is atomic and refuses to replace an existing output.
                    os.link(temp, out); temp.unlink()
                    receipt = dict(identity, bytes=out.stat().st_size, sha256=file_sha,
                                   shape=list(voxel.shape), dtype=str(voxel.dtype), finite=True,
                                   tensor_C_order_bytes_sha256=hashlib.sha256(voxel.tobytes(order='C')).hexdigest(),
                                   raw_slice_sha=raw_slice_sha,
                                   generation_seconds_including_slice_write_hash=time.monotonic() - per_started,
                                   process_peak_RSS_KiB_cumulative=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    atomic_json(receipt_path, receipt)
                    result['records'].append(dict(receipt, reused=False))
                    print(json.dumps({'completed': len(result['records']), 'target_id': target,
                                      'generation_seconds': receipt['generation_seconds_including_slice_write_hash']}), flush=True)
        result['status'] = ('FULL_CANONICAL_COMPLETE' if a.shard_count == 1 else 'SHARD_COMPLETE') if a.full_canonical else 'BOUNDED_SUBSET_COMPLETE'
        result['GPU_initialized'] = torch.cuda.is_initialized()
        if result['GPU_initialized']:
            raise RuntimeError('CPU-only contract violated')
    except BaseException as exc:
        result['status'] = 'FAILED_PARTIAL'; result['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        atomic_json(a.report, result)
        for lock in locks:
            lock.close()
    csv_path = a.report.with_suffix('.inputs.csv')
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['relative_path', 'category', 'target', 'split', 'bytes', 'sha256', 'encoding_id'])
        writer.writeheader()
        for r in result['records']:
            writer.writerow({'relative_path': r['relative_path'], 'category': 'input', 'target': r['target_id'],
                             'split': r['protocol_split'], 'bytes': r['bytes'], 'sha256': r['sha256'], 'encoding_id': ENCODING_ID})
    print(json.dumps({'status': result['status'], 'files': len(result['records']), 'manifest': str(csv_path),
                      'elapsed_seconds': result['elapsed_seconds'], 'GPU_initialized': False}))


if __name__ == '__main__':
    main()
