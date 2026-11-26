#!/usr/bin/env python3
"""Read-only CPU reconstruction for two fixed R13 development identities."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True

TARGETS = ('zurich_city_10_a_0554', 'zurich_city_10_a_0475')
REP_SHA = 'f02493e7440e12828b487b272d0c17f96a1a2a6c785fd8a6764981b09825403a'
INPUT_SHA = {
    'zurich_city_10_a_0554': 'dc47271d7ffb5945236ba5b0c939c1a8b3b0240ca069354bc297cd4cffee3951',
    'zurich_city_10_a_0475': 'df939c9efd8cd8786bd9fa4ed73e3b528950d502b660248923f8da53d17663be',
}
RAW_SHA = {
    'train_events/zurich_city_10_a/events/left/events.h5': '9e9b62d167735a67f97c4480c504f3ade378702733a9f23b9022bed1c82e77b7',
    'train_events/zurich_city_10_a/events/left/rectify_map.h5': '7d3efd1c2745f8967a2d9e88e2f954ed56853b826ba488a7f6c3072c051d9038',
    'train_optical_flow/zurich_city_10_a/flow/forward_timestamps.txt': '67df5b6c6a028b7c12f7271e861b8cacc4db27733830e504ebea8cc59d227f43',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def lower_bound(values, target):
    """Global H5 timestamp binary search, independent of ms_to_idx/EventSlicer."""
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if int(values[mid]) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def cpu_checks():
    cases = [([], 1, 0), ([1, 3, 3, 5], 3, 1), ([1, 3, 3, 5], 4, 3),
             ([1, 3, 3, 5], 0, 0), ([1, 3, 3, 5], 6, 4)]
    for values, target, expected in cases:
        if lower_bound(values, target) != expected:
            raise ValueError('Binary search regression')
    return len(cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--representation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Require a fresh audit directory')
    if sha(a.representation) != REP_SHA:
        raise ValueError('Upstream representation source changed')
    started = time.monotonic()
    checks = cpu_checks()
    receipts = {t: json.loads((a.root / '.native_encoding_receipts' / (t + '.json')).read_text())
                for t in TARGETS}
    hashes = {}
    for t, rec in receipts.items():
        relative = 'saved_flow_data/event_tensors/10bins/left/zurich_city_10_a/' + t + '.npy'
        if rec['target_id'] != t or rec['relative_path'] != relative or rec['sha256'] != INPUT_SHA[t]:
            raise ValueError('Receipt identity mismatch')
        if set(rec['source_hashes']) != set(RAW_SHA):
            raise ValueError('Raw source set changed')
        for rel, expected in rec['source_hashes'].items():
            if rel not in hashes:
                p = a.root / rel
                hashes[rel] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
            if hashes[rel] != expected or hashes[rel]['sha256'] != RAW_SHA[rel]:
                raise ValueError('Raw source mismatch: ' + rel)
    import numpy as np
    import torch
    import h5py
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location('pinned_native_rep', a.representation)
    rep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rep)
    seq = 'zurich_city_10_a'
    stamps = np.loadtxt(a.root / 'train_optical_flow' / seq / 'flow/forward_timestamps.txt',
                        dtype=np.int64, delimiter=',')
    raw = a.root / 'train_events' / seq / 'events/left'
    records = []
    a.output.mkdir(parents=True)
    with h5py.File(raw / 'events.h5', 'r') as f, h5py.File(raw / 'rectify_map.h5', 'r') as r:
        offset = int(f['t_offset'][()]) if 't_offset' in f else 0
        times = f['events/t']
        rectify = r['rectify_map'][()]
        for target in TARGETS:
            rec = receipts[target]
            index = int(target.rsplit('_', 1)[1])
            pair = stamps[index - 1]
            if rec['timestamp_index_1based'] != index or pair.tolist() != rec['timestamp_pair_us']:
                raise ValueError('Timestamp identity mismatch')
            beg, end = int(pair[0]) - offset, int(pair[1]) - offset
            lo, hi = lower_bound(times, beg), lower_bound(times, end)
            if not 0 <= lo < hi <= len(times):
                raise ValueError('Empty or invalid slice')
            if (lo and int(times[lo - 1]) >= beg) or int(times[lo]) < beg:
                raise ValueError('Invalid inclusive lower boundary')
            if int(times[hi - 1]) >= end or (hi < len(times) and int(times[hi]) < end):
                raise ValueError('Invalid exclusive upper boundary')
            ev = {k: f['events/' + k][lo:hi] for k in ('p', 'x', 'y', 't')}
            if np.any(ev['t'][1:] < ev['t'][:-1]):
                raise ValueError('Unsorted timestamps inside slice')
            ev['t'] = ev['t'] + np.int64(offset)
            raw_digests = {k: {'dtype': str(v.dtype), 'count': len(v),
                              'sha256_C_order_bytes': hashlib.sha256(v.tobytes(order='C')).hexdigest()}
                           for k, v in ev.items()}
            if raw_digests != rec['raw_slice_sha']:
                raise ValueError('Independent slice differs from generation receipt')
            xy = rep.rectify_events(ev['x'], ev['y'], rectify)
            ts = (ev['t'] - ev['t'][0]).astype('float32')
            if ts[-1] <= 0:
                raise ValueError('Degenerate event interval')
            ts = ts / ts[-1]
            tensor_events = {'p': torch.from_numpy(ev['p'].astype('float32')),
                             't': torch.from_numpy(ts), 'x': torch.from_numpy(xy[:, 0]),
                             'y': torch.from_numpy(xy[:, 1])}
            voxel = rep.VoxelGrid((10, 480, 640)).convert_CHW(tensor_events).numpy()
            original = a.root / rec['relative_path']
            if sha(original) != rec['sha256']:
                raise ValueError('Stored input file differs from receipt')
            saved = np.load(original, allow_pickle=False)
            same = np.array_equal(voxel, saved)
            if not same or not np.isfinite(voxel).all():
                raise ValueError('CPU reconstruction is not exactly the stored input')
            payload = hashlib.sha256(voxel.tobytes(order='C')).hexdigest()
            if payload != rec['tensor_C_order_bytes_sha256']:
                raise ValueError('Stored payload receipt mismatch')
            out = a.output / (target + '.npy')
            with out.open('xb') as out_file:
                np.save(out_file, voxel, allow_pickle=False)
            records.append({'target_id': target, 'timestamp_pair_us': pair.tolist(),
                            'event_index_range_half_open': [lo, hi], 'raw_slice_sha': raw_digests,
                            'receipt_sha256': sha(a.root / '.native_encoding_receipts' / (target + '.json')),
                            'stored_file_sha256': sha(original), 'reconstructed_file_sha256': sha(out),
                            'payload_sha256': payload, 'shape': list(voxel.shape), 'dtype': str(voxel.dtype),
                            'array_equal': same, 'max_absolute_difference': 0.0})
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only diagnostic initialized CUDA')
    result = {'status': 'PASS', 'schema': 'flow_fixed_two_event_source_reconstruction_v1',
              'script_sha256': sha(__file__), 'representation_sha256': REP_SHA,
              'binary_search_cpu_cases': checks, 'raw_source_files': hashes, 'records': records,
              'torch': str(torch.__version__), 'numpy': np.__version__, 'h5py': h5py.__version__,
              'cuda_initialized': torch.cuda.is_initialized(), 'full_model_forwards': 0,
              'seconds': time.monotonic() - started,
              'scope': 'Two fixed development identities; independent H5 timestamp slicing with pinned upstream voxel math. No GT, saved input, weights, or source data changed; no whole-dataset semantics claim.'}
    with (a.output / 'result.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('TWO_INPUT_CPU_RECONSTRUCTION_PASS', flush=True)


if __name__ == '__main__':
    main()
