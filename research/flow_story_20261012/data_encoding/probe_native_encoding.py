#!/usr/bin/env python3
"""Archive CPU-only, read-only three-arm encoding probe; JSON stdout, no NPY writes.

Run using stdin: prepend PLAN = <literal planned rows>, then this source.
All arms use the same event ordering; full native has no rectified-event prefilter.
Only target slices (not entire events.h5 files) are hashed. No GPU initialization.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import resource
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import torch
import h5py
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ROOT = Path('/root/private_data/SothisAI/dataset/Console/DSEC/main/DSEC')
SOURCE = Path('/root/private_data/work/sdformer_codex/SDformer')
PINS = {
    'tools/prepare_dsec_single_sequence.py': '41813faece7b5dfe8a61f002164e62e5cb6f947ef9c4d05895a675a3d8de3d9b',
    'third_party/SDformerFlow/DSEC_dataloader/event_representations.py': 'f02493e7440e12828b487b272d0c17f96a1a2a6c785fd8a6764981b09825403a',
    'third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_preprocess.py': '63fa4eaf500910d798e2d81fc17de85a0d4b6cd52f0ae6ade4d841be16256294',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(candidate, reference):
    diff = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return {'array_equal': bool(np.array_equal(candidate, reference)),
            'different_elements': int(np.count_nonzero(candidate != reference)),
            'max_absolute_difference': float(diff.max()),
            'mean_absolute_difference': float(diff.mean()),
            'relative_L1': float(diff.sum() / max(np.abs(reference.astype(np.float64)).sum(), 1e-30))}


def tensor_meta(array):
    stream = io.BytesIO(); start = time.monotonic(); np.save(stream, array, allow_pickle=False)
    value = stream.getvalue()
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'finite': bool(np.isfinite(array).all()),
            'npy_bytes': len(value), 'npy_sha256': hashlib.sha256(value).hexdigest(),
            'serialize_seconds': time.monotonic() - start,
            'abs_sum': float(np.abs(array.astype(np.float64)).sum()),
            'nonzero': int(np.count_nonzero(array))}


def main():
    for name, expected in PINS.items():
        if sha(SOURCE / name) != expected:
            raise ValueError('Source pin mismatch: ' + name)
    spec = importlib.util.spec_from_file_location('archive_helper', SOURCE / 'tools/prepare_dsec_single_sequence.py')
    prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
    rows = PLAN['records']
    if not 1 <= len(rows) <= 32 or any(r['protocol_split'] != 'train' for r in rows):
        raise ValueError('Probe requires bounded predeclared train members')
    started = time.monotonic(); records = []
    for row in rows:
        target = row['target_id']; seq = row['sequence']; index = row['target_index_1based']
        stamp = ROOT / 'train_optical_flow' / seq / 'flow/forward_timestamps.txt'
        pair = np.loadtxt(stamp, dtype=np.int64, delimiter=',')[index - 1]
        eventdir = ROOT / 'train_events' / seq / 'events/left'
        eventpath, rectpath = eventdir / 'events.h5', eventdir / 'rectify_map.h5'
        archived = ROOT / 'saved_flow_data/event_tensors/10bins/left' / seq / row['v2_target_filename']
        per_started = time.monotonic()
        with h5py.File(eventpath, 'r') as h5, h5py.File(rectpath, 'r') as rh5:
            slicer = prep.EventSlicer(h5); slicer.t_offset = np.int64(slicer.t_offset)
            rectify = rh5['rectify_map'][()]
            vg = prep.VoxelGrid((10, 480, 640))
            before = time.monotonic()
            legacy = prep.make_voxel_chunk(slicer, rectify, vg, int(pair[0]), int(pair[1]), torch.device('cpu'))
            legacy_seconds = time.monotonic() - before
            before = time.monotonic()
            events = slicer.get_events(int(pair[0]), int(pair[1]))
            if events is None or len(events['t']) < 2:
                raise ValueError('Native reference undefined for empty/single-event interval: ' + target)
            x, y = prep.rectify_events(events['x'], events['y'], rectify).T
            slice_rectify_seconds = time.monotonic() - before
            keep = (x >= 0) & (x < 640) & (y >= 0) & (y < 480)

            def convert(mask):
                t = events['t'][mask]
                if len(t) < 2 or t[-1] == t[0]:
                    raise ValueError('Degenerate native time domain: ' + target)
                # Exact upstream DSEC_dataset_preprocess.py:185-198 operation order.
                t = (t - t[0]).astype('float32')
                t = t / t[-1]
                p = events['p'][mask].astype('float32')
                ev = {'p': torch.from_numpy(p), 't': torch.from_numpy(t),
                      'x': torch.from_numpy(x[mask]), 'y': torch.from_numpy(y[mask])}
                return vg.convert_CHW(ev).numpy()

            before = time.monotonic(); relative_filtered = convert(keep)
            relative_filtered_seconds = time.monotonic() - before
            before = time.monotonic(); native = convert(slice(None))
            native_seconds = time.monotonic() - before
            expected_shape = (10, 480, 640)
            for arr in (legacy, relative_filtered, native):
                if arr.shape != expected_shape or arr.dtype != np.float32 or not np.isfinite(arr).all():
                    raise ValueError('Nonfinite/malformed probe tensor: ' + target)
            saved = np.load(archived, allow_pickle=False)
            raw_slices = {key: {'dtype': str(events[key].dtype), 'shape': list(events[key].shape),
                          'sha256_C_order_bytes': hashlib.sha256(events[key].tobytes(order='C')).hexdigest()}
                          for key in ('p', 'x', 'y', 't')}
            relative_exact = events['t'] - events['t'][0]
            stat = eventpath.stat()
            rec = {'target_id': target, 'protocol_split': 'train', 'sequence': seq,
                   'timestamp_index_1based': index, 'timestamp_pair_us': pair.tolist(),
                   'events': len(events['t']), 'in_image_events': int(keep.sum()),
                   'first_last_raw_timestamp_us': [int(events['t'][0]), int(events['t'][-1])],
                   'first_last_in_image_timestamp_us': [int(events['t'][keep][0]), int(events['t'][keep][-1])],
                   'absolute_float32_ulp_us': float(np.spacing(np.float32(events['t'][0]))),
                   'relative_cast_max_error_us': float(np.max(np.abs(relative_exact.astype(np.float64) - relative_exact.astype(np.float32)))),
                   'raw_h5': {'path': str(eventpath), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                              'full_file_SHA_computed': False, 'target_slice_hashes': raw_slices},
                   'timestamps_sha256': sha(stamp), 'rectify_sha256': sha(rectpath),
                   'archived_npy_sha256': sha(archived),
                   'legacy_cpu_vs_archived': compare(legacy, saved),
                   'time_only_vs_legacy': compare(relative_filtered, legacy),
                   'full_native_vs_legacy': compare(native, legacy),
                   'full_native_vs_time_only': compare(native, relative_filtered),
                   'legacy_tensor': tensor_meta(legacy),
                   'time_only_tensor': tensor_meta(relative_filtered),
                   'full_native_tensor': tensor_meta(native),
                   'timing': {'old_helper_including_slice_rectify_seconds': legacy_seconds,
                              'separate_slice_rectify_seconds': slice_rectify_seconds,
                              'time_only_convert_seconds': relative_filtered_seconds,
                              'native_convert_seconds': native_seconds,
                              'native_production_seconds_estimate_this_target': slice_rectify_seconds + native_seconds},
                   'process_peak_RSS_KiB_cumulative': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            rec['whole_probe_target_seconds_including_all_arms_comparisons_hashes'] = time.monotonic() - per_started
            records.append(rec)
            print(json.dumps({'progress': len(records), 'target_id': target,
                              'native_seconds': rec['timing']['native_production_seconds_estimate_this_target']}),
                  file=sys.stderr, flush=True)
    print(json.dumps({'schema_version': 'flow_encoding_probe_v1', 'scope': 'predeclared CPU train-only three-arm diagnostic, no AEE',
          'source_pins': PINS, 'canonical_sha256': PLAN['canonical_sha256'], 'selection': PLAN['selection'],
          'parameters': {'bins': 10, 'height': 480, 'width': 640, 'native_time': '(t-t[0]).astype(float32); t/t[-1]',
                         'polarity': '2*p-1 in signed voxel, not separate polarity grids',
                         'spatial': 'native rectify_map and tap mask; no whole-event prefilter in full_native',
                         'post_voxel_normalization': 'none; train wrapper applies nonzero minmax after crop/polarity split'},
          'torch': str(torch.__version__), 'numpy': np.__version__, 'cuda_initialized': torch.cuda.is_initialized(),
          'CPU_threads': 1, 'persistent_array_files_written': 0,
          'elapsed_seconds': time.monotonic() - started,
          'timing_scope_warning': 'Native target estimate excludes reopening metadata files, output transport, full-output hashes; prior legacy arm may warm HDF5 cache. First sample may include JIT warmup. Not a production-throughput claim.',
          'records': records}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
