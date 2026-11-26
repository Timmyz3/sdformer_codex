#!/usr/bin/env python3
"""Verify saved CPU predictions and inspect all-pixel final-flow fidelity.

No GT files, network or Torch are opened. Distances to tiled_all_keep are
teacher-output fidelity, NOT valid-GT AEE or task accuracy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARMS = ('native_dense', 'tiled_all_keep', 'independent_zero', 'motion_zero',
        'independent_mean', 'independent_trained', 'motion_trained')
TARGETS = ('zurich_city_01_a_0083', 'zurich_city_05_a_0165',
           'zurich_city_01_a_0030', 'zurich_city_05_a_0352')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect(results_path):
    results_path = Path(results_path)
    data = json.loads(results_path.read_text())
    require(data.get('schema') == 'flow_fast_runtime_results_v1'
            and data.get('status') == 'FAST_WARMSTART_RUNTIME_COMPLETE'
            and data.get('complete_frames') == 4, 'Complete four-frame results required')
    require(tuple(data['targets']) == TARGETS and len(data['frames']) == 4
            and tuple(f['target_id'] for f in data['frames']) == TARGETS, 'Fixed membership differs')
    require(data['runtime']['device'] == 'cpu' and data['runtime']['GPU_initialized'] is False,
            'Unexpected inference environment')
    records = []
    for frame in data['frames']:
        target = frame['target_id']; require(set(frame['arms']) == set(ARMS), 'Seven result arms required')
        artifact = frame['predictions_artifact']
        require(artifact['file'] == 'predictions_'+target+'.npz', 'Unexpected prediction archive path')
        path = results_path.parent/artifact['file']
        actual_sha = sha(path); require(actual_sha == artifact['sha256'], 'NPZ SHA differs: '+target)
        with np.load(path, allow_pickle=False) as archive:
            require(len(archive.files) == len(ARMS) and set(archive.files) == set(ARMS), 'NPZ seven keys differ')
            predictions = {}
            for arm in ARMS:
                prediction = archive[arm]
                require(prediction.shape == (2, 480, 640) and prediction.dtype == np.dtype('float32')
                        and bool(np.isfinite(prediction).all()), 'Invalid FP32 prediction: '+target+'/'+arm)
                raw_sha = hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest()
                require(raw_sha == frame['arms'][arm]['final_flow_sha256'], 'Raw prediction SHA differs: '+target+'/'+arm)
                predictions[arm] = prediction
        teacher = predictions['tiled_all_keep'].astype(np.float64)
        values = {}
        for arm, prediction in predictions.items():
            difference = prediction.astype(np.float64)-teacher
            endpoint_distance = np.sqrt(difference[0]**2+difference[1]**2)
            values[arm] = {
                'all_grid_mean_endpoint_distance_to_tiled_teacher_pixels': float(endpoint_distance.mean(dtype=np.float64)),
                'all_grid_max_endpoint_distance_to_tiled_teacher_pixels': float(endpoint_distance.max()),
                'all_grid_max_abs_component_difference_to_tiled_teacher_pixels': float(np.abs(difference).max()),
                'array_equal_to_tiled_teacher': bool(np.array_equal(prediction, predictions['tiled_all_keep'])),
                'raw_prediction_sha256': frame['arms'][arm]['final_flow_sha256']}
        native = values['native_dense']; recorded = frame['native_to_tiled_conversion']
        # Reproduce the runner's FP32 subtraction for its saved check; the
        # independent fidelity metrics above intentionally use FP64 differences.
        native_max_fp32 = float(np.abs(predictions['native_dense']-predictions['tiled_all_keep']).max())
        require(native['array_equal_to_tiled_teacher'] == recorded['final_flow_exact']
                and native_max_fp32 == recorded['final_flow_max_abs'],
                'Saved native-to-tiled conversion differs from arrays')
        records.append({'target_id': target, 'sequence': frame['sequence'], 'all_grid_pixels': 480*640,
            'prediction_archive': str(path.resolve()), 'prediction_archive_sha256': actual_sha,
            'shape': [2, 480, 640], 'dtype': 'float32', 'arms': values})
    summaries = {}
    for arm in ARMS:
        rows = [r['arms'][arm] for r in records]
        summaries[arm] = {'frame_count': len(rows), 'all_grid_pixels_per_frame': 480*640,
            'frame_equal_mean_endpoint_distance_to_tiled_teacher_pixels': math.fsum(
                r['all_grid_mean_endpoint_distance_to_tiled_teacher_pixels'] for r in rows)/len(rows),
            'maximum_abs_component_difference_over_all_frames_pixels': max(
                r['all_grid_max_abs_component_difference_to_tiled_teacher_pixels'] for r in rows),
            'maximum_endpoint_distance_over_all_frames_pixels': max(
                r['all_grid_max_endpoint_distance_to_tiled_teacher_pixels'] for r in rows),
            'frames_array_equal_to_tiled_teacher': sum(r['array_equal_to_tiled_teacher'] for r in rows)}
    return {'schema': 'flow_fast_final_prediction_fidelity_v1', 'status': 'VERIFIED_FIDELITY_COMPLETE',
        'source_results': str(results_path.resolve()), 'source_results_sha256': sha(results_path),
        'inspection_source_sha256': sha(__file__), 'reference_arm': 'tiled_all_keep',
        'scope': 'All 480x640 pixels, including invalid/unlabelled GT pixels; final-flow teacher fidelity, NOT valid-GT AEE or task accuracy.',
        'aggregation': 'Per-frame mean Euclidean endpoint distance, then equal average over four frames. Distances computed in FP64 from saved FP32 flows.',
        'population': 'Same four training-side development frames/two sequences; no population-level claim.',
        'no_GT_reopened': True, 'no_network_inference': True, 'all_4_NPZ_and_28_raw_prediction_hashes_verified': True,
        'summaries': summaries, 'frames': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output already exists')
    report = inspect(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output),
                      'summaries': report['summaries']}, allow_nan=False))


if __name__ == '__main__':
    main()
