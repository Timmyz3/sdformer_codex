#!/usr/bin/env python3
"""Independent R12 raw-input/NPZ metric audit; never imports experiment code."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import numpy as np

ARMS = ('native', 'FF', 'CC', 'fine_s2b2')
REGIONS = ('all_valid', 'B', 'L', 'B_and_L', 'B_only', 'L_only_B_known',
           'neither_B_known', 'B_unknown', 'voxel_nonzero', 'voxel_zero',
           'B_voxel_nonzero', 'B_voxel_zero', 'L_voxel_nonzero', 'L_voxel_zero',
           'B_unknown_voxel_nonzero', 'B_unknown_voxel_zero')
ATOMS = ('B_and_L', 'B_only', 'L_only_B_known', 'neither_B_known', 'B_unknown')
CANONICAL_SHA = '6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22'
ROLES_SHA = '74fea9aeaeb94cd6399273bf508494989e7c191f8cc4d22ffd9b72c96da4834f'
REGION_SHA = '8d64b18d3c69adb6c2a1bc146f55c254a2f6bbf02e50366b069f1d1d57fa8d26'
MEMBER_SHA = 'cfc84c52158213599717d1aaafb19c73506ff08edbf54831d2b839bfa9da75d5'
SNAPSHOT_SHA = '9b6c98313f610abc53b374335f4e8fa336cb02e48ffd7641f95f13902e387894'
TOLERANCE = {'rtol': 1e-12, 'error_sum_atol': 1e-10, 'AEE_atol': 1e-12,
             'integer_counts_hashes_shapes_dtypes': 'exact', 'empty_region_sum': 'exact_zero'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def region_masks(gt32, valid, events, boundary_threshold, large_threshold):
    """Four direct valid neighbors, strict thresholds, no border/GT-hole edges."""
    gt = np.asarray(gt32, dtype=np.float64)
    magnitude = np.sqrt(np.square(gt).sum(axis=0))
    eligible = np.zeros_like(valid)
    eligible[1:-1, 1:-1] = (valid[1:-1, 1:-1] & valid[:-2, 1:-1] &
        valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:])
    maximum_jump = np.zeros(valid.shape, dtype=np.float64)
    center = gt[:, 1:-1, 1:-1]
    neighbors = (gt[:, :-2, 1:-1], gt[:, 2:, 1:-1], gt[:, 1:-1, :-2], gt[:, 1:-1, 2:])
    for neighbor in neighbors:
        maximum_jump[1:-1, 1:-1] = np.maximum(maximum_jump[1:-1, 1:-1],
            np.sqrt(np.square(center - neighbor).sum(axis=0)))
    boundary = eligible & (maximum_jump > boundary_threshold)
    large = valid & (magnitude > large_threshold)
    activity = np.any(events != 0, axis=0)
    masks = {'all_valid': valid, 'B': boundary, 'L': large,
        'B_and_L': boundary & large, 'B_only': boundary & ~large,
        'L_only_B_known': large & eligible & ~boundary,
        'neither_B_known': valid & eligible & ~boundary & ~large,
        'B_unknown': valid & ~eligible, 'voxel_nonzero': valid & activity,
        'voxel_zero': valid & ~activity}
    for name in ('B', 'L', 'B_unknown'):
        masks[name + '_voxel_nonzero'] = masks[name] & activity
        masks[name + '_voxel_zero'] = masks[name] & ~activity
    require(set(masks) == set(REGIONS), 'All sixteen partitions')
    require(np.array_equal(sum(masks[name].astype(np.int8) for name in ATOMS), valid.astype(np.int8)),
            'Five mutually exclusive atoms cover every valid pixel exactly once')
    for name, mask in masks.items():
        require(mask.dtype == np.bool_ and not np.any(mask & ~valid), 'Invalid partition ' + name)
    return masks


def frame_metrics(prediction, gt, masks):
    # Cast both operands BEFORE subtraction. The metric has no loss epsilon.
    pred64, gt64 = np.asarray(prediction, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    error = np.sqrt(np.square(pred64 - gt64).sum(axis=0))
    require(np.isfinite(error).all(), 'Nonfinite endpoint error')
    result = {}
    for name in REGIONS:
        mask = masks[name]
        n = int(mask.sum())
        total = float(error[mask].sum(dtype=np.float64))
        result[name] = {'valid_pixels': n, 'error_sum': total, 'AEE': total / n if n else None}
    return result


def summarize(frames):
    result = {'frame_count': len(frames), 'regions': {}}
    for region in REGIONS:
        values = [frame['regions'][region] for frame in frames]
        present = [value for value in values if value['valid_pixels']]
        count = sum(value['valid_pixels'] for value in values)
        error = sum(value['error_sum'] for value in values)
        result['regions'][region] = {'frames_with_region': len(present),
            'empty_region_frames': len(values) - len(present), 'valid_pixels': count,
            'error_sum': error,
            'frame_mean_AEE': sum(value['AEE'] for value in present) / len(present) if present else None,
            'pixel_weighted_AEE': error / count if count else None}
    return result


def compare(expected, saved, location, maxima):
    """Check only numerical/count tree, with tolerances frozen before R12 reads."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            require(key in saved, 'Missing field ' + location + '/' + key)
            compare(value, saved[key], location + '/' + key, maxima)
    elif expected is None:
        require(saved is None, 'Empty region differs at ' + location)
    elif isinstance(expected, int):
        require(type(saved) is int and expected == saved, 'Exact count differs at ' + location)
    else:
        require(type(saved) in (float, int) and math.isfinite(saved), 'Invalid value ' + location)
        kind = 'error_sum' if location.endswith('/error_sum') else 'AEE'
        atol = TOLERANCE[kind + '_atol']
        difference = abs(expected - saved)
        maxima[kind] = max(maxima[kind], difference)
        require(math.isclose(expected, saved, rel_tol=TOLERANCE['rtol'], abs_tol=atol),
                'Numerical difference exceeds frozen tolerance at ' + location)
        if location.endswith('/error_sum') and expected == 0:
            require(saved == 0, 'Exact zero sum differs at ' + location)


def self_test():
    cases = []
    valid = np.ones((5, 5), dtype=bool)
    gt = np.zeros((2, 5, 5), dtype=np.float32)
    gt[0, 2, 2] = 1
    events = np.zeros((2, 5, 5), dtype=np.float32)
    events[:, 2, 2] = (1, -1)
    masks = region_masks(gt, valid, events, 1., 1.)
    require(not masks['B'].any() and not masks['L'].any(), 'Threshold equality must not pass')
    masks = region_masks(gt, valid, events, .5, .5)
    require(masks['B'].sum() == 5 and masks['L'].sum() == 1 and masks['B_unknown'].sum() == 16,
            'Direct-neighbor geometry')
    require(masks['voxel_nonzero'][2, 2], 'Opposite voxel signs are activity despite zero sum')
    cases.append('strict_thresholds_direct_neighbors_border_and_voxel_proxy')
    hole = valid.copy(); hole[1, 2] = False
    masks = region_masks(gt, hole, events, .5, .5)
    require(not masks['B'][2, 2] and masks['B_unknown'][2, 2] and not masks['all_valid'][1, 2],
            'GT hole creates unknown boundary, not edge')
    cases.append('GT_hole_unknown_and_exclusive_full_valid_partition')
    zero = frame_metrics(gt, gt, masks)
    require(all(value['error_sum'] == 0 and value['AEE'] in (None, 0.) for value in zero.values()),
            'No epsilon in metric')
    p = np.full((2, 1, 1), 1e8, dtype=np.float32)
    g = np.ones((2, 1, 1), dtype=np.float32)
    ones = {name: np.ones((1, 1), dtype=bool) for name in REGIONS}
    v = frame_metrics(p, g, ones)['all_valid']['AEE']
    require(v == math.sqrt(2 * (100000000. - 1.) ** 2), 'Subtract in FP64')
    require(v != float(np.sqrt(np.square(p - g).sum(axis=0))[0, 0]), 'Toy discriminates FP32 subtraction')
    cases.append('metric_no_epsilon_and_cast_before_subtraction')
    frames = [{'regions': {r: {'valid_pixels': n, 'error_sum': n * e, 'AEE': e if n else None}
               for r in REGIONS}} for n, e in [(1, 1.), (3, 3.), (0, 0.)]]
    s = summarize(frames)['regions']['all_valid']
    require(s['frame_mean_AEE'] == 2. and s['pixel_weighted_AEE'] == 2.5 and s['empty_region_frames'] == 1,
            'Frame equal, pixel weighted and empty distinct')
    cases.append('frame_equal_vs_pixel_weighted_with_empty_region')
    for expected, saved, label in [(1, 2, 'count'), (None, 0., 'empty'), (1., 1.0001, 'AEE'),
                                    (1., True, 'boolean_AEE'), (0., False, 'boolean_zero_AEE')]:
        try:
            compare(expected, saved, label, {'error_sum': 0., 'AEE': 0.})
        except AssertionError:
            pass
        else:
            raise AssertionError('Bad metric accepted: ' + label)
    cases.append('count_empty_boolean_and_numerical_drift_rejected')
    return {'status': 'PASS', 'cases': cases, 'tests': len(cases), 'numpy': np.__version__,
            'python': platform.python_version(), 'script_sha256': sha(__file__), 'tolerances': TOLERANCE,
            'scope': 'Synthetic arrays only; no R12 result, prediction or new check quality read.'}


def audit(args):
    require(len(args.result) == 4, 'All four complete results required together')
    root = Path(__file__).resolve().parents[2]
    r12 = root / 'flow_story_20261016/ordinary_recovery_round12'
    members_path = r12 / 'check_membership_v1.json'
    require(sha(members_path) == MEMBER_SHA, 'Frozen members changed')
    members = json.loads(members_path.read_text()); ids = members['groups']['check']
    receipt_path = args.snapshot / 'receipt.json'
    require(sha(receipt_path) == SNAPSHOT_SHA, 'Input snapshot receipt differs')
    receipt = json.loads(receipt_path.read_text())
    region_path = args.snapshot / 'regions_gatefit4573.json'
    require(sha(region_path) == REGION_SHA == receipt['region_sha256'], 'Region contract SHA')
    contract = json.loads(region_path.read_text())
    require(contract['schema'] == 'flow_a_train_fitted_regions_v1' and contract['definition'] ==
            'max_L2_flow_jump_to_four_direct_valid_neighbors_all_four_required_no_dilation', 'Region semantics')
    require(contract['boundary_quantile'] == .95 and contract['large_quantile'] == .9, 'Fixed quantile recipe')
    roles_path = root / 'flow_story_20261011/preflight/data_protocol/artifacts/train_auxiliary_roles.csv'
    canonical_path = roles_path.with_name('common_targets.json')
    require(sha(roles_path) == ROLES_SHA == contract['auxiliary_roles_sha256'], 'Roles source SHA')
    require(sha(canonical_path) == CANONICAL_SHA == contract['canonical_manifest_sha256'], 'Canonical source SHA')
    with roles_path.open(newline='') as f:
        roles = {row['target_id']: row['auxiliary_role'] for row in csv.DictReader(f)}
    records = json.loads(canonical_path.read_text())['records']
    ordered_fit = [row['target_id'] for row in records if roles.get(row['target_id']) == 'gate_fit']
    require(len(ordered_fit) == 4573 and ordered_fit == contract['fit_target_ids'], 'Full ordered train fit')
    require(hashlib.sha256(('\n'.join(ordered_fit) + '\n').encode()).hexdigest() == contract['fit_target_ids_sha256'],
            'Region fitting identity hash')
    require(not set(ids) & set(ordered_fit) and all(roles[t] == 'gate_development' for t in ids), 'Check region-fit separation')
    thresholds = [contract[k] for k in ('boundary_threshold_pixels', 'large_displacement_threshold_pixels')]
    require(all(math.isfinite(t) and t >= 0 for t in thresholds), 'Invalid thresholds')
    inputs = {target: {} for target in ids}; evidence = {target: {} for target in ids}
    for entry in receipt['all_files']:
        target, kind = entry['target_id'], entry['kind']
        require(target in inputs and kind in ('events', 'gt', 'mask') and kind not in inputs[target], 'Unique fixed input')
        path = args.snapshot / entry['relative_path']
        require(args.snapshot.resolve() in path.resolve().parents, 'Input path escapes snapshot')
        checksum = sha(path)
        require(checksum == entry['sha256'] == entry['expected_sha256'] and path.stat().st_size == entry['bytes'], 'Input integrity')
        arr = np.load(path, allow_pickle=False)
        shape, dtype = {'events': ((10, 480, 640), 'float32'), 'gt': ((2, 480, 640), 'float32'),
                        'mask': ((480, 640), 'bool')}[kind]
        require(arr.shape == shape and str(arr.dtype) == dtype and np.isfinite(arr).all(), 'Input array contract')
        inputs[target][kind] = arr; evidence[target][kind] = checksum
    require(len(receipt['all_files']) == 24 and all(set(v) == {'events', 'gt', 'mask'} for v in inputs.values()), 'All 24 inputs')
    masks = {}; supports = {}
    for target, arrays in inputs.items():
        require(arrays['mask'].any(), 'Empty all-valid input')
        masks[target] = region_masks(arrays['gt'], arrays['mask'], arrays['events'], *thresholds)
        supports[target] = {r: {'valid_pixels': int(m.sum()), 'mask_payload_sha256': hashlib.sha256(m.tobytes()).hexdigest()}
                            for r, m in masks[target].items()}
    raw = {}; sources = {}; cells = {}; pred_files = []; maxima = {'error_sum': 0., 'AEE': 0.}
    # Read all statuses before any metric calculation so incomplete sets are rejected.
    for path in args.result:
        result = json.loads(path.read_text()); arm = result['arm']
        require(arm in ARMS and arm not in raw, 'Distinct four arms')
        require(result['status'] == 'ROUND12_FAIR_RECOVERY_COMPLETE' and result['mode'] == 'recover' and
                result['completed_optimizer_steps'] == 200, 'Fixed 200 endpoint required')
        require(result['binding']['check_ids'] == ids, 'Ordered frozen check identities')
        raw[arm] = result; sources[arm] = {'path': str(path), 'sha256': sha(path)}
    for path in args.result:
        arm = json.loads(path.read_text())['arm']; result = raw[arm]
        for phase in ('initial', 'endpoint'):
            evaluation = result['evaluations'][phase]
            modes = (arm,) if phase == 'initial' else ARMS
            require([frame['target_id'] for frame in evaluation['frames']] == ids, 'Eight ordered frame identities')
            for mode in modes:
                key = ('initial' if phase == 'initial' else arm) + '/' + mode
                require(key not in cells, 'Distinct state/layout'); cells[key] = []
            for frame in evaluation['frames']:
                target = frame['target_id']; sequence = target.rsplit('_', 1)[0]
                require(frame['sequence'] == sequence and frame['input_evidence'] == evidence[target], 'Frame/input provenance')
                name = frame['prediction_file']; require(Path(name).name == name, 'Prediction filename')
                prediction_path = path.parent / name
                require(sha(prediction_path) == frame['prediction_file_sha256'], 'NPZ SHA')
                require(set(frame['arms']) == set(modes), 'All expected layouts in frame')
                pred_files.append({'arm': arm, 'phase': phase, 'target_id': target,
                                   'file': name, 'sha256': frame['prediction_file_sha256']})
                with np.load(prediction_path, allow_pickle=False) as payload:
                    require(set(payload.files) == set(modes), 'NPZ array keys')
                    for mode in modes:
                        pred = payload[mode]
                        require(pred.shape == (2, 480, 640) and pred.dtype == np.float32 and np.isfinite(pred).all(), 'Prediction shape/dtype/finite')
                        require(hashlib.sha256(pred.tobytes()).hexdigest() == frame['arms'][mode]['prediction_sha256'], 'Prediction payload SHA')
                        metric = frame_metrics(pred, inputs[target]['gt'], masks[target])
                        saved = frame['arms'][mode]['metrics']
                        require(saved['target_id'] == target and saved['sequence'] == sequence and set(saved['regions']) == set(REGIONS), 'Metric identity/regions')
                        compare(metric, saved['regions'], arm + '/' + phase + '/' + mode + '/' + target, maxima)
                        key = ('initial' if phase == 'initial' else arm) + '/' + mode
                        cells[key].append({'target_id': target, 'sequence': sequence, 'regions': metric})
            for mode in modes:
                key = ('initial' if phase == 'initial' else arm) + '/' + mode
                frames = cells[key]; whole = summarize(frames)
                whole['per_sequence'] = {s: summarize([f for f in frames if f['sequence'] == s])
                                         for s in sorted({f['sequence'] for f in frames})}
                require(set(whole['per_sequence']) == set(evaluation['summaries'][mode]['per_sequence']), 'Both sequence summaries')
                compare(whole, evaluation['summaries'][mode], key + '/summary', maxima)
    require(len(cells) == 20 and sum(len(v) for v in cells.values()) == 160 and len(pred_files) == 64, 'All20state-layout/160predictions/64NPZ')
    return {'schema': 'r12_independent_raw_gt_audit_v1', 'status': 'PASS', 'script_sha256': sha(__file__),
            'numpy': np.__version__, 'python': platform.python_version(), 'tolerances': TOLERANCE,
            'inputs': sources, 'snapshot_receipt_sha256': sha(receipt_path), 'regions_sha256': REGION_SHA,
            'thresholds': dict(zip(('boundary', 'large_displacement'), thresholds)),
            'original_input_SHA': evidence, 'region_support_and_mask_SHA': supports, 'predictions': pred_files,
            'prediction_count': 160, 'state_layout_count': 20, 'frame_region_metric_count': 2560,
            'summary_scope_region_count': 960, 'max_absolute_differences': maxima,
            'recomputed_frames': cells,
            'scope': 'Independent original GT/event/mask partitions and saved prediction metrics, all 16 regions, all160 endpoints and saved overall/per-sequence summaries. No evaluator/collector imports, threshold fitting, model execution, or quality-gate modification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', action='append', type=Path, default=[])
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); require(not args.output.exists(), 'Fresh output required')
    if args.self_test:
        require(not args.result and args.snapshot is None, 'Self-test must not read real input/results')
        report = self_test()
    else:
        require(args.snapshot is not None, 'Snapshot required'); report = audit(args)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'tests', 'prediction_count', 'max_absolute_differences') if k in report}))


if __name__ == '__main__':
    main()
