#!/usr/bin/env python3
"""R13 independent raw-array quality reduction. No runner/analyzer imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ('native', 'FF', 'CC', 'fine_s2b2')
CONDITIONS = ('reference', 'diversity')
REGIONS = ('all_valid', 'B', 'L', 'B_and_L', 'B_only', 'L_only_B_known',
           'neither_B_known', 'B_unknown', 'voxel_nonzero', 'voxel_zero',
           'B_voxel_nonzero', 'B_voxel_zero', 'L_voxel_nonzero', 'L_voxel_zero',
           'B_unknown_voxel_nonzero', 'B_unknown_voxel_zero')
MANIFEST_SHA = 'eba4544fe99c082d7d099f90d03470d7abad22b01b68d3eb4d022e61e78151d1'
ANALYSIS_SHA = '5ec51175fbf18b9d2f4fab329247b011700f4cf109b50bf73e905708698c2d7b'
REGION_SHA = '2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b'
ATOL, RTOL = 1e-10, 1e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_sha(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def masks_from_original(gt, valid, events, boundary_threshold, large_threshold):
    """Independent formula: valid four-neighbor vector jumps and voxel support."""
    vectors = np.asarray(gt, dtype=np.float64)
    height, width = valid.shape
    magnitude = np.sqrt(vectors[0] * vectors[0] + vectors[1] * vectors[1])
    known = np.zeros((height, width), dtype=bool)
    jump = np.zeros((height, width), dtype=np.float64)
    if height >= 3 and width >= 3:
        known[1:-1, 1:-1] = (valid[1:-1, 1:-1] & valid[:-2, 1:-1] & valid[2:, 1:-1]
                            & valid[1:-1, :-2] & valid[1:-1, 2:])
        center = vectors[:, 1:-1, 1:-1]
        for neighbor in (vectors[:, :-2, 1:-1], vectors[:, 2:, 1:-1],
                         vectors[:, 1:-1, :-2], vectors[:, 1:-1, 2:]):
            dx = center[0] - neighbor[0]
            dy = center[1] - neighbor[1]
            np.maximum(jump[1:-1, 1:-1], np.sqrt(dx * dx + dy * dy), out=jump[1:-1, 1:-1])
    boundary = known & (jump > boundary_threshold)
    large = valid & (magnitude > large_threshold)
    active = np.any(events != 0, axis=0)
    out = {'all_valid': valid, 'B': boundary, 'L': large,
           'B_and_L': boundary & large, 'B_only': boundary & ~large,
           'L_only_B_known': large & known & ~boundary,
           'neither_B_known': valid & known & ~boundary & ~large,
           'B_unknown': valid & ~known,
           'voxel_nonzero': valid & active, 'voxel_zero': valid & ~active}
    for name in ('B', 'L', 'B_unknown'):
        out[name + '_voxel_nonzero'] = out[name] & active
        out[name + '_voxel_zero'] = out[name] & ~active
    require(set(out) == set(REGIONS), 'Missing region')
    require(all(mask.dtype == np.bool_ and not np.any(mask & ~valid) for mask in out.values()), 'Invalid region support')
    return out


def measure(prediction, gt, masks):
    dx = prediction[0].astype(np.float64) - gt[0].astype(np.float64)
    dy = prediction[1].astype(np.float64) - gt[1].astype(np.float64)
    endpoint_error = np.sqrt(dx * dx + dy * dy)
    out = {}
    for name in REGIONS:
        support = masks[name]
        count = int(np.count_nonzero(support))
        error = float(np.sum(endpoint_error[support], dtype=np.float64))
        out[name] = {'valid_pixels': count, 'error_sum': error, 'AEE': error / count if count else None}
    return out


def aggregate(values):
    present = [v for v in values if v['valid_pixels'] > 0]
    count = sum(v['valid_pixels'] for v in values)
    error = sum(v['error_sum'] for v in values)
    return {'frames_with_region': len(present), 'empty_region_frames': len(values) - len(present),
            'valid_pixels': count, 'error_sum': error,
            'frame_mean_AEE': sum(v['AEE'] for v in present) / len(present) if present else None,
            'pixel_weighted_AEE': error / count if count else None}


def compare(candidate, reference):
    all_pass, scopes = True, {}
    for scope in candidate:
        scopes[scope] = {}
        for region, limit in (('all_valid', .03), ('B', .05), ('L', .05)):
            x = candidate[scope][region]['frame_mean_AEE']
            y = reference[scope][region]['frame_mean_AEE']
            require((x is None) == (y is None), 'Comparison support mismatch')
            passed = x is None or x <= y * (1 + limit)
            scopes[scope][region] = {'candidate': x, 'reference': y, 'limit': limit,
                'relative_change': x / y - 1 if y not in (None, 0) else None,
                'evidence': x is not None, 'pass': passed}
            all_pass = all_pass and passed
    return {'scopes': scopes, 'pass': all_pass}


def four_family_gates(summary):
    normal = {a: compare(summary[a + '/' + a], summary['initial/' + a]) for a in ('native', 'FF')}
    out = {}
    for arm in ('CC', 'fine_s2b2'):
        own = summary[arm + '/' + arm]
        families = {'normal_references': normal,
            'matched_recovered_references': {a: compare(own, summary[a + '/' + a]) for a in ('native', 'FF')},
            'own_checkpoint_references': {a: compare(own, summary[arm + '/' + a]) for a in ('native', 'FF')},
            'shared_fine_preserved': {a: compare(summary[arm + '/' + a], summary[a + '/' + a]) for a in ('native', 'FF')}}
        families['ordinary_base_quality_pass'] = all(v['pass'] for group in families.values() for v in group.values())
        out[arm] = families
    return out


class NumericAudit:
    def __init__(self):
        self.count = 0
        self.maximum_absolute = 0.0
        self.maximum_relative = 0.0
        self.maximum_location = None

    def check(self, actual, recorded, location):
        self.count += 1
        if actual is None or recorded is None:
            require(actual is None and recorded is None, 'Null mismatch: ' + location)
            return
        require(isinstance(recorded, (int, float)) and math.isfinite(recorded), 'Nonfinite reported metric: ' + location)
        difference = abs(actual - recorded)
        if difference > self.maximum_absolute:
            self.maximum_absolute, self.maximum_location = difference, location
        self.maximum_relative = max(self.maximum_relative, difference / abs(recorded) if recorded else (0.0 if not difference else difference))
        require(math.isclose(actual, recorded, rel_tol=RTOL, abs_tol=ATOL), 'Metric mismatch: ' + location)


def load_snapshot(root, receipt_path, expected_ids):
    receipt = json.loads(receipt_path.read_text())
    files = receipt['all_files']
    require(len(files) == 24, 'Exactly 24 source arrays required')
    cache, evidence = {}, {}
    kinds = {'events': ((10, 480, 640), np.dtype('float32')),
             'gt': ((2, 480, 640), np.dtype('float32')),
             'mask': ((480, 640), np.dtype('bool'))}
    for item in files:
        tid, kind = item['target_id'], item['kind']
        require(tid in expected_ids and kind in kinds, 'Unexpected source member')
        require(kind not in cache.setdefault(tid, {}), 'Duplicate source array')
        relative = Path(item['relative_path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source path')
        path = root / relative
        checksum = sha(path)
        require(checksum == item['sha256'] == item['expected_sha256'] and path.stat().st_size == item['bytes'], 'Original source bytes differ')
        value = np.load(path, allow_pickle=False)
        shape, dtype = kinds[kind]
        require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), 'Source shape/dtype/finite differs')
        cache[tid][kind] = value
        evidence.setdefault(tid, {})[kind] = checksum
    require(set(cache) == set(expected_ids) and all(set(v) == set(kinds) for v in cache.values()), 'Incomplete source snapshot')
    require(all(v['mask'].any() for v in cache.values()), 'Empty all-GT frame')
    return cache, evidence


def summarize_frames(frames, sequences):
    sets = {'overall': frames}
    sets.update({seq: [f for f in frames if f['sequence'] == seq] for seq in sequences})
    require(len(frames) == 8 and all(len(sets[seq]) == 4 for seq in sequences), 'Check scope changed')
    return {scope: {name: aggregate([f['regions'][name] for f in group]) for name in REGIONS}
            for scope, group in sets.items()}


def reduce(a):
    freeze = json.loads(a.freeze.read_text())
    require(freeze['schema'] == 'r13_independent_quality_freeze_v1', 'Unknown freeze schema')
    for name, digest in freeze['files'].items():
        require(sha(HERE / name) == digest, 'Independent audit source changed: ' + name)
    require(sha(a.membership) == MANIFEST_SHA and sha(HERE.parent / 'ANALYSIS_PROTOCOL.md') == ANALYSIS_SHA, 'Pre-result protocol changed')
    require(sha(a.regions) == REGION_SHA, 'Training-only region definition changed')
    member, region = json.loads(a.membership.read_text()), json.loads(a.regions.read_text())
    ids = [r['target_id'] for r in member['check_members']]
    sequences = sorted(member['split_sequences']['check'])
    require(len(ids) == 8 and len(set(ids)) == 8 and len(sequences) == 2, 'Check population changed')
    require(region['fit_target_ids'] == [r['target_id'] for r in member['train_members_diversity']] and region['fit_uses_final_valid'] is False,
            'Region training support changed')
    paths = {(condition, arm): a.run_root / condition / 'recover' / arm / 'results.json'
             for condition in CONDITIONS for arm in ARMS}
    require(all(path.is_file() for path in paths.values()), 'Wait for all eight complete results before quality reduction')
    raw = {key: json.loads(path.read_text()) for key, path in paths.items()}
    require(all(z.get('status') == 'ROUND13_COVERAGE_RECOVERY_COMPLETE' for z in raw.values()), 'Incomplete experiment set')
    cache, source_evidence = load_snapshot(a.input_root, a.input_receipt, ids)
    masks = {tid: masks_from_original(v['gt'], v['mask'], v['events'],
             region['boundary_threshold_pixels'], region['large_displacement_threshold_pixels']) for tid, v in cache.items()}
    numeric = NumericAudit()
    reductions, predictions, input_records, common = {}, {}, {}, {}
    npz_count, prediction_count, region_count, formal_forwards = 0, 0, 0, 0
    for condition in CONDITIONS:
        cells, recorded_summaries = {}, {}
        for arm in ARMS:
            z = raw[(condition, arm)]
            path = paths[(condition, arm)]
            require(z['condition'] == condition and z['arm'] == arm and z['mode'] == 'recover', 'Arm/condition mismatch')
            require(z['completed_optimizer_steps'] == 200 and len(z['steps']) == 200, 'Wrong optimizer updates')
            require(z['binding']['membership_sha256'] == MANIFEST_SHA and z['binding']['region_sha256'] == REGION_SHA, 'Wrong data/region binding')
            require(z['binding']['check_ids'] == ids, 'Wrong check IDs')
            expected_order = [r['target_id'] for r in member['train_order_' + condition]]
            require(z['binding']['fit_ids'] == expected_order and [v['target_id'] for v in z['steps']] == expected_order,
                    'Training curriculum changed')
            require([v['step'] for v in z['steps']] == list(range(1, 201)), 'Optimizer step order changed')
            require(all(z[k] for k in ('frozen_state_unchanged', 'BN_buffers_unchanged', 'original_modules_restored', 'all_hooks_removed')),
                    'Execution closure failed')
            require(z['full_forward_count'] == (244 if arm == 'native' else 246) and len(z['forward_records']) == z['full_forward_count'],
                    'Formal forward count differs')
            formal_forwards += z['full_forward_count']
            for field in ('runner_sha256', 'execute_sha256', 'wrapper_sha256', 'runner_checks_sha256', 'source_pins_sha256',
                          'checkpoint_sha256', 'membership_sha256', 'runtime_manifest_sha256', 'BN_buffers_sha256',
                          'BN_receipt_sha256', 'recipe', 'loss_sha256', 'allowed_parameter_count', 'region_sha256'):
                common.setdefault(field, z['binding'][field])
                require(common[field] == z['binding'][field], 'Mixed common contract: ' + field)
            common.setdefault('initial_model_state_sha256', z['initial_model_state_sha256'])
            require(z['initial_model_state_sha256'] == common['initial_model_state_sha256'], 'Unequal common initial state')
            input_records[condition + '/' + arm] = {'result_path': str(path), 'result_sha256': sha(path)}
            for phase in ('initial', 'endpoint'):
                frames = z['evaluations'][phase]['frames']
                modes = (arm,) if phase == 'initial' else ARMS
                require([f['target_id'] for f in frames] == ids, 'Frame order differs')
                for mode in modes:
                    key = ('initial' if phase == 'initial' else arm) + '/' + mode
                    require(key not in cells, 'Duplicate state/layout')
                    cells[key] = []
                    recorded_summaries[key] = z['evaluations'][phase]['summaries'][mode]
                for frame in frames:
                    tid = frame['target_id']
                    require(frame['sequence'] == tid.rsplit('_', 1)[0] and set(frame['arms']) == set(modes), 'Frame sequence/layouts differ')
                    require(frame['input_evidence'] == source_evidence[tid], 'Raw source SHA differs from evaluated frame')
                    filename = frame['prediction_file']
                    require(Path(filename).name == filename and filename.endswith('.npz'), 'Unsafe prediction filename')
                    npz = path.parent / filename
                    require(sha(npz) == frame['prediction_file_sha256'], 'Prediction archive SHA differs')
                    with np.load(npz, allow_pickle=False) as payload:
                        require(set(payload.files) == set(modes), 'Prediction archive layouts differ')
                        npz_count += 1
                        for mode in modes:
                            pred = payload[mode]
                            require(pred.shape == (2, 480, 640) and pred.dtype == np.float32 and np.isfinite(pred).all(), 'Invalid prediction')
                            psha = payload_sha(pred)
                            require(psha == frame['arms'][mode]['prediction_sha256'], 'Prediction payload SHA differs')
                            key = ('initial' if phase == 'initial' else arm) + '/' + mode
                            predictions[(condition, key, tid)] = psha
                            independent = measure(pred, cache[tid]['gt'], masks[tid])
                            original = frame['arms'][mode]['metrics']
                            require(original['target_id'] == tid and original['sequence'] == frame['sequence'] and set(original['regions']) == set(REGIONS),
                                    'Recorded metric support differs')
                            for name in REGIONS:
                                for field in ('valid_pixels', 'error_sum', 'AEE'):
                                    numeric.check(independent[name][field], original['regions'][name][field],
                                        '/'.join((condition, key, tid, name, field)))
                                region_count += 1
                            cells[key].append({'target_id': tid, 'sequence': frame['sequence'], 'regions': independent})
                            prediction_count += 1
        require(len(cells) == 20, 'Wrong state/layout count')
        summaries = {key: summarize_frames(frames, sequences) for key, frames in cells.items()}
        for key, scopes in summaries.items():
            for scope, values in scopes.items():
                report = recorded_summaries[key] if scope == 'overall' else recorded_summaries[key]['per_sequence'][scope]
                require(report['frame_count'] == (8 if scope == 'overall' else 4), 'Recorded scope frame count differs')
                for name in REGIONS:
                    for field, value in values[name].items():
                        numeric.check(value, report['regions'][name][field], '/'.join((condition, key, scope, name, field)))
        own = {}
        for arm in ARMS:
            before, after = cells['initial/' + arm], cells[arm + '/' + arm]
            pairs = [{'target_id': b['target_id'], 'initial_AEE': b['regions']['all_valid']['AEE'],
                      'post_AEE': c['regions']['all_valid']['AEE'],
                      'signed_gain': b['regions']['all_valid']['AEE'] - c['regions']['all_valid']['AEE']} for b, c in zip(before, after)]
            total = sum(v['signed_gain'] for v in pairs)
            for v in pairs:
                v['signed_gain_share_percent'] = 100 * v['signed_gain'] / total if total else None
            own[arm] = {'per_member': pairs, 'signed_total_gain': total,
                'wins': sum(v['signed_gain'] > 0 for v in pairs), 'ties': sum(v['signed_gain'] == 0 for v in pairs),
                'losses': sum(v['signed_gain'] < 0 for v in pairs),
                'own_layout_relative_initial': compare(summaries[arm + '/' + arm], summaries['initial/' + arm])}
        reductions[condition] = {'summaries': summaries, 'candidate_gates': four_family_gates(summaries),
                                 'own_layout_recovery': own, 'all_endpoints': cells}
    for arm in ARMS:
        for tid in ids:
            require(predictions[('reference', 'initial/' + arm, tid)] == predictions[('diversity', 'initial/' + arm, tid)],
                    'Initial same-layout prediction differs across coverage conditions')
    require((npz_count, prediction_count, region_count, formal_forwards) == (128, 320, 5120, 1964), 'Final raw reduction counts differ')
    paired = {}
    for key in reductions['reference']['summaries']:
        paired[key] = {}
        for scope in reductions['reference']['summaries'][key]:
            paired[key][scope] = {}
            for name in REGIONS:
                r = reductions['reference']['summaries'][key][scope][name]['frame_mean_AEE']
                d = reductions['diversity']['summaries'][key][scope][name]['frame_mean_AEE']
                require((r is None) == (d is None), 'Paired condition support differs')
                paired[key][scope][name] = {'reference': r, 'diversity': d,
                    'D_minus_R': d - r if r is not None else None,
                    'D_relative_R': d / r - 1 if r not in (None, 0) else None}
    result = {'schema': 'r13_independent_raw_quality_reduction_v1', 'status': 'PASS',
        'auditor_sha256': sha(__file__), 'freeze_sha256': sha(a.freeze), 'analysis_protocol_sha256': ANALYSIS_SHA,
        'membership_sha256': MANIFEST_SHA, 'region_sha256': REGION_SHA, 'input_receipt_sha256': sha(a.input_receipt),
        'input_results': input_records, 'common_contract': common, 'source_array_sha256': source_evidence,
        'conditions': reductions, 'paired_D_minus_R': paired,
        'counts': {'source_arrays': 24, 'results': 8, 'prediction_npz': npz_count, 'prediction_payloads': prediction_count,
                   'state_layouts': 40, 'regions_per_frame': 16, 'frame_region_metrics': region_count,
                   'scope_region_summaries': 1920, 'formal_full_forwards': formal_forwards, 'formal_optimizer_updates': 1600},
        'metric_reproduction': {'atol': ATOL, 'rtol': RTOL, 'numeric_comparisons': numeric.count,
            'maximum_absolute_difference': numeric.maximum_absolute, 'maximum_relative_difference': numeric.maximum_relative,
            'maximum_absolute_location': numeric.maximum_location},
        'common_initial_state_and_32_predictions_match': True,
        'scope': 'Independent CPU recomputation from all fixed raw GT/valid/voxel arrays and all predictions. Train-side new-pipeline sequence holdout only; no official-test, significance, flow-specific increment or hardware-benefit claim.'}
    require(not a.output_dir.exists(), 'Refuse output overwrite')
    a.output_dir.mkdir(parents=True)
    write(a.output_dir / 'independent_reduction.json', result)
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'metric_reproduction': result['metric_reproduction'],
                      'quality_gates': {c: {arm: reductions[c]['candidate_gates'][arm]['ordinary_base_quality_pass']
                                           for arm in ('CC', 'fine_s2b2')} for c in CONDITIONS}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run-root', 'input-root', 'input-receipt', 'regions', 'output-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--membership', type=Path, default=HERE.parent / 'data_protocol/manifest_v1.json')
    parser.add_argument('--freeze', type=Path, default=HERE / 'freeze_v1.json')
    reduce(parser.parse_args())


if __name__ == '__main__':
    main()
