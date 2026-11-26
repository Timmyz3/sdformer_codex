"""CPU metric/partition/member contracts, independent of the native model."""
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np

CANONICAL_SHA = '6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22'
AUX_ROLES_SHA = '74fea9aeaeb94cd6399273bf508494989e7c191f8cc4d22ffd9b72c96da4834f'
REGION_SCHEMA = 'flow_a_train_fitted_regions_v1'
REGION_DEFINITION = 'max_L2_flow_jump_to_four_direct_valid_neighbors_all_four_required_no_dilation'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest()


def canonical_rows(path):
    if sha(path) != CANONICAL_SHA:
        raise ValueError('Unrecognized canonical manifest SHA')
    records = json.loads(Path(path).read_text())['records']
    ids = [r['target_id'] for r in records]
    train = [r for r in records if r['protocol_split'] == 'train']
    valid = [r for r in records if r['protocol_split'] == 'valid']
    if len(ids) != len(set(ids)) or len(train) != 6000 or len(valid) != 2152:
        raise ValueError('Canonical members must be distinct 6000 train / 2152 valid')
    if {r['sequence'] for r in train} & {r['sequence'] for r in valid}:
        raise ValueError('Train/valid sequences overlap')
    return records


def auxiliary_roles(path, records):
    if sha(path) != AUX_ROLES_SHA:
        raise ValueError('Unrecognized aux_policy_v2 role manifest SHA')
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    roles = {r['target_id']: r['auxiliary_role'] for r in rows}
    train = {r['target_id'] for r in records if r['protocol_split'] == 'train'}
    if len(rows) != len(roles) or set(roles) != train:
        raise ValueError('Aux roles must cover each train target exactly once')
    if {role: sum(v == role for v in roles.values()) for role in set(roles.values())} != {
            'bn_calibration': 703, 'gate_development': 724, 'gate_fit': 4573}:
        raise ValueError('Unexpected aux_policy_v2 counts')
    return roles


def select_rows(records, roles, scope, requested_ids=None):
    if scope == 'valid_all':
        if requested_ids:
            raise ValueError('valid_all forbids sample limits or target lists')
        return [r for r in records if r['protocol_split'] == 'valid']
    role_for_scope = {'train_gate_dev': 'gate_development', 'train_gate_fit': 'gate_fit',
                      'train_bn_pool': 'bn_calibration'}
    if scope not in role_for_scope:
        raise ValueError('Unknown train-side evaluation scope')
    allowed = [r for r in records if roles.get(r['target_id']) == role_for_scope[scope]]
    if requested_ids is None:
        return allowed
    if not requested_ids or len(set(requested_ids)) != len(requested_ids):
        raise ValueError('Explicit train-side target IDs must be nonempty and unique')
    by_id = {r['target_id']: r for r in allowed}
    if not set(requested_ids) <= set(by_id):
        raise ValueError('Target list contains valid IDs or other auxiliary roles')
    return [by_id[target] for target in requested_ids]


def paths_for(root, row):
    name, sequence = row['v2_target_filename'], row['sequence']
    if Path(name).name != name or Path(sequence).name != sequence or not name.endswith('.npy'):
        raise ValueError('Unsafe canonical filename')
    root = Path(root)
    return {'events': root / 'event_tensors/10bins/left' / sequence / name,
            'gt': root / 'gt_tensors' / name, 'mask': root / 'mask_tensors' / name}


def load_arrays(root, row, include_events=True, evidence=None, expected_hashes=None):
    paths = paths_for(root, row)
    arrays = {}
    for key, path in paths.items():
        if key == 'events' and not include_events:
            continue
        payload = path.read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        if expected_hashes is not None and checksum != expected_hashes.get(key):
            raise ValueError('Array differs from accepted/source inventory: ' + str(path))
        arrays[key] = np.load(io.BytesIO(payload), allow_pickle=False)
        if evidence is not None:
            evidence[key] = checksum
    expected = {'events': (10, 480, 640), 'gt': (2, 480, 640), 'mask': (480, 640)}
    dtypes = {'events': 'float32', 'gt': 'float32', 'mask': 'bool'}
    for key, value in arrays.items():
        if value.shape != expected[key] or str(value.dtype) != dtypes[key] or not np.isfinite(value).all():
            raise ValueError('Unexpected shape/type or nonfinite array: ' + str(paths[key]))
    if not np.logical_or(arrays['mask'] == 0, arrays['mask'] == 1).all():
        raise ValueError('GT validity mask must be binary')
    arrays['mask'] = arrays['mask'].astype(bool)
    if not arrays['mask'].any():
        raise ValueError('Empty GT frame cannot silently count as zero AEE')
    return arrays, paths


def region_features(gt, valid):
    """Flow units remain original input pixels; GT holes never create edges."""
    gt, valid = np.asarray(gt, dtype=np.float64), np.asarray(valid, dtype=bool)
    if gt.shape != (2,) + valid.shape or not np.isfinite(gt).all():
        raise ValueError('Bad GT/valid shapes or nonfinite GT')
    magnitude = np.sqrt(np.square(gt).sum(axis=0))
    eligible = np.zeros_like(valid)
    score = np.zeros(valid.shape, dtype=np.float64)
    eligible[1:-1, 1:-1] = (valid[1:-1, 1:-1] & valid[:-2, 1:-1] & valid[2:, 1:-1]
                              & valid[1:-1, :-2] & valid[1:-1, 2:])
    center = gt[:, 1:-1, 1:-1]
    for neighbor in (gt[:, :-2, 1:-1], gt[:, 2:, 1:-1], gt[:, 1:-1, :-2], gt[:, 1:-1, 2:]):
        score[1:-1, 1:-1] = np.maximum(score[1:-1, 1:-1], np.sqrt(np.square(center - neighbor).sum(axis=0)))
    return magnitude, score, eligible


def validate_region_contract(contract, records, roles, require_complete_fit=False):
    if contract.get('schema') != REGION_SCHEMA or contract.get('definition') != REGION_DEFINITION:
        raise ValueError('Unknown region definition/schema')
    if contract.get('canonical_manifest_sha256') != CANONICAL_SHA or contract.get('auxiliary_roles_sha256') != AUX_ROLES_SHA:
        raise ValueError('Region fitting manifests differ from fixed protocol')
    targets = contract['fit_target_ids']
    if not targets or len(targets) != len(set(targets)) or contract['fit_target_ids_sha256'] != ids_sha(targets):
        raise ValueError('Invalid region-fit target list/hash')
    fit_ids = [r['target_id'] for r in records if roles.get(r['target_id']) == 'gate_fit']
    if not set(targets) <= set(fit_ids):
        raise ValueError('Region thresholds must fit gate-fit train targets only')
    if require_complete_fit and targets != fit_ids:
        raise ValueError('Final valid evaluation requires thresholds fitted on all 4573 ordered gate-fit targets')
    for key in ('boundary_threshold_pixels', 'large_displacement_threshold_pixels'):
        if not np.isfinite(contract[key]) or contract[key] < 0:
            raise ValueError('Region thresholds must be finite and nonnegative')
    # Quantile recipe is predeclared rather than selected by final valid errors.
    if contract.get('boundary_quantile') != 0.95 or contract.get('large_quantile') != 0.90:
        raise ValueError('Unexpected predeclared region quantiles')
    return contract


def partitions(gt, valid, events, contract):
    magnitude, boundary_score, eligible = region_features(gt, valid)
    boundary = eligible & (boundary_score > contract['boundary_threshold_pixels'])
    large = valid & (magnitude > contract['large_displacement_threshold_pixels'])
    # This is an encoded-voxel activity proxy, not raw-event presence/count.
    activity = np.any(events != 0, axis=0)
    masks = {'all_valid': valid, 'B': boundary, 'L': large, 'B_and_L': boundary & large,
             'B_only': boundary & ~large, 'L_only_B_known': large & eligible & ~boundary,
             'neither_B_known': valid & eligible & ~boundary & ~large,
             'B_unknown': valid & ~eligible,
             'voxel_nonzero': valid & activity, 'voxel_zero': valid & ~activity}
    for name in ('B', 'L', 'B_unknown'):
        masks[name + '_voxel_nonzero'] = masks[name] & activity
        masks[name + '_voxel_zero'] = masks[name] & ~activity
    return masks


def crop_slices(geometry):
    if geometry == 'crop_center9':
        return slice(96, 384), slice(128, 512)
    if geometry in ('full9', 'full15'):
        return slice(None), slice(None)
    raise ValueError('Unknown geometry')


class MetricAccumulator:
    """Frame-equal primary; empty strata excluded and counted explicitly."""
    def __init__(self):
        self.frames = []
        self.seen = set()

    def update(self, target, sequence, prediction, gt, masks):
        if target in self.seen:
            raise ValueError('Duplicate metric target')
        prediction, gt = np.asarray(prediction, dtype=np.float64), np.asarray(gt, dtype=np.float64)
        if prediction.shape != gt.shape or gt.shape[0] != 2 or not np.isfinite(prediction).all():
            raise ValueError('Prediction shape or finite check failed')
        if not np.isfinite(gt).all() or not masks['all_valid'].any():
            raise ValueError('Nonfinite GT or empty whole-frame valid mask')
        error = np.sqrt(np.square(prediction - gt).sum(axis=0))
        out = {'target_id': target, 'sequence': sequence, 'regions': {}}
        for name, mask in masks.items():
            if mask.shape != error.shape or not np.issubdtype(mask.dtype, np.bool_) or np.any(mask & ~masks['all_valid']):
                raise ValueError('Invalid region mask')
            n = int(mask.sum()); total = float(error[mask].sum(dtype=np.float64))
            out['regions'][name] = {'valid_pixels': n, 'error_sum': total,
                                    'AEE': total / n if n else None}
        self.frames.append(out); self.seen.add(target)
        return out

    @staticmethod
    def summarize_frames(frames):
        if not frames:
            return {'frame_count': 0, 'regions': {}}
        result = {'frame_count': len(frames), 'regions': {}}
        for name in frames[0]['regions']:
            parts = [frame['regions'][name] for frame in frames]
            present = [part for part in parts if part['valid_pixels']]
            n = sum(part['valid_pixels'] for part in parts)
            total = sum(part['error_sum'] for part in parts)
            result['regions'][name] = {
                'frames_with_region': len(present), 'empty_region_frames': len(parts) - len(present),
                'valid_pixels': n, 'error_sum': total,
                'frame_mean_AEE': sum(part['AEE'] for part in present) / len(present) if present else None,
                'pixel_weighted_AEE': total / n if n else None}
        return result

    def summary(self):
        result = self.summarize_frames(self.frames)
        result['per_sequence'] = {sequence: self.summarize_frames([f for f in self.frames if f['sequence'] == sequence])
                                  for sequence in sorted({f['sequence'] for f in self.frames})}
        result['metric_contract'] = 'Primary: mean of per-frame mean L2 endpoint error over every valid GT pixel. Pixel-weighted and per-sequence summaries separately. Empty strata are null and counted; empty all-GT frames are rejected.'
        return result
