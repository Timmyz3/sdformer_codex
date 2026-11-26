#!/usr/bin/env python3
"""Recompute MLP-only work shapes from frozen breadth logs; never estimate cycles."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def observed(frame):
    mlp = {k: v for k, v in frame['linear_execution'].items() if '.mlp.' in k}
    other = {k: v for k, v in frame['linear_execution'].items() if '.mlp.' not in k}
    require(len(mlp) in (24, 48), 'Expected 12 two-FC MLPs, optionally two factors each')
    return {
        'mlp_leaf_rows': sum(v['rows'] for v in mlp.values()),
        'mlp_scalar_products': sum(v['rows'] * v['in_features'] * v['out_features'] for v in mlp.values()),
        'other_linear_scalar_products': sum(v['rows'] * v['in_features'] * v['out_features'] for v in other.values()),
    }


def candidate(frame):
    records = frame.get('tile_execution', [])
    if not records:
        return None
    policy = {r['layer_name']: r for r in frame['policy_records']}
    require(len(records) == len(policy) == 12, 'Missing MLP masks')
    total = 0
    for r in records:
        t, b, h, w, c = r['shape_TBHWC']
        require((t, b) == (10, 1), 'Expected complete T10 B1')
        mask = r['keep']
        expected = sum(min(9, h - 9*y) * min(9, w - 9*x)
                       for y, row in enumerate(mask) for x, keep in enumerate(row) if keep)
        p = policy[r['layer_name']]
        require(mask == p['unit_keep'], 'Recorded policy and execution masks differ')
        require(expected == r['kept_real_pixels'] == p['kept_real_pixels'], 'Mask/pixel count differs')
        require(sum(sum(row) for row in mask) == p['kept_tiles'], 'Tile count differs')
        require(p['kept_tiles'] == int(p['tile_count'] * p['keep_fraction_requested']), 'Unit quota differs')
        require(t * (h*w - expected) == r['candidate_cancelled_fc_rows_per_layer'], 'Candidate row count differs')
        require(r['mode'] == 'dense_masked' and r['observed']['fc1_rows'] == r['observed']['fc2_rows'] == t*h*w,
                'Quality path must be fully executed dense-masked here')
        total += 2 * t * expected * c * r['hidden_channels']
    return total


def audit(value):
    targets = value['population']['target_ids']
    require(len(targets) == len(set(targets)) == 32, 'Expected fixed32')
    baseline = observed(value['arms']['dense_D']['frames'][0])
    rows = []
    for case in value['cases']:
        arm = value['arms'].get(case['id'], {})
        if arm.get('status') != 'PASS':
            continue
        frames = arm['frames']
        require([f['target_id'] for f in frames] == targets, 'PASS lacks ordered fixed32')
        executions = [observed(f) for f in frames]
        require(all(v == executions[0] for v in executions), 'Regular execution shapes vary')
        require(executions[0]['other_linear_scalar_products'] == baseline['other_linear_scalar_products'],
                'Non-MLP Linear work changed; this comparison needs review')
        candidates = [candidate(f) for f in frames]
        require(all(v is None for v in candidates) or all(v is not None for v in candidates), 'Partial mask evidence')
        rows.append({'case_id': case['id'], 'method': case['method'], 'bn': case['bn'],
            'parameter_count': arm['parameter_count'], **executions[0],
            'observed_mlp_scalar_product_ratio': executions[0]['mlp_scalar_products'] / baseline['mlp_scalar_products'],
            'candidate_mlp_ratio_min': min(candidates) / baseline['mlp_scalar_products'] if candidates[0] is not None else None,
            'candidate_mlp_ratio_mean': sum(candidates) / len(candidates) / baseline['mlp_scalar_products'] if candidates[0] is not None else None,
            'candidate_mlp_ratio_max': max(candidates) / baseline['mlp_scalar_products'] if candidates[0] is not None else None})
    return {'schema': 'flow_breadth_service_shape_audit_v1', 'baseline_per_frame': baseline, 'cases': rows,
        'scope': 'Observed Linear rows times input/output width, plus mask-derived candidate MLP-only products. No sparsity, precision, PSN/BN, convolution, score, transfer, PE-slot or cycle pricing. Not full-model cost.',
        'candidate_equivalence': 'Initial14 actual_skip pairs all fail tolerance. Candidate masks cannot inherit dense_masked AEE as deployed execution evidence.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--input-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    raw = a.input.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == a.input_sha256, 'Input SHA differs')
    require(not a.output_dir.exists(), 'Use a new audit directory')
    result = audit(json.loads(raw))
    result.update(input_sha256=a.input_sha256, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output_dir.mkdir(parents=True)
    (a.output_dir / 'service_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (a.output_dir / 'service_audit.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result['cases'][0]))
        writer.writeheader()
        writer.writerows(result['cases'])
    print(json.dumps({'checked_complete_cases': len(result['cases']), 'baseline_per_frame': result['baseline_per_frame']}))


if __name__ == '__main__':
    main()
