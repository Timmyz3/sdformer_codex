#!/usr/bin/env python3
"""Fit our predeclared GT-region thresholds on gate-fit TRAIN targets, CPU only.

This is a diagnostic region definition, not a published DSEC boundary standard.
No model, checkpoint, CUDA, or final-valid data are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.dont_write_bytecode = True
from flow_eval_contract import (CANONICAL_SHA, AUX_ROLES_SHA, REGION_SCHEMA, REGION_DEFINITION,
    canonical_rows, auxiliary_roles, select_rows, load_arrays, region_features, sha, ids_sha)


def sample_values(values, target, feature):
    """Fixed <=4096 uniformly sampled candidates per target; no error-dependent selection."""
    seed = int.from_bytes(hashlib.sha256(('flow-a-regions-v1:' + feature + ':' + target).encode()).digest()[:8], 'little')
    if len(values) <= 4096:
        return values
    indices = np.random.Generator(np.random.PCG64(seed)).choice(len(values), 4096, replace=False)
    return values[np.sort(indices)]


def main():
    root = Path(__file__).resolve().parents[2] / 'flow_story_20261011/preflight/data_protocol/artifacts'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=root / 'common_targets.json')
    p.add_argument('--roles', type=Path, default=root / 'train_auxiliary_roles.csv')
    p.add_argument('--target-id', action='append', default=None,
                   help='Explicit gate-fit subset for train-side exploration only; final valid requires full4573 fit')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Do not overwrite threshold artifact; select a new output')
    records = canonical_rows(a.manifest); roles = auxiliary_roles(a.roles, records)
    selected = select_rows(records, roles, 'train_gate_fit', a.target_id)
    all_fit = select_rows(records, roles, 'train_gate_fit')
    values = {'boundary': [], 'large': []}; evidence = []
    begin = time.perf_counter()
    for i, row in enumerate(selected):
        loaded_hashes = {}
        arrays, paths = load_arrays(a.data_root, row, include_events=False, evidence=loaded_hashes)
        mag, jump, eligible = region_features(arrays['gt'], arrays['mask'])
        for key, candidate in (('boundary', jump[eligible]), ('large', mag[arrays['mask']])):
            values[key].append(sample_values(candidate, row['target_id'], key))
        evidence.append({'target_id': row['target_id'], 'valid_pixels': int(arrays['mask'].sum()),
                         'boundary_eligible_pixels': int(eligible.sum()),
                         'gt_sha256': loaded_hashes['gt'], 'mask_sha256': loaded_hashes['mask']})
        if (i + 1) % 100 == 0:
            print(json.dumps({'fit_targets_completed': i + 1, 'of': len(selected)}), flush=True)
    arrays = {key: np.concatenate(parts) for key, parts in values.items()}
    if any(not len(value) for value in arrays.values()):
        raise ValueError('No eligible boundary or valid displacement samples; thresholds cannot be fabricated')
    b = float(np.quantile(arrays['boundary'], 0.95, method='linear'))
    l = float(np.quantile(arrays['large'], 0.90, method='linear'))
    result = {
        'schema': REGION_SCHEMA, 'definition': REGION_DEFINITION,
        'source': 'Our preregistered train-fitted diagnostic partition; not a paper-reported threshold or exact semantic motion boundary',
        'canonical_manifest_sha256': CANONICAL_SHA, 'auxiliary_roles_sha256': AUX_ROLES_SHA,
        'fit_target_ids': [r['target_id'] for r in selected],
        'fit_target_ids_sha256': ids_sha([r['target_id'] for r in selected]),
        'fit_scope': 'all4573_gate_fit' if selected == all_fit else 'explicit_gate_fit_subset_exploration_only',
        'fit_uses_final_valid': False, 'boundary_quantile': 0.95, 'large_quantile': 0.90,
        'boundary_threshold_pixels': b, 'large_displacement_threshold_pixels': l,
        'comparison': 'strict_greater_than; ties can reduce intended tail coverage and constant-zero fields create no B/L',
        'sampling': {'per_target_feature_cap': 4096, 'seed_prefix': 'flow-a-regions-v1:',
                     'generator': 'NumPy PCG64, SHA-derived 64-bit seed; uniform without replacement',
                     'quantile_method': 'linear',
                     'meaning': 'Fixed capped per-frame samples approximate pooled training feature quantiles; not exact full-pixel quantiles. Record actual downstream coverage.'},
        'sample_counts': {key: len(value) for key, value in arrays.items()},
        'sample_value_sha256': {key: hashlib.sha256(value.astype('<f8').tobytes()).hexdigest() for key, value in arrays.items()},
        'empty_boundary_candidate_frames': sum(e['boundary_eligible_pixels'] == 0 for e in evidence),
        'data_evidence': evidence, 'wall_seconds': time.perf_counter() - begin,
        'numpy_version': np.__version__, 'script_sha256': sha(__file__),
        'contract_module_sha256': sha(Path(__file__).with_name('flow_eval_contract.py')),
        'thresholds_fixed_before_validation': True}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'output': str(a.output), 'sha256': sha(a.output), 'fit_scope': result['fit_scope'],
                      'boundary_threshold_pixels': b, 'large_displacement_threshold_pixels': l}))


if __name__ == '__main__':
    main()
