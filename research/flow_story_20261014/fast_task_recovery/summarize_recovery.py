#!/usr/bin/env python3
"""Audit completed paired recovery receipts; no model, GT or GPU access."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def aggregate(frames, arm, sequence=None):
    rows = [f for f in frames if sequence is None or f['sequence'] == sequence]
    regions = rows[0]['arms'][arm]['metrics']['regions']
    out = {}
    for region in regions:
        parts = [f['arms'][arm]['metrics']['regions'][region] for f in rows]
        for p in parts:
            n, total, value = p['valid_pixels'], p['error_sum'], p['AEE']
            require((n == 0 and total == 0 and value is None) or
                    (n > 0 and math.isclose(total/n, value, rel_tol=1e-12)), 'Frame arithmetic')
        values = [p['AEE'] for p in parts if p['AEE'] is not None]
        n = sum(p['valid_pixels'] for p in parts)
        out[region] = {'frames_with_region': len(values), 'valid_pixels': n,
                       'frame_mean_AEE': math.fsum(values)/len(values) if values else None,
                       'pixel_weighted_AEE': math.fsum(p['error_sum'] for p in parts)/n if n else None}
    return out


def main():
    data = {a: json.loads((ROOT/(a+'_v3/results.json')).read_text()) for a in ('R0', 'R1')}
    bank = json.loads((ROOT/'prepare_v3/mask_bank.json').read_text())
    for key in ('binding', 'runtime', 'mask_bank_sha256', 'initial_fast_state_sha256'):
        require(data['R0'][key] == data['R1'][key], 'Paired contract: '+key)
    require(data['R0']['mask_bank_sha256'] == sha(ROOT/'prepare_v3/mask_bank.json'), 'Bank SHA')
    require(data['R0']['endpoint_dev8']['summaries']['warmstart'] ==
            data['R1']['endpoint_dev8']['summaries']['warmstart'], 'Warm summaries differ')
    audit = {'schema': 'flow_recovery_summary_v1', 'scope': 'Exploratory backbone-seen dev8; not official test',
             'results_sha256': {}, 'npz_sha256': {}, 'arms': {}, 'pairwise': {}}
    for arm, d in data.items():
        require(d['status'] == 'FAST_TASK_RECOVERY_COMPLETE' and d['completed_optimizer_steps'] == 200,
                'Recovery incomplete')
        require(d['original_model_unchanged'] and d['BN_buffers_unchanged'], 'Native state changed')
        require(all(d[k]['all_flow_outputs_exact'] and d[k]['branch_output_exact']
                    for k in ('initial_numerical_pair', 'final_numerical_pair')), 'Numerical guards')
        require(sha(ROOT/(arm+'_v3/weights.pt')) == d['weights_sha256'], 'Weights SHA')
        require(len(d['steps']) == 200 and [s['target_id'] for s in d['steps']] == bank['fit_ids']*10,
                'Training exposure/order')
        frames = d['endpoint_dev8']['frames']
        require([f['target_id'] for f in frames] == bank['dev_ids'], 'Dev identities')
        audit['results_sha256'][arm] = sha(ROOT/(arm+'_v3/results.json'))
        audit['npz_sha256'][arm] = {}
        for f in frames:
            p = ROOT/(arm+'_v3/dev_'+f['target_id']+'.npz')
            audit['npz_sha256'][arm][p.name] = sha(p)
            with np.load(p, allow_pickle=False) as arrays:
                require(set(arrays.files) == {'warmstart', 'recovered'}, 'NPZ arms')
                for name in arrays.files:
                    pred = arrays[name]
                    require(pred.shape == (2, 480, 640) and pred.dtype == np.float32 and
                            np.isfinite(pred).all(), 'Prediction tensor')
                    require(hashlib.sha256(pred.tobytes()).hexdigest() == f['arms'][name]['prediction_sha256'],
                            'Prediction SHA')
        for name in ('warmstart', 'recovered'):
            s = aggregate(frames, name)
            for region, values in s.items():
                for key, expected in values.items():
                    got = d['endpoint_dev8']['summaries'][name]['regions'][region][key]
                    require((expected is None and got is None) or
                            (expected is not None and math.isclose(expected, got, rel_tol=1e-12)), 'Summary arithmetic')
            key = 'warmstart' if name == 'warmstart' else arm
            audit['arms'][key] = {'regions': s, 'per_sequence': {
                seq: aggregate(frames, name, seq) for seq in sorted({f['sequence'] for f in frames})}}
    for f0, f1 in zip(data['R0']['endpoint_dev8']['frames'], data['R1']['endpoint_dev8']['frames']):
        require(f0['arms']['warmstart'] == f1['arms']['warmstart'], 'Warm predictions or service differ')
        for name in ('warmstart', 'recovered'):
            e0, e1 = (f['arms'][name]['execution'] for f in (f0, f1))
            for key in ('mode', 'shape_TBHWC', 'keep', 'slow_points', 'fast_points', 'observed', 'input_sha256'):
                require(e0[key] == e1[key], 'Endpoint actual service differs: '+key)
    for left, right in (('R0', 'warmstart'), ('R1', 'warmstart'), ('R1', 'R0')):
        contrast = {}
        for region in audit['arms'][left]['regions']:
            contrast[region] = {}
            for metric in ('frame_mean_AEE', 'pixel_weighted_AEE'):
                a, b = (audit['arms'][x]['regions'][region][metric] for x in (left, right))
                contrast[region][metric] = {'difference': a-b if a is not None else None,
                    'relative_percent': (a/b-1)*100 if a is not None and b else None}
            pairs = []
            for f0, f1 in zip(data['R0']['endpoint_dev8']['frames'], data['R1']['endpoint_dev8']['frames']):
                lookup = {'warmstart': f0['arms']['warmstart'], 'R0': f0['arms']['recovered'],
                          'R1': f1['arms']['recovered']}
                a, b = (lookup[x]['metrics']['regions'][region]['AEE'] for x in (left, right))
                pairs.append({'target_id': f0['target_id'], 'sequence': f0['sequence'],
                              'difference': a-b if a is not None else None})
            contrast[region]['paired_frames'] = pairs
        audit['pairwise'][left+'_minus_'+right] = contrast
    audit['checks_passed'] = True
    audit['source_sha256'] = sha(Path(__file__))
    out = ROOT/'summary_v1'
    require(not out.exists(), 'New summary directory required')
    out.mkdir()
    (out/'summary.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'checks_passed': True, 'summary': str(out/'summary.json'),
                      'R1_minus_warmstart': audit['pairwise']['R1_minus_warmstart']['all_valid']}))


if __name__ == '__main__':
    main()
