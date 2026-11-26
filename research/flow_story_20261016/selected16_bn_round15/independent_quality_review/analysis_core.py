"""Fixed selected16 candidate vs same-W current78, private24 descriptive only."""
import hashlib
import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATH_PATH = HERE.parents[1] / 'coverage_recovery_round13/independent_quality_review/reduce_raw_quality.py'
MATH_SHA = '26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088'
assert hashlib.sha256(MATH_PATH.read_bytes()).hexdigest() == MATH_SHA
_spec = importlib.util.spec_from_file_location('r13_independent_math', MATH_PATH)
METRIC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(METRIC)
REGIONS = METRIC.REGIONS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def deltas(candidate, reference):
    result = {}
    for scope in candidate:
        result[scope] = {}
        for region in REGIONS:
            x, y = candidate[scope][region]['frame_mean_AEE'], reference[scope][region]['frame_mean_AEE']
            require((x is None) == (y is None), 'Paired region presence')
            result[scope][region] = {'candidate': x, 'reference': y,
                'delta': x-y if x is not None else None,
                'relative_change': x/y-1 if y not in (None, 0) else None}
    return result


def analyze(selected_frames, current_frames, private_frames, check_members, execution_audit_pass):
    require(type(execution_audit_pass) is bool, 'Actual execution Boolean required')
    ids = [r['target_id'] for r in check_members]
    seqs = sorted({r['sequence'] for r in check_members})
    require(len(ids) == len(set(ids)) == 8 and len(seqs) == 2
            and all(sum(r['sequence'] == s for r in check_members) == 4 for s in seqs), 'Fixed8/two sequences/four each')
    states = {'selected16/native': selected_frames, 'current78/native': current_frames, 'private24/native': private_frames}
    for name, frames in states.items():
        require([f['target_id'] for f in frames] == ids and [f['sequence'] for f in frames] == [r['sequence'] for r in check_members], 'Fixed member order: ' + name)
        for row, current in zip(frames, current_frames):
            require(set(row['regions']) == set(current['regions']) == set(REGIONS), 'All16 original regions')
            for region, value in row['regions'].items():
                require(set(value) == {'valid_pixels', 'error_sum', 'AEE'}, 'Region metric shape')
                n, total, aee = value['valid_pixels'], value['error_sum'], value['AEE']
                require(type(n) is int and n >= 0 and n == current['regions'][region]['valid_pixels'], 'Original paired support')
                require(math.isfinite(total) and total >= 0 and ((aee is None) == (n == 0)), 'Finite nonnegative EPE/support')
                if n:
                    require(math.isfinite(aee) and aee >= 0 and abs(aee-total/n) <= 1e-10, 'Finite mean/sum')
                else:
                    require(total == 0, 'Empty-region sum')
    summaries = {name: METRIC.summarize_frames(rows, seqs) for name, rows in states.items()}
    gate = METRIC.compare(summaries['selected16/native'], summaries['current78/native'])
    positive = all(c['evidence'] for rs in gate['scopes'].values() for c in rs.values())
    per_frame = {}
    for refname in ('current78/native', 'private24/native'):
        rows = []
        for selected, ref in zip(selected_frames, states[refname]):
            item = {'target_id': selected['target_id'], 'sequence': selected['sequence'], 'regions': {}}
            for region in REGIONS:
                x, y = selected['regions'][region]['AEE'], ref['regions'][region]['AEE']
                item['regions'][region] = {'selected16': x, 'reference': y, 'delta': x-y if x is not None else None,
                    'relative_change': x/y-1 if y not in (None, 0) else None}
            rows.append(item)
        per_frame[refname] = rows
    reasons = []
    if not execution_audit_pass:
        reasons.append('EXECUTION_NOT_VERIFIED')
    if not positive:
        reasons.append('CURRENT_REFERENCE_GATE_UNSUPPORTED')
    if not gate['pass']:
        reasons.append('SELECTED16_CONVERSION_QUALITY_LOSS')
    return {'schema': 'flow_selected16_bn_independent_quality_v1', 'all_endpoints': states, 'summaries': summaries,
        'selected16_vs_current78_gate': gate, 'conversion_gate_positive_support': positive,
        'all_region_deltas': {ref: deltas(summaries['selected16/native'], summaries[ref]) for ref in ('current78/native', 'private24/native')},
        'per_frame_region_deltas': per_frame,
        'next_step_scope': {'recommendation': 'STOP_BN_SUBSET_SWEEP_REVIEW_DEPLOYMENT_CONTRACT' if reasons else 'SELECTED16_CONVERSION_SCOPE_CHECK_PASSED',
            'reasons': reasons, 'checkpoint_step': 271482, 'additional_BN_subset_grid': False,
            'private24_used_to_rescue_current78_gate': False, 'prior_K_R14_gate_changed': False,
            'automatic_execution': False, 'H0_H1_admitted': False, 'recovery_stability_proven': False, 'compression_action_quality_proven': False},
        'scope': 'Eight native selected16-fixed/62current predictions, same original271482 and ownCMA176. Current78 is primary denominator; private24 diagnostic only. Reused development8, allGT/all16, zero training/calibration, no flow-specific increment or hardware benefit claim.'}
