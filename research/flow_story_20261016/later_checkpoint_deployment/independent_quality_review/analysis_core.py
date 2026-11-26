"""Pre-result pure analysis for one later-checkpoint deployment state."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATH_PATH = HERE.parents[1] / 'coverage_recovery_round13/independent_quality_review/reduce_raw_quality.py'
MATH_SHA = '26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088'
assert hashlib.sha256(MATH_PATH.read_bytes()).hexdigest() == MATH_SHA
_spec = importlib.util.spec_from_file_location('frozen_independent_math', MATH_PATH)
METRIC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(METRIC)
REGIONS = METRIC.REGIONS
REFERENCES = ('original/native', 'original/FF', 'smallLR/native', 'smallLR/nativeW_FF')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def analyze(candidate_frames, reference_frames, check_members, execution_audit_pass):
    require(type(execution_audit_pass) is bool, 'Execution audit must be explicit Boolean')
    require(set(candidate_frames) == {'native', 'FF'}, 'Exactly two candidate execution layouts')
    require(set(reference_frames) == set(REFERENCES), 'Historical state references are fixed')
    ids = [r['target_id'] for r in check_members]
    sequences = sorted({r['sequence'] for r in check_members})
    require(len(ids) == len(set(ids)) == 8 and len(sequences) == 2, 'Eight fixed members/two sequences')
    states = {'later/' + key: frames for key, frames in candidate_frames.items()}
    states.update(reference_frames)
    for key, frames in states.items():
        require([f['target_id'] for f in frames] == ids, 'Target order: ' + key)
        require([f['sequence'] for f in frames] == [r['sequence'] for r in check_members], 'Sequence order: ' + key)
        for frame in frames:
            require(set(frame['regions']) == set(REGIONS), 'All16 region presence: ' + key)
            for region, metrics in frame['regions'].items():
                require(set(metrics) == {'valid_pixels', 'error_sum', 'AEE'}, 'Metric shape')
                n = metrics['valid_pixels']
                require(type(n) is int and n >= 0, 'Support must be nonnegative integer')
                require((metrics['AEE'] is None) == (n == 0), 'Empty-region null semantics')
                require(n != 0 or metrics['error_sum'] == 0, 'Empty-region sum')
                if n:
                    require(metrics['error_sum'] >= 0 and metrics['AEE'] >= 0, 'Nonnegative EPE')
                    require(abs(metrics['AEE'] - metrics['error_sum'] / n) <= 1e-10, 'AEE/sum consistency')
    for key, frames in states.items():
        for frame, original in zip(frames, states['original/native']):
            for region in REGIONS:
                require(frame['regions'][region]['valid_pixels'] == original['regions'][region]['valid_pixels'],
                        'Changed raw region support: ' + key + '/' + region)
    summaries = {key: METRIC.summarize_frames(frames, sequences) for key, frames in states.items()}
    pairings = {
        'FF_vs_sameW_native': ('later/FF', 'later/native'),
        'native_vs_original_native': ('later/native', 'original/native'),
        'native_vs_smallLR_native': ('later/native', 'smallLR/native'),
        'FF_vs_original_FF': ('later/FF', 'original/FF'),
        'FF_vs_smallLR_nativeW_FF': ('later/FF', 'smallLR/nativeW_FF'),
    }
    comparisons, all_region_deltas, frame_deltas = {}, {}, {}
    for name, (candidate, reference) in pairings.items():
        comparisons[name] = METRIC.compare(summaries[candidate], summaries[reference])
        all_region_deltas[name] = {}
        for scope in summaries[candidate]:
            all_region_deltas[name][scope] = {}
            for region in REGIONS:
                x = summaries[candidate][scope][region]['frame_mean_AEE']
                y = summaries[reference][scope][region]['frame_mean_AEE']
                all_region_deltas[name][scope][region] = {'candidate': x, 'reference': y,
                    'delta': x-y if x is not None else None, 'relative_change': x/y-1 if y not in (None, 0) else None}
        frame_deltas[name] = []
        for current, old in zip(states[candidate], states[reference]):
            row = {'target_id': current['target_id'], 'sequence': current['sequence'], 'regions': {}}
            for region in REGIONS:
                x, y = current['regions'][region]['AEE'], old['regions'][region]['AEE']
                row['regions'][region] = {'candidate': x, 'reference': y, 'delta': x-y if x is not None else None}
            frame_deltas[name].append(row)
    support = all(cell['evidence'] for name in ('FF_vs_sameW_native', 'native_vs_original_native')
                  for regions in comparisons[name]['scopes'].values() for cell in regions.values())
    reasons = []
    if not execution_audit_pass:
        reasons.append('NO_GO_EXECUTION')
    if not support:
        reasons.append('NO_GO_UNSUPPORTED_GATE')
    if not comparisons['FF_vs_sameW_native']['pass']:
        reasons.append('NO_GO_FF_CONVERSION')
    if not comparisons['native_vs_original_native']['pass']:
        reasons.append('NO_GO_STATE_QUALITY_ANCHOR')
    return {
        'schema': 'flow_later_checkpoint_independent_analysis_core_v1',
        'all_endpoints': states, 'summaries': summaries, 'comparisons': comparisons,
        'all_region_deltas': all_region_deltas, 'per_frame_region_deltas': frame_deltas,
        'resource_decision': {
            'recommendation': 'NO_GO' if reasons else 'GO_ONE_ORDINARY_RECOVERY_DIAGNOSTIC',
            'reasons': reasons, 'engineering_audit_pass': execution_audit_pass,
            'gate_support_observed': support,
            'FF_conversion_preservation_pass': comparisons['FF_vs_sameW_native']['pass'],
            'original_native_state_quality_anchor_pass': comparisons['native_vs_original_native']['pass'],
            'smallLR_reference_used_for_selection': False,
            'automatic_execution': False, 'H0_H1_admitted': False,
            'recovery_stability_proven': False,
        },
        'scope': 'One untrained checkpoint-plus-own-BN deployment state, two execution layouts. Resource rule only; no training-progress causal claim, no compressed-base/recovery stability, no H0/H1 admission, no hardware claim.',
    }
