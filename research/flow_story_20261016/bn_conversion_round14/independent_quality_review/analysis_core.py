"""Pre-result same-W BN conversion accounting; no model/data loading."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATH_PATH = HERE.parents[1] / 'coverage_recovery_round13/independent_quality_review/reduce_raw_quality.py'
MATH_SHA = '26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088'
assert hashlib.sha256(MATH_PATH.read_bytes()).hexdigest() == MATH_SHA
_spec = importlib.util.spec_from_file_location('r13_independent_math', MATH_PATH)
METRIC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(METRIC)
REGIONS = METRIC.REGIONS
STEPS = ('135930', '271482')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def region_deltas(candidate, reference):
    result = {}
    for scope in candidate:
        result[scope] = {}
        for region in REGIONS:
            x = candidate[scope][region]['frame_mean_AEE']
            y = reference[scope][region]['frame_mean_AEE']
            require((x is None) == (y is None), 'Paired support differs')
            result[scope][region] = {'candidate': x, 'reference': y,
                'delta': x-y if x is not None else None,
                'relative_change': x/y-1 if y not in (None, 0) else None}
    return result


def analyze(current_frames, deployment_frames, check_members, execution_audit_pass):
    require(type(execution_audit_pass) is bool, 'Explicit execution audit Boolean required')
    require(set(current_frames) == set(deployment_frames) == set(STEPS), 'Exactly original135930 and271482')
    ids = [r['target_id'] for r in check_members]
    sequences = sorted({r['sequence'] for r in check_members})
    require(len(ids) == len(set(ids)) == 8 and len(sequences) == 2, 'Eight fixed targets/two sequences')
    require(all(sum(r['sequence'] == seq for r in check_members) == 4 for seq in sequences), 'Four members per sequence')
    states = {step + '/current': current_frames[step] for step in STEPS}
    states.update({step + '/deployment': deployment_frames[step] for step in STEPS})
    for name, frames in states.items():
        require([f['target_id'] for f in frames] == ids, 'Member order: ' + name)
        require([f['sequence'] for f in frames] == [r['sequence'] for r in check_members], 'Sequence mapping: ' + name)
        for frame, original in zip(frames, states['135930/deployment']):
            require(set(frame['regions']) == set(REGIONS), 'All16 regions required')
            for region, value in frame['regions'].items():
                require(set(value) == {'valid_pixels', 'error_sum', 'AEE'}, 'Metric shape')
                n = value['valid_pixels']
                require(type(n) is int and n >= 0 and n == original['regions'][region]['valid_pixels'], 'Same raw region support')
                require((value['AEE'] is None) == (n == 0), 'Empty-region semantics')
                if not n:
                    require(value['error_sum'] == 0, 'Empty-region sum')
                else:
                    require(value['AEE'] >= 0 and value['error_sum'] >= 0
                            and abs(value['AEE']-value['error_sum']/n) <= 1e-10, 'Nonfinite/inconsistent EPE')
    summaries = {key: METRIC.summarize_frames(rows, sequences) for key, rows in states.items()}
    gates, all_deltas, frame_deltas, evidence = {}, {}, {}, {}
    for step in STEPS:
        current, deployed = summaries[step + '/current'], summaries[step + '/deployment']
        gates[step] = METRIC.compare(deployed, current)
        evidence[step] = all(cell['evidence'] for regions in gates[step]['scopes'].values() for cell in regions.values())
        all_deltas[step] = region_deltas(deployed, current)
        frame_deltas[step] = []
        for d, c in zip(deployment_frames[step], current_frames[step]):
            row = {'target_id': d['target_id'], 'sequence': d['sequence'], 'regions': {}}
            for region in REGIONS:
                x, y = d['regions'][region]['AEE'], c['regions'][region]['AEE']
                row['regions'][region] = {'deployment': x, 'current_reference': y,
                    'conversion_delta': x-y if x is not None else None,
                    'conversion_relative': x/y-1 if y not in (None, 0) else None}
            frame_deltas[step].append(row)
    reasons = []
    if not execution_audit_pass:
        reasons.append('EXECUTION_NOT_VERIFIED')
    if not evidence['271482']:
        reasons.append('LATER_CONVERSION_GATE_UNSUPPORTED')
    if not gates['271482']['pass']:
        reasons.append('LATER_BN_DEPLOYMENT_QUALITY_LOSS')
    return {
        'schema': 'flow_round14_independent_bn_conversion_analysis_v1',
        'all_endpoints': states, 'summaries': summaries, 'same_W_conversion_gates': gates,
        'same_W_all_region_deltas': all_deltas, 'same_W_frame_region_deltas': frame_deltas,
        'between_checkpoint_description': {mode: region_deltas(summaries['271482/' + mode], summaries['135930/' + mode])
                                           for mode in ('current', 'deployment')},
        'conversion_gate_positive_support': evidence,
        'next_step_scope': {
            'recommendation': 'ADDRESS_ENGINEERING_OR_BN_CONTRACT_BEFORE_RECOVERY' if reasons else 'CONSIDER_ONE_FIXED_ORDINARY_RECOVERY_DIAGNOSTIC',
            'reasons': reasons, 'preselected_later_checkpoint': '271482',
            '135930_used_to_rescue_failed_later_state': False, 'K_prior_resource_gate_changed': False,
            'automatic_execution': False, 'H0_H1_admitted': False, 'recovery_stability_proven': False,
        },
        'scope': 'Same-W full current-stat reference versus cached24private deployment candidate, all8/all16. Current-stat is explicit project evaluation semantics, not asserted upstream default. Zero training/calibration. Existing K gate is unchanged; no flow-specific increment, native training sufficiency, hardware benefit or causal training-progress claim.',
    }
