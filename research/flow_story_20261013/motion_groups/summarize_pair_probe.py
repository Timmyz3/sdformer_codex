#!/usr/bin/env python3
"""Summarize one paired-output diagnostic; never pool CPU smoke with GPU results."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = 'flow_motion_pair_summary_v1'
REGIONS = ('all_valid', 'B', 'L', 'B_unknown')
CONDITIONS = ('motion', 'rotated90', 'horizontal_local')
METRICS = ('AEE_base', 'AEE_a', 'AEE_b', 'AEE_ab', 'AEE_additive', 'delta_a', 'delta_b',
           'delta_ab', 'interaction', 'metric_only_interaction', 'nonlinear_network_AEE_excess',
           'flow_superposition_residual_mean', 'relative_superposition_residual')
PAIRED_METRICS = ('delta_ab', 'interaction', 'metric_only_interaction',
                  'nonlinear_network_AEE_excess', 'flow_superposition_residual_mean')
NOTES = [
    'One input artifact only. CPU smoke and GPU diagnostic are separate evidence and are never pooled.',
    'Within each frame/layer/condition/region, average anchors first; then equally average nonempty frames. No pair-level significance test or independent-pair sample-size claim.',
    'Matched differences are motion minus rotated90 at the same anchor, then anchor mean within frame and equal frame mean. Negative values are arithmetic contrasts, not a deployment benefit claim.',
    'interaction = metric_only_interaction + nonlinear_network_AEE_excess. The metric-only term comes from the L2 endpoint norm on the additive-prediction counterfactual.',
    'The network term and vector superposition residual include the full downstream graph, including current-stat non-MLP BN. They do not establish motion-specific or SNN-specific coupling.',
    'Dense final flow is an offline same-target teacher. Native MLP producers still run; output branches are zeroed. No runtime selector, producer cancellation, saved cycles, energy or throughput is demonstrated.',
    'Motion and rotated90 share anchor, full-tile work and distance. Horizontal is an equal-work local control; it can duplicate another endpoint and need not match diagonal distance.',
    'Empty or missing regions are null, never zero. Incomplete source runs are excluded in full; an incomplete frame inside a claimed complete run is excluded from all its stage summaries and explicitly flagged.',
    'All-valid GT uses the entire valid frame. These are existing train-side development targets, not held-out final validation; B/L thresholds are diagnostic masks, not semantic boundary labels.',
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def average(values):
    values = [v for v in values if v is not None]
    return math.fsum(values)/len(values) if values else None


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10)


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite/non-numeric '+label)
    return float(value)


def region_values(value, base, label):
    """Return all-null for absent/empty regions; reject inconsistent measured numbers."""
    if value is None:
        return {k: None for k in METRICS}, 'missing'
    n = value.get('valid_pixels')
    require(type(n) is int and n >= 0, 'Invalid GT count: '+label)
    if base is not None:
        require(n == base['valid_pixels'], 'Region changes its GT population: '+label)
    if not n:
        require(value.get('AEE') is None and value.get('interaction') is None, 'Empty region has numeric result: '+label)
        return {k: None for k in METRICS}, 'empty'
    require(isinstance(value.get('AEE'), dict), 'Missing AEE fields: '+label)
    ae = {k: finite(value['AEE'].get(k), label+'.AEE.'+k) for k in ('base', 'a', 'b', 'ab', 'additive')}
    out = {'AEE_'+k: v for k, v in ae.items()}
    for key in METRICS[5:]:
        out[key] = None if key == 'relative_superposition_residual' and value.get(key) is None else finite(value.get(key), label+'.'+key)
    require(all(v >= 0 for v in ae.values()) and out['flow_superposition_residual_mean'] >= 0
            and (out['relative_superposition_residual'] is None or out['relative_superposition_residual'] >= 0), 'Negative norm: '+label)
    if base is not None:
        require(close(ae['base'], finite(base['AEE'], label+'.base')), 'Dense baseline differs: '+label)
    identities = {
        'delta_a': ae['a']-ae['base'], 'delta_b': ae['b']-ae['base'], 'delta_ab': ae['ab']-ae['base'],
        'interaction': ae['ab']-ae['a']-ae['b']+ae['base'],
        'metric_only_interaction': ae['additive']-ae['a']-ae['b']+ae['base'],
        'nonlinear_network_AEE_excess': ae['ab']-ae['additive']}
    require(all(close(out[k], v) for k, v in identities.items()) and
            close(out['interaction'], out['metric_only_interaction']+out['nonlinear_network_AEE_excess']),
            'Interaction decomposition differs: '+label)
    return out, 'present'


def stage_number(name):
    match = re.fullmatch(r'sttmultires_unet\.encoders\.swin3d\.layers\.([02])\.swin_blocks\.0\.mlp', name)
    require(match is not None, 'Unexpected intervention layer: '+str(name))
    return int(match.group(1))


def frame_complete(frame, layers):
    actual = frame.get('layers', [])
    if {item.get('layer') for item in actual} != set(layers) or len(actual) != len(layers):
        return 'missing_or_duplicate_declared_layer'
    for item in actual:
        if item.get('empty_intervention_exact') is not True:
            return 'empty_intervention_not_verified'
        selection = item.get('selection', {})
        if selection.get('selected_anchors') != len(item.get('pairs', [])):
            return 'partial_anchor_records'
        for pair in item.get('pairs', []):
            if set(pair.get('conditions', {})) != set(CONDITIONS):
                return 'partial_condition_records'
            if any(not isinstance(v.get('regions'), dict) for v in pair['conditions'].values()):
                return 'missing_condition_region_record'
    return None


def check_pair(pair, item):
    shape = item['shape_TBHWC']
    require(len(shape) == 5 and shape[:2] == [10, 1] and shape[4] == item['C'], 'Unexpected MLP geometry')
    anchor = pair['anchor_yx']; ends = pair['endpoints_yx']
    require(set(ends) == set(CONDITIONS) and pair['each_member_valid_feature_points'] == 81, 'Pair population differs')
    for point in (anchor, *ends.values()):
        require(len(point) == 2 and all(type(v) is int and v >= 0 for v in point)
                and (point[0]+1)*9 <= shape[2] and (point[1]+1)*9 <= shape[3], 'Tail/outside tile entered full-cost pair')
    offsets = {c: (end[0]-anchor[0], end[1]-anchor[1]) for c, end in ends.items()}
    require(all(max(abs(y), abs(x)) == 1 for y, x in offsets.values()), 'Not a nonzero one-hop pair')
    my, mx = offsets['motion']
    require(offsets['rotated90'] == (mx, -my) and offsets['horizontal_local'] == (0, 1), 'Direction controls differ')
    distances = {c: y*y+x*x for c, (y, x) in offsets.items()}
    require(distances == pair['squared_distance_in_tiles'], 'Recorded distance differs')
    for condition, body in pair['conditions'].items():
        duplicates = sorted(c for c in CONDITIONS if c != condition and ends[c] == ends[condition])
        require(sorted(body.get('same_pair_as_other_control', [])) == duplicates, 'Duplicate-control metadata differs')
        for key, points in (('a_execution', 81), ('b_execution', 81), ('ab_execution', 162)):
            records = body.get(key, [])
            require(len(records) == 1, 'Missing/multiple intervention execution records')
            record = records[0]
            require(record.get('shape_TBHWC') == shape and record.get('masked_feature_points') == points
                    and record.get('logical_omitted_fc_rows_each') == 10*points
                    and record.get('producer_calls_cancelled') == 0 and record.get('actual_PE_cycles_saved') is None,
                    'Output omission mislabeled as actual producer/cycle saving')
    return distances


def summarize(data, *, source_path=None, source_sha256=None):
    require(data.get('schema') == 'flow_motion_pair_probe_v1', 'Unexpected input schema')
    require(data.get('hardware_claim') is False and data.get('producer_cancellation_executed') is False
            and data.get('training_performed') is False, 'Unexpected hardware/training claim')
    runtime = data.get('runtime', {})
    gpu = runtime.get('GPU_inference_executed')
    device = runtime.get('device')
    if data.get('status') == 'PAIR_DIAGNOSTIC_COMPLETE':
        require(type(gpu) is bool and isinstance(device, str) and gpu == device.startswith('cuda'), 'CPU/GPU evidence label differs')
        require(data.get('inference_executed') is True and data.get('BN_buffers_unchanged') is True, 'Missing completed-run invariance evidence')
    label = 'GPU_DIAGNOSTIC' if gpu is True else 'CPU_SMOKE_ONLY' if gpu is False else 'NO_COMPLETED_INFERENCE'
    output = {'schema': SCHEMA, 'status': 'SUMMARY_COMPLETE', 'source_path': source_path, 'source_sha256': source_sha256,
              'source_status': data.get('status'), 'contract_sha256': data.get('contract_sha256'),
              'evidence_label': label, 'runtime': runtime, 'checkpoint_sha256': data.get('native_preflight', {}).get('checkpoint_sha256'),
              'BN_scope': data.get('BN_scope', {}).get('scope'), 'notes': NOTES,
              'aggregation': 'anchor mean within frame, then equal frame mean; paired motion-minus-rotated90 on the same anchor',
              'statistical_significance_claim': False, 'runtime_or_hardware_benefit_claim': False,
              'excluded_frames': [], 'included_targets': [], 'included_frame_count': 0, 'stages': {}, 'frame_statistics': []}
    if data.get('status') != 'PAIR_DIAGNOSTIC_COMPLETE':
        output.update(status='SOURCE_INCOMPLETE_EXCLUDED',
                      excluded_frames=[{'target_id': f.get('target_id'), 'reason': 'entire_incomplete_source_run_excluded'} for f in data.get('frames', [])])
        return output
    contract = data['contract']; layers = data['layers']; stages = contract['stages']
    require(len(set(layers)) == len(layers) and sorted(stage_number(n) for n in layers) == sorted(stages), 'Declared stage/layer set differs')
    require(len(data['targets']) == len(set(data['targets'])) == contract['target_count'], 'Declared target set differs')
    frames = data.get('frames', [])
    require(len({f['target_id'] for f in frames}) == len(frames) and set(f['target_id'] for f in frames) <= set(data['targets']), 'Duplicate/undeclared frames')
    for missing in sorted(set(data['targets'])-{f['target_id'] for f in frames}):
        output['excluded_frames'].append({'target_id': missing, 'reason': 'declared_target_missing'})
    buffers = {s: {'conditions': {c: {r: [] for r in REGIONS} for c in CONDITIONS},
                   'paired': {r: [] for r in REGIONS}, 'coverage': [],
                   'duplicate_horizontal': {'motion': 0, 'rotated90': 0, 'neither': 0},
                   'motion_distance_anchor_counts': {'cardinal': 0, 'diagonal': 0}} for s in stages}
    for frame in frames:
        reason = frame_complete(frame, layers)
        if reason:
            output['excluded_frames'].append({'target_id': frame['target_id'], 'reason': reason})
            continue
        output['included_targets'].append(frame['target_id'])
        base = frame['base']['regions']
        require(base['all_valid']['valid_pixels'] > 0, 'Whole-frame GT is empty')
        for item in frame['layers']:
            s = stage_number(item['layer']); bag = buffers[s]; pairs = item['pairs']; sel = item['selection']
            require(sel['requested_anchors'] == contract['pairs_per_layer'] and sel['selected_anchors'] == min(sel['requested_anchors'], sel['eligible_common_anchors']), 'Anchor coverage differs from declared selection')
            require(len({tuple(p['anchor_yx']) for p in pairs}) == len(pairs), 'Duplicate anchor in a frame/layer')
            bag['coverage'].append({'target_id': frame['target_id'], **sel})
            values = {c: {r: [] for r in REGIONS} for c in CONDITIONS}
            paired = {r: [] for r in REGIONS}
            reasons = {c: {r: {'missing': 0, 'empty': 0, 'present': 0} for r in REGIONS} for c in CONDITIONS}
            for pair in pairs:
                distances = check_pair(pair, item)
                bag['motion_distance_anchor_counts']['cardinal' if distances['motion'] == 1 else 'diagonal'] += 1
                duplicate = [c for c in ('motion', 'rotated90') if pair['endpoints_yx'][c] == pair['endpoints_yx']['horizontal_local']]
                bag['duplicate_horizontal'][duplicate[0] if duplicate else 'neither'] += 1
                current = {}
                for c in CONDITIONS:
                    current[c] = {}
                    for r in REGIONS:
                        v, why = region_values(pair['conditions'][c]['regions'].get(r), base.get(r), frame['target_id']+'/'+c+'/'+r)
                        reasons[c][r][why] += 1; values[c][r].append(v); current[c][r] = v
                for r in REGIONS:
                    if current['motion'][r]['AEE_base'] is not None and current['rotated90'][r]['AEE_base'] is not None:
                        require(close(current['motion'][r]['delta_a'], current['rotated90'][r]['delta_a']), 'Matched controls do not share anchor intervention')
                        paired[r].append({k: current['motion'][r][k]-current['rotated90'][r][k] for k in PAIRED_METRICS})
            for c in CONDITIONS:
                for r in REGIONS:
                    row = {'target_id': frame['target_id'], 'sequence': frame['sequence'], 'stage': s, 'condition': c, 'region': r,
                           'selected_anchor_count': len(pairs), 'region_anchor_counts': reasons[c][r],
                           'valid_GT_pixels_in_frame': base.get(r, {}).get('valid_pixels'),
                           'means': {k: average([v[k] for v in values[c][r]]) for k in METRICS}}
                    bag['conditions'][c][r].append(row); output['frame_statistics'].append(row)
            for r in REGIONS:
                bag['paired'][r].append({'target_id': frame['target_id'], 'sequence': frame['sequence'],
                    'matched_anchor_count': len(paired[r]), 'selected_anchor_count': len(pairs),
                    'means': {k: average([v[k] for v in paired[r]]) for k in PAIRED_METRICS}})
    for s, bag in buffers.items():
        def fold(rows, metrics):
            return {'frames_total': len(rows), 'frames_with_values': sum(any(v is not None for v in row['means'].values()) for row in rows),
                    'means': {k: average([row['means'][k] for row in rows]) for k in metrics},
                    'frames_per_metric': {k: sum(row['means'][k] is not None for row in rows) for k in metrics}}
        conditions = {c: {r: fold(bag['conditions'][c][r], METRICS) for r in REGIONS} for c in CONDITIONS}
        for c in CONDITIONS:
            for r in REGIONS:
                rows = bag['conditions'][c][r]
                conditions[c][r].update(region_anchor_counts={k: sum(v['region_anchor_counts'][k] for v in rows) for k in ('present', 'empty', 'missing')},
                    frames_without_selected_anchors=sum(v['selected_anchor_count'] == 0 for v in rows))
        matched = {r: {**fold(bag['paired'][r], PAIRED_METRICS), 'matched_anchor_count': sum(v['matched_anchor_count'] for v in bag['paired'][r]),
                       'per_frame': bag['paired'][r]} for r in REGIONS}
        output['stages'][str(s)] = {'conditions': conditions, 'motion_minus_rotated90': matched,
            'coverage_per_frame': bag['coverage'], 'duplicate_horizontal_anchor_counts': bag['duplicate_horizontal'],
            'motion_distance_anchor_counts': bag['motion_distance_anchor_counts'],
            'same_stage_full_tile_cost': {'points_per_member': 81, 'T': 10, 'logical_fc_rows_each_single': 810,
                                        'logical_fc_rows_each_pair': 1620, 'actual_producer_cancellations': 0, 'measured_cycle_saving': None}}
    if output['excluded_frames']:
        output['status'] = 'SUMMARY_PARTIAL_FRAME_EXCLUSIONS'
    else:
        require(data.get('complete_frames') == len(frames) == contract['target_count'], 'Completed frame count differs')
        require(data.get('pair_anchors') == sum(len(layer['pairs']) for frame in frames for layer in frame['layers']),
                'Completed anchor count differs')
    output['included_frame_count'] = len(output['included_targets'])
    return output


def markdown(summary):
    def fmt(value):
        return 'null' if value is None else f'{value:.8f}'
    out = ['# Paired output-omission diagnostic', '',
           f"Evidence: **{summary['evidence_label']}**. Status: `{summary['status']}`. Source: `{summary['source_path']}`.", '',
           'All numbers below average anchors within a frame, then average frames equally. No independent-pair significance test is made.', '',
           '| Stage | Condition | Region | Frames with values | ΔAEE(ab) | Interaction | Metric-only | Network excess | Vector residual |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for stage, data in summary['stages'].items():
        for condition in CONDITIONS:
            for region in REGIONS:
                row = data['conditions'][condition][region]; values = row['means']
                nums = [values[k] for k in ('delta_ab', 'interaction', 'metric_only_interaction', 'nonlinear_network_AEE_excess', 'flow_superposition_residual_mean')]
                out.append('| '+ ' | '.join([stage, condition, 'allGT' if region == 'all_valid' else region,
                                             str(row['frames_with_values'])]+[fmt(v) for v in nums])+' |')
    out += ['', '## Matched motion minus rotated90', '',
            'Same anchor and distance; differences are formed before averaging. Negative signs are contrasts, not measured runtime benefits.', '',
            '| Stage | Region | Frames | Matched anchors (descriptive only) | Δ(delta_ab) | Δ(interaction) | Δ(metric-only) | Δ(network excess) | Δ(vector residual) |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for stage, data in summary['stages'].items():
        for region, row in data['motion_minus_rotated90'].items():
            out.append('| '+' | '.join([stage, 'allGT' if region == 'all_valid' else region, str(row['frames_with_values']),
                                        str(row['matched_anchor_count'])]+[fmt(row['means'][k]) for k in PAIRED_METRICS])+' |')
    out += ['', '## Coverage and interpretation', '']
    for stage, data in summary['stages'].items():
        d = data['duplicate_horizontal_anchor_counts']
        out.append(f"- Stage {stage}: horizontal duplicates motion on {d['motion']} anchors and rotated90 on {d['rotated90']}; distinct on {d['neither']}. Duplicate endpoints are not independent controls. Distance strata: {data['motion_distance_anchor_counts']}.")
    out.append(f"- Included frames: {summary.get('included_frame_count', 0)}. Excluded frames: {len(summary['excluded_frames'])}.")
    for frame in summary['excluded_frames']:
        out.append(f"- Excluded `{frame['target_id']}`: `{frame['reason']}`.")
    out.extend('- '+note for note in summary['notes'])
    out += ['', 'AEE, interaction, network excess and vector residual units are output pixels; relative residual in JSON is dimensionless. Null fields are not zero. Full per-frame values and coverage are in the companion JSON.', '']
    return '\n'.join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-sha256')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    digest = file_sha(args.input)
    require(args.input_sha256 is None or args.input_sha256 == digest, 'Input SHA differs')
    outputs = [args.output_dir/'pair_summary.json', args.output_dir/'PAIR_SUMMARY.md']
    require(not any(p.exists() for p in outputs), 'Summary outputs already exist; preserve prior evidence')
    result = summarize(json.loads(args.input.read_text()), source_path=str(args.input.resolve()), source_sha256=digest)
    result['summarizer_sha256'] = file_sha(__file__)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    outputs[1].write_text(markdown(result))
    print(json.dumps({'status': result['status'], 'evidence_label': result['evidence_label'],
                      'included_frames': result.get('included_frame_count', 0), 'outputs': [str(p) for p in outputs]}))


if __name__ == '__main__':
    main()
