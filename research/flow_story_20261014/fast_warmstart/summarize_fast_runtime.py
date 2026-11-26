#!/usr/bin/env python3
"""Recompute recorded fast/slow CPU pilot statistics; never run inference or read GT.

Self-contained stdlib. Arithmetic helpers follow the audited Oct13 summary, but
this schema separately accounts for native slow FC, real fast LN+FC and mean fill.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ARMS = ('native_dense', 'tiled_all_keep', 'independent_zero', 'motion_zero',
        'independent_mean', 'independent_trained', 'motion_trained')
POLICIES = ('independent', 'motion', 'rotated90', 'horizontal_local')
REFERENCE_ARMS = ('independent_zero', 'independent_mean', 'independent_trained', 'motion_trained')
DIRECTIONS = ('direction_correct', 'direction_wrong', 'direction_abstain', 'direction_GT_ineligible')
REGIONS = ('all_valid', 'B', 'L', 'B_unknown', 'voxel_zero', 'voxel_nonzero') + DIRECTIONS
TARGETS = ['zurich_city_01_a_0083', 'zurich_city_05_a_0165',
           'zurich_city_01_a_0030', 'zurich_city_05_a_0352']
LAYER = 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp'
SHAPE = [10, 1, 30, 40, 384]
POINTS = [[81, 81, 81, 81, 36]]*3 + [[27, 27, 27, 27, 12]]
CHECKPOINT = 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'

def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, name, nonnegative=True):
    require(type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0), 'Invalid number: '+name)
    return value


def integer(value, name):
    require(type(value) is int and value >= 0, 'Invalid integer: '+name)
    return value


def hash_value(value, name):
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'Invalid SHA: '+name)


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)


def mean(values):
    values = [x for x in values if x is not None]
    return math.fsum(values)/len(values) if values else None


def difference(left, right):
    return None if left is None or right is None else left-right


def metric_frame(row, target, sequence, required_regions=REGIONS):
    require(row['target_id'] == target and row['sequence'] == sequence, 'Metric membership differs')
    regions = row['regions']
    require(set(required_regions) <= set(regions), 'Required regions missing')
    for name, value in regions.items():
        n = integer(value['valid_pixels'], name+'.pixels')
        total = number(value['error_sum'], name+'.error_sum')
        if n:
            require(close(number(value['AEE'], name+'.AEE'), total/n), 'Frame AEE differs from sufficient statistics')
        else:
            require(total == 0 and value['AEE'] is None, 'Empty region must be zero sum/null AEE')
    n = regions['all_valid']['valid_pixels']
    require(n > 0 and all(v['valid_pixels'] <= n for v in regions.values()), 'Invalid all-GT population')
    require(regions['voxel_zero']['valid_pixels']+regions['voxel_nonzero']['valid_pixels'] == n,
            'Voxel populations do not partition all GT')
    return row


def aggregate(frames):
    names = set(frames[0]['regions'])
    require(all(set(f['regions']) == names for f in frames), 'Region set changed across frames')
    out = {'frame_count': len(frames), 'regions': {}}
    for name in sorted(names):
        pieces = [f['regions'][name] for f in frames]
        n = sum(v['valid_pixels'] for v in pieces); total = math.fsum(v['error_sum'] for v in pieces)
        count = sum(v['valid_pixels'] > 0 for v in pieces)
        out['regions'][name] = {'frames_with_region': count, 'empty_region_frames': len(pieces)-count,
            'valid_pixels': n, 'error_sum': total, 'frame_mean_AEE': mean(v['AEE'] for v in pieces),
            'pixel_weighted_AEE': total/n if n else None}
    return out


def compare_summary(stored, recomputed, label):
    require(stored['frame_count'] == recomputed['frame_count']
            and set(stored['regions']) == set(recomputed['regions']), 'Summary coverage differs: '+label)
    for name, values in recomputed['regions'].items():
        for key, expected in values.items():
            actual = stored['regions'][name][key]
            if expected is None:
                require(actual is None, 'Empty summary is numeric: '+label)
            elif isinstance(expected, int):
                require(type(actual) is int and actual == expected, 'Summary count differs: '+label)
            else:
                require(close(number(actual, label+'.'+key), expected), 'Summary arithmetic differs: '+label)


def counter_check(counters):
    for key in ('state_visits', 'transition_count', 'include_transitions_attempted', 'exclude_transitions',
                'include_transitions_within_cap', 'include_transitions_over_cap', 'score_additions',
                'score_comparisons', 'equal_score_tie_comparisons', 'peak_reachable_states'):
        integer(counters[key], 'DP.'+key)
    n = counters['state_visits']
    require(counters['transition_count'] == 2*n and counters['exclude_transitions'] == n
            and counters['include_transitions_attempted'] == n
            and counters['include_transitions_within_cap']+counters['include_transitions_over_cap'] == n,
            'DP transition accounting differs')


def execution(record, mode, branch, all_keep=False):
    require(len(record['execution']) == 1, 'Exactly one MLP invocation required')
    item = record['execution'][0]
    require(item['layer_name'] == LAYER and item['shape_TBHWC'] == SHAPE
            and item['mode'] == mode and item['branch'] == branch
            and item['all_keep_shortcut'] is False, 'Execution contract differs')
    keep = item['keep']
    require(len(keep) == 4 and all(len(r) == 5 and all(type(x) is bool for x in r) for r in keep), 'Invalid tile mask')
    ids = [r*5+c for r in range(4) for c in range(5) if keep[r][c]]
    points = sum(POINTS[i//5][i%5] for i in ids)
    require(points == (1200 if all_keep else 900)
            and item['slow_points'] == points and item['fast_region_points'] == 1200-points
            and item['slow_tiles'] == len(ids), 'Actual point budget differs')
    slow_rows = 12000 if mode == 'reference' else 10*points
    slow_calls = 20 if mode == 'reference' else len(ids)
    fast_rows = (12000 if mode == 'reference' else 10*(1200-points)) if branch == 'trained' else 0
    fast_calls = ((20 if mode == 'reference' else 20-len(ids)) if branch == 'trained' else 0)
    mean_elements = ((12000 if mode == 'reference' else 10*(1200-points))*384) if branch == 'mean' else 0
    require(item['observed'] == {'slow_fc1_rows': slow_rows, 'slow_fc2_rows': slow_rows,
            'fast_fc_rows': fast_rows, 'slow_calls': slow_calls, 'fast_calls': fast_calls,
            'mean_fill_elements': mean_elements}, 'Observed slow/fast service differs')
    require(item['fast_LN_rows'] == fast_rows and item['fast_real_MACs_observed_rows_times_C2'] == fast_rows*384*384,
            'Fast LN/real-MAC accounting differs')
    for key in ('input_sha256', 'output_sha256'): hash_value(item[key], key)
    for key in ('input_stride', 'output_stride'):
        require(len(item[key]) == 5 and all(type(x) is int and x > 0 for x in item[key]), 'Invalid tensor stride')
    number(record['wall_seconds_CPU'], 'inference wall')
    hash_value(record['final_flow_sha256'], 'final flow')
    return item, ids


def selector_cost(record, arm, target, sequence):
    policy, branch = arm.split('_', 1)
    item, ids = execution(record, 'actual', branch)
    source = record['runtime_selector']
    proposal, selection, common = source['proposal'], source['selection'], source['common_budget']
    require(source['policy'] == policy and source['input_shape_TBHWC'] == SHAPE
            and source['input_sha256'] == item['input_sha256'] and source['points_per_tile'] == POINTS
            and source['dense_spatial_points'] == 1200 and source['nominal_slow_cap'] == 900
            and source['GT_or_final_flow_used'] is False and source['upstream_attention_already_executed'] is True
            and source['all_four_proposals_computed_for_cost_matching'] is True, 'Runtime selector contract differs')
    require(selection['feasible'] is True and selection['slow_mask'] == item['keep']
            and selection['selected_tile_ids'] == ids and selection['slow_service_cost'] == 900
            and selection['achieved_cost'] == selection['target_cost'] == common['target_cost'] == 900
            and common['cap'] == 900 and common['unused_cap'] == 0, 'Exact selection differs')
    reachable = common['per_policy_reachability']
    require(set(reachable) == set(POLICIES), 'All-four planner work missing')
    common_set = set.intersection(*(set(reachable[p]['costs']) for p in POLICIES))
    require(common_set and max(common_set) == 900 and sorted(common_set) == common['common_reachable_costs'], 'Reachability differs')
    for value in reachable.values(): counter_check(value['counters'])
    counter_check(selection['counters'])
    reach_work = sum(v['counters']['transition_count'] for v in reachable.values())
    require(common['reachability_transition_count'] == reach_work, 'Reachability work omitted')
    groups = proposal['groups']; by_id = {g['group_id']: g for g in groups}
    require(len(by_id) == len(groups) and proposal['policy'] == policy, 'Proposal identity differs')
    require(sorted(i for g in groups for i in g['members']) == list(range(20))
            and all(len(g['members']) in (1, 2) for g in groups), 'Groups do not partition tiles')
    for group in groups:
        require(group['slow_service_cost'] == sum(POINTS[i//5][i%5] for i in group['members']), 'Tail cost differs')
    chosen = selection['selected_group_ids']
    require(len(set(chosen)) == len(chosen) and set(chosen) <= set(by_id), 'Invalid selected groups')
    selected = [by_id[i] for i in chosen]
    require(sorted(i for g in selected for i in g['members']) == ids
            and selection['selected_group_count'] == len(selected)
            and selection['selected_tile_count'] == len(ids), 'Group closure differs')
    flow, scores = source['flow_dxdy_pixels'], source['scores']
    require(len(flow) == len(scores) == 4 and all(len(r) == 5 for r in flow+scores), 'Head grid differs')
    for row, score_row in zip(flow, scores):
        for vector, score in zip(row, score_row):
            require(len(vector) == 2, 'Flow width differs')
            for v in vector: number(v, 'runtime flow', False)
            # The online norm was computed in FP32; Python hypot is FP64.
            require(math.isclose(number(score, 'score'), math.hypot(*vector), rel_tol=2e-6, abs_tol=1e-7), 'Head magnitude score differs')
    head = source['coarse_head_shape_cost']
    expected = {'input_shape_TBHWC': SHAPE, 'tile_count': 20, 'spatial_points_per_tile': POINTS,
        'temporal_reduction_additions': 9*1200*384, 'temporal_mean_divisions': 1200*384,
        'spatial_reduction_additions': 384*(1200-20), 'spatial_mean_divisions': 20*384,
        'linear_real_MACs': 20*(384*16+16*2), 'linear_bias_additions': 20*18,
        'GELU_scalar_evaluations': 20*16, 'costs_are_measured_cycles': False}
    require(all(head.get(k) == v for k, v in expected.items()), 'Head cost differs')
    return {'target_id': target, 'sequence': sequence, 'arm': arm, 'policy': policy, 'branch': branch,
        'slow_points': 900, 'full_spatial_points': 1200, 'slow_fraction': .75, 'fast_region_points': 300,
        'retained_tile_calls': len(ids), 'available_tiles': 20,
        'paired_groups_selected': sum(len(g['members']) == 2 for g in selected),
        'paired_groups_available': sum(len(g['members']) == 2 for g in groups),
        'observed': item['observed'], 'fast_LN_rows': item['fast_LN_rows'],
        'fast_real_MACs': item['fast_real_MACs_observed_rows_times_C2'],
        'head_shape_cost_nominal': head, 'reachability_DP_all_four': {p: v['counters'] for p, v in reachable.items()},
        'selected_policy_DP': selection['counters'],
        'DP_transition_count_all_reachability_plus_one_selection': reach_work+selection['counters']['transition_count'],
        'controller_wall_seconds_CPU': number(source['controller_wall_seconds_CPU'], 'controller wall'),
        'inference_wall_seconds_CPU': record['wall_seconds_CPU'], 'measured_PE_cycles': None,
        'cost_note': 'Equal slow FC rows only; real fast LN/FC, mean fill, head and DP are separate services.'}


def direction_check(regions):
    all_gt = regions['all_valid']
    require(sum(regions[k]['valid_pixels'] for k in DIRECTIONS) == all_gt['valid_pixels'], 'Direction populations do not cover GT')
    require(close(math.fsum(regions[k]['error_sum'] for k in DIRECTIONS), all_gt['error_sum']), 'Direction error sums do not cover GT')


def summarize(data, source_path=None, source_sha256=None):
    require(data.get('schema') == 'flow_fast_runtime_results_v1'
            and data.get('status') == 'FAST_WARMSTART_RUNTIME_COMPLETE', 'Only complete fast runtime accepted')
    require(data.get('complete_frames') == 4 and data.get('inference_executed') is True
            and data.get('BN_buffers_unchanged') is True and data.get('hardware_claim') is False
            and data.get('backbone_or_gate_recovery_performed') is False, 'Completion/scope differs')
    require(data['runtime']['device'] == 'cpu' and data['runtime']['GPU_initialized'] is False
            and data['runtime']['TF32'] is False and data['BN_scope']['scope'] == 'MLP_CMA176'
            and data['BN_scope']['non_bn_state_unchanged'] is True, 'CPU/BN scope differs')
    require(data['native_preflight']['checkpoint_sha256'] == CHECKPOINT
            and data['native_preflight']['geometry'] == 'full9', 'Checkpoint or geometry differs')
    require(data['contract']['schema'] == 'flow_fast_runtime_contract_v1'
            and data['contract']['target_count'] == 4, 'Contract scope differs')
    hash_value(data['contract_sha256'], 'contract')
    for value in data['contract']['source_pins'].values(): hash_value(value, 'source pin')
    for key in ('head_weights_sha256', 'head_fit_cache_sha256', 'fast_weights_sha256', 'fast_cache_sha256'):
        hash_value(data['contract'][key], key)
    frames = data['frames']
    require(data['targets'] == TARGETS and len(frames) == 4 and [f['target_id'] for f in frames] == TARGETS
            and set(data['arms']) == set(ARMS) and set(data['input_evidence']) == set(TARGETS), 'Four-frame/seven-arm membership differs')
    rows = {arm: [] for arm in ARMS}; costs = []; references = []; conversions = []; pairs = []
    for fi, frame in enumerate(frames):
        target, sequence = frame['target_id'], frame['sequence']
        require(sequence == target.rsplit('_', 1)[0] and set(frame['arms']) == set(ARMS), 'Frame membership differs')
        evidence = data['input_evidence'][target]
        require(set(evidence) == {'events', 'gt', 'mask'}, 'Input evidence missing')
        for key, value in evidence.items(): hash_value(value, key)
        artifact = frame['predictions_artifact']; hash_value(artifact['sha256'], 'predictions')
        require(artifact['file'] == 'predictions_'+target+'.npz', 'Prediction artifact name differs')
        definition = frame['direction_error_strata']
        labels = definition['tile_labels_0ineligible_1correct_2wrong_3abstain']
        require(definition['GT_used_by_inference'] is False and len(labels) == 4
                and all(len(r) == 5 and all(type(v) is int and 0 <= v <= 3 for v in r) for r in labels), 'Direction label scope differs')
        counts = None; input_hashes = set(); selectors = []
        for arm in ARMS:
            record = frame['arms'][arm]
            metric = metric_frame(record['metrics'], target, sequence); direction_check(metric['regions']); rows[arm].append(metric)
            current = {r: v['valid_pixels'] for r, v in metric['regions'].items()}
            require(counts is None or counts == current, 'Arms use different GT populations'); counts = current
            number(record['wall_seconds_CPU'], 'CPU wall'); hash_value(record['final_flow_sha256'], 'flow')
            if arm == 'native_dense':
                require(record['execution'] == [] and 'runtime_selector' not in record, 'Native graph changed')
            elif arm == 'tiled_all_keep':
                item, _ = execution(record, 'actual', 'zero', all_keep=True); input_hashes.add(item['input_sha256'])
                require('runtime_selector' not in record, 'All-keep unexpectedly invokes selector')
            else:
                costs.append(selector_cost(record, arm, target, sequence))
                item = record['execution'][0]; input_hashes.add(item['input_sha256']); selectors.append(record['runtime_selector'])
                if fi == 0 and arm in REFERENCE_ARMS:
                    ref = record['same_shape_reference']; other = ref['reference']
                    other_item, _ = execution(other, 'reference', arm.split('_', 1)[1])
                    require(ref['final_flow_exact'] is True and ref['final_flow_max_abs'] == 0
                            and record['final_flow_sha256'] == other['final_flow_sha256']
                            and all(item[k] == other_item[k] for k in ('input_sha256', 'output_sha256', 'keep', 'input_stride', 'output_stride')),
                            'Actual/reference numerical evidence differs')
                    references.append({'target_id': target, 'arm': arm, 'branch_exact': True,
                        'final_flow_exact': True, 'max_abs': 0, 'reference_observed': other_item['observed']})
                else:
                    require('same_shape_reference' not in record, 'Unexpected reference coverage')
        require(len(input_hashes) == 1, 'Pre-MLP input differs across replaced arms')
        for key in ('flow_dxdy_pixels', 'descriptor_sha256', 'common_budget'):
            require(all(s[key] == selectors[0][key] for s in selectors), 'Common online signal/budget differs: '+key)
        for family in ('independent', 'motion'):
            names = [family+'_zero', family+'_trained'] + (['independent_mean'] if family == 'independent' else [])
            require(all(frame['arms'][a]['execution'][0]['keep'] == frame['arms'][names[0]]['execution'][0]['keep'] for a in names), 'Branch substitution changed masks')
        conversion = frame['native_to_tiled_conversion']; maximum = number(conversion['final_flow_max_abs'], 'native delta')
        require(type(conversion['final_flow_exact']) is bool and conversion['final_flow_exact'] == (maximum == 0)
                and conversion['final_flow_exact'] == (frame['arms']['native_dense']['final_flow_sha256'] == frame['arms']['tiled_all_keep']['final_flow_sha256']), 'Native conversion evidence differs')
        conversions.append({'target_id': target, 'sequence': sequence, **conversion,
            'allGT_AEE_tiled_minus_native': difference(frame['arms']['tiled_all_keep']['metrics']['regions']['all_valid']['AEE'],
                                                      frame['arms']['native_dense']['metrics']['regions']['all_valid']['AEE'])})
        for left, right in (('motion_trained', 'independent_trained'), ('motion_zero', 'independent_zero'),
                            ('independent_trained', 'independent_zero'), ('independent_trained', 'independent_mean'),
                            ('motion_trained', 'motion_zero')):
            pairs.append({'target_id': target, 'sequence': sequence, 'contrast': left+'_minus_'+right,
                'region_delta_AEE': {r: difference(frame['arms'][left]['metrics']['regions'][r]['AEE'], frame['arms'][right]['metrics']['regions'][r]['AEE']) for r in rows[left][-1]['regions']}})
    summaries = {}
    for arm in ARMS:
        saved = data['arms'][arm]; require(saved['frames'] == rows[arm], 'Duplicate frame tables disagree')
        summary = aggregate(rows[arm]); compare_summary(saved['summary'], summary, arm)
        seqs = sorted({r['sequence'] for r in rows[arm]})
        require(set(saved['summary']['per_sequence']) == set(seqs), 'Per-sequence coverage differs')
        summary['per_sequence'] = {}
        for seq in seqs:
            part = aggregate([r for r in rows[arm] if r['sequence'] == seq]); compare_summary(saved['summary']['per_sequence'][seq], part, arm+'/'+seq)
            summary['per_sequence'][seq] = part
        summaries[arm] = summary
    for arm, summary in summaries.items():
        for region, values in summary['regions'].items():
            for ref in ('native_dense', 'tiled_all_keep', 'independent_zero', 'independent_mean'):
                values['delta_frame_mean_AEE_vs_'+ref] = difference(values['frame_mean_AEE'], summaries[ref]['regions'][region]['frame_mean_AEE'])
    return {'schema': 'flow_fast_runtime_summary_v1', 'status': 'AUDITED_SUMMARY_COMPLETE',
        'source_path': source_path, 'source_sha256': source_sha256, 'contract_sha256': data['contract_sha256'],
        'validation': {'complete_frames': 4, 'arm_count': 7, 'summaries_and_sequences_recomputed': True,
            'direction_strata_partition_all_GT': True, 'same_family_zero_trained_masks': True,
            'all_partial_arms_slow_points_900_FC_rows_9000': True, 'trained_fast_FC_LN_rows_3000': True,
            'first_frame_four_same_mask_references_exact': True, 'BN_buffers_unchanged': True,
            'same_preMLP_input_for_all_wrapped_arms': True, 'native_preMLP_input_independently_captured': False},
        'scope': 'Train-side backbone-seen four frames/two sequences; fit20-disjoint teacher-feature warmstart only, no backbone/gate recovery; not formal validation.',
        'audit_scope': 'Recorded sufficient statistics and execution evidence checked; GT/NPZ arrays not reopened and no inference performed.',
        'runtime': data['runtime'], 'wall_seconds': number(data['wall_seconds'], 'total CPU wall'),
        'targets': TARGETS, 'sequences': sorted({f['sequence'] for f in frames}), 'BN_scope': 'MLP_CMA176',
        'hardware_benefit_claim': False, 'statistical_significance_claim': False,
        'cost_scope': 'Equal slow work, not equal total cost. Motion/independent trained arms share slow/fast FC dimensions; controller work remains separately reported.',
        'summaries': summaries, 'per_frame_metrics': rows, 'per_frame_contrasts': pairs,
        'mean_contrasts': {name: {region: {'delta_AEE': mean(p['region_delta_AEE'][region] for p in pairs if p['contrast'] == name),
            'frames': sum(p['region_delta_AEE'][region] is not None for p in pairs if p['contrast'] == name)} for region in rows[ARMS[0]][0]['regions']}
            for name in sorted({p['contrast'] for p in pairs})},
        'cost_rows': costs, 'native_to_tiled_conversion': conversions, 'first_frame_references': references,
        'direction_scope': 'Posthoc logical tile-mean direction: >=1px, eight bins; not skip-risk truth or pixel correspondence. No GT enters online policy.',
        'provenance': {'checkpoint_sha256': CHECKPOINT, 'input_evidence': data['input_evidence'],
            'source_pins': data['contract']['source_pins'], **{k: data['contract'][k] for k in ('head_weights_sha256', 'head_fit_cache_sha256', 'fast_weights_sha256', 'fast_cache_sha256')}},
        'predictions_artifacts': {f['target_id']: f['predictions_artifact'] for f in frames}}


def prior_comparison(summary, prior, path, digest):
    require(prior.get('schema') == 'flow_runtime_group_cpu_results_v1' and prior.get('status') == 'RUNTIME_GROUP_CPU_COMPLETE', 'Prior result incomplete')
    require(prior['targets'] == TARGETS and prior['native_preflight']['checkpoint_sha256'] == CHECKPOINT
            and prior['native_preflight']['geometry'] == 'full9' and prior['BN_scope']['scope'] == 'MLP_CMA176', 'Prior comparison contract differs')
    require(prior['input_evidence'] == summary['provenance']['input_evidence'], 'Prior input evidence differs')
    rows = []
    for current, old in (('native_dense', 'native_dense'), ('tiled_all_keep', 'tiled_all_keep'),
                         ('independent_zero', 'independent'), ('motion_zero', 'motion')):
        for frame, previous in zip(summary['per_frame_metrics'][current], prior['frames']):
            metric = metric_frame(previous['arms'][old]['metrics'], previous['target_id'], previous['sequence'], REGIONS[:6])
            require(frame['target_id'] == metric['target_id'] and frame['sequence'] == metric['sequence'], 'Prior target order differs')
            deltas = {}; exact = True
            for region, value in metric['regions'].items():
                require(value['valid_pixels'] == frame['regions'][region]['valid_pixels'], 'Prior region population differs')
                exact &= value == frame['regions'][region]
                deltas[region] = difference(frame['regions'][region]['AEE'], value['AEE'])
            rows.append({'target_id': frame['target_id'], 'current_arm': current, 'prior_arm': old,
                         'recorded_metrics_exact': bool(exact), 'current_minus_prior_AEE': deltas})
    return {'source_path': path, 'source_sha256': digest, 'rows': rows,
        'all_recorded_common_region_metrics_exact': all(r['recorded_metrics_exact'] for r in rows),
        'note': 'Descriptive cross-run numerical comparison only; differences are reported, not forced to zero.'}


def markdown(value):
    def fmt(x): return 'null' if x is None else f'{x:.6f}'
    lines = ['# Fast 分支特征蒸馏后的整网 CPU 诊断', '',
        f"状态 `{value['status']}`；4 帧、{len(value['sequences'])} 个序列、7 臂；总 CPU 秒 {value['wall_seconds']:.3f}。",
        f"源结果 SHA `{value['source_sha256']}`。", '',
        '这是固定中期 backbone 上的 fit20 teacher-MLP 特征 warmstart，开发帧与 fast 拟合帧分离，但 backbone 已见这些训练侧序列。'
        '没有 gate/backbone 恢复，不是完整 DynamicSwin 复现，也不做总体或显著性结论。', '',
        'AEE 为非空帧等权平均；空区为 null。像素加权、所有细分区域、逐序列和逐帧数据保留在 JSON。'
        '方向正确/错误/弃权/GT 不适用是事后 GT tile 均值方向分区，不是跳过风险真值；其像素数和误差和均覆盖全部有效 GT。'
        'voxel_zero 不等于已核实的 raw 无事件。', '',
        '| 臂 | 区域 | 有区帧 | AEE | Δvs tiled全开 | Δvs I_zero | Δvs I_mean |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for region in REGIONS:
            r = value['summaries'][arm]['regions'][region]
            lines.append('| '+' | '.join([arm, region, str(r['frames_with_region']), fmt(r['frame_mean_AEE']),
                *(fmt(r['delta_frame_mean_AEE_vs_'+ref]) for ref in ('tiled_all_keep', 'independent_zero', 'independent_mean'))])+' |')
    lines += ['', '## 同帧对照', '', '以下为左臂减右臂；负数表示这些帧的误差较低。', '',
        '| 对照 | allGT ΔAEE | B ΔAEE | L ΔAEE | 方向错误 ΔAEE |', '|---|---:|---:|---:|---:|']
    for name, values in value['mean_contrasts'].items():
        lines.append('| '+name+' | '+' | '.join(fmt(values[r]['delta_AEE']) for r in ('all_valid', 'B', 'L', 'direction_wrong'))+' |')
    lines += ['', '## 实际执行服务', '',
        '全部局部臂均保留 900/1200 空间点；两条原生 FC 各执行 9000 行，T10/全通道保留。'
        'trained 额外执行 3000 行实值 LN+Linear（442,368,000 个实值 MAC）；mean 执行 1,152,000 个逻辑常量填充值；zero 没有 fast FC。'
        '这些臂只有慢分支工作量相同，总成本不同。M_trained 与 I_trained 的慢/快 FC 行数相同，控制器工作量仍分别记账。'
        '改动仅作用于 stage2 的第一个 MLP；attention、外部残差及完整网格上的其他 current-stat BN 仍执行。'
        '没有测量 PE 周期、访存事务或能耗，不可据此宣称全网省 25%。', '',
        '| Target | 臂 | 慢 tile调用 | 配对组 保留/可用 | 慢FC1/FC2行 | fast FC/LN行 | 全部可达规划+所选DP转移 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in value['cost_rows']:
        o = row['observed']
        lines.append(f"| {row['target_id']} | {row['arm']} | {row['retained_tile_calls']} | {row['paired_groups_selected']}/{row['paired_groups_available']} | {o['slow_fc1_rows']}/{o['slow_fc2_rows']} | {o['fast_fc_rows']}/{row['fast_LN_rows']} | {row['DP_transition_count_all_reachability_plus_one_selection']} |")
    lines += ['', '粗流 head 的 nominal real MAC、归约、GELU 及完整 DP 计数逐行保存在 JSON；没有把 head FC 混入慢/快分支 FC 行数。'
        'mean 填充计数是逻辑元素数，不是实测 SRAM 写事务。', '', '## 数值与来源核验', '',
        '首帧 independent_zero、independent_mean、independent_trained、motion_trained 的实际分支输出和整网预测均与同 mask 双分支 reference 精确一致。'
        '其他帧未额外重跑 reference。每帧 I/M 的 zero 与 trained mask 相同，I_mean 也同 mask；所有包装臂的输入 SHA 相同，BN 不变。', '',
        '| Target | native→tiled精确 | 最大预测差 | tiled−native allGT AEE |', '|---|---|---:|---:|']
    for row in value['native_to_tiled_conversion']:
        lines.append(f"| {row['target_id']} | {row['final_flow_exact']} | {fmt(row['final_flow_max_abs'])} | {fmt(row['allGT_AEE_tiled_minus_native'])} |")
    if 'prior_zero_comparison' in value:
        prior = value['prior_zero_comparison']
        lines += ['', f"与 10/13 既有零分支结果比较：公共区域逐帧 metrics 全部逐值相同 = `{prior['all_recorded_common_region_metrics_exact']}`。"
                  '差值保留在 JSON；不同不被强行归零。']
    lines += ['', '汇总器只重算保存的误差充分统计和执行记录，不重新读取 GT/预测 NPZ，也不运行网络。'
        '预测 NPZ 的已记录 SHA 原样保留，未在本汇总中重新计算。', '']
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--input-sha256')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--prior-results', type=Path); p.add_argument('--prior-sha256')
    a = p.parse_args(argv)
    require(not a.output_dir.exists(), 'Output directory already exists')
    actual = sha(a.input); require(a.input_sha256 is None or actual == a.input_sha256, 'Input SHA differs')
    value = summarize(json.loads(a.input.read_text()), str(a.input.resolve()), actual)
    if a.prior_results:
        prior_sha = sha(a.prior_results); require(a.prior_sha256 is None or prior_sha == a.prior_sha256, 'Prior SHA differs')
        value['prior_zero_comparison'] = prior_comparison(value, json.loads(a.prior_results.read_text()), str(a.prior_results.resolve()), prior_sha)
    else:
        require(a.prior_sha256 is None, '--prior-sha256 needs --prior-results')
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'summary.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    (a.output_dir/'SUMMARY.md').write_text(markdown(value))
    print(json.dumps({'status': value['status'], 'frames': 4, 'arms': 7, 'output_dir': str(a.output_dir)}))


if __name__ == '__main__':
    main()


