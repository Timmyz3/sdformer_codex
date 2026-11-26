#!/usr/bin/env python3
"""Standard-library audit/summary of one COMPLETE four-frame CPU skip pilot."""
import argparse
import hashlib
import json
import math
from pathlib import Path

POLICIES = ('independent', 'motion', 'rotated90', 'horizontal_local')
ARMS = ('native_dense', 'tiled_all_keep') + POLICIES
REGIONS = ('all_valid', 'B', 'L', 'B_unknown', 'voxel_zero', 'voxel_nonzero')
LAYER = 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp'
SHAPE = [10, 1, 30, 40, 384]
POINTS = [[81, 81, 81, 81, 36]]*3 + [[27, 27, 27, 27, 12]]
# First4 sequence-round-robin members of the sealed breadth gate_dev32 list
# (gate_dev32.json SHA41309684f927a10ef24b479178ec73a473da9d2ee31880011646f5f682d80509).
TARGETS = ['zurich_city_01_a_0083', 'zurich_city_05_a_0165',
           'zurich_city_01_a_0030', 'zurich_city_05_a_0352']


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


def metric_frame(row, target, sequence):
    require(row['target_id'] == target and row['sequence'] == sequence, 'Metric membership differs')
    regions = row['regions']
    require(set(REGIONS) <= set(regions), 'Required regions missing')
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


def execution(record, mode):
    records = record['execution']; require(len(records) == 1, 'Only one replaced MLP allowed')
    item = records[0]; capture = record['capture']
    require(item['layer_name'] == LAYER and item['shape_TBHWC'] == SHAPE and item['mode'] == mode
            and item['all_keep_shortcut'] is False and capture['shape'] == SHAPE, 'Execution layer/shape/mode differs')
    keep = item['keep']
    require(len(keep) == 4 and all(len(r) == 5 and all(type(x) is bool for x in r) for r in keep), 'Invalid mask')
    ids = [r*5+c for r in range(4) for c in range(5) if keep[r][c]]
    points = sum(POINTS[i//5][i%5] for i in ids)
    require(item['kept_real_pixels'] == points and item['kept_tiles'] == len(ids), 'Kept geometry differs')
    observed = item['observed']; rows = 12000 if mode == 'tiled_all_then_mask' else 10*points
    calls = 20 if mode == 'tiled_all_then_mask' else len(ids)
    require(observed == {'fc1_rows': rows, 'fc2_rows': rows, 'native_mlp_calls': calls}
            and record['native_FC_observed'] == observed and item['expected_fc_rows_per_layer'] == rows,
            'Observed native FC service differs')
    require(item['input_sha256'] == capture['input_sha256'] and item['output_sha256'] == capture['branch_sha256']
            and item['output_stride'] == capture['output_stride'], 'Wrapper/capture evidence differs')
    for key in ('input_sha256', 'branch_sha256'): hash_value(capture[key], key)
    number(record['wall_seconds_CPU'], 'inference wall')
    return item, ids, points


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


def selector_cost(record, policy, target, sequence):
    item, ids, points = execution(record, 'tiled_kept_only')
    source = record['runtime_selector']; proposal = source['proposal']; selection = source['selection']; common = source['common_budget']
    require(source['policy'] == policy and source['input_shape_TBHWC'] == SHAPE
            and source['input_sha256'] == record['capture']['input_sha256']
            and source['points_per_tile'] == POINTS and source['dense_spatial_points'] == 1200
            and source['nominal_slow_cap'] == 900 and source['GT_or_final_flow_used'] is False
            and source['upstream_attention_already_executed'] is True
            and source['all_four_proposals_computed_for_cost_matching'] is True, 'Runtime selector contract differs')
    require(selection['feasible'] is True and selection['slow_mask'] == item['keep']
            and selection['selected_tile_ids'] == ids and selection['slow_service_cost'] == points
            and selection['achieved_cost'] == points and selection['target_cost'] == points
            and common['target_cost'] == points and 0 < points <= 900
            and common['cap'] == 900 and common['unused_cap'] == 900-points, 'Exact actual work differs')
    reachable = common['per_policy_reachability']
    require(set(reachable) == set(POLICIES), 'Four reachability planners missing')
    common_set = set.intersection(*(set(reachable[p]['costs']) for p in POLICIES))
    require(common_set and max(common_set) == points and sorted(common_set) == common['common_reachable_costs'],
            'Common budget is not maximum common reachable cost')
    for value in reachable.values(): counter_check(value['counters'])
    counter_check(selection['counters'])
    reach_work = sum(v['counters']['transition_count'] for v in reachable.values())
    require(common['reachability_transition_count'] == reach_work, 'Reachability work omitted')
    groups = proposal['groups']; by_id = {g['group_id']: g for g in groups}
    require(len(by_id) == len(groups) and proposal['policy'] == policy, 'Group identity differs')
    flat = [i for g in groups for i in g['members']]
    require(sorted(flat) == list(range(20)) and all(len(g['members']) in (1, 2) for g in groups), 'Groups do not partition all20 tiles')
    for g in groups:
        require(g['slow_service_cost'] == sum(POINTS[i//5][i%5] for i in g['members']), 'Group tail cost differs')
    selected_ids = selection['selected_group_ids']
    require(len(set(selected_ids)) == len(selected_ids) and set(selected_ids) <= set(by_id), 'Bad selected groups')
    selected = [by_id[i] for i in selected_ids]
    require(sorted(i for g in selected for i in g['members']) == ids
            and selection['selected_group_count'] == len(selected)
            and selection['selected_tile_count'] == len(ids), 'Selection split/omitted group members')
    head = source['coarse_head_shape_cost']
    expected = {'input_shape_TBHWC': SHAPE, 'tile_count': 20, 'spatial_points_per_tile': POINTS,
        'temporal_reduction_additions': 9*1200*384, 'temporal_mean_divisions': 1200*384,
        'spatial_reduction_additions': 384*(1200-20), 'spatial_mean_divisions': 20*384,
        'linear_real_MACs': 20*(384*16+16*2), 'linear_bias_additions': 20*18,
        'GELU_scalar_evaluations': 20*16, 'costs_are_measured_cycles': False}
    require(all(head.get(k) == v for k, v in expected.items()), 'Head nominal cost differs from declared geometry')
    return {'target_id': target, 'sequence': sequence, 'policy': policy, 'common_actual_slow_points': points,
        'retained_point_fraction': points/1200, 'nominal_cap': 900, 'unused_cap': 900-points,
        'fc1_rows': item['observed']['fc1_rows'], 'fc2_rows': item['observed']['fc2_rows'],
        'retained_tile_calls': len(ids), 'available_tiles': 20,
        'selected_paired_groups': sum(len(g['members']) == 2 for g in selected),
        'available_paired_groups': sum(len(g['members']) == 2 for g in groups),
        'selected_groups': len(selected), 'available_groups': len(groups),
        'reachability_DP_all_four': {p: v['counters'] for p, v in reachable.items()},
        'selected_policy_DP': selection['counters'],
        'DP_transition_count_all_reachability_plus_one_selection': reach_work+selection['counters']['transition_count'],
        'head_shape_cost_nominal': head, 'controller_wall_seconds_CPU': number(source['controller_wall_seconds_CPU'], 'controller wall'),
        'inference_wall_seconds_CPU': record['wall_seconds_CPU'], 'PE_cycles_or_energy_measured': False}


def summarize(data, source_path=None, source_sha256=None):
    require(data.get('schema') == 'flow_runtime_group_cpu_results_v1'
            and data.get('status') == 'RUNTIME_GROUP_CPU_COMPLETE', 'Only COMPLETE runtime CPU pilot accepted')
    require(data.get('complete_frames') == 4 and data.get('inference_executed') is True
            and data.get('BN_buffers_unchanged') is True and data.get('training_performed') is False
            and data.get('hardware_claim') is False and data.get('upstream_attention_cancelled') is False,
            'Completion/invariance/scope differs')
    runtime = data['runtime']
    require(runtime['device'] == 'cpu' and runtime['GPU_initialized'] is False and runtime['autocast'] is False
            and runtime['TF32'] is False and data['BN_scope']['scope'] == 'MLP_CMA176', 'Runtime/BN scope differs')
    frames = data['frames']; targets = data['targets']
    require(len(frames) == len(targets) == len(set(targets)) == 4
            and targets == TARGETS and [f['target_id'] for f in frames] == targets
            and set(data['arms']) == set(ARMS), 'Fixed four-frame/six-arm membership differs')
    require(data['numerical_reference_frames'] == 1, 'Reference evidence coverage differs')
    summaries = {}; costs = []; numerical = []; refs = []; paired = []; metric_rows = {a: [] for a in ARMS}
    for frame_index, frame in enumerate(frames):
        target, seq = frame['target_id'], frame['sequence']
        require(target.startswith(seq+'_') and set(frame['arms']) == set(ARMS), 'Missing/foreign frame arm')
        evidence = data['input_evidence'][target]
        require(set(evidence) == {'events', 'gt', 'mask'}, 'Raw input evidence incomplete')
        for key, value in evidence.items(): hash_value(value, target+'.'+key)
        region_counts = None; input_hashes = set(); common_budgets = []
        for arm in ARMS:
            record = frame['arms'][arm]
            metric = metric_frame(record['metrics'], target, seq); metric_rows[arm].append(metric)
            counts = {k: v['valid_pixels'] for k, v in metric['regions'].items()}
            require(region_counts is None or counts == region_counts, 'Arms used different GT region populations')
            region_counts = counts
            capture = record['capture']; require(capture['shape'] == SHAPE, 'Captured target geometry differs')
            hash_value(capture['input_sha256'], 'upstream input'); input_hashes.add(capture['input_sha256'])
            if arm == 'native_dense':
                require(record['execution'] == [] and record['native_FC_observed'] == {'fc1_rows': 12000, 'fc2_rows': 12000}, 'Native service differs')
            elif arm == 'tiled_all_keep':
                _, kept, points = execution(record, 'tiled_all_then_mask')
                require(points == 1200 and len(kept) == 20, 'All-keep reference is partial')
            else:
                costs.append(selector_cost(record, arm, target, seq))
                common_budgets.append(record['runtime_selector']['common_budget'])
                if frame_index == 0:
                    reference = record['same_shape_reference']; other = reference['execution']
                    ref_item, _, _ = execution(other, 'tiled_all_then_mask')
                    require(reference['final_flow_exact'] is True and reference['final_flow_max_abs'] == 0
                            and record['capture'] == other['capture']
                            and ref_item['keep'] == record['execution'][0]['keep'], 'First-frame same-mask reference failed')
                    refs.append({'target_id': target, 'policy': arm, 'branch_exact': True,
                                 'final_flow_exact': True, 'max_abs': 0, 'input_sha256': capture['input_sha256']})
        require(len(input_hashes) == 1 and all(v == common_budgets[0] for v in common_budgets), 'Upstream input/common budgets differ')
        work = [frame['arms'][a]['execution'][0]['kept_real_pixels'] for a in POLICIES]
        require(len(set(work)) == 1 and frame['common_actual_slow_points'] == work[0], 'Policies used different real work')
        conversion = frame['native_vs_all_tiled']; maximum = number(conversion['final_flow_max_abs'], 'Native conversion maxabs')
        require(type(conversion['final_flow_exact']) is bool and conversion['final_flow_exact'] == (maximum == 0), 'Native conversion metadata differs')
        numerical.append({'target_id': target, 'sequence': seq, **conversion,
            'allGT_AEE_tiled_minus_native': difference(frame['arms']['tiled_all_keep']['metrics']['regions']['all_valid']['AEE'],
                                                     frame['arms']['native_dense']['metrics']['regions']['all_valid']['AEE'])})
        for control in ('independent', 'rotated90', 'horizontal_local'):
            paired.append({'target_id': target, 'sequence': seq, 'contrast': 'motion_minus_'+control,
                'region_delta_AEE': {r: difference(frame['arms']['motion']['metrics']['regions'][r]['AEE'],
                                                  frame['arms'][control]['metrics']['regions'][r]['AEE']) for r in REGIONS}})
    for arm in ARMS:
        saved = data['arms'][arm]
        require(saved['frames'] == metric_rows[arm], 'Per-arm frame rows differ from original per-frame metrics')
        computed = aggregate(metric_rows[arm]); compare_summary(saved['summary'], computed, arm)
        sequences = sorted({f['sequence'] for f in metric_rows[arm]})
        require(set(saved['summary']['per_sequence']) == set(sequences), 'Sequence summary population differs')
        computed['per_sequence'] = {}
        for sequence in sequences:
            subset = aggregate([f for f in metric_rows[arm] if f['sequence'] == sequence])
            compare_summary(saved['summary']['per_sequence'][sequence], subset, arm+'/'+sequence)
            computed['per_sequence'][sequence] = subset
        summaries[arm] = computed
    for arm, summary in summaries.items():
        for region, values in summary['regions'].items():
            for reference in ('native_dense', 'tiled_all_keep'):
                values['delta_frame_mean_AEE_vs_'+reference] = difference(values['frame_mean_AEE'], summaries[reference]['regions'][region]['frame_mean_AEE'])
    seqs = sorted({f['sequence'] for f in frames})
    return {'schema': 'flow_runtime_group_summary_v1', 'status': 'AUDITED_SUMMARY_COMPLETE',
        'source_path': source_path, 'source_sha256': source_sha256, 'contract_sha256': data['contract_sha256'],
        'audit_scope': 'Recomputed recorded sufficient statistics and checked stored execution evidence; no new raw-pixel/forward evaluation.',
        'validation': {'complete_frames': 4, 'arm_count': 6, 'stored_summaries_recomputed': True,
                       'all_four_policies_equal_actual_FC_rows_each_frame': True,
                       'first_frame_four_replayed_pairs_exact': True, 'same_upstream_inputs': True, 'BN_buffers_unchanged': True},
        'runtime': runtime, 'BN_scope': 'MLP_CMA176', 'targets': targets, 'sequences': seqs,
        'scope': 'Train-side backbone-seen 4-frame CPU zero-recovery exploration; not independent validation or a reproduction/ASIC result.',
        'hardware_benefit_claim': False, 'statistical_significance_claim': False,
        'metric_aggregation': 'Equal mean of nonempty per-frame mean EPE; pixel-weighted and sequence summaries retained separately.',
        'summaries': summaries, 'per_frame_motion_contrasts': paired,
        'mean_motion_contrasts': {c: {r: {'delta_AEE': mean(p['region_delta_AEE'][r] for p in paired if p['contrast'] == c),
                                                  'frames': sum(p['region_delta_AEE'][r] is not None for p in paired if p['contrast'] == c)} for r in REGIONS}
                                  for c in ('motion_minus_independent', 'motion_minus_rotated90', 'motion_minus_horizontal_local')},
        'cost_rows': costs, 'native_to_tiled_numerical_conversion': numerical, 'first_frame_same_shape_references': refs,
        'provenance': {'head_weights_sha256': data['contract']['head_weights_sha256'],
                       'head_fit_cache_sha256': data['contract']['head_fit_cache_sha256'],
                       'input_evidence': data['input_evidence'], 'runner_source_pins': data['contract']['source_pins']}}


def markdown(value):
    def fmt(x): return 'null' if x is None else f'{x:.6f}'
    lines = ['# 前置粗流指导单 MLP 取消：CPU pilot 汇总', '',
        f"状态：`{value['status']}`。4帧、{len(value['sequences'])}个序列、6臂；源 SHA `{value['source_sha256']}`。", '',
        '全部结果来自既有训练侧、backbone 已见的开发帧，零恢复；4帧并非4个独立序列，不做显著性、精度复现或硬件收益结论。'
        '以下仅复算记录的逐帧误差和/像素数，并核执行证据，没有重新读取 GT 或运行网络。', '',
        '主数为各非空帧 AEE 等权均值；空区 null，像素加权及逐序列数保存在 JSON。'
        'B/L 是固定诊断分区；voxel_zero 指编码 voxel 不活跃，不能等同独立核实的 raw 无事件。', '',
        '| 臂 | 区域 | 有区帧 | AEE | Δvs tiled全开 | Δvs native |',
        '|---|---|---:|---:|---:|---:|']
    for arm in ARMS:
        for region in REGIONS:
            r = value['summaries'][arm]['regions'][region]
            lines.append('| '+' | '.join([arm, '全部有效GT' if region == 'all_valid' else region,
                str(r['frames_with_region']), fmt(r['frame_mean_AEE']), fmt(r['delta_frame_mean_AEE_vs_tiled_all_keep']),
                fmt(r['delta_frame_mean_AEE_vs_native_dense'])])+' |')
    lines += ['', '## 同帧 motion 与对照', '',
        '差值为 motion − 对照；负数只表示这几个样本的 EPE 较低。各区域的逐帧差和平均差均保存在 JSON。', '',
        '| Target | 对照 | allGT | B | L | B_unknown | voxel_zero | voxel_nonzero |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for p in value['per_frame_motion_contrasts']:
        lines.append('| '+' | '.join([p['target_id'], p['contrast'][len('motion_minus_'):]]+[fmt(p['region_delta_AEE'][r]) for r in REGIONS])+' |')
    lines += ['', '## 实际执行与控制开销', '',
        '同帧四策略恰好保留同一真实点数；完整层为1200点、每个 FC 12000行。'
        '实际 tile 数可不同，不能从等点数推出相同发射/packing/控制成本。FC1与FC2行宽不同，下面每列是行数而非 MAC 或 PE 周期。', '',
        '| Target | 策略 | 保留点 | 实际tile调用/20 | FC1行 | FC2行 | 选中双点组/可用双点组 | DP transitions |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in value['cost_rows']:
        lines.append('| '+' | '.join([r['target_id'], r['policy'], str(r['common_actual_slow_points']),
            f"{r['retained_tile_calls']}/20", str(r['fc1_rows']), str(r['fc2_rows']),
            f"{r['selected_paired_groups']}/{r['available_paired_groups']}",
            str(r['DP_transition_count_all_reachability_plus_one_selection'])])+' |')
    cost = value['cost_rows'][0]['head_shape_cost_nominal']
    lines += ['', f"每次在线决策的 head 名义实数 MAC={cost['linear_real_MACs']}，时间归约加法={cost['temporal_reduction_additions']}，"
        f"空间归约加法={cost['spatial_reduction_additions']}，GELU标量={cost['GELU_scalar_evaluations']}；其余除法/偏置/缓冲计数见 JSON。",
        '这是形状算术，不是时钟测量。全部四策略 proposal/reachability 与当前策略 selection DP 都在线付费；'
        'DP 表含状态/转移/比较计数，但不能将 Python Fraction、分配、排序及访存成本视作零。'
        'head/控制器不混入目标 MLP 的 FC 行数。共同 cap900 可能有剩余额度；实用点数及余量逐帧保留。', '',
        '## 数值转换与 replay 范围', '',
        '首帧四策略均检查同 mask、同输入与分支 SHA、相同输出 stride 和最终 flow 精确一致。'
        '后3帧只有 actual 执行和服务审计，未额外声称它们逐帧 replay 等价。native 与 tiled全开差异独立列出，不能归因于分组选择。', '',
        '| Target | native→tiled全开 exact | 最大flow绝对差 | allGT ΔAEE |',
        '|---|---|---:|---:|']
    for r in value['native_to_tiled_numerical_conversion']:
        lines.append(f"| {r['target_id']} | {r['final_flow_exact']} | {fmt(r['final_flow_max_abs'])} | {fmt(r['allGT_AEE_tiled_minus_native'])} |")
    lines += ['', '仅 stage2 首 MLP 的完整T10、全部通道、9×9真尾块生产者被取消；attention和其他原生模块继续执行。'
        '外部 current-stat BN仍在完整网格上计算。该 CPU pilot 没有 RTL/DC/周期校准证据，不能报告加速器节省。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-sha256')
    parser.add_argument('--output-dir', type=Path, required=True)
    a = parser.parse_args()
    require(not a.output_dir.exists(), 'Output directory exists; preserve earlier summary')
    source_sha = sha(a.input)
    require(a.input_sha256 is None or source_sha == a.input_sha256, 'Input SHA differs')
    data = json.loads(a.input.read_text())
    result = summarize(data, str(a.input), source_sha)
    result['summarizer_sha256'] = sha(__file__)
    text = markdown(result)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (a.output_dir/'RUNTIME_GROUP_SUMMARY.md').write_text(text)
    print(json.dumps({'status': result['status'], 'output_dir': str(a.output_dir), 'frames': 4, 'arms': 6}))


if __name__ == '__main__':
    main()
