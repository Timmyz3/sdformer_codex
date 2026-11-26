#!/usr/bin/env python3
"""Read queue JSON evidence and produce a bounded recovery-quality snapshot.

Standard library only; never loads checkpoints, images, GT or GPU libraries.
Only whole completed candidate evaluations enter quality tables. Deltas require
the same stage, fixed member list, BN policy, region protocol and per-frame GT
coverage. Source results are never modified.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

ARMS = ('dense', 'lowrank_param25', 'svs_hidden25', 'magnitude_hidden25')
BNS = ('P_paper', 'D_candidate_CMA176')
START_SHA = 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
REGION_COLUMNS = ('B', 'L', 'B_unknown', 'voxel_zero')
POPULATION_DIGESTS = {
    'gate_dev32': '1158593d27329b8c2f59b9124cbc24e8cd15d473a0cef06f7fa3a31dd9fd619b',
    'check64': '0e0d5fc0ad7238575d7e714b8c788b21ced79c49ad5b35f1b3e2f663e2074a93',
}


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def finite(value): return type(value) in (int, float) and math.isfinite(value)


def near(a, b):
    return a is None and b is None or finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def read_source(path):
    if not path.exists(): return None, {'path': str(path), 'status': 'MISSING'}
    raw = path.read_bytes()
    evidence = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    try:
        value = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))
        require(isinstance(value, dict), 'JSON root must be an object')
        evidence.update(status='READ', reported_status=value.get('status'))
        return value, evidence
    except (ValueError, UnicodeDecodeError) as exc:
        evidence.update(status='INVALID_JSON', error=str(exc))
        return None, evidence


def metric_summary(group, ids):
    """Validate actual MetricAccumulator keys against its saved frame records."""
    frames = group['frames']; summary = group['summary']
    require([f['target_id'] for f in frames] == ids and len(ids) == len(set(ids)), 'Incomplete/changed frame identity')
    require(summary['frame_count'] == len(ids), 'Summary frame_count differs')
    regions = summary['regions']
    require(isinstance(regions, dict) and 'all_valid' in regions, 'Missing all_valid summary')
    coverage = []
    for frame in frames:
        require(set(frame['regions']) == set(regions), 'Frame region keys differ from summary')
        require(type(frame['regions']['all_valid']['valid_pixels']) is int and
                frame['regions']['all_valid']['valid_pixels'] > 0, 'Frame has no valid GT')
        coverage.append({'target_id': frame['target_id'], 'sequence': frame['sequence'],
                         'region_pixels': {k: v['valid_pixels'] for k, v in frame['regions'].items()}})
    for key, row in regions.items():
        parts = [f['regions'][key] for f in frames]
        for part in parts:
            n, error = part['valid_pixels'], part['error_sum']
            require(type(n) is int and n >= 0 and finite(error) and error >= 0, 'Invalid saved region counts/error')
            require(near(part['AEE'], error/n if n else None) and (n > 0 or error == 0), 'Frame AEE is not error_sum/count')
        present = [p for p in parts if p['valid_pixels']]
        pixels = sum(p['valid_pixels'] for p in parts); error = sum(p['error_sum'] for p in parts)
        require(row['frames_with_region'] == len(present) and row['empty_region_frames'] == len(parts)-len(present)
                and row['valid_pixels'] == pixels and near(row['error_sum'], error), 'Summary region coverage differs')
        require(near(row['frame_mean_AEE'], sum(p['AEE'] for p in present)/len(present) if present else None)
                and near(row['pixel_weighted_AEE'], error/pixels if pixels else None), 'Summary metric does not match frames')
    require(isinstance(summary.get('per_sequence'), dict), 'Missing actual per_sequence summary')
    require(sum(v['frame_count'] for v in summary['per_sequence'].values()) == len(ids), 'Sequence frame count differs')
    return summary, coverage


def candidate_rows(value, arm, stage, source):
    require(value.get('schema') == 'flow_recovery_candidate_evaluation_v1', 'Unknown candidate evaluation schema')
    require(value.get('status') == 'CANDIDATE_EVALUATION_COMPLETE', 'Whole candidate evaluation not complete')
    require(value['stage'] == stage and value.get('geometry') == 'full9', 'Stage/geometry differs')
    identity = value['identity']; training = identity['recovery_training']
    require(identity['arm'] == arm and identity['source_checkpoint_sha256'] == START_SHA and
            identity.get('final_valid_used') is False, 'Arm/common checkpoint/final-valid identity differs')
    checkpoint = identity['candidate_checkpoint_sha256']
    require(type(checkpoint) is str and len(checkpoint) == 64 and
            identity['candidate_sidecar']['checkpoint_sha256'] == checkpoint, 'Candidate checkpoint identity differs')
    expected_visits = 0 if stage == 'initial' else 4573
    require(training['visited_targets'] == expected_visits and
            type(training['completed_optimizer_steps']) is int and training['completed_optimizer_steps'] >= 0 and
            type(training['zero_valid_crops']) is int and training['zero_valid_crops'] >= 0 and
            training['completed_optimizer_steps']+training['zero_valid_crops'] == expected_visits,
            'Recovery fixed-visit/update accounting differs')
    populations = value['populations']; expected_pop = {'gate_dev32'} if stage == 'initial' else {'gate_dev32', 'check64'}
    require(set(populations) == expected_pop, 'Wrong declared populations for stage')
    all_ids = [t for ids in populations.values() for t in ids]
    require(len(all_ids) == len(set(all_ids)), 'Evaluation populations overlap')
    cal = value['calibration']
    require(value.get('candidate_BN_calibration_completed') is True and
            value.get('all_evaluation_BN_states_unchanged') is True and
            cal.get('non_BN_state_unchanged') is True and cal.get('count_per_BN') == 176 and
            len(cal['fit_ids']) == len(set(cal['fit_ids'])) == 176 and
            not set(cal['fit_ids']) & set(all_ids), 'Candidate-own BN176/immutability evidence incomplete')
    expected_groups = {b+'__'+p for b in BNS for p in populations}
    require(set(value['evaluations']) == expected_groups, 'Missing or extra BN/population group')
    preflight = value['native_preflight']
    # Actual frozen evaluator fields, not inferred from names or metric values.
    binding = {k: preflight.get(k) for k in ('data_contract', 'source_sha256', 'regions_sha256', 'region_native_source_binding')}
    require(all(binding[k] is not None for k in ('data_contract', 'source_sha256', 'regions_sha256')),
            'Missing data/source/region comparison binding')
    rows = []
    for population, ids in populations.items():
        require(len(ids) == (32 if population == 'gate_dev32' else 64) and
                digest(ids) == POPULATION_DIGESTS[population], 'Wrong fixed population identity/count')
        for bn in BNS:
            group = value['evaluations'][bn+'__'+population]
            require(group.get('status') == 'COMPLETE' and group['BN_policy'] == bn and
                    group['population'] == population, 'Incomplete/mislabeled evaluation group')
            summary, coverage = metric_summary(group, ids)
            rows.append({'arm': arm, 'stage': stage, 'population': population, 'BN_policy': bn,
                'status': 'COMPLETE', 'source': source, 'checkpoint_sha256': checkpoint,
                'source_checkpoint_sha256': START_SHA, 'training': training,
                'recovery_contract_sha256': identity['recovery_contract_sha256'],
                'architecture_sha256': identity['architecture_sha256'],
                'population_ids_sha256': digest(ids), 'coverage_sha256': digest(coverage),
                'comparison_binding_sha256': digest(binding), 'metric_summary': summary,
                'coverage': coverage, 'all_valid_frame_count': summary['regions']['all_valid']['frames_with_region'],
                'all_valid_pixel_count': summary['regions']['all_valid']['valid_pixels'],
                'coverage_scope': 'Saved per-frame all_valid GT counts complete; masks not independently reread.',
                'delta_reference': None, 'delta_regions': {}})
    return rows


def attach_deltas(rows):
    lookup = {(r['arm'], r['stage'], r['population'], r['BN_policy']): r for r in rows}
    for row in rows:
        dense = lookup.get(('dense', row['stage'], row['population'], row['BN_policy']))
        if dense is None:
            row['delta_status'] = 'PENDING_MATCHED_COMPLETE_DENSE'; continue
        checks = ('population_ids_sha256', 'coverage_sha256', 'comparison_binding_sha256', 'source_checkpoint_sha256')
        if any(row[k] != dense[k] for k in checks):
            row['delta_status'] = 'REFUSED_BINDING_OR_GT_COVERAGE_MISMATCH'; continue
        row['delta_status'] = 'MATCHED_DENSE'
        row['delta_reference'] = {k: dense[k] for k in
            ('arm', 'stage', 'population', 'BN_policy', 'checkpoint_sha256', 'source')}
        for region, part in row['metric_summary']['regions'].items():
            base = dense['metric_summary']['regions'][region]
            row['delta_regions'][region] = {metric: None if part[metric] is None else part[metric]-base[metric]
                for metric in ('frame_mean_AEE', 'pixel_weighted_AEE')}


def summarize(run_dir):
    run_dir = Path(run_dir).resolve(); sources = {}; completed = []; pending = []
    auxiliary = {}
    for name in ('queue_status.json', 'smoke_audit.json', 'svs_stability.json'):
        value, source = read_source(run_dir/name); sources[name] = source; auxiliary[name] = value
    for arm in ARMS:
        for stage in ('initial', 'endpoint'):
            name = arm+'_'+stage+'_eval.json'; value, source = read_source(run_dir/name); sources[name] = source
            state = {'arm': arm, 'stage': stage, 'source': source}
            if value is None:
                state.update(status=source['status'], reason='文件尚未生成' if source['status']=='MISSING' else source.get('error'))
                pending.append(state); continue
            if value.get('status') != 'CANDIDATE_EVALUATION_COMPLETE':
                state.update(status='PENDING', reported_status=value.get('status'), reason='候选评估未整体完成，已有局部数值不入表')
                pending.append(state); continue
            try:
                completed.extend(candidate_rows(value, arm, stage, source))
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                state.update(status='INVALID_COMPLETE_RESULT', reason=str(exc)); pending.append(state)
    attach_deltas(completed)
    queue = auxiliary['queue_status.json'] or {}; smoke = auxiliary['smoke_audit.json'] or {}
    stability = auxiliary['svs_stability.json'] or {}
    eval_complete = not pending and len(completed) == 24
    smoke_complete = smoke.get('status') == 'ALL_FOUR_ARMS_PASS' and {r.get('arm') for r in smoke.get('arms', [])} == set(ARMS)
    stability_complete = stability.get('schema') == 'flow_recovery_svs_stability_v1' and stability.get('sampling_only_comparison') is True
    deltas_complete = bool(completed) and all(r['delta_status']=='MATCHED_DENSE' for r in completed)
    all_complete = eval_complete and deltas_complete and smoke_complete and stability_complete and queue.get('experiment_status') == 'FOUR_ARMS_COMPLETE'
    return {'schema': 'flow_recovery_summary_v1',
        'status': 'RECOVERY_EXPERIMENT_COMPLETE' if all_complete else 'RECOVERY_EVIDENCE_INCOMPLETE',
        'snapshot_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'run_directory': str(run_dir),
        'completion': {'evaluations_complete': eval_complete, 'complete_candidate_files': 8-len(pending),
            'complete_quality_groups': len(completed), 'expected_quality_groups': 24,
            'matched_dense_deltas_complete': deltas_complete,
            'smoke_four_arms_complete': smoke_complete, 'SVS_sampling_stability_available': stability_complete,
            'queue_status': queue.get('status'), 'queue_experiment_status': queue.get('experiment_status'),
            'active_job': queue.get('active_job')},
        'source_checkpoint_sha256': START_SHA, 'source_checkpoint_optimizer_steps': 135930,
        'metric_definition': 'all_valid frame_mean_AEE is mean of frame-wise AEE over all valid GT; pixel_weighted_AEE reported separately.',
        'delta_definition': 'candidate minus dense, same stage/population/BN and matching member/GT coverage/region-source binding.',
        'scope': {'training_side_only': True, 'official_test_result': False, 'final_valid_used': False,
            'interim_checkpoint': True, 'paper_reproduction_claim': False, 'hardware_cycle_or_energy_claim': False,
            'endpoint_target_visits_per_arm': 4573, 'new_backbone_unseen_data_claim': False,
            'recovery_is_one_bounded_pass_not_full_original_method_schedule': True},
        'quality_rows': completed, 'pending_candidates': pending,
        'queue': queue, 'smoke': smoke, 'SVS_stability': stability,
        'sources': sources, 'summarizer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def number(value, signed=False):
    if value is None: return '空区域'
    return format(value, '+.6f' if signed else '.6f')


def quality_cell(row, region, metric='frame_mean_AEE'):
    part = row['metric_summary']['regions'].get(region)
    if part is None: return '未报告'
    value = number(part[metric])
    if row['delta_status'] == 'MATCHED_DENSE':
        return value+' / '+number(row['delta_regions'][region][metric], signed=True)
    return value+' / '+('待 dense' if row['delta_status'].startswith('PENDING') else '不可比')


def markdown(report):
    c = report['completion']; rows = report['quality_rows']
    lines = ['# 四臂有限恢复结果', '',
        '完成状态：'+('全部实验完成。' if report['status']=='RECOVERY_EXPERIMENT_COMPLETE' else '尚未全部完成；仅下表完整结果可用。'),
        '候选文件 {}/8，质量组 {}/24；队列状态 `{}`，当前任务 `{}`。'.format(
            c['complete_candidate_files'], c['complete_quality_groups'], c['queue_status'], c['active_job']), '',
        '共同起点是中途训练 checkpoint step 135930，SHA `{}`。恢复终点固定访问 4,573 个 gate-fit target；实际更新数与空 GT crop 数分别列出。这里是训练侧有限恢复探索，old dev32 已用于前期探索，check64 是后续开发检查；均不属于官方测试结果，也不支持论文复现、训练充分或未见数据泛化结论。'.format(START_SHA), '',
        '每格为 AEE / ΔAEE；Δ只相对同阶段、同集合、同 BN 的 dense。全 GT 主指标先逐帧求 AEE 再平均，像素加权指标单列。P_paper 与候选自身 CMA176 的 D 分开。GT 数量来自保存的逐帧统计，此入口不重读 GT；B_unknown 保留为未知，不能当作平坦区。', '']
    for stage, population in (('initial', 'gate_dev32'), ('endpoint', 'gate_dev32'), ('endpoint', 'check64')):
        for bn in BNS:
            group = [r for r in rows if (r['stage'],r['population'],r['BN_policy']) == (stage,population,bn)]
            title = ('初始' if stage=='initial' else '固定终点')+' · '+('old dev32' if population=='gate_dev32' else 'check64')+' · '+bn
            lines.extend(['## '+title, ''])
            if not group:
                lines.extend(['待完成。', '']); continue
            extra = [k for k in REGION_COLUMNS if any(k in r['metric_summary']['regions'] for r in group)]
            cols = ['臂', '全GT帧均 / Δ', '全GT像素加权 / Δ']+[k+'帧均 / Δ' for k in extra]+['有效GT 帧/像素']
            lines += ['| '+' | '.join(cols)+' |', '| '+' | '.join(['---']*len(cols))+' |']
            for arm in ARMS:
                row = next((r for r in group if r['arm']==arm), None)
                if row is None: continue
                cells = [arm, quality_cell(row,'all_valid'), quality_cell(row,'all_valid','pixel_weighted_AEE')]
                cells += [quality_cell(row,k) for k in extra]
                cells += [str(row['all_valid_frame_count'])+'/'+str(row['all_valid_pixel_count'])]
                lines.append('| '+' | '.join(cells)+' |')
            lines.append('')
            support = group[0]['metric_summary']['regions']
            lines.append('分区覆盖：'+'；'.join('{}：{} 帧 / {} 像素'.format(k,support[k]['frames_with_region'],support[k]['valid_pixels']) for k in extra)+'。')
            absent = [a for a in ARMS if not any(r['arm']==a for r in group)]
            if absent: lines.append('待完成：'+', '.join(absent)+'。')
            lines.append('')
    lines += ['## 检查点与恢复账', '', '| 臂 | 阶段 | target访问 | 更新 | 空GT crop | 候选checkpoint SHA256 |', '| --- | --- | --- | --- | --- | --- |']
    seen = set()
    for r in rows:
        key = (r['arm'],r['stage'])
        if key in seen: continue
        seen.add(key); t=r['training']
        lines.append('| {} | {} | {} | {} | {} | `{}` |'.format(*key,t['visited_targets'],t['completed_optimizer_steps'],t['zero_valid_crops'],r['checkpoint_sha256']))
    lines += ['', 'Smoke：`{}`；SVS fit2→fit20 稳定性：{}。'.format(report['smoke'].get('status','待完成'),'已生成' if c['SVS_sampling_stability_available'] else '待完成')]
    aggregate = report['SVS_stability'].get('aggregate_channel_overlap_not_PE_cost', {})
    if aggregate:
        lines.append('SVS 保留通道重合率（通道数口径）：'+'；'.join('{}：{:.6f}'.format(k,v['channel_micro_kept_overlap']) for k,v in aggregate.items())+'。这不是精度或 PE 服务收益。')
    if report['pending_candidates']:
        lines += ['', '待完成或需核对的文件：', '']
        for p in report['pending_candidates']:
            lines.append('- `{}` {}：{}。'.format(p['arm'],p['stage'],str(p['reason']).replace('\n',' ')))
    lines += ['', '本汇总不报告 ASIC 周期、能耗或加速比。全部实际分区、逐序列统计、输入文件 SHA 和队列/Smoke/SVS 原始字段保存在相邻 JSON 中。', '']
    return '\n'.join(lines)


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(text, encoding='utf-8'); os.replace(temp,path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path)
    a=p.parse_args(argv); require(a.run_dir.is_dir(), 'Run directory does not exist')
    out=a.output_dir or a.run_dir/'summary'; report=summarize(a.run_dir)
    atomic(out/'recovery_summary.json',json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    atomic(out/'RECOVERY_SUMMARY.md',markdown(report))
    print(json.dumps({'status':report['status'],'completion':report['completion'],'output_dir':str(out)},ensure_ascii=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
