#!/usr/bin/env python3
"""Posthoc four-state descriptions; no policy optimization or deployment claim."""
import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_SHA = '0a4c7f16b346d19b614cacf86739436adc278675bb57b21b13fd80d3f80b8dfe'
REGIONS = ('all_valid', 'B', 'L', 'B_unknown')
CONDITIONS = ('motion', 'rotated90', 'horizontal_local')
EPS = 1e-6  # numerical comparison tolerance, not an accepted application AEE budget


def avg(xs):
    return math.fsum(xs)/len(xs) if xs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA and not args.output_dir.exists()
    data = json.loads(args.input.read_text())
    assert data['status'] == 'PAIR_DIAGNOSTIC_COMPLETE' and data['complete_frames'] == 8
    rows = []
    for frame in data['frames']:
        for layer in frame['layers']:
            stage = int(layer['layer'].split('.layers.')[1].split('.')[0])
            for pair in layer['pairs']:
                for condition in CONDITIONS:
                    body = pair['conditions'][condition]
                    for region in REGIONS:
                        value = body['regions'][region]
                        if not value['valid_pixels']:
                            continue
                        ae = value['AEE']
                        da, db, dab = (ae[k]-ae['base'] for k in ('a', 'b', 'ab'))
                        assert abs(dab-da-db-value['interaction']) < 1e-10
                        joint = dab > EPS
                        row = {
                            'target_id': frame['target_id'], 'sequence': frame['sequence'],
                            'stage': stage, 'anchor_yx': pair['anchor_yx'],
                            'condition': condition, 'region': region,
                            'delta_a': da, 'delta_b': db, 'delta_ab': dab,
                            'interaction': value['interaction'],
                            'joint_higher_than_dense': joint,
                            'both_single_not_higher_joint_higher': da <= EPS and db <= EPS and joint,
                            'joint_higher_one_slow_can_restore_dense': joint and min(da, db) <= EPS,
                            'positive_interaction': value['interaction'] > EPS,
                            'best_single_delta_GT_oracle': min(da, db),
                            'oracle_one_slow_improvement_over_both_fast': dab-min(da, db),
                            'duplicate_control_endpoints': body['same_pair_as_other_control'],
                        }
                        rows.append(row)
    categories = ('joint_higher_than_dense', 'both_single_not_higher_joint_higher',
                  'joint_higher_one_slow_can_restore_dense', 'positive_interaction')
    groups = []
    for stage in (0, 2):
        for condition in CONDITIONS:
            for region in REGIONS:
                chosen = [r for r in rows if (r['stage'],r['condition'],r['region']) == (stage,condition,region)]
                frames = []
                for target in data['targets']:
                    frows = [r for r in chosen if r['target_id'] == target]
                    if not frows:
                        continue
                    frames.append({'target_id': target, 'sequence': frows[0]['sequence'],
                        'anchors': len(frows), 'category_fractions': {k:avg([float(r[k]) for r in frows]) for k in categories},
                        'delta_ab_mean': avg([r['delta_ab'] for r in frows]),
                        'interaction_mean': avg([r['interaction'] for r in frows])})
                groups.append({'stage': stage, 'condition': condition, 'region': region,
                    'anchor_count_descriptive_only': len(chosen), 'nonempty_frames': len(frames),
                    'raw_category_counts': {k:sum(r[k] for r in chosen) for k in categories},
                    'frame_equal_category_fractions': {k:avg([f['category_fractions'][k] for f in frames]) for k in categories},
                    'per_frame': frames})
    result = {'schema': 'flow_four_state_posthoc_description_v1', 'source_sha256': digest,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'tolerance_AEE_pixels': EPS, 'analysis_pre_registered': False,
        'quality_budget_or_significance_claim': False, 'runtime_selection_or_hardware_claim': False,
        'scope': 'Existing offline teacher-guided zero-output interventions; same repeatedly explored dev8; no new forward',
        'state_definition': 'base=both slow, a=only a fast/zero, b=only b fast/zero, ab=both fast/zero',
        'oracle_warning': 'Best single uses GT after inference and need not be predictable. One-slow costs more than both-fast; counts are not a matched-cost gain.',
        'confound_warning': 'Includes current-stat BN and downstream nonlinearities. Ordinary nearby pairs can also interact. Duplicate endpoints are not independent evidence.',
        'groups': groups, 'rows': rows}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'four_states.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines = ['# 四状态后验描述', '',
        '这些是对既有8帧诊断的后验描述，不是新前向、独立验证、可部署选择或同成本收益。', '',
        '“不高于dense”仅指该分区AEE差≤1e−6像素的数值分类，不等于通过应用质量门。每个anchor只用于覆盖计数，不作为独立统计样本。', '',
        '| Stage | 方向 | 区域 | 非空帧/anchor | 联合高于dense | 两个单独均不高、联合高 | 联合高但保留一个可恢复 | 正交互 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        n=g['raw_category_counts']
        lines.append('| '+' | '.join([str(g['stage']),g['condition'],g['region'],
            f"{g['nonempty_frames']}/{g['anchor_count_descriptive_only']}"]+[str(n[k]) for k in categories])+' |')
    lines += ['',
        '例如stage2/L运动组中，8个联合高于dense的anchor里有6个可由GT事后选出的单块保留恢复到dense以内。这提示要检验“至少保一块”，并不证明哪块应由运行时保留，也不证明运动配对比普通邻域更好。', '',
        '该分类仍有强反例：stage0全GT中运动组有5/16个“两单独不高、联合高”，水平控制也有4/16个；stage2全GT没有此类anchor。不可将层/区域条件推广为普遍规律。', '',
        '完整逐anchor、逐帧和帧等权类别比例保存在four_states.json。部署结论还需同fast、同共同候选支持、充分恢复、运行时无GT与同实际总成本对照。', '']
    (args.output_dir/'FOUR_STATES.md').write_text('\n'.join(lines))
    print(json.dumps({'status':'COMPLETE','source_sha256':digest,'rows':len(rows),'groups':len(groups)}))


if __name__ == '__main__':
    main()
