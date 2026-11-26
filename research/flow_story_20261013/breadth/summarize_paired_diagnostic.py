#!/usr/bin/env python3
"""Recompute fixed32 diagnostics against one explicit within-run baseline."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import summarize_breadth as common

HELPER_SHA = '05af4e8bae066a32e8be35c0c734958c7aa949efd46dbcfd3bd52e05821c5f9b'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--input-sha256', required=True)
    p.add_argument('--expected-schema', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--comparison-purpose', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    assert sha(common.__file__) == HELPER_SHA
    assert sha(a.input) == a.input_sha256
    d = json.loads(a.input.read_text())
    assert d['schema'] == a.expected_schema and d['status'] == 'BREADTH_CASES_FINISHED'
    targets = d['population']['target_ids']
    assert len(targets) == len(set(targets)) == 32
    assert d['population']['population'] == 'train_gate_development'
    assert len(d['cases']) == d['complete_cases'] == len(d['arms'])
    reference = d['arms'][a.baseline]
    assert reference['case']['method'] == 'dense'
    rows = []
    for case in d['cases']:
        entry = d['arms'][case['id']]
        assert entry['status'] == 'PASS' and entry['BN_buffers_unchanged'] is True
        assert [r['target_id'] for r in entry['frames']] == targets
        calculated = common.regions_for(entry['frames'])
        delta = common.pair_regions(entry['frames'], reference['frames'])
        for region, values in calculated.items():
            stored = entry['summary']['regions'][region]
            assert stored['valid_pixels'] == values['valid_pixels']
            for key in ['frame_mean_AEE', 'pixel_weighted_AEE']:
                x, y = stored[key], values[key]
                assert (x is None and y is None) or math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10)
            rows.append({'case_id': case['id'], 'BN': case['bn'], 'region': region,
                         **values, **delta[region]})
    result = {'schema': 'flow_explicit_paired_diagnostic_summary_v1',
              'input_file': str(a.input.resolve()), 'input_sha256': a.input_sha256,
              'source_schema': a.expected_schema, 'baseline_id': a.baseline,
              'comparison_purpose': a.comparison_purpose,
              'checkpoint_sha256': d['checkpoint_sha256'], 'geometry': d['geometry'],
              'target_ids': targets, 'complete_cases': len(d['cases']), 'regions': rows,
              'summary_script_sha256': sha(__file__), 'helper_sha256': HELPER_SHA,
              'scope': 'Mid-training checkpoint, fixed train-side32, no recovery; no final-test or hardware gain claim.'}
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (a.output_dir/'regions.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ['# 配对诊断结果', '', a.comparison_purpose, '',
             '共同中期权重、固定32个训练侧目标，零恢复。所有有效GT保留；不构成最终测试、完整方法复现或硬件收益。', '',
             '| Case | 全GT AEE | 相对指定dense的Δ | B区Δ | L区Δ |', '|---|---:|---:|---:|---:|']
    for case in d['cases']:
        by_region = {r['region']: r for r in rows if r['case_id'] == case['id']}
        values = [by_region['all_valid']['frame_mean_AEE']] + [by_region[k]['frame_mean_delta_AEE'] for k in ['all_valid', 'B', 'L']]
        lines.append('| '+case['id']+' | '+' | '.join(f'{v:.6f}' if v is not None else '—' for v in values)+' |')
    lines += ['', '其他全部分区、覆盖像素数、空帧、像素加权指标见regions.csv；L空区域帧不当作零误差。', '',
              f'输入 SHA：`{a.input_sha256}`；baseline：`{a.baseline}`。']
    (a.output_dir/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete_cases': len(d['cases']), 'regions': len(rows), 'baseline': a.baseline}))


if __name__ == '__main__':
    main()
