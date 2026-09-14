"""Join actual component service and separate full-network quality records."""
from pathlib import Path
import csv
import json

H = Path(__file__).resolve().parent


def read(path):
    return json.loads((H / path).read_text())


def main():
    quality = read('quality/comparison.json')['means']
    base = read('spatial_winograd/comparison.json')['sets']
    pruned = read('spatial_winograd/pruned_replay/comparison.json')['arms']
    moment = read('spatial_moment3/comparison.json')['comparisons']
    rows = []

    def add(function, implementation, workload, stall, cycles, source):
        rows.append(dict(function=function, implementation=implementation,
                         workload=workload, backpressure=bool(stall),
                         cold_component_cycles=cycles,
                         valid825_frame_AEE=quality[function]['frame_mean'],
                         valid825_pixel_AEE=quality[function]['pixel_mean'],
                         evidence=source))

    for workload in ('sequences', 'held', 'disjoint'):
        for row in base[workload]['services']:
            if row['model_resident']:
                continue
            for name, key in [('expanded_OS', 'direct_os'), ('ordinary_factor', 'ordinary'), ('general_Winograd', 'winograd')]:
                add('spatial_q11', name, workload, row['stall'], row[key], 'spatial_winograd/comparison.json')
        for function in ('moment', 'native_tap'):
            for row in pruned[function]['sets'][workload]['services']:
                if row['model_resident']:
                    continue
                for name, key in [('ordinary_factor', 'ordinary'), ('general_Winograd', 'general_winograd')]:
                    add(function, name, workload, row['stall'], row[key], 'spatial_winograd/pruned_replay/comparison.json')
        for row in moment:
            if row['stage'] == workload and row['consumer']:
                add('moment', 'three_product', workload, row['stall'], row['moment3'], 'spatial_moment3/comparison.json')
        control = read('spatial_moment3/unconstrained/SUMMARY.json')['results']['consumer']['sets'][workload]['summaries']
        for row in control:
            if row['repeat'] == 0:
                add('unconstrained', 'three_product', workload, row['stall'], row['service'], 'spatial_moment3/unconstrained/SUMMARY.json')
        box = read('spatial_box2/SUMMARY.json')['results']['consumer']['sets'][workload]['summaries']
        for row in box:
            if row['repeat'] == 0:
                add('moment', 'source_box_two_tap', workload, row['stall'], row['service'], 'spatial_box2/SUMMARY.json')
    with (H / 'component_quality.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = [row for row in rows if row['workload'] == 'sequences' and not row['backpressure']]
    text = ['# 跨18序列固定36tile：组件服务与独立网络质量', '',
            '周期覆盖固定36tile的原生source到最后I24，模型加载一次、逐tile加载source/origin/start；AEE覆盖完整825帧网络，两个覆盖范围不同。不同function行是有损方案取舍，不是同函数加速比。', '',
            '| 函数 | 执行 | 组件周期 | 825逐帧AEE |', '|---|---|---:|---:|']
    for row in selected:
        text.append(f"| {row['function']} | {row['implementation']} | {row['cold_component_cycles']:,} | {row['valid825_frame_AEE']:.6f} |")
    text += ['', '同环境NB0逐帧AEE为1.447936665574317；候选使用既有matched-dense学生/粗头，整段优势不能归因于本次剪枝或分解。', '',
             '全部周期来自Verilator组件仿真；共同算术与服务合同不等于独立裁剪后同面积或同Fmax。展开OS权重容量、通用Winograd额外状态、各路径所有配置/恢复费用见各报告。无VCS/DC/PT/Formality/PPA、无整网RTL周期或FPS。', '',
             'unconstrained为两相位函数，已与其独立展开gold核对；尚未实现同函数直接两相位RTL。moment与unconstrained的三项硬件相同，不把控制少一项说成moment独占。', '',
             '[完整ready/BP及两套同帧64tile表](component_quality.csv)、[825质量明细](quality/QUALITY_REPORT.md)、[独立评审](review_stage_final.md)。']
    (H / 'NET_RESULT.md').write_text('\n'.join(text) + '\n')
    print(f'Wrote {len(rows)} measured component/quality rows; no cross-function speedup claimed.')


if __name__ == '__main__':
    main()
