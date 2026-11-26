#!/usr/bin/env python3
"""CPU/stdlib shape ledger; no model, checkpoint, data array, CUDA or simulator."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HEAD = HERE / 'coarse_head.py'
TRACE = REPO / 'research/flow_story_20261012/deployment_contract/whole_model_tile_probe.json'
MLP_SOURCE = REPO / 'SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py'
PSN_SOURCE = MLP_SOURCE.with_name('Spiking_submodules.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_record(path):
    path = path.resolve()
    raw = path.read_bytes()
    return {'path': str(path.relative_to(REPO)), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def source_counter():
    # Execute only these three pure shape helpers; never import torch/coarse_head.
    selected = {'_require', '_shape', 'shape_operation_counts'}
    tree = ast.parse(HEAD.read_text(), filename=str(HEAD))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    require({n.name for n in nodes} == selected, 'Pure source counter helpers changed')
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HEAD), 'exec'), namespace)
    return namespace['shape_operation_counts']


def audit_stage(layer, stage, counter):
    t, b, h, w, c = layer['shape_TBHWC']
    f = layer['hidden_channels']
    require(t == 10 and b == 1 and layer['tile_hw'] == [9, 9], 'Wrong native geometry')
    require(f == 4*c, 'This audit covers original native MLP, not a narrowed/factored candidate')
    areas = [min(9, h-y)*min(9, w-x) for y in range(0, h, 9) for x in range(0, w, 9)]
    n, points = len(areas), h*w
    require(sum(areas) == points and layer['kept_real_pixels'] == points, 'Tail/all-keep coverage differs')
    require(layer['observed']['fc1_rows'] == layer['observed']['fc2_rows'] == t*points, 'Actual FC hooks differ')
    counts = counter(layer['shape_TBHWC'])
    expected = {
        'temporal_reduction_additions': (t-1)*points*c,
        'temporal_mean_divisions': points*c,
        'spatial_reduction_additions': sum(v-1 for v in areas)*c,
        'spatial_mean_divisions': n*c,
        'linear_real_MACs': n*(c*16+16*2),
        'linear_bias_additions': n*18,
        'GELU_scalar_evaluations': n*16,
        'trainable_parameters': c*16+16+16*2+2,
    }
    require(all(counts[k] == v for k, v in expected.items()), 'Independent/source operation count mismatch')
    require([v for row in counts['spatial_points_per_tile'] for v in row] == areas, 'Source tail sizes differ')
    require(counts['output_shape_BtilesYtilesX2'] == [1, (h+8)//9, (w+8)//9, 2], 'Head output shape differs')
    source_tensor_ledger = [
        ('u', t*points*c, t*points*c, 0),
        ('temporal_mean', points*c, points*c, points*c),
        ('pooled_C_descriptor', n*c, n*c, n*c),
        ('linear1_output', n*16, n*16, n*16),
        ('GELU_output', n*16, n*16, n*16),
        ('flow_dxdy', n*2, 0, n*2),
    ]
    tensor_ledger = [{'tensor': name, 'elements': elems, 'logical_payload_bytes_FP32': elems*4,
                     'logical_read_bytes_FP32': reads*4, 'logical_write_bytes_FP32': writes*4}
                    for name, elems, reads, writes in source_tensor_ledger]
    return {
        'stage': stage, 'layer_name': layer['layer_name'], 'shape_TBHWC': layer['shape_TBHWC'], 'hidden_F': f,
        'tile_count': n, 'real_spatial_points': points,
        'tile_area_histogram': {str(k): v for k, v in sorted(Counter(areas).items(), reverse=True)},
        'source_counter_matches_independent_formulas': True,
        'head': {**expected, 'aggregate_additions': expected['temporal_reduction_additions']+expected['spatial_reduction_additions'],
                 'mean_divisions_or_reciprocal_multiplications': expected['temporal_mean_divisions']+expected['spatial_mean_divisions'],
                 'unique_parameter_payload_bytes_FP32': expected['trainable_parameters']*4},
        'native_slow_MLP_one_block': {
            'PSN1_real_MAC': t*t*points*c, 'PSN2_real_MAC': t*t*points*f,
            'PSN_real_MAC_total': t*t*points*(c+f),
            'FC1_binary_weighted_AC_terms': t*points*c*f,
            'FC2_binary_weighted_AC_terms': t*points*f*c,
            'FC_binary_weighted_AC_terms_total': 2*t*points*c*f,
            'actual_FC_rows_each_from_prior_GPU_hooks': t*points,
            'PSN_threshold_elements': t*points*(c+f),
            'BN_affine_elements': t*points*(c+f), 'residual_add_elements': t*points*c,
            'scope': 'Nominal dense terms, not activity-weighted issued AC or cycles; BN/threshold implementation unresolved',
        },
        'head_source_materialized_tensor_ledger': tensor_ledger,
        'head_logical_tensor_read_bytes_FP32': sum(x['logical_read_bytes_FP32'] for x in tensor_ledger),
        'head_logical_tensor_write_bytes_FP32': sum(x['logical_write_bytes_FP32'] for x in tensor_ledger),
        'descriptor_accounting_assumption': {
            'scope': 'Proposed logical array ABI, not current Python object storage or measured SRAM traffic',
            'flow_read_bytes_FP32x2_once': 8*n,
            'existing_score_read_bytes_FP32_once': 4*n,
            'slow_cost_read_bytes_UINT64_once': 8*n,
            'membership_UINT32_ids': n, 'membership_write_bytes_once': 4*n, 'membership_read_bytes_once': 4*n,
            'decision_bitmap_bits': n, 'bitmap_write_bytes_once': (n+7)//8, 'bitmap_read_bytes_once': (n+7)//8,
            'group_record_count_range': [(n+1)//2, n],
            'member_issue_descriptor_count': n, 'member_issue_descriptor_bytes': None,
            'omitted': ['Group record/address handle field widths', 'sorting/quantization/conflict work',
                        'repeated scans, bank alignment and transfers', 'score producer', 'queue/backpressure state'],
            'no_double_count': 'flow_dxdy write is already in head tensor ledger; flow read here belongs to policy consumer',
        },
    }


def audit():
    trace = json.loads(TRACE.read_text())
    require(trace['status'] == 'ENGINEERING_PROBE_COMPLETE' and trace['geometry'] == [480, 640]
            and trace['window'] == [2, 9, 9], 'Expected old full9 executed trace')
    all_keep = next(c for c in trace['cases'] if c['pattern'] == 'all_keep')
    layers = all_keep['arms']['actual_skip']['layers']
    require(len(layers) == 12, 'Expected all twelve executed native MLPs')
    counter = source_counter()
    stages = []
    for stage in (0, 2):
        matches = [x for x in layers if x['layer_name'].endswith(f'.layers.{stage}.swin_blocks.0.mlp')]
        require(len(matches) == 1, 'Representative stage block missing')
        layer = matches[0]
        # The native graph dimensions must agree in every saved arm/case.
        for case in trace['cases']:
            for arm in case['arms'].values():
                other = next(x for x in arm['layers'] if x['layer_name'] == layer['layer_name'])
                require(other['shape_TBHWC'] == layer['shape_TBHWC'] and other['hidden_channels'] == layer['hidden_channels'],
                        'Old trace geometry differs across cases')
        stages.append(audit_stage(layer, stage, counter))
    require(stages[0]['shape_TBHWC'] == [10, 1, 120, 160, 96], 'Stage0 contract changed')
    return {
        'schema': 'flow_coarse_head_shape_audit_v1', 'status': 'CPU_SHAPE_AUDIT_PASS',
        'execution': 'stdlib only; source pure counter AST + independent arithmetic; no PyTorch forward/GPU/checkpoint load',
        'sources': [source_record(p) for p in (Path(__file__), HEAD, TRACE, MLP_SOURCE, PSN_SOURCE)],
        'geometry_trace_provenance': {k: trace[k] for k in
            ('scope', 'checkpoint_sha256', 'training', 'input_sha256', 'target_id', 'script_sha256', 'tile_module_sha256')},
        'stages': stages,
        'measured_cycles': False, 'net_speedup_claim': False, 'quantized_precision_verified': False,
    }


def markdown(result):
    a, b = result['stages']
    def fmt(v):
        return f'{v:,}' if isinstance(v, int) else str(v)
    def table_row(label, get):
        return '| '+label+' | '+fmt(get(a))+' | '+fmt(get(b))+' |'
    lines = ['# 粗流头与原生 MLP 的 CPU 形状账本', '',
        '2026-10-13。`CPU_SHAPE_AUDIT_PASS`；仅标准库读取源码/旧 trace、执行纯形状函数并独立复算，没有 PyTorch forward、GPU、checkpoint 或输入数组读取。没有测量周期、能耗或净加速。粗流头仍为未训练原型，本账本不含预测质量或真实门控结果。', '',
        '每列为一个 stage 的一个 MLP block。stage2 尺寸从旧 full9 真实 forward trace 取出，并核对各 arm/case 一致；该 trace 是单输入、36-step 工程权重，只借用几何/FC hooks，不继承精度或训练充分性结论。', '',
        '| 项目 | stage0 | stage2 |', '| --- | ---: | ---: |',
        table_row('u 形状 [T,B,H,W,C]', lambda s: s['shape_TBHWC']),
        table_row('原生 hidden F', lambda s: s['hidden_F']),
        table_row('真实空间点 / tile 数', lambda s: f"{s['real_spatial_points']:,} / {s['tile_count']}"),
        table_row('tile 面积:个数', lambda s: s['tile_area_histogram']),
        table_row('head 两 Linear real MAC', lambda s: s['head']['linear_real_MACs']),
        table_row('head Tmean + tilemean 加法', lambda s: s['head']['aggregate_additions']),
        table_row('head mean 除法或倒数乘法', lambda s: s['head']['mean_divisions_or_reciprocal_multiplications']),
        table_row('head bias 加法 / GELU 标量', lambda s: f"{s['head']['linear_bias_additions']:,} / {s['head']['GELU_scalar_evaluations']:,}"),
        table_row('head 参数个数 / FP32载荷 byte', lambda s: f"{s['head']['trainable_parameters']:,} / {s['head']['unique_parameter_payload_bytes_FP32']:,}"),
        table_row('slow PSN1 / PSN2 real MAC', lambda s: f"{s['native_slow_MLP_one_block']['PSN1_real_MAC']:,} / {s['native_slow_MLP_one_block']['PSN2_real_MAC']:,}"),
        table_row('slow 两 FC binary weighted AC 名义项', lambda s: s['native_slow_MLP_one_block']['FC_binary_weighted_AC_terms_total']),
        table_row('实际旧 hooks：每个 FC 的输入行数', lambda s: s['native_slow_MLP_one_block']['actual_FC_rows_each_from_prior_GPU_hooks']),
        table_row('slow BN 仿射元素 / residual 加法', lambda s: f"{s['native_slow_MLP_one_block']['BN_affine_elements']:,} / {s['native_slow_MLP_one_block']['residual_add_elements']:,}"), '',
        '公式：head MAC=`N(C·16+16·2)`；聚合加法=`(T−1)HWC+C(HW−N)`；mean除法=`HWC+NC`。slow PSN real MAC=`T²HW(C+F)`；两FC名义AC=`2THWCF`。这些操作种类不能合并成同成本MAC，更不能相除产生周期加速比。GELU内部实现未展开。', '',
        '## 逻辑物化与描述符', '',
        '以下 FP32 张量表采用与源码表达式对应的分步物化假设：Tmean、tilemean、Linear、GELU、Linear。每张量一次生产/消费；不含权重反复读取或可能的contiguous/layout复制。它是审查用逻辑载荷，不是实际 SRAM/DMA 字节、缓存行为或峰值存活量。', '',
        '| 张量/流量 | stage0 byte | stage2 byte |', '| --- | ---: | ---: |']
    for i, row in enumerate(a['head_source_materialized_tensor_ledger']):
        lines.append(table_row(row['tensor']+' 载荷', lambda s, i=i: s['head_source_materialized_tensor_ledger'][i]['logical_payload_bytes_FP32']))
    lines += [table_row('分步逻辑读总量（无参数）', lambda s: s['head_logical_tensor_read_bytes_FP32']),
              table_row('分步逻辑写总量（无参数）', lambda s: s['head_logical_tensor_write_bytes_FP32']), '',
        '普通融合可以边产生 u 边累计各 tile/channel 的完整T统计，消除部分中间物化或额外回读；必须给独立/配对两方同样机会。有限缓冲、到达顺序和浮点归约次序改变仍待验证，不能将本表全部读写视作不可避免开销。u 的生产者及完整T到达均已付出，粗流头不能反过来取消它。', '',
        '描述符另用显式假设：flow为FP32×2，已有score为FP32，slow cost为uint64，member ID为uint32，keep为bit；均一次读/写。当前Python字典没有这个硬件ABI，本表不声称已实际物化。', '',
        '| 逻辑策略接口 | stage0 | stage2 |', '| --- | ---: | ---: |',
        table_row('flow / score / cost 一次读 byte', lambda s: ' / '.join(fmt(s['descriptor_accounting_assumption'][k]) for k in
            ('flow_read_bytes_FP32x2_once','existing_score_read_bytes_FP32_once','slow_cost_read_bytes_UINT64_once'))),
        table_row('member ID 一次写 / 一次读 byte', lambda s: f"{s['descriptor_accounting_assumption']['membership_write_bytes_once']:,} / {s['descriptor_accounting_assumption']['membership_read_bytes_once']:,}"),
        table_row('keep bitmap 一次写 / 一次读 byte', lambda s: f"{s['descriptor_accounting_assumption']['bitmap_write_bytes_once']} / {s['descriptor_accounting_assumption']['bitmap_read_bytes_once']}"),
        table_row('group记录数范围 / member issue记录数', lambda s: f"{s['descriptor_accounting_assumption']['group_record_count_range']} / {s['tile_count']}"), '',
        'flow 写已计在 head 张量表，策略端只追加读取，不重复计写；地址句柄/组记录的编码宽度尚未定，完整 issue descriptor 字节留空。配对冲突、排序、阈值、预算扫描次数及反压没有被此表收费完成。', '',
        '## 结论边界与下一步接入', '',
        '- head 的 Linear MAC 较少不等于总开销小：必须保留约化、GELU、u就绪等待和真实端口/缓存费用。',
        '- slow MLP 的 BN、阈值、bias、PSN状态/布局和残差仍需具体实现；外部attention/BN/decoder完整保留。粗流坐标/时间合同由单独审计确定，本账本不以H/W比例证明物理支持范围。',
        '- 若用DynamicSwin，LN+Linear fast branch还需收费；这里是原生dense MLP对照，未评估窄化或低秩底座。低秩第二因子为real MAC，不能套本表FC AC。',
        '- 下步把真实head来源、score-ready时刻、pair/mask及各类服务接入相同资源的普通执行器；补SRAM、队列、端口及数值合同后才谈净周期。旧16lane核与256乘积/issue模板不能混用。', '',
        '复算（只输出stdout）：`python3 audit_coarse_cost.py --format json`；`--format markdown` 可重建本报告。', '',
        '## 来源与 SHA256', '', '| 来源 | SHA256 |', '| --- | --- |']
    for source in result['sources']:
        lines.append(f"| `{source['path']}` | `{source['sha256']}` |")
    lines += ['', '源 trace checkpoint：`'+result['geometry_trace_provenance']['checkpoint_sha256']+'`；本脚本未加载该权重。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=('json', 'markdown'), default='json')
    args = parser.parse_args()
    result = audit()
    print(markdown(result) if args.format == 'markdown' else json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
