#!/usr/bin/env python3
"""T48：把 T45–T47 的头条数字装订成论文用的单一账本（不重新测量，只做汇总与口径对齐）。

## 为什么需要

论文要引的数字现在散在 4 份报告 + 5 个 JSON 里，任何一处口径漂移都会让"三足鼎立"
或"16.94×"这类结论复活错误版本（T45 的 55.6%/60.2% 就是这么来的）。
本脚本把它们从**各自的权威 JSON**里读出来装订，并新增一项重解释：

**二值操作数 ⇒ 乘法退化 select ⇒ 输入驻留（input-stationary）数据流下，
SOP_B 恰好等于权重字读取次数。** 于是
`权重读取放大 = SOP_B / 权重总数`，直接量出"要多少次权重读才能出一次推理"，
也就是 weight-stationary 能省掉的倍数。

用法：/opt/anaconda3/envs/pytorch310/bin/python t48_ledger.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / 'results'

SUPER = (('enc.', 'encoder.swin3d'), ('frontend.', 'frontend.patch_embed'),
         ('decoder.', 'decoder'), ('bottleneck.', 'bottleneck.resblocks'),
         ('pred.', 'pred'))
ORDER = ['encoder.swin3d', 'frontend.patch_embed', 'decoder',
         'bottleneck.resblocks', 'pred']


def super_of(group: str) -> str:
    for pre, name in SUPER:
        if group.startswith(pre):
            return name
    return 'other:' + group


def main():
    t45b = json.loads((R / 't45b_mapping_cost_ep34.json').read_text())
    t45c2 = json.loads((R / 't45c2_density_operand_width.json').read_text())
    t46 = json.loads((R / 't46_zero_decomposition.json').read_text())
    t46b = json.loads((R / 't46b_polyphase_deconv.json').read_text())
    t47 = json.loads((R / 't47_ppa.json').read_text())

    # 每个超级组的权重数（从 T45b 的逐组 params 汇总，不重算）
    params = {k: 0 for k in ORDER}
    for g, v in t45b['groups'].items():
        s = super_of(g)
        params[s] = params.get(s, 0) + int(v.get('params', 0))

    groups = []
    for s in ORDER:
        g = t46['groups'][s]
        p = params[s]
        groups.append({
            'super': s,
            'mac_A': g['mac_A'], 'mac_A_pct': 100 * g['mac_A'] / t46['sum_detail_macs_A'],
            'sop_B': g['sop_B'], 'sop_B_pct': 100 * g['sop_B'] / t46['real_work_sop_B'],
            'real_frac_of_A': 100 * g['sop_B'] / g['mac_A'],
            'data_zero': g['data_zero'], 'struct_zero': g['struct_zero'],
            'params': p,
            'weight_read_amplification_input_stationary': g['sop_B'] / p if p else None,
        })

    A = t46['sum_detail_macs_A']
    W = t46['real_work_sop_B']
    Zs = t46['struct_zero_zero_insertion']
    Zd = t46['data_zero_sparsity']
    P = int(t45b['total_params'])

    ledger = {
        'provenance': {
            'checkpoint': 'ep34 (m2041_ep34_quant_binding_inputs)',
            'resolution': '480x640, T=10',
            'convention_A': '零插值 + 普通卷积（cuDNN/ptflops/hook 的标准数法）',
            'convention_B': '输入散射（无零插值）+ 逐抽头零跳过',
            'sources': {
                'convention_A_and_rates': 'results/t45b_mapping_cost_ep34.json',
                'cross_validation_and_operand_width': 'results/t45c2_density_operand_width.json',
                'three_way_decomposition': 'results/t46_zero_decomposition.json',
                'polyphase_equivalence': 'results/t46b_polyphase_deconv.json',
                'rtl_ppa': 'results/t47_ppa.json',
            },
        },
        'totals': {
            'mac_A_dense': A, 'mac_A_dense_G': A / 1e9,
            'sop_B_real_work': W, 'sop_B_real_work_G': W / 1e9,
            'data_zero': Zd, 'data_zero_G': Zd / 1e9,
            'struct_zero': Zs, 'struct_zero_G': Zs / 1e9,
            'params': P,
            'real_work_frac_of_A': W / A,
            'data_zero_frac_of_A': Zd / A,
            'struct_zero_frac_of_A': Zs / A,
            'data_zero_frac_of_all_zero': Zd / (Zd + Zs),
            'struct_zero_frac_of_all_zero': Zs / (Zd + Zs),
            'total_sop_A': t45b['total_sops'],
            'dense_over_real_ratio': A / W,
            'weight_read_amplification_input_stationary': W / P,
        },
        'groups': groups,
        'operand_width': {
            'multibit_sop_G': t45c2['multibit_sop'] / 1e9,
            'multibit_frac': t45c2['multibit_frac_strict'],
            'boundary_sop_G': t45c2['boundary_sop'] / 1e9,
            'boundary_frac': t45c2['boundary_frac'],
            'binary_frac': 1 - t45c2['multibit_plus_boundary_frac'],
            'multibit_ops': t45c2['multibit_ops'],
            'boundary_ops': t45c2['boundary_ops'],
            'criterion': 'bulk = mean_abs/(max*density)；bulk>=0.99 ⇒ 0/θ 脉冲',
            'network_sop_B_corrected': t45c2['network_sop_B_corrected'],
        },
        'polyphase': {
            'equivalence': 'exact 0 elementwise error on real ep34 weights + real input spikes',
            'phase_taps': {'00': 1, '01': 2, '10': 2, '11': 4},
            'taps_total': 9, 'zero_insert_taps_per_input_elem': 36, 'ratio': 4,
            'per_stage': [{'conv': c['conv'], 'Cout': c['Cout'],
                           'struct_zero_allT_allCout_G':
                               c['structural_zero_allT_allCout'] / 1e9,
                           'nonzero_allT_allCout_G': c['mac_nonzero_allT_allCout'] / 1e9}
                          for c in t46b['cases']],
            'structural_zero_dense_G': sum(c['structural_zero_allT_allCout']
                                           for c in t46b['cases']) / 1e9,
            'nonzero_total_G': sum(c['mac_nonzero_allT_allCout']
                                   for c in t46b['cases']) / 1e9,
        },
        'rtl_ppa': {
            'device': 'xczu5ev-sfvc784-2-e', 'mode': 'out_of_context',
            'clk_period_ns': 4.0,
            'z': t47['variants']['z'], 'p': t47['variants']['p'],
            'comparison': t47['comparison'],
            'boundary': '玩具规模 H=4,W=5,CIN=4 单输出通道；LUT 比不可外推，槽位比可外推',
        },
        'levers': [
            {'rank': 1, 'lever': 'spike-driven 映射（口径 B + 零跳过）',
             'gain': 'A/B = %.2f×' % (A / W), 'measured': True,
             'cost': '0（纯映射决策）'},
            {'rank': 2, 'lever': '相位分解（decoder stride-2 转置卷积）',
             'gain': '结构零 %.2fG（A 的 %.1f%%）；槽位 4×；LUT −29%%'
                     % (Zs / 1e9, 100 * Zs / A), 'measured': True,
             'cost': '0 误差、0 重训练；不改非零乘法次数'},
            {'rank': 3, 'lever': 'C1 + SGLR（组级 BFP + 精确证书 + 逐 t 退休）',
             'gain': '~5.9%（覆盖 17.03% × −34.86% 比特）', 'measured': True,
             'cost': '需显式模块化约束 B'},
            {'rank': 4, 'lever': 'k=4 结构稀疏 @ 同一门路',
             'gain': '~3.1%（17.03% × −18.0%）', 'measured': True,
             'cost': '+1.0% AEE（T40f 纪律下已实测）'},
            {'rank': 5, 'lever': '抽头轴/通道前缀精确证书',
             'gain': '0.06%（锁步 0.88%×7.69%）', 'measured': True, 'cost': '判负'},
            {'rank': 6, 'lever': '位平面证书搬到全网',
             'gain': '0%（94.17% 无位平面可裁）', 'measured': True, 'cost': '判负'},
        ],
        'attribution_rule': {
            'statement': 'A/B ≡ 4 是对稠密引擎说的；对已带零跳过的引擎，非零乘法次数不变',
            'nonrelabeled_by_polyphase': [
                '非零乘法次数（active×9×Cout）', '活动加权 SOP（55.856G）',
            ],
            'what_polyphase_actually_buys': [
                '抽头槽位/地址生成/有效判定 36→9（4×，RTL 实测）',
                '输入缓冲 4·Cin·H·W → Cin·H·W（4×，解析）',
                '控制路径 LUT −29%（RTL 实测 219→155）',
            ],
            'cost_shift_when_operand_is_binary': (
                '乘法退化成 select ⇒ 0 DSP ⇒ 代价从乘法器转到**权重读取**。'
                '输入驻留数据流下权重读取次数 ≡ SOP_B = %.2fG 字/推理 = 权重集（%.3fM）的 %.0f×'
                % (W / 1e9, P / 1e6, W / P)),
        },
    }
    (R / 't48_ledger.json').write_text(json.dumps(ledger, indent=1) + '\n')

    t = ledger['totals']
    print('=== 总量（口径 A / B）===')
    print('A dense MAC      %10.2fG' % t['mac_A_dense_G'])
    print('B 真工作量       %10.3fG  (%.2f%% of A)' % (t['sop_B_real_work_G'],
                                                     100 * t['real_work_frac_of_A']))
    print('数据零           %10.2fG  (%.2f%% of A)' % (t['data_zero_G'],
                                                     100 * t['data_zero_frac_of_A']))
    print('结构零           %10.2fG  (%.2f%% of A)' % (t['struct_zero_G'],
                                                     100 * t['struct_zero_frac_of_A']))
    print('A/B 比           %10.2f×' % t['dense_over_real_ratio'])
    print('\n=== 每组（含权重读取放大）===')
    print('%-20s %9s %9s %9s %9s %9s %11s'
          % ('super', 'A(G)', 'A%', 'B(G)', 'B%', '真%ofA', '放大(SOP/权重)'))
    for g in groups:
        print('%-20s %9.2f %8.2f%% %9.3f %8.2f%% %8.2f%% %11.1f'
              % (g['super'], g['mac_A'] / 1e9, g['mac_A_pct'], g['sop_B'] / 1e9,
                 g['sop_B_pct'], g['real_frac_of_A'],
                 g['weight_read_amplification_input_stationary']))
    print('%-20s %9.1f' % ('全网', t['weight_read_amplification_input_stationary']))
    print('\n=== 操作数宽度 ===')
    ow = ledger['operand_width']
    print('真多比特 %.3fG (%.2f%%) / 边界带 %.3fG (%.2f%%) / 0-θ 脉冲 %.2f%%'
          % (ow['multibit_sop_G'], 100 * ow['multibit_frac'],
             ow['boundary_sop_G'], 100 * ow['boundary_frac'],
             100 * ow['binary_frac']))
    print('\n=== 相位分解（RTL 实测）===')
    rp = ledger['rtl_ppa']
    print('Z: %d LUT / %d FF / %d DSP   P: %d LUT / %d FF / %d DSP'
          % (rp['z']['lut_logic'], rp['z']['ff'], rp['z']['dsp'],
             rp['p']['lut_logic'], rp['p']['ff'], rp['p']['dsp']))
    print('槽位比 %.3f×；LUT 比 %.3f×' % (rp['comparison']['slot_ratio_z_over_p'],
                                        rp['comparison']['lut_ratio_z_over_p']))
    print('\nwrote results/t48_ledger.json')


if __name__ == '__main__':
    main()
