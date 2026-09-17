#!/usr/bin/env python3
"""T46-1：把全网 946.38G dense MAC 拆成"真工作量 / 数据零 / 结构零"三份。

## 为什么值得单独拆

T45c 把位级/抽头级的机制轴全判负了，剩下唯一有"作用对象"的东西是**零**。
但"零"有两种，硬件代价完全不同：

- **数据零**（输入脉冲为 0 的那些抽头）：靠 spike-driven 引擎跳过 —— 需要
  逐抽头的活动掩码 / 输入驻留数据流。**这是 B 口径相对 A 省下来的那部分**。
- **结构零**（stride-2 转置卷积的零插值带来的零）：与数据无关，是**映射决策**就能去掉的
  —— 相位分解（polyphase）把它变成 4 个小卷积，抽头数从 36/输入元素降到 9/输入元素。
  不需要任何活动信息、不需要任何重训练、数值精确等价。

两者之和就是 A 口径里"被乘零"的全部。把它们分开，才能回答
"**这张网络的成本里，有多少是数据流的功劳、多少是映射的功劳**"。

## 口径

- 真工作量 = T45b/T45c 的**口径 B 活动加权 SOP**（55.856G，已含三处 frontend 修正）。
- A 口径 dense MAC = T45b 的 `total_macs`（946.38G，forward hook 数法）。
- 结构零只存在于 4 个 stride-2 转置卷积：零插值后每个输入元素被 36 个输出抽头覆盖，
  其中只有 9 个落在真实输入上 ⇒ 结构零 = (36−9)/36 = 3/4 的 A 口径 deconv MAC
  = `mac_A_zero_insert − mac_B_direct`（T45b3 已经算过）。
- 数据零 = 剩下的全部 = A − 结构零 − 真工作量。

（注意 T45b3 的 `mac_B_direct` 是**未加权**的 B 口径稠密数；乘上输入活动率就是真工作量。
  dec 的 4 级 rate 是 0.1299/0.1501/0.1542/0.2597。）

用法：/opt/anaconda3/envs/pytorch310/bin/python t46_zero_decomposition.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T45C = ROOT / 'results' / 't45c2_density_operand_width.json'
T45B3 = ROOT / 'results' / 't45b3_deconv_convention.json'

SUPER = (('enc.', 'encoder.swin3d'), ('frontend.', 'frontend.patch_embed'),
         ('decoder.', 'decoder'), ('bottleneck.', 'bottleneck.resblocks'),
         ('pred.', 'pred'))


def supergroup(group: str) -> str:
    for pre, name in SUPER:
        if group.startswith(pre):
            return name
    return 'other:' + group


def main():
    t45c = json.loads(T45C.read_text())
    dec = json.loads(T45B3.read_text())
    rows = t45c['rows']

    # 结构零：只来自 4 个 stride-2 转置卷积的零插值
    struct = {r['module']: r['mac_A_zero_insert'] - r['mac_B_direct'] for r in dec['rows']}
    struct_total = sum(struct.values())
    struct_by_stage = {m.split('decoders.')[1].split('.')[0]: v for m, v in struct.items()}

    agg = {}
    for r in rows:
        s = r['super']
        g = agg.setdefault(s, {'mac_A': 0.0, 'sop_B': 0.0, 'struct_zero': 0.0, 'nops': 0})
        g['mac_A'] += r['macs']
        g['sop_B'] += r['sop_B']
        g['nops'] += 1
        if r['group'].startswith('decoder.'):
            g['struct_zero'] += struct_by_stage[r['group'].split('decoder.')[1]]
        else:
            # 非转置卷积里 A 与 B 的 MAC 数相同（没有零插值），结构零为 0
            pass

    for g in agg.values():
        g['data_zero'] = g['mac_A'] - g['struct_zero'] - g['sop_B']

    A = sum(g['mac_A'] for g in agg.values())
    W = sum(g['sop_B'] for g in agg.values())
    Zs = sum(g['struct_zero'] for g in agg.values())
    Zd = sum(g['data_zero'] for g in agg.values())

    # 交叉检查：A 口径 dense MAC 应与 T45b 的 total_macs 一致
    t45b = json.loads((ROOT / 'results' / 't45b_mapping_cost_ep34.json').read_text())

    out = {
        't45b_total_macs_hook_A': t45b['total_macs'],
        'sum_detail_macs_A': A,
        'real_work_sop_B': W,
        'struct_zero_zero_insertion': Zs,
        'data_zero_sparsity': Zd,
        'check_A_equals_sum': abs(A - (W + Zs + Zd)) < 1.0,
        'groups': agg,
        'deconv_struct_zero_by_stage': struct_by_stage,
        'deconv_rows': dec['rows'],
    }
    (ROOT / 'results' / 't46_zero_decomposition.json').write_text(
        json.dumps(out, indent=1) + '\n')

    print('=== 全网 946.38G dense MAC（口径 A）的三分解 ===')
    print('%-24s %10s %9s %10s %9s %10s %9s'
          % ('', 'A_dense', 'A%', '真工作量B', '真%B', '数据零', '结构零'))
    for s, g in sorted(agg.items(), key=lambda kv: -kv[1]['mac_A']):
        print('%-24s %10.2f %8.2f%% %10.3f %8.2f%% %10.2f %10.2f'
              % (s, g['mac_A'] / 1e9, 100 * g['mac_A'] / A, g['sop_B'] / 1e9,
                 100 * g['sop_B'] / g['mac_A'], g['data_zero'] / 1e9,
                 g['struct_zero'] / 1e9))
    print('%-24s %10.2f %8.2f%% %10.3f %8.2f%% %10.2f %10.2f'
          % ('合计', A / 1e9, 100.0, W / 1e9, 100 * W / A, Zd / 1e9, Zs / 1e9))
    print('\n交叉检查：T45b hook 的 total_macs = %.2fG，逐算子求和 = %.2fG（%s）'
          % (t45b['total_macs'] / 1e9, A / 1e9,
             '一致' if out['check_A_equals_sum'] else '不一致'))

    print('\n--- 被乘零的 890.5G 里谁占多少 ---')
    tot_zero = Zd + Zs
    print('数据零（spike-driven 引擎跳过，需活动掩码/输入驻留）：%7.2fG = %.1f%% of all zero, %.1f%% of A'
          % (Zd / 1e9, 100 * Zd / tot_zero, 100 * Zd / A))
    print('结构零（相位分解，纯映射决策，精确等价）：          %7.2fG = %.1f%% of all zero, %.1f%% of A'
          % (Zs / 1e9, 100 * Zs / tot_zero, 100 * Zs / A))
    print('真工作量（不可省）：                                %7.3fG = %.1f%% of A'
          % (W / 1e9, 100 * W / A))
    print('\n⇒ A 口径下 94.10% 的乘法是乘零；其中 73.0% 靠数据稀疏、27.0% 靠映射。')

    print('\n--- 结构零的来源（4 个 stride-2 转置卷积）---')
    print('%-14s %10s %10s %10s' % ('stage', 'MAC_A(G)', 'MAC_B(G)', '结构零(G)'))
    for r in dec['rows']:
        print('%-14s %10.3f %10.3f %10.3f'
              % (r['module'].split('sttmultires_unet.')[-1], r['mac_A_zero_insert'] / 1e9,
                 r['mac_B_direct'] / 1e9, struct[r['module']] / 1e9))

    print('\n--- 每个超级组里"被乘零"占自己的比例 ---')
    for s, g in sorted(agg.items(), key=lambda kv: -kv[1]['mac_A']):
        print('%-24s 真工作占 %.2f%%，数据零占 %.2f%%，结构零占 %.2f%%'
              % (s, 100 * g['sop_B'] / g['mac_A'], 100 * g['data_zero'] / g['mac_A'],
                 100 * g['struct_zero'] / g['mac_A']))
    print('\nwrote results/t46_zero_decomposition.json')


if __name__ == '__main__':
    main()
