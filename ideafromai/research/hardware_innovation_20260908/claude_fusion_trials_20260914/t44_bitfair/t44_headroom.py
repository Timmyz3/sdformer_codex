#!/usr/bin/env python3
"""T44a：**余量塑形（margin shaping）的供数天花板** —— 花 GPU 之前先量"奖品有多大"。

## 问题

T44 的设计（见 T40_REPORT.md §4 门核一行）是：**抄全 BitFair 的"可学习阈值 + 软门退火训练"，
但把推理期的启发式预测换成 C1 的精确界**。训练侧唯一能动的量是让网络"决定得更自信"，
即把 `δ_t = V_t − thr_t` 相对证书宽度 `L1_t = Σ|A_q[t,c]|` 放大（ρ_t = |δ_t|/L1_t 变大）
⇒ 证书更早锁定 ⇒ 供数下降。推理期仍走精确证书 ⇒ **判决零差**（T40f 的精度约束自动满足）。

本脚本量的是这个方向的**天花板**：若训练能把每个判决的余量放大 κ 倍（判决不变），
供数能降到多少？

> ⚠ **这是逐样本 oracle 上界，不是可达点**：κ 需要知道每个样本的 V 与 thr 才能施加，
> 真实训练只能改变 δ 的**分布尺度**。任何**固定**的 thr 偏移都不等价——常数偏移会移动
> 工作点、改变正负判决比例（那正是 T40f 警告的退化方向）。
> 与 T16 的区别（重要）：T16 的 λ=2 是 **S 尺度等价**（A 与 thr 同缩）⇒ ρ=|δ|/L1 不变
> ⇒ 供数中性（17.5%→17.3%）；本脚本**固定 L1 只动 δ**。两者不是同一个操作。
> ⇒ 本脚本只回答"供数对 ρ 在分布尺度方向上有多敏感"；可达性须由 T44b（固定 thr 偏移的
> 真实 Pareto）与 T44c（带 L_bit 正则的训练）实测。

## 口径

决策保持的余量放大：`δ = V − thr`，令 `thr' = V − κ·(V − thr)` ⇒ `δ' = κ·δ`。
κ>0 时 `V ≥ thr' ⟺ V ≥ thr`，所以判决**逐比特不变**（脚本内 assert 校验）。
供数用 T40f 的 `Σ_j |need_j|`（SGLR 逐 lane 退役）口径，A 取稠密 A 的 top-4 掩码
（= T40f 的 `mag`，13.307 bits/组，与 T39 变体 A 的 13.65 同量级）。

用法：python t44_bitfair/t44_headroom.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 't32_lever'))
from t32_lever import cert_planes, trace_setup  # noqa: E402
import t19_all_layers as t19  # noqa: E402
sys.path.insert(0, str(ROOT / 't40_arch'))
from t40f_masklayout import topk_mask, union_tab, bits_of  # noqa: E402

T, LIDS, NSID, KEEP = 10, [8, 14, 20, 28], 10, 4
GROUPS = 10000
KAPPAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]


def main():
    z = np.load(t19.PARAMS)
    prms = {lid: {'lid': lid, 'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
                  'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
                  'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
                  'theta_src': z['L%d_theta_src' % lid]} for lid in LIDS}
    src = t19.parse_sources(list(range(NSID)))
    cache = {}
    for lid in LIDS:
        for sid in range(NSID):
            if (sid, lid) in src:
                pk, C, br = src[(sid, lid)]
                cache[(sid, lid)] = trace_setup(pk, C, br, prms[lid])

    per_layer = {}
    for lid in LIDS:
        A = np.rint(prms[lid]['A'] * 4096).astype(np.int64)
        A = np.where(A >= (1 << 15), A - (1 << 16), A)
        A4 = np.where(topk_mask(A, KEEP), A, 0)
        supports = np.array([sum((1 << c) for c in range(T) if A4[t, c] != 0)
                             for t in range(T)], np.int64)
        tab = union_tab(supports)

        acc = {k: [0, 0] for k in KAPPAS}      # [bits, groups]
        flips = {k: 0 for k in KAPPAS}
        n_dec = 0
        for key in sorted(cache):
            if key[1] != lid:
                continue
            Yq, thr_g, _ = cache[key]
            Yq, thr_g = Yq[:GROUPS], thr_g[:GROUPS]
            V = Yq @ A4.T                                   # 精确终态值（T41 已证 = 证书终态判决）
            _, dec_ref, _ = cert_planes(Yq, thr_g, A4)
            n_dec += dec_ref.size
            for k in KAPPAS:
                thr_k = V - np.rint(k * (V - thr_g)).astype(np.int64)
                b, _, g = bits_of(Yq, thr_k, A4, tab)
                acc[k][0] += b
                acc[k][1] += g
                _, dec_k, _ = cert_planes(Yq, thr_k, A4)
                flips[k] += int((dec_ref != dec_k).sum())
        row = {'bits_per_group': {}, 'gain_vs_k1_pct': {}, 'flips': {}}
        base = acc[1.0][0] / acc[1.0][1]
        for k in KAPPAS:
            v = acc[k][0] / acc[k][1]
            row['bits_per_group']['%g' % k] = v
            row['gain_vs_k1_pct']['%g' % k] = 100.0 * (base - v) / base
            row['flips']['%g' % k] = flips[k]
        row['n_decisions'] = n_dec
        row['groups'] = acc[1.0][1]
        row['base_bits_per_group'] = base
        per_layer['L%d' % lid] = row
        print('L%-3d base %.4f bits/组 | ' % (lid, base)
              + ' '.join('κ=%g: %.3f (%+.1f%%)' % (k, row['bits_per_group']['%g' % k],
                                                   row['gain_vs_k1_pct']['%g' % k])
                         for k in KAPPAS), flush=True)
        assert sum(flips.values()) == 0, '余量放大改变了判决（%s）' % flips

    keys = list(per_layer['L8']['bits_per_group'])
    agg = {
        'kappas': KAPPAS,
        'bits_per_group': {k: float(np.mean([per_layer[r]['bits_per_group'][k] for r in per_layer]))
                           for k in keys},
        'gain_vs_k1_pct': {k: float(np.mean([per_layer[r]['gain_vs_k1_pct'][k] for r in per_layer]))
                           for k in keys},
        'base_bits_per_group': float(np.mean([per_layer[r]['base_bits_per_group']
                                              for r in per_layer])),
        'total_flips_all_kappa': 0,
        'per_layer': per_layer,
    }
    agg['verdict_note'] = (
        '⚠ 这是**逐样本 oracle 上界**，不是可达点，绝不能当"训练能拿到的收益"引用。'
        'κ 是"把每个 (组,t) 的余量 δ=V−thr 各自放大 κ 倍且保号"——逐样本操作，'
        '需要知道每个样本的 V 与 thr；任何**固定**的 thr 偏移都给不出这个效果'
        '（常数偏移会移动工作点、改变正负判决比例，即 T40f 警告的退化方向）。'
        '真实训练只能改变 δ 的**分布尺度**，不能逐样本设定。'
        '另注意与 T16 的区别：T16 的 λ=2 是 S 尺度等价（A 与 thr 同缩）⇒ ρ=|δ|/L1 不变 ⇒ '
        '供数中性（实测 17.5%→17.3%）；本脚本**固定 L1 只动 δ**，是另一个操作，故结论不同。'
        '所有 κ 下判决翻转恒为 0（构造保证：δ′=κδ，κ>0 保号）。'
        '⇒ 本脚本只回答一件事：**供数对 ρ 在"分布尺度"方向上有多敏感**。'
        'κ=1.25 就给 +18%，说明方向值得追；但可达性必须由 T44b（固定 thr 偏移的真实 Pareto）'
        '与 T44c（带 L_bit 正则的训练）实测，不能由本表的数推。')
    (ROOT / 'results' / 't44_headroom.json').write_text(json.dumps(agg, indent=1) + '\n')
    print('\n=== 4 层平均（基准 %.4f bits/组） ===' % agg['base_bits_per_group'])
    for k in keys:
        print('  κ=%-5s %8.4f bits/组   %+7.2f%%' % (k, agg['bits_per_group'][k],
                                                     agg['gain_vs_k1_pct'][k]))
    print('wrote results/t44_headroom.json')


if __name__ == '__main__':
    main()
