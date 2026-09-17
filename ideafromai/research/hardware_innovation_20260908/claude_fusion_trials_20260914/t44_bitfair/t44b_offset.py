#!/usr/bin/env python3
"""T44b：固定阈值偏移（= BitFair 的 θ^l 原形）的真实 (供数, 判决) Pareto。

## 为什么必须先做这一步

T44a 的 κ 曲线是**逐样本 oracle**，不可达。BitFair 实际能施加的操作是
`θ^l = θ0^l + θx^l`——一个**固定的、逐层的阈值偏移**（推理期就是常数）。它改变的是
工作点：thr 上移 ⇒ 判 0 更多 ⇒ 早停更早，但**漏报正判决**；thr 下移反之。
这正是 T40f 警告的方向（用精度换供数的退化风险），所以必须实测 (供数, 漏报) 曲线，
而不是假设它能白拿。

## 参数化

`Δ_t = α · anchor_t`，`thr'_t = thr_t − Δ_t`，其中 `anchor_t = median_s |V_{s,t} − thr_t|`
（该层 t 的典型余量）。α>0 ⇒ thr 下移 ⇒ 判 1 更多；α<0 ⇒ 判 0 更多。
用中位余量作锚点，使 α=1 恰是"偏移等于典型余量"的尺度——判决开始大面积翻转的地方。

锚点在**同一批 traces** 上算（诊断用，非训练；训练侧用的是 BitFair 的 θ0 初始化 + 可学习偏移）。

给论文的对照口径：BitFair 的早停是**单边启发式**（`P_k ≤ θ` 就预测 ReLU 零），
无误差界；本表的 flip 列就是"它省下的供数要付多少判决代价"。

用法：python t44_bitfair/t44b_offset.py
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
ALPHAS = [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


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
                Yq, thr_g, _ = trace_setup(pk, C, br, prms[lid])
                cache[(sid, lid)] = (Yq[:GROUPS], thr_g[:GROUPS])

    per_layer = {}
    for lid in LIDS:
        A = np.rint(prms[lid]['A'] * 4096).astype(np.int64)
        A = np.where(A >= (1 << 15), A - (1 << 16), A)
        A4 = np.where(topk_mask(A, KEEP), A, 0)
        supports = np.array([sum((1 << c) for c in range(T) if A4[t, c] != 0)
                             for t in range(T)], np.int64)
        tab = union_tab(supports)

        keys = [k for k in sorted(cache) if k[1] == lid]
        V = np.concatenate([cache[k][0] @ A4.T for k in keys])
        thr = np.concatenate([cache[k][1] for k in keys])
        Yall = np.concatenate([cache[k][0] for k in keys])
        anchor = np.median(np.abs(V - thr), axis=0).astype(np.int64)     # 逐 t 锚点
        _, dec_ref, _ = cert_planes(Yall, thr, A4)
        n_dec = dec_ref.size

        row = {'alpha': ALPHAS, 'bits_per_group': {}, 'flip_pct': {},
               'miss_pct': {}, 'extra_pct': {}, 'anchor': anchor.tolist(),
               'groups': int(Yall.shape[0]), 'n_decisions': int(n_dec)}
        for a in ALPHAS:
            thr_a = thr - np.rint(a * anchor).astype(np.int64)
            b, _, g = bits_of(Yall, thr_a, A4, tab)
            _, dec, _ = cert_planes(Yall, thr_a, A4)
            row['bits_per_group']['%g' % a] = b / g
            row['flip_pct']['%g' % a] = 100.0 * int((dec != dec_ref).sum()) / n_dec
            row['miss_pct']['%g' % a] = 100.0 * int((dec_ref & ~dec).sum()) / n_dec
            row['extra_pct']['%g' % a] = 100.0 * int((~dec_ref & dec).sum()) / n_dec
        row['base_bits_per_group'] = row['bits_per_group']['0']
        per_layer['L%d' % lid] = row
        print('L%-3d base %.3f bits/组 | ' % (lid, row['base_bits_per_group'])
              + ' '.join('α=%+.2g: %.2f/%.2f%%' % (a, row['bits_per_group']['%g' % a],
                                                   row['flip_pct']['%g' % a])
                         for a in ALPHAS), flush=True)

    keys = ['%g' % a for a in ALPHAS]
    agg = {
        'alpha': ALPHAS,
        'base_bits_per_group': float(np.mean([per_layer[r]['base_bits_per_group']
                                              for r in per_layer])),
        'bits_per_group': {k: float(np.mean([per_layer[r]['bits_per_group'][k] for r in per_layer]))
                           for k in keys},
        'flip_pct': {k: float(np.mean([per_layer[r]['flip_pct'][k] for r in per_layer]))
                     for k in keys},
        'miss_pct': {k: float(np.mean([per_layer[r]['miss_pct'][k] for r in per_layer]))
                     for k in keys},
        'extra_pct': {k: float(np.mean([per_layer[r]['extra_pct'][k] for r in per_layer]))
                      for k in keys},
        'per_layer': per_layer,
    }
    agg['verdict_note'] = (
        '这是 BitFair 的 θ^l **原形**（固定逐层偏移）在 C1 口径下的真实 Pareto：'
        'α<0（thr 上移、判 0 更多）省供数但漏报；α>0 供数上升。'
        '判读要看**省 1 bit 要付多少 flip**——若斜率差（flip 涨得比供数省得快），'
        '则"抄全 BitFair"这条线在 C1 上不成立，只剩 T44c 的"训练塑形 + 精确界推理"'
        '那一支（推理期零差，只买供数）。'
        '⚠ 锚点用同一批 traces 的中位余量，是诊断参数化，不是训练超参。')
    (ROOT / 'results' / 't44b_offset.json').write_text(json.dumps(agg, indent=1) + '\n')
    print('\n=== 4 层平均（基准 %.4f bits/组） ===' % agg['base_bits_per_group'])
    for k in keys:
        print('  α=%-6s %8.4f bits/组   flip %6.3f%%  (漏报 %5.3f%% / 虚报 %5.3f%%)'
              % (k, agg['bits_per_group'][k], agg['flip_pct'][k],
                 agg['miss_pct'][k], agg['extra_pct'][k]))
    print('wrote results/t44b_offset.json')


if __name__ == '__main__':
    main()
