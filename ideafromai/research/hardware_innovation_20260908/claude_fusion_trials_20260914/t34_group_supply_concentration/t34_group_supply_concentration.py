#!/usr/bin/env python3
"""T34：供数在"组"上的集中度 —— 数量维（token/块/时间步剪枝）能否买供数？

动机：数量维轴（Q1 token 剪枝/合并、Q2 自适应深度、Q3 adaptive-T）的共同承诺是
"减少被服务的对象数 → 按比例减少供数"。但 C1 的证书是**逐组动态**的：低信息量的组
（无脉冲 ⇒ Y≈0）在 MSB-first 递推里**早就锁定**，本来就只花极少平面。
若供数高度集中在少数"难组"上，则剪掉那些"易组"能省的供数 ≈ 0（**次可加性**），
剪枝买到的是**数据通路周期**而非**供数**——这正是 §G 对数量维的关键限定。

本脚本用真实 traces 量化：逐组 plane 数的分布与供数集中度。
指标：①组平面数分布；②top-k% 组承载的供数份额；③"剪掉 plane≤阈值 的组"能省多少供数。

口径：bits/组 = 10 × planes/组（主口径）。自有代码；capture 只读。
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

T = 10
G = 4000
LIDS = [8, 14, 20, 28]
NSID = 10
TOPS = [1, 5, 10, 20, 50]


def main():
    z = np.load(t19.PARAMS)
    prms = {}
    for lid in LIDS:
        prms[lid] = {'lid': lid, 'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
                     'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
                     'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
                     'theta_src': z['L%d_theta_src' % lid]}
    src = t19.parse_sources(list(range(NSID)))
    print('parsed', len(src), 'pairs; layers', LIDS, flush=True)

    parts = []
    for lid in LIDS:
        for sid in range(NSID):
            key = (sid, lid)
            if key not in src:
                continue
            packed, C, br = src[key]
            Yq, thr_g, A_q = trace_setup(packed, C, br, prms[lid])
            del src[key]
            p, _, _ = cert_planes(Yq, thr_g, A_q)
            parts.append(np.asarray(p, np.float64))
        print('  L%02d done' % lid, flush=True)

    P = np.concatenate(parts)
    tot = P.sum()
    base = float(P.mean())
    print('\n== 逐组平面数分布（n=%d 组）==' % P.size)
    print('  mean %.4f  median %.1f  min %.1f  max %.1f  std %.3f'
          % (base, np.median(P), P.min(), P.max(), P.std()))

    print('\n== 集中度：top-k%% 组承载的供数份额 ==')
    Ps = np.sort(P)[::-1]
    n = Ps.size
    conc = {}
    for k in TOPS:
        m = max(1, int(n * k / 100))
        share = Ps[:m].sum() / tot
        conc[k] = float(share)
        print('  top %2d%% 的组（n=%5d）→ 供数份额 %.1f%%' % (k, m, 100 * share))

    print('\n== 次可加性检验：剪掉"易组"能省多少供数 ==')
    sub = {}
    for thr in [0, 1, 2, 3]:
        mask = P <= thr
        frac_groups = mask.mean()
        saved = P[mask].sum() / tot
        sub[thr] = {'groups_cut': float(frac_groups), 'supply_saved': float(saved)}
        print('  剪掉 plane<=%d 的组：占 %.1f%% 的组 → 仅省 %.2f%% 供数'
              % (thr, 100 * frac_groups, 100 * saved))

    out = {'baseline_planes': base, 'n_groups': int(P.size),
           'mean': base, 'median': float(np.median(P)), 'max': float(P.max()),
           'concentration_tops': conc, 'subadditivity': sub,
           'layers': LIDS, 'n_sid': NSID, 'G': G}
    (ROOT / 'results' / 't34_group_supply_concentration.json').write_text(
        json.dumps(out, indent=1) + '\n')
    print('\nwrote results/t34_group_supply_concentration.json')


if __name__ == '__main__':
    main()
