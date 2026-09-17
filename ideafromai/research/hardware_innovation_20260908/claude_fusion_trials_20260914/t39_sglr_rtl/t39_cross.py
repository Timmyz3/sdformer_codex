#!/usr/bin/env python3
"""T39 交叉核对：SGLR 的 numpy 口径（T38b 的 gating） vs RTL 实测。

T38b 报的是 4 层（L8/14/20/28）× 10 序列的合计 34.86%；RTL 只在 s0_stage0/L8
上跑（t25 的 tau 只对这一层生成）。本脚本按同一 numpy 口径给出**逐层**分解，
确认 RTL 的 24.3% 是层差异而非实现差异。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 't32_lever'))
from t32_lever import cert_planes, trace_setup, mask_keep  # noqa: E402
import t19_all_layers as t19  # noqa: E402

T, LIDS, NSID, KEEP = 10, [8, 14, 20, 28], 10, 4
POPTBL = np.array([bin(i).count('1') for i in range(1 << T)], np.int8)


def gating(Aw, msb, jfw):
    """按发送顺序（j 递减）逐平面统计：实际比特 vs SGLR 门控后比特。"""
    sup = (Aw != 0)
    masks = np.array([sum((1 << k) for k in range(T) if sup[t, k]) for t in range(T)], np.int64)
    act = gate = 0
    for g in range(msb.size):
        top, bot = int(msb[g]), int(jfw[g].min())
        if top - bot <= 0:
            continue
        act += (top - bot) * T
        for j in range(top - 1, bot - 1, -1):
            uu = 0
            for t_ in range(T):
                if jfw[g, t_] <= j:
                    uu |= int(masks[t_])
            gate += int(POPTBL[uu])
    return act, gate


def main():
    z = np.load(t19.PARAMS)
    prms = {lid: {'lid': lid, 'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
                  'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
                  'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
                  'theta_src': z['L%d_theta_src' % lid]} for lid in LIDS}
    src = t19.parse_sources(list(range(NSID)))

    per = {}
    for lid in LIDS:
        A_q = np.rint(prms[lid]['A'] * 4096).astype(np.int64)
        A_q = np.where(A_q >= (1 << 15), A_q - (1 << 16), A_q)
        A4 = mask_keep(A_q, KEEP)
        a = g = 0
        first_sid = {}
        for sid in range(NSID):
            key = (sid, lid)
            if key not in src:
                continue
            packed, C, br = src[key]
            Yq, thr_g, _ = trace_setup(packed, C, br, prms[lid])
            del src[key]
            msb = np.array([int(np.abs(Yq[i]).max()).bit_length() for i in range(Yq.shape[0])])
            pl, _, jf = cert_planes(Yq, thr_g, A4)
            aa, gg = gating(A4, msb, jf)
            a += aa; g += gg
            if sid == 0:
                first_sid = {'act': int(aa), 'gate': int(gg),
                             'planes': float(np.clip(pl, 0, None).mean()),
                             'saving': float(1 - gg / aa) if aa else 0.0}
        per['L%02d' % lid] = {
            'bits_actual': int(a), 'bits_gated': int(g),
            'saving': float(1 - g / a),
            'planes_per_group': float(a / (T * NSID * (g and 1) * 10000)),
            'sid0': first_sid,
        }
        print('L%02d  bits %9d -> %9d  省 %.2f%%   sid0 省 %.2f%% (planes/组 %.3f)'
              % (lid, a, g, 100 * (1 - g / a), 100 * first_sid['saving'],
                 first_sid['planes']))

    ta, tg = sum(v['bits_actual'] for v in per.values()), sum(v['bits_gated'] for v in per.values())
    print('合计 bits %d -> %d  省 %.2f%%' % (ta, tg, 100 * (1 - tg / ta)))
    out = {'per_layer': per, 'total_saving': float(1 - tg / ta),
           'note': 'numpy 口径（T38b gating）逐层分解；L08 对应 RTL 的 s0_stage0。'}
    (ROOT / 'results' / 't39_cross_check.json').write_text(json.dumps(out, indent=1) + '\n')
    print('wrote results/t39_cross_check.json')


if __name__ == '__main__':
    main()
