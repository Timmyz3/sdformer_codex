#!/usr/bin/env python3
"""T40 探针①b：z 空间结果的**分解与验证**（在相信 t40_zspace 的结论前必须过这一关）。

`t40_zspace.py` 测出判决基传输只用 3.42 bits/组（lane 基 13.31 的 0.257×）。
本脚本回答三个问题：
  V1 判决是否一致：z 空间证书（区间不跨 thr 即锁）最终是否给出与 lane 基相同的符号？
  V2 收益从哪来：把 lane 基的两部分拆开——
       (a) 逐判决深度 Σ_t depth_Y(t)（若判决能各自退役的**下界**）
       (b) 支撑并集税 = Σ_j |need_j| − Σ_t depth_Y(t)
     若 (a) ≈ z 空间成本，则 3.9× 全部来自"lane 是比判决更粗的退役单位"。
  V3 尺度合理性：depth 分布、e_z 与 e 的关系（e_z ≈ e + log2 L1 是否成立）。

用法：python t40_arch/t40_zdiag.py
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


def main():
    z = np.load(t19.PARAMS)
    prms = {lid: {'lid': lid, 'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
                  'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
                  'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
                  'theta_src': z['L%d_theta_src' % lid]} for lid in LIDS}
    src = t19.parse_sources(list(range(NSID)))

    acc = {k: 0 for k in ('grp', 'y_sglr', 'y_perdec', 'z', 'z_d0', 'z_dec', 'n_dec')}
    dz_all, dy_all, ez_minus_e = [], [], []
    agree_bad = 0
    for lid in LIDS:
        A_q = np.rint(prms[lid]['A'] * 4096).astype(np.int64)
        A_q = np.where(A_q >= (1 << 15), A_q - (1 << 16), A_q)
        A4 = mask_keep(A_q, KEEP)
        sup = (A4 != 0)
        masks = np.array([sum((1 << k) for k in range(T) if sup[t, k]) for t in range(T)], np.int64)
        L1 = np.abs(A4).sum(1)
        for sid in range(NSID):
            key = (sid, lid)
            if key not in src:
                continue
            packed, C, br = src[key]
            Yq, thr_g, _ = trace_setup(packed, C, br, prms[lid])
            del src[key]
            n = Yq.shape[0]
            msb = np.array([int(np.abs(Yq[i]).max()).bit_length() for i in range(n)])
            _, _, jf = cert_planes(Yq, thr_g, A4)
            Z = Yq @ A4.T
            ez = np.array([int(np.abs(Z[i]).max()).bit_length() for i in range(n)])
            ez_minus_e.extend((ez - msb).tolist())
            for i in range(n):
                # Y 空间：SGLR（支撑并集）与逐判决深度下界
                top, bot = int(msb[i]), int(jf[i].min())
                dy = msb[i] - jf[i]
                dy_all.extend(dy.tolist())
                acc['grp'] += 1
                for j in range(top - 1, bot - 1, -1):
                    uu = 0
                    for t_ in range(T):
                        if jf[i, t_] <= j:
                            uu |= int(masks[t_])
                    acc['y_sglr'] += int(POPTBL[uu])
                acc['y_perdec'] += int(dy.sum())
                # Z 空间
                for t in range(T):
                    zt, th = int(Z[i, t]), int(thr_g[i, t])
                    j = int(ez[i])
                    while j > 0:
                        lo = (zt >> j) << j
                        if lo >= th or lo + ((1 << j) - 1) < th:
                            break
                        j -= 1
                    d = int(ez[i]) - j
                    acc['z'] += d
                    acc['z_d0'] += int(d == 0)
                    dz_all.append(d)
                    # V1：j=0 时已精确；这里验证"锁定时的符号"与最终符号一致
                    lo = (zt >> j) << j
                    if (lo >= th) != (zt >= th):
                        agree_bad += 1
                    acc['z_dec'] += int(zt >= th)
                    acc['n_dec'] += 1
    g = acc['grp']
    out = {
        'groups': g,
        'y_sglr_bits_per_group': acc['y_sglr'] / g,
        'y_per_decision_bits_per_group': acc['y_perdec'] / g,
        'z_bits_per_group': acc['z'] / g,
        'support_union_tax_bits_per_group': (acc['y_sglr'] - acc['y_perdec']) / g,
        'z_depth0_fraction': acc['z_d0'] / acc['n_dec'],
        'z_over_y_sglr': acc['z'] / acc['y_sglr'],
        'z_over_y_perdec': acc['z'] / acc['y_perdec'],
        'ez_minus_e_mean': float(np.mean(ez_minus_e)),
        'dec_sign_mismatch_at_lock': agree_bad,
        'y_depth_hist': np.bincount(np.clip(np.array(dy_all), 0, 12)).tolist(),
        'z_depth_hist': np.bincount(np.clip(np.array(dz_all), 0, 12)).tolist(),
        'note': 'V1 判决一致（dec_sign_mismatch_at_lock 须为 0）；V2 若 z ≈ y_per_decision，'
                '则 3.9× 全部来自"lane 是比判决更粗的退役单位"（支撑并集税）。',
    }
    (ROOT / 'results' / 't40_zdiag.json').write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out.items():
        print('%-36s %s' % (k, v))
    print('wrote results/t40_zdiag.json')


if __name__ == '__main__':
    main()
