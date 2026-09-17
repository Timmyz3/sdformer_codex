#!/usr/bin/env python3
"""T40 探针①：**判决基（z 空间）传输**能不能赢过当前的 lane 基（Y 空间）传输？

motivation（来自体系结构调研里"把权重挪到生产者/预计算"这一族思路，如 Phi / 结构化重参数化）：
当前 C1 传 Y 的位平面，消费者做 dot=A·Y，证书区间宽度 = (P_t−N_t)·(2^m−1) ≈ Σ_c|A[t][c]|·2^m。
若生产者**先算** z_t = Σ_c A[t][c]·Y_c 再传 z，则区间宽度只剩 2^m（系数全为 1）。
单看"界宽度"→ z 空间紧 ~Σ|A|≈5200 倍，似乎能早 12 个平面锁定。

但**每拍是 10 位**：Y 空间一个平面（10 位）同时推进 10 个判决的界；z 空间一个平面
（10 位）只推进各自的界。本脚本实测两种口径的 bits/组，判 z 空间是否真的更优。

口径与 t39_cross.py 的 gating() 一致（同 A4、同 thr、同 j 从 msb 起扫），
额外把 Y 空间换成 **SGLR + sop 粗区间**（T39 已实测的当前设计点），逐层给数。
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


def z_space(Aw, Yq, thr_g):
    """z 空间：生产者先算 z=A·Y（精确整数），BFP 打包后 MSB-first 传。

    判决 t 的界：读 j 位后 z ∈ [vtop·2^j, vtop·2^j + 2^j − 1]（系数全 1）。
    锁 ⟺ thr 不在该区间。锁深度 d_t = e_z − j_lock；该判决之后不再发。
    bits = Σ_j #{t : d_t ≥ j}（每拍每位 1 bit）。
    """
    Z = Yq @ Aw.T                          # (n, T) 精确整数
    n = Z.shape[0]
    e_z = np.array([int(np.abs(Z[i]).max()).bit_length() for i in range(n)], np.int64)
    bits = 0
    depths = np.zeros((n, T), np.int64)
    for i in range(n):
        thr = thr_g[i]
        for t in range(T):
            zt = int(Z[i, t])
            j = e_z[i]
            while j > 0:
                top = zt >> j                   # 算术右移（二补码）
                lo = top << j
                hi = lo + ((1 << j) - 1)
                if lo >= thr[t] or hi < thr[t]:
                    break
                j -= 1
            depths[i, t] = e_z[i] - j
        if depths[i].max() > 0:
            for j in range(1, int(depths[i].max()) + 1):
                bits += int((depths[i] >= j).sum())
    return bits, depths, e_z


def y_bits_reference(Aw, msb, jfw):
    """Y 空间参照：逐平面 need 并集（= T38b gating，含 sop 粗区间的 j>=msb 段）。"""
    sup = (Aw != 0)
    masks = np.array([sum((1 << k) for k in range(T) if sup[t, k]) for t in range(T)], np.int64)
    bits = 0
    for g in range(msb.size):
        top, bot = int(msb[g]), int(jfw[g].min())
        if top - bot <= 0:
            continue
        for j in range(top - 1, bot - 1, -1):
            uu = 0
            for t_ in range(T):
                if jfw[g, t_] <= j:
                    uu |= int(masks[t_])
            bits += int(POPTBL[uu])
    return bits


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
        by = bz = ngrp = 0
        for sid in range(NSID):
            key = (sid, lid)
            if key not in src:
                continue
            packed, C, br = src[key]
            Yq, thr_g, _ = trace_setup(packed, C, br, prms[lid])
            del src[key]
            msb = np.array([int(np.abs(Yq[i]).max()).bit_length() for i in range(Yq.shape[0])])
            _, _, jf = cert_planes(Yq, thr_g, A4)
            by += y_bits_reference(A4, msb, jf)
            bzv, _, _ = z_space(A4, Yq, thr_g)
            bz += bzv
            ngrp += Yq.shape[0]
        per['L%02d' % lid] = {'y_bits': int(by), 'z_bits': int(bz), 'groups': int(ngrp),
                              'y_per_group': by / ngrp, 'z_per_group': bz / ngrp,
                              'z_over_y': bz / by}
        print('L%02d  n=%6d  Y %8.3f bits/组 (SGLR+粗区间)   Z %8.3f bits/组   Z/Y=%.2fx'
              % (lid, ngrp, by / ngrp, bz / ngrp, bz / by))

    ty = sum(v['y_bits'] for v in per.values())
    tz = sum(v['z_bits'] for v in per.values())
    ng = sum(v['groups'] for v in per.values())
    print('合计  Y %.3f bits/组  vs  Z %.3f bits/组  →  Z/Y = %.2fx'
          % (ty / ng, tz / ng, tz / ty))
    out = {'per_layer': per, 'y_per_group': ty / ng, 'z_per_group': tz / ng,
           'z_over_y': tz / ty,
           'note': '判决基（生产者预算 z=A·Y）传输 vs lane 基传输。Z/Y>1 表示判决基更差：'
                   'Y 空间一个平面（10bit）同时推进 10 个判决的界，Z 空间一个平面只推进'
                   '各自的界；界宽紧 Σ|A|≈5200 倍的纸面优势被这个 10 路 fan-out 抵消。'}
    (ROOT / 'results' / 't40_zspace.json').write_text(json.dumps(out, indent=1) + '\n')
    print('wrote results/t40_zspace.json')


if __name__ == '__main__':
    main()
