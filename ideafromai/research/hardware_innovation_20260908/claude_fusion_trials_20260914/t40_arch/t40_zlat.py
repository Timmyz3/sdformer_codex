#!/usr/bin/env python3
"""T40 探针①c：z 空间方案的**延迟尾**（bits 只是均值，拍数看最差判决）。

z 空间的 bits 由 Σ_t depth_t 决定，但**拍数 = max_t depth_t**（一个组要等最慢判决）。
本脚本给出 max_t depth 的分布，与 lane 基（拍数 = 组平面数，由最慢判决决定）对照，
确认 z 空间在尾部分位是否可接受。

用法：python t40_arch/t40_zlat.py
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


def main():
    z = np.load(t19.PARAMS)
    prms = {lid: {'lid': lid, 'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
                  'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
                  'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
                  'theta_src': z['L%d_theta_src' % lid]} for lid in LIDS}
    src = t19.parse_sources(list(range(NSID)))

    zmax, yplanes, n = [], [], 0
    for lid in LIDS:
        A_q = np.rint(prms[lid]['A'] * 4096).astype(np.int64)
        A_q = np.where(A_q >= (1 << 15), A_q - (1 << 16), A_q)
        A4 = mask_keep(A_q, KEEP)
        for sid in range(NSID):
            key = (sid, lid)
            if key not in src:
                continue
            packed, C, br = src[key]
            Yq, thr_g, _ = trace_setup(packed, C, br, prms[lid])
            del src[key]
            msb = np.array([int(np.abs(Yq[i]).max()).bit_length() for i in range(Yq.shape[0])])
            pl, _, jf = cert_planes(Yq, thr_g, A4)
            Z = Yq @ A4.T
            m = Yq.shape[0]
            ez = np.array([int(np.abs(Z[i]).max()).bit_length() for i in range(m)])
            for i in range(m):
                d = np.zeros(T, np.int64)
                for t in range(T):
                    zt, th = int(Z[i, t]), int(thr_g[i, t])
                    j = int(ez[i])
                    while j > 0:
                        lo = (zt >> j) << j
                        if lo >= th or lo + ((1 << j) - 1) < th:
                            break
                        j -= 1
                    d[t] = int(ez[i]) - j
                zmax.append(int(d.max()))
            # lane 基拍数 = 组平面数（由最慢判决决定）
            yplanes.extend(np.clip(msb - jf.min(1), 0, None).tolist())
            n += m

    zmax = np.array(zmax)
    yp = np.array(yplanes)
    q = [50, 90, 99, 99.9, 100]
    out = {
        'groups': int(zmax.size),
        'z_maxdepth_mean': float(zmax.mean()),
        'z_maxdepth_pct': [float(np.percentile(zmax, p)) for p in q],
        'z_maxdepth_max': int(zmax.max()),
        'y_planes_mean': float(yp.mean()),
        'y_planes_pct': [float(np.percentile(yp, p)) for p in q],
        'y_planes_max': int(yp.max()),
        'z_zero_plane_fraction': float((zmax == 0).mean()),
        'note': '拍数 = 1(sop) + max_t depth。z 空间 bits 均值低（3.42 bits/组）但拍数由最慢'
                '判决决定；lane 基拍数 = 组平面数。',
    }
    (ROOT / 'results' / 't40_zlat.json').write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out.items():
        print('%-26s %s' % (k, v))
    print('wrote results/t40_zlat.json')


if __name__ == '__main__':
    main()
