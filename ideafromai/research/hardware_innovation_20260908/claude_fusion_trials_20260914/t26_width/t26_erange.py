#!/usr/bin/env python3
"""T26a：全网指数 e 与幅值上界实测（决定门核区间数据通路的**必需位宽**）。

T26 的 48b 从何而来：e 是 5 bit 端口（e≤31），|Y|≤2^e，V=Y·A_q 的幅值 ≤ L1_t·2^e。
L1_t 上界 ~2^15.3（A_q 是 f12 量化的 10 项和）→ 最坏 15.3+31+1(sign) ≈ 48 bit。
所以 48 位是"e≤31 全量程"的保守尺寸，不是浪费。

要收窄位宽，必须知道真实 e 的**全局上界**（12 层 × 40 序列），据此给设计余量。
本脚本复用 t19 的 capture 解析器与参数，逐 (layer,sid) 重算每组的
e_g = bit_length(max|Y_q|)（组级块浮点指数）并记录全局最大。

用法：python t26_width/t26_erange.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t19_all_layers import (CAP, HEADER, LIDS, PARAMS, T,  # noqa: E402
                            parse_sources)

G = 20000


def main():
    z = np.load(PARAMS)
    sids = list(range(40))
    src = parse_sources(sids)
    print('parsed', len(src), '(sid,lid) pairs', flush=True)
    rows = []
    msb_max = 0
    l1_row_max = 0
    for lid in LIDS:
        e_lid, l1_lid, y_lid = 0, 0, 0
        ntr = 0
        for sid in sids:
            if (sid, lid) not in src:
                continue
            packed, C, br = src[sid, lid]
            S = np.unpackbits(packed.reshape(-1, br), axis=1,
                              bitorder='little')[:, :C].astype(np.float64)
            N = S.shape[0]
            P = N // T
            W = z['L%d_W' % lid].astype(np.float64)
            A = z['L%d_A' % lid]
            theta = float(np.asarray(z['L%d_theta_src' % lid]).ravel()[0])
            A_q = np.rint(A * 4096).astype(np.int64)
            l1_lid = max(l1_lid, int(np.abs(A_q).sum(1).max()))   # 逐行 L1
            # 组级块浮点指数：与 t19 同口径——Y 先量化到 signed24/f14
            Y = (S @ W.T) * theta                          # (N, H)
            Yq = np.clip(np.rint(Y.reshape(P, T, -1) * (1 << 14)),
                         -(1 << 23), (1 << 23) - 1)
            msb = np.abs(Yq).reshape(P, -1).max(1)          # 每组 max|Yq|
            e = np.array([int(v).bit_length() for v in msb])   # = e+1（e 为 MSB 下标）
            e_lid = max(e_lid, int(e.max()))
            y_lid = max(y_lid, int(np.abs(Yq).max()))
            ntr += 1
            del src[sid, lid], S, Y, Yq
        msb_max = max(msb_max, e_lid)
        l1_row_max = max(l1_row_max, l1_lid)
        rows.append({'lid': lid, 'traces': ntr, 'msb_max': e_lid,
                     'e_max': e_lid - 1, 'max_absYq': y_lid, 'L1_row_max': l1_lid,
                     'worst_bits': int(l1_lid * (1 << (e_lid - 1))).bit_length() + 1})
        print(f'L{lid:02d}: traces={ntr} e_max={e_lid - 1:2d} max|Yq|={y_lid:>12,d} '
              f'L1_row_max={l1_lid:>7,d} worst_width={rows[-1]["worst_bits"]}',
              flush=True)

    out = {'e_max_global': msb_max - 1, 'L1_row_max_global': l1_row_max,
           'worst_width_global': int(l1_row_max * (1 << (msb_max - 1))).bit_length() + 1,
           'per_layer': rows,
           'note': 'e = MSB 下标 = bit_length(组内 max|Yq_f14|)-1（组内 10 行）'}
    (ROOT / 'results' / 't26_erange.json').write_text(json.dumps(out, indent=1) + '\n')
    print(f'\nGLOBAL e_max = {msb_max - 1}  L1_row_max = {l1_row_max}  '
          f'worst_width = {out["worst_width_global"]}')
    print('wrote results/t26_erange.json')


if __name__ == '__main__':
    main()
