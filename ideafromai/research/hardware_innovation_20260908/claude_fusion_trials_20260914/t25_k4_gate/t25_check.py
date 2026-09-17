#!/usr/bin/env python3
"""T25 校验：加法树门核（full / k4 变体）零差检查。

自足重建（不依赖 S/W/随机种子）：从 stim_bf.txt 的 (sign, e, planes) 直接还原
Y_q[s] = −sign_s·2^e + Σ_{j<e} plane_j[s]·2^j（与 t5 生成器同一编码），
thr 从 <variant>/tau_t*.hex 的 64b 字取（bit63=Dflag，低 48 位=thr，U 侧方向折入）。
期望 = 全深度整数判决 dec = Dflag ? (Σ A_q·Y_q ≥ thr) : ~(Σ A_q·Y_q ≥ thr)。

用法：python t25_k4_gate/t25_check.py <variant> <stim> <mode...>
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_A(hexpath):
    vals = [int(x, 16) for x in hexpath.read_text().split()]
    vals = np.array([v - (1 << 16) if v >= (1 << 15) else v for v in vals], np.int64)
    return vals.reshape(10, 10)


def load_tau(d, H):
    tau = np.zeros((10, H), np.int64)
    dflag = np.zeros((10, H), np.int64)
    for t in range(10):
        vals = [int(x, 16) for x in (d / f'tau_t{t}.hex').read_text().split()]
        v = np.array(vals, np.uint64)
        dflag[t] = (v >> np.uint64(63)).astype(np.int64)
        thr = (v & np.uint64((1 << 48) - 1)).astype(np.int64)
        thr = np.where(thr >= (1 << 47), thr - (1 << 48), thr)
        tau[t] = thr
    return tau, dflag


def parse_stim(path):
    groups = []
    with open(path) as f:
        for ln in f:
            if ln[0] == 'G':
                _, h, sign, e = ln.split()
                groups.append({'h': int(h), 'sign': int(sign, 16), 'e': int(e), 'planes': []})
            elif ln[0] == 'B':
                groups[-1]['planes'].append(int(ln.split()[1], 16))
    return groups


def main():
    variant = sys.argv[1]
    stim = Path(sys.argv[2])
    modes = sys.argv[3:]
    d = ROOT / 't25_k4_gate' / variant
    A_q = load_A(d / 'a.hex')
    groups = parse_stim(stim)
    H = max(g['h'] for g in groups) + 1
    thr, dflag = load_tau(d, H)

    # 期望判决（全深度整数）
    Yq = np.zeros((len(groups), 10), np.int64)
    for gi, g in enumerate(groups):
        e = g['e']
        mag = np.zeros(10, np.int64)
        for j, pl in enumerate(reversed(g['planes'])):      # planes 顺序 e-1 … 0
            mag += ((pl >> np.arange(10)) & 1) << j
        sign = np.array([(g['sign'] >> s) & 1 for s in range(10)], np.int64)
        Yq[gi] = -sign * (1 << e) + mag
    V = Yq @ A_q.T                                          # (G,10)
    hs = np.array([g['h'] for g in groups])
    thr_g, df_g = thr[:, hs].T, dflag[:, hs].T
    raw = V >= thr_g
    exp = np.where(df_g > 0, raw, ~raw)

    nex = (A_q != 0).sum(1)
    print(f'{variant}: terms/row={nex.tolist()} groups={len(groups)} H={H}')
    ok = True
    for mode in modes:
        out = d / f'rtl_{mode}.txt'
        lines = [ln for ln in out.read_text().splitlines() if ln and ln[0] != '#']
        assert len(lines) == len(groups), (mode, len(lines), len(groups))
        got = np.array([int(ln.split()[1], 16) for ln in lines], np.int64)
        dec = ((got[:, None] >> np.arange(10)[None, :]) & 1).astype(np.int64)
        mism = int((dec != exp).sum())
        ngrp = int((dec != exp).any(1).sum())
        cyc = np.array([int(ln.split()[2]) for ln in lines])
        print(f'  {mode:9s} dec_mismatches={mism} groups_mismatch={ngrp} '
              f'mean_cycles={cyc.mean():.4f}')
        ok &= mism == 0
    print('ALL MATCH' if ok else 'MISMATCH!')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
