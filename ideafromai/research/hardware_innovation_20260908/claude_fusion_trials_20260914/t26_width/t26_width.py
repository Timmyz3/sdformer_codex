#!/usr/bin/env python3
"""T26 数据通路宽度实测：48b 区间通路是不是真的需要 48 位？

方法：对真实 stim（4 traces × 2 变体）逐组重放 T25/T5 门核的**逐拍递推**，记录
所有中间量（vtop/nv/vmin/vmax/pmsk/nmsk）与 thr 的实测幅值上界，反推所需有符号
位宽。若实测远小于 48，则收窄位宽是不掉判决的纯面积收益（再做 RTL 零差验证）。

用法：python t26_width/t26_width.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

T = 10
STIM_DIRS = ['s0_stage0', 's10_stage0']        # 与 t25 tau 常数同层（H=384）


def load_A(d):
    v = np.array([int(x, 16) for x in (d / 'a.hex').read_text().split()], np.int64)
    v = np.where(v >= (1 << 15), v - (1 << 16), v)
    return v.reshape(T, T)


def load_pn(d):
    v = np.array([int(x, 16) for x in (d / 'pn.hex').read_text().split()], np.int64)
    v = np.where(v >= (1 << 47), v - (1 << 48), v)
    return v[0::2], v[1::2]                       # P_t, N_t



def parse_stim(path):
    groups = []
    with open(path) as f:
        for ln in f:
            if ln[0] == 'G':
                _, h, sign, e = ln.split()
                groups.append({'h': int(h), 'sign': int(sign, 16), 'e': int(e),
                               'planes': []})
            elif ln[0] == 'B':
                groups[-1]['planes'].append(int(ln.split()[1], 16))
    return groups


def load_tau(d, H):
    tau = np.zeros((T, H), np.int64)
    for t in range(T):
        v = np.array([int(x, 16) for x in (d / f'tau_t{t}.hex').read_text().split()],
                     np.uint64)
        thr = (v & np.uint64((1 << 48) - 1)).astype(np.int64)
        tau[t] = np.where(thr >= (1 << 47), thr - (1 << 48), thr)
    return tau


def bits(v):
    """有符号表示所需最小位宽（含符号位）：bits(0)=1, bits(1)=2, bits(-1)=2 ..."""
    a = int(np.abs(v).max())
    return max(1, a.bit_length() + 1)


def replay(groups, A_q, P_t, N_t, thr, H):
    """逐组逐拍重放递推，返回各中间量幅值上界。"""
    mx = {k: 0 for k in ('vtop', 'nv', 'vmin', 'vmax', 'pmsk', 'nmsk', 'thr')}
    e_max = 0
    n_with_m = 0
    for g in groups:
        e = g['e']
        if e == 0:
            m = 0
        else:
            m = e - 1
        n_with_m += (m > 0)
        e_max = max(e_max, e)
        sign = np.array([(g['sign'] >> s) & 1 for s in range(T)], np.int64)
        dot_sign = A_q @ sign
        vtop = -dot_sign
        pmsk = (P_t << m) - P_t
        nmsk = (N_t << m) - N_t
        thr_t = thr[:, g['h']]
        mx['vtop'] = max(mx['vtop'], int(np.abs(vtop).max()))
        mx['pmsk'] = max(mx['pmsk'], int(np.abs(pmsk).max()))
        mx['nmsk'] = max(mx['nmsk'], int(np.abs(nmsk).max()))
        mx['thr'] = max(mx['thr'], int(np.abs(thr_t).max()))
        mm = m
        for pl in reversed(g['planes']):          # planes 顺序 e-1 … 0
            plane = np.array([(pl >> s) & 1 for s in range(T)], np.int64)
            lutw = A_q @ plane
            nv = (vtop << 1) + lutw
            vmin = (nv << mm) + nmsk
            vmax = (nv << mm) + pmsk
            mx['nv'] = max(mx['nv'], int(np.abs(nv).max()))
            mx['vmin'] = max(mx['vmin'], int(np.abs(vmin).max()))
            mx['vmax'] = max(mx['vmax'], int(np.abs(vmax).max()))
            vtop = nv
            if mm != 0:
                pmsk = (pmsk - P_t) >> 1
                nmsk = (nmsk - N_t) >> 1
                mm -= 1
    return mx, e_max, n_with_m


def main():
    out = {}
    for variant in ('full', 'k4'):
        d = ROOT / 't25_k4_gate' / variant
        A_q = load_A(d)
        P_t, N_t = load_pn(d)
        agg = {k: 0 for k in ('vtop', 'nv', 'vmin', 'vmax', 'pmsk', 'nmsk', 'thr')}
        e_max = 0
        ngrp = 0
        nz = 0
        for sd in STIM_DIRS:
            p = ROOT / 'results' / 't5_rtl' / sd / 'stim_bf.txt'
            groups = parse_stim(p)
            H = max(g['h'] for g in groups) + 1
            thr = load_tau(d, H)
            mx, em, nm = replay(groups, A_q, P_t, N_t, thr, H)
            for k in agg:
                agg[k] = max(agg[k], mx[k])
            e_max = max(e_max, em)
            ngrp += len(groups)
            nz += nm
        req = {k: bits(v) for k, v in agg.items()}
        need = max(req.values())
        out[variant] = {'max_abs': agg, 'required_signed_bits': req,
                        'binding_width': need, 'e_max': e_max,
                        'groups': ngrp, 'groups_with_m_gt0': nz}
        print(f'== {variant} ==  groups={ngrp} e_max={e_max} '
              f'(groups with m>0: {nz})')
        for k in agg:
            print(f'   {k:6s} max|.|={agg[k]:>18,d}  needs {req[k]:2d} bits')
        print(f'   -> binding width = {need} bits (48b 当前)')

    (ROOT / 'results' / 't26_width.json').write_text(json.dumps(out, indent=1) + '\n')
    print('\nwrote results/t26_width.json')


if __name__ == '__main__':
    main()
