#!/usr/bin/env python3
"""T39 逐组模型：在 **RTL 实际用到的同一批组**上复现证书递推 + SGLR 门控。

目的：把"RTL 的 24.3%"与"numpy 口径的 33.6%"的差异定位到**组样本**而非实现。
本脚本完全照 RTL 的时序写（t25_gen 的 SV 模板语义）：
  sop : m_chk=e−1, vtop=−dot10(sign), pmsk=P·(2^m−1), nmsk=N·(2^m−1)
  plane pi: nv=(vtop<<1)+dot10(plane_pi)
            vmin=nv<<m_chk + nmsk ; vmax=nv<<m_chk + pmsk
            lock ⟸ ~locked & (vmin≥thr | vmax<thr)
            pmsk←(pmsk−P)>>>1 ; nmsk←(nmsk−N)>>>1 ; m_chk←m_chk−1
  SGLR: need(pi) = OR_{t: ~locked[t]} sup_t ；bits += popcount(need)
逐组与 RTL 的 `R <dec> <fed>` 与 `S <fed> <bits_gated> <bits_full>` 比对。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 't39_sglr_rtl'
REF = ROOT / 't25_k4_gate' / 'k4'
STIM = ROOT / 'results' / 't5_rtl' / 's0_stage0' / 'stim_bf.txt'
T = 10
POPTBL = np.array([bin(i).count('1') for i in range(1 << T)], np.int8)


def s16(v):
    v = np.asarray(v, np.int64)
    return v - (1 << 16) * (v >= (1 << 15))


def load_A_q():
    vals = [int(x, 16) for x in (REF / 'a.hex').read_text().split()]
    return s16(vals).reshape(T, T)


def load_tau(H):
    thr = np.zeros((T, H), np.int64)
    dflag = np.zeros((T, H), np.int64)
    for t in range(T):
        v = np.array([int(x, 16) for x in (REF / f'tau_t{t}.hex').read_text().split()], np.uint64)
        dflag[t] = (v >> np.uint64(63)).astype(np.int64)
        w = (v & np.uint64((1 << 48) - 1)).astype(np.int64)
        thr[t] = np.where(w >= (1 << 47), w - (1 << 48), w)
    return thr, dflag


def parse_stim(path):
    gs = []
    with open(path) as f:
        for ln in f:
            if ln[0] == 'G':
                _, h, sign, e = ln.split()
                gs.append({'h': int(h), 'sign': int(sign, 16), 'e': int(e), 'pl': []})
            elif ln[0] == 'B':
                gs[-1]['pl'].append(int(ln.split()[1], 16) & 0x3ff)
    return gs


def dot(A_q, t, w):
    """dot10(t,w)=Σ_{c∈w} A_q[t][c]，w 为 10bit 掩码。"""
    m = [(w >> c) & 1 for c in range(T)]
    return int(sum(A_q[t][c] for c in range(T) if m[c]))


def run_group(A_q, P, N, thr_t, df_t, g, preplane=False):
    e = g['e']
    sign = g['sign']
    pl = g['pl']
    vtop = np.array([-dot(A_q, t, sign) for t in range(T)], np.int64)
    locked = np.zeros(T, bool)
    raw = np.zeros(T, bool)
    # [可选] 发送前粗区间：全部 e 个低位未知，V ∈ vtop·2^e + [N(2^e−1), P(2^e−1)]
    if preplane:
        m0 = (1 << e) - 1
        vmin0 = (vtop << e) + N * m0
        vmax0 = (vtop << e) + P * m0
        lock0 = (vmin0 >= thr_t) | (vmax0 < thr_t)
        raw = np.where(lock0, vmin0 >= thr_t, raw)
        locked = lock0
    m_chk = 0 if e == 0 else e - 1
    pmsk = np.array([(P[t] << m_chk) - P[t] for t in range(T)], np.int64)
    nmsk = np.array([(N[t] << m_chk) - N[t] for t in range(T)], np.int64)
    sup = np.array([sum(1 << c for c in range(T) if A_q[t][c] != 0) for t in range(T)], np.int64)
    fed = bits_g = bits_f = 0
    n_plane = max(e, 1)
    for pi in range(n_plane):
        if locked.all():
            break
        need = 0
        for t in range(T):
            if not locked[t]:
                need |= int(sup[t])
        bits_g += int(POPTBL[need])
        bits_f += T
        plane = pl[pi] if pi < len(pl) else 0
        nv = (vtop << 1) + np.array([dot(A_q, t, plane) for t in range(T)], np.int64)
        vmin = (nv << m_chk) + nmsk
        vmax = (nv << m_chk) + pmsk
        newlock = (~locked) & ((vmin >= thr_t) | (vmax < thr_t))
        vtop = nv
        if m_chk != 0:
            pmsk = (pmsk - P) >> 1
            nmsk = (nmsk - N) >> 1
            m_chk -= 1
        raw = np.where(newlock, vmin >= thr_t, raw)
        locked |= newlock
        fed += 1
    dec = np.where(df_t > 0, raw, ~raw).astype(np.int64)
    return dec, fed, bits_g, bits_f, locked


def main():
    A_q = load_A_q()
    gs = parse_stim(STIM)
    H = max(g['h'] for g in gs) + 1
    thr, dflag = load_tau(H)
    P = A_q.clip(min=0).sum(1)
    N = A_q.clip(max=0).sum(1)

    # 消融对照：T39 设计点（SGLR+粗区间）与仅 SGLR 变体各一份 RTL 输出
    args = sys.argv[1:]
    preplane = '--preplane' in args
    prefix = 'np_' if '--prefix-np' in args else ''
    modes = ('bf_cert',) if '--cert-only' in args else ('bf_cert', 'bf_full')

    rtl = {}
    for mode in modes:
        lines = [ln for ln in (D / f'rtl_{prefix}{mode}_g1.txt').read_text().splitlines()
                 if ln and ln[0] != '#']
        st = [ln for ln in (D / f'stats_{prefix}{mode}_g1.txt').read_text().splitlines()
              if ln and ln[0] != '#']
        rtl[mode] = {
            'dec': np.array([int(ln.split()[1], 16) for ln in lines], np.int64),
            'fed': np.array([int(ln.split()[2]) for ln in lines], np.int64),
            'bits_g': np.array([int(ln.split()[2]) for ln in st], np.int64),
            'bits_f': np.array([int(ln.split()[3]) for ln in st], np.int64),
        }
        assert len(lines) == len(gs) == len(st), (mode, len(lines), len(gs), len(st))

    res = {}
    for mode in modes:
        dec_m = np.zeros(len(gs), np.int64)
        fed_m = np.zeros(len(gs), np.int64)
        bg_m = np.zeros(len(gs), np.int64)
        bf_m = np.zeros(len(gs), np.int64)
        nolock = 0
        for i, g in enumerate(gs):
            d, fed, bg, bf, lk = run_group(A_q, P, N, thr[:, g['h']], dflag[:, g['h']], g,
                                           preplane=preplane)
            if not lk.all():
                nolock += 1
            dec_m[i] = sum(int(d[t]) << t for t in range(T))
            fed_m[i], bg_m[i], bf_m[i] = fed, bg, bf
        r = rtl[mode]
        # bf_full 不省平面（send=0x3ff），其 bits_g==bits_f 是构造使然，不纳入 bits 比对；
        # 拍的"省"只在 bf_cert（提前收）里体现。
        exact = mode == 'bf_cert'
        res[mode] = {
            'dec_mismatch': int((dec_m != r['dec']).sum()),
            'fed_mismatch': int((fed_m != r['fed']).sum()) if exact else None,
            'bits_g_mismatch': int((bg_m != r['bits_g']).sum()) if exact else None,
            'fed_mean': float(fed_m.mean()), 'fed_rtl_mean': float(r['fed'].mean()),
            'bits_g_total': int(bg_m.sum()), 'bits_f_total': int(bf_m.sum()),
            'saving': float(1 - bg_m.sum() / bf_m.sum()),
            'rtl_saving': float(1 - r['bits_g'].sum() / r['bits_f'].sum()) if exact else None,
            'groups_not_fully_locked': nolock,
        }
        print('%s: dec_mism=%d%s | 模型省 %.2f%%%s'
              % (mode, res[mode]['dec_mismatch'],
                 '' if exact else ' (fed/bits 不比对)',
                 100 * res[mode]['saving'],
                 '' if exact else ''))
    checked = [v for m, v in res.items() if m == 'bf_cert']
    ok = (all(v['dec_mismatch'] == 0 for v in res.values())
          and all(v['fed_mismatch'] == 0 and v['bits_g_mismatch'] == 0 for v in checked))
    print('MODEL==RTL 逐组逐比特:', 'PASS' if ok else 'FAIL')
    out = {'preplane': bool(preplane), 'prefix': prefix, 'per_mode': res,
           'model_vs_rtl_exact': bool(ok),
           'stim': str(STIM.relative_to(ROOT)), 'n_groups': len(gs),
           'note': 's0_stage0（T5 链 2 万组）；bf_cert 逐组逐比特对齐 RTL。'
                   'preplane=True → 含 sop 拍粗区间判定；prefix=np_ → 仅 SGLR 变体。'}
    tag = ('np' if prefix else 'pp') + ('_full' if not preplane else '')
    (ROOT / 'results' / f't39_model_{tag}.json').write_text(json.dumps(out, indent=1) + '\n')
    print('wrote results/t39_model_%s.json' % tag)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
