#!/usr/bin/env python3
"""T21e/T21f：微调后网络的真实供数比（在 sd5ai 上运行，避免 215MB 传输）。

输入：capture npz（微调网络 40 样本自有 capture）+ 门参数 npz（微调后门
参数）。A 已在参数里是 top-k 掩码态。与 t19/t21_supply 同口径（证书递推、
逐 (sid,lid) 一条 trace、G=20000）。

用法：python t21e_supply_finetuned.py [capture.npz params.npz out.json]
（缺省 = T21b k=5 的三个路径）。产出 JSON（summary + traces）。
"""
import json
import sys
from pathlib import Path

import numpy as np

args = sys.argv[1:]
CAP = Path(args[0]) if len(args) > 0 else Path('/tmp/t21b_cap.npz')
PARAMS = Path(args[1]) if len(args) > 1 else Path('/tmp/t21b_gate_params.npz')
JOUT = Path(args[2]) if len(args) > 2 else Path('/tmp/t21e_result.json')
print('CAP=%s PARAMS=%s OUT=%s' % (CAP, PARAMS, JOUT), flush=True)
T, G = 10, 20000
LIDS = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]


def j_first_exact(Yq, thr_g, A_q, P_t, N_t):
    Gn = Yq.shape[0]
    j_first = np.full((Gn, T), -1, np.int8)
    frozen = np.zeros((Gn, T), bool)
    for j in range(23, -1, -1):
        Vtop = (Yq >> j) @ A_q.T
        Vmin = (Vtop << j) + N_t * ((1 << j) - 1)
        Vmax = (Vtop << j) + P_t * ((1 << j) - 1)
        lock = (Vmin >= thr_g) | (Vmax < thr_g)
        newly = lock & ~frozen
        j_first[newly] = j
        frozen |= lock
    assert frozen.all()
    return j_first


def one_trace(packed, prm, sid):
    S = np.unpackbits(packed, axis=1, bitorder='little')[:, :prm['W'].shape[1]]
    N = S.shape[0]
    P = N // T
    W = prm['W'].astype(np.float64)
    A = prm['A'].astype(np.float64)
    gamma, beta = prm['gamma'], prm['beta']
    bias, center = prm['bias'], prm['center']
    theta_src = float(np.asarray(prm['theta_src']).ravel()[0])
    H = W.shape[0]
    R = A.sum(1).reshape(T, 1)
    direction = np.sign(gamma)
    Sf = S.astype(np.float64)

    tau = np.zeros((T, H))
    for lo in range(0, H, 32):
        hi = min(H, lo + 32)
        Y = (Sf @ W[lo:hi].T).reshape(T, P, hi - lo) * theta_src
        mu = Y.mean((0, 1))
        var = ((Y - mu) ** 2).mean((0, 1))
        tau[:, lo:hi] = (mu * R + np.sqrt(var + 1e-5) / gamma[lo:hi]
                         * (1.0 + center - bias - beta[lo:hi] * R)) * direction[None, lo:hi]

    A_q = np.rint(A * 4096).astype(np.int64)
    A_q = np.where(A_q >= (1 << 15), A_q - (1 << 16), A_q)
    P_t = A_q.clip(min=0).sum(1)
    N_t = A_q.clip(max=0).sum(1)
    tau_q = np.rint(tau * (1 << 14)).astype(np.int64)
    thr = np.where(direction[None, :] < 0, -(tau_q << 12) + 1, tau_q << 12)
    thr = np.where(thr >= (1 << 47), thr - (1 << 48), thr)
    thr = np.where(thr < -(1 << 47), thr + (1 << 48), thr)

    rng = np.random.default_rng(20260914 + sid)
    ps = rng.integers(0, P, G)
    hs = rng.integers(0, H, G)
    Ysub = np.einsum('tgc,gc->tg',
                     Sf.reshape(T, P, S.shape[1])[:, ps, :], W[hs]) * theta_src
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64)
    Yq = Yq.T
    thr_g = thr[:, hs].T
    msb_g = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(G)])
    j_star = j_first_exact(Yq, thr_g, A_q, P_t, N_t).min(1)
    return float((1 + np.maximum(np.maximum(msb_g - j_star, 0), 1)).mean() / 24)


def main():
    z = np.load(PARAMS)
    cap = np.load(CAP)
    prms = {}
    for lid in LIDS:
        prms[lid] = {f: z['L%d_%s' % (lid, f)]
                     for f in ('W', 'A', 'gamma', 'beta', 'bias', 'center', 'theta_src')}
    res = []
    for lid in LIDS:
        for sid in range(40):
            key = 's%02d_L%02d' % (sid, lid)
            if key not in cap:
                continue
            r = one_trace(cap[key], prms[lid], sid)
            res.append({'lid': lid, 'sid': sid, 'ratio': r})
            if sid % 10 == 9:
                print('L%02d s%02d %.1f%%' % (lid, sid, 100 * r), flush=True)
        print('L%02d mean %.2f%%' % (lid, 100 * np.mean([x['ratio'] for x in res if x['lid'] == lid])), flush=True)
    by = {}
    for x in res:
        by.setdefault(x['lid'], []).append(x['ratio'])
    summary = {'layers': [{'lid': lid, 'mean': float(np.mean(v)),
                           'min': float(np.min(v)), 'max': float(np.max(v))}
                          for lid, v in sorted(by.items())],
               'network_mean': float(np.mean([np.mean(v) for v in by.values()])),
               'traces': len(res)}
    JOUT.write_text(json.dumps({'summary': summary, 'traces': res}, indent=1))
    print('NETWORK %.2f%% (%d traces)' % (100 * summary['network_mean'], len(res)))
    print('saved', JOUT)


if __name__ == '__main__':
    main()
