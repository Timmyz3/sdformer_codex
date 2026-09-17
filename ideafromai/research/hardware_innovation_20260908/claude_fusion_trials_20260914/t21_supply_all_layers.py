#!/usr/bin/env python3
"""T21 配套：masked-A（top-k 逐行）下全部 12 个 fc1 层的证书供数比。

T20 只测了 stage0/3 block0；本脚本用 t19 的参数/解析器把 k∈{5,4,3,2} 扫到
全部 12 层 × 10 序列，给出与 T21a AEE 配对的全网供数数字。
口径与 t19/t20 相同：原始 capture S + Ā 网络（R̄/tau/thr 随 Ā 重算），
静态近似（部署态 S 会随 Ā 变化，微调后需重capture）。
产出 results/t21_supply.json。
"""
import json
import struct
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CAP = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07'
           '/results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture'
           '_s40_r1_20260901')
PARAMS = ROOT / 't19_gate_params.npz'
JOUT = ROOT / 'results' / 't21_supply.json'
T, G, SIDS = 10, 20000, 10
KS = [5, 4, 3, 2]
HEADER = struct.Struct('<8sHH11I')
LIDS = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]


def parse_sources(sids):
    want = {(sid, lid) for sid in sids for lid in LIDS}
    chunks = {k: [] for k in want}
    meta = {}
    with (CAP / 'fc_frames.bin').open('rb') as f:
        while raw := f.read(HEADER.size):
            magic, ver, hs, lid, sid, fi, start, n, C, br, nnz, rb, cb, _ = \
                HEADER.unpack(raw)
            if (sid, lid) not in chunks:
                f.seek(cb, 1)
                continue
            payload = zlib.decompress(f.read(cb))
            if (sid, lid) in meta and meta[sid, lid][0] != C:
                raise ValueError('C changed')
            meta[sid, lid] = (C, br)
            chunks[sid, lid].append(payload[:n * br])
    out = {}
    for k, lst in chunks.items():
        if lst:
            C, br = meta[k]
            out[k] = (np.concatenate([np.frombuffer(p, np.uint8) for p in lst]), C, br)
    return out


def sparsify(A, k):
    if k >= A.shape[1]:
        return A
    idx = np.argpartition(np.abs(A), -k, axis=1)[:, -k:]
    m = np.zeros_like(A, dtype=bool)
    np.put_along_axis(m, idx, True, axis=1)
    return A * m


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


def one_trace(packed, C, br, prm, sid, k):
    S = np.unpackbits(packed.reshape(-1, br), axis=1, bitorder='little')[:, :C]
    N = S.shape[0]
    P = N // T
    W = prm['W'].astype(np.float64)
    A = sparsify(prm['A'].astype(np.float64), k)
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
                     Sf.reshape(T, P, C)[:, ps, :], W[hs]) * theta_src
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64)
    Yq = Yq.T
    thr_g = thr[:, hs].T
    msb_g = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(G)])

    j_star = j_first_exact(Yq, thr_g, A_q, P_t, N_t).min(1)
    cyc = (1 + np.maximum(np.maximum(msb_g - j_star, 0), 1)).mean()
    return float(cyc / 24)


def main():
    z = np.load(PARAMS)
    prms = {}
    for lid in LIDS:
        prms[lid] = {
            'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
            'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
            'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
            'theta_src': z['L%d_theta_src' % lid],
        }
    sids = list(range(SIDS))
    src = parse_sources(sids)
    print('parsed', len(src), 'pairs', flush=True)
    res = {}
    for k in KS:
        per_layer = []
        for lid in LIDS:
            ratios = []
            for sid in sids:
                if (sid, lid) not in src:
                    continue
                packed, C, br = src[sid, lid]
                ratios.append(one_trace(packed, C, br, prms[lid], sid, k))
            per_layer.append({'lid': lid, 'mean': float(np.mean(ratios)),
                              'min': float(np.min(ratios)), 'max': float(np.max(ratios))})
            print('k=%d L%02d %.1f%%' % (k, lid, 100 * per_layer[-1]['mean']), flush=True)
        allm = [x['mean'] for x in per_layer]
        res['k%d' % k] = {'layers': per_layer,
                          'network_mean': float(np.mean(allm)),
                          'network_minmax': [float(np.min(allm)), float(np.max(allm))]}
        print('k=%d NETWORK %.2f%% (layers %.1f–%.1f%%)'
              % (k, 100 * res['k%d' % k]['network_mean'],
                 100 * np.min(allm), 100 * np.max(allm)), flush=True)
    # k=10 baseline from t19 (subset mean over same sids not stored; use full-40 means)
    t19 = json.load(open(ROOT / 'results' / 't19_result.json'))
    by = {}
    for r in t19:
        by.setdefault(r['lid'], []).append(r['ratio_measured'])
    res['k10'] = {'layers': [{'lid': lid, 'mean': float(np.mean(v))}
                             for lid, v in sorted(by.items())],
                  'network_mean': float(np.mean([np.mean(v) for v in by.values()]))}
    print('k=10 NETWORK (t19, 40 sids) %.2f%%' % (100 * res['k10']['network_mean']))
    JOUT.write_text(json.dumps(res, indent=1))
    print('saved', JOUT)


if __name__ == '__main__':
    main()
