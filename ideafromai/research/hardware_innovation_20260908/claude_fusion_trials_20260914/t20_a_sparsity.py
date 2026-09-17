#!/usr/bin/env python3
"""T20（自提 idea）：证书感知的 A 矩阵稀疏化——供数/翻转交换率静态测量。

动机（T16 解析模型）：锁定平面下界 j_first ≥ floor(log2(ρ+1))，ρ=|δ|/L1_t，
L1_t = Σ_s|A_q[t,s]|。A 稀疏化（每行保留 top-k 幅值）直接降 L1 → ρ 升 →
供数按 ~log2(c) 平面下降。代价：门函数改变（判决翻转），恢复路径=sd5ai 微调。

静态实验（本脚本）：对 stage0/3 block0 两层 × 40 序列，k∈{10..2}：
- Ā = 每行保留 |a| 前 k 大，其余置零；
- 用 Ā 重算 R̄/tau/thr（部署网络=Ā 网络，f14 教训：以部署链自身为基准）；
- 测：证书拍比、判决翻转率（vs k=10 原网络）、ρ 中位；
- 对照 T16 预测：拍比下降应 ≈ log2(L1_10/L1_k) 平面/组。

若翻转率在可用 k 下 <~1%（微调可恢复量级）→ sd5ai 微调实验立项；
否则杀，记录为"A 结构自由度已被网络功能锁死"的证据。自有代码。
"""
import json
from pathlib import Path

import numpy as np

from t17_more_traces import parse_sources, j_first_exact, j_first_analytical, T, G

ROOT = Path(__file__).resolve().parent
BN = ROOT.parents[0] / 'bn_state'
JOUT = ROOT / 'results' / 't20_result.json'
KS = [10, 9, 8, 7, 6, 5, 4, 3, 2]


def sparsify(A, k):
    if k >= A.shape[1]:
        return A.copy()
    Ab = np.abs(A)
    order = np.argsort(-Ab, axis=1, kind='stable')
    keep = order[:, :k]
    mask = np.zeros_like(A, bool)
    rows = np.arange(A.shape[0])[:, None]
    mask[rows, keep] = True
    return np.where(mask, A, 0.0)


def one_trace(S, st, sid, A_mod_list):
    """A_mod_list: [(k, Ā)]；一次重算 tau 基础量（mu/var 与 A 无关），各 k 复用。"""
    z = np.load(BN / f'trace_s0_stage{st}.npz')
    W = z['W'].astype(np.float64)
    gamma, beta = z['gamma'], z['beta']
    bias, center = z['bias'], z['center']
    A0 = z['A']
    N, C = S.shape
    P = N // T
    H = W.shape[0]
    direction = np.sign(gamma)
    Sf = S.astype(np.float64)

    # mu/var 与 A 无关（Y=S@W.T）；R 依赖 A
    mu = np.zeros(H)
    var = np.zeros(H)
    for lo in range(0, H, 32):
        hi = min(H, lo + 32)
        Y = (Sf @ W[lo:hi].T).reshape(T, P, hi - lo)
        mu[lo:hi] = Y.mean((0, 1))
        var[lo:hi] = ((Y - mu[lo:hi]) ** 2).mean((0, 1))

    rng = np.random.default_rng(20260914 + sid)
    ps = rng.integers(0, P, G)
    hs = rng.integers(0, H, G)
    Ysub = np.einsum('tgc,gc->tg', Sf.reshape(T, P, C)[:, ps, :], W[hs]).T  # (G,T)
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64)
    msb_g = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(G)])
    dir_g = direction[hs]

    out = {}
    dec_ref = None
    for k, Am in A_mod_list:
        R = Am.sum(1).reshape(T, 1)
        tau = (mu[None] * R + np.sqrt(var + 1e-5)[None] / gamma[None]
               * (1.0 + center - bias - beta[None] * R)) * direction[None]
        A_q = np.rint(Am * 4096).astype(np.int64)
        A_q = np.where(A_q >= (1 << 15), A_q - (1 << 16), A_q)
        P_t = A_q.clip(min=0).sum(1)
        N_t = A_q.clip(max=0).sum(1)
        L1 = np.abs(A_q).sum(1)
        tau_q = np.rint(tau * (1 << 14)).astype(np.int64)
        thr = np.where(direction[None] < 0, -(tau_q << 12) + 1, tau_q << 12)
        thr = np.where(thr >= (1 << 47), thr - (1 << 48), thr)
        thr = np.where(thr < -(1 << 47), thr + (1 << 48), thr)
        thr_g = thr[:, hs].T

        Vfull = Yq @ A_q.T
        dec = np.where(dir_g[:, None] > 0, Vfull >= thr_g, ~(Vfull >= thr_g))
        if k == KS[0]:
            dec_ref = dec
            flips = 0.0
        else:
            flips = float((dec != dec_ref).mean())

        jf = j_first_exact(Yq, thr_g, A_q, P_t, N_t)
        j_star = jf.min(1)
        cyc = 1 + np.maximum(np.maximum(msb_g - j_star, 0), 1)
        delta = Vfull - thr_g
        out[k] = {
            'ratio': float(cyc.mean() / 24),
            'flip_rate': flips,
            'rho_median': float(np.median(np.abs(delta) / L1[None, :])),
            'L1_mean': float(L1.mean()),
        }
    return out


def main():
    src = parse_sources(list(range(40)))
    print('parsed', len(src), 'pairs', flush=True)
    res = {}
    for st in (0, 3):
        A0 = np.load(BN / f'trace_s0_stage{st}.npz')['A']
        A_mods = [(k, sparsify(A0, k)) for k in KS]
        per_k = {k: {'ratio': [], 'flip': [], 'rho': [], 'L1': []} for k in KS}
        for (sid, sst), S in sorted(src.items()):
            if sst != st:
                continue
            r = one_trace(S, st, sid, A_mods)
            for k, v in r.items():
                per_k[k]['ratio'].append(v['ratio'])
                per_k[k]['flip'].append(v['flip_rate'])
                per_k[k]['rho'].append(v['rho_median'])
                per_k[k]['L1'].append(v['L1_mean'])
        res['stage%d' % st] = {
            str(k): {
                'ratio_mean': float(np.mean(per_k[k]['ratio'])),
                'flip_mean': float(np.mean(per_k[k]['flip'])),
                'flip_max': float(np.max(per_k[k]['flip'])),
                'rho_median': float(np.mean(per_k[k]['rho'])),
                'L1_mean': float(np.mean(per_k[k]['L1'])),
            } for k in KS}
        print('stage%d done' % st, flush=True)
        for k in KS:
            d = res['stage%d' % st][str(k)]
            print('  k=%2d ratio=%.1f%% flip=%.3f%% (max %.3f%%) rho=%.0f L1=%.2f'
                  % (k, d['ratio_mean'] * 100, d['flip_mean'] * 100,
                     d['flip_max'] * 100, d['rho_median'], d['L1_mean']), flush=True)
    JOUT.write_text(json.dumps(res, indent=1))
    print('saved', JOUT)


if __name__ == '__main__':
    main()
