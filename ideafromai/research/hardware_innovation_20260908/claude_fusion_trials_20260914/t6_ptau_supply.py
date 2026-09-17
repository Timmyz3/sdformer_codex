"""T6（C1 卡）：ptau 可实现版供数实测——T5 的 oracle 边界闭合。

T5 的 16.6-17.0% 拍比用终态 tau（BN 全局矩，oracle）。部署态 thr 是静态常数，
与 trace 终态矩有分布差。本试验用块前缀矩 ptau（B=32，含当前块，与生产树
bn_state/build_traces.py 同口径，只读其定义不复用代码）作为 tau 扰动的真实幅度，
按 thr_k = tau_final + k·(ptau − tau_final)，k ∈ {0, 0.5, 1, 2, 4} 扫供数敏感度：
- k=1 即 ptau 直用；k>1 外推放大；
- 每口径跑 T5 同款位平面证书模型（BF 共享指数 + 组级终止），输出拍比与
  判决相对 k=0 全深判决的偏差（网络级效应，生产实测 k=1 为 0.30%）。
采样组与 T5 同种子（可直接对照）。样本内数值试验，非 RTL。自有代码。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HW = ROOT.parents[0]
TRACES = sorted((HW / 'bn_state').glob('trace_*.npz'))
G_PER_TRACE = 20000
SEED = 20260914
B_BLOCK = 32
KS = (0.0, 0.5, 1.0, 2.0, 4.0)


def one_trace(trace):
    z = np.load(trace)
    C = int(z['W'].shape[1])
    S = np.unpackbits(z['source_packed'], axis=1, bitorder='little')[:, :C].astype(np.float64)
    N = S.shape[0]
    T, P = 10, N // 10
    H = z['W'].shape[0]
    W = z['W'].astype(np.float64)
    A, gamma, beta = z['A'], z['gamma'], z['beta']
    bias, center = z['bias'], z['center']
    theta = 1.0
    R = A.sum(1).reshape(T, 1)
    direction = np.sign(gamma)
    nb = (P + B_BLOCK - 1) // B_BLOCK
    ends = np.minimum((np.arange(nb) + 1) * B_BLOCK, P) - 1

    # 终态 tau + 块前缀 ptau（按 h 分块控内存），均 f14 量化后折方向得 thr
    tau_q = np.zeros((T, H), np.int64)
    ptau_q = np.zeros((T, nb, H), np.int64)
    for lo in range(0, H, 32):
        hi = min(H, lo + 32)
        Y = (S @ W[lo:hi].T).reshape(T, P, hi - lo)
        mu = Y.mean((0, 1))
        var = ((Y - mu) ** 2).mean((0, 1))
        tau = (mu * R + np.sqrt(var + 1e-5) / gamma[lo:hi]
               * (theta + center - bias - beta[lo:hi] * R)) * direction[None, lo:hi]
        tau_q[:, lo:hi] = np.rint(tau * (1 << 14))
        cs = np.cumsum(Y.sum(0), 0)                     # (P, hi-lo)
        cq = np.cumsum((Y * Y).sum(0), 0)
        denom = ((ends + 1) * T)[:, None]
        pmu = cs[ends] / denom
        pvar = np.maximum(cq[ends] / denom - pmu * pmu, 0)
        ptau = (pmu[None] * R[:, None] + np.sqrt(pvar + 1e-5)[None] / gamma[lo:hi][None]
                * (theta + center - bias - beta[lo:hi] * R)[:, None]) * direction[None, lo:hi]
        ptau_q[:, :, lo:hi] = np.rint(ptau * (1 << 14))
        del Y
    dneg = (direction < 0)[None, :]                       # (1,H)
    # U 侧 thr（审计修正）：D=−1 取反 +1；48bit 合同（同 t5）
    thr_fin = np.where(dneg, -(tau_q << 12) + 1, tau_q << 12)
    thr_fin = np.rint(thr_fin).astype(np.int64)
    thr_pre = np.where(dneg[None, :, :], -(ptau_q << 12) + 1, ptau_q << 12)  # (T,nb,H)
    thr_pre = np.rint(thr_pre).astype(np.int64)

    A_q = np.rint(A.astype(np.float64) * 4096).astype(np.int64)
    P_t = A_q.clip(min=0).sum(1)
    N_t = A_q.clip(max=0).sum(1)
    Dflag = (direction > 0).astype(np.int64)

    # 采样组（与 T5 同种子）
    rng = np.random.default_rng(SEED)
    ps = rng.integers(0, P, G_PER_TRACE)
    hs = rng.integers(0, H, G_PER_TRACE)
    blocks = np.minimum(ps // B_BLOCK, nb - 1)
    Ssub = S.reshape(T, P, C)[:, ps, :]
    Wsub = W[hs]
    Ysub = np.einsum('tgc,gc->tg', Ssub, Wsub)
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64).T  # (G,T)
    msb_g = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(G_PER_TRACE)])

    thr_g_fin = thr_fin[:, hs].T                          # (G,T) k=0
    thr_g_pre = thr_pre[:, blocks, hs].T                  # (T,nb,H) 索引 -> (G,T)
    D_g = np.broadcast_to(Dflag[hs][:, None], (G_PER_TRACE, T))

    Vfull = np.einsum('gs,ts->gt', Yq, A_q)
    dec_ref = np.where(D_g > 0, Vfull >= thr_g_fin, Vfull < thr_g_fin)

    rows = []
    for k in KS:
        thr_k = np.rint(thr_g_fin + k * (thr_g_pre - thr_g_fin)).astype(np.int64)
        j_first = np.full((G_PER_TRACE, T), -1, np.int8)
        dec_raw = np.zeros((G_PER_TRACE, T), bool)
        frozen = np.zeros((G_PER_TRACE, T), bool)
        for j in range(23, -1, -1):
            Ytop = Yq >> j
            Vtop = np.einsum('gs,ts->gt', Ytop, A_q)
            Vmin = (Vtop << j) + N_t[None, :] * ((1 << j) - 1)
            Vmax = (Vtop << j) + P_t[None, :] * ((1 << j) - 1)
            lock = (Vmin >= thr_k) | (Vmax < thr_k)
            newly = lock & ~frozen
            j_first[newly] = j
            dec_raw[newly] = Vmin[newly] >= thr_k[newly]
            frozen |= lock
        assert frozen.all()
        dec_k = np.where(D_g > 0, dec_raw, ~dec_raw)
        j_star = j_first.min(1)
        planes_bf = np.maximum(msb_g - j_star, 0)
        cyc_bf = 1 + np.maximum(planes_bf, 1)
        cyc_fx = 1 + np.maximum(23 - j_star, 1)
        rows.append({
            'k': k,
            'bf_cert_cycle_ratio': float(cyc_bf.mean() / 24.0),
            'fx_cert_cycle_ratio': float(cyc_fx.mean() / 24.0),
            'bf_planes_mean': float(planes_bf.mean()),
            'dec_vs_k0_mismatch_frac': float((dec_k != dec_ref).mean()),
        })

    delta = thr_g_pre - thr_g_fin
    return {
        'trace': str(trace.relative_to(HW)),
        'groups': G_PER_TRACE, 'H': H, 'P': P, 'nb': nb,
        'thr_delta_abs_mean': float(np.abs(delta).mean()),
        'thr_delta_abs_p99': float(np.quantile(np.abs(delta), 0.99)),
        'thr_abs_mean': float(np.abs(thr_g_fin).mean()),
        'ks': rows,
    }


def main():
    results = [one_trace(t) for t in TRACES]
    out = {'traces': results, 'ks': KS,
           'note': ('thr_k = thr_final + k·(thr_ptau − thr_final)，ptau=B32 块前缀矩（含当前块，'
                    '与 bn_state/build_traces.py 同口径）。k=1 即 ptau 直用；供数=BF 共享指数+组级'
                    '证书终止拍比（vs FX 全深 25 拍/组）。dec_vs_k0 为网络级判决偏差（生产实测 '
                    'k=1 全判决 0.30%）。样本内数值试验，非 RTL。')}
    (ROOT / 'results' / 't6_ptau_supply.json').write_text(json.dumps(out, indent=1) + '\n')
    for r in results:
        for row in r['ks']:
            print('%-28s k=%.1f bf_cert=%.4f planes=%.2f dec_mismatch=%.5f' %
                  (r['trace'], row['k'], row['bf_cert_cycle_ratio'],
                   row['bf_planes_mean'], row['dec_vs_k0_mismatch_frac']))


if __name__ == '__main__':
    main()
