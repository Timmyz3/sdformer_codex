#!/usr/bin/env python3
"""T31：逐元素跨层 lock-depth 相关性——跨层级联路线的决定性 gate。

背景：T30 从"层均值"层面看跨层无供数信号（相邻层 r=0.788 ≈ 远层 r=0.739，
相关全来自序列级全局难度因子）。但架构挖掘代理的卡片 3（SEENN 式跨层级联）
给的门是**逐元素级**：

    同一 token/空间位置，相邻层 lock-depth 的 Pearson r ≥ 0.4 → 跨层可预测性存在；
    r < 0.2 → 整条"跨层级联"路线直接杀。

本脚本下沉到**逐组**（T19 的 (ps,hs) 采样：同一空间位置 token、同一隐藏通道）：
对每个 sid，12 个层用**同一 RNG 种子**采样同一批组（group g 在各层指向同一
(ps[g],hs[g])），计算各层逐组 lock-depth（MSB-first 证书锁定的平面数），
再做层间 Pearson r。

对照：①同 sid 内打乱组序（置换零假设，应为 ~0）证明估计量有效；
②层均值层面的 r（复现 T30 的"全局因子"现象）作对比。

口径与 T19/T27 一致：组级 MSB-first 证书递推，planes_g = msb_g − j_star_g
（j_star = 该组 10 个判决全部锁定的平面；常数偏移不影响 Pearson）。
自有代码；capture 只读。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t19_all_layers as t19  # noqa: E402

T, G = 10, 20000
LIDS = t19.LIDS


def per_group_planes(packed, C, br, prm, sid):
    """复刻 t19.one_trace 的前半（Yq/thr/A_q 同口径），返回逐组 plane 数。"""
    S = np.unpackbits(packed.reshape(-1, br), axis=1, bitorder='little')[:, :C]
    N = S.shape[0]
    P = N // T
    W = prm['W'].astype(np.float64)
    A, gamma, beta = prm['A'], prm['gamma'], prm['beta']
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
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1)
    Yq = Yq.astype(np.int64).T
    thr_g = thr[:, hs].T
    msb = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(G)])

    jf = t19.j_first_exact(Yq, thr_g, A_q, P_t, N_t)
    j_star = jf.min(1)
    planes = (msb - j_star).astype(np.int16)
    return planes, msb, j_star


def main():
    z = np.load(t19.PARAMS)
    prms = {}
    for lid in LIDS:
        mod = str(z['L%d_module' % lid])
        prms[lid] = {
            'lid': lid,
            'stage': int(mod.split('layers.')[1].split('.')[0]),
            'block': int(mod.split('swin_blocks.')[1].split('.')[0]),
            'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
            'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
            'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
            'theta_src': z['L%d_theta_src' % lid],
        }
    sids = list(range(40))
    src = t19.parse_sources(sids)
    print('parsed', len(src), '(sid,lid) pairs', flush=True)

    nL, nS = len(LIDS), len(sids)
    PL = np.zeros((nL, nS, G), np.int16)
    MSB = np.zeros((nL, nS), np.float64)
    for li, lid in enumerate(LIDS):
        for si, sid in enumerate(sids):
            key = (sid, lid)
            if key not in src:
                print('  MISSING', key, flush=True)
                continue
            packed, C, br = src[key]
            planes, msb, j_star = per_group_planes(packed, C, br, prms[lid], sid)
            PL[li, si] = planes
            MSB[li, si] = msb.mean()
            del src[key]
        print('L%02d done: planes/group mean %.3f  msb %.2f'
              % (lid, PL[li].mean(), MSB[li].mean()), flush=True)

    print('\n== 逐组 lock-depth（planes/group）逐层均值 ==', flush=True)
    for li, lid in enumerate(LIDS):
        print('  L%02d st%d blk%d: %.3f  (跨 sid std %.3f)'
              % (lid, prms[lid]['stage'], prms[lid]['block'],
                 PL[li].mean(), PL[li].mean(1).std()))

    # ---- 层间相关：逐 sid 内、逐组（去层均值 → Pearson）----
    adj_r, far_r, dist_r = [], [], {}
    null_r = []
    for si in range(nS):
        Pw = PL[:, si, :].astype(np.float64)          # nL × G
        Pw = Pw - Pw.mean(1, keepdims=True)
        sd = Pw.std(1, keepdims=True)
        sd[sd == 0] = 1.0
        Z = Pw / sd
        Cc = Z @ Z.T / G                              # nL × nL Pearson
        for i in range(nL - 1):
            adj_r.append(Cc[i, i + 1])
        for i in range(nL):
            for j in range(i + 1, nL):
                d = j - i
                dist_r.setdefault(d, []).append(Cc[i, j])
                if d >= 3:
                    far_r.append(Cc[i, j])
        # 置换零假设：把 layer 0 的组序打乱后与 layer 1 相关
        rng = np.random.default_rng(999 + si)
        perm = rng.permutation(G)
        a = Z[0]
        b = Z[1][perm]
        null_r.append(float(a @ b / G))

    adj_r = np.array(adj_r)
    far_r = np.array(far_r)
    null_r = np.array(null_r)
    print('\n== Q1 逐元素跨层 Pearson r（40 sid 内）==', flush=True)
    print('  相邻层对 r: mean %.3f  median %.3f  范围 %.3f–%.3f  (>0.4 占比 %.1f%%)'
          % (adj_r.mean(), np.median(adj_r), adj_r.min(), adj_r.max(),
             100 * (adj_r > 0.4).mean()))
    print('  远层对 (|i-j|>=3) r: mean %.3f  median %.3f'
          % (far_r.mean(), np.median(far_r)))
    print('  置换零假设 r: mean %.4f  std %.4f  |r|max %.4f'
          % (null_r.mean(), null_r.std(), np.abs(null_r).max()))
    print('  按层距:', end='')
    for d in sorted(dist_r):
        print(' d=%d %.3f' % (d, np.mean(dist_r[d])), end='')
    print(flush=True)

    # ---- 对照：层均值层面的 r（复现 T30 的全局因子）----
    lm = PL.mean(2)                                   # nL × nS
    lm_c = lm - lm.mean(1, keepdims=True)
    Cs = np.corrcoef(lm_c)
    adj_seq = [Cs[i, i + 1] for i in range(nL - 1)]
    far_seq = [Cs[i, j] for i in range(nL) for j in range(i + 3, nL)]
    print('\n== Q2 对照：层均值层面（跨 40 sid）相关 —— 复现 T30 ==', flush=True)
    print('  相邻层 r(层均值): mean %.3f' % np.mean(adj_seq))
    print('  远层 r(层均值):   mean %.3f' % np.mean(far_seq))

    out = {
        'planes_per_layer_mean': [float(PL[i].mean()) for i in range(nL)],
        'msb_per_layer_mean': [float(MSB[i].mean()) for i in range(nL)],
        'adj_r_mean': float(adj_r.mean()), 'adj_r_median': float(np.median(adj_r)),
        'adj_r_min': float(adj_r.min()), 'adj_r_max': float(adj_r.max()),
        'adj_frac_gt_0.4': float((adj_r > 0.4).mean()),
        'far_r_mean': float(far_r.mean()), 'far_r_median': float(np.median(far_r)),
        'null_r_mean': float(null_r.mean()), 'null_r_std': float(null_r.std()),
        'null_absmax': float(np.abs(null_r).max()),
        'dist_r': {int(d): float(np.mean(v)) for d, v in dist_r.items()},
        'seq_level_adj_r_mean': float(np.mean(adj_seq)),
        'seq_level_far_r_mean': float(np.mean(far_seq)),
    }
    (ROOT / 'results' / 't31_lockdepth.json').write_text(
        json.dumps(out, indent=1) + '\n')
    print('\nwrote results/t31_lockdepth.json')


if __name__ == '__main__':
    main()
