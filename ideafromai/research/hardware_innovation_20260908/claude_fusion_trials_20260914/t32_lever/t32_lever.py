#!/usr/bin/env python3
"""T32：证书供数的训练杠杆类型判定——幅度 vs 结构（决定卡 3 是否成立）。

动机：训练侧挖掘代理给出旗舰新轴"卡 3 = IBP/认证训练族"，其数学同一性是
`L1_t = Σ|A[t,c]|` 就是区间界宽系数、`ρ=|δ|/L1` 就是认证半径 → 结论是
"直接对 L1_t 做正则"。但 T16/T30 已给两条**反向**证据：
  - T16：λ=2（把 δ 放大 2×）供数仅 17.5% → 17.3%；
  - T30：跨层 log ρ vs 供数 r = −0.054。
即供数对 ρ 只是 **log 敏感**。若如此，则"均匀缩 A 的幅度"很可能**零收益**——
因为整层输出被 BN 尺度吸收时 ρ 不变。本脚本用真实 traces 做三类杠杆的对照：

  杠杆 S（尺度等价）：A→2A 且 thr→2·thr ⇒ 判决与平面数**应当逐字节不变**
                     （证明"整层缩幅 + BN 重标定"= 零供数收益，纯解析）。
  杠杆 M（仅幅度）：A→A/2，thr 不变 ⇒ 判决大量翻转（AEE 崩），供数下降多少？
  杠杆 K（结构置零）：逐行清零 C−keep 个最小 |A_q| ⇒ 与 T20/T22 同定义，
                     比"同等供数节省下"的判决翻转代价。

判读：若 K 在**小翻转**下拿到大部分供数收益、而 M 要靠大翻转才换到，则
认证训练必须走**结构**（= 已有 top-k 轴），"幅度正则"不成立 → 卡 3 需要重定向。

口径：bits/组 = 10 × planes/组（T29 方法学修正后的主口径；不用拍比）。
自有代码；capture 只读。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t19_all_layers as t19  # noqa: E402

T = 10
G = 10000
LIDS = [8, 14, 20, 28]          # stage0..3 各一层
NSID = 20
KEEPS = [10, 8, 6, 5, 4, 3]
SHINKS = [1.0, 0.75, 0.5]       # 杠杆 M：A 缩放（thr 不变）
D = 24                          # signed24 词宽（bits/组 分母无关，仅报 planes）


def cert_planes(Yq, thr_g, A_q):
    """组级 MSB-first 证书递推：返回 (逐组 plane 数, 逐判决 raw 判决, 逐判决 j_first)。"""
    A_q = np.asarray(A_q, np.int64)
    P_t = A_q.clip(min=0).sum(1)
    N_t = A_q.clip(max=0).sum(1)
    Gn = Yq.shape[0]
    jf = np.full((Gn, T), -1, np.int8)
    frozen = np.zeros((Gn, T), bool)
    dec = np.zeros((Gn, T), bool)
    for j in range(23, -1, -1):
        Vtop = (Yq >> j) @ A_q.T
        Vmin = (Vtop << j) + N_t * ((1 << j) - 1)
        Vmax = (Vtop << j) + P_t * ((1 << j) - 1)
        lock = (Vmin >= thr_g) | (Vmax < thr_g)
        newly = lock & ~frozen
        jf[newly] = j
        dec[newly] = Vmin[newly] >= thr_g[newly]
        frozen |= lock
    assert frozen.all()
    msb = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(Gn)])
    planes = (msb - jf.min(1)).astype(np.int64)
    return planes, dec, jf


def trace_setup(packed, C, br, prm):
    """复刻 t19 的 Yq/thr/A_q 构造。"""
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
    tau_q = np.rint(tau * (1 << 14)).astype(np.int64)
    thr = np.where(direction[None, :] < 0, -(tau_q << 12) + 1, tau_q << 12)
    thr = np.where(thr >= (1 << 47), thr - (1 << 48), thr)
    thr = np.where(thr < -(1 << 47), thr + (1 << 48), thr)
    rng = np.random.default_rng(20260914)          # 与 T19/T31 同种子
    ps = rng.integers(0, P, G)
    hs = rng.integers(0, H, G)
    Ysub = np.einsum('tgc,gc->tg', Sf.reshape(T, P, C)[:, ps, :], W[hs]) * theta_src
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64).T
    return Yq, thr[:, hs].T, A_q


def mask_keep(A_q, keep):
    """逐行保留 |A_q| 最大的 keep 列，其余清零（T20/T22 定义）。"""
    C = A_q.shape[1]
    if keep >= C:
        return A_q
    order = np.argsort(np.abs(A_q), axis=1, kind='stable')
    drop = order[:, :C - keep]
    out = A_q.copy()
    np.put_along_axis(out, drop, 0, axis=1)
    return out


def main():
    z = np.load(t19.PARAMS)
    prms = {}
    for lid in LIDS:
        prms[lid] = {'lid': lid, 'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
                     'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
                     'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
                     'theta_src': z['L%d_theta_src' % lid]}
    sids = list(range(NSID))
    src = t19.parse_sources(sids)
    print('parsed', len(src), 'pairs;  layers', LIDS, flush=True)

    agg = {f'S|s={s}': [] for s in SHINKS}          # 尺度等价（thr 同缩）
    agg.update({f'M|s={s}': [] for s in SHINKS if s != 1.0})
    agg.update({f'K|keep={k}': [] for k in KEEPS})
    flips = {k: [] for k in agg}
    eq_check = []

    for lid in LIDS:
        for sid in sids:
            key = (sid, lid)
            if key not in src:
                continue
            packed, C, br = src[key]
            Yq, thr_g, A_q = trace_setup(packed, C, br, prms[lid])
            del src[key]
            p0, d0, _ = cert_planes(Yq, thr_g, A_q)
            agg['K|keep=10'].append(p0.mean())
            flips['K|keep=10'].append(0.0)
            agg['S|s=1.0'].append(p0.mean())
            flips['S|s=1.0'].append(0.0)

            # 杠杆 S：整层 2×/4×/… 与 thr 同缩（尺度等价）
            for s in SHINKS:
                if s == 1.0:
                    continue
                num, den = {0.75: (3, 4), 0.5: (1, 2)}[s]
                Aq_s = np.rint(A_q.astype(np.float64) * num / den).astype(np.int64)
                thr_s = np.rint(thr_g.astype(np.float64) * num / den).astype(np.int64)
                ps_, ds_, _ = cert_planes(Yq, thr_s, Aq_s)
                agg[f'S|s={s}'].append(ps_.mean())
                flips[f'S|s={s}'].append(float((ds_ != d0).mean()))
                eq_check.append(int((ps_ != p0).sum()))

            # 杠杆 M：仅缩 A，thr 不变（判决会崩）
            for s in SHINKS:
                if s == 1.0:
                    continue
                if s == 0.5:
                    Aq_m = np.floor_divide(A_q, 2)
                else:
                    Aq_m = np.rint(A_q.astype(np.float64) * 0.75).astype(np.int64)
                pm_, dm_, _ = cert_planes(Yq, thr_g, Aq_m)
                agg[f'M|s={s}'].append(pm_.mean())
                flips[f'M|s={s}'].append(float((dm_ != d0).mean()))

            # 杠杆 K：结构置零（逐行保留 keep 个）
            for k in KEEPS:
                if k >= C:
                    continue
                pk_, dk_, _ = cert_planes(Yq, thr_g, mask_keep(A_q, k))
                agg[f'K|keep={k}'].append(pk_.mean())
                flips[f'K|keep={k}'].append(float((dk_ != d0).mean()))
        print('  L%02d done' % lid, flush=True)

    base = float(np.mean(agg['K|keep=10']))
    print('\n== 基准（真实 A，keep=10）planes/组 = %.4f，bits/组 = %.2f =='
          % (base, 10 * base))
    print('  尺度等价性检验（S 杠杆，应与基准逐组零差）: 失配组数总和 = %d'
          % int(np.sum(eq_check)))
    print('\n%-14s %10s %10s %10s %12s' %
          ('杠杆', 'planes/组', 'Δ供数', '判决翻转', '翻转/Δ供数'))
    print('%-14s %10.4f %10s %10s %12s' % ('baseline', base, '—', '0.00%', '—'))
    rows = []
    for k in agg:
        if k == 'K|keep=10':
            continue
        m = float(np.mean(agg[k]))
        fl = float(np.mean(flips[k]))
        d = m / base - 1
        rows.append({'lever': k, 'planes': m, 'd_supply': d, 'flip': fl,
                     'flip_per_dsupply': (fl / abs(d)) if d else float('inf')})
    for r in sorted(rows, key=lambda r: r['lever']):
        print('%-14s %10.4f %9.2f%% %9.2f%% %12.3f'
              % (r['lever'], r['planes'], 100 * r['d_supply'], 100 * r['flip'],
                 r['flip_per_dsupply'] if np.isfinite(r['flip_per_dsupply']) else -1))

    out = {'baseline_planes': base, 'baseline_bits': 10 * base,
           'scale_equivariance_mismatch_groups': int(np.sum(eq_check)),
           'rows': rows, 'n_layers': LIDS, 'n_sid': NSID, 'G': G}
    (ROOT / 'results' / 't32_lever.json').write_text(json.dumps(out, indent=1) + '\n')
    print('\nwrote results/t32_lever.json')


if __name__ == '__main__':
    main()
