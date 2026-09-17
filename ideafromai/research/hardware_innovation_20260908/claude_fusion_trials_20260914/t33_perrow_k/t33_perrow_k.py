#!/usr/bin/env python3
"""T33：逐行（逐判决）非均匀 k 分配——卡 4（RigL/LTP per-row k）的本地供给侧验证。

动机：T20–T22 只做了**均匀** k（每个 MLP 的 10 个判决行共享同一个 keep 数）。
训练侧挖掘代理的卡 4 指出：LTP 允许**每行 k 不同**，而 A_q 的行是"判决 t' 的系数行"，
其 L1[t']=Σ_t|A[t',t]| 正是该判决的界系数 → 行间异质性应可被利用。

本脚本在花 GPU 之前，先在真实 traces 上比较**同等 slot 预算**下三种分配规则的
供数 / 判决翻转（AEE 代理）：

  uniform   ：每行 keep = S/10（= T20–T22 口径）
  prop      ：keep ∝ L1 行（大 L1 行给更多 slot，"保住重权"）
  inv       ：keep ∝ 1/L1 行（大 L1 行给更少 slot，"收紧松界"）
  opt-desc  ：按 L1 降序贪心灌满（prop 的极限）
  opt-asc   ：按 L1 升序贪心灌满（inv 的极限）

口径：bits/组 = 10×planes/组（主口径）；翻转率按 raw 判决与 baseline 比。
自有代码；capture 只读。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 't32_lever'))
from t32_lever import cert_planes, trace_setup, mask_keep  # noqa: E402
import t19_all_layers as t19  # noqa: E402

T = 10
G = 4000
LIDS = [8, 14, 20, 28]
NSID = 10
BUDGETS = [50, 40, 30, 20]


def alloc_uniform(L1, S):
    return np.full(T, S / T)


def alloc_prop(L1, S):
    w = L1 / L1.sum()
    return w * S


def alloc_inv(L1, S):
    w = (1.0 / np.maximum(L1, 1e-9))
    w = w / w.sum()
    return w * S


def alloc_greedy(L1, S, ascending):
    """极限形式：把 slot 按 L1 升/降序灌满（先给一行的 10 个，再给下一行）。"""
    order = np.argsort(L1) if ascending else np.argsort(-L1)
    keep = np.ones(T, int)
    rem = S - T
    for idx in order:
        if rem <= 0:
            break
        take = min(rem, T - 1)
        keep[idx] += take
        rem -= take
    return keep


def slots_to_keep(frac, S):
    """把连续份额转成整数 keep∈[1,10] 且总和恰为 S。"""
    keep = np.clip(np.rint(frac), 1, T).astype(int)
    while keep.sum() > S:
        cand = np.where(keep > 1)[0]
        if len(cand) == 0:
            break
        j = cand[np.argmax(keep[cand] - frac[cand])]
        keep[j] -= 1
    while keep.sum() < S:
        cand = np.where(keep < T)[0]
        if len(cand) == 0:
            break
        j = cand[np.argmin(keep[cand] - frac[cand])]
        keep[j] += 1
    return keep


def mask_per_row(A_q, keep):
    C = A_q.shape[1]
    out = A_q.copy()
    for r in range(A_q.shape[0]):
        k = int(keep[r])
        if k >= C:
            continue
        order = np.argsort(np.abs(A_q[r]), kind='stable')
        out[r, order[:C - k]] = 0
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
    print('parsed', len(src), 'pairs; layers', LIDS, flush=True)

    rules = ('uniform', 'prop', 'inv', 'opt-desc', 'opt-asc')
    agg = {(r, S): [] for r in rules for S in BUDGETS}
    flips = {(r, S): [] for r in rules for S in BUDGETS}
    base_planes = []

    for lid in LIDS:
        for sid in sids:
            key = (sid, lid)
            if key not in src:
                continue
            packed, C, br = src[key]
            Yq, thr_g, A_q = trace_setup(packed, C, br, prms[lid])
            del src[key]
            p0, d0, _ = cert_planes(Yq, thr_g, A_q)
            base_planes.append(p0.mean())
            L1 = np.abs(A_q).sum(1).astype(np.float64)      # 逐判决行的 L1

            for S in BUDGETS:
                for r in rules:
                    if r == 'uniform':
                        frac = alloc_uniform(L1, S)
                    elif r == 'prop':
                        frac = alloc_prop(L1, S)
                    elif r == 'inv':
                        frac = alloc_inv(L1, S)
                    else:
                        keep = alloc_greedy(L1, S, ascending=(r == 'opt-asc'))
                    if r in ('uniform', 'prop', 'inv'):
                        keep = slots_to_keep(frac, S)
                    p_, d_, _ = cert_planes(Yq, thr_g, mask_per_row(A_q, keep))
                    agg[(r, S)].append(p_.mean())
                    flips[(r, S)].append(float((d_ != d0).mean()))
        print('  L%02d done' % lid, flush=True)

    base = float(np.mean(base_planes))
    print('\n== baseline planes/组 = %.4f (bits/组 %.1f) ==' % (base, 10 * base))
    print('\n%-8s %4s %10s %10s %11s %14s' %
          ('rule', 'S', 'planes/组', 'Δ供数', '判决翻转', '翻转/Δ供数'))
    rows = []
    for S in BUDGETS:
        for r in rules:
            m = float(np.mean(agg[(r, S)]))
            fl = float(np.mean(flips[(r, S)]))
            d = m / base - 1
            rows.append({'rule': r, 'slots': S, 'planes': m, 'd_supply': d, 'flip': fl})
            print('%-8s %4d %10.4f %9.2f%% %10.2f%% %14.3f'
                  % (r, S, m, 100 * d, 100 * fl, (fl / abs(d)) if d else -1))
        print()

    out = {'baseline_planes': base, 'G': G, 'n_sid': NSID, 'layers': LIDS, 'rows': rows}
    (ROOT / 'results' / 't33_perrow_k.json').write_text(json.dumps(out, indent=1) + '\n')
    print('wrote results/t33_perrow_k.json')


if __name__ == '__main__':
    main()
