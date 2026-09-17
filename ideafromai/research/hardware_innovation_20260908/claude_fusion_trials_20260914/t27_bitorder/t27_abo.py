#!/usr/bin/env python3
"""T27a：BitFair 式自适应位序（ABO）在 C1 证书门核上的照抄试验。

BitFair（JETCAS'26, arXiv:2607.05445）的核心之一是 **adaptive bit ordering**：
不按 MSB-first，而是用贪心搜索定一个**层级**位遍历顺序，声称比 MSB-first 快
最多 2.12×（他们的早停是"学习阈值预测 ReLU 零"，启发式）。

T9 曾结论"位平面序最优"，但它只比较了 **位平面序 vs 通道序**，没有比较
**位的遍历顺序本身**。本脚本把门核的区间证书推广到**任意位序**（不再是
`vtop' = 2·vtop + dot`，而是 `vtop += dot(A, plane_p)·2^p`，未读位仍是 [0,R_k]），
然后在真实 trace 上做贪心搜索，看 MSB-first 是否真的是最优。

推广式（对任意已读集合 S，未读集 U）：
    R_k = Σ_{p∈U} 2^p
    Ypart_k = Σ_c A[t][c]·( −sign_c·2^e + Σ_{p∈S} bit_p[c]·2^p )
    vmin = Ypart_k + N_t·R_k ,  vmax = Ypart_k + P_t·R_k     （N_t≤0≤P_t）
    lock  = (vmin ≥ thr) | (vmax < thr)
MSB-first 即 S = {e−1,…,e−k} ⇒ R_k = 2^{e−k}−1，与 T5/T25 同式。

用法：python t27_bitorder/t27_abo.py [stage0|stage3]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T = 10
MAXJ = 17                      # 位位置上界（实测 e_max=16 → 位 0..16）


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


def load_consts(d):
    A = np.array([int(x, 16) for x in (d / 'a.hex').read_text().split()], np.int64)
    A = np.where(A >= (1 << 15), A - (1 << 16), A).reshape(T, T)
    pn = np.array([int(x, 16) for x in (d / 'pn.hex').read_text().split()], np.int64)
    pn = np.where(pn >= (1 << 47), pn - (1 << 48), pn)
    return A, pn[0::2], pn[1::2]


def load_thr(d, H):
    thr = np.zeros((T, H), np.int64)
    for t in range(T):
        v = np.array([int(x, 16) for x in (d / f'tau_t{t}.hex').read_text().split()],
                     np.uint64)
        x = (v & np.uint64((1 << 48) - 1)).astype(np.int64)
        thr[t] = np.where(x >= (1 << 47), x - (1 << 48), x)
    return thr


def build(d, stim, H):
    """预计算每个 (组, 位位置 j<e) 的贡献，供任意位序重排。"""
    A, P_t, N_t = load_consts(d)
    groups = parse_stim(stim)
    thr = load_thr(d, H)
    G = len(groups)
    e = np.array([g['e'] for g in groups], np.int64)
    hs = np.array([g['h'] for g in groups], np.int64)
    sign = np.array([[-(g['sign'] >> s) & 1 for s in range(T)] for g in groups],
                    np.int64)
    # bitj[g, j, c] = 位 j 的取值（j 从 0=LSB 到 e-1=MSB）
    bitj = np.zeros((G, MAXJ + 1, T), np.int64)
    for g, grp in enumerate(groups):
        eg = grp['e']
        for k, pl in enumerate(grp['planes']):       # planes 顺序 e-1 … 0
            j = eg - 1 - k
            bitj[g, j] = [(pl >> s) & 1 for s in range(T)]
    # 贡献 C[g,j,t] = dot(A[t], bitplane_j)·2^j，仅在 j<e 时有效
    C = np.einsum('tc,gjc->gjt', A, bitj) * (1 << np.arange(MAXJ + 1))[None, :, None]
    valid = np.arange(MAXJ + 1)[None, :] < e[:, None]
    C *= valid[:, :, None]
    base = -(sign @ A.T) * (1 << e)[:, None]         # −dot(A,sign)·2^e
    Yint = -(sign * (1 << e)[:, None]) \
        + (bitj * (1 << np.arange(MAXJ + 1))[None, :, None]).sum(1)
    thr_g = thr[:, hs].T
    return dict(A=A, P_t=P_t, N_t=N_t, C=C, valid=valid, base=base,
                thr=thr_g, e=e, G=G, Yint=Yint)


def cycles_for_order(S, order):
    """给定层位序 order（位位置的列表，按处理先后），算每组的供给平面数。

    关键口径：**按组记数**（e 是逐组的，10 个 lane 共享）——只计该组自己有效的
    位位置（`sel`），无效位不占该组的拍。故 order=MSB-first 退化为逐组 MSB-first
    （= T5 口径）；这是"每组一支位序器、共享同一排列、各自跳过越界位"的实现。

    返回 (planes[G], locked_all[G])；planes 下限 1（与 T5/T19 口径一致）。"""
    G = S['G']
    C, valid, base, thr = S['C'], S['valid'], S['base'], S['thr']
    P_t, N_t = S['P_t'], S['N_t']
    part = base.copy()                               # (G,10)
    remaining = ((1 << (S['e'])) - 1).astype(np.int64)   # Σ_{j<e} 2^j
    planes = np.zeros(G, np.int64)
    cnt = np.zeros(G, np.int64)                      # 逐组已供平面数
    done = np.zeros(G, bool)
    for p in order:
        sel = valid[:, p]                            # 该组含位 p（p < e_g）
        if not sel.any():
            continue
        part = part + C[:, p, :] * sel[:, None]
        remaining = remaining - np.where(sel, 1 << p, 0)
        cnt = cnt + sel
        vmin = part + N_t[None, :] * remaining[:, None]
        vmax = part + P_t[None, :] * remaining[:, None]
        lock = (vmin >= thr) | (vmax < thr)
        newly = lock.all(1) & ~done
        planes[newly] = cnt[newly]
        done |= lock.all(1)
        if done.all():
            break
    planes[~done] = cnt[~done]
    planes = np.maximum(planes, 1)
    return planes, done


def greedy_order(S, verbose=True):
    """BitFair 式贪心（ETR 目标，AccLoss≡0 因证书零差）：每步在前缀末尾试插每个
    剩余位，**用"前缀 + MSB-first 补齐"的完整位序**评估最终平均供出平面数，
    取最小者。（用单比特前缀评估是退化的——未锁组统一记 1 拍。）"""
    order = []
    while True:
        best_p, best_mean = None, float('inf')
        for p in range(MAXJ, -1, -1):
            if p in order:
                continue
            rest = [q for q in range(MAXJ, -1, -1)
                    if q != p and q not in order]
            cand = order + [p] + rest
            c, _ = cycles_for_order(S, cand)
            m = float(c.mean())
            if m < best_mean - 1e-9:
                best_mean, best_p = m, p
        rest0 = [q for q in range(MAXJ, -1, -1) if q not in order]
        cur = float(cycles_for_order(S, order + rest0)[0].mean())
        if best_p is None or best_mean >= cur - 1e-9:
            break                        # 前缀已无改进空间
        order.append(best_p)
        if verbose:
            print(f'  step {len(order):2d}: pick bit {best_p:2d}  '
                  f'(full-order mean planes {best_mean:.4f})', flush=True)
        if len(order) >= S['e'].max():
            break
    order += [q for q in range(MAXJ, -1, -1) if q not in order]
    return order


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'stage0'
    d = ROOT / 't25_k4_gate' / 'k4'
    stim = ROOT / 'results' / 't5_rtl' / f's0_{stage}' / 'stim_bf.txt'
    H = 384 if stage == 'stage0' else 3072
    S = build(d, stim, H)
    print(f'{stage}: {S["G"]} groups, e range {S["e"].min()}–{S["e"].max()}',
          flush=True)

    msb_first = list(range(MAXJ, -1, -1))
    c_msb, done = cycles_for_order(S, msb_first)
    print(f'MSB-first: mean planes/group = {c_msb.mean():.4f}  '
          f'(all locked: {done.all()})')

    print('greedy search:')
    order = greedy_order(S)
    c_abo, done2 = cycles_for_order(S, order)
    print(f'ABO order (from MSB side): {order[:8]}...')
    print(f'ABO greedy: mean planes/group = {c_abo.mean():.4f}  '
          f'(all locked: {done2.all()})')

    # 反序（LSB-first）作下界参考
    c_lsb, _ = cycles_for_order(S, list(range(MAXJ + 1)))
    print(f'LSB-first: mean planes/group = {c_lsb.mean():.4f}')

    # 随机排列搜索：排除"贪心停在局部最优"的可能
    rng = np.random.default_rng(0)
    best_rand, best_perm = float('inf'), None
    NPERM = 400
    for _ in range(NPERM):
        perm = list(rng.permutation(MAXJ + 1))
        m = float(cycles_for_order(S, perm)[0].mean())
        if m < best_rand:
            best_rand, best_perm = m, perm
    print(f'random search ({NPERM} perms): best mean planes = {best_rand:.4f}  '
          f'(MSB-first {c_msb.mean():.4f})')

    # 转置局部搜索：MSB-first 的任意单次两位对换是否更优（比贪心更强的局部最优检验）
    base_order = list(range(MAXJ, -1, -1))
    n_explored, n_improved, best_tr = 0, 0, c_msb.mean()
    for i in range(len(base_order)):
        for j in range(i + 1, len(base_order)):
            cand = base_order.copy()
            cand[i], cand[j] = cand[j], cand[i]
            m = float(cycles_for_order(S, cand)[0].mean())
            n_explored += 1
            if m < best_tr - 1e-9:
                n_improved += 1
                best_tr = m
    print(f'transposition search: {n_explored} single swaps, '
          f'{n_improved} improved, best = {best_tr:.4f} '
          f'(MSB-first {c_msb.mean():.4f})')

    out = {'stage': stage, 'groups': int(S['G']),
           'e_min': int(S['e'].min()), 'e_max': int(S['e'].max()),
           'mean_planes_msb': float(c_msb.mean()),
           'mean_planes_abo': float(c_abo.mean()),
           'mean_planes_lsb': float(c_lsb.mean()),
           'mean_planes_rand': float(best_rand),
           'nperm_rand': NPERM,
           'transpos_explored': n_explored,
           'transpos_improved': n_improved,
           'mean_planes_transpos_best': float(best_tr),
           'abo_order': [int(p) for p in order]}
    (ROOT / 'results' / f't27_abo_{stage}.json').write_text(
        json.dumps(out, indent=1) + '\n')
    rel = (c_msb.mean() - c_abo.mean()) / c_msb.mean() * 100
    print(f'\nABO vs MSB-first: {rel:+.2f}% planes '
          f'({"ABO better" if rel > 0 else "MSB-first better"})')
    print(f'wrote results/t27_abo_{stage}.json')


if __name__ == '__main__':
    main()
