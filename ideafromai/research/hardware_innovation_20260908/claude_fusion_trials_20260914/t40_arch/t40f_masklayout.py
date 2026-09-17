#!/usr/bin/env python3
"""T40f：**掩码铺排**（保留哪一个通道）作为供数杠杆——判定为**负结果（退化解）**。

## 结论（2026-09-16 复核后，四种铺排同一批 traces）

纯 `Σ_j |need_j|` 目标**没有可用的铺排解**：它有无成本的退化解——**让门不放电**。
判决越少 ⇒ 早期锁定越多 ⇒ 要传的平面越少。实测（4 层 × 10 序列 × 1 万组，4 层均值）：

| 铺排 | bits/组 | 相对 mag | flip vs 稠密A | 判决数保留 |
|---|---|---|---|---|
| `mag`（稠密 A 的 top-4 幅度掩码，稠密值） | 13.307 | — | 2.05% | 59.0% |
| `t21f`（**T25/T39 设计点**，支撑集同 mag、值微调） | 13.496 | **+1.38%** | **1.98%** | 60.1% |
| `block`（2×5 块对角） | 7.497 | −43.7% | 3.27% | **8.8%** |
| `greedy`（坐标下降直接最小化 Σ\\|need_j\\|） | 1.337 | −89.8% | 3.60% | **2.6%** |

三条读数：
1. **学到的铺排买到的是精度，不是供数**：t21f 的支撑集与 mag **逐行完全相同**
   （`t21f_vs_mag_support_diff = 0`），微调只改了值——结果是供数**贵了 1.38%**、
   flip 从 2.05% 降到 1.98%、保留率 59.0%→60.1%。⇒ k=4 下"留哪几个"没有供数头寸。
2. **greedy/block 的"收益"是假的**：greedy 在 L14/L28 把判决数保留压到 **0.0%**
   （一个从不放电的门当然不用传任何比特），block 只剩 8.8%。
3. **【判退化要看量级】** 保留率是**稀疏正事件**占稠密 A 正判决的比例，
   **连合法的 t21f 也只有 ~60%** ——所以不能拿"60%"当精度达标线，
   要看 2.6% / 8.8% 这种量级。

⇒ **铺排轴关闭。** 方法论产出（比杠杆本身重要）：**供数轴上的任何搜索都必须带精度约束**
（AEE、或至少判决率/漏报率），否则优化器一定收敛到"不放电"。

## 保留下来的实现

⚠ **口径（2026-09-16 复核）**：本脚本的 A_ref（精度参考 + 值池）取自 `t19_gate_params.npz`
（`t19.PARAMS`，**稠密**微调 A，10 非零/行）；`t21f` 行另用 `results/t21b_k4_gate_params.npz`
（T25/T39 RTL 实际用的 k=4 微调 A）**自己的值**。判决保留率一律相对**稠密 A 的正判决数**。

四种铺排（同 k=4、同一批真实 traces），**同时报判决率与漏报/虚报**：
  A) `mag`   ：稠密 A 每行 |A_q| 最大的 4 个（朴素幅度，**用稠密 A 的值**）
  B) `t21f`  ：t21b k=4 微调 A 的**支撑集 + 自己的微调值**（= T25/T39 的实际设计点）
  C) `greedy`：逐行坐标下降，直接最小化实测 Σ_j |need_j|（在子样本上搜，全量上评）
  D) `block` ：块对角启发式（10 通道切 2 组各 5，每行只在本组内取 4）

⚠ 边界：**AEE 必须回 GPU 复测**；但本轮已能判定 C/D 不是候选（判决塌陷）。

用法：python t40_arch/t40f_masklayout.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 't32_lever'))
from t32_lever import cert_planes, trace_setup  # noqa: E402
import t19_all_layers as t19  # noqa: E402

T, LIDS, NSID, KEEP = 10, [8, 14, 20, 28], 10, 4
K4_PARAMS = ROOT / 'results' / 't21b_k4_gate_params.npz'   # T25/T39 的设计点（t21f k=4 微调）
BIT = (1 << np.arange(T)).astype(np.int64)
POP = np.array([bin(i).count('1') for i in range(1 << T)], np.int32)
SUB_GROUPS = 3000          # 搜索用子样本
SUB_TRACES = 2


def union_tab(masks):
    tab = np.zeros(1 << T, np.int32)
    for s in range(1, 1 << T):
        low = s & -s
        tab[s] = tab[s ^ low] | int(masks[low.bit_length() - 1])
    return tab


def bits_of(Yq, thr_g, A4, tab):
    """Σ_j |need_j|（bits/组）与该 trace 的逐判决深度和。"""
    n = Yq.shape[0]
    msb = np.array([int(np.abs(Yq[i]).max()).bit_length() for i in range(n)])
    _, _, jf = cert_planes(Yq, thr_g, A4)
    tot = 0
    for j in range(int(msb.max()) - 1, -1, -1):
        valid = msb > j
        if not valid.any():
            continue
        idx = ((jf <= j) * BIT[None, :]).sum(1)
        tot += int(POP[tab[idx]][valid].sum())
    return tot, int(np.clip(msb[:, None] - jf, 0, None).sum()), int(n)


def make_tab(mask, A):
    A4 = np.where(mask, A, 0)
    sup = (A4 != 0)
    masks = np.array([sum((1 << c) for c in range(T) if sup[t, c]) for t in range(T)],
                     np.int64)
    return A4, union_tab(masks)


def score(mask, A_val, trs):
    A4, tab = make_tab(mask, A_val)
    b = d = g = 0
    for Yq, thr_g in trs:
        bb, dd, gg = bits_of(Yq, thr_g, A4, tab)
        b += bb
        d += dd
        g += gg
    return b / g, d / g


def flips(mask, A_val, A_ref, trs):
    """判决失配率：用 `np.where(mask, A_val, 0)` 的判决 vs **稠密参考 A_ref** 的判决。"""
    A4 = np.where(mask, A_val, 0)
    f = tot = 0
    for Yq, thr_g in trs:
        _, dec_ref, _ = cert_planes(Yq, thr_g, A_ref)
        _, dec_m, _ = cert_planes(Yq, thr_g, A4)
        f += int((dec_ref != dec_m).sum())
        tot += dec_ref.size
    return f / tot


def diag(mask, A_val, A_ref, trs):
    """退化诊断：判决率 / 漏报 / 虚报。纯供数目标有退化解——**把判决压没**也能省比特。
    返回 (rate_pct, miss_pct, extra_pct, pos, tot)，pos/tot 供"相对稠密 A"的保留率使用。"""
    A4 = np.where(mask, A_val, 0)
    pos = tot = miss = extra = 0
    for Yq, thr_g in trs:
        _, dec_ref, _ = cert_planes(Yq, thr_g, A_ref)
        _, dec_m, _ = cert_planes(Yq, thr_g, A4)
        miss += int((dec_ref & ~dec_m).sum())
        extra += int((~dec_ref & dec_m).sum())
        pos += int(dec_m.sum())
        tot += dec_m.size
    return 100.0 * pos / tot, 100.0 * miss / tot, 100.0 * extra / tot, pos, tot


def topk_mask(A, k):
    mag = np.abs(A)
    m = np.zeros_like(A, bool)
    for t in range(T):
        m[t, np.argsort(-mag[t])[:k]] = True
    return m


def q16(Af):
    """fp → 16b 有符号量化（与 RTL 门核同一口径）。"""
    a = np.rint(np.asarray(Af) * 4096).astype(np.int64)
    return np.where(a >= (1 << 15), a - (1 << 16), a)


def t21f_mask(lid, A=None):
    """t21b k=4 微调 A 的**支撑集**（T25/T39 的实际铺排；与稠密 A 的 top-4 幅度掩码逐行一致，
    差别只在**值**被微调过）。值池由调用方决定。"""
    return q16(np.load(K4_PARAMS)['L%d_A' % lid]) != 0


def main():
    z = np.load(t19.PARAMS)
    prms = {lid: {'lid': lid, 'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
                  'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
                  'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
                  'theta_src': z['L%d_theta_src' % lid]} for lid in LIDS}
    src = t19.parse_sources(list(range(NSID)))
    cache = {}
    for lid in LIDS:
        for sid in range(NSID):
            key = (sid, lid)
            if key in src:
                pk, C, br = src[key]
                cache[key] = trace_setup(pk, C, br, prms[lid])[:2]

    per_layer = {}
    for lid in LIDS:
        A = q16(prms[lid]['A'])
        full = [cache[k] for k in sorted(cache) if k[1] == lid]
        sub = [(Y[:SUB_GROUPS], th[:SUB_GROUPS]) for Y, th in full[:SUB_TRACES]]

        m_mag = topk_mask(A, KEEP)
        m_gr = m_mag.copy()
        best = score(m_gr, A, sub)[0]
        # 逐行坐标下降：每行在**冻结的 base** 上枚举全部 (移出, 移入) 对，取该行最优后
        # 才落盘。不能像早期版本那样在枚举 out 的同时改写 m_gr —— 那时 `out` 已成快照，
        # 再把它置 False 是空操作而 `inv` 仍被置 True，popcount 会漂到 5（k 约束被破坏）。
        for _ in range(6):
            improved = False
            for t in range(T):
                base = m_gr.copy()
                best_row = (best, None)
                for out in np.where(base[t])[0]:
                    for inv in range(T):
                        if base[t, inv]:
                            continue
                        cand = base.copy()
                        cand[t, out] = False
                        cand[t, inv] = True
                        v = score(cand, A, sub)[0]
                        if v < best_row[0] - 1e-9:
                            best_row = (v, cand)
                if best_row[1] is not None:
                    best, m_gr, improved = best_row[0], best_row[1], True
            if not improved:
                break
        assert (m_gr.sum(1) == KEEP).all(), 'greedy 破坏了 k=%d 约束' % KEEP

        m_bl = np.zeros_like(A, bool)
        for t in range(T):
            grp = list(range(0, T // 2)) if t < T // 2 else list(range(T // 2, T))
            m_bl[t, sorted(grp, key=lambda c: -abs(A[t, c]))[:KEEP]] = True

        m_k4 = t21f_mask(lid)
        A_k4 = q16(np.load(K4_PARAMS)['L%d_A' % lid])   # t21f 自己的值（与稠密 A 同支撑、值已微调）
        row_groups = sum(Y.shape[0] for Y, _ in full)

        # 保留率分母 = **稠密 A 自身的判决数**（参考 = A_ref = 稠密 A）
        _, _, _, pos_dense, _ = diag(np.ones_like(A, bool), A, A, full)

        row = {}
        for tag, (m, av) in (('mag', (m_mag, A)), ('t21f', (m_k4, A_k4)),
                             ('greedy', (m_gr, A)), ('block', (m_bl, A))):
            b, d = score(m, av, full)
            row['%s_bits_per_group' % tag] = b
            row['%s_perdec' % tag] = d
            row['%s_flip_pct' % tag] = 100.0 * flips(m, av, A, full)
            rate, miss, extra, pos, _ = diag(m, av, A, full)
            row['%s_dec_rate_pct' % tag] = rate
            row['%s_miss_pct' % tag] = miss
            row['%s_extra_pct' % tag] = extra
            row['%s_dec_retained_pct' % tag] = 100.0 * pos / max(pos_dense, 1)
        row['dense_dec_rate_pct'] = 100.0 * pos_dense / (row_groups * T)
        row['greedy_gain_vs_mag_pct'] = 100.0 * (row['mag_bits_per_group']
                                                 - row['greedy_bits_per_group']) \
            / row['mag_bits_per_group']
        row['block_gain_vs_mag_pct'] = 100.0 * (row['mag_bits_per_group']
                                                - row['block_bits_per_group']) \
            / row['mag_bits_per_group']
        row['t21f_gain_vs_mag_pct'] = 100.0 * (row['mag_bits_per_group']
                                               - row['t21f_bits_per_group']) \
            / row['mag_bits_per_group']
        row['mask_changed_cells'] = int((m_gr != m_mag).sum())
        row['t21f_vs_mag_support_diff'] = int((m_k4 != m_mag).sum())
        row['k_ok'] = bool((m_mag.sum(1) == KEEP).all() and (m_gr.sum(1) == KEEP).all()
                           and (m_bl.sum(1) == KEEP).all() and (m_k4.sum(1) == KEEP).all())
        row['mag_masks'] = [int(sum((1 << c) for c in range(T) if m_mag[t, c]))
                            for t in range(T)]
        row['greedy_masks'] = [int(sum((1 << c) for c in range(T) if m_gr[t, c]))
                               for t in range(T)]
        row['t21f_masks'] = [int(sum((1 << c) for c in range(T) if m_k4[t, c]))
                             for t in range(T)]
        row['groups'] = int(sum(Y.shape[0] for Y, _ in full))
        per_layer['L%d' % lid] = row
        for tag in ('t21f', 'greedy', 'block'):
            print('L%-3d mag %.4f | %s %.4f (%+.2f%%) 判决保留 %.1f%% 漏报 %.2f%% '
                  '| flip %.2f%% | 保留率[dense 分母] mag/t21f/gr/bl %.0f/%.0f/%.0f/%.0f%%'
                  % (lid, row['mag_bits_per_group'], tag, row['%s_bits_per_group' % tag],
                     row['%s_gain_vs_mag_pct' % tag],
                     row['%s_dec_retained_pct' % tag], row['%s_miss_pct' % tag],
                     row['%s_flip_pct' % tag], row['mag_dec_retained_pct'],
                     row['t21f_dec_retained_pct'], row['greedy_dec_retained_pct'],
                     row['block_dec_retained_pct']))

    keys = [k for k in per_layer['L8']
            if k not in ('mag_masks', 'greedy_masks', 't21f_masks')]
    agg = {k: float(np.mean([per_layer[r][k] for r in per_layer])) for k in keys}
    agg['per_layer'] = per_layer
    agg['verdict'] = ('NEGATIVE（退化解）：纯 Σ_j|need_j| 目标**没有**可用的铺排解——'
                      'greedy 把供数压下 89.8%% 的代价是**判决数只剩 2.6%%**（相对稠密 A），'
                      'block −43.7%% 的代价是只剩 8.8%%。两者都**不可能 AEE 兼容**。'
                      '同 k=4 下，学到的铺排（t21f，= T25/T39 设计点）与朴素幅度掩码（mag）'
                      '**支撑集逐行完全相同**，其微调买到的是**精度**（flip 2.05%%→1.98%%）'
                      '而**不是**供数（反而 +1.38%%）。'
                      '⇒ 铺排轴**关闭**，根因：供数目标必须带精度约束（AEE/判决率），'
                      '否则最小化 Σ|need_j| 的退化解是"让门不放电"。')
    agg['note'] = ('⚠ 口径：A_ref（精度参考 + 值池）= t19_gate_params.npz 的**稠密**微调 A（10 非零/行）。'
                   'mag = 其每行 |A_q| top-4 掩码（朴素幅度，稠密值）；'
                   't21f = results/t21b_k4_gate_params.npz（**支撑集与 mag 相同**、值经微调，'
                   '即 T25/T39 的实际设计点）；greedy/block 见 docstring。'
                   '判决保留率分母 = 稠密 A 自身的正判决数（注意：稀疏正事件，'
                   '**连合法的 t21f 也只有 ~60%%**，故判退化要看 2.6%%/8.8%% 这种量级，'
                   '不要拿 60%% 当"精度达标"）。'
                   'AEE 仍须回 GPU 复测，但本轮已证明 greedy/block **不可能是 AEE 兼容的**。')
    (ROOT / 'results' / 't40f_masklayout.json').write_text(json.dumps(agg, indent=1) + '\n')
    print('\n=== 4 层平均 ===')
    for k, v in agg.items():
        if k not in ('per_layer', 'note'):
            print('%-30s %s' % (k, v))
    print('wrote results/t40f_masklayout.json')


if __name__ == '__main__':
    main()
