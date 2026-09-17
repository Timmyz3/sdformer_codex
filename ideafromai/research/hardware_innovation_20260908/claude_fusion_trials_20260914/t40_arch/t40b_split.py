#!/usr/bin/env python3
"""T40b：把 z 空间对 lane 基的 3.89× 拆成**两个独立可归因的部分**。

t40_zdiag 给出的 y_per_decision = -6.84 bits/组 是错的：`dy = msb − jf` 在
"判决于 sop 粗检查（jf > msb）就锁定"时为负，被直接求和。本脚本修正为
`depth_Y = max(msb − jf, 0)`（jf > msb ⇔ 0 个平面，与 T39 的 k=0 档一致），
并做**配对**比较（同一 (组, 判决) 同时算 depth_Y 与 depth_Z），从而分离：

  (1) **共享税**  y_sglr / Σ_t depth_Y：Y 的一个平面比特同时服务 |sup| 个判决，
      已锁定的判决仍被迫跟着收比特。这是"lane 比判决粗"的退役单位税。
  (2) **基底增益** Σ_t depth_Y / Σ_t depth_Z：两者都是**逐判决独立退役**、
      各自可自由选深度，唯一的差别是证书构造（操作数箱式包络 vs 线性像）。
      解析上二者应相等（e_z − e ≈ log2 L1），故这一项必须实测确认。

另给出配对差 (depth_Y − depth_Z) 的分布：若 ≈ 0 则解析成立、直方图差异来自
端点/组级尺度；若 ≈ 11.4（log2 L1）则两侧收敛**速率**不同，需要改写解析模型。

用法：python t40_arch/t40b_split.py
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

T, LIDS, NSID, KEEP = 10, [8, 14, 20, 28], 10, 4
POP = np.array([bin(i).count('1') for i in range(1 << T)], np.int32)
BIT = (1 << np.arange(T)).astype(np.int64)


def chan_union_table(masks):
    """TAB[10bit 未决判决集合] -> 通道并集掩码。"""
    tab = np.zeros(1 << T, np.int32)
    for s in range(1, 1 << T):
        low = s & -s
        t = low.bit_length() - 1
        tab[s] = tab[s ^ low] | int(masks[t])
    return tab


def z_depths(Z, thr, ez, maxd=40):
    """逐 (组, 判决) 的 z 空间深度：最大的 j ≤ ez 使区间不跨 thr。"""
    n, T_ = Z.shape
    done = np.zeros((n, T_), bool)
    depth = np.zeros((n, T_), np.int64)
    for d in range(0, maxd):
        if done.all():
            break
        j = np.maximum(ez[:, None] - d, 0)
        active = ~done
        lo = (Z >> j) << j
        w = (np.int64(1) << j) - 1
        lock = (lo >= thr) | (lo + w < thr)
        new = active & lock
        depth[new] = d
        done |= new
    depth[~done] = maxd
    return depth


def main():
    z = np.load(t19.PARAMS)
    prms = {lid: {'lid': lid, 'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
                  'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
                  'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
                  'theta_src': z['L%d_theta_src' % lid]} for lid in LIDS}
    src = t19.parse_sources(list(range(NSID)))

    acc = dict(grp=0, y_sglr=0, y_perdec=0, z_perdec=0, n_dec=0,
               z_gt_y=0, z_lt_y=0, z_eq_y=0, dy_pos=0, dz_pos=0,
               both_pos=0, sum_dy_both=0, sum_dz_both=0)
    diffs, dy_hist, dz_hist, l1s, ezme = [], [], [], [], []
    for lid in LIDS:
        A_q = np.rint(prms[lid]['A'] * 4096).astype(np.int64)
        A_q = np.where(A_q >= (1 << 15), A_q - (1 << 16), A_q)
        A4 = mask_keep(A_q, KEEP)
        sup = (A4 != 0)
        masks = np.array([sum((1 << k) for k in range(T) if sup[t, k])
                          for t in range(T)], np.int64)
        tab = chan_union_table(masks)
        L1 = np.abs(A4).sum(1)
        l1s.append(np.log2(np.maximum(L1, 1)))
        for sid in range(NSID):
            key = (sid, lid)
            if key not in src:
                continue
            packed, C, br = src[key]
            Yq, thr_g, _ = trace_setup(packed, C, br, prms[lid])
            del src[key]
            n = Yq.shape[0]
            msb = np.array([int(np.abs(Yq[i]).max()).bit_length() for i in range(n)])
            _, _, jf = cert_planes(Yq, thr_g, A4)
            depth_Y = np.clip(msb[:, None] - jf, 0, None).astype(np.int64)
            Z = Yq @ A4.T
            ez = np.array([int(np.abs(Z[i]).max()).bit_length() for i in range(n)])
            ezme.extend((ez - msb).tolist())
            depth_Z = z_depths(Z, thr_g, ez)
            # (2) z 侧：SGLR 共享掩码（逐平面 |need_j|）——仅计 msb > j 的组
            top = int(msb.max())
            for j in range(top - 1, -1, -1):
                valid = msb > j
                if not valid.any():
                    continue
                unres = (jf <= j) * BIT[None, :]
                idx = unres.sum(1)
                acc['y_sglr'] += int(POP[tab[idx]][valid].sum())
            acc['grp'] += int(n)
            acc['y_perdec'] += int(depth_Y.sum())
            acc['z_perdec'] += int(depth_Z.sum())
            acc['n_dec'] += int(n * T)
            d = depth_Y - depth_Z
            diffs.append(d.ravel())
            dy_hist.append(depth_Y.ravel())
            dz_hist.append(depth_Z.ravel())
            acc['z_gt_y'] += int((depth_Z > depth_Y).sum())
            acc['z_lt_y'] += int((depth_Z < depth_Y).sum())
            acc['z_eq_y'] += int((depth_Z == depth_Y).sum())
            acc['dy_pos'] += int((depth_Y > 0).sum())
            acc['dz_pos'] += int((depth_Z > 0).sum())
            b = (depth_Y > 0) & (depth_Z > 0)
            acc['both_pos'] += int(b.sum())
            acc['sum_dy_both'] += int(depth_Y[b].sum())
            acc['sum_dz_both'] += int(depth_Z[b].sum())

    g = acc['grp']
    diffs = np.concatenate(diffs)
    dyh = np.concatenate(dy_hist)
    dzh = np.concatenate(dz_hist)
    both = (dyh > 0) & (dzh > 0)
    out = {
        'groups': g,
        'decisions': int(acc['n_dec']),
        # 三项成本（bits/组）
        'y_sglr_bits_per_group': acc['y_sglr'] / g,
        'y_perdec_bits_per_group': acc['y_perdec'] / g,
        'z_perdec_bits_per_group': acc['z_perdec'] / g,
        # 两项归因
        'sharing_tax_ratio': acc['y_sglr'] / acc['y_perdec'],
        'basis_gain_ratio': acc['y_perdec'] / acc['z_perdec'],
        'total_ratio': acc['y_sglr'] / acc['z_perdec'],
        'z_depth0_fraction': 1.0 - acc['dz_pos'] / acc['n_dec'],
        'y_depth0_fraction': 1.0 - acc['dy_pos'] / acc['n_dec'],
        'z_worse_than_y_fraction': acc['z_gt_y'] / acc['n_dec'],
        'z_better_than_y_fraction': acc['z_lt_y'] / acc['n_dec'],
        'z_equal_y_fraction': acc['z_eq_y'] / acc['n_dec'],
        'paired_diff_mean': float(diffs.mean()),
        'paired_diff_pct': [float(np.percentile(diffs, p)) for p in (1, 25, 50, 75, 99)],
        'both_positive_count': int(acc['both_pos']),
        'both_positive_mean_dy': acc['sum_dy_both'] / max(acc['both_pos'], 1),
        'both_positive_mean_dz': acc['sum_dz_both'] / max(acc['both_pos'], 1),
        'log2_L1_mean': float(np.mean(np.concatenate(l1s))),
        'ez_minus_e_mean': float(np.mean(ezme)),
        'y_depth_hist': np.bincount(np.clip(dyh, 0, 15), minlength=16).tolist(),
        'z_depth_hist': np.bincount(np.clip(dzh, 0, 15), minlength=16).tolist(),
        'note': 'sharing_tax = y_sglr/y_perdec（共享平面比特的税）；'
                'basis_gain = y_perdec/z_perdec（同为逐判决退役，仅证书构造不同）。'
                'paired_diff = depth_Y − depth_Z，若 ≈0 则两侧收敛速率相同。',
    }
    (ROOT / 'results' / 't40b_split.json').write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out.items():
        print('%-30s %s' % (k, v))
    print('wrote results/t40b_split.json')


if __name__ == '__main__':
    main()
