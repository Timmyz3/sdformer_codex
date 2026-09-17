#!/usr/bin/env python3
"""T37：papers 库精读提出的三个"最便宜否证测试"（代理收敛的三条门）。

动机：T36 精读 21 篇文献后，多个代理各自独立建议**同三条廉价 numpy 门**，
用来否证三个看似可套的机制。本脚本一次跑完：

  门 1（Bishop/quantity 廉价上界）：**全零组占比**。
      quantity 维（token/块剪枝）能拿到的最便宜节省 = 整组（10 lane）全零的组。
      测得占比 f → 供数侧"免费剪枝"上界即 f（且受 T34 次可加性再打折）。

  门 2（MCBP BSTC / 零游程编码头寸）：**逐平面 lane 非零密度**。
      BSTC 压缩的是"高平面 ~65-80% 稀疏"的位片。我方 Y 逐平面若近满密度
      （mean ≈ 1.0），则切片级零/非零前缀编码**无头寸**（且前缀仍是同端口元数据税）。

  门 3（BitL 组内临界路径前提）：**逐组 max/mean 判决平面数**。
      BitL 的全部收益建立在"组内最慢/最密的那一个元素独吞周期"（其测得
      group 级稀疏只有理想 2× 的 0.1%）。我方组级证书本就取 jf.min（最慢判决
      = max 平面数）→ 若 max/mean ≈ 1，前提不成立；若 ≫2，则组内分区或有头寸。
      这是**粒度税（granularity tax）的第一次定量**。

  门 4（MCBP BSCR 复现）：A_q 按 m=4 行分块后的**重复非零列向量**计数。
      BSCR 的机制 = 合并重复列向量（= Prosperity 同轴的乘积复用）。我方已测
      逐 dot 非零复用 0.00%；此处换 A_q 常数侧口径再确认一次。

口径：bits/组 = 10 × planes/组（T29 主口径）。自有代码；capture 只读。
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

T = 10
G = 4000
LIDS = [8, 14, 20, 28]
NSID = 10


def main():
    z = np.load(t19.PARAMS)
    prms = {}
    for lid in LIDS:
        prms[lid] = {'lid': lid, 'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
                     'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
                     'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
                     'theta_src': z['L%d_theta_src' % lid]}
    src = t19.parse_sources(list(range(NSID)))

    gate1 = []                      # 全零组占比
    lane_den = []                   # 逐平面 lane 非零密度
    gate3 = []                      # 逐组 max/mean 判决平面数
    gate3_max, gate3_mean = [], []
    bscr = []                       # A_q m=4 分块重复非零列向量

    for lid in LIDS:
        A_q = np.rint(prms[lid]['A'] * 4096).astype(np.int64)
        A_q = np.where(A_q >= (1 << 15), A_q - (1 << 16), A_q)

        # 门 4：A_q 10×10，按 m=4 行分块，块内统计重复非零列向量
        m = 4
        n_blk, n_rep, n_col_nz = 0, 0, 0
        for lo in range(0, T, m):
            blk = A_q[lo:lo + m]                       # (<=4, 10)
            for k in range(blk.shape[1]):
                col = blk[:, k]
                nz = col[col != 0]
                n_col_nz += nz.size
                if nz.size >= 2:
                    n_rep += nz.size - np.unique(nz).size
            n_blk += 1
        bscr.append((n_col_nz, n_rep))

        for sid in range(NSID):
            key = (sid, lid)
            if key not in src:
                continue
            packed, C, br = src[key]
            Yq, thr_g, A_q_ = trace_setup(packed, C, br, prms[lid])
            del src[key]

            # 门 1：整组全零（10 lane 全 0）
            gate1.append(float((Yq == 0).all(1).mean()))

            # 门 2：逐平面 lane 非零密度（只统计 msb 以下的实质平面）
            msb = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(Yq.shape[0])])
            G_n = Yq.shape[0]
            for j in range(23, -1, -1):
                act = msb > j                            # 该平面在组内实质存在
                if not act.any():
                    continue
                bits = ((Yq[act] >> j) & 1) != 0         # (n_act, 10) 逐 lane
                lane_den.append(bits.mean())

            # 门 3：逐判决平面数 = msb − jf，取组内 max/mean
            planes, dec, jf = cert_planes(Yq, thr_g, A_q_)
            per_dec = msb[:, None] - jf                  # (G_n, 10) 逐判决平面数
            pmax = per_dec.max(1).astype(np.float64)
            pmean = per_dec.mean(1)
            ok = pmean > 0
            gate3.append(float(np.median(pmax[ok] / pmean[ok])))
            gate3_max.append(float(pmax[ok].mean()))
            gate3_mean.append(float(pmean[ok].mean()))
        print('  L%02d done' % lid, flush=True)

    f_zero = float(np.mean(gate1))
    dens = float(np.mean(lane_den))
    med_ratio = float(np.median(gate3))
    nz_b, rep_b = (sum(x[0] for x in bscr), sum(x[1] for x in bscr))

    print('\n== 门 1：全零组占比（quantity 免费上界）==')
    print('  %.2f%% 的组 10 lane 全零 → 供数侧免费剪枝上界（此后仍受 T34 次可加性打折）'
          % (100 * f_zero))
    print('\n== 门 2：逐平面 lane 非零密度（BSTC/零游程头寸）==')
    print('  均值 %.4f（1.0 = 每平面 10 lane 全非零，零编码无头寸）' % dens)
    print('\n== 门 3：逐组 max/mean 判决平面数（BitL 临界路径前提）==')
    print('  median(max/mean) = %.3f；max 均值 %.2f，mean 均值 %.2f'
          % (med_ratio, np.mean(gate3_max), np.mean(gate3_mean)))
    print('  （≈1 ⇒ 组内无"一个判决独吞周期"，BitL 前提不成立）')
    print('\n== 门 4：A_q m=4 分块内重复非零列向量（BSCR/乘积复用同轴）==')
    print('  非零块内元素 %d，重复 %d → 复用率 %.4f%%' % (nz_b, rep_b, 100 * rep_b / max(nz_b, 1)))

    out = {'all_zero_group_frac': f_zero,
           'plane_lane_density_mean': dens,
           'group_max_over_mean_planes_median': med_ratio,
           'group_max_planes_mean': float(np.mean(gate3_max)),
           'group_mean_planes_mean': float(np.mean(gate3_mean)),
           'bscr_block_nonzero': int(nz_b), 'bscr_block_repeats': int(rep_b),
           'bscr_reuse_rate': float(rep_b / max(nz_b, 1)),
           'layers': LIDS, 'n_sid': NSID, 'G': G}
    (ROOT / 'results' / 't37_paper_probes.json').write_text(json.dumps(out, indent=1) + '\n')
    print('\nwrote results/t37_paper_probes.json')


if __name__ == '__main__':
    main()
