#!/usr/bin/env python3
"""T35：乘积复用潜力（Prosperity HPCA'25 式 product sparsity）——C1 门核的互补机制门。

动机：互补机制挖掘代理把 **Prosperity**（HPCA'25, arXiv:2503.03379,
github.com/dubcyfor3/Prosperity）列为首选：spiking matmul 里存在大量**相同**的部分积，
运行时检测重复并复用一份结果（声称 result-preserving、非近似）。它省的是
**算子数/周期/能量**，与 C1 省的"每组多少位"**正交** → 是真正的互补轴候选。

我方门核的算子结构：每 (组 g, 判决 t') 是长度 10 的 dot：`V[t']=Σ_k Y[g,k]·A_q[t',k]`
（A_q 为**固定常数** 10×10、量化到 4096 标度；Y 逐组变化）。
⇒ 复用有两条来源：
  (a) **A_q 内重复系数**：同列 k 上 `A_q[t',k]==A_q[t'',k]` ⇒ 乘积 `Y·A_q[t',k]` 可跨判决复用。
      这是**离线可预分析**的（A 是部署常量）→ 硬件只是路由，零运行时开销。
  (b) **组内重复乘积**：同一 dot 的 10 个乘积里出现相同值。

本脚本用真实 traces 量化 (a)(b) 的复用率。若复用率低，则 Prosperity 式机制**无门**。

口径：自有代码；capture 只读。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 't32_lever'))
from t32_lever import trace_setup  # noqa: E402
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

    aq_stats, prod_stats, col_dup = [], [], []
    for lid in LIDS:
        A_q = np.rint(prms[lid]['A'] * 4096).astype(np.int64)
        A_q = np.where(A_q >= (1 << 15), A_q - (1 << 16), A_q)   # T32/t19 口径
        # (a) A_q 内重复系数（离线可预分析）
        nz = A_q[A_q != 0]
        aq_stats.append((nz.size, np.unique(nz).size))
        # 同列重复：col k 上 A_q[:,k] 内重复的对数占比
        dcol = []
        for k in range(A_q.shape[1]):
            c = A_q[:, k]
            dcol.append(1.0 - np.unique(c).size / c.size)
        col_dup.append(float(np.mean(dcol)))

        for sid in range(NSID):
            key = (sid, lid)
            if key not in src:
                continue
            packed, C, br = src[key]
            Yq, thr_g, A_q_ = trace_setup(packed, C, br, prms[lid])
            del src[key]
            # (b) 逐 (组,判决) 的 10 个乘积
            P = np.einsum('tk,gk->gtk', A_q_, Yq)      # (G, T, 10)
            flat = P.reshape(-1, 10)
            uniq_per_dot = np.array([np.unique(r).size for r in flat])
            prod_stats.append((flat.size, uniq_per_dot.sum()))
        print('  L%02d done' % lid, flush=True)

    tot_p, tot_u = (sum(x[0] for x in prod_stats), sum(x[1] for x in prod_stats))
    aq_p, aq_u = (sum(x[0] for x in aq_stats), sum(x[1] for x in aq_stats))
    reuse_prod = 1.0 - tot_u / tot_p
    print('\n== (a) A_q 常数内重复（离线可预分析）==')
    print('  非零系数 %d 个，distinct %d → 重复率 %.1f%%' % (aq_p, aq_u, 100 * (1 - aq_u / aq_p)))
    print('  同列内重复（跨判决可复用）均值 %.1f%%' % (100 * np.mean(col_dup)))
    print('\n== (b) 逐 dot 的 10 个乘积内重复 ==')
    print('  乘积总数 %d，去重后 %d → 可复用率 %.2f%%' % (tot_p, tot_u, 100 * reuse_prod))
    print('  （即 Prosperity 式运行时复用最多省 %.2f%% 的乘积）' % (100 * reuse_prod))

    out = {'A_q_nonzero': int(aq_p), 'A_q_distinct': int(aq_u),
           'A_q_dup_rate': float(1 - aq_u / aq_p),
           'same_column_dup_mean': float(np.mean(col_dup)),
           'products_total': int(tot_p), 'products_distinct': int(tot_u),
           'product_reuse_rate': float(reuse_prod),
           'layers': LIDS, 'n_sid': NSID, 'G': G}
    (ROOT / 'results' / 't35_product_reuse.json').write_text(json.dumps(out, indent=1) + '\n')
    print('\nwrote results/t35_product_reuse.json')


if __name__ == '__main__':
    main()
