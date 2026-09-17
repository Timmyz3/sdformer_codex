#!/usr/bin/env python3
"""T38：文献机制**逐篇照抄 → 找根因 → 改进版 → 改进版实测**的迁移电池。

动机（用户反馈）：T36 精读做成了"分析判死"——每篇一句"落在已关闭轴"就丢掉，
21 篇只产出几条结论。用户要求走**完整迁移路径**：
  ① 照抄（真实试跑）→ ② 效果不好**找根因** → ③ 针对根因**改进出新方法** → ④ 改进版再实测。
"性能不好"不是终点，**根因往往是新机制的入口**。

本脚本对四个可量化的失败机制做这件事：

  [A] BitL（MICRO'25）——"组内临界路径"照抄失败（T37 门3：我方无权重位稀疏可 pivot）。
      根因：BitL 靠**权重稀疏**换遍历序；我方 A_q 稠密 → 无 pivot 对象。
      ★ 改进版（新机制候选）：**support-aware lane gating**——把"临界路径"从
        权重位稀疏换到**未锁判决的支撑并集**上。k=4 掩码后每个判决只用 4 个 lane，
        判决逐个锁定 → 每平面只需发"未锁判决还在用的 lane"。
        这是**精确**的（不发没人用的比特），且与 k 掩码**互补**。
        本脚本测其 oracle 上界（假设掩码可零成本推导）。

  [B] MCBP（MICRO'25）BSTC——"高平面 65–80% 稀疏"照抄失败（T37 门2：全平面密度 0.4595）。
      根因假设：MSB-first **已把高平面全零部分吃掉**，剩下的是**中间最稠密**的平面。
      → 测**逐平面密度剖面**，看稀疏到底在哪几层、还够不够 BSTC 用。

  [C] MCBP BSCR——重复列向量复用 0.2506%（T35/T37 已判）。
      根因：精确值碰撞不存在（连续权重）。但 BSCR 的**结构**是无损的。
      ★ 改进版：把"值重复"换成"**支撑重复**"（k=4 下哪些判决行共用同一组 lane）
        → 若重复率高，可共享 lane 门控 / 共享部分和。测支撑碰撞率。

  [D] 数量维（ToMe/Scrooge/SATA 等）——T34 测得**次可加**（剪 26.6% 组只省 6.10%）。
      根因假设：T34 剪的是**最便宜**的组（plane≤k），便宜组本就占供数少。
      → 测**剪最贵组**的 oracle 上界：若也次可加，则次可加性是"组难度的分布"决定，
        与剪谁无关（轴死透）；若大幅超线性，则"剪难 token"才是真杠杆（但与精度冲突）。

口径：bits/组（= Σ lane 数），分母用 planes×10 便于与 T5/T19 对齐；T29 主口径。自有代码；capture 只读。
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
KEEP = 4


POPTBL = np.array([bin(i).count('1') for i in range(1 << T)], np.int8)


def main():
    z = np.load(t19.PARAMS)
    prms = {}
    for lid in LIDS:
        prms[lid] = {'lid': lid, 'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
                     'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
                     'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
                     'theta_src': z['L%d_theta_src' % lid]}
    src = t19.parse_sources(list(range(NSID)))

    dens_prof = np.zeros(24)
    dens_n = np.zeros(24)
    agg = {'dense': [0, 0], 'k4': [0, 0]}          # [bits_actual, bits_gated]
    supp_coll = {'pairs': 0, 'same': 0}
    all_planes = []                                 # 逐组 plane 数（用于 D）

    for lid in LIDS:
        A_q = np.rint(prms[lid]['A'] * 4096).astype(np.int64)
        A_q = np.where(A_q >= (1 << 15), A_q - (1 << 16), A_q)
        A4 = mask_keep(A_q, KEEP)

        # [C] 支撑碰撞（k=4）——按层统计判决行两两同一支撑
        sup = (A4 != 0)
        for i in range(T):
            for j in range(i + 1, T):
                supp_coll['pairs'] += 1
                supp_coll['same'] += int(np.array_equal(sup[i], sup[j]))

        for sid in range(NSID):
            key = (sid, lid)
            if key not in src:
                continue
            packed, C, br = src[key]
            Yq, thr_g, A_q_ = trace_setup(packed, C, br, prms[lid])
            del src[key]
            Gn = Yq.shape[0]
            msb = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(Gn)])

            # [B] 逐平面非零密度剖面（在 msb 以下、即真实会发的平面里）
            for j in range(24):
                act = msb > j
                if act.any():
                    dens_prof[j] += (((Yq[act] >> j) & 1) != 0).sum()
                    dens_n[j] += act.sum() * T

            planes_full, dec, jf = cert_planes(Yq, thr_g, A_q_)
            all_planes.append(planes_full)

            # [A] 用 k=4 掩码的证书（供数与决策都会变；仅作 gating 口径）
            planes4, _, jf4 = cert_planes(Yq, thr_g, A4)

            for tag, Aw, jfw, pl in (('dense', A_q_, jf, planes_full),
                                     ('k4', A4, jf4, planes4)):
                sup = (Aw != 0)
                masks = np.array([int(''.join('1' if sup[t, k] else '0' for k in range(T)[::-1]), 2)
                                  for t in range(T)], np.int64)   # bit k ↔ lane k
                jmin = jfw.min(1)
                for g in range(Gn):
                    top = msb[g]
                    bot = jmin[g]
                    if top - bot <= 0:
                        continue
                    n_pl = top - bot
                    agg[tag][0] += n_pl * T
                    # 逐平面：**未锁**判决（发第 j 平面时尚未解析：jf <= j）的支撑并集
                    bits = 0
                    for j in range(bot, top):
                        uu = 0
                        for t_ in range(T):
                            if jfw[g, t_] <= j:
                                uu |= int(masks[t_])
                        bits += int(POPTBL[uu])
                    agg[tag][1] += bits

        print('  L%02d done' % lid, flush=True)

    # [D] 剪最贵组 vs 剪最便宜组
    P = np.concatenate(all_planes, 0).astype(np.float64)
    P = np.clip(P, 0, None)
    tot = P.sum()
    order_asc = np.argsort(P)                 # 便宜在前
    order_desc = np.argsort(-P)               # 贵在前
    drop_curve = {}
    for q in (0.10, 0.25, 0.50):
        n = int(q * P.size)
        cheap = P[order_asc[:n]].sum() / tot
        exp_ = P[order_desc[:n]].sum() / tot
        drop_curve['q=%.2f' % q] = {'drop_cheapest_frac': float(cheap),
                                    'drop_most_expensive_frac': float(exp_)}

    d_act, d_gate = agg['dense']
    k_act, k_gate = agg['k4']
    print('\n== [A] BitL 迁移：support-aware lane gating（oracle，零反馈成本）==')
    print('  dense A_q：实际 %d bits，gated %d bits → 省 %.2f%%（掩码全 1，无头寸 = 照抄失败的根因）'
          % (d_act, d_gate, 100 * (1 - d_gate / d_act)))
    print('  k=4 A_q  ：实际 %d bits，gated %d bits → **省 %.2f%%**'
          % (k_act, k_gate, 100 * (1 - k_gate / k_act)))
    print('\n== [B] 逐平面非零密度剖面（BSTC 根因）==')
    prof = dens_prof / np.maximum(dens_n, 1)
    print('  j=0..11:', np.round(prof[:12], 3).tolist())
    print('  j=12..23:', np.round(prof[12:], 3).tolist())
    print('  （低 j = 低位平面；MSB-first 先发高位 → 看哪段最密）')
    print('\n== [C] k=4 支撑碰撞（BSCR 的"结构版"改进）==')
    print('  判决行两两对 %d，支撑完全相同 %d → **%.2f%%**'
          % (supp_coll['pairs'], supp_coll['same'],
             100 * supp_coll['same'] / max(supp_coll['pairs'], 1)))
    print('\n== [D] 数量维：剪最贵组 vs 剪最便宜组（供数份额）==')
    for k_, v in drop_curve.items():
        print('  剪掉 %s 的组：最便宜组省 %.2f%%，最贵组省 %.2f%%'
              % (k_, 100 * v['drop_cheapest_frac'], 100 * v['drop_most_expensive_frac']))

    out = {'lane_gating': {'dense_actual': int(d_act), 'dense_gated': int(d_gate),
                           'k4_actual': int(k_act), 'k4_gated': int(k_gate),
                           'dense_saving': float(1 - d_gate / max(d_act, 1)),
                           'k4_saving': float(1 - k_gate / max(k_act, 1))},
           'plane_density_profile': [float(x) for x in prof],
           'support_collision_k4': float(supp_coll['same'] / max(supp_coll['pairs'], 1)),
           'quantity_drop_curve': drop_curve,
           'layers': LIDS, 'n_sid': NSID, 'G': G, 'keep': KEEP}
    (ROOT / 'results' / 't38_migration.json').write_text(json.dumps(out, indent=1) + '\n')
    print('\nwrote results/t38_migration.json')


if __name__ == '__main__':
    main()
