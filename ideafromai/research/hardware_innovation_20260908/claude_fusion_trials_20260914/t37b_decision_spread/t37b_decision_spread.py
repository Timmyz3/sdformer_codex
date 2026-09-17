#!/usr/bin/env python3
"""T37b：组内判决平面数散度的**归因**（T37 门 3 的后续）。

T37 门 3 实测：逐组 max/mean(逐判决平面数) 的 median = 2.857（max 3.46 / mean 1.19）。
这有**两种互斥解释**，决定它是否是可训练头寸：

  H_struct：组内存在**系统性难判决**（某个 t' 恒难）→ 训练侧"均衡 margin"可压
            （= 新旋钮，潜在收益 ~ (1−1.19/3.46) = −66%！须先证实）。
  H_noise ：per-decision 平面数近似 iid → max 只是"10 个抽样的最大值"这个
            **序统计量**效应，与结构无关 → 不可训练、不可利用（= 只是粒度税的度量）。

判据（方差分解，对标 T31 的做法）：
  把 per_dec[g, t']（G×10）分解为  组效应 a_g + 判决效应 b_t' + 残差。
  • 判决效应 b_t' 的方差占比 = **可训练头寸**（哪个输出时间步恒难）。
  • 组效应 a_g 的方差占比 = 全局难度（改不了，T30/T31 已证跨层无头寸）。
  • 残差占比 = 序统计量噪声。

同时报两个**oracle 上界**（若 H_struct 成立则可达）：
  supply_actual  = mean_g(max_t' per_dec) / 24
  supply_homog   = mean_g(mean_t' per_dec) / 24   ← 组内完全齐平的理想
  supply_iidnull = E[max of 10 iid draws matching the marginal] / 24  ← 纯序统计量基线

若 supply_homog 相对 supply_actual 有大空间 AND b_t' 方差占比显著，则新轴成立；
若实际 max ≈ iid 零假设的 max，则纯噪声 → 关闭。

口径：bits/组 = 10 × planes/组（分母 24 词宽，与 T5/T19 口径一致）。自有代码；capture 只读。
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
D = 24
RNG = np.random.default_rng(20260916)


def main():
    z = np.load(t19.PARAMS)
    prms = {}
    for lid in LIDS:
        prms[lid] = {'lid': lid, 'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
                     'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
                     'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
                     'theta_src': z['L%d_theta_src' % lid]}
    src = t19.parse_sources(list(range(NSID)))

    PD, PMAX = [], []                  # 逐判决平面数（拼接）、逐组 max
    per_t_mean = np.zeros(T)           # 判决效应 b_t'
    n_t = 0
    for lid in LIDS:
        for sid in range(NSID):
            key = (sid, lid)
            if key not in src:
                continue
            packed, C, br = src[key]
            Yq, thr_g, A_q_ = trace_setup(packed, C, br, prms[lid])
            del src[key]
            msb = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(Yq.shape[0])])
            planes, dec, jf = cert_planes(Yq, thr_g, A_q_)
            per_dec = (msb[:, None] - jf).astype(np.float64)      # (G,10)
            PD.append(per_dec)
            PMAX.append(per_dec.max(1))
            per_t_mean += per_dec.mean(0)
            n_t += 1
        print('  L%02d done' % lid, flush=True)

    PD = np.concatenate(PD, 0)                 # (N, 10)
    PMAX = np.concatenate(PMAX, 0)             # (N,)
    N = PD.shape[0]
    per_t_mean /= n_t

    # ---- 方差分解 ----
    a_g = PD.mean(1)                            # 组效应
    b_t = PD.mean(0)                            # 判决效应
    grand = PD.mean()
    resid = PD - a_g[:, None] - b_t[None, :] + grand
    v_tot = PD.var()
    v_grp = (a_g[:, None] - grand).var()
    v_t = (b_t[None, :] - grand).var()
    v_res = resid.var()

    # ---- 三个供数口径 ----
    supply_actual = PMAX.mean() / D
    supply_homog = a_g.mean() / D
    # iid 零假设：按边际分布重抽 10 个求 max（保留组数 N）
    marg = PD.ravel()
    draws = RNG.choice(marg, size=(N, T))
    supply_iidnull = draws.max(1).mean() / D

    print('\n== 方差分解（per_dec 的方差来源）==')
    print('  组效应 a_g     占比 %.4f' % (v_grp / v_tot))
    print('  判决效应 b_t\'  占比 %.4f' % (v_t / v_tot))
    print('  残差（序统计量噪声）占比 %.4f' % (v_res / v_tot))
    print('  判决效应 b_t\' 逐 t 值:', np.round(b_t, 3).tolist(), '（极差 %.3f）'
          % (b_t.max() - b_t.min()))
    print('\n== 三个供数口径（planes/组 ÷ 24）==')
    print('  实测 actual      %.4f（mean max = %.3f）' % (supply_actual, PMAX.mean()))
    print('  组内齐平 homog   %.4f（= 组效应均值 %.3f）' % (supply_homog, a_g.mean()))
    print('  iid 零假设 null  %.4f（= 10 抽样 max 期望 %.3f）'
          % (supply_iidnull, draws.max(1).mean()))
    print('\n判读：若 actual ≈ null 且 b_t\' 方差占比小 → 2.86× 是**序统计量**，非可训练头寸。')

    out = {'supply_actual': float(supply_actual), 'supply_homog': float(supply_homog),
           'supply_iid_null': float(supply_iidnull),
           'max_mean_per_group': float(PMAX.mean()), 'group_effect_mean': float(a_g.mean()),
           'var_frac_group': float(v_grp / v_tot), 'var_frac_decision': float(v_t / v_tot),
           'var_frac_resid': float(v_res / v_tot),
           'per_t_prime_mean': [float(x) for x in b_t],
           'max_over_mean_median': float(np.median(PMAX / np.maximum(a_g, 1e-9))),
           'layers': LIDS, 'n_sid': NSID, 'G': G, 'N': int(N)}
    (ROOT / 'results' / 't37b_decision_spread.json').write_text(json.dumps(out, indent=1) + '\n')
    print('\nwrote results/t37b_decision_spread.json')


if __name__ == '__main__':
    main()
