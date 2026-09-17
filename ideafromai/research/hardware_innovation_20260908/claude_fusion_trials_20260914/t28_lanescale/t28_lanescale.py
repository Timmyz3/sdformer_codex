#!/usr/bin/env python3
"""T28：BitMoD（HPCA'25）式**逐 lane 静态尺度**移植——证书界被组最大 lane 绑死，
能否用逐 lane 尺度（零逐拍元数据）把界收窄？

C1 现状：组共享指数 e = max_c msb(|Y_c|)，证书残差 R_k = Σ_{未读 p} 2^p 对**所有 lane 同宽**。
若 lane c 的有效位宽只有 s_c+1（高位恒零），则该 lane 可只按 s_c 计残差：
    residual_c = Σ_{p ∉ S, p ≤ caps_c} 2^p
    vmin_t = Ypart_t + Σ_c min(A[t][c],0)·residual_c
    vmax_t = Ypart_t + Σ_c max(A[t][c],0)·residual_c
关键：caps_c 若是**编译期常数**（逐 lane 上水位）则零逐拍元数据——正是 T1B 说的"逐词归一化"
的可实现形态，而 T8 的元数据税不适用。三种 caps 口径：
  oracle  : caps[g,c] = msb(|Y_{g,c}|)          —— 逐组逐 lane 精确上界（收益天花板）
  static  : caps[g,c] = max_g msb(|Y_{g,c}|)    —— 逐 lane 常数上水位（标定自本数据集）
  p999    : caps[g,c] = 99.9 分位 msb            —— 更激进的水位（对未见样本有失稳风险）

用法：python t28_lanescale/t28_lanescale.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 't27_bitorder'))
from t27_abo import build, MAXJ, T          # noqa: E402


def cycles_with_caps(S, order, caps):
    """caps[g,c] = lane c 的最高有效位位置（其上恒零）。按组记数。"""
    G = S['G']
    C, valid, base, thr, Yint = (S['C'], S['valid'], S['base'], S['thr'],
                                 S['Yint'])
    A = S['A']
    Aneg = np.minimum(A, 0)                      # A[t][c], t 判决, c lane
    Apos = np.maximum(A, 0)
    pow2 = (1 << np.arange(MAXJ + 1))
    unread = np.ones((G, MAXJ + 1), bool)        # 初始全未读
    part = base.copy()
    planes = np.zeros(G, np.int64)
    cnt = np.zeros(G, np.int64)
    done = np.zeros(G, bool)
    for p in order:
        sel = valid[:, p]
        if not sel.any():
            continue
        part = part + C[:, p, :] * sel[:, None]
        unread[sel, p] = False
        cnt = cnt + sel
        ur = np.where(unread, pow2[None, :], 0)
        cum = np.cumsum(ur, axis=1)              # (G, MAXJ+1)
        resid = cum[np.arange(G)[:, None], caps]  # (G, 10) lane c 的残差
        lo = np.einsum('tc,gc->gt', Aneg, resid)  # ≤0
        hi = np.einsum('tc,gc->gt', Apos, resid)  # ≥0
        vmin = part + lo
        vmax = part + hi
        lock = (vmin >= thr) | (vmax < thr)
        newly = lock.all(1) & ~done
        planes[newly] = cnt[newly]
        done |= lock.all(1)
        if done.all():
            break
    planes[~done] = cnt[~done]
    planes = np.maximum(planes, 1)
    return planes, done


def main():
    stage = 'stage0'
    d = ROOT / 't25_k4_gate' / 'k4'
    stim = ROOT / 'results' / 't5_rtl' / f's0_{stage}' / 'stim_bf.txt'
    S = build(d, stim, 384)
    G, TS = S['G'], T
    msb = np.zeros((G, TS), np.int64)
    absY = np.abs(S['Yint'])
    nz = absY > 0
    msb[nz] = (np.floor(np.log2(np.maximum(absY, 1)))).astype(np.int64)[nz]
    e = S['e']

    msb_oracle = msb
    width = e[:, None] - 1                       # 该组可用位位置上界（p<e）
    msb_static = np.minimum(np.tile(msb.max(0), (G, 1)), width)
    p999 = np.percentile(msb, 99.9, axis=0).astype(np.int64)
    msb_p999 = np.minimum(np.tile(p999, (G, 1)), width)

    msb_first = list(range(MAXJ, -1, -1))
    c0, _ = cycles_with_caps(S, msb_first,
                             np.tile(e[:, None] - 1, (1, TS)))
    c_or, _ = cycles_with_caps(S, msb_first, msb_oracle)
    c_st, _ = cycles_with_caps(S, msb_first, msb_static)
    c_p9, _ = cycles_with_caps(S, msb_first, msb_p999)

    # 逐组 e 与逐 lane msb 的落差（收益空间的直接度量）
    drop = e[:, None] - msb
    print(f'{stage}: {G} 组 × {TS} lane')
    print(f'  组指数 e: {e.min()}–{e.max()} (mean {e.mean():.2f})')
    print(f'  lane msb 均值: {msb.mean():.2f};  e−msb: '
          f'mean {drop.mean():.2f}, p50 {np.median(drop):.0f}, '
          f'p90 {np.percentile(drop,90):.0f}, frac(drop≥2) '
          f'{(drop>=2).mean()*100:.1f}%')
    print(f'  per-lane static 上水位 S_c: {msb_static[0]}')
    print(f'\n平均供出平面/组（MSB-first）:')
    print(f'  共享 e（现状）      {c0.mean():.4f}')
    print(f'  oracle 逐组逐 lane  {c_or.mean():.4f}  '
          f'({(c0.mean()-c_or.mean())/c0.mean()*100:+.2f}%)')
    print(f'  静态逐 lane 上水位   {c_st.mean():.4f}  '
          f'({(c0.mean()-c_st.mean())/c0.mean()*100:+.2f}%)')
    print(f'  p99.9 逐 lane 水位   {c_p9.mean():.4f}  '
          f'({(c0.mean()-c_p9.mean())/c0.mean()*100:+.2f}%)')

    out = {'stage': stage, 'groups': int(G),
           'e_mean': float(e.mean()), 'msb_mean': float(msb.mean()),
           'drop_mean': float(drop.mean()),
           'frac_drop_ge2': float((drop >= 2).mean()),
           'planes_shared': float(c0.mean()),
           'planes_oracle': float(c_or.mean()),
           'planes_static': float(c_st.mean()),
           'planes_p999': float(c_p9.mean()),
           'static_watermark': [int(x) for x in msb_static[0]]}
    (ROOT / 'results' / 't28_lanescale.json').write_text(
        json.dumps(out, indent=1) + '\n')
    print('\nwrote results/t28_lanescale.json')


if __name__ == '__main__':
    main()
