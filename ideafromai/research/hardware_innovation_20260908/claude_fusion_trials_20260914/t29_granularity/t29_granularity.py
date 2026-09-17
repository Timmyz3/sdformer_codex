#!/usr/bin/env python3
"""T29：供数粒度扫描——BitL（MICRO'25）的"位串↔位并"动态切换 + DF-BETA（TRETS'25）
的"检查粒度 >1 位更优"，落到 C1 门核上：**每拍发 g 个平面**（端口 10·g 位）。

动机（架构会议挖掘的方法学警告）：17% 是**拍比**（4.2 拍/组 vs FX 24 拍），位传输比是 6.3×。
**任何只改调度、不改比特数的机制，都能靠加宽端口把拍比刷下去**——故本表同时报
**bits/组**（当前 42b）作为主口径，拍比只作部署态辅助。

模型：g 个平面一拍 ⇒ 证书在颗粒边界才判一次 ⇒ 锁定发生在其后的第一个 g 边界。
    planes_sent(g) = g·ceil(p_lock/g) ≥ p_lock ，cycles(g) = ceil(p_lock/g) ，bits = 10·planes_sent
p_lock 取 T27/T28 的逐组供给平面数（口径 = T5）。

用法：python t29_granularity/t29_granularity.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 't27_bitorder'))
from t27_abo import build, cycles_for_order, MAXJ   # noqa: E402

FX_CYCLES, FX_BITS = 11, 24          # T24/T26 诚实 FX 基线：11 拍/组 @ 24b 端口


def main():
    d = ROOT / 't25_k4_gate' / 'k4'
    stim = ROOT / 'results' / 't5_rtl' / 's0_stage0' / 'stim_bf.txt'
    S = build(d, stim, 384)
    plock, _ = cycles_for_order(S, list(range(MAXJ, -1, -1)))
    G = len(plock)
    print(f'stage0: {G} 组，p_lock 均值 {plock.mean():.4f} 平面/组 '
          f'(范围 {plock.min()}–{plock.max()})')

    rows = []
    for g in (1, 2, 4, 8):
        cyc = np.ceil(plock / g)
        planes_sent = g * cyc
        bits = 10 * planes_sent
        rows.append((g, 10 * g, cyc.mean(), planes_sent.mean(), bits.mean()))

    print(f'\n{"g":>2} {"端口":>5} {"拍/组":>7} {"平面/组":>8} {"bits/组":>8} '
          f'{"拍比/FX":>9} {"bits比/FX":>10}')
    for g, port, cyc, pl, bits in rows:
        print(f'{g:>2} {port:>5} {cyc:>7.3f} {pl:>8.3f} {bits:>8.1f} '
              f'{cyc/FX_CYCLES*100:>8.1f}% {bits/(FX_CYCLES*FX_BITS)*100:>9.1f}%')
    print(f'{"FX":>2} {FX_BITS:>5} {FX_CYCLES:>7.3f} {FX_CYCLES:>8.3f} '
          f'{FX_CYCLES*FX_BITS:>8.0f} {"100.0%":>9} {"100.0%":>10}')

    # 自适应粒度的上界：按组选 g 使 cycles 最小（不计选择开销）——收益天花板
    best_cyc = np.minimum.reduce([np.ceil(plock / g) for g in (1, 2, 4, 8)])
    print(f'\n自适应 g 上界（按组取最小拍，零开销）：{best_cyc.mean():.3f} 拍/组 '
          f'(vs g=1 {np.ceil(plock).mean():.3f})')

    out = {'groups': int(G), 'fx_cycles': FX_CYCLES, 'fx_bits': FX_BITS,
           'p_lock_mean': float(plock.mean()),
           'sweep': [{'g': g, 'port_bits': port, 'cycles': float(cyc),
                      'planes': float(pl), 'bits': float(bits)}
                     for g, port, cyc, pl, bits in rows],
           'adaptive_g_cycles': float(best_cyc.mean())}
    (ROOT / 'results' / 't29_granularity.json').write_text(
        json.dumps(out, indent=1) + '\n')
    print('wrote results/t29_granularity.json')


if __name__ == '__main__':
    main()
