#!/usr/bin/env python3
"""T30：跨层数据分析——12 个门路层的供数比/margin 是否有跨层结构？

动机（新轴：跨层，架构挖掘代理正在做文献侧，本脚本做数据侧，互补不重叠）：
"层间级联早停 / 预算分配 / 共享门核仲裁" 这类机制成立的前提是——**层与层之间存在可利用的
统计结构**。若无结构（各层供数比独立），则跨层机制无信号、直接排除，省掉后续文献与 RTL。
本脚本用 T19 的 480 traces（12 层 × 40 序列）回答三个问题：
  Q1 各层供数比差异有多大？（谁最难）
  Q2 相邻层供数比跨序列相关吗？（级联预测信号）
  Q3 ρ=|δ|/L1 与供数比的关系是否跨层稳定？（T16 解析模型的跨层再验证）

用法：python t30_crosslayer/t30_crosslayer.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
res = json.load(open(ROOT / 'results' / 't19_result.json'))

lids = sorted({r['lid'] for r in res})
sids = sorted({r['sid'] for r in res})
print(f'{len(lids)} 层 × {len(sids)} 序列 = {len(res)} traces')
print(f'layers: {lids}')

R = np.full((len(lids), len(sids)), np.nan)
RH = np.full((len(lids), len(sids)), np.nan)
JS = np.full((len(lids), len(sids)), np.nan)
for r in res:
    i, j = lids.index(r['lid']), sids.index(r['sid'])
    R[i, j] = r['ratio_measured']
    RH[i, j] = r['rho_median']
    JS[i, j] = r['j_star_mean']

print('\nQ1 逐层供数比（跨 40 序列）:')
print(f'{"lid":>4} {"stage":>5} {"block":>5} {"供数比均值":>9} {"±std":>7} '
      f'{"j*":>6} {"rho中位":>10}')
stage_of = {r['lid']: r['stage'] for r in res}
blk_of = {r['lid']: r['block'] for r in res}
for i, lid in enumerate(lids):
    print(f'{lid:>4} {stage_of[lid]:>5} {blk_of[lid]:>5} {R[i].mean():>9.4f} '
          f'{R[i].std():>7.4f} {JS[i].mean():>6.2f} {RH[i].mean():>10.0f}')
print(f'{"ALL":>4} {"":>5} {"":>5} {np.nanmean(R):>9.4f} '
      f'{np.nanstd(R):>7.4f}')
print(f'层间极差: {R.mean(1).max()-R.mean(1).min():.4f} '
      f'({(R.mean(1).max()-R.mean(1).min())/R.mean()*100:.1f}% of mean)')

print('\nQ2 跨层相关（跨 40 序列的 Pearson r）:')
C = np.corrcoef(R)
adj = [C[i, i + 1] for i in range(len(lids) - 1)]
print(f'  相邻层 r: mean {np.mean(adj):.3f}, range {min(adj):.3f}–{max(adj):.3f}')
off = C[np.triu_indices(len(lids), 1)]
far = [C[i, j] for i in range(len(lids)) for j in range(len(lids))
       if abs(i - j) >= 3]
print(f'  全部非同层 r: mean {off.mean():.3f}; |i-j|≥3: mean {np.mean(far):.3f}')
print(f'  最高相关层对: ', end='')
iu = np.triu_indices(len(lids), 1)
k = np.argmax(C[iu])
print(f'{lids[iu[0][k]]}–{lids[iu[1][k]]} r={C[iu][k]:.3f}')

print('\nQ3 ρ vs 供数比（跨层稳定性）:')
for i, lid in enumerate(lids):
    print(f'  lid {lid:>2}: rho中位 {RH[i].mean():>9.0f}  '
          f'供数比 {R[i].mean():.4f}  j* {JS[i].mean():>5.2f}')
rr = np.corrcoef(np.log(RH.ravel()), R.ravel())[0, 1]
print(f'  log(rho) 与供数比的总体 r = {rr:.3f}')

out = {'n_layers': len(lids), 'n_seqs': len(sids),
       'per_layer': [{'lid': int(lids[i]), 'ratio_mean': float(R[i].mean()),
                      'ratio_std': float(R[i].std()),
                      'rho_median_mean': float(RH[i].mean()),
                      'jstar_mean': float(JS[i].mean())}
                     for i in range(len(lids))],
       'adj_corr_mean': float(np.mean(adj)),
       'adj_corr_min': float(min(adj)), 'adj_corr_max': float(max(adj)),
       'all_corr_mean': float(off.mean()),
       'far_corr_mean': float(np.mean(far)),
       'layer_span': float(R.mean(1).max() - R.mean(1).min()),
       'logrho_vs_ratio_r': float(rr)}
(ROOT / 'results' / 't30_crosslayer.json').write_text(
    json.dumps(out, indent=1) + '\n')
print('\nwrote results/t30_crosslayer.json')
