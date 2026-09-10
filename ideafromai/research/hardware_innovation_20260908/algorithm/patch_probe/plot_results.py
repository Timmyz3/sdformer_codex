"""Plot the measured production-mask tradeoff; operation counts are not cycles."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

p = Path(__file__).resolve().parent
font = '/usr/share/fonts/google-noto-cjk/NotoSerifCJK-Regular.ttc'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                     'font.size': 10, 'axes.unicode_minus': False})
data = json.loads((p/'production_mask_valid10_summary.json').read_text())
base = data['parent']['counts']
pair = base['remaining_conv1_active_terms']+base['conv2_active_terms']
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
names = ('temporal_0.25','temporal_0.5','temporal_0.75')
remain = np.array([[data[n]['counts']['produced_time_tile_fraction'],
                   data[n]['counts']['remaining_conv1_active_terms']/base['remaining_conv1_active_terms'],
                   (data[n]['counts']['remaining_conv1_active_terms']+data[n]['counts']['conv2_active_terms'])/pair]
                  for n in names])
xx = np.arange(3)
for i, (label, color) in enumerate(zip(('生产时间块','conv1活跃乘加','两层合计活跃乘加'),
                                      ('#849fbb','#cb762d','#34654d'))):
    bars = axes[0].bar(xx+(i-1)*.25, remain[:,i]*100, .24, label=label, color=color)
    axes[0].bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
axes[0].set_xticks(xx, ('训练分位25%','训练分位50%','训练分位75%'))
axes[0].set_ylim(0,116)
axes[0].set_ylabel('相对母模型的剩余量 / %')
axes[0].set_title('少生产时间块 ≠ 等比例少算')
axes[0].legend(loc='upper right', fontsize=8)
for mode,label,color,marker in [('spatial','跨T共享空间掩码','#8b5b81','s'),
                               ('temporal','逐T生产掩码','#27758b','o')]:
    rows=[data[f'{mode}_{q}'] for q in (.25,.5,.75)]
    x=[(r['counts']['remaining_conv1_active_terms']+r['counts']['conv2_active_terms'])/pair*100 for r in rows]
    y=[r['AEE_frame_mean'] for r in rows]
    axes[1].plot(x,y,marker=marker,color=color,label=label)
axes[1].scatter([100],[data['parent']['AEE_frame_mean']],marker='*',s=110,color='#222222',label='母模型')
axes[1].set_xlabel('两层活跃乘加剩余量 / %')
axes[1].set_ylabel('十帧 AEE')
axes[1].set_title('固定训练阈值，真实后续网络')
axes[1].legend(fontsize=8)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
fig.suptitle('最贵 patch 残差块：输入端生产掩码探针（未梯度训练；非周期/PPA）',fontsize=12)
fig.savefig(p/'production_mask_tradeoff.svg')
fig.savefig(p/'production_mask_tradeoff.png',dpi=150)
