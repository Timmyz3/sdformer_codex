"""Show actual temporal supports; no fabricated circuit or speedup diagram."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

HERE = Path(__file__).resolve().parent
directory = HERE/'dependency'
variants = json.loads((directory/'fit.json').read_text())['variants']
variants['reordered_masked_k4'] = json.loads((directory/'reordered_masked_k4.json').read_text())
variants.update(json.loads((directory/'reassigned_groups.json').read_text())['variants'])
panels = [
    ('native', '原完整矩阵', 10),
    ('row34', '逐行稀疏', 5),
    ('continuous334', '连续 3/3/4 分组', 4),
    ('masked_k4', '已有的四阶因果 mask', 4),
    ('reordered_masked_k4', '同一 mask，优化时间排列', 4),
    ('reassigned_fit334', '普通分组，重新分配输出行', 4),
]
plt.rcParams.update({'font.family': 'Noto Serif CJK JP', 'font.size': 10,
                     'axes.unicode_minus': False, 'svg.fonttype': 'path'})
fig, axes = plt.subplots(2, 3, figsize=(10.5, 9.5))
cmap = ListedColormap(['#f0f3f5', '#236a78'])
for ax, (name, title, yslots) in zip(axes.flat, panels):
    support = np.asarray(variants[name]['support'], dtype=bool)
    ax.imshow(support, cmap=cmap, vmin=0, vmax=1, interpolation='none')
    ax.set_xticks(range(10), range(10), fontsize=8)
    ax.set_yticks(range(10), range(10), fontsize=8)
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.7)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_xlabel('原输入时间 s')
    ax.set_ylabel('原输出时间 t')
    ax.set_title(f'{title}\n{int(support.sum())} 项；最佳保留策略需 {yslots} 个 Y', fontsize=11, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.suptitle('时间依赖的形状与状态：同为 34 项，存活期仍可不同', fontsize=15, y=.99)
fig.text(.5, .012,
         '色块表示存在连续系数，不表示 ATLIF 幅度为 1。Y 槽包含当前输入，另需一个工作 U。\n'
         '状态在规定策略内选择最佳输入顺序；尚未计卷积生产、端口、背压与位宽。',
         ha='center', va='bottom', fontsize=9, color='#444444')
fig.subplots_adjust(left=.06, right=.98, bottom=.10, top=.88,
                    wspace=.30, hspace=.55)
fig.savefig(HERE/'dependency_structure.svg', bbox_inches='tight')
fig.savefig(HERE/'dependency_structure.png', dpi=160, bbox_inches='tight')
print(HERE/'dependency_structure.svg')
