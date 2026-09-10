"""Draw the actual quantized temporal connectivity of the compared students."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

HERE = Path(__file__).resolve().parent
shared = [2, 3, 7]
fig = plt.figure(figsize=(10, 6.2), layout='constrained')
grid = fig.add_gridspec(2, 2, height_ratios=[4, 1.3])
titles = ['Row-wise sparse PSN', 'Three shared columns + private tails']
variants = ['row34', 'common3_diagonal_34']
for col, (name, title) in enumerate(zip(variants, titles)):
    data = np.load(HERE/'integer_deployment'/f'{name}.npz')
    a = data['temporal_int16']
    support = a != 0
    codes = support.astype(int)
    if col:
        codes[:, shared] *= 2
    ax = fig.add_subplot(grid[0, col])
    ax.imshow(codes, cmap=ListedColormap(['#f7f7f5', '#d28137', '#247c98']),
              vmin=0, vmax=2, interpolation='none')
    for t, s in zip(*np.nonzero(support)):
        ax.text(s, t, '+' if a[t, s] > 0 else '−', ha='center', va='center',
                color='white', fontsize=13, weight='bold')
    ax.set_xticks(np.arange(10), [f'{s}' for s in range(10)])
    ax.set_yticks(np.arange(10), [f'{t}' for t in range(10)])
    ax.set_xticks(np.arange(-.5, 10), minor=True)
    ax.set_yticks(np.arange(-.5, 10), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_xlabel('Conv1 output time column s (continuous Yi)')
    ax.set_ylabel('PSN output time t (theta × gate)')
    ax.set_title(title+'\n34 signed coefficients; rank 10', fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bars = fig.add_subplot(grid[1, col])
    degree = support.sum(0)
    colors = ['#247c98' if col and s in shared else '#d28137' for s in range(10)]
    bars.bar(np.arange(10), degree, color=colors, width=.7)
    bars.set_xticks(np.arange(10))
    bars.set_ylim(0, 11.5)
    bars.set_yticks([0, 1, 5, 10])
    bars.set_ylabel('Temporal\nconsumers')
    bars.set_xlabel('Original time column s')
    for s, n in enumerate(degree):
        bars.text(s, n+.2, str(n), ha='center', fontsize=9)
    bars.spines[['top', 'right']].set_visible(False)
fig.suptitle('Change the dependency graph before scheduling expensive Conv production', fontsize=14)
fig.supxlabel('Blue: shared context columns. Each orange tail in the right panel serves only its own time output.\n'
              'The 32 spatial/channel consumers in each P4 × H8 request group still share that producer.', fontsize=10)
fig.savefig(HERE/'integer_dependency.svg')
fig.savefig(HERE/'integer_dependency.png', dpi=160)
print(HERE/'integer_dependency.svg')
