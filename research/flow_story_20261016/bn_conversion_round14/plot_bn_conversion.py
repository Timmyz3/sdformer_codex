#!/usr/bin/env python3
"""Plot every frozen development identity from independent R14 quality output."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
source = HERE / 'independent_quality_review/actual_reduction_v1.json'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'd97d9ed7b5ea84c5c002d432aed6b5e50377291bba3033127df98ae28b5a2f41'
data = json.loads(source.read_text())
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
for ax, checkpoint in zip(axes, ('135930', '271482')):
    current = data['all_endpoints'][checkpoint + '/current']
    deployed = data['all_endpoints'][checkpoint + '/deployment']
    assert len(current) == len(deployed) == 8
    assert [r['target_id'] for r in current] == [r['target_id'] for r in deployed]
    ca = [r['regions']['all_valid']['AEE'] for r in current]
    da = [r['regions']['all_valid']['AEE'] for r in deployed]
    for i, (c, d) in enumerate(zip(ca, da)):
        ax.plot([i - .10, i + .10], [c, d], color='#aeb6bf', linewidth=1.3, zorder=1)
    ax.scatter([i - .10 for i in range(8)], ca, color='#1967a3', s=38, label='All 78 BN: current statistics', zorder=2)
    ax.scatter([i + .10 for i in range(8)], da, color='#c95b18', marker='s', s=34, label='24 MLP BN: fixed statistics', zorder=2)
    ax.set_xticks(range(8), [r['target_id'].replace('zurich_city_', '').replace('_a_', 'a\n') for r in current])
    ax.set_yscale('log')
    ax.set_ylim(.75, 95)
    ax.set_yticks([1, 2, 5, 10, 20, 50], ['1', '2', '5', '10', '20', '50'])
    ax.grid(axis='y', which='major', color='#dce1e6', linewidth=.7)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    cm = data['summaries'][checkpoint + '/current']['overall']['all_valid']['frame_mean_AEE']
    dm = data['summaries'][checkpoint + '/deployment']['overall']['all_valid']['frame_mean_AEE']
    ax.set_title(f'Checkpoint {checkpoint}\nAll-frame mean: {cm:.4f} → {dm:.4f} ({(dm/cm-1)*100:+.2f}%)', fontsize=11)
    ax.set_xlabel('Frozen development identity (all eight retained)')
axes[0].set_ylabel('All-valid-GT AEE per frame (log scale)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.01))
fig.text(.5, .015, 'Same weights within each panel; zero training. Reused development subset, not official DSEC test results.', ha='center', fontsize=9, color='#4b5563')
fig.tight_layout(rect=(0, .045, 1, .92))
fig.savefig(HERE / 'bn_conversion_all8.png', dpi=180)
fig.savefig(HERE / 'bn_conversion_all8.pdf')
plt.close(fig)
