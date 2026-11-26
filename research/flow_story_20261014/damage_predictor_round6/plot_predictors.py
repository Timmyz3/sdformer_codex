#!/usr/bin/env python3
"""All fixed ROUND6 policies, same k1 action; no selective method filtering."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.analysis.read_text())
    values = data['check_summaries']
    names = list(values)
    base = values['coarse_baseline']
    labels = {n: n.replace('psn1_spatial_neighbor_hamming', 'PSN Hamming')
               .replace('psn1_activity', 'PSN activity').replace('amplitude_abs_mean', 'Amplitude')
               .replace('pooling_residual_mse', 'Pooling residual')
               .replace('posthoc_GT_best', 'GT best (offline)')
               .replace('offline_branch_MSE', 'Branch MSE (offline)')
               .replace('fixed_uniform_index3', 'Fixed index 3')
               .replace('fit_position_mean', 'Fitted position')
               .replace('coarse_baseline', 'Coarse baseline') for n in names}
    fig, axes = plt.subplots(1, 3, figsize=(13, 9), sharey=True)
    for ax, region, title in zip(axes, ('all_valid', 'B', 'L'), ('All valid GT', 'Boundary', 'Large displacement')):
        reference = base['regions'][region]['frame_mean_AEE']
        delta = [(values[n]['regions'][region]['frame_mean_AEE']/reference-1)*100 for n in names]
        color = ['#b33e47' if n == 'flow11_task' else '#458c78' if n == 'flow11_MSE'
                 else '#999999' if 'offline' in n or 'posthoc' in n else '#577fa4' for n in names]
        bars = ax.barh(np.arange(len(names)), delta, height=.58, color=color, alpha=.85)
        for n, bar in zip(names, bars):
            if n in ('offline_branch_MSE', 'posthoc_GT_best'):
                bar.set_hatch('//')
        for s, marker in zip(sorted(base['per_sequence']), ('o', 'x')):
            b = base['per_sequence'][s][region]['frame_mean_AEE']
            points = [((values[n]['per_sequence'][s][region]['frame_mean_AEE']/b-1)*100)
                      if b and values[n]['per_sequence'][s][region]['frame_mean_AEE'] is not None else np.nan for n in names]
            ax.scatter(points, np.arange(len(names)), marker=marker, s=24, color='#222222', linewidths=.9,
                       label=s.replace('zurich_city_', 'Sequence '), zorder=3)
        ax.axvline(0, color='#555555', linewidth=.8)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        ax.set_title(title)
        ax.set_xlabel('AEE change vs coarse (%)\nNegative = improvement')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(np.arange(len(names)), [labels[n] for n in names])
    axes[0].invert_yaxis()
    axes[-1].legend(loc='upper right', fontsize=8)
    fig.suptitle('ROUND6: fixed fit20 to check8 member experiment', fontsize=15, y=.985)
    fig.text(.5, .015, 'Bars: frame-equal mean; markers: sequence means. Same restored producer service, scorer costs differ.\n'
             'Two previously explored training-side sequences; no official-test or net-cycle claim. Hatched rows use offline labels.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .96))
    stem = a.analysis.parent/'round6_all_policies'
    fig.savefig(stem.with_suffix('.png'), dpi=170)
    fig.savefig(stem.with_suffix('.pdf'))
    print(str(stem))


if __name__ == '__main__':
    main()
