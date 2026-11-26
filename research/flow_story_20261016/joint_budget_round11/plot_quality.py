#!/usr/bin/env python3
"""Descriptive figures retaining all fixed policies, budgets, and members."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    z = json.loads(a.analysis.read_text())
    raw = json.loads(a.raw.read_text())
    assert hashlib.sha256(a.raw.read_bytes()).hexdigest() == z['input_sha256']
    names = raw['binding']['policy_names']
    summary = z['summaries']
    def mean(n, r, scope='overall'):
        return summary[n]['scopes'][scope][r]['frame_mean_AEE']
    cells = np.array([[100*(mean(n+'_k'+str(k), r)/mean('CC', r)-1)
                       for k in (2, 4) for r in ('all_valid', 'B', 'L')]
                      for n in names])
    limit = max(abs(cells.min()), abs(cells.max()), 0.1)
    fig, ax = plt.subplots(figsize=(11, 8.7))
    im = ax.imshow(cells, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(range(6), ['k2: allGT', 'k2: boundary', 'k2: large',
                            'k4: allGT', 'k4: boundary', 'k4: large'])
    ax.set_yticks(range(len(names)), names)
    ax.axvline(2.5, color='black', lw=1.5)
    for y in range(len(names)):
        for x in range(6):
            ax.text(x, y, '%+.3f'%cells[y, x], ha='center', va='center', fontsize=9,
                    color='white' if abs(cells[y, x])>limit*.65 else 'black')
    primary = names.index('flow11_task')
    ax.add_patch(plt.Rectangle((-.5, primary-.5), 6, 1, fill=False, lw=2, color='black'))
    ax.set_title('R11: all 16 fixed policies, both actual joint budgets\n'
                 'Frame-mean AEE change from CC (%); lower is better')
    fig.colorbar(im, ax=ax, label='AEE change from CC (%)')
    fig.text(.5, .015, '8 train-side members / 2 historically exposed sequences; no official-test or net-cycle claim.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, 1))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output_dir/'all_policies_both_budgets.png', dpi=170)
    fig.savefig(a.output_dir/'all_policies_both_budgets.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.8))
    labels = [f['target_id'].replace('zurich_city_', '') for f in raw['frames']]
    def value(f, n):
        return f['arms'][n]['metrics']['regions']['all_valid']['AEE']
    for col, k in enumerate((2, 4)):
        n = 'flow11_task_k'+str(k)
        best = z['per_budget_gates'][str(k)]['best_comparator_by_overall_allGT']+'_k'+str(k)
        for row, ref in enumerate(('CC', best)):
            delta = [value(f, n)-value(f, ref) for f in raw['frames']]
            ax = axes[row, col]
            ax.bar(range(8), delta, color=['#3b7fa3' if x<=0 else '#b45750' for x in delta])
            ax.axhline(0, color='#333333', lw=.8)
            ax.set_xticks(range(8), labels, rotation=60, ha='right')
            ax.set_ylabel('Primary minus reference AEE')
            ax.set_title('k%d: flow11_task vs %s'%(k, ref.replace('_k'+str(k), '')))
            ax.grid(axis='y', alpha=.18)
    fig.suptitle('R11: every member retained; negative differences favor the primary', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(a.output_dir/'paired_members_both_budgets.png', dpi=170)
    fig.savefig(a.output_dir/'paired_members_both_budgets.pdf')
    plt.close(fig)
    receipt = {'scope':'Descriptive plots only; no changed gate or new hypothesis test.',
               'analysis_sha256':hashlib.sha256(a.analysis.read_bytes()).hexdigest(),
               'raw_sha256':z['input_sha256'],
               'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(a.output_dir.glob('*')) if p.suffix in ('.png', '.pdf')}}
    (a.output_dir/'plot_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
