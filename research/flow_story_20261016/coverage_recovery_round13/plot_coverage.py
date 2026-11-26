#!/usr/bin/env python3
"""Plot every own-layout endpoint from the completed, frozen R13 analysis."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    assert data['evaluation_endpoint_count'] == 320
    assert data['state_layout_count'] == 40
    assert data['initial_same_layout_prediction_SHA_and_metrics_cross_condition_exact']
    assert not args.output_dir.exists()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    arms = ('native', 'FF', 'CC', 'fine_s2b2')
    labels = ('Native', 'All fine', 'All coarse', 'Block 2 fine')
    series = (('Initial', '#909090'), ('40 identities x 5', '#c3742a'),
              ('200 identities x 1', '#2875a7'))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    plotted = []
    for ax, region, title in zip(axes, ('all_valid', 'B', 'L'),
                                 ('All valid GT', 'Boundary proxy B', 'Large displacement L')):
        for j, (name, color) in enumerate(series):
            values = []
            for arm in arms:
                condition = 'diversity' if j == 2 else 'reference'
                key = ('initial' if j == 0 else arm) + '/' + arm
                entry = data['summaries'][condition][key]['overall'][region]
                assert entry['frame_mean_AEE'] is not None
                values.append(entry['frame_mean_AEE'])
                plotted.append({'series': name, 'arm': arm, 'region': region, **entry})
            ax.bar(np.arange(4) + (j - 1) * .24, values, width=.23, color=color, label=name)
        ax.set_title(title)
        ax.set_xticks(range(4), labels, rotation=25, ha='right')
        ax.set_ylabel('Frame-mean AEE (pixels); lower is better')
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', color='#dddddd', linewidth=.5)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, 1.01))
    fig.text(.015, .015,
             'R13 development check: 8 frames / 2 sequences; 200 updates per arm. '
             'Each arm uses its own layout.\n'
             'This plot does not replace the four quality-gate families, '
             'per-sequence results, or cross-layout checks. Not official DSEC test.',
             fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, .92))
    args.output_dir.mkdir(parents=True)
    fig.savefig(str(args.output_dir / 'own_layout_AEE.png'), dpi=180)
    fig.savefig(str(args.output_dir / 'own_layout_AEE.pdf'))
    plt.close(fig)
    receipt = {'analysis_sha256': hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'plotted_values': plotted,
               'scope': 'All own-layout endpoints; descriptive development check, no significance or gate claim.'}
    (args.output_dir / 'plot_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
