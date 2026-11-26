"""Plot all eight fixed frames and all twelve singleton gains; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quality', type=Path, required=True)
    p.add_argument('--quality-sha256', required=True)
    p.add_argument('--output-prefix', type=Path, required=True)
    a = p.parse_args()
    digest = hashlib.sha256(a.quality.read_bytes()).hexdigest()
    assert digest == a.quality_sha256
    q = json.loads(a.quality.read_text())
    assert q['schema'] == 'flow_r17_dense_independent_raw_quality_v1' and q['status'] == 'COMPLETE_RAW_REDUCTION'
    assert q['raw_predictions'] == 120 and len(q['frames']) == len(q['action_values']) == 8
    arms = ['fine_%d_%d' % (r, c) for r in range(3) for c in range(4)]
    labels, native, coarse, gains = [], [], [], []
    for frame, values in zip(q['frames'], q['action_values']):
        assert frame['target_id'] == values['target_id']
        labels.append(frame['target_id'].replace('zurich_city_', ''))
        e = {arm: v['all_valid']['AEE'] for arm, v in frame['arms'].items()}
        native.append(e['native'])
        coarse.append(e['CC'])
        g = [values['regions']['all_valid']['singleton_gains'][arm] for arm in arms]
        assert np.allclose(g, [e['CC']-e[arm] for arm in arms], atol=1e-12, rtol=1e-12)
        gains.append(g)
    gains = np.array(gains)
    assert gains.shape == (8, 12) and np.isfinite(gains).all()
    limit = max(float(np.abs(gains).max()), 1e-12)
    fig, (ax, heat) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={'width_ratios': [1.0, 2.1]})
    y = np.arange(8)
    for i, (n, c) in enumerate(zip(native, coarse)):
        ax.plot([n, c], [i, i], color='#bcc6cf', lw=2, zorder=1)
    ax.scatter(native, y, c='#164e79', s=45, label='Native', zorder=3)
    ax.scatter(coarse, y, c='#bc4b2e', marker='x', s=50, label='Target all coarse', zorder=4)
    ax.set_yticks(y, labels)
    ax.set_ylim(7.6, -.6)
    ax.set_xlabel('AEE on all valid GT (pixels)')
    ax.set_title('All fixed H8 frames', loc='left', fontweight='bold')
    ax.grid(axis='x', color='#dfe4e8', lw=.7)
    ax.legend(loc='best', frameon=False, fontsize=8)
    values = heat.imshow(gains, cmap='RdBu', norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect='auto')
    heat.set_yticks(y, labels)
    heat.set_xticks(range(12), [a.replace('fine_', '').replace('_', ',') for a in arms])
    heat.set_xlabel('Fine tile (row, column); other 19 target tiles remain coarse')
    heat.set_title('Measured conditional gain for every singleton', loc='left', fontweight='bold')
    for row in range(8):
        for col in range(12):
            v = gains[row, col]
            heat.text(col, row, f'{v:.3g}', ha='center', va='center', fontsize=7,
                      color='white' if abs(v) > .60*limit else '#171d24')
    cb = fig.colorbar(values, ax=heat, fraction=.033, pad=.02)
    cb.set_label('AEE(CC) - AEE(single fine tile), pixels')
    fig.suptitle('Dense reference action screen — training-side evidence only', x=.05, ha='left', fontsize=15, fontweight='bold')
    fig.text(.05, .025, 'All 96 signed gains are shown in the frozen frame/tile order. Positive means improvement over CC.\n'
             'Each action is evaluated on the complete valid output domain. These dense executions establish no hardware saving or deployable selector advantage.',
             fontsize=9, color='#4b5560')
    fig.subplots_adjust(left=.09, right=.96, top=.87, bottom=.18, wspace=.42)
    for suffix in ('.png', '.pdf'):
        dest = a.output_prefix.with_suffix(suffix)
        assert not dest.exists()
        fig.savefig(dest, dpi=180, bbox_inches='tight')
    receipt = {'schema': 'flow_r17_all8_all96_figure_v1', 'quality_sha256': digest,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'frame_order': labels, 'tile_order': arms, 'gains': gains.tolist(),
               'frames': 8, 'signed_gains': 96, 'GT_regions_plotted': ['all_valid'],
               'all16_regions_retained_in_source_quality': True, 'quality_subset_selection': False,
               'hardware_benefit_claim': False, 'flow_predictor_benefit_claim': False,
               'artifacts': {s: hashlib.sha256(a.output_prefix.with_suffix(s).read_bytes()).hexdigest() for s in ('.png', '.pdf')}}
    dest = a.output_prefix.with_suffix('.json')
    assert not dest.exists()
    dest.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'frames': 8, 'signed_gains': 96, 'source_quality_sha256': digest}))


if __name__ == '__main__':
    main()
