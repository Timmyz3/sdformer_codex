#!/usr/bin/env python3
"""Source-checked original-D versus CMA176-D fixed32 breadth heatmaps."""
import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import summarize_breadth as summary

ROWS = (
    ('uniform', 'tile', 'uniform'), ('random', 'tile', 'seeded_random'),
    ('u_meanabs', 'tile', 'u_meanabs'), ('u_l2', 'tile', 'u_l2'),
    ('temporal_change', 'tile', 'temporal_change'), ('tp_factored', 'tile', 'tp_factored'),
    ('static_magnitude', 'static', 'magnitude'), ('static_random', 'static', 'random'),
)
RATIOS = (.75, .5, .25)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load(path, checksum):
    raw = path.read_bytes()
    summary.require(hashlib.sha256(raw).hexdigest() == checksum, 'Input SHA differs: '+str(path))
    return json.loads(raw)


def implementation_pins(value):
    pins = {}
    for name in ('policies.py', 'static_methods.py'):
        candidates = [value.get(field, {}).get(name) for field in
                      ('method_implementation_sha256', 'extended_implementation_sha256')]
        candidates = [v for v in candidates if v is not None]
        summary.require(candidates and len(set(candidates)) == 1 and len(candidates[0]) == 64,
                        'Missing or conflicting implementation SHA: '+name)
        pins[name] = candidates[0]
    return pins


def validate_pair(original, calibrated):
    for value in (original, calibrated):
        summary.require(value.get('geometry') == 'full9', 'Only full9 may be compared')
        summary.require(value.get('precision') == 'FP32, TF32 off, no autocast', 'Precision differs')
        summary.require(value.get('training_performed') is False, 'Evaluation performed weight training')
        summary.require(value.get('population', {}).get('population') == 'train_gate_development', 'Not train gate-dev')
    for key in ('checkpoint_sha256', 'checkpoint_run_contract_sha256', 'data_contract', 'source_sha256',
                'population', 'input_file_sha256'):
        summary.require(original.get(key) is not None and original[key] == calibrated.get(key), 'Paired '+key+' differs')
    summary.require(implementation_pins(original) == implementation_pins(calibrated), 'Paired implementation SHA differs')
    step = original['checkpoint_sidecar']['completed_optimizer_steps']
    summary.require(step == calibrated['checkpoint_sidecar']['completed_optimizer_steps'], 'Checkpoint step differs')
    summary.require('BN_artifact' not in original, 'Original-D panel already has calibration')
    artifact = calibrated.get('BN_artifact', {})
    metadata = artifact.get('metadata', {})
    summary.require(len(artifact.get('sha256', '')) == 64, 'Missing calibrated artifact SHA')
    summary.require(metadata.get('schema') == 'flow_breadth_bn_calibration_v1' and metadata.get('status') == 'COMPLETE'
                    and metadata.get('method') == 'cma_native_batch_stats_reset', 'Not complete CMA calibration')
    ids = metadata.get('fit_target_ids', [])
    summary.require(len(ids) == len(set(ids)) == 176 and ids == metadata.get('completed_ids')
                    and metadata.get('fit_target_ids_sha256') == digest(ids), 'Incomplete fixed176 calibration')
    summary.require(not set(ids).intersection(original['population']['target_ids']), 'Calibration/evaluation IDs overlap')
    summary.require(metadata.get('checkpoint_sha256') == original['checkpoint_sha256']
                    and metadata.get('data_contract_sha256') == digest(original['data_contract'])
                    and metadata.get('source_sha256') == original['source_sha256']
                    and metadata.get('geometry') == 'full9', 'Calibration binding differs')
    summary.require(metadata.get('non_bn_state_unchanged') is True and metadata.get('optimizer_steps') == 0
                    and metadata.get('fit_uses_final_valid') is False and metadata.get('fit_role') == 'bn_calibration',
                    'Calibration scope differs')


def panel(value, checksum, label, calibrated=False):
    result = summary.summarize(value, checksum)
    indexed = {r['case_id']: r for r in result['cases']}
    dense = indexed.get('dense_D')
    summary.require(dense is not None and dense['method'] == 'dense' and dense['bn'] == 'D', 'Complete dense_D required')
    cells = []
    for row, (name, method, mechanism) in enumerate(ROWS):
        for col, ratio in enumerate(RATIOS):
            matches = [c for c in value['cases'] if c.get('bn') == 'D' and c.get('method') == method
                       and c.get('policy' if method == 'tile' else 'static_method') == mechanism
                       and c.get('keep_fraction' if method == 'tile' else 'keep_ratio') == ratio]
            summary.require(len(matches) <= 1, 'Ambiguous matching case')
            case = matches[0] if matches else None
            complete = indexed.get(case['id']) if case else None
            status = value['arms'].get(case['id'], {}).get('status', 'NOT_EXECUTED') if case else 'MISSING_CASE'
            if case and method == 'tile':
                summary.require(case.get('execution') == 'dense_masked', 'Dynamic graph is not dense-masked')
            if complete:
                summary.require(complete['same_bn_baseline_id'] == 'dense_D', 'Wrong paired BN baseline')
                if calibrated:
                    applied = value['arms'][case['id']].get('BN_buffer_load', {})
                    summary.require(applied.get('count_per_BN') == 176 and applied.get('non_bn_state_unchanged') is True,
                                    'Case lacks actual calibrated-buffer load evidence')
                delta = complete['delta_same_bn_frame_AEE']
                summary.require(isinstance(delta, (int, float)) and math.isfinite(delta), 'Invalid plotted delta')
            else:
                delta = None
                if status == 'PASS':
                    status = 'INVALID_EVIDENCE'
            cells.append({'panel': label, 'row': row, 'column': col, 'method': name, 'keep_fraction': ratio,
                          'case_id': case['id'] if case else None, 'status': status, 'delta_AEE': delta})
    if calibrated:
        applied = value['arms']['dense_D'].get('BN_buffer_load', {})
        summary.require(applied.get('count_per_BN') == 176 and applied.get('non_bn_state_unchanged') is True,
                        'Dense reference lacks actual calibrated-buffer load evidence')
    return {'label': label, 'dense_frame_AEE': dense['frame_mean_AEE'], 'cells': cells,
            'excluded_evidence_errors': result['evidence_errors']}


def prepare(original, calibrated, original_sha, calibrated_sha):
    validate_pair(original, calibrated)
    panels = [panel(original, original_sha, 'Original D'), panel(calibrated, calibrated_sha, 'CMA176 D', True)]
    values = [c['delta_AEE'] for p in panels for c in p['cells'] if c['delta_AEE'] is not None]
    summary.require(bool(values), 'No complete method cells')
    bound = max(abs(x) for x in values) or 1e-6
    return {'schema': 'flow_breadth_heatmap_v1', 'metric': 'Mean of 32 paired per-frame all-valid-GT AEE differences; pixels',
            'original_result_sha256': original_sha, 'calibrated_result_sha256': calibrated_sha,
            'checkpoint_sha256': original['checkpoint_sha256'],
            'checkpoint_step': original['checkpoint_sidecar']['completed_optimizer_steps'],
            'target_ids': original['population']['target_ids'], 'geometry': 'full9',
            'calibration_artifact_sha256': calibrated['BN_artifact']['sha256'],
            'paired_method_source_sha256': implementation_pins(original),
            'color_scale': {'type': 'shared linear symmetric zero-centered', 'vmin': -bound, 'vmax': bound, 'clipped': False},
            'panels': panels}


def render(result, output_dir):
    # Scientific static artifacts; no torch import, CUDA, remote access or image generation.
    os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir()+'/flow_breadth_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#d7d7d7')
    norm = Normalize(result['color_scale']['vmin'], result['color_scale']['vmax'], clip=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(11.8, 7.3))
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, .065), left=.18, right=.93, bottom=.23, top=.76, wspace=.13)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    for i, (ax, data) in enumerate(zip(axes, result['panels'])):
        matrix = np.full((len(ROWS), len(RATIOS)), np.nan)
        for cell in data['cells']:
            if cell['delta_AEE'] is not None:
                matrix[cell['row'], cell['column']] = cell['delta_AEE']
        img = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(3), ['75%', '50%', '25%'])
        ax.xaxis.tick_top()
        ax.set_xlabel('Nominal keep', labelpad=10)
        ax.xaxis.set_label_position('top')
        ax.set_yticks(range(8), [r[0] for r in ROWS] if i == 0 else [])
        ax.tick_params(length=0)
        ax.set_title(data['label']+'\nDense AEE = %.4f px' % data['dense_frame_AEE'], fontsize=12, pad=45)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.2)
        ax.tick_params(which='minor', length=0)
        ax.axhline(5.5, color='#444444', linewidth=1.3)
        for cell in data['cells']:
            v = cell['delta_AEE']
            if v is None:
                label, color = 'N/A', '#555555'
            else:
                red, green, blue, _ = cmap(norm(v))
                color = 'white' if .2126*red+.7152*green+.0722*blue < .48 else '#111111'
                label = '%+.3f' % v
            ax.text(cell['column'], cell['row'], label, ha='center', va='center', color=color, fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(img, cax=fig.add_subplot(grid[0, 2]))
    cbar.set_label('Paired $\Delta$AEE vs own dense D (px)\nPositive = greater error')
    fig.suptitle('Compression sensitivity under two frozen BN states', fontsize=15, y=.965)
    fig.text(.5, .914, 'Same checkpoint step %s | full9 | FP32 | all valid GT | shared unclipped linear color scale'
             % format(result['checkpoint_step'], ','), ha='center', fontsize=10)
    footnotes = ('Interim training-side 32 targets; zero recovery training. This is exploratory quality, not final-valid reproduction.\n'
                 'Nominal keep is not equal service: tile rounding/tails and hidden-width pruning use different budgets.\n'
                 'Dynamic rows use dense-masked inference; actual-skip execution equivalence remains pending.\n'
                 'Grey N/A = missing, failed or incomplete evidence; no imputation. Calibration is common dense-model CMA on 176 disjoint inputs.')
    fig.text(.055, .052, footnotes, ha='left', va='bottom', fontsize=9, linespacing=1.55, color='#333333')
    fig.savefig(output_dir/'breadth_delta_aee.png', dpi=240)
    fig.savefig(output_dir/'breadth_delta_aee.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'calibrated'):
        p.add_argument('--'+name, type=Path, required=True)
        p.add_argument('--'+name+'-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    summary.require(not a.output_dir.exists(), 'Use a new output directory')
    result = prepare(load(a.original, a.original_sha256), load(a.calibrated, a.calibrated_sha256),
                     a.original_sha256, a.calibrated_sha256)
    result['plot_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['summary_source_sha256'] = hashlib.sha256(Path(summary.__file__).read_bytes()).hexdigest()
    a.output_dir.mkdir(parents=True)
    render(result, a.output_dir)
    (a.output_dir/'plot_evidence.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    cells = [c for panel in result['panels'] for c in panel['cells']]
    with (a.output_dir/'plot_cells.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(cells[0])); writer.writeheader(); writer.writerows(cells)
    print(json.dumps({'png': str(a.output_dir/'breadth_delta_aee.png'), 'pdf': str(a.output_dir/'breadth_delta_aee.pdf'),
                      'plotted_cells': sum(c['delta_AEE'] is not None for c in cells), 'total_cells': len(cells)}))


if __name__ == '__main__':
    main()
