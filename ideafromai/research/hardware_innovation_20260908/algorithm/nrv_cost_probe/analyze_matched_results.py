"""Read-only marginal and local-versus-shuffled attribution for r2."""
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'r2_matched'
AXES = ('local_nr4_marginal', 'shuffled_nr4_marginal')
summary = json.loads((OUT/'summary.json').read_text())
training = json.loads((OUT/'train_stats.json').read_text())


def variation(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a/a.sum()-b/b.sum()).sum()/2)


histograms = {}
for axis in (*AXES, 'distill'):
    directory = ROOT if axis == 'distill' else OUT
    frames = json.loads((directory/(axis+'_valid10_frames.json')).read_text())
    histograms[axis] = {m: np.sum([f['sources'][m]['class_histogram'] for f in frames], axis=0)
                        for m in training}

report = {'variants': {}, 'layers': {}}
for axis in AXES:
    row = summary[axis]
    c = row['source_counts']['class']
    report['variants'][axis] = {
        'AEE_frame_mean': row['AEE_frame_mean'], 'AEE_pixel_mean': row['AEE_pixel_mean'],
        'valid_source_nonzero': c['scalar_events'], 'valid_true_NR4_distinct': c['merged_destinations'],
        'valid_distinct_per_nonzero': c['merged_destinations']/c['scalar_events'],
        'valid_source_zero_rate': row['source_zero_rate'], 'valid_spike_rate': row['spike_rate'],
        'train_distinct_per_nonzero': sum(training[m][axis]['true_nr4_distinct'] for m in training)
                                      /sum(training[m][axis]['source_nonzero'] for m in training)}
for module in training:
    report['layers'][module] = {
        'train_TV_vs_distill': {axis: training[module][axis]['marginal_total_variation_from_train_distill'] for axis in AXES},
        'valid_TV_vs_distill': {axis: variation(histograms[axis][module], histograms['distill'][module]) for axis in AXES},
        'valid_TV_local_vs_shuffled': variation(histograms[AXES[0]][module], histograms[AXES[1]][module]),
        'valid_class_histograms': {axis: histograms[axis][module].tolist() for axis in (*AXES, 'distill')}}
a, b = [report['variants'][axis] for axis in AXES]
report['local_vs_shuffled'] = {
    'AEE_frame_delta': a['AEE_frame_mean']-b['AEE_frame_mean'],
    'AEE_pixel_delta': a['AEE_pixel_mean']-b['AEE_pixel_mean'],
    'NR4_distinct_ratio': a['valid_true_NR4_distinct']/b['valid_true_NR4_distinct'],
    'nonzero_ratio': a['valid_source_nonzero']/b['valid_source_nonzero'],
    'merge_ratio_factor': a['valid_distinct_per_nonzero']/b['valid_distinct_per_nonzero']}
report['decision'] = ('Stop this fixed-dictionary/source-projection NR4 training version: local grouping '
                      'did not beat shuffled grouping under approximately matched marginals. This does '
                      'not reject the GP execution substrate or all trainable source representations.')
report['limits'] = ['one fixed seed and coefficient setting; no valid825',
                    'train marginals measured on saved integer-student upstream inputs; valid runs all six updated sources together',
                    'marginal penalty is approximate, with actual per-layer TV reported',
                    'NR4 counters precede output-specific weight-zero packing and are not service cycles']
(OUT/'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report['local_vs_shuffled'], indent=2))
