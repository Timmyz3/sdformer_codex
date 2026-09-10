"""Summarize the completed fixed-setting probe; no model runs or parameter fits."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
summaries = json.loads((ROOT / 'summary.json').read_text())
weights = json.loads((ROOT / 'teacher_weight_sparsity.json').read_text())
initial = json.loads((ROOT.parent / 'direct_code_integer/summary.json').read_text())
expected = initial['integer_student']
for key in ('frames', 'valid_pixels', 'AEE_frame_mean', 'AEE_pixel_mean'):
    assert summaries['start_integer'][key] == expected[key], key
assert summaries['start_integer']['source_counts']['class']['merged_destinations'] == 10232609
assert summaries['start_integer']['source_counts']['time']['merged_destinations'] == 13461891

results = {'scope': 'Same ten complete validation frames; six S2 sources. Counts are pre-weight-zero NR4 opportunities, not service cycles.',
           'starting_model_and_prior_NR_count_reproduced': True, 'variants': {}, 'comparisons': {}, 'layers': {}}
for name, result in summaries.items():
    c, t = result['source_counts']['class'], result['source_counts']['time']
    results['variants'][name] = {
        'AEE_frame_mean': result['AEE_frame_mean'], 'AEE_pixel_mean': result['AEE_pixel_mean'],
        'source_zero_rate': result['source_zero_rate'], 'spike_rate': result['spike_rate'],
        'class_merged_destinations': c['merged_destinations'],
        'class_nonzero_source_events': c['scalar_events'],
        'merged_per_nonzero_source': c['merged_destinations'] / c['scalar_events'],
        'member_merge_savings': 1 - c['merged_destinations'] / c['scalar_events'],
        'class_vs_same_variant_time_reduction': 1 - c['merged_destinations'] / t['merged_destinations']}
for baseline in ('start_integer', 'distill', 'spike_cost'):
    a, b = results['variants']['nr4_cost'], results['variants'][baseline]
    results['comparisons']['nr4_cost_vs_' + baseline] = {
        'AEE_frame_delta': a['AEE_frame_mean'] - b['AEE_frame_mean'],
        'AEE_pixel_delta': a['AEE_pixel_mean'] - b['AEE_pixel_mean'],
        'merged_destination_ratio': a['class_merged_destinations'] / b['class_merged_destinations'],
        'nonzero_source_ratio': a['class_nonzero_source_events'] / b['class_nonzero_source_events'],
        'merge_ratio_factor': a['merged_per_nonzero_source'] / b['merged_per_nonzero_source']}
for module in weights:
    row = {'teacher_weights': weights[module], 'variants': {}}
    for name in summaries:
        frames = json.loads((ROOT / (name + '_valid10_frames.json')).read_text())
        sources = [frame['sources'][module] for frame in frames]
        nonzero = sum(s['class']['scalar_events'] for s in sources)
        merged = sum(s['class']['merged_destinations'] for s in sources)
        hist = [sum(s['class_histogram'][k] for s in sources) for k in range(8)]
        row['variants'][name] = {'class_merged_destinations': merged, 'nonzero_source_events': nonzero,
                                 'merged_per_nonzero_source': merged / nonzero,
                                 'source_zero_rate': hist[0] / sum(hist), 'class_histogram': hist}
    results['layers'][module] = row
results['interpretation'] = [
    'NR4 cost improves both AEE and transactions over this equal-budget distillation control.',
    'Ordinary spike regularization has fewer transactions but worse AEE; no matched-accuracy or matched-rate sweep was run.',
    'Nearly all NR4 transaction reduction is explained by fewer nonzero source events. Merge efficiency does not improve over distillation.',
    'Fixed mapping and dictionary do not introduce a new inference structure. Weight-zero packing and complete service remain unmeasured.',
    'No valid825 evaluation or second GPU training was launched.'
]
(ROOT / 'analysis.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(results['comparisons'], indent=2))
