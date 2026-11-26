#!/usr/bin/env python3
"""Post-result descriptive concentration audit. Never creates a new policy."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

raw_path = HERE/'collect_v1/results.json'
pred_path = HERE/'analysis_v1/predictions_before_check_evaluation.json'
analysis_path = HERE/'analysis_v1/analysis.json'
raw = json.loads(raw_path.read_text())
pred = json.loads(pred_path.read_text())
analysis = json.loads(analysis_path.read_text())
assert pred['input_sha256'] == sha(raw_path)
assert analysis['prediction_artifact_sha256'] == sha(pred_path)
rows = []
for f in raw['frames']:
    if f['group'] != 'check':
        continue
    tid = f['target_id']
    chosen = pred['selected_arms']['flow11_task'][tid]
    coarse = f['arms']['coarse_stage02']['metrics']['regions']['all_valid']['AEE']
    value = f['arms'][chosen]['metrics']['regions']['all_valid']['AEE']
    rows.append({'target_id': tid, 'selected_arm': chosen, 'coarse_AEE': coarse,
                 'flow11_task_AEE': value, 'paired_AEE_gain': coarse-value})
assert len(rows) == 8
total = sum(r['paired_AEE_gain'] for r in rows)
largest = max(rows, key=lambda r: r['paired_AEE_gain'])
for r in rows:
    r['signed_total_gain_share_percent'] = 100*r['paired_AEE_gain']/total if total else None
result = {
    'schema': 'flow_round10_post_result_concentration_v1',
    'source_sha256': {str(p): sha(p) for p in (raw_path, pred_path, analysis_path, Path(__file__))},
    'rows': rows,
    'largest_gain_member': largest['target_id'],
    'largest_gain_signed_share_percent': largest['signed_total_gain_share_percent'],
    'other_seven_mean_AEE_gain': (total-largest['paired_AEE_gain'])/7,
    'scope': 'Post-result description, not a new gate or sample exclusion. All eight remain in every main result. No significance or causal inference.'
}
output = HERE/'analysis_v1/concentration.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

summaries = analysis['check_summaries']
names = list(summaries)
base = summaries['coarse_baseline']['regions']['all_valid']['frame_mean_AEE']
values = [100*(summaries[n]['regions']['all_valid']['frame_mean_AEE']/base-1) for n in names]
colors = ['#237a8d' if n == 'flow11_task' else '#d0a568' if n in ('offline_branch_MSE','posthoc_GT_best') else '#a7b4c5' for n in names]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 7), gridspec_kw={'width_ratios':[1.25,1]})
axes[0].barh(np.arange(len(names)), values, color=colors)
axes[0].set_yticks(np.arange(len(names)), names, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_xlabel('All-GT AEE change versus CC (%) — lower is better')
axes[0].set_title('All 19 fixed rules / references')
gains = [r['paired_AEE_gain'] for r in rows]
axes[1].barh(np.arange(8), gains, color=['#237a8d' if v >= 0 else '#ad4537' for v in gains])
axes[1].set_yticks(np.arange(8), [r['target_id'].replace('zurich_city_', '') for r in rows], fontsize=9)
axes[1].invert_yaxis()
axes[1].set_xlabel('CC AEE − flow11-task AEE (positive = improvement)')
axes[1].set_title('Every check member; no outlier removal')
for ax in axes:
    ax.axvline(0, color='#4b5563', lw=.8)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
fig.suptitle('R10: ordinary B5 ties the flow candidate; activity is stronger overall', fontsize=12)
fig.text(.02, .025, '05a_0229 contributes %.2f%% of signed total gain. The other seven remain reported.\nGold bars use offline information and are not deployable policies; this is train-side exploration.' % result['largest_gain_signed_share_percent'], fontsize=9)
fig.tight_layout(rect=(0,.08,1,.96))
fig.savefig(HERE/'analysis_v1/all_rules_and_concentration.png', dpi=180)
fig.savefig(HERE/'analysis_v1/all_rules_and_concentration.pdf')
print(json.dumps({k:result[k] for k in ('largest_gain_member','largest_gain_signed_share_percent','other_seven_mean_AEE_gain')}))
