#!/usr/bin/env python3
"""Result-triggered diagnostics; no new selectors, tuning or promotion rule."""
import hashlib
import json
from pathlib import Path
import numpy as np
import predictor_models as pm

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    raw = json.loads((HERE/'collect_v1/results.json').read_text())
    pred = json.loads((HERE/'analysis_v1/predictions_before_check_evaluation.json').read_text())
    analysis = json.loads((HERE/'analysis_v1/analysis.json').read_text())
    native = json.loads((HERE/'native_check_v1/results.json').read_text())
    fine = json.loads((HERE/'matched_fine_check_v1/results.json').read_text())
    frames = raw['frames']; fit = [f for f in frames if f['group'] == 'fit']
    check = [f for f in frames if f['group'] == 'check']
    cache = {}
    for f in frames:
        path = HERE/'collect_v1'/f['predecision_input_file']
        assert sha(path) == f['predecision_input_file_sha256']
        with np.load(path, allow_pickle=False) as z:
            cache[f['target_id']] = (z['no_GT_cell_descriptor'][0], z['cell_GT_flow'][0], z['cell_GT_valid'][0].astype(bool))
    truth = np.concatenate([cache[f['target_id']][1][cache[f['target_id']][2]] for f in fit])
    constants = {'fit_mean': truth.mean(0), 'fit_median': np.median(truth, axis=0), 'zero': np.zeros(2)}
    head = {k: np.asarray(v) if k in ('mean','std','beta','intercept') else v for k,v in pred['final_heads']['flow'].items()}
    qualities = {}
    shifts = []
    for group, members in [('fit', fit), ('check', check)]:
        entries = []
        for f in members:
            tid = f['target_id']; x, y, valid = cache[tid]
            outputs = {k: np.broadcast_to(v, y.shape) for k,v in constants.items()}
            outputs['final_fit20_head'] = pm.predict_fp32(head, x)
            if group == 'fit':
                outputs['OOF_head'] = np.asarray(pred['fit_OOF_flow_prediction'][tid])
            errors = {k: float(np.linalg.norm(v[valid]-y[valid], axis=-1).mean()) for k,v in outputs.items()}
            entries.append({'target_id': tid, 'sequence': f['sequence'], 'valid_cells': int(valid.sum()), 'cell_mean_AEE': errors})
            z = np.abs((x-head['mean'])/head['std'])
            shifts.append({'target_id': tid, 'group': group, 'max_abs_fit_standardized_descriptor': float(z.max()),
                           'fraction_abs_z_over5': float((z>5).mean())})
        qualities[group] = {'frames': entries, 'frame_mean_cell_AEE': {k: float(np.mean([e['cell_mean_AEE'][k] for e in entries])) for k in entries[0]['cell_mean_AEE']}}
    nsummary = native['summaries']['native_dense']['regions']
    fsummary = fine['summaries']['fine_stage02']['regions']
    anchor = {}
    for region in nsummary:
        n = nsummary[region]['frame_mean_AEE']; ff = fsummary[region]['frame_mean_AEE']
        c = analysis['check_summaries']['coarse_baseline']['regions'][region]['frame_mean_AEE']
        oracle = analysis['check_summaries']['posthoc_GT_best']['regions'][region]['frame_mean_AEE']
        candidate = analysis['check_summaries']['flow11_task']['regions'][region]['frame_mean_AEE']
        anchor[region] = {'native': n, 'matched_fine_stage02': ff, 'coarse_stage02': c,
                          'GT_best_k1': oracle, 'flow11_task': candidate,
                          'fine_vs_native_relative': ff/n-1 if n else None,
                          'coarse_vs_fine_relative': c/ff-1 if ff else None,
                          'coarse_vs_native_relative': c/n-1 if n else None,
                          'GT_k1_vs_native_relative': oracle/n-1 if n else None}
    # Keep every frame's quality and array equality. Different arrays can share
    # AEE; neither metric nor local branch equality implies whole-flow equality.
    nf = {f['target_id']: f for f in native['frames']}; ff = {f['target_id']: f for f in fine['frames']}
    per_frame = []
    for f in check:
        tid=f['target_id']
        with np.load(HERE/'native_check_v1'/nf[tid]['prediction_file'],allow_pickle=False) as z:
            n=z['native_dense']
        with np.load(HERE/'matched_fine_check_v1'/ff[tid]['prediction_file'],allow_pickle=False) as z:
            q=z['fine_stage02']
        per_frame.append({'target_id':tid, 'sequence':f['sequence'], 'native_vs_matched_fine_prediction_exact': bool(np.array_equal(n,q)),
                          'native_vs_matched_fine_max_abs_flow_difference': float(np.abs(n-q).max()),
                          'native_regions':nf[tid]['arms']['native_dense']['metrics']['regions'],
                          'matched_fine_regions':ff[tid]['arms']['fine_stage02']['metrics']['regions'],
                          'coarse_regions':f['arms']['coarse_stage02']['metrics']['regions']})
    result={'schema':'flow_round6_result_triggered_diagnostics_v1',
            'scope':'Post-result diagnostic. Constants fitted only to fit20, no selector is changed. In-sample final-head fit quality is separate from OOF quality. All cell-mean head errors are NOT dense optical-flow AEE. Native/fine diagnostic uses fixed allcheck8, not outcome-based member selection.',
            'sources':{str(p.relative_to(HERE)):sha(p) for p in [HERE/'collect_v1/results.json',HERE/'analysis_v1/analysis.json',HERE/'analysis_v1/predictions_before_check_evaluation.json',HERE/'native_check_v1/results.json',HERE/'matched_fine_check_v1/results.json']},
            'script_sha256':sha(Path(__file__)), 'fit_only_constants':constants,
            'head_quality':qualities,'descriptor_shift':shifts,'anchors':anchor,'per_frame_anchors':per_frame,
            'main_advancement_unchanged':analysis['advancement'], 'new_prediction_methods':False}
    out=HERE/'post_result_diagnostics.json';assert not out.exists()
    out.write_text(json.dumps(pm.serializable(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(pm.serializable({'head_quality':{k:v['frame_mean_cell_AEE'] for k,v in qualities.items()},
                                     'anchors':{r:anchor[r] for r in ('all_valid','B','L')}}),indent=2))


if __name__=='__main__':main()
