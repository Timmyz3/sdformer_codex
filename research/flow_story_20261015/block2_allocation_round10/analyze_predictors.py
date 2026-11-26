#!/usr/bin/env python3
"""R10 block2: frozen R6 fit/check predictor recipe; all k1 endpoints were actually executed."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from source_contract import load_source, source_path
pm = load_source("predictor_models", "_r10_frozen_predictor_models")

HERE = Path(__file__).resolve().parent
BASE_ARM = 'coarse_stage02'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    Path(path).write_text(json.dumps(pm.serializable(value), indent=2, allow_nan=False) + '\n')


def reg(frame, name, region='all_valid'):
    return frame['arms'][name]['metrics']['regions'][region]


def arm(index):
    r, c = pm.TILES[index]
    return f'fine_r{r}_c{c}'


def targets(frame):
    base = reg(frame, BASE_ARM)['AEE']
    require(base > 0, 'Positive coarse AEE needed for fixed relative target')
    task = np.asarray([(base-reg(frame, arm(i))['AEE']) / base for i in range(12)])
    branch = {tuple(v['tile']): v['MSE'] for v in frame['offline_fine_coarse_branch_mse']}
    return {'task': task, 'MSE': np.asarray([branch[t] for t in pm.TILES])}


def load_input(frame, root, training):
    path = root / frame['predecision_input_file']
    require(sha(path) == frame['predecision_input_file_sha256'], 'Input cache SHA differs')
    with np.load(path, allow_pickle=False) as z:
        x = z['no_GT_cell_descriptor'][0].astype(np.float32)
        require(x.shape == (10, 14, 384) and np.isfinite(x).all(), 'Descriptor shape/values differ')
        result = {'cells': x, 'input_sha256': sha(path)}
        if training:
            result['truth'] = z['cell_GT_flow'][0]
            result['valid'] = z['cell_GT_valid'][0].astype(bool)
            require(np.isfinite(result['truth'][result['valid']]).all(), 'Nonfinite valid head target')
    return result


def fit_projections(ids, cache):
    all_x = np.concatenate([cache[t]['cells'].reshape(-1, 384) for t in ids])
    xx = np.concatenate([cache[t]['cells'][cache[t]['valid']] for t in ids])
    yy = np.concatenate([cache[t]['truth'][cache[t]['valid']] for t in ids])
    return {'flow': pm.ridge_fit(xx, yy, 10.), 'PCA': pm.pca_fit(all_x)}


def family_for(frame, cache, heads):
    x = cache[frame['target_id']]['cells']
    flow = pm.predict_fp32(heads['flow'], x)
    pca = pm.predict_fp32(heads['PCA'], x)
    return pm.feature_families(frame['ordinary_scores'], x, flow, pca), flow, pca


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--results-sha256', required=True)
    p.add_argument('--manifest', type=Path, default=HERE/'selection_manifest_v1.json')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    require(sha(a.results) == a.results_sha256, 'Complete result SHA differs')
    require(not a.output_dir.exists(), 'Refuse output overwrite')
    raw = json.loads(a.results.read_text())
    manifest = json.loads(a.manifest.read_text())
    require(raw.get('status') == 'ROUND10_BLOCK2_SINGLETON_COLLECTION_COMPLETE' and
            raw.get('full_forward_count') == 366 and raw.get('completed_targets') == 28,
            'Complete fixed 28-member, 366-forward collection required')
    require(raw.get('original_model_unchanged') and raw.get('BN_buffers_unchanged'), 'Complete unchanged model required')
    require(raw.get('binding', {}).get('target') == 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp', 'R10 block2 target required')
    frames = raw['frames']
    by_id = {f['target_id']: f for f in frames}
    require(len(frames) == len(by_id) == 28, 'Frame count or uniqueness differs')
    require(set(by_id) == set(sum(manifest['groups'].values(), [])), 'Manifest membership differs')
    fit = [by_id[t] for t in manifest['groups']['fit']]
    check = [by_id[t] for t in manifest['groups']['check']]
    fit_seqs = sorted({f['sequence'] for f in fit})
    check_seqs = sorted({f['sequence'] for f in check})
    require(len(fit) == 20 and len(check) == 8 and len(fit_seqs) == 10 and len(check_seqs) == 2,
            'Fixed split differs')
    require(not(set(fit_seqs) & set(check_seqs)), 'Sequence leakage')
    require(all(f['group'] == 'fit' for f in fit) and all(f['group'] == 'check' for f in check), 'Group differs')
    cache = {f['target_id']: load_input(f, a.results.parent, f['group'] == 'fit') for f in frames}
    fit_features, oof_flow, folds = {}, {}, []
    for index in range(5):
        hold_seqs = fit_seqs[2*index:2*index+2]
        train_ids = [f['target_id'] for f in fit if f['sequence'] not in hold_seqs]
        held = [f for f in fit if f['sequence'] in hold_seqs]
        require(len(train_ids) == 16 and len(held) == 4, 'OOF head split differs')
        heads = fit_projections(train_ids, cache)
        for f in held:
            features, flow, _ = family_for(f, cache, heads)
            fit_features[f['target_id']] = features
            oof_flow[f['target_id']] = flow
        folds.append({'fold': index, 'train_ids': train_ids, 'held_sequences': hold_seqs,
                      'held_ids': [f['target_id'] for f in held], 'models': heads})
    final_heads = fit_projections([f['target_id'] for f in fit], cache)
    check_features, check_flow, check_pca = {}, {}, {}
    for f in check:
        features, flow, pca = family_for(f, cache, final_heads)
        check_features[f['target_id']], check_flow[f['target_id']], check_pca[f['target_id']] = features, flow, pca
    fit_targets = {f['target_id']: targets(f) for f in fit}
    policies, models, scores, rounding = {}, {}, {}, {}
    for family in ('B5', 'rich37', 'PCA11', 'flow11'):
        for target in ('task', 'MSE'):
            name = family + '_' + target
            x = np.concatenate([fit_features[f['target_id']][family] for f in fit])
            y = np.concatenate([fit_targets[f['target_id']][target] for f in fit])
            model = pm.ridge_fit(x, y, 1.)
            models[name], policies[name], scores[name], rounding[name] = model, {}, {}, {}
            for f in check:
                tid = f['target_id']; xx = check_features[tid][family]
                score = pm.predict_fp32(model, xx); reference = pm.predict(model, xx)
                policies[name][tid] = arm(pm.top1(score)); scores[name][tid] = score
                rounding[name][tid] = {'FP64_selector_reference_arm': arm(pm.top1(reference)),
                                      'same_selected_arm': pm.top1(score) == pm.top1(reference),
                                      'max_abs_score_difference': float(np.max(np.abs(score-reference)))}
    small = [f for f in fit if f['sequence'] in fit_seqs[:2]]
    require(len(small) == 4, 'Fixed small4 differs')
    for target in ('task', 'MSE'):
        name = 'B5_small4_' + target
        x = np.concatenate([fit_features[f['target_id']]['B5'] for f in small])
        y = np.concatenate([fit_targets[f['target_id']][target] for f in small])
        models[name] = pm.ridge_fit(x, y, 1.)
        scores[name] = {f['target_id']: pm.predict_fp32(models[name], check_features[f['target_id']]['B5']) for f in check}
        policies[name] = {t: arm(pm.top1(s)) for t, s in scores[name].items()}
    mean_gain = np.mean([fit_targets[f['target_id']]['task'] for f in fit], 0)
    policies['fit_position_mean'] = {f['target_id']: arm(pm.top1(mean_gain)) for f in check}
    policies['fixed_uniform_index3'] = {f['target_id']: arm(3) for f in check}
    for key in ('amplitude_abs_mean', 'pooling_residual_mse', 'psn1_activity', 'psn1_spatial_neighbor_hamming'):
        policies[key] = {}
        for f in check:
            ordinary = {tuple(v['tile']): v for v in f['ordinary_scores']}
            policies[key][f['target_id']] = arm(pm.top1([ordinary[t][key] for t in pm.TILES]))
    # Archive all deployable decisions BEFORE any access to check GT/branch
    # labels by the evaluation code. These heads cannot take GT arguments.
    a.output_dir.mkdir(parents=True, exist_ok=False)
    pred = {'schema': 'flow_round10_fit_only_decisions_v1', 'input_sha256': a.results_sha256,
            'manifest_sha256': sha(a.manifest), 'script_sha256': sha(__file__),
            'model_source_sha256': sha(source_path('predictor_models')), 'head_OOF_folds': folds,
            'final_heads': final_heads, 'selector_models': models, 'selected_arms': policies,
            'scores': scores, 'selector_FP32_vs_FP64': rounding,
            'check_features': check_features, 'fit_features': fit_features,
            'check_flow_prediction': check_flow, 'check_PCA_prediction': check_pca,
            'fit_OOF_flow_prediction': oof_flow, 'small4_ids': [f['target_id'] for f in small],
            'position_mean_gain': mean_gain, 'check_GT_used_for_predictions': False,
            'scope': 'FP64 offline solves, folded FP32 CPU prediction; shared actual prefix and executed singleton replay. No ASIC arithmetic/cycle claim.'}
    write(a.output_dir/'predictions_before_check_evaluation.json', pred)
    # Evaluation starts here. Counterfactual/GT references never enter models.
    analysis_source = source_path('summarizer')
    require(sha(analysis_source) == '0974218298cfb6c7933b85b35d82da0119aabeab3421360a258f417cf256348a', 'Frozen summarizer differs')
    spec = importlib.util.spec_from_file_location('_r5_summary', analysis_source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    policies['coarse_baseline'] = {f['target_id']: BASE_ARM for f in check}
    policies['offline_branch_MSE'] = {f['target_id']: arm(pm.top1(targets(f)['MSE'])) for f in check}
    policies['posthoc_GT_best'] = {f['target_id']: arm(pm.top1(targets(f)['task'])) for f in check}
    summaries = {}
    for name, choices in policies.items():
        summaries[name] = {'regions': module.summarize(check, choices),
                          'per_sequence': {s: module.summarize([f for f in check if f['sequence'] == s], choices) for s in check_seqs},
                          'selected_arms': choices,
                          'wins_vs_coarse': sum(reg(f, choices[f['target_id']])['AEE'] < reg(f, BASE_ARM)['AEE'] for f in check)}
    head_quality = {}
    for group, members, predictions in [('fit_OOF', fit, oof_flow), ('check', check, check_flow)]:
        vals = []
        for f in members:
            with np.load(a.results.parent/f['predecision_input_file'], allow_pickle=False) as z:
                valid = z['cell_GT_valid'][0].astype(bool); truth = z['cell_GT_flow'][0]
                e = np.linalg.norm(predictions[f['target_id']][valid]-truth[valid], axis=-1)
                vals.append({'target_id': f['target_id'], 'sequence': f['sequence'], 'valid_cells': int(valid.sum()),
                             'cell_mean_AEE': float(e.mean()) if len(e) else None})
        head_quality[group] = {'frames': vals, 'frame_mean_cell_AEE': float(np.mean([v['cell_mean_AEE'] for v in vals if v['cell_mean_AEE'] is not None])),
                               'not_dense_flow_AEE': True}
    candidate = summaries['flow11_task']
    ordinary_names = [n for n in summaries if n not in ('flow11_task', 'coarse_baseline', 'offline_branch_MSE', 'posthoc_GT_best')]
    c_aee = candidate['regions']['all_valid']['frame_mean_AEE']
    all_better = all(c_aee < summaries[n]['regions']['all_valid']['frame_mean_AEE'] for n in ordinary_names)
    beats_coarse = c_aee < summaries['coarse_baseline']['regions']['all_valid']['frame_mean_AEE']
    seq_nonreverse = all(candidate['per_sequence'][s]['all_valid']['frame_mean_AEE'] <= summaries[n]['per_sequence'][s]['all_valid']['frame_mean_AEE']
                        for s in check_seqs for n in ('psn1_spatial_neighbor_hamming', 'flow11_MSE'))
    guards = []
    for region in ('B', 'L'):
        for scope in ['overall'] + check_seqs:
            c = candidate['regions'][region] if scope == 'overall' else candidate['per_sequence'][scope][region]
            b = summaries['coarse_baseline']['regions'][region] if scope == 'overall' else summaries['coarse_baseline']['per_sequence'][scope][region]
            guards.append({'region': region, 'scope': scope,
                           'relative_change': c['frame_mean_AEE']/b['frame_mean_AEE']-1 if b['frame_mean_AEE'] else None,
                           'pass': c['frame_mean_AEE'] <= b['frame_mean_AEE']*1.01 if b['frame_mean_AEE'] is not None else True})
    result = {'schema': 'flow_round10_predictor_analysis_v1', 'input_sha256': a.results_sha256,
              'prediction_artifact_sha256': sha(a.output_dir/'predictions_before_check_evaluation.json'),
              'check_summaries': summaries, 'head_quality': head_quality,
              'advancement': {'candidate': 'flow11_task', 'all_ordinary_names': ordinary_names,
                              'beats_all_fixed_ordinary_allGT': all_better, 'beats_lower_cost_coarse_allGT': beats_coarse,
                              'sequence_nonreverse': seq_nonreverse,
                              'region_guards': guards, 'passed': all_better and beats_coarse and seq_nonreverse and all(g['pass'] for g in guards)},
              'scope': 'New train-side check members, two previously explored sequences, backbone-seen. No new network forwards; actual singleton replay. Not independent official test, significance, full algorithm reproduction or hardware speedup.',
              'hardware_benefit_claim': False}
    write(a.output_dir/'analysis.json', result)
    lines = ['# ROUND10预定block2预测器检查', '', '固定fit20→check8；全部真实singleton重放，完整scope见PROTOCOL。', '',
             '| 规则 | 全GT AEE | B | L | 全GT胜/coarse |', '|---|---:|---:|---:|---:|']
    for name, s in summaries.items():
        values = [s['regions'][r]['frame_mean_AEE'] for r in ('all_valid','B','L')]
        lines.append('| '+name+' | '+' | '.join(f'{v:.9f}' if v is not None else 'empty' for v in values)+f" | {s['wins_vs_coarse']}/8 |")
    lines += ['', '预定flow11-task晋升：'+str(result['advancement']['passed']), '',
              '逐序列、全部分区、预测头误差、区域门与失败点在analysis.json；拟合系数、折身份、FP32决定及前置features在predictions_before_check_evaluation.json。']
    (a.output_dir/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status': 'ROUND10_ANALYSIS_COMPLETE', 'advancement': result['advancement']['passed'], 'output': str(a.output_dir)}))


if __name__ == '__main__':
    main()
