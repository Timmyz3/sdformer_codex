"""CPU-only, fit64 sequence-OOF counter features and bounded decision programs."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import os
import time
import numpy as np

B = Path(__file__).resolve().parent
RESEARCH = B.parents[2]
R33 = RESEARCH / 'flow_story_20261021/round33_predictability'
FIT = R33 / 'packs/fit_tune_128_v1/fit.npz'
QUALITY = R33 / 'packs/fit_tune_128_v1/QUALITY_V1.json'
spec = importlib.util.spec_from_file_location('r33_ridge', R33/'algorithm/predictors.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
VOL = 192000
COLS = np.array([8*c+j for c in range(16) for j in (2, 3)])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    assert not p.exists(), str(p)
    p.write_text(json.dumps(M.jsonable(obj), indent=2, allow_nan=False)+'\n')


def leaf(cost, weight):
    sums = (cost*weight[:, None]).sum(axis=0)
    action = int(np.argmin(sums))
    return dict(action=action, cost_sum=float(sums[action]), rows=len(cost),
                weight=int(weight.sum()), action_cost_sums=sums)


def fit_tree(x, cost, weight, depth=0):
    node = leaf(cost, weight)
    if depth == 2 or len(x) < 16:
        return node
    best = node['cost_sum']
    choice = None
    # Weights are globally normalized for a numerically scale-independent
    # 1e-15 tolerance, but integer support is retained in saved metadata.
    total_weight = float(weight.sum())
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind='stable')
        xx = x[order, j]
        cumulative = np.cumsum(cost[order]*weight[order, None], axis=0)
        for k in range(8, len(x)-7):
            if xx[k-1] == xx[k]:
                continue
            left, right = cumulative[k-1], cumulative[-1]-cumulative[k-1]
            value = float(left.min()+right.min())
            if value/total_weight < best/total_weight-1e-15:
                best = value
                choice = (j, int(xx[k-1]))
    if choice is None:
        return node
    j, threshold = choice
    mask = x[:, j] <= threshold
    node.update(feature=j, threshold=threshold,
                left=fit_tree(x[mask], cost[mask], weight[mask], depth+1),
                right=fit_tree(x[~mask], cost[~mask], weight[~mask], depth+1))
    return node


def predict_tree(node, x):
    actions, comparisons = [], []
    for row in x:
        n, count = node, 0
        while 'feature' in n:
            n = n['left'] if row[n['feature']] <= n['threshold'] else n['right']
            count += 1
        actions.append(n['action'])
        comparisons.append(count)
    return np.asarray(actions, dtype=np.int64), comparisons


def fit_policies(d, keep):
    x, w, y = d['x'][keep], d['valid_pixels'][keep], d['delta_aee'][keep]
    aux = d['aux_g1'][keep]
    c = (aux[:, :2]*w[:, None]).sum(axis=0)/w.sum()
    p = (aux[:, 2:].reshape(-1, 6, 2)*c).sum(axis=2)
    count = d['counts'][keep]
    models = {}
    for target_name, target in [('direct', y), ('projection', p)]:
        models['X128_ridge_'+target_name] = M.train_ridge(x, target, w, .1)
        models['count32_ridge_'+target_name] = M.train_ridge(count, target, w, .1)
        models['count32_tree_'+target_name] = fit_tree(count, target, w)
        models['static_'+target_name] = leaf(target, w)
    return dict(c=c, models=models, fit_ids=d['ids'][keep], fit_groups=sorted(set(d['groups'][keep])))


def predict_policies(fit, d, held):
    ans = {}
    for name, model in fit['models'].items():
        if '_ridge_' in name:
            x = d['x'][held] if name.startswith('X128') else d['counts'][held]
            scores = M.ridge_predict(model, x)
            ans[name] = dict(actions=np.argmin(scores, axis=1), scores=scores)
        elif '_tree_' in name:
            actions, comparisons = predict_tree(model, d['counts'][held])
            ans[name] = dict(actions=actions, comparisons=comparisons)
        else:
            ans[name] = dict(actions=np.full(int(held.sum()), model['action']))
    return ans


def main():
    start = time.monotonic()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    freeze = json.loads((B/'SOURCE_FREEZE_V1.json').read_text())
    for name, digest in freeze['files'].items():
        assert sha(RESEARCH/name) == digest, name
    with np.load(FIT, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    assert len(d['ids']) == 64 and set(d['roles']) == {'fit'}
    assert len(set(d['groups'])) == 8
    counts = np.rint(d['x'][:, COLS]*VOL).astype(np.int64)
    assert np.max(np.abs(counts/VOL-d['x'][:, COLS])) < 1e-15
    assert counts.min() >= 0 and counts.max() <= VOL
    d['counts'] = counts
    out = B/'actual_v1'
    out.mkdir(exist_ok=False)
    folds, actions = [], {}
    for group in sorted(set(d['groups'])):
        held = d['groups'] == group
        fit = fit_policies(d, ~held)
        assert not set(fit['fit_ids']) & set(d['ids'][held])
        predictions = predict_policies(fit, d, held)
        for name, row in predictions.items():
            actions.setdefault(name, np.full(64, -99, dtype=np.int64))[held] = row['actions']
        folds.append(dict(held_group=group, held_ids=d['ids'][held], fit=fit, predictions=predictions))
    assert all(np.all(a >= 0) for a in actions.values())
    save(out/'PREDICTIONS_BEFORE_QUALITY_V1.json', dict(ids=d['ids'], groups=d['groups'],
         counts=counts, column_indices=COLS, cell_volume=VOL, folds=folds, actions=actions,
         source_sha256=sha(__file__), input_sha256=sha(FIT), quality_not_read_yet=True))
    final = fit_policies(d, np.ones(64, dtype=bool))
    save(out/'FINAL_FIT_MODELS_V1.json', dict(**final, in_sample_quality_not_evaluated=True,
         purpose='Only future deployment interface; all current quality is outer-OOF'))
    # QUALITY container has fit+tune rows. Discard all non-fit identities before
    # touching their states; no tune metric is used for fitting or selection.
    wanted = set(d['ids'].tolist())
    raw = {f['target_id']: f for f in json.loads(QUALITY.read_text())['frames'] if f['target_id'] in wanted}
    assert set(raw) == wanted and all(f['role'] == 'fit' for f in raw.values())
    for i, tid in enumerate(d['ids']):
        states = raw[tid]['states']
        a0 = states['background_Q4']['all_valid']['AEE']
        assert abs(a0-d['background_AEE'][i]) < 1e-10
        for a in range(6):
            assert abs(states[f'upgrade_{a}']['all_valid']['AEE']-a0-d['delta_aee'][i, a]) < 1e-10
    actions.update(Q4=np.full(64, -1), oracle6=np.argmin(d['delta_aee'], axis=1),
                   uniform6=np.full(64, -2))
    region_names = list(next(iter(raw.values()))['states']['background_Q4'])
    results = {}
    for name, selected in actions.items():
        frames, tails = [], []
        for i, tid in enumerate(d['ids']):
            a = int(selected[i])
            states = raw[tid]['states']
            if a == -2:
                regions = {}
                for reg in region_names:
                    rows = [states[f'upgrade_{k}'][reg] for k in range(6)]
                    n = rows[0]['valid_pixels']
                    es = float(np.mean([r['error_sum'] for r in rows]))
                    regions[reg] = dict(valid_pixels=n, error_sum=es, AEE=es/n if n else None)
            else:
                regions = states['background_Q4' if a == -1 else f'upgrade_{a}']
            for reg in ['all_valid', 'B', 'L']:
                ref, cand = states['background_Q4'][reg], regions[reg]
                if cand['valid_pixels'] >= 256 and (cand['AEE'] > 1.2*ref['AEE'] or cand['AEE']-ref['AEE'] > 1):
                    tails.append(dict(target_id=tid, region=reg, candidate=cand, Q4=ref))
            regret = regions['all_valid']['AEE']-d['background_AEE'][i]-d['delta_aee'][i].min()
            frames.append(dict(target_id=tid, sequence=d['groups'][i], action=a,
                               regret_vs_oracle6=float(regret), regions=regions))
        summaries = {}
        for seq in ['overall']+sorted(set(d['groups'])):
            rows = [f for f in frames if seq == 'overall' or f['sequence'] == seq]
            reg_summary = {}
            for reg in region_names:
                n = sum(f['regions'][reg]['valid_pixels'] for f in rows)
                es = sum(f['regions'][reg]['error_sum'] for f in rows)
                means = [f['regions'][reg]['AEE'] for f in rows if f['regions'][reg]['valid_pixels']]
                reg_summary[reg] = dict(valid_pixels=n, error_sum=es, AEE=es/n if n else None,
                                       frame_mean_AEE=float(np.mean(means)) if means else None)
            rw = sum(f['regret_vs_oracle6']*f['regions']['all_valid']['valid_pixels'] for f in rows)
            summaries[seq] = dict(regions=reg_summary,
                 pixel_weighted_regret_vs_oracle6=rw/reg_summary['all_valid']['valid_pixels'],
                 frame_mean_regret_vs_oracle6=float(np.mean([f['regret_vs_oracle6'] for f in rows])),
                 action_counts={str(k):sum(f['action'] == k for f in rows) for k in range(-2, 6)})
        results[name] = dict(summaries=summaries, frames=frames, high_support_tails=tails)
    comparisons = {}
    for base in ['static_direct', 'X128_ridge_direct', 'X128_ridge_projection', 'count32_tree_direct']:
        comparisons[base] = {}
        for name, result in results.items():
            a = result['summaries']['overall']['regions']['all_valid']['AEE']
            b = results[base]['summaries']['overall']['regions']['all_valid']['AEE']
            seqdiff = {g:result['summaries'][g]['regions']['all_valid']['AEE']-
                        results[base]['summaries'][g]['regions']['all_valid']['AEE'] for g in sorted(set(d['groups']))}
            comparisons[base][name] = dict(delta_AEE=a-b, relative=a/b-1, per_sequence_delta=seqdiff)
    save(out/'RESULTS_V1.json', dict(status='COMPLETE_FIT_ONLY_EXPLORATORY_LOSO',
         source_sha256=sha(__file__), quality_source_sha256=sha(QUALITY), ids=d['ids'],
         results=results, comparisons=comparisons, new_GPU_forwards=0, training_backbone=False,
         new_independent_confirmation=False, hardware_cost_not_in_this_file=True))
    save(out/'EXECUTION_RECEIPT_V1.json', dict(status='COMPLETE',
         utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), wall_seconds=time.monotonic()-start,
         files={p.name:sha(p) for p in out.iterdir()}, new_GPU_forwards=0))
    for name, result in results.items():
        o = result['summaries']['overall']
        print(name, o['regions']['all_valid']['AEE'], o['pixel_weighted_regret_vs_oracle6'], flush=True)


if __name__ == '__main__':
    main()
