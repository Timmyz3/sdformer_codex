#!/usr/bin/env python3
"""Fit-only R21 pilot, frozen decisions before any development quality is opened.

Input manifest contains native arrays for 16 fit + 8 development members; only fit
members have GT/mask references and signed-gain labels. Never reads confirm data.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import structure_models as m

HERE = Path(__file__).resolve().parent
G0 = HERE.parents[1]/'ordinary_controls_round18/implementation/g0_core.py'
G0_SHA = None  # Explicitly supplied by the frozen manifest, validated before import.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def checked(spec):
    m.require(set(spec) == {'path', 'sha256'}, 'Explicit source path/SHA')
    p = Path(spec['path'])
    m.require(p.is_file() and sha(p) == spec['sha256'], 'Actual bytes differ '+str(p))
    return p


def serial(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, dict): return {k: serial(w) for k, w in v.items()}
    if isinstance(v, (tuple, list)): return [serial(w) for w in v]
    return v


def write(path, value):
    Path(path).write_text(json.dumps(serial(value), indent=2, allow_nan=False)+'\n')


def motion_controls(q):
    r = q-m.pool_expand3(q[None])[0]
    out = {k: [] for k in ('motion_magnitude', 'motion_pool_residual', 'motion_neighbor_difference')}
    for y, x in m.TILES:
        a, b = q[9*y:9*y+9, 9*x:9*x+9], r[9*y:9*y+9, 9*x:9*x+9]
        out['motion_magnitude'].append(float(np.linalg.norm(a, axis=-1).mean()))
        out['motion_pool_residual'].append(float(np.square(b).sum(-1).mean()))
        out['motion_neighbor_difference'].append(float((np.linalg.norm(a[1:]-a[:-1], axis=-1).mean()+
                                                        np.linalg.norm(a[:, 1:]-a[:, :-1], axis=-1).mean())/2))
    return out


def fit_pca2(x):
    a = x.reshape(-1, 384).astype(np.float64)
    mean, std = a.mean(0), a.std(0)
    std = np.where(std > 1e-6, std, 1.)
    z = (a-mean)/std
    values, vec = np.linalg.eigh(z.T@z/len(z))
    w = vec[:, np.argsort(values)[::-1][:2]]
    for i in range(2):
        if w[np.argmax(np.abs(w[:, i])), i] < 0: w[:, i] *= -1
    w = w/std[:, None]
    return dict(weight=w.astype(np.float32), bias=(-mean@w).astype(np.float32),
                source='fit-only ordinary PCA2; squared residual norm invariant to orthogonal basis rotation within a fixed subspace')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    m.require(not a.output.exists() and sha(a.manifest) == a.manifest_sha256, 'Fresh output and pinned input')
    inp = json.loads(a.manifest.read_text())
    m.require(inp['schema'] == 'flow_r21_fit_only_inputs_v1', 'Explicit fit-only schema')
    for spec in inp['frozen_sources']: checked(spec)
    m.require(any(Path(s['path']).resolve() == Path(__file__).resolve() for s in inp['frozen_sources']), 'Actual entry frozen')
    m.require(any(Path(s['path']).resolve() == Path(m.__file__).resolve() for s in inp['frozen_sources']), 'Actual math frozen')
    roles = json.loads(checked(inp['roles']).read_text())
    audit = json.loads(checked(inp['execution_audit']).read_text())
    m.require(audit['status'] == 'PASS' and audit['quality_analysis_allowed'], 'Actual execution must be independently accepted')
    result_path = checked(inp['execution_result'])
    result_sha = sha(result_path)
    execution = json.loads(result_path.read_text())
    m.require(audit['source_result_sha256'] == result_sha and execution['graph_id'] == inp['graph_id'], 'Audited actual graph')
    native_records = {r['target_id']: r for r in execution['forward_records'] if r['arm'] == 'native'}
    records = inp['members']
    fit = [r for r in records if r['role'] == 'fit']
    dev = [r for r in records if r['role'] == 'development']
    m.require(len(records) == 24 and len(fit) == 16 and len(dev) == 8, 'Only fixed16/8, no confirm')
    for group, rows in (('fit', fit), ('development', dev)):
        expected = [r['target_id'] for r in roles['groups'][group]]
        m.require([r['target_id'] for r in rows] == expected, 'Predeclared membership/order '+group)
    m.require(all('fit_GT' not in r and 'fit_mask' not in r and 'gain' not in r for r in dev), 'Development cannot carry labels')
    labels = json.loads(checked(inp['fit_gain_labels']).read_text())
    m.require(labels['schema'] == 'flow_r21_fit_gain_labels_v1' and labels['background'] == inp['background'], 'Own-graph fit labels')
    m.require(labels['source_result_sha256'] == result_sha and labels['execution_audit_sha256'] == sha(checked(inp['execution_audit']))
              and labels['graph_id'] == inp['graph_id'] and labels['tile_order'] == [list(t) for t in m.TILES], 'Fit labels bound to this exact graph/audit/order')
    m.require(set(labels['signed_gain_by_id']) == {r['target_id'] for r in fit}, 'Exactly fit labels; never development')
    gpath = checked(inp['ordinary_score_source'])
    spec = importlib.util.spec_from_file_location('_r21_ordinary_g0', gpath)
    g0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g0)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    m.require(not torch.cuda.is_initialized(), 'CPU fit only')
    start = time.monotonic()
    a.output.mkdir(parents=True, exist_ok=False)
    prefix, ordinary, source_receipts, truths, counts = {}, {}, {}, {}, {}
    for row in records:
        tid = row['target_id']; path = checked(row['native_archive'])
        record = native_records[tid]
        m.require(path.name == record['prediction_file'] and sha(path) == record['prediction_file_sha256'], 'Native archive is an actual audited record')
        with np.load(path, allow_pickle=False) as z:
            u, sn1 = z['u'], z['sn1']
            m.require(u.shape == sn1.shape == (10, 1, 30, 40, 384), 'Native original-grid tensors')
            m.require(u.dtype == sn1.dtype == np.float32 and np.isfinite(u).all(), 'Finite FP32 prefix')
            m.require(hashlib.sha256(np.ascontiguousarray(u).tobytes()).hexdigest() == record['tensor_records']['u']['sha256'] and
                      hashlib.sha256(np.ascontiguousarray(sn1).tobytes()).hexdigest() == record['tensor_records']['sn1']['sha256'], 'Actual predecision payloads')
            prefix[tid] = u[:, 0].mean(0, dtype=np.float32)
            ordinary[tid] = g0.score_frame(u, sn1)['scores']
        source_receipts[tid] = dict(native_archive_sha256=sha(path),
                                   time_mean_payload_sha256=hashlib.sha256(prefix[tid].tobytes()).hexdigest())
        if row['role'] == 'fit':
            m.require(row['fit_GT']['sha256'] == record['input_evidence']['gt'] and
                      row['fit_mask']['sha256'] == record['input_evidence']['mask'], 'Fit labels use the actual audited original GT/mask')
            gt = np.load(checked(row['fit_GT']), allow_pickle=False)
            mask = np.load(checked(row['fit_mask']), allow_pickle=False)
            truths[tid], counts[tid] = m.pixel_labels(gt, mask)
    fit_ids, dev_ids = [r['target_id'] for r in fit], [r['target_id'] for r in dev]
    sequence_by_id = {r['target_id']: r['sequence'] for r in records}
    sequences = sorted({r['sequence'] for r in fit})
    m.require(len(sequences) == 8 and all(sum(r['sequence'] == s for r in fit) == 2 for s in sequences), 'Eight2-member folds')
    def fit_q(ids):
        return m.ridge_motion(np.stack([prefix[t] for t in ids]), np.stack([truths[t] for t in ids]), np.stack([counts[t] for t in ids]))
    oof, folds = {}, []
    for seq in sequences:
        train = [t for t in fit_ids if sequence_by_id[t] != seq]
        hold = [t for t in fit_ids if sequence_by_id[t] == seq]
        model = fit_q(train)
        m.require(len(train) == 14 and len(hold) == 2 and not set(train)&set(hold), 'No same-sequence q fitting')
        for tid in hold: oof[tid] = m.project(model, prefix[tid])
        folds.append(dict(held_sequence=seq, train_ids=train, held_ids=hold, projection=model))
    final_q = fit_q(fit_ids)
    dev_q = {t: m.project(final_q, prefix[t]) for t in dev_ids}
    train_inputs = m.spatial_inputs(np.stack([prefix[t] for t in fit_ids]), np.stack([oof[t] for t in fit_ids]))
    dev_inputs = m.spatial_inputs(np.stack([prefix[t] for t in dev_ids]), np.stack([dev_q[t] for t in dev_ids]))
    y = np.asarray([labels['signed_gain_by_id'][t] for t in fit_ids], np.float32)
    m.require(y.shape == (16, 12) and np.isfinite(y).all(), 'Own-graph relative signed gains')
    all_scores, train_records = {}, {}
    shared_scalers = []
    for kind, field in (('generic_raw', 'raw'), ('generic_group', 'ordinary'), ('same_motion', 'motion'), ('structured', 'motion')):
        fitted = m.fit_predictor(kind, train_inputs[field], train_inputs['context'], y.reshape(-1))
        score = m.infer_predictor(fitted, dev_inputs[field], dev_inputs['context']).reshape(8, 12)
        all_scores[kind] = {t: score[i] for i, t in enumerate(dev_ids)}
        modelpath = a.output/(kind+'_state.pt')
        torch.save(fitted['model'].state_dict(), modelpath)
        train_records[kind] = {k: v for k, v in fitted.items() if k != 'model'}
        train_records[kind].update(state_path=modelpath.name, state_file_sha256=sha(modelpath), state_payload_sha256=m.state_sha(fitted['model']))
        if kind in ('same_motion', 'structured'):
            shared_scalers.append((fitted['spatial_scaler'], fitted['context_scaler']))
        print(json.dumps(dict(kind=kind, steps=fitted['steps'], parameters=fitted['parameter_count'], fit_loss=fitted['fit_loss'])), flush=True)
    for (xs, cs), (xt, ct) in zip(shared_scalers, shared_scalers[1:]):
        m.require(all(np.array_equal(xs[k], xt[k]) and np.array_equal(cs[k], ct[k]) for k in ('mean', 'std')), 'Exactly same G2/F scales')
    for name in g0.SCORES:
        all_scores['G0_'+name] = {t: np.asarray([v[name] for v in ordinary[t]]) for t in dev_ids}
    for tid in dev_ids:
        for name, scores in motion_controls(dev_q[tid]).items():
            all_scores.setdefault(name, {})[tid] = scores
    pca = fit_pca2(np.stack([prefix[t] for t in fit_ids]))
    all_scores['PCA2_pool_residual'] = {t: motion_controls(m.project(pca, prefix[t]))['motion_pool_residual'] for t in dev_ids}
    prior = y.mean(0)
    all_scores['fit_position_prior'] = {t: prior.copy() for t in dev_ids}
    decisions = {name: {t: {'selected_index': m.top1(scores[t]), 'tile': list(m.TILES[m.top1(scores[t])])} for t in dev_ids}
                 for name, scores in all_scores.items()}
    result = dict(schema='flow_r21_frozen_dev_predictions_v1', status='FROZEN_BEFORE_DEVELOPMENT_QUALITY',
                  inputs_sha256=a.manifest_sha256, background=inp['background'], fit_ids=fit_ids, development_ids=dev_ids,
                  q_folds=folds, final_q_projection=final_q, ordinary_PCA2=pca, training=train_records,
                  scores=all_scores, decisions=decisions, source_receipts=source_receipts,
                  exact_tie_rule='first row-major maximum; no epsilon', CUDA_initialized=torch.cuda.is_initialized(),
                  projection_fit_valid_pixel_counts={t: int(counts[t].sum()) for t in fit_ids},
                  quality_metrics_read_roles=['fit'], development_GT_paths_present=False,
                  wall_seconds=time.monotonic()-start, claim='Exploratory development predictions only; no result or hardware advantage yet')
    write(a.output/'predictions.json', result)
    print('FROZEN '+sha(a.output/'predictions.json'), flush=True)


if __name__ == '__main__': main()
