#!/usr/bin/env python3
"""R22: fit-only original-grid motion representation; no allocation/quality claim.

Native input collection is independent of whether a compressed action is feasible.
Development GT paths are forbidden; export predictions before external evaluation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
CORE = HERE.parents[1]/'motion_structure_round21/design/structure_models.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def checked(spec):
    require(set(spec) == {'path', 'sha256'}, 'Pinned path/SHA')
    p = Path(spec['path'])
    require(p.is_file() and sha(p) == spec['sha256'], 'Actual asset bytes differ '+str(p))
    return p


def serial(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k: serial(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [serial(v) for v in x]
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--inputs-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists() and sha(args.inputs) == args.inputs_sha256, 'Fresh output/pinned manifest')
    inp = json.loads(args.inputs.read_text())
    require(inp['schema'] == 'flow_r22_motion_projection_fit_inputs_v1', 'Fit-only motion-proxy schema')
    for pin in inp['source_pins']: checked(pin)
    require(any(Path(q['path']).resolve() == Path(__file__).resolve() for q in inp['source_pins']), 'Freeze entry')
    require(any(Path(q['path']).resolve() == CORE.resolve() for q in inp['source_pins']), 'Freeze exact shared math')
    spec = importlib.util.spec_from_file_location('_r22_frozen_structure_math', CORE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    import torch
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), 'CPU projection fitting only')
    roles_path = checked(inp['roles'])
    require(sha(roles_path) == 'e6f216f6b017fc2cbba262ec51c70474f342e593e1549058a2200970949e10a1', 'Original frozen32 identity roles')
    roles = json.loads(roles_path.read_text())
    audited_path = checked(inp['execution_result'])
    execution = json.loads(audited_path.read_text())
    audit_path = checked(inp['execution_audit'])
    audit = json.loads(audit_path.read_text())
    require(audit['schema'] == 'flow_r22_representation_independent_execution_audit_v1' and
            audit['status'] == 'PASS' and audit['representation_artifacts_allowed'] and
            not audit['quality_analysis_allowed'] and audit['source_result_sha256'] == sha(audited_path),
            'Independent native-representation admission; development quality remains sealed')
    require(execution['graph_id'] == inp['graph_id'] == 'r22_W271482_all78current_native_fit16dev8' and
            execution['status'] == 'REPRESENTATION_NATIVE26_COMPLETE' and execution['full_forward_count'] == 26 and
            execution['binding']['selector_roles_sha256'] == sha(roles_path), 'Actual complete native26 graph and original role binding')
    records = inp['members']
    fit = [r for r in records if r['role'] == 'fit']
    dev = [r for r in records if r['role'] == 'development']
    require(len(records) == 24 and len(fit) == 16 and len(dev) == 8, 'Only16fit/8dev')
    for role, subset in (('fit', fit), ('development', dev)):
        require([r['target_id'] for r in subset] == [r['target_id'] for r in roles['groups'][role]], 'Original identity/order '+role)
        for row, canonical in zip(subset, roles['groups'][role]):
            require(row['sequence'] == canonical['sequence'], 'LOO sequence must be the canonical sequence')
    require(all('fit_GT' not in r and 'fit_mask' not in r for r in dev), 'No development GT/mask references')
    native_list = [r for r in execution['forward_records'] if r['arm'] == 'native']
    native = {r['target_id']: r for r in native_list}
    require(len(execution['forward_records']) == 26 and len(native_list) == len(native) == 24 and
            set(native) == {r['target_id'] for r in records}, 'Exactly24 unique original native endpoints')
    for row in records:
        rec = native[row['target_id']]
        require(rec['quality_endpoint'] and rec['execution'] == 'native' and rec['selector_role'] == row['role'] and
                execution['binding']['selector_role_map'][row['target_id']] == row['role'], 'Actual endpoint identity/role')
    cache, truth, counts, evidence = {}, {}, {}, {}
    begin = time.monotonic()
    for row in records:
        tid = row['target_id']; record = native[tid]; path = checked(row['native_archive'])
        require(path.name == record['prediction_file'] and sha(path) == record['prediction_file_sha256'], 'Actual native output')
        with np.load(path, allow_pickle=False) as z:
            u = z['u']
            require(u.shape == (10, 1, 30, 40, 384) and u.dtype == np.float32 and np.isfinite(u).all(), 'Native prefix')
            require(hashlib.sha256(np.ascontiguousarray(u).tobytes()).hexdigest() == record['tensor_records']['u']['sha256'], 'Audited native prefix payload')
            cache[tid] = u[:, 0].mean(0, dtype=np.float32)
        evidence[tid] = dict(native_archive_sha256=sha(path), prefix_mean_payload_sha256=hashlib.sha256(cache[tid].tobytes()).hexdigest())
        if row['role'] == 'fit':
            require(row['fit_GT']['sha256'] == record['input_evidence']['gt'] and row['fit_mask']['sha256'] == record['input_evidence']['mask'], 'Fit-only original labels')
            truth[tid], counts[tid] = m.pixel_labels(np.load(checked(row['fit_GT']), allow_pickle=False), np.load(checked(row['fit_mask']), allow_pickle=False))
    fit_ids = [r['target_id'] for r in fit]
    dev_ids = [r['target_id'] for r in dev]
    seq = {r['target_id']: r['sequence'] for r in fit}
    sequences = sorted(set(seq.values()))
    require(len(sequences) == 8 and all(sum(s == q for s in seq.values()) == 2 for q in sequences), 'Fixed8x2 folds')
    def train(ids):
        return m.ridge_motion(np.stack([cache[t] for t in ids]), np.stack([truth[t] for t in ids]), np.stack([counts[t] for t in ids]))
    def priors(ids):
        n = np.stack([counts[t] for t in ids]).sum(0)
        s = np.stack([truth[t]*counts[t][..., None] for t in ids]).sum(0)
        global_vector = s.sum((0, 1))/n.sum()
        position = s/np.maximum(n, 1)[..., None]
        position[n == 0] = global_vector
        return {'global_vector': global_vector.astype(np.float32), 'position': position.astype(np.float32)}
    args.output.mkdir(parents=True, exist_ok=False)
    predictions, folds = {}, []
    for sequence in sequences:
        train_ids = [t for t in fit_ids if seq[t] != sequence]
        held_ids = [t for t in fit_ids if seq[t] == sequence]
        require(len(train_ids) == 14 and len(held_ids) == 2, '14-to2 OOF')
        model, ordinary = train(train_ids), priors(train_ids)
        folds.append(dict(sequence=sequence, train_ids=train_ids, held_ids=held_ids, projection=model))
        for tid in held_ids:
            predictions[tid] = dict(q=m.project(model, cache[tid]), global_vector=np.broadcast_to(ordinary['global_vector'], (30, 40, 2)).copy(), position_prior=ordinary['position'])
    final, ordinary = train(fit_ids), priors(fit_ids)
    for tid in dev_ids:
        predictions[tid] = dict(q=m.project(final, cache[tid]), global_vector=np.broadcast_to(ordinary['global_vector'], (30, 40, 2)).copy(), position_prior=ordinary['position'])
    prediction_files = []
    for row in records:
        tid = row['target_id']; vals = predictions[tid]
        vals['q_pool_expand3'] = m.pool_expand3(vals['q'][None])[0]
        for name, val in vals.items(): require(val.shape == (30, 40, 2) and val.dtype == np.float32 and np.isfinite(val).all(), 'Finite full-grid proxy '+name)
        path = args.output/(tid+'_motion.npz')
        np.savez_compressed(path, **vals)
        prediction_files.append(dict(target_id=tid, role=row['role'], file=path.name, sha256=sha(path),
                                     payloads={name:hashlib.sha256(np.ascontiguousarray(val).tobytes()).hexdigest() for name,val in vals.items()}))
    out = dict(schema='flow_r22_motion_projection_predictions_v1', status='FROZEN_BEFORE_DEVELOPMENT_QUALITY',
               inputs_sha256=args.inputs_sha256, source_result_sha256=sha(audited_path), execution_audit_sha256=sha(audit_path),
               graph_id=inp['graph_id'], fit_ids=fit_ids, development_ids=dev_ids, q_folds=folds, final_q_projection=final,
               final_ordinary_priors=ordinary, predictions=prediction_files, source_evidence=evidence,
               fit_valid_pixels={t:int(counts[t].sum()) for t in fit_ids}, device='CPU', CUDA_initialized=torch.cuda.is_initialized(),
               wall_seconds=time.monotonic()-begin, selector_training=False, original_model_training=False,
               scope='Fit-only motion representation availability; no compression allocation, new task accuracy or flow-specific advantage established')
    outpath = args.output/'predictions.json'
    outpath.write_text(json.dumps(serial(out), indent=2, allow_nan=False)+'\n')
    print('FROZEN '+sha(outpath), flush=True)


if __name__ == '__main__': main()
