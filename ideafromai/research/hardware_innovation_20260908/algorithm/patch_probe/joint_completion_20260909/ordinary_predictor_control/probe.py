"""Ordinary prefix ridge predictor, fixed JOINT students, train16/valid4 only.

This is a stronger ordinary control, not a new mechanism. Fit each H from its
three observed FP32 norm1 columns to the ten margins of the SAME saved student.
Use centered ridge with lambda = 1e-3 * trace(Cpp) / 3, without validation tuning.
Rows already fully determined by the prefix retain their exact expression.
Runtime coefficients/intercept/radius are FP32; fit and covariance are Float64.
Future Y is used only for train targets/residual calibration and exact fallback.

Source words are the captured P4 OR. Request counts are exact for this group
envelope, without W-zero filtering, lane cancellation, finite ports or cycles.
Ridge is not an A-prefix partial sum: failure must recompute A or pay a prefix
correction. Both Y-stored fallback and a favorable retained-value bound are
reported, with scalar FMAs (not compiled CMVM service).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
JOINT = HERE.parent
PATCH = JOINT.parent
BITS = 1 << np.arange(10, dtype=np.int64)
MODELS = {
    'row34_optimized_packed': JOINT/'optimized_prefix_train16/row34_packed_word.npz',
    'common3_group': JOINT/'local_train16/common3_group.npz',
}


def load_data():
    capture = torch.load(PATCH/'partial_completion/capture.pt', map_location='cpu', weights_only=False)
    out = {}
    for split in ('train', 'valid'):
        rows = [r for r in capture['samples'] if r['split'] == split]
        y = torch.stack([r['Y'] for r in rows]).permute(0, 3, 2, 4, 1).contiguous()
        y = y.reshape(len(rows), 64, 12, 8, 4, 10).reshape(-1, 8, 4, 10)
        words = np.stack([r['source_words'].numpy() for r in rows]).reshape(-1, 864).astype(np.uint16)
        hist = np.stack([np.bincount(w, minlength=1024) for w in words])
        visits = np.stack([np.count_nonzero(words & b, axis=1) for b in BITS], axis=1)
        channels = (torch.arange(len(y)) % 12)[:, None]*8 + torch.arange(8)[None]
        out[split] = dict(y=y, channels=channels, hist=np.repeat(hist, 12, axis=0),
                          visits=np.repeat(visits, 12, axis=0), files=[r['file'] for r in rows])
    return out, capture['metadata']


def by_channel(values):
    return values.reshape(-1, 12, 8, 4, 10).permute(1, 2, 0, 3, 4).reshape(96, -1, 10)


def load_model(path):
    with np.load(path) as f:
        d = {k: f[k].copy() for k in f.files}
    a = torch.from_numpy(d['temporal_q14'].astype(np.float32)/16384)
    b = torch.from_numpy(d['temporal_bias'].astype(np.float32))
    known = torch.zeros(10, dtype=torch.float32)
    known[d['prefix'].tolist()] = 1
    remaining = a*(1-known)
    mean = torch.from_numpy(d['mean'].astype(np.float32))
    cov = torch.from_numpy(d['covariance'].astype(np.float32))
    gamma = torch.from_numpy(d['gamma'].astype(np.float32))
    offset = mean@remaining.T + b-float(d['theta_output'])
    variance = torch.einsum('ts,csu,tu->ct', remaining, cov, remaining)
    radius = variance.clamp_min(1e-6).sqrt()*gamma
    return dict(raw=d, a=a, b=b, theta=float(d['theta_output']), gamma=gamma,
                known=known, remaining=remaining, offset=offset, radius=radius,
                prefix=d['prefix'].tolist(), exact_rows=~remaining.ne(0).any(1))


def original_values(model, y, channels):
    return (y*model['known'])@model['a'].T + model['offset'][channels][:, :, None, :]


def ridge_values(table, model, y, channels):
    pred = torch.einsum('bhpk,bhkt->bhpt', y[..., model['prefix']], table['beta'][channels])
    pred = pred + table['intercept'][channels][:, :, None, :]
    if model['exact_rows'].any():
        exact = original_values(model, y, channels)
        pred[..., model['exact_rows']] = exact[..., model['exact_rows']]
    return pred


@torch.no_grad()
def fit_ridge(model, data):
    y = by_channel(data['y']).numpy().astype(np.float64)
    x = y[..., model['prefix']]
    # Targets are the actual Float32 complete student, not a different A or integer Y.
    full = data['y']@model['a'].T + model['b']-model['theta']
    target = by_channel(full).numpy().astype(np.float64)
    mx, mm = x.mean(1), target.mean(1)
    xc, mc = x-mx[:, None], target-mm[:, None]
    cpp = np.einsum('hni,hnj->hij', xc, xc)/x.shape[1]
    cpm = np.einsum('hni,hnt->hit', xc, mc)/x.shape[1]
    lam = 1e-3*np.trace(cpp, axis1=1, axis2=2)/3
    beta = np.zeros((96, 3, 10), np.float64)
    for h in range(96):
        if lam[h] > 0:
            beta[h] = np.linalg.solve(cpp[h]+lam[h]*np.eye(3), cpm[h])
    intercept = mm-np.einsum('hi,hit->ht', mx, beta)
    # Standard exact-prefix bypass, based only on A's structural dependencies.
    exact = model['exact_rows'].numpy()
    beta[:, :, exact] = model['a'].numpy()[exact][:, model['prefix']].T[None]
    intercept[:, exact] = (model['b'].numpy()-model['theta'])[exact]
    table = dict(beta=torch.from_numpy(beta.astype(np.float32)),
                 intercept=torch.from_numpy(intercept.astype(np.float32)))
    pred = ridge_values(table, model, data['y'], data['channels'])
    residual = by_channel(full-pred).numpy().astype(np.float64)
    residual_mean = residual.mean(1)
    sigma = np.sqrt(np.mean((residual-residual_mean[:, None])**2, axis=1))
    radius = np.maximum(sigma, 1e-3)*model['gamma'].numpy()[None]
    radius[:, exact] = 0
    table['radius'] = torch.from_numpy(radius.astype(np.float32))
    table['sigma'] = sigma
    table['residual_mean'] = residual_mean
    table['lambda'] = lam
    table['cpp_eigenvalues'] = np.linalg.eigvalsh(cpp)
    return table


def numerical_costs(model, table, contexts, predictor):
    a = model['a'].numpy()
    original_n = int(np.count_nonzero(a[:, model['prefix']]))
    if predictor == 'original':
        per_h = np.full(96, original_n, dtype=np.int64)
        additional = 0
    else:
        per_h = np.count_nonzero(table['beta'].numpy(), axis=(1, 2))
        additional = int((~model['exact_rows']).sum())*96*3*4
    samples_per_h = contexts//12*4
    undecided_rows = int((~model['exact_rows']).sum())
    return dict(prediction_fmas=int(per_h.sum()*samples_per_h),
                prediction_fmas_per_p_h_min=int(per_h.min()),
                prediction_fmas_per_p_h_max=int(per_h.max()),
                intercept_initializations=contexts*8*4*10,
                abs_radius_compares=contexts*8*4*undecided_rows,
                additional_runtime_coefficient_bytes=additional,
                compact_offset_radius_bytes=96*undecided_rows*2*4,
                runtime_table_bytes_excluding_existing_A=additional+96*undecided_rows*2*4,
                table_format='FP32; exact-prefix rows reuse existing shared A/b and need no fitted table',
                arithmetic_label='scalar nonzero FMA count; no CMVM/CSE, RF, ports, or cycle claim')


@torch.no_grad()
def measure(model, table, data, predictor):
    a = model['a']
    totals = dict(gates=0, own_positive=0, own_negative=0, wrong=0, fn=0, fp=0,
                  accepted=0, accepted_wrong=0, needed_columns=0,
                  source_column_visits=0, full_source_column_visits=0,
                  prefix_w_requests=0, tail_w_requests=0, full_one_scan_w_requests=0,
                  fallback_full_A_fmas=0, retained_prediction_fallback_fmas=0,
                  prediction_squared_error=0.)
    need_all, gate_all, accept_all = [], [], []
    prefix_word = int(BITS[model['prefix']].sum())
    words = np.arange(1024)
    exact = model['exact_rows']
    a_n = a.ne(0).sum(1).numpy()
    tail_n = model['remaining'].ne(0).sum(1).numpy()
    # A ridge prediction can be corrected to the exact margin; those coefficients are not free.
    correction = model['a'].numpy()[:, model['prefix']].T[None]-table['beta'].numpy()
    correction_n = np.count_nonzero(correction, axis=1)
    correction_n[:, exact.numpy()] = 0
    retained_n = np.minimum(a_n[None], correction_n+tail_n[None])
    radius = model['radius'] if predictor == 'original' else table['radius']
    for start in range(0, len(data['y']), 128):
        sl = slice(start, start+128)
        y, channels = data['y'][sl], data['channels'][sl]
        full = y@a.T + model['b']-model['theta']
        pred = (original_values(model, y, channels) if predictor == 'original'
                else ridge_values(table, model, y, channels))
        accept = pred.abs() >= radius[channels][:, :, None, :]
        gate = torch.where(accept, pred >= 0, full >= 0)
        own = full >= 0
        wrong = gate != own
        pending = ~accept
        by_lane = (pending[..., None] & a.ne(0)).any(-2)
        need = by_lane.any(dim=(1, 2)) | model['known'].bool()
        tail = need & ~model['known'].bool()
        tail_words = tail.numpy().astype(np.int64)@BITS
        hist = data['hist'][sl]
        totals['gates'] += gate.numel()
        totals['own_positive'] += int(own.sum())
        totals['own_negative'] += int((~own).sum())
        totals['wrong'] += int(wrong.sum())
        totals['fn'] += int((~gate & own).sum())
        totals['fp'] += int((gate & ~own).sum())
        totals['accepted'] += int(accept.sum())
        totals['accepted_wrong'] += int((accept & wrong).sum())
        totals['needed_columns'] += int(need.sum())
        totals['source_column_visits'] += int((data['visits'][sl]*need.numpy()).sum())
        totals['full_source_column_visits'] += int(data['visits'][sl].sum())
        totals['prefix_w_requests'] += int(hist[:, (words & prefix_word) != 0].sum())
        totals['tail_w_requests'] += int((hist*((tail_words[:, None] & words[None]) != 0)).sum())
        totals['full_one_scan_w_requests'] += int(hist[:, 1:].sum())
        pending_np = pending.numpy()
        totals['fallback_full_A_fmas'] += int((pending_np*a_n).sum())
        if predictor == 'original':
            totals['retained_prediction_fallback_fmas'] += int((pending_np*tail_n).sum())
        else:
            totals['retained_prediction_fallback_fmas'] += int((pending_np*retained_n[channels.numpy()][:, :, None]).sum())
        totals['prediction_squared_error'] += float((pred.double()-full.double()).square().sum())
        need_all.append(need.numpy()); gate_all.append(gate.numpy()); accept_all.append(accept.numpy())
    totals['two_stage_w_requests'] = totals['prefix_w_requests']+totals['tail_w_requests']
    totals.update(gate_error=totals['wrong']/totals['gates'],
                  false_negative_rate=totals['fn']/max(1, totals['own_positive']),
                  false_positive_rate=totals['fp']/max(1, totals['own_negative']),
                  accepted_fraction=totals['accepted']/totals['gates'],
                  accepted_error=totals['accepted_wrong']/max(1, totals['accepted']),
                  source_column_ratio=totals['source_column_visits']/totals['full_source_column_visits'],
                  two_stage_w_ratio=totals['two_stage_w_requests']/totals['full_one_scan_w_requests'],
                  prediction_mse=totals['prediction_squared_error']/totals['gates'])
    costs = numerical_costs(model, table, len(data['y']), predictor)
    costs['with_Y_stored_full_row_fallback_fmas'] = costs['prediction_fmas']+totals['fallback_full_A_fmas']
    costs['retained_value_arithmetic_lower_bound_fmas'] = costs['prediction_fmas']+totals['retained_prediction_fallback_fmas']
    costs['full_student_A_fmas'] = totals['gates']//10*int(a.ne(0).sum())
    costs['retained_value_warning'] = 'Favorable arithmetic bound requires retained per-gate values; state is not free. Ridge uses min(full A row, corrected prediction).'
    totals['costs'] = costs
    return totals, dict(need_column=np.concatenate(need_all), gate=np.concatenate(gate_all), accepted=np.concatenate(accept_all))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    data, metadata = load_data()
    result = dict(scope='Fixed FP32 Conv1/norm1 students; real train16/valid4, 64 native P4, C96/T10; no AEE/GPU claim',
                  source='partial_completion/capture.pt; P4 OR source_words; no integer Yi or unseen validation labels in fit',
                  fit=dict(method='centered per-H ridge to ten complete student margins',
                           regularizer='1e-3 * trace(Cpp) / 3; one fixed choice',
                           training_observations_per_H=16*64*4,
                           residual_std='train residual after FP32 coefficient/intercept evaluation; centered std, floor 1e-3',
                           acceptance='abs(predicted margin) >= original gamma[t] * residual_std[h,t]',
                           exact_prefix_rows='retain original exact coefficients; no ridge uncertainty for already known outputs',
                           numeric='Float64 fit, FP32 coefficient/offset/radius and runtime; same Float32 full-student truth'),
                  train_files=data['train']['files'], valid_files=data['valid']['files'], axes={})
    for name, path in MODELS.items():
        model = load_model(path)
        table = fit_ridge(model, data['train'])
        axis = dict(model_file=str(path), prefix=model['prefix'], theta_output=model['theta'],
                    gamma=model['gamma'].tolist(), exact_prefix_rows=model['exact_rows'].nonzero().flatten().tolist(),
                    lambda_min=float(table['lambda'].min()), lambda_max=float(table['lambda'].max()),
                    fitted_sigma_mean=float(table['sigma'][:, ~model['exact_rows'].numpy()].mean()),
                    original_sigma_mean=float((model['radius']/model['gamma']).numpy()[:, ~model['exact_rows'].numpy()].mean()),
                    metrics={})
        np.savez_compressed(args.output/(name+'_ridge_table.npz'),
                            coefficient=table['beta'].numpy(), intercept=table['intercept'].numpy(), radius=table['radius'].numpy(),
                            sigma=table['sigma'], residual_mean=table['residual_mean'], ridge_lambda=table['lambda'],
                            cpp_eigenvalues=table['cpp_eigenvalues'], prefix=np.array(model['prefix']),
                            exact_prefix_rows=model['exact_rows'].numpy(), temporal_q14=model['raw']['temporal_q14'],
                            temporal_bias=model['raw']['temporal_bias'], theta_output=np.array(model['theta']), gamma=model['gamma'].numpy())
        for predictor in ('original', 'ridge'):
            axis['metrics'][predictor] = {}
            for split in ('train', 'valid'):
                metrics, decisions = measure(model, table, data[split], predictor)
                axis['metrics'][predictor][split] = metrics
                if split == 'valid':
                    np.savez_compressed(args.output/(name+'_'+predictor+'_valid_decisions.npz'), **decisions)
                    if predictor == 'original':
                        old_path = path.with_name(path.stem+'_valid_decisions.npz')
                        with np.load(old_path) as old:
                            axis['original_reproduction'] = {k+'_mismatches': int(np.count_nonzero(decisions[k] != old[k])) for k in ('gate', 'need_column')}
                print(name, predictor, split, json.dumps({k:metrics[k] for k in ('wrong','fn','gate_error','source_column_ratio','two_stage_w_ratio')}), flush=True)
        old, new = axis['metrics']['original']['valid'], axis['metrics']['ridge']['valid']
        axis['valid_ridge_vs_original'] = {k+'_ratio': new[k]/old[k] for k in ('source_column_visits','two_stage_w_requests')}
        axis['valid_ridge_vs_original']['prediction_fma_ratio'] = new['costs']['prediction_fmas']/old['costs']['prediction_fmas']
        result['axes'][name] = axis
        (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    result['wall_seconds'] = time.monotonic()-started
    result['complete'] = True
    (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('DONE', result['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
