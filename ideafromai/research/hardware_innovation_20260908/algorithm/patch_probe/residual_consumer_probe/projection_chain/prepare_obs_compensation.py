"""Fixed-mask group-OBS compensation on the existing four anchor captures.

CPU only. Decode actual theta*g over allT10; each residual source has2560
vectors (4frames x10times x64anchors), not complete training-frame tensors.
The masks, original W, P=U diag(fixedBNgain), rank32 and source theta are fixed.
For a prescribed selected-column change dG, minimize
    ||d X_centered||_F^2/N + ridge ||d||_F^2
over retained columns only. ridge=.01*mean(diag(covariance)) is this probe's
fixed choice, not a mandatory OBC constant. The solution is
    dR = -dG H_GR H_RR^-1.

This borrows the fixed-group OBS subproblem described in OBC (NeurIPS2022,
https://arxiv.org/html/2208.11580v2, sections3-4 / block sparsity), not its
complete mask search, compression framework, or network recovery. No BN bias
is changed. Free intercept -d*mean is diagnostic only; root separately runs
full-frame sequential bias correction and the fixed real-GT recovery budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
AXES = ('ordinary_rank32', 'source_group_zero', 'consumer_nullspace', 'independent_sparse_L')


def save_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert)+'\n')


def read_arrays(path):
    with np.load(path) as z:
        return {key: z[key].copy() for key in z.files}


def decode_anchors(capture, label, theta):
    matrices, frames = [], []
    for path in sorted(capture.glob('[0-9][0-9]_*.npz')):
        with np.load(path) as z:
            if str(z['split']) != 'train':
                raise ValueError('OBS initialization must use training captures only.')
            words = z[label+'_source_gate_words']
            anchor = z['anchor_mask'].astype(bool)
            selected = words.transpose(0, 2, 1)[anchor]
            if selected.shape != (64, 864) or selected.max() > 1023:
                raise ValueError('Expected64 actual anchors and864 source columns packed overT10.')
            x = ((selected[None] >> np.arange(10, dtype=np.uint16)[:, None, None]) & 1)
            x = x.reshape(640, 864).T.astype(np.float64)*float(theta)
            matrices.append(x)
            frames.append(dict(file=str(z['frame_name']), anchors=int(anchor.sum()),
                vectors=x.shape[1], scalar_entries=x.size, source_nonzeros=int(np.count_nonzero(x))))
    if len(matrices) != 4:
        raise ValueError('Exactly the existing four captured training frames are required.')
    return np.concatenate(matrices, axis=1), frames


def stationarity(delta, h, selected, retained):
    left = delta[:, retained]@h[np.ix_(retained, retained)]
    right = delta[:, selected]@h[np.ix_(selected, retained)]
    residual = left+right
    scale = np.linalg.norm(left)+np.linalg.norm(right)
    return dict(max_abs=float(np.abs(residual).max()), frobenius=float(np.linalg.norm(residual)),
                relative=float(np.linalg.norm(residual)/scale) if scale else 0.)


def local_error(base, effective, x, mean, centered, ridge):
    delta = effective-base
    error = delta@centered
    target = base@centered
    centered_total = float(np.square(error).sum()/x.shape[1])
    penalty = float(ridge*np.square(delta).sum())
    mean_error = delta@mean
    target_energy = float(np.square(target).sum())
    return dict(output_dimensions=base.shape[0], vectors=x.shape[1],
        centered_MSE=float(np.square(error).mean()), centered_RMSE=float(np.sqrt(np.square(error).mean())),
        centered_relative_RMSE=float(np.sqrt(np.square(error).sum()/target_energy)) if target_energy else None,
        mean_error=mean_error, free_intercept_diagnostic=-mean_error,
        uncentered_RMSE=float(np.sqrt(np.square(delta@x).mean())),
        mean_error_RMS=float(np.sqrt(np.square(mean_error).mean())),
        ridge_data_sum_per_vector=centered_total, ridge_penalty=penalty,
        ridge_objective=centered_total+penalty,
        objective='sum_output E[(delta*x_centered)^2] + ridge*||delta||_F^2; fixed selected-column penalty included',
        intercept_applied=False)


def kkt_small_check():
    """Compare the block formula with independent constrained KKT and LS solves."""
    x = .75*np.array([[1, 0, 1, 1, 0, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 1, 0, 0], [1, 1, 0, 0, 1, 1, 0, 1, 0]], np.float64)
    xc = x-x.mean(axis=1, keepdims=True)
    covariance = xc@xc.T/x.shape[1]
    ridge = .01*np.diag(covariance).mean()
    h = covariance+ridge*np.eye(5)
    g, r = np.array([1, 3]), np.array([0, 2, 4])
    w = np.array([[.4, -.7, .2, .3, -.8], [.1, .5, -.4, -.2, .9]], np.float64)
    dg = -w[:, g]
    delta = np.zeros_like(w)
    delta[:, g] = dg
    delta[:, r] = -dg@np.linalg.solve(h[np.ix_(r, r)], h[np.ix_(r, g)]).T
    constraint = np.eye(5)[g]
    kkt = np.block([[h, constraint.T], [constraint, np.zeros((2, 2))]])
    kkt_rhs = np.vstack((np.zeros((5, 2)), dg.T))
    solved = np.linalg.solve(kkt, kkt_rhs)[:5].T
    design = np.vstack((xc[r].T/np.sqrt(x.shape[1]), np.sqrt(ridge)*np.eye(len(r))))
    target = np.vstack((-(dg@xc[g]).T/np.sqrt(x.shape[1]), np.zeros((len(r), 2))))
    least_squares = np.linalg.lstsq(design, target, rcond=None)[0].T
    record = dict(formula_vs_full_KKT_max_abs=float(np.max(np.abs(delta-solved))),
        formula_vs_augmented_LS_retained_max_abs=float(np.max(np.abs(delta[:, r]-least_squares))),
        required_deltaG_max_abs=float(np.max(np.abs(delta[:, g]+w[:, g]))),
        selected_new_weight_max_abs=float(np.max(np.abs((w+delta)[:, g]))),
        stationarity=stationarity(delta, h, g, r))
    if max(record['formula_vs_full_KKT_max_abs'], record['formula_vs_augmented_LS_retained_max_abs'],
           record['required_deltaG_max_abs'], record['selected_new_weight_max_abs']) > 1e-11:
        raise RuntimeError('The independent KKT/LS sign and constraint check failed.')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=HERE/'capture_train4')
    parser.add_argument('--parameters', type=Path, default=HERE/'consumer_pruning_diverse10/parameters.npz')
    parser.add_argument('--output', type=Path, default=HERE/'obs_compensation_train4')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    parameters = read_arrays(args.parameters)
    captured = read_arrays(args.capture/'parameters.npz')
    output = dict(axes=np.asarray(AXES), training_files=parameters['selected_train_frames'],
        U=parameters['U'], V=parameters['V'], rank=np.array(32),
        initialization_scope=np.array('original captured W/PW, fixed-mask centered groupOBS; no BN bias updates or GT training'),
        ridge_factor=np.array(.01))
    result = dict(complete=False, parameters=str(args.parameters), capture=str(args.capture),
        prior=dict(title='Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning',
            venue='NeurIPS2022', url='https://arxiv.org/html/2208.11580v2', sections='3-4, fixed-group update / block sparsity',
            scope='Only the prescribed fixed-mask quadratic subproblem; not complete OBC, SparseGPT or network recovery.'),
        formula='deltaR=-deltaG H_GR H_RR^-1; H=XcXc^T/N + .01*mean(diag(XcXc^T/N))*I',
        ridge_scope='0.01 is this fixed probe choice, not an OBC requirement; no sweep.',
        source_scope='Each branch:4 training frames x10times x64 captured actual anchors =2560 vectors; original theta*g, C/kh/kw columns; not full training frames.',
        bias_scope='No bias changes. Free intercept -delta*mean is local diagnostic only; full-frame sequential correction is a separate root-run step.',
        nonlinear_scope='Both branches use original teacher source captures. Changing r0 may change actual r1 sources, so these local errors do not predict full-network AEE.',
        schema=dict(W_shadow='{axis}__{r0|r1}_W_shadow:Float32[96,96,3,3]',
            effective_W='{axis}__{r0|r1}_effective_W:Float32[96,96,3,3]',
            independent_L='independent_sparse_L__{r0|r1}_L and_effective_L:Float32[32,864]; selected columns exactly0',
            projected_kernel='{axis}__{r0|r1}_effective_projected_kernel:Float64[32,864], diagnostics; not extra deployed kernels in W axes',
            null_shadow='Selected shadow columns remain originalW; only retained columns receive compensation. Existing forward reapplies fixedP projection.'),
        small_check=kkt_small_check(), branches={})
    save_json(args.output/'result.json', result)
    for label in ('r0', 'r1'):
        theta = float(captured[label+'_theta'])
        x, frames = decode_anchors(args.capture, label, theta)
        if [row['file'] for row in frames] != parameters['selected_train_frames'].tolist():
            raise ValueError('OBS capture frames differ from the fixed mask-selection identities.')
        mean = x.mean(axis=1)
        xc = x-mean[:, None]
        covariance = xc@xc.T/x.shape[1]
        ridge = .01*float(np.diag(covariance).mean())
        h = covariance+ridge*np.eye(864)
        mask = np.repeat(parameters[label+'_selected_source_mask'], 9)
        g, r = np.flatnonzero(mask), np.flatnonzero(~mask)
        if len(g) != 432 or len(r) != 432:
            raise ValueError('The fixed6/12 complete C8 source mask changed.')
        hrr = h[np.ix_(r, r)]
        transfer = cho_solve(cho_factor(hrr, lower=True), h[np.ix_(r, g)]).T
        w = parameters[label+'_original_W'].astype(np.float64).reshape(96, 864)
        p = parameters[label+'_P'].astype(np.float64)
        pinv = np.linalg.pinv(p, rcond=1e-12)
        k = p@w
        output.update({label+'_P': p, label+'_selected_columns': g, label+'_retained_columns': r,
            label+'_selected_groups': parameters[label+'_selected_groups'], label+'_theta': np.array(theta),
            label+'_source_mean': mean, label+'_covariance_diagonal': np.diag(covariance),
            label+'_ridge': np.array(ridge)})
        branch = dict(theta=theta, frames=frames, vectors=x.shape[1], source_columns=x.shape[0],
            source_nonzeros=int(np.count_nonzero(x)), source_zero_fraction=float(np.mean(x == 0)),
            selected_groups=parameters[label+'_selected_groups'],
            covariance_mean_diagonal=float(np.diag(covariance).mean()), ridge=ridge,
            retained_H_min_eigenvalue=float(np.linalg.eigvalsh(hrr)[0]),
            transfer_stationarity_max_abs=float(np.max(np.abs(transfer@hrr-h[np.ix_(g, r)]))),
            axes={})
        projected_ideal, projected_exported = {}, {}
        for axis in AXES:
            base = k if axis == 'independent_sparse_L' else w
            delta = np.zeros_like(base)
            if axis == 'source_group_zero': delta[:, g] = -w[:, g]
            elif axis == 'consumer_nullspace': delta[:, g] = -pinv@k[:, g]
            elif axis == 'independent_sparse_L': delta[:, g] = -k[:, g]
            uncompensated = base+delta
            delta[:, r] = -delta[:, g]@transfer
            effective64 = base+delta
            prefix = axis+'__'+label
            shadow = w.copy()
            if axis == 'source_group_zero':
                shadow = effective64.copy()
                shadow[:, g] = 0
            elif axis == 'consumer_nullspace':
                shadow[:, r] = effective64[:, r]
            shadow32 = shadow.astype(np.float32)
            effective_w32 = shadow32.copy()
            if axis == 'consumer_nullspace':
                selected = shadow32[:, g].astype(np.float64)
                effective_w32[:, g] = (selected-pinv@(p@selected)).astype(np.float32)
            output[prefix+'_W_shadow'] = shadow32.reshape(96, 96, 3, 3)
            output[prefix+'_effective_W'] = effective_w32.reshape(96, 96, 3, 3)
            if axis == 'independent_sparse_L':
                effective32 = effective64.astype(np.float32)
                effective32[:, g] = 0
                output[prefix+'_L'] = effective32
                output[prefix+'_effective_L'] = effective32
                projected_ideal[axis] = effective64
                projected_exported[axis] = effective32.astype(np.float64)
            else:
                effective32 = effective_w32
                projected_ideal[axis] = p@effective64
                projected_exported[axis] = p@effective_w32.astype(np.float64)
            exported = effective32.astype(np.float64)
            output[prefix+'_effective_projected_kernel'] = projected_exported[axis]
            output[prefix+'_free_intercept_diagnostic'] = -(exported-base)@mean
            before = local_error(base, uncompensated, x, mean, xc, ridge)
            after = local_error(base, effective64, x, mean, xc, ridge)
            rounded = local_error(base, exported, x, mean, xc, ridge)
            record = dict(output_space='P-projected32 channels' if axis == 'independent_sparse_L' else 'rawConv2 96 channels',
                uncompensated=before, compensated_Float64=after, exported_Float32=rounded,
                centered_MSE_ratio_after_before=after['centered_MSE']/before['centered_MSE'] if before['centered_MSE'] else None,
                stationarity_Float64=stationarity(delta, h, g, r),
                stationarity_exported_Float32=stationarity(exported-base, h, g, r),
                retained_delta_l2=float(np.linalg.norm(delta[:, r])),
                original_coefficient_l2=float(np.linalg.norm(base)), original_max_abs=float(np.max(np.abs(base))),
                effective_coefficient_l2=float(np.linalg.norm(exported)), effective_max_abs=float(np.max(np.abs(exported))),
                coefficient_l2_ratio=float(np.linalg.norm(exported)/np.linalg.norm(base)),
                selected_shadow_original_W_max_abs=float(np.max(np.abs(shadow32[:, g].astype(np.float64)-w[:, g]))),
                selected_effective_W_nonzeros=int(np.count_nonzero(effective_w32[:, g])),
                selected_projected_kernel_max_abs_Float64=float(np.max(np.abs(projected_ideal[axis][:, g]))),
                selected_projected_kernel_max_abs_exported=float(np.max(np.abs(projected_exported[axis][:, g]))),
                parameter_count_change_from_same_axis=0,
                independent_L_extra_effective_coefficients=int(32*len(r)) if axis == 'independent_sparse_L' else 0,
                BN_bias_modified=False)
            if axis != 'independent_sparse_L':
                record['fixed_BN_centered_RMSE'] = float(np.sqrt(np.square(
                    parameters[label+'_bn_gain'][:, None]*((exported-w)@xc)).mean()))
            record['projected_centered_RMSE'] = float(np.sqrt(np.square((projected_exported[axis]-k)@xc).mean()))
            if after['ridge_objective'] > before['ridge_objective']+1e-10*max(1., before['ridge_objective']):
                raise RuntimeError('The fixed-mask quadratic objective increased: '+label+'/'+axis)
            branch['axes'][axis] = record
            print('OBS_LOCAL', label, axis, json.dumps(dict(before_RMSE=before['centered_RMSE'],
                after_RMSE=after['centered_RMSE'], coefficient_norm_ratio=record['coefficient_l2_ratio'])), flush=True)
        candidates = AXES[1:]
        branch['three_axis_projected_kernel_relation'] = dict(
            explanation='Same real local K after compensation: P*dWzero_G=P*dWnull_G=dL_G, and retained-column solve commutes with P. Gate-producing W and future actual source changes remain different.',
            Float64_max_pairwise_abs=max(float(np.max(np.abs(projected_ideal[a]-projected_ideal[b])))
                                        for a in candidates for b in candidates),
            exported_max_pairwise_abs=max(float(np.max(np.abs(projected_exported[a]-projected_exported[b])))
                                         for a in candidates for b in candidates))
        result['branches'][label] = branch
        save_json(args.output/'result.json', result)
    np.savez_compressed(args.output/'initial_parameters.npz', **output)
    result.update(complete=True, initial_parameters='initial_parameters.npz', wall_seconds=time.monotonic()-started,
        next_step='Full-frame train4 sequential bias correction and fixed64-step GT recovery are separate tasks; no AEE or network-function recovery is established here.')
    save_json(args.output/'result.json', result)
    print('DONE', args.output/'initial_parameters.npz', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
