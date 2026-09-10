"""Fixed rank-2 local controls from old control3 train4 moments; CPU only.

x is the saved actual proj.sn input (including the residual), not the upstream
r1 identity alone. No new network/source capture, GT training or AEE evaluation.
The ordinary coordinate basis is I_10; the shared basis is the saved source As.
Both get an exact one-to-one 1024-state assignment before fixed-P refinement.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def assignment(target, covariance, basis):
    cross = target @ covariance @ basis.T
    variance = np.einsum('ti,ij,tj->t', basis, covariance, basis)
    teacher_variance = np.einsum('ti,ij,tj->t', target, covariance, target)
    slopes = cross / variance[None, :]
    costs = teacher_variance[:, None] - cross * slopes
    dp = np.full(1024, np.inf)
    dp[0] = 0
    previous, chosen = np.full(1024, -1, int), np.full(1024, -1, int)
    for mask in range(1023):
        t = bin(mask).count('1')
        for j in range(10):
            if mask & (1 << j):
                continue
            nxt = mask | (1 << j)
            value = dp[mask] + costs[t, j]
            if value < dp[nxt]:
                dp[nxt], previous[nxt], chosen[nxt] = value, mask, j
    permutation, mask = np.zeros(10, int), 1023
    for t in range(9, -1, -1):
        permutation[t] = chosen[mask]
        mask = previous[mask]
    return permutation, slopes[np.arange(10), permutation]


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', type=Path, default=here / 'affine_shared_temporal_control_diverse10/parameters.npz')
    parser.add_argument('--output', type=Path, default=here / 'shared_temporal_lowrank_control')
    args = parser.parse_args()
    data = dict(np.load(args.parameters))
    mean = data['mean_x']
    covariance = (data['covariance_x'] + data['covariance_x'].T) / 2
    source = data['source_A']
    target = data['original_proj_A'].astype(np.float64)
    bias = data['original_proj_bias'].reshape(10).astype(np.float64)
    eigenvalues, q = np.linalg.eigh(covariance)
    assert eigenvalues.min() > 0
    root = (q * np.sqrt(eigenvalues)) @ q.T
    root_inverse = (q / np.sqrt(eigenvalues)) @ q.T
    teacher_variance = float(np.trace(target @ covariance @ target.T))
    saved = dict(mean_x=mean, covariance_x=covariance, source_A=source,
                 original_proj_A=target, original_proj_bias=bias,
                 effective_proj_center=data['effective_proj_center'],
                 proj_theta=data['proj_theta'], source_theta=data['source_theta'],
                 vectors=data['vectors'])
    results = {}

    def rank2(matrix):
        left, values, right = np.linalg.svd(matrix @ root, full_matrices=False)
        return left[:, :2] * values[:2], right[:2] @ root_inverse

    def record(name, basis, permutation, slope, left, right, **extra):
        base = slope[:, None] * basis[permutation]
        fitted = base + left @ right
        error = target - fitted
        fitted_bias = bias + error @ mean
        per_row = np.diag(error @ covariance @ error.T)
        # Independent covariance-realizing centered pseudo-input check.
        pseudo = root * np.sqrt(10)
        direct = np.square(error @ pseudo).mean()
        assert abs(direct - per_row.mean()) < 1e-12
        mean_error = np.max(np.abs(fitted @ mean + fitted_bias - (target @ mean + bias)))
        assert mean_error < 1e-12
        results[name] = dict(sum10_error=float(per_row.sum()), mean_output_MSE=float(per_row.mean()),
                            teacher_variance_fraction=float(per_row.sum() / teacher_variance),
                            actual_A_rank=int(np.linalg.matrix_rank(fitted)),
                            mean_bias_error_max=float(mean_error), permutation=permutation.tolist(),
                            slope=slope.tolist(), native_bias=fitted_bias.tolist(), **extra)
        for key, value in dict(A=fitted, bias=fitted_bias, base_A=base,
                               permutation=permutation, slope=slope,
                               residual_left=left, residual_right=right).items():
            saved[name + '_' + key] = value

    eye, natural = np.eye(10), np.arange(10)
    zeros_left, zeros_right = np.zeros((10, 2)), np.zeros((2, 10))
    record('ordinary_r2', eye, natural, np.zeros(10), *rank2(target))
    for name, basis, use_assignment in [('shared', source, True), ('identity_permuted', eye, True),
                                       ('identity_diagonal', eye, False)]:
        if use_assignment:
            permutation, slope = assignment(target, covariance, basis)
        else:
            permutation = natural.copy()
            slope = np.diag(target @ covariance) / np.diag(covariance)
        selected = basis[permutation]
        record(name + '_base', basis, permutation, slope, zeros_left, zeros_right)
        left, right = rank2(target - slope[:, None] * selected)
        record(name + '_fixed_r2', basis, permutation, slope, left, right)
        current, previous = slope.copy(), np.inf
        variance = np.einsum('ti,ij,tj->t', selected, covariance, selected)
        for step in range(1000):
            left, right = rank2(target - current[:, None] * selected)
            current = np.einsum('ti,ij,tj->t', target - left @ right, covariance, selected) / variance
            error = target - current[:, None] * selected - left @ right
            value = float(np.trace(error @ covariance @ error.T))
            assert value <= previous + 1e-10
            if np.isfinite(previous) and previous - value <= 1e-12 * max(1, previous):
                break
            previous = value
        record(name + '_joint_r2', basis, permutation, current,
               *rank2(target - current[:, None] * selected),
               iterations=step + 1, converged=step < 999,
               optimum='Local fixed-P alternating least squares; not global joint optimum.')

    # Re-express the ordinary rank-2 map in the already-available Q'=As*x domain.
    # This must not weaken its function-space optimum.
    gq = source @ covariance @ source.T
    eq, qq = np.linalg.eigh((gq + gq.T) / 2)
    cq, cqi = (qq * np.sqrt(eq)) @ qq.T, (qq / np.sqrt(eq)) @ qq.T
    lq, sq, rq = np.linalg.svd((target @ np.linalg.inv(source)) @ cq, full_matrices=False)
    ordinary_from_q = ((lq[:, :2] * sq[:2]) @ rq[:2] @ cqi) @ source
    error = target - ordinary_from_q
    q_error = float(np.trace(error @ covariance @ error.T))
    assert abs(q_error - results['ordinary_r2']['sum10_error']) < 1e-11
    summary = dict(scope=__doc__.strip(), parameters=str(args.parameters), vectors=int(data['vectors']),
                   teacher_total_variance=teacher_variance, rank_budget=2,
                   objective='E|| (Ap-Ahat)(x-mean_x) ||_2^2; sum over ten output times. All native biases corrected by (Ap-Ahat)mean_x; center and theta unchanged.',
                   method='Weighted SVD of (Ap-base)Cov^(1/2); exact conditional rank2 solution. Assignments use scalar-LS costs and 1024-state DP once. Fixed-P refinement alternates rank2 SVD and ten scalar closed-form slope updates; no rank, validation or restart search.',
                   ordinary_rank2_existing_Q_coordinate_error=q_error,
                   covariance_eigen_min_max=[float(eigenvalues.min()), float(eigenvalues.max())],
                   source_A_condition=float(np.linalg.cond(source)), results=results,
                   local_cost='Rank2 uses 2x10 input products plus 10x2 output products per channel/spatial T10 vector, plus ten body merges for the mixed models. Static gain may fold into threshold/direction and residual coefficients. No cost-free residual, Q-prime production, raw-input retention, CMVM, state, or new AEE is asserted.',
                   conclusion='Ordinary permuted diagonal+rank2 beats shared-As+rank2 on these old calibration moments. This initialization does not establish a shared-specific advantage; neither result evaluates the new128-frame restored student.')
    saved['scope'] = np.array(summary['scope'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), **saved)
    args.output.with_suffix('.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({name: result['sum10_error'] for name, result in results.items()}, indent=2))


if __name__ == '__main__':
    main()
