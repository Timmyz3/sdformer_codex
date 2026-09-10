"""Independent CPU implementation of the Gyro output-grouping branch.

Primary descriptions:
  HiNM (2024), section 4: https://arxiv.org/html/2407.20496v1
  FlexHiNM-GP (ICLR 2026), appendix F / algorithm 3:
  https://proceedings.iclr.cc/paper_files/paper/2026/file/
  9e33fdc35b68781132e836964a326bf3-Paper-Conference.pdf

This is NOT author code or a complete HiNM/VENOM/FlexHiNM reproduction.
Inherited: equal per-group sampling, balanced clustering of importance-score
features, actual-pruning assignment costs, Hungarian matching, and the optional
stop-on-nonimprovement rule. The paper's example sampling list is used by default.

Local concretizations: exact capacity rather than approximately balanced groups;
deterministic farthest-point centroid initialization; whole C16 0/16 pruning;
stable lower-block-index ties; supplied joint-consumer cost; bounded Lloyd steps.
Input permutation, inner 2:4, region search, OBS weight compensation and Hard
Concrete/gradual training are OUTSIDE this module. Raw block scores are used as
features by default; no implicit row normalization or replacement by W squared.

Permutation convention: perm[new_h] = old_h. Hence W1_new=W1[perm],
tau_new=tau[:,perm], theta_new=theta[...,perm] when h-dependent, and
W2_new=W2[:,perm]. The inverse maps a permuted activation back to old h order.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


PAPER_SAMPLE_SCHEDULE = (8, 1, 4, 1, 2, 1, 1)


def shared_keep(block_saliency, group_indices, keep_blocks=12):
    """Boolean [N,B] mask for groups [N,K]; accepts any K <= group size.

    Mask selection is always the same additive target, including when the
    optional callback evaluates its genuinely joint nonlinear consequences.
    """
    score = np.asarray(block_saliency, dtype=np.float64)
    groups = np.asarray(group_indices, dtype=np.int64)
    summed = score[groups].sum(axis=1)
    chosen = np.argsort(-summed, axis=1, kind='stable')[:, :keep_blocks]
    keep = np.zeros_like(summed, dtype=bool)
    np.put_along_axis(keep, chosen, True, axis=1)
    return keep


def additive_pruning_cost(block_saliency, group_indices, keep_blocks=12):
    """Realized pruned saliency after recomputing each candidate's shared mask."""
    score = np.asarray(block_saliency, dtype=np.float64)
    groups = np.asarray(group_indices, dtype=np.int64)
    keep = shared_keep(score, groups, keep_blocks)
    return (score[groups].sum(axis=1) * ~keep).sum(axis=1)


def diagonal_second_order_blocks(weight, hessian_diagonal, block_size=16):
    """Sum 0.5*w^2*H_diag per block; OBD/diagonal-GGN, NOT OBS/oBERT.

    The caller supplies curvature, not squared gradients of a zero teacher loss.
    Whole-block reconstruction scores with within-block correlations can instead
    be supplied directly to gyro_group as block_saliency.
    """
    w = np.asarray(weight, dtype=np.float64)
    curvature = np.asarray(hessian_diagonal, dtype=np.float64)
    assert w.shape[1] % block_size == 0 and np.all(curvature >= 0)
    return (.5 * np.square(w) * curvature).reshape(
        w.shape[0], w.shape[1] // block_size, block_size).sum(axis=2)


def balanced_clusters(features, cluster_count, cluster_size, iterations=4):
    """Exact-cardinality Lloyd clustering using an assignment to repeated slots.

    Returned indices refer to rows of features. This deterministic implementation
    is our capacity-constrained realization, not a recovered author routine.
    """
    x = np.asarray(features, dtype=np.float64)
    n = len(x)
    assert n == cluster_count * cluster_size
    if cluster_size == 1:
        return np.arange(n, dtype=np.int64)[:, None]
    # One scalar scaling leaves the Euclidean clustering objective unchanged.
    scale = np.max(np.abs(x))
    if scale:
        x = x / scale
    norm = np.einsum('nd,nd->n', x, x)
    chosen = np.zeros(n, dtype=bool)
    nearest = np.full(n, np.inf)
    centers = []
    index = int(np.argmax(norm))
    for _ in range(cluster_count):
        centers.append(x[index].copy())
        chosen[index] = True
        nearest = np.minimum(nearest, np.sum((x - x[index]) ** 2, axis=1))
        nearest[chosen] = -np.inf
        index = int(np.argmax(nearest))
    centers = np.asarray(centers)
    previous = None
    for _ in range(iterations):
        distance = norm[:, None] + np.sum(centers ** 2, axis=1)[None] - 2 * x @ centers.T
        row, slot = linear_sum_assignment(np.repeat(distance, cluster_size, axis=1))
        assigned = np.empty(n, dtype=np.int64)
        assigned[slot] = row
        groups = np.sort(assigned.reshape(cluster_count, cluster_size), axis=1)
        if previous is not None and np.array_equal(groups, previous):
            break
        centers = x[groups].mean(axis=1)
        previous = groups.copy()
    return groups


def gyro_group(block_saliency, *, group_size=8, keep_blocks=12, block_size=16,
               sample_schedule=PAPER_SAMPLE_SCHEDULE, seed=9523,
               group_cost=None, callback_batch=256, features=None,
               initial_groups=None, cluster_iterations=4,
               stop_on_nonimprove=True, deduplicate_costs=True):
    """Return balanced groups, their shared mask, permutation and search history.

    group_cost(groups[N,group_size]) -> nonnegative cost[N]. All callback groups
    have the COMPLETE group size, are sorted in original-h coordinates, and must
    be evaluated on the SAME fixed train examples and the mask from shared_keep.
    No validation inputs or random per-call subsamples belong in that callback.

    Default objective is equation-4-style realized additive pruning cost. With a
    callback, BOTH Hungarian assignment and acceptance use the supplied joint
    cost; clustering still uses the same supplied features. Setting
    stop_on_nonimprove=False retains the best grouping and spends the entire
    common schedule: useful matched-budget control, but not algorithm 3's stop.
    Identical complete groups may share one deterministic cost evaluation; this
    changes neither candidate masks nor the assignment problem (notably s=8).
    """
    score = np.asarray(block_saliency, dtype=np.float64)
    assert score.ndim == 2 and np.isfinite(score).all() and np.all(score >= 0)
    h, blocks = score.shape
    assert h % group_size == 0 and 0 <= keep_blocks <= blocks
    assert cluster_iterations >= 1 and callback_batch >= 1
    schedule = tuple(int(s) for s in sample_schedule)
    assert all(1 <= s <= group_size for s in schedule)
    count = h // group_size
    groups = (np.arange(h).reshape(count, group_size) if initial_groups is None
              else np.asarray(initial_groups, dtype=np.int64).copy())
    assert groups.shape == (count, group_size)
    assert np.array_equal(np.sort(groups.ravel()), np.arange(h))
    groups.sort(axis=1)
    feat = score if features is None else np.asarray(features, dtype=np.float64)
    assert feat.ndim == 2 and len(feat) == h and np.isfinite(feat).all()
    rng = np.random.default_rng(seed)
    evaluations = 0
    requests = 0

    def evaluate(candidates):
        nonlocal evaluations, requests
        requests += len(candidates)
        if deduplicate_costs:
            candidates, inverse = np.unique(candidates, axis=0, return_inverse=True)
        out = []
        for first in range(0, len(candidates), callback_batch):
            batch = candidates[first:first + callback_batch]
            value = (additive_pruning_cost(score, batch, keep_blocks)
                     if group_cost is None else group_cost(batch))
            value = np.asarray(value, dtype=np.float64)
            assert value.shape == (len(batch),) and np.isfinite(value).all()
            assert np.all(value >= 0), 'group cost must be a nonnegative fixed-example distortion'
            out.append(value)
            evaluations += len(batch)
        values = np.concatenate(out)
        return values[inverse] if deduplicate_costs else values

    current = float(evaluate(groups).sum())
    initial_cost = current
    history = []
    for iteration, samples in enumerate(schedule):
        local = np.stack([rng.permutation(group_size) for _ in range(count)])
        selected = np.take_along_axis(groups, local[:, :samples], axis=1).ravel()
        remainder = np.take_along_axis(groups, local[:, samples:], axis=1)
        labels = balanced_clusters(feat[selected], count, samples, cluster_iterations)
        clusters = selected[labels]
        candidates = np.concatenate((
            np.broadcast_to(remainder[:, None], (count, count, group_size - samples)),
            np.broadcast_to(clusters[None], (count, count, samples))), axis=2)
        candidates.sort(axis=2)
        costs = evaluate(candidates.reshape(-1, group_size)).reshape(count, count)
        row, column = linear_sum_assignment(costs)
        proposed = float(costs[row, column].sum())
        accepted = proposed < current
        before = current
        if accepted:
            groups = candidates[row, column].copy()
            current = proposed
        history.append(dict(iteration=iteration, samples_per_group=samples,
                            before=before, proposed=proposed, accepted=bool(accepted),
                            after=current, assignment_candidates=count * count))
        if not accepted and stop_on_nonimprove:
            break
    keep = shared_keep(score, groups, keep_blocks)
    block_mask_original = np.zeros((h, blocks), dtype=bool)
    block_mask_original[groups] = keep[:, None, :]
    perm = groups.ravel()
    assert np.array_equal(np.sort(perm), np.arange(h))
    return dict(groups=groups, perm=perm, inverse_perm=np.argsort(perm),
                keep_blocks=keep, block_mask_original=block_mask_original,
                mask_original=np.repeat(block_mask_original, block_size, axis=1),
                initial_cost=initial_cost, final_cost=current,
                final_additive_cost=float(additive_pruning_cost(score, groups, keep_blocks).sum()),
                history=history, group_evaluations=evaluations,
                logical_group_cost_requests=requests, deduplicate_costs=deduplicate_costs,
                objective='joint_callback' if group_cost is not None else 'realized_additive_pruned_saliency',
                stop_rule='paper_nonimprovement_stop' if stop_on_nonimprove else 'local_fixed_budget_best_so_far',
                sample_schedule=schedule,
                provenance='independent output-branch implementation; C16 local specialization; not author artifact')
