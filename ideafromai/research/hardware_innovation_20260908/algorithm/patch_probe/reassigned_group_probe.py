"""Strong ordinary group control: decouple input groups from output labels.

Every 3/3/4 input partition has 3/3/4 output slots. Hungarian assignment
minimizes full train32 membrane MSE for those slots. Exhausting all 2100
partitions retains exactly 34 connections and at most four resident Y.
Unlike row34, this control does not require the original diagonal.
"""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path,
                        default=Path(__file__).parent/'dependency')
    args = parser.parse_args()
    started = time.monotonic()
    fit = json.loads((args.directory/'fit.json').read_text())
    moments = np.load(args.directory/'train_moments.npz')
    mean, cov = moments['mean'], moments['covariance']
    a = np.asarray(fit['variants']['native']['weight'], dtype=np.float64)
    bias = np.asarray(fit['variants']['native']['bias'], dtype=np.float64)
    cache = {}
    for size in (3, 4):
        for group in itertools.combinations(range(10), size):
            cols = list(group)
            coef = np.linalg.lstsq(cov[np.ix_(cols, cols)], cov[cols] @ a.T,
                                   rcond=1e-10)[0].T
            delta = a.copy()
            delta[:, cols] -= coef
            error = np.einsum('ij,jk,ik->i', delta, cov, delta).clip(0)
            cache[group] = error, coef

    def assign(groups):
        slots = [group for group in groups for _ in group]
        costs = np.array([cache[group][0] for group in slots]).T
        rows, chosen = linear_sum_assignment(costs)
        support_groups = tuple(slots[col] for col in chosen)
        return float(costs[rows, chosen].sum()), support_groups

    partitions = []
    for four in itertools.combinations(range(10), 4):
        remain = [x for x in range(10) if x not in four]
        for three in itertools.combinations(remain, 3):
            other = tuple(x for x in remain if x not in three)
            if three > other:
                continue
            groups = (three, other, four)
            cost, assigned = assign(groups)
            partitions.append((cost, groups, assigned))
    best = min(partitions)
    contiguous = ((0, 1, 2), (3, 4, 5), (6, 7, 8, 9))
    cont_cost, cont_assigned = assign(contiguous)
    variants = {}
    for name, (cost, groups, assignments) in (
        ('reassigned_continuous334', (cont_cost, contiguous, cont_assigned)),
        ('reassigned_fit334', best),
    ):
        weight = np.zeros_like(a)
        row_errors = []
        for row, group in enumerate(assignments):
            error, coef = cache[group]
            weight[row, list(group)] = coef[row]
            row_errors.append(float(error[row]))
        fitted_bias = (bias+(a-weight) @ mean).astype(np.float32)
        weight = weight.astype(np.float32)
        variants[name] = dict(
            weight=weight.tolist(), bias=fitted_bias.tolist(),
            support=(weight != 0).tolist(),
            input_groups=[list(x) for x in groups],
            output_rows_by_group=[[row for row, chosen in enumerate(assignments)
                                   if chosen == group] for group in groups],
            input_order=[x for group in groups for x in group],
            allocated_connections=34,
            actual_connections=int(np.count_nonzero(weight)),
            matrix_rank=int(np.linalg.matrix_rank(weight)),
            training_membrane_MSE=cost/10,
            per_row_MSE=row_errors,
            retained_diagonal=int(np.count_nonzero(np.diag(weight))),
            state_expectation='four resident Y including the current input, plus one working U; original output t labels restored',
        )
    result = dict(variants=variants, partitions=len(partitions),
        assignment='Hungarian allocation of output rows to 3/3/4 group capacities; diagonal not required',
        selection='full train32 moments only; no validation data',
        seconds=time.monotonic()-started,
        claim='stronger ordinary grouped-PSN control; no AEE or hardware timing result')
    (args.directory/'reassigned_groups.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({name: {k:v for k,v in variant.items()
                            if k not in ('weight','bias','support','per_row_MSE')}
                      for name, variant in variants.items()}, indent=2))


if __name__ == '__main__':
    main()
