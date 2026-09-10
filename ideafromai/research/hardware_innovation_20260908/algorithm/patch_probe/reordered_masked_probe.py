"""Fit the strongest train-MSE permutation of the ordinary causal k4 mask.

All ten inputs are already available in this workload. Permuting both axes
retains the original output labels and gives exactly 34 coefficient slots.
This is a stronger masked-PSN control, not a new neuron or a timing result.
The subset/last-three DP is exact over all 10! permutations; validation data
are never used to select an order or fit coefficients.
"""
import argparse
from functools import lru_cache
import itertools
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path,
                        default=Path(__file__).parent/'dependency')
    args = parser.parse_args()
    started = time.monotonic()
    fit = json.loads((args.directory/'fit.json').read_text())
    moments = np.load(args.directory/'train_moments.npz')
    mean, cov = moments['mean'], moments['covariance']
    original = fit['variants']['native']
    a = np.asarray(original['weight'], dtype=np.float64)
    bias = np.asarray(original['bias'], dtype=np.float64)
    n = len(a)
    cached = {}
    for row in range(n):
        other = [i for i in range(n) if i != row]
        for size in range(1, 5):
            for previous in itertools.combinations(other, size-1):
                cols = tuple(sorted((*previous, row)))
                ix = list(cols)
                coef = np.linalg.lstsq(cov[np.ix_(ix, ix)], cov[ix] @ a[row],
                                       rcond=1e-10)[0]
                delta = a[row].copy()
                delta[ix] -= coef
                error = max(0., float(delta @ cov @ delta))
                cached[row, cols] = error, coef
    full = (1 << n)-1

    @lru_cache(None)
    def solve(done, last):
        if done == full:
            return 0., ()
        candidates = []
        for row in range(n):
            if done & (1 << row):
                continue
            cols = tuple(sorted((*last, row)))
            suffix_error, suffix = solve(done | (1 << row), (*last, row)[-3:])
            candidates.append((cached[row, cols][0]+suffix_error, (row, *suffix)))
        return min(candidates)

    total_error, order = solve(0, ())
    weight = np.zeros_like(a)
    row_errors = np.zeros(n)
    for step, row in enumerate(order):
        cols = tuple(sorted(order[max(0, step-3):step+1]))
        row_errors[row], coefficient = cached[row, cols]
        weight[row, list(cols)] = coefficient
    fitted_bias = (bias+(a-weight) @ mean).astype(np.float32)
    weight = weight.astype(np.float32)
    natural = tuple(range(n))
    natural_error = sum(cached[row, natural[max(0, row-3):row+1]][0]
                        for row in natural)/n
    result = dict(
        name='reordered_masked_k4', input_order=list(order),
        output_labels='original t; row perm[i] is emitted at arrival step i',
        weight=weight.tolist(), bias=fitted_bias.tolist(),
        support=(weight != 0).tolist(),
        allocated_connections=34,
        actual_connections=int(np.count_nonzero(weight)),
        matrix_rank=int(np.linalg.matrix_rank(weight)),
        training_membrane_MSE=total_error/n,
        per_row_MSE=row_errors.tolist(),
        natural_masked_k4_recomputed_MSE=natural_error,
        state_expectation='ordinary order-4 control: at most four Y including the current input, plus one working U; full producer/consumer cost unmeasured',
        selection='exact subset/last-three DP over all 10! orders, full train32 moments only',
        dp_states=solve.cache_info().currsize,
        seconds=time.monotonic()-started,
        claim='permuted known mask; no new neuron, no AEE result or speedup',
    )
    target = args.directory/'reordered_masked_k4.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('weight', 'bias', 'support', 'per_row_MSE')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
