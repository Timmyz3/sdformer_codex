"""Exact examples for a new-model representation, not a frozen H67 experiment."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import random
import sys


def dense(base, residual, p, t):
    return base[t] + residual.get((p, t), F(0))


def main():
    assert sys.version_info[:2] == (3, 12)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rng = random.Random(0x1d3f)
    trials = decisions = 0
    for t_count in (2, 10):
        # Identity plus a signed superdiagonal is always full rank.
        matrix = [[F(int(s == t)) - F(1, 2)*int(s == t+1)
                   for s in range(t_count)] for t in range(t_count)]
        for _ in range(100):
            p_count = 7
            base = [F(rng.randint(-3, 3), 4) for _ in range(t_count)]
            residual = {(p, t): F(rng.randint(-3, 3), 3)
                        for p in range(p_count) for t in range(t_count)
                        if rng.randrange(4) == 0}
            all_x = [dense(base, residual, p, t)
                     for p in range(p_count) for t in range(t_count)]
            mass = p_count * sum(base) + sum(residual.values())
            square = p_count * sum(x*x for x in base) + sum(
                2*base[t]*e + e*e for (p, t), e in residual.items())
            assert mass == sum(all_x)
            assert square == sum(x*x for x in all_x)
            mu = mass / len(all_x)
            assert square / len(all_x) - mu*mu >= 0
            # Supply a positive normalization denominator; exact rational tests
            # of affine BN propagation do not claim a hardware sqrt proof.
            denom, gamma, beta = F(rng.randint(1, 5), 2), F(rng.randint(-3, 3)), F(1, 3)
            b_norm = [gamma*(x-mu)/denom+beta for x in base]
            e_norm = {key: gamma*value/denom for key, value in residual.items()}
            theta0 = F(3, 4)
            theta_extra = {(p, t): F(1, 8) for p in range(p_count)
                           for t in range(t_count) if rng.randrange(9) == 0}
            default_u = [sum(a*v for a, v in zip(row, b_norm)) for row in matrix]
            default_z = [theta0*int(u >= theta0) for u in default_u]
            for p in range(p_count):
                dense_normalized = [gamma*(dense(base, residual, p, t)-mu)/denom+beta
                                    for t in range(t_count)]
                for t, row in enumerate(matrix):
                    u = sum(a*v for a, v in zip(row, dense_normalized))
                    encoded_u = default_u[t] + sum(a*e_norm.get((p, s), F(0))
                                                  for s, a in enumerate(row))
                    assert u == encoded_u
                    theta = theta0 + theta_extra.get((p, t), F(0))
                    actual = theta*int(u >= theta)
                    exception = theta*int(encoded_u >= theta)-default_z[t]
                    assert actual == default_z[t]+exception
                    decisions += 1
            trials += 1
    # All gates are active while continuous amplitudes differ: bits alone fail.
    theta = [F(1, 4), F(1, 4), F(1, 4), F(3, 4)]
    x = [F(1), F(1), F(1), F(3)]
    mu = sum(x)/4
    var = sum(v*v for v in x)/4-mu*mu
    assert (mu, var) == (F(3, 2), F(3, 4))
    normalized = [v-mu+1 for v in x]  # supplied denominator 1 for exact check
    amplitude = [th*int(v>=th) for v, th in zip(normalized, theta)]
    assert amplitude == theta
    assert amplitude != [F(1)]*4 and amplitude != [F(0)]*4
    periodic = [0, 1, 0, 1]
    wrong_motion_prediction = periodic[-2:]+periodic[:-2]
    assert wrong_motion_prediction == periodic
    result = {
        'status': 'PASS_RATIONAL_REPRESENTATION_EXAMPLES',
        'full_domain_moment_cases': trials, 'full_rank_time_amplitude_decisions': decisions,
        'T': [2, 10], 'amplitudes_example': [str(v) for v in amplitude],
        'counterexamples_preserved': ['all active gates do not determine unequal theta amplitudes',
                                     'absent residual is a nonzero current default state',
                                     'periodic wrong displacement can have zero residual'],
        'normalization_scope': 'Exact sum/sumsq and affine propagation with supplied positive denominator; no sqrt implementation',
        'PPA_ADMISSION': 0, 'RTL_SPEEDUP_ADMISSION': 0, 'FROZEN_FP_EQUIVALENCE': 0,
        'AEE_ADMISSION': 0}
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
