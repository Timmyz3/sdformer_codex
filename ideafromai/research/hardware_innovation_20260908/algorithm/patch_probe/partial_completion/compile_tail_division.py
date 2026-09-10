"""Compile exact private-tail raw-Y thresholds using the existing MAC widths.

Three signed 24x16 products implement an active constant-division query, plus
shifts/adds and one quotient correction. This is arithmetic compilation, not
three physical cycles: ports, pipelines and sharing still require scheduling.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREFIX = [2, 3, 7]
TAIL = [s for s in range(10) if s not in PREFIX]


def compile_divisor(divisor, bound):
    d = abs(int(divisor))
    nmax = (bound + 2) * d - 1
    shift = nmax.bit_length()
    reciprocal = (1 << shift) // d
    signed_r = reciprocal if reciprocal < 32768 else reciprocal - 65536
    return dict(divisor=d, diagonal_sign=1 if divisor > 0 else -1,
                Y_bound=bound, numerator_clip=d * (bound + 1),
                maximum_adjusted_n=nmax, reciprocal_fraction_bits=shift,
                reciprocal_unsigned16=reciprocal, reciprocal_signed16=signed_r,
                reciprocal_high_bit=int(reciprocal >= 32768))


def compiled_threshold(numerator, config):
    num = np.asarray(numerator, dtype=np.int64)
    d, shift = config['divisor'], config['reciprocal_fraction_bits']
    limit = config['numerator_clip']
    clipped = np.clip(num, -limit, limit)
    n = np.where(clipped >= 0, clipped + d - 1, -clipped)
    # Two signed 24x16 products form n * unsigned16(reciprocal). The unsigned
    # coefficient's high bit is handled by a shift/add, not silently treated
    # as a positive signed16 value.
    lo, hi = n & ((1 << 23) - 1), n >> 23
    r = config['reciprocal_signed16']
    product = lo * r + ((hi * r) << 23)
    if config['reciprocal_high_bit']:
        product += n << 16
    q0 = product >> shift
    correction_product = q0 * d  # Third product: both inputs fit signed16.
    q = q0 + (n - correction_product >= d)
    ceiling = np.where(clipped >= 0, q, -q)
    threshold = config['diagonal_sign'] * ceiling
    expected = -np.floor_divide(-num, d)
    expected = np.clip(expected, -config['Y_bound'] - 1, config['Y_bound'] + 1)
    expected *= config['diagonal_sign']
    if np.any(threshold != expected):
        raise RuntimeError('reciprocal correction differs from exact signed division')
    if np.any(np.abs(threshold) > 32767):
        raise RuntimeError('private threshold does not fit this compiled INT16 format')
    return threshold, dict(cases=int(num.size),
        active_division_queries=int((np.abs(num) < limit).sum()),
        constant_clamp_queries=int((np.abs(num) >= limit).sum()),
        reciprocal_corrections=int((n - correction_product >= d).sum()),
        max_product_abs=int(np.abs(product).max(initial=0)),
        max_adjusted_n=int(n.max(initial=0)))


def main():
    data = dict(np.load(HERE/'integer_deployment/common3_diagonal_34.npz'))
    a = data['temporal_int16'].astype(np.int64)
    bound = int(data['Y_abs_bound'].max())
    configs = {t: compile_divisor(a[t, t], bound) for t in TAIL}
    boundary_cases = 0
    for t, c in configs.items():
        # Every possible INT16 raw-Y threshold transition, including equality
        # and adjacent numerator values, not random favourable points.
        multiples = np.arange(-bound - 2, bound + 3, dtype=np.int64) * c['divisor']
        values = (multiples[:, None] + np.array([-1, 0, 1])).ravel()
        _, checks = compiled_threshold(values, c)
        boundary_cases += checks['cases']
    frames = []
    tau = data['threshold_positive'][data['full_entry']]
    for path in sorted((HERE/'integer_valid10').glob('capture_*.npz')):
        sample = np.load(path)
        y = sample['Yi'].reshape(64, 12, 10, 8, 4).transpose(0, 2, 1, 3, 4).reshape(64, 10, 96, 4)
        core = np.einsum('ts,gshp->gthp', a[:, PREFIX], y[:, PREFIX], dtype=np.int64)
        checks = []
        for t in TAIL:
            numerator = tau[t][None, :, None] - core[:, t]
            threshold, row = compiled_threshold(numerator, configs[t])
            reference = core[:, t] + a[t, t] * y[:, t] >= tau[t][None, :, None]
            actual = y[:, t] >= threshold if a[t, t] > 0 else y[:, t] <= threshold
            if np.any(reference != actual):
                raise RuntimeError('compiled raw-Y threshold changed a real integer gate')
            checks.append(dict(time=t, **row))
        frames.append(dict(capture=path.name, rows=checks))
    totals = {key: sum(row[key] for frame in frames for row in frame['rows'])
              for key in ('cases', 'active_division_queries', 'constant_clamp_queries', 'reciprocal_corrections')}
    result = dict(
        scope='exact raw-Y predicate compilation for seven private tails after the three shared Yi columns; not an extra accuracy approximation',
        predicate='d>0: Yi >= ceil((TUi-Ucore)/d); d<0: Yi <= -ceil((TUi-Ucore)/abs(d))',
        proof='For 0<=n<2^F and r=floor(2^F/d), q0=floor(n*r/2^F) is the true floor quotient or one below. One remainder comparison corrects it. Sign handling yields ceiling; clipping only selects predicates constant on the legal raw-Y domain.',
        Y_bound=bound, threshold_bits=16, constants=configs,
        signed_MAC_mapping='n=lo23+hi*2^23; n*r_u16=lo23*r_s16+(hi*r_s16)*2^23+highbit(r_u16)*(n<<16). Correct q0 using q0*d. All products use signed24x16 inputs and fit the 48bit working sum.',
        cost=dict(products_per_active_query=3,
                  other_work='input clip/sign handling, fixed shifts/adds, one quotient-correction comparison, sign restoration; pipeline latency and parameter/source ports not closed',
                  constant_clamp='queries outside the legal Yi range can bypass reciprocal products after the range test; actual materialized-query policy is determined by the tail controller'),
        all_threshold_transition_cases=boundary_cases,
        real_query_totals=totals, frames=frames,
        validation='all tested exact-division transitions and real raw-Y gate comparisons agree; no RTL, physical cycles or PPA',
    )
    (HERE/'tail_division_compile.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(boundary_cases=boundary_cases, real=totals, constants=configs), indent=2))


if __name__ == '__main__':
    main()
