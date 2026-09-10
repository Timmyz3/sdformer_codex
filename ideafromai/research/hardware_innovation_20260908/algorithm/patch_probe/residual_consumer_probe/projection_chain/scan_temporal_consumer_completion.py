"""Fixed 4/16, H8, T10/P2 consumer-completion opportunity; CPU/NumPy only.

Example: python scan_temporal_consumer_completion.py --shared SHARED_CAPTURE
         --raw RAW_DIAG_CAPTURE --output result.json
Captures must be from temporal_consumer_capture.py. This is a new, explicitly
defined Float64 reference on captured coordinates, not proof of equivalence to
the network's FP32 arithmetic. No rank, order, threshold, or seed search.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHECK, RANK, TICKS, H_BLOCK, POSITIONS = 4, 16, 10, 8, 2
STATIC_COEFFICIENT_FIELDS = ('UF_F64_nonzero_coefficients',
    'UF_tail_F64_nonzero_coefficients', 'UF_F64_vs_local_FP32_zero_pattern_differences')


def time_vector(value):
    return np.broadcast_to(np.asarray(value, np.float64).reshape(-1), (TICKS,)).copy()


def ceil_power2(value):
    mantissa, exponent = np.frexp(value)
    exponent = exponent - (mantissa == .5)
    cap = np.where(value == 0, 0., np.ldexp(np.ones_like(value), exponent))
    return cap, exponent.astype(np.int64)


def reference_inputs(frame, params, metadata, kind):
    permutation = np.asarray(params['permutation'], np.int64)
    if sorted(permutation.tolist()) != list(range(TICKS)):
        raise ValueError('This scan requires the saved one-to-one time permutation')
    F = np.asarray(params['F_fp32'], np.float64)
    Z = np.asarray(frame.get('Z_anchor', frame['Z']), np.float64)
    beta = np.asarray(params['BN_offset_fp32'], np.float64).reshape(-1)
    if kind == 'shared_singleQ':
        if metadata['mode'] != 'single_q':
            raise ValueError('Shared capture must identify its actual single_q function')
        coordinates = np.asarray(frame['Q_As_I'], np.float64)[permutation]
        x = np.asarray(frame['AZ_anchor'], np.float64)[permutation]
        gain = time_vector(params['shared_d'])
        # Q is unbiased As*I: source bias/center must NOT be added again.
        rowsum = np.asarray(params['As'], np.float64).sum(1)[permutation]
        offset = rowsum[:, None] * beta[None, :]
    else:
        if 'raw_L' in params or 'raw_R' in params:
            raise ValueError('raw diagonal+R2 is not one of the two scanned axes')
        if metadata['mode'] not in ('native', 'raw_factorized'):
            raise ValueError('Raw-diagonal capture must identify native or raw_factorized arithmetic')
        coordinates = np.asarray(frame['I'], np.float64)[permutation]
        # RawTemporalForward executes e.float(); the capture stores original e64.
        # A native diagonal-only Ap likewise stores that diagonal as FP32.
        x = Z[permutation]
        gain = time_vector(np.asarray(params['raw_e'], np.float32))
        offset = np.broadcast_to(beta, (TICKS, len(beta)))
    bias = time_vector(params['consumer_bias'])
    center = (np.zeros(TICKS) if metadata['consumer_center_mode'] == 'zero'
              else time_vector(params['consumer_center']))
    theta = float(np.asarray(params['consumer_theta']).item())
    rhs = (theta - bias) + center
    tau = np.zeros(TICKS, np.float64)
    np.divide(rhs, gain, out=tau, where=gain != 0)
    if not np.isfinite(tau).all():
        raise ValueError('Nonfinite compiled Float64 threshold')
    base = coordinates + offset[:, :, None, None]
    return dict(base=base, x=x, Z=Z, F=F, tau=tau, gain=gain,
                constant_gate=(bias-center >= theta), theta=theta, bias=bias, center=center)


def scan_core(base, x, F, tau, gain, constant_gate):
    """Serial Float64 full/prefix sums; a shared H8 outward-rounded tail bound.

    The prefix is the *same* Float64 accumulator as in the reference. For the
    remaining 12 products/additions, 64*eps*(max_H8|prefix|+radius) exceeds the
    standard gamma_26 error bound. Every positive radius addition, guard and
    endpoint rounds outward. All-zero tails need no arithmetic error allowance.
    No complete sum or complete gate participates in accept.
    """
    T, H, G, P = base.shape
    if T != TICKS or P != POSITIONS or H % H_BLOCK or F.shape != (H, RANK) or x.shape != (T, RANK, G, P):
        raise ValueError('Expected complete T10/P2, H8 blocks and rank16')
    B, tail = H//H_BLOCK, RANK-CHECK
    full, active = base.copy(), []
    for j in range(RANK):
        coefficient, value = F[None, :, j, None, None], x[:, None, j]
        full = full + coefficient*value
        active.append((coefficient != 0) & (value != 0))
        if j == CHECK-1:
            prefix = full.copy()
    caps, cap_exp = ceil_power2(np.abs(F[:, CHECK:]).reshape(B, H_BLOCK, tail).max(1))
    radius = np.zeros((T, B, G, P), np.float64)
    terms = np.zeros_like(radius, np.int32)
    scale_terms = nontrivial_scales = 0
    nonconstant = gain != 0
    for r, j in enumerate(range(CHECK, RANK)):
        value = np.abs(x[:, j])[:, None]
        term = caps[None, :, r, None, None]*value
        nz = term != 0
        radius = np.where(nz, np.nextafter(radius+term, np.inf), radius)
        enabled = nz & nonconstant[:, None, None, None]
        terms += enabled
        scale_terms += int(enabled.sum())
        nontrivial_scales += int((enabled & (cap_exp[None, :, r, None, None] != 0)).sum())
    prefix_max = np.abs(prefix).reshape(T, B, H_BLOCK, G, P).max(2)
    magnitude = np.nextafter(prefix_max+radius, np.inf)
    guard = np.nextafter(np.ldexp(magnitude, -46), np.inf)  # 64*eps64
    radius_guarded = np.nextafter(radius+guard, np.inf)
    radius_guarded = np.where(radius == 0, 0., radius_guarded)
    expanded = np.repeat(radius_guarded, H_BLOCK, axis=1)
    lower = np.where(expanded == 0, prefix, np.nextafter(prefix-expanded, -np.inf))
    upper = np.where(expanded == 0, prefix, np.nextafter(prefix+expanded, np.inf))
    if not np.isfinite(full).all() or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError('The bounded reference assumes finite Float64 arithmetic')
    bounds_errors = int(((full < lower) | (full > upper)).sum())
    if bounds_errors:
        raise ArithmeticError('Full serial reference escaped an outward-rounded interval')
    threshold = tau[:, None, None, None]
    positive = (gain > 0)[:, None, None, None]
    variable = nonconstant[:, None, None, None]
    constant = constant_gate[:, None, None, None]
    reference = np.where(variable, np.where(positive, full >= threshold, full <= threshold), constant)
    one = np.where(positive, lower >= threshold, upper <= threshold) & variable
    zero = np.where(positive, upper < threshold, lower > threshold) & variable
    one |= (~variable) & constant
    zero |= (~variable) & (~constant)
    accepted, predicted = one | zero, one
    errors = int((accepted & (predicted != reference)).sum())
    if errors:
        raise ArithmeticError('Wrong retirement in the declared Float64 reference domain')
    active = np.stack(active)
    issued = active.copy()
    issued[CHECK:] &= ~accepted[None]
    ordinary_vectors = active.reshape(RANK, T, B, H_BLOCK, G, P).any(axis=(1, 3, 5))
    issued_vectors = issued.reshape(RANK, T, B, H_BLOCK, G, P).any(axis=(1, 3, 5))
    unresolved_union = (~accepted).reshape(T, B, H_BLOCK, G, P).any(axis=(0, 2, 4))
    has_tail_work = active[CHECK:].any(0)
    check_blocks = (radius != 0) & nonconstant[:, None, None, None]
    early_comparisons = int(np.broadcast_to(variable, base.shape).sum())
    early_comparisons += int(((~one) & variable).sum())  # second side only if first failed
    ordinary_comparisons = int(np.broadcast_to(variable, base.shape).sum())
    final_comparisons = int(((~accepted) & variable).sum())
    stats = dict(gates=int(reference.size), accepted=int(accepted.sum()),
        accepted_one=int((accepted & predicted).sum()), accepted_zero=int((accepted & ~predicted).sum()),
        constant_gain_gates=int((~np.broadcast_to(variable, base.shape)).sum()),
        alpha=float(accepted.mean()), strict_retirement_errors=errors, interval_errors=bounds_errors,
        accepted_with_nonzero_tail_work=int((accepted & has_tail_work).sum()),
        ordinary_zero_tail_gates=int((~has_tail_work).sum()),
        full_F64_nonzero_gates=int(reference.sum()),
        negative_gain_equal_threshold_gates=int(((full == threshold) & (gain < 0)[:, None, None, None]).sum()),
        T10_P2_H8_groups=int(unresolved_union.size), unresolved_groups=int(unresolved_union.sum()),
        completed_groups=int((~unresolved_union).sum()),
        ordinary_F_products_after_zero_filter=int(active.sum()),
        conditional_F_products_after_zero_filter=int(issued.sum()),
        saved_F_products_after_zero_filter=int(active.sum()-issued.sum()),
        ordinary_F_H8_vectors_after_zero_filter=int(ordinary_vectors.sum()),
        conditional_F_H8_vectors_after_zero_filter=int(issued_vectors.sum()),
        saved_F_H8_vectors_after_zero_filter=int(ordinary_vectors.sum()-issued_vectors.sum()),
        bound=dict(tail_abs_values=int(x[nonconstant, CHECK:].size),
            nonzero_dyadic_scale_terms=scale_terms, nontrivial_dyadic_scales=nontrivial_scales,
            radius_additions=int(np.maximum(terms-1, 0).sum()),
            early_comparisons=early_comparisons, failed_final_comparisons=final_comparisons,
            ordinary_final_comparisons=ordinary_comparisons,
            extra_comparisons=early_comparisons+final_comparisons-ordinary_comparisons,
            floating_guard_prefix_abs=H_BLOCK*int(check_blocks.sum()),
            floating_guard_H8_max_comparisons=(H_BLOCK-1)*int(check_blocks.sum()),
            floating_guard_additions=2*int(check_blocks.sum()),
            floating_guard_dyadic_scales=int(check_blocks.sum()),
            interval_endpoint_add_sub=2*H_BLOCK*int(check_blocks.sum())))
    return dict(stats=stats, reference=reference, accepted=accepted, predicted=predicted,
        full=full, prefix=prefix, lower=lower, upper=upper, caps=caps, cap_exp=cap_exp)


def cost_account(inputs, params, core, kind):
    T, H, G, P = inputs['base'].shape
    D, N, tail = np.asarray(params['U']).shape[0], G*P, RANK-CHECK
    accepted = core['stats']['accepted']
    full_F, common_U = N*T*H*RANK, N*T*D*H
    saved_dense_F = accepted*tail
    UF = np.asarray(params['U'], np.float64) @ inputs['F']
    UF32 = np.asarray(params['U'], np.float32) @ np.asarray(params['F_fp32'], np.float32)
    z_occurrences = (inputs['Z'] != 0).sum(axis=(0, 2, 3))
    UF_live_full = int(np.dot((UF != 0).sum(0), z_occurrences))
    UF_live_tail = int(np.dot((UF[:, CHECK:] != 0).sum(0), z_occurrences[CHECK:]))
    baseline_UF = N*T*D*(tail if kind == 'raw_diagonal' else RANK)
    inverse_credit = 0 if kind == 'raw_diagonal' else N*D*T*T
    return dict(sampled_anchors=N, dense_full_F_products=full_F,
        dense_common_U_products=common_U, dense_F_products_saved=saved_dense_F,
        dense_extra_continuous_products=baseline_UF,
        dense_removed_inverse_products=inverse_credit,
        dense_product_delta_vs_own_full_baseline=baseline_UF-inverse_credit-saved_dense_F,
        UF_products_after_F64_coefficient_and_Z_zero_filter=UF_live_full,
        UF_tail_products_after_F64_coefficient_and_Z_zero_filter=UF_live_tail,
        UF_F64_nonzero_coefficients=int((UF != 0).sum()),
        UF_tail_F64_nonzero_coefficients=int((UF[:, CHECK:] != 0).sum()),
        UF_F64_vs_local_FP32_zero_pattern_differences=int(((UF != 0) != (UF32 != 0)).sum()),
        filtered_component_delta=(UF_live_tail if kind == 'raw_diagonal' else UF_live_full)-inverse_credit-core['stats']['saved_F_products_after_zero_filter'],
        dense_own_full_path_products=(full_F+common_U if kind == 'raw_diagonal'
            else full_F+common_U+N*(T*T*RANK+D*T*T)),
        dense_own_split_path_products=(full_F-saved_dense_F+common_U+N*T*D*tail
            if kind == 'raw_diagonal' else full_F-saved_dense_F+common_U+N*(T*T*RANK+T*D*RANK)),
        filtered_own_full_path_components=(core['stats']['ordinary_F_products_after_zero_filter']+common_U
            +(N*(T*T*RANK+D*T*T) if kind == 'shared_singleQ' else 0)),
        filtered_own_split_path_components=(core['stats']['conditional_F_products_after_zero_filter']+common_U
            +(N*T*T*RANK+UF_live_full if kind == 'shared_singleQ' else UF_live_tail)),
        own_zero_control_cost_break_even_alpha=(T*D*tail if kind == 'raw_diagonal' else T*D*RANK-D*T*T)/(T*H*tail))


def add_counts(left, right):
    for key, value in right.items():
        if isinstance(value, dict):
            add_counts(left.setdefault(key, {}), value)
        elif key not in ('alpha', 'own_zero_control_cost_break_even_alpha', *STATIC_COEFFICIENT_FIELDS):
            left[key] = left.get(key, 0) + value


def scan_capture(directory, kind):
    directory = Path(directory)
    metadata = json.loads((directory/'capture.json').read_text())
    with np.load(directory/'parameters.npz') as data:
        params = {key: data[key] for key in data.files}
    rows, totals, total_cost = [], {}, {}
    for entry in metadata['frames']:
        with np.load(directory/entry['capture']) as data:
            frame = {key: data[key] for key in data.files}
        values = reference_inputs(frame, params, metadata, kind)
        core = scan_core(**{key: values[key] for key in ('base', 'x', 'F', 'tau', 'gain', 'constant_gate')})
        actual = np.asarray(frame['actual_consumer_theta_g'], np.float64)
        if actual.shape != core['reference'].shape:
            raise ValueError('Actual gates and the captured T,H,G,P coordinates differ in shape')
        stats = core['stats']
        actual_gate = actual != 0
        reference_output = core['reference'] * values['theta']
        stats.update(F64_reference_vs_actual_FP32_gate_differences=int((core['reference'] != actual_gate).sum()),
            F64_reference_vs_actual_FP32_output_differences=int((reference_output != actual).sum()),
            accepted_vs_actual_FP32_gate_differences=int((core['accepted'] & (core['predicted'] != actual_gate)).sum()))
        amplitude_error = float(np.abs(actual-actual_gate*values['theta']).max())
        if values['theta'] == 0:
            raise ValueError('Zero-amplitude capture does not reveal the underlying native gate')
        direct_affine = (core['full']*values['gain'][:, None, None, None]
                         + values['bias'][:, None, None, None]) - values['center'][:, None, None, None]
        stats['tau_reference_vs_direct_affine_F64_differences'] = int(
            ((direct_affine >= values['theta']) != core['reference']).sum())
        cost = cost_account(values, params, core, kind)
        add_counts(totals, stats)
        add_counts(total_cost, cost)
        rows.append(dict(frame=entry['frame'], capture=entry['capture'],
            source_shape=list(values['base'].shape), theta=values['theta'],
            actual_theta_g_max_abs_error=amplitude_error, **stats, cost=cost))
        print(kind, entry['frame'], 'alpha', round(stats['alpha'], 6),
              'F_saved', stats['saved_F_products_after_zero_filter'],
              'H8_vectors_saved', stats['saved_F_H8_vectors_after_zero_filter'], flush=True)
    totals['alpha'] = totals['accepted']/totals['gates'] if totals else None
    return dict(kind=kind, directory=str(directory), capture_mode=metadata['mode'],
        consumer_center_mode=metadata['consumer_center_mode'], frames=rows,
        totals=totals, cost_totals=total_cost,
        static_coefficient_compile={key: cost[key] for key in STATIC_COEFFICIENT_FIELDS},
        permutation=np.asarray(params['permutation']).tolist(),
        gain=values['gain'].tolist(), tau=values['tau'].tolist(),
        H8_tail_caps=core['caps'].tolist(), H8_tail_cap_exponents=core['cap_exp'].tolist())


def self_test():
    rng = np.random.default_rng(610)
    gates_checked = cases = 0
    for kind in ('shared_singleQ', 'raw_diagonal'):
        for boundary in (False, True):
            T, H, G, P = 10, 16, 2, 2
            I = rng.integers(-8, 9, (T, H, G, P)).astype(np.float32)/4
            Z = rng.integers(-5, 6, (T, RANK, G, P)).astype(np.float32)/8
            A = np.eye(T, dtype=np.float32)
            F = rng.integers(-4, 5, (H, RANK)).astype(np.float32)/8
            beta = rng.integers(-2, 3, H).astype(np.float32)/4
            gain = np.array([1, -1, .5, -.25, 2, -2, 0, 1, -.5, 0])
            permutation = np.arange(T-1, -1, -1)
            center = np.arange(T, dtype=np.float64)/8
            bias = np.zeros(T)
            if boundary:
                F.fill(0); beta.fill(0); I.fill(2)
                bias = 1.375+center-gain*2
            params = dict(F_fp32=F, BN_offset_fp32=beta, As=A,
                consumer_bias=bias, consumer_theta=np.array(1.375), consumer_center=center,
                source_bias=np.full(T, 999.), source_center=np.full(T, -999.),
                permutation=permutation, shared_d=gain, raw_e=gain,
                U=np.ones((32, H), np.float32))
            frame = dict(I=I, Z=Z, Q_As_I=I.copy(), AZ_anchor=Z.copy())
            metadata = dict(mode='single_q' if kind == 'shared_singleQ' else 'native', consumer_center_mode='nonzero')
            value = reference_inputs(frame, params, metadata, kind)
            core = scan_core(**{key: value[key] for key in ('base', 'x', 'F', 'tau', 'gain', 'constant_gate')})
            scalar = np.empty_like(core['reference'])
            for t in range(T):
                for h in range(H):
                    for g in range(G):
                        for p in range(P):
                            s = float(I[permutation[t], h, g, p])+float(beta[h])
                            for j in range(RANK):
                                s = s+float(F[h, j])*float(Z[permutation[t], j, g, p])
                            assert s == core['full'][t, h, g, p]
                            scalar[t, h, g, p] = (s >= value['tau'][t] if gain[t] > 0 else
                                s <= value['tau'][t] if gain[t] < 0 else value['constant_gate'][t])
            assert np.array_equal(scalar, core['reference'])
            if boundary:
                assert core['accepted'].all() and core['reference'].all()
            costs = cost_account(value, params, core, kind)
            assert costs['dense_extra_continuous_products'] == G*P*T*32*(12 if kind == 'raw_diagonal' else 16)
            gates_checked += scalar.size; cases += 1
    # Large cancellation and mixed exponents: interval containment, not a seed search.
    x = np.ldexp(rng.choice([-1., 1.], (10, 16, 2, 2)), rng.integers(-20, 21, (10, 16, 2, 2)))
    F = np.ldexp(rng.choice([-1., 1.], (16, 16)), rng.integers(-20, 21, (16, 16)))
    base = rng.normal(size=(10, 16, 2, 2))*1e8
    result = scan_core(base, x, F, np.zeros(10), np.array([1., -1.]*5), np.zeros(10, bool))
    assert result['stats']['strict_retirement_errors'] == 0
    print(json.dumps(dict(self_test='PASS', scalar_cases=cases, scalar_gates=gates_checked,
        mixed_exponent_interval_gates=int(result['reference'].size),
        checks='Nontrivial P, both signs, exact threshold equality, zero gain, nonunit theta, ignored source bias/center, zero tail, serial scalar reference, interval containment; fixed checkpoint only')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared', type=Path)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.shared is None or args.raw is None:
        parser.error('--shared and --raw are both required; do not omit a control')
    result = dict(fixed=dict(T=10, R=16, checkpoint=4, H_block=8, P=2, latent_order=list(range(16))),
        numerical_definition='Captured FP32 inputs/constants promoted to Float64. Shared uses captured Q_As_I and AZ_anchor at saved P and saved d64; raw uses I and Z at P and rounds saved raw_e64 to its actually executed FP32 coefficient before promotion. Offset is As.sum(Float64)*BN_offset for shared, BN_offset for raw. Full sum starts at coordinate+offset then performs j0..15 multiply/add in Float64. Gate compares to rounded Float64 tau=(theta-bias+center)/gain using >= for positive and <= for negative gain. Zero gain uses its constant gate. This defines a new reference; direct-affine-F64 and actual-FP32 differences are reported, never hidden.',
        exactness_scope='Outward H8 ceilPow2 radius plus a 64*eps64 rounding guard encloses the complete serial Float64 sum. Zero errors certify retirement only in that reference domain. Not a network-FP32, RTL, fixed-point, or flow-AEE equivalence proof.',
        reuse_scope='All R16 Z and, for shared, all AZ are already produced and paid. No future full gate or F-tail dot is used for accept. Source theta remains embodied in captured Z; output theta is independent of the compiled decision threshold.',
        count_scope='F products and logical H8 coefficient-vector uses give BOTH axes exact F-zero/latent-zero filtering. One vector use is OR across T10/P2/H8 at one sampled group and one latent j. Bound abs values can be shared across H8 blocks; no free physical broadcast/port assumption is made. Nextafter is the CPU proof wrapper; error-guard operations are separately counted, not silently free hardware.',
        cost_scope='Dense formulas compare products at equal unit cost with control/state cost zero; NOT cycles. Dense U is included; input-dependent U activity is not measured. Filtered component delta additionally checks F64-folded UF coefficient zeros and actual raw Z zeros. Source GP, final V work, dtype-dependent cost, thresholds, bounds/comparisons, partial-state traffic, and physical mapping remain outside these totals. Reassociated continuous readout has not been evaluated in the network. Values from distinct students are not a same-function speedup.',
        state=dict(per_anchor_I_or_Q_numbers=960, per_anchor_Z_numbers=160,
            per_anchor_partial_continuous_numbers=320, per_anchor_gate_bits=960,
            per_anchor_accept_bits=960, H8_tail_cap_exponents=144,
            raw_UF_tail_coefficients=384, shared_UF_coefficients=512,
            note='Logical state, not peak RF or SRAM bytes. The partial continuous accumulator overlaps unresolved gate state; address reuse must be scheduled. P2 doubles per-anchor numbers. Baseline also gets legal streaming/address reuse.'),
        necessary_dense_gates=dict(raw_own_alpha_gt=1/3, shared_own_alpha_gt=1/6,
            shared_vs_raw_full_alpha_gt=7/12, shared_vs_profitable_raw_split_alpha_gap_gt=.25,
            assumptions='H96/D32/T10/R16/check4; nonconstant gains, no zero operands, equal product cost, no control/state tax. Raw chooses min(full reuse, prefix split). Actual α may include empty tails, so use zero-filtered savings too.'),
        not_scanned=['ordinary independent Ap: retains equal split/bound permissions and its native full-FZ/PSN alternatives', 'raw diagonal+rank2: equal permissions, no fabricated measurements'])
    result['axes'] = dict(shared_singleQ=scan_capture(args.shared, 'shared_singleQ'),
                          raw_diagonal=scan_capture(args.raw, 'raw_diagonal'))
    shared, raw = result['axes']['shared_singleQ'], result['axes']['raw_diagonal']
    if [r['frame'] for r in shared['frames']] != [r['frame'] for r in raw['frames']]:
        raise ValueError('Cross-student comparison requires the same captured frame list')
    sc, rc = shared['cost_totals'], raw['cost_totals']
    result['cross_student_components'] = dict(
        alpha_shared_minus_raw=shared['totals']['alpha']-raw['totals']['alpha'],
        dense_shared_best_minus_raw_best=min(sc['dense_own_full_path_products'], sc['dense_own_split_path_products'])
            -min(rc['dense_own_full_path_products'], rc['dense_own_split_path_products']),
        filtered_shared_best_minus_raw_best=min(sc['filtered_own_full_path_components'], sc['filtered_own_split_path_components'])
            -min(rc['filtered_own_full_path_components'], rc['filtered_own_split_path_components']),
        scope='One whole-capture full-or-split choice per student, not an oracle per-frame choice. Products only, control/state and distinct upstream/source work omitted. Filtered components retain dense U/AZ/inverse and apply F/UF/latent-zero filtering as explicitly counted; not complete service or AEE.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('saved', args.output)


if __name__ == '__main__':
    main()
