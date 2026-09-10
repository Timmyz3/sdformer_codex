"""Compile the saved U8/VQ5 students into a distinct exact-integer function.

No GPU, RTL, activation quantizer, or PPA. The saved train-only residual
moments define the predictor; they are not silently re-estimated on validation.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ('shared56_u8_vq5', 'private56_u8_vq5', 'shared48_u8_vq5')


def fraction(value):
    return Fraction.from_float(float(value))


def power2(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def ceil_fraction(value):
    return -((-value.numerator) // value.denominator)


def signed_width(low, high):
    low, high = int(np.min(low)), int(np.max(high))
    bits = 1
    while low < -(1 << (bits-1)) or high >= (1 << (bits-1)):
        bits += 1
    return bits


def interval_times_a(a, low, high):
    positive = np.maximum(a, 0).sum(1)[:, None]
    negative = np.minimum(a, 0).sum(1)[:, None]
    return positive*low + negative*high, positive*high + negative*low


def domains(u, v, a, shared):
    """Tight binary-source prefix bounds plus any-reduction-order bounds."""
    zlo, zhi = np.minimum(u, 0).sum(0), np.maximum(u, 0).sum(0)
    zabs = np.maximum(-zlo, zhi)
    ylo_all, yhi_all, effective = [], [], np.zeros((u.shape[0], v.shape[1]), np.int64)
    for r in range(u.shape[1]):
        effective += u[:, r, None]*v[r]
        ylo_all.append(np.minimum(effective, 0).sum(0))
        yhi_all.append(np.maximum(effective, 0).sum(0))
    ylo_all, yhi_all = np.stack(ylo_all), np.stack(yhi_all)
    ylo, yhi = ylo_all[-1], yhi_all[-1]
    ulo, uhi = interval_times_a(a, ylo, yhi)
    alo, ahi = [], []
    for lo, hi in zip(ylo_all, yhi_all):
        l, h = interval_times_a(a, lo, hi)
        alo.append(l); ahi.append(h)
    alo, ahi = np.stack(alo), np.stack(ahi)
    # V->A accumulates time columns. Source times have independent binary
    # supports; latent channels share the same source, already used above.
    cp, cn = np.maximum(a, 0).cumsum(1), np.minimum(a, 0).cumsum(1)
    valo = cp[:, :, None]*ylo + cn[:, :, None]*yhi
    vahi = cp[:, :, None]*yhi + cn[:, :, None]*ylo
    qlo, qhi = interval_times_a(a, zlo, zhi)
    y_triangle = zabs @ np.abs(v)
    q_triangle = np.abs(a).sum(1)[:, None]*zabs
    # These bounds also cover arbitrary GEMM reduction orders, products,
    # shared/tail partitions, and deleted private terms. No rounding needed.
    u_triangle = np.abs(a).sum(1)[:, None]*y_triangle
    routes = dict(
        Zi_signed_bits=signed_width(zlo, zhi),
        Yi_ordered_latent_prefix_bits=signed_width(ylo_all, yhi_all),
        Yi_any_order_triangle_bits=signed_width(-y_triangle, y_triangle),
        A_then_V_Q_signed_bits=signed_width(qlo, qhi),
        A_then_V_U_ordered_prefix_bits=signed_width(alo, ahi),
        V_then_A_U_ordered_prefix_bits=signed_width(valo, vahi),
        U_any_order_triangle_bits=signed_width(-u_triangle, u_triangle),
        Zi_min=int(zlo.min()), Zi_max=int(zhi.max()),
        Yi_final_min=int(ylo.min()), Yi_final_max=int(yhi.max()),
        Yi_any_order_absolute_bound=int(y_triangle.max()),
        Q_any_order_absolute_bound=int(q_triangle.max()),
        U_final_min=int(ulo.min()), U_final_max=int(uhi.max()),
        U_any_order_absolute_bound=int(u_triangle.max()),
        FP64_all_integer_products_and_reductions_exact=bool(
            max(int(y_triangle.max()), int(q_triangle.max()), int(u_triangle.max())) <= 2**53),
        INT48_U_any_reduction_order=bool(np.max(u_triangle) <= 2**47-1),
        proof='All source bits independently 0/1. Ordered bounds retain common-source latent correlation; triangle bounds cover arbitrary summation order and pruned private subsets.')
    return routes, dict(integer_z_lower=zlo, integer_z_upper=zhi,
        integer_y_lower=ylo, integer_y_upper=yhi,
        integer_y_shared_lower=ylo_all[shared-1], integer_y_shared_upper=yhi_all[shared-1],
        integer_u_lower=ulo, integer_u_upper=uhi,
        integer_y_any_order_abs=y_triangle, integer_u_any_order_abs=u_triangle)


def compile_model(filename):
    with np.load(filename) as data:
        original = {key: data[key].copy() for key in data.files}
    u = original['u_int8'].astype(np.int64)
    e = original['u_scale_exponent'].astype(int)
    sign = original['v_sign'].astype(np.int64)
    shift = original['v_shift'].astype(int)
    nz = original['v_nonzero'].astype(bool)
    gain = original['bn_scale'].astype(np.float64)
    bias = original['bn_bias'].astype(np.float64)
    b = original['temporal_bias'].astype(np.float64).reshape(-1)
    theta = float(original['theta_output'])
    if not np.all(gain > 0):
        raise ValueError('This declared deployment needs positive BN1 gains.')
    if not np.all(nz.any(0)):
        raise ValueError('An all-zero V output needs a separate constant-output compilation.')
    uq_expected = u*np.exp2(e)[None, :]
    if not np.array_equal(uq_expected, original['u'].astype(np.float64)*float(original['theta_source'])):
        raise ValueError('Saved U8/theta restoration does not match the exported source contract.')
    if not np.array_equal(sign*np.exp2(shift)*nz, original['v'].astype(np.float64)):
        raise ValueError('Saved V does not match sign/shift/nonzero.')
    exponent = e[:, None] + shift
    eh = np.where(nz, exponent, 1000).min(0)
    aligned_shift = np.where(nz, exponent-eh, 0)
    v = sign*np.left_shift(np.ones_like(sign), aligned_shift)*nz
    aq = np.rint(original['a'].astype(np.float64)*16384).astype(np.int64)
    if not (aq.min() >= -32768 and aq.max() <= 32767):
        raise ValueError('Aq does not fit signed16.')
    shared = int(original['shared_rank'])
    domain, extra = domains(u, v, aq, shared)
    if not domain['FP64_all_integer_products_and_reductions_exact']:
        raise ValueError('The declared FP64 GPU integer emulation bound failed.')
    tcount, hcount = aq.shape[0], gain.size
    kappa = [fraction(gain[h])*power2(int(eh[h])-14) for h in range(hcount)]
    offsets = [[fraction(bias[h])*Fraction(int(aq[t].sum()),16384)+fraction(b[t])-fraction(theta)
                for h in range(hcount)] for t in range(tcount)]
    full_threshold = np.array([[ceil_fraction(-offsets[t][h]/kappa[h])
                                for h in range(hcount)] for t in range(tcount)], np.int64)
    arrays = dict(original)
    arrays.update(extra, integer_v_align_coeff=v,
        integer_v_align_shift=aligned_shift.astype(np.uint8),
        integer_y_exponent=eh.astype(np.int16),
        integer_a_q14=aq.astype(np.int16),
        integer_full_threshold=full_threshold,
        integer_gain_sign=np.ones(hcount, np.int8),
        integer_source_theta_folded=np.array(True),
        integer_v_code5=((sign < 0).astype(np.uint8)*16+(shift+15).clip(0,15).astype(np.uint8)),
        integer_v_nonzero=nz,
        integer_u_fraction_bits=np.array(14),
        integer_kappa_fp64=np.array([float(x) for x in kappa]),
        integer_full_offset_fp64=np.array([[float(x) for x in row] for row in offsets]))
    mean = original['completion_mean'].astype(np.float64)
    covariance = original['completion_covariance'].astype(np.float64)
    gamma = float(original['gamma'])
    entries, radius_zero, pos_neg_overlap = [], 0, 0
    for t in range(tcount):
        idx = np.flatnonzero(aq[t])
        patterns = np.arange(1 << len(idx))
        alive = 1-((patterns[:, None] >> np.arange(len(idx))) & 1)
        residual_a = alive * (aq[t, idx]/16384.)[None, :]
        mu = residual_a @ mean[idx]
        cov = covariance[:, idx[:, None], idx[None, :]]
        variance = np.einsum('ps,hsu,pu->ph', residual_a, cov, residual_a)
        radius = gamma*np.sqrt(np.maximum(variance, 0.))
        positive, negative = np.zeros(mu.shape, np.int64), np.zeros(mu.shape, np.int64)
        for pattern in patterns:
            for h in range(hcount):
                center = offsets[t][h]+fraction(mu[pattern,h])
                positive[pattern,h] = ceil_fraction((fraction(radius[pattern,h])-center)/kappa[h])
                negative[pattern,h] = ((-fraction(radius[pattern,h])-center)/kappa[h]).__floor__()
        arrays.update({f'integer_empty_indices_t{t}':idx.astype(np.int16),
            f'integer_threshold_pos_t{t}':positive, f'integer_threshold_neg_t{t}':negative,
            f'integer_predictor_mean_t{t}':mu, f'integer_predictor_radius_t{t}':radius})
        entries.append(len(patterns))
        radius_zero += int((radius == 0).sum())
        pos_neg_overlap += int((positive <= negative).sum())
    arrays['numeric_scope'] = np.array(
        'New integer factor deployment: theta-folded U8; exact aligned VQ5; RNE A Q14; '
        'integer Yi/U without intermediate rounding; positive BN folded into integer thresholds. '
        'Predictor uses saved train16 residual moments and Aq-recomputed FP64 constants; '
        'not FP32/QDQ equivalence and not newly calibrated integer residual moments.')
    arrays['integer_accept_rule'] = np.array('positive=Ushared>=pos; negative=Ushared<=neg; positive wins ties; failed uses Ufull>=full')
    arrays['integer_domain_json'] = np.array(json.dumps(domain))
    threshold_arrays = [full_threshold] + [arrays[f'integer_threshold_{s}_t{t}']
                                           for t in range(tcount) for s in ('pos','neg')]
    threshold_bits = signed_width(min(int(x.min()) for x in threshold_arrays),
                                  max(int(x.max()) for x in threshold_arrays))
    pairs = sum(entries)*hcount
    bytes_ = dict(U8_payload=int(u.size), U_scale_exponents_int16=int(e.size*2),
        VQ5_nonzero_payload_bits=int(nz.sum()*5),
        V_dense_nonzero_bitmap_bytes=int((nz.size+7)//8),
        V_H8_payload_zero_word64_bytes=int(nz.reshape(nz.shape[0],-1,8).any(-1).sum()*8),
        A_int16_dense_bytes=int(aq.size*2), A_int16_nonzero_bytes=int(np.count_nonzero(aq)*2),
        full_threshold_software_int64_bytes=int(full_threshold.nbytes),
        predictor_threshold_software_int64_bytes=int(pairs*2*8),
        predictor_threshold_uniform48_payload_bytes=int(pairs*2*6),
        predictor_threshold_actual_width_packed_bits=int(pairs*2*threshold_bits),
        prior_FP32_mean_radius_payload_bytes=int(pairs*2*4),
        P4_R_Zi_actual_bound_payload_bytes=int((4*10*u.shape[1]*domain['Zi_signed_bits']+7)//8),
        P4_H96_Yi_32bit_payload_bytes=int(4*10*hcount*4),
        P4_H96_U_48bit_payload_bytes=int(4*10*hcount*6),
        note='Payload arithmetic only; 64/256-bit row alignment, cache tags/ports and state banking still belong to the service model. Expanded V coefficients in NPZ are software helpers, not the Q5 hardware table.')
    summary = dict(input=str(filename), structure=str(original['structure']),
        numeric_scope=str(arrays['numeric_scope']), source_theta=float(original['theta_source']),
        output_theta=theta, recovery_updates=int(original['recovery_updates']),
        saved_residual_statistics_observations_per_time_channel=int(original['completion_spatial_observations']),
        a_q14_max_abs_change=float(np.max(np.abs(aq/16384.-original['a']))),
        a_support_changes=int(np.count_nonzero((aq != 0) != (original['a'] != 0))),
        source_scale_exponent_range=[int(e.min()),int(e.max())],
        y_exponent_range=[int(eh.min()),int(eh.max())],
        aligned_v_shift_range=[int(aligned_shift[nz].min()),int(aligned_shift[nz].max())],
        compact_table_entries_per_row=entries, compact_table_total_entries=sum(entries),
        predictor_radius_zero_entries=radius_zero, threshold_overlap_entries=pos_neg_overlap,
        threshold_signed_bits=threshold_bits, domain=domain, storage=bytes_)
    return arrays, summary, offsets, kappa


def predict(arrays, ushared, empty):
    pos, neg = np.zeros(ushared.shape, bool), np.zeros(ushared.shape, bool)
    for t in range(10):
        idx = arrays[f'integer_empty_indices_t{t}']
        code = (empty[:,idx].transpose(0,2,1)*(1 << np.arange(len(idx)))).sum(-1)
        pos[:,t] = ushared[:,t] >= arrays[f'integer_threshold_pos_t{t}'][code]
        neg[:,t] = ushared[:,t] <= arrays[f'integer_threshold_neg_t{t}'][code]
    return pos, pos | neg


def functional(arrays, source_directory, offsets, kappa):
    indices = np.array([0,1,2,15,31,32,62,63])
    blocks, provenance = [], []
    for filename in sorted(source_directory.glob('capture_*.npz')):
        with np.load(filename) as data:
            words = data['source_gate_words'][indices].astype(np.int64)
            blocks.append(((words[:,:,:,None] >> np.arange(10)) & 1).transpose(0,3,2,1))
            provenance.append(dict(file=str(data['file']), path=str(filename),
                group_ids=data['group_ids'][indices].tolist()))
    source = np.concatenate(blocks)
    empty = ~source.astype(bool).any(-1)
    u, v, a = arrays['u_int8'].astype(np.int64), arrays['integer_v_align_coeff'], arrays['integer_a_q14'].astype(np.int64)
    shared = int(arrays['shared_rank'])
    z = source @ u
    ysh, yt = z[...,:shared] @ v[:shared], z[...,shared:] @ v[shared:]
    y = ysh+yt
    va = np.einsum('ts,gsph->gtph', a, y)
    shared_u = np.einsum('ts,gsph->gtph', a, ysh)
    q = np.einsum('ts,gspr->gtpr', a, z)
    av, shared_av = q @ v, q[...,:shared] @ v[:shared]
    full_gate = va >= arrays['integer_full_threshold'][None,:,None,:]
    positive, accepted = predict(arrays, shared_u, empty)
    mixed = positive | (~accepted & full_gate)
    # The only pruning information is an already computed shared decision.
    need_y = np.einsum('gtph,ts->gsph', (~accepted).astype(np.int32), (a != 0).astype(np.int32)) > 0
    need_y &= ~empty[...,None]
    support = (v[shared:] != 0).reshape(-1,2,v.shape[1]).any(1)
    need_z = np.einsum('gsph,rh->gspr',need_y.astype(np.int32),support.astype(np.int32)) > 0
    ztail = z[...,shared:]*np.repeat(need_z,2,-1)
    sparse_y = ysh+ztail @ v[shared:]
    sparse_u = np.einsum('ts,gsph->gtph',a,sparse_y)
    sparse_gate = positive | (~accepted & (sparse_u >= arrays['integer_full_threshold'][None,:,None,:]))
    # FP64 represents every integer product/reduction exactly under the
    # published triangle domain. This tests the intended GPU arithmetic too.
    y64 = z.astype(np.float64) @ v.astype(np.float64)
    va64 = np.einsum('ts,gsph->gtph',a.astype(np.float64),y64)
    av64 = np.einsum('ts,gspr->gtpr',a.astype(np.float64),z.astype(np.float64)) @ v.astype(np.float64)
    # Boundary checks use exact rationals of the saved constants, including
    # negative numerators. They are separate from integer-route equality.
    boundary_checks = boundary_errors = 0
    for t in range(10):
        for h in range(v.shape[1]):
            full = int(arrays['integer_full_threshold'][t,h])
            for value in (full-1,full,full+1):
                boundary_checks += 1
                boundary_errors += (value >= full) != (kappa[h]*value+offsets[t][h] >= 0)
            mu, radius = arrays[f'integer_predictor_mean_t{t}'], arrays[f'integer_predictor_radius_t{t}']
            pp, nn = arrays[f'integer_threshold_pos_t{t}'], arrays[f'integer_threshold_neg_t{t}']
            for pattern in range(len(mu)):
                center = offsets[t][h]+fraction(mu[pattern,h])
                rr = fraction(radius[pattern,h])
                for value in (int(pp[pattern,h])-1,int(pp[pattern,h]),int(nn[pattern,h]),int(nn[pattern,h])+1):
                    ptest, ntest = value >= pp[pattern,h], value <= nn[pattern,h]
                    exact = kappa[h]*value+center
                    boundary_checks += 1
                    boundary_errors += (ptest != (exact >= rr)) or (ntest != (exact <= -rr))
    differences = dict(A_then_V_vs_V_then_A=int(np.count_nonzero(av != va)),
        shared_A_then_V_vs_V_then_A=int(np.count_nonzero(shared_u != shared_av)),
        pruned_private_vs_full_fallback_mixed_gate=int(np.count_nonzero(sparse_gate != mixed)),
        FP64_V_then_A_vs_int64=int(np.count_nonzero(va64 != va)),
        FP64_A_then_V_vs_int64=int(np.count_nonzero(av64 != va)),
        rational_threshold_boundary=int(boundary_errors))
    if any(differences.values()):
        raise AssertionError(differences)
    return dict(scope='Four existing actual-source captures, eight fixed native P4 groups each; 32 P4, full T10/K864/H96. No new GPU, network AEE or old-gate equivalence.',
        source=provenance, source_shape=list(source.shape), actual_gate_count=int(mixed.size),
        nonempty_source_TP=int((~empty).sum()), accepted_gates=int(accepted.sum()),
        conditional_vs_this_integer_full_gate_differences=int(np.count_nonzero(mixed != full_gate)),
        needed_tail_Z_values=int(np.repeat(need_z,2,-1).sum()),
        actual_maxabs_Zi=int(np.abs(z).max()),actual_maxabs_Yi=int(np.abs(y).max()),
        actual_maxabs_Ui=int(np.abs(va).max()),
        threshold_boundary_cases=boundary_checks,differences=differences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=HERE/'flow_recovery64')
    parser.add_argument('--output',type=Path,default=HERE/'integer_factors')
    parser.add_argument('--source',type=Path,default=HERE.parent.parent/'partial_completion'/'integer_valid10')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result = dict(scope='CPU integer compilation and actual-source local numerical checks; new deployment function, no service or PPA result.',models={})
    for name in NAMES:
        arrays, summary, offsets, kappa = compile_model(args.input/(name+'.npz'))
        summary['functional'] = functional(arrays,args.source,offsets,kappa)
        output = args.output/(name+'.npz')
        np.savez_compressed(output,**arrays)
        summary['output'] = str(output)
        result['models'][name] = summary
        print(name,json.dumps(dict(domain=summary['domain'],threshold_bits=summary['threshold_signed_bits'],
                                   functional=summary['functional']['differences'])),flush=True)
    result['complete'] = True
    (args.output/'compile_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
