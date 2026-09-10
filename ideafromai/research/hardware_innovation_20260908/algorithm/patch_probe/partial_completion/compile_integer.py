"""Common integer format for row34 and common-column conditional production.

Quantization defines new students. The integer predicates are compiled from
explicit affine-BN and frozen residual constants, not claimed equivalent to
the original FP32 GPU BN. No fitted thresholds use validation observations.
"""
from fractions import Fraction as F
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
Q = 1 << 14


def compile_pair(rate, center, radius, bound):
    """p=rate*U+center; positive iff p>=radius, negative iff p<-radius."""
    if rate > 0:
        positive = math.ceil((radius-center)/rate)
        negative = math.ceil((-radius-center)/rate)-1
    else:
        positive = math.floor((radius-center)/rate)
        negative = math.floor((-radius-center)/rate)+1
    # Outside the statically legal accumulator range, the adjacent sentinel
    # has exactly the same decision. No huge threshold register is required.
    low, high = -bound-1, bound+1
    return min(high,max(low,positive)), min(high,max(low,negative))


def compile_variant(name, params, source, wq, wscale, means, covariance, const_bits):
    original_a = np.asarray(params['weight'], dtype=np.float64)
    rounded = np.rint(original_a*Q)
    aq = np.clip(rounded,-32768,32767).astype(np.int16)
    a = aq.astype(np.float64)/Q
    bias = np.asarray(params['bias'], dtype=np.float64).reshape(10)
    theta = float(source['output_theta'])
    bn_bias = source['bn_bias']
    cq = 1 << const_bits
    offset_i = np.rint(bn_bias*cq).astype(np.int64)
    bias_i = np.rint(bias*cq).astype(np.int64)
    theta_i = int(np.rint(theta*cq))
    shifts = np.rint(np.log2(wscale)).astype(np.int32)+const_bits
    source_bound = np.abs(wq.astype(np.int64)).reshape(96,-1).sum(1)
    u_bound = np.abs(aq.astype(np.int64)).sum(1)[:,None]*source_bound[None]
    sign = (source_bound != 0).astype(np.int8)
    full_sums = aq.astype(np.int64).sum(1)
    base_threshold = (theta_i-bias_i)[:,None]*Q-full_sums[:,None]*offset_i[None]
    constant_gate = np.zeros((10,96),dtype=bool)
    slots, entry_row, entry_mask, entry_observed_qsum = [], [], [], []
    global_to_entry = np.zeros((1024,10),dtype=np.int16)
    full_entry = []
    for t in range(10):
        columns = np.flatnonzero(aq[t])
        start = len(slots)
        for local in range(1 << len(columns)):
            observed = [s for j,s in enumerate(columns) if local & (1 << j)]
            mask = sum(1 << int(s) for s in observed)
            slots.append((t,observed))
            entry_row.append(t)
            entry_mask.append(mask)
            entry_observed_qsum.append(int(aq[t,observed].astype(np.int64).sum()))
        full_entry.append(len(slots)-1)
        for seen in range(1024):
            local = sum(((seen >> int(s)) & 1) << j for j,s in enumerate(columns))
            global_to_entry[seen,t] = start+local
    pos = np.zeros((len(slots),96),dtype=np.int64)
    neg = np.zeros_like(pos)
    residual_mean, residual_radius, sum_remaining = [], [], []
    boundary_checks = 0
    predicate_abs_bound = 0
    for entry,(t,observed) in enumerate(slots):
        remaining = a[t].copy()
        remaining[observed] = 0.
        mu = float(remaining @ means)
        radius = 3.*math.sqrt(max(0.,float(remaining @ covariance @ remaining)))
        mu_i = int(np.rint(mu*cq))
        rad_i = math.ceil(radius*cq)
        residual_mean.append(mu_i)
        residual_radius.append(rad_i)
        sum_remaining.append(int(full_sums[t])-entry_observed_qsum[entry])
        for h in range(96):
            rate = F(1 << int(shifts[h]))
            center = F((int(bias_i[t])+mu_i-theta_i)*Q+
                       int(offset_i[h])*entry_observed_qsum[entry])
            rad = F(rad_i*Q)
            if not sign[h]:
                constant_gate[t,h] = base_threshold[t,h] <= 0
                continue
            bound = int(u_bound[t,h])
            predicate_abs_bound=max(predicate_abs_bound,int(rate*bound+abs(center)+rad))
            positive, negative = compile_pair(rate,center,rad,bound)
            pos[entry,h],neg[entry,h] = positive,negative
            # Independent direct rational affine evaluation at both compiled
            # boundaries, including equality, and at legal range endpoints.
            values = {-bound,bound,0,positive-1,positive,positive+1,
                      negative-1,negative,negative+1}
            for value in values:
                if not -bound <= value <= bound:
                    continue
                p = rate*value+center
                factored = ((value << int(shifts[h]))-int(base_threshold[t,h])+
                            (mu_i << 14)-int(offset_i[h])*sum_remaining[entry])
                if p != factored:
                    raise RuntimeError('BN offset factorization differs from observed-column reference')
                expected = (factored >= rad,factored < -rad)
                actual = ((value >= positive,value <= negative) if rate > 0 else
                          (value <= positive,value >= negative))
                if actual != expected:
                    raise RuntimeError((name,t,h,entry,value,actual,expected))
                boundary_checks += 1
    output = HERE/'integer_deployment'
    output.mkdir(exist_ok=True)
    np.savez_compressed(output/(name+'.npz'),weight_int8=wq,
        weight_scale=wscale,temporal_int16=aq,temporal_fractional_bits=np.array(14),
        threshold_positive=pos,threshold_negative=neg,positive_gain=sign,
        constant_gate=constant_gate,entry_row=np.asarray(entry_row,dtype=np.int8),
        entry_observed_mask=np.asarray(entry_mask,dtype=np.int16),
        entry_observed_qsum=np.asarray(entry_observed_qsum,dtype=np.int32),
        entry_mean_int=np.asarray(residual_mean,dtype=np.int64),entry_radius_int=np.asarray(residual_radius,dtype=np.int64),
        entry_remaining_qsum=np.asarray(sum_remaining,dtype=np.int32),
        state_to_entry=global_to_entry,full_entry=np.asarray(full_entry,dtype=np.int16),
        temporal_bias=bias,original_bn_scale=source['bn_scale'],bn_bias=bn_bias,
        constant_fractional_bits=np.array(const_bits),bn_offset_int=offset_i,
        temporal_bias_int=bias_i,theta_decision_int=np.array(theta_i),
        base_threshold=base_threshold,accumulator_left_shift=shifts,
        theta_source=source['source_theta'],theta_output=source['output_theta'],
        Y_abs_bound=source_bound,U_abs_bound=u_bound)
    return dict(name=name,actual_connections=int(np.count_nonzero(aq)),
        rank_after_A_quantization=int(np.linalg.matrix_rank(a)),
        saturated_A_coefficients=int(np.count_nonzero(rounded!=aq)),
        temporal_int16_min=int(aq.min()),temporal_int16_max=int(aq.max()),
        residual_entries=len(slots),
        Y_abs_bound=int(source_bound.max()),U_abs_bound=int(u_bound.max()),
        fits_INT24_Y=bool(source_bound.max() < 2**23),fits_INT48_U=bool(u_bound.max() < 2**47),
        predicate_abs_bound=predicate_abs_bound,fits_INT48_predicate=predicate_abs_bound < 2**47,
        threshold_abs_max=int(max(np.abs(pos).max(),np.abs(neg).max())),
        positive_gain_channels=int((sign>0).sum()),negative_gain_channels=int((sign<0).sum()),
        zero_gain_channels=int((sign==0).sum()),
        rational_boundary_cases=boundary_checks,
        uncompressed_two_threshold_table_bytes_at48bits=int(pos.size*2*6),
        factorized_storage_bytes_at_48bit_base_32bit_constants=int(960*6+(3*len(slots)+96)*4+96),
        factorized_expression='Z=(Ui << shift_h)-base_t_h+(mean_R_int << 14)-bn_offset_h_int*sum_R_Aq; positive if Z>=radius_R_int<<14, negative if Z<-(radius_R_int<<14)',
        state_addressing='physical row-local bit selection; exported 1024x10 LUT is GPU indexing convenience, not an admitted ROM',
        format_note='INT24/INT48 bounds prove capacity only; no circuit timing or PPA; literal table storage is one implementation, not an optimized baseline')


def main():
    source=np.load(HERE/'shared_column_deployment_source.npz')
    # Ordinary fixed-BN gain folding is granted to BOTH structures before
    # quantization. Remaining per-channel gain is now a dyadic shift.
    weight=(source['weight'].astype(np.float64)*float(source['source_theta'])*
            source['bn_scale'][:,None,None,None])
    peak=np.abs(weight).reshape(96,-1).max(1)
    exponents=np.zeros(96,dtype=np.int32)
    nonzero=peak!=0
    exponents[nonzero]=np.ceil(np.log2(peak[nonzero]/127.)).astype(np.int32)
    scales=np.ldexp(np.ones(96),exponents)
    const_bits=max(16,int(-exponents.min()))
    wq=np.clip(np.rint(weight/scales[:,None,None,None]),-127,127).astype(np.int8)
    row34=json.loads((HERE.parent/'dependency/fit.json').read_text())['variants']['row34']
    common=json.loads((HERE/'shared_column_result.json').read_text())['selected']['common3_diagonal_34']
    moments=np.load(HERE.parent/'dependency/train_moments.npz')
    results=[compile_variant(name,params,source,wq,scales,moments['mean'],moments['covariance'],const_bits)
             for name,params in [('row34',row34),('common3_diagonal_34',common)]]
    # The selected model may have no negative BN gains; exercise the reversed
    # strict/equality branch separately without pretending negative payloads
    # were observed in this checkpoint.
    signed_checks=0
    for rate in [F(3,8),F(-3,8)]:
        for center in [F(-3,4),F(0),F(5,4)]:
            for radius in [F(0),F(1,2),F(2)]:
                p,n=compile_pair(rate,center,radius,32)
                for value in range(-32,33):
                    margin=rate*value+center
                    expect=(margin>=radius,margin < -radius)
                    actual=((value>=p,value<=n) if rate>0 else (value<=p,value>=n))
                    if actual!=expect:
                        raise RuntimeError('signed/equality boundary failure')
                    signed_checks+=1
    report=dict(
        format='source theta*g -> BN-gain-folded dyadic per-H INT8 W -> INT24 Yi -> signed INT16 Q14 A / INT48 Ui -> fixed-constant shift/offset predicate -> theta*g',
        source='loaded current isolated patch parameters; bn_scale/bias explicitly computed in Float64 before quantization',
        constant_fractional_bits=const_bits,
        numerical_definition='A and BN-folded W are quantized new students. BN offset, temporal bias, decision theta and residual mean round even to the stated common constant format; radius rounds upward. Exact integer ceiling/floor thresholds preserve these integer predicates. Original emitted theta amplitude is retained.',
        full_gate='remaining empty: positive or negative decision covers every legal Ui with no uncertain gap',
        statistical_policy='gamma3; residual mean/covariance inherited from original train32 Y and recomputed with each quantized A; not recalibrated on quantized Conv; lossy early acceptance',
        ordinary_control='both row34 and common3 use identical BN-folded W quantization, bit widths, compiler, raw-zero handling and residual estimator',
        weight_zero_fraction=float((wq==0).mean()),weight_scale_min=float(scales.min()),weight_scale_max=float(scales.max()),
        variants=results,signed_boundary_diagnostic_cases=signed_checks,
        validation='rational boundary checks only so far, no new network AEE or RTL equivalence',
        exclusions=['full-network accuracy in this integer format','finite bank/port service and output backpressure',
                    'threshold table ports, factorization and state/weight allocation','VCS/DC/PT/Formality/PPA'])
    (HERE/'integer_deployment/compile_result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
