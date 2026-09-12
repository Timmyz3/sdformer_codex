"""Fixed shared time decoder on the actual updated-I24 / proj-gate fork.

No training, parameter/rank sweep, GPU, or cycle claim. D and its intercept
are rounded ONCE to integer I24-state units. Exact residuals are retained
without saturation; binary and residual U sums merge before original RNE.
"""
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
NEW=HERE.parent
BASE=NEW.parents[1]
FULL=BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
sys.path.insert(0,str(NEW))
from rebase_probe import execute
from probe import rne24
T,K,R,H=10,96,32,96
WINDOWS={'corner':(0,0),'interior':(120,160)}


def signed_width(a):
    lo,hi=int(np.min(a)),int(np.max(a))
    return max(1,max(hi,~lo,0).bit_length()+1)


def gate_words(words,ys,xs):
    w=words[:,ys,:][:,:,xs]
    return ((w[None]>>np.arange(T)[:,None,None,None])&1).astype(np.int64).reshape(T,K,-1)


def data_window(full,ys,xs):
    return full[:,:,ys,:][:,:,:,xs].astype(np.int64).reshape(T,K,-1)


def flatten(a):
    return a.transpose(1,0,2).reshape(a.shape[1],-1)


def unflatten(a,p):
    return a.reshape(a.shape[0],T,p).transpose(1,0,2)


def bit_statistics(a):
    # Per (channel, spatial point) T10 word. Highest bit is NEGATIVE.
    x=a.reshape(T,-1)
    stats=dict(elements=int(a.size),minimum=int(a.min()),maximum=int(a.max()),
        signed_width_global=signed_width(a),zero_values=int(np.count_nonzero(a==0)),
        zero_fraction=float(np.mean(a==0)),
        outside_signed24=int(np.count_nonzero((a<-(1<<23))|(a>=(1<<23)))),
        widths={},bitplane_rows=0,nonempty_bitplane_rows=0,set_bit_events=0,
        majority_correction_events=0,majority_default_groups=0,
        negative_high_bit_events=0,exact_bitplane_reconstruction_differences=0)
    reconstructed=np.zeros_like(x)
    widths=[]
    for j in range(x.shape[1]):
        col=x[:,j]
        w=signed_width(col)
        widths.append(w);stats['widths'][str(w)]=stats['widths'].get(str(w),0)+1
        stats['bitplane_rows']+=w
        for bit in range(w):
            b=(col>>bit)&1
            pop=int(b.sum())
            stats['nonempty_bitplane_rows']+=int(pop>0)
            stats['set_bit_events']+=pop
            stats['majority_correction_events']+=min(pop,T-pop)
            stats['majority_default_groups']+=int(pop>T//2)
            coefficient=-(1<<bit) if bit==w-1 else (1<<bit)
            if bit==w-1:stats['negative_high_bit_events']+=pop
            reconstructed[:,j]+=coefficient*b
    stats['exact_bitplane_reconstruction_differences']=int(np.count_nonzero(reconstructed!=x))
    assert np.array_equal(reconstructed,x)
    stats['mean_T10_signed_width']=float(np.mean(widths))
    stats['serial_bitplane_MAC8_or_AAC8_issues']=stats['set_bit_events']*(R//8)
    stats['majority_shared_AAC8_plus_default_flush_issues']=(
        stats['majority_correction_events']+2*stats['majority_default_groups'])*(R//8)
    # An ideal per-channel/P2 width-coded packet, including one width byte.
    p=a.shape[2]
    assert p%2==0
    packed=0
    for p0 in range(0,p,2):
        for k in range(K):
            block=a[:,k,p0:p0+2]
            packed+=(20*signed_width(block)+7)//8+1
    stats['ideal_P2_T10_width_payload_bytes_including_1B_header']=packed
    stats['raw_I24_packed_bytes']=int(a.size*3)
    return stats


def vector_issues(a,u):
    output_channels,input_channels=u.shape
    groups=np.any(u.reshape(output_channels//8,8,input_channels)!=0,axis=1)
    return int(sum(np.count_nonzero(a[:,k])*int(groups[:,k].sum()) for k in range(input_channels)))


def cost(x,g,b,res,u,D,c,z,v):
    """Issue/operand accounting, deliberately not a scheduled cycle model.

    Fused-coefficient binary U(Dg): weight K[t,s,h,k]=D[t,s]*U[h,k].
    The original gate convolution has DIFFERENT weights and is NOT credited.
    Also expose the smaller-fanout factored implementation U*g then D*(U*g),
    whose middle precision exceeds the existing signed24 MAC input in general.
    """
    p=x.shape[2]
    dnz=D!=0
    groups=np.any(u.reshape(R//8,8,K)!=0,axis=1)
    raw=vector_issues(x,u);continuous=vector_issues(res,u)
    decoder_updates=0;binary=0;decoder_nonzero_constants=int(np.count_nonzero(c))
    coefficient_bits=0;fused_coefficient_bytes=0;fused_words_per_once_read=0
    max_combined_coefficient_width=1
    for t in range(T):
        for s in range(T):
            if D[t,s]==0:continue
            decoder_updates+=int(g[s].sum())
            for k in range(K):
                if not np.any(g[s,k]):continue
                w=D[t,s]*u[:,k]
                bits=signed_width(w)
                max_combined_coefficient_width=max(max_combined_coefficient_width,bits)
                fused_coefficient_bytes+=(R*bits+7)//8
                fused_words_per_once_read+=(R*bits+255)//256
                binary+=int(g[s,k].sum())*int(groups[:,k].sum())
    const_rows=int(np.count_nonzero(c))
    constant_Umerge=const_rows*p*(R//8)
    # Best possible vector packing for decoder/subtraction; no decode/index fee.
    is_raw=not np.any(D) and not np.any(c)
    decode8=(decoder_updates+7)//8
    init8=(const_rows*K*p+7)//8
    subtract8=0 if is_raw else (int(x.size)+7)//8
    merge8=T*p*(R//8)
    v_mac=vector_issues(unflatten(z,p),v)
    gu=vector_issues(g,u) if np.any(D) else 0
    factored_time_mul8=int(dnz.sum())*p*(R//8)
    # 32 columns *10 times per point of U*g; 48-bit storage, no truncation.
    gu_state_bytes=T*p*R*6
    original_coeff_bytes=u.size*2
    raw_coeff_words=(original_coeff_bytes+31)//32
    gate_operand_bytes=K*p*2
    raw_source_bytes=int(x.size*3)
    c_u=(u.sum(axis=1)[None,:,None]*c[:,None,None])
    c_u_width=signed_width(c_u) if c_u.size else 1
    body=dict(
        scope='necessary MAC8/AAC8 arithmetic work and optimistic packed operands, NOT cycles or RTL',
        raw_U_MAC8_issues=raw,residual_U_MAC8_issues=continuous,
        original_V_MAC8_issues_common=v_mac,
        original_V_dense_MAC8_issues_reference=T*p*(R*H//8),
        decoder_Dg_scalar_adds=decoder_updates,
        decoder_integer_ADD8_ideal_issues=decode8,
        decoder_constant_initialization_ADD8_ideal_issues=init8,
        residual_formation_SUB8_ideal_issues=subtract8,
        binary_U_Dg_fused_AAC8_issues=binary,
        raw_U_bias_constant_merge_ADD8_issues=constant_Umerge,
        raw_accumulator_merge_ADD8_issues=merge8 if np.any(D) else 0,
        binary_fused_coefficient_max_signed_width=max_combined_coefficient_width,
        binary_fused_coeff_payload_once_per_live_s_k_t_bytes=fused_coefficient_bytes,
        binary_fused_CR256_once_per_live_s_k_t_transactions=fused_words_per_once_read,
        raw_U_coefficient_payload_bytes=original_coeff_bytes,
        raw_U_CR256_ideal_once_transactions=raw_coeff_words,
        original_x_read_bytes_common=raw_source_bytes,
        additional_proj_gate_read_bytes_if_not_forwarded=gate_operand_bytes if np.any(D) else 0,
        D_plus_intercept_I32_storage_bytes=(int((D.size+c.size)*4) if np.any(D) else (0 if is_raw else int(c.size*4))),
        preprojected_constant_signed_width=c_u_width,
        preprojected_constant_packed_bytes_once=(T*R*c_u_width+7)//8 if np.any(c) else 0,
        preprojected_constant_CR256_transactions_once=(T*R*c_u_width+255)//256 if np.any(c) else 0,
        factored_binary_Ug_AAC8_issues=gu,
        factored_D_Ug_wide_MUL8_issues=factored_time_mul8,
        factored_Ug_48bit_state_bytes_if_materialized=gu_state_bytes,
        decoding_and_residual_live_state_note='Perfect on-producer subtraction assumed; no second x/r SRAM traversal charged. Actual state/ports cannot be inferred from this lower bound.',
        fused_binary_read_note='Each live s/k/t coefficient row read once across every spatial point: optimistic unbounded coefficient reuse, not a legal 96RF schedule.',
        negative_high_bit_preserved=True,gate_convolution_compute_saved=0)
    overhead=decode8+init8+subtract8+constant_Umerge+(merge8 if np.any(D) else 0)
    body['fused_U_total_arithmetic_ideal_issues']=continuous+binary+overhead
    body['fused_UV_total_arithmetic_ideal_issues']=continuous+binary+overhead+v_mac
    body['raw_UV_total_MAC8_issues']=raw+v_mac
    body['fused_UV_arithmetic_ratio_vs_raw']=(continuous+binary+overhead+v_mac)/(raw+v_mac)
    body['factored_wide_arithmetic_ideal_issues']=continuous+gu+factored_time_mul8+overhead+v_mac
    body['factored_wide_ratio_vs_raw']=(continuous+gu+factored_time_mul8+overhead+v_mac)/(raw+v_mac)
    return body


def full_anchor_ranges(full,words,D,c,constant):
    """Same captured frame: complete actual stride2-anchor range, not replay.

    Only b+r identity and residual/source ranges; U/V functional comparison
    remains the fixed calibration and disjoint original windows below.
    """
    modes={'raw_I24':(np.zeros_like(D),np.zeros_like(c)),
           'constant_offset_residual':(np.zeros_like(D),constant),
           'shared_Dg_residual':(D,c)}
    rows={m:dict(values=0,minimum=2**63-1,maximum=-2**63,zeros=0,
        outside_signed24=0,identity_reconstruction_differences=0) for m in modes}
    xs=np.arange(0,320,2)
    for y0 in range(0,240,16):
        ys=np.arange(y0,min(y0+16,240),2)
        x=data_window(full,ys,xs);g=gate_words(words,ys,xs)
        for mode,(dm,cm) in modes.items():
            b=(dm@g.reshape(T,-1)+cm[:,None]).reshape(x.shape)
            residual=x-b
            assert np.array_equal(b+residual,x)
            a=rows[mode]
            a['values']+=int(residual.size)
            a['minimum']=min(a['minimum'],int(residual.min()))
            a['maximum']=max(a['maximum'],int(residual.max()))
            a['zeros']+=int(np.count_nonzero(residual==0))
            a['outside_signed24']+=int(np.count_nonzero((residual<-(1<<23))|(residual>=(1<<23))))
    for row in rows.values():
        row['zero_fraction']=row['zeros']/row['values']
        row['global_signed_width']=signed_width(np.asarray([row['minimum'],row['maximum']]))
    return rows


def evaluate(x,g,u,v,bias,D,c,gold=None):
    p=x.shape[2]
    b=(D@g.reshape(T,-1)+c[:,None]).reshape(x.shape)
    residual=x-b
    assert np.array_equal(b+residual,x)
    ur=u@flatten(residual)
    ub=u@flatten(b)
    original=u@flatten(x)
    # Independent algebra check via binary U*g and time D. Integer D has no
    # intervening decode RNE, so this particular commute is exactly legal.
    ug=unflatten(u@flatten(g),p)
    ub_factored=(D@ug.reshape(T,-1)).reshape(T,R,p)
    ub_factored+=u.sum(axis=1)[None,:,None]*c[:,None,None]
    assert np.array_equal(flatten(ub_factored),ub)
    merged=ur+ub
    assert np.array_equal(merged,original)
    for a in (ur,ub,merged):
        assert np.min(a)>=-(1<<47) and np.max(a)<(1<<47)
    z=rne24(merged,16)
    vy=v@z
    assert np.min(vy)>=-(1<<47) and np.max(vy)<(1<<47)
    out=rne24(rne24(vy,15)+bias[:,None],0)
    reference,counts=execute(u,v,flatten(x),bias)
    assert np.array_equal(out,reference)
    if gold is not None:assert np.array_equal(out,gold)
    return dict(
        input_values=int(x.size),output_values=int(out.size),
        identity_reconstruction_differences=0,U_raw_accumulator_differences=0,
        separately_factored_binary_U_Dg_differences=0,original_output_differences=0,
        captured_output_differences=0 if gold is not None else None,
        decoder_b_range=[int(b.min()),int(b.max())],
        original_U_acc_range=[int(original.min()),int(original.max())],
        residual_U_acc_range=[int(ur.min()),int(ur.max())],
        binary_U_acc_range=[int(ub.min()),int(ub.max())],
        factored_Ug_signed_width=signed_width(ug),
        raw=bit_statistics(x),residual=bit_statistics(residual),reference_saturation_counts=counts,
        cost=cost(x,g,b,residual,u,D,c,z,v))


def main():
    report=dict(
        scope=__doc__,gate='full_proj_words, observed from proj.sn on the SAME updated source consumed by continuous PED; no sn1/sn2 substitution',
        identity='AT-LIF {0,theta}; theta folded into gate-branch W. The continuous source is signed24 I24, never treated as binary.',
        calibration=dict(y=list(range(32,96,4)),x=list(range(32,96,4)),points=256,
            scope='one fixed frame, spatial calibration only; no held-out frame claim'),
        decoder='one shared 10x10 linear time decoder + 10 intercepts per student, fit LS then round coefficients to integer I24 units once; no clipping/format/rank sweep',
        control='raw I24 and same-calibration per-time mean offset. Both use original signed16 U32/V96, exponents16/15, both original RNE/sat24 and PED bias.',
        exact_merge='integer residual + binary partial sum merge BEFORE original U RNE, then ONE original V/RNE/bias',
        model_limit='Arithmetic/packed operand lower bounds only; no scheduled cycles, VCS, PPA, network AEE, training.',
        axes={})
    ys=np.arange(32,96,4);xs=np.arange(32,96,4)
    for axis in ('ordinary','lifting_raw'):
        print('loading '+axis,flush=True)
        with np.load(FULL/'capture'/axis/'parameters.npz') as z:
            u=z['U_ped_q16'].astype(np.int64);v=z['V_ped_q16'].astype(np.int64);bias=z['PED_bias_q24'].astype(np.int64)
        with np.load(FULL/'capture_full_producers'/axis/'000_zurich_city_09_a_0001.npz') as z:
            full=z['full_updated_I24'];words=z['full_proj_words']
        with np.load(FULL/'capture'/axis/'000_zurich_city_09_a_0001.npz') as z:
            gold={label:z[label+'_continuous_q24'].transpose(1,0,2,3).reshape(H,-1) for label in WINDOWS}
        xc=data_window(full,ys,xs);gc=gate_words(words,ys,xs)
        design=np.concatenate([gc.reshape(T,-1).T,np.ones((K*len(ys)*len(xs),1))],axis=1)
        target=xc.reshape(T,-1).T
        fit,_,rank,singular=np.linalg.lstsq(design,target,rcond=None)
        D=np.rint(fit[:T].T).astype(np.int64);c=np.rint(fit[T]).astype(np.int64)
        constant=np.rint(target.mean(0)).astype(np.int64)
        np.savez_compressed(HERE/(axis+'_decoder.npz'),D_integer=D,c_integer=c,constant_control=constant,
            calibration_y=ys,calibration_x=xs)
        record=dict(design_rank=int(rank),design_singular_values=singular.tolist(),
            D_nonzero=int(np.count_nonzero(D)),D_signed_width=signed_width(D),c_signed_width=signed_width(c),
            gate_density_calibration=float(gc.mean()),windows={})
        record['same_frame_full_stride2_anchor_ranges']=full_anchor_ranges(full,words,D,c,constant)
        for label,(wy,wx) in [('calibration',(None,None)),*list(WINDOWS.items())]:
            if label=='calibration':x,g=xc,gc
            else:
                ay=np.arange(wy,wy+8,2);ax=np.arange(wx,wx+8,2)
                x,g=data_window(full,ay,ax),gate_words(words,ay,ax)
            modes={}
            for mode,dm,cm in (
                ('raw_I24',np.zeros((T,T),np.int64),np.zeros(T,np.int64)),
                ('constant_offset_residual',np.zeros((T,T),np.int64),constant),
                ('shared_Dg_residual',D,c)):
                modes[mode]=evaluate(x,g,u,v,bias,dm,cm,gold.get(label))
            # Raw arm does not actually decode/subtract/merge zeros.
            modes['raw_I24']['cost']['executed_raw_control']=True
            record['windows'][label]=dict(gate_density=float(g.mean()),modes=modes)
            print(axis+' '+label+' '+json.dumps({m:dict(zero=r['residual']['zero_fraction'],
                width=r['residual']['mean_T10_signed_width'],
                ratio=r['cost']['fused_UV_arithmetic_ratio_vs_raw'],
                wide_ratio=r['cost']['factored_wide_ratio_vs_raw']) for m,r in modes.items()}),flush=True)
        report['axes'][axis]=record
        del full,words
    (HERE/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
