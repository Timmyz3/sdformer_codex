"""Three fixed lossy PED source representations on saved real producer data.

CPU opportunity and exact integer output comparisons only; not AEE/cycles.
D and its intercept are existing single-frame-fit parameters, never refit here.
"""
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
OPEN=HERE.parents[1];BASE=OPEN.parent
NEW=OPEN/'new_interface_selection'
FULL=BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
sys.path.insert(0,str(NEW))
from rebase_probe import execute

MODES=('i24_q8_s11','Dg_q8_s11','Dg_only')
STEP=1<<11


def round_step(a):
    quotient,remainder=np.divmod(a,STEP)
    quotient+=((2*remainder>STEP)|((2*remainder==STEP)&((quotient&1)!=0)))
    return quotient


def q8(a):
    return np.clip(round_step(a),-128,127)


def representation(x,g,D,c,mode):
    b=(D@g.reshape(10,-1)+c[:,None]).reshape(x.shape)
    values=x if mode=='i24_q8_s11' else x-b
    quant=np.zeros_like(values) if mode=='Dg_only' else q8(values)
    estimate=(np.zeros_like(b) if mode=='i24_q8_s11' else b)+quant*STEP
    return np.clip(estimate,-(1<<23),(1<<23)-1),quant,b


def flatten(x):return x.transpose(1,0,2).reshape(x.shape[1],-1)


def main():
    out=HERE/'representations';out.mkdir(exist_ok=True)
    report=dict(inference=False,AEE_measured=False,training=False,quant_step_state_units=STEP,
        quant_step_physical=STEP/(1<<14),quant_range=[-128,127],RNE=True,
        source_gate='Actual same updated-I24 producer full_proj_words; no sn1/sn2 or future flow.',
        decoder='Existing D_integer/c_integer fitted on256 anchors of one old valid calibration frame; no refitting.',
        scope='Complete anchor-source statistics plus actual integer R24 U/V outputs for two fixed local4x4 windows; not whole-layer schedule.',axes={})
    for axis,rank_mode in [('ordinary','original_ordered'),('lifting_raw','activation_whitened')]:
        capture=FULL/'capture_full_producers'/axis/'000_zurich_city_09_a_0001.npz'
        with np.load(capture) as z:full=z['full_updated_I24'];words=z['full_proj_words']
        with np.load(NEW/'spike_residual_probe'/(axis+'_decoder.npz')) as z:decoder={k:z[k] for k in z.files}
        D,c=decoder['D_integer'],decoder['c_integer'];np.savez_compressed(out/(axis+'_decoder.npz'),**decoder)
        with np.load(NEW/(axis+'_rebase_parameters.npz')) as z:u=z[rank_mode+'_U'][:24].astype(np.int64);v=z[rank_mode+'_V'][:,:24].astype(np.int64);bias=z['PED_bias_q24']
        totals={m:dict(elements=0,quant_zero=0,quant_clipped=0,empty_P2_T10_source_words=0,P2_T10_source_words=0,
            input_abs_error_sum=0,input_max_abs_error=0,decoded_min=2**63-1,decoded_max=-(2**63),reconstruction_clipped24=0) for m in MODES}
        for y in range(0,240,16):
            x=full[:,:,y:y+16:2,::2].astype(np.int64)
            w=words[:,y:y+16:2,::2];g=((w[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
            for mode in MODES:
                estimate,quant,b=representation(x,g,D,c,mode);a=totals[mode]
                raw=x if mode=='i24_q8_s11' else x-b
                a['elements']+=x.size;a['quant_zero']+=int(np.count_nonzero(quant==0))
                rounded=round_step(raw)
                a['quant_clipped']+=int(np.count_nonzero((rounded < -128)|(rounded > 127))) if mode!='Dg_only' else 0
                groups=quant.reshape(10,96,x.shape[2],80,2)
                a['empty_P2_T10_source_words']+=int(np.count_nonzero(~np.any(groups,axis=(0,4))))
                a['P2_T10_source_words']+=96*x.shape[2]*80
                error=np.abs(estimate-x);a['input_abs_error_sum']+=int(error.sum());a['input_max_abs_error']=max(a['input_max_abs_error'],int(error.max()))
                a['decoded_min']=min(a['decoded_min'],int(estimate.min()));a['decoded_max']=max(a['decoded_max'],int(estimate.max()))
                merged=(np.zeros_like(b) if mode=='i24_q8_s11' else b)+quant*STEP
                a['reconstruction_clipped24']+=int(np.count_nonzero((merged<-(1<<23))|(merged>=(1<<23))))
        window_results={}
        for label,(y,x0) in [('corner',(0,0)),('interior',(120,160))]:
            x=full[:,:,y:y+8:2,x0:x0+8:2].astype(np.int64).reshape(10,96,-1)
            w=words[:,y:y+8:2,x0:x0+8:2].reshape(96,-1)
            g=((w[None]>>np.arange(10)[:,None,None])&1).astype(np.int64)
            gold,_=execute(u,v,flatten(x),bias)
            checks={}
            for mode in MODES:
                estimate,quant,b=representation(x,g,D,c,mode);actual,clips=execute(u,v,flatten(estimate),bias)
                e=actual.astype(np.float64)-gold
                checks[mode]=dict(output_values=actual.size,changed_values=int(np.count_nonzero(e)),
                    output_q24_RMSE=float(np.sqrt(np.mean(e*e))),output_q24_max_abs=float(np.abs(e).max()),
                    output_physical_RMSE=float(np.sqrt(np.mean(e*e)))/(1<<14),clip_counts=clips)
                if mode=='Dg_only':
                    # New R24 actually uses the SAME producer gate. There is no
                    # intermediate decode rounding; the existing U RNE remains
                    # after the complete D(Ug)+Uc sum.
                    points=x.shape[2];ug=(u@flatten(g)).reshape(24,10,points).transpose(1,0,2)
                    raw=(D@ug.reshape(10,-1)).reshape(10,24,points)+c[:,None,None]*u.sum(1)[None,:,None]
                    direct=u@flatten(estimate)
                    assert np.array_equal(flatten(raw),direct)
                    checks[mode]['factorized_U_Dg_integer_accumulator_differences']=0
                    checks[mode]['Ug_observed_max_abs']=int(np.abs(ug).max())
                    checks[mode]['D_Ug_plus_Uc_observed_max_abs']=int(np.abs(raw).max())
            window_results[label]=checks
        for mode,a in totals.items():
            a['quant_zero_fraction']=a['quant_zero']/a['elements'];a['empty_P2_T10_fraction']=a['empty_P2_T10_source_words']/a['P2_T10_source_words']
            a['input_physical_MAE']=a['input_abs_error_sum']/a['elements']/(1<<14)
            a['stored_quant_payload_bytes']=0 if mode=='Dg_only' else a['elements']
            a['original_I24_payload_bytes']=a['elements']*3
            a['D_constant_bytes']=0 if mode=='i24_q8_s11' else int((D.size+c.size)*4)
            a['gate_payload_bytes_if_not_forwarded']=0 if mode=='i24_q8_s11' else int(a['elements']/10*2)
        ug_bound=np.abs(u).sum(1)
        decoded_bound=np.abs(D).sum(1)+np.abs(c)
        projected_bound=(np.abs(D).sum(1)[:,None]*ug_bound[None,:]+np.abs(c)[:,None]*np.abs(u.sum(1))[None,:])
        assert np.max(decoded_bound)<(1<<23) and np.max(projected_bound)<(1<<47)
        report['axes'][axis]=dict(PED_R24_mode=rank_mode,D_nonzero=int(np.count_nonzero(D)),
            decoder_file=axis+'_decoder.npz',D_max_abs=int(np.abs(D).max()),statistics=totals,windows=window_results,
            Dg_only_factorization=dict(decoded_abs_bound_all_binary_g=int(decoded_bound.max()),
                Ug_abs_bound=int(ug_bound.max()),D_Ug_plus_Uc_abs_bound=int(projected_bound.max()),
                reconstruction_sat24_redundant_for_all_binary_g=True,
                U_RNE_placement='After complete D(Ug)+Uc. No permission to insert intermediate U-g or D-g RNE.',
                coefficient_boundary='Actual D requires more than16 signed bits; split or wider multiply, metadata/psum life and gate-ready schedule remain uncharged here.'),
            cost_boundary='Int8 payload excludes decoder/subtract/RNE, rank24 MACs, gate-ready dependencies and state; dense D decode and18bit coefficients must be charged by hardware. No speedup inferred.')
    (out/'cpu_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({a:{m:dict(zero=s['quant_zero_fraction'],empty_word=s['empty_P2_T10_fraction'],MAE=s['input_physical_MAE']) for m,s in ar['statistics'].items()} for a,ar in report['axes'].items()}))


if __name__=='__main__':main()
