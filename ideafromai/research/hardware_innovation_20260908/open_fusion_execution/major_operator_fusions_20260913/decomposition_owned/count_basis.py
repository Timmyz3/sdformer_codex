"""Shared signed-count dictionary + 2:4 residual; actual bitplane execution."""
from pathlib import Path
import json
import numpy as np
from prototype import HERE,load_capture,error

def group(w):
    return w.transpose(0,2,3,1).reshape(w.shape[0],-1,4)

def ungroup(w):
    return w.reshape(96,3,3,96).transpose(0,3,1,2)

def fit(w,signed=True):
    z=group(w)
    signs=np.ones((8 if signed else 1,4))
    if signed:signs[:,1:]=1-2*((np.arange(8)[:,None]>>np.arange(3))&1)
    costs=[];all_a=[];all_res=[];all_pairs=[]
    # For fixed sign, two approximated coordinates share the scalar a.
    # Their optimum is the closest pair of signed weight values, at midpoint.
    for s in signs:
        t=z*s;order=np.argsort(t,axis=-1);ts=np.take_along_axis(t,order,axis=-1)
        k=np.argmin(np.diff(ts,axis=-1)**2,axis=-1)
        k0=np.take_along_axis(order,k[...,None],axis=-1)[...,0]
        k1=np.take_along_axis(order,(k+1)[...,None],axis=-1)[...,0]
        a=(np.take_along_axis(t,k0[...,None],axis=-1)[...,0]+np.take_along_axis(t,k1[...,None],axis=-1)[...,0])/2
        res=z-a[...,None]*s
        np.put_along_axis(res,k0[...,None],0,axis=-1);np.put_along_axis(res,k1[...,None],0,axis=-1)
        costs.append(np.sum((z-a[...,None]*s-res)**2,axis=(0,2)))
        all_a.append(a);all_res.append(res);all_pairs.append(np.stack([k0,k1],axis=-1))
    chosen=np.argmin(np.asarray(costs),axis=0)
    a=np.stack(all_a)[chosen,:,np.arange(z.shape[1])].T
    res=np.stack(all_res)[chosen,:,np.arange(z.shape[1]),:].transpose(1,0,2)
    ss=signs[chosen]
    pair=np.stack(all_pairs)[chosen,:,np.arange(z.shape[1]),:].transpose(1,0,2)
    return a,ss,ungroup(res),pair

def execute_pair(x,a,s,retained,pair,theta,bias):
    """Equivalent decomposition: direct exceptions + selected signed pair count.

    Excludes the exception bits from the count BEFORE coefficient application.
    It therefore avoids a count term followed by its fullwidth cancellation.
    The pair ID differs per output/group; selector and descriptor cost remain.
    """
    g=(x/theta).reshape(-1,96,3,3).transpose(0,2,3,1).reshape(-1,216,4).astype(np.int8)
    y=x@retained.reshape(96,-1).T+bias
    pair_events=0;pair_nonzeros=0;pair_cancel=0;pair_shift=0
    groups=np.arange(216)[None,:]
    for o in range(96):
        p0,p1=pair[o,:,0],pair[o,:,1]
        g0=g[:,np.arange(216),p0];g1=g[:,np.arange(216),p1]
        small=g0*s[np.arange(216),p0]+g1*s[np.arange(216),p1]
        value=np.zeros(len(x),dtype=np.float64)
        for bit in range(2):
            digit=np.sign(small)*((np.abs(small).astype(np.int8)>>bit)&1)
            value+=(digit@(a[o]*theta))*(1<<bit)
        y[:,o]+=value
        pair_events+=int(np.count_nonzero(small));pair_nonzeros+=int(g0.sum()+g1.sum())
        pair_cancel+=int(np.count_nonzero((g0+g1==2)&(small==0)))
        pair_shift+=int(np.count_nonzero(np.abs(small)==2))
    residual_aac=float(np.mean((x!=0)@np.count_nonzero(retained,axis=0).reshape(-1)))
    live_group=float(np.mean(np.count_nonzero(g.any(-1),axis=1)))
    dense=float(np.mean(np.count_nonzero(x,axis=1))*96)
    ledger=dict(selected_pair_AAC_per_position=pair_events/len(x),retained_AAC_per_position=residual_aac,
        total_fullwidth_AAC_per_position=residual_aac+pair_events/len(x),dense_AAC_per_position=dense,
        saved_AAC_fraction=1-(residual_aac+pair_events/len(x))/dense,
        eliminated_opposite_pair_per_position=pair_cancel/len(x),shifted_equal_pair_per_position=pair_shift/len(x),
        selected_pair_input_spikes_per_position=pair_nonzeros/len(x),
        nonempty_source_groups_per_position=live_group,consumer_pair_decode_requests_per_position=live_group*96,
        source_fourbit_codes_per_position=216,continuous_MAC_per_position=0)
    return y,ledger

def execute(x,a,s,res,theta,bias):
    # Source is theta*g. Count remains exact small signed integer; theta folds
    # into a/res, not into the integer counter.
    g=(x/theta).reshape(-1,96,3,3).transpose(0,2,3,1).reshape(-1,216,4)
    if not np.array_equal(g,g.astype(bool)):raise ValueError('Expected exact theta*g')
    counts=np.sum(g*s[None],axis=-1).astype(np.int8)
    magn=np.abs(counts).astype(np.int8)
    y=x@res.reshape(96,-1).T+bias
    coef=a*theta
    planes=[]
    for bit in range(3):
        digit=np.sign(counts)*((magn>>bit)&1)
        planes.append(digit)
        y+=(digit@coef.T)*(1<<bit)
    return y,counts,planes

def main():
    w,theta,bias,frames=load_capture();out=HERE/'count_results';out.mkdir(exist_ok=True)
    current_path=HERE.parent/'root_owned/sttmultires_unet_encoders_swin3d_patch_embed_residual_encoding_resblocks_0_conv2_0.npz'
    with np.load(current_path) as cur:
        assert np.array_equal(w,cur['weight'])
        current=('matched_dense_firstframe',cur['input'].reshape(-1,864).astype(np.float64),cur['output'].reshape(-1,96).astype(np.float64))
    rows=[]
    for signed in [False,True]:
        a,s,res,pair=fit(w,signed)
        for bits in [32,8]:
            aa=a.copy();rr=res.copy()
            scales={}
            if bits==8:
                for key,z in [('a',aa),('residual',rr)]:
                    scale=np.maximum(np.max(np.abs(z),axis=tuple(range(1,z.ndim)),keepdims=True)/127,1e-30)
                    z[:]=np.rint(z/scale).clip(-127,127)*scale;scales[key]=scale
            aa=aa.astype(np.float32).astype(np.float64);rr=rr.astype(np.float32).astype(np.float64)
            wh=ungroup(aa[:,:,None]*s[None])+rr
            name=('signed' if signed else 'unsigned')+f'_count_2of4_w{bits}'
            # Reconstruct the retained entries, then store those original
            # coefficients rather than the residual differences. The pair
            # approximation stays aa*sign. Both arms are quantized equally.
            retained_group=group(w).copy()
            np.put_along_axis(retained_group,pair,0,axis=-1)
            retained=ungroup(retained_group)
            if bits==8:
                rs=np.maximum(np.max(np.abs(retained),axis=(1,2,3),keepdims=True)/127,1e-30)
                retained=np.rint(retained/rs)*rs
            retained=retained.astype(np.float32).astype(np.float64)
            approx_pair=np.zeros_like(group(w));np.put_along_axis(approx_pair,pair,np.take_along_axis(aa[:,:,None]*s[None],pair,axis=-1),axis=-1)
            pair_wh=ungroup(approx_pair)+retained
            pair_vals=[]
            coeff_nz=np.count_nonzero(aa,axis=0);res_nz=np.count_nonzero(rr,axis=0).reshape(-1)
            vals=[]
            for fname,x,y in frames+[current]:
                yy,count,planes=execute(x,aa,s,rr,theta,bias)
                dense_ref=x@wh.reshape(96,-1).T+bias
                exactdiff=float(np.max(np.abs(yy-dense_ref)))
                if exactdiff>1e-10:raise AssertionError((name,exactdiff))
                hist=np.bincount((count+4).reshape(-1),minlength=9)
                first_active=float(np.mean(np.count_nonzero(x,axis=1)))
                residual_aac=float(np.mean((x!=0)@res_nz))
                bitplane_aac=[float(np.mean(np.count_nonzero(p,axis=0)@coeff_nz)/len(p)) for p in planes]
                # Above one scalar dot is total contributions across rows.
                code_nonzero=float(np.mean(np.count_nonzero(count,axis=1)))
                vals.append(dict(file=fname,**error(yy,y),factor_vs_reconstruction_max_abs=exactdiff,
                    count_hist_minus4_to4=hist.tolist(),count_min=int(count.min()),count_max=int(count.max()),
                    first_count_update_ops_per_position=first_active,first_count_fixed_scan_small_adds_per_position=3*216,
                    live_count_codes_per_position=code_nonzero,residual_AAC_per_position=residual_aac,
                    bitplane_AAC_per_position=bitplane_aac,total_fullwidth_AAC_per_position=residual_aac+sum(bitplane_aac),
                    continuous_MAC_per_position=0,count_encode_groups_per_position=216,
                    residual_merge_adds_per_position=96,
                    fullwidth_AAC_vs_original=float((residual_aac+sum(bitplane_aac))/(first_active*96))))
                pair_y,ledger=execute_pair(x,aa,s,retained,pair,theta,bias)
                err=float(np.max(np.abs(pair_y-(x@pair_wh.reshape(96,-1).T+bias))))
                if err>1e-10:raise AssertionError((name,'pair',err))
                pair_vals.append(dict(file=fname,**error(pair_y,y),factor_vs_reconstruction_max_abs=err,**ledger))
            row=dict(name=name,coefficient_bits=bits,weight_relative_l2=float(np.linalg.norm(wh-w)/np.linalg.norm(w)),
                local_holdout2_relative_l2_mean=float(np.mean([r['relative_l2'] for r in vals[2:4]])),frames=vals,
                count_dictionary_patterns_used=int(len(np.unique(s,axis=0))),
                count_state_width_signed_bits=4,count_storage_per_position_bytes=108,
                coefficient_storage_bytes=(aa.size+np.count_nonzero(rr))*bits/8+216/8*4+82944/4*4/8+96*4,
                first_layer='216 groups of four input channels at the same spatial tap; per group static shared sign basis.',
                second_layer='Exact sign-magnitude 3 bitplanes; each set plane selects a static shifted theta*a coefficient. No continuous multiplier.')
            rows.append(row)
            pair_name=('signed' if signed else 'unsigned')+f'_selected_pair_w{bits}'
            pair_row=dict(name=pair_name,coefficient_bits=bits,frames=pair_vals,
                local_holdout2_relative_l2_mean=float(np.mean([r['relative_l2'] for r in pair_vals[2:4]])),
                weight_relative_l2=float(np.linalg.norm(pair_wh-w)/np.linalg.norm(w)),
                coefficient_storage_bytes=(aa.size+w.size/2)*bits/8+96*4+(96*8 if bits==8 else 0),
                metadata_bytes=dict(pair_index_3bits=96*216*3/8,group_sign_bits=216*4/8,optional_global_pair_code_LUT_3bits=16*6*16*3/8),
                meaning='Same selected signed-pair approximation, but exceptions stored as original coefficients. Per-consumer pair selection creates integer {-2,-1,0,1,2}; exact bitplane add/shift, no continuous MAC.')
            rows.append(pair_row)
            np.savez_compressed(out/(pair_name+'.npz'),a=aa.astype(np.float32),signs=s.astype(np.int8),retained=retained.astype(np.float32),pair=pair.astype(np.int8),
                theta=np.array(theta),bias=bias.astype(np.float32),weight=pair_wh.astype(np.float32),coefficient_bits=np.array(bits))
            np.savez_compressed(out/(name+'.npz'),a=aa.astype(np.float32),signs=s.astype(np.int8),residual=rr.astype(np.float32),
                theta=np.array(theta),bias=bias.astype(np.float32),weight=wh.astype(np.float32),coefficient_bits=np.array(bits),
                **{'scale_'+k:v for k,v in scales.items()})
            print(name,'hold2',row['local_holdout2_relative_l2_mean'],'current',vals[-1]['relative_l2'],'AAC',vals[-1]['total_fullwidth_AAC_per_position'],flush=True)
            print(pair_name,'hold2',pair_row['local_holdout2_relative_l2_mean'],'current',pair_vals[-1]['relative_l2'],'AAC',pair_vals[-1]['total_fullwidth_AAC_per_position'],'saved',pair_vals[-1]['saved_AAC_fraction'],flush=True)
    report=dict(complete=True,current_weight_exactly_equal_historical_capture=True,current_capture=str(current_path),
        numerical_contract='Exact theta*g required; signed count is integer [-4,4]; theta folded into output coefficients; FP32 or coefficient-only W8, no intermediate fullwidth MAC.',
        scope='Real original W, no GT training; first4 historical train captures plus one current matched frame. Local output error not AEE. No latency/PPA claim.',
        cost_boundary='Exact observed fullwidth bitplane coefficient-add counts and residual AAC, separately small integer counter work, encoder groups and merges. Port sharing and physical addresses not yet implemented.',rows=rows)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
