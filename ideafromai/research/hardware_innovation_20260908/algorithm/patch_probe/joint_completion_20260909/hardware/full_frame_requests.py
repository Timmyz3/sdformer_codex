"""Actual full-frame convolution requests and captured-window numeric checks.

Input is evaluate_network's gates.npz, consumer_windows.npz and parameters.npz.
Counts have explicit P4/H8 coefficient addresses and source layouts. They are
not a completion-time simulation; FP32 and INT8 coefficient beats differ.
"""
from pathlib import Path
import argparse
import json
import math
import time

import numpy as np

P, HG, ROW_CHUNK = 4, 8, 8


def scalar(z,key,default=None):
    return z[key].item() if key in z else default


def unpack(z,key,shape_key):
    shape=tuple(int(x) for x in z[shape_key])
    return np.unpackbits(z[key].reshape(-1),bitorder=str(scalar(z,'bitorder','little')),
                         count=int(np.prod(shape))).reshape(shape).astype(bool)


def request_counts(gates,w,prefix,lane_need=None):
    """[T,C,Y,X], [H,C,3,3], optional [T,H,Y,X] exact demand for Yi.

    All coefficient requests include source and real W zero intersection.
    A coefficient address is (H8*Cin*9 + c*9 + kh*3+kw)*vector_bytes.
    Distinct prefix/tail reuse epochs cannot share the same fetch for free.
    """
    t,cin,ny,nx=gates.shape
    hout=w.shape[0];nh=hout//HG;ng=nx//P
    assert w.shape==(hout,cin,3,3) and hout%HG==0 and nx%P==0
    padded=np.pad(gates,((0,0),(0,0),(1,1),(1,1)))
    wz=w.reshape(nh,HG,cin,3,3)!=0
    names=('time_vector_uses','full_T_vector_uses','prefix_vector_uses',
           'tail_vector_uses','needed_time_vector_uses','active_scalar_terms',
           'needed_scalar_terms')
    total={name:np.zeros(nh,dtype=np.int64) for name in names}
    source_nrv=np.zeros((ny,ng),dtype=np.int64)
    headers=np.zeros((t,ny,ng),bool)
    group_need=(np.ones((t,nh,ny,ng),bool) if lane_need is None else
                lane_need.reshape(t,nh,HG,ny,ng,P).any((2,5)))
    for y0 in range(0,ny,ROW_CHUNK):
        y1=min(ny,y0+ROW_CHUNK);nr=y1-y0
        for kh in range(3):
            for kw in range(3):
                src=padded[:,:,y0+kh:y1+kh,kw:kw+nx].reshape(t,cin,nr,ng,P)
                op=src.any(-1)
                union=op.any(0)
                source_nrv[y0:y1]+=union.sum(0,dtype=np.int64)
                headers[:,y0:y1]|=op.any(1)
                coeff=wz[:,:,:,kh,kw]
                any_h=coeff.any(1).astype(np.int64)
                time_counts=op.sum((0,2,3),dtype=np.int64)
                scalar_counts=src.sum((0,2,3,4),dtype=np.int64)
                total['time_vector_uses']+=any_h@time_counts
                total['full_T_vector_uses']+=any_h@union.sum((1,2),dtype=np.int64)
                total['prefix_vector_uses']+=any_h@op[prefix].any(0).sum((1,2),dtype=np.int64)
                total['active_scalar_terms']+=coeff.sum(1,dtype=np.int64)@scalar_counts
                if lane_need is None:
                    tail=op.copy();tail[prefix]=False
                    total['tail_vector_uses']+=any_h@tail.any(0).sum((1,2),dtype=np.int64)
                    total['needed_time_vector_uses']+=any_h@time_counts
                    total['needed_scalar_terms']+=coeff.sum(1,dtype=np.int64)@scalar_counts
                    continue
                for h in range(nh):
                    wanted=lane_need[:,h*HG:(h+1)*HG,y0:y1].reshape(t,HG,nr,ng,P)
                    # Equal 8-bit W masks share a demand reduction. This is CPU
                    # counting, not a free runtime mask-reduction circuit.
                    patterns=(coeff[h].astype(np.int64)*(1<<np.arange(HG))[:,None]).sum(0)
                    for pattern in np.unique(patterns):
                        if not pattern:continue
                        cs=np.flatnonzero(patterns==pattern)
                        hs=np.flatnonzero((pattern>>np.arange(HG))&1)
                        wanted_p=wanted[:,hs].any(1)
                        matched=src[:,cs]&wanted_p[:,None]
                        needed_op=matched.any(-1)
                        total['needed_time_vector_uses'][h]+=int(needed_op.sum())
                        needed_op[prefix]=False
                        total['tail_vector_uses'][h]+=int(needed_op.any(0).sum())
                        total['needed_scalar_terms'][h]+=int((src[:,cs].sum(1,dtype=np.int64)*wanted[:,hs].sum(1,dtype=np.int64)).sum())
    active_h=wz.any((1,2,3,4))
    header_prefix=headers[prefix].any(0)
    tail_need=group_need.copy();tail_need[prefix]=False
    tail_header=(tail_need&headers[:,None]).any(0)&active_h[:,None,None]
    full_reads=int(source_nrv.sum())*int(active_h.sum())
    time_reads=int((headers.sum(0,dtype=np.int64)*source_nrv).sum())*int(active_h.sum())
    staged_reads=int((header_prefix*source_nrv).sum())*int(active_h.sum())+int((tail_header*source_nrv).sum())
    ckh_live=wz.any((1,4)).sum(1,dtype=np.int64) # [H8,kh]
    valid_kh=np.array([[0<=y+kh-1<ny for y in range(ny)] for kh in range(3)])
    words_per_hg_y=ckh_live@valid_kh.astype(np.int64)
    assembled_full=int((headers[:,None]*words_per_hg_y[None,:,:,None]).sum())
    assembled_needed=int((headers[:,None]*group_need*words_per_hg_y[None,:,:,None]).sum())
    vector_bytes=HG*w.dtype.itemsize
    out={k:int(v.sum()) for k,v in total.items()}
    out['staged_vector_uses']=out['prefix_vector_uses']+out['tail_vector_uses']
    out.update(coefficient_dtype=str(w.dtype),coefficient_vector_bytes=vector_bytes,
               coefficient_vector_read64_beats=math.ceil(vector_bytes/8),
               weight_bytes=int(w.nbytes),weight_scalar_nonzeros=int(wz.sum()),
               weight_H8_vectors_with_nonzero=int(wz.any(1).sum()),
               weight_H8_vectors=int(nh*cin*9),
               time_major_coefficient_read64_beats=out['time_vector_uses']*math.ceil(vector_bytes/8),
               full_T_epoch_coefficient_read64_beats=out['full_T_vector_uses']*math.ceil(vector_bytes/8),
               staged_coefficient_read64_beats=out['staged_vector_uses']*math.ceil(vector_bytes/8),
               source_NRV64_records=int(source_nrv.sum()),
               source_NRV64_max_records_per_P4=int(source_nrv.max()),
               source_NRV64_build_writes=int(source_nrv.sum()),
               source_NRV64_full_epoch_reads=full_reads,
               source_NRV64_time_major_replay_reads=time_reads,
               source_NRV64_staged_replay_reads=staged_reads,
               assembled_source64_header_filtered_time_major_reads=assembled_full,
               assembled_source64_header_filtered_needed_time_major_reads=assembled_needed,
               source_nonzero_time_P4=int(headers.sum()),
               per_H8={k:v.tolist() for k,v in total.items()})
    return out


def need_from_accept(accepted,a,prefix):
    """Only current accepted bits and static graph; never inspect future Y."""
    need=np.empty_like(accepted)
    for s in range(len(a)):
        if s in prefix:need[s]=True
        else:
            consumers=np.flatnonzero(a[:,s]!=0)
            need[s]=(~accepted[consumers]).any(0) if len(consumers) else False
    return need


def x_word_count(x0,x1,word_pixels):
    return (x1-1)//word_pixels-x0//word_pixels+1 if x1>x0 else 0


def source_layouts(gates,w):
    """Explicit source-address alternatives; do not sum different layouts."""
    t,cin,ny,nx=gates.shape;nh=w.shape[0]//HG
    # Conv1 source ready in words [y,c,x//6]: six pixels x complete T10,
    # 60 payload bits/64-bit word. P4 halo gathers up to two adjacent words.
    assert t<=10
    yvalid=np.array([min(ny,y+2)-max(0,y-1) for y in range(ny)])
    x6=np.array([x_word_count(max(0,x-1),min(nx,x+P+1),6) for x in range(0,nx,P)])
    x64=np.array([x_word_count(max(0,x-1),min(nx,x+P+1),64) for x in range(0,nx,P)])
    raw_tiled_reads=int(yvalid.sum()*x6.sum()*cin)
    assembled_words=int(yvalid.sum()*(nx//P)*cin)
    # Conv2 per-T three-row bit cache. A c/kh source word is reused by the
    # three kw offsets; groups of up to4 H8 output contexts share its broadcast.
    kw_live=(w.reshape(nh,HG,cin,3,3)!=0).any(1) # [H8,C,kh,kw]
    words_for_mask=np.zeros(8,dtype=np.int64)
    for mask in range(1,8):
        for x in range(0,nx,P):
            addresses={(x+p+kw-1)//64 for p in range(P) for kw in range(3)
                       if ((mask>>kw)&1) and 0<=x+p+kw-1<nx}
            words_for_mask[mask]+=len(addresses)
    one_h8_reads=0;h32_reads=0
    for y in range(ny):
        for kh in range(3):
            if not 0<=y+kh-1<ny:continue
            masks=(kw_live[:,:,kh].astype(np.int64)*(1<<np.arange(3))).sum(-1)
            one_h8_reads+=int(words_for_mask[masks].sum())*t
            for h0 in range(0,nh,4):
                shared=kw_live[h0:h0+4,:,kh].any(0)
                mask=(shared.astype(np.int64)*(1<<np.arange(3))).sum(-1)
                h32_reads+=int(words_for_mask[mask].sum())*t
    h32_stripes=math.ceil(w.shape[0]/32)
    row_words=cin*math.ceil(nx/64)
    global_gate_words=t*ny*row_words
    return dict(
        alternative_layouts_not_additive=True,
        packed_capture='T,C,Y,X little bitorder is a serialization; hardware layouts below explicitly repack the same bits',
        Conv1_temporal_word=dict(
            address='word64=((y*Cin+c)*ceil(W/6)+floor(x/6)); bit=10*(x%6)+t',
            frame_storage_bytes=ny*cin*math.ceil(nx/6)*8,
            source_ready_endpoint=True,
            ordinary_P4_halo_read64=raw_tiled_reads,
            local_assembled60bit_word_writes64=assembled_words,
            H32_stripes=h32_stripes,
            H32_coefficient_bytes=min(32,w.shape[0])*cin*9*w.dtype.itemsize,
            ordinary_P4_halo_read64_H32_stripes=raw_tiled_reads*h32_stripes,
            local_assembled60bit_word_writes64_H32_stripes=assembled_words*h32_stripes,
            local_max_assembled_source_bytes=3*cin*8,
            per_H8_time_major_assembled_source_word_reads64=assembled_words*t*nh,
            once_per_P4_assembled_scan_read64=assembled_words,
            construction='source-ready packed temporal words -> actual halo word reads -> one60-bit aligned six-pixel c/kh word; NRV compaction/decoding is additional. All-zero rows are known only after read unless paid metadata says otherwise.',
            stripe_policy='For actualFP32 W1, H32 stripe110592B fits128KiB but full331776B does not: three whole-space source passes. Rebuild the boundedP4 NRV once per stripe and reuse it across four serialH8 contexts; no simultaneous four full-T Y contexts are granted.',
            limit='No free complete-T three-row cache or across-P4 residency is assumed. Ordinary larger line/broadcast buffers may improve this tiled point and must be compared under equal capacity.'),
        Conv2_time_row_cache=dict(
            address='backing word64=(((t*H+y)*Cin+c)*ceil(W/64)+floor(x/64)); local row index=y%3',
            cache_bytes=3*row_words*8,
            backing_word64_loads_per_output_stripe=global_gate_words,
            local_reads64_H8_contexts_no_broadcast=one_h8_reads,
            local_reads64_broadcast_to_four_H8_in_each_H32_stripe=h32_reads,
            H32_stripes=h32_stripes,H32_coefficient_bytes=min(32,w.shape[0])*cin*9*w.dtype.itemsize,
            backing_word64_loads_H32_stripes=global_gate_words*h32_stripes,
            corresponding_cache_write64_beats=global_gate_words*h32_stripes,
            kw_weight_mask='read addresses include only kw with some real nonzero W in receiving H8/H32; a varying mask needs actual static metadata/selection, not a free oracle. Dense-all weights compile to constant111.',
            one_output_row_arrival_latency='first valid two source rows must be loaded before y=0; then one next source row, with border zeros generated. No free preloaded frame.',
            broadcast_state_limit='H32 source broadcast needs all four H8 recipients to accept before reuse; P4xH32 one-time FP32 psum=512B, versus128B for H8. Port/arithmetic serialization and stalls are not inferred from these read counts.',
            precision='W bytes use actual parameter dtype. ForFP32, H32 stripe110592B fits128KiB; this is storage capacity, not area.'),
    )


def error(ref,observed):
    diff=np.asarray(ref,dtype=np.float64)-np.asarray(observed,dtype=np.float64)
    return dict(values=int(diff.size),max_abs=float(np.max(np.abs(diff))),
                mean_abs=float(np.mean(np.abs(diff))),rms=float(np.sqrt(np.mean(diff*diff))),
                nonfinite=int(np.count_nonzero(~np.isfinite(diff))))


def window_convolution(gates,w,theta,origins,size):
    t,cin,ny,nx=gates.shape
    padded=np.pad(gates,((0,0),(0,0),(1,1),(1,1)))
    ys=np.empty((len(origins),t,len(w),size,size),np.float64)
    for n,(y,x) in enumerate(origins):
        # Invalid outside-image output is never requested by the capture.
        src=padded[:,:,int(y):int(y)+size+2,int(x):int(x)+size+2]
        patches=np.lib.stride_tricks.sliding_window_view(src,(3,3),axis=(-2,-1))
        ys[n]=np.einsum('tcxyij,hcij->thxy',patches.astype(np.float64)*theta,w.astype(np.float64),optimize=True)
    return ys


def window_checks(path,source,gate,params,theta1,theta2):
    if not path.exists():return dict(available=False)
    z=np.load(path)
    origins=z['window_origins_yx']
    layout=str(scalar(z,'windows_layout',scalar(z,'layout','N,T,C,Y,X')))
    def get(name):
        a=z[name]
        if a.ndim==6 and a.shape[2]==1:a=a[:,:,0]
        if a.shape[0]!=len(origins) and a.shape[2]==len(origins):a=a.transpose(2,0,1,3,4)
        assert a.ndim==5 and a.shape[0]==len(origins),(name,a.shape,layout)
        return a
    raw1=get('conv1_raw');raw2=get('conv2_raw');size=raw1.shape[-1]
    observed_gate=get('sn2_gate')
    if observed_gate.dtype!=np.bool_:
        unit=observed_gate/theta2
        amplitude_invalid=int(np.count_nonzero((unit!=0)&(unit!=1)))
        observed_gate=unit!=0
    else:amplitude_invalid=0
    bit_windows=np.stack([gate[:,:,int(y):int(y)+size,int(x):int(x)+size] for y,x in origins])
    c1=window_convolution(source,params['W1'],theta1,origins,size)
    c2=window_convolution(gate,params['W2'],theta2,origins,size)
    if 'conv1_bias' in params and params['conv1_bias'].size:
        c1+=params['conv1_bias'][None,None,:,None,None]
    if 'conv2_bias' in params and params['conv2_bias'].size:
        c2+=params['conv2_bias'][None,None,:,None,None]
    out=dict(available=True,layout=layout,origins=origins.tolist(),window_size=size,
             gate_unpack_differences=int(np.count_nonzero(observed_gate!=bit_windows)),
             gate_non_theta_amplitudes=amplitude_invalid,
             conv1_FP64_reference_vs_GPU=error(c1,raw1),conv2_FP64_reference_vs_GPU=error(c2,raw2),
             precision_boundary='direct real-value FP64 convolution checks shape/weights/halo; cuDNN FP32/TF32 reduction is not claimed bit-equivalent')
    for i,raw in ((1,raw1),(2,raw2)):
        req=[f'bn{i}_{k}' for k in ('gamma','beta','mean','var','eps')]
        if not all(k in params for k in req):continue
        gain=params[f'bn{i}_gamma'].astype(np.float64)/np.sqrt(params[f'bn{i}_var'].astype(np.float64)+float(params[f'bn{i}_eps']))
        bias=params[f'bn{i}_beta'].astype(np.float64)-params[f'bn{i}_mean'].astype(np.float64)*gain
        normalized=raw.astype(np.float64)*gain[None,None,:,None,None]+bias[None,None,:,None,None]
        if i==1:out['fixed_BN1_using_captured_raw']=error(normalized,get('norm1_Y'))
        else:out['BN2_plus_shortcut_using_captured_raw']=error(normalized+get('identity'),get('block_output'))
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture-dir',type=Path,required=True)
    ap.add_argument('--parameters',type=Path,required=True)
    ap.add_argument('--predictor',type=Path)
    ap.add_argument('--prefix',type=int,nargs='+',default=[2,3,7])
    ap.add_argument('--execution-mode',choices=['exact','conditional'])
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();started=time.monotonic()
    z=np.load(args.capture_dir/'gates.npz');params=np.load(args.parameters)
    source=unpack(z,'source_gate_bits','source_gate_shape')
    gate=unpack(z,'output_gate_bits','output_gate_shape')
    assert source.shape==gate.shape and len(source.shape)==4
    theta1=float(scalar(z,'theta_source'));theta2=float(scalar(z,'theta_output'))
    prefix=[int(t) for t in (z['prefix'] if 'prefix' in z else args.prefix)]
    execution_mode=args.execution_mode or ('exact' if args.capture_dir.parent.name.endswith(('_exact','_full')) else 'conditional')
    need=None;need_check=None;need_mode='full'
    if args.predictor and 'accepted_bits' in z:
        learned=np.load(args.predictor)
        accepted=unpack(z,'accepted_bits','accepted_shape')
        prefix=learned['prefix'].tolist()
        need=need_from_accept(accepted,learned['temporal_q14'],prefix)
        need_mode='individual_lane_from_actual_accepted'
        if 'need_column_bits' in z:
            saved_need=unpack(z,'need_column_bits','need_column_shape')
            t,h,ny,nx=need.shape
            observed=need.reshape(t,h//HG,HG,ny,nx//P,P).any((2,5)).transpose(2,3,1,0)
            need_check=int(np.count_nonzero(observed!=saved_need))
    elif 'need_column_bits' in z:
        columns=unpack(z,'need_column_bits','need_column_shape') # [Y,X/P,H/HG,T]
        ny,ng,nh,t=columns.shape
        shaped=columns.transpose(3,2,0,1)[:,:,None,:,:,None]
        if not columns.all():
            need=np.broadcast_to(shaped,(t,nh,HG,ny,ng,P)).reshape(t,nh*HG,ny,ng*P)
        need_mode='all_or_none_P4_H8_column_from_actual_predictor'
    psn=None
    if args.predictor and 'accepted_gate_counts_H_T' in z:
        learned=np.load(args.predictor)
        mask=learned['temporal_q14']!=0
        prefix=learned['prefix'].tolist()
        pn=mask[:,prefix].sum(1);nn=mask.sum(1);tn=nn-pn
        positions=source.shape[2]*source.shape[3]
        failed=positions-z['accepted_gate_counts_H_T'].astype(np.int64)
        prefix_all=int(positions*source.shape[1]*pn.sum())
        psn=dict(prefix_all=prefix_all,full_terms=int(positions*source.shape[1]*nn.sum()),
                 ideal_retained_C_terms=prefix_all+int((failed*tn[None]).sum()),
                 keep_Y_recompute_terms=prefix_all+int((failed*nn[None]).sum()),
                 keep_Y_skip_prefix_only_fallback_terms=prefix_all+int((failed*(tn>0)[None]*nn[None]).sum()),
                 scope='same-forward accepted counts; nonzero-A scalar terms before known-constant-Y/zero-Y/CSE, not cycles or INT24 execution ofFP32',
                 final_gate_decisions=int(source.size),initial_prediction_checks=int(source.size))
        psn['execution_mode']=execution_mode
        psn['ordinary_exact_actual_terms']=psn['full_terms'] if execution_mode=='exact' else None
        psn['scope']+='; exact mode owes only ordinary full PSN, its prefix/fallback figures are alternative diagnostics and are NOT charged to its baseline'
    c1=request_counts(source,params['W1'],prefix,need)
    c2=request_counts(gate,params['W2'],prefix)
    c1['source_NRV64_build_writes_H32_stripes']=c1['source_NRV64_build_writes']*math.ceil(len(params['W1'])/32)
    result=dict(
        frame=str(scalar(z,'frame_name',args.capture_dir.name)),shape=list(source.shape),prefix=prefix,
        execution_mode=execution_mode,
        scope='whole spatial frame, every C/3x3/P4/H8 and original T; exact source/W request counts under declared layouts, no completion cycles',
        theta_source=theta1,theta_output=theta2,need_column_reconstruction_differences=need_check,
        conditional_demand_mode=need_mode,
        conditional_lane_enable_closed=need_mode=='individual_lane_from_actual_accepted',
        conditional_limit='saved need_column supplies exact whole-column schedule; without per-lane accept its counts are an upper envelope for ordinary destination-lane enable. Final gate zeros never select upstream cancellation.',
        Conv1=c1,Conv2=c2,
        PSN_prediction_and_recompute=psn,
        source_ports=dict(Conv1=source_layouts(source,params['W1'])['Conv1_temporal_word'],
                          Conv2=source_layouts(gate,params['W2'])['Conv2_time_row_cache']),
        windows=window_checks(args.capture_dir/'consumer_windows.npz',source,gate,params,theta1,theta2),
        chain_common=dict(BN2_elements=int(gate.size),shortcut_additions=int(gate.size),
             gate_materialization_bytes=math.ceil(gate.size/8),continuous_FP32_output_bytes=int(gate.size*4),
             identity='independent continuous tensor, not inferred from source bits'),
        not_inferred=['physical clock or same-area PPA','FP32 arithmetic serviced by old INT24 PE','free source packing/NRV metadata or inter-context broadcast','whole chain speed from vector or byte ratios'],
        wall_seconds=time.monotonic()-started,
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:{q:result[k][q] for q in ('coefficient_dtype','time_vector_uses','full_T_vector_uses','staged_vector_uses','active_scalar_terms')} for k in ('Conv1','Conv2')}))


if __name__=='__main__':main()
