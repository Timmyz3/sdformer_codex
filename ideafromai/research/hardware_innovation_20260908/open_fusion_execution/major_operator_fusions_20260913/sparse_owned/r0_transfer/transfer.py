"""Native r0 conv2: two fixed nonzero-fill candidates + Gram causal control.
No training or AEE. Actual first-frame samples; even positions calibrate, odd
positions are local heldout. No hyperparameter/format/budget sweep.
"""
from pathlib import Path
from itertools import combinations
import json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
NEW=HERE.parents[1]
SOURCE=NEW/'root_owned/sttmultires_unet_encoders_swin3d_patch_embed_residual_encoding_resblocks_0_conv2_0.npz'
TARGET='sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0'


def err(a,b):
    delta=a-b
    return dict(relative_squared_error=float(np.sum(delta*delta)/np.sum(b*b)),RMSE=float(np.sqrt(np.mean(delta*delta))),maxabs=float(np.max(np.abs(delta))))


def fit_options(w,x,n,fill):
    # Both families use the same Gram, ridge and minimum-weight-distance tie.
    opts=list(combinations(range(4),n));h=w.shape[0];ng=w.shape[1]//4;slots=n+int(fill)
    values=np.zeros((ng,len(opts),h,slots));matrix=np.zeros((ng,len(opts),h,4));error=np.zeros((ng,len(opts),h));weight_error=np.zeros_like(error)
    for g in range(ng):
        wg=w[:,g*4:g*4+4];xx=x[:,g*4:g*4+4];gram=xx.T@xx
        ridge=max(float(np.trace(gram))/4,1)*1e-4
        for o,indices in enumerate(opts):
            basis=np.eye(4)[:,indices]
            if fill:
                # Actual fill/override basis: c*(pop-g_s)+v*g_s, exact cancellation.
                base=np.ones(4);base[list(indices)]=0
                basis=np.column_stack((base,basis))
            coef=np.linalg.solve(basis.T@(gram+ridge*np.eye(4))@basis,basis.T@(gram+ridge*np.eye(4))@wg.T).T
            candidate=(coef@basis.T).astype(np.float32).astype(float)
            values[g,o]=coef.astype(np.float32).astype(float);matrix[g,o]=candidate
            error[g,o]=np.sum((xx@(candidate-wg).T)**2,axis=0)
            # Magnitude support selection uses pre-refit Euclidean projection.
            projected=np.linalg.lstsq(basis,wg.T,rcond=None)[0].T@basis.T
            weight_error[g,o]=np.sum((projected-wg)**2,axis=1)
    return dict(values=values,matrix=matrix,error=error,weight_error=weight_error,support=np.array(opts),n=n,fill=fill)


def operand_counts(xg,indices,fill):
    # xg: bundles(2 tokens x T10),20,4. n-e may differ by output lane;
    # its <=3-bit mux feeds existing multiplier width and no RF reads.
    selected=xg[:,:,indices] # B,20,H8,n
    if fill:
        base=xg.sum(-1)[:,:,None]-selected.sum(-1)
        return np.concatenate((base[:,:,:,None],selected),axis=-1)
    return selected


def service(activation,coeff):
    active=np.any(activation*coeff[None,None,:,:]!=0,axis=2) # B,T,slots
    # A request can only use source activity, before coefficients return.
    requests=np.any(activation!=0,axis=(1,2)) # B,slots
    return int(requests.sum()),int(active.sum())


def choose(options,x,request_aware,magnitude=False):
    ng,_,h=options['error'].shape;chosen=np.argmin(options['weight_error'] if magnitude else options['error'],axis=1)
    x=x.reshape(-1,20,ng,4)
    switches=0;audits=[]
    for g in range(ng):
        for hb in range(h//8):
            lanes=np.arange(hb*8,hb*8+8);cur=chosen[g,lanes].copy()
            original=cur.copy()
            floor=float(np.min(options['error'][g,:,lanes],axis=1).sum())
            # Fixed 10% summed local-SSE allowance + tiny absolute stabilizer.
            cap=1.10*floor+1e-12
            def measure(c):
                coeff=options['values'][g,c,lanes];inds=options['support'][c]
                requests,issues=service(operand_counts(x[:,:,g],inds,options['fill']),coeff)
                sse=float(options['error'][g,c,lanes].sum())
                return 2*requests+issues,requests,issues,sse
            initial=measure(cur)
            if request_aware:
                # One fixed pass over eight output lanes. No restarts/sweeps.
                for lane in range(8):
                    best=measure(cur);best_choice=int(cur[lane])
                    for candidate in range(len(options['support'])):
                        trial=cur.copy();trial[lane]=candidate
                        score=measure(trial)
                        if score[3]<=cap and (score[0],score[3],candidate)<(best[0],best[3],best_choice):best=score;best_choice=candidate
                    cur[lane]=best_choice
            final=measure(cur);chosen[g,lanes]=cur;switches+=int(np.count_nonzero(cur!=original))
            audits.append(dict(group=g,H8=hb,initial_requests=initial[1],final_requests=final[1],initial_issues=initial[2],final_issues=final[2],initial_SSE=initial[3],final_SSE=final[3],SSE_limit=cap))
    return chosen,dict(changed_exceptions_or_supports=switches,groups=audits)


def decode(options,choice):
    ng,h=choice.shape;w=np.zeros((h,ng*4));values=np.zeros((ng,h,options['n']+int(options['fill'])),np.float32);idx=np.zeros((ng,h,options['n']),np.uint8)
    for g in range(ng):
        w[:,g*4:g*4+4]=options['matrix'][g,choice[g],np.arange(h)]
        values[g]=options['values'][g,choice[g],np.arange(h)];idx[g]=options['support'][choice[g]]
    return w.astype(np.float32),values,idx


def serialize(values,idx,fill):
    """Real little-endian FP32 words and minimal fixed-width support codes."""
    n=idx.shape[-1]
    if fill:
        table=np.arange(4,dtype=np.uint8)[:,None];encoding='2-bit exception index'
    elif n==2:
        table=np.array(list(combinations(range(4),2)),dtype=np.uint8);encoding='3-bit pair code (six legal codes, two unused)'
    elif n==3:
        table=np.array([[j for j in range(4) if j!=i] for i in range(4)],dtype=np.uint8);encoding='2-bit omitted index'
    else:raise ValueError(n)
    bits=3 if n==2 and not fill else 2
    meta=bytearray()
    for hb in range(12):
        for g in range(216):
            word=0
            for lane in range(8):
                matches=np.flatnonzero(np.all(table==idx[g,hb*8+lane],axis=1))
                assert len(matches)==1
                word|=int(matches[0])<<(bits*lane)
            meta.extend(word.to_bytes(bits,'little'))
    payload=values.reshape(216,12,8,-1).transpose(1,0,3,2).astype('<f4').tobytes()
    return payload,bytes(meta),table,bits,encoding


def source_packets(x):
    masks=(x.astype(np.uint8)*np.array([1,2,4,8],np.uint8)).sum(-1).astype(np.uint8)
    packet=np.zeros((len(x),216,16),np.uint8)
    packet[:,:,:10]=(masks[:,::2,:]|(masks[:,1::2,:]<<4)).transpose(0,2,1)
    return packet


def packed_counts(x,values,idx,fill):
    # H8 coefficient slot = one aligned CR256 word. Metadata comes from an
    # actual bitstream and one32B response latch; straddles really fetch twice.
    x=x.reshape(-1,20,216,4);req=issues=metareq=decodes=0
    payload,meta,table,meta_bits,encoding=serialize(values,idx,fill)
    packets=source_packets(x)
    out=np.zeros((len(x),20,96),np.float64)
    slots=values.shape[-1]
    for hb in range(12):
        for b in range(len(x)):
            last_meta=-1;response=b''
            for g in range(216):
                # Fixed source packet prepacking is a shared collector contract,
                # not a claim that native im2col gathering/packet creation is free.
                packet=packets[b,g]
                nibble=np.empty(20,np.uint8);nibble[::2]=packet[:10]&15;nibble[1::2]=packet[:10]>>4
                a=((nibble[:,None]>>np.arange(4))&1)[None,:,:]
                assert np.array_equal(a[0],x[b,:,g])
                if not np.any(a):continue
                lanes=slice(hb*8,hb*8+8)
                # Packed metadata bytes are addressed in H8->K4 order.
                addr=(hb*216+g)*meta_bits
                codebytes=bytearray()
                for byteaddr in range(addr,addr+meta_bits):
                    word=byteaddr//32
                    if word!=last_meta:
                        metareq+=1;last_meta=word;response=meta[word*32:(word+1)*32]
                    codebytes.append(response[byteaddr%32])
                codeword=int.from_bytes(codebytes,'little')
                indices=table[[(codeword>>(meta_bits*lane))&((1<<meta_bits)-1) for lane in range(8)]]
                assert np.array_equal(indices,idx[g,lanes]);decodes+=1
                operands=operand_counts(a,indices,fill)[0]
                for slot in range(slots):
                    # Pre-request activity uses only source operands. No free
                    # coefficient inspection, including when a future fit is zero.
                    if not np.any(operands[:,:,slot]):continue
                    offset=((hb*216+g)*slots+slot)*32
                    coefficient=np.frombuffer(payload[offset:offset+32],dtype='<f4').astype(float)
                    req+=1
                    products=operands[:,:,slot]*coefficient[None,:]
                    issues+=int(np.count_nonzero(np.any(products!=0,axis=1)))
                    out[b,:,lanes]+=products
    return out.reshape(-1,96),dict(CR256_value_requests=req,CR256_metadata_requests=metareq,SIMD8_MAC_issues=issues,
        modeled_CR_response_plus_issue_slots=2*(req+metareq)+issues,
        support_decode8lane_issues=decodes,modeled_CR_response_issue_plus_support_decode_slots=2*(req+metareq)+issues+decodes,
        metadata_encoding=encoding,weight_value_bytes=len(payload),metadata_bytes=len(meta),
        source_packet_16B_reads=len(x)*216*12,source_packet_nibble_decodes=len(x)*216*12*20,
        scope='Actual serialized FP32 coefficient-word/minimal support metadata reads and issued products on sampled P2T10 bundles; one metadata response latch, no coefficient reuse across bundles. H8 outer uses20 accumulator vectors and reads every source packet12 times including zero packets. One support decode issue per nonempty group is separately charged. Shared source packing/gathering, arbitration, source latency and destination writes excluded; these are counts, not whole-layer latency.')


def main():
    with np.load(SOURCE) as z:data={k:z[k] for k in z.files}
    # profile_current.py:73 saves bias iff module.bias is not None.
    assert 'bias' not in data
    w=data['weight'].reshape(96,864).astype(float);native=data['input'].transpose(1,0,2).astype(float)
    assert np.array_equal(np.unique(native),[0.,1.])
    cal=native[::2].reshape(-1,864);test=native[1::2].reshape(-1,864)
    report=dict(scope=__doc__,source=str(SOURCE),target=TARGET,weight_shape=[96,96,3,3],theta=1.,bias='original Conv bias=None (profile_current.py:73 captures any present bias); all-zero adapter bridge',
        calibration_positions=data['positions'][::2].tolist(),holdout_positions=data['positions'][1::2].tolist(),
        original_CPU_vs_GPU=err(native.reshape(-1,864)@w.T,data['output'].transpose(1,0,2).reshape(-1,96)),
        permission='All legal supports with equal local Gram refit, ridge1e-4; ordinary2/3 also receive the identical one-pass request optimizer and10% local-SSE allowance. Only magnitude candidate intentionally fixes its support before refit.',
        request_objective='2*physical_CR256_value_requests + issued_SIMD8_MAC; same one-pass optimizer. Minimal metadata formats (ordinary24:3 bits, ordinary34:2 bits, shift14:2 bits) and support decode are measured separately; their request counts do not depend on selected support. Use runtime pop-g_s and g_s, never downstream gold.',
        NB0_diverse10=1.45460286107,NB0_valid825_frame_mean=1.44535253468097,training=False,AEE=False,rows=[])
    candidates=[('ordinary24',2,False,True,False),('ordinary34',3,False,True,False),('shift14_magnitude',1,True,False,True),('shift14_gram',1,True,False,False),('shift14_request',1,True,True,False)]
    cached={}
    for name,n,fill,request_aware,magnitude in candidates:
        key=(n,fill)
        if key not in cached:cached[key]=fit_options(w,cal,n,fill)
        options=cached[key];chosen,audit=choose(options,cal,request_aware,magnitude)
        effective,values,indices=decode(options,chosen)
        row=dict(name=name,fill=fill,coefficient_bits=32,calibration=err(cal@effective.T,cal@w.T),holdout=err(test@effective.T,test@w.T),weight=err(effective,w),request_optimization=audit)
        for split,x in [('calibration',cal),('holdout',test)]:
            actual,counters=packed_counts(x,values,indices,fill)
            reference=x@effective.astype(float).T
            delta=np.max(np.abs(actual-reference));assert delta<2e-6,(name,delta)
            row[split+'_execution']=dict(**counters,packed_vs_effective_maxabs=float(delta),packed_vs_effective_relative_squared_error=err(actual,reference)['relative_squared_error'])
        payload,metadata,_,_,encoding=serialize(values,indices,fill)
        np.savez_compressed(HERE/(name+'.npz'),weight=effective.reshape(96,96,3,3),bias=np.zeros(96,np.float32),
            values=values,indices=indices,fill=np.asarray(fill),theta=np.asarray(1.),coefficient_bits=np.asarray(32),name=np.asarray(name),
            value_stream=np.frombuffer(payload,np.uint8),metadata_stream=np.frombuffer(metadata,np.uint8),metadata_encoding=np.asarray(encoding))
        report['rows'].append(row)
        print(name,row['holdout'],row['holdout_execution'],flush=True)
        (HERE/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (HERE/'results.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
