"""Bounded calibrated PED basis replacement; no dynamic rank or training.

Two students fit separately on a predetermined 16x16 anchor grid, then evaluate
the two original disjoint windows. Real q16/q15 coefficients, signed48 sums,
original RNE/sat24 boundaries and bias. New R32 is explicitly NOT lossless.
"""
from pathlib import Path
import json
import numpy as np
from probe import FULL, HERE, RANKS, rne24, error, fee


def execute(u,v,x,bias):
    a=u.astype(np.int64)@x.astype(np.int64)
    assert np.max(np.abs(a))<2**47
    z=rne24(a,16)
    b=v.astype(np.int64)@z
    assert np.max(np.abs(b))<2**47
    y=rne24(b,15)
    out=rne24(y+bias[:,None],0)
    return out,dict(U_acc_maxabs=int(np.max(np.abs(a))),V_acc_maxabs=int(np.max(np.abs(b))),
        U_sat=int(np.count_nonzero((np.rint(a/65536)<-2**23)|(np.rint(a/65536)>2**23-1))),
        V_sat=int(np.count_nonzero((np.rint(b/32768)<-2**23)|(np.rint(b/32768)>2**23-1))),
        bias_sat=int(np.count_nonzero((y+bias[:,None]<-2**23)|(y+bias[:,None]>2**23-1))))


def balance_quantize(u,v):
    # Offline per-latent gauge balancing is folded into coefficients. No new
    # runtime scales or ports. Both SVD controls receive the identical rule.
    um=np.max(np.abs(u),axis=1)*65536
    vm=np.max(np.abs(v),axis=0)*32768
    scale=np.sqrt(vm/np.maximum(um,1e-30))
    uq=np.rint(u*scale[:,None]*65536)
    vq=np.rint(v/scale[None,:]*32768)
    counts=dict(U_coefficient_clips=int(np.count_nonzero((uq<-32768)|(uq>32767))),
        V_coefficient_clips=int(np.count_nonzero((vq<-32768)|(vq>32767))),
        U_raw_coefficient_maxabs=float(np.max(np.abs(uq))),V_raw_coefficient_maxabs=float(np.max(np.abs(vq))),
        offline_latent_scale=scale.tolist())
    return np.clip(uq,-32768,32767).astype(np.int16),np.clip(vq,-32768,32767).astype(np.int16),counts


def main():
    result=dict(scope=__doc__,calibration=dict(frame='000_zurich_city_09_a_0001',
        y=list(range(32,96,4)),x=list(range(32,96,4)),anchors=256,T=10,
        note='No held-out spatial-window overlap; SAME frame, NOT held-out sequences or AEE.'),
        no_extra_runtime_scale=True,coefficient_format='Original signed16 with fixed U exponent16/V exponent15',
        basis_contract='Original R32/row-norm R24 vs weight-only SVD vs activation-whitened latent SVD. Reassociation crosses original U RNE, so full-rank replacement error is measured, never assumed zero.',
        fee_contract='Same fee() necessary MAC8/ideal packed ports and reservation proxy as first probe, NOT cycles or PPA.',axes={})
    for axis in ('ordinary','lifting_raw'):
        with np.load(FULL/'capture'/axis/'parameters.npz') as z:q={k:z[k] for k in z.files}
        with np.load(FULL/'capture_full_producers'/axis/'000_zurich_city_09_a_0001.npz') as z:xall=z['full_updated_I24']
        with np.load(FULL/'capture'/axis/'000_zurich_city_09_a_0001.npz') as z:gold={k:z[k+'_continuous_q24'] for k in ('corner','interior')}
        uq,vq=q['U_ped_q16'],q['V_ped_q16']
        u=uq.astype(float)/65536; v=vq.astype(float)/32768
        xcal=xall[:,:,32:96:4,32:96:4].transpose(1,0,2,3).reshape(96,-1)
        zcal=rne24(uq.astype(np.int64)@xcal.astype(np.int64),16).astype(float)
        normalized=zcal/(2**23)
        gram=normalized@normalized.T/normalized.shape[1]
        eig=np.linalg.eigvalsh(gram)
        jitter=max(0.,float(eig[-1]*1e-12-eig[0]))
        chol=np.linalg.cholesky(gram+np.eye(32)*jitter)
        left,s,rt=np.linalg.svd(v@chol,full_matrices=False)
        ua=(np.sqrt(s)[:,None]*rt)@np.linalg.solve(chol,u)
        va=left*np.sqrt(s)[None,:]
        lw,sw,rw=np.linalg.svd(v@u,full_matrices=False)
        uw=np.sqrt(sw[:32])[:,None]*rw[:32]; vw=lw[:,:32]*np.sqrt(sw[:32])[None,:]
        models={}
        model_params={}
        for label,un,vn in [('weight_svd',uw,vw),('activation_whitened',ua,va)]:
            a,b,metadata=balance_quantize(un,vn)
            models[label]=(a,b)
            metadata['real_full32_operator_relative_error']=float(np.linalg.norm(vn@un-v@u)/np.linalg.norm(v@u))
            model_params[label]=metadata
        order=np.argsort(-(np.linalg.norm(uq.astype(float),axis=1)*np.linalg.norm(vq.astype(float),axis=0)),kind='stable')
        models['original_ordered']=(uq[order],vq[:,order])
        ar=dict(latent_gram_eigen_range=eig[[0,-1]].tolist(),latent_gram_jitter=jitter,
            source_values_calibrated=int(xcal.size),models=model_params,windows={},
            fee={str(r):fee(r,False) for r in RANKS})
        np.savez_compressed(HERE/(axis+'_rebase_parameters.npz'),
            **{label+'_'+part:a for label,(un,vn) in models.items() for part,a in [('U',un),('V',vn)]},
            PED_bias_q24=q['PED_bias_q24'],U_ped_exponent=np.array(16),V_ped_exponent=np.array(15),
            calibration_y=np.arange(32,96,4),calibration_x=np.arange(32,96,4))
        for label,(oy,ox) in [('calibration',(32,32)),('corner',(0,0)),('interior',(120,160))]:
            x=xcal if label=='calibration' else xall[:,:,oy:oy+8:2,ox:ox+8:2].transpose(1,0,2,3).reshape(96,-1)
            ref,ref_counts=execute(uq,vq,x,q['PED_bias_q24'])
            if label!='calibration':
                exp=gold[label].transpose(1,0,2,3).reshape(96,-1)
                assert np.array_equal(ref,exp),(axis,label,'original capture mismatch')
            wr=dict(output_values=int(ref.size),original32_capture_differences=0 if label!='calibration' else None,
                original_counts=ref_counts,models={})
            for m,(un,vn) in models.items():
                rows={}
                for r in RANKS:
                    out,counts=execute(un[:r],vn[:,:r],x,q['PED_bias_q24'])
                    rows[str(r)]=dict(error=error(out,ref),counts=counts)
                wr['models'][m]=rows
            ar['windows'][label]=wr
        result['axes'][axis]=ar
    (HERE/'rebase_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({a:{w:{m:{r:rr['error']['nrmse'] for r,rr in mr.items()} for m,mr in wr['models'].items()} for w,wr in ar['windows'].items()} for a,ar in result['axes'].items()},indent=2))


if __name__=='__main__':main()
