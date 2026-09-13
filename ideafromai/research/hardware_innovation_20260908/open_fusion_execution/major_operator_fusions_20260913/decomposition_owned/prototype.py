"""Real patch convolution decompositions. CPU NumPy only; no cycle claims."""
from pathlib import Path
import argparse, json, os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
CAP=BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/capture_train4'

def sparse_nm(w,n=2,m=4):
    # Group consecutive input channels separately at every spatial tap.
    o,c,h,k=w.shape
    z=w.transpose(0,2,3,1).reshape(o,h,k,c//m,m)
    ii=np.argsort(np.abs(z),axis=-1)[...,-n:]
    out=np.zeros_like(z);np.put_along_axis(out,ii,np.take_along_axis(z,ii,axis=-1),axis=-1)
    return out.reshape(o,h,k,c).transpose(0,3,1,2)

def svd_pair(w,rank,spatial=False,metric=None):
    o,c,h,k=w.shape
    mat=w.transpose(0,3,1,2).reshape(o*k,c*h) if spatial else w.reshape(o,-1)
    if metric is not None:mat=mat@metric
    u,s,v=np.linalg.svd(mat,full_matrices=False)
    a=u[:,:rank]*np.sqrt(s[:rank]);b=np.sqrt(s[:rank,None])*v[:rank]
    if metric is not None:b=np.linalg.solve(metric.T,b.T).T
    if spatial:
        return {'first':b.reshape(rank,c,h,1),'second':a.reshape(o,k,rank).transpose(0,2,1)[:,:,None,:]}
    return {'first':b.reshape(rank,c,h,k),'second':a[:,:,None,None]}

def tucker(w,rank,iters=6):
    o,c,h,k=w.shape
    a=np.linalg.svd(w.reshape(o,-1),full_matrices=False)[0][:,:rank]
    b=np.linalg.svd(w.transpose(1,0,2,3).reshape(c,-1),full_matrices=False)[0][:,:rank]
    for _ in range(iters):
        z=np.einsum('ochw,ci->oihw',w,b,optimize=True)
        a=np.linalg.svd(z.reshape(o,-1),full_matrices=False)[0][:,:rank]
        z=np.einsum('ochw,oj->cjhw',w,a,optimize=True)
        b=np.linalg.svd(z.reshape(c,-1),full_matrices=False)[0][:,:rank]
    core=np.einsum('ochw,oj,ci->jihw',w,a,b,optimize=True)
    return {'first':b.T[:,:,None,None],'core':core,'second':a[:,:,None,None]}

def reconstruct(f):
    a,b=f['first'],f['second']
    if 'core' in f:
        w=np.einsum('oj,jihw,ic->ochw',b[:,:,0,0],f['core'],a[:,:,0,0],optimize=True)
    elif b.shape[-1]>1:
        w=np.einsum('orw,rch->ochw',b[:,:,0,:],a[:,:,:,0],optimize=True)
    else:w=np.einsum('or,rchw->ochw',b[:,:,0,0],a,optimize=True)
    return w+f.get('residual',0)

def hybrid(w,rank,spatial,iters=10):
    resid=np.zeros_like(w)
    for _ in range(iters):
        f=svd_pair(w-resid,rank,spatial)
        resid=sparse_nm(w-reconstruct(f))
    f['residual']=resid
    return f

def quantize(f,bits):
    out={};scales={}
    for name,z in f.items():
        if bits==32:out[name]=z.astype(np.float32).astype(np.float64);continue
        limit=(1<<(bits-1))-1
        scale=np.maximum(np.max(np.abs(z),axis=tuple(range(1,z.ndim)),keepdims=True)/limit,1e-30)
        out[name]=np.rint(z/scale).clip(-limit,limit)*scale
        scales[name]=scale
    return out,scales

def counts(f):
    first=f['first'];second=f['second'];r=first.shape[0]
    nz={k:int(np.count_nonzero(v)) for k,v in f.items()}
    # Per output position under a complete same-resolution streaming image.
    # Spatial first result is shared among neighboring final positions.
    aac=nz['first']+nz.get('residual',0)
    mac=nz['second']+nz.get('core',0)
    return dict(first_spike_coefficients=nz['first'],residual_spike_coefficients=nz.get('residual',0),
        spike_weighted_terms_per_position_at_unit_activity=aac,
        continuous_MAC_per_position=mac,coefficients=sum(nz.values()),
        intermediate_continuous_values_per_position=r+(f['core'].shape[0] if 'core' in f else 0),
        residual_merge_adds_per_position=second.shape[0] if 'residual' in f else 0,
        extra_horizontal_continuous_shift_values=(2*r if second.shape[-1]>1 else 0),
        exact_scope='Arithmetic coefficients under streaming reuse; no clock, memory-port, schedule, or PPA claim.')

def load_capture():
    with np.load(CAP/'parameters.npz') as z:
        w=z['r0_W'].astype(np.float64);theta=float(z['r0_theta']);bias=z['r0_bias']
    frames=[]
    for p in sorted(CAP.glob('[0-9]*.npz')):
        with np.load(p) as z:
            words=z['r0_source_gate_words']
            x=(((words[None]>>np.arange(10)[:,None,None,None])&1)*theta).transpose(0,1,3,2).reshape(-1,864)
            y=z['r0_conv_raw'].transpose(0,2,3,1).reshape(-1,96).astype(np.float64)
            frames.append((str(z['frame_name']),x.astype(np.float64),y))
    return w,theta,bias,frames

def error(y,ref):
    d=y-ref
    return dict(rmse=float(np.sqrt(np.mean(d*d))),relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(ref),1e-30)),max_abs=float(np.max(np.abs(d))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'results');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    w,theta,bias,frames=load_capture()
    x=np.concatenate([f[1] for f in frames[:2]])
    cov=x.T@x/len(x);ridge=float(np.trace(cov)/len(cov)*1e-5)
    metric=np.linalg.cholesky(cov+ridge*np.eye(cov.shape[0]))
    base=float(np.linalg.norm(w));rows=[]
    baseline=[dict(file=name,**error(xx@w.reshape(96,-1).T+bias,yy),active_fraction=float(np.mean(xx!=0))) for name,xx,yy in frames]
    candidates=[]
    for rank in [8,16,24,32,40,48,64,80]:
        candidates.append((f'flat_svd_r{rank}',svd_pair(w,rank)))
        candidates.append((f'activation_svd_r{rank}',svd_pair(w,rank,metric=metric)))
    for rank in [16,24,32,48,64,80,96,128]:candidates.append((f'spatial_r{rank}',svd_pair(w,rank,True)))
    for rank in [8,16,24,32,40,48,64]:candidates.append((f'tucker_r{rank}',tucker(w,rank)))
    for rank in [8,16,24,32,48]:
        candidates.append((f'flat_nm_r{rank}',hybrid(w,rank,False)))
        candidates.append((f'spatial_nm_r{rank}',hybrid(w,rank,True)))
    for name,fac in candidates:
        for bits in [32,8]:
            f,scales=quantize(fac,bits);wh=reconstruct(f);rows_frame=[]
            for frame,xx,yy in frames:
                rows_frame.append(dict(file=frame,**error(xx@wh.reshape(96,-1).T+bias,yy)))
            label=f'{name}_w{bits}'
            row=dict(name=label,coefficient_bits=bits,weight_relative_l2=float(np.linalg.norm(wh-w)/base),cost=counts(f),
                frames=rows_frame,holdout2_relative_l2_mean=float(np.mean([r['relative_l2'] for r in rows_frame[2:]])))
            rows.append(row)
            np.savez_compressed(args.output/(label+'.npz'),**{k:v.astype(np.float32) for k,v in f.items()},bias=bias.astype(np.float32),
                theta=np.array(theta),coefficient_bits=np.array(bits),name=np.array(label),**{'scale_'+k:v for k,v in scales.items()})
        print(name,flush=True)
    for n in [1,2,3]:
        wh=sparse_nm(w,n);name=f'structure_{n}of4'
        vals=[dict(file=name,**error(xx@wh.reshape(96,-1).T+bias,yy)) for name,xx,yy in frames]
        rows.append(dict(name=name,coefficient_bits=32,weight_relative_l2=float(np.linalg.norm(wh-w)/base),
            holdout2_relative_l2_mean=float(np.mean([r['relative_l2'] for r in vals[2:]])),frames=vals,
            cost=dict(spike_weighted_terms_per_position_at_unit_activity=int(np.count_nonzero(wh)),continuous_MAC_per_position=0,coefficients=int(np.count_nonzero(wh)))))
        np.savez_compressed(args.output/(name+'.npz'),weight=wh.astype(np.float32),bias=bias.astype(np.float32))
    report=dict(complete=True,target='sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0',
        source=str(CAP),weight_shape=list(w.shape),theta=theta,baseline_capture_validation=baseline,
        fit_frames=[f[0] for f in frames[:2]],local_holdout_frames=[f[0] for f in frames[2:]],
        scope='Actual captured Conv2 output only; historical train4 parent, not AEE or current matched10. Factors fitted to W except activation SVD uses first2 frames. No GT training.',
        same_precision='FP32 or per-output-row symmetric W8 factors; continuous states not quantized. Not an integer pipeline claim.',ridge=ridge,rows=rows)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DONE',len(rows),flush=True)

if __name__=='__main__':main()
