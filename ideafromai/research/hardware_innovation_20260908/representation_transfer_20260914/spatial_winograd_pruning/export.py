"""Exact bounded integer projection, fixed coefficients only; no fitting or GPU."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,csv,itertools,sys
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent/'spatial_winograd_inputs'
sys.path.insert(0,str(H.parent/'spatial_r16_integer'))
from export import factor_chain,rne_shift,identity_to_j

def project(a,v,B=1023):
    # Signed diagonal transform is isometric; sum(h)=0 is the only equality.
    a=np.asarray(a,np.int64);v=np.asarray(v,np.int64);old=a*v
    x=np.arange(-B,B+1,dtype=np.int64);summ=-x
    low=np.maximum(-B,summ-B);high=np.minimum(B,summ+B)
    center=(old[1]-old[2]+summ)//2
    y=np.stack([np.clip(center,low,high),np.clip(center+1,low,high)],axis=1)
    z=summ[:,None]-y
    cost=(x[:,None]-old[0])**2+(y-old[1])**2+(z-old[2])**2
    pos=int(np.argmin(cost));i,j=divmod(pos,2)
    h=np.array([x[i],y[i,j],z[i,j]],np.int64);out=h*v
    assert h.sum()==0 and np.max(abs(out))<=B
    # Independent discrete convex certificate: no feasible unit exchange helps.
    err=h-old
    for ii in range(3):
        for jj in range(3):
            if ii!=jj and h[ii]<B and h[jj]>-B:
                assert 2*(int(err[ii])-int(err[jj]))+2>=0
    return out,int(cost[i,j])

def check_small():
    cases=0
    B=2;feasible=[np.array(x,np.int64) for x in itertools.product(range(-B,B+1),repeat=3) if sum(x)==0]
    for v in [(1,1,1),(1,-1,1)]:
        for a in itertools.product(range(-B,B+1),repeat=3):
            p,c=project(a,v,B);oracle=min(int(np.square(h-np.asarray(a)*v).sum()) for h in feasible)
            assert c==oracle and int(np.dot(p,v))==0;cases+=1
    return cases

def err(a,b):
    d=np.asarray(a,np.float64)-np.asarray(b,np.float64);b=np.asarray(b,np.float64)
    return dict(relative_L2=float(np.linalg.norm(d.ravel())/max(np.linalg.norm(b.ravel()),1e-300)),RMSE=float(np.sqrt(np.mean(d*d))),MAE=float(np.mean(abs(d))),max_abs=float(np.max(abs(d))),different=int(np.count_nonzero(d)))

def transformed(q):
    return np.stack([2*q[:,:,0],q.sum(2),q[:,:,0]-q[:,:,1]+q[:,:,2],2*q[:,:,2]],axis=2)

def main():
    small=check_small()
    with np.load(S/'factors.npz') as ar:f={k:ar[k].copy() for k in ar.files}
    with np.load(S/'gold_tiles.npz') as ar:g={k:ar[k].copy() for k in ar.files}
    q=f['q2'].astype(np.int64);q1=f['q1'].astype(np.int64)
    assert q.shape==(96,16,3) and int(f['q2_bits'])==11 and np.max(abs(q))<=1023
    scale_gain=f['output_scale'].astype(np.float64)*f['BN_gain'].astype(np.float64);weights=np.square(scale_gain)
    proj=np.empty((2,96,16,3),np.int64);distance=np.empty((2,96,16),np.int64)
    for mi,v in enumerate([(1,1,1),(1,-1,1)]):
        for o in range(96):
            for r in range(16):proj[mi,o,r],distance[mi,o,r]=project(q[o,r],v)
    candidates={'moment':q.copy(),'native_tap':q.copy()};choices={'moment':np.empty((12,16),np.int8),'native_tap':np.empty((12,16),np.int8)};records=[]
    for og in range(12):
        oo=slice(og*8,og*8+8)
        for r in range(16):
            m_cost=(distance[:,oo,r]*weights[None,oo]).sum(1);mi=int(np.argmin(m_cost))
            candidates['moment'][oo,r]=proj[mi,oo,r];choices['moment'][og,r]=mi+1
            t_cost=(np.square(q[oo,r])*weights[oo,None]).sum(0);tap=int(np.argmin(t_cost))
            candidates['native_tap'][oo,r,tap]=0;choices['native_tap'][og,r]=tap
            records.append(dict(og=og,rank=r,moment_component=mi+1,m1_weighted_squared_error=float(m_cost[0]),m2_weighted_squared_error=float(m_cost[1]),moment_integer_squared_error=int(distance[mi,oo,r].sum()),native_tap=tap,native_weighted_squared_error=float(t_cost[tap])))
    za=np.maximum(-f['z_lower'],f['z_upper']).astype(np.int64)
    reports={};frozen={}
    # Finish both model parameter files before emitting the longer gold arrays.
    for name,qq in candidates.items():
        d=H/name;d.mkdir(exist_ok=True);U=transformed(qq)
        for og in range(12):
            for r in range(16):
                k=int(choices[name][og,r]);a=U[og*8:og*8+8,r,k] if name=='moment' else qq[og*8:og*8+8,r,k]
                assert np.all(a==0)
        assert np.max(abs(qq))<=1023 and np.max(abs(U))<=3069
        assert not np.any(U[:,:,[0,3]]&1) and np.array_equal(U[:,:,1]+U[:,:,2],U[:,:,0]+U[:,:,3])
        W=np.einsum('orx,rcy->ocyx',qq,q1)
        lo=np.minimum(W,0).sum((1,2,3));hi=np.maximum(W,0).sum((1,2,3));prefix=(abs(qq)*za[None,:,None]).sum((1,2))
        mp=(abs(U)*(2*za)[None,:,None]).sum(1)
        rec=np.maximum(mp[:,0]+mp[:,1]+mp[:,2],mp[:,1]+mp[:,2]+mp[:,3])
        wide=np.maximum(-lo,hi)*abs(f['a_q40'].astype(np.int64))+((1<<31)+abs(f['b_q20'].astype(np.int64)))*(1<<20)
        assert max(int(prefix.max()),int(mp.max()),int(rec.max()))<2**31 and int(wide.max())<2**63
        ff={k:v.copy() for k,v in f.items()};ff.update(q2=qq.astype(np.int16),q2_bits=np.array(11),expanded_int32=W.astype(np.int32),p_lower=lo,p_upper=hi,p_any_prefix_abs=prefix,wide_abs_bound=wide,winograd_q2=U.astype(np.int16),winograd_M_prefix_abs=mp,winograd_reconstruction_abs=rec,pruning_choice=choices[name])
        for k in ['q1','a_q40','b_q20','output_scale','theta','BN_gain','BN_offset','bias','first_scale']:assert np.array_equal(ff[k],f[k]),k
        np.savez_compressed(d/'factors.npz',**ff);frozen[name]=ff
        fields=['q1','q2','winograd_q2','a_q40','b_q20','output_scale','theta','bias','BN_gain','BN_offset','first_scale','pruning_choice']
        (d/'frozen_parameters.json').write_text(json.dumps(dict(schema='spatial_r16_q8_q11_pruned_i24_v1',arm=name,arrays={k:dict(shape=list(ff[k].shape),dtype=str(ff[k].dtype),values=ff[k].tolist()) for k in fields}),separators=(',',':'))+'\n')
        qgroup=qq.reshape(12,8,16,3);ugroup=U.reshape(12,8,16,4)
        report=dict(passed=True,arm=name,selection='fixed-coefficient BN*scale weighted L2; no source/valid fitting',groups=192,target='one m1/m2 zero N8 vector per group' if name=='moment' else 'one shared native tap zero per group',choice_histogram={str(int(k)):int(np.count_nonzero(choices[name]==k)) for k in np.unique(choices[name])},native_zero_N8_groups=int(np.count_nonzero(~np.any(qgroup,axis=1))),winograd_zero_N8_groups=int(np.count_nonzero(~np.any(ugroup,axis=1))),native_scalar_zeros=int(np.count_nonzero(qq==0)),winograd_scalar_zeros=int(np.count_nonzero(U==0)),weighted_coefficient_error=err((qq-q)*scale_gain[:,None,None],np.zeros_like(q)),weighted_coefficient_relative_L2=float(np.linalg.norm(((qq-q)*scale_gain[:,None,None]).ravel())/np.linalg.norm((q*scale_gain[:,None,None]).ravel())),bounds=dict(q2=[int(qq.min()),int(qq.max())],U=[int(U.min()),int(U.max())],Z=[int(f['z_lower'].min()),int(f['z_upper'].max())],D_abs=int((2*za).max()),expanded=[int(W.min()),int(W.max())],p_final=[int(lo.min()),int(hi.max())],p_any_prefix_abs=int(prefix.max()),M_any_prefix_abs=int(mp.max()),reconstruction_abs=int(rec.max()),wide_abs=int(wide.max())),small_exact_projection_cases=small,full_projected_triples=3072,full_discrete_optimality_certificates=3072,integer_constraints=True,network_AEE='not run',RTL='not run',training=False)
        # Error against zero has undefined relative scale: retain only absolute fields.
        report['weighted_coefficient_error'].pop('relative_L2')
        reports[name]=report
    with (H/'group_choices.csv').open('w',newline='') as out:
        wr=csv.DictWriter(out,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    print('PARAMETERS_READY moment/ and native_tap/',flush=True)
    for name,ff in frozen.items():
        d=H/name;qq=ff['q2'].astype(np.int64);U=ff['winograd_q2'].astype(np.int64);W=ff['expanded_int32'].astype(np.int64)
        ps=[];wides=[];i24s=[];Ms=[];per=[]
        for idx,words in enumerate(g['source_words']):
            gates=((words[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
            z,p=factor_chain(gates,q1,qq);assert np.array_equal(z,g['z_halo_int'][idx])
            direct=np.stack([np.stack([gates[:,:,yy:yy+3,xx:xx+3].reshape(10,864)@W.reshape(96,864).T for xx in range(2)],axis=2) for yy in range(2)],axis=2)
            assert np.array_equal(p,direct)
            D=np.stack([z[:,:,:,0]-z[:,:,:,2],z[:,:,:,1]+z[:,:,:,2],z[:,:,:,2]-z[:,:,:,1],z[:,:,:,1]-z[:,:,:,3]],axis=3)
            M=np.einsum('trym,orm->toym',D,U)
            twice=np.stack([M[:,:,:,0]+M[:,:,:,1]+M[:,:,:,2],M[:,:,:,1]-M[:,:,:,2]-M[:,:,:,3]],axis=3)
            assert not np.any(twice&1) and np.array_equal(twice//2,p)
            for stripe in [0,1]:
                m=np.einsum('trym,orm->toym',D[:,stripe*8:stripe*8+8],U[:,stripe*8:stripe*8+8])
                assert not np.any((m[:,:,:,0]+m[:,:,:,1]+m[:,:,:,2])&1)
                assert not np.any((m[:,:,:,1]-m[:,:,:,2]-m[:,:,:,3])&1)
            j=identity_to_j(g['identity_fp32_bits'][idx].view(np.float32));assert np.array_equal(j,g['J_q20'][idx])
            wide=p*ff['a_q40'][None,:,None,None].astype(np.int64)+(j+ff['b_q20'][None,:,None,None].astype(np.int64))*(1<<20)
            i24=rne_shift(wide,26,24)
            ps.append(p.astype(np.int32));wides.append(wide);i24s.append(i24.astype(np.int32));Ms.append(M.astype(np.int32))
            per.append(dict(tile=int(g['tile_ids'][idx]),p_error=err(p,g['p_int'][idx]),i24_error=err(i24,g['i24'][idx]),bn_scaled_error=err((p-g['p_int'][idx])*scale_gain[None,:,None,None],np.zeros_like(p))))
            per[-1]['bn_scaled_error'].pop('relative_L2')
        gold={k:g[k].copy() for k in ['tile_ids','source_words','output_origin_yx','z_halo_int','identity_fp32_bits','J_q20']}
        gold.update(p_int=np.stack(ps),wide_int64=np.stack(wides),i24=np.stack(i24s),winograd_M_int=np.stack(Ms))
        np.savez_compressed(d/'gold_tiles.npz',**gold)
        report=reports[name];report.update(tiles=len(ps),raw_values=int(gold['p_int'].size),factor_vs_expanded_differences=0,factor_vs_winograd_differences=0,odd_reconstructions=0,odd_stripe_reconstructions=0,identity_J_differences=0,p_error_vs_q11_parent=err(gold['p_int'],g['p_int']),wide_error_vs_q11_parent=err(gold['wide_int64'],g['wide_int64']),i24_error_vs_q11_parent=err(gold['i24'],g['i24']),bn_scaled_output_error=err((gold['p_int'].astype(np.int64)-g['p_int'].astype(np.int64))*scale_gain[None,None,:,None,None],np.zeros_like(gold['p_int'])))
        report['bn_scaled_output_error'].pop('relative_L2')
        (d/'stats.json').write_text(json.dumps(report,indent=2)+'\n');(d/'tile_stats.jsonl').write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in per))
        manifest=dict(arm=name,parent='../spatial_winograd_inputs',factors='factors.npz',gold='gold_tiles.npz',input_fields=['tile_ids','source_words','output_origin_yx','identity_fp32_bits'],oracle_only_fields=['z_halo_int','p_int','J_q20','wide_int64','i24','winograd_M_int'],array_shapes={k:list(v.shape) for k,v in gold.items()},array_dtypes={k:str(v.dtype) for k,v in gold.items()},model_quality='root evaluates separately; local errors never used for selection')
        (d/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(dict(arm=name,passed=True,choices=report['choice_histogram'],U_zero_groups=report['winograd_zero_N8_groups'],i24_error=report['i24_error_vs_q11_parent'])),flush=True)
    (H/'SUMMARY.json').write_text(json.dumps(dict(passed=True,arms=reports,projection_tie='ascending h0, then clipped floor before clipped ceil; equal mode costs choose m1; equal tap costs choose smallest tap',no_activity_used_for_selection=True,no_training=True,no_GPU_or_RTL=True),indent=2)+'\n')

if __name__=='__main__':main()
