"""One-shot U-weight sign+low-rank compensation; real captured PED replay.

CPU only, no trained recovery, no cycle/PPA claim. Every parameter setting is
fixed by weight-only rules before reading output errors. All edits stay HERE.
"""
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OPEN=HERE.parents[1]
FULL=OPEN.parent/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
SOURCE=FULL.parent/'consumer_service.py'
spec=importlib.util.spec_from_file_location('original_consumer_service',SOURCE)
cs=importlib.util.module_from_spec(spec);spec.loader.exec_module(cs)
LIMIT=1<<23


def rne(x,e,counts,label):
    x=np.asarray(x,np.int64)
    assert np.max(np.abs(x),initial=0)<(1<<47),(label,'signed48 overflow')
    if e>0:
        d=1<<int(e);q,r=np.divmod(x,d)
        pre=q+((2*r>d)|((2*r==d)&((q&1)!=0)))
    else:pre=x<<(-int(e))
    counts[label]=dict(acc_maxabs=int(np.max(np.abs(x),initial=0)),sat24_values=int(np.count_nonzero((pre<-LIMIT)|(pre>=LIMIT))),written24_values=int(x.size),exponent=int(e))
    return cs.rne24(x,e)


def err(y,ref):
    d=y.astype(np.float64)-ref.astype(np.float64)
    return dict(nrmse=float(np.linalg.norm(d)/max(np.linalg.norm(ref),1)),rms_q24=float(np.sqrt(np.mean(d*d))),rms_real_f14=float(np.sqrt(np.mean(d*d))/16384),max_abs_q24=int(np.max(abs(d))),max_abs_real_f14=float(np.max(abs(d))/16384),different_values=int(np.count_nonzero(d)),values=int(ref.size))


def fit_sign(u,r):
    s=np.where(u>=0,1,-1).astype(np.int8)
    alpha=np.rint(np.abs(u.astype(np.float64)).mean(axis=1)).astype(np.int16)
    d=u.astype(np.float64)/65536-(alpha[:,None].astype(float)*s)/65536
    left,sv,right=np.linalg.svd(d,full_matrices=False)
    spectrum=dict(singular_values=sv.tolist(),sign_only_relative_weight_error=float(np.linalg.norm(d)/np.linalg.norm(u/65536)),residual_tail_energy_fraction=float(np.sum(sv[r:]**2)/np.sum(sv**2)))
    m=dict(kind='sign_residual',rank=r,sign=s,alpha=alpha,V=None,spectrum=spectrum)
    if r:
        A=left[:,:r]*np.sqrt(sv[:r])[None,:]
        B=np.sqrt(sv[:r])[:,None]*right[:r]
        # Deterministic per-latent gauge balancing; no activation fitting.
        scale=np.sqrt(np.max(abs(B),axis=1)/np.maximum(np.max(abs(A),axis=0),1e-30))
        A=A*scale[None,:];B=B/scale[:,None]
        assert np.max(abs(A))*65536<=32767,'A16 does not fit declared fixed exponent'
        eB=min(30,int(np.floor(np.log2(32767/np.max(abs(B))))))
        aq=np.rint(A*65536);bq=np.rint(B*(2**eB))
        assert np.min(aq)>=-32768 and np.max(aq)<=32767
        assert np.min(bq)>=-32768 and np.max(bq)<=32767
        m.update(A=aq.astype(np.int16),B=bq.astype(np.int16),B_exponent=eB,gauge_scale=scale)
        effective=alpha[:,None]*s+(aq@bq)/(2**eB)
    else:
        m.update(A=np.empty((32,0),np.int16),B=np.empty((0,96),np.int16),B_exponent=0,gauge_scale=np.empty(0))
        effective=alpha[:,None]*s
    m['effective_weight_relative_error']=float(np.linalg.norm(effective-u)/np.linalg.norm(u))
    return m


def fit_lowbit(u,bits):
    lim=(1<<(bits-1))-1
    initial=np.max(abs(u.astype(float)),axis=1)/lim
    code=np.clip(np.rint(u/initial[:,None]),-lim,lim).astype(np.int8)
    scale=np.rint(np.sum(u.astype(float)*code,axis=1)/np.maximum(np.sum(code.astype(float)**2,axis=1),1)).astype(np.int16)
    effective=scale[:,None].astype(np.int64)*code
    return dict(kind='lowbit',bits=bits,code=code,scale=scale,V=None,effective_weight_relative_error=float(np.linalg.norm(effective-u)/np.linalg.norm(u)))


def execute(m,x,bias):
    counts={};v=m['V']
    if m['kind']=='dense':
        latent=rne(m['U'].astype(np.int64)@x,16,counts,'original_U_RNE')
    elif m['kind']=='lowbit':
        small=m['code'].astype(np.int64)@x
        numerator=m['scale'][:,None].astype(np.int64)*small
        counts['U_small_weight_reduction']=dict(maxabs=int(np.max(abs(small))),free_scale=False)
        latent=rne(numerator,16,counts,'original_U_RNE')
    else:
        signed_sum=m['sign'].astype(np.int64)@x
        numerator=m['alpha'][:,None].astype(np.int64)*signed_sum
        counts['U_sign_reduction']=dict(maxabs=int(np.max(abs(signed_sum))),free_scale=False)
        if m['rank']:
            rz=rne(m['B'].astype(np.int64)@x,m['B_exponent'],counts,'new_residual_B_RNE')
            extra=m['A'].astype(np.int64)@rz
            assert np.max(abs(extra))<(1<<47)
            counts['residual_A_acc']=dict(maxabs=int(np.max(abs(extra))))
            numerator=numerator+extra
        latent=rne(numerator,16,counts,'original_U_RNE')
    y=rne(v.astype(np.int64)@latent,15,counts,'original_V_RNE')
    out=rne(y+bias[:,None],0,counts,'original_bias_sat24')
    return out,counts


def cost(m):
    common=96*3+16 # unchanged packed signed24 bias plus four 32b mode/rank/exponent fields
    vbytes=m['V'].size*2
    if m['kind']=='dense':
        rank=m['U'].shape[0]
        return dict(coefficient_bytes=int(m['U'].size*2+vbytes+common),weight_body_bytes=int(m['U'].size*2+vbytes),scale_bytes=0,metadata_bytes=16,common_bias_bytes=288,mult16x24=int(m['U'].size+m['V'].size),small_weight_products=0,small_weight_bits=0,sign_terms=0,scale16x_wide_multiplies=0,accum_add_or_sub=rank*95+96*(rank-1)+96,rne24_dot_values=rank+96,bias_sat24_values=96,extra_latent24_bytes=0,original_U_latent24_bytes=rank*3)
    if m['kind']=='lowbit':
        ubytes=(m['code'].size*m['bits']+7)//8
        return dict(coefficient_bytes=int(ubytes+64+vbytes+common),weight_body_bytes=int(ubytes+vbytes),scale_bytes=64,metadata_bytes=16,common_bias_bytes=288,mult16x24=3072,small_weight_products=3072,small_weight_bits=m['bits'],sign_terms=0,scale16x_wide_multiplies=32,accum_add_or_sub=6112,rne24_dot_values=128,bias_sat24_values=96,extra_latent24_bytes=0,original_U_latent24_bytes=96)
    r=m['rank'];extra_meta=4 if r else 0
    return dict(coefficient_bytes=int(384+64+256*r+vbytes+common+extra_meta),weight_body_bytes=int(384+256*r+vbytes),scale_bytes=64,metadata_bytes=16+extra_meta,common_bias_bytes=288,mult16x24=3072+128*r,small_weight_products=0,small_weight_bits=0,sign_terms=3072,scale16x_wide_multiplies=32,accum_add_or_sub=6112+127*r,rne24_dot_values=128+r,bias_sat24_values=96,extra_latent24_bytes=3*r,original_U_latent24_bytes=96)


def main():
    start=time.time();result=dict(plan='PLAN.md frozen before output errors',scope='U32x96 only; two old students, two 4x4 windows, all T10; no fitting on outputs',repetitions=1,reference_rne_source=str(SOURCE),reference_rne_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),source_layout='x=updated_I24 at output_origin*2; crop via captured gate_origin, stride2; columns are T,H,W',count_contract='Necessary operations per one 96-input vector; counts are NOT equivalent-cost operations, cycles, bandwidth or PPA. Coefficient bytes include packed weights, all explicit row scales, residual exponents, unchanged bias, fixed16B mode/rank/original-exponent metadata. No controller/SRAM macro padding.',precision='original U e16 then V e15 RNE/sat24; candidate adds residual B RNE/sat24 before A, then combines q16 numerators before original U RNE; all accumulators checked signed48',axes={})
    flat=[];fixture={}
    for axis in ['ordinary','lifting_raw']:
        pp=FULL/'capture'/axis/'parameters.npz';cp=FULL/'capture'/axis/'000_zurich_city_09_a_0001.npz'
        with np.load(pp,allow_pickle=False) as f:q={k:f[k] for k in f.files}
        u=q['U_ped_q16'];v=q['V_ped_q16'];bias=q['PED_bias_q24']
        assert int(q['U_ped_exponent'])==16 and int(q['V_ped_exponent'])==15
        models={'original32':dict(kind='dense',U=u,V=v)}
        for r in [0,4,8,16]:models['sign_residual_r'+str(r)]=fit_sign(u,r)
        for b in [4,8]:models['W'+str(b)]=fit_lowbit(u,b)
        for m in models.values():
            if m['V'] is None:m['V']=v
        rp=OPEN/'new_interface_selection'/(axis+'_rebase_parameters.npz')
        with np.load(rp,allow_pickle=False) as f:
            for name in ['original_ordered','weight_svd','activation_whitened']:
                models[name+'24']=dict(kind='dense',U=f[name+'_U'][:24],V=f[name+'_V'][:,:24])
        params={'original_U':u,'original_V':v,'bias_q24':bias,'original_U_exponent':np.array(16),'original_V_exponent':np.array(15)}
        for name,m in models.items():
            for key,val in m.items():
                if isinstance(val,np.ndarray):params[name+'_'+key]=val
            if m['kind']=='sign_residual':params[name+'_B_exponent']=np.array(m['B_exponent'])
        np.savez_compressed(HERE/(axis+'_parameters_probe_only.npz'),**params)
        ar=dict(parameters_path=str(pp),capture_path=str(cp),models={},windows={})
        for name,m in models.items():
            ar['models'][name]=dict(cost=cost(m),effective_U_relative_error=m.get('effective_weight_relative_error'),spectrum=m.get('spectrum'),B_exponent=m.get('B_exponent'),coefficient_saturation_count=0)
        with np.load(cp,allow_pickle=False) as f:
            geo=json.loads(str(f['window_geometry_json']))
            for window in ['corner','interior']:
                g=geo[window];dy=2*g['output_origin'][0]-g['gate_origin'][0];dx=2*g['output_origin'][1]-g['gate_origin'][1]
                x=f[window+'_updated_I24'][:,:,dy:dy+8:2,dx:dx+8:2].transpose(1,0,2,3).reshape(96,-1).astype(np.int64)
                expected=f[window+'_continuous_q24'].transpose(1,0,2,3).reshape(96,-1)
                ref,cnt=execute(models['original32'],x,bias)
                assert np.array_equal(ref,expected),(axis,window,'reference capture mismatch')
                fixture[axis+'_'+window+'_x']=x.astype(np.int32);fixture[axis+'_'+window+'_reference']=ref.astype(np.int32)
                wr=dict(input_vectors=x.shape[1],output_values=ref.size,original_capture_differences=0,models={})
                for name,m in models.items():
                    out,c=execute(m,x,bias);e=err(out,ref)
                    wr['models'][name]=dict(error=e,counts=c)
                    fixture[axis+'_'+window+'_'+name]=out.astype(np.int32)
                    flat.append(dict(axis=axis,window=window,model=name,**e,**cost(m)))
                ar['windows'][window]=wr
        result['axes'][axis]=ar
    np.savez_compressed(HERE/'replay_fixture_probe_only.npz',**fixture)
    result['elapsed_wall_s']=time.time()-start
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (HERE/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    for axis,a in result['axes'].items():
        print(axis)
        for name,m in a['models'].items():
            print(name,'bytes',m['cost']['coefficient_bytes'],'mult16',m['cost']['mult16x24'],'nrmse',[a['windows'][w]['models'][name]['error']['nrmse'] for w in ['corner','interior']])
    print('elapsed_s',result['elapsed_wall_s'])


if __name__=='__main__':main()
