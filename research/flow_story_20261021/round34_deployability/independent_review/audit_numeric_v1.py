"""Independent numerical reconstruction, without importing numeric/core.py."""
from pathlib import Path
import hashlib,json,os
import numpy as np
B=Path(__file__).resolve().parent.parent;R33=B.parent/'round33_predictability';A=B/'numeric/actual_v1'
F=np.float32;METHODS=('direct_raw','signed_g1','signed_g4','direct_aug_g1','direct_aug_g4','direct_residual_g1','direct_residual_g4')
COUNT=dict(raw_features=0,ridge_traces=0,integer_accumulator_elements=0,quality_frame_regions=0,quality_sequence_regions=0)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def eq(a,b):assert np.array_equal(np.asarray(a),np.asarray(b)),(np.asarray(a).shape,np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
def close(a,b):assert np.allclose(a,b,rtol=3e-11,atol=3e-10),(a,b)
def total(a,axis=None):return np.add.reduce(a,axis=axis,dtype=F)

def features(raw):
    result=np.empty(160,F);t=np.linspace(-1,1,10,dtype=F)
    for iy in range(4):
        for ix in range(4):
            cell=iy*4+ix;q=raw[:,iy*120:(iy+1)*120,ix*160:(ix+1)*160].copy(order='C')
            mags=np.abs(q);mt=total(mags,axis=(1,2));mass=total(mt);s=total(q,axis=0)
            result[cell*8:cell*8+8]=[
                total(np.where(q>0,q,F(0)))/F(q.size),total(np.where(q<0,-q,F(0)))/F(q.size),
                F((q>0).sum())/F(q.size),F((q<0).sum())/F(q.size),
                (total(np.abs(s))/F(s.size))/F(10),total(np.abs(q[1:]-q[:-1]))/F(q[1:].size),
                total(t*mt)/mass if mass!=0 else F(0),total((t*t)*mt)/mass if mass!=0 else F(0)]
            mid=F(.5)*(q[1:]+q[:-1]);dt=q[1:,1:-1,1:-1]-q[:-1,1:-1,1:-1]
            dx=F(.5)*(mid[:,1:-1,2:]-mid[:,1:-1,:-2]);dy=F(.5)*(mid[:,2:,1:-1]-mid[:,:-2,1:-1])
            time_energy=total(dt*dt)
            for axis,gradient in enumerate((dx,dy)):
                num=total(gradient*dt);denom=np.sqrt(F(total(gradient*gradient)*time_energy))
                result[128+cell*2+axis]=num/denom if denom>F(1e-12) else F(0)
    return result

def scale_for(maximum):
    m=np.asarray(maximum,float);out=np.ones_like(m)
    positive=m>0;out[positive]=np.exp2(np.ceil(np.log2(m[positive]*2/32767)))
    return out.astype(F)
def quant(x,scale):
    rounded=np.rint(np.asarray(x,F)/np.asarray(scale,F));sat=np.abs(rounded)>32767
    return np.minimum(np.maximum(rounded,-32767),32767).astype(np.int16),sat
def parts(aux,g):
    d=2*g*g;return aux[:,:d,None]*aux[:,d:].reshape(len(aux),6,d).transpose(0,2,1)
def signed(aux,g,dtype):
    p=parts(aux,g)
    if dtype==np.float64:return p.sum(axis=1)
    acc=np.zeros((len(aux),6),F)
    for k in range(p.shape[1]):acc=np.add(acc,p[:,k,:])
    return acc

def suite(model,x,mode,cal,trace=None,check_cal=False):
    dtype=np.float64 if mode=='fp64' else F;xx=np.asarray(x,dtype);outputs={};auxs={};satdetails={}
    def affine(section,name,inp):
        key=section+'/'+name;keytrace=section+'_'+name;r={k:np.asarray(model[section][name][k],dtype) for k in ['mean','scale','intercept','coef']}
        z=(inp-r['mean'])/r['scale']
        if mode=='fp64':return z@r['coef']+r['intercept']
        expected={'standardized':z}
        if mode=='fp32':
            out=np.zeros((len(z),len(r['intercept'])),F)
            for k in range(len(r['coef'])):out=np.add(out,np.multiply(z[:,k,None],r['coef'][k]))
            out=np.add(out,r['intercept'])
        else:
            c=cal[key];sx=F(c['input_scale']);sw=np.asarray(c['weight_scale'],F)
            qw,satw=quant(r['coef'],sw);eq(qw,np.asarray(c['weight_q'],np.int16));eq(satw,c['weight_saturation'])
            eq(sw,scale_for(np.max(np.abs(r['coef']),axis=0)))
            assert c['accumulator_absolute_bound']==z.shape[1]*32767**2<np.iinfo(np.int64).max and c['fit_rows']==64
            if check_cal:
                eq(sx,scale_for(np.max(np.abs(z))));eq(c['fit_standardized_absmax'],np.max(np.abs(z)))
                eq(c['coefficient_absmax'],np.max(np.abs(r['coef']),axis=0))
            qi,sati=quant(z,sx)
            # Sequential exact int64 sum is independent of the implementation's matmul.
            acc=np.zeros((len(z),qw.shape[1]),np.int64)
            for k in range(z.shape[1]):acc+=qi[:,k,None].astype(np.int64)*qw[k].astype(np.int64)
            out=np.add(np.multiply(np.multiply(acc.astype(F),sx),sw),r['intercept'])
            expected.update(input_q=qi,input_saturation=sati,int64_accumulator=acc)
            COUNT['integer_accumulator_elements']+=acc.size;satdetails[key]=sati.sum(axis=1).tolist()
        expected['output']=out
        if trace is not None:
            for k,v in expected.items():
                stored=trace[keytrace+'__'+k];assert stored.dtype==v.dtype;eq(stored,v)
            COUNT['ridge_traces']+=1
        return out
    outputs['direct_raw']=affine('models','direct_raw',xx)
    for g in [1,4]:
        aux=affine('auxiliary',f'g{g}',xx);auxs[f'g{g}']=aux;ss=signed(aux,g,dtype);outputs[f'signed_g{g}']=ss
        augmented=np.concatenate([xx,aux,parts(aux,g).transpose(0,2,1).reshape(len(xx),-1)],axis=1)
        outputs[f'direct_aug_g{g}']=affine('models',f'direct_aug_g{g}',augmented)
        outputs[f'direct_residual_g{g}']=ss+affine('models',f'direct_residual_g{g}',augmented)
    return outputs,auxs,satdetails

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';out=B/'independent_review/NUMERIC_INDEPENDENT_AUDIT_V1.json';assert not out.exists()
    freeze=read(B/'numeric/SOURCE_FREEZE_V1.json');receipt=read(B/'numeric/EXECUTION_RECEIPT_V1.json')
    for p,h in freeze['files'].items():assert sha(B/'numeric'/p)==h
    assert receipt['source_freeze_sha256']==sha(B/'numeric/SOURCE_FREEZE_V1.json') and receipt['exit_code']==0 and not receipt['GPU_used']
    for f in read(A/'INPUT_BINDINGS_V1.json')['files'].values():assert sha(f['path'])==f['sha256'] and Path(f['path']).stat().st_size==f['bytes']
    members=read(R33/'data/MEMBERSHIP_V1.json')['records'];byid={r['target_id']:r for r in members};feat=load(A/'FEATURES_FP32_V1.npz')
    raw=read(A/'RAW_INPUT_BINDINGS_V1.json')['records'];assert len(raw)==120 and {r['target_id'] for r in raw}==set(byid)
    for role in ['fit','tune','confirm']:eq(feat[role+'_ids'],[r['target_id'] for r in members if r['role']==role])
    for record in raw:
        tid=record['target_id'];r=byid[tid];spec=next(f for f in r['source_files'] if f['category']=='input');p=Path(record['path'])
        assert sha(p)==record['sha256']==spec['sha256'] and p.stat().st_size==int(spec['bytes'])
        z=np.load(p,allow_pickle=False);assert z.shape==(10,480,640) and z.dtype==F
        expected=features(z);role=r['role'];ix=feat[role+'_ids'].tolist().index(tid)
        eq(expected,feat[role+'_x160'][ix]);COUNT['raw_features']+=1
    cal=read(A/'CALIBRATION_V1.json');cf=read(A/'CALIBRATION_FREEZE_V1.json');sat=read(A/'SATURATION_V1.json');bridge=read(A/'BRIDGE_ERROR_V1.json')
    assert cf['calibration_sha256']==sha(A/'CALIBRATION_V1.json') and cf['source_freeze_sha256']==sha(B/'numeric/SOURCE_FREEZE_V1.json')
    assert cf['fit_ids']==feat['fit_ids'].tolist() and not cf['evaluation_inference_started']
    hw={c['action']:c for c in read(R33/'hardware/results/SUMMARY_V1.json')['candidates']};summary={};all_saturations={};maxerr={}
    for width,branch in [(128,'voxel128_v1'),(160,'directional160_v1')]:
        mp=R33/f'fitted/{branch}/MODEL_V1.json';model=read(mp);c=cal[str(width)]
        assert c['fit_ids']==feat['fit_ids'].tolist() and c['model_sha256']==sha(mp) and c['calibration_roles']==['fit'] and not c['GT_or_true_response_used']
        assert c['feature_file_sha256']==sha(A/'FEATURES_FP32_V1.npz');summary[str(width)]={};all_saturations[str(width)]={}
        for role in ['fit','tune','confirm']:
            folder='confirm' if role=='confirm' else 'fit_tune';pack=load(R33/f'packs/{folder}_{width}_v1/{role}.npz')
            ids=pack['ids'].tolist();x=feat[role+'_x160'][:,:width];pred=read(A/f'{role}_{width}/PREDICTIONS_V1.json')
            assert pred['ids']==ids and pred['frozen_original_model_sha256']==sha(mp)
            assert pred['calibrated_model_sha256']==sha(A/'CALIBRATION_V1.json') and not pred['runtime_GT_or_counterfactuals_used']
            quality=read(A/f'{role}_{width}/QUALITY_V1.json');qp=R33/f'packs/{folder}_{width}_v1/QUALITY_V1.json'
            pa=read(R33/'independent_review'/('CONFIRM_PACK_INDEPENDENT_AUDIT_V1.json' if role=='confirm' else 'FIT_TUNE_PACK_INDEPENDENT_AUDIT_V1.json'))
            assert pa['status']=='PASS' and quality['cached_quality_sha256']==sha(qp)==pa['quality_sha256']
            rr={r['target_id']:r for r in read(qp)['frames']};regions=list(rr[ids[0]]['states']['background_Q4']);assert len(regions)==16
            states=['background_Q4']+[f'upgrade_{a}' for a in range(6)]
            errors=np.array([[[rr[tid]['states'][s][r]['error_sum'] for r in regions] for s in states] for tid in ids]);ns=np.array([[rr[tid]['states']['background_Q4'][r]['valid_pixels'] for r in regions] for tid in ids])
            ai=regions.index('all_valid');q4=errors[:,0,ai].sum()/ns[:,ai].sum();refchoice={};role_summary={};all_saturations[str(width)][role]={}
            for mode in ['fp64','fp32','mixed_int16']:
                trace=load(A/f'{role}_{width}/{mode.upper()}_TRACE_V1.npz') if mode!='fp64' else None
                scores,auxes,ss=suite(model,pack['x'] if mode=='fp64' else x,mode,c['scales'],trace,check_cal=mode=='mixed_int16' and role=='fit')
                stored=pred['results'][mode]
                for name,a in auxes.items():
                    if mode=='fp64':close(stored['predicted_auxiliary'][name],a)
                    else:eq(np.asarray(stored['predicted_auxiliary'][name],F),a)
                if mode=='mixed_int16':
                    for layer,counts in ss.items():eq(sat[str(width)][role][mode][layer]['input_saturation_by_identity'],counts)
                    all_saturations[str(width)][role]={layer:[{'target_id':ids[i],'count':n} for i,n in enumerate(counts) if n] for layer,counts in ss.items() if sum(counts)}
                for method,s in scores.items():
                    if mode=='fp64':close(stored['scores'][method],s)
                    else:
                        eq(np.asarray(stored['scores'][method],F),s)
                        old=np.asarray(pred['results']['fp64']['scores'][method]);err=np.abs(s.astype(float)-old)
                        close(bridge[str(width)][role][mode][method]['score_error']['max_absolute'],err.max())
                    for allow,label,key in [(False,'six','fixed_six_choices'),(True,'allow_Q4','allow_Q4_choices')]:
                        choice=s.argmin(axis=1)
                        if allow:choice=np.where(s.min(axis=1)>=0,-1,choice)
                        eq(stored[key][method],choice);name=f'{mode}__{method}__{label}'
                        if mode=='fp64':refchoice[method,label]=choice
                        else:
                            flips=int(np.sum(choice!=refchoice[method,label]));bk='allow_Q4' if allow else 'fixed_six'
                            assert bridge[str(width)][role][mode][method][bk+'_flips']==flips
                            assert bridge[str(width)][role][mode][method][bk+'_flipped_ids']==[tid for tid,b in zip(ids,choice!=refchoice[method,label]) if b]
                        ee=errors[np.arange(len(ids)),choice+1];ticks=np.array([4*hw[6 if a<0 else int(a)]['dynamic_affine_BN_ticks'] for a in choice]);tails=[]
                        for i,tid in enumerate(ids):
                            v=quality['frames'][i]['policies'][name];assert v['action']==choice[i];close(v['four_operator_expected_ticks'],ticks[i])
                            for j,r in enumerate(regions):
                                assert v['regions'][r]['valid_pixels']==ns[i,j];close(v['regions'][r]['error_sum'],ee[i,j]);COUNT['quality_frame_regions']+=1
                            for r in ['all_valid','B','L']:
                                j=regions.index(r)
                                if ns[i,j]>=256 and (ee[i,j]>1.2*errors[i,0,j] or ee[i,j]-errors[i,0,j]>ns[i,j]):tails.append((tid,r))
                        assert tails==[(v['target_id'],v['region']) for v in quality['comparisons'][name+'__vs_Q4']['high_support_tail_review']]
                        for seq in ['overall']+sorted(set(pack['groups'].tolist())):
                            ix=np.ones(len(ids),bool) if seq=='overall' else pack['groups']==seq
                            for j,r in enumerate(regions):
                                n=int(ns[ix,j].sum());e=float(ee[ix,j].sum());v=quality['summaries'][name][seq][r]
                                assert v['valid_pixels']==n;close(v['error_sum'],e)
                                if n:close(v['pixel_weighted_AEE'],e/n)
                                else:assert v['pixel_weighted_AEE'] is None
                                COUNT['quality_sequence_regions']+=1
                            dd=quality['decisions'][name][seq];close(dd['upgrade_fraction'],np.mean(choice[ix]>=0));close(dd['operator_only_service']['mean_four_conv_ticks'],ticks[ix].mean())
                        aee=float(ee[:,ai].sum()/ns[:,ai].sum());ret=quality['retention'][name]
                        oldchoice=refchoice[method,label];old=float(errors[np.arange(len(ids)),oldchoice+1,ai].sum()/ns[:,ai].sum())
                        close(ret['Q4_AEE'],q4);close(ret['FP64_AEE'],old);close(ret['new_AEE'],aee)
                        if q4>old:close(ret['retained_positive_improvement'],(q4-aee)/(q4-old))
                        else:assert ret['retained_positive_improvement'] is None
                        role_summary[name]={'AEE':aee,'relative_vs_Q4':aee/q4-1,'upgraded':int((choice>=0).sum()),'tails':len(tails),'retention':ret['retained_positive_improvement']}
            summary[str(width)][role]=role_summary
    report=dict(schema='flow_r34_numeric_independent_audit_v1',status='PASS',auditor_sha256=sha(__file__),source_freeze_sha256=sha(B/'numeric/SOURCE_FREEZE_V1.json'),
        summary_sha256=sha(A/'SUMMARY_V1.json'),calibration_sha256=sha(A/'CALIBRATION_V1.json'),counts=COUNT,all_role_head_quality=summary,input_saturation_events=all_saturations,
        features_recomputed_from_original_raw=True,all_intermediate_quantized_inputs_weights_and_exact_int64_accumulators_independently_recomputed=True,
        scope='Mixed FP32/int16 readout only; no integer feature or full accelerator claim. All120 identities exposed. Cached quality independently re-reduced from original-array-audited R33 metrics.',GPU_calls=0,settings_changed=False)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':'PASS','counts':COUNT,'saturation':all_saturations}))

if __name__=='__main__':main()
