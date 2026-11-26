"""Independent fit-only bias trajectory and original-array quality audit."""
from pathlib import Path
import importlib.util,json,os
import numpy as np
import audit_numeric_v1 as N
B=N.B;R33=N.R33;D=B/'bias_control';A=D/'actual_v1'
spec=importlib.util.spec_from_file_location('r33_independent_masks',R33/'independent_review/audit_pack_actual_v1.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
def objective(e,b):return float(np.sqrt(np.sum((e-b)**2,axis=1)).mean())
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';out=B/'independent_review/BIAS_INDEPENDENT_AUDIT_V1.json';assert not out.exists()
    freeze=N.read(D/'SOURCE_FREEZE_V1.json')
    for p,h in freeze['files'].items():assert N.sha(D/p)==h
    for bind in freeze['bound'].values():assert N.sha(bind['path'])==bind['sha256']
    model=N.read(A/'BIAS_MODEL_V1.json');bindings=N.read(A/'SOURCE_BINDINGS_V1.json');cf=N.read(A/'BIAS_FREEZE_V1.json')
    assert cf['bias_model_sha256']==N.sha(A/'BIAS_MODEL_V1.json') and not cf['evaluation_started']
    memberpath=R33/'data/MEMBERSHIP_V1.json';members=N.read(memberpath)['records'];assert bindings['membership_sha256']==N.sha(memberpath)==model['membership_sha256']
    assert model['fit_ids']==[r['target_id'] for r in members if r['role']=='fit']
    assert model['epsilon']==1e-12 and model['max_iterations']==100 and model['stop_displacement']==1e-10 and not model['tune_confirm_used_for_estimation']
    runrows={};paths={};fiterrors=[];fitcounts=[]
    for phase in ['fit_tune','confirm']:
        run=R33/f'actual/{phase}_v1';rr=N.read(run/'results.json');runrows[phase]={r['target_id']:r for r in rr['evaluations']['background_Q4']['frames']}
    for r in members:
        tid=r['target_id'];phase='confirm' if r['role']=='confirm' else 'fit_tune';src={}
        for f in r['source_files']:
            p=R33/f'data/{phase}_source_snapshot'/f['relative_path'];assert N.sha(p)==f['sha256'] and p.stat().st_size==int(f['bytes'])
            assert bindings['records'][tid][f['category']]['sha256']==f['sha256'];src[f['category']]=p
        row=runrows[phase][tid];p=R33/f'actual/{phase}_v1'/row['prediction_file'];assert N.sha(p)==row['prediction_file_sha256']==bindings['records'][tid]['Q4_prediction']['sha256']
        src['prediction']=p;paths[tid]=src
        if r['role']=='fit':
            gt=np.load(src['gt'],allow_pickle=False);mask=np.load(src['mask'],allow_pickle=False);f=N.load(p)['head3']
            error=f.astype(float)-gt.astype(float);e=np.stack([error[0,mask],error[1,mask]],axis=1)
            fiterrors.append(e);fitcounts.append(len(e))
    e=np.concatenate(fiterrors);del fiterrors
    assert fitcounts==model['fit_valid_pixel_counts'] and len(e)==model['fit_pooled_valid_pixels']
    assert N.hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest()==model['fit_error_payload_sha256']
    b=e.mean(axis=0);trajectory=model['trajectory'];N.close(trajectory[0]['bias'],b);N.close(trajectory[0]['objective'],objective(e,b));maxdiff=0.
    for k,row in enumerate(trajectory[1:],1):
        assert row['iteration']==k
        norm=np.sqrt(np.sum((e-b)**2,axis=1));ww=np.reciprocal(np.maximum(norm,1e-12))
        new=np.sum(e*ww[:,None],axis=0)/ww.sum();step=float(np.linalg.norm(new-b));maxdiff=max(maxdiff,float(np.max(np.abs(new-np.asarray(row['bias'])))))
        N.close(row['bias'],new);N.close(row['displacement'],step);N.close(row['objective'],objective(e,new))
        assert row['objective']<=trajectory[k-1]['objective']+1e-12
        if k<len(trajectory)-1:assert step>=1e-10
        b=new
    assert len(trajectory)-1==model['iterations']<=100;assert model['converged']==(step<1e-10)
    N.close(model['bias'],b);N.close(model['zero_bias_objective'],objective(e,np.zeros(2)));N.close(model['fitted_bias_objective'],objective(e,b))
    residual=e-b;norm=np.sqrt(np.sum(residual**2,axis=1));gradient=np.mean(residual/np.maximum(norm[:,None],1e-12),axis=0);del e
    regionpath=R33.parents[1]/'flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json';regions=N.read(regionpath)
    tables={};counts=dict(original_source_files=360,Q4_predictions=120,raw_frame_region_checks=0,aggregate_region_checks=0)
    for role in ['fit','tune','confirm']:
        q=N.read(A/f'{role}_QUALITY_V1.json');assert q['bias_model_sha256']==N.sha(A/'BIAS_MODEL_V1.json') and q['fit_is_resubstitution']==(role=='fit')
        rows=[r for r in members if r['role']==role];rawquality={r['target_id']:r for r in N.read(R33/f'packs/{"confirm" if role=="confirm" else "fit_tune"}_128_v1/QUALITY_V1.json')['frames']}
        actual=[];base=[];numbers=[];ids=[];groups=[]
        for i,r in enumerate(rows):
            tid=r['target_id'];p=paths[tid];gt=np.load(p['gt'],allow_pickle=False);mask=np.load(p['mask'],allow_pickle=False);events=np.load(p['input'],allow_pickle=False);f=N.load(p['prediction'])['head3']
            masks=M.masks_from_sources(gt,mask,events,regions['boundary_threshold_pixels'],regions['large_displacement_threshold_pixels']);names=list(masks)
            ff=f.astype(float);corrected=ff-b[:,None,None];err=np.sqrt(np.sum((corrected-gt.astype(float))**2,axis=0));zero=np.sqrt(np.sum((ff-gt.astype(float))**2,axis=0))
            assert q['frames'][i]['target_id']==tid
            assert N.hashlib.sha256(np.ascontiguousarray(corrected).tobytes()).hexdigest()==q['corrected_payloads'][i]['corrected_float64_payload_sha256']
            er=[];ba=[];ns=[]
            for name,m in masks.items():
                n=int(m.sum());s=float(err[m].sum());s0=float(zero[m].sum());ns.append(n);er.append(s);ba.append(s0)
                for policy,error in [('Q4',s0),('bias_corrected_Q4',s)]:
                    v=q['frames'][i]['states'][policy][name];assert v['valid_pixels']==n;N.close(v['error_sum'],error)
                    if n:N.close(v['AEE'],error/n)
                    else:assert v['AEE'] is None
                    counts['raw_frame_region_checks']+=1
                assert q['frames'][i]['states']['Q4'][name]==rawquality[tid]['states']['background_Q4'][name]
            actual.append(er);base.append(ba);numbers.append(ns);ids.append(tid);groups.append(r['sequence'])
        actual=np.array(actual);base=np.array(base);numbers=np.array(numbers);groups=np.array(groups);ai=names.index('all_valid');tails=[]
        seqd={};leave={}
        for seq in ['overall']+list(dict.fromkeys(groups.tolist())):
            ix=np.ones(len(rows),bool) if seq=='overall' else groups==seq
            for policy,es in [('Q4',base),('bias_corrected_Q4',actual)]:
                for j,name in enumerate(names):
                    n=int(numbers[ix,j].sum());s=float(es[ix,j].sum());v=q['summaries'][policy][seq][name]
                    assert v['valid_pixels']==n;N.close(v['error_sum'],s)
                    if n:N.close(v['pixel_weighted_AEE'],s/n)
                    else:assert v['pixel_weighted_AEE'] is None
                    counts['aggregate_region_checks']+=1
            if seq!='overall':
                seqd[seq]=float((actual[ix,ai]-base[ix,ai]).sum()/numbers[ix,ai].sum())
                leave[seq]=float((actual[~ix,ai]-base[~ix,ai]).sum()/numbers[~ix,ai].sum())
        for i,tid in enumerate(ids):
            for name in ['all_valid','B','L']:
                j=names.index(name)
                if numbers[i,j]>=256 and (actual[i,j]>1.2*base[i,j] or actual[i,j]-base[i,j]>numbers[i,j]):tails.append((tid,name))
        assert tails==[(v['target_id'],v['region']) for v in q['comparisons']['bias_corrected_Q4__vs_Q4']['high_support_tail_review']]
        tables[role]=dict(AEE=float(actual[:,ai].sum()/numbers[:,ai].sum()),Q4_AEE=float(base[:,ai].sum()/numbers[:,ai].sum()),
            relative_vs_Q4=float(actual[:,ai].sum()/base[:,ai].sum()-1),sequence_delta_vs_Q4=seqd,leave_one_sequence_delta_vs_Q4=leave,tail_items=tails,
            regions={name:dict(AEE=float(actual[:,j].sum()/numbers[:,j].sum()),relative_vs_Q4=float(actual[:,j].sum()/base[:,j].sum()-1)) if numbers[:,j].sum() else None for j,name in enumerate(names)},
            folded_uncorrected_AEE=q['summaries']['uncorrected_folded_fixed_six']['overall']['all_valid']['pixel_weighted_AEE'])
    result=dict(schema='flow_r34_bias_independent_audit_v1',status='PASS',auditor_sha256=N.sha(__file__),mask_helper_sha256=N.sha(Path(M.__file__)),
        source_freeze_sha256=N.sha(D/'SOURCE_FREEZE_V1.json'),bias_model_sha256=N.sha(A/'BIAS_MODEL_V1.json'),summary_sha256=N.sha(A/'SUMMARY_V1.json'),counts=counts,
        fit_pooled_valid_pixels=model['fit_pooled_valid_pixels'],iterations=model['iterations'],converged=model['converged'],bias=b.tolist(),max_iteration_difference=maxdiff,
        final_mean_unit_residual=gradient.tolist(),independent_raw_quality=tables,GPU_calls=0,
        scope='One fit-only FP64 global output correction. A legitimate cheap competing explanation, not an equalized comparison against bias-corrected upgraded actions. No postprocess cycles measured.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','counts':counts,'bias':b.tolist(),'iterations':model['iterations'],'gradient':gradient.tolist(),'AEE':{k:v['AEE'] for k,v in tables.items()}}))
if __name__=='__main__':main()
