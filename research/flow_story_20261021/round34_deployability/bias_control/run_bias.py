"""One fit-only 2D geometric-median output correction; real cached Q4 arrays."""
from pathlib import Path
import sys,os,time,hashlib,json
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'numeric'))
import run_numeric as u

def objective(errors,b):
    d=errors-b
    return float(np.mean(np.sqrt(d[:,0]*d[:,0]+d[:,1]*d[:,1]),dtype=np.float64))

def weiszfeld(errors):
    e=np.asarray(errors,dtype=np.float64)
    if e.ndim!=2 or e.shape[1]!=2 or len(e)==0 or not np.all(np.isfinite(e)):raise ValueError('finite pooled Nx2 errors')
    b=np.mean(e,axis=0,dtype=np.float64);trace=[dict(iteration=0,bias=b.copy(),objective=objective(e,b),displacement=None)]
    converged=False
    for k in range(1,101):
        d=e-b;distance=np.sqrt(d[:,0]*d[:,0]+d[:,1]*d[:,1]);weights=1/np.maximum(distance,1e-12)
        updated=np.sum(e*weights[:,None],axis=0,dtype=np.float64)/np.sum(weights,dtype=np.float64)
        displacement=float(np.linalg.norm(updated-b));b=updated
        trace.append(dict(iteration=k,bias=b.copy(),objective=objective(e,b),displacement=displacement))
        if displacement<1e-10:converged=True;break
    return b,trace,converged

def measure64(pred,gt,masks):
    p=np.asarray(pred,dtype=np.float64);g=np.asarray(gt,dtype=np.float64)
    assert p.shape==g.shape and np.all(np.isfinite(p));d=p-g;error=np.sqrt(d[0]*d[0]+d[1]*d[1]);out={}
    for key,mask in masks.items():
        n=int(mask.sum());s=float(np.sum(error[mask],dtype=np.float64))
        out[key]=dict(valid_pixels=n,error_sum=s,AEE=s/n if n else None)
    return out

def main():
    start=time.time();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):assert os.environ.get(k)=='2'
    freeze=u.check_freeze(HERE/'SOURCE_FREEZE_V1.json')
    for b in freeze['bound'].values():assert u.sha(b['path'])==b['sha256']
    out=HERE/'actual_v1';assert not out.exists();out.mkdir()
    memberpath=u.R33/'data/MEMBERSHIP_V1.json';assert u.sha(memberpath)==u.MEMBERSHIP_SHA;members=u.read(memberpath)['records']
    reducer=u.imported('r34_bias_raw_sources',u.R33.parent/'independent_review/reduce_e1_quality_v1.py')
    ev=u.imported('r34_bias_old_evaluator',u.R33/'evaluate.py');raw=reducer.raw
    regionpath=u.R33.parents[1]/'flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json'
    assert u.sha(regionpath)==reducer.REGIONS_SHA;regions=u.read(regionpath)
    sources={};bindings={};pred={}
    # Bind every source byte before estimating the single fit-only vector.
    for phase in ('fit_tune','confirm'):
        run=u.R33/'actual'/f'{phase}_v1';result=u.read(run/'results.json');assert result['status']=='R33_COMPLETE'
        byid={f['target_id']:f for f in result['evaluations']['background_Q4']['frames']}
        for rec in members:
            if (rec['role']=='confirm')!=(phase=='confirm'):continue
            tid=rec['target_id'];sources[tid]={};bindings[tid]={}
            for f in rec['source_files']:
                path=reducer.child(u.R33/'data'/f'{phase}_source_snapshot',f['relative_path'])
                assert path.stat().st_size==int(f['bytes']) and u.sha(path)==f['sha256']
                sources[tid][f['category']]=path;bindings[tid][f['category']]=dict(path=str(path),sha256=f['sha256'])
            pred[tid]=reducer.prediction(run,byid[tid]);bindings[tid]['Q4_prediction']=dict(path=str(pred[tid]),sha256=u.sha(pred[tid]))
    u.write(out/'SOURCE_BINDINGS_V1.json',dict(source_freeze_sha256=u.sha(HERE/'SOURCE_FREEZE_V1.json'),membership_sha256=u.sha(memberpath),records=bindings))
    def arrays(tid):
        src={k:np.load(v,allow_pickle=False) for k,v in sources[tid].items()}
        with np.load(pred[tid],allow_pickle=False) as z:prediction=z['head3'].copy()
        return src,prediction
    pooled=[];fit_ids=[];fit_counts=[]
    for rec in members:
        if rec['role']!='fit':continue
        tid=rec['target_id'];src,f=arrays(tid);valid=src['mask'];assert valid.dtype==np.bool_
        e=(f.astype(np.float64)-src['gt'].astype(np.float64))[:,valid].T.copy();pooled.append(e);fit_ids.append(tid);fit_counts.append(len(e))
    errors=np.concatenate(pooled);del pooled
    b,trace,converged=weiszfeld(errors)
    model=dict(schema='flow_r34_fit_constant_output_bias_v1',bias=b,fit_ids=fit_ids,fit_valid_pixel_counts=fit_counts,fit_pooled_valid_pixels=len(errors),
        fit_error_payload_sha256=hashlib.sha256(np.ascontiguousarray(errors).tobytes()).hexdigest(),zero_bias_objective=objective(errors,np.zeros(2)),
        fitted_bias_objective=objective(errors,b),iterations=len(trace)-1,converged=converged,trajectory=trace,
        epsilon=1e-12,max_iterations=100,stop_displacement=1e-10,initializer='pooled_error_mean',sample_weight='equal_each_valid_GT_pixel',
        tune_confirm_used_for_estimation=False,runtime='float64(Q4)-bias',membership_sha256=u.sha(memberpath))
    u.write(out/'BIAS_MODEL_V1.json',model);u.write(out/'BIAS_FREEZE_V1.json',dict(bias_model_sha256=u.sha(out/'BIAS_MODEL_V1.json'),evaluation_started=False))
    print('FIT_BIAS_FROZEN',b.tolist(),'iterations',len(trace)-1,'converged',converged,flush=True)
    del errors
    hardware=u.read(u.R33/'hardware/results/SUMMARY_V1.json');q4op=4*next(v['static_affine_BN_ticks'] for v in hardware['candidates'] if v['action']==6)
    allresults={};zero_parity=0
    for role in ('fit','tune','confirm'):
        phase='confirm' if role=='confirm' else 'fit_tune';qpath=u.R33/'packs'/f'{phase}_128_v1'/'QUALITY_V1.json'
        cached={f['target_id']:f for f in u.read(qpath)['frames']}
        folded=u.read(u.HERE/'folded_constant_v/actual_v1'/f'{role}_QUALITY_V1.json');fc={f['target_id']:f for f in folded['frames']}
        frames=[];payloads=[]
        for rec in members:
            if rec['role']!=role:continue
            tid=rec['target_id'];src,f=arrays(tid)
            masks=raw.region_masks(src['gt'],src['mask'],src['input'],regions['boundary_threshold_pixels'],regions['large_displacement_threshold_pixels'])
            zero=measure64(f.astype(np.float64)-np.zeros((2,1,1),np.float64),src['gt'],masks)
            reference=raw.measure(f,src['gt'],masks)
            assert zero==reference==cached[tid]['states']['background_Q4'];zero_parity+=len(zero)
            corrected=f.astype(np.float64)-b[:,None,None];quality=measure64(corrected,src['gt'],masks)
            states=dict(Q4=zero,bias_corrected_Q4=quality,
                uncorrected_folded_fixed_six=fc[tid]['policies']['mixed_int16__fixed_six']['regions'],
                uncorrected_folded_allow_Q4=fc[tid]['policies']['mixed_int16__allow_Q4']['regions'])
            frames.append(dict(target_id=tid,sequence=rec['sequence'],role=role,states=states))
            payloads.append(dict(target_id=tid,corrected_float64_payload_sha256=hashlib.sha256(np.ascontiguousarray(corrected).tobytes()).hexdigest()))
        seqs=list(dict.fromkeys(f['sequence'] for f in frames));summary={}
        for policy in frames[0]['states']:
            summary[policy]={scope:{region:ev.mc.aggregate([f['states'][policy][region] for f in frames if scope=='overall' or f['sequence']==scope]) for region in ev.REGIONS} for scope in ['overall']+seqs}
        comparisons={}
        for candidate,reference in [('bias_corrected_Q4','Q4'),('bias_corrected_Q4','uncorrected_folded_fixed_six'),
            ('uncorrected_folded_fixed_six','bias_corrected_Q4'),('uncorrected_folded_allow_Q4','bias_corrected_Q4')]:
            comparisons[candidate+'__vs_'+reference]=ev.comparison(frames,summary,candidate,reference,seqs)
        result=dict(role=role,frames=frames,summaries=summary,comparisons=comparisons,corrected_payloads=payloads,zero_bias_raw_and_cache_exact=True,
            bias_model_sha256=u.sha(out/'BIAS_MODEL_V1.json'),fit_is_resubstitution=role=='fit',all_data_previously_exposed=True,
            baseline_cost=dict(four_Q4_operator_service_ticks=q4op,postprocess_component_adds_per_identity=614400,two_FP64_constants_bytes=16,
                selector=False,upgrade=False,postprocess_cycles_measured=False,output_buffer_traffic_not_modeled=True),
            comparison_cost_scope='controlled operators plus explicitly unmodeled postprocess; no total-cycle Pareto claim')
        u.write(out/f'{role}_QUALITY_V1.json',result)
        allresults[role]=dict(AEE={name:v['overall']['all_valid']['pixel_weighted_AEE'] for name,v in summary.items()},
            vs_Q4=comparisons['bias_corrected_Q4__vs_Q4']['aggregate_2pct_review'],
            tail_items=comparisons['bias_corrected_Q4__vs_Q4']['high_support_tail_review'],
            sequence_gain_counts=comparisons['bias_corrected_Q4__vs_Q4']['sequence_gain_counts'])
    u.write(out/'SUMMARY_V1.json',dict(schema='flow_r34_bias_control_result_v1',status='COMPLETE',bias=b,iterations=len(trace)-1,converged=converged,
        results=allresults,zero_bias_exact_frame_region_comparisons=zero_parity,GPU_used=False,new_flow_forwards=0,
        CPU_seconds=time.time()-start,no_bias_applied_to_upgrades=True))
    print('BIAS_CONTROL_COMPLETE',round(time.time()-start,3),flush=True)

if __name__=='__main__':main()
