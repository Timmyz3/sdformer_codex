"""CPU-only R34 bridge. Refuses overwrite; source freeze precedes real execution."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
import core
HERE=Path(__file__).resolve().parent
R33=HERE.parents[1]/'round33_predictability'
MEMBERSHIP_SHA='d860612a3ae3e4620fa2a9a616df9f8e93774efadcf7dc564881e27b7e7da8cc'
MODEL_SHA={128:'164bf51189371edb4030800165f050b75d0238a30a00683cc3c214ea14a80ae4',160:'a9649640e0443f07f082e02778c61264e1703f047314846f886cc88cb51d0654'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def j(v):
    if isinstance(v,dict):return {k:j(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [j(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v
def write(p,v):
    assert not p.exists(),p
    p.write_text(json.dumps(j(v),indent=2,allow_nan=False)+'\n')
def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def bind(paths):return {k:dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for k,p in paths.items()}
def check_freeze(path):
    f=read(path)
    for name,digest in f['files'].items():assert sha(path.parent/name)==digest,(path,name)
    return f
def flatten(d,prefix=''):
    out={}
    for k,v in d.items():
        key=(prefix+'__'+k).strip('_').replace('/','_')
        if isinstance(v,dict):out.update(flatten(v,key))
        else:out[key]=np.asarray(v)
    return out
def error(a,b):
    d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    return dict(max_absolute=float(np.max(np.abs(d))),mean_absolute=float(np.mean(np.abs(d))),
        root_mean_square=float(np.sqrt(np.mean(d*d))),exact_elements=int(np.sum(d==0)),elements=d.size)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();start=time.time()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):assert os.environ.get(name)=='2'
    assert not a.output.exists();check_freeze(HERE/'SOURCE_FREEZE_V1.json');a.output.mkdir(parents=True)
    memberpath=R33/'data/MEMBERSHIP_V1.json';assert sha(memberpath)==MEMBERSHIP_SHA;members=read(memberpath)['records']
    assert len(members)==120 and len({r['target_id'] for r in members})==120
    check_freeze(R33/'algorithm/SOURCE_FREEZE_V1.json');check_freeze(R33/'directional_branch/SOURCE_FREEZE_V1.json')
    scalar=imported('r34_old_evaluator',R33/'evaluate.py')
    original={128:imported('r34_original128',R33/'algorithm/predictors.py'),160:imported('r34_original160',R33/'directional_branch/predictors.py')}
    paths={'membership':memberpath,'source_freeze':HERE/'SOURCE_FREEZE_V1.json','evaluator':R33/'evaluate.py',
        'scalar_math':scalar.CORE,'hardware':R33/'hardware/results/SUMMARY_V1.json'}
    packs={};models={};role_ids={};qualitypaths={}
    for width in (128,160):
        folder='voxel128_v1' if width==128 else 'directional160_v1';mp=R33/'fitted'/folder/'MODEL_V1.json'
        assert sha(mp)==MODEL_SHA[width];models[width]=original[width].load_model(mp);paths[f'model_{width}']=mp
        paths[f'predictor_{width}']=Path(original[width].__file__)
        for phase in ('fit_tune','confirm'):
            pdir=R33/'packs'/f'{phase}_{width}_v1';check_freeze(pdir/'PACK_FREEZE_V1.json')
            paths[f'packfreeze_{width}_{phase}']=pdir/'PACK_FREEZE_V1.json';paths[f'quality_{width}_{phase}']=pdir/'QUALITY_V1.json'
            for role in (('fit','tune') if phase=='fit_tune' else ('confirm',)):
                p=pdir/f'{role}.npz';paths[f'pack_{width}_{role}']=p
                with np.load(p,allow_pickle=False) as z:packs[width,role]={k:z[k].copy() for k in z.files}
                ids=packs[width,role]['ids'].astype(str).tolist()
                expected=[r['target_id'] for r in members if r['role']==role];assert ids==expected
                assert models[width]['metadata']['fit_ids']==[r['target_id'] for r in members if r['role']=='fit']
                if role in role_ids:assert ids==role_ids[role]
                role_ids[role]=ids;qualitypaths[role]=pdir/'QUALITY_V1.json'
    write(a.output/'INPUT_BINDINGS_V1.json',dict(schema='flow_r34_numeric_bindings_v1',files=bind(paths),new_quality_not_used_to_choose_format=True))
    # Original input file hashes are checked before their values are decoded.
    sources={};feature_rows={};inputchecks=[]
    for r in members:
        spec=next(f for f in r['source_files'] if f['category']=='input')
        rel=Path(spec['relative_path']);assert not rel.is_absolute() and '..' not in rel.parts
        phase='confirm' if r['role']=='confirm' else 'fit_tune'
        path=R33/'data'/f'{phase}_source_snapshot'/rel
        assert path.stat().st_size==int(spec['bytes']) and sha(path)==spec['sha256']
        sources[r['target_id']]=path;inputchecks.append(dict(target_id=r['target_id'],role=r['role'],path=str(path),sha256=spec['sha256']))
    write(a.output/'RAW_INPUT_BINDINGS_V1.json',dict(records=inputchecks,GPU_used=False))
    for i,r in enumerate(members):
        z=np.load(sources[r['target_id']],allow_pickle=False);assert z.dtype==np.float32 and z.shape==(10,480,640)
        feature_rows[r['target_id']]=core.voxel_features(z)
        if (i+1)%20==0:print('FP32_RAW_FEATURES',i+1,flush=True)
    X={role:np.stack([feature_rows[tid] for tid in ids]) for role,ids in role_ids.items()}
    np.savez_compressed(a.output/'FEATURES_FP32_V1.npz',**{f'{role}_x160':x for role,x in X.items()},
        **{f'{role}_ids':np.asarray(ids) for role,ids in role_ids.items()})
    feature_errors={}
    for width in (128,160):
        feature_errors[str(width)]={role:dict(**error(X[role][:,:width],packs[width,role]['x']),
            unequal_to_cast_only=int(np.sum(X[role][:,:width]!=packs[width,role]['x'].astype(np.float32)))) for role in role_ids}
    write(a.output/'FEATURE_ERROR_V1.json',feature_errors)
    calibration={};calibration_info={}
    # This loop receives only X_fit and frozen weights. No label/calibration target API.
    for width in (128,160):
        _,_,calibration[width]=core.run_suite(models[width],X['fit'][:,:width],mode='mixed_int16',calibrate=True)
        calibration_info[str(width)]=dict(fit_ids=role_ids['fit'],model_sha256=MODEL_SHA[width],
            feature_file_sha256=sha(a.output/'FEATURES_FP32_V1.npz'),scales=calibration[width],
            calibration_roles=['fit'],GT_or_true_response_used=False)
    write(a.output/'CALIBRATION_V1.json',calibration_info)
    write(a.output/'CALIBRATION_FREEZE_V1.json',dict(calibration_sha256=sha(a.output/'CALIBRATION_V1.json'),
        source_freeze_sha256=sha(HERE/'SOURCE_FREEZE_V1.json'),fit_ids=role_ids['fit'],evaluation_inference_started=False))
    all_results={};bridge={};saturation={}
    for width in (128,160):
        bridge[str(width)]={};saturation[str(width)]={}
        for role in ('fit','tune','confirm'):
            dest=a.output/f'{role}_{width}';dest.mkdir();x=X[role][:,:width]
            fp64=original[width].predict_suite(models[width],packs[width,role]['x'])
            if role!='fit':
                old=read(R33/'evaluations'/f'{role}_{width}_v1'/'PREDICTIONS_V1.json')['results']
                for family in ('scores','predicted_auxiliary','fixed_six_choices','allow_Q4_choices'):
                    for method,value in fp64[family].items():assert np.array_equal(value,np.asarray(old[family][method])),(role,width,family,method)
            results={'fp64':fp64};bridge[str(width)][role]={};saturation[str(width)][role]={}
            for mode in ('fp32','mixed_int16'):
                result,trace,_=core.run_suite(models[width],x,mode=mode,calibration=calibration[width]);results[mode]=result
                np.savez_compressed(dest/f'{mode.upper()}_TRACE_V1.npz',**flatten(trace))
                sat={layer:dict(input_saturation_by_identity=t.get('input_saturation',np.zeros((len(x),1),bool)).sum(axis=1).tolist(),
                    input_elements_per_identity=t['standardized'].shape[1],output_error_to_fp32=None) for layer,t in trace.items()}
                saturation[str(width)][role][mode]=sat;diff={}
                for method in core.METHODS:
                    diff[method]=dict(score_error=error(result['scores'][method],fp64['scores'][method]),
                        fixed_six_flips=int(np.sum(result['fixed_six_choices'][method]!=fp64['fixed_six_choices'][method])),
                        allow_Q4_flips=int(np.sum(result['allow_Q4_choices'][method]!=fp64['allow_Q4_choices'][method])),
                        fixed_six_flipped_ids=[tid for tid,b in zip(role_ids[role],result['fixed_six_choices'][method]!=fp64['fixed_six_choices'][method]) if b],
                        allow_Q4_flipped_ids=[tid for tid,b in zip(role_ids[role],result['allow_Q4_choices'][method]!=fp64['allow_Q4_choices'][method]) if b])
                diff['auxiliary_errors']={g:error(result['predicted_auxiliary'][g],fp64['predicted_auxiliary'][g]) for g in ('g1','g4')}
                bridge[str(width)][role][mode]=diff
            np.savez_compressed(dest/'ALL_NUMERIC_ARRAYS_V1.npz',**flatten(results),features_fp32=x,features_fp64=packs[width,role]['x'],ids=np.asarray(role_ids[role]))
            write(dest/'PREDICTIONS_V1.json',dict(schema='flow_r34_numeric_predictions_v1',width=width,role=role,ids=role_ids[role],results=results,
                calibrated_model_sha256=sha(a.output/'CALIBRATION_V1.json'),feature_source_sha256=sha(a.output/'FEATURES_FP32_V1.npz'),
                frozen_original_model_sha256=MODEL_SHA[width],runtime_GT_or_counterfactuals_used=False))
            all_results[width,role]=results
    write(a.output/'BRIDGE_ERROR_V1.json',bridge);write(a.output/'SATURATION_V1.json',saturation)
    # Only now decode quality: all formats, scales and choices have already been persisted.
    hardware=read(paths['hardware']);headline={}
    for width in (128,160):
        headline[str(width)]={}
        for role in ('fit','tune','confirm'):
            pack=packs[width,role];byid={f['target_id']:f for f in read(qualitypaths[role])['frames']};frames=[byid[tid] for tid in role_ids[role]]
            policies={}
            def policy(name,choices,regime,control='dynamic'):
                policies[name]=dict(weights=scalar.policy_weights(choices),choices=np.asarray(choices),observable='runtime_voxel_only' if control=='dynamic' else 'constant',
                    regime=regime,operator_control=control,label_source=None,stochastic_expectation=False)
            policy('Q4',np.full(len(frames),-1),'Q4_only','static')
            for mode,result in all_results[width,role].items():
                for method in core.METHODS:
                    policy(mode+'__'+method+'__six',result['fixed_six_choices'][method],'fixed_six')
                    policy(mode+'__'+method+'__allow_Q4',result['allow_Q4_choices'][method],'allow_Q4')
            allframes,summary,decision,details,seqs=scalar.summarize_policies(frames,pack,policies,hardware)
            comparisons={};retention={}
            for mode in ('fp64','fp32','mixed_int16'):
                for method in core.METHODS:
                    for regime in ('six','allow_Q4'):
                        name=f'{mode}__{method}__{regime}';ref=f'fp64__{method}__{regime}'
                        comparisons[name+'__vs_Q4']=scalar.comparison(allframes,summary,name,'Q4',seqs)
                        if mode!='fp64':comparisons[name+'__vs_FP64']=scalar.comparison(allframes,summary,name,ref,seqs)
                        q=summary['Q4']['overall']['all_valid']['pixel_weighted_AEE'];old=summary[ref]['overall']['all_valid']['pixel_weighted_AEE'];new=summary[name]['overall']['all_valid']['pixel_weighted_AEE']
                        retention[name]=dict(Q4_AEE=q,FP64_AEE=old,new_AEE=new,FP64_gain=q-old,new_gain=q-new,
                            retained_positive_improvement=(q-new)/(q-old) if q-old>0 else None,
                            engineering_90pct_reference_met=(q-new)/(q-old)>=.9 if q-old>0 else None,
                            reference_is_not_new_independent_validation=True,
                            tail_vs_Q4=len(comparisons[name+'__vs_Q4']['high_support_tail_review']))
            dest=a.output/f'{role}_{width}'
            write(dest/'QUALITY_V1.json',dict(schema='flow_r34_numeric_actual_cached_quality_v1',role=role,width=width,
                cached_quality_sha256=sha(qualitypaths[role]),all_original_action_quality_unchanged=True,
                summaries=summary,decisions=decision,frames=details,comparisons=comparisons,retention=retention,
                feature_or_head_cost_in_operator_service=False,GPU_used=False,new_model_forwards=0))
            headline[str(width)][role]=retention
    write(a.output/'SUMMARY_V1.json',dict(schema='flow_r34_numeric_summary_v1',status='COMPLETE',headlines=headline,
        all_roles_are_previously_exposed=True,FP32_raw_features_recomputed=True,integer_scope='mixed_readout_only',
        calibration_fit_only=True,CPU_seconds=time.time()-start,GPU_used=False,new_flow_forwards=0))
    print('NUMERIC_COMPLETE',round(time.time()-start,3),flush=True)

if __name__=='__main__':main()
