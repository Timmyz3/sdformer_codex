"""One deterministic X128 fixed-direction fold; frozen numeric core only."""
from pathlib import Path
import sys,os,time
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import core
import run_numeric as u

def fold(ridge):
    coef=np.asarray(ridge['coef'],dtype=np.float64);intercept=np.asarray(ridge['intercept'],dtype=np.float64)
    assert coef.shape==(128,14) and intercept.shape==(14,)
    c=intercept[:2].copy()
    return dict(mean=np.asarray(ridge['mean'],dtype=np.float64),scale=np.asarray(ridge['scale'],dtype=np.float64),
        coef=coef[:,2::2]*c[0]+coef[:,3::2]*c[1],intercept=intercept[2::2]*c[0]+intercept[3::2]*c[1],
        input_width=128,output_width=6,fit_rows=64,alpha=ridge['alpha']),c

def main():
    start=time.time();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):assert os.environ.get(k)=='2'
    freeze=u.check_freeze(HERE/'SOURCE_FREEZE_V1.json');u.check_freeze(u.HERE/'SOURCE_FREEZE_V1.json')
    for b in freeze['bound'].values():assert u.sha(b['path'])==b['sha256']
    out=HERE/'actual_v1';assert not out.exists();out.mkdir()
    p=u.imported('r34_fold_original',u.R33/'algorithm/predictors.py');mp=u.R33/'fitted/voxel128_v1/MODEL_V1.json'
    source=p.load_model(mp);ridge,c=fold(source['auxiliary']['g1'])
    u.write(out/'FOLDED_MODEL_V1.json',dict(schema='flow_r34_folded_fixed_direction_v1',ridge=ridge,constant_direction=c,
        original_model_sha256=u.sha(mp),original_auxiliary='g1',fit_ids=source['metadata']['fit_ids'],no_refit=True,
        online_v_needed=False,directional32_needed=False,coefficient_count=768,intercept_count=6))
    features=np.load(u.HERE/'actual_v1/FEATURES_FP32_V1.npz',allow_pickle=False)
    assert features['fit_ids'].astype(str).tolist()==source['metadata']['fit_ids']
    calibration=core.calibrate_ridge(ridge,features['fit_x160'][:,:128])
    u.write(out/'CALIBRATION_V1.json',dict(fit_ids=features['fit_ids'],calibration=calibration,
        numerical_source_sha256=u.sha(u.HERE/'core.py'),folded_model_sha256=u.sha(out/'FOLDED_MODEL_V1.json'),fit_only=True))
    u.write(out/'CALIBRATION_FREEZE_V1.json',dict(calibration_sha256=u.sha(out/'CALIBRATION_V1.json'),tune_confirm_inference_started=False))
    ev=u.imported('r34_fold_eval',u.R33/'evaluate.py');hardware=u.read(u.R33/'hardware/results/SUMMARY_V1.json')
    ablation=u.read(u.HERE.parent/'ordinary/direction_ablation_v1/PREDICTIONS_BEFORE_QUALITY_V1.json')['predictions'];summary_all={}
    for role in ('fit','tune','confirm'):
        phase='confirm' if role=='confirm' else 'fit_tune';packdir=u.R33/'packs'/f'{phase}_128_v1'
        with np.load(packdir/f'{role}.npz',allow_pickle=False) as z:pack={k:z[k].copy() for k in z.files}
        ids=pack['ids'].astype(str).tolist();assert ids==features[f'{role}_ids'].astype(str).tolist()
        aux=p.ridge_predict(source['auxiliary']['g1'],pack['x']);_,response=p.split_aux(aux,1)
        unfolded=(c[None,None,:]*response).sum(axis=2);folded=p.ridge_predict(ridge,pack['x'])
        if role!='fit':
            key='128_old_confirm' if role=='confirm' else '128_tune';assert ids==ablation[key]['ids']
            assert np.array_equal(unfolded,np.asarray(ablation[key]['scores']['fit_constant_direction']))
        fp32,t32=core.affine_fp32(ridge,features[f'{role}_x160'][:,:128]);mixed,tm=core.affine_mixed(ridge,features[f'{role}_x160'][:,:128],calibration)
        modes={'unfolded_fp64':unfolded,'folded_fp64':folded,'fp32':fp32,'mixed_int16':mixed};policies={};decisions={};bridge={}
        def policy(name,choices,regime,control='dynamic'):
            policies[name]=dict(weights=ev.policy_weights(choices),choices=choices,observable='runtime_voxel_only',regime=regime,
                operator_control=control,label_source=None,stochastic_expectation=False)
        policy('Q4',np.full(len(ids),-1),'Q4_only','static')
        for mode,scores in modes.items():
            choices=dict(fixed_six=core.choose(scores),allow_Q4=core.choose(scores,True));decisions[mode]=choices
            for regime,action in choices.items():policy(mode+'__'+regime,action,regime)
            bridge[mode]=dict(score_error_to_unfolded_FP64=u.error(scores,unfolded),
                fixed_six_flips=int(np.sum(choices['fixed_six']!=core.choose(unfolded))),
                allow_Q4_flips=int(np.sum(choices['allow_Q4']!=core.choose(unfolded,True))))
        bridge['mixed_input_saturation_by_identity']=tm['input_saturation'].sum(axis=1)
        np.savez_compressed(out/f'{role}_ARRAYS_V1.npz',**u.flatten(dict(scores=modes,choices=decisions,fp32_trace=t32,mixed_trace=tm)),ids=pack['ids'])
        raw={f['target_id']:f for f in u.read(packdir/'QUALITY_V1.json')['frames']};frames=[raw[i] for i in ids]
        fs,summary,decision,detail,seqs=ev.summarize_policies(frames,pack,policies,hardware);comp={}
        for mode in modes:
            for regime in ('fixed_six','allow_Q4'):
                name=mode+'__'+regime;comp[name+'__vs_Q4']=ev.comparison(fs,summary,name,'Q4',seqs)
                if mode!='unfolded_fp64':comp[name+'__vs_unfolded_FP64']=ev.comparison(fs,summary,name,'unfolded_fp64__'+regime,seqs)
        u.write(out/f'{role}_QUALITY_V1.json',dict(role=role,summaries=summary,decisions=decision,frames=detail,comparisons=comp,bridge=bridge,
            role_already_exposed=True,fit_is_resubstitution=role=='fit',allow_Q4_is_new_diagnostic_extension=True))
        summary_all[role]=dict(bridge=bridge,AEE={k:v['overall']['all_valid']['pixel_weighted_AEE'] for k,v in summary.items()},
            primary={'fixed_six':comp['mixed_int16__fixed_six__vs_Q4']['aggregate_2pct_review'],
                'allow_Q4':comp['mixed_int16__allow_Q4__vs_Q4']['aggregate_2pct_review']})
    u.write(out/'SUMMARY_V1.json',dict(status='COMPLETE',results=summary_all,CPU_seconds=time.time()-start,GPU_used=False,new_flow_forwards=0,
        numerical_contract_unchanged=True,no_refit=True,coefficient_count=768,intercept_count=6))
    print('FOLDED_CONSTANT_V_COMPLETE',round(time.time()-start,3))

if __name__=='__main__':main()
