"""Apply the same fixed numeric bridge to two new frozen ordinary controls."""
from pathlib import Path
import os,time
import numpy as np
import core
import run_numeric as u

def infer(model,x,mode,scales=None,calibrate=False):
    family=model['family'];choice=model['selection'];auxridge=family['auxiliary'];resridge=family['residual'][str(choice['alpha'])]
    if mode=='fp64':
        p=u.imported('r34_ext_original',u.R33/'algorithm/predictors.py')
        aux=p.ridge_predict(auxridge,x);s=p.signed_scores(aux,1);aug=p.augmented_features(x,aux,1)
        return s+choice['shrink']*p.ridge_predict(resridge,aug),dict(aux=aux),None
    x=np.asarray(x,dtype=np.float32);trace={}
    if calibrate:scales={}
    if mode=='fp32':aux,t=core.affine_fp32(auxridge,x)
    else:
        if calibrate:scales['aux']=core.calibrate_ridge(auxridge,x)
        aux,t=core.affine_mixed(auxridge,x,scales['aux'])
    trace['aux']=t;s=core.signed(aux,1);aug=np.concatenate((x,aux,core.products(aux,1).reshape(len(x),-1)),axis=1)
    if mode=='fp32':res,t=core.affine_fp32(resridge,aug)
    else:
        if calibrate:scales['residual']=core.calibrate_ridge(resridge,aug)
        res,t=core.affine_mixed(resridge,aug,scales['residual'])
    trace['residual']=t
    return s+np.float32(choice['shrink'])*res,trace,scales

def main():
    start=time.time();p=u.HERE;out=p/'ordinary_extension_actual_v1';assert not out.exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):assert os.environ.get(k)=='2'
    freeze=u.check_freeze(p/'ORDINARY_EXTENSION_FREEZE_V1.json');u.check_freeze(p/'SOURCE_FREEZE_V1.json')
    for b in freeze['bound'].values():assert u.sha(b['path'])==b['sha256']
    out.mkdir();features=np.load(p/'actual_v1/FEATURES_FP32_V1.npz',allow_pickle=False)
    ordinary=p.parent/'ordinary/actual_v1';old=u.read(ordinary/'RESULTS_V1.json');models={w:u.read(ordinary/f'MODEL_{w}_V1.json') for w in (128,160)}
    cal={}
    for w in (128,160):
        assert models[w]['fit_ids']==features['fit_ids'].astype(str).tolist()
        _,_,cal[w]=infer(models[w],features['fit_x160'][:,:w],'mixed_int16',calibrate=True)
    u.write(out/'CALIBRATION_V1.json',dict(fit_only=True,fit_ids=features['fit_ids'],scales={str(w):cal[w] for w in (128,160)},models={str(w):u.sha(ordinary/f'MODEL_{w}_V1.json') for w in (128,160)}))
    u.write(out/'CALIBRATION_FREEZE_V1.json',dict(calibration_sha256=u.sha(out/'CALIBRATION_V1.json'),tune_confirm_inference_started=False))
    ev=u.imported('r34_ext_eval',u.R33/'evaluate.py');hardware=u.read(u.R33/'hardware/results/SUMMARY_V1.json');allsummary={}
    for w in (128,160):
        allsummary[str(w)]={}
        for role in ('fit','tune','confirm'):
            phase='confirm' if role=='confirm' else 'fit_tune';packdir=u.R33/'packs'/f'{phase}_{w}_v1'
            with np.load(packdir/f'{role}.npz',allow_pickle=False) as z:pack={k:z[k].copy() for k in z.files}
            ids=pack['ids'].astype(str).tolist();assert ids==features[f'{role}_ids'].astype(str).tolist()
            modes={};traces={};policies={};bridges={}
            def policy(name,choices,regime,control='dynamic'):
                policies[name]=dict(weights=ev.policy_weights(choices),choices=choices,observable='runtime_voxel_only',regime=regime,operator_control=control,label_source=None,stochastic_expectation=False)
            policy('Q4',np.full(len(ids),-1),'Q4_only','static')
            for mode in ('fp64','fp32','mixed_int16'):
                scores,t,_=infer(models[w],pack['x'] if mode=='fp64' else features[f'{role}_x160'][:,:w],mode,cal[w]);traces[mode]=t
                modes[mode]=dict(scores=scores,fixed_six=core.choose(scores),allow_Q4=core.choose(scores,True))
                if mode=='fp64' and role!='fit':
                    oldrole='old_confirm' if role=='confirm' else 'tune'
                    assert np.array_equal(scores,np.asarray(old['results'][str(w)]['phases'][oldrole]['scores']))
                for regime in ('fixed_six','allow_Q4'):policy(mode+'__'+regime,modes[mode][regime],regime)
                if mode!='fp64':bridges[mode]=dict(score_error=u.error(scores,modes['fp64']['scores']),
                    fixed_six_flips=int(np.sum(modes[mode]['fixed_six']!=modes['fp64']['fixed_six'])),
                    allow_Q4_flips=int(np.sum(modes[mode]['allow_Q4']!=modes['fp64']['allow_Q4'])),
                    saturation_by_layer={k:np.sum(v.get('input_saturation',np.zeros((len(ids),1),bool)),axis=1) for k,v in t.items()})
            np.savez_compressed(out/f'{role}_{w}_ARRAYS_V1.npz',**u.flatten(dict(results=modes,trace=traces)),ids=pack['ids'])
            raw={f['target_id']:f for f in u.read(packdir/'QUALITY_V1.json')['frames']};frames=[raw[i] for i in ids]
            fs,summary,decisions,detail,seqs=ev.summarize_policies(frames,pack,policies,hardware);comparisons={}
            for mode in modes:
                for regime in ('fixed_six','allow_Q4'):
                    name=mode+'__'+regime;comparisons[name+'__vs_Q4']=ev.comparison(fs,summary,name,'Q4',seqs)
                    if mode!='fp64':comparisons[name+'__vs_FP64']=ev.comparison(fs,summary,name,'fp64__'+regime,seqs)
            u.write(out/f'{role}_{w}_QUALITY_V1.json',dict(summaries=summary,decisions=decisions,frames=detail,comparisons=comparisons,bridges=bridges,
                role=role,fit_is_resubstitution=role=='fit',old_confirm_is_exposed=True))
            allsummary[str(w)][role]=dict(bridges=bridges,AEE={k:v['overall']['all_valid']['pixel_weighted_AEE'] for k,v in summary.items()})
    u.write(out/'SUMMARY_V1.json',dict(status='COMPLETE',results=allsummary,CPU_seconds=time.time()-start,GPU_used=False,new_flow_forwards=0,
        numerical_contract_unchanged=True,no_new_fit=True))
    print('ORDINARY_NUMERIC_COMPLETE',round(time.time()-start,3))

if __name__=='__main__':main()
