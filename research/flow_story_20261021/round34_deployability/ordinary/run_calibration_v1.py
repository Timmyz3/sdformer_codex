"""One bounded fit-only nested residual calibration; exposed-data diagnosis."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import os
import time

import numpy as np

B=Path(__file__).resolve().parent
R33=B.parents[1]/'round33_predictability'
MEMBER_SHA='d860612a3ae3e4620fa2a9a616df9f8e93774efadcf7dc564881e27b7e7da8cc'
PRED_SHA='35663dd6b999621652d9d185d72b967e33c380e1064a09ec75540d2759cd61c3'
METHODS=[dict(name='zero_residual',alpha=None,shrink=0.)]+[
    dict(name=f'alpha_{a:g}_shrink_{s:g}',alpha=a,shrink=s)
    for s in [.25,1.] for a in [10.,1.,.1]]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    assert not Path(p).exists(),str(p)
    Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


assert sha(R33/'algorithm/predictors.py')==PRED_SHA
spec=importlib.util.spec_from_file_location('r33_frozen_ridge',R33/'algorithm/predictors.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def subset(d,mask):return {k:v[mask] for k,v in d.items()}


def fit_family(d):
    x,y,w=d['x'],d['delta_aee'],d['valid_pixels']
    aux=d['aux_g1']
    oof,trace=M.crossfit_auxiliary(x,aux,w,d['ids'],d['groups'],.1)
    am=M.train_ridge(x,aux,w,.1)
    aug=M.augmented_features(x,oof,1)
    residual=y-M.signed_scores(oof,1)
    rms={str(a):M.train_ridge(aug,residual,w,a) for a in [10.,1.,.1]}
    return dict(auxiliary=am,residual=rms,auxiliary_OOF_trace=trace,fit_ids=d['ids'].tolist(),fit_groups=sorted(set(d['groups'].tolist())))


def predict_family(f,x):
    aux=M.ridge_predict(f['auxiliary'],x)
    signed=M.signed_scores(aux,1)
    aug=M.augmented_features(x,aux,1)
    residual={k:M.ridge_predict(v,aug) for k,v in f['residual'].items()}
    return np.stack([signed if p['shrink']==0 else signed+p['shrink']*residual[str(p['alpha'])] for p in METHODS]),aux


def grouped_predictions(d):
    scores=np.empty((len(METHODS),len(d['ids']),6));trace=[]
    for seq in sorted(set(d['groups'])):
        held=d['groups']==seq;train=~held
        f=fit_family(subset(d,train));s,_=predict_family(f,d['x'][held]);scores[:,held,:]=s
        assert not set(f['fit_ids'])&set(d['ids'][held])
        trace.append(dict(held_group=str(seq),held_ids=d['ids'][held].tolist(),training=f))
    return scores,trace


def objective(scores,d):
    choices=np.argmin(scores,axis=2)
    values=np.take_along_axis(d['delta_aee'][None,:,:],choices[:,:,None],axis=2)[:,:,0]
    cost=(values*d['valid_pixels'][None,:]).sum(axis=1)/d['valid_pixels'].sum()
    k=int(np.argmin(cost))
    return k,[dict(**p,weighted_delta_AEE=float(v)) for p,v in zip(METHODS,cost)]


def summarize(scores,d,raw_frames,hardware):
    byid={f['target_id']:f for f in raw_frames};ids=d['ids'].tolist()
    assert all(i in byid for i in ids)
    byaction={v['action']:v for v in hardware['candidates']}
    out={}
    for allow in [False,True]:
        decisions=M.choose_actions(scores,allow)
        metrics=[];tails=[];cost=[]
        for i,tid in enumerate(ids):
            action=int(decisions[i]);frame=byid[tid]
            state='background_Q4' if action<0 else f'upgrade_{action}'
            vals=frame['states'][state];assert len(vals)==16
            for reg in ['all_valid','B','L']:
                r=frame['states']['background_Q4'][reg];c=vals[reg]
                if r['valid_pixels']>=256 and (c['AEE']>r['AEE']*1.2 or c['AEE']-r['AEE']>1):
                    tails.append(dict(target_id=tid,region=reg,candidate=c,reference=r))
            t=4*byaction[6 if action<0 else action]['dynamic_affine_BN_ticks'];cost.append(t)
            metrics.append(dict(target_id=tid,sequence=str(d['groups'][i]),action=action,regions=vals,operator_only_ticks=t))
        summaries={}
        for seq in ['overall']+list(dict.fromkeys(d['groups'].tolist())):
            frames=[x for x in metrics if seq=='overall' or x['sequence']==seq];rows={}
            for reg in frames[0]['regions']:
                n=sum(x['regions'][reg]['valid_pixels'] for x in frames)
                es=sum(x['regions'][reg]['error_sum'] for x in frames)
                means=[x['regions'][reg]['AEE'] for x in frames if x['regions'][reg]['valid_pixels']]
                rows[reg]=dict(valid_pixels=n,error_sum=es,pixel_weighted_AEE=es/n if n else None,frame_mean_AEE=float(np.mean(means)) if means else None)
            summaries[seq]=rows
        out['allow_Q4' if allow else 'fixed_six']=dict(choices=decisions.tolist(),upgrade_fraction=float(np.mean(decisions>=0)),summaries=summaries,
            frames=metrics,high_support_tail_review=tails,mean_four_operator_only_ticks=float(np.mean(cost)),predictor_cost_included=False,
            whole_network=False)
    return out


def main():
    began=time.monotonic();out=B/'actual_v1';out.mkdir(exist_ok=False)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert sha(R33/'data/MEMBERSHIP_V1.json')==MEMBER_SHA
    members=read(R33/'data/MEMBERSHIP_V1.json')
    hardware=read(R33/'hardware/results/SUMMARY_V1.json')
    sources={'membership':MEMBER_SHA,'predictor_source':PRED_SHA,'hardware':sha(R33/'hardware/results/SUMMARY_V1.json')}
    results={}
    for dim,branch in [(128,'voxel128_v1'),(160,'directional160_v1')]:
        fitpath=R33/f'packs/fit_tune_{dim}_v1/fit.npz';d=arrays(fitpath)
        assert d['ids'].tolist()==[r['target_id'] for r in members['records'] if r['role']=='fit']
        assert len(d['ids'])==64 and d['x'].shape==(64,dim) and set(d['roles'])=={'fit'}
        sources[f'fit_{dim}']=sha(fitpath)
        # Outer development folds choose calibration only inside their training sequences.
        nested=np.empty((64,6));outer=[]
        for seq in sorted(set(d['groups'])):
            held=d['groups']==seq;train=subset(d,~held)
            inner,itrace=grouped_predictions(train);chosen,losses=objective(inner,train)
            f=fit_family(train);pred,_=predict_family(f,d['x'][held]);nested[held]=pred[chosen]
            heldids=set(d['ids'][held].tolist())
            assert not heldids&set(f['fit_ids'])
            assert all(not heldids&set(t['training']['fit_ids']) for t in itrace)
            outer.append(dict(held_group=str(seq),held_ids=sorted(heldids),chosen=METHODS[chosen],inner_losses=losses,inner_trace=itrace,outer_fit=f))
        inner,trace=grouped_predictions(d);chosen,losses=objective(inner,d)
        family=fit_family(d)
        original=read(R33/f'fitted/{branch}/MODEL_V1.json')
        for key in ['mean','scale','intercept','coef']:
            assert np.array_equal(family['auxiliary'][key],original['auxiliary']['g1'][key]),key
        model=dict(schema='r34_bounded_residual_calibration_v1',feature_dimension=dim,candidates=METHODS,selection=METHODS[chosen],
            selection_index=chosen,selection_objective='fit-only grouped out-of-fold pixel weighted actual delta AEE, fixed six',
            fit_ids=d['ids'].tolist(),fit_groups=sorted(set(d['groups'].tolist())),fit_pack_sha256=sha(fitpath),family=family,
            source_sha256=sha(__file__),candidate_losses=losses,selection_trace=trace,old_confirm_is_exposed=True)
        mp=out/f'MODEL_{dim}_V1.json';write(mp,M.jsonable(model))
        # No tune or old-confirm labels used above this point.
        write(out/f'FIT_NESTED_{dim}_V1.json',M.jsonable(dict(scores=nested,trace=outer,ids=d['ids'],final_selection=METHODS[chosen])))
        phase_results={}
        for phase,path in [('fit_loso',fitpath),('tune',R33/f'packs/fit_tune_{dim}_v1/tune.npz'),('old_confirm',R33/f'packs/confirm_{dim}_v1/confirm.npz')]:
            q=arrays(path);full,aux=predict_family(family,q['x']);pred=nested if phase=='fit_loso' else full[chosen]
            qp=R33/f'packs/{"confirm" if phase=="old_confirm" else "fit_tune"}_{dim}_v1/QUALITY_V1.json'
            raw=read(qp)['frames'];sources[f'{phase}_{dim}_pack']=sha(path);sources[f'{phase}_{dim}_quality']=sha(qp)
            summary=summarize(pred,q,raw,hardware)
            phase_results[phase]=dict(ids=q['ids'].tolist(),scores=pred.tolist(),model_sha256=sha(mp),quality=summary,
                all_final_candidate_scores=None if phase=='fit_loso' else full.tolist(),new_independent_confirmation=False)
        results[str(dim)]=dict(final_selection=METHODS[chosen],fit_selection_losses=losses,phases=phase_results)
        print('BRANCH_DONE',dim,METHODS[chosen],flush=True)
    write(out/'RESULTS_V1.json',dict(schema='r34_ordinary_bounded_calibration_results_v1',status='COMPLETE',source_sha256=sha(__file__),sources=sources,
        results=results,GPU_used=False,new_flow_forwards=0,old_confirm_is_exposed_development=True,fit_only_hyperparameter_selection=True))
    write(out/'EXECUTION_RECEIPT_V1.json',dict(status='COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-began,
        files={p.name:sha(p) for p in out.iterdir()},source_sha256=sha(__file__),GPUs_used=0))


if __name__=='__main__':main()
