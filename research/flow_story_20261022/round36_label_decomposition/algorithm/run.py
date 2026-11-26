"""Run one frozen R36 fit-only decomposition; no tune/confirm or GPU access."""
from pathlib import Path
import os,sys,time,json,hashlib,importlib.util
import numpy as np
import core
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];R33=ROOT/'research/flow_story_20261021/round33_predictability';R34=R33.parent/'round34_deployability'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def default(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,default=default,allow_nan=False)+'\n')
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    start=time.time();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):assert os.environ.get(k)=='2'
    frozen=read(HERE/'SOURCE_FREEZE_V1.json')
    for name,h in frozen['files'].items():assert sha(HERE/name)==h
    for item in frozen['bound'].values():assert sha(item['path'])==item['sha256']
    out=HERE/'actual_v1';assert not out.exists();out.mkdir()
    base=load('r36_base',R33/'algorithm/predictors.py');ev=load('r36_eval',R33/'evaluate.py')
    with np.load(R33/'packs/fit_tune_128_v1/fit.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    assert data['x'].shape==(64,128) and np.all(data['roles'].astype(str)=='fit')
    groups=np.asarray(data['groups'],str);ids=np.asarray(data['ids'],str)
    assert len(set(ids))==64 and len(set(groups))==8 and all(np.sum(groups==g)==8 for g in set(groups))
    fit_quality=read(R34/'bias_control/extension_v1/actual_v1/fit_ALL_ACTION_QUALITY_V1.json')['uncorrected']
    assert len(fit_quality)==64 and [f['target_id'] for f in fit_quality]==ids.tolist()
    raw_y=np.asarray([[f['states'][state]['all_valid']['AEE']-f['states']['background_Q4']['all_valid']['AEE'] for state in ev.STATES[1:]] for f in fit_quality])
    assert np.allclose(raw_y,data['delta_aee'],atol=1e-12,rtol=0)
    assert np.array_equal(data['valid_pixels'],[f['states']['background_Q4']['all_valid']['valid_pixels'] for f in fit_quality])
    result=core.run_oof(data,base)
    for fold in result['folds']:assert len(fold['fit_ids'])==56 and len(fold['held_ids'])==8 and len(fold['fit_groups'])==7
    write(out/'FOLDS_MODELS_V1.json',dict(schema='flow_r36_oof_label_decomposition_v1',alpha=.1,folds=result['folds'],source_freeze_sha256=sha(HERE/'SOURCE_FREEZE_V1.json')))
    arrays=dict(ids=ids,groups=groups,valid_pixels=data['valid_pixels'],c_by_identity=result['c_by_identity'],static_choices=result['static_choices'])
    arrays.update({'scores__'+k:v for k,v in result['scores'].items()});arrays.update({'choices__'+k:v for k,v in result['choices'].items()});arrays.update({'label__'+k:v for k,v in result['labels'].items()})
    np.savez_compressed(out/'OOF_ARRAYS_V1.npz',**arrays)
    policies={}
    def add(name,choices,regime='fixed_six',observable='fold_frozen_runtime_X128_only',control='dynamic'):
        a=np.asarray(choices,np.int64);policies[name]=dict(weights=ev.policy_weights(a),choices=a,observable=observable,regime=regime,operator_control=control,label_source=None,stochastic_expectation=False)
    for name,a in result['choices'].items():add(name,a)
    add('fit_static',result['static_choices'],observable='training_fold_only_constant',control='static')
    add('Q4',np.full(64,-1),regime='Q4_only',observable='constant',control='static')
    add('oracle_six',base.choose_actions(data['delta_aee']),observable='offline_GT_six_counterfactuals')
    add('oracle_seven',base.choose_actions(data['delta_aee'],True),regime='allow_Q4',observable='offline_GT_seven_counterfactuals')
    policies['uniform_six_expectation']=dict(weights=np.tile(np.r_[0,np.ones(6)/6],(64,1)),choices=None,observable='uniform_random_expectation',regime='fixed_six',operator_control='dynamic',label_source=None,stochastic_expectation=True)
    allframes,summary,decisions,frames,seqs=ev.summarize_policies(fit_quality,data,policies,read(R33/'hardware/results/SUMMARY_V1.json'))
    comparisons={name+'__vs_Q4':ev.comparison(allframes,summary,name,'Q4',seqs) for name in policies if name!='Q4'}
    for other in policies:
        if other!='projection':comparisons['projection__vs_'+other]=ev.comparison(allframes,summary,'projection',other,seqs)
    leave={}
    for other in ('direct','online_signed_g1','fit_static','Q4'):
        leave[other]={}
        for removed in seqs:
            selected=[f for f in allframes if f['sequence']!=removed];leave[other][removed]={}
            for reg in ev.REGIONS:
                p=ev.mc.aggregate([f['states']['projection'][reg] for f in selected]);d=ev.mc.aggregate([f['states'][other][reg] for f in selected]);pa,da=p['pixel_weighted_AEE'],d['pixel_weighted_AEE']
                leave[other][removed][reg]=dict(projection=pa,reference=da,delta=ev.mc.difference(pa,da),relative=ev.mc.ratio(pa,da))
    labels=result['labels'];weight=np.asarray(data['valid_pixels'],np.float64);y=labels['y'];pchoice=result['choices']['projection'];dchoice=result['choices']['direct']
    regret_delta=y[np.arange(64),dchoice]-y[np.arange(64),pchoice];changed=pchoice!=dchoice
    explanatory={};model_for_label={'y':'direct','p':'projection','r':'residual_only_diagnostic'}
    for scope in ['overall']+seqs:
        m=np.ones(64,bool) if scope=='overall' else groups==scope;w=weight[m];scope_result={}
        def avg(a):return float(np.sum(a*w[:,None])/(w.sum()*a.shape[1]))
        for label,name in model_for_label.items():
            truth=labels[label][m];pred=result['scores'][name][m];tc=truth-truth.mean(axis=1,keepdims=True);pc=pred-pred.mean(axis=1,keepdims=True)
            mean=float(np.sum(truth*w[:,None])/(6*w.sum()))
            scope_result[label]=dict(pixel_weighted_MSE=avg((pred-truth)**2),frame_mean_MSE=float(np.mean((pred-truth)**2)),
                action_centered_pixel_weighted_MSE=avg((pc-tc)**2),action_centered_target_second_moment=avg(tc**2),
                action_centered_prediction_second_moment=avg(pc**2),target_total_variance=avg((truth-mean)**2),
                zero_predictor_pixel_weighted_MSE=avg(truth**2),target_mean=mean)
        dif=regret_delta[m];ch=changed[m];cm=changed&m
        trueorder=np.sign(y[m,:,None]-y[m,None,:]);triu=np.triu(np.ones((6,6),bool),1)
        order={}
        for name in ('direct','projection','online_signed_g1'):
            score=result['scores'][name][m];predorder=np.sign(score[:,:,None]-score[:,None,:]);wrong=(predorder[:,triu]!=trueorder[:,triu]);order[name]=dict(wrong_pairs=int(wrong.sum()),total_pairs=int(wrong.size),pixel_weighted_disagreement=float(np.average(wrong.mean(axis=1),weights=w)))
        scope_result['residual_perturbation']=dict(changed_actions=int(ch.sum()),identity_count=int(m.sum()),
            direct_worse_than_projection_identities=int(np.sum(dif>0)),direct_better_than_projection_identities=int(np.sum(dif<0)),ties=int(np.sum(dif==0)),
            actual_direct_minus_projection_regret_pixel_weighted=float(np.average(dif,weights=w)),actual_delta_frame_mean=float(np.mean(dif)),
            changed_only_pixel_weighted_delta=float(np.average(regret_delta[cm],weights=weight[cm])) if cm.any() else None,
            adding_predicted_residual_is_exactly_direct=True)
        scope_result['true_pairwise_action_order_error']=order;explanatory[scope]=scope_result
    output=dict(schema='flow_r36_fit_only_oof_quality_v1',source_freeze_sha256=sha(HERE/'SOURCE_FREEZE_V1.json'),summaries=summary,decisions=decisions,
        comparisons=comparisons,frames=frames,leave_one_sequence_projection_vs_reference=leave,explanatory_diagnostics=explanatory,
        uniform_six_actual_tail_risk=ev.uniform_tail_risk(fit_quality),data_scope='Only original64 fit,8seq LOSO; no new confirmation',
        all16_GT_regions=True,ordinary_function_class_same=True,tail_flags_not_new_gates=True,old_operator_costs_not_fullnet=True)
    write(out/'EVALUATION_V1.json',output)
    result_summary={name:dict(AEE=s['overall']['all_valid']['pixel_weighted_AEE'],frame_mean_AEE=s['overall']['all_valid']['frame_mean_AEE'],
        B=s['overall']['B']['pixel_weighted_AEE'],L=s['overall']['L']['pixel_weighted_AEE'],regret_six_pixel_weighted=decisions[name]['overall']['regret_six_action_pixel_weighted'],
        regret_six_frame_mean=decisions[name]['overall']['regret_six_action_frame_mean'],action_histogram=decisions[name]['overall']['action_histogram_expected'],
        tails_vs_Q4=0 if name=='Q4' else len(comparisons[name+'__vs_Q4']['high_support_tail_review'])) for name,s in summary.items()}
    write(out/'SUMMARY_V1.json',dict(status='COMPLETE',policies=result_summary,explanatory_overall=explanatory['overall'],fold_equality={f['held_group']:f['equality'] for f in result['folds']},
        source_freeze_sha256=sha(HERE/'SOURCE_FREEZE_V1.json'),CPU_seconds=time.time()-start,GPU_used=False,fit_identities=64,folds=8,hyperparameter_search=False,tune_confirm_quality_access=False))
    print('R36_FIT_ONLY_COMPLETE',round(time.time()-start,3),flush=True)
if __name__=='__main__':main()
