"""Fixed alpha=.1, train-fold-only label decomposition. No IO/GPU dependencies."""
import numpy as np
ALPHA=0.1
SCORE_METHODS=('direct','projection','residual_only_diagnostic','recombined','online_signed_g1')

def fold_predict(data,held_group,base):
    x=np.asarray(data['x'],dtype=np.float64);y=np.asarray(data['delta_aee'],dtype=np.float64)
    aux=np.asarray(data['aux_g1'],dtype=np.float64);w=np.asarray(data['valid_pixels'],dtype=np.float64)
    ids=np.asarray(data['ids'],dtype=str);groups=np.asarray(data['groups'],dtype=str)
    held=groups==held_group;fit=~held
    assert held.any() and fit.any() and x.shape==(len(y),128) and aux.shape==(len(y),14)
    assert y.shape==(len(y),6) and np.all(w>0)
    train_w=w[fit]/w[fit].sum();c=np.sum(aux[fit,:2]*train_w[:,None],axis=0)
    d_train=aux[fit,2:].reshape(-1,6,2);p=np.sum(d_train*c[None,None,:],axis=2)
    targets={'direct':y[fit],'projection':p,'residual_only_diagnostic':y[fit]-p}
    models={name:base.train_ridge(x[fit],target,w[fit],ALPHA) for name,target in targets.items()}
    aux_model=base.train_ridge(x[fit],aux[fit],w[fit],ALPHA)
    scores={name:base.ridge_predict(model,x[held]) for name,model in models.items()}
    scores['recombined']=scores['projection']+scores['residual_only_diagnostic']
    aux_pred=base.ridge_predict(aux_model,x[held]);v,dp=base.split_aux(aux_pred,1)
    scores['online_signed_g1']=np.sum(v[:,None,:]*dp,axis=2)
    folded_prediction=np.sum(c[None,None,:]*dp,axis=2)
    eq={}
    for key in ('mean','scale'):
        assert np.array_equal(models['direct'][key],models['projection'][key]) and np.array_equal(models['direct'][key],models['residual_only_diagnostic'][key])
    for key in ('coef','intercept'):
        error=np.max(np.abs(models['direct'][key]-models['projection'][key]-models['residual_only_diagnostic'][key]));eq[key+'_max_abs_error']=float(error)
        assert np.allclose(models['direct'][key],models['projection'][key]+models['residual_only_diagnostic'][key],rtol=1e-10,atol=1e-11)
    eq['prediction_max_abs_error']=float(np.max(np.abs(scores['direct']-scores['recombined'])))
    eq['projection_equals_folded_aux_prediction_max_abs_error']=float(np.max(np.abs(scores['projection']-folded_prediction)))
    assert np.allclose(scores['direct'],scores['recombined'],rtol=1e-10,atol=1e-11)
    assert np.allclose(scores['projection'],folded_prediction,rtol=1e-10,atol=1e-11)
    choices={name:base.choose_actions(score) for name,score in scores.items()}
    assert np.array_equal(choices['direct'],choices['recombined'])
    static=int(np.argmin(np.sum(y[fit]*train_w[:,None],axis=0)))
    return dict(held_mask=held,held_ids=ids[held].tolist(),fit_ids=ids[fit].tolist(),fit_groups=sorted(set(groups[fit])),held_group=str(held_group),
        c=c,models=models,aux_model=aux_model,scores=scores,choices=choices,static_action=static,equality=eq)

def run_oof(data,base):
    n=len(data['ids']);scores={k:np.empty((n,6),np.float64) for k in SCORE_METHODS};c=np.empty((n,2),np.float64)
    static=np.empty(n,np.int64);folds=[];seen=np.zeros(n,np.int64)
    for group in sorted(set(np.asarray(data['groups'],str))):
        f=fold_predict(data,group,base);held=f.pop('held_mask');seen[held]+=1;c[held]=f['c'];static[held]=f['static_action']
        for name in scores:scores[name][held]=f['scores'][name]
        folds.append(f)
    assert np.all(seen==1)
    choices={k:base.choose_actions(v) for k,v in scores.items()};assert np.array_equal(choices['direct'],choices['recombined'])
    # GT/counterfactual labels below are diagnostic-only after all held scores are fixed.
    p=np.sum(np.asarray(data['aux_g1'],np.float64)[:,2:].reshape(n,6,2)*c[:,None,:],axis=2)
    return dict(scores=scores,choices=choices,c_by_identity=c,static_choices=static,folds=folds,
        labels=dict(y=np.asarray(data['delta_aee'],np.float64),p=p,r=np.asarray(data['delta_aee'],np.float64)-p))
