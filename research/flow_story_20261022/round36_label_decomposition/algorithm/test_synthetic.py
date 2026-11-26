from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import core
HERE=Path(__file__).resolve().parent
p=HERE.parents[3]/'research/flow_story_20261021/round33_predictability/algorithm/predictors.py';s=importlib.util.spec_from_file_location('r36_testbase',p);base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
rng=np.random.default_rng(20261022);n=32
x=rng.normal(size=(n,128));x[:,3]=2
aux=rng.normal(size=(n,14));y=rng.normal(size=(n,6))
data=dict(x=x,aux_g1=aux,delta_aee=y,valid_pixels=np.arange(1,n+1)*17,ids=np.asarray(['s%d'%i for i in range(n)]),groups=np.repeat(['a','b','c','d'],8))
f=core.fold_predict(data,'a',base);held=data['groups']=='a'
mut={k:np.array(v,copy=True) for k,v in data.items()};mut['aux_g1'][held]*=1e8;mut['delta_aee'][held]+=1e9;mut['valid_pixels'][held]*=100000
q=core.fold_predict(mut,'a',base)
assert np.array_equal(f['c'],q['c'])
for k in f['scores']:assert np.array_equal(f['scores'][k],q['scores'][k]) and np.array_equal(f['choices'][k],q['choices'][k])
for name in f['models']:
 for field in ('coef','intercept','mean','scale'):assert np.array_equal(f['models'][name][field],q['models'][name][field])
# Independent primal ridge solution for direct and p/r decomposition.
fit=~held;w=data['valid_pixels'][fit].astype(float);w/=w.sum();model=f['models']['direct'];a=(x[fit]-model['mean'])/model['scale'];yc=y[fit]-model['intercept'];primal=np.linalg.solve(a.T@(w[:,None]*a)+.1*np.eye(128),a.T@(w[:,None]*yc))
assert np.allclose(primal,model['coef'],atol=1e-11,rtol=1e-10)
out=core.run_oof(data,base);assert len(out['folds'])==4
res=dict(status='PASS',seed=20261022,held_label_aux_weight_perturbation_predictions_exact=True,independent_primal_ridge_max_error=float(np.max(abs(primal-model['coef']))),
    fold_equality=f['equality'],GPU_used=False,real_data_read=False,source_sha256=hashlib.sha256((HERE/'core.py').read_bytes()).hexdigest())
(HERE/'SYNTHETIC_CHECKS_V1.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
