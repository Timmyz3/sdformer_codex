"""Independent same-IOS FP32 arithmetic and per-identity budget verification."""
from pathlib import Path
import hashlib
import json
import numpy as np

B=Path(__file__).resolve().parent
P=B.parent/'round38_frozen_survival'
H=P/'hardware_budget'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def affine(model,x):
    coef=np.asarray(model['coef'],np.float32)
    z=(x-np.asarray(model['mean'],np.float32))/np.asarray(model['scale'],np.float32)
    out=np.zeros(len(model['intercept']),np.float32)
    for j in range(len(out)):
        acc=np.float32(0)
        for k in range(len(z)):
            acc=np.float32(acc+np.float32(z[k]*coef[k,j]))
        out[j]=np.float32(acc+np.float32(model['intercept'][j]))
    return out

r=read(H/'actual_v2/ACTUAL_PREFIX_BUDGET_V1.json')
assert r['status']=='PASS_20_ACTUAL_FROZEN_PREFIX_REPLAYS'
for path,digest in r['bindings'].items():assert sha(Path(path))==digest,path
plan=read(P/'ACTION_PLAN_V1.json')
refs=np.load(P/'FEATURES_AND_SCORES_V1.npz',allow_pickle=False)
fold=read(P/'model_sources/FOLDED_MODEL_V1.json')['ridge']
ordinary=read(P/'model_sources/ORDINARY_MODEL_128_V1.json')
old=B.parents[1]/'flow_story_20261021/round33_predictability/hardware/results/SUMMARY_V1.json'
cost={int(x['action']):4*x['dynamic_affine_BN_ticks'] for x in read(old)['candidates']}
bindings={};margins=[];zeros=[];max_diff=0.
for i,row in enumerate(r['rows']):
    assert row['target_id']==plan['records'][i]['target_id']==refs['ids'][i]
    fd=Path(row['feature_directory'])
    if not fd.is_absolute():fd=H/fd
    vector=fd/'feature.f32';image=fd/'feature.f32.ios'
    x=np.fromfile(vector,dtype='<f4')
    memory=np.fromfile(image,dtype='<f4')
    assert x.shape==(128,) and memory.shape==(131072,)
    assert np.array_equal(x,memory[100000:100128])
    directory=Path(row['directory'])
    assert (directory/'zurich_city_07_a_0010.f32.ios').resolve()==image.resolve()
    z=np.load(row['input_path'],allow_pickle=False).reshape(10,4,120,4,160)
    zero=int(np.sum(~np.any(z!=0,axis=(0,2,4))))
    assert zero==row['zero_mass_cells']
    assert row['feature_service']['feature_ready_tick']==113938624-32*zero
    assert row['feature_service']['DMA_bytes']==12288000 and row['feature_service']['DMA_transactions']==40
    zeros.append(zero)
    computed={'folded128':affine(fold,x)}
    aux=affine(ordinary['family']['auxiliary'],x)
    products=(aux[:2][None,:]*aux[2:].reshape(6,2)).astype(np.float32)
    signed=(products[:,0]+products[:,1]).astype(np.float32)
    augmented=np.concatenate([x,aux,products.reshape(-1)])
    residual=affine(ordinary['family']['residual'][str(ordinary['selection']['alpha'])],augmented)
    computed['ordinary128']=(signed+np.float32(ordinary['selection']['shrink'])*residual).astype(np.float32)
    for name,m in row['models'].items():
        scores=np.asarray(m['actual_head']['scores'],np.float32)
        assert np.array_equal(scores,computed[name]),(row['target_id'],name)
        a=int(np.argmin(scores))
        assert a==m['primary_action']==int(refs['choice__'+name+'_fp32'][i])
        assert plan['records'][i]['policy_actions'][name+'_fp32_fixed6']==f'upgrade_{a}'
        head=8599 if name=='folded128' else 26646
        assert head==m['actual_head']['action_ready_tick']
        assert m['prefix_ticks']==row['feature_service']['feature_ready_tick']+head
        assert m['scoped_total']==m['prefix_ticks']+cost[a]
        assert m['static360_four_isolated_ticks']==1032538860
        assert m['static360_remaining_ticks']==m['scoped_total']-1032538860>0
        events_path=directory/(name+'_head_events.json')
        events=read(events_path)
        first=next(e for e in events if e['space']=='IOS' and not e['write'])
        assert first['address']==400000
        param=directory/(name+'_parameters.bin')
        assert sha(param)==m['parameter_sha256']
        for p in [events_path,param]:bindings[str(p.relative_to(P))]=sha(p)
        if name=='folded128':margins.append(m['static360_remaining_ticks'])
    for p in [vector,image]:bindings[str(p.relative_to(P))]=sha(p)
    max_diff=max(max_diff,float(np.max(np.abs(x-refs['x128_fp32'][i]))))
assert len(margins)==20 and min(margins)==184991 and max(margins)==187075
assert sum(zeros)==4
out=B/'R38_PREFIX_BUDGET_INDEPENDENT_V1.json'
assert not out.exists()
out.write_text(json.dumps(dict(status='PASS_BOUNDED_PREFIX_AND_STATIC_BUDGET',
 auditor='root independent of prefix replay implementer',identities=20,
 same_IOS_head_recomputations=40,zero_mass_cells_total=sum(zeros),
 feature_max_abs_diff_from_primary=max_diff,all_primary_actions_equal=True,
 static360_min_folded_margin=min(margins),static360_max_folded_margin=max(margins),
 full_network=False,new_RTL_for_X128=False,
 timing_scope='Inherited serialized FP32 finite model; scalar RF/routing/FPU and stem limitations unchanged.',
 feature_trace_reexecuted=False,
 source_sha256=sha(Path(__file__)),prefix_report_sha256=sha(H/'actual_v2/ACTUAL_PREFIX_BUDGET_V1.json'),
 action_plan_sha256=sha(P/'ACTION_PLAN_V1.json'),bindings=bindings),indent=2)+'\n')
print('R38_PREFIX_INDEPENDENT_PASS',sha(out))
