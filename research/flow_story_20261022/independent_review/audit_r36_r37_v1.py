"""Independent primal refits, exhaustive greedy splits and fit-only raw metrics."""
from pathlib import Path
import json,hashlib,os
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];RESEARCH=ROOT/'research'
R33=RESEARCH/'flow_story_20261021/round33_predictability';R34=R33.parent/'round34_deployability'
R36=HERE.parent/'round36_label_decomposition/algorithm';R37=HERE.parent/'round37_cheap_selector/algorithm'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-11)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def ridge(x,y,w):
    p=w/w.sum();mu=np.sum(x*p[:,None],axis=0)
    sd=np.sqrt(np.sum((x-mu)**2*p[:,None],axis=0));sd=np.where(sd>1e-12,sd,1.)
    z=(x-mu)/sd;inter=np.sum(y*p[:,None],axis=0)
    coef=np.linalg.solve(z.T@(p[:,None]*z)+.1*np.eye(x.shape[1]),z.T@(p[:,None]*(y-inter)))
    return dict(mean=mu,scale=sd,intercept=inter,coef=coef)
def pred(m,x):return ((x-m['mean'])/m['scale'])@m['coef']+m['intercept']
def check_model(m,saved):
    for k in m:close(m[k],saved[k])
    assert saved['alpha']==.1

def tree(x,y,w,depth=0):
    sums=np.sum(y*w[:,None],axis=0);a=int(sums.argmin());best=float(sums[a]);choice=None
    if depth<2:
        for j in range(x.shape[1]):
            for cut in np.unique(x[:,j])[:-1]:
                left=x[:,j]<=cut
                if left.sum()<8 or (~left).sum()<8:continue
                val=float(np.sum(y[left]*w[left,None],axis=0).min()+np.sum(y[~left]*w[~left,None],axis=0).min())
                if val/w.sum()<best/w.sum()-1e-15:best=val;choice=(j,int(cut),left)
    node=dict(action=a,cost_sum=float(sums[a]),rows=len(x),weight=int(w.sum()),action_cost_sums=sums)
    if choice is not None:
        j,cut,left=choice;node.update(feature=j,threshold=cut,left=tree(x[left],y[left],w[left],depth+1),right=tree(x[~left],y[~left],w[~left],depth+1))
    return node
def compare_tree(a,b):
    assert set(a)==set(b)
    for k in a:
        if k in ('left','right'):compare_tree(a[k],b[k])
        elif k in ('cost_sum','action_cost_sums'):close(a[k],b[k])
        else:assert a[k]==b[k],(k,a[k],b[k])
def tree_predict(m,x):
    out=[]
    for r in x:
        n=m
        while 'feature' in n:n=n['left'] if r[n['feature']]<=n['threshold'] else n['right']
        out.append(n['action'])
    return np.array(out)

def metrics(action,errors,counts,groups,names,ids):
    es=errors[:,1:].mean(axis=1) if action is None else errors[np.arange(len(ids)),action+1]
    ai=names.index('all_valid');base=errors[:,0];oracle=errors[:,1:,ai].min(axis=1);summary={};tails=[]
    for seq in ['overall']+sorted(set(groups)):
        ix=np.ones(len(ids),bool) if seq=='overall' else groups==seq;rr={}
        for j,rn in enumerate(names):
            n=int(counts[ix,j].sum());err=float(es[ix,j].sum());nz=ix&(counts[:,j]>0)
            rr[rn]=dict(valid_pixels=n,error_sum=err,AEE=err/n if n else None,frame_mean_AEE=float((es[nz,j]/counts[nz,j]).mean()) if nz.any() else None)
        summary[seq]=dict(regions=rr,regret=float((es[ix,ai]-oracle[ix]).sum()/counts[ix,ai].sum()),frame_regret=float(((es[ix,ai]-oracle[ix])/counts[ix,ai]).mean()))
    for i,tid in enumerate(ids):
        for rn in ['all_valid','B','L']:
            j=names.index(rn);n=counts[i,j]
            if n>=256 and (es[i,j]>1.2*base[i,j] or es[i,j]-base[i,j]>n):tails.append([str(tid),rn])
    return dict(summary=summary,tails=tails,errors=es)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out=HERE/'R36_R37_INDEPENDENT_AUDIT_V1.json';assert not out.exists()
    f36=read(R36/'SOURCE_FREEZE_V1.json');f37=read(R37/'SOURCE_FREEZE_V1.json')
    for k,h in f36['files'].items():assert sha(R36/k)==h
    for x in f36['bound'].values():assert sha(x['path'])==x['sha256']
    for k,h in f37['files'].items():assert sha(RESEARCH/k)==h
    data=load(R33/'packs/fit_tune_128_v1/fit.npz');x=data['x'];w=data['valid_pixels'].astype(float);y=data['delta_aee'];aux=data['aux_g1'];ids=data['ids'];groups=data['groups']
    assert x.shape==(64,128) and set(data['roles'])=={'fit'}
    qpath=R34/'bias_control/extension_v1/actual_v1/fit_ALL_ACTION_QUALITY_V1.json';q=read(qpath)['uncorrected'];assert len(q)==64 and [r['target_id'] for r in q]==ids.tolist()
    states=['background_Q4']+[f'upgrade_{i}' for i in range(6)];names=list(q[0]['states'][states[0]]);assert len(names)==16
    counts=np.array([[r['states'][states[0]][n]['valid_pixels'] for n in names] for r in q]);errors=np.array([[[r['states'][s][n]['error_sum'] for n in names] for s in states] for r in q]);ai=names.index('all_valid')
    assert np.array_equal(counts[:,ai],w);close((errors[:,1:,ai]-errors[:,:1,ai])/w[:,None],y)
    # Independently rebuild count32 from all64 original raw input arrays only.
    members={r['target_id']:r for r in read(R33/'data/MEMBERSHIP_V1.json')['records']};count=[];rawbind={}
    for tid in ids:
        rec=members[tid];f=next(f for f in rec['source_files'] if f['category']=='input');p=R33/'data/fit_tune_source_snapshot'/f['relative_path'];assert sha(p)==f['sha256'];v=np.load(p,allow_pickle=False);row=[]
        for cy in range(4):
            for cx in range(4):
                a=v[:,cy*120:(cy+1)*120,cx*160:(cx+1)*160];row.extend([int(np.count_nonzero(a>0)),int(np.count_nonzero(a<0))])
        count.append(row);rawbind[str(tid)]=f['sha256']
    count=np.asarray(count);columns=np.array([8*c+j for c in range(16) for j in (2,3)]);close(count/192000,x[:,columns])
    a36=load(R36/'actual_v1/OOF_ARRAYS_V1.npz');m36=read(R36/'actual_v1/FOLDS_MODELS_V1.json');p37=read(R37/'actual_v1/PREDICTIONS_BEFORE_QUALITY_V1.json');assert p37['quality_not_read_yet']
    assert np.array_equal(count,p37['counts']);scores36={k:np.empty((64,6)) for k in ['direct','projection','residual_only_diagnostic','recombined','online_signed_g1']};actions37={k:np.full(64,-99,dtype=int) for k in p37['actions']};static=np.empty(64,int);cs=np.empty((64,2));p_label=np.empty((64,6));ridge_count=0;tree_count=0
    for g in sorted(set(groups)):
        held=groups==g;tr=~held;ww=w[tr]/w[tr].sum();c=np.sum(aux[tr,:2]*ww[:,None],axis=0);cs[held]=c
        pp=np.sum(aux[tr,2:].reshape(-1,6,2)*c[None,None,:],axis=2);p_label[held]=np.sum(aux[held,2:].reshape(-1,6,2)*c[None,None,:],axis=2)
        sf36=next(f for f in m36['folds'] if f['held_group']==g);sf37=next(f for f in p37['folds'] if f['held_group']==g)
        assert sf36['fit_ids']==ids[tr].tolist() and sf36['held_ids']==ids[held].tolist();assert sf37['fit']['fit_ids']==ids[tr].tolist() and sf37['held_ids']==ids[held].tolist();close(c,sf36['c']);close(c,sf37['fit']['c'])
        own={}
        for name,target in [('direct',y[tr]),('projection',pp),('residual_only_diagnostic',y[tr]-pp)]:
            model=ridge(x[tr],target,w[tr]);own[name]=model;check_model(model,sf36['models'][name]);scores36[name][held]=pred(model,x[held]);ridge_count+=1
        am=ridge(x[tr],aux[tr],w[tr]);check_model(am,sf36['aux_model']);ap=pred(am,x[held]);ridge_count+=1;scores36['online_signed_g1'][held]=np.sum(ap[:,:2,None].transpose(0,2,1)*ap[:,2:].reshape(-1,6,2),axis=2)
        scores36['recombined'][held]=scores36['projection'][held]+scores36['residual_only_diagnostic'][held]
        for k in ['coef','intercept']:close(own['direct'][k],own['projection'][k]+own['residual_only_diagnostic'][k])
        close(scores36['projection'][held],np.sum(ap[:,2:].reshape(-1,6,2)*c,axis=2))
        static[held]=np.sum(y[tr]*ww[:,None],axis=0).argmin();assert sf36['static_action']==static[held][0]
        for target_name,target in [('direct',y[tr]),('projection',pp)]:
            for label,features in [('X128',x),('count32',count)]:
                key=f'{label}_ridge_{target_name}';model=ridge(features[tr],target,w[tr]);check_model(model,sf37['fit']['models'][key]);predicted=pred(model,features[held]);close(predicted,sf37['predictions'][key]['scores']);actions37[key][held]=predicted.argmin(axis=1);ridge_count+=1
            key='count32_tree_'+target_name;tm=tree(count[tr],target,w[tr]);compare_tree(tm,sf37['fit']['models'][key]);actions37[key][held]=tree_predict(tm,count[held]);tree_count+=1
            key='static_'+target_name;sa=int(np.sum(target*w[tr,None],axis=0).argmin());assert sa==sf37['fit']['models'][key]['action'];actions37[key][held]=sa
    close(a36['c_by_identity'],cs);close(a36['label__p'],p_label);close(a36['label__y'],y);close(a36['label__r'],y-p_label);assert np.array_equal(static,a36['static_choices'])
    actions36={k:s.argmin(axis=1) for k,s in scores36.items()}
    for k,s in scores36.items():close(s,a36['scores__'+k]);assert np.array_equal(actions36[k],a36['choices__'+k])
    for k,v in actions37.items():assert np.array_equal(v,p37['actions'][k])
    assert np.array_equal(actions36['direct'],actions37['X128_ridge_direct']) and np.array_equal(actions36['projection'],actions37['X128_ridge_projection'])
    oracle=y.argmin(axis=1);oracle7=np.where(y.min(axis=1)>=0,-1,oracle);actions36.update(fit_static=static,Q4=np.full(64,-1),oracle_six=oracle,oracle_seven=oracle7,uniform_six_expectation=None);actions37.update(Q4=np.full(64,-1),oracle6=oracle,uniform6=None)
    e36=read(R36/'actual_v1/EVALUATION_V1.json');e37=read(R37/'actual_v1/RESULTS_V1.json');audits={};metric_count=0
    for roundname,actions,ev in [('R36',actions36,e36),('R37',actions37,e37)]:
        audits[roundname]={}
        for method,act in actions.items():
            m=metrics(act,errors,counts,groups,names,ids)
            for scope,z in m['summary'].items():
                s=ev['summaries'][method][scope] if roundname=='R36' else ev['results'][method]['summaries'][scope]['regions']
                for rn,v in z['regions'].items():
                    assert v['valid_pixels']==s[rn]['valid_pixels'];close(v['error_sum'],s[rn]['error_sum'])
                    if v['valid_pixels']:close(v['AEE'],s[rn]['pixel_weighted_AEE'] if roundname=='R36' else s[rn]['AEE']);close(v['frame_mean_AEE'],s[rn]['frame_mean_AEE'])
                    metric_count+=1
                saved=ev['decisions'][method][scope] if roundname=='R36' else ev['results'][method]['summaries'][scope]
                close(z['regret'],saved['regret_six_action_pixel_weighted'] if roundname=='R36' else saved['pixel_weighted_regret_vs_oracle6']);close(z['frame_regret'],saved['regret_six_action_frame_mean'] if roundname=='R36' else saved['frame_mean_regret_vs_oracle6'])
            if method!='Q4':
                tails=ev['comparisons'][method+'__vs_Q4']['high_support_tail_review'] if roundname=='R36' else ev['results'][method]['high_support_tails'];assert m['tails']==[[v['target_id'],v['region']] for v in tails]
            audits[roundname][method]=dict(summaries=m['summary'],mean_policy_tail_items=m['tails'])
    # Random uniform tail risk is expectation over actual action outcomes, not
    # a tail test on averaged AEE. R37 frozen output only contains the latter.
    risk=[];expected_any=0.
    for i,tid in enumerate(ids):
        union=set();rr={}
        for rn in ['all_valid','B','L']:
            j=names.index(rn);n=counts[i,j];bad=[]
            if n>=256:bad=[a for a in range(6) if errors[i,a+1,j]>1.2*errors[i,0,j] or errors[i,a+1,j]-errors[i,0,j]>n]
            rr[rn]=dict(trigger_actions=bad,probability=len(bad)/6);union.update(bad)
        p=len(union)/6;expected_any+=p;risk.append(dict(target_id=str(tid),sequence=str(groups[i]),regions=rr,any_primary_probability=p))
    residual=scores36['direct'].argmin(axis=1);proj=scores36['projection'].argmin(axis=1);delta=y[np.arange(64),residual]-y[np.arange(64),proj];diagnostics={}
    for label,predicted in [('y',scores36['direct']),('p',scores36['projection']),('r',scores36['residual_only_diagnostic'])]:
        truth={'y':y,'p':p_label,'r':y-p_label}[label];center=truth-truth.mean(axis=1,keepdims=True);pc=predicted-predicted.mean(axis=1,keepdims=True);weight=w[:,None]/(6*w.sum());mse=float(np.sum(weight*(predicted-truth)**2));zero=float(np.sum(weight*truth**2));ac=float(np.sum(weight*(pc-center)**2));tar=float(np.sum(weight*center**2));saved=e36['explanatory_diagnostics']['overall'][label]
        close(mse,saved['pixel_weighted_MSE']);close(zero,saved['zero_predictor_pixel_weighted_MSE']);close(ac,saved['action_centered_pixel_weighted_MSE']);close(tar,saved['action_centered_target_second_moment'])
        diagnostics[label]=dict(MSE=mse,zero_MSE=zero,MSE_over_zero=mse/zero,action_centered_MSE=ac,action_centered_zero_MSE=tar,action_centered_MSE_over_zero=ac/tar)
    perturb=dict(changed=int(np.sum(residual!=proj)),direct_worse=int(np.sum(delta>0)),direct_better=int(np.sum(delta<0)),weighted_direct_minus_projection=float(np.average(delta,weights=w)))
    close(perturb['weighted_direct_minus_projection'],e36['explanatory_diagnostics']['overall']['residual_perturbation']['actual_direct_minus_projection_regret_pixel_weighted'])
    result=dict(status='PASS_NUMERICS_WITH_R37_SCOPE_CAVEATS',auditor_sha256=sha(__file__),R36_source_freeze_sha256=sha(R36/'SOURCE_FREEZE_V1.json'),R37_source_freeze_sha256=sha(R37/'SOURCE_FREEZE_V1.json'),R36_summary_sha256=sha(R36/'actual_v1/SUMMARY_V1.json'),R37_results_sha256=sha(R37/'actual_v1/RESULTS_V1.json'),fit_quality_sha256=sha(qpath),fit_pack_sha256=sha(R33/'packs/fit_tune_128_v1/fit.npz'),raw_count_input_sha256=rawbind,
        counts=dict(independent_primal_ridges=ridge_count,independent_exhaustive_greedy_trees=tree_count,raw_count_inputs=64,all_region_aggregates=metric_count),
        independent_quality=audits,label_predictability=diagnostics,residual_change=perturb,uniform_actual_tail_risk=dict(frames=risk,expected_any_primary_bad_frames=expected_any),
        scope_caveats=['R36 only isolated64fit quality decoded; R37 original combined fit_tune JSON was decoded after predictions froze, but nonfit values not used in fitting/evaluation/selection; do not claim never read/decoded','R37 uniform high_support_tails tests expectedAEE only; actual randomized risk independently provided here','All64 fit previously used for development; OOF is mechanistic diagnostic, not new independent confirmation','R36 cached operator costs omit actual full feature/head prefix and are not deployment-cost comparison'],GPU_calls=0,new_network_training=False)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=result['status'],sha256=sha(out),counts=result['counts'],perturb=perturb,diagnostics=diagnostics,uniform_expected_bad=expected_any)))

if __name__=='__main__':main()
