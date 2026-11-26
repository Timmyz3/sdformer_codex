"""Independent R34 nested-calibration mathematics and cached-outcome audit.

Does not import the calibration implementation or its ridge helper. It
reconstructs auxiliary OOF fits only to verify archived coefficients, never
changes the deployment model or selected settings.
"""
from pathlib import Path
import hashlib,json,os
import numpy as np

B=Path(__file__).resolve().parent.parent
R33=B.parent/'round33_predictability'
CAND=[('zero_residual',None,0.)]+[(f'alpha_{a:g}_shrink_{s:g}',a,s) for s in [.25,1.] for a in [10.,1.,.1]]
COUNTS={'archived_families':0,'archived_ridges':0,'auxiliary_OOF_splits':0,'outer_selection_folds':0,'per_frame_regions':0,'aggregated_regions':0}
MAX={'coefficient_difference':0.,'weighted_stationarity':0.}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def close(a,b):assert np.allclose(a,b,rtol=2e-9,atol=5e-10),(np.max(np.abs(np.asarray(a)-np.asarray(b))),np.asarray(a).shape)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def subset(d,ids):
    lookup={v:i for i,v in enumerate(d['ids'].tolist())};ix=[lookup[v] for v in ids]
    return {k:v[ix] for k,v in d.items()}
def predict(m,x):return ((x-np.asarray(m['mean']))/np.asarray(m['scale']))@np.asarray(m['coef'])+np.asarray(m['intercept'])
def signed(z):return (z[:,:2,None]*z[:,2:].reshape(len(z),6,2).transpose(0,2,1)).sum(axis=1)
def augmented(x,z):
    products=z[:,:2,None]*z[:,2:].reshape(len(z),6,2).transpose(0,2,1)
    return np.concatenate([x,z,products.transpose(0,2,1).reshape(len(z),12)],axis=1)
def ref_ridge(x,y,w,a):
    p=w.astype(float)/sum(w);mean=np.sum(p[:,None]*x,axis=0)
    scale=np.sqrt(np.sum(p[:,None]*(x-mean)**2,axis=0));scale[scale<=1e-12]=1
    intercept=np.sum(p[:,None]*y,axis=0);xx=(x-mean)/scale
    aa=xx*np.sqrt(p[:,None]);bb=(y-intercept)*np.sqrt(p[:,None])
    coef=aa.T@np.linalg.solve(aa@aa.T+a*np.eye(len(x)),bb)
    return dict(mean=mean,scale=scale,intercept=intercept,coef=coef)
def check_ridge(saved,x,y,w,a):
    ref=ref_ridge(x,y,w,a)
    assert saved['alpha']==a and saved['fit_rows']==len(x)
    assert saved['input_width']==x.shape[1] and saved['output_width']==y.shape[1]
    for key,v in ref.items():
        close(saved[key],v)
        if key=='coef':MAX['coefficient_difference']=max(MAX['coefficient_difference'],float(np.max(np.abs(np.asarray(saved[key])-v))))
    p=w/sum(w);xx=(x-ref['mean'])/ref['scale'];coef=np.asarray(saved['coef'])
    grad=xx.T@(p[:,None]*(xx@coef-(y-ref['intercept'])))+a*coef
    residual=float(np.max(np.abs(grad)));assert residual<1e-8,residual
    MAX['weighted_stationarity']=max(MAX['weighted_stationarity'],residual);COUNTS['archived_ridges']+=1

def audit_family(f,full,expected_ids,cache):
    assert f['fit_ids']==expected_ids
    d=subset(full,expected_ids);groups=d['groups'];w=d['valid_pixels'];x=d['x'];y=d['aux_g1']
    assert f['fit_groups']==sorted(set(groups.tolist()))
    check_ridge(f['auxiliary'],x,y,w,.1)
    oof=np.zeros_like(y);seen=set()
    assert len(f['auxiliary_OOF_trace'])==len(set(groups))
    for t in f['auxiliary_OOF_trace']:
        seq=t['held_group'];assert seq not in seen;seen.add(seq)
        held=groups==seq;fit=~held
        assert t['fit_ids']==d['ids'][fit].tolist() and t['held_ids']==d['ids'][held].tolist()
        assert t['fit_groups']==sorted(set(groups[fit].tolist()))
        assert not set(t['fit_ids'])&set(t['held_ids'])
        key=tuple(t['fit_ids'])
        if key not in cache:cache[key]=ref_ridge(x[fit],y[fit],w[fit],.1)
        oof[held]=predict(cache[key],x[held]);COUNTS['auxiliary_OOF_splits']+=1
    xx=augmented(x,oof);target=d['delta_aee']-signed(oof)
    for a in [10.,1.,.1]:check_ridge(f['residual'][str(a)],xx,target,w,a)
    COUNTS['archived_families']+=1

def scores(f,x):
    z=predict(f['auxiliary'],x);ss=signed(z);xx=augmented(x,z)
    return np.stack([ss if s==0 else ss+s*predict(f['residual'][str(a)],xx) for _,a,s in CAND])

def audit_grouped(trace,full,ids,cache):
    d=subset(full,ids);out=np.zeros((7,len(ids),6));seen=set()
    assert len(trace)==len(set(d['groups']))
    for t in trace:
        seq=t['held_group'];assert seq not in seen;seen.add(seq);held=d['groups']==seq
        assert t['held_ids']==d['ids'][held].tolist()
        audit_family(t['training'],full,d['ids'][~held].tolist(),cache)
        out[:,held,:]=scores(t['training'],d['x'][held])
    return out

def selected_index(s,d,losses):
    actions=s.argmin(axis=2);deltas=np.take_along_axis(d['delta_aee'][None],actions[:,:,None],axis=2)[:,:,0]
    obj=(deltas*d['valid_pixels'][None]).sum(axis=1)/sum(d['valid_pixels'])
    assert len(losses)==7
    for saved,(name,a,h),v in zip(losses,CAND,obj):
        assert saved['name']==name and saved['alpha']==a and saved['shrink']==h;close(saved['weighted_delta_AEE'],v)
    return int(np.argmin(obj))

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out=B/'independent_review/ORDINARY_INDEPENDENT_AUDIT_V1.json';assert not out.exists()
    actual=B/'ordinary/actual_v1';source=B/'ordinary/run_calibration_v1.py'
    freeze=read(B/'ordinary/SOURCE_FREEZE_V1.json');receipt=read(actual/'EXECUTION_RECEIPT_V1.json');res=read(actual/'RESULTS_V1.json')
    for p,h in freeze['files'].items():assert sha(B/p)==h
    assert sha(source)==receipt['source_sha256']==res['source_sha256']
    assert receipt['GPUs_used']==0 and res['GPU_used'] is False and res['new_flow_forwards']==0
    for p,h in receipt['files'].items():assert sha(actual/p)==h
    members=read(R33/'data/MEMBERSHIP_V1.json');fitids=[r['target_id'] for r in members['records'] if r['role']=='fit']
    assert res['sources']['membership']==sha(R33/'data/MEMBERSHIP_V1.json')
    assert res['sources']['predictor_source']==sha(R33/'algorithm/predictors.py')
    assert res['sources']['hardware']==sha(R33/'hardware/results/SUMMARY_V1.json')
    hw={r['action']:r for r in read(R33/'hardware/results/SUMMARY_V1.json')['candidates']}
    table={}
    for dim in [128,160]:
        p=R33/f'packs/fit_tune_{dim}_v1/fit.npz';d=load(p);assert d['ids'].tolist()==fitids
        m=read(actual/f'MODEL_{dim}_V1.json');nested=read(actual/f'FIT_NESTED_{dim}_V1.json');cache={}
        assert m['fit_pack_sha256']==sha(p)==res['sources'][f'fit_{dim}'] and m['fit_ids']==fitids
        assert m['source_sha256']==sha(source) and m['old_confirm_is_exposed']
        assert [(r['name'],r['alpha'],r['shrink']) for r in m['candidates']]==CAND
        audit_family(m['family'],d,fitids,cache)
        allcv=audit_grouped(m['selection_trace'],d,fitids,cache)
        k=selected_index(allcv,d,m['candidate_losses']);assert k==m['selection_index']
        assert m['selection']==m['candidates'][k]==res['results'][str(dim)]['final_selection']
        independent_nested=np.zeros((64,6));seen=set()
        for t in nested['trace']:
            seq=t['held_group'];assert seq not in seen;seen.add(seq);held=d['groups']==seq;train=subset(d,d['ids'][~held].tolist())
            assert t['held_ids']==sorted(d['ids'][held].tolist())
            s=audit_grouped(t['inner_trace'],d,train['ids'].tolist(),cache)
            ck=selected_index(s,train,t['inner_losses']);assert t['chosen']==m['candidates'][ck]
            audit_family(t['outer_fit'],d,train['ids'].tolist(),cache)
            independent_nested[held]=scores(t['outer_fit'],d['x'][held])[ck];COUNTS['outer_selection_folds']+=1
        assert len(seen)==8;close(nested['scores'],independent_nested);assert nested['ids']==fitids
        table[str(dim)]={'chosen':m['selection'],'roles':{}}
        for phase,role,folder in [('fit_loso','fit','fit_tune'),('tune','tune','fit_tune'),('old_confirm','confirm','confirm')]:
            packpath=R33/f'packs/{folder}_{dim}_v1/{role}.npz';q=load(packpath)
            qp=packpath.parent/'QUALITY_V1.json';qa=read(R33/'independent_review'/('CONFIRM_PACK_INDEPENDENT_AUDIT_V1.json' if role=='confirm' else 'FIT_TUNE_PACK_INDEPENDENT_AUDIT_V1.json'))
            assert qa['status']=='PASS' and sha(qp)==qa['quality_sha256']
            assert res['sources'][f'{phase}_{dim}_pack']==sha(packpath) and res['sources'][f'{phase}_{dim}_quality']==sha(qp)
            saved=res['results'][str(dim)]['phases'][phase]
            s=independent_nested if phase=='fit_loso' else scores(m['family'],q['x'])[k]
            close(saved['scores'],s);assert saved['ids']==q['ids'].tolist()
            if phase!='fit_loso':close(saved['all_final_candidate_scores'],scores(m['family'],q['x']))
            raw={r['target_id']:r for r in read(qp)['frames']};ids=q['ids'].tolist();regions=list(raw[ids[0]]['states']['background_Q4']);assert len(regions)==16
            nn=np.array([[raw[i]['states']['background_Q4'][r]['valid_pixels'] for r in regions] for i in ids]);ai=regions.index('all_valid')
            base=np.array([[raw[i]['states']['background_Q4'][r]['error_sum'] for r in regions] for i in ids]);roleout={}
            for allow,label in [(False,'fixed_six'),(True,'allow_Q4')]:
                choices=s.argmin(axis=1)
                if allow:choices=np.where(s.min(axis=1)>=0,-1,choices)
                ss=saved['quality'][label];assert ss['choices']==choices.tolist()
                ee=np.array([[raw[i]['states']['background_Q4' if a<0 else f'upgrade_{a}'][r]['error_sum'] for r in regions] for i,a in zip(ids,choices)])
                tails=[]
                for j,i in enumerate(ids):
                    assert ss['frames'][j]['target_id']==i and ss['frames'][j]['action']==choices[j]
                    for ri,r in enumerate(regions):
                        value=ss['frames'][j]['regions'][r];assert value['valid_pixels']==nn[j,ri];close(value['error_sum'],ee[j,ri]);COUNTS['per_frame_regions']+=1
                        if nn[j,ri]:close(value['AEE'],ee[j,ri]/nn[j,ri])
                        else:assert value['AEE'] is None
                    for r in ['all_valid','B','L']:
                        ri=regions.index(r);n=nn[j,ri]
                        if n>=256 and (ee[j,ri]>1.2*base[j,ri] or ee[j,ri]-base[j,ri]>n):tails.append((i,r))
                assert tails==[(r['target_id'],r['region']) for r in ss['high_support_tail_review']]
                seqrows={};leaveout={}
                for seq in ['overall']+list(dict.fromkeys(q['groups'].tolist())):
                    ix=np.ones(len(ids),bool) if seq=='overall' else q['groups']==seq
                    for ri,r in enumerate(regions):
                        n=int(nn[ix,ri].sum());e=float(ee[ix,ri].sum());v=ss['summaries'][seq][r]
                        assert v['valid_pixels']==n;close(v['error_sum'],e)
                        if n:close(v['pixel_weighted_AEE'],e/n)
                        else:assert v['pixel_weighted_AEE'] is None
                        nonempty=ix&(nn[:,ri]>0)
                        if nonempty.any():close(v['frame_mean_AEE'],np.mean(ee[nonempty,ri]/nn[nonempty,ri]))
                        else:assert v['frame_mean_AEE'] is None
                        COUNTS['aggregated_regions']+=1
                    if seq!='overall':
                        seqrows[seq]=float((ee[ix,ai]-base[ix,ai]).sum()/nn[ix,ai].sum())
                        leaveout[seq]=float((ee[~ix,ai]-base[~ix,ai]).sum()/nn[~ix,ai].sum())
                ticks=[4*hw[6 if a<0 else int(a)]['dynamic_affine_BN_ticks'] for a in choices]
                close(ss['mean_four_operator_only_ticks'],np.mean(ticks));close(ss['upgrade_fraction'],np.mean(choices>=0))
                aee=float(ee[:,ai].sum()/nn[:,ai].sum());b=float(base[:,ai].sum()/nn[:,ai].sum())
                roleout[label]=dict(AEE=aee,delta_vs_Q4=aee-b,relative_vs_Q4=aee/b-1,upgraded=int((choices>=0).sum()),
                    sequence_delta_vs_Q4=seqrows,leave_one_sequence_delta_vs_Q4=leaveout,tail_items=tails,
                    regional_AEE={r:float(ee[:,ri].sum()/nn[:,ri].sum()) if nn[:,ri].sum() else None for ri,r in enumerate(regions)})
            table[str(dim)]['roles'][phase]=roleout
    report=dict(schema='flow_r34_ordinary_independent_audit_v1',status='PASS',auditor_sha256=sha(__file__),result_sha256=sha(actual/'RESULTS_V1.json'),
        source_sha256=sha(source),counts=COUNTS,max_numeric_discrepancy=MAX,independent_results=table,new_GPU_calls=0,
        new_settings_selected=False,old_confirm_is_exposed_development=True,
        scope='Independent mathematical reconstruction of every archived family/OOF/selection; selected raw outcomes from previously original-array-audited packs. Predictor cost excluded from archived four-operator ticks.')
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':'PASS','counts':COUNTS,'max':MAX,'selected':{k:v['chosen'] for k,v in table.items()}}))

if __name__=='__main__':main()
