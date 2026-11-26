"""Independent frozen-head/selected-outcome evaluation audit; no model fitting."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def close(a,b):assert np.allclose(a,b,rtol=2e-11,atol=2e-10),(a,b)
def ridge(m,x):return ((x-np.asarray(m['mean']))/np.asarray(m['scale']))@np.asarray(m['coef'])+np.asarray(m['intercept'])
def contraction(aux,g):
    d=2*g*g;return (aux[:,:d,None]*aux[:,d:].reshape(len(aux),6,d).transpose(0,2,1)).sum(axis=1)
def choose(s,allow):
    a=np.argmin(s,axis=1);return np.where(np.min(s,axis=1)>=0,-1,a) if allow else a

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['evaluation','pack-audit','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.output.exists()
    ev=read(a.evaluation);assert ev['status']=='COMPLETE' and ev['role'] in ['tune','confirm']
    pa=read(a.pack_audit);assert pa['status']=='PASS' and ev['sources']['quality']['sha256']==pa['quality_sha256']
    assert ev['sources']['pack_freeze']['sha256'] in [pa['pack128_freeze_sha256'],pa['pack160_freeze_sha256']]
    for value in ev['sources'].values():assert sha(value['path'])==value['sha256'] and Path(value['path']).stat().st_size==value['bytes']
    assert sha(a.evaluation.parent/'BINDINGS_V1.json')==ev['bindings_sha256']
    model=read(ev['sources']['model']['path']);pred=read(ev['sources']['predictions']['path'])
    with np.load(ev['sources']['pack']['path'],allow_pickle=False) as z:pack={k:z[k].copy() for k in z.files}
    ids=pack['ids'].astype(str).tolist();groups=pack['groups'].astype(str).tolist();N=len(ids)
    assert N==(16 if ev['role']=='tune' else 40) and set(pack['roles'].astype(str))=={ev['role']}
    assert ids==pred['ids']==[r['target_id'] for r in ev['per_identity']]
    assert not set(ids)&set(model['metadata']['fit_ids'])
    quality=read(ev['sources']['quality']['path']);raw={r['target_id']:r for r in quality['frames']}
    states=['background_Q4']+[f'upgrade_{i}' for i in range(6)]
    regions=list(raw[ids[0]]['states']['background_Q4']);assert len(regions)==16
    # Independently execute stored affine heads from X only. Auxiliary outputs
    # are predictions, never their offline GT labels, for all seven real heads.
    x=pack['x'];scores={'direct_raw':ridge(model['models']['direct_raw'],x)};auxpred={}
    for g in [1,4]:
        aux=ridge(model['auxiliary'][f'g{g}'],x);auxpred[g]=aux
        dim=2*g*g;products=aux[:,:dim,None]*aux[:,dim:].reshape(N,6,dim).transpose(0,2,1)
        signed=products.sum(axis=1)
        aug=np.concatenate([x,aux,products.transpose(0,2,1).reshape(N,-1)],axis=1)
        scores[f'signed_g{g}']=signed
        scores[f'direct_aug_g{g}']=ridge(model['models'][f'direct_aug_g{g}'],aug)
        scores[f'direct_residual_g{g}']=signed+ridge(model['models'][f'direct_residual_g{g}'],aug)
    assert set(scores)==set(model['methods'])
    for name,v in scores.items():
        close(v,pred['results']['scores'][name])
        for allow,key in [(False,'fixed_six_choices'),(True,'allow_Q4_choices')]:assert np.array_equal(choose(v,allow),pred['results'][key][name])
    policies={}
    def pair(name,s):
        for allow,label in [(False,'six'),(True,'allow_Q4')]:policies[name+'__'+label]=choose(s,allow)
    for name,s in scores.items():pair(name,s)
    policies['Q4']=np.full(N,-1)
    for allow,label,key in [(False,'six','fit_static_six_action'),(True,'allow_Q4','fit_static_allow_Q4_action')]:
        policies['fit_static__'+label]=np.full(N,model['metadata'][key])
    policies['uniform_six_expectation']=None
    pair('oracle',pack['delta_aee'])
    for g in [1,4]:
        dim=2*g*g;true=pack[f'aux_g{g}'];pr=auxpred[g]
        pair(f'ideal_signed_g{g}',contraction(true,g))
        pair(f'offline_predv_trued_g{g}',contraction(np.concatenate([pr[:,:dim],true[:,dim:]],axis=1),g))
        pair(f'offline_truev_predd_g{g}',contraction(np.concatenate([true[:,:dim],pr[:,dim:]],axis=1),g))
    assert set(policies)==set(ev['policy_definitions']) and len(policies)==32
    ns=np.asarray([[raw[i]['states']['background_Q4'][r]['valid_pixels'] for r in regions] for i in ids])
    errors=np.asarray([[[raw[i]['states'][s][r]['error_sum'] for r in regions] for s in states] for i in ids])
    selected={};weights={};allidx=regions.index('all_valid');Bidx=regions.index('B');Lidx=regions.index('L')
    hardware=read(ev['sources']['hardware']['path']);costs={c['action']:c for c in hardware['candidates']}
    summaries={};counts=dict(policy_identity_regions=0,sequence_region_aggregates=0)
    for name,choices in policies.items():
        w=np.tile(np.r_[0.,np.full(6,1/6)],(N,1)) if choices is None else np.eye(7)[choices+1]
        weights[name]=w;sel=np.einsum('na,nar->nr',w,errors);selected[name]=sel
        definition=ev['policy_definitions'][name]
        if name=='Q4' or name.startswith('fit_static'):assert definition['operator_control']=='static'
        else:assert definition['operator_control']=='dynamic'
        control=definition['operator_control'];ticks=np.asarray([costs[c][control+'_affine_BN_ticks']*4 for c in [6,0,1,2,3,4,5]])
        percost=w@ticks
        for i,tid in enumerate(ids):
            detail=ev['per_identity'][i]['policies'][name]
            assert detail['action']==(None if choices is None else int(choices[i]))
            close(w[i],detail['action_probabilities']);close(percost[i],detail['four_operator_expected_ticks'])
            for j,r in enumerate(regions):
                actual=detail['regions'][r];assert actual['valid_pixels']==ns[i,j]
                close(actual['error_sum'],sel[i,j])
                if ns[i,j]:close(actual['AEE'],sel[i,j]/ns[i,j])
                else:assert actual['AEE'] is None
                counts['policy_identity_regions']+=1
        summary={}
        for scope in ['overall']+ev['sequences']:
            ix=np.ones(N,dtype=bool) if scope=='overall' else np.asarray(groups)==scope
            sums=sel[ix].sum(axis=0);number=ns[ix].sum(axis=0);summary[scope]={}
            for j,r in enumerate(regions):
                saved=ev['quality_summaries'][name][scope][r]
                assert saved['valid_pixels']==number[j];close(saved['error_sum'],sums[j])
                aee=sums[j]/number[j] if number[j] else None
                if aee is None:assert saved['pixel_weighted_AEE'] is None
                else:close(saved['pixel_weighted_AEE'],aee)
                nframe=ns[ix,j]>0
                fm=np.mean(sel[ix,j][nframe]/ns[ix,j][nframe]) if nframe.any() else None
                if fm is None:assert saved['frame_mean_AEE'] is None
                else:close(saved['frame_mean_AEE'],fm)
                summary[scope][r]=aee;counts['sequence_region_aggregates']+=1
            ds=ev['decision_summaries'][name][scope]
            close(ds['upgrade_fraction'],np.mean(1-w[ix,0]))
            close(ds['operator_only_service']['mean_four_conv_ticks'],np.mean(percost[ix]))
            close(ds['operator_only_service']['max_possible_four_conv_ticks'],max(max(ticks[wi>0]) for wi in w[ix]))
            chosen_delta=sel[ix,allidx]/ns[ix,allidx]-errors[ix,0,allidx]/ns[ix,allidx]
            for k,truth in [('six',pack['delta_aee'][ix].min(axis=1)),('seven',np.minimum(pack['delta_aee'][ix].min(axis=1),0))]:
                regret=chosen_delta-truth
                close(ds[f'regret_{k}_action_pixel_weighted'],np.average(regret,weights=ns[ix,allidx]))
                close(ds[f'regret_{k}_action_frame_mean'],regret.mean())
        summaries[name]=summary
    # Compare complete per-action uniform tail probabilities, including union.
    for i,row in enumerate(ev['uniform_six_tail_risk']['frames']):
        assert row['target_id']==ids[i];union=set()
        for r,j in [('all_valid',allidx),('B',Bidx),('L',Lidx)]:
            support=ns[i,j];bad=[]
            if support>=256:
                b=errors[i,0,j]/support
                for act in range(6):
                    c=errors[i,act+1,j]/support
                    if c>b*1.2 or c-b>1:bad.append(act)
            assert row['regions'][r]['triggering_actions']==bad
            close(row['regions'][r]['trigger_probability'],len(bad)/6);union.update(bad)
        assert row['any_primary_region_triggering_actions']==sorted(union)
        close(row['any_primary_region_trigger_probability'],len(union)/6)
    # Compact independently calculated interpretation table with all opponents.
    table={};base=selected['Q4'];base_aee=base[:,allidx].sum()/ns[:,allidx].sum()
    for name in ['Q4']+[m+'__'+reg for m in model['methods']+['fit_static','oracle','ideal_signed_g1','ideal_signed_g4'] for reg in ['six','allow_Q4']]+['uniform_six_expectation']:
        sel=selected[name];w=weights[name];overall=summaries[name]['overall'];seqrows={};leaveout={};tails=[]
        for seqname in ev['sequences']:
            ix=np.asarray(groups)==seqname
            gain=(base[ix,allidx]-sel[ix,allidx]).sum()
            seqrows[seqname]=dict(identities=int(ix.sum()),valid_pixels=int(ns[ix,allidx].sum()),
                delta_AEE=float((sel[ix,allidx]-base[ix,allidx]).sum()/ns[ix,allidx].sum()),error_sum_gain=float(gain))
            other=~ix
            leaveout[seqname]=float((sel[other,allidx]-base[other,allidx]).sum()/ns[other,allidx].sum())
        for i,tid in enumerate(ids):
            for r,j in [('all_valid',allidx),('B',Bidx),('L',Lidx)]:
                if ns[i,j]>=256:
                    c,b=sel[i,j]/ns[i,j],base[i,j]/ns[i,j]
                    if c>b*1.2 or c-b>1:tails.append(dict(target_id=tid,region=r,valid_pixels=int(ns[i,j]),delta_AEE=float(c-b),relative_change=float(c/b-1) if b else None))
        table[name]=dict(AEE=overall['all_valid'],delta_vs_Q4=overall['all_valid']-base_aee,
            relative_vs_Q4=overall['all_valid']/base_aee-1,B_AEE=overall['B'],L_AEE=overall['L'],B_and_L_AEE=overall['B_and_L'],
            upgrade_identities_or_expected=float((1-w[:,0]).sum()),
            four_operator_mean_ticks=ev['decision_summaries'][name]['overall']['operator_only_service']['mean_four_conv_ticks'],
            sequence_changes=seqrows,leave_one_sequence_delta_vs_Q4=leaveout,high_support_tails_vs_Q4=tails,
            mean_policy_tails_only_for_uniform=name=='uniform_six_expectation')
    out=dict(schema='flow_r33_actual_evaluation_independent_audit_v1',status='PASS',role=ev['role'],feature_width=int(pack['x'].shape[1]),
        evaluation_sha256=sha(a.evaluation),source_pack_audit_sha256=sha(a.pack_audit),auditor_sha256=sha(__file__),
        counts=counts,identities=N,sequences=ev['sequences'],heads_replayed_from_X_only=7,policies_checked=32,regions_checked=16,
        independently_recomputed_table=table,uniform_risk=ev['uniform_six_tail_risk'],
        model_fitting=False,model_or_setting_changed=False,GPU_used=False,
        scope='Selected actual raw-output metrics independently reduced from already original-array-audited packs. Operator costs exclude feature/head/full-network. No official-test or superiority assertion.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':'PASS','role':ev['role'],'width':out['feature_width'],'identities':N,'policies':32}))

if __name__=='__main__':main()
