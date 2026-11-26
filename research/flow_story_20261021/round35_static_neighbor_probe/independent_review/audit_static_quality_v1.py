"""Independent R35 original-array static360 quality and matched-cache comparisons.

No neural forward or fitting. Older 40 policies use the R34 independently
recomputed cache, checked again against its independent aggregate evidence.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os
import numpy as np

HERE = Path(__file__).resolve().parent
B = HERE.parent
ROOT = HERE.parents[3]
R34 = B.parent / 'round34_deployability'
R33 = B.parent / 'round33_predictability'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b): np.testing.assert_allclose(a,b,rtol=3e-13,atol=3e-12)
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def module(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def measure(errors,counts,names,ids,groups,baseline):
    """All16 regions overall/sequence/leave-one; fixed old primary tail rule."""
    seqs=list(dict.fromkeys(groups.tolist())); summaries={};leave={};tails=[]
    def sums(ix):
        out={}
        for j,rn in enumerate(names):
            n=int(counts[ix,j].sum());s=float(errors[ix,j].sum());bs=float(baseline[ix,j].sum())
            out[rn]=dict(valid_pixels=n,error_sum=s,AEE=s/n if n else None,
                         reference_AEE=bs/n if n else None,delta=(s-bs)/n if n else None,
                         relative=s/bs-1 if n and bs else None)
        return out
    summaries['overall']=sums(np.ones(len(ids),dtype=bool))
    for seq in seqs:summaries[seq]=sums(groups==seq);leave[seq]=sums(groups!=seq)
    for i,tid in enumerate(ids):
        for rn in ['all_valid','B','L']:
            j=names.index(rn);n=int(counts[i,j]);v=errors[i,j];bv=baseline[i,j]
            if n>=256 and (v>1.2*bv or v-bv>n):
                tails.append(dict(target_id=tid,sequence=str(groups[i]),region=rn,valid_pixels=n,AEE=float(v/n),reference_AEE=float(bv/n)))
    return dict(summaries=summaries,leave_one_sequence=leave,high_support_tail_review=tails)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=B/'actual/static360_v1')
    ap.add_argument('--output',type=Path,default=HERE/'STATIC360_RAW_QUALITY_INDEPENDENT_V1.json');a=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.output.exists()
    result=read(a.run/'results.json');assert result['status']=='R35_STATIC360_COMPLETE'
    assert result['model_forwards']==result['forward_attempts']==124 and result['formal_model_forwards']==120 and result['engineering_model_forwards']==4
    assert result['final_state_sha256']=='7be9fed7c55332a150b4c9f67e863d5e68db2bb0d1829a5344703427a48ff018'
    assert result['completed_optimizer_steps']==0 and not result['GT_arrays_read'] and not result['quality_computed']
    contract=read(B/'ACTION_CONTRACT_V1.json');assert sha(B/'ACTION_CONTRACT_V1.json')==result['action_contract_sha256']
    memberspath=R33/'data/MEMBERSHIP_V1.json';members=read(memberspath)['records']
    assert sha(memberspath)==result['membership_sha256']==contract['membership_sha256']
    assert result['selected_ids']==contract['target_ids']==[r['target_id'] for r in members]
    frames=result['evaluations']['static360']['frames'];assert [r['target_id'] for r in frames]==result['selected_ids']
    refs=read(B/'Q4_PARITY_REFERENCES_V1.json')
    for role in ['fit','tune','confirm']:
        rr=result['parity']['Q4_'+role];assert rr['target_id']==refs['roles'][role]['target_id']
        assert rr['actual_input']==refs['roles'][role]['actual_input']
        for h in range(4):assert rr['heads'][f'head{h}']['sha256']==refs['roles'][role]['heads'][f'head{h}']['sha256']
    assert len(result['parity']['static360_dense']['comparison']['checks'])==137
    assert all(v['different_bytes']==0 and v['actual_sha256']==v['reference_sha256'] for v in result['parity']['static360_dense']['comparison']['checks'].values())
    priorpath=R34/'independent_review/MATCHED_BIAS_INDEPENDENT_AUDIT_V1.json';prior=read(priorpath);assert prior['status']=='PASS'
    prior_summary=R34/'bias_control/extension_v1/actual_v1/SUMMARY_V1.json';assert sha(prior_summary)==prior['summary_sha256']
    biaspath=R34/'bias_control/actual_v1/BIAS_MODEL_V1.json';assert sha(biaspath)==prior['bias_model_sha256'];bias=np.array(read(biaspath)['bias'])
    regionpath=ROOT/'research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json'
    regions=read(regionpath);helperpath=R33/'independent_review/audit_pack_actual_v1.py'
    helper=module(helperpath,'independent_old_regions');assert sha(helperpath)==prior['mask_helper_sha256']
    phase_results={};oldbg={}
    for phase in ['fit_tune','confirm']:
        p=R33/f'actual/{phase}_v1/results.json';phase_results[phase]=sha(p)
        oldbg[phase]={r['target_id']:r for r in read(p)['evaluations']['background_Q4']['frames']}
    lookup={r['target_id']:r for r in frames};allrows=[];pred_hashes={};source_hashes={};corrected_hashes={}
    for r in members:
        tid=r['target_id'];fr=lookup[tid];phase='confirm' if r['role']=='confirm' else 'fit_tune';source={}
        assert fr['role']==r['role'] and fr['sequence']==r['sequence'] and fr['actual_input']==oldbg[phase][tid]['actual_input']
        for f in r['source_files']:
            p=R33/f'data/{phase}_source_snapshot'/f['relative_path'];assert sha(p)==f['sha256'] and p.stat().st_size==int(f['bytes'])
            source[f['category']]=np.load(p,allow_pickle=False);source_hashes[str(p.relative_to(ROOT))]=sha(p)
        p=a.run/fr['prediction_file'];assert sha(p)==fr['prediction_file_sha256'];pred=load(p)['head3']
        assert pred.shape==(2,480,640) and pred.dtype==np.float32 and np.isfinite(pred).all()
        assert hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest()==fr['prediction_sha256']==fr['heads']['head3']['sha256']
        pred_hashes[tid]=sha(p)
        masks=helper.masks_from_sources(source['gt'],source['mask'],source['input'],regions['boundary_threshold_pixels'],regions['large_displacement_threshold_pixels'])
        names=list(masks);assert len(names)==16;counts=[int(m.sum()) for m in masks.values()];vals={}
        for cond,b in [('uncorrected',np.zeros(2)),('same_bias_corrected',bias)]:
            corrected=pred.astype(np.float64)-b[:,None,None]
            err=np.sqrt(np.sum((corrected-source['gt'].astype(np.float64))**2,axis=0))
            vals[cond]=[float(err[m].sum()) for m in masks.values()]
            if cond=='same_bias_corrected':corrected_hashes[tid]=hashlib.sha256(np.ascontiguousarray(corrected).tobytes()).hexdigest()
        allrows.append(dict(target_id=tid,role=r['role'],sequence=r['sequence'],counts=counts,error_sums=vals))
    tables={};bindings={};comparisons=0
    for role in ['fit','tune','confirm']:
        rows=[r for r in allrows if r['role']==role];ids=[r['target_id'] for r in rows];groups=np.array([r['sequence'] for r in rows]);counts=np.array([r['counts'] for r in rows]);tables[role]={}
        for cond in ['uncorrected','same_bias_corrected']:
            ep=R34/f'bias_control/extension_v1/actual_v1/{role}_{cond}_EVALUATION_V1.json';ev=read(ep);bindings[str(ep.relative_to(ROOT))]=sha(ep)
            assert [r['target_id'] for r in ev['frames']]==ids
            old={}
            for method in ev['summaries']:
                er=np.array([[fr['policies'][method]['regions'][rn]['error_sum'] for rn in names] for fr in ev['frames']]);old[method]=er
                for i,fr in enumerate(ev['frames']):
                    assert [fr['policies'][method]['regions'][rn]['valid_pixels'] for rn in names]==counts[i].tolist()
                pv=prior['independent_policy_quality'][role][cond][method]
                for j,rn in enumerate(names):
                    n=counts[:,j].sum()
                    if n:close(er[:,j].sum()/n,pv['regions'][rn]['AEE'])
            assert len(old)==40
            current=np.array([r['error_sums'][cond] for r in rows]);base=old['Q4']
            static=measure(current,counts,names,ids,groups,base);versus={}
            for method,er in old.items():
                versus[method]=measure(current,counts,names,ids,groups,er);comparisons+=1
            tables[role][cond]=dict(static360=static,static360_vs_each_frozen_policy=versus,
                unchanged_policy_scope='All40 R34 policies;37 observable choices fixed. Offline oracle and expectation retained as diagnostics. Static360 no predictor. Fixed-six cost comparison is primary; allow-Q4 budget differs.')
    output=dict(schema='flow_r35_independent_static360_raw_quality_v1',status='PASS',auditor_sha256=sha(__file__),
        raw_result_sha256=sha(a.run/'results.json'),contract_sha256=sha(B/'ACTION_CONTRACT_V1.json'),membership_sha256=sha(memberspath),
        original_results_sha256=phase_results,prior_independent_audit_sha256=sha(priorpath),prior_complete_summary_sha256=sha(prior_summary),
        bias_model_sha256=sha(biaspath),mask_helper_sha256=sha(helperpath),region_contract_sha256=sha(regionpath),
        counts=dict(new_static_predictions=120,raw_region_condition_values=120*16*2,older_policy_comparisons=comparisons),
        input_source_sha256=source_hashes,static_prediction_sha256=pred_hashes,corrected_payload_sha256=corrected_hashes,
        reused_policy_evaluation_sha256=bindings,region_names=names,raw_static_frames=allrows,independent_quality=tables,
        GPU_calls=0,new_fits=0,scope='Original120 exposed identities. Old40-policy cache independently audited in R34; aggregates rebound here. All new static360 arrays recomputed. Same frozen output bias for both sides. No full-network or measured RTL latency claim.')
    write(a.output,output)
    print(json.dumps(dict(status='PASS',output=str(a.output),sha256=sha(a.output),key={role:{cond:table['static360']['summaries']['overall']['all_valid'] for cond,table in rt.items()} for role,rt in tables.items()})))

if __name__=='__main__':main()
