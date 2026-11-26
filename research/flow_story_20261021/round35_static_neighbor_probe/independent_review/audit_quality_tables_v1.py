"""Bind independent static raw math to all R35 evaluator tables and scope."""
from pathlib import Path
import json,os
import numpy as np
import audit_static_quality_v1 as raw

B=raw.B;R34=raw.R34;HERE=raw.HERE

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out=HERE/'QUALITY_TABLES_INDEPENDENT_AUDIT_V1.json';assert not out.exists()
    rp=HERE/'STATIC360_RAW_QUALITY_INDEPENDENT_V1.json';d=raw.read(rp);assert d['status']=='PASS'
    qdir=B/'quality/actual_v1';quality=raw.read(qdir/'QUALITY_V1.json');summary=raw.read(qdir/'SUMMARY_V1.json');assert summary['status']=='COMPLETE'
    assert quality['bias_model_sha256']==d['bias_model_sha256']
    assert [r['target_id'] for r in quality['frames']]==[r['target_id'] for r in d['raw_static_frames']]
    names=d['region_names'];rawchecks=0;framechecks=0;aggregatechecks=0;tables={};binding={}
    for r,q in zip(d['raw_static_frames'],quality['frames']):
        assert r['role']==q['role'] and r['sequence']==q['sequence']
        assert q['corrected_payload_sha256']==d['corrected_payload_sha256'][r['target_id']]
        for old,new in [('uncorrected','original'),('same_bias_corrected','same_bias_corrected')]:
            for j,rn in enumerate(names):
                x=q[new][rn];n=r['counts'][j];s=r['error_sums'][old][j];assert x['valid_pixels']==n;raw.close(x['error_sum'],s)
                if n:raw.close(x['AEE'],s/n)
                else:assert x['AEE'] is None
                rawchecks+=1
    for role in ['fit','tune','confirm']:
        rows=[r for r in d['raw_static_frames'] if r['role']==role];ids=[r['target_id'] for r in rows];groups=np.array([r['sequence'] for r in rows]);counts=np.array([r['counts'] for r in rows]);seqs=list(dict.fromkeys(groups.tolist()));tables[role]={}
        for oldcond,newcond in [('uncorrected','original'),('same_bias_corrected','same_bias_corrected')]:
            p=qdir/f'{role}_{newcond}_EVALUATION_V1.json';e=raw.read(p);binding[str(p.relative_to(raw.ROOT))]=raw.sha(p)
            old=raw.read(R34/f'bias_control/extension_v1/actual_v1/{role}_{oldcond}_EVALUATION_V1.json')
            assert e['original_policy_choices']==old['unchanged_observable_policy_choices']
            assert e['uniform_six_actual_tail_risk']==old['uniform_six_actual_tail_risk']
            assert [f['target_id'] for f in e['frames']]==ids
            errors={k:np.array([[f['policies'][k]['regions'][rn]['error_sum'] for rn in names] for f in old['frames']]) for k in old['summaries']}
            errors['static360']=np.array([r['error_sums'][oldcond] for r in rows]);assert len(errors)==41
            means={}
            for method,es in errors.items():
                for i,f in enumerate(e['frames']):
                    for j,rn in enumerate(names):
                        z=f['policies'][method]['regions'][rn];assert z['valid_pixels']==counts[i,j];raw.close(z['error_sum'],es[i,j]);framechecks+=1
                for scope in ['overall']+seqs:
                    ix=np.ones(len(rows),bool) if scope=='overall' else groups==scope
                    for j,rn in enumerate(names):
                        n=int(counts[ix,j].sum());total=float(es[ix,j].sum());z=e['summaries'][method][scope][rn];assert z['valid_pixels']==n;raw.close(z['error_sum'],total)
                        positive=ix&(counts[:,j]>0);fm=float(np.mean(es[positive,j]/counts[positive,j])) if positive.any() else None
                        if n:raw.close(z['pixel_weighted_AEE'],total/n);raw.close(z['frame_mean_AEE'],fm)
                        else:assert z['pixel_weighted_AEE'] is None and z['frame_mean_AEE'] is None
                        aggregatechecks+=1
                z=e['summaries'][method]['overall']['all_valid'];means[method]=dict(AEE=z['pixel_weighted_AEE'],frame_mean_AEE=z['frame_mean_AEE'])
                raw.close(summary['results'][role][newcond][method]['AEE'],z['pixel_weighted_AEE']);raw.close(summary['results'][role][newcond][method]['frame_mean_AEE'],z['frame_mean_AEE'])
            checks={}
            for method in ['X128__folded_constant_v__six','X128__calibrated_ordinary__six','X160__signed_g1__six','X128__direct_aug_g1__six','X160__direct_aug_g1__six']:
                m=raw.measure(errors[method],counts,names,ids,groups,errors['static360'])
                if method+'__vs_static360' in e['comparisons']:
                    stored=e['comparisons'][method+'__vs_static360']
                    assert [(r['target_id'],r['region']) for r in m['high_support_tail_review']]==[(r['target_id'],r['region']) for r in stored['high_support_tail_review']]
                m['frame_mean_AEE']=means[method]['frame_mean_AEE'];m['static_frame_mean_AEE']=means['static360']['frame_mean_AEE']
                checks[method]=m
            ordinary=[k for k in errors if k.endswith('__six') and ('__direct_' in k or '__calibrated_ordinary__' in k)]
            best=min(ordinary,key=lambda k:means[k]['AEE'])
            tables[role][newcond]=dict(policy_overall=means,selected_vs_static360=checks,best_observed_ordinary_fixed_six=best,
                ordinary_choice_scope='Descriptive best of frozen ordinary controls; not a new deployable selector or universal generic bound')
    hwp=HERE/'hardware_actual_work_audit_v1.json';hw=raw.read(hwp);budgetp=HERE/'hardware_per_identity_budget_v1.json';budget=raw.read(budgetp)
    assert hw['status'].startswith('PASS') and budget['status']=='PASS_ALL120_FIXED_SIX_IDENTITIES'
    assert len(budget['rows'])==120 and all(r['static360_four_isolated_ticks']<r['dynamic_scoped_budget'] for r in budget['rows'])
    final=dict(status='PASS',schema='flow_r35_quality_tables_independent_v1',auditor_sha256=raw.sha(__file__),
        raw_audit_sha256=raw.sha(rp),raw_quality_sha256=raw.sha(qdir/'QUALITY_V1.json'),summary_sha256=raw.sha(qdir/'SUMMARY_V1.json'),evaluator_files=binding,
        actual_hardware_audit_sha256=raw.sha(hwp),per_identity_budget_sha256=raw.sha(budgetp),
        counts=dict(new_raw_region_values=rawchecks,all_policy_frame_region_values=framechecks,all_policy_aggregate_region_values=aggregatechecks),
        independent_scope_tables=tables,new_GPU_calls=0,new_fits=0,
        cost_scope='Original outputs match measured-model uncorrected service. Same-b correction is a common unmeasured additive operation; no absolute corrected-system cycles. One static360 mask, all120 previously exposed; fixed-six primary; not full-network timing or all static layouts.')
    raw.write(out,final);print(json.dumps(dict(status='PASS',output=str(out),sha256=raw.sha(out),counts=final['counts'])))

if __name__=='__main__':main()
