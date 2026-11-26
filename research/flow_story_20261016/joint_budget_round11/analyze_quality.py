#!/usr/bin/env python3
"""Frozen R11 descriptive quality analysis; never fits or changes a decision."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
POLICIES=('B5_task','B5_MSE','rich37_task','rich37_MSE','PCA11_task','PCA11_MSE',
          'flow11_task','flow11_MSE','B5_small4_task','B5_small4_MSE',
          'fit_position_mean','fixed_uniform_index3','amplitude_abs_mean',
          'pooling_residual_mse','psn1_activity','psn1_spatial_neighbor_hamming')
BUDGETS=(2,4)
ANCHORS=('native','FF','CC','fine_s2b2')
PRIMARY='flow11_task'
SEQUENCE_CONTROLS=('B5_task','PCA11_task','psn1_activity','psn1_spatial_neighbor_hamming','flow11_MSE')
REGIONS=('all_valid','B','L','B_and_L','B_only','L_only_B_known','neither_B_known','B_unknown',
         'voxel_nonzero','voxel_zero','B_voxel_nonzero','B_voxel_zero','L_voxel_nonzero',
         'L_voxel_zero','B_unknown_voxel_nonzero','B_unknown_voxel_zero')

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def arm(policy,k):return policy+'_k'+str(k)
def metric(frame,name,region):return frame['arms'][name]['metrics']['regions'][region]

def summarize(frames,name):
    regions=list(frames[0]['arms']['native']['metrics']['regions'])
    out={}
    for region in regions:
        values=[metric(f,name,region) for f in frames]
        nonempty=[v for v in values if v['valid_pixels']>0]
        pixels=sum(v['valid_pixels'] for v in values)
        error=sum(v['error_sum'] for v in values)
        out[region]={'frames_with_region':len(nonempty),'valid_pixels':pixels,'error_sum':error,
            'frame_mean_AEE':sum(v['AEE'] for v in nonempty)/len(nonempty) if nonempty else None,
            'pixel_weighted_AEE':error/pixels if pixels else None}
    return out

def mean(summary,scope,region='all_valid'):
    return summary['scopes'][scope][region]['frame_mean_AEE']

def relative(value,reference):
    return value/reference-1 if value is not None and reference is not None and reference!=0 else None

def guard(value,reference,tolerance):
    if reference is None or value is None:
        require(reference is None and value is None,'Mismatched region support')
        return {'pass':True,'evidence':False,'relative_change':None,'candidate':value,'reference':reference}
    return {'pass':value<=reference*(1+tolerance),'evidence':True,
            'relative_change':relative(value,reference),'candidate':value,'reference':reference}

def evaluate_budget(summaries,k,sequences):
    candidate=summaries[arm(PRIMARY,k)]
    comparator_names=[p for p in POLICIES if p!=PRIMARY]
    # Fixed order breaks exact aggregate ties; list all ties transparently.
    best=min(comparator_names,key=lambda p:mean(summaries[arm(p,k)],'overall'))
    best_summary=summaries[arm(best,k)]
    best_value=mean(best_summary,'overall');value=mean(candidate,'overall')
    best_ties=[p for p in comparator_names if mean(summaries[arm(p,k)],'overall')==best_value]
    application={r:guard(mean(candidate,'overall',r),mean(summaries['native'],'overall',r),limit)
                 for r,limit in [('all_valid',.03),('B',.05),('L',.05)]}
    sequence_native={s:{r:guard(mean(candidate,s,r),mean(summaries['native'],s,r),limit)
                       for r,limit in [('all_valid',.03),('B',.05),('L',.05)]} for s in sequences}
    cc_guards={s:{r:guard(mean(candidate,s,r),mean(summaries['CC'],s,r),.01) for r in ('B','L')}
               for s in ['overall']+list(sequences)}
    sequence_comparisons={s:{p:mean(candidate,s)<=mean(summaries[arm(p,k)],s) for p in SEQUENCE_CONTROLS} for s in sequences}
    strong_regions={r:guard(mean(candidate,'overall',r),mean(best_summary,'overall',r),0) for r in ('B','L')}
    strict_all=all(value<mean(summaries[arm(p,k)],'overall') for p in comparator_names)
    beats_cc=value<mean(summaries['CC'],'overall')
    pilot=all(v['pass'] for v in application.values()) and strict_all and beats_cc and all(
        all(v.values()) for v in sequence_comparisons.values()) and all(
        all(g['pass'] for g in v.values()) for v in cc_guards.values())
    improvement=(best_value-value)/best_value if best_value>0 else None
    margin=improvement is not None and improvement>=.01
    best_regions=all(v['pass'] for v in strong_regions.values())
    main=pilot and margin and best_regions
    return {'budget':k,'candidate':arm(PRIMARY,k),'application_overall_native':application,
        'sequence_native_warnings_not_added_to_overall_gate':sequence_native,
        'strictly_beats_all_15_comparators':strict_all,'beats_cheaper_CC':beats_cc,
        'best_comparator_by_overall_allGT':best,'exact_best_ties_in_fixed_order':best_ties,
        'improvement_over_best_fraction':improvement,'B_L_vs_same_best_comparator':strong_regions,
        'sequence_allGT_nonreverse':sequence_comparisons,'B_L_vs_CC_guards':cc_guards,
        'pilot_gate_passed':pilot,'main_quality_increment_gate_passed':main,
        'main_unmet_components':[key for key,ok in [('pilot_conjunction',pilot),('one_percent_margin',margin),('B_L_not_worse_than_best_comparator',best_regions)] if not ok],
        'classification':'main_quality_candidate_only' if main else 'limited_pilot_only' if pilot else 'failed_fixed_gate',
        'limited_pilot_meaning':'Pilot conjunction holds but at least one additional main component fails; not necessarily only a sub-1% margin.',
        'hardware_gate_evaluated':False,'scope':'Same selected producer service at this k; total scoring/queue/bank costs not matched. No independent-sequence or official-test claim.'}

def analyze(frames):
    require(len(frames)==8 and len({f['target_id'] for f in frames})==8,'Exactly eight unique members')
    seqs=sorted({f['sequence'] for f in frames})
    require(len(seqs)==2 and all(sum(f['sequence']==s for f in frames)==4 for s in seqs),'4+4 members required')
    names=list(ANCHORS)+[arm(p,k) for k in BUDGETS for p in POLICIES]
    regions=list(frames[0]['arms']['native']['metrics']['regions'])
    require(set(regions)==set(REGIONS),'All sixteen predetermined regions required')
    for f in frames:
        require(set(f['arms'])==set(names),'All 36 fixed arms required')
        for n in names:
            metrics=f['arms'][n]['metrics']['regions']
            require(set(metrics)==set(regions),'Region coverage differs')
            for region,v in metrics.items():
                require(v['valid_pixels']==metric(f,'native',region)['valid_pixels'],'Paired region support differs')
                require(v['valid_pixels']>=0 and math.isfinite(v['error_sum']) and v['error_sum']>=0,'Invalid count/sum')
                if v['valid_pixels']:
                    require(v['AEE'] is not None and math.isfinite(v['AEE']) and math.isclose(v['error_sum']/v['valid_pixels'],v['AEE'],rel_tol=1e-10,abs_tol=1e-10),'AEE reduction differs')
                else:
                    require(v['error_sum']==0 and v['AEE'] is None,'Empty region must have zero sum and undefined AEE')
    scopes={'overall':frames};scopes.update({s:[f for f in frames if f['sequence']==s] for s in seqs})
    summaries={n:{'scopes':{s:summarize(members,n) for s,members in scopes.items()},
        'allGT_wins_ties_losses_vs_CC':{
            'wins':sum(metric(f,n,'all_valid')['AEE']<metric(f,'CC','all_valid')['AEE'] for f in frames),
            'ties':sum(metric(f,n,'all_valid')['AEE']==metric(f,'CC','all_valid')['AEE'] for f in frames),
            'losses':sum(metric(f,n,'all_valid')['AEE']>metric(f,'CC','all_valid')['AEE'] for f in frames)}} for n in names}
    contributions={}
    for k in BUDGETS:
        rows=[{'target_id':f['target_id'],'CC_minus_primary_AEE':metric(f,'CC','all_valid')['AEE']-metric(f,arm(PRIMARY,k),'all_valid')['AEE'],
               'primary_minus_native_AEE':metric(f,arm(PRIMARY,k),'all_valid')['AEE']-metric(f,'native','all_valid')['AEE']} for f in frames]
        total=sum(r['CC_minus_primary_AEE'] for r in rows)
        for r in rows:r['signed_gain_share_percent']=100*r['CC_minus_primary_AEE']/total if total else None
        contributions[str(k)]={'per_frame':rows,'signed_total_gain':total}
    transitions={p:{'k4_better_frames':sum(metric(f,arm(p,4),'all_valid')['AEE']<metric(f,arm(p,2),'all_valid')['AEE'] for f in frames),
                    'k4_minus_k2_frame_mean_AEE':mean(summaries[arm(p,4)],'overall')-mean(summaries[arm(p,2)],'overall')} for p in POLICIES}
    return {'summaries':summaries,'per_budget_gates':{str(k):evaluate_budget(summaries,k,seqs) for k in BUDGETS},
            'allGT_concentration':contributions,'actual_budget_change':transitions,'regions':regions,
            'scope':'All36 prespecified endpoints/rules and all nonempty regions retained. k2/k4 analyzed separately, no best-after-check budget selection. No joint optimality, additive gain, hardware speedup or significance claim.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--results-sha256',required=True)
    p.add_argument('--manifest',type=Path,default=HERE/'membership_v1.json')
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    require(sha(a.results)==a.results_sha256,'Raw SHA differs');require(not a.output_dir.exists(),'No output overwrite')
    raw=json.loads(a.results.read_text());manifest=json.loads(a.manifest.read_text())
    require(raw['status']=='ROUND11_FROZEN_JOINT_BUDGET_COMPLETE','Complete R11 result required')
    require(raw['completed_targets']==8 and 53<=raw['full_forward_count']<=293,'Forward/member bounds differ')
    require(all(raw[k] for k in ('original_model_unchanged','BN_buffers_unchanged','original_modules_restored')),'State closure differs')
    require([f['target_id'] for f in raw['frames']]==manifest['groups']['check'],'Frozen identities/order differ')
    result=analyze(raw['frames']);result.update(schema='flow_round11_quality_analysis_v1',input_sha256=a.results_sha256,
        script_sha256=sha(__file__),manifest_sha256=sha(a.manifest),full_forward_count=raw['full_forward_count'])
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# R11 固定预测器的真实联合 k2/k4 比较','',result['scope'],'',
           '| 规则 | allGT AEE | B | L | allGT 对native | 胜/平/负 CC |','|---|---:|---:|---:|---:|---|']
    native=mean(result['summaries']['native'],'overall')
    for name,s in result['summaries'].items():
        vals=[mean(s,'overall',r) for r in ('all_valid','B','L')];ratio=relative(vals[0],native);w=s['allGT_wins_ties_losses_vs_CC']
        lines.append('| '+name+' | '+' | '.join('%.9f'%v if v is not None else 'empty' for v in vals)+' | '+('%+.6f%%'%(100*ratio) if ratio is not None else 'undefined')+' | %d/%d/%d |'%(w['wins'],w['ties'],w['losses']))
    for k,g in result['per_budget_gates'].items():
        lines+=['','预算 k='+k+'：'+g['classification']+'；最强固定对照 '+g['best_comparator_by_overall_allGT']+'。完整固定门、分序列native警戒与全部区域见 analysis.json。']
    (a.output_dir/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'ROUND11_ANALYSIS_COMPLETE','per_budget':{k:g['classification'] for k,g in result['per_budget_gates'].items()}}))

if __name__=='__main__':main()
