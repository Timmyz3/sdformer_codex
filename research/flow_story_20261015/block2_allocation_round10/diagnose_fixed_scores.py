#!/usr/bin/env python3
"""Descriptive fixed-score audit after decisions; no new model or policy."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
FAMILIES=('B5','rich37','PCA11','flow11')
BASE='coarse_stage02'
TILES=tuple((r,c) for r in range(3) for c in range(4))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def ranks(x):
    x=np.asarray(x);out=np.empty(len(x),float)
    for value in np.unique(x):
        indices=np.flatnonzero(x==value);out[indices]=np.count_nonzero(x<value)+(len(indices)-1)/2
    return out


def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])


def main():
    rp=HERE/'collect_v1/results.json';pp=HERE/'analysis_v1/predictions_before_check_evaluation.json'
    ap=HERE/'analysis_v1/analysis.json'
    raw=json.loads(rp.read_text());p=json.loads(pp.read_text());a=json.loads(ap.read_text())
    assert raw['status']=='ROUND10_BLOCK2_SINGLETON_COLLECTION_COMPLETE'
    assert p['input_sha256']==sha(rp) and a['prediction_artifact_sha256']==sha(pp)
    frames=[]
    for frame in raw['frames']:
        if frame['group']!='check':continue
        tid=frame['target_id'];base=frame['arms'][BASE]['metrics']['regions']['all_valid']['AEE']
        names=['fine_r%d_c%d'%t for t in TILES]
        gain=np.asarray([base-frame['arms'][n]['metrics']['regions']['all_valid']['AEE'] for n in names])
        mses={tuple(x['tile']):x['MSE'] for x in frame['offline_fine_coarse_branch_mse']}
        mse=np.asarray([mses[t] for t in TILES]);scores={}
        for family in FAMILIES:
            for target in ('task','MSE'):
                key=family+'_'+target;pred=np.asarray(p['scores'][key][tid],float)
                choice=p['selected_arms'][key][tid];index=names.index(choice)
                scores[key]={'rank_correlation_with_task_gain':correlation(ranks(pred),ranks(gain)),
                             'rank_correlation_with_branch_MSE':correlation(ranks(pred),ranks(mse)),
                             'selected_arm':choice,'actual_allGT_gain':float(gain[index]),
                             'loss_to_measured_GT_best':float(np.max(gain)-gain[index])}
        chosen=p['selected_arms']['flow11_task'][tid]
        frames.append({'target_id':tid,'sequence':frame['sequence'],'positive_gain_candidates':int((gain>0).sum()),
                       'all12_actual_gains':gain.tolist(),'best_measured_gain':float(gain.max()),
                       'offline_branchMSE_vs_task_rank_correlation':correlation(ranks(mse),ranks(gain)),
                       'fixed_scores':scores,'flow11_task_same_choice_as':{key:choices[tid]==chosen for key,choices in p['selected_arms'].items() if key!='flow11_task'}})
    assert len(frames)==8
    summary={}
    for family in FAMILIES:
        for target in ('task','MSE'):
            key=family+'_'+target;values=[f['fixed_scores'][key] for f in frames]
            rho=[v['rank_correlation_with_task_gain'] for v in values if v['rank_correlation_with_task_gain'] is not None]
            summary[key]={'mean_within_frame_rank_correlation_with_task_gain':float(np.mean(rho)) if rho else None,
                          'nonconstant_rank_frames':len(rho),'positive_selected_gain_frames':sum(v['actual_allGT_gain']>0 for v in values),
                          'mean_actual_selected_gain':float(np.mean([v['actual_allGT_gain'] for v in values])),
                          'mean_loss_to_measured_GT_best':float(np.mean([v['loss_to_measured_GT_best'] for v in values]))}
    result={'schema':'flow_round10_fixed_score_description_v1','source_sha256':{str(path):sha(path) for path in (rp,pp,ap,Path(__file__))},
            'frames':frames,'fixed_score_summaries':summary,
            'positive_gain_candidates_of96':sum(f['positive_gain_candidates'] for f in frames),
            'frames_with_any_positive_gain':sum(f['best_measured_gain']>0 for f in frames),
            'scope':'Descriptive within-frame ranking of already fixed scores. No fitting, tuning, significance, new selector, additive joint-gain, native-quality or hardware claim.'}
    output=HERE/'analysis_v1/fixed_score_diagnostic.json';assert not output.exists()
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'positive_gain_candidates_of96':result['positive_gain_candidates_of96'],'fixed_score_summaries':summary},indent=2))


if __name__=='__main__':main()
