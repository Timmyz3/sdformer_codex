#!/usr/bin/env python3
"""Prespecified full-cohort R8 description; no deployable selector or GPU."""
import hashlib
import json
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
OLD=HERE.parent.with_name(HERE.parent.name.replace('20261015','20261014'))
SOURCES={'R8':HERE/'run_v1/results.json','CC':OLD/'damage_predictor_round6/collect_v1/results.json',
         'FF':OLD/'damage_predictor_round6/matched_fine_check_v1/results.json',
         'native':OLD/'damage_predictor_round6/native_check_v1/results.json',
         'R7':OLD/'repair_scope_round7/run_v1/results.json'}
ARMS=['fine_s2b'+str(i) for i in range(6)]
OLDARMS={'native':('native','native_dense'),'FF':('FF','fine_stage02'),
         'CC':('CC','coarse_stage02'),'CF':('R7','C0_F2')}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    raw={k:json.loads(p.read_text()) for k,p in SOURCES.items()}
    r8=raw['R8'];assert r8['status']=='ROUND8_SINGLE_LAYER_DIAGNOSTIC_COMPLETE'
    assert r8['full_forward_count']==56 and len(r8['frames'])==8
    assert all(r8[k] for k in ('original_model_unchanged','BN_buffers_unchanged','original_modules_restored'))
    assert all(x['exact_archived_prediction'] for x in r8['anchor_replays'].values())
    assert all(x['flow_exact'] and x['all8_branches_exact'] for x in r8['dual_checks'].values())
    assert all(r['original_state_sha256']==r8['original_state_sha256'] for r in raw.values())
    indexed={k:{f['target_id']:f for f in r['frames']} for k,r in raw.items()}
    frames=[]; keys=list(OLDARMS)+ARMS+['GT_best_of_six']
    for f in r8['frames']:
        tid=f['target_id'];metrics={a:f['arms'][a]['metrics']['regions'] for a in ARMS}
        for key,(source,arm) in OLDARMS.items():
            prior=indexed[source][tid]
            assert prior['input_evidence']==f['input_evidence'] and prior['timestamps']==f['timestamps']
            metrics[key]=prior['arms'][arm]['metrics']['regions']
        for a in ARMS:
            ex=f['arms'][a]['execution'];assert ex['summed_fc1_rows']==ex['summed_fc2_rows']==164200
        chosen=min(ARMS,key=lambda a:(metrics[a]['all_valid']['AEE'],a))
        metrics['GT_best_of_six']=metrics[chosen]
        regions={}
        for reg in metrics['CC']:
            n=metrics['CC'][reg]['valid_pixels'];assert all(m[reg]['valid_pixels']==n for m in metrics.values())
            regions[reg]={'valid_pixels':n,'AEE':{k:metrics[k][reg]['AEE'] for k in keys},
                          'error_sum':{k:metrics[k][reg]['error_sum'] for k in keys}}
        frames.append({'target_id':tid,'sequence':f['sequence'],'GT_best_arm':chosen,'regions':regions})
    summaries={}
    for scope in ['overall']+sorted({f['sequence'] for f in frames}):
        selected=frames if scope=='overall' else [f for f in frames if f['sequence']==scope]
        summary={}
        for reg in frames[0]['regions']:
            data=[f['regions'][reg] for f in selected if f['regions'][reg]['valid_pixels']]
            count=sum(d['valid_pixels'] for d in data)
            fm={k:mean(d['AEE'][k] for d in data) if data else None for k in keys}
            summary[reg]={'nonempty_frames':len(data),'valid_pixels':count,'frame_mean_AEE':fm,
                          'pixel_weighted_AEE':{k:sum(d['error_sum'][k] for d in data)/count if count else None for k in keys},
                          'relative_to_native_percent':{k:100*(fm[k]/fm['native']-1) if data else None for k in keys}}
        summaries[scope]=summary
    s=summaries['overall']
    passed={a:all(s[reg]['relative_to_native_percent'][a]<=gate for reg,gate in [('all_valid',3),('B',5),('L',5)]) for a in ARMS}
    best=min(ARMS,key=lambda a:(s['all_valid']['frame_mean_AEE'][a],a))
    candidate=min([a for a in ARMS if passed[a]],key=lambda a:(s['all_valid']['frame_mean_AEE'][a],a)) if any(passed.values()) else None
    improvements={a:{'better_than_CC_frames':sum(f['regions']['all_valid']['AEE'][a]<f['regions']['all_valid']['AEE']['CC'] for f in frames),
                     'sum_CC_minus_arm_AEE':sum(f['regions']['all_valid']['AEE']['CC']-f['regions']['all_valid']['AEE'][a] for f in frames),
                     'per_frame_CC_minus_arm_AEE':{f['target_id']:f['regions']['all_valid']['AEE']['CC']-f['regions']['all_valid']['AEE'][a] for f in frames}} for a in ARMS}
    result={'schema':'flow_round8_layer_analysis_v1','source_sha256':{str(p):sha(p) for p in SOURCES.values()},
            'script_sha256':sha(Path(__file__)),'frames':frames,'summaries':summaries,'quality_gate_pass':passed,
            'best_fixed_development_arm':best,'selected_ordinary_candidate':candidate,'CC_improvements':improvements,
            'GT_reference_caveat':'Per-frame GT selection from these six measured outputs only, no runtime policy, not a universal bound.',
            'scope':'Reused post-result development cohort; fixed layer is ordinary baseline, no independent validation or hardware benefit'}
    path=HERE/'analysis.json';assert not path.exists();path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'summary':{reg:s[reg] for reg in ('all_valid','B','L')},'pass':passed,'candidate':candidate,'improvements':improvements},indent=2))


if __name__=='__main__':main()
