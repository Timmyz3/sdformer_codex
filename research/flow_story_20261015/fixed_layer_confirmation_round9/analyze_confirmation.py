#!/usr/bin/env python3
"""Prespecified paired confirmation of one fixed ordinary layer; no selection."""
import hashlib
import json
from pathlib import Path
from statistics import mean
HERE=Path(__file__).resolve().parent
ARMS=('native','FF','CC','fine_s2b2')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    path=HERE/'run_v1/results.json';raw=json.loads(path.read_text());membership=json.loads((HERE/'membership_v1.json').read_text())
    assert raw['status']=='ROUND9_FIXED_LAYER_CONFIRMATION_COMPLETE' and raw['full_forward_count']==69
    assert all(raw[k] for k in ('original_model_unchanged','BN_buffers_unchanged','original_modules_restored'))
    assert all(v['exact_archived_prediction'] for v in raw['anchor_replays'].values())
    assert all(v['flow_exact'] and v['all8_branches_exact'] for v in raw['dual_checks'].values())
    assert [f['target_id'] for f in raw['frames']]==membership['target_ids']
    frames=[]
    for f in raw['frames']:
        metrics={a:f['arms'][a]['metrics']['regions'] for a in ARMS};regions={}
        for reg in metrics['native']:
            n=metrics['native'][reg]['valid_pixels'];assert all(m[reg]['valid_pixels']==n for m in metrics.values())
            e={a:metrics[a][reg]['AEE'] for a in ARMS}
            regions[reg]={'valid_pixels':n,'AEE':e,'error_sum':{a:metrics[a][reg]['error_sum'] for a in ARMS},
                          'CC_minus_fixed_AEE':e['CC']-e['fine_s2b2'] if n else None,
                          'relative_to_native_percent':{a:100*(e[a]/e['native']-1) if n and e['native'] else None for a in ARMS}}
        frames.append({'target_id':f['target_id'],'sequence':f['sequence'],'regions':regions})
    summaries={}
    for scope in ['overall']+sorted({f['sequence'] for f in frames}):
        selected=frames if scope=='overall' else [f for f in frames if f['sequence']==scope];summary={}
        for reg in frames[0]['regions']:
            data=[f['regions'][reg] for f in selected if f['regions'][reg]['valid_pixels']];n=sum(d['valid_pixels'] for d in data)
            fm={a:mean(d['AEE'][a] for d in data) if data else None for a in ARMS}
            summary[reg]={'nonempty_frames':len(data),'valid_pixels':n,'frame_mean_AEE':fm,
                          'pixel_weighted_AEE':{a:sum(d['error_sum'][a] for d in data)/n if n else None for a in ARMS},
                          'relative_to_native_percent':{a:100*(fm[a]/fm['native']-1) if n and fm['native'] else None for a in ARMS},
                          'CC_minus_fixed_mean_AEE':mean(d['CC_minus_fixed_AEE'] for d in data) if data else None,
                          'fixed_better_than_CC_frames':sum(d['CC_minus_fixed_AEE']>0 for d in data)}
        summaries[scope]=summary
    overall=summaries['overall']; gates={a:all(overall[r]['relative_to_native_percent'][a]<=limit for r,limit in [('all_valid',3),('B',5),('L',5)]) for a in ARMS}
    result={'schema':'flow_round9_fixed_confirmation_analysis_v1','source_sha256':{str(path):sha(path),str(HERE/'membership_v1.json'):sha(HERE/'membership_v1.json')},
            'script_sha256':sha(Path(__file__)),'frames':frames,'summaries':summaries,'quality_gate_pass':gates,
            'fixed_candidate':'fine_s2b2','new_candidate_selected':False,
            'scope':'Fixed ordinary baseline only; new members in historically exposed sequences, no runtime damage gate, significance, official test, or measured hardware benefit'}
    output=HERE/'analysis.json';assert not output.exists();output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'gates':gates,'overall':{r:overall[r] for r in ('all_valid','B','L')},
                     'per_sequence_allGT':{s:v['all_valid'] for s,v in summaries.items() if s!='overall'}},indent=2))


if __name__=='__main__':main()
