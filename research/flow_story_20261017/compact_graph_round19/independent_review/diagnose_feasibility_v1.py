#!/usr/bin/env python3
"""Full fixed-k1 feasibility population, never a GT-selected deployment policy."""
import argparse
import json
import math
from pathlib import Path
import reduce_raw_v1 as raw

SINGLETONS=tuple(f'fine_{r}_{c}' for r in range(3) for c in range(4))

def gate(candidate,native):
    rows={}
    for region,limit in (('all_valid',.03),('B',.05),('L',.05)):
        c,n=candidate[region],native[region]
        raw.require(c['valid_pixels']==n['valid_pixels'],'Same actual support')
        cn,nn=c['AEE'],n['AEE'];positive=n['valid_pixels']>0
        raw.require((cn is None)==(nn is None)==(not positive),'Actual empty support preserved')
        if positive:raw.require(all(math.isfinite(v) and v>=0 for v in (cn,nn)),'Finite nonnegative errors')
        passed=positive and cn<=nn*(1+limit)
        rows[region]={'valid_pixels':n['valid_pixels'],'candidate_AEE':cn,'native_AEE':nn,
          'relative_change':cn/nn-1 if positive and nn!=0 else None,'limit':limit,'positive_support':positive,'pass':passed}
    return {'regions':rows,'all_three_pass':all(v['pass'] for v in rows.values())}

def analyze(frames):
    raw.require(len(frames)==len({f['target_id'] for f in frames})==8,'All eight fixed frames')
    out=[]
    for f in frames:
        native=f['arms']['native'];actions={a:gate(f['arms'][a],native) for a in SINGLETONS}
        passing=[a for a in SINGLETONS if actions[a]['all_three_pass']]
        out.append({'target_id':f['target_id'],'sequence':f['sequence'],'CC':gate(f['arms']['CC'],native),
          'singleton_actions':actions,'feasible_singleton_count':len(passing),'all_feasible_singleton_ids':passing})
    return {'frames':out,'singleton_actions_checked':96,'CC_actions_checked':8,
      'zero_feasible_singleton_frames':[f['target_id'] for f in out if f['feasible_singleton_count']==0],
      'scope':'GT-dependent fixed-set feasibility diagnostic, not deployable selection or evidence about other budgets/actions; all H8 exposed development.'}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('freeze','quality'):p.add_argument('--'+n,type=Path,required=True);p.add_argument('--'+n+'-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw.require(raw.sha(a.freeze)==a.freeze_sha256 and raw.sha(a.quality)==a.quality_sha256 and not a.output.exists(),'Fixed source/new output')
    freeze=raw.read(a.freeze)
    for name,digest in freeze['source_sha256'].items():raw.require(raw.sha(raw.HERE/name)==digest,'Frozen supplementary source')
    raw.preflight(raw.HERE/'IO_FREEZE_V1.json',freeze['main_IO_freeze_sha256'])
    q=raw.read(a.quality)
    raw.require(q['schema']=='flow_r19_independent_compact_quality_v1' and q['status']=='COMPLETE_NEW_GRAPH_RAW_REDUCTION'
      and q['IO_freeze_sha256']==freeze['main_IO_freeze_sha256'] and q['graph_id']==raw.GRAPH,'Own complete quality only')
    out={'schema':'flow_r19_fixed_k1_feasibility_v1','status':'COMPLETE','graph_id':raw.GRAPH,
      'quality_sha256':a.quality_sha256,'pre_result_freeze_sha256':a.freeze_sha256,**analyze(q['frames'])}
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:out[k] for k in ('status','singleton_actions_checked','zero_feasible_singleton_frames')}))
