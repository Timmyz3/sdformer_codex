#!/usr/bin/env python3
"""Compare unchanged R18 implementation scores with independent R19 raw reduction."""
import argparse
import json
from pathlib import Path
import numpy as np
import reduce_raw_v1 as raw

require=raw.require

def compare_arrays(own,reported,g0):
    require(len(own)==len(reported)==12,'All score tiles')
    maximum=0.;values=0
    for x,y in zip(own,reported):
        require(set(y)=={'tile',*g0.SCORES} and x['tile']==y['tile'],'Complete named scores')
        for name in g0.SCORES:
            require(np.isfinite(y[name]) and y[name]>=0 and np.isclose(x[name],y[name],atol=1e-10,rtol=1e-12),'Independent score agreement')
            maximum=max(maximum,abs(x[name]-y[name]));values+=1
    require(all(g0.select(own,n)==g0.select(reported,n) for n in g0.SCORES),'Exact choice even within tolerance')
    return maximum,values

def compare(a):
    freeze=raw.preflight(a.freeze,a.freeze_sha256);ext=freeze['external_sources']
    require(raw.sha(a.quality)==a.quality_sha256 and raw.sha(a.scores)==a.scores_sha256,'Actual supplied artifacts')
    q=raw.read(a.quality);s=raw.read(a.scores)
    require(q['schema']=='flow_r19_independent_compact_quality_v1' and q['status']=='COMPLETE_NEW_GRAPH_RAW_REDUCTION'
      and q['graph_id']==raw.GRAPH and q['IO_freeze_sha256']==a.freeze_sha256,'Own complete raw reduction')
    require(s['schema']=='flow_r19_g0_cached_ordinary_v1' and s['status']=='G0_CPU_COMPLETE' and s['graph_id']==raw.GRAPH,'Complete actual implementation scores')
    require(s['R19_result_sha256']==q['source_result_sha256'] and s['R19_execution_audit_sha256']==q['execution_audit_sha256'],'Exact same new graph execution')
    rp=raw.REPO/freeze['actual_result_relative_path']
    require(raw.sha(rp)==q['source_result_sha256'],'Original new graph metadata bytes')
    natives={r['target_id']:r for r in raw.read(rp)['forward_records'] if r['arm']=='native'}
    require(len(natives)==8,'Eight actual source captures')
    require(s['root_source_freeze_sha256']==ext['runner_source_freeze']['sha256'] and s['math_sha256']==ext['implementation_G0_math']['sha256'],'Fixed graph and unchanged G0 math')
    require(s['freeze_sha256']==ext['implementation_score_freeze']['sha256'] and s['independent_source_review_sha256']==ext['implementation_score_review']['sha256'],'Reviewed score sources')
    sf=raw.read(raw.REPO/ext['implementation_score_freeze']['relative_path'])
    require(s['source_sha256']==sf['source_sha256'],'Actual thin IO source hashes')
    require(s['h8_sha256']==raw.H8_SHA and len(s['frames'])==8 and s['score_values']==672 and s['score_selections']==56
      and s['policy_frame_cells']==160 and s['policy_count']==20 and s['selector_score_comparisons']==616,'Complete policy population')
    require(all(s[k]==0 for k in ('model_forwards','PSN_calls','FC_calls','GPU_calls','optimizer_steps'))
      and s['GT_loaded'] is False and s['quality_loaded'] is False,'Scores independent of labels')
    _,_,g0=raw.frozen_math()
    require(s['score_names']==list(g0.SCORES) and s['tiles']==[list(t) for t in g0.TILES],'Fixed names/order')
    require([f['target_id'] for f in s['frames']]==[f['target_id'] for f in q['frames']],'Same fixed H8 order')
    checks=[];count=0;policies=0
    for frame in s['frames']:
        tid=frame['target_id'];own=q['ordinary_scores'][tid]
        original=natives[tid]
        require(frame['input_archive']==original['prediction_file'] and
          frame['input_archive_sha256']==original['prediction_file_sha256'] and
          frame['input_tensor_records']=={k:original['tensor_records'][k] for k in ('u','sn1')},'Actual current native cache provenance')
        diff,n=compare_arrays(own,frame['scores'],g0);count+=n
        w=g0.policy_weights(own)
        expectedkeys=set(w)-{'uniform_expectation'}|{'uniform_random_expected'}
        require(set(frame['policies'])==expectedkeys,'All twenty policies')
        for name,weights in w.items():
            p=frame['policies']['uniform_random_expected' if name=='uniform_expectation' else name]
            require(p['weights']==weights,'Same actual endpoint weights')
            if name=='uniform_expectation':
                require(p['selected_tile'] is None and p['mask'] is None and p['kind']=='expected_uniform_randomized_k1'
                  and p['rational_weight']=={'numerator':1,'denominator':12}
                  and p['endpoint_arms']==[f'fine_{r}_{c}' for r,c in g0.TILES],'Random expectation, not ensemble')
            else:
                r,c=g0.TILES[weights.index(1)]
                require(p['selected_tile']==[r,c] and p['endpoint_arm']==f'fine_{r}_{c}' and p['kind']=='deterministic_k1'
                  and p['mask']==[[i==r and j==c for j in range(5)] for i in range(4)],'Real singleton mask')
            policies+=1
        checks.append({'target_id':tid,'maximum_absolute_score_difference':diff,
          'choices':{name:list(g0.TILES[g0.select(own,name)]) for name in g0.SCORES}})
    require(count==672 and policies==160,'Complete independent double implementation comparison')
    return {'schema':'flow_r19_independent_double_score_check_v1','status':'PASS','graph_id':raw.GRAPH,
      'quality_sha256':a.quality_sha256,'score_sha256':a.scores_sha256,'source_result_sha256':q['source_result_sha256'],
      'execution_audit_sha256':q['execution_audit_sha256'],'independent_IO_freeze_sha256':a.freeze_sha256,
      'score_values':count,'score_selections':56,'policy_frame_weights':policies,
      'maximum_absolute_score_difference':max(f['maximum_absolute_score_difference'] for f in checks),'frames':checks,
      'scope':'Two fixed mathematical implementations on current native inputs; no new model/GT reduction/score tuning.'}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('freeze','quality','scores'):p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Fresh comparison artifact')
    out=compare(a);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({k:out[k] for k in ('status','score_values','score_selections','maximum_absolute_score_difference')}))
