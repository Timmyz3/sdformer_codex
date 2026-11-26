#!/usr/bin/env python3
"""R19 new compact-graph original-array reduction; repeated endpoints excluded."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
R19=HERE.parent
REPO=HERE.parents[3]
DATE=HERE.parents[1]
R13=REPO/'research/flow_story_20261016/coverage_recovery_round13'
H8=R13/'next_action_headroom/implementation_h0/h8_manifest.json'
SNAPSHOT=R13/'next_action_headroom/h8_input_snapshot_v1'
REGIONS=R13/'preparation/regions_v1.json'
MATH=R13/'independent_quality_review/reduce_raw_quality.py'
PURE=DATE/'current_bn_round16/independent_review'
G0=DATE/'ordinary_controls_round18/independent_review'
H8_SHA='1083d2a3f9b940aee0ebf271cb30b062eeed7bc4bcbf51429d573172573569f5'
SNAPSHOT_SHA='7f0cc5d74ce171aaa82cc602ec877e8fd58456e7481a9a158bc397cb8efa7b5d'
REGION_SHA='2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b'
MATH_SHA='26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088'
PURE_SHA='96960f6b23d402e56ca6bc99d707eab4d1d23c9a496f1220d60b1bdaf7f783ee'
G0_SHA='110b715051179d75a20f4ca14ab829323b7b9722faceb5c53a57061cd7a96740'
GRAPH='r19_W271482_all78current_s2b2_R16V2_compact'
ARMS=('native','FF','CC')+tuple(f'fine_{r}_{c}' for r in range(3) for c in range(4))

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def payload(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def frozen_math():
    require(sha(MATH)==MATH_SHA,'Original-array metric source')
    require(sha(PURE/'PILOT_QUALITY_PURE_FREEZE_V1.json')==PURE_SHA,'Fixed signed-action mathematics')
    require(sha(G0/'PURE_FREEZE_V1.json')==G0_SHA,'Fixed G0 descriptor/policy mathematics')
    for base,filename in ((PURE,'PILOT_QUALITY_PURE_FREEZE_V1.json'),(G0,'PURE_FREEZE_V1.json')):
        for name,digest in read(base/filename)['files'].items():require(sha(base/name)==digest,'Frozen pure source '+name)
    metric=load(MATH,'r19_original_array_metrics');signed=load(PURE/'pilot_quality_core.py','r19_signed_action_core');g0=load(G0/'analysis_core.py','r19_ordinary_rule_core')
    require(tuple(signed.ARMS)==ARMS and set(signed.REGIONS)==set(metric.REGIONS)==set(g0.REGIONS),'Same full actions/regions')
    return metric,signed,g0

def preflight(path,digest):
    require(sha(path)==digest,'Independent R19 IO freeze')
    freeze=read(path);require(freeze['schema']=='flow_r19_independent_quality_io_freeze_v1','IO schema')
    for name,h in freeze['own_files'].items():require(sha(HERE/name)==h,'Own fixed IO source '+name)
    for name,spec in freeze['external_sources'].items():require(sha(REPO/spec['relative_path'])==spec['sha256'],'External fixed source '+name)
    for path,h in ((H8,H8_SHA),(SNAPSHOT/'receipt.json',SNAPSHOT_SHA),(REGIONS,REGION_SHA)):
        require(sha(path)==h,'Same identities/inputs/regions')
    frozen_math()
    return freeze

def plan(members):
    out=[]
    for i,m in enumerate(members):
        for arm in ARMS:
            out.append((m['target_id'],arm,'native' if arm=='native' else 'compact',True,arm))
            if i==0 and arm=='CC':out.extend((m['target_id'],f'CC_repeat{j}','compact',False,'CC') for j in (1,2))
    require(len(out)==122 and sum(v[3] for v in out)==120,'Fixed 122/120+2 plan')
    return out

def validate_complete(result,audit):
    require(result['schema']=='flow_r19_compact_graph_v1' and result['status']=='COMPACT_GRAPH122_COMPLETE','No partial quality')
    require(result['graph_id']==result['binding']['graph_id']==GRAPH,'This independently evaluated compact graph')
    require(audit['schema']=='flow_r19_compact_graph_independent_execution_audit_v1' and audit['status']=='PASS'
      and audit['quality_analysis_allowed'] is True,'Actual execution and numerical admission required')
    require(result['full_forward_attempts']==result['full_forward_count']==122,'Complete single bounded run')
    required={'full_forwards':122,'native':8,'compact':114,'expanded':0,'compact_target_calls':114,
      'quality_endpoints':120,'determinism_replays':2,'FC1_rows':67021520,'FC2_rows':67021520,
      'BN_calls':9516,'archives':122,'head_payloads':488,'capture_payloads':98,'optimizer_steps':0,'calibration_forwards':0}
    require(all(audit['counts'][k]==v for k,v in required.items()),'All exact actual execution counts')
    require(len(audit['numerical_checks'])==10 and all(v['pass'] for v in audit['numerical_checks']),'All numerical checks independently passed')
    kinds=[v['kind'] for v in audit['numerical_checks']]
    require(kinds.count('allfine_vs_native')==8 and kinds.count('CC_exact_replay')==2,'Eight anchors and two replays')
    require(result['training_performed'] is False and result['optimizer_steps']==result['calibration_forwards']==0,'Zero fitting/calibration')
    require(all(result[k] is True for k in ('all_states_unchanged','all_hooks_removed','original_modules_restored')),'Actual state closure')

def reduce(a):
    freeze=preflight(a.freeze,a.freeze_sha256);ext=freeze['external_sources']
    rp=a.result_dir/'results.json';result=read(rp)
    ap=REPO/freeze['actual_execution_audit_relative_path'];audit=read(ap);validate_complete(result,audit)
    require(audit['source_result_sha256']==sha(rp),'Audited actual result')
    require(audit['auditor_sha256']==ext['execution_auditor']['sha256'] and audit['pre_result_freeze_sha256']==ext['execution_audit_freeze']['sha256'],'Frozen actual auditor')
    require(audit['source_freeze_sha256']==ext['runner_source_freeze']['sha256']==result['binding']['source_freeze_sha256'],'Frozen runner')
    b=result['binding'];require(b['plan']=='compact122' and b['BN_mode']=='NATIVE_DENSE_CURRENT' and b['BN']=='all78_current'
      and b['other_11_MLPs']=='native' and b['h8_sha256']==H8_SHA and b['region_sha256']==REGION_SHA,'Same source execution semantics')
    require(b['independent_admission_sha256']==ext['source_admission']['sha256'],'Frozen new-graph source admission')
    require(b['inputs_sha256']==ext['runner_inputs']['sha256'],'Exact execution inputs manifest')
    members=read(H8)['members'];ids=[m['target_id'] for m in members];expected=plan(members)
    require(len(result['forward_records'])==len(audit['predictions'])==122,'Complete new artifacts')
    evidence={}
    for item in read(SNAPSHOT/'receipt.json')['all_files']:evidence.setdefault(item['target_id'],{})[item['kind']]=item['sha256']
    metric,signed,g0=frozen_math();predictions={};ordinary_scores={};weights={};head_count=0;captures_count=0
    for i,(r,ir,wanted) in enumerate(zip(result['forward_records'],audit['predictions'],expected),1):
        keys=('target_id','arm','execution','quality_endpoint','endpoint_arm')
        require(tuple(r[k] for k in keys)==tuple(ir[k] for k in keys)==wanted and r['forward_id']==ir['forward_id']==i,'Ordered actual identity/graph/replay role')
        require(r['input_evidence']==evidence[r['target_id']],'Actual event/GT/mask source bytes')
        name=Path(r['prediction_file']);require(len(name.parts)==1 and name.suffix=='.npz','Artifact basename only')
        path=a.result_dir/name;require(name.name==Path(ir['file']).name and sha(path)==r['prediction_file_sha256']==ir['file_sha256'],'Audited prediction bytes')
        with np.load(path,allow_pickle=False) as z:
            heads={f'head{j}' for j in range(4)}
            caps=set()
            if r['arm']=='native':caps={'u','branch','sn1','sn2'}
            elif r['arm']=='FF' or (r['target_id']==ids[0] and r['arm'].startswith('CC')):caps={'u','branch','sn1','sn2','sn1_packed','sn2_packed'}
            require(set(z.files)==heads|caps,'Exact primary/replay archive population')
            for j in range(4):
                x=z[f'head{j}'];meta=r['head_records'][j]
                require(x.dtype==np.float32 and np.isfinite(x).all() and list(x.shape)==meta['shape'] and payload(x)==meta['sha256'],'Four finite original heads')
                head_count+=1
            for name in caps:
                x=z[name];meta=r['tensor_records'][name]
                require(x.dtype==np.float32 and np.isfinite(x).all() and list(x.shape)==meta['shape'] and payload(x)==meta['sha256'],'Original numerical capture '+name)
                captures_count+=1
            final=z['head3'];require(final.shape==(1,2,480,640) and payload(final)==ir['head3_sha256'],'Final native coordinate output')
            if r['quality_endpoint']:
                key=(r['target_id'],r['endpoint_arm']);require(key not in predictions,'No duplicate quality endpoint')
                predictions[key]=final[0].copy()
            if r['arm']=='native':
                scores=g0.independently_score(z['u'],z['sn1'])
                ordinary_scores[r['target_id']]=scores;weights[r['target_id']]=g0.policy_weights(scores)
    require(head_count==488 and captures_count==98 and len(predictions)==120 and len(ordinary_scores)==8,'Complete new-graph observations before GT')
    # These are fresh reductions from the original arrays, never R17 labels.
    cache,snapshot_evidence=metric.load_snapshot(SNAPSHOT,SNAPSHOT/'receipt.json',ids)
    require(snapshot_evidence==evidence,'Original input snapshot matches all actual forwards')
    region=read(REGIONS);require(region['fit_uses_final_valid'] is False and len(region['fit_target_ids'])==200,'Unchanged training-only regions')
    frames=[]
    for member in members:
        tid=member['target_id'];src=cache[tid]
        masks=metric.masks_from_original(src['gt'],src['mask'],src['events'],region['boundary_threshold_pixels'],region['large_displacement_threshold_pixels'])
        frames.append({'target_id':tid,'sequence':member['sequence'],'arms':{arm:metric.measure(predictions[(tid,arm)],src['gt'],masks) for arm in ARMS}})
    action=signed.analyze(frames)
    wrap=lambda arm:{reg:{'AEE':cell['frame_mean_AEE'],'valid_pixels':cell['valid_pixels']} for reg,cell in action['summaries'][arm]['overall'].items()}
    action['overall_protection_vs_native']={arm:signed.protection(wrap(arm),wrap('native')) for arm in ARMS[1:]}
    policies=g0.analyze(frames,weights)
    return {'schema':'flow_r19_independent_compact_quality_v1','status':'COMPLETE_NEW_GRAPH_RAW_REDUCTION',
      'graph_id':GRAPH,'source_result_sha256':sha(rp),'execution_audit_sha256':sha(ap),'IO_freeze_sha256':a.freeze_sha256,
      'metric_source_sha256':MATH_SHA,'region_sha256':REGION_SHA,'snapshot_sha256':SNAPSHOT_SHA,
      'raw_prediction_archives':122,'raw_quality_predictions':120,'numerical_replays_excluded_from_quality':2,
      'raw_head_payloads':488,'raw_capture_payloads':98,'original_input_arrays':24,
      'ordinary_scores':ordinary_scores,'ordinary_policies':policies,**action,
      'scope':'New actual compact graph, exposed H8 development. No dense-quality transfer, restored equivalence, learned flow increment or end-to-end hardware benefit claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--result-dir',type=Path);p.add_argument('--output',type=Path);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    if a.preflight_only:preflight(a.freeze,a.freeze_sha256);print('R19_INDEPENDENT_SOURCE_PREFLIGHT_PASS')
    else:
        require(a.result_dir and a.output and not a.output.exists(),'Fresh complete reduction output')
        out=reduce(a);a.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
        print(json.dumps({k:out[k] for k in ('status','raw_quality_predictions','numerical_replays_excluded_from_quality','frame_region_cells','summary_cells')}))
