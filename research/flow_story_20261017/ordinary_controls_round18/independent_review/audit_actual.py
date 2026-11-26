#!/usr/bin/env python3
"""Independently recompute R18 cached scores and actual-endpoint expectations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import analysis_core as core

HERE=Path(__file__).resolve().parent
R18=HERE.parent
R17=R18.parent/'dense_reference_round17'
RESULT_SHA='0b4ef2744c9da8fd3513b65141b4133f381cf267db68083e4bb854165ae07a2e'
AUDIT_SHA='6ed07d00330df7824e4f8cc754a1c735bded4b3bf5bf89565f9114506594276f'
QUALITY_SHA='d3a0901a24dd6f15407dcb9425ecda9be2daa33c947ba279bd2b30e19f6e7990'
PURE_SHA='110b715051179d75a20f4ca14ab829323b7b9722faceb5c53a57061cd7a96740'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def payload(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
require=core.require

def preflight(path,digest):
    require(sha(path)==digest,'Independent IO freeze bytes')
    freeze=read(path);require(freeze['schema']=='flow_r18_independent_io_freeze_v1','IO freeze schema')
    require(sha(HERE/'PURE_FREEZE_V1.json')==PURE_SHA,'Frozen independent pure mathematics')
    pure=read(HERE/'PURE_FREEZE_V1.json')
    for name,h in pure['files'].items():require(sha(HERE/name)==h,'Pure source '+name)
    for name,h in freeze['own_files'].items():require(sha(HERE/name)==h,'Independent IO source '+name)
    for name,h in freeze['implementation_files'].items():require(sha(R18/'implementation'/name)==h,'Implementation source '+name)
    return freeze

def validate_header(x,freeze):
    require(x['schema']=='flow_r18_g0_cached_ordinary_v1' and x['status']=='G0_CPU_COMPLETE','Only complete fixed CPU score run')
    require(x['source_sha256']==freeze['implementation_source_sha256'],'Actual implementation sources')
    require(x['freeze_sha256']==freeze['implementation_files']['SOURCE_FREEZE_V1.json'],'Implementation freeze')
    require(x['independent_source_review_sha256']==freeze['own_files']['SOURCE_REVIEW_V1.json'],'Independent source review')
    require(x['R17_result_sha256']==RESULT_SHA and x['R17_execution_audit_sha256']==AUDIT_SHA,'Actual fixed R17 source')
    for key in ('model_forwards','PSN_calls','FC_calls','GPU_calls','optimizer_steps'):
        require(x[key]==0,'No new computation '+key)
    require(x['GT_loaded'] is False and x['quality_loaded'] is False,'No labels in scoring')
    require(x['score_names']==list(core.SCORES) and x['tiles']==[list(p) for p in core.TILES],'Fixed score and tile order')
    require(x['policy_count']==20 and x['score_policy_count']==7 and x['score_values']==672 and x['score_selections']==56
      and x['policy_frame_cells']==160 and x['selector_score_comparisons']==616,'Complete score/policy population')
    require(len(x['frames'])==len(x['archive_reads'])==8,'Complete eight source records')

def check_policy(actual,weights,name):
    require(actual['weights']==weights,'Exact policy distribution '+name)
    if name=='uniform_expectation':
        require(actual['kind']=='expected_uniform_randomized_k1' and actual['selected_tile'] is None and actual['mask'] is None,'Expected random policy, no actual choice')
        require(actual['rational_weight']=={'numerator':1,'denominator':12} and actual['endpoint_arms']==[f'fine_{r}_{c}' for r,c in core.TILES],'Uniform actual endpoint definition')
    else:
        index=weights.index(1);r,c=core.TILES[index]
        require(actual['kind']=='deterministic_k1' and actual['selected_tile']==[r,c] and actual['endpoint_arm']==f'fine_{r}_{c}','Actual selected endpoint '+name)
        expected=[[i==r and j==c for j in range(5)] for i in range(4)]
        require(actual['mask']==expected and all(type(v) is bool for row in actual['mask'] for v in row),'Exact 4x5 mask; tails coarse')

def audit(a):
    freeze=preflight(a.freeze,a.freeze_sha256)
    actual=read(a.scores);validate_header(actual,freeze)
    resultpath=R17/'actual_v1/run_v1/results.json';auditpath=R17/'independent_execution_audit/ACTUAL_EXECUTION_AUDIT_V1.json'
    qualitypath=R17/'independent_review/ACTUAL_RAW_QUALITY_V1.json'
    require(sha(resultpath)==RESULT_SHA and sha(auditpath)==AUDIT_SHA and sha(qualitypath)==QUALITY_SHA,'Original audited artifacts unchanged')
    result=read(resultpath);old_audit=read(auditpath)
    require(old_audit['status']=='PASS' and old_audit['quality_analysis_allowed'] is True,'Audited full R17 population')
    require(actual['h8_sha256']==result['binding']['h8_sha256'],'Actual original H8 metadata')
    native=[r for r in result['forward_records'] if r['arm']=='native']
    require(len(native)==8 and [f['target_id'] for f in actual['frames']]==[r['target_id'] for r in native],'Original H8 ordering')
    diagnostics=[];weights_by_id={};checked=0;max_abs=0.
    for f,r,io in zip(actual['frames'],native,actual['archive_reads']):
        p=resultpath.parent/r['prediction_file']
        require(f['input_archive']==p.name and f['input_archive_sha256']==r['prediction_file_sha256']==sha(p),'Native capture archive bytes')
        require(f['input_tensor_records']=={k:r['tensor_records'][k] for k in ('u','sn1')},'Exact tensor descriptors')
        require(Path(io['path']).name==p.name and io['sha256']==sha(p) and io['payload_names_loaded']==['u','sn1'],'Actual archive read record')
        with np.load(p,allow_pickle=False) as z:
            u=z['u'];sn1=z['sn1']
        for name,arr in (('u',u),('sn1',sn1)):
            require(arr.dtype==np.float32 and list(arr.shape)==r['tensor_records'][name]['shape']
              and payload(arr)==r['tensor_records'][name]['sha256'],'Original full tensor '+name)
        require(io['payload_elements_each']==u.size and io['uncompressed_payload_bytes']==u.nbytes+sn1.nbytes,'Loaded payload accounting')
        computed=core.independently_score(u,sn1)
        require(len(f['scores'])==12,'All twelve score tiles')
        frame_abs=0.
        for own,reported in zip(computed,f['scores']):
            require(set(reported)=={'tile',*core.SCORES} and own['tile']==reported['tile'],'All seven named scores in order')
            for name in core.SCORES:
                v=reported[name];require(np.isfinite(v) and v>=0 and np.isclose(v,own[name],atol=1e-10,rtol=1e-12),'Independent score math '+name)
                frame_abs=max(frame_abs,abs(v-own[name]));checked+=1
        expected=core.policy_weights(computed)
        require(f['ledger_tile_multiplicity']==12 and f['selector']['score_comparisons']==77
          and f['selector']['comparisons_per_score']==11 and f['selector']['selected_score_policies']==7
          and f['selector']['fixed_policies']==12 and f['selector']['expectation_policies']==1,'Complete fixed selector bookkeeping')
        require(set(f['policies'])==set(core.SCORES)|{f'fixed_{r}_{c}' for r,c in core.TILES}|{'uniform_random_expected'},'Twenty policies with no extras')
        for name,w in expected.items():
            key='uniform_random_expected' if name=='uniform_expectation' else name
            check_policy(f['policies'][key],w,name)
        # Recomputed rank must agree even if an allowed summation error could swap it.
        require(all(core.select(computed,n)==core.select(f['scores'],n) for n in core.SCORES),'Exact top1 rank agreement')
        weights_by_id[f['target_id']]=expected
        diagnostics.append({'target_id':f['target_id'],'maximum_score_absolute_difference':frame_abs,
          'source_u_sha256':r['tensor_records']['u']['sha256'],'source_sn1_sha256':r['tensor_records']['sn1']['sha256'],
          'selections':{n:list(core.TILES[core.select(computed,n)]) for n in core.SCORES},'recomputed_scores':computed})
        max_abs=max(max_abs,frame_abs)
    require(checked==672,'All actual score values audited')
    require(actual['total_loaded_payload_bytes']==sum(v['uncompressed_payload_bytes'] for v in actual['archive_reads'])==294912000,'Full original capture payload read count')
    quality=read(qualitypath)
    require(quality['status']=='COMPLETE_RAW_REDUCTION' and quality['raw_predictions']==120 and quality['frame_region_cells']==1920 and quality['summary_cells']==2160,'Prior independent original-GT population')
    require([f['target_id'] for f in quality['frames']]==[r['target_id'] for r in native],'Quality H8 identity order')
    analysis=core.analyze(quality['frames'],weights_by_id)
    return {'schema':'flow_r18_independent_score_quality_v1','status':'PASS_COMPLETE_INDEPENDENT_SCORE_AND_QUALITY',
      'score_result_sha256':sha(a.scores),'IO_freeze_sha256':a.freeze_sha256,'pure_freeze_sha256':PURE_SHA,
      'R17_result_sha256':RESULT_SHA,'R17_execution_audit_sha256':AUDIT_SHA,'R17_independent_quality_sha256':QUALITY_SHA,
      'score_values_independently_recomputed':checked,'score_selections_independently_checked':56,
      'source_capture_payloads_independently_read':16,'maximum_score_absolute_difference':max_abs,
      'score_audit':diagnostics,'R17_reference_summaries':{arm:quality['summaries'][arm] for arm in ('native','CC')},
      'new_model_forwards':0,'new_GT_array_reduction':False,'uniform_expected_error_not_flow_ensemble':True,**analysis}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--scores',type=Path);p.add_argument('--output',type=Path);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    if a.preflight_only:preflight(a.freeze,a.freeze_sha256);print('INDEPENDENT_SOURCE_PREFLIGHT_PASS')
    else:
        require(a.scores and a.output and not a.output.exists(),'Fresh independent output')
        out=audit(a);a.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
        print(json.dumps({k:out[k] for k in ('status','score_values_independently_recomputed','score_selections_independently_checked','frame_region_cells','summary_cells')}))
