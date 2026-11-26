#!/usr/bin/env python3
"""Full fixed R17 dense-only screen: independent original-array quality reduction.

The execution audit and every prediction archive are validated before any
original GT array is loaded. A failed prefix cannot reach quality reduction.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import sys
PURE_ROOT=Path(__file__).resolve().parents[2]/'current_bn_round16/independent_review'
sys.path.insert(0,str(PURE_ROOT))
import pilot_quality_core as core

HERE=Path(__file__).resolve().parent
R17=HERE.parent
REPO=HERE.parents[3]
R13=REPO/'research/flow_story_20261016/coverage_recovery_round13'
H8=R13/'next_action_headroom/implementation_h0/h8_manifest.json'
SNAPSHOT=R13/'next_action_headroom/h8_input_snapshot_v1'
REGIONS=R13/'preparation/regions_v1.json'
MATH=R13/'independent_quality_review/reduce_raw_quality.py'
SHA_H8='1083d2a3f9b940aee0ebf271cb30b062eeed7bc4bcbf51429d573172573569f5'
SHA_SNAPSHOT='7f0cc5d74ce171aaa82cc602ec877e8fd58456e7481a9a158bc397cb8efa7b5d'
SHA_REGIONS='2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b'
SHA_MATH='26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088'
SHA_PURE='96960f6b23d402e56ca6bc99d707eab4d1d23c9a496f1220d60b1bdaf7f783ee'

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def payload(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def load_math():
    require(sha(MATH)==SHA_MATH,'Fixed independent metric mathematics')
    spec=importlib.util.spec_from_file_location('_r16_original_array_metrics',MATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(set(module.REGIONS)==set(core.REGIONS),'Identical sixteen-region definition')
    return module

def source_preflight(freeze_path,expected_sha):
    require(sha(freeze_path)==expected_sha,'Exact independent IO freeze')
    freeze=read(freeze_path)
    require(freeze['schema']=='flow_r17_dense_quality_io_freeze_v1','IO freeze schema')
    for name,digest in freeze['own_files'].items():require(sha(HERE/name)==digest,'Independent source '+name)
    require(sha(PURE_ROOT/'PILOT_QUALITY_PURE_FREEZE_V1.json')==SHA_PURE,'Unchanged pure analysis')
    pure=read(PURE_ROOT/'PILOT_QUALITY_PURE_FREEZE_V1.json')
    for name,digest in pure['files'].items():require(sha(PURE_ROOT/name)==digest,'Pure source '+name)
    for label,spec in freeze['external_sources'].items():require(sha(REPO/spec['relative_path'])==spec['sha256'],'External source '+label)
    for path,digest in ((H8,SHA_H8),(SNAPSHOT/'receipt.json',SHA_SNAPSHOT),(REGIONS,SHA_REGIONS),(MATH,SHA_MATH)):
        require(sha(path)==digest,'Identity/input/region/math source '+str(path))
    return freeze

def plan(members):
    out=[(row['target_id'],arm,'native' if arm=='native' else 'expanded')
         for row in members for arm in core.ARMS]
    require(len(out)==120,'Fixed 8x15 dense-only plan')
    return out


def reduce(a):
    freeze=source_preflight(a.freeze,a.freeze_sha256)
    result_path=a.result_dir/'results.json';result=read(result_path)
    audit_path=REPO/freeze['actual_audit_relative_path'];audit=read(audit_path)
    source=freeze['external_sources']
    require(audit['schema']=='flow_r17_dense_reference_independent_execution_audit_v1' and audit['status']=='PASS','Complete independently audited execution required')
    require(audit['quality_analysis_allowed'] is True,'Explicit complete-only quality interpretation')
    require(audit['source_result_sha256']==sha(result_path),'Actual result audited')
    require(audit['auditor_sha256']==source['execution_auditor']['sha256'] and
            audit['pre_result_freeze_sha256']==source['execution_audit_freeze']['sha256'],'Actual fixed auditor')
    require(audit['source_freeze_sha256']==source['pilot_source_freeze']['sha256'],'Actual pilot source binding')
    expected_counts={'full_forwards':120,'native':8,'expanded':112,'compact':0,'FC1_rows':66960000,'FC2_rows':66960000,
                     'BN_calls':9360,'archives':120,'head_payloads':480,
                     'capture_payloads':64,'optimizer_steps':0,'calibration_forwards':0}
    require(all(audit['counts'][key]==value for key,value in expected_counts.items()),'Complete fixed execution budget')
    require(audit['counts']['observed_FC_modules']==2880 and audit['counts']['compact_target_calls']==0
            and audit['counts']['BN_input_elements']>0,'Full dense observer inventory')
    require(len(audit['numerical_checks'])==8 and all(v['pass'] for v in audit['numerical_checks']),'All eight FF-native checks')
    require(result['schema']=='flow_r17_dense_reference_v1' and result['status']=='DENSE_REFERENCE120_COMPLETE','No partial quality interpretation')
    require(result['full_forward_count']==result['full_forward_attempts']==120,'No failed/replayed attempts')
    require(result['optimizer_steps']==result['calibration_forwards']==0 and result['training_performed'] is False,'Zero training/calibration')
    b=result['binding']
    require(b['plan']=='dense120' and b['BN_mode']=='REFERENCE_DENSE_CURRENT' and b['compact_forwards']==0 and b['other_11_MLPs']=='native','Fixed execution mechanism')
    require(b['h8_sha256']==SHA_H8 and b['region_sha256']==SHA_REGIONS,'Same H8/D200 regions')
    require(b['source_freeze_sha256']==source['pilot_source_freeze']['sha256'],'Runner source')
    require(b['independent_admission_sha256']==source['independent_admission']['sha256'],'Finite source admission')
    require(all(result[k] is True for k in ('all_states_unchanged','all_hooks_removed','original_modules_restored')),'Actual closure')
    checks=result['numerical_checks'];require(len(checks)==8 and all(c['pass'] for c in checks),'All numerical gates')
    members=read(H8)['members'];ids=[r['target_id'] for r in members];expected_plan=plan(members)
    require(len(result['forward_records'])==len(audit['predictions'])==120,'Complete raw prediction records')
    evidence={}
    receipt=read(SNAPSHOT/'receipt.json')
    for f in receipt['all_files']:evidence.setdefault(f['target_id'],{})[f['kind']]=f['sha256']
    predictions={};archives=0;heads=0
    # Read and verify all inference artifacts before permitting GT reads.
    for i,(record,independent,expected) in enumerate(zip(result['forward_records'],audit['predictions'],expected_plan),1):
        key=tuple(record[k] for k in ('target_id','arm','execution'))
        require(key==expected and record['forward_id']==i,'Ordered exact plan')
        require(tuple(independent[k] for k in ('target_id','arm','execution'))==expected and independent['forward_id']==i,'Ordered audited prediction')
        require(record['input_evidence']==evidence[key[0]],'Actual input source snapshot')
        relative=Path(record['prediction_file']);require(len(relative.parts)==1 and relative.suffix=='.npz','Safe prediction file')
        path=a.result_dir/relative
        require(relative.name==Path(independent['file']).name and sha(path)==record['prediction_file_sha256']==independent['file_sha256'],'Actual raw archive binding')
        with np.load(path,allow_pickle=False) as z:
            for j in range(4):
                arr=z[f'head{j}'];meta=record['head_records'][j]
                require(np.isfinite(arr).all() and arr.dtype==np.float32 and list(arr.shape)==meta['shape'] and payload(arr)==meta['sha256'],'Four raw head arrays')
                heads+=1
            final=z['head3']
            require(final.shape==(1,2,480,640),'Native final full-resolution flow')
            require(payload(final)==independent['head3_sha256'],'Independent final head audit')
            require((key[0],key[1]) not in predictions,'Unique actual endpoint')
            predictions[(key[0],key[1])]=final[0].copy()
        archives+=1
    require(archives==120 and heads==480 and len(predictions)==120,'Complete output population')
    # First numerical access to original events/GT/masks occurs only here.
    math=load_math();cache,source_evidence=math.load_snapshot(SNAPSHOT,SNAPSHOT/'receipt.json',ids)
    require(source_evidence==evidence,'Raw inputs agree with execution evidence')
    region=read(REGIONS)
    require(region['fit_uses_final_valid'] is False and len(region['fit_target_ids'])==200,'Fixed training-only region thresholds')
    frames=[]
    for row in members:
        tid=row['target_id'];original=cache[tid]
        masks=math.masks_from_original(original['gt'],original['mask'],original['events'],
                     region['boundary_threshold_pixels'],region['large_displacement_threshold_pixels'])
        metrics={arm:math.measure(predictions[(tid,arm)],original['gt'],masks) for arm in core.ARMS}
        frames.append({'target_id':tid,'sequence':row['sequence'],'arms':metrics})
    analysis=core.analyze(frames)
    overall={arm:{name:{'AEE':cell['frame_mean_AEE'],'valid_pixels':cell['valid_pixels']}
                  for name,cell in analysis['summaries'][arm]['overall'].items()} for arm in core.ARMS}
    analysis['overall_protection_vs_native']={arm:core.protection(overall[arm],overall['native'])
                                               for arm in core.ARMS[1:]}
    return {'schema':'flow_r17_dense_independent_raw_quality_v1','status':'COMPLETE_RAW_REDUCTION',
            'source_result_sha256':sha(result_path),'execution_audit_sha256':sha(audit_path),
            'independent_io_freeze_sha256':a.freeze_sha256,'math_sha256':SHA_MATH,'region_sha256':SHA_REGIONS,
            'h8_sha256':SHA_H8,'input_snapshot_sha256':SHA_SNAPSHOT,
            'raw_predictions':120,'raw_reference_predictions_numeric_only':0,'prediction_archives':archives,'source_arrays':24,
            'primary_metric_comparison':'Runner intentionally contains no AEE; this is original-array independent reduction, not comparison to an invented primary table.',
            **analysis}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--result-dir',type=Path);p.add_argument('--output',type=Path);p.add_argument('--source-preflight-only',action='store_true')
    a=p.parse_args()
    if a.source_preflight_only:
        source_preflight(a.freeze,a.freeze_sha256);print('SOURCE_PREFLIGHT_PASS_NO_RAW_QUALITY')
    else:
        require(a.result_dir and a.output and not a.output.exists(),'Fresh complete reduction output')
        result=reduce(a)
        with a.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        print(json.dumps({k:result[k] for k in ('status','raw_predictions','frame_region_cells','summary_cells')},indent=2))
