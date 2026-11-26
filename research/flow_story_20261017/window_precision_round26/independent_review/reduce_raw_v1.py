#!/usr/bin/env python3
"""R26 original-array quality. Fixed40 endpoints; two Q6 repeats excluded."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
R26=HERE.parent
REPO=HERE.parents[3]
R13=REPO/'research/flow_story_20261016/coverage_recovery_round13'
H8=R13/'next_action_headroom/implementation_h0/h8_manifest.json'
SNAPSHOT=R13/'next_action_headroom/h8_input_snapshot_v1'
REGIONS=R13/'preparation/regions_v1.json'
MATH=R13/'independent_quality_review/reduce_raw_quality.py'
H8_SHA='1083d2a3f9b940aee0ebf271cb30b062eeed7bc4bcbf51429d573172573569f5'
SNAPSHOT_SHA='7f0cc5d74ce171aaa82cc602ec877e8fd58456e7481a9a158bc397cb8efa7b5d'
REGION_SHA='2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b'
MATH_SHA='26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088'
PURE_SHA='9bd9a36cc78581db3aa26b6010439cc42529958ab75c1d1cd1d7dce54c5b3616'
GRAPH='r26_W271482_all78current_s2b2_fullspatial_weightplanes_H8'
ARMS=('native','Q8ref','Q8serial','Q6','Q4')
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def payload(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def frozen_math():
 require(sha(MATH)==MATH_SHA and sha(HERE/'ANALYSIS_FREEZE_V1.json')==PURE_SHA,'Immutable metric and pre-result math')
 for name,h in read(HERE/'ANALYSIS_FREEZE_V1.json')['files'].items():require(sha(HERE/name)==h,'Fixed pure source '+name)
 metric=load(MATH,'r26_original_array_metrics');core=load(HERE/'analysis_core.py','r26_fixed_precision_science')
 require(core.ARMS==ARMS and set(core.REGIONS)==set(metric.REGIONS),'Complete five arms/full16')
 return metric,core

def preflight(p,digest):
 require(sha(p)==digest,'Independent IO freeze bytes');f=read(p)
 require(f['schema']=='flow_r26_independent_quality_io_freeze_v1','Fixed IO schema')
 for name,h in f['own_files'].items():require(sha(HERE/name)==h,'Own fixed source '+name)
 for name,spec in f['external_sources'].items():require(sha(REPO/spec['relative_path'])==spec['sha256'],'External fixed source '+name)
 for p,h in ((H8,H8_SHA),(SNAPSHOT/'receipt.json',SNAPSHOT_SHA),(REGIONS,REGION_SHA)):require(sha(p)==h,'Original identities/inputs/regions')
 frozen_math();return f

def plan(members):
 items=[]
 for i,m in enumerate(members):
  for arm in ARMS:
   items.append((m['target_id'],arm,'native' if arm=='native' else 'integer_quantized',True,arm))
   if i==0 and arm=='Q6':items.extend((m['target_id'],f'Q6_repeat{j}','integer_quantized',False,'Q6') for j in (1,2))
 require(len(items)==42 and sum(x[3] for x in items)==40,'Fixed42/40+2');return items

def validate_complete(r,a):
 require(r['schema']=='flow_r26_window_precision_v1' and r['status']=='WINDOW_PRECISION42_COMPLETE','Complete fixed precision graph')
 require(r['graph_id']==r['binding']['graph_id']==GRAPH,'Actual new graph')
 require(a['schema']=='flow_r26_window_precision_independent_execution_audit_v1' and a['status']=='PASS' and a['quality_analysis_allowed'] is True,'Actual independent complete execution/numerical audit')
 require(r['full_forward_count']==r['full_forward_attempts']==42,'No missing/extra actual model calls')
 required={'full_forwards':42,'native':8,'integer_reference':8,'bitserial':26,'quality_endpoints':40,'determinism_replays':2,'full_grid_FC1_rows':23436000,'full_grid_FC2_rows':23436000,'original_FC1_rows':23028000,'original_FC2_rows':23028000,'target_row_plane_FC1_work':1990560,'target_row_plane_FC2_work':1990560,'model_int_mm_calls':432,'backend_operator_calls':4,'BN_calls':3276,'archives':42,'head_payloads':168,'float_capture_payloads':108,'integer_capture_payloads':38,'saved_payloads':314,'quantization_archives':1,'quantization_payloads':4,'optimizer_steps':0,'calibration_forwards':0}
 require(all(a['counts'][k]==v for k,v in required.items()),'Exact execution raw/row/plane/BN inventory')
 require(a['graph_id']==GRAPH,'Independently audited graph')
 require(len(a['numerical_checks'])==10 and all(c['pass'] for c in a['numerical_checks']),'Actual10 raw checks')
 kinds=[x['kind'] for x in a['numerical_checks']]
 require(kinds.count('Q8serial_vs_Q8ref_exact')==8 and kinds.count('Q6_exact_replay')==2,'Eight exact Q8 graphs and two Q6 replays')
 require(r['training_performed'] is False and r['optimizer_steps']==r['calibration_forwards']==0,'Zero recovery/calibration')
 require(all(r[k] is True for k in ('all_states_unchanged','all_hooks_removed','original_modules_restored')),'Actual state/module closure')
 require(len(r['backend_operator_smoke'])==4 and all(x['passed_exact_column_sum'] and x['model_forward_count']==0 for x in r['backend_operator_smoke']),'Four separate integer operator smoke checks')
 # The execution auditor independently verifies detailed integer rows/planes;
 # this I/O independently recounts the complete 42 archives/314 raw arrays.
 require(len(r['forward_records'])==len(a['predictions'])==42,'Complete actual artifacts')

def reduce(args):
 f=preflight(args.freeze,args.freeze_sha256);ext=f['external_sources'];rp=args.result_dir/'results.json';r=read(rp)
 ap=REPO/f['actual_execution_audit_relative_path'];a=read(ap);validate_complete(r,a)
 require(a['source_result_sha256']==sha(rp),'Independently audited actual result')
 require(a['auditor_sha256']==ext['execution_auditor']['sha256'] and a['pre_result_freeze_sha256']==ext['execution_audit_freeze']['sha256'],'Frozen execution audit source')
 require(a['source_freeze_sha256']==ext['runner_source_freeze']['sha256']==r['binding']['source_freeze_sha256'],'Actual runner source freeze')
 b=r['binding'];require(b['plan']=='precision42' and b['BN']=='all78_current' and b['BN_mode']=='ALL78_NATIVE_CURRENT' and b['other_11_MLPs']=='native','Full native grid/BN, fixed target FCs only')
 require(b['h8_sha256']==H8_SHA and b['region_sha256']==REGION_SHA and b['inputs_sha256']==ext['runner_inputs']['sha256'],'Exact identities/regions/runtime input contract')
 require(b['primitive_sha256']==ext['primitive']['sha256'] and b['data_only_helper_sha256']==ext['data_helper']['sha256'] and b['independent_admission_sha256']==ext['source_admission']['sha256'],'Exact primitive/data-only helper/admission')
 require(b['old_spatial_compact_called'] is False and b['expected_full_grid_logical_FC_rows_each']==23436000 and b['expected_target_row_plane_work_each']==1990560 and b['expected_native_BN_calls']==3276,'New full-grid plane work, not old row savings')
 members=read(H8)['members'];ids=[m['target_id'] for m in members];expected=plan(members)
 evidence={}
 for z in read(SNAPSHOT/'receipt.json')['all_files']:evidence.setdefault(z['target_id'],{})[z['kind']]=z['sha256']
 require(r['data_evidence']==[dict(target_id=t,input_evidence=evidence[t]) for t in ids],'Complete original event/GT/mask byte contract')
 predictions={};heads=floats=integers=0;pair_hashes={};q6_hashes=None
 for fid,(record,ir,wanted) in enumerate(zip(r['forward_records'],a['predictions'],expected),1):
  keys=('target_id','arm','execution','quality_endpoint','endpoint_arm')
  require(tuple(record[k] for k in keys)==tuple(ir[k] for k in keys)==wanted and record['forward_id']==ir['forward_id']==fid,'Exact fixed actual order')
  require(record['input_evidence']==evidence[record['target_id']],'Per-forward original inputs')
  require(record['state_and_BN_unchanged'] and record['original_modules_restored_and_hooks_removed'],'Per-forward closure')
  name=Path(record['prediction_file']);require(len(name.parts)==1 and name.suffix=='.npz','Artifact basename')
  path=args.result_dir/name;require(name.name==Path(ir['file']).name and sha(path)==record['prediction_file_sha256']==ir['file_sha256'],'Actual archive bytes')
  captured=record['arm'] in ('native','Q8ref','Q8serial') or (record['target_id']==ids[0] and record['endpoint_arm']=='Q6')
  caps={'u','branch','sn1','sn2'} if captured else set()
  ints={'fc1_integer','fc2_integer'} if captured and record['arm']!='native' else set()
  allkeys={f'head{j}' for j in range(4)}|caps|ints
  hashes={}
  with np.load(path,allow_pickle=False) as z:
   require(set(z.files)==set(record['saved_array_names'])==allkeys,'Exact primary/replay raw inventory')
   for key in sorted(allkeys):
    x=z[key];meta=record['head_records'][int(key[-1])] if key.startswith('head') else record['tensor_records'][key]
    dtype=np.int32 if key in ints else np.float32
    require(x.dtype==dtype and np.isfinite(x).all() and list(x.shape)==meta['shape'] and payload(x)==meta['sha256'],'Original finite float/int payload '+key)
    require(meta['dtype']==('torch.int32' if key in ints else 'torch.float32'),'Real dtype metadata')
    hashes[key]=payload(x)
    if key.startswith('head'):heads+=1
    elif key in ints:integers+=1
    else:floats+=1
   final=z['head3'];require(final.shape==(1,2,480,640) and hashes['head3']==ir['head3_sha256'],'Original output coordinates')
   if record['quality_endpoint']:
    key=(record['target_id'],record['endpoint_arm']);require(key not in predictions,'Unique quality endpoint');predictions[key]=final[0].copy()
  if record['arm']=='Q8ref':pair_hashes[record['target_id']]=hashes
  elif record['arm']=='Q8serial':require(hashes==pair_hashes[record['target_id']],'Own all10 Q8 raw equality')
  elif record['arm']=='Q6' and record['target_id']==ids[0]:q6_hashes=hashes
  elif record['arm'].startswith('Q6_repeat'):require(hashes==q6_hashes and len(hashes)==10,'Own all10 Q6 replay equality')
 require((heads,floats,integers,len(predictions))==(168,108,38,40),'Full314 original payloads and40 quality endpoints')
 # Quantized arrays are engineering sources, not extra quality endpoints.
 receipt=r['quantization_receipt'];archive=r['quantization_archive']
 for d in (receipt,archive):require(len(Path(d['path']).parts)==1 and sha(args.result_dir/d['path'])==d['sha256'],'Quantization source bytes')
 require(a['quantization']['receipt_sha256']==receipt['sha256'] and a['quantization']['archive_sha256']==archive['sha256'],'Independently audited quantization bytes')
 qr=read(args.result_dir/receipt['path']);require(qr['schema']=='flow_r26_weight_quantization_v1' and qr['arrays_sha256']==archive['sha256'] and qr['original_allcurrent_state_sha256']==r['initial_allcurrent_state_sha256'],'Own source quantization state binding')
 with np.load(args.result_dir/archive['path'],allow_pickle=False) as z:
  require(set(z.files)==set(qr['arrays_records'])=={'fc1__q8','fc1__scale','fc2__q8','fc2__scale'},'Exactly two q8/scale sources')
  for name in z.files:
   x=z[name];m=qr['arrays_records'][name];require(str(x.dtype)==m['dtype'] and list(x.shape)==m['shape'] and payload(x)==m['sha256'] and np.isfinite(x).all(),'Raw q8/scale payload')
 # This is the first GT numerical read in the quality program.
 metric,core=frozen_math();cache,snapshot_evidence=metric.load_snapshot(SNAPSHOT,SNAPSHOT/'receipt.json',ids)
 require(snapshot_evidence==evidence,'Original input snapshot source closure')
 cfg=read(REGIONS);require(cfg['fit_uses_final_valid'] is False and len(cfg['fit_target_ids'])==200,'Original training-only regions')
 frames=[]
 for m in members:
  tid=m['target_id'];src=cache[tid];masks=metric.masks_from_original(src['gt'],src['mask'],src['events'],cfg['boundary_threshold_pixels'],cfg['large_displacement_threshold_pixels'])
  frames.append(dict(target_id=tid,sequence=m['sequence'],arms={arm:metric.measure(predictions[tid,arm],src['gt'],masks) for arm in ARMS}))
 quality=core.analyze(frames,serial8_exact=True)
 return dict(schema='flow_r26_independent_precision_quality_v1',status='COMPLETE_NEW_GRAPH_RAW_REDUCTION',graph_id=GRAPH,
  source_result_sha256=sha(rp),execution_audit_sha256=sha(ap),IO_freeze_sha256=args.freeze_sha256,metric_source_sha256=MATH_SHA,region_sha256=REGION_SHA,snapshot_sha256=SNAPSHOT_SHA,
  raw_prediction_archives=42,raw_quality_predictions=40,raw_head_payloads=heads,raw_float_capture_payloads=floats,raw_integer_capture_payloads=integers,raw_quantization_source_arrays=4,original_input_arrays=24,**quality)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True);p.add_argument('--result-dir',type=Path);p.add_argument('--output',type=Path);p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
 if args.preflight_only:preflight(args.freeze,args.freeze_sha256);print('R26_INDEPENDENT_SOURCE_PREFLIGHT_PASS')
 else:
  require(args.result_dir and args.output and not args.output.exists(),'New complete reduction output');out=reduce(args);args.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({k:out[k] for k in ('status','raw_quality_predictions','Q8_conversion_anchor_pass','Q6_primary_pass','Q4_diagnostic_pass','resource_decision')}))
