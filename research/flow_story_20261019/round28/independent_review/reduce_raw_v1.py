#!/usr/bin/env python3
"""R28 two-state new128 raw predictions; reuse pinned R27 GT/masks/references."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import q4_analysis as core
import raw_quality_math as raw

HERE=Path(__file__).resolve().parent
R27=core.R27
require=core.require
REFERENCE_SHA='2855b04808568ce34afc1fe097c14739011f73a2542e8331e71f777a37908a0e'
MEMBERS_SHA='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'
ANALYSIS_SHA='7b9bc437cb42901fc794bc2caf519b600c9a809be380c8dac82cf9f3733730af'
GRAPH='flow_r28_all12mlp_patch4conv_integer_Q4_complete9_tailQ8_all78current_v1'
AUDIT_SCHEMA='flow_r28_independent_actual_execution_audit_v1'
REFERENCE_PATH=R27/'actual_quality_v1/ACTUAL_RAW_QUALITY_V1.json'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def safe(root,rel):
 p=Path(rel);require(not p.is_absolute() and '..' not in p.parts,'Safe relative artifact path');return Path(root)/p
def descriptor(d):
 p=Path(d['path']);require(sha(p)==d['sha256'],'Bound actual bytes '+str(p));return p
def audit_dependency(freeze_path,name):
 # The actual auditor explicitly pins sibling R27 sources. This exception
 # applies only to frozen source dependencies, never prediction/data paths.
 base=Path(freeze_path).resolve().parent;p=(base/Path(name)).resolve()
 require(p.is_relative_to(base.parent),'Auditor dependency remains inside this flow-date tree')
 return p

def source_preflight(path):
 f=read(path);require(f['schema']=='flow_r28_raw_quality_io_freeze_v1','Frozen new quality IO')
 for name,h in f['files'].items():require(sha(HERE/name)==h,'Own frozen source '+name)
 for d in f['dependencies'].values():descriptor(d)
 require(Path(raw.__file__).resolve()==R27/'raw_quality_math.py','Original raw math import')
 for key in ('auditor','audit_freeze'):descriptor(f[key])
 af=Path(f['audit_freeze']['path'])
 for name,h in read(af)['files'].items():
  p=audit_dependency(af,name)
  require(sha(p)==h,'Frozen auditor dependency '+name)
 return f

def validate_audit(a,f,ids):
 require(a['schema']==AUDIT_SCHEMA and a['status']=='PASS' and a['quality_analysis_allowed'] is True,'Complete actual Q4 execution audit')
 require(a['auditor_sha256']==f['auditor']['sha256'] and a['auditor_freeze_sha256']==f['audit_freeze']['sha256'] and a['pre_result_freeze_sha256']==ANALYSIS_SHA,'Actual auditor/source/prospective rule bound')
 require(a['entry_scope_correction_sha256']==f['dependencies']['entry_scope_correction']['sha256'],'Prospective uniform/mixed distinction')
 for k,n in dict(optimizer_updates=512,quality_predictions=128,quality_archives=128,formal_model_forwards=1155).items():require(a['counts'][k]==n,'Complete fixed count '+k)
 require(set(a['source_results'])=={'Q4'},'Single real new Q4 arm')
 rows=a['predictions'];by={(r['state'],r['target_id']):r for r in rows}
 require(len(rows)==len(by)==128 and set(by)=={(s,t) for s in core.NEW_STATES for t in ids},'Exact two new states x64 unique predictions')
 return by

def bind(inp,f):
 require(inp['schema']=='flow_r28_raw_quality_inputs_v1','R28 raw input schema')
 require(sha(REFERENCE_PATH)==REFERENCE_SHA and sha(R27/'membership_v1.json')==MEMBERS_SHA,'Fixed old quality/identities')
 ref=read(REFERENCE_PATH);members=read(R27/'membership_v1.json');ids=[x['target_id'] for x in members['dev_members']]
 require(len(ids)==len(set(ids))==64 and [x['target_id'] for x in ref['frames']]==ids,'Fixed64 identity order')
 require(not set(ids)&set(members['sealed_ids']+members['H8_historical_ids']),'No sealed/H8 identities')
 ap=descriptor(inp['execution_audit']);audit=read(ap);by=validate_audit(audit,f,ids)
 rp=descriptor(inp['run']);result=read(rp)
 require(audit['source_results']['Q4']==inp['run']['sha256'],'Actual Q4 result binding')
 require(result['schema']=='flow_r28_bounded_recovery_v1' and result['graph_id']==GRAPH and result['arm']=='Q4' and result['status']=='FIXED512_AND128_DEVELOPMENT_PREDICTIONS_COMPLETE','Declared complete new graph')
 require(result['completed_optimizer_steps']==512 and result['sources']['members']==MEMBERS_SHA,'Fixed endpoint/membership')
 require(all(result[k] is False for k in ('development_GT_read','quality_computed','quality_selection','formal_resume_from_smoke')),'No dev quality selection or smoke continuation')
 ledger=safe(rp.parent,result['training_ledger_file']);require(sha(ledger)==result['training_ledger_sha256'],'Actual ledger bytes')
 steps=[json.loads(x) for x in ledger.read_text().splitlines() if x.strip()];require(len(steps)==512,'512 ledger records')
 for a,b in zip(steps,members['train_order']):require(all(a[k]==b[k] for k in ('step','target_id','sequence','pass_index','crop')),'Matched order/crop')
 old_audit=read(descriptor(ref['execution_audit']));require(old_audit['status']=='PASS' and old_audit['quality_analysis_allowed'] is True,'Actual completed R27 reference audit')
 old_by={(r['state'],r['target_id']):r for r in old_audit['predictions']}
 # Only source bytes of old predictions are rechecked; their scalar metrics
 # were independently measured by R27 and are explicitly reused, not recast as new.
 for arm in ('native','Q8'):
  oldpath=descriptor(ref['source_results'][arm]);old=read(oldpath)
  require(old_audit['source_results'][arm]==ref['source_results'][arm]['sha256'],'Old actual result bound')
  require(all(result[k]==old[k] for k in ('initial_state_sha256','teacher_initial_state_sha256','recipe')),'Same original W/teacher/ordinary recovery opportunity')
  for phase,state in [('initial','original_native' if arm=='native' else 'initial_Q8'),('endpoint','recovered_'+arm)]:
   for row in old['evaluations'][phase]['frames']:
    a=old_by[(state,row['target_id'])];p=safe(oldpath.parent,row['prediction_file'])
    require(p.resolve()==Path(a['file']).resolve() and sha(p)==row['prediction_file_sha256']==a['file_sha256'] and row['prediction_sha256']==a['head3_sha256'],'Old actual reference prediction provenance')
 snapshots={x['target_id']:x for x in ref['shared_source_artifacts']}
 records={x['target_id']:x for x in members['records'] if x['role']=='development'}
 require(len(snapshots)==len(records)==64 and set(snapshots)==set(ids),'All64 reused GT/mask snapshots')
 shared={};evidence={}
 for row in members['dev_members']:
  tid=row['target_id'];s=snapshots[tid];rec=records[tid]
  evidence[tid]={{'input':'events','gt':'gt','mask':'mask'}[x['category']]:dict(relative_path=x['relative_path'],sha256=x['sha256']) for x in rec['source_files']}
  require(s['sequence']==row['sequence'] and s['input_evidence']==evidence[tid],'Shared GT original accepted source')
  p=Path(s['file']);p=p if p.is_absolute() else safe(R27,p)
  require(sha(p)==s['file_sha256'],'Original frozen shared GT/mask archive')
  shared[tid]=dict(path=p,record=s)
 bound={}
 for phase,state in [('initial','initial_Q4'),('endpoint','recovered_Q4')]:
  rows=result['evaluations'][phase]['frames'];require([(x['target_id'],x['sequence']) for x in rows]==[(x['target_id'],x['sequence']) for x in members['dev_members']],'Exact64 new state order')
  for row in rows:
   tid=row['target_id'];a=by[(state,tid)]
   require(row['sequence']==a['sequence'] and row['input_evidence']==a['input_evidence']==evidence[tid],'Same new prediction source')
   p=safe(rp.parent,row['prediction_file']);require(p.resolve()==Path(a['file']).resolve() and sha(p)==row['prediction_file_sha256']==a['file_sha256'],'Audited new actual archive')
   with np.load(p,allow_pickle=False) as z:
    require(z.files==['head3'],'Only head3 quality payload');v=z['head3']
    require(v.dtype==np.float32 and v.shape==(2,480,640) and np.isfinite(v).all(),'Fullgrid pixel units float32')
    require(raw.payload_sha(v)==row['prediction_sha256']==a['head3_sha256'],'New actual payload SHA')
   bound[(state,tid)]=dict(path=p,sha256=a['file_sha256'],payload_sha256=a['head3_sha256'])
 return ref,members,shared,bound,rp,ap

def run(a):
 require(not a.output.exists(),'Fresh R28 independent quality output')
 f=source_preflight(a.freeze);inp=read(a.inputs)
 ref,members,shared,bound,rp,ap=bind(inp,f)
 # First GT numerical access occurs only after the complete audit and every
 # new128 prediction and all64 old snapshot archives have closed.
 frames=[]
 for row in members['dev_members']:
  tid=row['target_id'];s=shared[tid];require(sha(s['path'])==s['record']['file_sha256'],'Shared archive unchanged')
  with np.load(s['path'],allow_pickle=False) as z:
   require(set(z.files)==set(s['record']['payloads'])=={'gt','mask','voxel_nonzero_bins'}|{'region__'+r for r in core.old.REGIONS},'Exact19 shared source payloads')
   arrays={k:z[k] for k in z.files}
  for k,v in arrays.items():
   d=s['record']['payloads'][k];require(list(v.shape)==d['shape'] and str(v.dtype)==d['dtype'] and raw.payload_sha(v)==d['sha256'],'Reused original payload '+k)
  gt=arrays['gt'];mask=arrays['mask'];masks={r:arrays['region__'+r] for r in core.old.REGIONS}
  require(gt.dtype==np.float32 and gt.shape==(2,480,640) and np.isfinite(gt).all() and mask.dtype==np.bool_ and mask.shape==(480,640),'Original GT geometry/dtype')
  require(np.array_equal(masks['all_valid'],mask) and all(v.dtype==np.bool_ and v.shape==mask.shape and not np.any(v&~mask) for v in masks.values()),'Same16 original GT-subset masks')
  states={}
  for state in core.NEW_STATES:
   d=bound[(state,tid)];require(sha(d['path'])==d['sha256'],'Prediction unchanged')
   with np.load(d['path'],allow_pickle=False) as z:
    pred=z['head3'];require(raw.payload_sha(pred)==d['payload_sha256'],'Prediction payload unchanged');states[state]=raw.measure(pred,gt,masks)
  frames.append(dict(target_id=tid,sequence=row['sequence'],states=states))
 out=core.analyze(ref,frames)
 out.update(schema='flow_r28_independent_raw_quality_v1',status='RAW_METRICS_COMPLETE',graph_id=GRAPH,
  source_result=dict(path=str(rp),sha256=sha(rp)),execution_audit=dict(path=str(ap),sha256=sha(ap)),
  reference_quality=dict(path=str(REFERENCE_PATH),sha256=REFERENCE_SHA),source_inputs_manifest_sha256=sha(a.inputs),io_freeze_sha256=sha(a.freeze),
  reused_GT_snapshot_archives=64,reused_reference_prediction_archives_byte_checked=256,independently_recomputed_new_predictions=128,
  scope_note='Only new Q4 raw predictions independently remeasured. Original GT/masks and old reference scalar quality are reused after SHA binding; no new training, threshold fit, selector or family-wide impossibility claim.')
 a.output.mkdir(parents=True);path=a.output/'ACTUAL_RAW_QUALITY_V1.json';write(path,out)
 print(json.dumps(dict(status=out['status'],decision=out['configuration_decision'],quality_sha256=sha(path),new_predictions=128,review_flags={k:len(v['high_support_tail_review']) for k,v in out['comparisons'].items()})))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',type=Path,default=HERE/'IO_FREEZE_V1.json');p.add_argument('--source-preflight-only',action='store_true');a=p.parse_args()
 if a.source_preflight_only:source_preflight(a.freeze);print('R28_RAW_QUALITY_SOURCE_PREFLIGHT_PASS')
 else:run(a)
