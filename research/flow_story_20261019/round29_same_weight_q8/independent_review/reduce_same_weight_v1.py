#!/usr/bin/env python3
"""R29 new64 predictions only; no training or new action eligibility gate."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import analysis_same_weight as core
import raw_quality_math as raw
HERE=Path(__file__).resolve().parent
R27=core.R27;R28=HERE.parents[1]/'round28/independent_review'
REF27=R27/'actual_quality_v1/ACTUAL_RAW_QUALITY_V1.json'
REF28=R28/'actual_quality_v1/ACTUAL_RAW_QUALITY_V1.json'
REF27_SHA='2855b04808568ce34afc1fe097c14739011f73a2542e8331e71f777a37908a0e'
REF28_SHA='1d4e43b0224c38bb2e7d7d389c8ea711cd37b7f747d298eb0f817c63e0ffaabd'
MEMBERS_SHA='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'
GRAPH='flow_r29_same_Q4_recovered_weights_all_Q8_all78current_v1'
AUDIT_SCHEMA='flow_r29_independent_actual_execution_audit_v1'
require=core.require
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def descriptor(d):
 p=Path(d['path']);require(sha(p)==d['sha256'],'Bound original bytes '+str(p));return p
def safe(root,rel):
 p=Path(rel);require(not p.is_absolute() and '..' not in p.parts,'Safe relative artifact');return Path(root)/p
def preflight(path):
 f=read(path);require(f['schema']=='flow_r29_same_weight_quality_io_freeze_v1','Frozen independent new64 IO')
 for n,h in f['files'].items():require(sha(HERE/n)==h,'Own frozen source '+n)
 for d in f['dependencies'].values():descriptor(d)
 descriptor(f['auditor']) # Thin auditor has one explicit source SHA, no separate freeze field.
 require(Path(raw.__file__).resolve()==R27/'raw_quality_math.py','Actual original raw metric math')
 return f
def validate_audit(a,f,ids,result_sha,checkpoint_sha,state_sha,q4_audit_sha):
 require(a['schema']==AUDIT_SCHEMA and a['status']=='PASS' and a['quality_analysis_allowed'] is True,'Actual complete67 evaluation audit')
 require(a['auditor_sha256']==f['auditor']['sha256'],'Actual thin auditor source binding')
 for k,n in dict(optimizer_updates=0,quality_predictions=64,formal_model_forwards=67).items():require(a['counts'][k]==n,'Fixed evaluation count '+k)
 require(a['source_result_sha256']==result_sha and a['Q4_source_checkpoint_sha256']==checkpoint_sha and a['source_state_sha256']==state_sha and a['R28_execution_audit_sha256']==q4_audit_sha,'Same actual Q4 source and new result')
 rows=a['predictions'];by={x['target_id']:x for x in rows}
 require(len(rows)==len(by)==64 and set(by)==set(ids) and all(x['state']==core.STATE for x in rows),'Only64 real new same-W Q8 predictions')
 return by
def bind(inp,f):
 require(inp['schema']=='flow_r29_same_weight_quality_inputs_v1','New64 input contract')
 require(sha(REF27)==REF27_SHA and sha(REF28)==REF28_SHA and sha(R27/'membership_v1.json')==MEMBERS_SHA,'Immutable measured reference quality/membership')
 q27,q28=read(REF27),read(REF28);m=read(R27/'membership_v1.json');ids=[x['target_id'] for x in m['dev_members']]
 require(len(ids)==len(set(ids))==64 and [x['target_id'] for x in q27['frames']]==[x['target_id'] for x in q28['frames']]==ids,'Exact64 reference population')
 require(not set(ids)&set(m['sealed_ids']+m['H8_historical_ids']),'No sealed/H8 input')
 q4path=descriptor(q28['source_result']);q4=read(q4path);q4audit=read(descriptor(q28['execution_audit']))
 require(q4audit['status']=='PASS' and q4audit['quality_analysis_allowed'] is True and q4audit['source_results']=={'Q4':sha(q4path)},'Original complete Q4 execution')
 weights=safe(q4path.parent,q4['weights_file']);require(sha(weights)==q4['weights_sha256']==q4audit['arms']['Q4']['checkpoint_sha256'],'Actual fixed512 Q4 weight bytes')
 rp=descriptor(inp['run']);r=read(rp);ap=descriptor(inp['execution_audit']);a=read(ap)
 by=validate_audit(a,f,ids,sha(rp),q4['weights_sha256'],q4['final_state_sha256'],q28['execution_audit']['sha256'])
 require(r['schema']=='flow_r29_same_weight_q8_evaluation_v1' and r['graph_id']==GRAPH and r['status']=='SAME_Q4_WEIGHT_Q8_EVALUATION_COMPLETE' and r['state_label']==core.STATE and r['arm']=='Q8','Exact declared new same-W graph')
 require(r['model_forwards']==67 and r['completed_optimizer_steps']==0 and all(r[k] is False for k in ('training_performed','teacher_constructed','optimizer_constructed','optimizer_restored','development_GT_read','quality_computed','quality_selection')),'Evaluation-only contract')
 s=r['source_Q4_checkpoint'];require(s['sha256']==q4['weights_sha256'] and s['state_sha256']==q4['final_state_sha256']==r['initial_state_sha256']==r['state_sha256']==r['final_state_sha256'] and s['source_result_sha256']==sha(q4path) and s['audit_sha256']==q28['execution_audit']['sha256'] and s['source_arm']=='Q4' and s['source_updates']==512,'Same477 Q4 weight package across all forwards')
 require(r['sources']['members']==MEMBERS_SHA,'Same original development membership')
 # R28's Q4 references remain old measured scalar quality, with actual raw
 # archive provenance checked again. No additional old quality claim is made.
 oldby={(x['state'],x['target_id']):x for x in q4audit['predictions']}
 for row in q4['evaluations']['endpoint']['frames']:
  old=oldby[('recovered_Q4',row['target_id'])];p=safe(q4path.parent,row['prediction_file'])
  require(p.resolve()==Path(old['file']).resolve() and sha(p)==row['prediction_file_sha256']==old['file_sha256'] and row['prediction_sha256']==old['head3_sha256'],'Same-W Q4 original reference artifact')
 snapshots={x['target_id']:x for x in q27['shared_source_artifacts']};records={x['target_id']:x for x in m['records'] if x['role']=='development'}
 require(len(snapshots)==len(records)==64 and set(snapshots)==set(ids),'All64 original shared snapshots')
 shared={};evidence={}
 for row in m['dev_members']:
  tid=row['target_id'];s=snapshots[tid]
  evidence[tid]={{'input':'events','gt':'gt','mask':'mask'}[x['category']]:dict(relative_path=x['relative_path'],sha256=x['sha256']) for x in records[tid]['source_files']}
  require(s['sequence']==row['sequence'] and s['input_evidence']==evidence[tid],'Accepted original GT/source membership')
  p=Path(s['file']);p=p if p.is_absolute() else safe(R27,p);require(sha(p)==s['file_sha256'],'Original GT/mask snapshot bytes');shared[tid]=dict(path=p,record=s)
 rows=r['evaluations']['diagnostic']['frames'];require([(x['target_id'],x['sequence']) for x in rows]==[(x['target_id'],x['sequence']) for x in m['dev_members']],'Exact64 new order')
 bound={}
 for row in rows:
  tid=row['target_id'];a=by[tid];require(a['sequence']==row['sequence'] and a['input_evidence']==row['input_evidence']==evidence[tid],'Same actual new source')
  p=safe(rp.parent,row['prediction_file']);require(p.resolve()==Path(a['file']).resolve() and sha(p)==row['prediction_file_sha256']==a['file_sha256'],'Actual new archive')
  with np.load(p,allow_pickle=False) as z:
   require(z.files==['head3'],'One real quality payload');v=z['head3'];require(v.dtype==np.float32 and v.shape==(2,480,640) and np.isfinite(v).all(),'Pixel float32 fullgrid')
   require(raw.payload_sha(v)==row['prediction_sha256']==a['head3_sha256'],'Actual new payload')
  bound[tid]=dict(path=p,sha256=a['file_sha256'],payload_sha256=a['head3_sha256'])
 return q27,q28,m,shared,bound,rp,ap
def run(args):
 require(not args.output.exists(),'Fresh independent quality output');f=preflight(args.freeze);inp=read(args.inputs)
 q27,q28,m,shared,bound,rp,ap=bind(inp,f)
 # No GT numeric read before complete execution + all new64 payload binding.
 frames=[]
 for row in m['dev_members']:
  tid=row['target_id'];s=shared[tid];require(sha(s['path'])==s['record']['file_sha256'],'Shared source unchanged')
  with np.load(s['path'],allow_pickle=False) as z:
   require(set(z.files)==set(s['record']['payloads'])=={'gt','mask','voxel_nonzero_bins'}|{'region__'+r for r in core.old.REGIONS},'Exact19 old source arrays');arrays={k:z[k] for k in z.files}
  for k,v in arrays.items():
   d=s['record']['payloads'][k];require(list(v.shape)==d['shape'] and str(v.dtype)==d['dtype'] and raw.payload_sha(v)==d['sha256'],'Actual old GT/mask payload '+k)
  gt=arrays['gt'];valid=arrays['mask'];masks={r:arrays['region__'+r] for r in core.old.REGIONS}
  require(gt.dtype==np.float32 and gt.shape==(2,480,640) and np.isfinite(gt).all() and valid.dtype==np.bool_ and valid.shape==(480,640),'Original GT format')
  require(np.array_equal(masks['all_valid'],valid) and all(v.dtype==np.bool_ and v.shape==valid.shape and not np.any(v&~valid) for v in masks.values()),'Complete16 same-support subsets')
  d=bound[tid];require(sha(d['path'])==d['sha256'],'New archive unchanged')
  with np.load(d['path'],allow_pickle=False) as z:
   pred=z['head3'];require(raw.payload_sha(pred)==d['payload_sha256'],'New payload unchanged');metric=raw.measure(pred,gt,masks)
  frames.append(dict(target_id=tid,sequence=row['sequence'],states={core.STATE:metric}))
 out=core.analyze(q27,q28,frames)
 out.update(schema='flow_r29_independent_raw_quality_v1',status='RAW_METRICS_COMPLETE',graph_id=GRAPH,source_result=dict(path=str(rp),sha256=sha(rp)),execution_audit=dict(path=str(ap),sha256=sha(ap)),reference_qualities={'R27':dict(path=str(REF27),sha256=REF27_SHA),'R28':dict(path=str(REF28),sha256=REF28_SHA)},source_inputs_manifest_sha256=sha(args.inputs),io_freeze_sha256=sha(args.freeze),independently_recomputed_new_predictions=64,reused_GT_snapshot_archives=64,reused_same_W_Q4_reference_archives_byte_checked=64)
 args.output.mkdir(parents=True);p=args.output/'ACTUAL_RAW_QUALITY_V1.json';write(p,out);print(json.dumps(dict(status=out['status'],quality_sha256=sha(p),new_predictions=64,new_qualification_gate=False)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',type=Path,default=HERE/'IO_FREEZE_V1.json');p.add_argument('--source-preflight-only',action='store_true');a=p.parse_args()
 if a.source_preflight_only:preflight(a.freeze);print('R29_QUALITY_SOURCE_PREFLIGHT_PASS')
 else:run(a)
