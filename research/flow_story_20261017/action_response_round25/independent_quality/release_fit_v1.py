#!/usr/bin/env python3
"""R25 audited fresh-graph fit-only gain release. Never loads dev GT/actions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;RUN=HERE.parent;DATE=RUN.parent;REPO=DATE.parents[1]
ROLES=DATE/'action_domain_round20/independent_review/SELECTOR_ROLES_V1.json'
ROLE_SHA='e6f216f6b017fc2cbba262ec51c70474f342e593e1549058a2200970949e10a1'
GRAPH='r25_W271482_all78current_s2b2_fixedH1_two_slot_T2_fit16dev8'
TILES=tuple((r,c) for r in range(3) for c in range(4));ARMS=('native','FF','T2')+tuple(f'fine_{r}_{c}' for r,c in TILES)
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def payload(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def write(p,d):
 p=Path(p);require(not p.exists(),'Fresh artifact');p.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n')
def preflight(path,digest):
 require(sha(path)==digest,'Actual independent release freeze');f=read(path)
 require(f['schema']=='flow_r25_independent_fit_release_freeze_v1','Release-only schema')
 for n,h in f['own_files'].items():require(sha(HERE/n)==h,'Own fixed source '+n)
 for k,v in f['external_sources'].items():require(sha(REPO/v['relative_path'])==v['sha256'],'External fixed source '+k)
 require(sha(ROLES)==ROLE_SHA,'Original32 roles')
 return f

def expected_plan(members):
 out=[]
 for i,m in enumerate(members):
  for arm in ARMS:
   out.append((m['target_id'],'fit' if i<16 else 'development',arm,True,arm))
   if i==0 and arm=='T2':out.extend((m['target_id'],'fit',f'T2_repeat{j}',False,'T2') for j in (1,2))
 require(len(out)==362,'Fixed362 plan');return out

def source_admission(f,result_dir,input_root):
 """Metadata and complete source/archive byte hashes only; no np.load."""
 rp=Path(result_dir)/'results.json';d=read(rp);ap=REPO/f['actual_execution_audit_relative_path'];a=read(ap);e=f['external_sources']
 require(d['schema']=='flow_r25_action_response_collection_v1' and d['status']=='ACTION_RESPONSE_COLLECTION362_COMPLETE','Complete new actual graph')
 require(d['graph_id']==d['binding']['graph_id']==GRAPH,'Exact new R25 graph')
 require(a['schema']=='flow_r25_action_response_independent_execution_audit_v1' and a['status']=='PASS' and a['quality_analysis_allowed'] is True,'Complete independent actual execution audit')
 require(a['source_result_sha256']==sha(rp) and a['source_freeze_sha256']==e['runner_source_freeze']['sha256']==d['binding']['source_freeze_sha256'],'Exact audited execution bytes')
 require(a['auditor_sha256']==e['execution_auditor']['sha256'] and a['pre_result_freeze_sha256']==e['execution_audit_freeze']['sha256'],'Frozen actual auditor')
 b=d['binding'];require(b['inputs_sha256']==e['runner_inputs']['sha256'] and b['independent_admission_sha256']==e['source_admission']['sha256'],'Actual input/admission')
 require(b['selector_roles_sha256']==ROLE_SHA and b['background']=='T2' and b['BN']=='all78_current' and b['other_11_MLPs']=='native' and b['confirmation_array_accesses']==0,'Roles/current state/target scope')
 require(d['full_forward_count']==d['full_forward_attempts']==362 and d['training_performed'] is False and d['optimizer_steps']==d['calibration_forwards']==0,'Single bounded untrained collection')
 counts=dict(full_forwards=362,native=24,compact=338,expanded=0,quality_endpoints=360,determinism_replays=2,FC1_rows=199803600,FC2_rows=199803600,BN_calls=28236,archives=362,head_payloads=515,capture_payloads=258,optimizer_steps=0,calibration_forwards=0)
 require(all(a['counts'][k]==v for k,v in counts.items()),'Complete actual endpoint/raw budget')
 checks=a['numerical_checks'];require(len(checks)==26 and all(v['pass'] for v in checks) and sum(v['kind']=='allfine_vs_native' for v in checks)==24 and sum(v['kind']=='T2_exact_replay' for v in checks)==2,'24 original FF anchors and2 exact replay')
 roles=read(ROLES);members=roles['groups']['fit']+roles['groups']['development'];meta={r['target_id']:r for r in roles['records']};ids=[r['target_id'] for r in members]
 require(b['selected_target_ids']==ids and b['selector_role_map']=={t:'fit' if i<16 else 'development' for i,t in enumerate(ids)},'Canonical exact24 identity/role mapping')
 paths={};evidence={}
 for row in members:
  tid=row['target_id'];paths[tid]={};evidence[tid]={}
  for item in meta[tid]['accepted_array_metadata']:
   kind='events' if item['category']=='input' else item['category'];relative=Path(item['relative_path']);require(not relative.is_absolute() and '..' not in relative.parts,'Safe input relative path')
   p=Path(input_root)/relative;require(p.stat().st_size==int(item['bytes']) and sha(p)==item['sha256'],'Selected original source bytes')
   paths[tid][kind]=p;evidence[tid][kind]=item['sha256']
 plan=expected_plan(members);records=d['forward_records'];require(len(records)==len(a['predictions'])==362,'All original archives')
 endpoints={}
 for fid,(r,ir,want) in enumerate(zip(records,a['predictions'],plan),1):
  keys=('target_id','selector_role','arm','quality_endpoint','endpoint_arm')
  require(tuple(r[k] for k in keys)==tuple(ir[k] for k in keys)==want and r['forward_id']==ir['forward_id']==fid,'Actual fixed roles/order/replays')
  require(r['execution']==ir['execution']==('native' if r['arm']=='native' else 'compact'),'Native versus compact role')
  require(r['input_evidence']==evidence[r['target_id']],'Actual GT/event/mask source provenance')
  name=Path(r['prediction_file']);require(len(name.parts)==1 and name.suffix=='.npz' and name.name==Path(ir['file']).name,'Safe original archive basename')
  p=Path(result_dir)/name;require(sha(p)==r['prediction_file_sha256']==ir['file_sha256'],'Complete actual archive bytes')
  require(ir['head3_sha256']==r['head_records'][3]['sha256'],'Audited final head')
  if r['quality_endpoint']:
   key=(r['target_id'],r['arm']);require(key not in endpoints,'Unique primary endpoint');endpoints[key]=(p,r)
 require(len(endpoints)==360,'All360 unique endpoints')
 return dict(result=d,audit=a,result_path=rp,audit_path=ap,members=members,paths=paths,evidence=evidence,endpoints=endpoints)

def head(state,tid,arm):
 p,r=state['endpoints'][(tid,arm)]
 require(sha(p)==r['prediction_file_sha256'],'Unchanged prediction archive')
 with np.load(p,allow_pickle=False) as z:
  a=z['head3'];require(a.shape==(1,2,480,640) and a.dtype==np.float32 and np.isfinite(a).all() and payload(a)==r['head_records'][3]['sha256'],'Finite actual final head3');return a[0].copy()

def all_valid_metric(prediction,gt,valid):
 require(prediction.shape==gt.shape==(2,480,640) and valid.shape==(480,640) and valid.dtype==np.bool_,'Original full output support')
 require(np.isfinite(prediction).all() and np.isfinite(gt[:,valid]).all(),'Finite full prediction/validGT')
 n=int(valid.sum());require(n>0,'Positive fit fullGT support')
 dx=prediction[0].astype(np.float64)-gt[0].astype(np.float64);dy=prediction[1].astype(np.float64)-gt[1].astype(np.float64)
 error=float(np.sqrt(dx*dx+dy*dy)[valid].sum(dtype=np.float64));return {'valid_pixels':n,'error_sum':error,'AEE':error/n}

def release(state,freeze_sha):
 members=[];relative={};details={}
 for i,row in enumerate(state['members']):
  tid=row['target_id'];out=dict(target_id=tid,sequence=row['sequence'],role='fit' if i<16 else 'development',native_archive=pin(state['endpoints'][(tid,'native')][0]))
  if i<16:
   out.update(background_archive=pin(state['endpoints'][(tid,'T2')][0]),action_archives=[pin(state['endpoints'][(tid,f'fine_{r}_{c}')][0]) for r,c in TILES])
   data={}
   for kind in ('gt','mask'):
    p=state['paths'][tid][kind];require(sha(p)==state['evidence'][tid][kind],'Original fit source unchanged');data[kind]=np.load(p,allow_pickle=False)
   require(data['gt'].dtype==np.float32 and data['mask'].dtype==np.bool_,'Original fit dtype')
   metrics={arm:all_valid_metric(head(state,tid,arm),data['gt'],data['mask']) for arm in ('native','T2')+ARMS[3:]}
   denominator=max(metrics['native']['AEE'],1e-6);gain=[metrics['T2']['AEE']-metrics[f'fine_{r}_{c}']['AEE'] for r,c in TILES]
   relative[tid]=[v/denominator for v in gain]
   details[tid]={'metrics':metrics,'native_denominator':denominator,'signed_gain':gain,'input_sha256':state['evidence'][tid],'action_head3_sha256':{a:state['endpoints'][(tid,a)][1]['head_records'][3]['sha256'] for a in ('native','T2')+ARMS[3:]}}
  members.append(out)
 labels=dict(schema='flow_r25_fit_gain_labels_v1',graph_id=GRAPH,background='T2',source_result_sha256=sha(state['result_path']),execution_audit_sha256=sha(state['audit_path']),tile_order=[list(t) for t in TILES],normalization='max(native_allGT_AEE,1e-6)',signed_relative_gain_by_id=relative,fit_details_by_id=details,fit_frame_count=16,fit_action_count=192,development_GT_values_loaded=False,development_action_values_loaded=False,sealed_array_accesses=0,independent_release_freeze_sha256=freeze_sha)
 release=dict(schema='flow_r25_fit_only_release_v1',graph_id=GRAPH,source_result_sha256=sha(state['result_path']),execution_audit_sha256=sha(state['audit_path']),members=members,role_sha256=ROLE_SHA,independent_release_freeze_sha256=freeze_sha,scope='Fit-only originalGT gain release; dev previously exposed by q but new action quality remains deferred until frozen predictions; sealed8 untouched')
 return labels,release

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True);p.add_argument('--result-dir',type=Path);p.add_argument('--input-root',type=Path);p.add_argument('--labels-output',type=Path);p.add_argument('--release-output',type=Path);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();f=preflight(a.freeze,a.freeze_sha256)
 if a.preflight_only:print('R25_FIT_RELEASE_SOURCE_PREFLIGHT_PASS')
 else:
  require(all((a.result_dir,a.input_root,a.labels_output,a.release_output)) and not a.labels_output.exists() and not a.release_output.exists(),'Fresh bound fit-only release')
  s=source_admission(f,a.result_dir,a.input_root);labels,metadata=release(s,a.freeze_sha256);write(a.labels_output,labels);write(a.release_output,metadata);print(json.dumps({'fit_labels_sha256':sha(a.labels_output),'fit_release_sha256':sha(a.release_output),'fit_labels':192,'development_GT_loaded':False}))
