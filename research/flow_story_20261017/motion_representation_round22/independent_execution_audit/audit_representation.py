#!/usr/bin/env python3
"""Independent native26 source/actual raw audit. Never loads GT arrays."""
import argparse,csv,hashlib,importlib.util,json,math
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;RUN=HERE.parent;DATE=RUN.parent;REPO=DATE.parents[1];OLD=REPO/'research/flow_story_20261016'
PRIOR=DATE/'current_bn_round16/pilot/independent_execution_audit/audit_pilot_v2.py'
PRIOR_SHA='1e660183e7f1c518524dcde2eb798385b2ad86be7d9ceb9eed443810580ff7f3'
ROLE=DATE/'action_domain_round20/independent_review/SELECTOR_ROLES_V1.json'
ROLE_SHA='e6f216f6b017fc2cbba262ec51c70474f342e593e1549058a2200970949e10a1'
ROOT_FREEZE='ff299b70c527dfd3f45d153e3693a5d38d1bff00cf4f4774ca6c5e5078b5ca79';INPUTS_SHA='f5e420832b14cd14f5749825fa1dc6ed555030a305cee893be0a02c3d8125b84'
RUNNER_SHA='33945f6af6d4e19d4a5d4bb65772af57324854a72b17a245b352659fc435fa39'
HELPER_SHA='7e650140ebd50a1af50aed0f46189a3c218683ad8c1e814e7c7a131326532411'
GRAPH='r22_W271482_all78current_native_fit16dev8'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
require(sha(PRIOR)==PRIOR_SHA,'Frozen original source state helpers')
spec=importlib.util.spec_from_file_location('_r22_independent_state',PRIOR);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
def source_pins():
 require(sha(RUN/'source_freeze_v1.json')==ROOT_FREEZE and sha(RUN/'inputs_v1.json')==INPUTS_SHA and sha(RUN/'implementation/run_representation.py')==RUNNER_SHA,'Exact R22 source/input')
 pins={}
 for v in read(RUN/'source_freeze_v1.json')['files']:
  remote=v['path']
  if '/research_20261017/' in remote:p=DATE/remote.split('/research_20261017/',1)[1]
  elif '/research_20261016/' in remote:p=OLD/remote.split('/research_20261016/',1)[1]
  else:raise ValueError('Unrecognized frozen source mapping')
  require(sha(p)==v['sha256'],'Actual frozen source '+str(p));pins[str(p)]=sha(p)
 pins.update(S.A14.source_pins())
 for p in (RUN/'source_freeze_v1.json',RUN/'inputs_v1.json',PRIOR,ROLE,Path(__file__),HERE/'SOURCE_REVIEW_V1.md'):pins[str(p)]=sha(p)
 return pins
def roles_and_sources(input_root,*,verify_files=True):
 require(sha(ROLE)==ROLE_SHA,'Exact identity-only frozen roles')
 d=read(ROLE);require(d['schema']=='flow_r20_next_selector_identity_roles_v1' and d['status']=='FROZEN_METADATA_ONLY','Roles schema')
 g=d['groups'];require(d['group_counts']=={'fit':16,'development':8,'sealed_confirmation':8} and set(g)==set(d['group_counts']),'Exact32 role populations')
 require(all(len(g[k])==v for k,v in d['group_counts'].items()),'Actual role counts')
 ids=[r['target_id'] for rows in g.values() for r in rows];require(len(ids)==len(set(ids))==32,'Disjoint roles')
 all_records={r['target_id']:r for r in d['records']};require(len(d['records'])==len(all_records)==32 and set(all_records)==set(ids),'Unique32 metadata records')
 for v in d['source_sha256'].values():require(sha(v['path'])==v['sha256'],'Frozen role-source metadata')
 canon={r['target_id']:r for r in read(d['source_sha256']['canonical']['path'])['records']}
 with Path(d['source_sha256']['deployment']['path']).open(newline='') as f:deployment={r['relative_path']:r for r in csv.DictReader(f)}
 selected=g['fit']+g['development'];rolemap={r['target_id']:role for role in ('fit','development') for r in g[role]}
 evidence=[];files=[]
 for row in selected:
  tid=row['target_id'];record=all_records[tid]
  require(row==canon[tid]==record['canonical'] and row['protocol_split']=='train' and record['role']==rolemap[tid],'Selected canonical/role identity')
  meta=record['accepted_array_metadata'];require(len(meta)==3 and {x['category'] for x in meta}=={'input','gt','mask'},'Exactly three accepted files')
  expected={'input':f"saved_flow_data/event_tensors/10bins/left/{row['sequence']}/{row['v2_target_filename']}",
   'gt':'saved_flow_data/gt_tensors/'+row['gt_filename'],'mask':'saved_flow_data/mask_tensors/'+row['mask_filename']}
  entry={}
  for v in meta:
   cat=v['category'];relative=v['relative_path'];require(relative==expected[cat] and v==deployment[relative],'Accepted exact array metadata')
   require(v['target']==tid and v['sequence']==row['sequence'] and v['split']=='train','Array identity/split')
   p=Path(input_root)/relative
   if verify_files:require(p.is_file() and p.stat().st_size==int(v['bytes']) and sha(p)==v['sha256'],'Actual allowed source bytes '+str(p))
   entry['events' if cat=='input' else cat]=v['sha256'];files.append({'target_id':tid,'role':rolemap[tid],'category':cat,'path':str(p),'bytes':int(v['bytes']),'sha256':v['sha256']})
  evidence.append({'target_id':tid,'selector_role':rolemap[tid],'input_evidence':entry})
 require(len(files)==len({x['path'] for x in files})==72,'Exactly72 unique allowed file accesses')
 return selected,rolemap,evidence,files
def plan(rows,roles):
 require(len(rows)==len(roles)==len({r['target_id'] for r in rows})==24 and {r['target_id'] for r in rows}==set(roles) and [roles[r['target_id']] for r in rows]==['fit']*16+['development']*8,'Fixed24 ordered roles')
 out=[]
 for i,r in enumerate(rows):
  base={'target_id':r['target_id'],'selector_role':roles[r['target_id']],'arm':'native','execution':'native','quality_endpoint':True,'endpoint_arm':'native'};out.append(base)
  if i==0:out.extend(dict(base,arm='native_repeat%d'%j,quality_endpoint=False) for j in (1,2))
 return [dict(r,forward_id=i+1) for i,r in enumerate(out)]
def check_records(records,rows,roles,bn):
 require(0<len(records)<=26,'Nonempty bounded prefix');totals=Counter()
 for r,w in zip(records,plan(rows,roles)):
  require(all(r[k]==v for k,v in w.items()),'Exact native26 role/order')
  require(r['target_execution']=={} and set(r['all12_observed_fc_rows'])==set(S.G.GEOMETRY),'No compact producer')
  n=0
  for name,shape in S.G.GEOMETRY.items():
   count=math.prod(shape[:-1]);require(r['all12_observed_fc_rows'][name]=={'fc1_rows':count,'fc2_rows':count},'All12 actual native rows');n+=count
  require(n==r['summed_fc1_rows']==r['summed_fc2_rows']==558000,'Native actual service sum')
  require(set(r['native_BN_calls'])==set(r['native_BN_inputs'])==set(bn),'All78 actual BN modules')
  for name,shape in bn.items():
   v=r['native_BN_inputs'][name];require(r['native_BN_calls'][name]==1 and len(v)==1 and v[0]['shape']==shape and v[0]['dtype']=='torch.float32','Full native BN geometry')
   require(len(v[0]['stride'])==len(shape) and all(type(x)is int and x>=0 for x in v[0]['stride']),'BN stride inventory')
   totals['BN_input_elements']+=math.prod(shape)
  require(r['state_and_BN_unchanged'] is True and r['original_modules_restored_and_hooks_removed'] is True,'Per-forward closure')
  totals.update(full_forwards=1,native=1,FC1_rows=n,FC2_rows=n,BN_calls=78,observed_FC_modules=24)
  totals['quality_endpoints' if r['quality_endpoint'] else 'determinism_replays']+=1
 if len(records)==26:require(totals['FC1_rows']==totals['FC2_rows']==14508000 and totals['BN_calls']==2028 and totals['quality_endpoints']==24 and totals['determinism_replays']==2,'Final independent counts')
 totals.update(compact=0,expanded=0,primitive_calls=0,optimizer_steps=0,calibration_forwards=0)
 return totals
def replay(a,b):
 keys={'head0','head1','head2','head3','u','branch','sn1','sn2'};require(set(a)==set(b)==keys,'Eight actual native arrays');checks={}
 for k,x in a.items():
  y=b[k];require(x.shape==y.shape and x.dtype==y.dtype,'Replay shape/dtype');dx=int(np.count_nonzero(np.ascontiguousarray(x).view(np.uint8)!=np.ascontiguousarray(y).view(np.uint8)))
  checks[k]={'bitwise_equal':dx==0,'different_bytes':dx,'shape':list(x.shape),'dtype':'torch.float32','actual_sha256':S.digest(x),'reference_sha256':S.digest(y)}
 return {'pass':all(v['bitwise_equal'] for v in checks.values()),'array_checks':checks}
def audit(path,input_root):
 sources=source_pins();state=S.source_state();rows,roles,evidence,sourcefiles=roles_and_sources(input_root);_,bn=S.A15.names_and_shapes()
 d=read(path);b=d['binding'];inp=read(RUN/'inputs_v1.json');n=len(d['forward_records']);complete=d['status']=='REPRESENTATION_NATIVE26_COMPLETE'
 require(d['schema']=='flow_r22_representation_v1' and (complete or d['status']=='INCOMPLETE_OR_NUMERICAL_FAILURE'),'Native26 complete or failed-prefix schema')
 require(d['graph_id']==b['graph_id']==GRAPH,'Native-only graph')
 require(d['full_forward_count']==n and n<=d['full_forward_attempts']<=min(n+1,26) and d['inference_executed'] is True,'Actual prefix counts')
 for k in ('training_performed','selector_training_performed','flow_specific_increment_claim','hardware_benefit_claim'):require(d[k] is False,'No unsupported action '+k)
 require(d['optimizer_steps']==d['calibration_forwards']==0,'No training/calibration')
 want={'plan':'native26','forward_plan':plan(rows,roles),'native_helper_sha256':HELPER_SHA,'selector_roles_sha256':ROLE_SHA,
  'selected_target_ids':[r['target_id'] for r in rows],'selector_role_map':roles,'confirmation_array_accesses':0,
  'inputs_sha256':INPUTS_SHA,'source_freeze_sha256':ROOT_FREEZE,'independent_admission_sha256':inp['independent_admission']['sha256'],
  'membership_sha256':inp['membership']['sha256'],'region_sha256':inp['regions']['sha256'],
  'native_forwards':26,'compact_forwards':0,'primitive_calls':0,'quality_endpoints':24,'determinism_replays':2,
  'expected_FC_rows_each':14508000,'expected_BN_calls':2028,'target':S.TARGET,'other_11_MLPs':'native','BN':'all78_current','BN_mode':'NATIVE_DENSE_CURRENT'}
 for k,v in want.items():require(b[k]==v,'Actual native/source binding '+k)
 for k,v in {'full_accepted_metadata_validated':True,'all_dataset_current_file_stat_checked':False,'allowed24_current_files_checked_separately':True,'allowed24_current_files_SHA_checked':72,'sealed_confirmation_array_paths_visited':0,'GT_array_loads':0}.items():require(b['data_preflight'][k]==v,'Restricted source verification '+k)
 require(b['precision']=='FP32; no AMP/TF32; T10/B1/full9' and b['optimizer_steps']==b['calibration_forwards']==0 and 0<b['wall_minutes']<=15,'Fixed precision/budget')
 cpu=b['checkpoint_cpu_audit']
 for k,v in {'checkpoint_sha256':S.CP_SHA,'checkpoint_step':271482,'source_state_sha256':S.SOURCE,'allcurrent_state_sha256':S.CURRENT,'source_tensor_count':711,'removed_BN_buffers':234,'allcurrent_tensor_count':477,'allcurrent_elements':54919238,'optimizer_restored':False,'external_BN_artifact_loaded':False,'source_BN_buffers_removed_before_inference':True}.items():require(cpu[k]==v,'Independent source state '+k)
 S.A14.verify_scope(d['BN_scope'],set(bn));require(d['initial_allcurrent_state_sha256']==S.CURRENT,'Independent original current477 state')
 if complete:require(n==d['full_forward_attempts']==26 and d['final_state_sha256']==S.CURRENT and all(d[k] is True for k in ('all_states_unchanged','all_hooks_removed','original_modules_restored')),'Complete native closure')
 require(d['runtime']['AMP'] is False and d['runtime']['TF32'] is False and d['data_evidence']==evidence,'Actual precision/selected source bytes')
 counts=check_records(d['forward_records'],rows,roles,bn);emap={r['target_id']:r['input_evidence'] for r in evidence};checks=[];predictions=[];first=None;seen=set()
 for r in d['forward_records']:
  require(r['input_evidence']==emap[r['target_id']],'Per-forward actual identity')
  name=r['prediction_file'];require(name==Path(name).name and name not in seen,'Unique new archive');seen.add(name)
  p=Path(path).parent/name;require(sha(p)==r['prediction_file_sha256'],'Actual archive SHA')
  keys={'head0','head1','head2','head3','u','branch','sn1','sn2'}
  with np.load(p,allow_pickle=False) as z:
   require(set(z.files)==keys and r['saved_payload_keys']==sorted(keys),'Complete eight raw tensors');arrays={k:z[k] for k in z.files}
  require(len(r['head_records'])==4 and set(r['tensor_records'])=={'u','branch','sn1','sn2'},'Exact native raw metadata')
  for i,h in enumerate(r['head_records']):
   v=arrays['head%d'%i];require(v.shape==(1,2,480,640) and v.dtype==np.float32 and np.isfinite(v).all() and h=={'shape':list(v.shape),'sha256':S.digest(v)},'Actual four flow heads')
  for key,meta in r['tensor_records'].items():
   v=arrays[key];shape=[10,1,30,40,1536 if key=='sn2' else 384]
   require(v.dtype==np.float32 and list(v.shape)==shape and np.isfinite(v).all() and meta=={'shape':shape,'dtype':'torch.float32','sha256':S.digest(v)},'Full native raw T/C capture')
   if key.startswith('sn'):require(np.isin(v,(0,1)).all(),'Binary native PSN')
  if r['forward_id']==1:first=arrays
  if r['arm'].startswith('native_repeat'):
   require(first is not None and r['target_id']==rows[0]['target_id'],'First own native reference')
   checks.append(dict(kind='native_exact_replay',forward_id=r['forward_id'],target_id=r['target_id'],arm=r['arm'],reference_forward_id=1,**replay(arrays,first)))
  counts.update(archives=1,head_payloads=4,capture_payloads=4)
  predictions.append({k:r[k] for k in ('forward_id','target_id','selector_role','arm','execution','quality_endpoint','endpoint_arm')}|{'file':str(p),'file_sha256':sha(p),'head3_sha256':S.digest(arrays['head3']),'u_sha256':S.digest(arrays['u'])})
 require(d['numerical_checks']==checks,'Independently recomputed two exact native replays');failed=[x for x in checks if not x['pass']]
 if complete:
  require(not failed and len(checks)==2 and counts['archives']==26 and counts['head_payloads']==counts['capture_payloads']==104,'Complete208 raw payloads')
  require(0<d['wall_seconds']<=900 and 0<d['max_allocated_bytes']<=d['max_reserved_bytes'],'Actual native resource report')
 elif failed:require(len(failed)==1 and failed[0]['forward_id']==n==d['full_forward_attempts'],'Immediate numerical stop')
 return {'schema':'flow_r22_representation_independent_execution_audit_v1','status':'PASS' if complete else 'PASS_PREFIX_EVIDENCE_ONLY',
  'representation_artifacts_allowed':complete,'quality_analysis_allowed':False,'admission':'NATIVE24_ARTIFACTS_ONLY_DEV_GT_REQUIRES_FROZEN_PREDICTIONS' if complete else 'NOT_ADMITTED_INCOMPLETE',
  'source_result_sha256':sha(path),'source_freeze_sha256':ROOT_FREEZE,'auditor_sha256':sha(__file__),'pre_result_freeze_sha256':sha(HERE/'PRE_RESULT_FREEZE_V1.json'),
  'source_pins':sources,'source_state':state,'counts':dict(counts),'numerical_checks':checks,'predictions':predictions,'input_evidence':evidence,
  'source_files':sourcefiles,'source_file_count':72,'sealed_array_accesses':0,'GT_array_values_loaded':False,
  'runtime':d['runtime'],'resources':{k:d.get(k) for k in ('wall_seconds','max_allocated_bytes','max_reserved_bytes')},
  'limitations':['Native-only source collection, no compact execution, selector labels, learned motion proxy or hardware benefit.',
   'Independent original PT/current477 reconstruction and72 actual source-byte hashes. Per-forward full-state closure remains frozen runtime assertions; no26 full-state snapshots replayed.',
   'All208 raw arrays and two exact8-array replays checked independently. Hidden current-BN reduction arithmetic and model computations not independently rerun.',
   'Fit/dev from backbone-training sequences, not unseen-scene confirmation. No GT values read; dev GT quality remains closed until frozen predictions; sealed8 arrays unvisited.']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Exclusive new audit result')
 for path,h in read(HERE/'PRE_RESULT_FREEZE_V1.json')['source_sha256'].items():require(sha(path)==h,'Frozen actual auditor inputs')
 out=audit(a.result,a.input_root)
 with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:out[k] for k in ('status','representation_artifacts_allowed','quality_analysis_allowed','counts','source_result_sha256')}))
if __name__=='__main__':main()
