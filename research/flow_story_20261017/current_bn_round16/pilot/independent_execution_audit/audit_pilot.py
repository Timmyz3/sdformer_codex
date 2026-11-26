#!/usr/bin/env python3
"""Independent source/count/raw numeric audit. Never imports the pilot runner.

Supports the exact fixed-plan prefix when original numerical gates stop it.
A consistent stopped prefix is NOT admission to quality analysis.
"""
import argparse, hashlib, importlib.util, json, math
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;PILOT=HERE.parent;R16=PILOT.parent
REPO=R16.parents[2];OLD=REPO/'research/flow_story_20261016'
ROOT_FREEZE='4eaf7cb81216e2bab6bcfa2235e16faa3f0947a1250cf6b93f53873394666b6c'
INPUTS_SHA='f2811e8279e35755835c1a67db36eb477f0ed22866bb6985152bdba12af341ea'
SOURCE='ad14a7d45091aa75d64df83e1a71bd89b35dfbbf49c9ab84861d5ae0aa270560'
CURRENT='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
CP_SHA='1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'
TARGET='sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp'
ARMS=('native','FF','CC')+tuple('fine_%d_%d'%(y,x) for y in range(3) for x in range(4))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(p,name,expected):
 require(sha(p)==expected,'Pinned independent helper')
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
A14=load(OLD/'bn_conversion_round14/independent_review/audit_conversion.py','_pilot_independent14','ced75fda67e2e868d85398d72204cf929b55bee86973be7b9c6cc3933ed00478')
A15=load(OLD/'selected16_bn_round15/independent_review/audit_selected16.py','_pilot_independent15','56cb16a85097b057401ff40714c9dce5b85fb701d7a124c8b8a05f4246af353f')
G=A14.G

def source_pins():
 require(sha(PILOT/'source_freeze_v1.json')==ROOT_FREEZE and sha(PILOT/'inputs_v1.json')==INPUTS_SHA,'Frozen exact execution sources/inputs')
 pinned={}
 for item in read(PILOT/'source_freeze_v1.json')['files']:
  remote=item['path']
  if '/research_20261017/' in remote:local=REPO/'research/flow_story_20261017'/remote.split('/research_20261017/',1)[1]
  elif '/research_20261016/' in remote:local=OLD/remote.split('/research_20261016/',1)[1]
  else:raise ValueError('Unknown frozen remote source mapping')
  require(sha(local)==item['sha256'],'Actual frozen source '+str(local));pinned[str(local)]=item['sha256']
 # These already-frozen old dependencies are hashed, never executed as runners.
 pinned.update(A14.source_pins())
 for p in (PILOT/'source_freeze_v1.json',PILOT/'inputs_v1.json',Path(__file__),HERE/'SOURCE_REVIEW_V1.md',HERE/'INDEPENDENT_ORCHESTRATION_CPU_V1.json'):
  pinned[str(p)]=sha(p)
 return pinned

def source_state():
 import torch
 torch.set_num_threads(2);require(not torch.cuda.is_initialized(),'CPU-only source state reconstruction')
 cp=OLD/'baseline_checkpoint_verification/source_snapshot/checkpoint_step_000271482.pt'
 require(sha(cp)==CP_SHA,'Actual original PT SHA')
 payload=torch.load(cp,map_location='cpu',weights_only=True);state=payload['model_state_dict']
 require(payload['training']['completed_optimizer_steps']==271482 and payload['schema_version']=='flow_a_native_encoding_v2_checkpoint_v1','Actual original step/schema')
 require(payload['contract_sha256']=='961f1700e76f3427c74d4e9af9799a684433ff00449b8386be6c1f9cdd75e934','Same encoding/source contract')
 require(len(state)==711 and sum(v.numel() for v in state.values())==55001588 and all(bool(torch.isfinite(v).all()) for v in state.values()),'Actual source711 inventory')
 current={k:v for k,v in state.items() if not k.endswith(('.running_mean','.running_var','.num_batches_tracked'))}
 require(len(current)==477 and sum(v.numel() for v in current.values())==54919238,'Allcurrent477 actual inventory')
 require(G.tensor_digest(state)==SOURCE and G.tensor_digest(current)==CURRENT,'Independent actual fullstate SHA')
 return {'checkpoint_sha256':CP_SHA,'original_state_sha256':SOURCE,'allcurrent_state_sha256':CURRENT,'source_tensors':711,'remaining_tensors':477,'removed_buffers':234,'state_elements':54919238,'actual_PT_read_CPU':True}

def plan(members):
 result=[]
 for i,row in enumerate(members):
  for arm in ARMS:
   result.append({'target_id':row['target_id'],'arm':arm,'execution':'native' if arm=='native' else 'compact'})
   if i==0 and arm in ('FF','CC','fine_0_0'):result.append({'target_id':row['target_id'],'arm':arm,'execution':'expanded'})
 require(len(result)==123,'Independent123')
 return [dict(r,forward_id=i+1) for i,r in enumerate(result)]

def cells(arm):
 out=[]
 for gy,y in enumerate(range(0,30,9)):
  for gx,x in enumerate(range(0,40,9)):
   fine=arm=='FF' or arm=='fine_%d_%d'%(gy,gx)
   ey,ex=min(30,y+9),min(40,x+9);stride=1 if fine else 3
   for yy in range(y,ey,stride):
    for xx in range(x,ex,stride):
     y1,x1=min(ey,yy+stride),min(ex,xx+stride);out.append([yy,y1,xx,x1,(y1-yy)*(x1-xx)])
 require(sum(r[4] for r in out)==1200,'Exact area/tail coverage')
 return out

def expected_target(r):return 12000 if r['execution'] in ('native','expanded') else 10*len(cells(r['arm']))

def continuous(a,b):
 require(a.shape==b.shape and a.dtype==b.dtype and np.isfinite(a).all() and np.isfinite(b).all(),'Finite same-shape/dtype numeric pair')
 d=np.abs(a.astype(np.float64)-b.astype(np.float64));bad=d>2e-5+2e-5*np.abs(b.astype(np.float64))
 return {'pass':not bool(bad.any()),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'outside_tolerance':int(bad.sum()),'different_elements':int((d!=0).sum()),'elements':int(a.size),'atol':2e-5,'rtol':2e-5}

def numerical(a,b):
 cs={'branch':continuous(a['branch'],b['branch'])};cs.update({'head%d'%i:continuous(a['head%d'%i],b['head%d'%i]) for i in range(4)})
 flips={sn:int(np.count_nonzero(a[sn]!=b[sn])) for sn in ('sn1','sn2')};same=bool(np.array_equal(a['u'],b['u']))
 return {'pass':same and all(v['pass'] for v in cs.values()) and not any(flips.values()),'same_u_exact':same,'continuous':cs,'PSN_flips':flips}

def check_reported(got,want):
 require(got['pass']==want['pass'] and got['same_u_exact']==want['same_u_exact'] and got['PSN_flips']==want['PSN_flips'],'Independent numeric boolean/flip reconstruction')
 require(set(got['continuous'])==set(want['continuous']),'Every branch/head comparison')
 for key,row in want['continuous'].items():
  for field,v in row.items():
   # GPU torch and CPU NumPy FP64 summations may have a final-bit mean change;
   # this audit tolerance concerns the reported diagnostic mean, never gates.
   if field=='mean_abs':require(math.isclose(got['continuous'][key][field],v,rel_tol=1e-12,abs_tol=1e-15),'Diagnostic FP64 mean agreement')
   else:require(got['continuous'][key][field]==v,'Exact numeric field '+key+'/'+field)

def audit(path):
 sources=source_pins();state=source_state();d=read(path);b=d['binding'];inputs=read(PILOT/'inputs_v1.json')
 mpath=OLD/'coverage_recovery_round13/data_protocol/manifest_v1.json';members=read(mpath)['check_members'];schedule=plan(members)
 require(len(members)==8 and len({r['target_id'] for r in members})==8,'Original8 members')
 require(d['schema']=='flow_r16_spatial_pilot_v1' and d['inference_executed'] is True and not d['training_performed'] and d['optimizer_steps']==d['calibration_forwards']==0,'Inference-only fixed pilot schema')
 require(b['plan']=='pilot123' and b['forward_plan']==schedule and b['target']==TARGET and b['other_11_MLPs']=='native' and b['BN']=='all78_current' and b['BN_mode']=='NATIVE_DENSE_CURRENT','Exact original plan/graph')
 for key,val in {'source_freeze_sha256':ROOT_FREEZE,'inputs_sha256':INPUTS_SHA,'primitive_sha256':inputs['primitive']['sha256'],'independent_admission_sha256':inputs['independent_admission']['sha256'],'membership_sha256':inputs['membership']['sha256'],'h8_sha256':inputs['h8']['sha256'],'region_sha256':inputs['regions']['sha256']}.items():require(b[key]==val,'Fixed binding '+key)
 require(sha(mpath)==b['membership_sha256'],'Actual membership bytes')
 cpu=b['checkpoint_cpu_audit'];require(cpu['checkpoint_sha256']==CP_SHA and cpu['checkpoint_step']==271482,'Same actual checkpoint')
 require(cpu['source_state_sha256']==SOURCE and cpu['allcurrent_state_sha256']==CURRENT,'Source preflight matches independently reconstructed actual tensors')
 require(b['precision']=='FP32; no AMP/TF32; T10/B1/full9' and b['optimizer_steps']==b['calibration_forwards']==0 and 0<b['wall_minutes']<=20,'Frozen precision/training/time contract')
 for key,val in {'source_tensor_count':711,'removed_BN_buffers':234,'allcurrent_tensor_count':477,'allcurrent_elements':54919238,'optimizer_restored':False,'external_BN_artifact_loaded':False,'source_BN_buffers_removed_before_inference':True}.items():require(cpu[key]==val,'Preflight inventory '+key)
 require(d['initial_allcurrent_state_sha256']==CURRENT,'Actual initial current477 digest')
 _,bn_shapes=A15.names_and_shapes();A14.verify_scope(d['BN_scope'],set(bn_shapes))
 require(d['runtime']['AMP'] is False and d['runtime']['TF32'] is False,'Actual FP32 no AMP/TF32')
 prior_path=OLD/'bn_conversion_round14/run_v2/results.json'
 require(sha(prior_path)=='c9ead13e411512f570fc6ca926fcfaaa67c2f93e04c700bdd7799c85fcb98404','Pinned cached current input evidence')
 prior=read(prior_path);evidence={r['target_id']:r['input_evidence'] for r in prior['checkpoints']['271482']['evaluations']['endpoint']['frames']}
 require(d['data_evidence']==[dict(target_id=r['target_id'],input_evidence=evidence[r['target_id']]) for r in members],'Same all8 input bytes')
 records=d['forward_records'];n=len(records);require(0<n<=123 and d['full_forward_count']==n and d['full_forward_attempts'] in (n,n+1),'Actual attempts/completed prefix')
 totals=Counter();prediction_rows=[];native_payload={};previous=None;numerics=[];seen_files=set();head_native_cache=[]
 for rec,wanted in zip(records,schedule):
  require(all(rec[k]==v for k,v in wanted.items()),'Unchanged ordered prefix')
  fid=rec['forward_id'];tid=rec['target_id'];mode=rec['execution'];arm=rec['arm']
  require(rec['input_evidence']==evidence[tid] and rec['state_and_BN_unchanged'] is True and rec['original_modules_restored_and_hooks_removed'] is True,'Source/closure record')
  require(set(rec['all12_observed_fc_rows'])==set(G.GEOMETRY),'All12 original FC observers')
  expected_sum=0
  for name,shape in G.GEOMETRY.items():
   expect=expected_target(rec) if name==TARGET else math.prod(shape[:-1]);expected_sum+=expect
   require(rec['all12_observed_fc_rows'][name]=={'fc1_rows':expect,'fc2_rows':expect},'Actual FC rows with fullT/tail '+name)
  require(rec['summed_fc1_rows']==rec['summed_fc2_rows']==expected_sum,'Recomputed per-forward rows')
  totals.update(FC1_rows=expected_sum,FC2_rows=expected_sum,observed_FC_modules=24,BN_calls=78,full_forwards=1)
  totals['expanded' if mode=='expanded' else 'actual']+=1;totals['compact_target_calls']+=mode=='compact'
  require(set(rec['native_BN_calls'])==set(rec['native_BN_inputs'])==set(bn_shapes),'All78 native BN coverage')
  for name,shape in bn_shapes.items():
   observed=rec['native_BN_inputs'][name]
   require(rec['native_BN_calls'][name]==1 and len(observed)==1 and observed[0]['shape']==shape and observed[0]['dtype']=='torch.float32','Actual full-domain BN geometry '+name)
   require(len(observed[0]['stride'])==len(shape) and all(type(s) is int and s>=0 for s in observed[0]['stride']),'Actual recorded native stride')
   totals['BN_input_elements']+=math.prod(shape)
  local=rec['target_execution']
  if mode=='compact':
   expected_cells=cells(arm);r=len(expected_cells);n1=10*r*1536;n2=10*r*384
   require(local['cells']==expected_cells and local['areas']==[v[4] for v in expected_cells],'Actual fullT true-tail representative cover')
   required={'representatives':r,'statistics_barriers':2,'FC1_retained_elements':n1,'FC2_retained_elements':n2,'full_T_and_all_channels':True,'native_MLP_shortcut':False,'BN_mode':'NATIVE_DENSE_CURRENT','full_domain_BN_elements':23040000,'expand_written_elements':23040000,'BN_output_dense_elements':23040000,'production_BN1_gather_read_elements':n1,'production_BN1_gather_written_elements':n1,'diagnostic_BN2_gather_read_elements':n2,'cell_exact_audit_read_elements':46080000,'each_cell_BN1_and_BN2_bitwise_equal':True}
   for k,v in required.items():require(local[k]==v,'Full-domain V2 actual cost/geometry '+k)
   for k in ('full_domain_BN_elements','expand_written_elements','BN_output_dense_elements','production_BN1_gather_read_elements','production_BN1_gather_written_elements','diagnostic_BN2_gather_read_elements','cell_exact_audit_read_elements'):totals[k]+=local[k]
  filename=rec['prediction_file'];require(filename==Path(filename).name and filename not in seen_files,'Unique sibling prediction archive');seen_files.add(filename)
  archive=Path(path).parent/filename;require(sha(archive)==rec['prediction_file_sha256'],'Actual lossless archive bytes')
  captures=fid<=7 or arm in ('native','FF');wanted_keys={'head%d'%i for i in range(4)}|({'u','branch','sn1','sn2'} if captures else set())
  with np.load(archive,allow_pickle=False) as raw:
   require(set(raw.files)==wanted_keys,'Exact saved payload inventory');arrays={k:raw[k] for k in raw.files}
  require(len(rec['head_records'])==4 and set(rec['tensor_records'])=={'u','branch','sn1','sn2'},'Complete metadata per inference')
  for i,head in enumerate(rec['head_records']):
   arr=arrays['head%d'%i];require(arr.dtype==np.float32 and arr.shape==(1,2,480,640) and np.isfinite(arr).all() and head=={'shape':list(arr.shape),'sha256':digest(arr)},'Actual full flow head/hash')
  for key,meta in rec['tensor_records'].items():
   shape=[10,1,30,40,1536 if key=='sn2' else 384]
   require(meta['shape']==shape and G.hexdigest(meta['sha256']),'Full expanded target metadata')
   if captures:
    arr=arrays[key];require(arr.dtype==np.float32 and list(arr.shape)==shape and np.isfinite(arr).all() and digest(arr)==meta['sha256'],'Actual target raw capture/hash')
    if key.startswith('sn'):require(np.isin(arr,(0,1)).all(),'Native binary PSN output')
  if mode=='native':native_payload[tid]=arrays
  else:require(rec['tensor_records']['u']['sha256']==digest(native_payload[tid]['u']),'Exact predecision target input per frame')
  if mode=='expanded':
   require(previous is not None and previous[0]['arm']==arm and previous[0]['execution']=='compact' and previous[0]['target_id']==tid,'Immediate same-mask pair')
   numeric=dict(target_id=tid,kind='compact_vs_expanded',arm=arm,**numerical(previous[1],arrays));numerics.append(numeric)
  elif arm=='FF':numerics.append(dict(target_id=tid,kind='allfine_vs_native',arm='FF',**numerical(arrays,native_payload[tid])))
  previous=(rec,arrays)
  totals.update(archives=1,head_payloads=4,capture_payloads=4 if captures else 0)
  prediction_rows.append({'forward_id':fid,'target_id':tid,'arm':arm,'execution':mode,'file':str(archive),'file_sha256':sha(archive),'head3_sha256':digest(arrays['head3'])})
 require(len(d['numerical_checks'])==len(numerics),'Actual numeric pair count')
 for actual,independent in zip(d['numerical_checks'],numerics):
  require(all(actual[k]==independent[k] for k in ('target_id','kind','arm')),'Numeric record pair identity');check_reported(actual,independent)
 failed=[v for v in numerics if not v['pass']]
 complete=n==123 and d['status']=='SPATIAL_PILOT123_COMPLETE'
 if complete:
  require(not failed and d['full_forward_attempts']==123 and d['final_state_sha256']==CURRENT and all(d[k] is True for k in ('all_states_unchanged','all_hooks_removed','original_modules_restored')),'Complete numeric/state gate')
  for k,v in {'FC1_rows':67600720,'FC2_rows':67600720,'BN_calls':9594,'compact_target_calls':112,'archives':123,'head_payloads':492,'capture_payloads':84,'actual':120,'expanded':3}.items():require(totals[k]==v,'Independent complete budget '+k)
  require(len(numerics)==11,'Full8fine+3reference comparisons')
 else:
  require(d['status']=='INCOMPLETE_OR_NUMERICAL_FAILURE','Explicit stopped-run status')
  require(bool(failed) and failed[-1]==numerics[-1] and n==numerics[-1].get('forward_id',n),'Failed original numeric gate retained')
  require(records[-1]['execution']=='expanded' or records[-1]['arm']=='FF','Stop immediately after checked pair')
  require(len(failed)==1 and 'Predeclared numerical contract failed' in d['error'],'No post-failure continuation/alternative failure claim')
 totals.update(optimizer_steps=0,calibration_forwards=0)
 return {'schema':'flow_r16_pilot_independent_execution_audit_v1','status':'PASS' if complete else 'PASS_PREFIX_EVIDENCE_ONLY','admission':'ADMITTED_FIXED_PILOT_QUALITY_ONLY' if complete else 'NOT_ADMITTED_NUMERICAL_FAILURE','quality_analysis_allowed':complete,
  'source_result_sha256':sha(path),'source_freeze_sha256':ROOT_FREEZE,'auditor_sha256':sha(__file__),'pre_result_freeze_sha256':sha(HERE/'AUDIT_FREEZE_BEFORE_RAW_READ_V1.json'),
  'source_pins':sources,'source_state':state,'counts':dict(totals),'numerical_checks':numerics,'predictions':prediction_rows,'input_evidence':d['data_evidence'],'runtime':d['runtime'],'saved_full_plan_budget':123,'actual_plan_prefix_length':n,
  'audit_freeze_timing':'Auditor source frozen after root reported the original numerical stop, before this auditor opened any actual result/archive. Original numerical thresholds and plan were already frozen before execution.',
  'limitations':['No new model inference or GT/AEE reading. Saved raw branch/heads/spikes independently recomputed only where retained by the original protocol.',
   'Actual original PT state independently reconstructed. Per-inference no-state/no-hook-change evidence is the frozen runner closure and digest checks; intermediate full model states were not saved for independent replay.',
   'BN/FC hook records are independently recounted against pinned actual geometry; this is execution evidence, not RTL/DC or cycle measurement.',
   'A valid failed prefix preserves the failure. It never grants quality analysis or permission to extend the fixed plan.']}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 require(not a.output.exists(),'Independent exclusive output')
 freeze=read(HERE/'AUDIT_FREEZE_BEFORE_RAW_READ_V1.json')
 for name,digest_ in freeze['source_sha256'].items():require(sha(name)==digest_,'Actual frozen auditor source')
 result=audit(a.result)
 with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:result[k] for k in ('status','admission','quality_analysis_allowed','counts','source_result_sha256')}))
if __name__=='__main__':main()
