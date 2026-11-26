#!/usr/bin/env python3
"""R17 independent exact-source/120-execution/raw numeric audit. No GT reading."""
import argparse,hashlib,importlib.util,json,math
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;RUN=HERE.parent;DATE=RUN.parent;REPO=DATE.parents[1]
OLD=REPO/'research/flow_story_20261016'
PRIOR=DATE/'current_bn_round16/pilot/independent_execution_audit/audit_pilot_v2.py'
PRIOR_SHA='1e660183e7f1c518524dcde2eb798385b2ad86be7d9ceb9eed443810580ff7f3'
ROOT_FREEZE='d7e1834fd6e8a19990bc0726f2db29352e959f646ed7d7c67d3bb13cc551ff00'
INPUTS_SHA='9cb7525278a5e424c9b7dd03f34bf8e224e6e76dcbea3e6237d5e9146d33fc38'
RUNNER_SHA='d720b4d0d75dde347754aa598a3a819057f68446c1d9b1d11899e95ffa52565e'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
require(sha(PRIOR)==PRIOR_SHA,'Frozen independent source/numeric helpers')
spec=importlib.util.spec_from_file_location('_r17_independent_helpers',PRIOR);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
ARMS=('native','FF','CC')+tuple('fine_%d_%d'%(y,x) for y in range(3) for x in range(4))

def plan(members):
 rows=[dict(target_id=r['target_id'],arm=arm,execution='native' if arm=='native' else 'expanded') for r in members for arm in ARMS]
 require(len(members)==8 and len(rows)==120,'Independent exact H8x15 budget')
 return [dict(r,forward_id=i+1) for i,r in enumerate(rows)]

def source_pins():
 require(sha(RUN/'source_freeze_v1.json')==ROOT_FREEZE and sha(RUN/'inputs_v1.json')==INPUTS_SHA and sha(RUN/'run_dense_reference.py')==RUNNER_SHA,'Exact new R17 source/input')
 pins={}
 for item in read(RUN/'source_freeze_v1.json')['files']:
  remote=item['path']
  if '/research_20261017/' in remote:p=DATE/remote.split('/research_20261017/',1)[1]
  elif '/research_20261016/' in remote:p=OLD/remote.split('/research_20261016/',1)[1]
  else:raise ValueError('Unrecognized remote source mapping')
  require(sha(p)==item['sha256'],'Actual frozen source '+str(p));pins[str(p)]=sha(p)
 pins.update(S.A14.source_pins())
 for p in (RUN/'source_freeze_v1.json',RUN/'inputs_v1.json',PRIOR,Path(__file__),HERE/'SOURCE_REVIEW_V1.md'):
  pins[str(p)]=sha(p)
 return pins

def check_count_records(records,members,bn_shapes):
 expected=plan(members);require(len(records)==120,'Only complete120 grants quality interpretation')
 totals=Counter()
 for r,want in zip(records,expected):
  require(all(r[k]==v for k,v in want.items()),'Fixed all-new120 order')
  require(set(r['all12_observed_fc_rows'])==set(S.G.GEOMETRY),'All12 FC observers')
  rows=0
  for name,shape in S.G.GEOMETRY.items():
   nr=math.prod(shape[:-1]);require(r['all12_observed_fc_rows'][name]=={'fc1_rows':nr,'fc2_rows':nr},'Native full-row FC producer '+name);rows+=nr
  require(rows==558000 and r['summed_fc1_rows']==r['summed_fc2_rows']==rows,'Actual dense per-forward rows')
  require(set(r['native_BN_calls'])==set(r['native_BN_inputs'])==set(bn_shapes),'All78 native BN observers')
  for name,shape in bn_shapes.items():
   entries=r['native_BN_inputs'][name]
   require(r['native_BN_calls'][name]==1 and len(entries)==1 and entries[0]['shape']==shape and entries[0]['dtype']=='torch.float32','All78 native BN full-domain input '+name)
   require(len(entries[0]['stride'])==len(shape) and all(type(s) is int and s>=0 for s in entries[0]['stride']),'Actual stride inventory')
   totals['BN_input_elements']+=math.prod(shape)
  require(r['state_and_BN_unchanged'] is True and r['original_modules_restored_and_hooks_removed'] is True,'Actual per-forward closure evidence')
  ex=r['target_execution']
  if r['execution']=='native':require(ex=={},'Native action no wrapper')
  else:require(set(ex)=={'expanded_input_sha256','reference_dense_only'} and ex['reference_dense_only'] is True and S.G.hexdigest(ex['expanded_input_sha256']),'Only frozen dense input intervention')
  totals.update(full_forwards=1,FC1_rows=rows,FC2_rows=rows,BN_calls=78,observed_FC_modules=24)
  totals[r['execution']]+=1
 require(totals['FC1_rows']==totals['FC2_rows']==66960000 and totals['BN_calls']==9360 and totals['native']==8 and totals['expanded']==112,'Independent whole-plan FC/BN sums')
 totals.update(compact=0,compact_target_calls=0,optimizer_steps=0,calibration_forwards=0)
 return totals

def audit(path):
 sources=source_pins();state=S.source_state();mpath,members,evidence=S.h8_inputs();_,bn_shapes=S.A15.names_and_shapes()
 d=read(path);b=d['binding'];inputs=read(RUN/'inputs_v1.json')
 require(d['schema']=='flow_r17_dense_reference_v1' and d['status']=='DENSE_REFERENCE120_COMPLETE','New R17 complete-only schema')
 require(d['full_forward_count']==d['full_forward_attempts']==120 and d['inference_executed'] is True and d['training_performed'] is False,'Exactly120 newly executed forwards')
 require(d['optimizer_steps']==d['calibration_forwards']==0 and d['hardware_benefit_claim'] is False and d['flow_specific_increment_claim'] is False,'No training/calibration/unsupported benefit')
 require(b['plan']=='dense120' and b['forward_plan']==plan(members) and b['target']==S.TARGET and b['other_11_MLPs']=='native' and b['BN']=='all78_current' and b['BN_mode']=='REFERENCE_DENSE_CURRENT' and b['compact_forwards']==0,'Fixed dense-only graph/action scope')
 for k,v in {'inputs_sha256':INPUTS_SHA,'source_freeze_sha256':ROOT_FREEZE,'primitive_sha256':'ad85fab7c0de64bad5b1fca15baca1af3347e32e226fb17ec18487485cc837ec','independent_admission_sha256':inputs['independent_admission']['sha256'],'membership_sha256':inputs['membership']['sha256'],'h8_sha256':inputs['h8']['sha256'],'region_sha256':inputs['regions']['sha256']}.items():require(b[k]==v,'Actual source binding '+k)
 require(b['precision']=='FP32; no AMP/TF32; T10/B1/full9' and b['optimizer_steps']==b['calibration_forwards']==0 and 0<b['wall_minutes']<=20,'Frozen compute/effort contract')
 cpu=b['checkpoint_cpu_audit']
 for k,v in {'checkpoint_sha256':S.CP_SHA,'checkpoint_step':271482,'source_state_sha256':S.SOURCE,'allcurrent_state_sha256':S.CURRENT,'source_tensor_count':711,'removed_BN_buffers':234,'allcurrent_tensor_count':477,'allcurrent_elements':54919238,'optimizer_restored':False,'external_BN_artifact_loaded':False,'source_BN_buffers_removed_before_inference':True}.items():require(cpu[k]==v,'Actual source477 inventory '+k)
 S.A14.verify_scope(d['BN_scope'],set(bn_shapes))
 require(d['initial_allcurrent_state_sha256']==d['final_state_sha256']==S.CURRENT and all(d[k] is True for k in ('all_states_unchanged','all_hooks_removed','original_modules_restored')),'Original current477 state and closure')
 require(d['runtime']['AMP'] is False and d['runtime']['TF32'] is False,'FP32 runtime')
 require(d['data_evidence']==[dict(target_id=r['target_id'],input_evidence=evidence[r['target_id']]) for r in members],'All original H8 input evidence')
 totals=check_count_records(d['forward_records'],members,bn_shapes);native=None;predictions=[];checks=[];seen=set()
 for rec in d['forward_records']:
  tid,arm,mode=rec['target_id'],rec['arm'],rec['execution'];require(rec['input_evidence']==evidence[tid],'Actual per-action event/GT-source identity')
  name=rec['prediction_file'];require(name==Path(name).name and name not in seen,'Unique new actual archive');seen.add(name)
  archive=Path(path).parent/name;require(sha(archive)==rec['prediction_file_sha256'],'Actual new archive SHA')
  captures=arm in ('native','FF');keys={'head%d'%i for i in range(4)}|({'u','branch','sn1','sn2'} if captures else set())
  with np.load(archive,allow_pickle=False) as z:
   require(set(z.files)==keys,'Exact new saved payload set');arrays={k:z[k] for k in z.files}
  require(len(rec['head_records'])==4 and set(rec['tensor_records'])=={'u','branch','sn1','sn2'},'All actual output metadata')
  for i,h in enumerate(rec['head_records']):
   arr=arrays['head%d'%i];require(arr.shape==(1,2,480,640) and arr.dtype==np.float32 and np.isfinite(arr).all() and h=={'shape':list(arr.shape),'sha256':S.digest(arr)},'Actual full native head/bytes')
  for key,meta in rec['tensor_records'].items():
   shape=[10,1,30,40,1536 if key=='sn2' else 384];require(meta['shape']==shape and S.G.hexdigest(meta['sha256']),'FullT target capture metadata')
   if captures:
    arr=arrays[key];require(list(arr.shape)==shape and arr.dtype==np.float32 and np.isfinite(arr).all() and S.digest(arr)==meta['sha256'],'Actual saved input/branch/spike bytes')
    if key.startswith('sn'):require(np.isin(arr,(0,1)).all(),'Native binary spikes')
  if mode=='native':native=(tid,arrays)
  else:
   require(native is not None and tid==native[0] and rec['tensor_records']['u']['sha256']==S.digest(native[1]['u']),'Every action same true predecision input')
   if arm=='FF':
    require(rec['target_execution']['expanded_input_sha256']==S.digest(arrays['u']),'FF dense input is original u')
    measured=dict(forward_id=rec['forward_id'],target_id=tid,kind='allfine_vs_native',arm='FF',**S.numerical(arrays,native[1]));require(measured['pass'],'Original allfine numerical gate');checks.append(measured)
  totals.update(archives=1,head_payloads=4,capture_payloads=4 if captures else 0)
  predictions.append({'forward_id':rec['forward_id'],'target_id':tid,'arm':arm,'execution':mode,'file':str(archive),'file_sha256':sha(archive),'head3_sha256':S.digest(arrays['head3'])})
 require(len(checks)==len(d['numerical_checks'])==8,'Independent all8 numerical comparisons')
 for actual,expected in zip(d['numerical_checks'],checks):
  require(all(actual[k]==expected[k] for k in ('target_id','kind','arm')),'Original numerical pair identity');S.check_reported(actual,expected)
 require(totals['archives']==120 and totals['head_payloads']==480 and totals['capture_payloads']==64,'Actual complete544 payload inventory')
 require(0<d['wall_seconds']<=1260 and 0<d['max_allocated_bytes']<=d['max_reserved_bytes'],'Actual finite engineering resource receipt')
 return {'schema':'flow_r17_dense_reference_independent_execution_audit_v1','status':'PASS','quality_analysis_allowed':True,'admission':'ADMITTED_FIXED_DENSE_ACTION_SCREEN_ONLY','source_result_sha256':sha(path),'source_freeze_sha256':ROOT_FREEZE,'auditor_sha256':sha(__file__),'pre_result_freeze_sha256':sha(HERE/'PRE_RESULT_FREEZE_V1.json'),'source_pins':sources,'source_state':state,'counts':dict(totals),'numerical_checks':checks,'predictions':predictions,'input_evidence':d['data_evidence'],'runtime':d['runtime'],'resources':{k:d[k] for k in ('wall_seconds','max_allocated_bytes','max_reserved_bytes')},
  'limitations':['Only this new120 set is audited; no R16 old prediction or quality result is imported. All FC rows remain native, so there is no hardware speedup or upper-bound claim.',
   'Real original PT state reconstructed on CPU; intermediate full model states are not retained. Per-forward closure is evidence from frozen assertions/digests, not independent model replay.',
   'Raw saved heads and all8 native/FF input/branch/spikes independently hashed and numerically remeasured. Other actions retain target hashes but not full target arrays.',
   'No GT arrays/AEE read here, no numerical neural forward, no GPU/RTL/DC. Training-side H8 is exploratory and not independent confirmation.']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New exclusive audit output')
 for file,digest in read(HERE/'PRE_RESULT_FREEZE_V1.json')['source_sha256'].items():require(sha(file)==digest,'Frozen auditor source')
 out=audit(a.result)
 with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:out[k] for k in ('status','quality_analysis_allowed','source_result_sha256','counts')}))
if __name__=='__main__':main()
