#!/usr/bin/env python3
"""Independent CPU execution audit for R14 all78-current/native16; no runner/AEE."""
import argparse
from collections import Counter
import hashlib,importlib.util,json,math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;RUN=HERE.parent;DATE=RUN.parent;REPO=DATE.parent.parent
R13=DATE/'coverage_recovery_round13';K=DATE/'later_checkpoint_deployment'
OLD=R13/'independent_execution_audit/audit_execution.py'
STEPS=('135930','271482')
CP={'135930':'d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30','271482':'1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'}
SOURCE={'135930':'2bd43221e02d376df89e2b90ddce5a14f05301f45dd5ce871304b2a71b8c2831','271482':'ad14a7d45091aa75d64df83e1a71bd89b35dfbbf49c9ab84861d5ae0aa270560'}
CURRENT={'135930':'8d4b5141731b0bed7bd12da67217ee5f68ad72288d03fd6c56726d8b6acd9ef5','271482':'b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'}
PT={'135930':R13/'source_snapshot/checkpoint_step_000135930.pt','271482':DATE/'baseline_checkpoint_verification/source_snapshot/checkpoint_step_000271482.pt'}
CACHE={'135930':R13/'raw/experiments_v1/diversity/recover/native/results.json','271482':K/'run_v1/results.json'}
CACHE_SHA={'135930':'254fb4ec5f18d30e4245b4c277ee1c5fb53beaf7fb3cf42cf60d0a4ee4390d0f','271482':'8e0dafe172a27b6d56b484bada005ec18d15e62ae18afdc7b0bff7870ba1a841'}
AUDIT={'135930':R13/'independent_execution_audit/ACTUAL_EXECUTION_AUDIT_V1.json','271482':K/'independent_review/ACTUAL_EXECUTION_AUDIT_V1.json'}
AUDIT_SHA={'135930':'eb77f69e87d80192157b1b3fd458289cdff56ea07a134bb1738076f9325d9a61','271482':'7455e073fcfb1ff981e228807cf917a4e4657a7ef988d072d7a16cdcbb77e9bd'}
ROOT_FREEZE_SHA='b8b1d76f2babd07f92180559e88f6f1a905c541ed570f0d4b36ace9ba7de6f07'
SUFFIX=('running_mean','running_var','num_batches_tracked')
OWN=('run_conversion_v2.py','conversion_core.py','test_conversion_v2_cpu.py','source_pins.json','PROTOCOL.md')
def require(v,msg):
 if not v:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def local(root,name):
 require(isinstance(name,str) and name==Path(name).name and name not in ('','.','..'),'Sibling artifact basename required')
 return root/name
require(sha(OLD)=='dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8','Independent geometry helper unchanged')
spec=importlib.util.spec_from_file_location('_r14_independent_geometry',OLD);G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)

def source_pins():
 freeze=RUN/'EXECUTION_FREEZE_V2.json';require(sha(freeze)==ROOT_FREEZE_SHA,'Root execution freeze actual bytes')
 sources={}
 for p in (freeze,K/'EXECUTION_FREEZE_V1.json'):
  for name,digest in read(p)['source_sha256'].items():require(sha(name)==digest,'Previously frozen actual source '+name);sources[name]=digest
  sources[str(p)]=sha(p)
 paths=[RUN/n for n in OWN]+[RUN/'cpu_results_v2.json',RUN/'runtime_inputs_v1.json',RUN/'INPUT_ASSET_BINDINGS_V1.json',
  HERE/'SOURCE_STATES_V1.json',HERE/'source_state_audit.py',HERE/'INDEPENDENT_CPU_REVIEW_V2.json',HERE/'EXECUTION_AUDIT_INPUT_SPEC_V1.md',
  HERE/'test_auditor_cpu.py',HERE/'AUDITOR_CPU_V1.json',OLD,R13/'data_protocol/manifest_v1.json',R13/'runtime_v1.json',R13/'preparation/regions_v1.json']
 for step in STEPS:
  require(sha(CACHE[step])==CACHE_SHA[step] and sha(AUDIT[step])==AUDIT_SHA[step],'Fixed prior actual cache/audit bytes')
  paths.extend((CACHE[step],AUDIT[step]))
 for path in paths:sources[str(path)]=sha(path)
 for relative,digest in read(RUN/'source_pins.json')['sources'].items():
  p=REPO/relative;require(sha(p)==digest,'Inherited helper bytes '+relative);sources[str(p)]=digest
 return sources

def schedule(m):
 rows=m['check_members'];require(len(rows)==8 and len({x['target_id'] for x in rows})==8,'Unique fixed8 membership')
 return [dict(forward_id=i*8+j+1,checkpoint_step=int(step),target_id=row['target_id'],layout='native',BN_scope='ALL78_CURRENT',mode='actual')
  for i,step in enumerate(STEPS) for j,row in enumerate(rows)]

def original_states():
 import torch
 torch.set_num_threads(2)
 require(not torch.cuda.is_initialized(),'CPU source audit only')
 earlier=read(HERE/'SOURCE_STATES_V1.json');require(earlier['status']=='PASS','Prior independently audited source')
 found={};names_by_step={}
 for step in STEPS:
  require(sha(PT[step])==CP[step],'Actual original checkpoint bytes')
  p=torch.load(PT[step],map_location='cpu',weights_only=True);state=p['model_state_dict']
  require(p['schema_version']=='flow_a_native_encoding_v2_checkpoint_v1' and p['training']['completed_optimizer_steps']==int(step),'Original checkpoint contract/step')
  require(p['contract_sha256']=='961f1700e76f3427c74d4e9af9799a684433ff00449b8386be6c1f9cdd75e934','Exact model/source contract')
  require(len(state)==711 and sum(v.numel() for v in state.values())==55001588,'Full source actual711 inventory')
  require(all(v.device.type=='cpu' and bool(torch.isfinite(v).all()) for v in state.values()),'Finite CPU source state')
  buffers={n for n in state if n.endswith(tuple('.'+s for s in SUFFIX))};names={n.rsplit('.',1)[0] for n in buffers}
  require(len(names)==78 and buffers=={n+'.'+s for n in names for s in SUFFIX},'All actual78 BN buffers')
  current={n:v for n,v in state.items() if n not in buffers}
  require(len(current)==477 and sum(v.numel() for v in current.values())==54919238,'True remaining477 state')
  require(G.tensor_digest(state)==SOURCE[step] and G.tensor_digest(current)==CURRENT[step],'Independent source/allcurrent actual state digests')
  prior=[x for x in earlier['source_states'] if x['checkpoint_step']==int(step)]
  require(len(prior)==1 and prior[0]['BN_names']==sorted(names) and prior[0]['source_native_state_sha256']==SOURCE[step] and prior[0]['all78_current_state_sha256']==CURRENT[step],'Prior independent source state agrees')
  found[step]={'checkpoint_sha256':CP[step],'source_state_sha256':SOURCE[step],'allcurrent_state_sha256':CURRENT[step],
   'state_tensors':477,'state_elements':54919238,'removed_BN_buffers':234,'source_state_independently_reconstructed':True}
  names_by_step[step]=names
  del p,state,current
 require(names_by_step['135930']==names_by_step['271482'],'Same source BN graph')
 return found,names_by_step

def verify_scope(scope,names):
 require(scope['scope']=='ALL78_CURRENT' and scope['BN_count']==78 and scope['private_BN_current']==24 and scope['external_BN_current']==54,'Current78 inventory')
 require(scope['frozen_BN_names']==[] and scope['current_stat_BN_names']==sorted(names),'Exact78 current/zero frozen names')
 for k in ('non_BN_parameters_and_affine_unchanged','source_BN_buffers_removed_before_inference'):require(scope[k] is True,'Current policy '+k)
 for k in ('calibration_performed','external_BN_artifact_loaded'):require(scope[k] is False,'No imported/calibrated BN '+k)
 sig=scope['BN_signature_before_inference'];require(set(sig)==names,'All78 actual signatures')
 for x in sig.values():require(x['training'] is False and x['tracking'] is False and all(x[s] is None for s in SUFFIX),'All78 eval/no buffers/current policy')

def verify_records(d,m):
 expected=schedule(m);actual=d['forward_records'];require(len(actual)==d['full_forward_count']==16,'Exactly actual16 records')
 counts=Counter();lookup={}
 for rec,wanted in zip(actual,expected):
  require(all(rec[k]==v for k,v in wanted.items()),'Actual schedule/order/native mode')
  ex=rec['execution'];counts.update(G.check_execution(ex,'native',False,'evaluation'))
  require(ex['anchor']=='native' and ex['mode']=='actual' and ex['layers']=={},'Native forward without producer wrapper')
  require(len(ex['head_records'])==4 and all(x['shape']==[1,2,480,640] and x['dtype']=='torch.float32' and x['finite'] is True and G.hexdigest(x['sha256']) for x in ex['head_records']),'All4 actual native heads')
  require(set(ex['branch_records'])==G.SELECTED,'All8 original branch/source observations')
  for n,v in ex['branch_records'].items():require(v['shape']==list(G.GEOMETRY[n]) and all(G.hexdigest(v[k]) for k in ('input_sha256','output_sha256')),'True branch geometry/SHA')
  lookup[(str(wanted['checkpoint_step']),wanted['target_id'])]=ex
 require(counts['FC1_rows']==counts['FC2_rows']==8928000 and counts['wrapped_layers']==counts['native_calls_in_wrappers']==0 and counts['observed_FC_modules']==384,'Independent16 total FC rows/modules')
 return counts,lookup

def read_prediction(path,archive_sha,payload_sha):
 import numpy as np
 require(sha(path)==archive_sha,'Actual prediction archive SHA')
 with np.load(path,allow_pickle=False) as z:
  require('native' in z.files,'Native raw array retained');a=z['native']
  require(a.dtype==np.float32 and a.shape==(2,480,640) and np.isfinite(a).all(),'Actual finite native FP32 shape')
  require(hashlib.sha256(a.tobytes()).hexdigest()==payload_sha,'Actual native payload SHA')
 return {'file':str(path),'sha256':archive_sha,'bytes':path.stat().st_size}

def verify_references(b,m,inputs):
 out={};frames_by_step={};artifacts=[]
 for step in STEPS:
  require(sha(CACHE[step])==CACHE_SHA[step] and sha(AUDIT[step])==AUDIT_SHA[step],'Actual fixed cache/audit source')
  old=read(CACHE[step]);a=read(AUDIT[step]);require(a['status']=='PASS','Previously independent audit PASS')
  descriptor=b['references'][step];phase='initial' if step=='135930' else 'endpoint'
  require(descriptor['cache_phase']==phase and descriptor['cache_result_sha256']==CACHE_SHA[step] and descriptor['execution_audit_sha256']==AUDIT_SHA[step],'Exact cache phase/source audit')
  require(descriptor['cache_result_path']==inputs['references'][step]['result']['path'] and descriptor['execution_audit_path']==inputs['references'][step]['execution_audit']['path'],'Frozen remote/cache path identity')
  require(descriptor['checkpoint_sha256']==CP[step] and descriptor['allcurrent_state_sha256']==CURRENT[step],'Same original W source')
  require(descriptor['cache_not_loaded_as_model'] is True and descriptor['no_quality_based_admission'] is True,'Reference-only metadata')
  if step=='135930':
   require(a['schema']=='flow_round13_independent_execution_audit_v1' and a['scope']=={'conditions':['reference','diversity'],'modes':['smoke','recover'],'runs':16},'Full16 previous audit')
   matches=[x for x in a['results'] if (x['condition'],x['mode'],x['arm'])==('diversity','recover','native')]
   require(len(matches)==1 and matches[0]['result_sha256']==CACHE_SHA[step],'Exact independently audited original initial cache')
   prediction_map={x['target_id']:x['array_sha256'] for x in matches[0]['predictions'] if x['label']=='initial' and x['layout']=='native'}
   state=a['common_initial_model_sha256'];bn=a['new_BN_actual_sha256']
   require(old['initial_model_state_sha256']==state and old['BN_scope']['non_bn_state_sha256']==CURRENT[step],'Source unchanged before original recovery')
  else:
   require(a['schema']=='flow_later_checkpoint_independent_execution_audit_v1' and a['source_result_sha256']==CACHE_SHA[step] and a['full_forwards']==193 and a['optimizer_updates']==0,'Fixed ownBN K source')
   require(a['source_state_sha256']==SOURCE[step] and a['source_non_BN_state_sha256']==CURRENT[step],'Independently verified K original W')
   prediction_map={x['target_id']:x['array_sha256'] for x in a['quality_predictions'] if x['layout']=='native'}
   state=a['deployment_state_sha256'];bn=a['BN_buffers_sha256']
  require(descriptor['cached_deployment_state_sha256']==state and descriptor['cached_BN_buffers_sha256']==bn,'Complete cached deployment identity')
  frames=old['evaluations'][phase]['frames'];require([x['target_id'] for x in frames]==[x['target_id'] for x in m['check_members']],'Cached exact8/order')
  expected=[];predictions=[]
  for f,row in zip(frames,m['check_members']):
   require(f['sequence']==row['sequence'] and f['arms']['native']['prediction_sha256']==prediction_map[row['target_id']],'Audited cache payload/member')
   item={k:f[k] for k in ('target_id','sequence','input_evidence','prediction_file','prediction_file_sha256')};item['prediction_sha256']=prediction_map[row['target_id']];expected.append(item)
   artifacts.append(read_prediction(local(CACHE[step].parent,f['prediction_file']),f['prediction_file_sha256'],item['prediction_sha256']))
   predictions.append({'target_id':row['target_id'],'sequence':row['sequence'],'array_sha256':item['prediction_sha256']})
  require(descriptor['cached_native_predictions']==expected,'All8 exact cached identity/source metadata')
  out[step]={'cache_result_sha256':CACHE_SHA[step],'execution_audit_sha256':AUDIT_SHA[step],'cache_phase':phase,'cached_deployment_state_sha256':state,'cached_BN_buffers_sha256':bn,'checkpoint_sha256':CP[step],'cached_native_predictions':predictions}
  frames_by_step[step]=expected
 return out,frames_by_step,artifacts

def audit(path):
 import numpy as np
 import torch
 d=read(path);b=d['binding'];m=read(R13/'data_protocol/manifest_v1.json');G.memberships(m);inputs=read(RUN/'runtime_inputs_v1.json')
 require(d['schema']=='flow_bn_conversion_round14_v1' and d['status']=='ROUND14_BN_CONVERSION_COMPLETE','Actual completed16 result')
 require(d['inference_executed'] is True and d['training_performed'] is False,'Inference-only current-native test')
 for k in ('optimizer_steps','calibration_forwards','dual_forwards'):require(d[k]==0 and b[k]==0,'Zero '+k)
 for k in ('flow_specific_increment_claim','hardware_benefit_claim'):require(d[k] is False,'No unsupported claim '+k)
 require(b['schema']=='flow_bn_conversion_round14_binding_v1' and b['source_sha256']=={n:sha(RUN/n) for n in OWN} and b['cpu_checks_sha256']==sha(RUN/'cpu_results_v2.json'),'Frozen runtime source/cpu receipt')
 require(b['inputs_manifest_sha256']==sha(RUN/'runtime_inputs_v1.json'),'Exact execution inputs')
 for key,file in [('membership_sha256',R13/'data_protocol/manifest_v1.json'),('runtime_manifest_sha256',R13/'runtime_v1.json'),('region_sha256',R13/'preparation/regions_v1.json')]:require(b[key]==sha(file),'Frozen common source '+key)
 require(b['check_ids']==[r['target_id'] for r in m['check_members']] and b['forward_plan']==schedule(m),'Fixed membership/schedule')
 require(b['total_full_forwards']==16 and b['BN_scope']=='ALL78_CURRENT' and b['precision']=='FP32/noAMP/noTF32/T10/B1/full9','Fixed16 complete policy')
 require(0<b['wall_minutes']<=15 and G.finite(d['wall_seconds']) and 0<d['wall_seconds']<=900,'Resource cap')
 require(d['runtime']['AMP'] is False and d['runtime']['TF32'] is False and d['runtime']['threads']==2,'FP32 recorded runtime')
 sources,names=original_states();require(set(d['checkpoints'])==set(STEPS) and set(b['checkpoint_cpu_audits'])==set(STEPS) and set(b['references'])==set(STEPS),'Both fixed source inventories')
 references,cache_frames,cache_artifacts=verify_references(b,m,inputs)
 counts,lookup=verify_records(d,m);artifacts=[];predictions=[];evidence=[];case_times={}
 for step in STEPS:
  item=d['checkpoints'][step];cpu=b['checkpoint_cpu_audits'][step]
  require(item['checkpoint_step']==cpu['checkpoint_step']==int(step) and item['checkpoint_sha256']==cpu['checkpoint_sha256']==CP[step],'Actual original W binding')
  require(item['source_state_sha256']==cpu['source_state_sha256']==SOURCE[step],'Loaded source digest independently agrees')
  require(item['initial_allcurrent_state_sha256']==item['final_allcurrent_state_sha256']==cpu['allcurrent_state_sha256']==CURRENT[step],'Complete unchanged current477 source state')
  for key,value in {'source_tensor_count':711,'removed_BN_buffers':234,'allcurrent_tensor_count':477,'allcurrent_elements':54919238,'optimizer_restored':False,'external_BN_artifact_loaded':False,'source_BN_buffers_removed_before_inference':True}.items():require(cpu[key]==value,'Actual source inventory '+key)
  require(cpu['BN_names']==sorted(names[step]) and cpu['contract_sha256']=='961f1700e76f3427c74d4e9af9799a684433ff00449b8386be6c1f9cdd75e934','Source BN graph/contract')
  verify_scope(item['BN_scope'],names[step])
  for key in ('state_unchanged','BN_buffers_unchanged','all_hooks_removed','original_modules_restored'):require(item[key] is True,'Per-source closure '+key)
  require(item['full_forward_count']==8 and set(item['evaluations'])=={'endpoint'},'Exactly8 per source, no hidden phases')
  frames=item['evaluations']['endpoint']['frames'];require(len(frames)==8 and [x['target_id'] for x in frames]==b['check_ids'],'All8 actual endpoint order')
  require(G.finite(item['wall_seconds']) and 0<item['wall_seconds']<=d['wall_seconds'],'Per-source wall')
  case_times[step]=item['wall_seconds']
  cached_runtime=read(CACHE[step])['runtime'];current_runtime=d['runtime']
  for key in ('torch','cuda'):require(current_runtime[key]==cached_runtime[key],'Paired runtime '+key)
  require(current_runtime['numpy']==cached_runtime.get('numpy',cached_runtime.get('NumPy')) and current_runtime['device']==cached_runtime.get('device',cached_runtime.get('device_name')),'Paired runtime/device')
  if 'cudnn' in cached_runtime:require(current_runtime['cudnn']==cached_runtime['cudnn'],'Paired cuDNN')
  for f,row,old in zip(frames,m['check_members'],cache_frames[step]):
   require(f['sequence']==row['sequence'] and f['input_evidence']==old['input_evidence'] and f['paired_cached_native']==old,'Same actual input/GT/mask/cache binding')
   require(set(f['arms'])=={'native'},'Only native retained endpoint')
   payload=f['arms']['native'];ex=lookup[(step,row['target_id'])]
   require(payload['execution']==ex and payload['prediction_sha256']==ex['head_records'][3]['sha256'],'Actual fullforward-to-prediction identity')
   target=local(path.parent,f['prediction_file'])
   require(target.name=='current_'+step+'_'+row['target_id']+'.npz','Unique actual checkpoint/prediction name')
   with np.load(target,allow_pickle=False) as z:require(z.files==['native'],'Exactly one actual native payload per archive')
   artifacts.append(read_prediction(target,f['prediction_file_sha256'],payload['prediction_sha256']))
   predictions.append({'checkpoint_step':int(step),'target_id':row['target_id'],'layout':'native','array_sha256':payload['prediction_sha256']})
   evidence.append({'checkpoint_step':int(step),'target_id':row['target_id'],'sequence':row['sequence'],'input_evidence':old['input_evidence']})
  sources[step].update(state_unchanged=True,all78_current_verified=True)
 require(d['data_evidence']==evidence,'All16 actual input source bindings')
 require(len(artifacts)==len(predictions)==16 and len({x['file'] for x in artifacts})==16,'16 unique actual archives/payloads')
 require({p.name for p in path.parent.glob('*.npz')}=={Path(x['file']).name for x in artifacts} and not list(path.parent.glob('*.pt')),'No extra BN/prediction/weight artifacts')
 for key in ('all_states_unchanged','all_BN_buffers_removed','all_hooks_removed','original_modules_restored'):require(d[key] is True,'Global closure '+key)
 alloc=d['max_allocated_bytes'];reserved=d['max_reserved_bytes'];require(type(alloc) is int and type(reserved) is int and 0<alloc<=reserved,'Actual GPU allocator resource record')
 require(sum(case_times.values())<=d['wall_seconds']+1e-6 and not torch.cuda.is_initialized(),'Time accounting/CPU-only audit')
 return {'schema':'flow_bn_conversion_round14_independent_execution_audit_v1','status':'PASS','auditor_sha256':sha(__file__),
  'source_result_sha256':sha(path),'source_result_path':str(path),'full_forwards':16,'optimizer_updates':0,'calibration_forwards':0,'dual_forwards':0,
  'quality_prediction_count':16,'quality_archive_count':16,'BN_scope':'ALL78_CURRENT','source_states':sources,'references':references,
  'quality_predictions':predictions,'quality_archives':artifacts,'cached_prediction_archives':cache_artifacts,
  'executed_FC1_rows':counts['FC1_rows'],'executed_FC2_rows':counts['FC2_rows'],'service_totals_from_actual_records':dict(counts),
  'all_states_unchanged':True,'all_BN_buffers_removed':True,'all_hooks_removed':True,'original_modules_restored':True,
  'resource_ledger':{'device':d['runtime']['device'],'wall_seconds':d['wall_seconds'],'per_source_inference_and_archive_seconds':case_times,'max_allocated_bytes':alloc,'max_reserved_bytes':reserved},
  'GPU_used':False,'SSH_used':False,'quality_metrics_read_or_ranked':False,
  'limitations':['Two original source PTs independently read and complete all78-current state reconstructed. Before/after GPU state matches runtime digests; no duplicate endpoint PT is saved.',
   'All16 final raw predictions and both prior cached native prediction sets reread. Intermediate4-head/8-branch tensors are checked by shapes and digest records, not independently replayed.',
   'Input event/GT/mask bytes not reread in this execution audit; exact SHA/order and prior accepted source identities are checked. Independent raw-GT quality audit is separate.',
   'No quality gate or claim from execution PASS. This BN conversion control does not establish recovery stability, H0 eligibility, flow-specific novelty or hardware cycles/energy.',
   'GPU host timing and allocator peaks are observed resources; neither is PE cycle evidence.']}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','audit'),default='prepare');p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',type=Path);p.add_argument('--result',type=Path);a=p.parse_args()
 require(not a.output.exists(),'No audit artifact overwrite')
 if a.stage=='prepare':
  m=read(R13/'data_protocol/manifest_v1.json');G.memberships(m)
  require(sum(G.expected_layer(n,'native')[0] for n in G.GEOMETRY)==558000,'Independent true FC row geometry')
  result={'schema':'flow_bn_conversion_round14_independent_execution_freeze_v1','status':'PREPARED_BEFORE_ACTUAL_RESULT_READ','auditor_sha256':sha(__file__),
   'source_sha256':source_pins(),'forward_schedule':schedule(m),'full_forwards':16,'optimizer_updates':0,'calibration_forwards':0,'dual_forwards':0,
   'quality_prediction_count':16,'quality_archive_count':16,'expected_FC_rows_each':8928000,'runner_modules_imported':False,'quality_metrics_read_or_ranked':False}
 else:
  require(a.freeze and a.result,'Explicit fixed auditor freeze/result');f=read(a.freeze)
  require(f['auditor_sha256']==sha(__file__) and f['source_sha256']==source_pins(),'Unchanged pre-result audit/source freeze')
  result=audit(a.result);result['audit_freeze_sha256']=sha(a.freeze)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':result['status'],'schema':result['schema'],'output_sha256':sha(a.output)}))
if __name__=='__main__':main()
