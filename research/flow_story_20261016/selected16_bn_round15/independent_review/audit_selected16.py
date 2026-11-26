#!/usr/bin/env python3
"""Independent CPU audit of fixed R15 selected16/native8; no runner or AEE."""
import argparse
from collections import Counter
import hashlib,importlib.util,json,math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;RUN=HERE.parent;DATE=RUN.parent;REPO=DATE.parents[1]
R13=DATE/'coverage_recovery_round13';R14=DATE/'bn_conversion_round14';K=DATE/'later_checkpoint_deployment'
A14FILE=R14/'independent_review/audit_conversion.py'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(v,msg):
 if not v:raise ValueError(msg)
require(sha(A14FILE)=='ced75fda67e2e868d85398d72204cf929b55bee86973be7b9c6cc3933ed00478','Frozen independent R14 helpers')
s=importlib.util.spec_from_file_location('_independent_r14_readonly',A14FILE);A=importlib.util.module_from_spec(s);s.loader.exec_module(A);G=A.G
CP=A.CP['271482'];SOURCE=A.SOURCE['271482'];CURRENT=A.CURRENT['271482'];SELECTED='6b4be9b8f136c4f978c892fff5532723828654f203c2d468e837a70ad0fd4451'
BN=K/'run_v1/bn_buffers.npz';BN_SHA='4d887a8fce408e3a33cb95bd120d2ed3d865541a39c034205f4172ab7bdd30dc'
BN_RECEIPT=K/'run_v1/bn_receipt.json';BN_RECEIPT_SHA='24d33ba100150775a733fe6ecfabd4e88a1c152ca080adbf1f387a30ce574b9c'
CACHE={'private24':K/'run_v1/results.json','current78':R14/'run_v2/results.json'}
CACHE_SHA={'private24':'8e0dafe172a27b6d56b484bada005ec18d15e62ae18afdc7b0bff7870ba1a841','current78':'c9ead13e411512f570fc6ca926fcfaaa67c2f93e04c700bdd7799c85fcb98404'}
AUDIT={'private24':K/'independent_review/ACTUAL_EXECUTION_AUDIT_V1.json','current78':R14/'independent_review/ACTUAL_EXECUTION_AUDIT_V1.json'}
AUDIT_SHA={'private24':'7455e073fcfb1ff981e228807cf917a4e4657a7ef988d072d7a16cdcbb77e9bd','current78':'6bdcd565ccff753447f9becb7066f6670c7216841d644b174998a0a77960da71'}
OWN=('run_selected16.py','selected_core.py','source_state_cpu.py','test_selected16_cpu.py','expected_state.json','source_pins.json','PROTOCOL.md')
ROOT_FREEZE_SHA='dac1427a406ec38328b07e178a6b68437eafa8f31858b3c7aef4fdabeab3a13b'
SHAPE_FILE=DATE.parent/'flow_story_20261014/damage_allocation_probe/hardware_contract_complete_bn_audit.json'
SHAPE_SHA='9a92bd764878d36b5ca67af8ef0a39806830e755dd0f9a91217c81e165e27567'
SUFFIX=('running_mean','running_var','num_batches_tracked');SCOPE='SELECTED16_FIXED_62_CURRENT'

def source_pins():
 f=RUN/'EXECUTION_FREEZE_V1.json';require(sha(f)==ROOT_FREEZE_SHA,'Root exact S8 execution freeze')
 sources={}
 for path in (f,R14/'independent_review/PRE_RESULT_FREEZE_V1.json'):
  for name,digest in read(path)['source_sha256'].items():require(sha(name)==digest,'Frozen source '+name);sources[name]=digest
  sources[str(path)]=sha(path)
 paths=[RUN/n for n in OWN]+[RUN/'cpu_results.json',RUN/'runtime_inputs_v1.json',HERE/'SOURCE_STATE_V1.json',HERE/'reconstruct_source.py',
  HERE/'EXECUTION_AUDIT_INPUT_SPEC_V1.md',HERE/'INDEPENDENT_CPU_REVIEW_V1.json',HERE/'test_auditor_cpu.py',HERE/'AUDITOR_CPU_V1.json',A14FILE,SHAPE_FILE,BN_RECEIPT]
 require(sha(SHAPE_FILE)==SHAPE_SHA,'Actual prior full9 external54 shape evidence')
 for label in CACHE:
  require(sha(CACHE[label])==CACHE_SHA[label] and sha(AUDIT[label])==AUDIT_SHA[label],'Fixed same-W cached source')
  paths.extend((CACHE[label],AUDIT[label]))
 for path in paths:sources[str(path)]=sha(path)
 for relative,digest in read(RUN/'source_pins.json')['sources'].items():
  p=REPO/relative;require(sha(p)==digest,'Inherited source bytes '+relative);sources[str(p)]=digest
 return sources

def names_and_shapes():
 require(sha(SHAPE_FILE)==SHAPE_SHA,'Fixed actual external54 source')
 evidence=read(SHAPE_FILE);expected={x['name']:x['native_input_shape'] for x in evidence['external_BN']['records']}
 require(len(expected)==54,'Full external54 inventory')
 for relative,digest in evidence['native_source_sha256'].items():require(sha(REPO/'SDformer/third_party/SDformerFlow'/relative)==digest,'Unchanged native shape source')
 for n,(t,b,h,w,c) in G.GEOMETRY.items():
  for idx,width in ((1,4*c),(2,c)):expected[n+'.bn'+str(idx)+'.norm_layer']=[t,b,width,h,w]
 fixed={n+'.bn'+str(idx)+'.norm_layer' for n in G.SELECTED for idx in (1,2)}
 require(len(expected)==78 and len(fixed)==16,'True native78 and selected16')
 return fixed,expected

def reconstruct():
 import numpy as np
 import torch
 torch.set_num_threads(2);require(not torch.cuda.is_initialized(),'CPU actual source reconstruction')
 pt=A.PT['271482'];require(sha(pt)==CP and sha(BN)==BN_SHA and sha(BN_RECEIPT)==BN_RECEIPT_SHA,'Actual same-W source bytes')
 p=torch.load(pt,map_location='cpu',weights_only=True);state=p['model_state_dict']
 require(p['training']['completed_optimizer_steps']==271482 and p['contract_sha256']=='961f1700e76f3427c74d4e9af9799a684433ff00449b8386be6c1f9cdd75e934','Actual source W contract')
 require(len(state)==711 and sum(v.numel() for v in state.values())==55001588 and all(v.device.type=='cpu' and bool(torch.isfinite(v).all()) for v in state.values()),'Finite711 source')
 require(G.tensor_digest(state)==SOURCE,'Actual source full state')
 fixed,shapes=names_and_shapes();allnames=set(shapes);private={n+'.bn'+str(i)+'.norm_layer' for n in G.GEOMETRY for i in (1,2)}
 buffers={n for n in state if n.endswith(tuple('.'+x for x in SUFFIX))};require(buffers=={n+'.'+x for n in allnames for x in SUFFIX},'True78 source buffers')
 pure={n:v for n,v in state.items() if n not in buffers};require(len(pure)==477 and G.tensor_digest(pure)==CURRENT,'Actual unchanged477 nonBN')
 chosen={}
 with np.load(BN,allow_pickle=False) as z:
  require(set(z.files)=={n+'.'+x for n in private for x in SUFFIX},'Own original72 keys')
  for key in z.files:
   value=z[key];old=state[key].numpy();require(value.shape==old.shape and value.dtype==old.dtype and np.isfinite(value).all(),'All actual own72 geometry/finite')
   if key.endswith('.running_var'):require((value>=0).all(),'Nonnegative own variance')
   if key.endswith('.num_batches_tracked'):require(int(value)==176,'Source own CMA176 count')
   if key.rsplit('.',1)[0] in fixed:chosen[key]=value.copy()
 tensors={n:torch.from_numpy(v) for n,v in chosen.items()};full=dict(pure);full.update(tensors)
 require(len(full)==525 and sum(v.numel() for v in full.values())==54944214 and G.tensor_digest(full)==SELECTED,'Actual selected525 state')
 arrays={n:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for n,v in sorted(chosen.items())}
 record={'source_state_sha256':SOURCE,'non_BN_state_sha256':CURRENT,'selected16_state_sha256':SELECTED,'selected16_tensor_count':525,'selected16_elements':54944214,
  'retained_buffer_count':48,'retained_buffer_elements':sum(v.size for v in chosen.values()),'retained_buffer_tensor_sha256':G.tensor_digest(tensors),
  'retained_buffer_payloads':arrays,'frozen_BN_names':sorted(fixed),'private_current_BN_names':sorted(private-fixed)}
 previous=read(HERE/'SOURCE_STATE_V1.json');require(previous['selected16_deployment_state_sha256']==SELECTED and previous['status']=='PASS','Prior independently rebuilt source agrees')
 return record,chosen,fixed,shapes

def schedule(m):
 members=m['check_members'];require(len(members)==8 and len({r['target_id'] for r in members})==8,'Unique fixed8')
 return [dict(forward_id=i+1,checkpoint_step=271482,target_id=r['target_id'],layout='native',BN_scope=SCOPE,mode='actual') for i,r in enumerate(members)]

def verify_scope(scope,chosen,fixed,shapes):
 require(scope['scope']==SCOPE,'Actual selected16 policy')
 for k,v in {'BN_count':78,'private_BN_frozen':16,'private_BN_current':8,'external_BN_current':54,'retained_buffer_count':48}.items():require(scope[k]==v,'Scope inventory '+k)
 require(scope['frozen_BN_names']==sorted(fixed) and scope['current_stat_BN_names']==sorted(set(shapes)-fixed),'Exact source16/62 names')
 for k in ('non_BN_parameters_and_affine_unchanged','own_CMA_source_loaded','source_unselected_BN_buffers_removed_before_inference'):require(scope[k] is True,'Scope closure '+k)
 require(scope['calibration_performed'] is False,'No new calibration')
 sig=scope['BN_signature_before_inference'];require(set(sig)==set(shapes),'All78 actual signature')
 for name in shapes:
  r=sig[name];require(r['training'] is False and r['tracking']==(name in fixed),'Eval/fixed/current flags')
  for field in SUFFIX:
   wanted=hashlib.sha256(chosen[name+'.'+field].tobytes()).hexdigest() if name in fixed else None
   require(r[field]==wanted,'Actual own selected48 or absent current buffer')

def verify_records(d,m,fixed,shapes):
 rows=d['forward_records'];require(len(rows)==d['full_forward_count']==8,'Actual complete8 forward records')
 totals=Counter();lookup={}
 for r,w in zip(rows,schedule(m)):
  require(all(r[k]==v for k,v in w.items()),'Actual original8 schedule')
  ex=r['execution'];totals.update(G.check_execution(ex,'native',False,'evaluation'))
  require(ex['anchor']=='native' and ex['mode']=='actual' and ex['layers']=={},'Native-only no producer wrapper')
  require(len(ex['head_records'])==4 and all(v['shape']==[1,2,480,640] and v['dtype']=='torch.float32' and v['finite'] is True and G.hexdigest(v['sha256']) for v in ex['head_records']),'Four fullresolution heads')
  require(set(ex['branch_records'])==G.SELECTED,'Eight native branches/prefixes')
  for n,v in ex['branch_records'].items():require(v['shape']==list(G.GEOMETRY[n]) and all(G.hexdigest(v[k]) for k in ('input_sha256','output_sha256')),'True branch geometry/digests')
  bn=ex['all78_BN_observations'];require(set(bn)==set(shapes),'Actual78 BN hook coverage')
  for name,shape in shapes.items():
   require(bn[name]=={'calls':1,'input_shapes':[shape],'current_stat':name not in fixed},'Exact per-native BN geometry/policy '+name)
   totals['current_BN_calls' if name not in fixed else 'fixed_BN_calls']+=1
   if name not in fixed:totals['current_BN_input_elements']+=math.prod(shape)
  lookup[w['target_id']]=ex
 require(totals['FC1_rows']==totals['FC2_rows']==4464000 and totals['observed_FC_modules']==192 and totals['wrapped_layers']==0,'Independent native8 FC totals')
 require(totals['current_BN_calls']==496 and totals['fixed_BN_calls']==128,'Actual62current/16fixed across8')
 return totals,lookup

def verify_caches(binding,m,inputs):
 result={};frames={};archives=[]
 for label in ('private24','current78'):
  require(sha(CACHE[label])==CACHE_SHA[label] and sha(AUDIT[label])==AUDIT_SHA[label],'Actual fixed cache/audit bytes')
  old=read(CACHE[label]);audit=read(AUDIT[label]);require(audit['status']=='PASS','Independent prior cache execution')
  if label=='private24':
   require(audit['schema']=='flow_later_checkpoint_independent_execution_audit_v1' and audit['source_result_sha256']==CACHE_SHA[label],'Actual K source')
   fs=old['evaluations']['endpoint']['frames'];state=audit['deployment_state_sha256'];bn=audit['BN_buffers_sha256'];preds=audit['quality_predictions']
   require(audit['source_state_sha256']==SOURCE and audit['source_non_BN_state_sha256']==CURRENT,'K actual original W')
  else:
   require(audit['schema']=='flow_bn_conversion_round14_independent_execution_audit_v1' and audit['source_result_sha256']==CACHE_SHA[label],'Actual R14 source')
   fs=old['checkpoints']['271482']['evaluations']['endpoint']['frames'];state=CURRENT;bn=None;preds=[v for v in audit['quality_predictions'] if v['checkpoint_step']==271482]
   require(audit['source_states']['271482']['allcurrent_state_sha256']==CURRENT and audit['source_states']['271482']['source_state_sha256']==SOURCE,'R14 actual unchanged W')
  desc=binding['references'][label];expected={}
  for key,value in {'cache_phase':'endpoint','cache_result_sha256':CACHE_SHA[label],'execution_audit_sha256':AUDIT_SHA[label],
   'checkpoint_sha256':CP,'cached_deployment_state_sha256':state,'cached_BN_buffers_sha256':bn}.items():require(desc[key]==value,'Cache binding '+key);expected[key]=value
  require(desc['cache_result_path']==inputs['references'][label]['result']['path'] and desc['execution_audit_path']==inputs['references'][label]['execution_audit']['path'],'Actual input manifest cache paths')
  require(desc['cache_not_loaded_as_model'] is True,'Cache not a weight source')
  if label=='private24':require(desc['allcurrent_state_sha256']==CURRENT and desc['no_quality_based_admission'] is True,'Private24 reference identity only')
  else:require(desc['preselected_W_not_quality_ranked'] is True,'Preselected W')
  for k in ('membership_sha256','runtime_manifest_sha256','region_sha256'):require(old['binding'][k]==binding[k],'Both cached input contracts')
  require([v['target_id'] for v in fs]==[v['target_id'] for v in m['check_members']],'Cached complete check8')
  predmap={v['target_id']:v['array_sha256'] for v in preds if v['layout']=='native'};expected_frames=[];predictions=[]
  for f,row in zip(fs,m['check_members']):
   require(f['sequence']==row['sequence'] and f['arms']['native']['prediction_sha256']==predmap[row['target_id']],'Cached audited native payload')
   item={k:f[k] for k in ('target_id','sequence','input_evidence','prediction_file','prediction_file_sha256')};item['prediction_sha256']=predmap[row['target_id']];expected_frames.append(item)
   archives.append(A.read_prediction(A.local(CACHE[label].parent,f['prediction_file']),f['prediction_file_sha256'],item['prediction_sha256']))
   predictions.append({'target_id':row['target_id'],'sequence':row['sequence'],'array_sha256':item['prediction_sha256']})
  require(desc['cached_native_predictions']==expected_frames,'Exact8 cache actual source descriptors')
  expected['cached_native_predictions']=predictions;result[label]=expected;frames[label]=expected_frames
 return result,frames,archives

def audit(path):
 import numpy as np
 import torch
 d=read(path);b=d['binding'];m=read(R13/'data_protocol/manifest_v1.json');G.memberships(m);inputs=read(RUN/'runtime_inputs_v1.json')
 require(d['schema']=='flow_selected16_bn_round15_v1' and d['status']=='ROUND15_SELECTED16_S8_COMPLETE','Complete actual S8 only')
 require(d['training_performed'] is False and d['inference_executed'] is True,'Actual inference-only S8')
 for k in ('optimizer_steps','calibration_forwards','dual_forwards'):require(d[k]==b[k]==0,'Zero '+k)
 for k in ('flow_specific_increment_claim','hardware_benefit_claim','automatic_FF_H0_admission'):require(d[k] is False,'No unsupported claim/admission')
 require(b['schema']=='flow_selected16_bn_round15_binding_v1' and b['source_sha256']=={n:sha(RUN/n) for n in OWN} and b['cpu_checks_sha256']==sha(RUN/'cpu_results.json'),'Actual frozen source/CPU')
 require(b['checkpoint_sha256']==CP and b['checkpoint_step']==271482 and b['inputs_manifest_sha256']==sha(RUN/'runtime_inputs_v1.json'),'Fixed W/runtime inputs')
 for k,p in [('membership_sha256',R13/'data_protocol/manifest_v1.json'),('runtime_manifest_sha256',R13/'runtime_v1.json'),('region_sha256',R13/'preparation/regions_v1.json')]:require(b[k]==sha(p),'Fixed common source '+k)
 require(b['total_full_forwards']==8 and b['forward_plan']==schedule(m) and b['check_ids']==[r['target_id'] for r in m['check_members']],'Fixed exact8 schedule')
 require(b['BN_scope']==SCOPE and b['precision']=='FP32/noAMP/noTF32/T10/B1/full9','Selected16 full native graph')
 observer=b['native_forward_observer'];require(observer['frozen_h0_core_sha256']=='48fccdd60cbc468c381d3caf6a316a6427dd68f09933731290befd729b9c58da' and observer['historical_module_globals_modified'] is False and observer['extra_full_forwards']==0,'Narrow observation adapter only')
 require(observer['adaptation']=='Private forward globals; only exact Unremoved forward hooks guard accepts the known78 observer pre-hook ids. All other guards unchanged; finally removes every observer.','Exact declared hook-only adaptation')
 require(0<b['wall_minutes']<=15 and G.finite(d['wall_seconds']) and 0<d['wall_seconds']<=900,'Bounded execution')
 require(d['runtime']['AMP'] is False and d['runtime']['TF32'] is False and d['runtime']['threads']==2,'FP32 runtime')
 for label in CACHE:
  old=read(CACHE[label])['runtime'];new=d['runtime']
  require(old['torch']==new['torch'] and old['cuda']==new['cuda'] and old.get('numpy',old.get('NumPy'))==new['numpy'] and old.get('device',old.get('device_name'))==new['device'],'Same cached runtime/device')
  if 'cudnn' in old:require(old['cudnn']==new['cudnn'],'Same cuDNN')
 state,chosen,fixed,shapes=reconstruct();require(b['selected_state_cpu_audit']==state,'Actual selected525 CPU audit vs independent values')
 require(b['own_BN_buffers_sha256']==BN_SHA and b['own_BN_receipt_sha256']==BN_RECEIPT_SHA,'Own same-W original statistics')
 cpu=b['checkpoint_cpu_audit'];require(cpu['checkpoint_sha256']==CP and cpu['checkpoint_step']==271482 and cpu['source_state_sha256']==SOURCE and cpu['allcurrent_state_sha256']==CURRENT,'Actual original CPU source binding')
 require(set(d['checkpoints'])=={'271482'},'Only one selected W');item=d['checkpoints']['271482']
 require(item['checkpoint_step']==271482 and item['checkpoint_sha256']==CP and item['source_state_sha256']==SOURCE,'Actual GPU original source')
 require(item['initial_selected16_state_sha256']==item['final_selected16_state_sha256']==SELECTED and item['full_forward_count']==8,'Unchanged complete525 state')
 verify_scope(item['BN_scope'],chosen,fixed,shapes)
 for k in ('state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed'):require(item[k] is True,'Per-source closure '+k)
 meta=d['retained_BN_buffers'];target=A.local(path.parent,meta['file']);require(sha(target)==meta['sha256'],'Actual retained48 archive')
 require(meta['keys']==sorted(chosen) and meta['buffer_tensor_sha256']==state['retained_buffer_tensor_sha256'] and meta['source_own72_sha256']==BN_SHA and meta['source_own_receipt_sha256']==BN_RECEIPT_SHA,'Retained own48 source metadata')
 with np.load(target,allow_pickle=False) as z:
  require(set(z.files)==set(chosen),'Exactly actual48 retained arrays')
  for name,value in chosen.items():require(z[name].dtype==value.dtype and z[name].shape==value.shape and np.array_equal(z[name],value),'Actual retained BN array equals own source')
 references,cached,cached_archives=verify_caches(b,m,inputs);totals,lookup=verify_records(d,m,fixed,shapes)
 require(set(item['evaluations'])=={'endpoint'},'No unplanned evaluation phase')
 frames=item['evaluations']['endpoint']['frames'];require(len(frames)==8 and [f['target_id'] for f in frames]==b['check_ids'],'Actual complete8 predictions')
 archives=[];predictions=[];evidence=[]
 for idx,(f,row) in enumerate(zip(frames,m['check_members'])):
  expected={label:cached[label][idx] for label in cached}
  require(f['sequence']==row['sequence'] and f['paired_cached_native']==expected and all(f['input_evidence']==v['input_evidence'] for v in expected.values()),'Exact same actual input/source')
  require(set(f['arms'])=={'native'},'No FF/coarse predictions')
  payload=f['arms']['native'];ex=lookup[row['target_id']]
  require(payload['execution']==ex and payload['prediction_sha256']==ex['head_records'][3]['sha256'],'Actual final flow3 identity')
  p=A.local(path.parent,f['prediction_file']);require(p.name=='selected16_271482_'+row['target_id']+'.npz','Actual checkpoint-specific prediction name')
  with np.load(p,allow_pickle=False) as z:require(z.files==['native'],'Only actual native payload')
  archives.append(A.read_prediction(p,f['prediction_file_sha256'],payload['prediction_sha256']))
  predictions.append({'checkpoint_step':271482,'target_id':row['target_id'],'layout':'native','array_sha256':payload['prediction_sha256']})
  evidence.append({'checkpoint_step':271482,'target_id':row['target_id'],'sequence':row['sequence'],'input_evidence':f['input_evidence']})
 require(d['data_evidence']==evidence,'All actual eight input evidence')
 require(len({v['file'] for v in archives})==8 and {p.name for p in path.parent.glob('*.npz')}=={Path(v['file']).name for v in archives}|{target.name} and not list(path.parent.glob('*.pt')),'Eight quality/oneBN/no extra state artifacts')
 for k in ('all_states_unchanged','BN_buffers_unchanged','all_hooks_removed','original_modules_restored'):require(d[k] is True,'Global closure '+k)
 alloc=d['max_allocated_bytes'];reserved=d['max_reserved_bytes'];require(type(alloc) is int and type(reserved) is int and 0<alloc<=reserved,'Actual allocator resources')
 require(not torch.cuda.is_initialized(),'CPU-only independent audit')
 return {'schema':'flow_selected16_round15_independent_execution_audit_v1','status':'PASS','auditor_sha256':sha(__file__),
  'source_result_sha256':sha(path),'source_result_path':str(path),'full_forwards':8,'optimizer_updates':0,'calibration_forwards':0,'dual_forwards':0,'FF_forwards':0,
  'quality_prediction_count':8,'quality_archive_count':8,'retained_BN_archive_count':1,'BN_scope':SCOPE,
  'checkpoint_sha256':CP,'source_state_sha256':SOURCE,'non_BN_state_sha256':CURRENT,'selected16_state_sha256':SELECTED,'state_tensors':525,'state_elements':54944214,
  'fixed_BN_count':16,'current_BN_count':62,'retained_BN_buffer_count':48,'retained_BN_values_match_own_source':True,'retained_BN_archive_sha256':sha(target),
  'own_BN_source_sha256':BN_SHA,'own_BN_receipt_sha256':BN_RECEIPT_SHA,'references':references,'quality_predictions':predictions,'quality_archives':archives,'cached_native_archives':cached_archives,
  'executed_FC1_rows':totals['FC1_rows'],'executed_FC2_rows':totals['FC2_rows'],'service_totals_from_actual_records':dict(totals),
  'all_states_unchanged':True,'BN_buffers_unchanged':True,'all_hooks_removed':True,'original_modules_restored':True,
  'resource_ledger':{'device':d['runtime']['device'],'wall_seconds':d['wall_seconds'],'max_allocated_bytes':alloc,'max_reserved_bytes':reserved},
  'GPU_used':False,'SSH_used':False,'quality_metrics_read_or_ranked':False,
  'limitations':['Actual original PT, own72 BN, saved48 BN and eight raw predictions independently read;525 full state reconstructed and matched to runtime before/after digests. No duplicate endpoint PT.',
   'Intermediate4-head/8-branch tensors checked by metadata/SHA, no CPU inference replay; all78 BN input shapes and calls checked against independently pinned full9 graph.',
   'Original input arrays not reread here, exact SHA/member order and cached raw prediction identities checked. Raw GT quality is another independent audit.',
   'Native8 only; no FF or compressed action verified, no H0 admission. Geometry/host timing/allocator values are not cycles, energy or flow-specific novelty.']}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','audit'),default='prepare');p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',type=Path);p.add_argument('--result',type=Path);a=p.parse_args()
 require(not a.output.exists(),'No audit overwrite')
 if a.stage=='prepare':
  m=read(R13/'data_protocol/manifest_v1.json');G.memberships(m);fixed,shapes=names_and_shapes()
  result={'schema':'flow_selected16_round15_independent_execution_freeze_v1','status':'PREPARED_BEFORE_ACTUAL_RESULT_READ','auditor_sha256':sha(__file__),
   'source_sha256':source_pins(),'forward_schedule':schedule(m),'full_forwards':8,'optimizer_updates':0,'calibration_forwards':0,'dual_forwards':0,'FF_forwards':0,
   'quality_prediction_count':8,'quality_archive_count':8,'retained_BN_archive_count':1,'expected_FC_rows_each':4464000,
   'selected16_state_sha256':SELECTED,'BN_input_shapes':shapes,'frozen_BN_names':sorted(fixed),'runner_modules_imported':False,'quality_metrics_read_or_ranked':False}
 else:
  require(a.freeze and a.result,'Explicit frozen sources/actual result required');f=read(a.freeze)
  require(f['auditor_sha256']==sha(__file__) and f['source_sha256']==source_pins(),'Unchanged pre-result auditor/source freeze')
  result=audit(a.result);result['audit_freeze_sha256']=sha(a.freeze)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':result['status'],'schema':result['schema'],'output_sha256':sha(a.output)}))
if __name__=='__main__':main()
