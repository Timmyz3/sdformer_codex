#!/usr/bin/env python3
"""Independent CPU audit of 271482 own-BN fixed193 deployment; no runner import/AEE."""
import argparse
from collections import Counter
import hashlib,importlib.util,json,math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;RUN=HERE.parent;DATE=RUN.parent;REPO=DATE.parent.parent
R13=DATE/'coverage_recovery_round13';VERIFY=DATE/'baseline_checkpoint_verification'
PT=VERIFY/'source_snapshot/checkpoint_step_000271482.pt'
CP_SHA='1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'
SOURCE_SHA='ad14a7d45091aa75d64df83e1a71bd89b35dfbbf49c9ab84861d5ae0aa270560'
NON_BN_SHA='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
OLD=R13/'independent_execution_audit/audit_execution.py'
def require(v,msg):
 if not v:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
require(sha(OLD)=='dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8','Frozen independent geometry checker changed')
spec=importlib.util.spec_from_file_location('_independent_geometry_only',OLD);G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
SUFFIX=('running_mean','running_var','num_batches_tracked')
OWN=('run_deployment.py','deployment_core.py','test_deployment_cpu.py','source_pins.json','PROTOCOL.md')
def local(parent,name):
 require(isinstance(name,str) and name==Path(name).name and name not in ('','..','.'),'Artifact basename only')
 return parent/name

def source_pins():
 rootfreeze=RUN/'EXECUTION_FREEZE_V1.json'
 require(sha(rootfreeze)=='e2c371616efef1d120b10c2407677c896aad9b2fe69f69fa52160d5f0a5ffe1c','Root fixed execution freeze unchanged')
 for p,digest in read(rootfreeze)['source_sha256'].items():require(sha(p)==digest,'Root frozen source bytes '+p)
 paths=[RUN/n for n in OWN]+[RUN/'cpu_results.json',HERE/'EXECUTION_AUDIT_INPUT_SPEC_V1.md',HERE/'LATER_SOURCE_TENSOR_AUDIT_V1.json',OLD,
  R13/'data_protocol/manifest_v1.json',R13/'runtime_v1.json',R13/'preparation/regions_v1.json',R13/'preparation/bn_receipt.json',
  R13/'raw/experiments_v1/diversity/recover/native/results.json',VERIFY/'run_contract.json',VERIFY/'checkpoint_step_000271482.json']
 for relative,digest in read(RUN/'source_pins.json')['sources'].items():
  p=REPO/relative;require(sha(p)==digest,'Inherited source bytes '+relative);paths.append(p)
 result={str(p):sha(p) for p in paths};result.update(read(rootfreeze)['source_sha256']);result[str(rootfreeze)]=sha(rootfreeze)
 return result

def schedule(m):
 rows=[{'phase':'calibration','target_id':r['target_id'],'layout':'native','mode':'actual'} for r in m['bn_members']]
 for i,r in enumerate(m['check_members']):
  for arm in ('native','FF'):
   x={'phase':'evaluation','target_id':r['target_id'],'layout':arm,'mode':'actual'};rows.append(x)
   if i==0 and arm=='FF':rows.append(dict(x,mode='dual_reference'))
 require(len(rows)==193,'Independent193 schedule')
 return [dict(r,forward_id=i+1) for i,r in enumerate(rows)]

def original_state():
 import torch
 require(not torch.cuda.is_initialized() and sha(PT)==CP_SHA,'CPU actual specified checkpoint')
 p=torch.load(PT,map_location='cpu',weights_only=True);s=p['model_state_dict']
 require(p['training']['completed_optimizer_steps']==271482 and p['contract']==read(VERIFY/'run_contract.json')['contract'],'Actual source state/contract')
 require(len(s)==711 and sum(t.numel() for t in s.values())==55001588,'Source tensor inventory')
 require(all(t.device.type=='cpu' and bool(torch.isfinite(t).all()) for t in s.values()),'Finite source tensors')
 require(G.tensor_digest(s)==SOURCE_SHA,'Actual original complete state digest')
 names={n.rsplit('.',1)[0] for n in s if n.endswith(tuple('.'+x for x in SUFFIX))}
 buffers={n+'.'+k for n in names for k in SUFFIX}
 require(len(names)==78 and len(buffers)==234 and G.tensor_digest(s,buffers)==NON_BN_SHA,'Actual nonBN source state digest')
 private={n+'.bn'+str(i)+'.norm_layer' for n in G.GEOMETRY for i in (1,2)}
 require(len(private)==24 and private<=names,'True private24 source names')
 return s,names,private

def verify_bn(root,d,m,state,names,private):
 import numpy as np
 import torch
 info=d['own_BN_receipt'];path=local(root,info['path']);require(sha(path)==info['sha256'],'Actual own BN receipt SHA')
 b=read(path);valuepath=local(root,d['own_BN_buffers']['path']);vsha=sha(valuepath)
 require(vsha==d['own_BN_buffers']['sha256']==b['buffers_sha256'],'Actual own BN arrays SHA')
 require(b['schema']=='flow_later_checkpoint_own_bn_v1' and b['status']=='OWN_BN_COMPLETE','Own BN status')
 require(b['checkpoint_sha256']==CP_SHA and b['checkpoint_step']==271482 and b['source_checkpoint_state_sha256']==SOURCE_SHA,'Own checkpoint BN source')
 require(b['membership_sha256']==sha(R13/'data_protocol/manifest_v1.json') and b['runtime_manifest_sha256']==d['binding']['runtime_manifest_sha256'],'Own BN identities/version')
 require(b['source_sha256']==d['binding']['source_sha256'],'Own BN source code version')
 require(b['bn_target_ids']==[r['target_id'] for r in m['bn_members']],'Actual176 identity sequence')
 require(b['BN_scope']=='MLP_271482_OWN_CMA176' and b['calibration_definition']=='all78_reset_CMA_train_nonBN_eval','Own all78 reset CMA semantics')
 require(b['calibration_forwards']==176 and b['calibration_bn_count']==78 and b['retained_mlp_bn_count']==24,'BN counts')
 require(b['old_CMA_loaded'] is False and b['GT_used'] is False and b['optimizer_steps']==0,'Zero training/input-only BN')
 require(b['non_bn_state_before_sha256']==b['non_bn_state_after_sha256']==NON_BN_SHA and b['non_bn_state_unchanged'] is True,'Actual source nonBN digest binding')
 require(set(b['retained_names'])==private,'Private24 true names')
 require(set(b['all78_reset_counters'])==names and set(b['all78_reset_counters'].values())=={0},'All78 reset counters')
 require(b['input_evidence']==read(R13/'preparation/bn_receipt.json')['input_evidence'],'Same legal actual176 event SHA/order, not old BN values')
 reset=b['all78_reset_signature'];require(set(reset)==names,'Reset78 signature coverage')
 for name in names:
  r=reset[name];require(r['training'] is True and r['tracking'] is True,'All78 calibration BN flags')
  for field in SUFFIX:
   x=state[name+'.'+field].numpy();expected=np.ones_like(x) if field=='running_var' else np.zeros_like(x)
   require(r[field]==hashlib.sha256(expected.tobytes()).hexdigest(),'Independent reset array bytes')
 with np.load(valuepath,allow_pickle=False) as z:
  require(set(z.files)=={n+'.'+k for n in private for k in SUFFIX},'Exact own72 array keys')
  vals={k:z[k].copy() for k in z.files}
 for key,x in vals.items():
  original=state[key].numpy();require(x.shape==original.shape and x.dtype==original.dtype and np.isfinite(x).all(),'True native BN shape/dtype/finite')
  if key.endswith('.num_batches_tracked'):require(int(x)==176,'Private counter176')
  if key.endswith('.running_var'):require((x>=0).all(),'Nonnegative BN variance')
 deployed={n:t for n,t in state.items() if n not in {x+'.'+k for x in names-private for k in SUFFIX}}
 deployed.update({k:torch.from_numpy(v.copy()) for k,v in vals.items()})
 digest=G.tensor_digest(deployed)
 require(d['deployment_initial_state_sha256']==d['deployment_final_state_sha256']==digest,'CPU rebuilt source+ownBN deployment digest')
 require(len(deployed)==549 and sum(t.numel() for t in deployed.values())==54963422,'Deployment actual state geometry')
 scope=d['BN_scope'];require(scope['scope']=='MLP_271482_OWN_CMA176' and scope['BN_count']==78 and scope['MLP_BN_count']==24,'Deployment BN scope')
 require(set(scope['frozen_BN_names'])==private and set(scope['current_stat_BN_names'])==names-private and scope['old_CMA_loaded'] is False,'Disjoint24/54 sets')
 sig=scope['BN_signature_before_inference'];require(set(sig)==names,'Deployment78 signatures')
 for name in names:
  require(sig[name]['training'] is False and sig[name]['tracking']==(name in private),'Deployment eval/current-stat flags')
  for field in SUFFIX:
   expected=hashlib.sha256(vals[name+'.'+field].tobytes()).hexdigest() if name in private else None
   require(sig[name][field]==expected,'True private72/current54 values')
 return {'BN_buffers_sha256':vsha,'BN_receipt_sha256':sha(path),'deployment_state_sha256':digest,'state_tensors':len(deployed),'state_elements':sum(t.numel() for t in deployed.values())}

def verify_records(d,m,bnnames):
 expected=schedule(m);actual=d['forward_records'];require(len(actual)==d['full_forward_count']==193,'Actual193 records')
 counts=Counter();lookup={}
 for r,w in zip(actual,expected):
  require(all(r[k]==v for k,v in w.items()),'Exact record order/identity/layout/mode')
  ex=r['execution'];dual=w['mode']=='dual_reference'
  if w['phase']=='calibration':
   require(set(ex['all12_observed_fc_rows'])==set(G.GEOMETRY),'Calibration actual12 FC hooks')
   total=0
   for n in G.GEOMETRY:
    wanted=G.expected_layer(n,'native')[0]
    require(ex['all12_observed_fc_rows'][n]=={'fc1_rows':wanted,'fc2_rows':wanted},'Calibration perFC geometry')
    total+=wanted
   require(ex['summed_fc1_rows']==ex['summed_fc2_rows']==total==558000,'Calibration row total')
   require(ex['gradient_enabled'] is False and ex['BN_mode']=='all78_reset_CMA_train_nonBN_eval','Calibration zero-gradient semantics')
   require(set(ex['all78_num_batches_tracked'])==bnnames and set(ex['all78_num_batches_tracked'].values())=={w['forward_id']},'Each native call all78 once')
   counts.update(FC1_rows=total,FC2_rows=total,observed_FC_modules=24)
  else:
   counts.update(G.check_execution(ex,w['layout'],dual,'evaluation'))
   require(ex['anchor']==w['layout'] and ex['mode']==w['mode'],'Evaluation anchor/source mode')
   require(set(ex['branch_records'])==G.SELECTED,'Eight original branch captures')
   for n,rbranch in ex['branch_records'].items():
    require(rbranch['shape']==list(G.GEOMETRY[n]) and all(G.hexdigest(rbranch[k]) for k in ('input_sha256','output_sha256')),'True branch geometry/digest')
  heads=ex['head_records'];require(len(heads)==4,'All4 head records')
  require(all(h['shape']==[1,2,480,640] and h['dtype']=='torch.float32' and h['finite'] is True for h in heads),'Four native fullresolution finite heads')
  lookup[(w['phase'],w['target_id'],w['layout'],w['mode'])]=ex
 require(counts['FC1_rows']==counts['FC2_rows']==107745600,'Independent193 aggregate FC rows')
 return counts,lookup

def verify_raw_pair(root,d,m,lookup):
 import numpy as np
 pair=d['first_frame_FF_dual'];first=m['check_members'][0]['target_id'];loaded=[];artifacts=[]
 for side,mode in [('actual','actual'),('dual','dual_reference')]:
  ex=lookup[('evaluation',first,'FF',mode)];require(pair[side+'_execution']==ex,'Raw pair actual forward binding')
  meta=pair[side+'_raw'];path=local(root,meta['file']);require(sha(path)==meta['file_sha256'],'Raw pair archive SHA')
  names=sorted(G.SELECTED);require(meta['branch_names_in_index_order']==names,'Raw8 fixed native names')
  expected={f'flow_{i}' for i in range(4)}|{f'{kind}_{i}' for kind in ('input','branch') for i in range(8)}
  require(set(meta['arrays'])==expected,'Raw20 metadata coverage')
  with np.load(path,allow_pickle=False) as z:
   require(set(z.files)==expected,'Raw20 payload coverage');arrays={k:z[k].copy() for k in z.files}
  for key,a in arrays.items():
   rec=meta['arrays'][key];require(a.dtype==np.float32 and np.isfinite(a).all(),'Raw FP32 finite')
   digest=hashlib.sha256(a.tobytes()).hexdigest();require(rec=={'shape':list(a.shape),'dtype':'float32','payload_sha256':digest},'Actual raw array SHA/shape/dtype')
   kind,idx=key.rsplit('_',1);idx=int(idx)
   if kind=='flow':
    require(a.shape==(1,2,480,640) and digest==ex['head_records'][idx]['sha256'],'Raw4 true head geometry/binding')
   else:
    name=names[idx];require(a.shape==G.GEOMETRY[name],'Raw native input/branch geometry')
    field='input_sha256' if kind=='input' else 'output_sha256'
    require(digest==ex['branch_records'][name][field],'Raw8 true source binding')
    if kind=='input':require(digest==ex['layers'][name]['input_sha256'],'Raw prefix wrapper binding')
  loaded.append(arrays);artifacts.append({'file':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
 require(all(np.array_equal(loaded[0][k],loaded[1][k]) for k in loaded[0]),'Independent20 raw array actual-dual exact equality')
 require(all(pair[k] is True for k in ('all4_flow_heads_exact','all8_inputs_exact','all8_branches_exact')),'Raw pair recorded verdict agrees')
 final=hashlib.sha256(loaded[0]['flow_3'][0].tobytes()).hexdigest()
 return artifacts,final

def verify_predictions(root,d,m,lookup,first_ff):
 import numpy as np
 frames=d['evaluations']['endpoint']['frames'];members=m['check_members']
 require(set(d['evaluations'])=={'endpoint'} and [x['target_id'] for x in frames]==[r['target_id'] for r in members],'All8 ordered quality members')
 baseline=read(R13/'raw/experiments_v1/diversity/recover/native/results.json')['evaluations']['initial']['frames']
 artifacts=[];preds=[]
 for f,row,old in zip(frames,members,baseline):
  require(f['sequence']==row['sequence'] and f['input_evidence']==old['input_evidence'],'Same actual GT/input/mask source binding')
  require(set(f['arms'])=={'native','FF'},'Exactly2 quality layouts')
  path=local(root,f['prediction_file']);require(sha(path)==f['prediction_file_sha256'],'Quality actual archive SHA')
  with np.load(path,allow_pickle=False) as z:
   require(set(z.files)=={'native','FF'},'Two actual quality payload keys')
   for arm in ('native','FF'):
    a=z[arm];require(a.dtype==np.float32 and a.shape==(2,480,640) and np.isfinite(a).all(),'Quality prediction true geometry/finite')
    digest=hashlib.sha256(a.tobytes()).hexdigest();payload=f['arms'][arm]
    require(digest==payload['prediction_sha256'] and payload['execution']==lookup[('evaluation',row['target_id'],arm,'actual')],'Quality actual forward/payload binding')
    require(digest==payload['execution']['head_records'][3]['sha256'],'Final flow3 quality payload identity')
    if f is frames[0] and arm=='FF':require(digest==first_ff,'First actual/dual raw final equals quality FF')
    preds.append({'target_id':row['target_id'],'layout':arm,'array_sha256':digest})
  artifacts.append({'file':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
 require(len(artifacts)==8 and len(preds)==16,'16 quality predictions/8 archives')
 return artifacts,preds

def audit(path,frozen):
 import torch
 m=read(R13/'data_protocol/manifest_v1.json');G.memberships(m);d=read(path);b=d['binding']
 require(d['schema']=='flow_later_checkpoint_deployment_v1' and d['status']=='LATER_CHECKPOINT_193_FORWARD_COMPLETE','Complete actual result required')
 require(d['training_performed'] is False and d['optimizer_steps']==0 and d['inference_executed'] is True,'Zero training actual193 inference')
 require(d['hardware_benefit_claim'] is False and d['flow_specific_increment_claim'] is False,'No unsupported scientific claims')
 require(d['runtime']['AMP'] is False and d['runtime']['TF32'] is False,'FP32 runtime')
 require(b['source_sha256']=={n:sha(RUN/n) for n in OWN} and b['cpu_checks_sha256']==sha(RUN/'cpu_results.json'),'Frozen actual source bindings')
 require(b['checkpoint_sha256']==CP_SHA and b['checkpoint_step']==271482 and b['contract_sha256']=='961f1700e76f3427c74d4e9af9799a684433ff00449b8386be6c1f9cdd75e934','Specified later source binding')
 require(b['run_contract_sha256']==sha(VERIFY/'run_contract.json') and b['sidecar_sha256']==sha(VERIFY/'checkpoint_step_000271482.json'),'Actual contract/sidecar binding')
 require(b['BN_scope']=='MLP_271482_OWN_CMA176' and b['precision']=='FP32/noAMP/noTF32/T10/allC/full9','Fixed deployed BN/precision')
 require(b['membership_sha256']==sha(R13/'data_protocol/manifest_v1.json') and b['region_sha256']==sha(R13/'preparation/regions_v1.json'),'Fixed membership/regions')
 require(b['runtime_manifest_sha256']==sha(R13/'runtime_v1.json'),'Same actual runtime manifest')
 require(b['membership_origin_checkpoint_sha256']==G.ORIGINAL_SHA,'Preserved cohort origin distinct from candidate model')
 require(b['bn_ids']==[r['target_id'] for r in m['bn_members']] and b['check_ids']==[r['target_id'] for r in m['check_members']],'Exact source member order')
 require(b['forward_plan']==schedule(m),'Independent complete planned schedule')
 for k,v in {'calibration_forwards':176,'evaluation_actual_forwards':16,'dual_forwards':1,'total_full_forwards':193,'optimizer_steps':0,'external_BN_current_stat':54,'private_BN_frozen':24}.items():require(b[k]==v,'Bound fixed budget '+k)
 require(b['old_BN_loaded'] is False and b['calibration_GT_used'] is False,'Own input-only statistics')
 previous_bn=read(R13/'preparation/bn_receipt.json')['input_evidence']
 previous_check=read(R13/'raw/experiments_v1/diversity/recover/native/results.json')['evaluations']['initial']['frames']
 expected_evidence=[{'phase':'calibration','target_id':row['target_id'],'sequence':row['sequence'],'files':{'input':e['sha256']}} for row,e in zip(m['bn_members'],previous_bn)]
 expected_evidence.extend({'phase':'evaluation','target_id':row['target_id'],'sequence':row['sequence'],'files':{'input':e['input_evidence']['events'],'gt':e['input_evidence']['gt'],'mask':e['input_evidence']['mask']}} for row,e in zip(m['check_members'],previous_check))
 require(d['data_evidence']==expected_evidence,'All184 preflight actual source bindings')
 require(0<b['wall_minutes']<=30 and 0<d['wall_seconds']<=1800,'Fixed bounded wall time')
 for k in ('non_bn_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed'):require(d[k] is True,'Closed execution '+k)
 state,names,private=original_state();require(d['source_checkpoint_state_sha256']==SOURCE_SHA,'GPU loaded actual CPU source state')
 cpu=d['checkpoint_cpu_audit'];require(cpu['checkpoint_step']==271482 and cpu['model_tensors']==711 and cpu['model_elements']==55001588 and cpu['all_model_tensors_CPU_finite'] is True and cpu['full_source_state_sha256']==SOURCE_SHA,'Runtime CPU source audit agrees with independent actual PT')
 bn=verify_bn(path.parent,d,m,state,names,private);counts,lookup=verify_records(d,m,names)
 raw,first=verify_raw_pair(path.parent,d,m,lookup);archives,predictions=verify_predictions(path.parent,d,m,lookup,first)
 expectedfiles={Path(x['file']).name for x in raw+archives}|{d['own_BN_buffers']['path']}
 require({p.name for p in path.parent.glob('*.npz')}==expectedfiles,'No unreported NPZs')
 require(not list(path.parent.glob('*.pt')),'No unreported training-state PT')
 alloc=d['max_allocated_bytes'];reserved=d['max_reserved_bytes'];require(type(alloc) is int and type(reserved) is int and 0<alloc<=reserved,'Real allocator peak contract')
 if 'calibration_seconds' in d or 'evaluation_seconds' in d:
  require(all(G.finite(d[k]) and d[k]>=0 for k in ('calibration_seconds','evaluation_seconds')) and d['calibration_seconds']+d['evaluation_seconds']<=d['wall_seconds']+1e-6,'Phase clocks within entry')
 require(not torch.cuda.is_initialized(),'Audit remained CPU')
 return {'schema':'flow_later_checkpoint_independent_execution_audit_v1','status':'PASS','auditor_sha256':sha(__file__),
  'source_result_sha256':sha(path),'source_result_path':str(path),'checkpoint_sha256':CP_SHA,**bn,
  'full_forwards':193,'optimizer_updates':0,'calibration_forwards':176,'quality_prediction_count':16,'quality_archive_count':8,'dual_forward_count':1,
  'source_state_independently_reconstructed':True,'non_bn_weights_unchanged':True,'BN_state_closure':True,'actual_dual_raw_arrays_exact':True,
  'source_state_sha256':SOURCE_SHA,'source_non_BN_state_sha256':NON_BN_SHA,'service_totals_from_actual_records':dict(counts),
  'quality_archives':archives,'quality_predictions':predictions,'dual_raw_archives':raw,'dual_raw_arrays':40,
  'resource_ledger':{'device':d['runtime']['device'],'wall_seconds':d['wall_seconds'],'max_allocated_bytes':alloc,'max_reserved_bytes':reserved,
   'calibration_seconds':d.get('calibration_seconds'),'evaluation_seconds':d.get('evaluation_seconds')},
  'GPU_used':False,'SSH_used':False,'quality_metrics_read_or_ranked':False,
  'limitations':['Calibration input arrays not reread here; accepted event SHA/order bound to prior audited176 source cohort.',
   'Source PT and new72 BN values directly read to reconstruct full deployment state; runtime before/after state digests match, no redundant endpoint PT saved.',
   'Calibration statistic arithmetic not replayed on CPU; all78 reset flags, native176 call counters, source states and output buffers verified.',
   'Full4/eight prefix/branch raw actual-dual arrays directly read and compared; other evaluation intermediate tensors checked by record digests.',
   'Rows/allocator/host timing are not hardware cycles, energy or paper-quality evidence.']}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','audit'),default='prepare');p.add_argument('--output',type=Path,required=True)
 p.add_argument('--freeze',type=Path);p.add_argument('--result',type=Path);a=p.parse_args();require(not a.output.exists(),'No audit artifact overwrite')
 if a.stage=='prepare':
  m=read(R13/'data_protocol/manifest_v1.json');G.memberships(m)
  require(sum(G.expected_layer(n,'native')[0] for n in G.GEOMETRY)==558000 and sum(G.expected_layer(n,'FF',True)[0] for n in G.GEOMETRY)==609600,'True native/dual FC geometry')
  result={'schema':'flow_later_checkpoint_independent_execution_freeze_v1','status':'PREPARED_BEFORE_ACTUAL_RESULT_READ','auditor_sha256':sha(__file__),
   'source_sha256':source_pins(),'checkpoint_sha256':CP_SHA,'forward_schedule':schedule(m),'full_forwards':193,'optimizer_updates':0,
   'expected_FC_rows_each':107745600,'quality_archive_count':8,'quality_prediction_count':16,'dual_raw_archive_count':2,'dual_raw_arrays':40,
   'runner_modules_imported':False,'quality_metrics_read_or_ranked':False}
 else:
  require(a.freeze and a.result,'Explicit frozen audit and actual result required');frozen=read(a.freeze)
  require(frozen['auditor_sha256']==sha(__file__) and frozen['source_sha256']==source_pins(),'Actual audit source freeze unchanged')
  result=audit(a.result,frozen);result['pre_result_freeze_sha256']=sha(a.freeze)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':result['status'],'schema':result['schema']}))
if __name__=='__main__':main()
