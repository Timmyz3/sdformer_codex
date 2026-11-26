#!/usr/bin/env python3
"""Fixed R28 Q4-recovered W evaluated in all-Q8 mode: 3 parity + 64 dev.

No optimizer, teacher, training, GT-array read or task-quality admission. The
original loader constructs the graph; every forward follows strict final-W load.
"""
import argparse,copy,fcntl,hashlib,importlib.util,json,signal,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
SMOKE_SHA='6dadb497bf97fc33929209a3feed0acedcac241036d27bbcda502d8f35244d83'
CORE_SHA='f4e4dae18a7e59fab286f2e59f23f72d20583a2b8de22dee4c78204bbfd2fe9a'
EVAL_SHA='5a5bf2abaca8fe23fc269e4dc830acbd3ee6537e52958c29c1520543756a4f95'
AUDITOR_SHA='eb433542a13b6f2df185b503b922ab98fe648d9e69c9d51b956eb5e8654df71b'
MEMBERS_SHA='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'
ORIGINAL_STATE='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
SOURCE_GRAPH='flow_r28_all12mlp_patch4conv_integer_Q4_complete9_tailQ8_all78current_v1'
GRAPH='flow_r29_same_Q4_recovered_weights_all_Q8_all78current_v1'
SCHEMA='flow_r29_same_weight_q8_evaluation_v1'
STAGE='diagnostic'

def require(ok,why):
 if not ok:raise ValueError(why)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def state_digest(state):
 h=hashlib.sha256()
 for name,x in sorted(state.items()):
  y=x.detach().cpu().contiguous();h.update(name.encode());h.update(str(y.dtype).encode());h.update(str(tuple(y.shape)).encode());h.update(y.numpy().tobytes())
 return h.hexdigest()

def validate_audit(audit):
 require(audit['schema']=='flow_r28_independent_actual_execution_audit_v1' and audit['status']=='PASS','Completed independent R28 actual execution audit')
 require(audit['graph_id']==SOURCE_GRAPH and audit['auditor_sha256']==AUDITOR_SHA,'Frozen R28 executed graph and auditor')
 require(set(audit['arms'])==set(audit['source_results'])=={'Q4'},'Only original R28 Q4 recovery source')
 require(audit['counts']==dict(optimizer_updates=512,quality_predictions=128,quality_archives=128,formal_model_forwards=1155),'Complete R28 actual execution counts')
 arm=audit['arms']['Q4']
 require(arm['updates']==512 and arm['model_forwards']==1155 and arm['allowed_parameters']==477 and arm['actual_optimizer_parameters']==453,'Fixed final Q4 training execution')
 require(arm['result_sha256']==audit['source_results']['Q4'],'Actual result identity bound')
 return arm

def validate_checkpoint(cp,audit,weights_sha,torch):
 arm=validate_audit(audit)
 require(weights_sha==arm['checkpoint_sha256'],'Only audited final Q4 checkpoint bytes')
 require(cp['schema']=='flow_r28_fixed512_checkpoint_v1' and cp['arm']=='Q4' and cp['completed_optimizer_steps']==512,'Exact fixed final Q4 checkpoint, no other arm/smoke')
 require(cp['initial_state_sha256']==ORIGINAL_STATE and cp['members_sha256']==MEMBERS_SHA and cp['fixed_endpoint_no_development_selection'] is True,'Original source and fixed member/endpoint contract')
 require(cp['sources']['runner']==EVAL_SHA and cp['sources']['smoke_helpers']==SMOKE_SHA and cp['sources']['core']==CORE_SHA and cp['sources']['members']==MEMBERS_SHA,'Executed checkpoint source versions')
 state=cp['model_state_dict']
 require(len(state)==477 and all(isinstance(v,torch.Tensor) and v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in state.values()),'Actual finite477 FP32 parameters')
 require(not any(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in state),'All78 current-stat BN; no running buffers')
 digest=state_digest(state)
 require(digest==cp['final_state_sha256'] and digest!=ORIGINAL_STATE,'Actual recovered state digest')
 return state,dict(sha256=weights_sha,state_sha256=digest,source_result_sha256=audit['source_results']['Q4'],source_arm='Q4',source_updates=512)

def prepare(a):
 import torch
 require(not torch.cuda.is_initialized() and not a.output.exists(),'CPU-only preflight and fresh output')
 require(sha(__file__)==a.runner_sha256,'Explicit R29 runner source')
 helper=a.r28_root/'implementation'
 for name,digest in [('run_smoke_v2.py',SMOKE_SHA),('run_recovery_v2.py',EVAL_SHA),('recovery_integer_core.py',CORE_SHA)]:
  require(sha(helper/name)==digest,'Frozen R28 helper '+name)
 require(sha(a.r28_root/'audit_actual_recovery_v1.py')==AUDITOR_SHA and sha(a.r28_audit)==a.r28_audit_sha256,'Actual R28 audit source and receipt bytes')
 audit=read(a.r28_audit);validate_audit(audit)
 weights_sha=sha(a.weights);require(weights_sha==audit['arms']['Q4']['checkpoint_sha256'],'Actual audited final PT before load')
 cp=torch.load(a.weights,map_location='cpu',weights_only=True)
 state,provenance=validate_checkpoint(cp,audit,weights_sha,torch);source_recipe=cp['recipe'];del cp
 provenance.update(path=str(a.weights),audit_path=str(a.r28_audit),audit_sha256=a.r28_audit_sha256)
 smoke=load(helper/'run_smoke_v2.py','_r29_frozen_smoke');evaluation=load(helper/'run_recovery_v2.py','_r29_frozen_evaluation')
 core=load(helper/'recovery_integer_core.py','_r29_frozen_integer_core')
 old=copy.copy(a);old.runner_sha256=SMOKE_SHA
 p,m,first,order=smoke.prepare(old);evaluation.validate_order(m)
 require(source_recipe==evaluation.recipe(smoke),'Unchanged R28 recovery recipe on source checkpoint')
 return smoke,evaluation,core,p,m,first,state,provenance

def execute(a,smoke,evaluation,core,p,m,first,state,result):
 import torch,numpy as np
 require(not torch.cuda.is_initialized(),'Global GPU lock precedes CUDA')
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261019)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_float32_matmul_precision('highest')
 device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261019);torch.cuda.reset_peak_memory_stats()
 start=time.monotonic();r13=p['modules'][0]
 model,functional,bn=smoke.make_model(p,'full9',device,torch)
 require(r13.state_digest(model)==ORIGINAL_STATE,'Original graph construction state, before replacement')
 require(set(model.state_dict())==set(state),'Same exact original parameter names before strict final-W load')
 model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
 final=r13.state_digest(model);require(final==result['source_Q4_checkpoint']['state_sha256'],'Every forward uses audited Q4-recovered W')
 signature=p['base'].bn_signature(model,torch)
 require(sum(isinstance(x,torch.nn.modules.batchnorm._BatchNorm) for x in model.modules())==78,'Same all78 BN graph')
 require(all(x.running_mean is None and x.running_var is None and x.num_batches_tracked is None and not x.track_running_stats for x in model.modules() if isinstance(x,torch.nn.modules.batchnorm._BatchNorm)),'No source/frozen BN running statistics used')
 result.update(status='RUNNING',state_sha256=final,initial_state_sha256=final,BN_scope_from_original_graph_construction=bn,
  original_graph_state_before_final_reload=ORIGINAL_STATE,runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),AMP=False,TF32=False))
 evidence=evaluation.source_evidence(m,first['target_id']);events=evaluation.event_only(p,first,evidence,np)
 chunk=p['base'].prepare_input(events,device,torch);del events
 result['parity_input']=dict(target_id=first['target_id'],sequence=first['sequence'],input_evidence=evidence,full_geometry=True)
 reference,refrecord=smoke.capture_full(model,functional,chunk,'Q8',core,torch,ste=True);result['model_forwards']+=1
 actual,actrecord=smoke.capture_full(model,functional,chunk,'Q8',core,torch,ste=False);result['model_forwards']+=1
 parity=smoke.compare_full(actual,reference,torch);del actual
 result['final_training_deployment_parity']=dict(**parity,reference_record=refrecord,actual_record=actrecord)
 write(a.output/'results.json',result);require(parity['pass'],'Same-W Q8 STE/deploy exact integer/PSN/head parity')
 dense,denserecord=smoke.capture_full(model,functional,chunk,'Q8',core,torch,ste=False,reference=True);result['model_forwards']+=1
 parity=smoke.compare_full(dense,reference,torch);del dense
 result['final_dense_integer_parity']=dict(**parity,dense_record=denserecord)
 write(a.output/'results.json',result);require(parity['pass'],'Same-W Q8 dense integer/deploy exact parity')
 parity_path=a.output/'final_parity_heads.npz'
 np.savez_compressed(parity_path,**{k:v.numpy() for k,v in reference.items() if k.startswith('head')})
 result.update(parity_heads_file=parity_path.name,parity_heads_sha256=sha(parity_path))
 del reference,chunk;torch.cuda.empty_cache()
 evaluation.evaluate(a,p,m,smoke,core,model,functional,STAGE,torch,np,result)
 require(result['model_forwards']==67 and len(result['evaluations'][STAGE]['frames'])==64,'Exactly3 parity plus64 development forwards')
 require(r13.state_digest(model)==final and p['base'].bn_signature(model,torch)==signature,'Same recovered state/BN after all67 forwards')
 result.update(status='SAME_Q4_WEIGHT_Q8_EVALUATION_COMPLETE',final_state_sha256=final,wall_seconds=time.monotonic()-start,
  max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),
  work_budget=dict(serial_model_forwards=66,dense_reference_forwards=1,serial_row_planes=2211264000,dense_logical_target_rows=4188000,BN_calls=5226,PSN_calls=7035))

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--r28-root',type=Path,required=True);parser.add_argument('--r26-root',type=Path,required=True)
 parser.add_argument('--members',type=Path,required=True);parser.add_argument('--r28-audit',type=Path,required=True)
 parser.add_argument('--r28-audit-sha256',required=True);parser.add_argument('--weights',type=Path,required=True)
 parser.add_argument('--runner-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--preflight-output',type=Path);parser.add_argument('--execute',action='store_true')
 parser.add_argument('--wall-minutes',type=float,default=20)
 a=parser.parse_args();a.arm='Q8';require(a.wall_minutes==20,'Fixed20 minute evaluation-only cap')
 smoke,evaluation,core,p,m,first,state,provenance=prepare(a)
 result=dict(schema=SCHEMA,graph_id=GRAPH,status='PREFLIGHT_PASS',arm='Q8',state_label='Q8_on_recovered_Q4',source_Q4_checkpoint=provenance,
  sources=dict(runner=sha(__file__),smoke_helpers=SMOKE_SHA,evaluation_helpers=EVAL_SHA,core=CORE_SHA,members=MEMBERS_SHA,R28_auditor=AUDITOR_SHA),
  loader_source_receipt=p['binding'],model_forwards=0,expected_model_forwards=67,completed_optimizer_steps=0,
  training_performed=False,teacher_constructed=False,optimizer_constructed=False,optimizer_restored=False,
  development_GT_read=False,quality_computed=False,quality_selection=False,wall_cap_minutes=20,
  parity_evidence='All28 integer/105 PSN/4 heads compared bytewise in memory; only4 parity heads persisted, plus64 development head3 archives')
 if not a.execute:
  require(a.preflight_output and not a.preflight_output.exists(),'Fresh preflight receipt');write(a.preflight_output,result);return
 lock=p['modules'][3].GPU_LOCK.open('a')
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(parents=True,exist_ok=False)
  def alarm(signum,frame):raise TimeoutError('R29 fixed20min evaluation cap')
  signal.signal(signal.SIGALRM,alarm);signal.alarm(1200)
  try:execute(a,smoke,evaluation,core,p,m,first,state,result)
  except BaseException as e:
   result.update(status='EVALUATION_FAILED',error=str(e),traceback=traceback.format_exc());write(a.output/'results.json',result);raise
  finally:signal.alarm(0)
  write(a.output/'results.json',result)
 finally:lock.close()
if __name__=='__main__':main()
