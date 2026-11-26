#!/usr/bin/env python3
"""R30 one fixed joint recovery; separate one-update M smoke. No dev GT read."""
import argparse,collections,copy,fcntl,hashlib,importlib.util,json,signal,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
import mixed_integer_core as core
import mask_contract as masks
CORE_SHA='ceec6924fd07a33fa7b806dce7ee4699a4a438a08d6a342b24137d216e7a13d5'
MASK_SHA='408920b7f5f6b06475e0b42f2803992b1a23e5cc3366249efc2c98f90c8a9a83'
SMOKE_SHA='6dadb497bf97fc33929209a3feed0acedcac241036d27bbcda502d8f35244d83'
HELPER_SHA='5a5bf2abaca8fe23fc269e4dc830acbd3ee6537e52958c29c1520543756a4f95'
MEMBERS_SHA=masks.MEMBERS_SHA
ORIGINAL='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
GRAPH='flow_r30_joint_Q4_Q8_M_all12mlp_patch4_currentBN_v1'
require=core.require
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,x):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def initial_references(path,m):
 d=read(path);require(d['schema']=='flow_r30_original_initial_references_v1' and set(d['states'])=={'initial_Q4','initial_Q8'},'Two exact original-state references')
 dev=[x['target_id'] for x in m['dev_members']];out={}
 for state,desc in d['states'].items():
  rp,ap=Path(desc['result_path']),Path(desc['audit_path'])
  require(sha(rp)==desc['result_sha256'] and sha(ap)==desc['audit_sha256'],'Existing result/audit bytes')
  r,a=read(rp),read(ap);arm=state.split('_')[1]
  require(a['status']=='PASS' and a['quality_analysis_allowed'] is True and a['source_results'][arm]==sha(rp),'Accepted complete original-arm execution')
  require(a['schema']==('flow_r28_independent_actual_execution_audit_v1' if arm=='Q4' else 'flow_r27_independent_actual_execution_audit_v1'),'Original actual audit schema')
  require(r['arm']==arm and r['status']=='FIXED512_AND128_DEVELOPMENT_PREDICTIONS_COMPLETE' and r['initial_state_sha256']==ORIGINAL and r['sources']['members']==MEMBERS_SHA,'Original W and same membership')
  ev=r['evaluations']['initial'];require(ev['state_sha256']==ORIGINAL and ev['model_forwards']==64 and [x['target_id'] for x in ev['frames']]==dev,'Same original64 endpoint identity')
  out[state]=dict(**desc,state_sha256=ORIGINAL,model_forwards_reused=64,new_forwards=0)
 return out

def prepare(a):
 import torch
 require(not torch.cuda.is_initialized() and not a.output.exists(),'Fresh CPU-only preflight')
 require(sha(__file__)==a.runner_sha256 and sha(HERE/'mixed_integer_core.py')==CORE_SHA and sha(HERE/'mask_contract.py')==MASK_SHA,'Explicit R30 sources')
 require(Path(core.__file__).resolve()==HERE/'mixed_integer_core.py' and masks.core is core and Path(masks.__file__).resolve()==HERE/'mask_contract.py','Actual local imported mathematical modules')
 base=a.r28_root/'implementation'
 require(sha(base/'run_smoke_v2.py')==SMOKE_SHA and sha(base/'run_recovery_v2.py')==HELPER_SHA,'Frozen loader/training/data helper sources')
 smoke=load(base/'run_smoke_v2.py','_r30_smoke_helpers');helper=load(base/'run_recovery_v2.py','_r30_formal_helpers')
 old=copy.copy(a);old.runner_sha256=SMOKE_SHA
 p,m,first,order=smoke.prepare(old);helper.validate_order(m)
 require(sha(a.mask_contract)==a.mask_contract_sha256,'Externally chosen mask contract bytes')
 contract=masks.validate(read(a.mask_contract))
 schedule=[masks.mode_for(x,m['sequences']) for x in m['train_order']]
 require(collections.Counter(schedule)=={'Q4':256,'Q8':128,'M':128},'Fixed512 mode count')
 per_id=collections.defaultdict(list)
 for row,mode in zip(m['train_order'],schedule):per_id[row['target_id']].append(mode)
 require(all(len(v)==2 and v.count('Q4')==1 for v in per_id.values()),'Every identity sees Q4 once and one other mode')
 refs={} if a.smoke else initial_references(a.reference_contract,m)
 return smoke,helper,p,m,first,contract,refs

def train(a,smoke,helper,p,m,contract,result,torch):
 device=torch.device('cuda:0');r13=p['modules'][0]
 model,functional,_=smoke.make_model(p,'crop_center9',device,torch)
 teacher,tf,_=smoke.make_model(p,'crop_center9',device,torch);teacher.requires_grad_(False)
 require(r13.state_digest(model)==r13.state_digest(teacher)==ORIGINAL,'Fresh original W for student and fixed teacher')
 params=core.trainable_parameters(model);signature=p['base'].bn_signature(model,torch)
 opt=torch.optim.AdamW(list(params.values()),lr=2.5e-5,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 require(len(opt.state)==0,'Fresh optimizer; never inherit smoke state')
 sys.path.insert(0,str(p['b'].source_root));from loss.flow_supervised import flow_loss_supervised
 require(Path(sys.modules['loss.flow_supervised'].__file__).resolve()==(p['b'].source_root/'loss/flow_supervised.py').resolve() and sha(Path(sys.modules['loss.flow_supervised'].__file__))==smoke.read(p['b'].source_manifest)['loss/flow_supervised.py'],'Actual original loss source')
 loss_fn=flow_loss_supervised(p['context'][5],device);rows={r['target_id']:r for r in m['train_members']}
 orders=m['train_order'][:1] if a.smoke else m['train_order'];ledger=a.output/'training_ledger.jsonl';none_first=None;seen=set();counts=collections.Counter();tic=time.monotonic()
 result.update(trainable_names=list(params),trainable_elements=sum(x.numel() for x in params.values()),teacher_initial_state_sha256=ORIGINAL)
 for order in orders:
  step=order['step'];mode='M' if a.smoke else masks.mode_for(order,m['sequences']);step_start=time.monotonic()
  selected,mask_record=masks.materialize(contract,'crop_center9',step=step) if mode=='M' else (None,None)
  row=rows[order['target_id']];chunk,gt,valid,events,evidence=smoke.training_arrays(p,row,order['crop'],torch,device);del events
  tf.reset_net(teacher)
  with helper.observed(teacher,core,torch) as (tbn,tcalls):
   with torch.no_grad():target=teacher(chunk)['flow']
  result['model_forwards']+=1;functional.reset_net(model);opt.zero_grad(set_to_none=True)
  with core.installed(model,mode,masks=selected,ste=True) as wrappers:
   with helper.observed(model,core,torch) as (bn,calls):output=model(chunk)['flow']
   result['model_forwards']+=1
   require(len(output)==4 and all(v.requires_grad and bool(torch.isfinite(v).all()) for v in output),'Four finite actual gradient heads')
   sup=loss_fn(output,gt,valid,gamma=None);kd=sum((v-t).abs().mean() for v,t in zip(output,target))/4;loss=sup+.1*kd
   require(bool(torch.isfinite(loss)),'Finite ordinary supervised/distillation objective');loss.backward()
   names,none,norms=helper.grad_stats(params,torch)
   if none_first is None:none_first=none
   require(none==none_first,'Stable actual no-gradient set across modes')
   preclip=float(torch.nn.utils.clip_grad_norm_(list(params.values()),100.,error_if_nonfinite=True));opt.step()
   work={n:copy.deepcopy(w.last_audit) for n,w in wrappers.items()}
  require(p['base'].bn_signature(model,torch)==signature,'Current BN buffers stay absent')
  detailed=mode not in seen or step==orders[-1]['step'];seen.add(mode);counts[mode]+=1
  record=dict(step=step,target_id=row['target_id'],sequence=row['sequence'],pass_index=order['pass_index'],crop=order['crop'],input_evidence=evidence,mode=mode,mask_layout=mask_record,
   supervised_loss_px=float(sup.detach()),teacher_mean_L1_px=float(kd.detach()),combined_loss=float(loss.detach()),preclip_grad_norm=preclip,
   gradients_finite=True,gradient_parameter_count=len(names),no_gradient_names=none,actual_optimizer_parameter_count=len(opt.state),
   teacher_actual_BN_count=sum(tbn.values()),student_actual_BN_count=sum(bn.values()),teacher_affine_calls=tcalls,student_affine_calls=calls,
   work=work if detailed else helper.work_summary(work),detailed_kernel_records=detailed,wall_seconds=time.monotonic()-step_start)
  if detailed:record.update(gradient_norms=dict(zip(names,norms)),teacher_BN_calls=tbn,student_BN_calls=bn)
  with ledger.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
  result['completed_optimizer_steps']=step
  if step==1 or step%16==0:print(json.dumps(dict(step=step,mode=mode,supervised_px=record['supervised_loss_px'],KD_px=record['teacher_mean_L1_px'],seconds=record['wall_seconds'])),flush=True);write(a.output/'results.json',result)
  del output,target,chunk,gt,valid,loss,sup,kd
 require(counts==({'M':1} if a.smoke else {'Q4':256,'Q8':128,'M':128}),'Exact predeclared student-mode opportunities')
 require(all(bool(torch.isfinite(q).all()) for q in params.values()),'Finite final master weights')
 final=r13.state_digest(model);require(final!=ORIGINAL and r13.state_digest(teacher)==ORIGINAL,'Actually updated student; unchanged teacher')
 state={k:v.detach().cpu() for k,v in model.state_dict().items()}
 cp=dict(schema='flow_r30_joint_checkpoint_v1',arm='joint',model_state_dict=state,optimizer_state_dict=opt.state_dict(),completed_optimizer_steps=len(orders),
  initial_state_sha256=ORIGINAL,final_state_sha256=final,members_sha256=MEMBERS_SHA,sources=result['sources'],recipe=result['recipe'],joint_contract=result['joint_contract'],
  torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),fixed_endpoint_no_development_selection=True,formal_resume_allowed=False)
 file=a.output/('smoke_weights.pt' if a.smoke else 'recovered_step512.pt');torch.save(cp,file)
 result.update(training_state_sha256=final,teacher_final_state_sha256=ORIGINAL,weights_file=file.name,weights_sha256=sha(file),training_ledger_file=ledger.name,training_ledger_sha256=sha(ledger),
  training_seconds=time.monotonic()-tic,actual_optimizer_parameter_count=len(opt.state),no_gradient_names=none_first,actual_student_mode_counts=dict(counts))
 del opt,teacher,params,cp;functional.reset_net(model);del model;torch.cuda.empty_cache()
 return state,final

def parity(a,smoke,helper,p,m,first,model,functional,mode,bound,mask_record,result,torch,np):
 events=helper.event_only(p,first,helper.source_evidence(m,first['target_id']),np);chunk=p['base'].prepare_input(events,torch.device('cuda:0'),torch);del events
 reference,rr=smoke.capture_full(model,functional,chunk,mode,bound,torch,ste=True);result['model_forwards']+=1
 actual,ar=smoke.capture_full(model,functional,chunk,mode,bound,torch,ste=False);result['model_forwards']+=1
 cmp=smoke.compare_full(actual,reference,torch);del actual
 record=dict(mode=mode,mask_layout=mask_record,input_target_id=first['target_id'],input_evidence=helper.source_evidence(m,first['target_id']),
  training_deployment=dict(**cmp,reference_record=rr,actual_record=ar))
 result.setdefault('final_parity',{})[mode]=record;write(a.output/'results.json',result);require(cmp['pass'],'Exact final STE/deployment '+mode)
 dense,dr=smoke.capture_full(model,functional,chunk,mode,bound,torch,ste=False,reference=True);result['model_forwards']+=1
 cmp=smoke.compare_full(dense,reference,torch);del dense
 record['dense_integer']=dict(**cmp,dense_record=dr);write(a.output/'results.json',result);require(cmp['pass'],'Exact final dense integer '+mode)
 file=a.output/('final_parity_'+mode+'_heads.npz');np.savez_compressed(file,**{k:v.numpy() for k,v in reference.items() if k.startswith('head')})
 record.update(heads_file=file.name,heads_sha256=sha(file));del reference,chunk;torch.cuda.empty_cache()

def execute(a,smoke,helper,p,m,first,contract,result):
 import torch,numpy as np
 require(not torch.cuda.is_initialized(),'Global lock before CUDA')
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261019)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_float32_matmul_precision('highest')
 device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261019);torch.cuda.reset_peak_memory_stats();start=time.monotonic();r13=p['modules'][0]
 selected,mask_record=masks.materialize(contract,'full9');mixed=masks.BoundCore(selected)
 result.update(status='RUNNING',initial_state_sha256=ORIGINAL,runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),AMP=False,TF32=False))
 if not a.smoke:
  model,functional,bn=smoke.make_model(p,'full9',device,torch);require(r13.state_digest(model)==ORIGINAL,'Original initialM source')
  result['BN_scope']=bn;a.arm='M';helper.evaluate(a,p,m,smoke,mixed,model,functional,'initial_M',torch,np,result)
  result['evaluations']['initial_M']['mask_layout']=mask_record;del model;torch.cuda.empty_cache()
 state,final=train(a,smoke,helper,p,m,contract,result,torch)
 model,functional,bn=smoke.make_model(p,'full9',device,torch);model.load_state_dict(state,strict=True);del state;model.eval().requires_grad_(False)
 require(r13.state_digest(model)==final,'Exact final recovered state at full geometry');signature=p['base'].bn_signature(model,torch)
 for mode in (('M',) if a.smoke else ('Q4','Q8','M')):
  bound=mixed if mode=='M' else masks.BoundCore();layout=mask_record if mode=='M' else None
  parity(a,smoke,helper,p,m,first,model,functional,mode,bound,layout,result,torch,np)
  if not a.smoke:
   a.arm=mode;helper.evaluate(a,p,m,smoke,bound,model,functional,'endpoint_'+mode,torch,np,result)
   result['evaluations']['endpoint_'+mode]['mask_layout']=layout
 require(result['model_forwards']==(5 if a.smoke else 1289) and r13.state_digest(model)==final and p['base'].bn_signature(model,torch)==signature,'Complete fixed model budget and unchanged final inference state')
 result.update(status='MIXED_ONE_UPDATE_AND_PARITY_COMPLETE' if a.smoke else 'JOINT512_AND256_DEVELOPMENT_PREDICTIONS_COMPLETE',
  final_state_sha256=final,training_performed=True,development_GT_read=False,quality_computed=False,quality_selection=False,
  wall_seconds=time.monotonic()-start,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('r28-root','r26-root','members','mask-contract','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--mask-contract-sha256',required=True);p.add_argument('--runner-sha256',required=True)
 p.add_argument('--reference-contract',type=Path);p.add_argument('--preflight-output',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true')
 p.add_argument('--wall-minutes',type=int,required=True);a=p.parse_args()
 require(a.wall_minutes==(20 if a.smoke else 60),'Predeclared20min smoke/60min formal cap')
 require(a.smoke or a.reference_contract is not None,'Formal original-Q4/Q8 evidence descriptors')
 smoke,helper,support,m,first,contract,refs=prepare(a)
 result=dict(schema='flow_r30_joint_recovery_v1',graph_id=GRAPH,status='PREFLIGHT_PASS',arm='joint',engineering_smoke=a.smoke,
  sources=dict(runner=sha(__file__),core=CORE_SHA,mask_helper=MASK_SHA,smoke_helpers=SMOKE_SHA,formal_helpers=HELPER_SHA,members=MEMBERS_SHA,mask_contract=a.mask_contract_sha256),
  recipe=helper.recipe(smoke),joint_contract=dict(mode_counts={'M':1} if a.smoke else {'Q4':256,'Q8':128,'M':128},schedule='Q4,Q4,Q8,M[(visit-1+sequence_rank+2*(pass-1))mod4]',
    seed=20261019,mask_contract=contract,mask_contract_sha256=a.mask_contract_sha256,smoke_state_never_resumed=True),
  original_initial_references=refs,loader_source_receipt=support['binding'],model_forwards=0,expected_model_forwards=5 if a.smoke else 1289,
  completed_optimizer_steps=0,training_performed=False,development_GT_read=False,quality_computed=False,quality_selection=False,formal_resume_from_smoke=False,wall_cap_minutes=a.wall_minutes)
 if not a.execute:
  require(a.preflight_output and not a.preflight_output.exists(),'Fresh CPU preflight receipt');write(a.preflight_output,result);return
 lock=support['modules'][3].GPU_LOCK.open('a')
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(parents=True,exist_ok=False)
  def alarm(signum,frame):raise TimeoutError('R30 fixed wall cap; no partial endpoint eligible')
  signal.signal(signal.SIGALRM,alarm);signal.alarm(a.wall_minutes*60)
  try:execute(a,smoke,helper,support,m,first,contract,result)
  except BaseException as e:
   result.update(status='JOINT_RECOVERY_FAILED',error=str(e),traceback=traceback.format_exc());write(a.output/'results.json',result);raise
  finally:signal.alarm(0)
  write(a.output/'results.json',result)
 finally:lock.close()
if __name__=='__main__':main()
