#!/usr/bin/env python3
"""R27 fixed512 ordinary recovery, then one fixed final development endpoint.

No development GT array is loaded here. The frozen one-step smoke is a source
helper only; every formal arm starts from original W271482 and a fresh AdamW.
"""
import argparse,contextlib,copy,fcntl,hashlib,json,signal,sys,time,traceback,types
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SMOKE_SHA='48563d36f1b448a66d18278e0448deee02a7650583182a628dcb3865923c6c5c'
CORE_SHA='19623636229b4dc5a85132e4acc8bf217f9f62a5e51fcab0fdd7394ee186f788'
MEMBERS_SHA='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'
GRAPH='flow_r27_all12mlp_patch4conv_integer_Q6_complete9_tailQ8_all78current_v1'
def require(ok,why):
 if not ok:raise ValueError(why)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p,name):
 import importlib.util
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def tensor_record(x):
 y=x.detach().cpu().contiguous();return dict(shape=list(y.shape),dtype=str(y.dtype),sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest())
def recipe(smoke):
 r=copy.deepcopy(smoke.RECIPE);r.pop('formal_updates_not_run');r.update(updates=512,train_members=256,passes=2,development_members=64,
  checkpoint_selection='fixed_final_step512_only',training_mode='eval_with_grad_currentBN_no_stochastic_depth',
  scope='bounded_ordinary_recovery_not_converged_original_training',formal_resume_from_smoke=False)
 return r

def validate_order(m):
 rows={r['target_id']:r for r in m['train_members']};ids=list(rows)
 require(len(ids)==256 and len(m['train_order'])==512,'Exactly256 train members and512 updates')
 passes=[]
 for i,entry in enumerate(m['train_order']):
  require(entry['step']==i+1 and entry['pass_index']==i//256+1,'Fixed ordered two-pass step indices')
  require(entry['target_id'] in rows and entry['sequence']==rows[entry['target_id']]['sequence'],'Training member sequence bound')
  c=entry['crop'];require(set(c)=={'top','left','height','width'} and c['height']==288 and c['width']==384
   and 0<=c['top']<=192 and 0<=c['left']<=256,'Fixed original crop geometry')
  require(entry['augmentation']=='fixed_crop_only_no_flip_or_rotation','No unrecorded augmentation')
 for k in range(2):
  seq=[r['target_id'] for r in m['train_order'][256*k:256*(k+1)]]
  require(len(set(seq))==256 and set(seq)==set(ids),'Every member once per pass');passes.append(seq)
 require(passes[0]==passes[1],'Same identity order in each pass')
 dev=[r['target_id'] for r in m['dev_members']]
 require(len(dev)==len(set(dev))==64 and not(set(dev)&set(ids)),'Distinct64 development identities')

def prepare(a):
 require(sha(__file__)==a.runner_sha256 and sha(HERE/'run_smoke.py')==SMOKE_SHA and sha(HERE/'recovery_integer_core.py')==CORE_SHA,'Frozen formal/helper/core source bytes')
 smoke=load(HERE/'run_smoke.py','_r27_formal_smoke_helpers')
 old=copy.copy(a);old.runner_sha256=SMOKE_SHA
 p,m,first,order=smoke.prepare(old);validate_order(m)
 return smoke,p,m,first,order

def work_summary(work):
 out={}
 for name,w in work.items():
  out[name]={k:v for k,v in w.items() if k!='kernel_calls'}
 return out

@contextlib.contextmanager
def observed(model,core,torch):
 """Hooks measure the original BN and the actually installed28 affine callables."""
 handles=[];bns={};affines={}
 try:
  for name,module in model.named_modules():
   if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
    bns[name]=0
    def before(m,ins,name=name):bns[name]+=1
    handles.append(module.register_forward_pre_hook(before))
  for name in core.TARGETS:
   affines[name]=[]
   def after(m,ins,out,name=name):
    affines[name].append(dict(input_shape=list(ins[0].shape),output_shape=list(out.shape),module_class=type(m).__name__))
   handles.append(model.get_submodule(name).register_forward_hook(after))
  yield bns,affines
  require(len(bns)==78 and all(v==1 for v in bns.values()),'Every allcurrent BN called once')
  require(len(affines)==28 and all(len(v)==1 for v in affines.values()),'Every target affine called once')
 finally:
  for h in handles:h.remove()
  require(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()),'All formal observer hooks removed')

def source_evidence(m,tid):
 rec=next(r for r in m['records'] if r['target_id']==tid)
 return {('events' if x['category']=='input' else x['category']):{'relative_path':x['relative_path'],'sha256':x['sha256']} for x in rec['source_files']}

def event_only(p,row,evidence,np):
 path=p['context'][0].checked_path(p['b'].data_root,evidence['events']['relative_path'])
 require(sha(path)==evidence['events']['sha256'],'Actual development event source bytes')
 events=np.load(path,allow_pickle=False)
 require(events.shape==(10,480,640) and events.dtype==np.float32 and np.isfinite(events).all(),'Native full event encoding')
 return events

def evaluate(a,p,m,smoke,core,model,functional,stage,torch,np,result):
 model.eval().requires_grad_(False);r13=p['modules'][0];before=r13.state_digest(model)
 signature=p['base'].bn_signature(model,torch);frames=[];start=time.monotonic()
 folder=a.output/'predictions'/stage;folder.mkdir(parents=True,exist_ok=False)
 for index,row in enumerate(m['dev_members']):
  evidence=source_evidence(m,row['target_id']);events=event_only(p,row,evidence,np)
  chunk=p['base'].prepare_input(events,torch.device('cuda:0'),torch);functional.reset_net(model)
  with core.installed(model,a.arm,ste=False) as wrappers:
   with observed(model,core,torch) as (bns,calls):
    with torch.no_grad():flows=model(chunk)['flow']
   work=work_summary({n:copy.deepcopy(w.last_audit) for n,w in wrappers.items()})
  result['model_forwards']+=1
  require(len(flows)==4 and all(tuple(v.shape)==(1,2,480,640) and v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in flows),'Four finite full-resolution FP32 pixel heads')
  heads={f'head{i}':tensor_record(v[0]) for i,v in enumerate(flows)}
  file=folder/(row['target_id']+'.npz');np.savez_compressed(file,head3=flows[3][0].detach().cpu().numpy())
  frame=dict(target_id=row['target_id'],sequence=row['sequence'],input_evidence=evidence,prediction_file=str(file.relative_to(a.output)),
   prediction_file_sha256=sha(file),prediction_sha256=heads['head3']['sha256'],heads=heads,actual_BN_calls=bns,actual_affine_calls=calls,work=work)
  frames.append(frame);del flows,chunk,events
  if (index+1)%8==0:print(json.dumps({'stage':stage,'development_predictions':index+1,'arm':a.arm}),flush=True)
 require(r13.state_digest(model)==before and p['base'].bn_signature(model,torch)==signature,'Inference state and absent BN buffers unchanged')
 result.setdefault('evaluations',{})[stage]=dict(state_sha256=before,frames=frames,model_forwards=64,wall_seconds=time.monotonic()-start,GT_arrays_read=False)
 write(a.output/'results.json',result)

def grad_stats(params,torch):
 names=[n for n,p in params.items() if p.grad is not None];none=[n for n,p in params.items() if p.grad is None]
 norms=torch.stack([torch.linalg.vector_norm(params[n].grad) for n in names]);require(bool(torch.isfinite(norms).all()),'All nonempty parameter gradients finite')
 return names,none,norms.detach().cpu().tolist()

def execute(a,smoke,p,m,first,first_order,result):
 import torch,numpy as np
 require(not torch.cuda.is_initialized(),'Lock precedes CUDA')
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261019)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_float32_matmul_precision('highest')
 device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261019)
 core=load(HERE/'recovery_integer_core.py','_r27_formal_core');r13=p['modules'][0]
 start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 full,fullfun,bn=smoke.make_model(p,'full9',device,torch);initial=r13.state_digest(full)
 result.update(status='RUNNING',initial_state_sha256=initial,BN_scope=bn,runtime={'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'AMP':False,'TF32':False})
 evaluate(a,p,m,smoke,core,full,fullfun,'initial',torch,np,result)
 del full;torch.cuda.empty_cache()
 model,functional,_=smoke.make_model(p,'crop_center9',device,torch)
 teacher,teacherfun,_=smoke.make_model(p,'crop_center9',device,torch);teacher.requires_grad_(False)
 require(r13.state_digest(model)==r13.state_digest(teacher)==initial,'Formal models start original W0; no smoke state')
 params=core.trainable_parameters(model);signature=p['base'].bn_signature(model,torch);teacher_initial=r13.state_digest(teacher)
 opt=torch.optim.AdamW(list(params.values()),lr=2.5e-5,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 require(len(opt.state)==0,'Fresh AdamW, no inherited optimizer')
 sys.path.insert(0,str(p['b'].source_root));from loss.flow_supervised import flow_loss_supervised
 require(Path(sys.modules['loss.flow_supervised'].__file__).resolve()==(p['b'].source_root/'loss/flow_supervised.py').resolve()
  and sha(Path(sys.modules['loss.flow_supervised'].__file__))==smoke.read(p['b'].source_manifest)['loss/flow_supervised.py'],'Frozen pixel-unit loss source')
 loss_fn=flow_loss_supervised(p['context'][5],device);rows={r['target_id']:r for r in m['train_members']}
 result.update(trainable_names=list(params),trainable_elements=sum(p.numel() for p in params.values()),teacher_initial_state_sha256=teacher_initial)
 ledger=a.output/'training_ledger.jsonl';first_none=None;step_start=time.monotonic()
 for order in m['train_order']:
  i=order['step'];row=rows[order['target_id']];tic=time.monotonic()
  chunk,gt,valid,_,evidence=smoke.training_arrays(p,row,order['crop'],torch,device)
  teacherfun.reset_net(teacher)
  with observed(teacher,core,torch) as (teacher_bn,teacher_calls):
   with torch.no_grad():targets=teacher(chunk)['flow']
  result['model_forwards']+=1;functional.reset_net(model);opt.zero_grad(set_to_none=True)
  with core.installed(model,a.arm,ste=True) as wrappers:
   with observed(model,core,torch) as (bns,calls):output=model(chunk)['flow']
   result['model_forwards']+=1
   require(len(output)==4 and all(v.requires_grad and bool(torch.isfinite(v).all()) for v in output),'Four actual gradient heads')
   sup=loss_fn(output,gt,valid,gamma=None);kd=sum((x-y).abs().mean() for x,y in zip(output,targets))/4;loss=sup+.1*kd
   require(bool(torch.isfinite(loss)),'Finite ordinary loss')
   if i==1 and a.arm=='native':require(float(kd.detach())==0,'Original native teacher first-step KD0')
   loss.backward();names,none,norms=grad_stats(params,torch)
   if first_none is None:first_none=none
   require(none==first_none,'Stable observed no-gradient set; no hidden update scope changes')
   preclip=float(torch.nn.utils.clip_grad_norm_(list(params.values()),100.,error_if_nonfinite=True));opt.step()
   work={n:copy.deepcopy(w.last_audit) for n,w in wrappers.items()}
  require(p['base'].bn_signature(model,torch)==signature,'Allcurrent buffers absent through recovery')
  record=dict(step=i,target_id=row['target_id'],sequence=row['sequence'],pass_index=order['pass_index'],crop=order['crop'],input_evidence=evidence,
   supervised_loss_px=float(sup.detach()),teacher_mean_L1_px=float(kd.detach()),combined_loss=float(loss.detach()),preclip_grad_norm=preclip,
   gradients_finite=True,gradient_parameter_count=len(names),no_gradient_names=none,actual_optimizer_parameter_count=len(opt.state),
   teacher_actual_BN_count=sum(teacher_bn.values()),student_actual_BN_count=sum(bns.values()),teacher_affine_calls=teacher_calls,student_affine_calls=calls,
   work=work if i in (1,512) else work_summary(work),wall_seconds=time.monotonic()-tic)
  if i in (1,512):record.update(gradient_norms=dict(zip(names,norms)),teacher_BN_calls=teacher_bn,student_BN_calls=bns)
  with ledger.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
  result['completed_optimizer_steps']=i
  if i==1 or i%16==0:print(json.dumps({'arm':a.arm,'step':i,'supervised_px':record['supervised_loss_px'],'KD_px':record['teacher_mean_L1_px'],'gradient_count':len(names),'seconds':record['wall_seconds']}),flush=True);write(a.output/'results.json',result)
  del output,targets,chunk,gt,valid,loss,sup,kd
 require(result['completed_optimizer_steps']==512 and all(bool(torch.isfinite(q).all()) for q in params.values()),'Fixed512 finite master weights')
 final=r13.state_digest(model);require(final!=initial and r13.state_digest(teacher)==teacher_initial,'Updated student/fixed teacher')
 state={k:v.detach().cpu() for k,v in model.state_dict().items()}
 checkpoint=dict(schema='flow_r27_fixed512_checkpoint_v1',model_state_dict=state,optimizer_state_dict=opt.state_dict(),arm=a.arm,recipe=result['recipe'],
  completed_optimizer_steps=512,initial_state_sha256=initial,final_state_sha256=final,members_sha256=MEMBERS_SHA,sources=result['sources'],
  torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),fixed_endpoint_no_development_selection=True)
 weights=a.output/'recovered_step512.pt';torch.save(checkpoint,weights)
 result.update(training_state_sha256=final,teacher_final_state_sha256=teacher_initial,weights_file=weights.name,weights_sha256=sha(weights),
  training_ledger_file=ledger.name,training_ledger_sha256=sha(ledger),training_seconds=time.monotonic()-step_start,
  actual_optimizer_parameter_count=len(opt.state),no_gradient_names=first_none)
 del opt,teacher,params,checkpoint;functional.reset_net(model);del model;torch.cuda.empty_cache()
 full,fullfun,_=smoke.make_model(p,'full9',device,torch);full.load_state_dict(state,strict=True);del state;full.eval().requires_grad_(False)
 require(r13.state_digest(full)==final,'Full-resolution final model uses exact recovered state')
 events=event_only(p,first,source_evidence(m,first['target_id']),np);chunk=p['base'].prepare_input(events,device,torch);del events
 reference,refrecord=smoke.capture_full(full,fullfun,chunk,a.arm,core,torch,ste=True);result['model_forwards']+=1
 actual,actrecord=smoke.capture_full(full,fullfun,chunk,a.arm,core,torch,ste=False);result['model_forwards']+=1
 parity=smoke.compare_full(actual,reference,torch);del actual
 result['final_training_deployment_parity']=dict(**parity,reference_record=refrecord,actual_record=actrecord)
 write(a.output/'results.json',result);require(parity['pass'],'Final STE/deploy fullgraph bitwise equality')
 if a.arm!='native':
  dense,denserecord=smoke.capture_full(full,fullfun,chunk,a.arm,core,torch,ste=False,reference=True);result['model_forwards']+=1
  parity=smoke.compare_full(dense,reference,torch);del dense
  result['final_dense_integer_parity']=dict(**parity,dense_record=denserecord)
  write(a.output/'results.json',result);require(parity['pass'],'Final integer reference/deploy fullgraph bitwise equality')
 np.savez_compressed(a.output/'final_parity_heads.npz',**{k:v.numpy() for k,v in reference.items() if k.startswith('head')})
 del reference,chunk;torch.cuda.empty_cache()
 evaluate(a,p,m,smoke,core,full,fullfun,'endpoint',torch,np,result)
 require(r13.state_digest(full)==final and result['model_forwards']==(1154 if a.arm=='native' else 1155),'Closed final state and exact model forward budget')
 result.update(status='FIXED512_AND128_DEVELOPMENT_PREDICTIONS_COMPLETE',final_state_sha256=final,
  training_performed=True,development_GT_read=False,quality_computed=False,quality_selection=False,formal_resume_from_smoke=False,
  parity_heads_sha256=sha(a.output/'final_parity_heads.npz'),wall_seconds=time.monotonic()-start,
  max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--r26-root',type=Path,required=True);p.add_argument('--members',type=Path,required=True);p.add_argument('--runner-sha256',required=True)
 p.add_argument('--arm',choices=('native','Q8','Q6'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight-output',type=Path)
 p.add_argument('--execute',action='store_true');p.add_argument('--wall-minutes',type=float,required=True)
 a=p.parse_args();require(a.wall_minutes==60,'Predeclared60min per arm; never quality-selected')
 smoke,support,m,first,order=prepare(a)
 result=dict(schema='flow_r27_bounded_recovery_v1',graph_id=GRAPH,status='PREFLIGHT_PASS',arm=a.arm,recipe=recipe(smoke),
  sources={'runner':sha(__file__),'smoke_helpers':SMOKE_SHA,'core':CORE_SHA,'members':MEMBERS_SHA,'R26_loader_only':smoke.R26_SHA},
  loader_source_receipt=support['binding'],model_forwards=0,completed_optimizer_steps=0,training_performed=False,development_GT_read=False,
  expected_model_forwards=1154 if a.arm=='native' else 1155,wall_cap_minutes=a.wall_minutes)
 if not a.execute:
  require(a.preflight_output and not a.preflight_output.exists(),'Fresh CPU preflight output');write(a.preflight_output,result);return
 lock=support['modules'][3].GPU_LOCK.open('a')
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(parents=True,exist_ok=False)
  def alarm(signum,frame):raise TimeoutError('R27 fixed ordinary-recovery wall cap; no partial endpoint eligible')
  signal.signal(signal.SIGALRM,alarm);signal.alarm(int(a.wall_minutes*60))
  try:execute(a,smoke,support,m,first,order,result)
  except BaseException as e:
   result.update(status='RECOVERY_FAILED',error=str(e),traceback=traceback.format_exc());write(a.output/'results.json',result);raise
  finally:signal.alarm(0)
  write(a.output/'results.json',result)
 finally:lock.close()
if __name__=='__main__':main()
