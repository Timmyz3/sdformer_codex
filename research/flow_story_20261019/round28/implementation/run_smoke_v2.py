#!/usr/bin/env python3
"""R28 one actual update, then full-resolution STE/deploy/dense-int parity.

This source is a bounded engineering smoke. Its endpoint never resumes formal
512-step recovery. No development GT or AEE is read/selected here.
"""
import argparse,copy,fcntl,hashlib,importlib.util,json,math,signal,sys,time,traceback,types
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CORE_SHA='f4e4dae18a7e59fab286f2e59f23f72d20583a2b8de22dee4c78204bbfd2fe9a'
R26_SHA='66ce0f8f2385ab49ed114adde799e2a11c1d88934727baad62cf20495da3379d'
R26_INPUTS_SHA='cc7ab752c7cd1e39ce8b2be023b7701e561ec5e127b600691e22baae109b3f4a'
R26_FREEZE_SHA='98c4b750bead5a12f6d89f9d84f8780f370bc9b84bb22d5f96cd4a594e172bc0'
MEMBERS_SHA='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'
RECIPE=dict(updates=1,formal_updates_not_run=512,seed=20261019,optimizer='AdamW',lr=2.5e-5,weight_decay=.01,
            betas=[.9,.999],eps=1e-8,clip_grad_norm=100.,teacher_weight=.1,all_parameters=477,
            quantized_affine_modules=28,flow_scaling=1.,crop=[288,384],augmentation='fixed_crop_only_no_flip_or_rotation',
            teacher='fixed_original_W271482_native_all78current',BN='all78current',dropout_mode='model.eval()',
            formal_resume_from_smoke=False)
def require(x,m):
 if not x:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def load_membership(path):
 # R28 reuses the exact frozen R27 input contract; only output identities are R28.
 require(sha(path)==MEMBERS_SHA,'Fixed original R27 recovery roles')
 members=read(path)
 require(members['schema']=='flow_r27_recovery_identity_v1' and members['seed']==20261019 and
         len(members['train_members'])==256 and len(members['dev_members'])==64 and len(members['train_order'])==512,'Fixed R27 256/64/512 input roles')
 return members

def prepare(a):
 import torch
 require(not torch.cuda.is_initialized() and not a.output.exists(),'CPU-only fresh preflight')
 require(sha(HERE/'recovery_integer_core.py')==CORE_SHA and sha(__file__)==a.runner_sha256,'Explicit R28 new source bytes')
 require(sha(a.r26_root/'implementation/run_precision.py')==R26_SHA,'Frozen source/loader helper')
 old=load(a.r26_root/'implementation/run_precision.py','_r28_r26_source_preflight_only')
 legacy=types.SimpleNamespace(inputs=a.r26_root/'inputs_v1.json',inputs_sha256=R26_INPUTS_SHA,
     freeze=a.r26_root/'source_freeze_v1.json',freeze_sha256=R26_FREEZE_SHA,plan='precision42',output=a.output,wall_minutes=20)
 support=old.prepare(legacy) # CPU source+originalW+H8 byte contracts only; never calls old execute/forward.
 members=load_membership(a.members)
 native,_,metrics,_,hashes,config,*_=support['context']
 require(config['metrics']['flow_scaling']==1 and config['loss']['lambda_mod']==1 and config['loss']['gamma'] is None,'Actual checkpoint pixel-unit native loss')
 canonical=metrics.canonical_rows(support['b'].canonical)
 canonical={r['target_id']:r for r in canonical} if isinstance(canonical,list) else canonical
 records={r['target_id']:r for r in members['records']}
 for role,key in (('train','train_members'),('development','dev_members')):
  for row in members[key]:
   require(row==canonical[row['target_id']],'Canonical role member changed')
   rec=records[row['target_id']];require(rec['canonical']==row and rec['role']==role and rec['sequence']==row['sequence'],'Exact role mapping')
   expected=native.target_files(row)
   for source in rec['source_files']:
    category=source['category'];path=native.checked_path(support['b'].data_root,source['relative_path'])
    require(source['relative_path']==expected[category] and source['sha256']==hashes[source['relative_path']]['sha256'],'Original accepted source mapping')
    require(sha(path)==source['sha256'],'Actual allowed320 source bytes')
 order=members['train_order'][0];row=next(r for r in members['train_members'] if r['target_id']==order['target_id'])
 require(order['step']==order['pass_index']==1 and order['sequence']==row['sequence'],'First fixed training position')
 crop=order['crop'];require(crop['height']==288 and crop['width']==384 and 0<=crop['top']<=192 and 0<=crop['left']<=256,'Original crop bound')
 require(order['augmentation']==RECIPE['augmentation'],'Fixed identical crop only')
 return support,members,row,order


def make_model(p,geometry,device,torch):
 b=copy.copy(p['b']);b.geometry=geometry
 loader=load(Path(p['rt']['breadth_root'])/'run_same_shape_reference.py','_r28_original_native_loader_'+geometry)
 scope=load(Path(p['rt']['breadth_root'])/'run_bn_scope.py','_r28_original_bn_inventory_'+geometry)
 model,functional=loader.load_model(b,p['context'],p['base'],torch,device)
 r13=p['modules'][0]
 require(r13.state_digest(model)==p['k'].SOURCE_STATE_SHA,'Actual original711 source before BN switch')
 bn=p['r14'].core.switch_allcurrent(model,p['base'],scope,r13.state_digest,torch,p['k'].SOURCE_NON_BN_SHA)
 require(r13.state_digest(model)==p['k'].SOURCE_NON_BN_SHA,'Actual original477 allcurrent source')
 model.eval()
 return model,functional,bn


def training_arrays(p,row,crop,torch,device):
 import numpy as np
 native,_,metrics,_,hashes,*_=p['context'];expected={}
 for key,relative in native.target_files(row).items():expected['events' if key=='input' else key]=hashes[relative]['sha256']
 evidence={}
 arrays,_=metrics.load_arrays(p['b'].data_root/'saved_flow_data',row,evidence=evidence,expected_hashes=expected)
 y,x=crop['top'],crop['left'];ys,xs=slice(y,y+288),slice(x,x+384)
 chunk=p['base'].prepare_input(arrays['events'][:,ys,xs],device,torch)
 gt=torch.from_numpy(arrays['gt'][:,ys,xs].copy()).unsqueeze(0).to(device=device,dtype=torch.float32)
 valid=torch.from_numpy(arrays['mask'][ys,xs].copy()).unsqueeze(0).unsqueeze(0).to(device=device,dtype=torch.float32)
 require(int(valid.sum())>0,'Fixed crop has no valid GT; no resampling')
 return chunk,gt,valid,arrays['events'],evidence


def capture_full(model,functional,chunk,arm,core,torch,*,ste,reference=False):
 """True values retained on CPU for exact pair comparison, no GT/quality."""
 functional.reset_net(model);captures={};handles=[];bn_names={};called={}
 def store(name,tensor):
  require(name not in captures,'One actual capture '+name)
  captures[name]=tensor.detach().cpu().contiguous().clone()
 try:
  for name,module in model.named_modules():
   if type(module).__name__=='PSN':
    handles.append(module.register_forward_hook(lambda m,ins,out,name=name:store('PSN::'+name,out)))
   if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
    bn_names[name]=0
    def bn_count(m,ins,name=name):bn_names[name]+=1
    handles.append(module.register_forward_pre_hook(bn_count))
  with core.installed(model,arm,ste=ste,reference=reference) as wrappers:
   for name,w in wrappers.items():
    w.capture_integer=True;called[name]=0
    def after(module,ins,out,name=name):
     called[name]+=1;store('integer::'+name,module.last_integer)
    handles.append(w.register_forward_hook(after))
   with torch.no_grad():flows=model(chunk)['flow']
   for i,v in enumerate(flows):store('head%d'%i,v)
   work={name:copy.deepcopy(w.last_audit) for name,w in wrappers.items()}
  require(len(bn_names)==78 and all(x==1 for x in bn_names.values()),'All78 complete native BN calls')
  require(arm=='native' or len(called)==28 and all(x==1 for x in called.values()),'All28 integer calls')
  return captures,dict(actual_BN_calls=bn_names,actual_target_calls=called,work=work,PSN_count=sum(k.startswith('PSN::') for k in captures),
    integer_count=sum(k.startswith('integer::') for k in captures),head_count=4,ste_entry=ste,dense_integer_reference=reference)
 finally:
  for handle in handles:handle.remove()
  require(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()),'All parity hooks removed')


def compare_full(actual,ref,torch):
 require(set(actual)==set(ref),'Same complete parity capture names')
 checks={}
 for name,x in actual.items():
  y=ref[name];require(x.shape==y.shape and x.dtype==y.dtype,'Same parity shape/dtype')
  xx,yy=x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)
  checks[name]=dict(shape=list(x.shape),dtype=str(x.dtype),actual_sha256=hashlib.sha256(xx.numpy().tobytes()).hexdigest(),
    reference_sha256=hashlib.sha256(yy.numpy().tobytes()).hexdigest(),different_bytes=int((xx!=yy).sum()),bitwise_equal=bool(torch.equal(xx,yy)))
 return dict(**{'pass':all(x['bitwise_equal'] for x in checks.values())},checks=checks)


def execute(a,p,members,row,order,result):
 import torch,numpy as np
 require(not torch.cuda.is_initialized(),'Global lock precedes CUDA')
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261019)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_float32_matmul_precision('highest')
 device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261019)
 core=load(HERE/'recovery_integer_core.py','_r28_new_integer_ste')
 model,functional,bn=make_model(p,'crop_center9',device,torch)
 teacher,teacher_functional,teacher_bn=make_model(p,'crop_center9',device,torch);teacher.requires_grad_(False)
 params=core.trainable_parameters(model);r13=p['modules'][0];initial=r13.state_digest(model)
 teacher_initial=r13.state_digest(teacher);bn_signature=p['base'].bn_signature(model,torch)
 optimizer=torch.optim.AdamW(list(params.values()),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'],betas=tuple(RECIPE['betas']),eps=RECIPE['eps'])
 sys.path.insert(0,str(p['b'].source_root));from loss.flow_supervised import flow_loss_supervised
 require(Path(sys.modules['loss.flow_supervised'].__file__).resolve()==(p['b'].source_root/'loss/flow_supervised.py').resolve()
         and sha(Path(sys.modules['loss.flow_supervised'].__file__))==read(p['b'].source_manifest)['loss/flow_supervised.py'],'Actual frozen original loss source')
 loss_fn=flow_loss_supervised(p['context'][5],device)
 chunk,gt,valid,full_events,evidence=training_arrays(p,row,order['crop'],torch,device)
 result.update(status='RUNNING',runtime={'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'AMP':False,'TF32':False},
   BN_scope=bn,initial_state_sha256=initial,teacher_initial_state_sha256=teacher_initial,training_input_evidence=evidence,
   training_row=row['target_id'],training_order=order,trainable_names=list(params),trainable_elements=sum(x.numel() for x in params.values()))
 write(a.output/'results.json',result)
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
 teacher_functional.reset_net(teacher)
 with torch.no_grad():target_flows=teacher(chunk)['flow']
 result['model_forwards']+=1;torch.cuda.synchronize();teacher_seconds=time.monotonic()-start
 functional.reset_net(model);optimizer.zero_grad(set_to_none=True);begin=time.monotonic()
 with core.installed(model,a.arm,ste=True) as wrappers:
  output=model(chunk)['flow'];result['model_forwards']+=1
  require(len(output)==4 and all(x.requires_grad and bool(torch.isfinite(x).all()) for x in output),'Actual full gradient flow')
  supervised=loss_fn(output,gt,valid,gamma=None)
  kd=sum((x-y).abs().mean() for x,y in zip(output,target_flows))/4
  require(a.arm!='native' or float(kd.detach())==0,'Initial native teacher/student exact KD sanity')
  loss=supervised+.1*kd;require(bool(torch.isfinite(loss)),'Finite combined ordinary loss')
  torch.cuda.synchronize();forward_seconds=time.monotonic()-begin;back=time.monotonic();loss.backward()
  gradient={name:None if q.grad is None else {'finite':bool(torch.isfinite(q.grad).all()),'L2':float(torch.linalg.vector_norm(q.grad))} for name,q in params.items()}
  require(all(v is None or v['finite'] for v in gradient.values()),'All actual gradients finite')
  clip=float(torch.nn.utils.clip_grad_norm_(list(params.values()),100.,error_if_nonfinite=True));optimizer.step()
  work={name:copy.deepcopy(w.last_audit) for name,w in wrappers.items()}
 torch.cuda.synchronize();back_seconds=time.monotonic()-back
 require(all(bool(torch.isfinite(q).all()) for q in params.values()),'All updated parameters finite')
 require(p['base'].bn_signature(model,torch)==bn_signature,'Current BN buffers stay absent')
 final=r13.state_digest(model);require(final!=initial,'One actual optimizer update changes model')
 require(r13.state_digest(teacher)==teacher_initial,'Teacher fixed')
 result.update(training_performed=True,completed_optimizer_steps=1,training_state_sha256=final,
    step={'supervised_loss_px':float(supervised.detach()),'teacher_mean_L1_px':float(kd.detach()),'combined_loss':float(loss.detach()),
      'preclip_grad_norm':clip,'gradient':gradient,'actual_optimizer_parameters_with_state':len(optimizer.state),
      'teacher_seconds':teacher_seconds,'student_forward_seconds':forward_seconds,'backward_optimizer_seconds':back_seconds,
      'training_seconds':time.monotonic()-start,'work':work,'max_allocated_bytes':torch.cuda.max_memory_allocated(),
      'max_reserved_bytes':torch.cuda.max_memory_reserved()})
 checkpoint={'schema':'flow_r28_smoke_one_step_v1','model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
   'optimizer_state_dict':optimizer.state_dict(),'completed_optimizer_steps':1,'recipe':RECIPE,'arm':a.arm,'members_sha256':MEMBERS_SHA,
   'initial_state_sha256':initial,'final_state_sha256':final,'formal_resume_allowed':False}
 torch.save(checkpoint,a.output/'smoke_weights.pt');result['weights_sha256']=sha(a.output/'smoke_weights.pt')
 write(a.output/'results.json',result)
 del loss,supervised,kd,output,target_flows,chunk,gt,valid,teacher,optimizer,params,checkpoint
 functional.reset_net(model);model.requires_grad_(False);state={k:v.detach().cpu() for k,v in model.state_dict().items()}
 del model;torch.cuda.empty_cache()
 full,ffunctional,_=make_model(p,'full9',device,torch);full.load_state_dict(state,strict=True);del state;full.eval().requires_grad_(False)
 require(r13.state_digest(full)==final,'Same updated state at full geometry')
 full_chunk=p['base'].prepare_input(full_events,device,torch);del full_events
 reference,refrecord=capture_full(full,ffunctional,full_chunk,a.arm,core,torch,ste=True);result['model_forwards']+=1
 actual,actrecord=capture_full(full,ffunctional,full_chunk,a.arm,core,torch,ste=False);result['model_forwards']+=1
 parity=compare_full(actual,reference,torch);del actual
 result['training_vs_deploy_full_parity']=dict(**parity,reference_record=refrecord,actual_record=actrecord)
 write(a.output/'results.json',result);require(parity['pass'],'Training STE / actual integer deployment exact full-graph gate')
 if a.arm!='native':
  dense,denserecord=capture_full(full,ffunctional,full_chunk,a.arm,core,torch,ste=False,reference=True);result['model_forwards']+=1
  parity=compare_full(dense,reference,torch);del dense
  result['bitplane_vs_dense_integer_full_parity']=dict(**parity,dense_reference_record=denserecord)
  write(a.output/'results.json',result);require(parity['pass'],'Actual bitplane / integer-reference exact full-graph gate')
 np.savez_compressed(a.output/'full_parity_heads.npz',**{k:v.numpy() for k,v in reference.items() if k.startswith('head')})
 result.update(status='SMOKE_ONE_UPDATE_AND_FULL_PARITY_COMPLETE',final_state_sha256=r13.state_digest(full),teacher_final_state_sha256=teacher_initial,
    all_current_BN_and_source_contract_preserved=True,parity_heads_sha256=sha(a.output/'full_parity_heads.npz'),
    max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),new_quality_computed=False)
 require(result['model_forwards']==(4 if a.arm=='native' else 5),'Declared smoke forward count')


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--r26-root',type=Path,required=True);parser.add_argument('--members',type=Path,required=True)
 parser.add_argument('--runner-sha256',required=True);parser.add_argument('--arm',choices=('Q4',),required=True)
 parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-output',type=Path)
 parser.add_argument('--execute',action='store_true');parser.add_argument('--wall-minutes',type=float,default=30)
 a=parser.parse_args();require(0<a.wall_minutes<=30,'Bounded one-step engineering smoke');p,m,row,order=prepare(a)
 result=dict(schema='flow_r28_one_step_smoke_v1',status='PREFLIGHT_PASS',arm=a.arm,recipe=RECIPE,
   sources={'runner':sha(__file__),'core':CORE_SHA,'members':MEMBERS_SHA,'R26_loader_only':R26_SHA},
   loader_source_receipt=p['binding'],model_forwards=0,completed_optimizer_steps=0,training_performed=False,
   development_GT_read=False,quality_selection=False,smoke_endpoint_not_for_formal_resume=True)
 if not a.execute:
  require(a.preflight_output and not a.preflight_output.exists(),'Fresh preflight output');write(a.preflight_output,result);return
 lock=p['modules'][3].GPU_LOCK.open('a')
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(parents=True,exist_ok=False)
  def alarm(signum,frame):raise TimeoutError('R28 one-step engineering wall cap')
  signal.signal(signal.SIGALRM,alarm);signal.alarm(int(a.wall_minutes*60))
  try:execute(a,p,m,row,order,result)
  except BaseException as e:
   result.update(status='SMOKE_FAILED',error=str(e),traceback=traceback.format_exc());write(a.output/'results.json',result);raise
  finally:signal.alarm(0)
  write(a.output/'results.json',result)
 finally:lock.close()

if __name__=='__main__':main()
