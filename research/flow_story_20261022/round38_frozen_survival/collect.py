"""R38 frozen four-policy survival; source hashes and legal actions fixed first."""
from pathlib import Path
import argparse,copy,fcntl,importlib.util,json,sys,signal,time,traceback,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prepare(a):
 import torch
 assert not torch.cuda.is_initialized() and not a.output.exists()
 assert sha(__file__)==a.runner_sha256 and sha(HERE/'SOURCE_FREEZE_V1.json')==a.source_freeze_sha256
 for name,h in read(HERE/'SOURCE_FREEZE_V1.json')['files'].items():
  assert '..' not in Path(name).parts and not Path(name).is_absolute() and sha(HERE/name)==h
 af=read(HERE/'ACTION_SOURCE_FREEZE_V1.json')
 for name,h in af['files'].items():assert sha(HERE/name)==h
 plan=read(HERE/'ACTION_PLAN_V1.json');assert sha(HERE/'ACTION_PLAN_V1.json')==read(HERE/'ACTION_PLAN_FREEZE_V1.json')['action_plan_sha256']
 assert plan['source_freeze_sha256']==sha(HERE/'ACTION_SOURCE_FREEZE_V1.json') and plan['membership_sha256']==sha(HERE/'data/PROPOSED_SOURCE_MEMBERSHIP_V1.json')
 assert plan['engineering_forwards']==2 and plan['max_forward_attempts']==84 and plan['wall_cap_seconds']==900 and plan['GT_arrays_decoded'] is False
 assert sha(a.r35_root/'SOURCE_FREEZE_V1.json')=='0b17618ed596e9577279036a1a68d1e7331a8adf0d911a7a256fedb7831bec12'
 sys.path.insert(0,str(a.r35_root));old=load(a.r35_root/'collect_static.py','_r38_old_pinned_prepare')
 args=copy.copy(a);args.contract=a.r35_root/'ACTION_CONTRACT_V1.json';args.contract_sha256='426787f32fe13297b6dc0f29539d9ff019b7489ce348a338a23fe368de0b6ac1'
 args.source_freeze=a.r35_root/'SOURCE_FREEZE_V1.json';args.source_freeze_sha256='0b17618ed596e9577279036a1a68d1e7331a8adf0d911a7a256fedb7831bec12';args.runner_sha256='a9bcfe9874272672736a944106e2ec5245e121ad742941a9869cd911195ff4cf'
 core,c,smoke,helper,support,oldrows,refs,_=old.prepare(args)
 assert sha(a.e3_root/'collect_probe.py')=='ab4222e012857275e82e7cf0b848d1ca8bca1dee4d007069ebabe702cc236077'
 sys.path.insert(0,str(a.e3_root));ct=load(a.e3_root/'probe_contract.py','_r38_probe_contract');old151=load(a.e3_root/'collect_probe.py','_r38_151_work');c151=ct.validate(read(a.e3_root/'ENDPOINT_CONTRACT_V1.json'),core)
 rows=read(HERE/'data/PROPOSED_SOURCE_MEMBERSHIP_V1.json')['records'];survey=read(HERE/'data/ENTRY_METADATA_SURVEY_V1.json');ids=[r['target_id'] for r in rows]
 assert len(ids)==len(set(ids))==20 and ids==[r['target_id'] for r in survey['selected_canonical']]==[r['target_id'] for r in plan['records']]
 assert not set(ids)&set(survey['excluded_union']);assert len(set(r['sequence'] for r in rows))==5
 assert all(sum(r['sequence']==s for r in rows)==4 for s in set(r['sequence'] for r in rows))
 for item in plan['records']:
  policies=item['policy_actions'];assert set(policies)=={'Q4','static360','folded128_fp32_fixed6','ordinary128_fp32_fixed6'} and policies['Q4']=='background_Q4' and policies['static360']=='static360'
  assert all(policies[n] in ['upgrade_'+str(i) for i in range(6)] for n in ('folded128_fp32_fixed6','ordinary128_fp32_fixed6'))
  assert item['unique_actions']==list(dict.fromkeys(policies.values()))
 assert plan['expected_formal_forwards']==sum(len(x['unique_actions']) for x in plan['records'])<=80
 native,_,metrics,_,_,*_=support['context'];can=metrics.canonical_rows(support['b'].canonical);can={r['target_id']:r for r in can} if isinstance(can,list) else can
 inventory={f['relative_path']:f for f in read(a.new_members.parent/'native_verification_files_v1.json')['files']}
 for rec in rows:
  assert rec['canonical']==can[rec['target_id']] and rec['canonical']['protocol_split']=='valid'
  files=native.target_files(rec['canonical']);assert {f['category'] for f in rec['source_files']}=={'input','gt','mask'}
  for f in rec['source_files']:
   assert f['relative_path']==files[f['category']];v=inventory[f['relative_path']];assert v['passed'] and v['all_elements_finite'] and v['sha256']==f['sha256']
   p=native.checked_path(support['b'].data_root,f['relative_path']);assert sha(p)==f['sha256'] and p.stat().st_size==int(f['bytes'])
 engineering=read(HERE/'ENGINEERING_REFERENCES_V1.json');oldrec=next(r for r in oldrows if r['target_id']==engineering['target_id'])
 assert engineering['target_id']==refs['roles']['fit']['target_id'] and engineering['heads']['background_Q4']==refs['roles']['fit']['heads']
 return dict(core=core,c=c,c151=c151,smoke=smoke,helper=helper,support=support,rows=rows,plan=plan,sc=old.sc,ct=ct,old151=old151,engineering=engineering,oldrec=oldrec)

def execute(a,p,result):
 import torch,numpy as np
 core,c,smoke,helper,support=(p[k] for k in ('core','c','smoke','helper','support'));assert not torch.cuda.is_initialized()
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261019);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_float32_matmul_precision('highest')
 device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261019);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 model,functional,bn=smoke.make_model(support,'full9',device,torch);cp=torch.load(a.weights,map_location='cpu',weights_only=False);state=c['endpoint']['state_sha256']
 assert cp['schema']=='flow_e0_rne4_joint_checkpoint_v1' and cp['completed_optimizer_steps']==512 and cp['final_state_sha256']==state
 model.load_state_dict(cp['model_state_dict'],strict=True);del cp;model.eval().requires_grad_(False);assert support['modules'][0].state_digest(model)==state
 bnsig=support['base'].bn_signature(model,torch);psnnames=[n for n,m in model.named_modules() if type(m).__name__=='PSN'];assert len(psnnames)==105
 result.update(status='RUNNING',BN_scope=bn,model_forwards=0,forward_attempts=0,formal_model_forwards=0,engineering_model_forwards=0,evaluations={},engineering={},runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),TF32=False,AMP=False))
 bounds={};exemplar={}
 def getbound(label):
  if label not in bounds:
   bounds[label]=p['sc'].BoundCore(core,c,True) if label=='static360' else p['ct'].BoundCore(core,p['c151'],-1 if label=='background_Q4' else int(label.split('_')[1]))
  return bounds[label]
 def input_for(rec):
  f=next(f for f in rec['source_files'] if f['category']=='input');path=support['b'].data_root/f['relative_path'];assert sha(path)==f['sha256'];z=np.load(path,allow_pickle=False);assert z.shape==(10,480,640) and z.dtype==np.float32 and np.isfinite(z).all()
  chunk=support['base'].prepare_input(z,device,torch);return chunk,f,helper.tensor_record(chunk)
 def forward(chunk,label,kind):
  bound=getbound(label);functional.reset_net(model);counts={n:0 for n in psnnames};hooks=[]
  with bound.installed(model,bound.mode,ste=False) as wraps:
   with helper.observed(model,bound,torch) as (bns,measured):
    try:
     for name in psnnames:
      def hook(m,ins,out,name=name):counts[name]+=1
      hooks.append(model.get_submodule(name).register_forward_hook(hook))
     assert result['forward_attempts']<84;result['forward_attempts']+=1
     with torch.no_grad():flows=model(chunk)['flow']
     result['model_forwards']+=1;result['formal_model_forwards' if kind=='formal' else 'engineering_model_forwards']+=1
    finally:
     for h in hooks:h.remove()
   work={n:copy.deepcopy(w.last_audit) for n,w in wraps.items()}
  assert all(x==1 for x in counts.values()) and len(flows)==4;heads={f'head{i}':helper.tensor_record(v[0]) for i,v in enumerate(flows)};pred=flows[3][0].detach().cpu().numpy().copy();del flows
  calls={n:len(v) for n,v in measured.items()}
  if label=='static360':checks=p['sc'].validate_work(work,calls,bns,c,core,True)
  else:checks=p['old151'].validate_work(work,calls,bns,p['c151'],core,-1 if label=='background_Q4' else int(label.split('_')[1]))
  digest=hashlib.sha256(json.dumps(work,sort_keys=True).encode()).hexdigest()
  if label in exemplar:assert digest==exemplar[label]
  else:exemplar[label]=digest;result.setdefault('first_work_by_action',{})[label]=work
  return pred,heads,dict(actual_BN_calls=bns,actual_affine_call_counts=calls,actual_PSN_calls=counts,work_checks=checks,work_sha256=digest)
 chunk,source,inputrec=input_for(p['oldrec'])
 for label in ('background_Q4','static360'):
  pred,heads,audit=forward(chunk,label,'engineering');assert all(heads[h]['sha256']==p['engineering']['heads'][label][h]['sha256'] for h in heads)
  result['engineering'][label]=dict(target_id=p['engineering']['target_id'],heads=heads,all_four_head_hashes_exact=True,actual_input=inputrec,**audit);del pred;write(a.output/'results.json',result)
 del chunk
 for i,(rec,planned) in enumerate(zip(p['rows'],p['plan']['records'])):
  chunk,source,inputrec=input_for(rec)
  for label in planned['unique_actions']:
   pred,heads,audit=forward(chunk,label,'formal');assert pred.shape==(2,480,640) and pred.dtype==np.float32 and np.isfinite(pred).all()
   path=a.output/'predictions'/label/(rec['target_id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,head3=pred)
   row=dict(target_id=rec['target_id'],sequence=rec['sequence'],role='new_identity_survival',action=label,policy_users=[name for name,action in planned['policy_actions'].items() if action==label],prediction_file=str(path.relative_to(a.output)),prediction_file_sha256=sha(path),prediction_sha256=heads['head3']['sha256'],heads=heads,input_source=source,actual_input=inputrec,**audit)
   result['evaluations'].setdefault(label,dict(frames=[]))['frames'].append(row);write(a.output/'records'/(rec['target_id']+'__'+label+'.json'),row);del pred
  del chunk;write(a.output/'progress.json',dict(identities_completed=i+1,formal=result['formal_model_forwards'],engineering=result['engineering_model_forwards'],attempts=result['forward_attempts']));write(a.output/'results.json',result);print('SURVIVAL',i+1,result['formal_model_forwards'],flush=True)
 assert result['formal_model_forwards']==p['plan']['expected_formal_forwards'] and result['engineering_model_forwards']==2
 assert result['forward_attempts']==result['model_forwards']==result['formal_model_forwards']+2<=84
 assert support['modules'][0].state_digest(model)==state and support['base'].bn_signature(model,torch)==bnsig
 result.update(status='R38_COMPLETE',final_state_sha256=state,completed_optimizer_steps=0,wall_seconds=time.monotonic()-start,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())

def main():
 ap=argparse.ArgumentParser()
 for n in ('e0-core','e3-root','r28-root','r26-root','members','new-members','weights','output','r35-root'):ap.add_argument('--'+n,type=Path,required=True)
 for n in ('runner-sha256','source-freeze-sha256'):ap.add_argument('--'+n,required=True)
 ap.add_argument('--preflight',type=Path);ap.add_argument('--execute',action='store_true');a=ap.parse_args();p=prepare(a)
 result=dict(schema='flow_r38_four_policy_collector_v1',status='CPU_PREFLIGHT_PASS',source_freeze_sha256=sha(HERE/'SOURCE_FREEZE_V1.json'),runner_sha256=sha(__file__),action_plan_sha256=sha(HERE/'ACTION_PLAN_V1.json'),membership_sha256=sha(HERE/'data/PROPOSED_SOURCE_MEMBERSHIP_V1.json'),selected_ids=[r['target_id'] for r in p['rows']],endpoint=p['c']['endpoint'],expected_formal_forwards=p['plan']['expected_formal_forwards'],expected_engineering_forwards=2,hardcap_attempts=84,wall_cap_seconds=900,GT_arrays_read=False,quality_computed=False,training_performed=False,resume_allowed=False)
 if not a.execute:
  assert a.preflight and not a.preflight.exists();write(a.preflight,result);return
 lock=p['support']['modules'][3].GPU_LOCK.open('a')
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(parents=True,exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('R38 hard900s cap')));signal.alarm(900)
  try:execute(a,p,result)
  except BaseException as error:result.update(status='R38_PARTIAL_FAILED',error=str(error),traceback=traceback.format_exc());write(a.output/'results.json',result);raise
  finally:signal.alarm(0)
  write(a.output/'results.json',result)
 finally:lock.close()
if __name__=='__main__':main()
