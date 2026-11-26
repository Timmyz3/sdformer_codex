#!/usr/bin/env python3
"""E0 ordinary quantizer diagnosis and one fixed RNE4 joint recovery. No dev GT read."""
import argparse,collections,copy,fcntl,hashlib,importlib.util,json,signal,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
import mixed_integer_core as core
import mask_contract as masks
CORE_SHA='ce07ae04c9f3d58738ae134cdc1c8276542a41a04584720b5d69359e220d1f9f'
MASK_SHA='215d8711d31d3daf2fcc0771b96acb64c47fe0fad9a2231b989a7133c4ed53cb'
SMOKE_SHA='6dadb497bf97fc33929209a3feed0acedcac241036d27bbcda502d8f35244d83'
HELPER_SHA='5a5bf2abaca8fe23fc269e4dc830acbd3ee6537e52958c29c1520543756a4f95'
MEMBERS_SHA=masks.MEMBERS_SHA
ORIGINAL='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
GRAPH='flow_e0_rne4_independent_scales_all12mlp_patch4_currentBN_v1'
R30_CORE_SHA='ceec6924fd07a33fa7b806dce7ee4699a4a438a08d6a342b24137d216e7a13d5'
SELECTION_SHA='60b7d6d525d4130f5f0efecaff95511808b2186c3aaba9e9f37a957db32ef31f'
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
 require(sha(__file__)==a.runner_sha256 and sha(HERE/'mixed_integer_core.py')==CORE_SHA and sha(HERE/'mask_contract.py')==MASK_SHA,'Explicit E0 sources')
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
 refs={} if (a.smoke or a.diagnostic) else {'initial_Q8':initial_references(a.reference_contract,m)['initial_Q8']}
 if a.diagnostic:
  require(a.selections and sha(a.selections)==SELECTION_SHA,'Root-frozen E0/E1 localization identities')
  select=read(a.selections);ids=select['localization_ids'];rows={x['target_id']:x for x in m['dev_members']}
  require(len(ids)==len(set(ids))==16 and all(x in rows for x in ids),'Exactly16 original development identities')
  require(select['localization_members']==[rows[x] for x in ids],'Exact canonical localization metadata')
  require(sha(a.r30_core)==R30_CORE_SHA,'Old Q8/M implementation bytes')
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
 cp=dict(schema='flow_e0_rne4_joint_checkpoint_v1',arm='joint',model_state_dict=state,optimizer_state_dict=opt.state_dict(),completed_optimizer_steps=len(orders),
  initial_state_sha256=ORIGINAL,final_state_sha256=final,members_sha256=MEMBERS_SHA,sources=result['sources'],recipe=result['recipe'],joint_contract=result['joint_contract'],
  torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),fixed_endpoint_no_development_selection=True,formal_resume_allowed=False,quantizer='independent_maxabs_rne4',checkpoint_role='endpoint_only_not_eligible_for_training_resume')
 file=a.output/('smoke_weights.pt' if a.smoke else 'recovered_step512.pt');tmp=file.with_suffix('.pt.partial');torch.save(cp,tmp);tmp.replace(file)
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
  result['BN_scope']=bn;result['weight_error_reports']={'initial':weight_error_report(model,a.output/'weight_error_initial.json',ORIGINAL)}
  for mode in ('Q4','M'):
   a.arm=mode;bound=mixed if mode=='M' else masks.BoundCore()
   helper.evaluate(a,p,m,smoke,bound,model,functional,'initial_'+mode,torch,np,result)
   result['evaluations']['initial_'+mode]['mask_layout']=mask_record if mode=='M' else None
  del model;torch.cuda.empty_cache()
 state,final=train(a,smoke,helper,p,m,contract,result,torch)
 model,functional,bn=smoke.make_model(p,'full9',device,torch);model.load_state_dict(state,strict=True);del state;model.eval().requires_grad_(False)
 require(r13.state_digest(model)==final,'Exact final recovered state at full geometry');result.setdefault('weight_error_reports',{})['final']=weight_error_report(model,a.output/'weight_error_final.json',final);signature=p['base'].bn_signature(model,torch)
 for mode in (('M',) if a.smoke else ('Q4','Q8','M')):
  bound=mixed if mode=='M' else masks.BoundCore();layout=mask_record if mode=='M' else None
  parity(a,smoke,helper,p,m,first,model,functional,mode,bound,layout,result,torch,np)
  if not a.smoke:
   a.arm=mode;helper.evaluate(a,p,m,smoke,bound,model,functional,'endpoint_'+mode,torch,np,result)
   result['evaluations']['endpoint_'+mode]['mask_layout']=layout
 require(result['model_forwards']==(5 if a.smoke else 1353) and r13.state_digest(model)==final and p['base'].bn_signature(model,torch)==signature,'Complete fixed model budget and unchanged final inference state')
 result.update(status='MIXED_ONE_UPDATE_AND_PARITY_COMPLETE' if a.smoke else 'RNE4_JOINT512_AND320_DEVELOPMENT_PREDICTIONS_COMPLETE',
  final_state_sha256=final,training_performed=True,development_GT_read=False,quality_computed=False,quality_selection=False,
  wall_seconds=time.monotonic()-start,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())

def weight_error_report(model,path,state_sha256):
 """Pure weight diagnostics, never a task-quality score or selector input."""
 import torch
 out={}
 for name in core.TARGETS:
  w=model.get_submodule(name).weight.detach().reshape(model.get_submodule(name).weight.shape[0],-1).cpu()
  q,s=core.quantize(w);q=q.cpu();s=s.cpu();a=w.abs().amax(1);s4=a/7.;div=torch.where(s4==0,torch.ones_like(s4),s4)
  q4=torch.round(w/div[:,None]).clamp(-8,7).to(torch.int8)
  vals={'Q8':q.float()*s[:,None], 'drop4':((q.to(torch.int16)>>4)<<4).float()*s[:,None],
        'midpoint':(((q.to(torch.int16)>>4)<<4).float()+8)*s[:,None],
        'rne4':((q4.to(torch.int16)<<4).to(torch.int8)).float()*(s4/16.)[:,None]}
  out[name]={}
  for variant,v in vals.items():
   e=(v-w).double();flat=e.reshape(len(e),-1)
   out[name][variant]=dict(elements=e.numel(),mean=float(e.mean()),MAE=float(e.abs().mean()),MSE=float(e.square().mean()),
    per_output_mean=flat.mean(1).tolist(),per_output_MAE=flat.abs().mean(1).tolist(),per_output_MSE=flat.square().mean(1).tolist())
 write(path,dict(schema='flow_e0_weight_error_v1',GT_read=False,task_quality=False,state_sha256=state_sha256,layers=out))
 return dict(file=path.name,sha256=sha(path),state_sha256=state_sha256)


def diagnose(a,smoke,helper,p,m,contract,result):
 import torch,numpy as np
 require(not torch.cuda.is_initialized(),'Global lock before CUDA')
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261019)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_float32_matmul_precision('highest')
 device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261019);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 old=load(a.r30_core,'_e0_frozen_r30_core')
 selected,layout=masks.materialize(contract,'full9');selection=read(a.selections);rows={r['target_id']:r for r in m['dev_members']};members=[rows[t] for t in selection['localization_ids']]
 r13=p['modules'][0];model,functional,bn=smoke.make_model(p,'full9',device,torch);model.requires_grad_(False)
 require(r13.state_digest(model)==ORIGINAL,'All E0 diagnoses use original W0')
 signature=p['base'].bn_signature(model,torch);result['weight_error_reports']={'initial':weight_error_report(model,a.output/'weight_error_initial.json',ORIGINAL)}
 result.update(status='RUNNING_DIAGNOSTIC',initial_state_sha256=ORIGINAL,BN_scope=bn,
  runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),AMP=False,TF32=False),
  diagnostic_selection_sha256=sha(a.selections),diagnostic_ids=selection['localization_ids'],quality_model_forwards=0,engineering_model_forwards=0,
  diagnostic_states=['original_Q8','original_M_drop4','original_M_midpoint','original_M_rne4'])
 specs=[('original_Q8','Q8',masks.BoundCore(engine=old)),('original_M_drop4','M',masks.BoundCore(selected,engine=old)),
        ('original_M_midpoint','M',masks.BoundCore(selected,variant='midpoint')),('original_M_rne4','M',masks.BoundCore(selected,variant='rne4'))]
 for stage,mode,bound in specs:
  folder=a.output/'predictions'/stage;folder.mkdir(parents=True,exist_ok=False);frames=[];tic=time.monotonic()
  for index,row in enumerate(members):
   evidence=helper.source_evidence(m,row['target_id']);events=helper.event_only(p,row,evidence,np);chunk=p['base'].prepare_input(events,device,torch);del events
   do_parity=a.diagnostic_parity and index==0 and stage in ('original_M_midpoint','original_M_rne4')
   if do_parity:
    actual,rec=smoke.capture_full(model,functional,chunk,mode,bound,torch,ste=False)
    result['model_forwards']+=1;heads={f'head{i}':helper.tensor_record(actual[f'head{i}'][0]) for i in range(4)}
    prediction=actual['head3'][0].numpy();work=helper.work_summary(rec['work']);bn_calls=rec['actual_BN_calls'];affine_counts=rec['actual_target_calls']
    comparisons={}
    for label,ste,reference in (('training',True,False),('dense_integer',False,True)):
     check,rr=smoke.capture_full(model,functional,chunk,mode,bound,torch,ste=ste,reference=reference)
     result['model_forwards']+=1;result['engineering_model_forwards']+=1;cmp=smoke.compare_full(check,actual,torch)
     comparisons[label]=dict(**cmp,record=rr);del check
     require(cmp['pass'],'Actual diagnostic full graph parity '+stage+' '+label)
    pf=a.output/(stage+'_parity_heads.npz');np.savez_compressed(pf,**{k:v.numpy() for k,v in actual.items() if k.startswith('head')})
    result.setdefault('diagnostic_parity',{})[stage]=dict(input_target_id=row['target_id'],deployment_record=rec,comparisons=comparisons,heads_file=pf.name,heads_sha256=sha(pf));del actual
   else:
    functional.reset_net(model)
    with bound.installed(model,mode,ste=False) as wrappers:
     with helper.observed(model,bound,torch) as (bn_calls,calls):
      with torch.no_grad():flows=model(chunk)['flow']
     work=helper.work_summary({n:copy.deepcopy(w.last_audit) for n,w in wrappers.items()})
    result['model_forwards']+=1
    require(len(flows)==4 and all(tuple(v.shape)==(1,2,480,640) and bool(torch.isfinite(v).all()) for v in flows),'Four finite native heads')
    heads={f'head{i}':helper.tensor_record(v[0]) for i,v in enumerate(flows)};prediction=flows[3][0].detach().cpu().numpy();affine_counts={n:len(v) for n,v in calls.items()};del flows
   result['quality_model_forwards']+=1
   file=folder/(row['target_id']+'.npz');np.savez_compressed(file,head3=prediction)
   frames.append(dict(target_id=row['target_id'],sequence=row['sequence'],input_evidence=evidence,prediction_file=str(file.relative_to(a.output)),
    prediction_file_sha256=sha(file),prediction_sha256=heads['head3']['sha256'],heads=heads,actual_BN_calls=bn_calls,actual_affine_call_counts=affine_counts,work=work))
   del prediction,chunk
   write(a.output/'diagnostic_progress.json',dict(stage=stage,completed_state_frames=len(frames),model_forwards=result['model_forwards'],quality_model_forwards=result['quality_model_forwards'],engineering_model_forwards=result['engineering_model_forwards']))
  result.setdefault('evaluations',{})[stage]=dict(graph_id='flow_e0_'+stage+'_allcurrent_v1',state_sha256=ORIGINAL,frames=frames,model_forwards=16,GT_arrays_read=False,mask_layout=layout if mode=='M' else None,wall_seconds=time.monotonic()-tic)
  write(a.output/'results.json',result);print(json.dumps(dict(stage=stage,frames=16)),flush=True)
  require(r13.state_digest(model)==ORIGINAL and p['base'].bn_signature(model,torch)==signature,'Unchanged W0/current-BN after each diagnostic arm')
 require(result['quality_model_forwards']==64 and result['model_forwards']==64+(4 if a.diagnostic_parity else 0),'Exact diagnostic count')
 result.update(status='FOUR_ORDINARY_STATES_16_IDENTITIES_COMPLETE',final_state_sha256=ORIGINAL,training_performed=False,development_GT_read=False,
  quality_computed=False,quality_selection=False,wall_seconds=time.monotonic()-start,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--task',choices=('diagnostic','recovery','smoke'),required=True)
 p.add_argument('--r30-core',type=Path);p.add_argument('--selections',type=Path);p.add_argument('--diagnostic-parity',action='store_true')
 for name in ('r28-root','r26-root','members','mask-contract','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--mask-contract-sha256',required=True);p.add_argument('--runner-sha256',required=True)
 p.add_argument('--reference-contract',type=Path);p.add_argument('--preflight-output',type=Path);p.add_argument('--execute',action='store_true')
 p.add_argument('--wall-minutes',type=int,required=True);a=p.parse_args();a.smoke=a.task=='smoke';a.diagnostic=a.task=='diagnostic'
 require(not a.diagnostic or a.r30_core is not None,'Exact old core file for diagnostic')
 require(a.wall_minutes==(20 if (a.smoke or a.diagnostic) else 60),'Predeclared20min smoke/60min formal cap')
 require(a.smoke or a.diagnostic or a.reference_contract is not None,'Formal original-Q4/Q8 evidence descriptors')
 smoke,helper,support,m,first,contract,refs=prepare(a)
 result=dict(schema='flow_e0_ordinary_quantizer_v1',graph_id=GRAPH,status='PREFLIGHT_PASS',arm='joint',engineering_smoke=a.smoke,task=a.task,diagnostic_parity_requested=a.diagnostic_parity,
  sources=dict(runner=sha(__file__),core=CORE_SHA,mask_helper=MASK_SHA,smoke_helpers=SMOKE_SHA,formal_helpers=HELPER_SHA,members=MEMBERS_SHA,mask_contract=a.mask_contract_sha256),
  recipe=helper.recipe(smoke),joint_contract=dict(mode_counts={} if a.diagnostic else ({'M':1} if a.smoke else {'Q4':256,'Q8':128,'M':128}),schedule='Q4,Q4,Q8,M[(visit-1+sequence_rank+2*(pass-1))mod4]',
    seed=20261019,mask_contract=contract,mask_contract_sha256=a.mask_contract_sha256,smoke_state_never_resumed=True),
  original_initial_references=refs,loader_source_receipt=support['binding'],model_forwards=0,expected_model_forwards=(64+(4 if a.diagnostic_parity else 0)) if a.diagnostic else (5 if a.smoke else 1353),
  expected_quality_predictions=64 if a.diagnostic else (0 if a.smoke else 320),formal_resume_allowed=False,
  completed_optimizer_steps=0,training_performed=False,development_GT_read=False,quality_computed=False,quality_selection=False,formal_resume_from_smoke=False,wall_cap_minutes=a.wall_minutes)
 if a.diagnostic:
  result['graph_id']='flow_e0_four_quantizer_diagnostic_allcurrent_v1'
  result['arm']='diagnostic'
  result['recipe']=dict(updates=0,identity_count=16,states=4,source_weight='original_W271482',
    ordinary_quantizers=['Q8','R30_drop4_M','midpoint_M','independent_RNE4_M'],
    masks='fixed_R30_full9',all_parameters_unchanged=True,BN='all78current')
  result['sources'].update(old_r30_core=R30_CORE_SHA,selections=SELECTION_SHA)
 if not a.execute:
  require(a.preflight_output and not a.preflight_output.exists(),'Fresh CPU preflight receipt');write(a.preflight_output,result);return
 lock=support['modules'][3].GPU_LOCK.open('a')
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(parents=True,exist_ok=False)
  def alarm(signum,frame):raise TimeoutError('E0 fixed wall cap; no partial endpoint eligible')
  signal.signal(signal.SIGALRM,alarm);signal.alarm(a.wall_minutes*60)
  try:
   if a.diagnostic:diagnose(a,smoke,helper,support,m,contract,result)
   else:execute(a,smoke,helper,support,m,first,contract,result)
  except BaseException as e:
   result.update(status='E0_DIAGNOSTIC_FAILED' if a.diagnostic else 'RNE4_JOINT_RECOVERY_FAILED',error=str(e),traceback=traceback.format_exc());write(a.output/'results.json',result);raise
  finally:signal.alarm(0)
  write(a.output/'results.json',result)
 finally:lock.close()
if __name__=='__main__':main()
