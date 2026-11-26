#!/usr/bin/env python3
"""Post-execution CPU training diagnostics; no development predictions or GT.

This does not select a checkpoint, rerun a model, alter a recipe, or establish
generalization. Every formal arm and its independent execution audit must finish.
"""
import argparse,collections,hashlib,json,math,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
CHECKPOINT_SHA='1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'
INITIAL_SHA='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
MEMBERS_SHA='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'
RUNNER_SHA='495f7dbfdc9d54f74ed2f40ef5cfb9c6fcc58da5a03998efca733845d0b6753a'
PREFIX='sttmultires_unet.encoders.swin3d.'
FC={PREFIX+f'layers.{s}.swin_blocks.{b}.mlp.{fc}.weight' for s,d in enumerate((2,2,6,2)) for b in range(d) for fc in ('fc1','fc2')}
CONV={PREFIX+f'patch_embed.residual_encoding.resblocks.{b}.conv{j}.0.weight' for b in (0,1) for j in (1,2)}
NONE={PREFIX+f'layers.{s}.swin_blocks.{b}.attn.attn_sn.spiking_neuron.{q}' for s,d in enumerate((2,2,6,2)) for b in range(d) for q in ('weight','bias')}
GROUPS=('FC','conv','BN','PSN','decoder','other')
def require(ok,why):
 if not ok:raise ValueError(why)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(state):
 h=hashlib.sha256()
 for name,t in sorted(state.items()):
  t=t.detach().cpu().contiguous()
  for value in (name.encode(),str(t.dtype).encode(),str(tuple(t.shape)).encode(),t.numpy().tobytes()):h.update(value)
 return h.hexdigest()
def dist(values):
 a=np.asarray(values,dtype=np.float64);require(a.ndim==1 and len(a)>0 and np.isfinite(a).all(),'Finite nonempty distribution')
 qs=np.quantile(a,[.05,.25,.5,.75,.95],method='linear')
 return dict(n=len(a),mean=float(a.mean()),population_std=float(a.std(ddof=0)),min=float(a.min()),max=float(a.max()),
             **{k:float(v) for k,v in zip(('p05','p25','median','p75','p95'),qs)})
def summarize_pass(rows):
 require(len(rows)==256 and len({x['target_id'] for x in rows})==256,'Complete256 distinct training visits')
 keys=('supervised_loss_px','teacher_mean_L1_px','combined_loss','preclip_grad_norm')
 require(all(x['gradients_finite'] and x['gradient_parameter_count']==453 for x in rows),'Actual finite453 gradients each update')
 norms=torch.tensor([x['preclip_grad_norm'] for x in rows],dtype=torch.float32)
 coef=torch.clamp(100./(norms+1e-6),max=1.)
 return dict(distributions={k:dist([x[k] for x in rows]) for k in keys},teacher_weighted_contribution=dist([.1*x['teacher_mean_L1_px'] for x in rows]),
  norm_gt_100_count=int((norms>100.).sum()),reconstructed_FP32_clip_coefficient_lt_one_count=int((coef<1.).sum()),
  minimum_reconstructed_clip_coefficient=float(coef.min()),wall_seconds_sum=float(sum(x['wall_seconds'] for x in rows)))
def classify(name,tensor,bn_names):
 """Mutually exclusive groups; BN and PSN take precedence over location."""
 if name in bn_names:return 'BN'
 if '.spiking_neuron.' in name:return 'PSN'
 if name.startswith(('sttmultires_unet.decoders.','sttmultires_unet.preds.')):return 'decoder'
 if name in FC or '.attn.linear_q.' in name or '.attn.linear_k.' in name or '.attn.proj.' in name or '.downsample.reduction.' in name:return 'FC'
 if name in CONV or (tensor.ndim==4 and name.endswith('.weight')):return 'conv'
 return 'other'
def displacement(before,after):
 require(before.dtype==after.dtype==torch.float32 and before.shape==after.shape and bool(torch.isfinite(after).all()),'Matching finite FP32 parameter')
 x=before.double();d=after.double()-x
 s0=float((x*x).sum());sd=float((d*d).sum());sf=float((after.double()**2).sum())
 numeric_changed=int((before!=after).sum());bytewise=bool(torch.equal(before.contiguous().view(torch.uint8),after.contiguous().view(torch.uint8)))
 return dict(elements=before.numel(),initial_squared_L2=s0,delta_squared_L2=sd,final_squared_L2=sf,
  absolute_delta_L2=math.sqrt(sd),relative_delta_L2=math.sqrt(sd/s0) if s0 else None,zero_initial_norm=s0==0,
  max_absolute_delta=float(d.abs().max()),changed_numeric_elements=numeric_changed,bytewise_unchanged=bytewise)
def aggregate_parameters(rows):
 require(rows,'Nonempty parameter aggregation')
 s0=sum(x['initial_squared_L2'] for x in rows);sd=sum(x['delta_squared_L2'] for x in rows)
 return dict(parameter_tensors=len(rows),elements=sum(x['elements'] for x in rows),initial_L2=math.sqrt(s0),absolute_delta_L2=math.sqrt(sd),
  relative_delta_L2=math.sqrt(sd/s0) if s0 else None,zero_initial_norm=s0==0,
  changed_parameter_tensors=sum(not x['bytewise_unchanged'] for x in rows),changed_numeric_elements=sum(x['changed_numeric_elements'] for x in rows),
  max_absolute_delta=max(x['max_absolute_delta'] for x in rows))
def run(a):
 require(not a.output.exists(),'Fresh diagnostics output')
 audit=read(a.audit);require(audit['schema']=='flow_r27_independent_actual_execution_audit_v1' and audit['status']=='PASS'
  and audit['counts']['optimizer_updates']==1536 and audit['counts']['quality_predictions']==384,'All three formal arms independently audited')
 require(audit['auditor_sha256']==sha(HERE.parent/'audit_actual_recovery.py'),'Bound independent auditor source')
 members_path=HERE.parent/'independent_review/membership_v1.json';require(sha(members_path)==MEMBERS_SHA,'Fixed membership')
 members=read(members_path);require(sha(a.checkpoint)==CHECKPOINT_SHA,'Original actual W271482')
 source=torch.load(a.checkpoint,map_location='cpu',weights_only=True);source=source['model_state_dict']
 bn_prefix={k.removesuffix('.running_mean') for k in source if k.endswith('.running_mean')}
 require(len(bn_prefix)==78,'Exact original78 BN inventory')
 bn_names={p+'.'+s for p in bn_prefix for s in ('weight','bias')}
 initial={k:v for k,v in source.items() if not k.endswith(('.running_mean','.running_var','.num_batches_tracked'))};del source
 require(len(initial)==477 and digest(initial)==INITIAL_SHA and bn_names<=set(initial),'Exact original477 parameter state')
 arms={}
 for arm in ('native','Q8','Q6'):
  folder=a.actual_root/('formal_'+arm+'_v1');result_path=folder/'results.json';result=read(result_path)
  require(sha(result_path)==audit['source_results'][arm] and result['sources']['runner']==RUNNER_SHA,'Audited final result/source identity')
  require(result['completed_optimizer_steps']==512 and result['initial_state_sha256']==INITIAL_SHA,'Fixed final endpoint and anchor')
  ledger=folder/result['training_ledger_file'];require(sha(ledger)==result['training_ledger_sha256'],'Audited actual ledger')
  steps=[json.loads(line) for line in ledger.read_text().splitlines()];require(len(steps)==512,'Full512 ledger')
  for x,y in zip(steps,members['train_order']):require(all(x[k]==y[k] for k in ('step','pass_index','target_id','sequence','crop')),'Actual fixed order/crop')
  weights=folder/result['weights_file'];require(sha(weights)==result['weights_sha256']==audit['arms'][arm]['checkpoint_sha256'],'Audited final weights bytes')
  cp=torch.load(weights,map_location='cpu',weights_only=True);final=cp['model_state_dict'];opt=cp['optimizer_state_dict']
  require(len(final)==477 and digest(final)==result['final_state_sha256'],'Actual final state digest')
  ids=opt['param_groups'][0]['params'];names=result['trainable_names'];require(len(ids)==len(names)==477,'Actual optimizer parameter mapping')
  active={name for idx,name in zip(ids,names) if idx in opt['state']};inactive=set(names)-active
  require(len(active)==453 and inactive==NONE,'Exact453 active and24 unused parameters')
  require(all(float(v['step'])==512 for v in opt['state'].values()),'All optimizer states step512')
  parameters={}
  for name,x in initial.items():
   row=displacement(x,final[name]);row.update(group=classify(name,x,bn_names),active=name in active,quantized_target=name in FC|CONV)
   parameters[name]=row
  require(all(parameters[n]['bytewise_unchanged'] for n in NONE),'All24 no-gradient tensors exactly unchanged')
  group_rows={g:[r for r in parameters.values() if r['active'] and r['group']==g] for g in GROUPS}
  groups={g:aggregate_parameters(rows) for g,rows in group_rows.items() if rows}
  require(sum(x['parameter_tensors'] for x in groups.values())==453,'Exhaustive nonoverlapping active groups')
  same_crop=sum(steps[i]['crop']==steps[i+256]['crop'] for i in range(256))
  arms[arm]=dict(result_sha256=sha(result_path),weights_sha256=sha(weights),ledger_sha256=sha(ledger),
   passes={'pass1':summarize_pass(steps[:256]),'pass2':summarize_pass(steps[256:])},
   fixed512_trajectory= {k:dist([x[k] for x in steps]) for k in ('supervised_loss_px','teacher_mean_L1_px','combined_loss','preclip_grad_norm')},
   same_identity_visits=256,same_crop_between_visits=same_crop,different_crop_between_visits=256-same_crop,
   active_parameter_groups=groups,all_active=aggregate_parameters([r for r in parameters.values() if r['active']]),
   quantized_target28=aggregate_parameters([r for r in parameters.values() if r['quantized_target']]),
   unused24_all_bytewise_unchanged=True,unused24_names=sorted(NONE),parameters=parameters)
  del cp,final,opt,parameters
 out=dict(schema='flow_r27_training_diagnostics_v1',status='COMPLETE',source_sha256=sha(__file__),execution_audit_sha256=sha(a.audit),
  original_checkpoint_sha256=CHECKPOINT_SHA,membership_sha256=MEMBERS_SHA,arms=arms,GT_values_read=False,development_predictions_read=False,
  model_forwards=0,optimizer_updates_performed=0,GPU_used=False,
  grouping='Mutually exclusive: BN affine by original buffers, then PSN, decoder/prediction remaining parameters, FC, conv, other. Quantized28 additionally reported as a clearly overlapping secondary subset.',
  limits=['Pass1/pass2 use the same identities with separately fixed crops; their distributions are descriptive trajectories, not paired same-input improvement.',
   'Relative parameter displacement is undefined (null) when initial group/tensor norm is zero; no arbitrary epsilon is inserted.',
   'These training diagnostics neither establish development quality nor select checkpoints or change the fixed recipe.',
   'Clipping is summarized by stored preclip norm and an explicitly reconstructed FP32 clip coefficient; no backward is rerun.'])
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':'COMPLETE','output_sha256':sha(a.output),'active_group_counts':{a:{g:v['parameter_tensors'] for g,v in x['active_parameter_groups'].items()} for a,x in arms.items()}}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--actual-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.set_num_interop_threads(1);run(a)
