"""Independent completed R38 execution structure and all original-array quality.

No fitting/neural forward. The 20 identity roles/actions were fixed before GT.
Only original-output costs may be paired directly with existing cycle evidence.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
B=ROOT/'research/flow_story_20261022/round38_frozen_survival'
R34=ROOT/'research/flow_story_20261021/round34_deployability'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def aggregate(errors,counts,names,ids,groups,baseline):
 def sums(ix):
  out={}
  for j,name in enumerate(names):
   n=int(counts[ix,j].sum());e=float(errors[ix,j].sum());b=float(baseline[ix,j].sum());valid=ix&(counts[:,j]>0)
   out[name]=dict(valid_pixels=n,error_sum=e,AEE=e/n if n else None,reference_AEE=b/n if n else None,delta=(e-b)/n if n else None,relative=e/b-1 if n and b else None,
    frame_mean_AEE=float(np.mean(errors[valid,j]/counts[valid,j])) if valid.any() else None,reference_frame_mean_AEE=float(np.mean(baseline[valid,j]/counts[valid,j])) if valid.any() else None)
  return out
 seqs=list(dict.fromkeys(groups.tolist()));summaries={'overall':sums(np.ones(len(ids),bool))};leave={};tails=[]
 for seq in seqs:summaries[seq]=sums(groups==seq);leave[seq]=sums(groups!=seq)
 for i,tid in enumerate(ids):
  for name in ['all_valid','B','L']:
   j=names.index(name);n=int(counts[i,j]);e=errors[i,j];b=baseline[i,j]
   if n>=256 and (e>1.2*b or e-b>n):tails.append(dict(target_id=tid,sequence=str(groups[i]),region=name,valid_pixels=n,AEE=float(e/n),reference_AEE=float(b/n)))
 return dict(summaries=summaries,leave_one_sequence=leave,high_support_tail_review=tails,
  sequences_improved_all_valid=sum(summaries[s]['all_valid']['delta']<0 for s in seqs),leave_one_all_valid_deltas={s:leave[s]['all_valid']['delta'] for s in seqs})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=B/'actual/survival_v1');ap.add_argument('--source',type=Path,default=B/'data/source_snapshot')
 ap.add_argument('--output',type=Path,default=HERE/'R38_RAW_QUALITY_INDEPENDENT_V1.json');a=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.output.exists()
 pre=read(HERE/'R38_ACTION_AND_COLLECTOR_PRELAUNCH_V1.json');assert pre['status']=='PASS_PRELAUNCH'
 for name,h in read(B/'SOURCE_FREEZE_V1.json')['files'].items():assert sha(B/name)==h
 result=read(a.run/'results.json');assert result['status']=='R38_COMPLETE'
 assert result['schema']=='flow_r38_four_policy_collector_v1'
 assert result['source_freeze_sha256']==sha(B/'SOURCE_FREEZE_V1.json')==pre['execution_source_freeze_sha256']
 assert result['runner_sha256']==sha(B/'collect.py')==pre['collector_sha256']
 assert result['action_plan_sha256']==sha(B/'ACTION_PLAN_V1.json')==pre['action_plan_sha256']
 assert result['membership_sha256']==sha(B/'data/PROPOSED_SOURCE_MEMBERSHIP_V1.json')
 assert result['formal_model_forwards']==71 and result['engineering_model_forwards']==2 and result['model_forwards']==result['forward_attempts']==73
 assert result['hardcap_attempts']==84 and result['wall_cap_seconds']==900 and 0<result['wall_seconds']<900
 assert not result['GT_arrays_read'] and not result['quality_computed'] and not result['training_performed'] and not result['resume_allowed'] and result['completed_optimizer_steps']==0
 assert result['final_state_sha256']==result['endpoint']['state_sha256']=='7be9fed7c55332a150b4c9f67e863d5e68db2bb0d1829a5344703427a48ff018'
 members=read(B/'data/PROPOSED_SOURCE_MEMBERSHIP_V1.json')['records'];plan=read(B/'ACTION_PLAN_V1.json');ids=[r['target_id'] for r in members]
 assert ids==result['selected_ids']==[r['target_id'] for r in plan['records']]
 expected={(p['target_id'],action) for p in plan['records'] for action in p['unique_actions']};frames={};workhashes={}
 for label,work in result['first_work_by_action'].items():workhashes[label]=hashlib.sha256(json.dumps(work,sort_keys=True).encode()).hexdigest();assert len(work)==28
 for label,ev in result['evaluations'].items():
  for fr in ev['frames']:
   key=(fr['target_id'],label);assert key in expected and key not in frames and fr['action']==label and fr['role']=='new_identity_survival'
   assert len(fr['heads'])==4 and all(v['shape']==[2,480,640] and v['dtype']=='torch.float32' for v in fr['heads'].values())
   assert len(fr['actual_PSN_calls'])==105 and set(fr['actual_PSN_calls'].values())=={1}
   assert len(fr['actual_affine_call_counts'])==28 and set(fr['actual_affine_call_counts'].values())=={1}
   assert len(fr['actual_BN_calls'])==78 and set(fr['actual_BN_calls'].values())=={1}
   assert fr['work_sha256']==workhashes[label];frames[key]=fr
 assert set(frames)==expected and len(frames)==71
 refs=read(B/'ENGINEERING_REFERENCES_V1.json')
 for action,eng in result['engineering'].items():
  assert action in ['background_Q4','static360'] and eng['target_id']==refs['target_id'] and eng['heads']==refs['heads'][action]
  assert eng['all_four_head_hashes_exact'] and eng['work_sha256']==workhashes[action]
  assert len(eng['actual_BN_calls'])==78 and set(eng['actual_BN_calls'].values())=={1}
 assert set(result['engineering'])=={'background_Q4','static360'}
 biaspath=R34/'bias_control/actual_v1/BIAS_MODEL_V1.json';bias=np.array(read(biaspath)['bias'],dtype=np.float64)
 assert sha(biaspath)==read(R34/'independent_review/BIAS_INDEPENDENT_AUDIT_V1.json')['bias_model_sha256']
 regpath=ROOT/'research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json';reg=read(regpath)
 assert sha(regpath)=='2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b'
 helperpath=ROOT/'research/flow_story_20261021/round33_predictability/independent_review/audit_pack_actual_v1.py';helper=module(helperpath,'_independent_R38_region_masks')
 sources={};predictions={};rawframes=[];policies=list(plan['records'][0]['policy_actions'])
 for rec,planned in zip(members,plan['records']):
  tid=rec['target_id'];source={}
  for spec in rec['source_files']:
   path=(B/'data/input_snapshot'/(tid+'.npy')) if spec['category']=='input' else a.source/spec['relative_path']
   assert sha(path)==spec['sha256'] and path.stat().st_size==int(spec['bytes']);sources[str(path)]=sha(path)
   source[spec['category']]=np.load(path,allow_pickle=False)
  masks=helper.masks_from_sources(source['gt'],source['mask'],source['input'],reg['boundary_threshold_pixels'],reg['large_displacement_threshold_pixels'])
  names=list(masks);counts=[int(m.sum()) for m in masks.values()];state_errors={};inputref=frames[tid,'background_Q4']['actual_input']
  for action in planned['unique_actions']:
   row=frames[tid,action];assert row['sequence']==rec['sequence'] and row['actual_input']==inputref
   assert row['input_source']==next(s for s in rec['source_files'] if s['category']=='input')
   assert row['policy_users']==[p for p,action2 in planned['policy_actions'].items() if action2==action]
   path=a.run/row['prediction_file'];assert sha(path)==row['prediction_file_sha256']
   with np.load(path,allow_pickle=False) as z:pred=z['head3'].copy();assert z.files==['head3']
   assert pred.shape==(2,480,640) and pred.dtype==np.float32 and np.isfinite(pred).all()
   assert digest(pred)==row['prediction_sha256']==row['heads']['head3']['sha256'];predictions[tid+'__'+action]=sha(path)
   state_errors[action]={}
   for cond,b in [('original',np.zeros(2)),('same_bias_corrected',bias)]:
    corrected=pred.astype(np.float64)-b[:,None,None];error=np.sqrt(np.sum((corrected-source['gt'].astype(np.float64))**2,axis=0))
    state_errors[action][cond]=[float(error[m].sum()) for m in masks.values()]
  rawframes.append(dict(target_id=tid,sequence=rec['sequence'],counts=counts,policy_actions=planned['policy_actions'],action_errors=state_errors))
 groups=np.array([r['sequence'] for r in members]);counts=np.array([r['counts'] for r in rawframes]);tables={}
 for cond in ['original','same_bias_corrected']:
  errors={p:np.array([r['action_errors'][r['policy_actions'][p]][cond] for r in rawframes]) for p in policies}
  tables[cond]={p:{'versus_'+q:aggregate(errors[p],counts,names,ids,groups,errors[q]) for q in policies} for p in policies}
 report=dict(schema='flow_r38_original_array_independent_quality_v1',status='PASS',auditor_sha256=sha(__file__),raw_result_sha256=sha(a.run/'results.json'),prelaunch_audit_sha256=sha(HERE/'R38_ACTION_AND_COLLECTOR_PRELAUNCH_V1.json'),membership_sha256=sha(B/'data/PROPOSED_SOURCE_MEMBERSHIP_V1.json'),action_plan_sha256=sha(B/'ACTION_PLAN_V1.json'),bias_model_sha256=sha(biaspath),region_contract_sha256=sha(regpath),region_helper_sha256=sha(helperpath),
  counts=dict(identities=20,sequences=5,source_arrays=60,unique_predictions=71,formal_forwards=71,engineering_forwards=2,actual_raw_region_condition_values=71*16*2,policy_frame_region_condition_values=20*4*16*2),execution=dict(model_forwards=result['model_forwards'],wall_seconds=result['wall_seconds'],max_allocated_bytes=result['max_allocated_bytes'],max_reserved_bytes=result['max_reserved_bytes'],state_sha256=result['final_state_sha256']),
  region_names=names,input_source_sha256=sources,prediction_sha256=predictions,raw_frames=rawframes,independent_quality=tables,GPU_calls=0,new_fits=0,
  scope='Frozen FP32 four-policy survival on20 newly admitted identities in five previously used repository-valid sequences; incomplete historical exposure cannot prove backbone-unseen. No action oracle/regret because only chosen actions were collected. Original outputs align to existing component cost; identical frozen bias adds a common unmodeled postprocess. Work structure/hash checked here; detailed kernel/service golden audit is separately attributed.')
 write(a.output,report)
 print(json.dumps(dict(status='PASS',audit_sha256=sha(a.output),key={c:{p:tables[c][p]['versus_Q4']['summaries']['overall']['all_valid'] for p in policies} for c in tables})))
if __name__=='__main__':main()
