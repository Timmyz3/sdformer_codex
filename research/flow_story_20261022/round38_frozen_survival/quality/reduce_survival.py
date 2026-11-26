"""R38 fixed four-policy quality; no head fitting, action selection, or GPU."""
from pathlib import Path
import os,sys,time,hashlib,json,collections
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;R38=HERE.parent
ROOT=R38.parents[2];R34=ROOT/'research/flow_story_20261021/round34_deployability';R33=R34.parent/'round33_predictability'
sys.path.insert(0,str(R34/'bias_control'))
import run_bias as bc
u=bc.u
POLICIES=('Q4','static360','folded128_fp32_fixed6','ordinary128_fp32_fixed6')

def main():
 start=time.monotonic();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):assert os.environ.get(k)=='2'
 freeze=u.read(HERE/'SOURCE_FREEZE_V1.json')
 for name,h in freeze['files'].items():assert u.sha(HERE/name)==h
 for item in freeze['bound'].values():assert u.sha(item['path'])==item['sha256']
 out=HERE/'actual_v1';assert not out.exists();out.mkdir()
 ev=u.imported('r38_old_eval',R33/'evaluate.py');red=u.imported('r38_old_raw',R33.parent/'independent_review/reduce_e1_quality_v1.py');raw=red.raw
 regionpath=ROOT/'research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json';assert u.sha(regionpath)==red.REGIONS_SHA;regions=u.read(regionpath)
 run=R38/'actual/survival_v1';result=u.read(run/'results.json');plan=u.read(R38/'ACTION_PLAN_V1.json');members=u.read(R38/'data/PROPOSED_SOURCE_MEMBERSHIP_V1.json')['records']
 assert result['status']=='R38_COMPLETE' and result['formal_model_forwards']==plan['expected_formal_forwards']==71 and result['engineering_model_forwards']==2
 assert result['model_forwards']==result['forward_attempts']==73 and result['completed_optimizer_steps']==0
 assert result['source_freeze_sha256']==u.sha(R38/'SOURCE_FREEZE_V1.json') and result['action_plan_sha256']==u.sha(R38/'ACTION_PLAN_V1.json')
 assert result['final_state_sha256']==result['endpoint']['state_sha256'] and result['GT_arrays_read'] is False and result['quality_computed'] is False
 assert all(v['all_four_head_hashes_exact'] for v in result['engineering'].values())
 ids=[r['target_id'] for r in members];assert len(ids)==len(set(ids))==20 and ids==result['selected_ids']==[r['target_id'] for r in plan['records']]
 rows={(r['target_id'],action):r for action,v in result['evaluations'].items() for r in v['frames']};assert len(rows)==71
 expected={(p['target_id'],a) for p in plan['records'] for a in p['unique_actions']};assert set(rows)==expected
 bpath=R34/'bias_control/actual_v1/BIAS_MODEL_V1.json';b=np.asarray(u.read(bpath)['bias'],dtype=np.float64)
 bindings={};quality=[];zero=0
 for rec,planned in zip(members,plan['records']):
  tid=rec['target_id'];src={};srcbind={}
  for f in rec['source_files']:
   cat=f['category'];path=R38/'data/input_snapshot'/(tid+'.npy') if cat=='input' else red.child(R38/'data/source_snapshot',f['relative_path'])
   assert path.stat().st_size==int(f['bytes']) and u.sha(path)==f['sha256']
   src[cat]=np.load(path,allow_pickle=False);srcbind[cat]=dict(path=str(path),sha256=f['sha256'])
  masks=raw.region_masks(src['gt'],src['mask'],src['input'],regions['boundary_threshold_pixels'],regions['large_displacement_threshold_pixels'])
  states={'original':{},'same_bias_corrected':{}};rowbind={}
  for action in planned['unique_actions']:
   row=rows[tid,action];assert row['sequence']==rec['sequence'] and row['role']=='new_identity_survival'
   path=red.prediction(run,row)
   with np.load(path,allow_pickle=False) as z:p=z['head3'].copy()
   original=raw.measure(p,src['gt'],masks);assert original==bc.measure64(p.astype(np.float64),src['gt'],masks);zero+=len(original)
   corrected=p.astype(np.float64)-b[:,None,None];changed=bc.measure64(corrected,src['gt'],masks)
   for policy,a in planned['policy_actions'].items():
    if a==action:states['original'][policy]=original;states['same_bias_corrected'][policy]=changed
   rowbind[action]=dict(path=str(path),sha256=u.sha(path),head3_sha256=row['prediction_sha256'],corrected_payload_sha256=hashlib.sha256(np.ascontiguousarray(corrected).tobytes()).hexdigest())
  assert all(set(v)==set(POLICIES) for v in states.values())
  quality.append(dict(target_id=tid,sequence=rec['sequence'],role=rec['role'],policy_actions=planned['policy_actions'],**states));bindings[tid]=dict(source=srcbind,predictions=rowbind)
 u.write(out/'QUALITY_V1.json',dict(schema='flow_r38_four_policy_raw_quality_v1',frames=quality,all16regions=True,zero_bias_exact_region_comparisons=zero,bias_model_sha256=u.sha(bpath),region_source_sha256=u.sha(regionpath)))
 u.write(out/'BINDINGS_V1.json',dict(source_freeze_sha256=u.sha(HERE/'SOURCE_FREEZE_V1.json'),collector_result_sha256=u.sha(run/'results.json'),action_plan_sha256=u.sha(R38/'ACTION_PLAN_V1.json'),records=bindings))
 results={}
 for condition in ('original','same_bias_corrected'):
  frames=[dict(target_id=f['target_id'],sequence=f['sequence'],role=f['role'],states=f[condition]) for f in quality];seqs=list(dict.fromkeys(f['sequence'] for f in frames));summary={}
  for policy in POLICIES:
   summary[policy]={scope:{r:ev.mc.aggregate([f['states'][policy][r] for f in frames if scope=='overall' or f['sequence']==scope]) for r in ev.REGIONS} for scope in ['overall']+seqs}
  comparisons={c+'__vs_'+r:ev.comparison(frames,summary,c,r,seqs) for c in POLICIES for r in POLICIES if c!=r}
  leave={}
  for c in POLICIES:
   for r in POLICIES:
    if c==r:continue
    key=c+'__vs_'+r;leave[key]={}
    for drop in seqs:
     selected=[f for f in frames if f['sequence']!=drop];leave[key][drop]={}
     for reg in ev.REGIONS:
      cm=ev.mc.aggregate([f['states'][c][reg] for f in selected]);rm=ev.mc.aggregate([f['states'][r][reg] for f in selected])
      leave[key][drop][reg]={metric:dict(candidate=cm[metric],reference=rm[metric],delta=ev.mc.difference(cm[metric],rm[metric]),relative=ev.mc.ratio(cm[metric],rm[metric])) for metric in ('pixel_weighted_AEE','frame_mean_AEE')}
  actions={policy:dict(collections.Counter(f['policy_actions'][policy] for f in quality)) for policy in POLICIES}
  u.write(out/(condition+'_EVALUATION_V1.json'),dict(schema='flow_r38_four_policy_evaluation_v1',condition=condition,summaries=summary,comparisons=comparisons,leave_one_sequence=leave,action_histograms=actions,frames=frames,
   no_oracle_or_regret=True,new_model_fits=0,primary_precision='frozen_FP32',bias_model_sha256=u.sha(bpath),population='20 record-unexposed identities, five previously used repository-valid sequences; finite exposure scan, not official test',
   postprocess_cost=dict(enabled=condition=='same_bias_corrected',adds_per_identity=614400 if condition=='same_bias_corrected' else 0,cycles_unmeasured=True,not_in_absolute_hardware_service=True)))
  results[condition]={name:dict(AEE=v['overall']['all_valid']['pixel_weighted_AEE'],frame_mean_AEE=v['overall']['all_valid']['frame_mean_AEE'],B=v['overall']['B']['pixel_weighted_AEE'],L=v['overall']['L']['pixel_weighted_AEE'],
   relative_Q4=ev.mc.ratio(v['overall']['all_valid']['pixel_weighted_AEE'],summary['Q4']['overall']['all_valid']['pixel_weighted_AEE']),relative_static360=ev.mc.ratio(v['overall']['all_valid']['pixel_weighted_AEE'],summary['static360']['overall']['all_valid']['pixel_weighted_AEE']),tails=0 if name=='Q4' else len(comparisons[name+'__vs_Q4']['high_support_tail_review'])) for name,v in summary.items()}
 u.write(out/'SUMMARY_V1.json',dict(status='COMPLETE',results=results,identities=20,sequences=5,formal_forwards=71,engineering_forwards=2,GPU_used=False,CPU_seconds=time.monotonic()-start,source_freeze_sha256=u.sha(HERE/'SOURCE_FREEZE_V1.json'),collector_result_sha256=u.sha(run/'results.json'),new_model_fits=0,no_oracle_or_regret=True,zero_bias_exact_region_comparisons=zero))
 print('R38_QUALITY_COMPLETE',round(time.monotonic()-start,3),flush=True)
if __name__=='__main__':main()
