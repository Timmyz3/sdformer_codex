"""Independent raw-event and FP32 head replay; never opens GT/quality arrays."""
from pathlib import Path
import hashlib, importlib.util, json, os
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
B=ROOT/'research/flow_story_20261022/round38_frozen_survival'
OLD=ROOT/'research/flow_story_20261021/round34_deployability'
OUT=Path(__file__).with_name('R38_ACTION_AND_COLLECTOR_PRELAUNCH_V1.json')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b):assert np.array_equal(a,b), (np.shape(a),np.shape(b))
def load_module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def affine(m,x):
 r={k:np.asarray(m[k],np.float32) for k in ['mean','scale','coef','intercept']}
 z=(x-r['mean'])/r['scale'];out=np.zeros((len(x),len(r['intercept'])),np.float32)
 for k in range(x.shape[1]):out=np.add(out,np.multiply(z[:,k,None],r['coef'][k]))
 return np.add(out,r['intercept'])

def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists()
 n=load_module(OLD/'independent_review/audit_numeric_v1.py','_independent_r34_numeric')
 for name in ['ACTION_SOURCE_FREEZE_V1.json','SOURCE_FREEZE_V1.json']:
  for file,h in read(B/name)['files'].items():assert sha(B/file)==h
 plan=read(B/'ACTION_PLAN_V1.json');freeze=read(B/'ACTION_PLAN_FREEZE_V1.json')
 assert freeze['action_plan_sha256']==sha(B/'ACTION_PLAN_V1.json') and not freeze['GPU_started'] and not freeze['new_GT_quality_read']
 for key,name in [('source_freeze_sha256','ACTION_SOURCE_FREEZE_V1.json'),('membership_sha256','data/PROPOSED_SOURCE_MEMBERSHIP_V1.json'),('input_receipt_sha256','data/LOCAL_INPUT_RECEIPT_V1.json'),('features_scores_sha256','FEATURES_AND_SCORES_V1.npz')]:assert plan[key]==sha(B/name)
 originals={'FOLDED_MODEL_V1.json':OLD/'numeric/folded_constant_v/actual_v1/FOLDED_MODEL_V1.json','ORDINARY_MODEL_128_V1.json':OLD/'ordinary/actual_v1/MODEL_128_V1.json','numeric_frozen.py':OLD/'numeric/core.py','predictors_frozen.py':OLD.parent/'round33_predictability/algorithm/predictors.py'}
 for f,p in originals.items():assert sha(B/'model_sources'/f)==sha(p)
 f=read(B/'model_sources/FOLDED_MODEL_V1.json');o=read(B/'model_sources/ORDINARY_MODEL_128_V1.json')
 assert f['fit_ids']==o['fit_ids'] and o['selection']['alpha']==1.0 and o['selection']['shrink']==0.25
 rows=read(B/'data/PROPOSED_SOURCE_MEMBERSHIP_V1.json')['records'];rr=read(B/'data/LOCAL_INPUT_RECEIPT_V1.json')['records'];ids=[r['target_id'] for r in rows]
 assert len(ids)==len(set(ids))==20 and ids==[r['target_id'] for r in rr]==[r['target_id'] for r in plan['records']] and not set(ids)&set(f['fit_ids'])
 with np.load(B/'FEATURES_AND_SCORES_V1.npz',allow_pickle=False) as a: saved={k:a[k].copy() for k in a.files}
 eq(saved['ids'],ids);features=[]
 for row,r in zip(rows,rr):
  spec=next(s for s in row['source_files'] if s['category']=='input');assert r['relative_path']==spec['relative_path'] and r['sha256']==spec['sha256']==sha(r['path'])
  raw=np.load(r['path'],allow_pickle=False);assert raw.shape==(10,480,640) and raw.dtype==np.float32 and np.isfinite(raw).all()
  features.append(n.features(raw)[:128])
 x=np.stack(features);eq(x,saved['x128_fp32'])
 fs=affine(f['ridge'],x)
 aux=affine(o['family']['auxiliary'],x)
 products=aux[:,:2,None]*aux[:,2:].reshape(20,6,2).transpose(0,2,1)
 signed=np.add(products[:,0],products[:,1])
 augmented=np.concatenate([x,aux,products.transpose(0,2,1).reshape(20,12)],axis=1)
 resid=affine(o['family']['residual'][str(o['selection']['alpha'])],augmented)
 ordinary=np.add(signed,np.multiply(np.float32(o['selection']['shrink']),resid))
 scores={'folded128':fs,'ordinary128':ordinary};actions={}
 for name,s in scores.items():
  assert s.dtype==np.float32;eq(s,saved[name+'_fp32']);choice=s.argmin(axis=1);eq(choice,saved['choice__'+name+'_fp32']);actions[name]=choice
  flips=int(np.sum(choice!=saved['choice__'+name+'_fp64']));assert plan['bridges'][name]['fixed6_flips']==flips
  assert plan['bridges'][name]['max_abs_score_error']==np.abs(s.astype(float)-saved[name+'_fp64']).max()
 for i,row in enumerate(plan['records']):
  expected={'Q4':'background_Q4','static360':'static360','folded128_fp32_fixed6':'upgrade_'+str(actions['folded128'][i]),'ordinary128_fp32_fixed6':'upgrade_'+str(actions['ordinary128'][i])}
  assert row['policy_actions']==expected and row['unique_actions']==list(dict.fromkeys(expected.values()))
 count=sum(len(r['unique_actions']) for r in plan['records']);assert count==plan['expected_formal_forwards']==71 and plan['engineering_forwards']==2 and count+2<=plan['max_forward_attempts']==84 and plan['wall_cap_seconds']==900
 engineering=read(B/'ENGINEERING_REFERENCES_V1.json')
 for p,h in engineering['source_bindings'].items():assert sha(ROOT/p)==h
 old35=ROOT/'research/flow_story_20261021/round35_static_neighbor_probe'
 q4=read(old35/'Q4_PARITY_REFERENCES_V1.json')['roles']['fit'];static=next(r for r in read(old35/'actual/static360_v1/results.json')['evaluations']['static360']['frames'] if r['target_id']==engineering['target_id'])
 assert engineering['target_id']==q4['target_id'];assert engineering['heads']['background_Q4']==q4['heads'] and engineering['heads']['static360']==static['heads']
 review=dict(schema='flow_r38_action_collector_independent_review_v1',status='PASS_PRELAUNCH',auditor_sha256=sha(__file__),independent_numeric_helper_sha256=sha(n.__file__),collector_sha256=sha(B/'collect.py'),execution_source_freeze_sha256=sha(B/'SOURCE_FREEZE_V1.json'),action_plan_sha256=sha(B/'ACTION_PLAN_V1.json'),metadata_audit_sha256=sha(OUT.with_name('R38_ENTRY_METADATA_INDEPENDENT_V1.json')),
  actual_checks=dict(raw_input_arrays=20,FP32_features=2560,FP32_score_values=240,FP32_head_choices=40,all_FP32_features_and_scores_exact=True,models_original_bytes_unchanged=True,GT_arrays_opened=0,quality_arrays_opened=0,new_fits=0,GPU_calls=0,formal_unique_forwards=71,old_identity_engineering_replays=2,total_expected_forwards=73,attempt_cap=84,wall_cap_seconds=900),
  observed_fp64_bridge=plan['bridges'],actions={k:v.tolist() for k,v in actions.items()},
  collector_source_review=['R35 pinned prepare transitively binds checkpoint, core, helper, canonical and full accepted native verification; new20 canonical/source hashes are separately admitted.','Single exact FP32 choice per frozen head; identical actions share one counterfactual output, with policy_users recorded.','Model eval/reset before every forward, 105 PSN once, 28 affine and 78 BN via frozen validators, per-action actual work digest constant and final state/BN unchanged.','Old first-fit Q4 and static360 four-head replay hashes independently bound to original executed references; no new dense parity claimed.','No GT array decoding or quality selection in collector; source files GT/mask only hashed.','External timeout must remain in executed launcher receipt; this source has internal900s execution alarm after prepare.'],
  scope='Source/actions PASS for bounded frozen-policy survival. Not execution PASS, new-sequence/official-test validation, full-network cost closure, or proof of response-supervision novelty.')
 OUT.write_text(json.dumps(review,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':review['status'],'counts':review['actual_checks'],'audit_sha256':sha(OUT)}))
if __name__=='__main__':main()
