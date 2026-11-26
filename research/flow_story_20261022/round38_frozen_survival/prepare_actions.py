"""Only20 raw event tensors; freeze FP32 policy choices before new GT access."""
from pathlib import Path
import sys,os,json,hashlib,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def write(p,v):p.write_text(json.dumps(v,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v),allow_nan=False)+'\n')
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):assert os.environ.get(k)=='2'
 freeze=read(HERE/'ACTION_SOURCE_FREEZE_V1.json')
 for name,h in freeze['files'].items():assert sha(HERE/name)==h
 assert not (HERE/'ACTION_PLAN_V1.json').exists()
 p=load('r38_p',HERE/'model_sources/predictors_frozen.py');num=load('r38_num',HERE/'model_sources/numeric_frozen.py')
 fold=read(HERE/'model_sources/FOLDED_MODEL_V1.json');ordinary=read(HERE/'model_sources/ORDINARY_MODEL_128_V1.json')
 members=read(HERE/'data/PROPOSED_SOURCE_MEMBERSHIP_V1.json')['records'];sources=read(HERE/'data/LOCAL_INPUT_RECEIPT_V1.json')['records'];assert [r['target_id'] for r in members]==[r['target_id'] for r in sources]
 ids=[r['target_id'] for r in members];assert len(ids)==20 and not set(ids)&set(fold['fit_ids']) and fold['fit_ids']==ordinary['fit_ids']
 x64=[];x32=[]
 for rec in sources:
  assert sha(rec['path'])==rec['sha256'];z=np.load(rec['path'],allow_pickle=False);assert z.shape==(10,480,640) and z.dtype==np.float32
  x64.append(p.voxel_features(z));x32.append(num.voxel_features(z)[:128])
 x64=np.stack(x64);x32=np.stack(x32)
 scores={};scores['folded128_fp64']=p.ridge_predict(fold['ridge'],x64);scores['folded128_fp32'],_=num.affine_fp32(fold['ridge'],x32)
 family=ordinary['family'];selection=ordinary['selection'];auxridge=family['auxiliary'];resridge=family['residual'][str(selection['alpha'])]
 aux64=p.ridge_predict(auxridge,x64);aug64=p.augmented_features(x64,aux64,1);scores['ordinary128_fp64']=p.signed_scores(aux64,1)+selection['shrink']*p.ridge_predict(resridge,aug64)
 aux32,_=num.affine_fp32(auxridge,x32);aug32=np.concatenate((x32,aux32,num.products(aux32,1).reshape(20,-1)),axis=1);rr,_=num.affine_fp32(resridge,aug32);scores['ordinary128_fp32']=num.signed(aux32,1)+np.float32(selection['shrink'])*rr
 choices={k:num.choose(v) for k,v in scores.items()};records=[]
 for i,rec in enumerate(members):
  actions=dict(Q4='background_Q4',static360='static360',folded128_fp32_fixed6='upgrade_'+str(choices['folded128_fp32'][i]),ordinary128_fp32_fixed6='upgrade_'+str(choices['ordinary128_fp32'][i]))
  schedule=list(dict.fromkeys(actions.values()));records.append(dict(target_id=rec['target_id'],sequence=rec['sequence'],policy_actions=actions,unique_actions=schedule))
 np.savez_compressed(HERE/'FEATURES_AND_SCORES_V1.npz',ids=np.asarray(ids),x128_fp64=x64,x128_fp32=x32,**scores,**{'choice__'+k:v for k,v in choices.items()})
 out=dict(schema='flow_r38_event_only_action_plan_v1',source_freeze_sha256=sha(HERE/'ACTION_SOURCE_FREEZE_V1.json'),membership_sha256=sha(HERE/'data/PROPOSED_SOURCE_MEMBERSHIP_V1.json'),
  input_receipt_sha256=sha(HERE/'data/LOCAL_INPUT_RECEIPT_V1.json'),features_scores_sha256=sha(HERE/'FEATURES_AND_SCORES_V1.npz'),records=records,
  expected_formal_forwards=sum(len(r['unique_actions']) for r in records),engineering_forwards=2,max_forward_attempts=84,wall_cap_seconds=900,
  primary='FP32 features and FP32 readouts chosen before any new quality',GT_arrays_decoded=False,new_fits=0,GPU_calls=0,
  bridges={name:dict(fixed6_flips=int(np.sum(choices[name+'_fp32']!=choices[name+'_fp64'])),max_abs_score_error=float(np.max(np.abs(scores[name+'_fp32']-scores[name+'_fp64'])))) for name in ('folded128','ordinary128')})
 assert out['expected_formal_forwards']<=80;write(HERE/'ACTION_PLAN_V1.json',out);write(HERE/'ACTION_PLAN_FREEZE_V1.json',dict(action_plan_sha256=sha(HERE/'ACTION_PLAN_V1.json'),GPU_started=False,new_GT_quality_read=False))
 print(json.dumps(dict(formal=out['expected_formal_forwards'],bridges=out['bridges'])))
if __name__=='__main__':main()
