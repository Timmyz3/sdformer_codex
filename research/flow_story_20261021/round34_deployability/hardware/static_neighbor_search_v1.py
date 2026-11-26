#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
P=Path(__file__).resolve().parent;O=P/'actual_v1';R33=P.parent.parent/'round33_predictability'
def read(p):return json.loads(p.read_text())
def service(n):
 p=O/f'STATIC_{n}_V1.json'
 if not p.exists():
  r=subprocess.run([str(P/'static_neighbor_supply'),str(n)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True);p.write_text(r.stdout)
 d=read(p);q8=30900+n*810;q4=768000-q8;assert d['result']['q8_rows']==q8 and d['result']['q4_rows']==q4;assert d['result']['arithmetic_accepted_beats']==(q8*8+q4*4)*7*6
 return d['result']['affine_plus_BN_completion_ticks']*4
assert service(151)==4*229694508
head=read(O/'HEAD_PREFIX_RESULTS_V1.json')['models'];f=np.load(O/'STREAM_FEATURES_V1.npz');scores=np.load(O/'STREAM_HEAD_SCORES_V1.npz');role=f['roles']=='confirm';ids=f['ids'][role];x=f['x160'][role]
dynamic=read(R33/'hardware/results/SUMMARY_V1.json')['candidates'];cost=np.asarray([r['dynamic_affine_BN_ticks']*4 for r in dynamic],dtype=np.float64)
rows=[]
for key in ['X128_folded_constant_v','X128_signed_g1','X160_signed_g1','X128_calibrated_residual_g1','X160_calibrated_residual_g1']:
 s=scores[key][role];width=128 if key.startswith('X128') else 160
 if width==160:features=np.asarray([read(O/(str(i)+'.json'))['feature_ready_tick'] for i in ids],dtype=np.float64);scope='actual full stream model per identity'
 else:
  zero_cells=np.sum((x[:,:128:8]==0)&(x[:,1:128:8]==0),axis=1);features=113938624-32*zero_cells;scope='X128 executed full first identity; per-identity exact fixed work plus observed zero-mass guard count, validated affected cases separately'
 for allow in [False,True]:
  a=np.argmin(s,axis=1);a=np.where(np.min(s,axis=1)>=0,6,a) if allow else a
  operator_mean=float(np.mean(cost[a]));prefix_mean=float(np.mean(features))+head[key]['action_ready_tick'];budget=operator_mean+prefix_mean
  lo,hi=151,910
  while lo<hi:
   mid=(lo+hi+1)//2
   if service(mid)<=budget:lo=mid
   else:hi=mid-1
  lower=service(lo);upper=service(lo+1) if lo<910 else None
  rows.append(dict(head=key,policy='allow_Q4' if allow else 'fixed6',identity_count=len(ids),upgrades=int(np.count_nonzero(a!=6)),feature_prefix_scope=scope,operator_four_sum_mean_ticks=operator_mean,feature_mean_model_ticks=float(np.mean(features)),head_action_model_ticks=head[key]['action_ready_tick'],added_serial_predictor_prefix_mean_ticks=prefix_mean,serial_predictor_plus_four_isolated_conv_budget=budget,static_tiles_within_budget=lo,static_four_operator_ticks=lower,static_next_tiles=lo+1 if lo<910 else None,static_next_four_operator_ticks=upper,remaining_budget_ticks=budget-lower,mask_order='fit static group2 first (302..452), then lowest remaining tile IDs',static_AEE_known=False,whole_network_latency=False,shared_stem_ready_unknown=True))
out=dict(schema='r34_static_neighbor_finite_supply_search_v1',status='SERVICES_EXECUTED_QUALITY_UNMEASURED',rows=rows,ordinary_resources_equal=True,model_arithmetic_not_RTL=True,new_static_AEE_fabricated=False)
(O/'STATIC_NEIGHBOR_SEARCH_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
