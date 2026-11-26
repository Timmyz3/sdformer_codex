#!/usr/bin/env python3
"""Independent hardware aggregation: read completed frozen records, no policy run.

Raw input reads only check the mass==0 guard, using all-zero sample membership;
no feature/readout/GT/interval fitting or quality calculations are performed.
"""
from pathlib import Path
import collections,hashlib,json
import numpy as np
P=Path(__file__).resolve().parent;ALG=P.parent/'algorithm';A=ALG/'actual_fit64_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
 assert not p.exists(),p
 p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
summary=read(A/'SUMMARY_V1.json');manifest=read(ALG/'INPUT_MANIFEST_V1.json');cost=read(P/'SOURCE_COST_BOUNDS_V1.json')
assert summary['status']=='PASS_NUMERICAL_SCREEN_COMPLETE'
assert summary['input_manifest_sha256']==sha(ALG/'INPUT_MANIFEST_V1.json')
rows=[];heads={'folded128':8599,'ordinary128':26646}
assert len(manifest['records'])==64
for rec in manifest['records']:
 path=A/'identities'/(rec['target_id']+'.json');d=read(path)
 assert d['target_id']==rec['target_id'] and d['sequence']==rec['sequence']
 inp=Path(rec['path']);assert sha(inp)==rec['sha256']==d['input_sha256']
 x=np.load(inp,allow_pickle=False);assert x.dtype==np.float32 and x.shape==(10,480,640) and np.isfinite(x).all()
 # Nonnegative finite FP32 masses are exactly zero iff every input is signed zero,
 # under the already-required gradual-underflow floating environment.
 zeros=sum(not np.any(x[:,r*120:(r+1)*120,c*160:(c+1)*160]!=np.float32(0)) for r in range(4) for c in range(4))
 assert zeros==d['zero_mass_cells']
 stages=d['stages'];assert [s['stage'] for s in stages]==list(range(len(stages)))
 for name,h in heads.items():
  q=d['decisions'][name];k=q['stage'];assert 0<=k<=16
  # Independently validate the claimed earliest certificate from saved score bounds.
  earliest=None;calls=0
  for st in stages:
   if name not in st['heads']:continue
   calls+=1;b=st['heads'][name];lo=np.array(b['lower'],dtype=np.float32);hi=np.array(b['upper'],dtype=np.float32)
   cert=next((a for a in range(6) if all(hi[a]<lo[j] if j<a else hi[a]<=lo[j] for j in range(6) if j!=a)),None)
   assert cert==b['certified_action']
   if cert is not None:
    assert earliest is None
    earliest=(st['stage'],cert)
  if q['certified']:
   assert earliest==(k,q['action']) and q['action']==q['reference_action'] and calls==k+1
  else:assert earliest is None
  # Derive the prescribed relaxed service anew rather than reading cached stage cost.
  first=55877120-32*zeros;body=4204844*k;refetch=144320*((k+3)//4)
  old=113938624-32*zeros+h;new=first+body+refetch
  if not q['certified']:new=max(old,new) # conservative screen, not actual fallback fee
  assert old==q['original_prefix_ticks'] and new==q['optimistic_prefix_ticks']
  rows.append({'target_id':rec['target_id'],'sequence':rec['sequence'],'policy':name,'zero_mass_cells_independently_raw_checked':int(zeros),'stage':k,'actual_interval_head_calls':calls,'certified':q['certified'],'original_prefix_ticks':old,'free_certificate_prefix_ticks':new,'saved_ticks':old-new,'identity_record_sha256':sha(path)})
scopes={}
for name,h in heads.items():
 scopes[name]={}
 for sequence in ['overall']+list(dict.fromkeys(r['sequence'] for r in manifest['records'])):
  rr=[r for r in rows if r['policy']==name and (sequence=='overall' or r['sequence']==sequence)]
  old=sum(r['original_prefix_ticks'] for r in rr);new=sum(r['free_certificate_prefix_ticks'] for r in rr)
  actual={'identities':len(rr),'stage_histogram':dict(sorted(collections.Counter(str(r['stage']) for r in rr).items(),key=lambda q:int(q[0]))),'zero_mass_cell_histogram':dict(sorted(collections.Counter(str(r['zero_mass_cells_independently_raw_checked']) for r in rr).items(),key=lambda q:int(q[0]))),'total_original_prefix_ticks':old,'total_optimistic_prefix_ticks':new,'original_mean_prefix_ticks':old/len(rr),'optimistic_mean_prefix_ticks':new/len(rr),'optimistic_saved_fraction':(old-new)/old,'minimum_identity_saved_fraction':min(r['saved_ticks']/r['original_prefix_ticks'] for r in rr),'total_actual_interval_head_calls':sum(r['actual_interval_head_calls'] for r in rr),'passes_10pct_necessary_gate_exact_integer':10*(old-new)>=old}
  parent=summary['results'][name][sequence]
  assert abs(actual['optimistic_saved_fraction']-parent['optimistic_mean_saved_fraction'])<1e-14
  assert actual['optimistic_mean_prefix_ticks']==parent['optimistic_mean_prefix_ticks']
  assert actual['original_mean_prefix_ticks']==parent['original_mean_prefix_ticks']
  assert actual['stage_histogram']==parent['stage_histogram']
  scopes[name][sequence]=actual
# Per-record zero correction is equal on both sides, but affects ratio denominator.
sensitivity={}
for name,h in heads.items():
 rr=[r for r in rows if r['policy']==name];sensitivity[name]={}
 for z in (0,16):
  old=64*(113938624-32*z+h)
  new=sum(max(113938624-32*z+h,55877120-32*z+4204844*r['stage']+144320*((r['stage']+3)//4)) if not r['certified'] else 55877120-32*z+4204844*r['stage']+144320*((r['stage']+3)//4) for r in rr)
  sensitivity[name][f'all_identities_have_{z}_zero_cells']={'old_ticks':old,'new_ticks':new,'saved_fraction':(old-new)/old,'passes_10pct_necessary_gate_exact_integer':10*(old-new)>=old}
out={'schema':'r39_independent_fit64_hardware_screen_v1','status':'PASS_RECORDS_AND_ARITHMETIC','independent_of_algorithm_runner':True,'algorithm_summary_sha256':sha(A/'SUMMARY_V1.json'),'source_cost_freeze_sha256':sha(P/'SOURCE_COST_FREEZE_V1.json'),'audit_source_sha256':sha(Path(__file__)),'raw_inputs_read_only_for_zero_guard':64,'GT_read':False,'new_selector_executions':0,'records':rows,'scopes':scopes,'zero_sensitivity':sensitivity,'permitted_conclusion':'fixed source-like two-pass rowband relaxation only; bounds, metadata, repeated heads/control are all free. No actual new hardware cycles, exact NumPy streaming proof, new quality, RTL or whole-network inference.'}
save(P/'FIT64_HARDWARE_SCREEN_V1.json',out)
print(json.dumps({'status':out['status'],'sha256':sha(P/'FIT64_HARDWARE_SCREEN_V1.json'),'overall':{k:v['overall'] for k,v in scopes.items()}},indent=2))
