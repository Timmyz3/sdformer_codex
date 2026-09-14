"""Native whole-frame dual-P events, independent of any RTL work counter."""
from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;R=H.parent/'consumer_packed'
w=np.load(H/'first_source_words.npy');f=np.load(H/'factors.npz');kl=np.any(f['q1']!=0,axis=0).reshape(96,9)
w=np.pad(w,((0,0),(1,1),(1,1)));lut=np.array([x.bit_count() for x in range(1024)],np.int16)
A=np.zeros((120,160),np.int64);D=A.copy();Q=A.copy()
for ky in range(3):
 for kx in range(3):
  pp=[]
  for py in range(2):
   for px in range(2):pp.append(w[:,ky+py:ky+py+240:2,kx+px:kx+px+320:2]*kl[:,ky*3+kx,None,None])
  A+=sum(lut[x].sum(axis=0) for x in pp)
  D+=(lut[pp[0]&pp[1]]+lut[pp[2]&pp[3]]).sum(axis=0)
  Q+=((pp[0]|pp[1]|pp[2]|pp[3])!=0).sum(axis=0)
checks=0
for file in ('results_64.json','results_full.json'):
 rows=json.loads((R/file).read_text())
 for r in rows:
  idx=slice(r['first_tile'],r['first_tile']+r['tiles']);a=int(A.ravel()[idx].sum());d=int(D.ravel()[idx].sum());q=int(Q.ravel()[idx].sum());ff=r['tiles'];up=a if r['mode']==14 else a-d
  expected=dict(core_first_issues=up,core_dual_updates=d if r['mode']==15 else 0,core_z_vector_reads=up+40*ff,core_z_writes=up+20*ff,core_local_source_reads=int(kl.sum())*ff)
  for k,v in expected.items():assert r[k]==v,(file,k,r[k],v);checks+=1
  assert r['core_weight_words']-r['core_second_weight_words']==q;checks+=1
 if file=='results_full.json':
  by={r['mode']:r for r in rows}
  for key in ('core_source_words','core_weight_words','core_second_weight_words','core_psum_reads','core_psum_writes','core_mac_issues','consumer_identity_words','consumer_conversion_issues','consumer_mul_issues','consumer_add_issues','consumer_round_issues'):
   assert by[14][key]==by[15][key];checks+=1
  assert by[14]['total_cycles']-by[15]['total_cycles']==3*int(D.sum());checks+=1
assert (R/'i24_consumer.sv').read_bytes()==(R.parent/'consumer_rtl/i24_consumer.sv').read_bytes()
o=dict(complete=True,RTL_rerun=False,GPU_rerun=False,full_first_scalar_updates=int(A.sum()),full_dual_overlaps=int(D.sum()),full_union_updates=int((A-D).sum()),first_Q1_weight_words=int(Q.sum()),expected_complete_cycle_saving=3*int(D.sum()),checks=checks,all_match=True,consumer_source_bytes_identical=True)
(H/'review_full_source_pairs.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
