"""Rebuild native events, bounded class counts, accesses and cycle obligations."""
from pathlib import Path
import json
from collections import Counter
import numpy as np
from prepare import readhex,native
H=Path(__file__).resolve().parent;S=H.parent/'pair_sparse'
cases=json.loads((H/'fixtures.json').read_text());rows=json.loads((H/'results.json').read_text())
checks=0;profiles={}
def eq(a,b):
 global checks
 checks+=1
 assert a==b,(a,b)
def profile(d,params=None):
 p=d if params is None else params
 source=readhex(d/'source.hex').reshape(96,4,4);origin=readhex(d/'origin.hex');q=readhex(p/'q1.hex').reshape(864,8)
 v=readhex(p/'q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
 events,z=native(source,q,origin)
 # Independent Python-int final dot, rather than NumPy @ used by preparation.
 raw=np.array([[sum(int(z[pos,r])*int(v[n,r]) for r in range(8)) for n in range(g*8,g*8+8)] for g in range(12) for pos in range(40)],np.int64)
 eq(bool(np.array_equal(raw.ravel(),readhex(d/'gold.hex'))),True)
 cls=(readhex(p/'class.hex')[:,None]>>(6*np.arange(4)))&63;reps=readhex(p/'representative.hex').reshape(32,8);ng=int(readhex(p/'ngroups.hex')[0])
 eq(bool(np.array_equal(np.any(q!=0,axis=1),readhex(p/'k_live.hex'))),True)
 eq(bool(np.all((cls<=32)|(cls==63))),True)
 counts=np.zeros((4,32,4,10),np.int64);direct=np.zeros_like(q);max_class=0
 for g in range(4):
  raw_keys=[tuple(map(int,vv)) for vv in q[:,2*g:2*g+2]]
  raw_frequency=Counter(raw_keys)
  for kk,key in enumerate(raw_keys):
   if key!=(0,0) and raw_frequency[key]>255:eq(int(cls[kk,g]),63)
   if 1<=cls[kk,g]<=32:eq(raw_frequency[key]<=255,True)
  for code in range(1,33):
   members=cls[:,g]==code;n=int(members.sum());max_class=max(max_class,n);eq(n==0 or 2<=n<=255,True)
   if n:counts[g,code-1]=events[members].sum(axis=0).reshape(4,10)
  for k in range(864):
   code=int(cls[k,g]);pair=q[k,2*g:2*g+2]
   if code==0:eq(bool(np.all(pair==0)),True)
   elif code==63:direct[k,2*g:2*g+2]=pair
   else:eq(bool(np.array_equal(pair,reps[code-1,2*g:2*g+2])),True)
 reconstructed=events.T@direct
 for g in range(4):reconstructed[:,2*g:2*g+2]+=counts[g].reshape(32,40).T@reps[:,2*g:2*g+2]
 eq(bool(np.array_equal(reconstructed,z)),True);eq(bool(np.all(counts<=255)),True)
 livek=np.any(q!=0,axis=1);active=livek&np.any(events,axis=1);K=int(livek.sum());A=int(active.sum())
 ev=events.reshape(864,4,10);lo=ev[:,[0,2]].reshape(864,20).astype(bool);hi=ev[:,[1,3]].reshape(864,20).astype(bool)
 U=int((lo|hi)[livek].sum());vlive=np.any(v.reshape(12,8,8)!=0,axis=1)
 M=int(((z!=0)*vlive.sum(axis=0)).sum());V=int((vlive&np.any(z!=0,axis=0)).sum())
 valid=int(sum(0<=origin[0]+y<240 and 0<=origin[1]+x<320 for y in range(4) for x in range(4)))
 common=dict(outputs=3840,source_words=valid*96,local_source_reads=K,second_weight_words=V,z_scalar_reads=M,mac_issues=M,psum_reads=480,psum_writes=480)
 base=dict(common,weight_words=A+V,first_issues=U,dual_updates=int((lo&hi)[livek].sum()),z_vector_reads=U+40,z_writes=U+20,metadata_reads=0,count_checks=0,count_bank_reads=0,count_bank_writes=0,aux_reads=0,aux_writes=0,aux_issues=0,aux_weight_words=0,aux_events=0,base_cycles=5437+K+2*A+3*U+M)
 grouped=(cls>0)&(cls<=32);need_count=active&np.any(grouped,axis=1);need_direct=active&np.any(cls==63,axis=1)
 direct_A=int(need_direct.sum());direct_U=int((lo|hi)[need_direct].sum())
 touched=np.zeros((4,32,5),bool);updates=0;read_vectors=0;read_banks=0;write_banks=0
 for k in range(864):
  if not need_count[k]:continue
  for b in range(5):
   if not np.any(ev[k,:,b*2:b*2+2]):continue
   updates+=1;reads=0
   for g in range(4):
    code=int(cls[k,g])
    if 1<=code<=32:
     reads+=2*int(touched[g,code-1,b]);write_banks+=2;touched[g,code-1,b]=True
   read_vectors+=int(reads>0);read_banks+=reads
 liveblocks=int(np.any(touched,axis=0).sum());retire_reads=int(touched.sum()*2);slots=int(np.any(touched,axis=(0,2)).sum())
 retire=0;packed=0;fallback=0
 for slot in range(32):
  for t in range(10):
   for pp in range(2):
    low=counts[:,slot,pp*2,t];high=counts[:,slot,pp*2+1,t]
    active_positions=int(np.any(low))+int(np.any(high))
    if not active_positions:continue
    if np.all(high<=127):retire+=1;packed+=1
    else:retire+=active_positions;fallback+=active_positions
 extra=A+2*updates+(ng+1+slots+2*liveblocks+3*retire if ng else 0)
 candidate=dict(common,weight_words=direct_A+slots+V,first_issues=direct_U+retire,dual_updates=int((lo&hi)[need_direct].sum()),z_vector_reads=direct_U+retire+40,z_writes=direct_U+retire+20,metadata_reads=A,count_checks=updates,count_bank_reads=read_banks+retire_reads,count_bank_writes=write_banks,aux_reads=read_vectors+liveblocks,aux_writes=updates,aux_issues=updates,aux_weight_words=slots,aux_events=retire,base_cycles=5437+K+2*direct_A+3*direct_U+M+extra)
 return {14:base,20:candidate,'detail':dict(max_class=max_class,max_observed_count=int(counts.max()),packed_retire=packed,scalar_retire=fallback,first_touch_read_vectors=read_vectors,retire_live_blocks=liveblocks,physical_psum_bank_reads=candidate['count_bank_reads']+3840,physical_psum_bank_writes=candidate['count_bank_writes']+3840)}
def main():
 for c in cases:profiles[c['name']]=profile(H/'fixtures'/c['name'])
 old={(r['fixture'],r['mode'],r['stall'],r['command']):r for r in json.loads((S/'results.json').read_text())}
 matched14=matched19=0
 for r in rows:
  if r['mode']==19:
   expected=old[r['fixture'],19,r['stall'],r['command']]
   for k,v in expected.items():eq(r[k],v)
   matched19+=1;continue
  pred=profiles[r['fixture']][r['mode']]
  for key,value in pred.items():
   if key!='base_cycles':eq(r[key],value)
  eq(r['cycles'],pred['base_cycles']+sum(r[k] for k in ['source_stalls','weight_stalls','output_stalls']))
  eq(sum(r['state_cycles']),r['cycles']);eq(r['configuration_cycles'],0 if r['command'] else 3361 if r['mode']==14 else 4258)
  if r['mode']==14 and (r['fixture'],14,r['stall'],r['command']) in old:
   for k,v in old[r['fixture'],14,r['stall'],r['command']].items():eq(r[k],v)
   matched14+=1
 sv=(H/'decomp_core.sv').read_text();eq('count_mem' in sv,False);eq(sv.count('p_mem[i][p_addr[i]]'),2)
 result=dict(passed=True,checks=checks,fixtures=len(cases),rtl_commands=len(rows),rtl_raw_values=sum(r['outputs'] for r in rows),old_direct_all_fields_reproduced=matched14,old19_reference_all_fields_reproduced=matched19,independent_gold_values=len(cases)*3840,independent_class_bound_and_state_port_accounting=True,alias_assertions='active Verilator checks: single bank access, count address<160, count8 no overflow, no count accesses after retirement, all480 outputs overwrite before drain',per_fixture={name:p['detail'] for name,p in profiles.items()})
 (H/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
 (H/'profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='per_fixture'},indent=2))

if __name__=="__main__":main()
