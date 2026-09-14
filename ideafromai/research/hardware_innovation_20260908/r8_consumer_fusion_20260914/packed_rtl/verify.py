"""Independent integer/transaction checks; no RTL cycle prediction used as measured results."""
from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent
rows=json.loads((H/'results.json').read_text())
def words(p):return np.array([int(x,16) for x in p.read_text().split()],dtype=np.uint32)
def signed(a,bits):
 a=a.astype(np.int64)&((1<<bits)-1);return np.where(a&(1<<(bits-1)),a-(1<<bits),a)
info=[]
for name in sorted({r['fixture'] for r in rows}):
 p=H/'fixtures'/name
 q=signed(words(p/'q1.hex'),3).reshape(864,8).T
 v=signed(words(p/'q2.hex'),16).reshape(12,8,8).transpose(0,2,1).reshape(96,8)
 s=((words(p/'source.hex')[None,:]>>np.arange(10)[:,None])&1).reshape(10,96,4,4)
 oy,ox=signed(words(p/'origin.hex'),16)
 valid=(np.arange(4)[:,None]+oy>=0)&(np.arange(4)[:,None]+oy<240)&(np.arange(4)[None,:]+ox>=0)&(np.arange(4)[None,:]+ox<320)
 s=s*valid[None,None]
 x=np.stack([s[:,:,a//2:a//2+3,a%2:a%2+3].reshape(10,864) for a in range(4)])
 z=x@q.T;zflat=z.reshape(40,8);gold=zflat@v.T
 goldwords=gold.reshape(4,10,12,8).transpose(2,0,1,3).reshape(-1)
 assert np.array_equal(goldwords,signed(words(p/'gold.hex'),32))
 live=np.any(q!=0,axis=0);assert np.array_equal(live,words(p/'k_live.hex').astype(bool))
 xx=x[:,:,live];scalar=int(np.count_nonzero(xx));paired=int(np.count_nonzero(xx[0]|xx[1])+np.count_nonzero(xx[2]|xx[3]));dual=scalar-paired
 ranklive=np.any(zflat!=0,axis=0);vv=np.any(v.reshape(12,8,8)!=0,axis=1)
 second=int(np.count_nonzero(vv&ranklive));qreads=int(np.count_nonzero(np.any(x,axis=(0,1))&live));mac=sum(np.count_nonzero(zflat[:,vv[g]]) for g in range(12))
 # Independent packed representation recurrence and half-preservation.
 zw=np.zeros((20,8),np.int64)
 for k in range(864):
  if not live[k]:continue
  for pr in range(2):
   for t in range(10):
    lo,hi=int(x[2*pr,t,k]),int(x[2*pr+1,t,k])
    if not (lo or hi):continue
    row=pr*10+t;low=zw[row]&8191;high=(zw[row]>>13)&8191
    zw[row]=((low+lo*q[:,k])&8191)|(((high+hi*q[:,k])&8191)<<13)
 for pidx in range(4):
  rec=signed(zw[(pidx//2)*10:(pidx//2+1)*10]>>(13*(pidx%2)),13)
  assert np.array_equal(rec,z[pidx])
 rr=[r for r in rows if r['fixture']==name]
 for r in rr:
  first=scalar if r['mode']==14 else paired;st=r['state_cycles']
  assert r['cycles']==sum(st) and r['outputs']==3840
  assert r['first_issues']==first and r['dual_updates']==(0 if r['mode']==14 else dual)
  assert r['z_writes']==first+20 and r['z_vector_reads']==first+40
  assert r['source_words']==int(valid.sum())*96 and r['local_source_reads']==int(live.sum())
  assert r['second_weight_words']==second and r['weight_words']==second+qreads
  assert r['mac_issues']==mac==r['z_scalar_reads'];assert r['psum_reads']==r['psum_writes']==480
  assert r['configuration_cycles']==(3361 if r['command']==0 else 0)
  assert st[1]==20 and st[2]==96 and st[3]==1536+r['source_stalls'] and st[4]==864
  assert st[5]==int(live.sum()) and st[6]==qreads+r['weight_stalls']-(st[12]-96)
  assert st[7]==first+qreads and st[8]==st[9]==first and st[10]==864 and st[11]==40
  assert st[12]>=96 and st[13]==480 and st[14]==mac and st[15]==st[16]==480
  assert st[17]==480+r['output_stalls'] and st[18]==1
  if not r['stall']:
   common=20+96+1536+864+int(live.sum())+2*qreads+864+40+96+480+mac+480+480+480+1
   assert r['cycles']==common+3*first
 for m in [14,15]:
  for sf in [0,1]:
   a=[r for r in rr if r['mode']==m and r['stall']==sf]
   assert len(a)==2 and all(a[0][k]==a[1][k] for k in a[0] if k not in ['command','configuration_cycles'])
 info.append({'fixture':name,'scalar_updates':scalar,'paired_updates':paired,'dual_updates':dual,'z_min':int(z.min()),'z_max':int(z.max()),'q2_mac':int(mac),
  'mode14_cycles':next(r['cycles'] for r in rr if r['mode']==14 and not r['stall']),
  'mode15_cycles':next(r['cycles'] for r in rr if r['mode']==15 and not r['stall'])})
# Carry segmentation finite boundary combinations, including cross-half overflow/borrow.
carry_cases=0
for lo in [-2592,-1,0,1,2592]:
 for hi in [-2592,-1,0,1,2592]:
  for qv in range(-3,4):
   for enlo,enhi in [(0,0),(0,1),(1,0),(1,1)]:
    a=(lo&8191)|((hi&8191)<<13);b=((qv if enlo else 0)&8191)|(((qv if enhi else 0)&8191)<<13)
    out=0;cin=0
    for bit in range(32):
     if bit in [0,13]:cin=0
     aa=(a>>bit)&1;bb=(b>>bit)&1;out|=(aa^bb^cin)<<bit;cin=(aa&bb)|((aa^bb)&cin)
    assert int(signed(np.array([out&8191]),13)[0])==lo+enlo*qv
    assert int(signed(np.array([(out>>13)&8191]),13)[0])==hi+enhi*qv
    carry_cases+=1
out={'runs':len(rows),'gold_values_recomputed':len(info)*3840,'packed_state_values_recomputed':len(info)*320,
 'carry_boundary_combinations':carry_cases,'integer_bound_z':[-2592,2592],'integer_bound_abs_p':864*3*32768*8,
 'scope':'Read-only full native source/factor algebra, packed recurrence, finite carry-boundary checks and all measured state/counter sums; no Fmax/PPA.', 'fixtures':info}
(H/'checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
