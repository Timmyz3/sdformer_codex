from pathlib import Path
import sys,json
import numpy as np
H=Path(__file__).resolve().parent; O=H.parent; B=O.parents[1]
sys.path.insert(0,str(O))
from prepare import readhex
PERM=[8,2,6,3,9,4,1,5,7,0]
src=np.load(B/'r8_consumer_fusion_20260914/data/first_source_words.npy',mmap_mode='r')
p=O/'fixtures/real_0';q=readhex(p/'q1.hex').reshape(864,8)
v=readhex(p/'q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
cl=(readhex(p/'class.hex')[:,None]>>(6*np.arange(4)))&63
ng=int(readhex(p/'ngroups.hex')[0]); live=np.any(q!=0,axis=1)
def events(tile):
 oy,ox=2*(tile//160)-1,2*(tile%160)-1
 a=np.zeros((864,4,10),np.int64)
 for k in range(864):
  c,tap=divmod(k,9)
  for pp in range(4):
   y=oy+pp//2+tap//3;x=ox+pp%2+tap%3
   if 0<=y<240 and 0<=x<320:a[k,pp]=(int(src[c,y,x])>>np.arange(10))&1
 return a
def model(e,order,q=q,v=v,cl=cl,ng=ng):
 live=np.any(q!=0,axis=1)
 ev=e[:,:,order];active=live&np.any(ev,axis=(1,2));K=int(live.sum());A=int(active.sum())
 grouped=(cl>0)&(cl<=32);direct=active&np.any(cl==63,axis=1)
 pair=ev[:,[0,2],:]|ev[:,[1,3],:];U=int(pair[live].sum());dU=int(pair[direct].sum());dA=int(direct.sum())
 z=ev.reshape(864,40).T@q
 vlive=np.any(v.reshape(12,8,8)!=0,axis=1);M=int(((z!=0)*vlive.sum(axis=0)).sum())
 touched=np.zeros((4,32,5),bool);counts=np.zeros((4,32,4,10),np.int64)
 updates=reads=writes=read_vectors=0
 for k in range(864):
  if not active[k] or not np.any(grouped[k]):continue
  for g in range(4):
   if grouped[k,g]:counts[g,cl[k,g]-1]+=ev[k]
  for b in range(5):
   if not np.any(ev[k,:,2*b:2*b+2]):continue
   updates+=1;rb=0
   for g in range(4):
    if grouped[k,g]:
     slot=cl[k,g]-1;rb+=2*int(touched[g,slot,b]);writes+=2;touched[g,slot,b]=True
   reads+=rb;read_vectors+=int(rb>0)
 slots=int(np.any(touched,axis=(0,2)).sum());blocks=int(np.any(touched,axis=0).sum());reads+=int(touched.sum())*2
 packed=scalar=0
 for slot in range(32):
  for t in range(10):
   for pp in range(2):
    lo,hi=counts[:,slot,2*pp,t],counts[:,slot,2*pp+1,t]
    n=int(np.any(lo))+int(np.any(hi))
    if n:
     if np.all(hi<=127):packed+=1
     else:scalar+=n
 retire=packed+scalar
 cycles=5437+K+2*dA+3*dU+M+A+2*updates+(ng+1+slots+2*blocks+3*retire if ng else 0)
 return dict(source_blocks=updates,retire_blocks=blocks,packed_retire=packed,scalar_retire=scalar,retire_issues=retire,count_bank_reads=reads,count_bank_writes=writes,first_touch_read_vectors=read_vectors,slots=slots,K=K,A=A,direct_A=dA,direct_U=dU,Q2_mac=M,core_cycles=cycles,native_core_cycles=5437+K+2*A+3*U+M)
def main():
 result={'permutation':PERM,'no_refit':True,'configuration_bits':40,'configuration_cycles':1,'sets':{}}
 for label,ids in [('cal_0_31',range(32)),('held_128_191',range(128,192)),('disjoint_4000_4063',range(4000,4064))]:
  tiles=[]
  for tile in ids:
   e=events(tile);a=model(e,list(range(10)));b=model(e,PERM)
   for key in ['packed_retire','scalar_retire','retire_issues','slots','K','A','direct_A','direct_U','Q2_mac','native_core_cycles']:assert a[key]==b[key],(tile,key)
   assert b['core_cycles']-a['core_cycles']==2*(b['source_blocks']-a['source_blocks'])+2*(b['retire_blocks']-a['retire_blocks'])
   tiles.append(dict(tile=tile,original=a,permuted=b))
  totals={arm:{k:sum(row[arm][k] for row in tiles) for k in tiles[0][arm]} for arm in ['original','permuted']}
  for arm in totals:
   totals[arm]['cold_service']=totals[arm]['core_cycles']+2721+1538*len(tiles)+(arm=='permuted')
  delta={k:totals['permuted'][k]-totals['original'][k] for k in totals['original']}
  result['sets'][label]=dict(tiles=tiles,totals=totals,delta_candidate_minus_original=delta)
 # Independent event/count model must reproduce the already measured held stream.
 old=json.loads((O/'results_stream_64.json').read_text())
 rr=[r for r in old if r['mode']==20 and r['stall']==0 and r['command']<64]
 assert len(rr)==64,len(rr)
 for key,col in [('core_cycles','cycles'),('source_blocks','aux_issues'),('retire_issues','aux_events'),('count_bank_reads','count_bank_reads'),('count_bank_writes','count_bank_writes')]:
  assert result['sets']['held_128_191']['totals']['original'][key]==sum(r[col] for r in rr),(key,col)
 result['original_held_matches_rtl']=True
 (H/'opportunity.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:{'totals':v['totals'],'delta':v['delta_candidate_minus_original']} for k,v in result['sets'].items()},indent=2))
if __name__=='__main__':main()
