#!/opt/anaconda3/bin/python3.12
from pathlib import Path
import struct,json
import numpy as np
HERE=Path(__file__).resolve().parent

def read_source(path=None):
 f=(Path(path) if path else HERE.parent/'source.bin').open('rb')
 def arr(ty,n):return np.frombuffer(f.read(np.dtype(ty).itemsize*n),dtype=ty).copy()
 def word():return struct.unpack('<I',f.read(4))[0]
 A=arr('<i2',100).reshape(10,10);tau=arr('<i8',10);D=arr('u1',1536).reshape(6,16,16)
 profiles=[]
 for _ in range(2):
  code=arr('<u8',word());cl=arr('<u8',word());roots=arr('<u2',12);canonical=arr('u1',96).reshape(6,16);rank=arr('u1',96).reshape(6,16)
  profiles.append(dict(code=code,cl=cl,roots=roots,canonical=canonical,rank=rank))
 cases=[]
 for _ in range(word()):
  real=word();name=f.read(word()).decode();x=arr('<i4',960).reshape(96,10);cases.append(dict(real=real,name=name,x=x))
 assert not f.read(1)
 return A,tau,D,profiles,cases

def trace(gates,q,mode,width):
 nodes=q['cl' if mode==3 else 'code'];records=[];pairs=[];labels=[]
 for g in range(6):
  ids=np.repeat(q['roots'][g+(6 if mode==3 else 0)],10).astype('int64')
  while np.any(ids>=16):
   var=np.array([int(nodes[j])>>32&15 if j>=16 else -1 for j in ids])
   channels=sorted(set(map(int,var[var>=0])),key=lambda c:int(q['rank'][g,c]))[:width]
   active=np.isin(var,channels)&(ids>=16)
   records.append({'g':g,'channels':[g*16+c for c in channels],'active_mask':sum(int(v)<<t for t,v in enumerate(active)),'t_channels':[int(g*16+var[t]) if active[t] else -1 for t in range(10)]})
   for t in np.flatnonzero(active):
    c=g*16+int(var[t]);pairs.append((int(t),c));v=int(nodes[ids[t]]);ids[t]=(v>>16&65535) if gates[c,t] else (v&65535)
  labels.extend(map(int,ids))
 assert len(pairs)==len(set(pairs))
 return {'batches':len(records),'channel_fetches':sum(len(r['channels']) for r in records),'needed_pairs':len(pairs),'scalar_mac_needed':10*len(pairs),'active_lane_histogram':dict(zip(*[a.tolist() for a in np.unique([int(r['active_mask']).bit_count() for r in records],return_counts=True)])),'records':records,'labels':labels}

def main():
 A,tau,D,profiles,cases=read_source();q=profiles[1];out=[]
 for case in cases:
  u=case['x'].astype('int64')@A.astype('int64').T;gates=u>=tau[None]
  raw=gates.T.reshape(10,6,16);distance=np.count_nonzero(raw[:,:,None]!=D[None],axis=-1);code=distance.argmin(-1).T
  for mode in [2,3]:
   a=trace(gates,q,mode,1);x=trace(gates,q,mode,4);gold=code if mode==2 else np.stack([q['canonical'][g,code[g]] for g in range(6)])
   assert np.array_equal(a['labels'],gold.ravel()) and a['labels']==x['labels']
   assert a['needed_pairs']==x['needed_pairs']
   out.append({'case':case['name'],'real':case['real'],'mode':mode,'one_channel':a,'frontier4':x})
 ag=[]
 for mode in [2,3]:
  rr=[r for r in out if r['real'] and r['mode']==mode]
  ag.append({'mode':mode,'cases':len(rr),'one_channel_batches':sum(r['one_channel']['batches'] for r in rr),'one_channel_full_T10_scalar_MAC':sum(r['one_channel']['batches']*100 for r in rr),'frontier4_batches':sum(r['frontier4']['batches'] for r in rr),'frontier4_channel_fetches':sum(r['frontier4']['channel_fetches'] for r in rr),'frontier4_source_words':sum(r['frontier4']['channel_fetches']*2 for r in rr),'needed_pairs':sum(r['frontier4']['needed_pairs'] for r in rr),'needed_scalar_MAC':sum(r['frontier4']['scalar_mac_needed'] for r in rr)})
 result={'status':'PASS','scope':'Actual integer X/A/tau and D-only entropy rank, graph logical trace, not RTL latency','source':str(HERE.parent/'source.bin'),'cases':out,'aggregate_real':ag,'new_interface':'Each producer valid t returns only its current channel; channel may be fetched again for another t later. No complete T10 provider claim.'}
 (HERE/'probe_frontier.json').write_text(json.dumps(result,separators=(',',':'))+'\n');print(json.dumps(ag))
if __name__=='__main__':main()
