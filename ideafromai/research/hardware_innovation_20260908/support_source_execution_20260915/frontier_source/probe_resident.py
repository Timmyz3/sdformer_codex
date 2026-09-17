#!/opt/anaconda3/bin/python3.12
from pathlib import Path
import json
import numpy as np
from probe_frontier import read_source,trace
HERE=Path(__file__).resolve().parent

def resident_trace(gates,q,mode):
 nodes=q['cl' if mode==3 else 'code'];tags=[-1]*4;fifo=0;records=[];pairs=[];labels=[];ever=set();refetch=hits=0;peak=0
 for g in range(6):
  ids=np.repeat(q['roots'][g+(6 if mode==3 else 0)],10).astype('int64')
  while np.any(ids>=16):
   var=np.array([int(nodes[j])>>32&15 if j>=16 else -1 for j in ids]);front=sorted(set(map(int,var[var>=0])),key=lambda c:int(q['rank'][g,c]));channels=([c for c in front if g*16+c in tags]+[c for c in front if g*16+c not in tags])[:4]
   slots={c:tags.index(g*16+c) for c in channels if g*16+c in tags};reserved=set(slots.values());hits+=len(slots);miss=[]
   for c in channels:
    if c in slots:continue
    free=[i for i in range(4) if i not in reserved and tags[i]<0]
    victim=free[0] if free else next((fifo+i)%4 for i in range(4) if (fifo+i)%4 not in reserved)
    slots[c]=victim;reserved.add(victim);tags[victim]=g*16+c;fifo=(victim+1)%4;miss.append(g*16+c)
    refetch+=g*16+c in ever;ever.add(g*16+c)
   active=np.isin(var,channels)&(ids>=16);peak=max(peak,sum(c>=0 for c in tags))
   records.append({'g':g,'channels':[g*16+c for c in channels],'miss_channels':miss,'active_mask':sum(int(v)<<t for t,v in enumerate(active)),'t_channels':[int(g*16+var[t]) if active[t] else -1 for t in range(10)]})
   for t in np.flatnonzero(active):
    c=g*16+int(var[t]);pairs.append((int(t),c));v=int(nodes[ids[t]]);ids[t]=(v>>16&65535) if gates[c,t] else (v&65535)
  labels.extend(map(int,ids))
 assert len(pairs)==len(set(pairs))
 return {'batches':len(records),'channel_refs':sum(len(r['channels']) for r in records),'channel_fetches':sum(len(r['miss_channels']) for r in records),'cache_hits':hits,'refetches':refetch,'peak_slots':peak,'needed_pairs':len(pairs),'scalar_mac_needed':10*len(pairs),'records':records,'labels':labels}
def main():
 A,tau,D,profiles,cases=read_source();q=profiles[1];rr=[]
 for c in cases:
  gates=c['x'].astype('int64')@A.astype('int64').T>=tau[None]
  for mode in [2,3]:
   x=resident_trace(gates,q,mode);base=trace(gates,q,mode,4);assert x['labels']==base['labels'] and x['needed_pairs']==base['needed_pairs']
   rr.append({'case':c['name'],'real':c['real'],'mode':mode,**x})
 ag=[]
 for mode in [2,3]:
  rows=[r for r in rr if r['real'] and r['mode']==mode];ag.append({'mode':mode,**{k:sum(r[k] for r in rows) for k in ['batches','channel_refs','channel_fetches','cache_hits','refetches','needed_pairs','scalar_mac_needed']}})
 (HERE/'probe_resident.json').write_text(json.dumps({'status':'PASS','cases':rr,'aggregate_real':ag,'scope':'fixed4 resident-first, D-rank fill, empty-first/FIFO unreferenced eviction; CPU opportunity, not RTL timing'},separators=(',',':'))+'\n');print(json.dumps(ag))

if __name__=='__main__':main()
