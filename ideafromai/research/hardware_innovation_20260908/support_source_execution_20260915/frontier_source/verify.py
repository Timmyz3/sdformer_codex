#!/opt/anaconda3/bin/python3.12
from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np
from probe_frontier import read_source,trace
from probe_resident import resident_trace
HERE=Path(__file__).resolve().parent
SETS={'small':('resident_cycles.csv',HERE.parent/'source.bin'),'expanded_old':('expanded_old_cycles.csv',HERE.parent/'source_class_adapt/expanded_sources/old_class/source.bin'),'expanded_new':('expanded_new_cycles.csv',HERE.parent/'source_class_adapt/expanded_sources/source.bin')}
BASE_FIELDS=['cycles','channels','scalar_mac','words','xwords','graph_words','prefetch_words','graph_hits','req_stall','out_stall']+[f'state{i}' for i in range(14)]
smallorig={(r['case'],int(r['mode']),int(r['bp'])):r for r in csv.DictReader((HERE.parent/'source_cycles.csv').open()) if r['order']=='1' and r['packed32']=='1' and r['prefetch']=='1'}
plain={(r['case'],int(r['mode']),int(r['bp']),int(r['frontier'])):r for r in csv.DictReader((HERE/'plain_cycles.csv').open())}
summary={};unique_valid=labels=0
previous=None
for name,(csvname,path) in SETS.items():
 A,tau,D,profiles,cases=read_source(path);q=profiles[1];rr=list(csv.DictReader((HERE/csvname).open()))
 assert len(rr)==(1340 if name=='small' else 12324),(name,len(rr))
 for r in rr:
  for k in list(r):
   if k!='case':r[k]=int(r[k])
 bycase=defaultdict(list)
 for r in rr:bycase[r['case']].append(r)
 if previous is not None and name=='expanded_new':
  aa,tt,dd,qq,cc=previous
  assert np.array_equal(A,aa) and np.array_equal(tau,tt) and np.array_equal(D,dd) and np.array_equal(q['code'],qq['code']) and np.array_equal(q['rank'],qq['rank']) and np.array_equal(q['roots'][:6],qq['roots'][:6])
  assert all(c['name']==z['name'] and np.array_equal(c['x'],z['x']) for c,z in zip(cases,cc))
 if name=='expanded_old':previous=(A,tau,D,q,cases)
 for c in cases:
  u=c['x'].astype('int64')@A.astype('int64').T;gates=u>=tau[None];expected={}
  raw=gates.T.reshape(10,6,16);idx=np.count_nonzero(raw[:,:,None]!=D[None],axis=-1).argmin(-1).T
  for m in [2,3]:
   for fr,res in [(0,0),(0,1),(1,0),(1,1)]:
    x=resident_trace(gates,q,m) if fr and res else trace(gates,q,m,4 if fr else 1)
    gold=idx if m==2 else np.stack([q['canonical'][g,idx[g]] for g in range(6)])
    assert np.array_equal(x['labels'],gold.ravel())
    if not fr:
     x['needed_pairs']=x['batches']*10;x['channel_refs']=x['channel_fetches'];x['cache_hits']=0;x['refetches']=0
    elif not res:
     x['channel_refs']=x['channel_fetches'];x['cache_hits']=0
     channels=[ch for r in x['records'] for ch in r['channels']];x['refetches']=len(channels)-len(set(channels))
    expected[(m,fr,res)]=x
  for r in bycase[c['name']]:
   assert r['cycles']==sum(r[f'state{i}'] for i in range(14))-1
   assert r['scalar_mac']==r['pairs']*10
   assert r['words']==r['graph_words']+r['xwords']+(30 if r['mode']==1 else 21)
   assert r['state8']==10*r['batches'] and r['state7']==r['state10']==r['batches']
   assert r['xwords']//2+r['cache_hits']==r['channels'] and r['xwords']%2==0
   assert r['peak_slots']<=4
   if r['mode']==1:
    assert r['channels']==r['batches']==64 and r['pairs']==640 and r['xwords']==128 and r['cache_hits']==r['refetches']==0
   else:
    ex=expected[(r['mode'],r['frontier'],r['resident'])]
    for actual,key in [('batches','batches'),('pairs','needed_pairs'),('channels','channel_refs'),('cache_hits','cache_hits'),('refetches','refetches')]:assert r[actual]==ex[key],(name,c['name'],r['mode'],r['frontier'],r['resident'],actual,r[actual],ex[key])
    assert r['xwords']==2*ex['channel_fetches']
   if name=='small' and not r['frontier']:
    old=smallorig[(r['case'],r['mode'],r['bp'])]
    for k in BASE_FIELDS:assert r[k]==int(old[k]),(name,r['case'],k,r[k],old[k])
   if name=='small' and not r['resident']:
    old=plain[(r['case'],r['mode'],r['bp'],r['frontier'])]
    for k in old:
     if k!='case':assert r[k]==int(old[k]),(r['case'],k)
   unique_valid+=r['pairs'];labels+=60
 aggregates=[]
 for split in ['all_real','frame0','unselected31']:
  for mode,fr,res,bp in sorted(set((r['mode'],r['frontier'],r['resident'],r['bp']) for r in rr)):
   selected=[r for r in rr if r['real'] and (split=='all_real' or (r['case'].startswith('train0_') if split=='frame0' else not r['case'].startswith('train0_'))) and (r['mode'],r['frontier'],r['resident'],r['bp'])==(mode,fr,res,bp)]
   if name=='small' and split=='unselected31':continue
   sums={k:sum(r[k] for r in selected) for k in ['cycles','channels','batches','pairs','scalar_mac','words','xwords','graph_words','prefetch_words','cache_hits','refetches','req_stall','out_stall']+[f'state{i}' for i in range(14)]}
   aggregates.append({'split':split,'mode':mode,'frontier':fr,'resident':res,'bp':bp,'commands':len(selected),**sums})
 summary[name]={'source':str(path),'commands':len(rr),'valid_U_and_gate_checks':sum(r['pairs'] for r in rr),'labels_checked':len(rr)*60,'aggregates':aggregates}
 if name=='expanded_old':oldrecords={(r['case'],r['mode'],r['frontier'],r['resident'],r['bp']):r for r in rr}
 if name=='expanded_new':
  for r in rr:
   if r['mode']!=3:assert r==oldrecords[(r['case'],r['mode'],r['frontier'],r['resident'],r['bp'])]
 print('PASS',name,len(rr),flush=True)
result={'status':'PASS','sets':summary,'total_commands':sum(s['commands'] for s in summary.values()),'actual_valid_U_and_gate_checks_each':unique_valid,'actual_classification_labels_checked':labels,'methods':'Direct integer dot and complete Hamming/canonical in TB; independent CPU path+4-slot FIFO simulation verifies per-case batches/pairs/hits/reloads/traffic. Old controls/plain exactly reproduced; expanded code/static records equal across old/new class fixtures.','limitations':['Every command resets currently; no uninterrupted post-DONE restart tested.','Frames are existing training cache. Frame0 selected adapted response pair;31 remaining training frames are not independent validation.','Provider returns only producer_active[t] values, not full rawT10 words.','Child prefetch remains old minimum-rank channel only, not all active frontier lanes; an unfinished supply interface.', 'No new AEE/Fmax/area or downstream FC1/joined-chain integration.']}
(HERE/'SUMMARY.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
print('PASS ALL',result['total_commands'],unique_valid,labels)
