#!/opt/anaconda3/bin/python3.12
from pathlib import Path
import csv,json
HERE=Path(__file__).resolve().parent
sets={'small':('small_cycles.csv','resident_cycles.csv',2680),'expanded_old':('expanded_old_cycles.csv','expanded_old_cycles.csv',24648),'expanded_new':('expanded_new_cycles.csv','expanded_new_cycles.csv',24648)}
summary={};total_pairs=0;total_labels=0

def read(path):
 return [{k:v if k=='case' else int(v) for k,v in r.items()} for r in csv.DictReader(path.open())]
def key(r):return tuple(r[k] for k in ['case','mode','frontier','resident','bp'])
for name,(filename,parentfile,count) in sets.items():
 rr=read(HERE/filename);parent={key(r):r for r in read(HERE.parent/parentfile)}
 assert len(rr)==count
 pairs={}
 for r in rr:
  base=parent[key(r)]
  if not r['active_pf']:
   for k,v in base.items():assert r[k]==v,(name,r['case'],k,r[k],v)
  for k in ['channels','batches','pairs','scalar_mac','xwords','cache_hits','refetches','peak_slots']:
   assert r[k]==base[k],(name,r['case'],r['mode'],k,r[k],base[k])
  if not r['frontier']:
   for k,v in base.items():assert r[k]==v,(name,r['case'],'ordinary changed',k)
  assert r['cycles']==sum(r[f'state{i}'] for i in range(14))-1
  assert r['words']==r['xwords']+r['graph_words']+(30 if r['mode']==1 else 21)
  assert r['scalar_mac']==r['pairs']*10 and r['state8']==r['batches']*10
  assert r['state7']==r['state10']==r['batches']
  for k in ['state0','state1','state2','state3','state5','state7','state8','state10','state11','state13']:assert r[k]==base[k]
  total_pairs+=r['pairs'];total_labels+=60
  pairs[(key(r),r['active_pf'])]=r
 aggregates=[]
 for split in ['all_real','frame0','unselected31']:
  if name=='small' and split=='unselected31':continue
  for mode,fr,res,bp,pf in sorted({tuple(r[k] for k in ['mode','frontier','resident','bp','active_pf']) for r in rr}):
   selected=[r for r in rr if r['real'] and (split=='all_real' or (r['case'].startswith('train0_') if split=='frame0' else not r['case'].startswith('train0_'))) and tuple(r[k] for k in ['mode','frontier','resident','bp','active_pf'])==(mode,fr,res,bp,pf)]
   sums={k:sum(r[k] for r in selected) for k in ['cycles','channels','batches','pairs','scalar_mac','words','xwords','graph_words','prefetch_words','cache_hits','refetches','graph_hits','req_stall','out_stall']+[f'state{i}' for i in range(14)]}
   aggregates.append({'split':split,'mode':mode,'frontier':fr,'resident':res,'bp':bp,'active_pf':pf,'commands':len(selected),**sums})
 changes=[]
 for mode,fr,res,bp in sorted({tuple(r[k] for k in ['mode','frontier','resident','bp']) for r in rr}):
  cohort=[r for r in rr if r['real'] and (name=='small' or not r['case'].startswith('train0_')) and tuple(r[k] for k in ['mode','frontier','resident','bp'])==(mode,fr,res,bp) and not r['active_pf']]
  deltas=[pairs[(key(r),1)]['cycles']-r['cycles'] for r in cohort]
  changes.append({'mode':mode,'frontier':fr,'resident':res,'bp':bp,'commands':len(cohort),'cycle_delta_sum':sum(deltas),'faster':sum(v<0 for v in deltas),'equal':sum(v==0 for v in deltas),'slower':sum(v>0 for v in deltas),'worst_delta':max(deltas),'best_delta':min(deltas)})
 summary[name]={'commands':len(rr),'valid_U_and_gate_checks_each':sum(r['pairs'] for r in rr),'classification_labels':len(rr)*60,'aggregates':aggregates,'per_case_delta_statistics':changes}
 if name=='expanded_old':old={tuple(r[k] for k in ['case','mode','frontier','resident','bp','active_pf']):r for r in rr}
 if name=='expanded_new':
  for r in rr:
   if r['mode']!=3:assert r==old[tuple(r[k] for k in ['case','mode','frontier','resident','bp','active_pf'])]
 print('PASS',name,len(rr),flush=True)
result={'status':'PASS','sets':summary,'total_commands':sum(v['commands'] for v in summary.values()),'valid_U_and_gate_checks_each':total_pairs,'classification_labels':total_labels,'verification':'Physical RTL MAC/gate/label/protocol checks; all old-PF records exactly reproduce parent; all source-plan counters match independently CPU-validated parent for both PF scopes; ordinary single-channel and static entire records unchanged; old/new class mirrors have identical code/static records.', 'resource_delta':'One runtime PF-scope bit and active/slot matching control, no new arrays, slots, pending capacity, adders or multipliers; synthesis/timing unmeasured.', 'scope':'Fixed existing32 training-cache frames×32 samples plus3 directed cases. frame0 selected new class pair;31 remaining frames are still training data. No end-to-end joined/AEE result here.'}
(HERE/'SUMMARY.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
print('PASS ALL',result['total_commands'],total_pairs,total_labels)
