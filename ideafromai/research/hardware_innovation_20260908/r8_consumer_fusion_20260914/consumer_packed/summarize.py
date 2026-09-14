import json,csv
from pathlib import Path
H=Path(__file__).resolve().parent;small=json.loads((H/'results.json').read_text())
s={'function':'same Q1/Q2 and exact FP32-to-I24 consumer','real8':[],'streams':{}}
fields=[k for k,v in small[0].items() if isinstance(v,int) and k not in ['mode','stall','command']]
for m in [14,15]:
 for stall in [0,1]:
  x=[a for a in small if a['mode']==m and a['stall']==stall and a['command']==0 and a['fixture'].startswith('real_')]
  row=dict(mode=m,stall=stall,**{k:sum(a[k] for a in x) for k in fields});row['without_static_and_parameter_stalls']=row['total_cycles']-row['static_words']-row['parameter_stalls'];s['real8'].append(row)
for suffix in ['64','full']:
 rows=json.loads((H/f'results_{suffix}.json').read_text());s['streams'][suffix]=rows
 with (H/f'stream_{suffix}_costs.csv').open('w') as f:w=csv.DictWriter(f,lineterminator='\n',fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (H/'real8_costs.csv').open('w') as f:w=csv.DictWriter(f,lineterminator='\n',fieldnames=list(s['real8'][0]));w.writeheader();w.writerows(s['real8'])
full=s['streams']['full'];s['full_reduction_fraction']=1-full[1]['total_cycles']/full[0]['total_cycles']
s['verified']={'small_jobs':len(small),'small_values_each_checkpoint':sum(x['outputs'] for x in small),'continuous64_jobs':len(s['streams']['64']),'full_jobs':len(full),'all_jobs':len(small)+len(s['streams']['64'])+len(full),'all_values_each_checkpoint':sum(x['outputs'] for x in small+s['streams']['64']+full)}
(H/'SUMMARY.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s['verified'],indent=2))
