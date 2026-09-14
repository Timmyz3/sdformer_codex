import csv,json
from pathlib import Path
H=Path(__file__).resolve().parent;r=json.loads((H/'results.json').read_text())
fields=['total_cycles','consumer_cycles','core_cycles','static_words','source_load_words','origin_words','external_source_words','padding_words','parameter_stalls','source_load_stalls','core_source_words','core_weight_words','core_z_reads','core_z_writes','core_psum_reads','core_psum_writes','core_first_issues','core_mac_issues','core_support_reads','core_local_source_reads','consumer_identity_words','consumer_coefficient_words','consumer_conversion_issues','consumer_mul_issues','consumer_add_issues','consumer_round_issues','consumer_output_words','consumer_join_wait_cycles','consumer_identity_stalls','consumer_output_stalls','consumer_saturations','consumer_conversion_saturations']
s={'identity_input':'IEEE_binary32','unit_runs':len(r),'unit_I24_outputs':sum(x['outputs'] for x in r),'unit_raw_outputs':sum(x['raw_outputs'] for x in r),'unit_J_outputs':sum(x['J_outputs'] for x in r),'real8':[]}
for m in [6,7,9,11]:
 for stall in [0,1]:
  x=[a for a in r if a['fixture'].startswith('real_') and a['mode']==m and a['stall']==stall and a['command']==0]
  a=dict(mode=m,stall=stall,**{k:sum(t[k] for t in x) for k in fields});a['without_repeated_static_and_parameter_stalls']=a['total_cycles']-a['static_words']-a['parameter_stalls'];s['real8'].append(a)
with (H/'real8_costs.csv').open('w') as f:
 w=csv.DictWriter(f,lineterminator='\n',fieldnames=list(s['real8'][0]));w.writeheader();w.writerows(s['real8'])
for suffix in ['64','full']:
 p=H/f'results_{suffix}.json'
 if p.exists():
  a=json.loads(p.read_text())
  if all(x.get('identity_input')=='IEEE_binary32' for x in a):
   s['stream_'+suffix]=a
   with (H/f'stream_{suffix}_costs.csv').open('w') as f:
    w=csv.DictWriter(f,lineterminator='\n',fieldnames=list(a[0]));w.writeheader();w.writerows(a)
(H/'SUMMARY.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps({k:v for k,v in s.items() if k not in ['real8','stream_64','stream_full']},indent=2))
