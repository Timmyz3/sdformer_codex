from pathlib import Path
import importlib.util,json
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('phase_verify',H.parent/'phase_borrow/verify.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
rows=json.loads((H/'results.json').read_text());cache={};checks=0
for r in rows:
 if r['fixture'] not in cache:cache[r['fixture']]=v.predict(r['fixture'])
 p=cache[r['fixture']];u=p['U'][r['mode']]
 assert r['core_first_issues']==u and r['core_mac_issues']==p['M'];checks+=2
 assert r['core_cycles']==4947+p['K']+2*p['Q']+3*u+p['M']+sum(r[k] for k in ['core_source_stalls','core_weight_stalls','core_output_stalls','core_borrow_waits']);checks+=1
for name in ['64','full']:
 p=H/f'results_{name}.json'
 if p.exists():rows+=json.loads(p.read_text())
for r in rows:
 t=r['retired_tiles'];assert r['core_psum_reads']==r['core_psum_writes']==0;checks+=1
 assert r['consumer_cycles']==3385*t+r['consumer_join_wait_cycles']+r['consumer_output_stalls']+r['consumer_wide_waits'];checks+=1
 assert r['total_cycles']==r['consumer_cycles']+r['static_words']+r['parameter_stalls']+r['source_load_words']+r['origin_words']+r['source_load_stalls']+2*t+1;checks+=1
 first=r.get('first_tile',r.get('tile_id',0));retained=sum(i%160!=0 for i in range(first+1,first+t))*768
 assert r['source_load_words']==1536*t-retained;checks+=1
 assert r['source_load_words']==r['external_source_words']+r['padding_words'];checks+=1
for r in json.loads((H/'results_64.json').read_text()):
 if r['mode']==2 and r['stall']==0:
  control=next(x for x in json.loads((H.parent/'dataflow/d1_forward/results_64.json').read_text()) if x['mode']==2 and x['stall']==0 and x['command']==r['command'])
  assert control['total_cycles']-r['total_cycles']==62*768+64*10;checks+=1
full=H/'results_full.json'
if full.exists():
 rr={r['mode']:r for r in json.loads(full.read_text())}
 counts=json.loads((H.parent/'phase_borrow/checks.json').read_text())['full_source_counts']
 assert rr[2]['total_cycles']-rr[3]['total_cycles']==counts['saved_cycles'];checks+=1
 for m in [2,3]:
  assert rr[m]['source_load_words']==14837760 and rr[m]['outputs']==73728000;checks+=2
out=dict(complete=True,command_records=len(rows),checks=checks,note='Combination measured, not multiplying ratios. Same halo/forward/416bit controls; no additional independent idea count.')
(H/'checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
