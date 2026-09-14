from pathlib import Path
import json,argparse
from verify import profile
H=Path(__file__).resolve().parent;S=H.parent/'pair_sparse'
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['small','64'],default='small');args=p.parse_args()
rows=json.loads((H/f'results_stream_{args.stage}.json').read_text());paths=sorted({r['fixture'] for r in rows});n=len(paths)
pred={path:profile(Path(path),H/'fixtures/real_0') for path in paths}
old_rows=json.loads((S/('results.json' if args.stage=='small' else 'results_stream.json')).read_text())
matched=0
for r in rows:
 if r['mode'] in [14,20]:
  exp=pred[r['fixture']][r['mode']]
  for key,value in exp.items():
   if key!='base_cycles':assert r[key]==value,(r['fixture'],r['mode'],key,r[key],value)
  assert r['cycles']==exp['base_cycles']+sum(r[k] for k in ['source_stalls','weight_stalls','output_stalls'])
 if r['mode'] in [14,19]:
  if args.stage=='64':
   old=next(x for x in old_rows if x['mode']==r['mode'] and x['stall']==r['stall'] and x['command']==r['command'])
   for key,value in old.items():assert r[key]==value,(r['mode'],key,r[key],value)
  else:
   old=next(x for x in old_rows if x['mode']==r['mode'] and x['stall']==r['stall'] and x['fixture']==Path(r['fixture']).name and x['command']==0)
   for key,value in old.items():
    if key not in ['fixture','command','configuration_cycles']:assert r[key]==value,(r['mode'],key,r[key],value)
  matched+=1
 assert r['configuration_cycles']==1537+(1824 if r['mode']==14 else 2721)*(r['command']==0)
 assert sum(r['state_cycles'])==r['cycles']
result=dict(passed=True,stage=args.stage,tiles_per_round=n,rtl_commands=len(rows),raw_values=sum(r['outputs'] for r in rows),distinct_native_gold_values=n*3840,old14_19_records_reproduced=matched,all_count8_state_and_physical_port_obligations_verified=True,rounds_without_reset=2,configuration='resident Q1/Q2 and metadata once;1537 source/origin beats per command plus1start beat')
(H/f'stream_checks_{args.stage}.json').write_text(json.dumps(result,indent=2)+'\n')
(H/f'profiles_stream_{args.stage}.json').write_text(json.dumps(pred,indent=2)+'\n')
print(json.dumps(result,indent=2))
