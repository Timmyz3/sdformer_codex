from pathlib import Path
import argparse,json,numpy as np
H=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('result',nargs='?',default='results64.json');args=parser.parse_args()
rows=json.loads((H/args.result).read_text());checks=0
for r in rows:
 e=np.load(H/f"expected_{r['arm']}_m{r['mode']}.npz");selection=slice(r['first_tile'],r['first_tile']+r['tiles'])
 for field in e.files:
  if field=='terminal_savings':continue
  expect=int(e[field][selection].sum());actual=r[field]
  if field=='core_cycles':actual-=r['core_source_stalls']+r['core_weight_stalls']+r['core_output_stalls']
  assert actual==expect,(r['arm'],r['mode'],r['stall'],field,actual,expect);checks+=1
 core_expected=int(e['core_cycles'][selection].sum())
 stall=sum(r[f] for f in ('core_source_stalls','core_weight_stalls','core_output_stalls','source_load_stalls','parameter_stalls'))
 assert r['total_cycles']==core_expected+1539*r['tiles']+1+(10656 if r['command']==0 else 0)+stall
 assert r['checked_outputs']==r['tiles']*3840 and r['retired_tiles']==r['tiles']
 checks+=3
out={'passed':True,'runs':len(rows),'checks':checks,'checked_outputs':sum(r['checked_outputs'] for r in rows),'source':'Full native source/mask arrays; vectorized geometry/pair support cycle prediction. No oracle supplied to RTL.'}
(H/(Path(args.result).stem+'_checks.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
