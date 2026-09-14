from pathlib import Path
import json,sys
H=Path(__file__).resolve().parent
p=json.loads((H/'profiles.json').read_text());out={};checks=commands=0
def eq(a,b,label=''):
 global checks
 checks+=1;assert a==b,(label,a,b)
def predict(e,mode):
 ex={k:v for k,v in e.items() if k not in ['base_states','base_cycles','winograd_mac','winograd_weight']}
 states=e['base_states'].copy()
 ex.update(transform_issues=0,transform_reads=0,transform_writes=0,reconstruction_issues=0,stripe_add_issues=0,cache_reads=e['q2_issues'],exact_halves=0)
 if mode:
  states[12]=768;states[13]=480;states[14]=e['winograd_mac'];states[15]=0
  for j in range(19,25):states[j]=40
  for j in [25,26,29,30,31,34]:states[j]=480
  for j in [27,28,32,33]:states[j]=240
  ex.update(q2_words=e['winograd_weight'],q2_issues=e['winograd_mac'],z_vector_reads=e['z_vector_reads']+80,z_scalar_reads=e['winograd_mac'],z_writes=e['z_writes']+80,cache_writes=768,
   transform_issues=160,transform_reads=80,transform_writes=80,reconstruction_issues=1920,stripe_add_issues=480,cache_reads=e['winograd_mac'],exact_halves=960)
  eq(sum(states),e['base_cycles']+e['winograd_mac']-e['q2_issues']+2832,'paid transform/reconstruction schedule')
 return ex,states
for stage in (sys.argv[1:] or ['small','short','held','disjoint','sequences']):
 names=(H/f'{stage}.txt').read_text().splitlines();out[stage]={}
 for mode in [0,1]:
  for stall in [0,1]:
   rows=[json.loads(l) for l in (H/f'raw_{stage}_m{mode}_s{stall}.jsonl').read_text().splitlines()]
   eq(len(rows),2*len(names));commands+=len(rows)
   for i,r in enumerate(rows):
    eq(r['fixture'],names[i%len(names)]);eq(r['command'],i);eq(r['mode'],mode);eq(r['stall'],stall)
    ex,states=predict(p[r['fixture']],mode)
    for k,v in ex.items():eq(r[k],v,(stage,i,mode,k))
    eq(r['outputs'],3840);eq(r['z_values'],1280);eq(r['d_values'],1280 if mode else 0)
    eq(r['configuration_cycles'],1537+((1344 if mode else 1152) if i==0 else 0))
    eq(r['cycles'],sum(states)+sum(r[k] for k in ['source_stalls','weight_stalls','output_stalls']))
    for j,v in enumerate(states):
     if j==3:eq(r['state_cycles'][j],v+r['source_stalls'])
     elif j==17:eq(r['state_cycles'][j],v+r['output_stalls'])
     elif j not in [6,12]:eq(r['state_cycles'][j],v,(stage,i,mode,'state',j))
    eq(r['state_cycles'][6]+r['state_cycles'][12],states[6]+states[12]+r['weight_stalls'])
    if not stall:eq(r['source_stalls']+r['weight_stalls']+r['output_stalls'],0)
    if i>=len(names):
     for k,v in r.items():
      if k not in ['command','configuration_cycles']:eq(v,rows[i-len(names)][k],('resident repeat',k))
   for repeat in [0,1]:
    rr=rows[repeat*len(names):(repeat+1)*len(names)]
    out[stage][f'm{mode}_s{stall}_r{repeat}']=dict(service=sum(r['cycles']+r['configuration_cycles']+1 for r in rr),core=sum(r['cycles'] for r in rr),configuration=sum(r['configuration_cycles'] for r in rr),**{k:sum(r[k] for r in rr) for k in ex})
summary=dict(passed=True,commands=commands,raw_values=commands*3840,z_values=commands*1280,d_values=commands//2*1280,checks=checks,sets=out,oracle_fixtures=len(p),M_scope='shared update path/prefix assertions and independent arithmetic, no per-M RTL monitor')
(H/'raw_verification.json').write_text(json.dumps(summary,separators=(',',':'))+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='sets'}));print(json.dumps({k:{m:q['service'] for m,q in v.items()} for k,v in out.items()}))
