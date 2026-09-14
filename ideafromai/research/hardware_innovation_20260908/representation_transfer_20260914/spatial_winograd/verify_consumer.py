from pathlib import Path
import ast,json,sys
H=Path(__file__).resolve().parent
p=json.loads((H/'profiles.json').read_text());out={};checks=commands=0
def eq(a,b,label=''):
 global checks
 checks+=1;assert a==b,(label,a,b)
node=next(n for n in ast.parse((H/'verify_raw.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='predict')
env={'eq':eq};exec(compile(ast.Module(body=[node],type_ignores=[]),'predict','exec'),env);predict=env['predict']
for stage in (sys.argv[1:] or ['small','short','held','disjoint','sequences']):
 names=(H/f'{stage}.txt').read_text().splitlines();out[stage]={}
 for mode in [0,1]:
  for stall in [0,1]:
   rows=[json.loads(l) for l in (H/f'consumer_{stage}_m{mode}_s{stall}.jsonl').read_text().splitlines()]
   eq(len(rows),2*len(names));commands+=len(rows)
   for i,r in enumerate(rows):
    eq(r['fixture'],names[i%len(names)]);eq(r['command'],i);eq(r['mode'],mode);eq(r['stall'],stall)
    ex,states=predict(p[r['fixture']],mode)
    for k,v in ex.items():eq(r[k],v,(stage,i,mode,k))
    for k in ['raw_values','j_values','wide_values','i24_values']:eq(r[k],3840)
    eq(r['z_values'],1280);eq(r['d_values'],1280 if mode else 0)
    eq(r['configuration_cycles'],1537+((1368 if mode else 1176) if i==0 else 0))
    eq(r['cycles'],sum(states)+sum(r[k] for k in ['source_stalls','weight_stalls','output_stalls']))
    eq(sum(r['state_cycles']),r['c_cycles']);eq(sum(r['state_cycles'])-r['state_cycles'][0],r['cycles'])
    for j,v in enumerate(states):
     if j==3:eq(r['state_cycles'][j],v+r['source_stalls'])
     elif j==17:eq(r['state_cycles'][j],v+r['output_stalls'])
     elif j not in [0,6,12]:eq(r['state_cycles'][j],v,(stage,i,mode,'state',j))
    eq(r['state_cycles'][6]+r['state_cycles'][12],states[6]+states[12]+r['weight_stalls'])
    fixed={'c_raw_words':480,'c_identity_words':480,'c_coefficient_words':24,'c_mul_issues':480,'c_add_issues':960,'c_round_issues':480,'c_output_words':480,'c_conversion_issues':480,'c_conversion_saturations':0,'c_wide_waits':0}
    for k,v in fixed.items():eq(r[k],v,(stage,i,mode,k))
    eq(r['c_cycles'],3385+r['c_join_wait_cycles']+r['c_output_stalls'])
    if not stall:
     eq(r['source_stalls']+r['weight_stalls']+r['c_identity_stalls']+r['c_output_stalls'],0)
     eq(r['c_cycles'],sum(states)+2425)
    if i>=len(names):
     for k,v in r.items():
      if k not in ['command','configuration_cycles']:eq(v,rows[i-len(names)][k],('resident repeat',k))
   for repeat in [0,1]:
    rr=rows[repeat*len(names):(repeat+1)*len(names)]
    out[stage][f'm{mode}_s{stall}_r{repeat}']=dict(service=sum(r['c_cycles']+r['configuration_cycles']+1 for r in rr),core=sum(r['cycles'] for r in rr),consumer=sum(r['c_cycles'] for r in rr),configuration=sum(r['configuration_cycles'] for r in rr),**{k:sum(r[k] for r in rr) for k in ex})
summary=dict(passed=True,commands=commands,raw_J20_wide_I24_each=commands*3840,z_values=commands*1280,d_values=commands//2*1280,checks=checks,sets=out,oracle_fixtures=len(p),scope='actual raw/Z/D/J20/wide64/I24 RTL; M prefix assertions and static path review')
(H/'consumer_verification.json').write_text(json.dumps(summary,separators=(',',':'))+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='sets'}));print(json.dumps({k:{m:q['service'] for m,q in v.items()} for k,v in out.items()}))
