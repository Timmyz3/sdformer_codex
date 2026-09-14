from pathlib import Path
import json
H=Path(__file__).resolve().parent;R=H.parent/'spatial_winograd/pruned_replay';rows=[];same=set()
for key in ['q1','q2','consumer']:assert (H/f'parameters/{key}.hex').read_bytes()==(R/f'moment/parameters/{key}.hex').read_bytes(),key
for stage in ['small','held','disjoint','sequences']:
 current=(H/f'{stage}.txt').read_text().splitlines();other=(R/f'moment/{stage}.txt').read_text().splitlines();assert len(current)==len(other)
 for a,b in zip(current,other):
  for field in ['source','origin','gold','identity','j','wide','i24','z']:
   assert (Path(a)/f'{field}.hex').read_bytes()==(Path(b)/f'{field}.hex').read_bytes(),(stage,a,b,field)
  same.add(a)
 for consumer in [False,True]:
  prefix='consumer' if consumer else 'raw';cycle='c_cycles' if consumer else 'cycles'
  for stall in [0,1]:
   c=[json.loads(v) for v in (H/f'{prefix}_{stage}_s{stall}.jsonl').read_text().splitlines()]
   general=[json.loads(v) for v in (R/f'moment/{prefix}_{stage}_m1_s{stall}.jsonl').read_text().splitlines()]
   ordinary=[json.loads(v) for v in (R/f'moment/{prefix}_{stage}_m0_s{stall}.jsonl').read_text().splitlines()]
   assert len(c)==len(general)==len(ordinary)==2*len(current)
   for i,(m,g) in enumerate(zip(c,general)):
    for key in ['q1_words','q2_words','q1_issues','q2_issues','source_words','z_vector_reads','z_scalar_reads','z_writes','psum_reads','psum_writes','transform_reads','transform_writes','stripe_add_issues','cache_reads']:
     assert m[key]==g[key],(stage,consumer,stall,i,key)
    assert g['transform_issues']-m['transform_issues']==40
    assert g['reconstruction_issues']-m['reconstruction_issues']==960
    assert g['cache_writes']-m['cache_writes']==192
    assert g['exact_halves']==960 and m['exact_halves']==0
    if not stall:assert g[cycle]-m[cycle]==1192,(stage,consumer,i,'exact fixed tax')
   n=len(current);service=lambda rr:sum(x[cycle]+x['configuration_cycles']+1 for x in rr[:n])
   row=dict(stage=stage,consumer=consumer,stall=stall,tiles=n,moment3=service(c),moment_general4=service(general),moment_ordinary=service(ordinary),
    improvement_vs_same_function_general=1-service(c)/service(general),improvement_vs_same_function_ordinary=1-service(c)/service(ordinary))
   tap=[json.loads(v) for v in (R/f'native_tap/{prefix}_{stage}_m0_s{stall}.jsonl').read_text().splitlines()]
   row['native_tap_different_function']=service(tap)
   rows.append(row)
report=dict(passed=True,same_function_fixture_matches=len(same),matched_fields=['q1/q2/consumer params','source','origin','gold raw','FP32 identity','J','wide','I24','original Z'],
 exact_ready_core_saving_per_tile=1192,exact_static_configuration_saving=192,removed=dict(D_ALU_cycles=40,Q2_cache_load_cycles=192,reconstruction_ALU_cycles=960,half_operations=960),
 comparisons=rows,native_tap_note='different function and quality; displayed as a rival, not same-function speed ratio')
(H/'comparison.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,same_function_fixture_matches=len(same),fullconsumer=[r for r in rows if r['consumer'] and r['stage']!='small']),separators=(',',':')))
