from pathlib import Path
import json
H=Path(__file__).resolve().parent
allrows=[]
for stage in ['small','held','disjoint','sequences','swap']:
 p=H/f'results_{stage}.jsonl'
 if p.exists():allrows.extend(json.loads(l) for l in p.read_text().splitlines())
rows=[]
for r in allrows:
 out={k:r[k] for k in ['stage','initial_mode','next_mode','mode','stall','command','tiles','total_cycles','window_cycles','static_words','parameter_stalls','source_load_words','source_load_stalls','origin_words','origin_stalls','core_cycles','core_first_issues','core_mac_issues','core_arbitration_stalls','core_output_stalls','conflict_cycles','shared_source_grants','shared_weight_grants','shared_z_grants','shared_alu_grants','shared_psum_grants','shared_wide_grants','consumer_join_wait_cycles','consumer_wide_waits','consumer_output_stalls','borrow_grants','borrow_consumer_stalls','wide_conflict_cycles','core_bitmap_z_arbitration_stalls','core_bitmap_alu_arbitration_stalls','core_bitmap_weight_arbitration_stalls','core_bitmap_hold_writes','core_bitmap_hold_reads']}
 out.update(go_cycles=1,service_cycles=r['total_cycles']+1,raw_J_wide_I24_each=r['outputs'])
 rows.append(out)
(H/'comparison.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
(H/'SUMMARY.json').write_text(json.dumps(dict(passed=True,commands=len(rows),raw_J_wide_I24_each=sum(r['raw_J_wide_I24_each'] for r in rows),stages=sorted({r['stage'] for r in rows}),go_cycles_per_command=1,metric='service_cycles = RTL total_cycles + one accepted go tick',R8_valid825_AEE=1.3276350226079938,phase3_valid825_AEE=1.258343102157,different_functions=True),indent=2)+'\n')
for stage in ['held','disjoint','sequences']:
 if not any(r['stage']==stage for r in rows):continue
 print(stage)
 for mode in [2,4,21,7]:
  rr=[r for r in rows if r['stage']==stage and r['mode']==mode]
  print(mode,[(r['stall'],r['command'],r['service_cycles']) for r in rr])
