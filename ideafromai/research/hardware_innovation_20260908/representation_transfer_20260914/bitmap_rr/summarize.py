from pathlib import Path
import json
H=Path(__file__).resolve().parent
keys=['total_cycles','window_cycles','core_cycles','core_source_stalls','core_weight_stalls','core_output_stalls','core_arbitration_stalls','core_z_vector_reads','core_z_scalar_reads','core_z_writes','core_bitmap_hold_reads','core_bitmap_hold_writes','core_bitmap_row_z_arbitration_stalls','core_bitmap_z_arbitration_stalls','core_bitmap_alu_arbitration_stalls','core_bitmap_weight_arbitration_stalls','shared_z_grants','core_first_issues','core_cache_reads','core_cache_writes','core_aux_reads','core_aux_issues','core_aux_weight_words','core_bitmap_native_reads','consumer_join_wait_cycles','conflict_cycles','both_compute_cycles','core_weight_words','shared_alu_grants']
out={}
for stage in ['held','disjoint']:
 rows=[json.loads(l) for l in (H/f'results_{stage}.jsonl').read_text().splitlines()]
 cold={r['mode']:r for r in rows if r['command']==r['stall']==0}
 delta={k:cold[7][k]-cold[8][k] for k in keys}
 assert delta['core_cycles']==-cold[7]['core_bitmap_hold_writes']+delta['core_arbitration_stalls']+delta['core_output_stalls']
 assert delta['total_cycles']==delta['window_cycles']==delta['core_cycles']//2
 gains=[]
 for ctrl in [2,4,21,8]:
  for stall,command in [(0,0),(0,1),(1,0),(1,1)]:
   a=next(r['total_cycles'] for r in rows if (r['mode'],r['stall'],r['command'])==(ctrl,stall,command))
   b=next(r['total_cycles'] for r in rows if (r['mode'],r['stall'],r['command'])==(7,stall,command))
   gains.append(dict(control=ctrl,stall=stall,command=command,saved_cycles=a-b,saved_percent=100*(a-b)/a))
 out[stage]=dict(cold={m:{k:r[k] for k in keys} for m,r in cold.items()},delta8to7=delta,mode7_gains=gains,cycles=[{k:r[k] for k in ['mode','stall','command','total_cycles']} for r in rows])
(H/'comparison.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
print(json.dumps({k:dict(mode7_cold=v['cold'][7]['total_cycles'],saved_vs8=-v['delta8to7']['total_cycles']) for k,v in out.items()}))
