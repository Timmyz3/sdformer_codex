"""Aggregate the final replay; every rate uses a named same-condition base."""
import csv,json
from pathlib import Path
from collections import defaultdict,Counter

HERE=Path(__file__).resolve().parent
rows=list(csv.DictReader((HERE/'source_cycles.csv').open()))
keys=('order','mode','bp','packed32','prefetch')
fields=('cycles','channels','scalar_mac','words','xwords','graph_words','prefetch_words','graph_hits','req_stall','out_stall')
aggregate=defaultdict(Counter)
for row in rows:
    if int(row['real']):
        key=tuple(int(row[k]) for k in keys)
        aggregate[key].update({k:int(row[k]) for k in fields+tuple('state'+str(s) for s in range(14))})
out=[]
for key,totals in aggregate.items():
    base=aggregate[key[0],1,key[2],1,1]
    out.append(dict(zip(keys,key))|dict(totals)|{
        'cycle_reduction_vs_static64_with_X_prefetch':1-totals['cycles']/base['cycles'],
        'word_change_vs_static64_with_X_prefetch':totals['words']/base['words']-1,
        'channel_reduction_vs_static64':1-totals['channels']/base['channels']})
summary={'tasks':len(rows),'real_source_packets':64,'real_frames':2,'real_partition':'training cache, fixed sampled P32/frame, not contiguous tile',
         'real_replays':sum(int(r['real']) for r in rows),'outputs_checked':len(rows)*60,
         'computed_producer_U_and_gate_checks_each':sum(int(r['channels'])*10 for r in rows),
         'metrics_are':'Verilator component cycles, not admitted VCS/DC/PT/FM RTL speedup or network AEE',
         'aggregates':out}
(HERE/'source_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines=['| 源执行 | ready周期 | 背压周期 | 实际源任务 | 128bit请求 ready/BP |',
       '|---|---:|---:|---:|---:|']
for label,m,fmt,pf in [('全96+下一X预取',0,1,1),('静态64+下一X预取',1,1,1),
                     ('普通code图64bit',2,0,0),('普通code图32bit',2,1,0),
                     ('普通code图32bit+预取',2,1,1),('响应class图32bit+预取',3,1,1)]:
    a=aggregate[1,m,0,fmt,pf];b=aggregate[1,m,1,fmt,pf]
    lines.append(f'| {label} | {a["cycles"]:,} | {b["cycles"]:,} | {a["channels"]:,} | {a["words"]:,} / {b["words"]:,} |')
(HERE/'source_table.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
print('tasks',summary['tasks'],'checked output labels',summary['outputs_checked'],'checked computed U/gate each',summary['computed_producer_U_and_gate_checks_each'])
