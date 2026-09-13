"""Independent traffic checks from convolution geometry, not an FSM simulator."""
from pathlib import Path
from collections import defaultdict
import json

HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'native_sparse/control_results.json').read_text())
errors=[]
ones={r['mode']:r for r in rows if r['fixture']=='one' and r['command']==0 and r['stall']==0}
# 12 N8 groups, 48 source-channel pairs, nine taps, four spatial outputs.
contexts=12*48*9*4
updates=contexts*10
expected={
    0:dict(source_words=2*contexts,weight_words=2*contexts,sum_issues=contexts),
    1:dict(source_words=96*4*4,weight_words=2*contexts,sum_issues=contexts),
    2:dict(source_words=2*contexts,weight_words=2*contexts//4,sum_issues=contexts//4),
}
for mode,values in expected.items():
    values.update(update_issues=updates,psum_reads=updates+480,psum_writes=updates+480)
    for key,value in values.items():
        if ones[mode][key]!=value: errors.append([mode,key,ones[mode][key],value])
groups=defaultdict(list)
for row in rows:groups[(row['fixture'],row['mode'])].append(row)
for key,group in groups.items():
    # Backpressure can change elapsed cycles, but not the actual demanded work.
    for field in ('source_words','weight_words','sum_issues','update_issues','outputs','psum_reads','psum_writes'):
        if len({r[field] for r in group})!=1:errors.append([key,'unstable_work',field])
    active={r['cycles']-r['source_stalls']-r['weight_stalls']-r['output_stalls'] for r in group}
    if len(active)!=1:errors.append([key,'unexplained_cycle_delta',sorted(active)])
result=dict(scope='Analytic all-one traffic and all control backpressure/restart invariance; not a rerun of RTL.',
            controls=len(rows),outputs=sum(r['outputs'] for r in rows),
            all_one_expected=expected,errors=errors)
(HERE/'native_counter_review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if errors:raise SystemExit(1)
