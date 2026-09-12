from pathlib import Path
from collections import defaultdict
import csv
import json
import statistics

HERE = Path(__file__).resolve().parent
source = json.loads((HERE/'results.json').read_text())
rows = source['rows']; groups = defaultdict(list)
for row in rows:
    if not row['real']: continue
    group = 'trained_2of4' if 'row_2of4' in row['name'] else 'trained_C16' if 'C16' in row['name'] else 'original_integer'
    route = 'time' if row['temporal'] else 'class'
    groups[group, row['flive'], route, row['merge'], row['reduce']].append(row)
means = []
for (group, flive, route, merge, reduce), members in groups.items():
    mean = dict(group=group, flive=flive, route=route, merge=merge, reduce=reduce,
                actual_capture_count=len(members), service_slots_mean=statistics.mean(x['service_slots'] for x in members))
    for key in ('source_R64_requests', 'physical_W8_reads', 'NR4_packets', 'NR4_admission_stalls',
                'W_arbiter_denied_requests', 'selected_events', 'S_writes', 'program_issues', 'issue_idle_slots'):
        mean[key+'_mean'] = statistics.mean(x['counts'].get(key, 0) for x in members)
    mean['S_valid_peak'] = max(x['counts'].get('S_valid_peak', 0) for x in members)
    means.append(mean)
comparisons = []
for group in sorted({x['group'] for x in means}):
    f1 = min((x for x in means if x['group'] == group and x['flive'] == 1), key=lambda x:x['service_slots_mean'])
    f2 = min((x for x in means if x['group'] == group and x['flive'] == 2), key=lambda x:x['service_slots_mean'])
    comparisons.append(dict(group=group, strongest_tested_F1=f1, strongest_tested_F2=f2,
        F2_over_F1=f2['service_slots_mean']/f1['service_slots_mean'],
        F2_service_reduction=1-f2['service_slots_mean']/f1['service_slots_mean']))
pairs = defaultdict(dict)
for x in rows: pairs[x['name'],x['merge'],x['reduce']][x['flive']] = x
pair_counts = defaultdict(lambda:dict(total=0,F2_faster=0,equal=0,F2_slower=0))
for pair in pairs.values():
    x,y = pair[1],pair[2]; c = pair_counts['actual' if x['real'] else 'directed']
    c['total'] += 1
    c['F2_faster' if y['service_slots']<x['service_slots'] else 'equal' if y['service_slots']==x['service_slots'] else 'F2_slower'] += 1
    for key in ('NR4_packets','selected_events','S_writes','logical_W_values','program_issues'):
        assert x['counts'].get(key,0)==y['counts'].get(key,0)
out = dict(total_runs=len(rows),actual_runs=sum(x['real'] for x in rows),directed_runs=sum(not x['real'] for x in rows),
    actual_distinct_capture_selections=len({x['name'].replace('_packed_class','').replace('_packed_time','') for x in rows if x['real']}),
    actual_representation_cases=len({x['name'] for x in rows if x['real']}),
    S_values_checked=sum(x['S_values_checked'] for x in rows),gate_bits_checked=sum(x['gate_bits_checked'] for x in rows),
    differences=sum(x['differences'] for x in rows),NR4_peak=max(x['counts'].get('NR4_peak',0) for x in rows),
    S_valid_peak=max(x['counts'].get('S_valid_peak',0) for x in rows),
    ordinary_restoration_runs=sum(x['tail']=='ordinary_inplace_restore_then_A' for x in rows),
    comparisons=comparisons,matched_configuration_pair_counts=dict(pair_counts),means=means,
    scope='One-tile4PE, two real output rows, fullC384 andT10. CPU start-to-final-gate service; common56S/twoNR4, same private/merge and scalar/member controls; ordinary existing restoration tail given to both. No full-layer/Gustav/RTL/PPA/newAEE claim.')
(HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
with (HERE/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(means[0]));w.writeheader();w.writerows(means)
print(json.dumps({k:v for k,v in out.items() if k!='means'},indent=2))
