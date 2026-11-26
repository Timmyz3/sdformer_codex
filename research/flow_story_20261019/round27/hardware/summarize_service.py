#!/usr/bin/env python3
"""Read the resource result; no simulation/GT/model calls or parameter choice."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
d=json.loads((HERE/'UNIFIED_SERVICE_V1.json').read_text())
out=HERE/'SERVICE_SUMMARY_V1.json'
assert not out.exists()
rows=[]
for c in d['comparisons']:
 rows.append(dict(scope=c['scope'],baseline=c['baseline'],PE_arithmetic_reduction_percent=c['PE_arithmetic_ticks']['reduction_fraction']*100,
  PE_reserved_reduction_percent=c['PE_reserved_ticks']['reduction_fraction']*100,
  summed_service_reduction_percent=c['ticks']['reduction_fraction']*100,
  additional_scorer_arithmetic_budget_for10=c['PE_arithmetic_ticks']['incremental_scorer_budget_for_10percent']))
base=next(x for x in d['results'] if x['scope']=='MLP12_patch4' and x['implementation']=='Q8_strong')
candidate=next(x for x in d['results'] if x['scope']=='MLP12_patch4' and x['implementation']=='Q6_serial')
F8=sum(r['matrix_PE_ticks'] for r in base['records'] if r['category']=='affine' and r['selected'])
F6=sum(r['matrix_PE_ticks'] for r in candidate['records'] if r['category']=='affine' and r['selected'])
C8=base['summary']['PE_arithmetic_ticks']-F8;C6=candidate['summary']['PE_arithmetic_ticks']-F6
small=[]
for label,select,n in [('PSN_T2',lambda r:r['category']=='PSN' and r['T']==2,2),('PSN_T10',lambda r:r['category']=='PSN' and r['T']==10,10),('prediction_Cout2',lambda r:r['category']=='affine' and r['name'].startswith('pred'),2)]:
 rr=[r for r in base['records'] if select(r)];ticks=sum(r['matrix_PE_ticks'] for r in rr)
 small.append(dict(group=label,operators=len(rr),output_lanes_used=n,output_lanes_reserved=16,empty_output_lane_fraction=1-n/16,
  primary_matrix_PE_ticks=ticks,fraction_of_whole_PE_arithmetic=ticks/base['summary']['PE_arithmetic_ticks'],
  exact_avoidable_cycles_not_established=True))
sensitivity=[]
for x in d['sensitivity']:
 b=next(r for r in x['results'] if r['implementation']=='Q8_strong')['summary'];c=next(r for r in x['results'] if r['implementation']=='Q6_serial')['summary']
 sensitivity.append(dict(label=x['label'],PE_arithmetic_reduction_percent=100*(1-c['PE_arithmetic_ticks']/b['PE_arithmetic_ticks']),
  PE_reserved_reduction_percent=100*(1-c['PE_reserved_ticks']/b['PE_reserved_ticks']),summed_service_reduction_percent=100*(1-c['ticks']/b['ticks']),
  additional_scorer_arithmetic_budget_for10=.9*b['PE_arithmetic_ticks']-c['PE_arithmetic_ticks']))
r=dict(schema='flow_r27_service_scope_summary_v1',comparisons=rows,sensitivity=sensitivity,smallN_common_occupancy=small,
  rho_decomposition=dict(F8_selected_integer_matrix_ticks=F8,F6_selected_integer_matrix_ticks=F6,C8_all_other_PE_ticks=C8,C6_all_other_PE_ticks=C6,
   equation='0.9*(rho*F8 + C8) - (rho*F6 + C6) >= A',rho_min_no_scorer=(C6-.9*C8)/(.9*F8-F6),
   definition='rho scales BOTH selected integer matrix costs relative to all other PE work, holding per-layer implementation ordering and the declared dot decomposition fixed. It is not a measured technology ratio, nor a per-layer utilization-independent law.'),
  default_selected_baseline_modes={mode:sum(r.get('mode')==mode and r['category']=='affine' and r['selected'] for r in base['records']) for mode in ('int8_parallel','bitserial')},
  quality_read=False,quality_qualification_not_implied=True,whole_net_hardware_speedup_measured=False)
out.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
