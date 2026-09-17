from pathlib import Path
import json
P=Path(__file__).resolve().parent
read=lambda name:[json.loads(x) for x in (P/name).read_text().splitlines()]
main=read('results_all.jsonl');swap=read('results_swap.jsonl');diag=read('results_diagnostics.jsonl')
assert len(main)==444 and len(swap)==36 and len(diag)==16
old_summary=json.loads((P.parent/'joined_fc1/SUMMARY.json').read_text())
compared=0
for name,rows in [('results_all.jsonl',main),('results_swap.jsonl',swap)]:
 old=[json.loads(x) for x in (P.parent/'joined_fc1'/name).read_text().splitlines()]
 for a,b in zip(old,rows):
  for k,v in a.items():
   if k not in ['bound_checks','tail_checks','tail_even_checks']:assert b[k]==v,(name,a['case'],a['mode'],k,v,b.get(k))
  assert b['normalized_value_checks']==a['bound_checks']
  assert b['threshold_checks']==2*b['normalized_value_checks']
  assert b['normalized_predicate_checks']==b['normalized_value_checks']
  compared+=1
assert compared==480
info=json.loads((P/'diagnostic_inputs.json').read_text());dexp={x['case']:x for x in info['diag_cases']}
old_diag=[json.loads(x) for x in (P.parent/'fused_datapath/results_fullrange.jsonl').read_text().splitlines()]
for r in diag:
 e=dexp[r['case']];assert not r['real']
 assert r['planes']==e['expected_cert_planes' if r['mode'] else 'expected_full_planes']
 assert r['early_groups']==(e['expected_early_groups'] if r['mode'] else 0)
 assert r['normalized_value_checks']==(r['planes']*160 if r['mode'] else 0)
 assert r['threshold_checks']==2*r['normalized_value_checks'] and r['normalized_predicate_checks']==r['normalized_value_checks']
 assert r['gate_checks']==r['y_checks']==30720 and r['u_checks']==(0 if r['mode'] else 30720)
 assert sum(r['state_cycles'])==r['cycles'] and sum(r['state_cycles'][8:20])==r['psn_service']
 if r['case']=='diagnostic_signed24_fullrange':
  a=next(x for x in old_diag if all(x[k]==r[k] for k in ['mode','bp','warm']))
  for k,v in a.items():
   if k!='bound_checks':assert r[k]==v,(k,r[k],v)
coverage={k:sum(r.get(k,0) for r in main+swap+diag) for k in ['gate_checks','y_checks','u_checks','normalized_value_checks','threshold_checks','normalized_predicate_checks','exponent_checks','table_checks','memory_read_checks','memory_write_checks']}
coverage['table_checks']+=sum(x['table_rows_written']*10 for x in diag)
resources=dict(old_summary['resources'])
resources.pop('fused_bound48_positions');resources.pop('fused_tail48_positions');resources.pop('fused_P_N_tail_bytes')
resources.update(fused_bound49_positions=160,tail_addsub_positions=0,union_addsub_positions=336,union_addsub48_positions=176,union_addsub49_positions=160,normalized_q_bytes=980,fused_P_N_minus_one_bytes=120,removed_tail_bytes=120,data_bytes_delta_over_joined=860,fused_added_data_arrays_excluding_exponents_control_bytes=5880,removed_nv_variable_left_shift_lanes=80,removed_tail_variable_left_shift_lanes=20,new_q_variable_arithmetic_right_shift_lanes=160,new_q_shift_width=49)
# Do not preserve the old field implying all union positions are 48bit.
resources.pop('union_addsub48_positions',None);resources['union_addsub48_positions']=176
summary={'status':'PASS','main_commands':444,'lifecycle_commands':36,'Y24_diagnostic_commands':16,'reported_commands':496,'same_cycle_records_vs_joined':480,'same_cycle_original_Y24_records':8,'real_aggregate_unchanged':old_summary['aggregate_real'],'comparisons_unchanged':old_summary['comparisons'],'coverage':coverage,'resources':resources,'diagnostic_inputs':info,
 'scope':'Same actual FC1/sole Y/H96 gate interfaces as joined_fc1. Arithmetic-normalized certificate has exactly identical per-command cycles and service counts; this is a combinational-path/state change only.',
 'normalized_arithmetic':'P/N register contents are P-1/N-1. GROUP q=tau+stored+(1-positive); PLANE sum=nv+stored+1; signed49 sums, stored q, and floor right shifts. No old nv variable shift or tail arithmetic/monitors.',
 'limits':['No mapped area, Fmax, energy, or end-to-end physical speed assertion; fixed cold-ready clock break-even remains95.683484%.','Native96 comparison paths and96 multipliers remain in the common union;336 add/sub positions mix176x48bit and160x49bit.','16049bit right shifters replace80nv48bit left shifters and20tail48bit initialization shifters; fewer ALU positions is not an area proof.','The16diagnostic commands are direct Y input into a standalone leaf using exactly the same normalized_bound module, not claimed FC1-attainable extreme data.','Source classifier and hardware model-version validation remain outside current interface; warm retains unchanged-model host contract.']}
if (P/'comb_split_check.json').exists():summary['comb_split_verification']=json.loads((P/'comb_split_check.json').read_text())
(P/'SUMMARY.json').write_text(json.dumps(summary,separators=(',',':'))+'\n')
print(json.dumps({'status':'PASS','commands':496,'unchanged_cycle_records':480,'coverage':coverage},separators=(',',':')))
