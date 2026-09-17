#!/opt/anaconda3/bin/python3.12
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
rr=[json.loads(x) for x in (HERE/'results_all.jsonl').read_text().splitlines()]
parent=[json.loads(x) for x in (HERE.parent/'results_all.jsonl').read_text().splitlines()]
lookup={(x['case'],x['mode'],x['bp'],x['warm']):x for x in parent}
assert len(rr)==296
for r in rr:
 b=lookup[(r['case'],r['mode'],r['bp'],r['warm'])]
 for k in ['planes','early_groups','param_words','y_rows_written','y_rows_read','table_rows_written','gate_checks','u_checks','y_checks','exponent_checks']:
  assert r[k]==b[k],(r['case'],k,r[k],b[k])
 sc=r['state_cycles'];assert sum(sc)==r['cycles'] and sum(sc[8:20])==r['psn_service']
 for i in [11,12,14,15,16,17,18]:assert sc[i]==0
 assert sc[8]==32 and sc[9]==320 and sc[10]==384 and sc[13]==r['planes']
 for i in [1,2,3,4,5,6,7]:assert sc[i]==b['state_cycles'][i],(r['case'],'loading changed',i)
 assert r['bound_checks']==(r['planes']*160 if r['mode'] else 0)
 if not r['bp']:assert r['psn_service']==r['planes']+384*2+32+320
 # Parent independent S/W dot and certificate traversal established these group depths.
 assert r['gate_checks']==r['y_checks']==30720 and r['u_checks']==(30720 if not r['mode'] else 0)
agg=[]
for mode in [0,1]:
 for bp in [0,1]:
  for warm in [0,1]:
   selected=[r for r in rr if r['real'] and r['mode']==mode and r['bp']==bp and r['warm']==warm]
   assert len(selected)==32
   out={k:sum(r[k] for r in selected) for k in ['cycles','psn_service','param_words','y_rows_written','y_rows_read','table_rows_written','planes','early_groups']}
   states=[sum(r['state_cycles'][i] for r in selected) for i in range(21)]
   out.update(mode=mode,bp=bp,warm=warm,commands=32,cycles_plus_go=out['cycles']+32,state_cycles=states,
     group_plus_plane_cycles=states[10]+states[13],Y_read_plus_Pinit=states[8]+states[9],output_cycles=states[19],cold_setup=states[1]+states[2]+sum(states[3:7]))
   agg.append(out)
summary={'status':'PASS','commands':len(rr),'cases':37,'real_cases':32,'scope':'Same post-Y integer workload and resource-budget-equal full/cert on the NEW fused datapath. Old80ALU/Claude leaf/native96MAC are different-resource/interface references, not ASIC speed comparisons.',
 'coverage':{k:sum(r[k] for r in rr) for k in ['gate_checks','u_checks','y_checks','bound_checks','exponent_checks']},
 'tail_values_checked':sum(r['planes']*20 for r in rr if r['mode']),
 'tail_even_differences_checked':sum((r['planes']-(384-r['early_groups']))*20 for r in rr if r['mode']),
 'table_values_checked':sum(r['table_rows_written']*10 for r in rr),
 'real_aggregate':agg,
 'native_reference':json.loads((HERE.parent/'SUMMARY.json').read_text())['native_reference'],
 'resources':{'prefix_add_sub_48_positions':160,'parallel_bound_add_48_positions':160,'tail_subtract_48_positions':20,'total_explicit_add_sub_48_positions':340,'multipliers':0,'bound_signed_comparison_lanes':160,'equality_lanes':160,'Y_persistent_bytes':92160,'Ybuf_bytes':2880,'lut_bytes':1280,'lut_data_copies':1,'lut_read_muxes_16bit_32to1':160,'A_bytes':200,'tau_bytes':5760,'flags_bytes':144,'v_bytes':480,'P_N_tail_bytes':240,'locked_gate_bytes':20,'removed_dot_holding_bytes':480,'removed_lower_flag_bytes':20,'data_array_bytes_excluding_exponents_control_instrumentation':103164},
 'claude_boundary':{'reference':'../../../claude_fusion_trials_20260914/t10_rtl/cert_gate_bitl.sv','source_review':'H1xT10 outputs, e/sign/planes externally supplied; initial offline LUT and tau memories; TB stops driving after locked. Not the current32 post-FC1 cases; no inherited historical cycles.','ours':'H8xT10 GROUP sign/tail initialization one clock; each actual plane one clock with registered prefix/gate; GROUP+PLANE achieves 1+planes per physical group, plus real Y/control/output/loading costs.'},
 'limits':['No timing, synthesis, power, or ASIC frequency/area assertion. LUTmux->two48add->variable shift->bound48add->comparison->alllocked is a longer combinational path than old80ALU layout.','No complete BitL import or CSE/controller search.','Full/cert share new hardware. Full has no per-plane certificate wait; model initialization still builds common P/N for retained-mode reuse.','Y CPU-computed source handoff is an explicit post-Y boundary; actual FC1 production or full downstream network not fused.','32real are8 projected-source tilesx4 H96 functions. Negative gain/constant/tie diagnostics synthetic.']}
summary['parent_reference']=json.loads((HERE.parent/'SUMMARY.json').read_text())['aggregate_real']
summary['clock_break_even']=[
 {'candidate':'fused_cert','reference':'native96MAC','calendar':'ready','candidate_cycles':97424,'reference_cycles':118344,'minimum_f_candidate_over_f_reference':97424/118344},
 {'candidate':'fused_cert','reference':'native96MAC','calendar':'BP_cold','candidate_cycles':103265,'reference_cycles':122550,'minimum_f_candidate_over_f_reference':103265/122550},
 {'candidate':'fused_full','reference':'parent80ALU_full','calendar':'ready','candidate_cycles':148195,'reference_cycles':285126,'minimum_f_candidate_over_f_reference':148195/285126},
 {'candidate':'fused_cert','reference':'parent80ALU_cert','calendar':'ready','candidate_cycles':97424,'reference_cycles':428572,'minimum_f_candidate_over_f_reference':97424/428572}]
summary['integration_not_done']={
 'boundary':'CPU source handoff of Y; actual FC1 producer is not connected',
 'output_packing':'80 gates per H8xT10 group -> original per-t H96 requires 960bits=120B or paid reuse plus actual handshakes; not implemented',
 'tau_loader':'same5760B: subset two48bit values/128bit word=480 words; native dense48bit=360 words;120-word gap must be normalized in a future common interface',
 'flags_loader':'subset10words vs native dense9words; A/tau shared-model residency must be charged once consistently'}
extra=[json.loads(x) for x in (HERE/'results_fullrange.jsonl').read_text().splitlines()]
info=json.loads((HERE/'fullrange_inputs.json').read_text())
assert len(extra)==8
for r in extra:
 assert not r['real'] and r['case']==info['case']
 assert r['planes']==info['expected_cert_planes' if r['mode'] else 'expected_full_planes']
 assert r['early_groups']==(info['expected_early_groups'] if r['mode'] else 0)
 sc=r['state_cycles'];assert sum(sc)==r['cycles'] and sum(sc[8:20])==r['psn_service']
 assert sc[8:11]==[32,320,384] and sc[13]==r['planes']
 assert all(sc[i]==0 for i in [11,12,14,15,16,17,18])
 assert r['param_words']==(490 if r['warm'] else 503)
 assert r['y_rows_written']==r['y_rows_read']==320 and r['table_rows_written']==(0 if r['warm'] else 64)
 assert r['gate_checks']==r['y_checks']==30720 and r['u_checks']==(0 if r['mode'] else 30720)
 assert r['bound_checks']==(160*r['planes'] if r['mode'] else 0) and r['exponent_checks']==384
 if not r['bp']:assert r['psn_service']==r['planes']+1120
summary['original_commands']=len(rr);summary['commands']+=len(extra);summary['cases']+=1
for k in summary['coverage']:summary['coverage'][k]+=sum(r[k] for r in extra)
summary['tail_values_checked']+=sum(r['planes']*20 for r in extra if r['mode'])
summary['tail_even_differences_checked']+=sum((r['planes']-(384-r['early_groups']))*20 for r in extra if r['mode'])
summary['table_values_checked']+=sum(r['table_rows_written']*10 for r in extra)
summary['fullrange_diagnostic']={'inputs':info,'commands':len(extra),'records':extra,'excluded_from_real_aggregate':True}
(HERE/'SUMMARY.json').write_text(json.dumps(summary,separators=(',',':'))+'\n')
print(json.dumps({'status':'PASS','commands':summary['commands'],'coverage':summary['coverage'],'real':agg},separators=(',',':')))
