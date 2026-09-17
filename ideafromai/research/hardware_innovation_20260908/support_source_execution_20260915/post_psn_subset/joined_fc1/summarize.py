from pathlib import Path
import json,csv
P=Path(__file__).resolve().parent
r=[json.loads(x) for x in (P/'results_all.jsonl').read_text().splitlines()]
swap=[json.loads(x) for x in (P/'results_swap.jsonl').read_text().splitlines()]
assert len(r)==444 and len(swap)==36
expected={x['case']:x for x in json.loads((P/'inputs.json').read_text())['cases']}
old={(x['case'],int(x['bp'])):x for x in csv.DictReader((P.parents[2]/'support_lut_execution_20260915/rtl_cycles.csv').open()) if x['mode']=='4'}
leaf={(x['case'],x['mode'],x['bp'],x['warm']):x for x in (json.loads(line) for line in (P.parent/'fused_datapath/results_all.jsonl').read_text().splitlines())}
checks=0
for x in r+swap:
 e=expected[x['case']];sc=x['state_cycles'];assert sum(sc)==x['cycles']
 assert x['config_words']==(0 if x['warm'] else 397)
 assert x['words']==x['config_words']+x['coeff_words']
 for k in ['updates','jobs','coeff_words']:assert x[k]==e['expected_'+k],(x['case'],k)
 assert sc[0]==1 and sc[3]>=320 and sc[10]>=1
 assert x['y_write_cycles']==x['updates']
 assert x['memory_write_checks']==x['y_write_banks']*12
 assert x['memory_read_checks']==x['y_read_cycles']*96
 assert x['y_write_banks']<=x['y_write_cycles']*8 and x['y_read_banks']<=x['y_read_cycles']*8
 assert x['source_stall']==0 and x['bank_rsp_stall']==0
 assert x['gate_checks']==30720 and x['u_checks']==(0 if x['mode']==2 else 30720)
 assert x['table_checks']==10*x['table_writes']
 assert x['psn_cycles']==sum(sc[5:10])+sum(sc[15:21])
 if x['mode']==0:
  assert x['mac']==e['expected_native_mac'] and x['planes']==x['pack_writes']==x['pack_reads']==0
  assert x['table_writes']==0 and x['y_checks']==30720
  if not x['warm']:
   b=old[x['case'],x['bp']]
   for k in ['fc_cycles','psn_cycles','boot_cycles','words','coeff_words','updates','mac','jobs','zero_jobs','peak_words','peak_desc','out_stall','bank_req_stall']:
    assert x[k]==int(b[k]),(x['case'],x['bp'],k)
   assert x['cycles']==int(b['cycles'])+sc[10]
   checks+=1
 else:
  assert x['mac']==0 and x['y_checks']==61440
  assert x['pack_writes']==sc[19]==384 and x['pack_reads']==320
  assert x['pack_read_cycles']==sc[20] and sc[20]==320+x['out_stall']
  assert x['planes']==e['expected_full_planes' if x['mode']==1 else 'expected_cert_planes']
  assert x['early_groups']==(e['expected_early_groups'] if x['mode']==2 else 0)
  assert x['exponent_checks']==384 and x['bound_checks']==(x['planes']*160 if x['mode']==2 else 0)
  assert sc[15]==32 and sc[16]==320 and sc[17]==384 and sc[18]==x['planes']
  assert x['build_cycles']==(85 if x['table_writes'] else 0)
  if x['series']=='all':
   b=leaf[x['case'],x['mode']-1,x['bp'],x['warm']]
   assert x['planes']==b['planes'] and x['early_groups']==b['early_groups']
   if not x['bp']:assert x['psn_cycles']==b['psn_service']+320
  if not x['bp']:assert x['psn_cycles']==x['planes']+1440
# All three algorithms enter the same FC1 with identical configuration/source timing.
for case in expected:
 for bp in [0,1]:
  for warm in [0,1]:
   z=[x for x in r if x['case']==case and x['bp']==bp and x['warm']==warm]
   assert len(z)==3
   for k in ['fc_cycles','boot_cycles','words','coeff_words','config_words','updates','jobs','zero_jobs','peak_words','peak_desc','bank_req_stall','y_write_banks','y_write_cycles']:
    assert len({x[k] for x in z})==1,(case,bp,warm,k)
# Native cold -> full warm must actually build a table, after that warm transitions must retain it.
for j in range(0,36,6):
 z=swap[j:j+6];assert [x['mode'] for x in z]==[0,1,2,0,1,2]
 assert [x['warm'] for x in z]==[0,1,1,1,0,1]
 assert [x['table_writes'] for x in z]==[0,64,0,0,64,0]
aggregate=[]
keys=['cycles','source_to_gate','fc_cycles','psn_cycles','build_cycles','boot_cycles','words','coeff_words','config_words','updates','mac','jobs','zero_jobs','planes','early_groups','table_writes','pack_writes','pack_reads','pack_read_cycles','y_read_cycles','y_write_cycles','y_read_banks','y_write_banks','out_stall','bank_req_stall','bank_rsp_stall','source_stall']
for bp in [0,1]:
 for warm in [0,1]:
  for mode in range(3):
   z=[x for x in r if x['real'] and x['bp']==bp and x['warm']==warm and x['mode']==mode];assert len(z)==32
   a={k:sum(x[k] for x in z) for k in keys};a.update(bp=bp,warm=warm,mode=mode,commands=32)
   a['state_cycles']=[sum(x['state_cycles'][i] for x in z) for i in range(21)]
   a['source_bytes']=32*320*12;a['output_bytes']=32*320*12;a['config_bytes']=16*a['config_words'];a['coeff_bytes']=16*a['coeff_words']
   a['Y_bank_read_bytes_including_held_access_cycles']=36*a['y_read_banks'];a['Y_bank_write_bytes']=36*a['y_write_banks']
   a['pack_write_bytes']=10*a['pack_writes'];a['pack_read_accepted_bytes']=12*a['pack_reads'];a['pack_read_held_cycles_bytes']=12*a['pack_read_cycles']
   aggregate.append(a)
comparisons=[]
for bp in [0,1]:
 for warm in [0,1]:
  n,f,c=[next(x for x in aggregate if x['mode']==m and x['bp']==bp and x['warm']==warm) for m in range(3)]
  comparisons.append({'bp':bp,'warm':warm,'native_cycles':n['cycles'],'full_cycles':f['cycles'],'cert_cycles':c['cycles'],'cert_saving_cycles':n['cycles']-c['cycles'],'cert_saving_fraction':1-c['cycles']/n['cycles'],'clock_break_even_f_cert_over_f_native':c['cycles']/n['cycles'],'same_union_cert_vs_full_saving':1-c['cycles']/f['cycles'],'cert_vs_native_psn_saving':1-c['psn_cycles']/n['psn_cycles']})
coverage={k:sum(x[k] for x in r+swap) for k in ['gate_checks','y_checks','u_checks','bound_checks','exponent_checks','table_checks','tail_checks','tail_even_checks','memory_read_checks','memory_write_checks']}
summary={'status':'PASS','main_commands':444,'lifecycle_commands':36,'reported_commands':480,'main_cases':37,'real_cases':32,'independent_real_sources':8,'diagnostics':5,'native_old_mode4_records_matched_including_lifecycle':checks,'coverage':coverage,'aggregate_real':aggregate,'comparisons':comparisons,
 'resources':{'Y_array_copies':1,'Y_bytes':92160,'Y_valid_bytes':320,'Y_row_bits':2304,'Y_read_ports':1,'Y_write_banks':8,'Y_write_bits_per_bank':288,'FC1_payload_words':24,'FC1_descriptors':4,'memory_banks':8,'memory_word_bits':128,'outstanding_per_bank':1,'external_memory_capacity_bytes':131072,'shared_first_addsub48_positions':96,'fused_second_prefix48_positions':80,'fused_bound48_positions':160,'fused_tail48_positions':20,'union_addsub48_positions':356,'native_multipliers_16x24':96,'fused_bound_comparison_lanes':160,'native_gate_comparison_lanes_retained':96,'fused_lut_bytes':1280,'fused_lut_16bit_32to1_muxes':160,'fused_lut_copies':1,'gatepack_bytes':120,'fused_ybuf_bytes':2880,'native_yhold_retained_bytes':288,'native_U_retained_bytes':5760,'fused_v_bytes':480,'fused_P_N_tail_bytes':240,'fused_gate_locked_bytes':20,'fused_exponents_bits':60,'fused_added_data_arrays_excluding_exponents_control_bytes':5020,'A_copies':1,'A_bytes':200,'tau_copies':1,'tau_bytes':5760,'flags_bytes':144,'cold_A_D_tau_flags_class_words':[13,12,360,9,3]},
 'scope':'Actual common mode4 FC1 input g-prime -> sole persistent Y -> native/full/cert -> identical per-t H96 gate interface. No source classifier or GPU/EDA/PPA claim.',
 'limits':['Warm requires unchanged model A/D/W-derived class/tau/flags and hblock; hardware validates only config_valid and hblock, host must cold-load on model change.','Native/full/cert use one union RTL with all retained arithmetic/state. This is not a standalone native vs fused equal-area comparison.','Gate output is common; native U and fused full group U are verification monitors, cert has no full-U contract.','The long fused combinational path remains unmapped. Clock thresholds are break-even conditions, not measured Fmax.','Only32real=8source tilesx4 H96 functions; no additional quality/network claim.']}
(P/'SUMMARY.json').write_text(json.dumps(summary,separators=(',',':'))+'\n')
print(json.dumps({'status':'PASS','commands':480,'coverage':coverage,'comparisons':comparisons},separators=(',',':')))
