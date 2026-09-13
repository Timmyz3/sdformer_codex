from pathlib import Path
import csv,json
H=Path(__file__).resolve().parent
stats=json.loads((H/'statistics.json').read_text())
rows=[{k:int(v) for k,v in row.items()} for row in csv.DictReader((H/'rtl_results.csv').open())]
assert len(rows)==30
cases=list(stats['functional_cases'].values())
for r in rows:
    s=cases[r['case']]
    expected_mode = (r['mode']==1) if r['mode']!=2 else s['auto_select_endpoint']
    assert r['selected_endpoint']==expected_mode
    assert r['event_add']==(s['endpoint_vector_add_cycles'] if expected_mode else s['direct_vector_add_cycles'])
    assert r['prefix']==(36 if expected_mode else 0)
    assert r['weight_reads']==s['live_K_groups']
    assert r['source_reads']==864*(2 if r['mode']==2 else 1)
    assert (r['clear'],r['round'],r['output_beats'],r['mismatched_beats'])==(40,40,40,0)
    if not r['stress']:
        assert r['source_stalls']==r['weight_stalls']==r['output_stalls']==0
        assert r['selection_cycles']==(865 if r['mode']==2 else 0)
summary={'scope':stats['scope'],'status':'PASS','commands':len(rows),'integer_values_compared':len(rows)*320,
         'real_rows':[r for r in rows if r['case']==0],
         'fixed_resource_contract':{'signed32_datapath_addsub_lanes':8,'state_words':320,'state_bytes':1280,
          'state_ports':'two 8-word reads and one 8-word write per cycle, reserved in every mode',
          'source_bits_per_beat':20,'weight_bits_per_beat':256,'continuous_output_bits_per_beat':192,
          'weight_register_bytes':32,'output_register_bytes':24,
          'source_mask_format':'bit p*10+t, source supplier supplies actual T10 data; D generated in RTL'},
         'same_integer_function':'sat24(RNE((sum_k S[p,t,k]*U_q16[n,k])/8)); delta prefix before RNE',
         'not_measured':['whole layer','FP32 original function equivalence','SNN/BN producer','V96/downstream consumers',
                        'capture-to-mask packing construction','DRAM physical timing','area','Fmax','energy','new accuracy']}
for stress in [0,1]:
    rr={r['mode']:r for r in rows if r['case']==0 and r['stress']==stress}
    summary[f'real_stress_{stress}']={
      'endpoint_over_direct_cycle_ratio':rr[1]['cycles']/rr[0]['cycles'],
      'direct_over_endpoint_speedup':rr[0]['cycles']/rr[1]['cycles'],
      'auto_extra_cycles_over_fixed_direct':rr[2]['cycles']-rr[0]['cycles']}
(H/'result_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
