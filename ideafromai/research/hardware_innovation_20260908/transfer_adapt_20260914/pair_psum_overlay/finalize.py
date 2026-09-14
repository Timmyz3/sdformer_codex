from pathlib import Path
import json
H=Path(__file__).resolve().parent
checks=[json.loads((H/name).read_text()) for name in ['verification.json','stream_checks_small.json','stream_checks_64.json','reconfigure_checks.json']]
assert all(r['passed'] for r in checks)
results=[]
for name in ['results.json','results_stream_small.json','results_stream_64.json','results_reconfigure.json']:results+=json.loads((H/name).read_text())
count={m:sum(r['mode']==m for r in results) for m in [14,19,20]}
summary=json.loads((H/'SUMMARY_stream_64.json').read_text())
a,b,c=[summary[f'm{m}_s0_repeat0'] for m in [14,19,20]]
stream=json.loads((H/'results_stream_64.json').read_text())
retire_blocks={m:sum(r['state_cycles'][26] for r in stream if r['mode']==m and r['stall']==0 and r['command']<64) for m in [19,20]}
assert b['cycles']-c['cycles']==2*(b['count_checks']-c['count_checks'])+2*(retire_blocks[19]-retire_blocks[20])
resources={
 'scope':'one-context K864/R8/N96/P4/T10 raw p; same new module14 native2P and20 overlay, prior19 separately rerun unchanged',
 'compute':{'wide_data_ALUs':8,'ALU_bits':32,'count_carry_cuts':[8,16,24],'native_z_cut':[13],'signed_multiplier_expressions':8,'multiplier':'19x13 ->32','packed_retire':'original a+2048b times signed3; a<=255,b<=127; sign/zero carry-in correction atbit13; otherwise scalar retire'},
 'shared_existing_arrays':{
  'psum':{'banks':8,'rows':480,'word_bits':32,'bytes':15360,'output_port_bits':256,'one_address_per_bank_per_cycle':True,'one_read_or_write_per_bank_per_cycle':True,'read_expressions_per_bank':1,'write_expressions_per_bank':1},
  'z':{'banks':8,'rows':20,'word_bits':26,'bytes':520,'vector_port_bits':208},
  'source':{'rows':1536,'bits':10,'bytes':1920},'native_window':{'bytes':20},'Q1':{'bytes':2592,'port_bits':24},'Q2':{'bytes':1536,'port_bits':128},'qblock':{'bytes':128}},
 'phase_overlay':{'independent_count_array_bytes':0,'removed_prior_count_array_bytes':6400,'count_region_existing_psum_bytes':5120,'rows_per_bank':160,'count_bytes_per_word':4,'rank_groups':4,'banks_per_group':2,'rows_per_class':5,'mapping':'row=class_slot*5+time_pair;bank=2*rank_group+time_in_pair;byte=P0/P1/P2/P3','lifetime':'after start until complete G retirement; then Q2 overwrites all480 rows before drain; valid bits clear every start','bank_address_change':'Four independently selected class addresses during count RMW, common group row during retire, common output row later. Per-bank address/control mux added; no simultaneous count+output service.'},
 'metadata_and_holds':{'class_bytes':2592,'representative_bytes':96,'count_live_bits':640,'prior_count_live_bits':1280,'count_hold_bits':256,'prior_count_hold_bits':160,'count_hold_extra_bytes':12,'other':'original k_live/v_live/support/z_hold/output holding;5bit block pending and8bit retirement pending;control mux/priority logic remains real'},
 'configuration':{'native_first_command':3361,'overlay_first_command':4258,'extra_static_class_beats':897,'stream_static_native':1824,'stream_static_overlay':2721,'source_origin_per_tile':1537,'start_per_tile':1,'class_contract':'Nonzero class frequency2..255;larger frequency or unselected pair ->direct63;static Q1-only compilation, no source-based admission'},
 '64_count_traffic':{'prior19_bank_reads':b['count_bank_reads'],'overlay20_bank_reads':c['count_bank_reads'],'prior19_read_bytes':b['count_bank_reads']*20//8,'overlay20_read_bytes':c['count_bank_reads']*4,'writes_equal_reads_for_this_stream':True,'note':'Fewer transactions, but wider bank words increase transferred count bits. No energy claim.'},
 'not_established':['equal area after synthesis','equal-area optimized resource allocation against416bit z threeP/fourP (later different-resource-point service results are in temporal_pairing/README.md)','I24 or RR integration','physical single-port SRAM timing','Fmax,energy,full-frame/network speedup'],
 'verification_only':'Verilator-only range/overflow/read-write/lifecycle/overwrite assertions and check_retired/check_stores are not candidate data resources'
}
(H/'resource_contract.json').write_text(json.dumps(resources,indent=2)+'\n')
final=dict(passed=True,rtl_commands=len(results),rtl_raw_outputs=sum(r['outputs'] for r in results),commands_by_mode=count,new_module14_and20_commands=count[14]+count[20],unchanged_old19_reference_commands=count[19],fixture_commands=checks[0]['rtl_commands'],small_stream_commands=checks[1]['rtl_commands'],stream64_commands=checks[2]['rtl_commands'],reconfiguration_commands=checks[3]['commands'],stream64_cold_service={str(m):summary[f'm{m}_s0_repeat0']['service_cycles'] for m in [14,19,20]},native_service_saved=a['service_cycles']-c['service_cycles'],native_service_reduction_percent=100*(a['service_cycles']-c['service_cycles'])/a['service_cycles'],prior19_service_saved=b['service_cycles']-c['service_cycles'],prior19_delta_decomposition=dict(count_update_transactions_removed=b['count_checks']-c['count_checks'],retire_read_blocks_removed=retire_blocks[19]-retire_blocks[20],cycles_per_removed_transaction=2,retire_arithmetic_issues_unchanged=c['aux_events']),remaining_scope='native2P/208bit z denominator only; no I24/RR/EDA/training/production/commit/hash')
(H/'final_checks.json').write_text(json.dumps(final,indent=2)+'\n')
print(json.dumps(final,indent=2))
