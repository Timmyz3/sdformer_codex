from pathlib import Path
import json
H=Path(__file__).resolve().parent;O=H.parent;N=O.parent
stages={k:json.loads((H/f'checks_{k}.json').read_text()) for k in ['small','short','held','disjoint']}
assert all(v['passed'] for v in stages.values())
op=json.loads((H/'opportunity.json').read_text()); comparisons={}
for stage,label,file in [('held','held_128_191','SUMMARY_64.json'),('disjoint','disjoint_4000_4063','SUMMARY_disjoint.json')]:
 s=stages[stage]['summary'];strong=json.loads((N/'strong_raw_control'/file).read_text())['m2_s0_repeat0']
 for mode,arm in [(20,'original'),(21,'permuted')]:
  actual=s[f'm{mode}_s0_repeat0'];pred=op['sets'][label]['totals'][arm]
  assert actual['cycles']==pred['core_cycles'] and actual['service_cycles']==pred['cold_service']
 a=s['m21_s0_repeat0'];comparisons[stage]={}
 for label,b in [('native14',s['m14_s0_repeat0']),('overlay20',s['m20_s0_repeat0']),('raw4P_different_resource_point',strong)]:
  before=b['service_cycles'];after=a['service_cycles']
  comparisons[stage][label]=dict(before=before,after=after,saved=before-after,saved_percent=100*(before-after)/before)
small=json.loads((H/'results_small.json').read_text());real={}
for m in [14,20,21]:
 rr=[r for r in small if Path(r['fixture']).name.startswith('real_') and r['mode']==m and r['stall']==0 and r['command']==0]
 real[str(m)]=dict(core=sum(r['cycles'] for r in rr),cold_service=sum(r['cycles']+r['configuration_cycles']+1 for r in rr))
sv=(H/'decomp_core.sv').read_text()
assert 'count_mem' not in sv and sv.count('p_mem[i][p_addr[i]]')==2
res=dict(inherits='../resource_contract.json',mode20_original_files_unchanged=True,mode21_extra_state_bits=40,mode21_extra_cold_configuration_beats=1,mode21_static_configuration_beats=2722,mode21_first_fixture_configuration_beats=4259,source_time_mapping='40 output bits, each selecting one of ten local source bits; same local-source read and L_GATHER cycle',output_mapping='combinational inverse from ten configured nibbles; existing one-address-per-bank DRAIN read uses original row identity mapped to internal T',inverse_mapping_extra_state_bits=0,psum_banks=8,psum_single_read_OR_write_per_bank=True,count_extra_array_bytes=0,count_region_bytes_in_existing_psum=5120,positive_count_only=True,second_z_domain=False,z_bytes=520,z_vector_port_bits=208,ALUs='8 x32, same cut8/cut13 paths',multipliers='8 signed19x13, same exact packed2P/scalar retirement',class_bytes=2592,representative_bytes=96,raw4P_comparison=dict(z_bytes=520,z_vector_port_bits=416,ALUs='8 x32',multipliers=8,consumer_borrow=False,class_metadata=False,meaning='different resource points; candidate has class tables, psum bank address muxes and permutation logic; equal area/timing/energy not established'))
(H/'resource_contract.json').write_text(json.dumps(res,indent=2)+'\n')
result=dict(passed=True,rtl_commands=sum(v['commands'] for v in stages.values()),raw_values_compared=sum(v['raw_values'] for v in stages.values()),original14_20_all_fields_reproduced=sum(v['original14_20_all_fields_reproduced'] for v in stages.values()),independent_gold_values=sum(v['independent_gold_values'] for v in stages.values()),both_held_opportunities_match_actual_RTL=True,real8=real,comparison= comparisons,stages={k:{a:b for a,b in v.items() if a!='summary'} for k,v in stages.items()})
(H/'final_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='stages'},indent=2))
