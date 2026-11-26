"""Reconcile actual simulation counters and DC reports without rerunning them."""
import hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE.parent/'rtl_results'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sim=json.loads((OUT/'verilator/ACTUAL_PLUS_SYNTHETIC_V1.json').read_text())
manifest=json.loads((OUT/'ACTUAL_VECTOR_MANIFEST_V1.json').read_text())
assert sim['status']=='PASS' and sim['case_count']==110+manifest['cases']==206
vectors=[(k,mode) for k in (1,16,96,128,129,256,384,768,864,1536,3072) for mode in (False,True) for _ in range(5)]
vectors += [(384 if x['fc']=='fc1' else 1536,x['mode']=='Q8') for x in manifest['records']]
expected_binary=sum(((k+127)//128)*(8 if mode else 4) for k,mode in vectors)
expected_parallel=sum((k+15)//16 for k,_ in vectors)
for run in sim['runs']:
 assert run['accepted_beats']==(expected_binary if run['engine']=='binary_popcount128' else expected_parallel)
 assert run['retired_rows']==206 and run['checked_int32_values']==206*16
 if not run['stalls']:assert run['cycles']==run['accepted_beats']+1
dc={}
for name in ('binary','parallel'):
 folder=OUT/('dc_'+name);r=json.loads((folder/'summary.json').read_text())
 assert r['clock_period_ns']==4 and r['worst_setup_slack_ns']>0
 text=(folder/'area.rpt').read_text()
 assert float(re.search(r'Total cell area:\s+([0-9.]+)',text).group(1))==r['area_library_units']
 assert 'This design has no violated constraints.' in (folder/'constraints.rpt').read_text()
 dc[name]=r
unique=len({(x['fc'],tuple(x['coordinate']),tuple(x['outputs']),x['mode']) for x in manifest['records']})
result=dict(schema='flow_r30_integer_datapath_evidence_v1',status='LOCAL_RTL_SIMULATION_AND_DC_COMPLETE',
 actual_transactions=manifest['cases'],unique_actual_operand_mode_cases=unique,synthetic_cases=110,
 total_int32_values_checked=sum(x['checked_int32_values'] for x in sim['runs']),simulations=sim,dc=dc,
 binary_to_parallel_cell_area_ratio=dc['binary']['area_library_units']/dc['parallel']['area_library_units'],
 per_K_steady_arithmetic_beats=[dict(K=k,outputs=16,serial_Q8=8*((k+127)//128),serial_Q4=4*((k+127)//128),ordinary_Q8=(k+15)//16) for k in (96,192,384,768,864,1536,3072)],
 weight_interface_bits=dict(binary=2048,ordinary=2048),activation_interface_bits=dict(binary=128,ordinary=16),
 accumulator_bits=32,output_lanes=16,source_arrays_are_R30_final_weights=False,
 scope='Exact integer streaming PE only; old audited R26 operands and synthetic cases. No SRAM, scale, FP32/native mode, currentBN, whole-network or chip-power claim.',
 source_sha256={str(p.relative_to(HERE.parent)):sha(p) for p in (
  HERE/'plane_pe.sv',HERE/'tb_plane_pe.cpp',HERE/'synth_pe.tcl',HERE/'export_actual_vectors.py',
  OUT/'actual_vectors_v1.txt',OUT/'ACTUAL_VECTOR_MANIFEST_V1.json',OUT/'verilator/ACTUAL_PLUS_SYNTHETIC_V1.json',
  OUT/'dc_binary/summary.json',OUT/'dc_parallel/summary.json')},
 model_forwards=0,GPU_used=False,GT_read=False)
p=OUT/'RTL_SUMMARY_V1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],area_ratio=result['binary_to_parallel_cell_area_ratio'],unique_actual=unique,total_values=result['total_int32_values_checked'])))
