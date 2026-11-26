#!/usr/bin/env python3
"""Collect actual simulator, Verilator and DC evidence; no synthesis/RTL runs."""
import csv,hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1]
OLD=R.parent.parent/'flow_story_20261011/preflight'
REPO=R.parents[2]
LIB=Path('/opt/tech/tsmc28_2026/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp7t35p140_180a/tcbn28hpcplusbwp7t35p140tt0p9v25c.db')
LEF=Path('/opt/tech/tsmc28_2026/TSMCHOME/digital/Back_End/lef/tcbn28hpcplusbwp7t35p140_110a/lef/tcbn28hpcplusbwp7t35p140.lef')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
macros={}
for m in re.finditer(r'^MACRO\s+(\S+)\s*\n(.*?)^END\s+\1\s*$',LEF.read_text(),re.M|re.S):
 d=re.search(r'\bSIZE\s+([\d.]+)\s+BY\s+([\d.]+)\s*;',m[2])
 if d:macros[m[1]]=float(d[1])*float(d[2])
result={'scope':'stage0 cycle model + representative dot/residual core; no whole-engine timing/area/accuracy/power validation',
        'cycle_script_sha256':sha(R/'cycle_sim.py'),'rtl_sha256':sha(R/'rtl/dot_residual.sv'),
        'library':str(LIB),'library_sha256':sha(LIB),'lef':str(LEF),'lef_sha256':sha(LEF),
        'area_crosscheck_note':'LEF110a vs timing DB180a: only actual mapped cell footprints checked, not release equivalence',
        'tools':{'Verilator':'4.028','DC':'V-2023.12-SP3'},'variants':{}}
for lanes in (4,16):
 sim=R/('results/verilator_lanes'+str(lanes));dc=R/('results/dc_lanes'+str(lanes))
 s=json.loads((sim/'summary.json').read_text());d=json.loads((dc/'summary.json').read_text())
 assert s['status']=='passed' and d['slack_ns']>=0
 tr=list(csv.DictReader((sim/'cycle_trace.csv').open()))
 assert sum(int(v['read_fire']) for v in tr)==s['read_handshakes']
 assert sum(int(v['write_fire']) for v in tr)==s['write_handshakes']
 refs=re.findall(r'^(\S+BWP7T35P140)\s+tcbn28hpcplusbwp7t35p140tt0p9v25c\s+([\d.]+)\s+(\d+)\s+([\d.]+)',(dc/'references.rpt').read_text(),re.M)
 assert refs
 cells=[]
 for name,area,n,total in refs:
  assert name in macros and abs(macros[name]-float(area))<1e-6
  cells.append({'name':name,'area_um2':float(area),'count':int(n)})
 total=sum(c['area_um2']*c['count'] for c in cells);assert abs(total-d['area_library_units'])<1e-4
 actual_refs=set(re.findall(r'^\s*(\w+BWP7T35P140)\s+\S+\s*\(',(dc/'mapped.v').read_text(),re.M))
 assert actual_refs<={c['name'] for c in cells}
 assert 'no violated constraints' in (dc/'constraints.rpt').read_text()
 assert 'FLOW_DOT_SYNTH_DONE' in (dc/'dc.log').read_text() and '\nError:' not in (dc/'dc.log').read_text()
 result['variants'][str(lanes)]={'rtl_simulation':s,'dc':d,'cell_area_um2':total,'mapped_cells':cells,'all_mapped_cell_area_units_checked':True}
cycles=json.loads((R/'results/cycle_results.json').read_text());assert cycles['script_sha256']==sha(R/'cycle_sim.py')
result['cycle_model_status']=cycles['status'];result['cycle_model_vs_unit_clock']=cycles['unit_clock_vs_event_exact_match']
paths=[R/'cycle_sim.py',R/'mask_example.json']
for pattern in ['rtl/*','tb/*','scripts/*','masks/*.json','results/*.json','results/*.log','results/verilator_lanes*/*.csv','results/verilator_lanes*/*.json','results/verilator_lanes*/*.log','results/dc_lanes*/*.rpt','results/dc_lanes*/*.json','results/dc_lanes*/*.log','results/dc_lanes*/*.sdc','results/dc_lanes*/mapped.v']:
 paths.extend(p for p in R.glob(pattern) if p.is_file() and p.name!='HARDWARE_EVIDENCE.json')
result['evidence_sha256']={str(p.relative_to(R)):sha(p) for p in sorted(set(paths))}
source=[REPO/'SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py',
 REPO/'SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_submodules.py',
 OLD/'rtl_issue_probe/rtl/risk_group_issue.sv',OLD/'rtl_issue_probe/results/FRONTEND_RESULTS.json',
 OLD/'execution_contract/execution_services_results.json',OLD/'cpu_numerics/native_mlp_results.json']
result['source_sha256']={str(p):sha(p) for p in source}
engineering=R.parent/'deployment_contract/whole_model_tile_probe.json'
if engineering.exists():result['source_sha256'][str(engineering)]=sha(engineering)
(R/'results/HARDWARE_EVIDENCE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{'area_um2':v['cell_area_um2'],'slack_ns':v['dc']['slack_ns'],'test_cases':v['rtl_simulation']['completed_cases']} for k,v in result['variants'].items()},indent=2))
