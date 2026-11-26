#!/usr/bin/env python3
"""Collect bounded frontend results and validate evidence links; no tool runs."""
import csv
import datetime
import hashlib
import json
import pathlib
import re

root = pathlib.Path(__file__).resolve().parents[1]
lef = pathlib.Path('/opt/tech/tsmc28_2026/TSMCHOME/digital/Back_End/lef/tcbn28hpcplusbwp7t35p140_110a/lef/tcbn28hpcplusbwp7t35p140.lef')
library = pathlib.Path('/opt/tech/tsmc28_2026/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp7t35p140_180a/tcbn28hpcplusbwp7t35p140tt0p9v25c.db')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

result = {'scope': 'RTL frontend only; descriptor handshakes are not PE execution, memory transactions, power or whole-network speed',
          'utc': datetime.datetime.utcnow().isoformat()+'Z',
          'tool_versions': {'verilator': '4.028', 'dc': 'V-2023.12-SP3'},
          'target_library': str(library), 'target_library_sha256': digest(library),
          'front_end_parameters': {'ADDR_W': 16, 'GROUP_W': 16, 'risk_bits': 8, 'group_members': [1, 4], 'decision': 'max(risks) >= threshold'},
          'variants': {}}

lef_text = lef.read_text()
macros = {}
for match in re.finditer(r'^MACRO\s+(\S+)\s*\n(.*?)^END\s+\1\s*$', lef_text, re.M|re.S):
    dims = re.search(r'\bSIZE\s+([\d.]+)\s+BY\s+([\d.]+)\s*;', match.group(2))
    if dims:
        macros[match.group(1)] = float(dims.group(1))*float(dims.group(2))
physical_check = {'lef': str(lef), 'lef_sha256': digest(lef),
                  'note': 'LEF release110a vs timing release180a; compare actual mapped cell footprints, not assume release equivalence', 'variants': {}}

for mode in ['dynamic','forced']:
    simulation = json.loads((root/'results'/mode/'simulation_summary.json').read_text())
    synthesis = json.loads((root/'results'/('dc_'+mode)/'summary.json').read_text())
    assert simulation['passed'] and synthesis['worst_setup_slack_ns'] >= 0
    rows = list(csv.DictReader((root/'results'/mode/'group_results.csv').open()))
    trace = list(csv.DictReader((root/'results'/mode/'issue_trace.csv').open()))
    aborted_trace = list(csv.DictReader((root/'results'/mode/'aborted_issue_trace.csv').open()))
    assert len(rows) == simulation['completed_groups']
    assert len(trace) + len(aborted_trace) == simulation['compute_issues'] + simulation['bypass_issues']
    assert sum(int(row['compute_count']) for row in rows) == simulation['completed_group_compute_issues']
    assert sum(int(row['bypass_count']) for row in rows) == simulation['completed_group_bypass_issues']
    assert sum(row['channel']=='compute' for row in aborted_trace) == simulation['accepted_compute_in_aborted_groups']
    assert sum(row['channel']=='bypass' for row in aborted_trace) == simulation['accepted_bypass_in_aborted_groups']
    references = (root/'results'/('dc_'+mode)/'references.rpt').read_text()
    matches = re.findall(r'^(\S+BWP7T35P140)\s+tcbn28hpcplusbwp7t35p140tt0p9v25c\s+([\d.]+)\s+(\d+)\s+([\d.]+)', references, re.M)
    cells = []
    for name, area, count, total in matches:
        cells.append({'cell':name,'liberty_unit_area':float(area),'count':int(count),
                      'lef_rectangle_um2':macros.get(name),
                      'area_matches':name in macros and abs(macros[name]-float(area)) < 1e-6})
    assert cells and all(c['area_matches'] for c in cells)
    netlist = (root/'results'/('dc_'+mode)/'mapped.v').read_text()
    netlist_refs = set(re.findall(r'^\s*(\w+BWP7T35P140)\s+\S+\s*\(', netlist, re.M))
    assert not netlist_refs-set(c['cell'] for c in cells)
    weighted_area = sum(c['count']*c['liberty_unit_area'] for c in cells)
    assert abs(weighted_area-synthesis['total_cell_area_library_units'])<1e-4
    physical_check['variants'][mode] = {'mapped_cell_types_checked':len(set(c['cell'] for c in cells)), 'reference_records_checked':len(cells), 'netlist_cell_reference_coverage_complete':True, 'weighted_reference_area':weighted_area, 'all_match':True, 'cells':cells}
    result['variants'][mode] = {'simulation':simulation,'synthesis':synthesis,
                               'cell_area_um2':synthesis['total_cell_area_library_units'],
                               'no_stall_cycles':{'one_member':2 if mode=='forced' else 3,'four_members':5 if mode=='forced' else 9}}

result['control_only_area_increment_um2'] = result['variants']['dynamic']['cell_area_um2']-result['variants']['forced']['cell_area_um2']
result['physical_area_check'] = physical_check
evidence = []
for pattern in ['rtl/*','tb/*','scripts/*','results/*/*.json','results/*/*.csv','results/dc_*/*.rpt','results/dc_*/*.sdc','results/dc_*/mapped.v']:
    evidence += [p for p in root.glob(pattern) if p.is_file()]
result['evidence_sha256'] = {str(p.relative_to(root)):digest(p) for p in sorted(set(evidence))}
(root/'results'/'FRONTEND_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'area_dynamic_um2':result['variants']['dynamic']['cell_area_um2'],
                  'area_forced_um2':result['variants']['forced']['cell_area_um2'],
                  'control_only_increment_um2':result['control_only_area_increment_um2'],
                  'mapped_cell_types_checked':{k:v['mapped_cell_types_checked'] for k,v in physical_check['variants'].items()},
                  'evidence_files':len(result['evidence_sha256'])},indent=2))
