"""Independent scope, recorded RTL timing, DC and budget consistency review."""
from pathlib import Path
import hashlib
import json
import re

B=Path(__file__).resolve().parent
H=B.parent/'round37_cheap_selector/hardware'
R33=B.parents[1]/'flow_story_20261021/round33_predictability'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

f=read(H/'FREEZE_V1.json')
for name,digest in f['files'].items():
    assert sha(H/name)==digest,name
rtl=[json.loads(s) for s in (H/'actual_v1/RTL_EXECUTION_V1.jsonl').read_text().splitlines()]
for r in rtl[:2]:
    assert r['accepted_beats']==192000 and r['counter_exact'] and r['action']==2
    assert r['first_output_valid_cycle']-r['last_accept_cycle']==26
assert rtl[0]['last_accept_cycle']-rtl[0]['first_accept_cycle']==191999*11
assert rtl[1]['input_ready_idle_cycles']>0 and rtl[1]['input_valid_backpressure_cycles']>0
assert rtl[1]['output_hold_cycles']==18 and rtl[2]['rejected']
add=read(H/'rtl_depth_addendum_v1/RECEIPT_V1.json')
assert add['status']=='PASS' and add['unchanged_RTL_sha256']==sha(H/'rtl/count_tree.sv')
assert add['unchanged_frozen_hardware_sha256']==sha(H/'FREEZE_V1.json')
for row in add['rows']:
    assert row['counts_exact']
    for run in row['runs']:
        if 'last_accept_cycle' not in run:continue
        assert run['counter_exact'] and run['accepted_beats']==192000
        assert run['action']==row['action']
        assert run['first_output_valid_cycle']-run['last_accept_cycle']==12+7*row['depth']
dc=read(H/'actual_v1/dc/summary.json')
area=(H/'actual_v1/dc/area.rpt').read_text()
assert float(re.search(r'Total cell area:\s+([0-9.]+)',area).group(1))==dc['area_library_units']
assert 'tcbn28hpcplusbwp7t35p140tt0p9v25c' in area
assert dc['clock_period_ns']==4 and dc['worst_setup_slack_ns']>0
assert 'no violated constraints' in (H/'actual_v1/dc/constraints.rpt').read_text()
assert not dc['whole_accelerator'] and not dc['PE_included'] and not dc['SRAM_included']
b=read(H/'actual_v1/STATIC_NEIGHBOR_BUDGET_V1.json')
original=read(R33/'hardware/results/SUMMARY_V1.json')
byaction={r['action']:r for r in original['candidates']}
assert len(b['rows'])==128
for r in b['rows']:
    conv=4*byaction[r['action']]['dynamic_affine_BN_ticks']
    assert conv==r['dynamic_four_isolated_ticks']
    assert r['scoped_total']==conv+r['actual_prefix_ticks']
    assert r['remaining_ticks']==r['scoped_total']-b['services']['155']>0
    assert r['static156_excess_ticks']==b['services']['156']-r['scoped_total']>0
assert not b['static155_quality_measured']
paths=['FREEZE_V1.json','rtl/count_tree.sv','rtl/test_count_tree.cpp','rtl/synth_count.tcl',
       'actual_v1/RTL_EXECUTION_V1.jsonl','rtl_depth_addendum_v1/RECEIPT_V1.json',
       'actual_v1/dc/summary.json','actual_v1/dc/area.rpt','actual_v1/dc/timing.rpt',
       'actual_v1/dc/constraints.rpt','actual_v1/STATIC_NEIGHBOR_BUDGET_V1.json']
out=B/'COUNT32_RTL_DC_BUDGET_INDEPENDENT_V1.json'
assert not out.exists()
out.write_text(json.dumps(dict(status='PASS_BOUNDED_RECORDED_MODULE_EVIDENCE',
 auditor='root, separate from hardware implementer; source inspection and independent evidence arithmetic',
 reran_synthesis=False,reran_RTL_simulation=False,
 frozen_hardware_files_verified=len(f['files']),source_sha256=sha(Path(__file__)),
 area_library_units=dc['area_library_units'],clock_constraint_ns=4,
 worst_setup_slack_ns=dc['worst_setup_slack_ns'],verified_RTL_depths=[0,1,2],
 intrinsic_RTL_beat_interval=11,model_serial_service_per_beat=12,
 scope=['New selector registers/classifiers/count/tree only; extra area paid.',
        'DMA/IOS outside RTL retain finite-model service; stem and full network excluded.',
        'Recorded RTL runs cover one actual input plus legal depth0/1 configurations; other OOF inputs have independent CPU packed execution.',
        'DC reports include high-fanout and constant/unconnected-port lint warnings; no violated constraints is not a post-layout timing claim.',
        'static155 has no measured AEE; no co-design quality win follows from its service.'],
 bindings={name:sha(H/name) for name in paths}),indent=2)+'\n')
print('RTL_DC_BUDGET_INDEPENDENT_PASS',sha(out))
