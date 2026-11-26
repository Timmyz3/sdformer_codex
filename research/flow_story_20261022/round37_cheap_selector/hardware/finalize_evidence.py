#!/usr/bin/env python3
"""Bind executed integer selector, RTL/DC, and two static service neighbors."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;O=P/'actual_v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cpu=read(O/'CPU_EXECUTION_V1.json')
old=P.parents[2]/'flow_story_20261021'
source=old/'round34_deployability/hardware/static_neighbor_supply.cpp'
assert sha(P/'static_neighbor_supply.cpp')==sha(source)
dynamic=read(old/'round33_predictability/hardware/results/SUMMARY_V1.json')['candidates']
dyn={int(r['action']):int(r['dynamic_affine_BN_ticks'])*4 for r in dynamic}
services={}
for n in (155,156):
    d=read(O/f'STATIC{n}_SERVICE_V1.json');r=d['result'];q8=30900+n*810;q4=768000-q8
    assert r['q8_rows']==q8 and r['q4_rows']==q4
    assert r['arithmetic_accepted_beats']==(q8*8+q4*4)*7*6
    bits=[i for i in range(910) if d['mask_words'][i//32]>>(i%32)&1]
    assert bits==sorted(list(range(302,453))+list(range(n-151)))
    services[n]=4*r['affine_plus_BN_completion_ticks']
rows=[]
for r in cpu['oof_replays']:
    total=r['full_prefix_ticks']+dyn[r['action']]
    assert services[155]<=total<services[156]
    rows.append(dict(target_id=r['target_id'],policy=r['policy'],fold=r['fold'],action=r['action'],
        actual_prefix_ticks=r['full_prefix_ticks'],dynamic_four_isolated_ticks=dyn[r['action']],
        scoped_total=total,static155_four_isolated_ticks=services[155],
        remaining_ticks=total-services[155],static156_excess_ticks=services[156]-total))
budget=dict(status='PASS_ALL_128_OOF_POLICY_IDENTITY_PAIRS',services=services,rows=rows,
    min_static155_margin=min(r['remaining_ticks'] for r in rows),
    max_static155_margin=max(r['remaining_ticks'] for r in rows),
    min_static156_excess=min(r['static156_excess_ticks'] for r in rows),
    mask_order='fit static group2 302..452 then lowest remaining tile IDs',
    source_sha256=sha(source),static155_quality_measured=False,
    assumptions=['Serialized selector prefix once plus four isolated conv services, no shared-stem overlap.',
                 'The static155 point uses the unchanged finite supply model and explicit new mask, not only work arithmetic.',
                 'No whole-network latency or static155 AEE is inferred.',
                 'The one-classifier countercondition has measured extraction service, but no additional static-mask search was run.'])
(O/'STATIC_NEIGHBOR_BUDGET_V1.json').write_text(json.dumps(budget,indent=2)+'\n')
rtl=[json.loads(x) for x in (O/'RTL_EXECUTION_V1.jsonl').read_text().splitlines()]
assert len(rtl)==3 and rtl[2]['rejected']
for r in rtl[:2]:assert r['counter_exact'] and r['accepted_beats']==192000 and r['action']==cpu['rows'][0]['service']['action']
assert rtl[0]['last_accept_cycle']-rtl[0]['first_accept_cycle']==191999*11
assert rtl[1]['input_ready_idle_cycles']>0 and rtl[1]['input_valid_backpressure_cycles']>0 and rtl[1]['output_hold_cycles']>=17
assert all(r['first_output_valid_cycle']-r['last_accept_cycle']==26 for r in rtl[:2])
dc=read(O/'dc/summary.json')
assert dc['clock_period_ns']==4 and dc['worst_setup_slack_ns']>0
assert 'This design has no violated constraints.' in (O/'dc/constraints.rpt').read_text()
receipt=dict(status='PASS_REAL_FULL_INPUT_RTL_AND_DC',primary_input=cpu['rows'][0]['target_id'],
    input_sha256=cpu['rows'][0]['input_sha256'],configuration='final_fit_count32_tree_direct',
    final_fit_model_used_only_for_hardware_interface_test=True,OOF_quality_not_assigned_to_final_fit=True,
    raw_full_input_runs=2,accepted_beats_per_run=192000,counts_per_run=32,recorded_runs=rtl,
    surrounding_service_reconciliation=dict(clear_ticks=32,first_clear_on_frame_start=True,
        intrinsic_RTL_acceptance_interval=11,CPU_model_ticks_per_beat=12,
        extra_surrounding_DMA_ingress_tick_per_beat=1,
        ROOT_and_two_nodes_until_output_valid_ticks=15,
        output_acceptance_service_tick=1,head_service_ticks=16,
        other_costs='DMA startup, IOS read and configuration transport are explicit finite-model services; no SRAM/DMA RTL.'),
    synthesis=dc,bindings={str(p.relative_to(P)):sha(p) for p in [P/'rtl/count_tree.sv',P/'rtl/test_count_tree.cpp',
        P/'rtl/synth_count.tcl',O/'RTL_EXECUTION_V1.jsonl',O/'dc/summary.json',O/'dc/area.rpt',O/'dc/timing.rpt',O/'dc/constraints.rpt',O/'dc/run.log']},
    scope=['New 192B register bank, classifier, popcount, add/compare and tree controller synthesized; existing memory capacity is not free physical reuse.',
           'One shared logical read datapath is multiplexed with idle-only debug counter observation; debug mux/output is included in synthesized area.',
           'Same configurable hardware serves ordinary y and projection p; no supervision-specific hardware advantage.',
           'One real final-fit configuration received full input RTL testing; all 128 OOF trees were replayed in the finite CPU model, not each in RTL.',
           'Finite FP32 input admission is contractual; RTL does not implement NaN/Inf rejection.',
           'Pre-layout synthesis only; no power, placed-and-routed area, PE/BN/stem or whole-system timing claim.',
           'DC reports benign constant-adder input/fanout warnings retained in full logs, and no timing/constraint violation.'])
(O/'RTL_DC_RECEIPT_V1.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(static155=services[155],static156=services[156],min_margin=budget['min_static155_margin'],
                     max_margin=budget['max_static155_margin'],dc=dc),indent=2))
