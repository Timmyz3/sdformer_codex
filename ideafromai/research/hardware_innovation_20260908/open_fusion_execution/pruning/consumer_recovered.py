"""Retiming of the actual fixed64-step recovered U/F deployment parameters.

Only the unchanged producer input/sn2 mask maps and raw I24 are reused.
Every updated I24, projection gate, and continuous PED oracle is rebuilt
from the newly deployed U/F and then checked against paid resident execution.
"""
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import consumer_resident as resident
common = resident.common


def main():
    resident.integer_chain.dense = resident.dense_adapter
    common.Machine = resident.ProbeMachine
    axis = 'ordinary'
    folder = common.FULL / 'capture' / axis
    data = common.read_npz(folder / '000_zurich_city_09_a_0001.npz')
    old_q = common.read_npz(folder / 'parameters.npz')
    old_results = json.loads((HERE/'consumer_resident_ready.json').read_text())['axes'][axis]
    producer = json.loads((HERE/'execution.json').read_text())['axes'][axis]
    recovery = json.loads((HERE/'paired_recovery/comparison.json').read_text())
    result = dict(scope=__doc__, evidence='CPU payload-executing issue/port slots; no RTL, PPA or whole-frame time.',
        ordinary_only=True, stress=False, new_training_in_this_script=False,
        new_q_source='paired_recovery/stage64/<mode>_deployed_constants.npz',
        source_inputs='Existing ordinary captured raw I24 and emitted masked sn2 gates; source, preview and sn2 unchanged by recovery.',
        common_supply_is_not_mask_X=True, integrated_chain_closed=False,
        excludes=['native projection service', 'dynamic global BN service', 'later network service'],
        serial_component_contract='Unchanged producer service + newly executed recovered-parameter consumer. Both gate egress and ingress and coefficient fills remain charged; no overlap or handoff claimed.',
        accuracy_contract='Accuracy is the already measured deployed diverse10 (10 frames); service covers two fixed local windows of zurich_city_09_a_0001 only, on the same deployed parameters. Not valid825.',
        unpruned_diverse10_AEE=recovery['unpruned_AEE_frame_mean'],
        resources=dict(state_bytes=131072, coefficient_bytes=131072, RF_vectors=96,
            lanes=8, accumulator_bits=48, state_ports='1R64/1W64', coefficient_port='1R256'),
        modes={})
    for mode in ['global_group2', 'phase_joint']:
        q = common.read_npz(HERE/'paired_recovery/stage64'/f'{mode}_deployed_constants.npz')
        changed = [k for k in old_q if not np.array_equal(q[k], old_q[k])]
        assert set(changed) == {'U_conv2_theta_q16', 'F_q16'}, changed
        record = dict(changed_deployed_fields=changed,
            coefficient_changes={k:dict(changed_elements=int(np.count_nonzero(q[k]!=old_q[k])),
                elements=int(q[k].size), old_nonzeros=int(np.count_nonzero(old_q[k])),
                recovered_nonzeros=int(np.count_nonzero(q[k]))) for k in changed},
            deployed_diverse10={k:recovery['axes'][mode][k] for k in
                ['AEE_frame_mean', 'AEE_pixel_mean', 'delta_unpruned', 'delta_untrained_mask', 'holdout9_delta_unpruned']},
            source_frame_AEE=next(row['AEE'] for row in recovery['axes'][mode]['paired']
                if row['file']=='zurich_city_09_a_0001.npy'), windows={})
        for label in ['corner', 'interior']:
            preview = common.read_npz(HERE/f'{axis}_{label}_{mode}.npz')['gate']
            old_original, _ = common.independent_gold(data, old_q, label, data[label+'_sn2_gate'])
            old_masked, _ = common.independent_gold(data, old_q, label, preview)
            expected, metadata = common.independent_gold(data, q, label, preview)
            value, report = common.run_case(data, q, label, preview, expected, False)
            report['independent_oracle'] = metadata
            report['changed_network_delta_vs_original_capture'] = common.metrics(
                (value['updated'], value['gate'], value['continuous']),
                (old_original['updated'], old_original['gate'], old_original['continuous']))
            common.enrich(report)
            report['recovery_delta_vs_untrained_same_mask'] = common.metrics(
                (value['updated'], value['gate'], value['continuous']),
                (old_masked['updated'], old_masked['gate'], old_masked['continuous']))
            old = old_results[label][mode]
            unpruned = old_results[label]['mask_engine_no_pruning']
            report['old_same_mask_consumer_service'] = old['service_slots']
            report['new_parameter_service_change_vs_old_same_mask'] = report['service_slots']/old['service_slots']-1
            report['unpruned_resident_consumer_service_reused'] = unpruned['service_slots']
            report['consumer_reduction_vs_unpruned_resident'] = 1-report['service_slots']/unpruned['service_slots']
            report['producer_service_reused'] = producer[label][mode]['service_slots']
            report['separate_serial_producer_consumer_sum'] = report['producer_service_reused']+report['service_slots']
            baseline = producer[label]['unpruned']['service_slots']+unpruned['service_slots']
            report['serial_baseline_existing_unpruned_producer_and_resident_consumer'] = baseline
            report['serial_reduction_vs_existing_unpruned_and_resident_consumer'] = 1-report['separate_serial_producer_consumer_sum']/baseline
            report['physical_port_bytes_change_vs_old_same_mask'] = {
                k:report['physical_port_bytes'][k]-old['physical_port_bytes'][k]
                for k in report['physical_port_bytes']}
            report['projection_gate_activity'] = dict(
                untrained_active=int(np.count_nonzero(old_masked['gate'])),
                recovered_active=int(np.count_nonzero(value['gate'])),
                complete_local_gate_count=int(value['gate'].size),
                future_native_projection_cost_unknown=True)
            record['windows'][label] = report
            result['modes'][mode] = record
            print(mode, label, old['service_slots'], '->', report['service_slots'],
                  'serial reduction', report['serial_reduction_vs_existing_unpruned_and_resident_consumer'],
                  'gold', report['checks'], flush=True)
            (HERE/'consumer_recovered_ready.json').write_text(json.dumps(result, indent=2)+'\n')
    summary = dict(scope=result['scope'], evidence=result['evidence'],
        accuracy_contract=result['accuracy_contract'], serial_component_contract=result['serial_component_contract'],
        excludes=result['excludes'], unpruned_diverse10_AEE=result['unpruned_diverse10_AEE'], rows=[])
    for mode, record in result['modes'].items():
        for label, report in record['windows'].items():
            summary['rows'].append(dict(mode=mode, window=label,
                deployed_diverse10_AEE=record['deployed_diverse10']['AEE_frame_mean'],
                deployed_diverse10_delta=record['deployed_diverse10']['delta_unpruned'],
                old_same_mask_consumer=report['old_same_mask_consumer_service'],
                recovered_consumer=report['service_slots'],
                consumer_reduction_vs_unpruned_resident=report['consumer_reduction_vs_unpruned_resident'],
                producer_service=report['producer_service_reused'],
                serial_component_sum=report['separate_serial_producer_consumer_sum'],
                serial_reduction=report['serial_reduction_vs_existing_unpruned_and_resident_consumer'],
                physical_port_bytes=report['physical_port_bytes'],
                physical_port_bytes_change_vs_old_same_mask=report['physical_port_bytes_change_vs_old_same_mask'],
                actual_vs_new_gold=report['checks'],
                local_error_vs_unpruned=report['changed_network_delta_vs_original_capture'],
                projection_gate_activity=report['projection_gate_activity']))
    (HERE/'consumer_recovered_summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
