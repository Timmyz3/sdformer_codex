"""Full local integer consumer comparison on the common source-resident MAC.

The generic RF supply improvement is given to every pruning arm. It is not
the phase-mask mechanism's novelty. This wrapper leaves the original model,
original results, and independent candidate oracle unchanged.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import consumer_execute as common
sys.path.insert(0, str(HERE.parent / 'ped_bitplanes'))
from probe import ProbeMachine, resident_mac, rne_sat
import integer_chain


def gather_live(m, k, input_base, k_count, positions, live):
    """Same RF staging as resident_mac; absent T/P values are not consumed."""
    for ip in range(positions):
        for t0 in (0, 8):
            times = list(range(t0, min(t0+8, 10)))
            if not any(live & (1 << (ip*10+t)) for t in times):
                continue
            values = []
            for t in times:
                if live & (1 << (ip*10+t)):
                    m.collect_i24(input_base + ((ip*10+t)*k_count+k)*3)
                    values.append(m.scalar_collector)
                else:
                    values.append(0)
            values += [0]*(8-len(values))
            r = 88+2*ip+t0//8
            m.wait_reg(r)
            m.advance(op=('ILOAD', r, values), tag='common_gather_RF_write')
    m.drain()


def dense_adapter(m, base, name, k_count, h_count, input_base,
                  output_base, positions, shift, live=None, bias=None):
    # U32/V96, including V bias, use the exact newly provided callable.
    if live is None:
        return resident_mac(m, base, name, k_count, h_count, input_base,
                            output_base, positions, shift, bias=bias)
    # F's live mask comes from the paid complete-K NRV directory. Preserve
    # its existing zero-output semantics, while giving F the same RF reuse.
    assert 1 <= positions <= 2
    start = m.time
    m.phase = 'resident_'+name
    active = [tp for tp in range(positions*10) if live & (1 << tp)]
    for h0 in range(0, h_count, 32):
        groups = min(32, h_count-h0)//8
        for r in range(positions*10*groups):
            m.advance(op=('clear', r, None), tag='output_clear')
        for k in range(k_count):
            if not active:
                continue
            gather_live(m, k, input_base, k_count, positions, live)
            for hg in range(groups):
                address = base[name]+(k*h_count+h0+hg*8)*2
                m.coefficient(address)
                weights = np.frombuffer(m.cword, '<i2', count=8, offset=address%32)
                if not weights.any():
                    continue
                for tp in active:
                    ip, t = divmod(tp, 10)
                    src, lane = 88+2*ip+t//8, t%8
                    if m.rf[src, lane] == 0:
                        m.advance(tag='ordinary_zero_bypass')
                        continue
                    dst = tp*groups+hg
                    m.wait_reg(dst)
                    m.advance(op=('IMAC_INDEX', dst, (address%32, src, lane)), tag='ordinary_MAC')
        m.drain()
        for tp in range(positions*10):
            for hg in range(groups):
                dst = tp*groups+hg
                integer_chain.complete(m, dst, shift)
                if bias is not None:
                    m.coefficient(base[bias]+(h0+hg*8)*4)
                    m.advance(op=('IADD_COEF', dst, None), tag=name+'_bias')
                    m.wait_reg(dst)
                    m.advance(op=('ISAT', dst, None), tag=name+'_bias_sat24')
                m.store_i24(dst, output_base+(tp*h_count+h0+hg*8)*3)
    m.mark(m.phase, start)


def check_empty_live(q):
    """Exercise F's all-empty boundary with nonzero backing input bytes."""
    m = ProbeMachine(False)
    blob, base, _ = integer_chain.coeffs(q)
    m.dma_input(blob, 0, True)
    m.dma_input(integer_chain.pack24(np.full((2, 10, 16), 17, np.int64)), integer_chain.LAT16)
    begin = dict(m.count)
    dense_adapter(m, base, 'F', 16, 96, integer_chain.LAT16,
                  integer_chain.UPDATED, 2, int(q['F_exponent']), live=0)
    value = integer_chain.read24(m, integer_chain.UPDATED, (2, 10, 96))
    assert np.count_nonzero(value) == 0
    assert m.count.get('SR64_reads', 0) == begin.get('SR64_reads', 0)
    assert m.count.get('IMAC_INDEX_issues', 0) == begin.get('IMAC_INDEX_issues', 0)
    return dict(nonzero_backing_input=17, live=0, output_nonzeros=0,
                source_reads=0, MAC_issues=0, output_values=int(value.size))


def check_late_interface():
    axis, label = 'ordinary', 'corner'
    folder = common.FULL / 'capture' / axis
    data = common.read_npz(folder / '000_zurich_city_09_a_0001.npz')
    q = common.read_npz(folder / 'parameters.npz')
    preview = common.read_npz(HERE / f'{axis}_{label}_mask_engine_no_pruning.npz')['gate']
    expected, _ = common.independent_gold(data, q, label, preview)
    candidate = dict(data)
    for field, key in [('updated_I24', 'updated'), ('proj_gate', 'gate'), ('continuous_q24', 'continuous')]:
        candidate[label+'_'+field] = expected[key]
    m = ProbeMachine(False)
    words = sum(preview[t].astype(np.uint16) << t for t in range(10)).transpose(1, 2, 0)
    m.dma_input(words.astype('<u2').tobytes(), common.GATE)
    value, report = integer_chain.run(candidate, q, label, preview, machine=m, late_v=True)
    geo = json.loads(str(data['window_geometry_json']))[label]
    oy, ox = geo['gate_origin']
    out_y, out_x = geo['output_origin']
    raw = np.stack([expected['updated'][:, :, 2*(out_y+dy)-oy, 2*(out_x+dx)-ox]
                    for dy in range(4) for dx in range(4)])
    u = rne_sat(raw @ q['U_ped_q16'].astype(np.int64).T, int(q['U_ped_exponent']))
    expected_u = u.reshape(4, 4, 10, 32).transpose(2, 3, 0, 1)
    assert np.array_equal(value['U_ped'], expected_u)
    assert value['PED_spill'].tobytes() == integer_chain.pack24(u)
    assert not any(k == 'resident_V_ped' for k in report['stages'])
    result = dict(scope='Fixed ordinary/corner late-V continuation interface check.',
        updated_and_gate_checks=report['checks'], U_values=int(u.size), U_differences=0,
        spill_bytes=report['PED_spill_bytes'], actual_spill_bytes_match_independent_U=True,
        V_execution_deferred=True, V_bias_coefficients_retained=report['V_coefficients_retained_across_native'],
        no_native_or_global_BN_execution=True)
    (HERE / 'consumer_resident_interfaces.json').write_text(json.dumps(result, indent=2)+'\n')


def summarize():
    ready = json.loads((HERE / 'consumer_resident_ready.json').read_text())
    producer = json.loads((HERE / 'execution.json').read_text())
    result = dict(scope='Complete local integer consumer comparison using the same RF-resident source supply for every arm.',
        evidence=ready['evidence'], generic_supply_is_not_phase_X=True,
        serial_sum_contract='Separate paid producer and fresh consumer executions, both coefficient fills and gate egress/re-ingress included; no integrated handoff, overlap, native projection, global BN, AEE or full-frame claim.',
        ready_rows=[], stress_rows=[])
    for axis, labels in ready['axes'].items():
        old = json.loads((HERE/f'consumer_execution_{axis}.json').read_text())['axes'][axis]
        for label, variants in labels.items():
            pbase = producer['axes'][axis][label]['unpruned']['service_slots']
            for name, report in variants.items():
                result['ready_rows'].append(dict(axis=axis, window=label, variant=name,
                    old_consumer=old[label][name]['service_slots'], resident_consumer=report['service_slots'],
                    common_supply_consumer_reduction=report['common_supply_service_reduction_vs_old_same_mask'],
                    phase_or_global_reduction_vs_resident_unpruned=report['service_reduction_vs_resident_unpruned'],
                    serial_component_sum=report['separate_serial_producer_consumer_sum'],
                    serial_mask_reduction_vs_existing_producer_and_resident_unpruned=report['serial_sum_reduction_vs_existing_producer_and_resident_unpruned'],
                    serial_total_change_vs_existing_producer_and_old_unpruned=1-report['separate_serial_producer_consumer_sum']/(pbase+old[label]['mask_engine_no_pruning']['service_slots']),
                    port_bytes=report['physical_port_bytes'], checks=report['checks'],
                    unchanged_candidate_errors=report['changed_network_delta_vs_original_capture']))
    path = HERE / 'consumer_resident_stress.json'
    if path.exists():
        stress = json.loads(path.read_text())
        old_stress = json.loads((HERE/'consumer_execution_stress.json').read_text())
        for axis, labels in stress['axes'].items():
            for label, variants in labels.items():
                for name, report in variants.items():
                    counterpart = ready['axes'][axis][label][name]
                    assert report['checks'] == counterpart['checks']
                    assert report['changed_network_delta_vs_original_capture'] == counterpart['changed_network_delta_vs_original_capture']
                    prior = old_stress['axes'][axis][label][name]['service_slots']
                    result['stress_rows'].append(dict(axis=axis, window=label, variant=name,
                        old_consumer=prior, resident_consumer=report['service_slots'],
                        common_supply_consumer_reduction=1-report['service_slots']/prior,
                        phase_or_global_reduction_vs_resident_unpruned=report['service_reduction_vs_resident_unpruned'],
                        ready_stress_values_equal_via_identical_independent_gold=True,
                        port_bytes=report['physical_port_bytes']))
    result['empty_live_checks'] = ready['empty_live_checks']
    path = HERE / 'consumer_resident_interfaces.json'
    if path.exists():
        result['late_V_interface_check'] = json.loads(path.read_text())
    (HERE/'consumer_resident_summary.json').write_text(json.dumps(result, indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stress', action='store_true', help='One fixed ordinary/interior pressure comparison.')
    ap.add_argument('--interface-check', action='store_true')
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    # Local process substitution only: neither source module is edited.
    integer_chain.dense = dense_adapter
    common.Machine = ProbeMachine
    if args.interface_check:
        check_late_interface()
        return
    if args.summarize:
        summarize()
        return
    producer = json.loads((HERE / 'execution.json').read_text())
    result = dict(scope=__doc__, evidence='CPU payload issue/port slots; not RTL/PPA/AEE.',
        stress=args.stress, lossy_pruning=True, new_training=False, integrated_chain_closed=False,
        ordinary_supply_improvement_is_common_baseline=True,
        source_staging='RF88..91 holds the current k T10/P2 sources; RF0..79 output accumulators; common24B gather staging/3B collector.',
        resources=dict(state_bytes=131072, coefficient_bytes=131072, RF_vectors=96, lanes=8,
            accumulator_bits=48, state_ports='1R64/1W64', coefficient_port='1R256'),
        no_native_projection_no_global_BN=True, axes={})
    axes = ['ordinary'] if args.stress else ['ordinary', 'lifting_raw']
    labels = ['interior'] if args.stress else ['corner', 'interior']
    names = ['mask_engine_no_pruning', 'global_group2', 'phase_joint']
    for axis in axes:
        folder = common.FULL / 'capture' / axis
        data = common.read_npz(folder / '000_zurich_city_09_a_0001.npz')
        q = common.read_npz(folder / 'parameters.npz')
        result.setdefault('empty_live_checks', {})[axis] = check_empty_live(q)
        old = json.loads((HERE / f'consumer_execution_{axis}.json').read_text())['axes'][axis]
        result['axes'][axis] = {}
        for label in labels:
            original, _ = common.independent_gold(data, q, label, data[label + '_sn2_gate'])
            checks = dict(updated=common.difference(original['updated'], data[label + '_updated_I24']),
                projection_gate=common.difference(original['gate'], data[label + '_proj_gate']),
                PED=common.difference(original['continuous'], data[label + '_continuous_q24']))
            assert all(v['differences'] == 0 for v in checks.values())
            ref = (original['updated'], original['gate'], original['continuous'])
            records = {}
            for name in names:
                preview = common.read_npz(HERE / f'{axis}_{label}_{name}.npz')['gate']
                expected, metadata = common.independent_gold(data, q, label, preview)
                value, report = common.run_case(data, q, label, preview, expected, args.stress)
                report['independent_oracle'] = metadata
                report['unpruned_oracle_vs_original_capture'] = checks
                report['changed_network_delta_vs_original_capture'] = common.metrics(
                    (value['updated'], value['gate'], value['continuous']), ref)
                common.enrich(report)
                assert report['changed_network_delta_vs_original_capture'] == old[label][name]['changed_network_delta_vs_original_capture']
                report['service_reduction_vs_resident_unpruned'] = (0.0 if name == names[0]
                    else 1-report['service_slots']/records[names[0]]['service_slots'])
                if not args.stress:
                    report['common_supply_service_reduction_vs_old_same_mask'] = 1-report['service_slots']/old[label][name]['service_slots']
                    prod = producer['axes'][axis][label][name]['service_slots']
                    prod_base = producer['axes'][axis][label]['unpruned']['service_slots']
                    report['separate_serial_producer_consumer_sum'] = prod+report['service_slots']
                    base_consumer = report['service_slots'] if name == names[0] else records[names[0]]['service_slots']
                    report['serial_sum_reduction_vs_existing_producer_and_resident_unpruned'] = 1-(prod+report['service_slots'])/(prod_base+base_consumer)
                records[name] = report
                result['axes'][axis][label] = records
                print(axis, label, name, report['service_slots'],
                      report['service_reduction_vs_resident_unpruned'], flush=True)
                path = HERE / ('consumer_resident_stress.json' if args.stress else 'consumer_resident_ready.json')
                path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
