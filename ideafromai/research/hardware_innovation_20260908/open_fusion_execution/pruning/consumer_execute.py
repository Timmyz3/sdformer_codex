"""Price phase-H8 pruning through the actual integer dual-consumer chain.

Candidate gold is independently reconstructed from the emitted sn2 gate map,
complete K864 weights, original I24, and existing integer RNE/saturation rules.
It is explicitly a changed-network oracle, never a replacement capture. The
unchanged oracle is checked against the original capture before any timing.

All cases start a fresh identical machine and pay the same gate-map DMA. This
is consumer service, not an integrated producer/consumer or whole-frame run.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
FULL = BASE / 'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
sys.path.insert(0, str(FULL / 'preview_sn2_chain'))
sys.path.insert(0, str(HERE.parent / 'review'))
from run_chain import Machine, GATE, read_npz, difference
from integer_chain import run as execute_integer
from phase_channel_probe import downstream, metrics


def independent_gold(data, q, label, preview_gate):
    """No Machine state or numerical outputs feed this integer oracle."""
    geo = json.loads(str(data['window_geometry_json']))[label]
    h, w = geo['gate_shape']
    oy, ox = geo['gate_origin']
    sy, sx = geo['source_origin']
    out_y, out_x = geo['output_origin']
    original = data[label + '_I24']
    updated = original[:, :, oy-sy:oy-sy+h, ox-sx:ox-sx+w].astype(np.int64).copy()
    points = [(2*(out_y+dy), 2*(out_x+dx)) for dy in range(4) for dx in range(4)]
    rows = []
    raw = []
    for y, x in points:
        feature = np.zeros((10, 96, 3, 3), np.float64)
        for ky in range(3):
            for kx in range(3):
                yy, xx = y+ky-1, x+kx-1
                if 0 <= yy < 240 and 0 <= xx < 320:
                    assert 0 <= yy-oy < h and 0 <= xx-ox < w
                    feature[:, :, ky, kx] = preview_gate[:, :, yy-oy, xx-ox]
        rows.append(feature.reshape(10, 864))
        raw.append(original[:, :, y-sy, x-sx])
    rows = np.stack(rows)
    raw = np.stack(raw).astype(np.float64)
    uacc = rows @ q['U_conv2_theta_q16'].T
    anchor_updated, anchor_gate, ped = downstream(uacc, raw, q)
    for ip, (y, x) in enumerate(points):
        updated[:, :, y-oy, x-ox] = anchor_updated[ip].astype(np.int64)
    perm = q['consumer_permutation'].astype(int)
    threshold = q['consumer_threshold'][:, None, None, None]
    direction = q['consumer_direction'][:, None, None, None]
    constant = q['consumer_constant'][:, None, None, None]
    gate = np.where(constant >= 0, constant.astype(bool),
        np.where(direction > 0, updated[perm] >= threshold, updated[perm] <= threshold))
    continuous = ped.reshape(4, 4, 10, 96).transpose(2, 3, 0, 1).astype(np.int64)
    assert np.array_equal(anchor_gate,
        np.stack([gate[:, :, y-oy, x-ox] for y, x in points]))
    return dict(updated=updated, gate=gate, continuous=continuous), dict(
        full_K=864, anchors=16, time_steps=10, original_I24_unchanged=True,
        oracle='Independent NumPy integer matrix execution, legal accumulators <2**47, FP64 exact integer storage; RNE and signed24 saturation at original boundaries.',
        K_gate_occurrences=int(rows.sum()), nonempty_K_per_anchor_records=int(np.count_nonzero(rows.any(axis=1))))


def run_case(data, q, label, preview_gate, expected, stress):
    # The existing runner's assertions retain their original implementation.
    # Only this clearly identified candidate data view contains NEW gold.
    candidate_view = dict(data)
    candidate_view[label + '_updated_I24'] = expected['updated']
    candidate_view[label + '_proj_gate'] = expected['gate']
    candidate_view[label + '_continuous_q24'] = expected['continuous']
    m = Machine(stress)
    # This already implemented scalar collector/prefetch is a common baseline
    # optimization; withholding it from the unpruned case would be unfair.
    m.forward_i24 = True
    words = sum(preview_gate[t].astype(np.uint16) << t for t in range(10)).transpose(1, 2, 0)
    m.phase = 'sn2_continuation_input'
    m.dma_input(words.astype('<u2').tobytes(), GATE)
    startup = m.time
    value, report = execute_integer(candidate_view, q, label, preview_gate,
        stress=stress, machine=m, late_v=False)
    report['integer_service_without_gate_DMA'] = report['service_slots']
    report['service_slots'] = m.time
    report['gate_DMA_service_slots'] = startup
    report['counts'] = dict(m.count)
    report['stages'] = dict(m.stages)
    report['scalar_I24_forwarding_common_baseline'] = True
    report['gold_provenance'] = 'Independent candidate oracle, not original capture.'
    return value, report


def enrich(report):
    """Physical bytes and anchor-only errors derived from saved paid events."""
    counts = report['counts']
    report['physical_port_bytes'] = dict(
        state_read=8*counts.get('SR64_reads', 0),
        state_write=8*counts.get('SW64_writes', 0),
        coefficient_read=32*counts.get('CR256_reads', 0),
        coefficient_fill=32*counts.get('CW256_writes', 0))
    report['backpressure_or_writeback_stall_slots'] = counts.get('port_or_writeback_wait', 0)
    delta = report['changed_network_delta_vs_original_capture']
    delta['anchor_gate_count'] = 16*10*96
    delta['anchor_gate_flip_fraction'] = delta['gate_flips']/delta['anchor_gate_count']
    delta['PED_real_RMSE_q14'] = delta['PED_q24_RMSE']/(1 << 14)
    if 'sn2_continuation_input' not in report['stages']:
        report['stages']['sn2_continuation_input'] = report['gate_DMA_service_slots']
    assert sum(report['stages'].values()) == report['service_slots']


def summarize():
    producer = json.loads((HERE / 'execution.json').read_text())
    result = dict(scope='Phase-H8 local consumer execution and separate serial component accounting.',
        evidence='CPU payload slot prototype; not RTL, PPA, whole-frame latency, or AEE.',
        serial_component_sum_contract='Producer service plus fresh-machine consumer service, including both gate egress and re-ingress and both coefficient fills. No overlap or integrated handoff is claimed.',
        rows=[])
    for path in sorted(HERE.glob('consumer_execution*.json')):
        document = json.loads(path.read_text())
        for axis, labels in document['axes'].items():
            for label, variants in labels.items():
                base = variants['mask_engine_no_pruning']
                prod_base = producer['axes'][axis][label]['mask_engine_no_pruning']['service_slots']
                existing_prod_base = producer['axes'][axis][label]['unpruned']['service_slots']
                for name, report in variants.items():
                    enrich(report)
                    row = dict(axis=axis, window=label, variant=name, stress=document['stress'],
                        consumer_service=report['service_slots'],
                        consumer_reduction=report['service_reduction_vs_same_engine_unpruned'],
                        port_bytes=report['physical_port_bytes'],
                        consumer_error=report['changed_network_delta_vs_original_capture'])
                    if not document['stress']:
                        prod = producer['axes'][axis][label][name]['service_slots']
                        row['producer_service'] = prod
                        row['serial_component_sum'] = prod+report['service_slots']
                        row['serial_sum_reduction_vs_same_engine_unpruned'] = 1-(prod+report['service_slots'])/(prod_base+base['service_slots'])
                        row['serial_sum_reduction_vs_existing_unpruned'] = 1-(prod+report['service_slots'])/(existing_prod_base+base['service_slots'])
                    result['rows'].append(row)
        path.write_text(json.dumps(document, indent=2) + '\n')
    stress_path = HERE / 'consumer_execution_stress.json'
    if stress_path.exists():
        stress = json.loads(stress_path.read_text())
        pairs = 0
        for axis, labels in stress['axes'].items():
            ready = json.loads((HERE / f'consumer_execution_{axis}.json').read_text())
            for label, variants in labels.items():
                for name, report in variants.items():
                    counterpart = ready['axes'][axis][label][name]
                    assert report['checks'] == counterpart['checks']
                    assert report['changed_network_delta_vs_original_capture'] == counterpart['changed_network_delta_vs_original_capture']
                    pairs += 1
        result['ready_stress_check'] = dict(pairs=pairs,
            every_executed_updated_gate_PED_exact_vs_identical_independent_gold=True,
            all_candidate_error_metrics_identical=True,
            trace='Period32: state reads blocked at slots24..31; writes blocked at28..31. Common RF writeback arbitration retained.')
    (HERE / 'consumer_summary.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axis', choices=['ordinary', 'lifting_raw'])
    ap.add_argument('--stress', action='store_true')
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    if args.summarize:
        summarize()
        return
    result = dict(scope=__doc__, evidence='CPU payload slot prototype; not RTL or PPA',
        lossy=True, training=False, new_AEE=False, stress=args.stress,
        fresh_machine_consumer_only=True, integrated_chain_closed=False,
        same_resources=dict(state_bytes=131072, coefficient_bytes=131072,
            RF_words=96, RF_lanes=8, state_ports='1R64 / 1W64', coefficient_port='1R256'),
        no_native_projection_no_global_BN=True, axes={})
    names = ['mask_engine_no_pruning', 'phase_joint', 'global_group2', 'phase_magnitude']
    for axis in ([args.axis] if args.axis else ['ordinary', 'lifting_raw']):
        folder = FULL / 'capture' / axis
        data = read_npz(folder / '000_zurich_city_09_a_0001.npz')
        q = read_npz(folder / 'parameters.npz')
        result['axes'][axis] = {}
        for label in ['corner', 'interior']:
            records = {}
            original_gold, _ = independent_gold(data, q, label, data[label + '_sn2_gate'])
            original_check = dict(updated=difference(original_gold['updated'], data[label + '_updated_I24']),
                projection_gate=difference(original_gold['gate'], data[label + '_proj_gate']),
                PED=difference(original_gold['continuous'], data[label + '_continuous_q24']))
            assert all(v['differences'] == 0 for v in original_check.values()), original_check
            original_metric_tuple = (original_gold['updated'], original_gold['gate'], original_gold['continuous'])
            for name in names:
                preview_gate = read_npz(HERE / f'{axis}_{label}_{name}.npz')['gate']
                expected, metadata = independent_gold(data, q, label, preview_gate)
                value, report = run_case(data, q, label, preview_gate, expected, args.stress)
                report['independent_oracle'] = metadata
                report['unpruned_oracle_vs_original_capture'] = original_check
                report['changed_network_delta_vs_original_capture'] = metrics(
                    (value['updated'], value['gate'], value['continuous']), original_metric_tuple)
                enrich(report)
                report['service_reduction_vs_same_engine_unpruned'] = (
                    0.0 if name == names[0] else 1-report['service_slots']/records[names[0]]['service_slots'])
                records[name] = report
                print(axis, label, name, report['service_slots'],
                    report['service_reduction_vs_same_engine_unpruned'],
                    report['changed_network_delta_vs_original_capture'], flush=True)
                result['axes'][axis][label] = records
                suffix = ('_' + args.axis if args.axis else '') + ('_stress' if args.stress else '')
                (HERE / ('consumer_execution' + suffix + '.json')).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
