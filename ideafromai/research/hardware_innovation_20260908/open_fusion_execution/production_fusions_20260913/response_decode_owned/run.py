"""C2 fixed response-local W8 unpack, three arms at two saved conditions."""
from pathlib import Path
import argparse
import copy
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
OPEN = HERE.parents[1]
PRIOR = OPEN / 'breadth_20260912/hardware'
sys.path.insert(0, str(PRIOR))
import run_packed as old
packed, stage, consumer = old.packed, old.stage, old.consumer


class ResponseMachine(packed.PackedMachine):
    DECODE_LATENCY = 2

    def __init__(self, stress=False):
        super().__init__(stress)
        self.response_decode_enabled = False
        self.response_events = []
        self.active_response_event = None
        self.source_staging_writes = 0
        self.source_weight_overlap_errors = 0

    def collect_i24(self, address):
        # Bookkeeping performs no operation and adds no cycles to any arm.
        # The original collect_i24 obtains all bytes from actual SR64.
        held = bytes(self.common_staging[24:40])
        super().collect_i24(address)
        self.common_staging[0:3] = int(self.scalar_collector).to_bytes(3, 'little', signed=True)
        assert self.common_staging[24:40] == held
        self.source_staging_writes += 1

    def unpack_weight(self, address, bits):
        if not self.response_decode_enabled:
            return super().unpack_weight(address, bits)
        assert bits == 8
        self.coefficient(address)
        offset = address % 32
        assert offset in (0, 8, 16, 24)
        assert self.caddr == address//32*32
        raw = bytes(self.cword[offset:offset+8])
        assert len(raw) == 8
        if self.active_response_event is not None:
            self.active_response_event['next_decode_slot'] = self.time
        self.packed_coeff_valid = False
        start = self.time
        # A paid controller issue occupies the shared single issue slot.
        # The second charged slot models response selection/sign-extension
        # latency, with no execution overlap or speculative data availability.
        self.advance(tag='response_W8_decode_issue')
        for _ in range(self.DECODE_LATENCY-1):
            self.advance(tag='response_W8_decode_wait')
        weight = np.frombuffer(raw, 'i1').astype('<i2')
        source_held = bytes(self.common_staging[:24])
        self.common_staging[24:40] = weight.tobytes()
        assert self.common_staging[:24] == source_held
        self.packed_coeff_valid = True
        event = dict(address=address, response_word=self.caddr, offset=offset,
                     issue_slot=start, available_slot=self.time,
                     declared_latency=self.DECODE_LATENCY, vector_MAC_uses=0,
                     source_gather_region_preserved=True,
                     first_MAC_slot=None, last_MAC_slot=None)
        assert event['available_slot'] == event['issue_slot']+2
        self.response_events.append(event)
        self.active_response_event = event
        self.count['response_W8_vectors_decoded'] += 1
        return bool(np.any(weight))

    def integer_op(self, kind, dst, args):
        if self.response_decode_enabled and kind == 'IMAC_PACKED_INDEX':
            event = self.active_response_event
            assert event is not None and self.packed_coeff_valid
            assert self.time >= event['available_slot']
            raw = self.coef[event['address']:event['address']+8]
            expected = np.frombuffer(raw, 'i1').astype('<i2').tobytes()
            assert bytes(self.common_staging[24:40]) == expected
            event['vector_MAC_uses'] += 1
            if event['first_MAC_slot'] is None:
                event['first_MAC_slot'] = self.time
            event['last_MAC_slot'] = self.time
        return super().integer_op(kind, dst, args)


def run(stress=False):
    axis, label = 'ordinary', 'interior'
    folder = stage.common.FULL/'capture'/axis
    data = stage.common.read_npz(folder/'000_zurich_city_09_a_0001.npz')
    live = stage.common.read_npz(folder/'live_parameters.npz')
    q = stage.common.read_npz(folder/'parameters.npz')
    stage.run_windows.Machine = ResponseMachine
    stage.run_windows.build_nrv = stage.preview_directory
    print('Executing common actual producer', 'stress' if stress else 'ready', flush=True)
    _, producer, prefix = stage.run_windows.window(data, live, label, False, stress, True, axis)
    archived = json.loads((stage.common.FULL/'preview_sn2_chain/windows.json').read_text())['axes'][axis][label]['expanded_fp32']['checks']
    aggregate_compatibility = []
    producer_compatible = producer['checks'].keys() == archived.keys()
    for endpoint, expected_fields in archived.items():
        actual_fields = producer['checks'][endpoint]
        producer_compatible &= actual_fields.keys() == expected_fields.keys()
        for key, expected in expected_fields.items():
            actual = actual_fields[key]
            if actual == expected:
                continue
            # Only a floating summary reduction may change its final bit
            # across NumPy runtimes. Payload mismatch counts and maxima stay
            # exact; this exception never licenses different gates or values.
            allowed = bool(key == 'rms' and abs(actual-expected) <= 2*abs(np.spacing(expected)))
            producer_compatible &= allowed
            aggregate_compatibility.append(dict(endpoint=endpoint, field=key,
                                                 actual=actual, archived=expected,
                                                 accepted_summary_only_2ULP=allowed))
    if not producer_compatible or producer['checks']['sn2']['differences'] != 0:
        (HERE/('stress_prefix_check_failure.json' if stress else 'ready_prefix_check_failure.json')).write_text(
            json.dumps(dict(actual=producer['checks'], expected=archived), indent=2)+'\n')
        raise AssertionError('Producer check mismatch; see owned prefix_check_failure.json')
    print('Producer complete', prefix.time, flush=True)
    packed.install()
    prior = json.loads((PRIOR/('ordinary_interior_stress.json' if stress else 'ordinary_interior.json')).read_text())
    result = dict(axis=axis, window=label, stress=stress, producer=producer, rows=[],
                  producer_aggregate_compatibility=aggregate_compatibility,
                  scope=prior['scope'], evidence='Actual CPU payload issue/port prototype, no RTL/PPA/AEE.',
                  parameters='stage_20260912/weight_compensation/ordinary_lowbit_gpu_parameters.npz W8 fields; independently checked against saved GPU deployment',
                  source_quantizer_not_changed=True,
                  response_decoder=dict(latency_slots=2, issue_slots=1, explicit_wait_slots=1,
                                        shared_staging_source=[0, 24], shared_staging_weight=[24, 40],
                                        logic='4:1 selection of 64bits from one CR256 response; eight signed8 to signed16 extensions; existing16B latch write and valid control',
                                        additional_RF=0, timing_validated_by_STA=False),
                  resources=prior['resource'])
    path = HERE/('stress.json' if stress else 'ready.json')
    for mode, inherited_mode in [('expanded16', 'W8_expanded16'), ('RF84_decode', 'W8_packed'), ('response_decode', 'W8_packed')]:
        nq, candidate, delta = old.parameters_and_gold(data, q, label, axis, inherited_mode)
        m = copy.deepcopy(prefix)
        m.forward_i24 = True
        m.response_decode_enabled = mode == 'response_decode'
        boundary = m.time
        values, report = consumer.run(candidate, nq, label, None, stress=stress, machine=m, rank=32)
        assert sum(m.stages.values()) == m.time
        if mode != 'response_decode':
            ref = next(x for x in prior['rows'] if x['mode'] == inherited_mode)
            assert m.time == ref['service_slots'], (mode, m.time, ref['service_slots'])
            assert dict(m.count) == ref['counts'], (mode, 'archived counts differ')
        if mode == 'response_decode':
            assert not m.count.get('IUNPACK_SIGNED_issues', 0)
            assert not m.count.get('packed_weight_RF_to_common_staging', 0)
            assert all(e['declared_latency'] == 2 for e in m.response_events)
            assert all(e['last_MAC_slot'] is None or e.get('next_decode_slot', m.time) > e['last_MAC_slot'] for e in m.response_events)
        blob, base, _ = packed.coeffs(nq)
        counts = dict(m.count)
        row = dict(mode=mode, inherited_function=inherited_mode,
                   service_slots=m.time, consumer_service_slots=m.time-boundary,
                   producer_end=boundary, consumer_begin=boundary, same_machine_handoff=True,
                   prefix_reuse='Full executed Machine state copied; no gate substitution or summed prior service table.',
                   consumer=report, counts=counts, stages=dict(m.stages), timeline=m.timeline,
                   physical_port_bytes=dict(SR64=8*counts['SR64_reads'], SW64=8*counts['SW64_writes'],
                                            CR256=32*counts['CR256_reads'], CW256=32*counts['CW256_writes']),
                   checks=report['checks'], candidate_delta=delta,
                   coefficient_pool_blob_bytes=len(blob), coefficient_layout=base,
                   source_weight_staging_checks=m.source_staging_writes,
                   original_RNE_and_bias_preserved=True, row_scale_two_16x24_products_preserved=True,
                   archived_same_point_exact_counts_reproduced=mode != 'response_decode')
        if mode == 'response_decode':
            row['decoder_audit'] = dict(vectors=len(m.response_events),
                                        actual_vector_MAC_uses=sum(e['vector_MAC_uses'] for e in m.response_events),
                                        earliest_available_delay=min(e['available_slot']-e['issue_slot'] for e in m.response_events),
                                        latest_available_delay=max(e['available_slot']-e['issue_slot'] for e in m.response_events),
                                        first_MAC_not_before_decode_complete=True,
                                        all_vectors_preserved_until_last_MAC=True,
                                        weight_latch_bytes=16, source_gather_region_bytes=24)
            (HERE/('stress_decoder_events.json' if stress else 'ready_decoder_events.json')).write_text('[\n'+',\n'.join(json.dumps(e, separators=(',', ':')) for e in m.response_events)+'\n]\n')
        result['rows'].append(row)
        path.write_text(json.dumps(result, indent=2)+'\n')
        print(mode, m.time, m.time-boundary, report['checks'], flush=True)
    expanded, rf84, response = result['rows']
    assert expanded['candidate_delta'] == rf84['candidate_delta'] == response['candidate_delta']
    for row in result['rows']:
        row['percent_vs_expanded16'] = 100*(row['service_slots']/expanded['service_slots']-1)
        row['consumer_percent_vs_expanded16'] = 100*(row['consumer_service_slots']/expanded['consumer_service_slots']-1)
    response['saved_slots_vs_RF84'] = rf84['service_slots']-response['service_slots']
    charged_decode = response['counts'].get('response_W8_decode_issue', 0)+response['counts'].get('response_W8_decode_wait', 0)
    response['optimistic_zero_decoder_cost_arithmetic_bound'] = dict(
        hypothetical_slots=response['service_slots']-charged_decode,
        expanded16_slots=expanded['service_slots'],
        gap_slots=response['service_slots']-charged_decode-expanded['service_slots'],
        measured=False,
        note='Subtract charged decoder slots only, not a new executed latency arm or a general placement impossibility proof; does not recompute changed backpressure.')
    result['complete'] = True
    path.write_text(json.dumps(result, indent=2)+'\n')
    print('RESPONSE_DECODE_COMPLETE', 'stress' if stress else 'ready', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stress', action='store_true')
    run(parser.parse_args().stress)
