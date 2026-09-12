"""One actual ordinary/corner/ready phase interface, exactly two fixed arms.

Both arms use the same complete LUT and H8 reconstruction in RF before U.
Only the phase arm introduces a real signed16 SRAM z boundary. Old Dg/RECON
results are read-only external controls, not rerun or modified here.
"""
from pathlib import Path
import copy
import json
import numpy as np
import binding
import kernel
import predictor
from phase_probe import reconstruct, rne_sat

HERE = Path(__file__).resolve().parent
TABLE = 98304
CODE = 114688
MASK20 = (1 << 20)-1
PRED = 40
SOURCE = 60
TEMP = 51
NEW_OPS = ('ILUT20_BROADCAST', 'ILUT20_LANE', 'IFLOOR11', 'IPHASE11')


class PhaseMachine(kernel.QuantMachine):
    def __init__(self, stress=False):
        super().__init__(stress)
        self.phase_lut_events = []
        self.phase_h8_events = []

    def integer_op(self, kind, dst, args):
        if kind in ('ILUT20_BROADCAST', 'ILUT20_LANE'):
            word, time_row, lane = map(int, args)
            assert 0 <= word < 1024 and 0 <= time_row < 10
            assert self.caddr == TABLE+word*32
            # The operand is extracted from the actual single CR256 response.
            bits = (int.from_bytes(self.cword, 'little') >> (20*time_row)) & MASK20
            signed = bits-(1 << 20) if bits & (1 << 19) else bits
            if kind == 'ILUT20_BROADCAST':
                assert word == 0
                value = np.full(8, signed, np.int64)
                self.count['phase_LUT_broadcast_values'] += 8
            else:
                assert 0 <= lane < 8 and self.ready[dst] <= self.time
                value = self.rf[dst].astype(np.int64).copy()
                value[lane] = signed
                self.count['phase_LUT_lane_RMW_RF_reads'] += 1
                self.count['phase_LUT_lane_values'] += 1
        elif kind in ('IFLOOR11', 'IPHASE11'):
            src = int(args)
            assert self.ready[src] <= self.time
            b = self.rf[src].astype(np.int64)
            value = np.floor_divide(b, 2048) if kind == 'IFLOOR11' else np.bitwise_and(b, 2047)
            self.count['phase_floor_or_mod_RF_reads'] += 1
        else:
            return super().integer_op(kind, dst, args)
        assert np.all(value >= -(1 << 47)) and np.all(value < (1 << 47))
        return value.astype(np.float64)


def new_issue(m, kind, dst, args, tag):
    """Same single issue and two-slot integer writeback as original ops."""
    assert kind in NEW_OPS
    m.wait_reg(dst)
    m.advance(op=(kind, dst, args), tag=tag)
    assert m.ready[dst] == m.time+1  # issue is the preceding slot
    m.count['phase_checked_two_slot_new_issues'] += 1


def table_image(p):
    assert np.all(p['step'] == 2048)
    g = ((np.arange(1024)[:, None] >> np.arange(10)) & 1).T
    b = p['D'].astype(np.int64) @ g+p['c'].astype(np.int64)[:, None]
    assert np.all(b >= -(1 << 19)) and np.all(b < (1 << 19))
    image = bytearray()
    for word in range(1024):
        packed = sum((int(b[t, word]) & MASK20) << (20*t) for t in range(10))
        raw = packed.to_bytes(32, 'little')
        # Offline parameter-image verification, not a timed table supplier.
        for t in range(10):
            field = (int.from_bytes(raw, 'little') >> (20*t)) & MASK20
            decoded = field-(1 << 20) if field & (1 << 19) else field
            assert decoded == b[t, word]
        image.extend(raw)
    assert len(image) == 32768 and TABLE+len(image) == 131072
    return bytes(image)


def load_table(m, image):
    if getattr(m, 'phase_table_loaded', False):
        return
    m.phase = 'phase_complete_LUT_cold_fill'
    m.dma_input(image, TABLE, True)
    assert bytes(m.coef[TABLE:TABLE+len(image)]) == image
    m.phase_table_loaded = True


def table_predict_h8(m, gate_address):
    """Fill b[10,8] in RF40..49, with equal c/g0 permission in both arms."""
    m.phase = 'phase_LUT_predict_H8'
    begin = m.time
    # Common c baseline travels through the complete table's actual row0.
    reads = m.count['CR256_reads']
    m.coefficient(TABLE)
    response_slot = m.time
    for t in range(10):
        new_issue(m, 'ILUT20_BROADCAST', PRED+t, (0, t, 0), 'phase_LUT_c_broadcast')
    m.phase_lut_events.append(dict(word=0, broadcast=True, begin=begin,
        response_ready=response_slot, end=m.time, CR256_reads=m.count['CR256_reads']-reads,
        extracted_fields=10, written_lanes=80))
    predictor._load_gate_h8(m, gate_address)
    for lane in range(8):
        m.wait_reg(predictor.GATE_REG)
        m.advance(tag='phase_gword_lane_select_and_zero_test')
        # Controller selection after its charged RF read; no host gate array.
        word = int(m.rf[predictor.GATE_REG, lane])
        assert 0 <= word < 1024
        m.count['phase_gword_selector_RF_reads'] += 1
        if word == 0:
            m.count['phase_gword_zero_bypass'] += 1
            continue
        reads = m.count['CR256_reads']; begin = m.time
        m.coefficient(TABLE+word*32)
        response_slot = m.time
        for t in range(10):
            new_issue(m, 'ILUT20_LANE', PRED+t, (word, t, lane), 'phase_LUT_b_lane_extract')
        m.phase_lut_events.append(dict(word=word, lane=lane, broadcast=False,
            begin=begin, response_ready=response_slot, end=m.time,
            CR256_reads=m.count['CR256_reads']-reads, extracted_fields=10, written_lanes=10))
    for t in range(10):
        m.wait_reg(PRED+t)
    m.count['phase_LUT_H8_calls'] += 1


def store_z16(m, reg, address):
    m.wait_reg(reg)
    values = m.rf[reg].astype(np.int64)
    assert np.all(values >= -256) and np.all(values <= 255)
    held = bytes(m.common_staging[:24])+bytes(m.common_staging[40:])
    payload = values.astype('<i2').tobytes()
    m.advance(tag='phase_z16_RF_to_staging')
    m.common_staging[24:40] = payload
    for offset in (0, 8):
        m.advance(write=(address+offset, bytes(m.common_staging[24+offset:32+offset])),
                  tag='phase_z16_SW64')
    assert held == bytes(m.common_staging[:24])+bytes(m.common_staging[40:])
    m.count['phase_z16_vectors_written'] += 1


def load_z16(m, reg, address):
    held = bytes(m.common_staging[:24])+bytes(m.common_staging[40:])
    for offset in (0, 8):
        m.common_staging[24+offset:32+offset] = m.read_word(address+offset)
    m.advance(tag='phase_z16_response_collect')
    values = np.frombuffer(m.common_staging, '<i2', 8, 24).astype(np.int64)
    m.wait_reg(reg)
    m.advance(op=('ILOAD', reg, values), tag='phase_z16_H8_source_RF_load')
    m.wait_reg(reg)
    assert np.all(m.rf[reg] >= -256) and np.all(m.rf[reg] <= 255)
    assert held == bytes(m.common_staging[:24])+bytes(m.common_staging[40:])
    m.count['phase_z16_vectors_read'] += 1


def encode_decode_h8(m, input_base, h, phase_arm):
    """Original q8 encoder; return observer copies, never feeding arithmetic."""
    q_seen = np.empty((10, 8), np.int64)
    for t in range(10):
        reg = SOURCE+t
        m.phase = 'phase_actual_I24_H8_read'
        m.load_i24(reg, input_base+(t*96+h)*3)
        m.phase = 'phase_original_q_RNE_clip8'
        m.advance(op=('ISUB_REG', reg, PRED+t), tag='phase_actual_predictor_subtract')
        m.wait_reg(reg)
        m.advance(op=('IRNE', reg, 11), tag='phase_original_RNE_to_q')
        m.wait_reg(reg)
        m.advance(op=('ICLIP8', reg, None), tag='phase_original_clip8')
        m.wait_reg(reg)
        q_seen[t] = m.rf[reg].astype(np.int64)  # observation only
        if phase_arm:
            m.phase = 'phase_z9_encode_store16'
            new_issue(m, 'IFLOOR11', TEMP, PRED+t, 'phase_b_floor2048')
            m.wait_reg(TEMP)
            m.advance(op=('IADD', reg, TEMP), tag='phase_z_equals_q_plus_floor_b')
            m.wait_reg(reg)
            store_z16(m, reg, CODE+(t*96+h)*2)
        else:
            m.phase = 'table_direct_RF_reconstruction'
            m.advance(op=('IRNE', reg, -11), tag='table_q_dyadic_shift')
            m.wait_reg(reg)
            m.advance(op=('IADD', reg, PRED+t), tag='table_RF_reconstruction_add_b')
            m.wait_reg(reg)
            m.advance(op=('ISAT', reg, None), tag='table_RF_reconstruction_sat24')
            m.wait_reg(reg)
    if phase_arm:
        for t in range(10):
            reg = SOURCE+t
            m.phase = 'phase_actual_z16_H8_read'
            load_z16(m, reg, CODE+(t*96+h)*2)
            m.phase = 'phase_RF_reconstruction'
            new_issue(m, 'IPHASE11', TEMP, PRED+t, 'phase_b_low11')
            m.wait_reg(TEMP)
            m.advance(op=('IRNE', reg, -11), tag='phase_z_dyadic_shift')
            m.wait_reg(reg)
            m.advance(op=('IADD', reg, TEMP), tag='phase_RF_reconstruction_add_phi')
            m.wait_reg(reg)
            m.advance(op=('ISAT', reg, None), tag='phase_RF_reconstruction_sat24')
            m.wait_reg(reg)
    return q_seen, m.rf[SOURCE:SOURCE+10].astype(np.int64).copy()


def u_h8(m, base, h):
    """Exact kernel.u_stage H8 resident MAC loop; sources already in RF60..69."""
    m.phase = 'phase_common_H8_U_resident_MAC'
    for k in range(h, h+8):
        for hg in range(3):
            address = base['U_ped']+(k*24+hg*8)*2
            m.coefficient(address)
            if not np.frombuffer(m.cword, '<i2', 8, address % 32).any():
                continue
            for t in range(10):
                dst, src, lane = t*3+hg, SOURCE+t, k % 8
                if m.rf[src, lane] == 0:
                    m.advance(tag='ordinary_zero_bypass')
                else:
                    m.wait_reg(dst)
                    m.advance(op=('IMAC_INDEX', dst, (address % 32, src, lane)), tag='U_resident_MAC')


def make_callback(p, image, phase_arm):
    def callback(m, base, q, positions, geo):
        load_table(m, image)
        us, peds = [], []
        for ip, (y, x) in enumerate(positions):
            input_base = binding.UPDATED+ip*2880
            gate_address = binding.PROJ+((y-geo['gate_origin'][0])*geo['gate_shape'][1]+x-geo['gate_origin'][1])*192
            observed_x = binding.read24(m, input_base, (10, 96))
            observed_words = np.frombuffer(m.state, '<u2', 96, gate_address).astype(np.int64)
            observed_q = np.empty((10, 96), np.int64)
            observed_recon = np.empty((10, 96), np.int64)
            m.phase = 'phase_common_retained_U_clear'
            for reg in range(30):
                m.wait_reg(reg)
                m.advance(op=('clear', reg, None), tag='Uq_Ug_clear')
            for h in range(0, 96, 8):
                begin = m.time
                table_predict_h8(m, gate_address+2*h)
                predicted = m.time
                q_seen, reconstruction = encode_decode_h8(m, input_base, h, phase_arm)
                observed_q[:, h:h+8] = q_seen
                observed_recon[:, h:h+8] = reconstruction
                reconstructed = m.time
                u_h8(m, base, h)
                m.phase_h8_events.append(dict(position=[y, x], channel=h,
                    start=begin, b_ready=predicted, RF_reconstruction_ready=reconstructed,
                    last_U_issue_end=m.time, source_RF=[60, 70], prediction_RF=[40, 50],
                    U_accumulator_RF=[0, 30], same_H8_b_retained=True))
            m.drain()
            m.phase = 'original_U_RNE_sat'
            for t in range(10):
                for hg in range(3):
                    binding.integer.complete(m, t*3+hg, int(q['U_ped_exponent']))
            actual_u = m.rf[:30].astype(np.int64).copy().reshape(10, 24)
            m.encoded_u_ready = m.time
            us.append(actual_u)
            out = binding.PED_V+ip*2880
            kernel.v_stage(m, base, q, out)
            peds.append(binding.read24(m, out, (10, 96)))
            # All following arrays are final numerical observers, not input.
            observed_g = np.stack([(observed_words >> t) & 1 for t in range(10)])
            expected_recon, expected_q, _ = reconstruct(observed_x, observed_g, p)
            assert np.array_equal(observed_q, expected_q)
            assert np.array_equal(observed_recon, expected_recon)
            m.count['phase_observed_q8_values_compared'] += 960
            m.count['phase_observed_RF_RECON_values_compared'] += 960
            if phase_arm:
                b = p['D'] @ observed_g+p['c'][:, None]
                expected_z = expected_q+np.floor_divide(b, 2048)
                physical_z = np.frombuffer(m.state, '<i2', 960, CODE).reshape(10, 96)
                assert np.array_equal(physical_z, expected_z)
                m.count['phase_observed_physical_CODE16_values_compared'] += 960
        return np.asarray(us), np.asarray(peds)
    return callback


def oracle(data, q, label, p):
    geo = json.loads(str(data['window_geometry_json']))[label]
    dy, dx = [2*a-b for a, b in zip(geo['output_origin'], geo['gate_origin'])]
    x = data[label+'_updated_I24'][:, :, dy:dy+8:2, dx:dx+8:2].astype(np.int64)
    g = data[label+'_proj_gate'][:, :, dy:dy+8:2, dx:dx+8:2].astype(np.int64)
    estimate, _, _ = reconstruct(x, g, p)
    u = rne_sat(np.einsum('rk,tkij->trij', q['U_ped_q16'].astype(np.int64), estimate,
                         dtype=np.int64), int(q['U_ped_exponent']))
    acc = np.einsum('hr,trij->thij', q['V_ped_q16'].astype(np.int64), u, dtype=np.int64)
    ped = rne_sat(rne_sat(acc, int(q['V_ped_exponent']))+q['PED_bias_q24'][None, :, None, None], 0)
    return dict(continuous=ped, U_ped=u)


def run():
    axis, label = 'ordinary', 'corner'
    data, live, q, params = binding.load(axis, label)
    p = params['full_g_q8']; image = table_image(p)
    archived = json.loads((HERE/'ordinary_corner_final.json').read_text())
    print('PHASE_PREFIX_START', axis, label, flush=True)
    _, producer, prefix = binding.run_prefix(PhaseMachine, data, live, axis, label, False)
    assert producer['service_slots'] == archived['producer']['service_slots']
    assert producer['counts'] == archived['producer']['counts']
    assert producer['checks'] == archived['producer']['checks']
    print('PHASE_PREFIX_DONE', prefix.time, flush=True)
    gold = oracle(data, q, label, p)
    result = dict(axis=axis, window=label, stress=False, producer=producer, rows=[], complete=False,
        scope=__doc__, exact_function='Existing old R24 full_g_q8; original q RNE/clip8, U/V RNE/bias/sat.',
        resources=dict(RF_vectors=96, lanes=8, bits=48, state_bytes=131072,
                       coefficient_bytes=131072, staging_bytes=64, issue=1, SR64=1, SW64=1, CR256=1),
        layout=dict(original_coefficient_bytes=41024, table_coefficient_range=[TABLE, 131072],
            table_rows=1024, row_bytes=32, fields_per_row=10, field_signed_bits=20,
            CODE_state_range=[CODE, CODE+1920], CODE_physical_bits=16, CODE_required_signed_bits=9,
            U_RF=[0, 30], b_RF=[40, 50], gate_RF=50, temp_RF=51, H8_source_RF=[60, 70],
            U_max_simultaneous_live_vectors=52, V_max_simultaneous_live_vectors=90,
            gate_collector_staging=[0, 16], existing_source_staging=[0, 24], CODE_staging=[24, 40],
            RECON_SRAM_materialized=False, phase_b_relookup=False),
        new_operations=dict(names=list(NEW_OPS), issue_slots=1, RF_writeback_latency_slots=2,
            latency_checked_on_every_issue=True, extra_hardware='Signed20 CR256 field selector plus sign extension/broadcast or RF lane update; signed floor-by2048 and low11 extraction datapaths.',
            timing_status='Model assumption, no RTL/STA/EDA validation; dependent waits and shared arbitration actually execute.'),
        prefix_archived_counts_reproduced=True, new_training=False, new_AEE=False, EDA=False,
        production_modified=False,
        attribution='Both new arms gain LUT prediction and H8 direct-RF reconstruction. Only the within-pair delta isolates the signed16 z SRAM boundary. Old full_expanded also materializes RECON, so its difference is a compound fusion change.')
    for arm, phase_arm in (('table_RF_reconstruct', False), ('phase_z16_SRAM', True)):
        m = copy.deepcopy(prefix)
        print('PHASE_ARM_START', arm, flush=True)
        _, report = binding.consumer_run(data, q, label, make_callback(p, image, phase_arm),
                                         machine=m, gold=gold)
        assert sum(m.stages.values()) == m.time
        assert m.count['phase_LUT_H8_calls'] == 16*12
        assert m.count['phase_observed_q8_values_compared'] == 15360
        assert m.count['phase_observed_RF_RECON_values_compared'] == 15360
        assert m.count.get('phase_z16_vectors_written', 0) == (1920 if phase_arm else 0)
        assert m.count.get('phase_z16_vectors_read', 0) == (1920 if phase_arm else 0)
        counts = dict(m.count)
        row = dict(arm=arm, service_slots=m.time, consumer_service_slots=report['service_slots'],
            producer_end=prefix.time, same_executed_prefix_copied=True, consumer=report,
            counts=counts, stages=dict(m.stages),
            port_bytes=dict(SR64=8*counts['SR64_reads'], SW64=8*counts['SW64_writes'],
                            CR256=32*counts['CR256_reads'], CW256=32*counts['CW256_writes']),
            representation_checks=dict(actual_RF_q8_values=15360, q8_differences=0,
                actual_RF_RECON_values=15360, RECON_differences=0,
                actual_physical_CODE16_values=15360 if phase_arm else 0, CODE16_differences=0,
                original_U_values=3840, U_differences=0, original_PED_values=15360, PED_differences=0),
            table_events=m.phase_lut_events, H8_timeline=m.phase_h8_events,
            checked_new_two_slot_issues=counts['phase_checked_two_slot_new_issues'],
            original_V_function='kernel.v_stage, unchanged P1 retained U / H48 outputs',
            U_function='Exact kernel.u_stage H8 resident MAC body, supplied by actual RF reconstruction',
            full_layer=False, full_network=False)
        result['rows'].append(row)
        (HERE/'phase_execution.json').write_text(json.dumps(result, indent=2)+'\n')
        print('PHASE_ARM_DONE', arm, m.time, report['service_slots'], row['port_bytes'], flush=True)
    first, phase = result['rows']
    # Same cold fill, physical gate reads, table selections and b reconstruction.
    for key in ('CR256', 'CW256'):
        assert first['port_bytes'][key] == phase['port_bytes'][key]
    for key in ('phase_LUT_H8_calls', 'phase_gword_zero_bypass', 'phase_LUT_lane_values',
                'phase_LUT_broadcast_values', 'phase_gword_selector_RF_reads'):
        assert first['counts'][key] == phase['counts'][key]
    result['within_pair'] = dict(phase_minus_table_RF_slots=phase['service_slots']-first['service_slots'],
        phase_percent_vs_table_RF=100*(phase['service_slots']/first['service_slots']-1),
        phase_consumer_percent_vs_table_RF=100*(phase['consumer_service_slots']/first['consumer_service_slots']-1),
        port_byte_delta={key: phase['port_bytes'][key]-first['port_bytes'][key] for key in first['port_bytes']},
        same_LUT_cold_fill_and_selections=True,
        attribution='Only this pair judges added physical16bit z materialization and phase decoding; no benefit assigned to mod identity.')
    controls = []
    for ref in archived['rows']:
        if ref['arm'] not in ('full_expanded', 'full_code8', 'fixed_code8', 'affine_code8'):
            continue
        controls.append(dict(arm=ref['arm'], source_file='ordinary_corner_final.json',
            service_slots=ref['service_slots'], consumer_service_slots=ref['consumer_service_slots'],
            port_bytes=ref['port_bytes'], rerun=False,
            new_table_RF_percent_vs_control=100*(first['service_slots']/ref['service_slots']-1),
            new_phase_percent_vs_control=100*(phase['service_slots']/ref['service_slots']-1)))
    result['archived_controls'] = controls
    result['complete'] = True
    (HERE/'phase_execution.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PHASE_EXECUTION_COMPLETE', json.dumps(result['within_pair']), flush=True)


if __name__ == '__main__':
    run()
