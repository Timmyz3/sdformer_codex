"""Paid complete-K864 source-address interface; common finite resident consumer."""
from pathlib import Path
import argparse
import json
import sys
from collections import Counter
import numpy as np

HERE = Path(__file__).resolve().parent
PRUNING = HERE.parent
sys.path.insert(0, str(PRUNING))
import consumer_resident as resident
common = resident.common
integer_chain = resident.integer_chain
ORIGINAL_DIRECTORY = integer_chain.directory
MASK_ADDRESS = 122880
CACHE = 64
GEOMETRY = 80
MASK = 92
NRV = 93


class IteratorMachine(resident.ProbeMachine):
    def __init__(self, stress=False):
        super().__init__(stress)
        self.directory_trace = []
        self.directory_payloads = []
        self.directory_arm = 'original_scan'

    def integer_op(self, kind, dst, args):
        if kind == 'IMASK48':
            raw = int.from_bytes(self.sword[:6], 'little')
            value = [(raw >> (12*p)) & 4095 for p in range(4)] + [0]*4
        elif kind == 'IWORD_PAIR':
            ip, reset = args
            value = np.zeros(8, np.int64) if reset else self.rf[dst].astype(np.int64).copy()
            value[4*ip:4*ip+4] = np.frombuffer(self.sword, '<u2', count=4)
        elif kind == 'IGROUP_PERMIT':
            group = args
            value = self.rf[dst].astype(np.int64).copy()
            assert self.ready[MASK] <= self.time
            valid = int(value[6])
            permission = 0
            for ip in range(2):
                phase = int(value[2+ip])
                if (valid >> ip) & 1 and not ((int(self.rf[MASK, phase]) >> group) & 1):
                    permission |= 1 << ip
            value[7] = permission
        elif kind == 'INRV_CACHED':
            k, src = args
            assert self.ready[src] <= self.time
            lane = (k//9) % 4
            mask = int(self.rf[src, lane]) | (int(self.rf[src, lane+4]) << 10)
            value = [k, mask, int(self.rf[dst, 2]) | mask, 0, 0, 0, 0, 0]
        else:
            return super().integer_op(kind, dst, args)
        return np.asarray(value, np.float64)


def metadata_bytes(mask):
    group = np.asarray(mask, np.uint8).reshape(4, 12, 8)
    assert np.all(group == group[:, :, :1]), 'mask must be H8-uniform'
    packed = sum(int(group[p, g, 0]) << (p*12+g) for p in range(4) for g in range(12))
    return packed.to_bytes(6, 'little')


def trace_directory(m, start, before, positions, n, live, dir_base):
    payload = bytes(m.state[dir_base:dir_base+n*8])
    records = np.frombuffer(payload, '<u4').reshape(-1, 2)
    assert all(int(records[i, 0]) < int(records[i+1, 0]) for i in range(len(records)-1))
    assert not len(records) or (records[:, 1] != 0).all()
    m.directory_payloads.append((tuple(map(tuple, positions)), int(live), payload))
    m.directory_trace.append(dict(positions=positions, service_slots=m.time-start,
        records=int(n), live=int(live), counts=dict(m.count-before)))


def directory(m, geo, positions, gate_base=common.GATE, dir_base=integer_chain.DIR,
              phase='integer_NRVs_from_actual_sn2'):
    start = m.time
    before = Counter(m.count)
    arm = m.directory_arm
    if arm == 'original_scan':
        n, live = ORIGINAL_DIRECTORY(m, geo, positions, gate_base, dir_base, phase)
        trace_directory(m, start, before, positions, n, live, dir_base)
        return n, live
    m.phase = phase
    h, w = geo['gate_shape']
    oy, ox = geo['gate_origin']
    use_static = arm in ('static_predicate', 'address_iterator')
    if use_static:
        m.read_word(MASK_ADDRESS)
        m.advance(op=('IMASK48', MASK, None), tag='mask48_unpack_four_phase_rows')
        m.wait_reg(MASK)
    # Addresses, global source phase, and validity are kept in existing RF.
    # Two explicitly charged scalar controller slots per position/offset
    # cover coordinate/bounds/phase and row-major source address generation.
    valid_offsets = 0
    for rem in range(9):
        ky, kx = divmod(rem, 3)
        bases, phases, valid = [0, 0], [0, 0], 0
        for ip, (y, x) in enumerate(positions):
            m.advance(tag='source_coordinate_bounds_phase')
            sy, sx = y+ky-1, x+kx-1
            m.advance(tag='source_address_generation')
            if 0 <= sy < 240 and 0 <= sx < 320:
                ly, lx = sy-oy, sx-ox
                assert 0 <= ly < h and 0 <= lx < w
                bases[ip] = gate_base+((ly*w+lx)*96)*2
                phases[ip] = 2*(sy % 2)+sx % 2
                valid |= 1 << ip
        if use_static and m.uniform_mask and arm == 'address_iterator':
            m.advance(tag='uniform_boundary_offset_bitmap_merge')
            if valid:
                valid_offsets |= 1 << rem
        r = GEOMETRY+rem
        m.wait_reg(r)
        m.advance(op=('ILOAD', r, bases+phases+[0, 0, valid, valid]), tag='source_geometry_RF_write')
    m.drain()
    m.wait_reg(NRV)
    m.advance(op=('ILOAD', NRV, [0]*8), tag='directory_live_init')
    m.wait_reg(NRV)
    n = 0
    for c0 in range(0, 96, 4):
        m.advance(tag='H4_channel_counter')
        if use_static and c0 % 8 == 0:
            if m.uniform_mask:
                # Compile-time uniformity is applied by the same equality
                # rule to ANY mask; the selected bit still comes from the
                # paid actual SRAM metadata decoded into RF92.
                m.advance(tag='uniform_H8_mask_bit_select')
                group_enabled = not ((int(m.rf[MASK, 0]) >> (c0//8)) & 1)
                if arm == 'address_iterator':
                    m.advance(tag='uniform_H8_bounds_bitmap_select')
                    retained_offsets = valid_offsets if group_enabled else 0
            else:
                retained_offsets = 0
                for rem in range(9):
                    r = GEOMETRY+rem
                    m.advance(op=('IGROUP_PERMIT', r, c0//8), tag='H8_source_phase_mask_lookup')
                    m.wait_reg(r)
                    if arm == 'address_iterator':
                        m.advance(tag='H8_retained_offset_bitmap_merge')
                        if int(m.rf[r, 7]):
                            retained_offsets |= 1 << rem
        if use_static and m.uniform_mask:
            m.advance(tag='uniform_H8_zero_group_branch')
            if not group_enabled:
                continue
        # Same H4 word collector in every optimized arm. It consumes the
        # physical SR64 responses, not the host gate tensor. Two words fit
        # in one 8-lane RF vector as eight independent uint16 values.
        for rem in range(9):
            geo_reg = GEOMETRY+rem
            permissions = int(m.rf[geo_reg, 7 if use_static and not m.uniform_mask else 6])
            m.advance(tag='H4_offset_word_dispatch')
            if not permissions:
                continue
            reset = True
            for ip in range(len(positions)):
                if not ((permissions >> ip) & 1):
                    continue
                address = int(m.rf[geo_reg, ip])+2*c0
                m.read_word(address)
                r = CACHE+rem
                m.wait_reg(r)
                m.advance(op=('IWORD_PAIR', r, (ip, reset)), tag='SR64_to_H4_cache_decode')
                m.wait_reg(r)
                reset = False
        # Original c-major NRV order: no free sort or sparse gather buffer.
        for c in range(c0, c0+4):
            offsets = retained_offsets if arm == 'address_iterator' else 511
            for rem in range(9):
                if arm == 'address_iterator' and not ((offsets >> rem) & 1):
                    continue
                m.advance(tag=('retained_offset_priority_encode' if arm == 'address_iterator'
                               else 'K864_scan_select_and_predicate'))
                permissions = int(m.rf[GEOMETRY+rem, 7 if use_static and not m.uniform_mask else 6])
                if not permissions:
                    continue
                k = c*9+rem
                m.advance(op=('INRV_CACHED', NRV, (k, CACHE+rem)), tag='integer_NRV_decode')
                m.wait_reg(NRV)
                mask = int(m.rf[NRV, 1])
                if mask:
                    payload = int(m.rf[NRV, 0]).to_bytes(4, 'little')+mask.to_bytes(4, 'little')
                    m.advance(write=(dir_base+n*8, payload), tag='integer_NRV_write')
                    n += 1
    live = int(m.rf[NRV, 2])
    m.mark(phase, start)
    trace_directory(m, start, before, positions, n, live, dir_base)
    return n, live


def run_case(data, q, label, preview, expected, mask, arm, stress):
    candidate = dict(data)
    for field, name in [('updated_I24', 'updated'), ('proj_gate', 'gate'), ('continuous_q24', 'continuous')]:
        candidate[label+'_'+field] = expected[name]
    m = IteratorMachine(stress)
    m.forward_i24 = True
    m.directory_arm = arm
    m.uniform_mask = bool(np.all(mask == mask[:1]))
    words = sum(preview[t].astype(np.uint16) << t for t in range(10)).transpose(1, 2, 0)
    m.phase = 'sn2_continuation_input'
    m.dma_input(words.astype('<u2').tobytes(), common.GATE)
    if arm in ('static_predicate', 'address_iterator'):
        m.phase = 'source_mask_cold_metadata'
        m.dma_input(metadata_bytes(mask), MASK_ADDRESS)
    value, report = integer_chain.run(candidate, q, label, preview, machine=m, late_v=False)
    report['integer_service_without_input_DMA'] = report['service_slots']
    report['service_slots'] = m.time
    report['original_consumer_state_high_water'] = report.get('state_high_water')
    if arm in ('static_predicate', 'address_iterator'):
        report['state_high_water'] = max(report.get('state_high_water', 0), MASK_ADDRESS+32)
    report['uniform_mask_compile_rule_applied'] = bool(m.uniform_mask and arm in ('static_predicate', 'address_iterator'))
    report['counts'] = dict(m.count)
    report['stages'] = dict(m.stages)
    report['physical_port_bytes'] = dict(state_read=8*m.count['SR64_reads'],
        state_write=8*m.count['SW64_writes'], coefficient_read=32*m.count['CR256_reads'],
        coefficient_fill=32*m.count['CW256_writes'])
    report['directory_service'] = sum(v['service_slots'] for v in m.directory_trace)
    dc = sum((Counter(v['counts']) for v in m.directory_trace), Counter())
    report['directory_counts'] = dict(dc)
    report['directory_read_bytes'] = 8*dc['SR64_reads']
    report['directory_write_bytes'] = 8*dc['SW64_writes']
    report['directory_trace'] = m.directory_trace
    report['state_budget'] = dict(RF_vectors=96, vector_lanes=8, logical_H4_cache_bytes=144,
        allocated_cache_vectors=9, geometry_vectors=9, mask_vectors=1,
        coefficient_bytes=131072, state_bytes=131072, static_mask_padded_state_bytes=32 if arm in ('static_predicate', 'address_iterator') else 0,
        scalar_iterator_controller_bits=64, cache_uses_existing_RF=True)
    assert sum(report['stages'].values()) == m.time
    return value, report, m.directory_payloads


def reference_directories(preview, geo, trace):
    oy, ox = geo['gate_origin']
    out=[]
    for positions, _, _ in trace:
        records=[]; live=0
        for k in range(864):
            c, rem=divmod(k,9); ky,kx=divmod(rem,3); mask=0
            for ip,(y,x) in enumerate(positions):
                sy,sx=y+ky-1,x+kx-1
                if 0<=sy<240 and 0<=sx<320:
                    word=sum(int(preview[t,c,sy-oy,sx-ox])<<t for t in range(10))
                    mask |= word << (10*ip)
            live |= mask
            if mask: records.append(k.to_bytes(4,'little')+mask.to_bytes(4,'little'))
        out.append((positions,live,b''.join(records)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--axis', choices=['ordinary', 'lifting_raw'])
    parser.add_argument('--window', choices=['corner', 'interior'])
    parser.add_argument('--mask', choices=['global_group2', 'phase_joint', 'row_phase_joint_pair'])
    parser.add_argument('--stress', action='store_true')
    parser.add_argument('--reuse-scans', action='store_true')
    args = parser.parse_args()
    masks0 = json.loads((PRUNING.parent/'review/phase_group8_masks.json').read_text())
    masks1 = json.loads((PRUNING/'paired_phase_masks.json').read_text())
    integer_chain.directory = directory
    integer_chain.dense = resident.dense_adapter
    axes = [args.axis] if args.axis else ['ordinary', 'lifting_raw']
    labels = [args.window] if args.window else ['corner', 'interior']
    names = [args.mask] if args.mask else ['global_group2', 'phase_joint', 'row_phase_joint_pair']
    document = dict(evidence='CPU payload slot prototype; not RTL/PPA or full-frame execution.',
        scope='Real emitted masked sn2 gates -> complete K864 U16/F/BN2+original rawI24 -> projection gate and real PED_U32/V96, fixed 16 anchors plus gate-window raw-I24 consumers.',
        no_native_projection_no_globalBN=True, masks_already_evaluated_no_training=True,
        same_RF_ports_and_resident_MAC=True, ordinary_word_coalescing_and_static_skip_are_not_X=True,
        stress=args.stress, rows=[])
    output = HERE/('results'+''.join('_'+s for s in [args.axis, args.window, args.mask] if s)+('_stress' if args.stress else '')+'.json')
    previous = json.loads(output.read_text()) if args.reuse_scans else None
    for axis in axes:
        capture = common.FULL/'capture'/axis
        data = common.read_npz(capture/'000_zurich_city_09_a_0001.npz')
        q = common.read_npz(capture/'parameters.npz')
        assert masks0[axis]['global_group2']['mask_uint8'] == masks1[axis]['global_joint_pair']['mask_uint8']
        for label in labels:
            original, _ = common.independent_gold(data, q, label, data[label+'_sn2_gate'])
            for key, field in [('updated','updated_I24'),('gate','proj_gate'),('continuous','continuous_q24')]:
                assert np.array_equal(original[key], data[label+'_'+field])
            for name in names:
                mask = np.asarray((masks1 if name == 'row_phase_joint_pair' else masks0)[axis][name]['mask_uint8'], np.uint8)
                preview = common.read_npz(PRUNING/f'{axis}_{label}_{name}.npz')['gate']
                # Verify the advertised zero proof directly on every captured
                # global source position, including padding boundary phases.
                geo = json.loads(str(data['window_geometry_json']))[label]
                oy, ox = geo['gate_origin']
                for yy in range(preview.shape[2]):
                    for xx in range(preview.shape[3]):
                        phase = 2*((oy+yy)%2)+(ox+xx)%2
                        assert not preview[:, mask[phase].astype(bool), yy, xx].any()
                expected, metadata = common.independent_gold(data, q, label, preview)
                gold_dirs = None
                record = dict(axis=axis, window=label, mask=name, independent_oracle=metadata, arms={})
                for arm in ['original_scan', 'coalesced_scan', 'static_predicate', 'address_iterator']:
                    if args.reuse_scans and arm in ('original_scan', 'coalesced_scan'):
                        old = next(r for r in previous['rows'] if (r['axis'],r['window'],r['mask']) == (axis,label,name))
                        record['arms'][arm] = old['arms'][arm]
                        continue
                    value, report, directories = run_case(data,q,label,preview,expected,mask,arm,args.stress)
                    if gold_dirs is None:
                        if args.reuse_scans:
                            gold_dirs = reference_directories(preview, geo, directories)
                        else:
                            gold_dirs = directories
                    assert directories == gold_dirs, (axis,label,name,arm,'NRV/live mismatch')
                    report['NRV_byte_exact_to_same_mask_original_scan'] = True
                    report['NRV_directory_calls'] = len(directories)
                    report['changed_network_delta_vs_original_capture'] = common.metrics(
                        (value['updated'],value['gate'],value['continuous']),
                        (original['updated'],original['gate'],original['continuous']))
                    record['arms'][arm] = report
                    print(axis,label,name,arm,report['service_slots'],report['directory_service'],report['directory_read_bytes'],flush=True)
                    document['rows'] = [r for r in document['rows'] if (r['axis'],r['window'],r['mask']) != (axis,label,name)]+[record]
                    output.write_text(json.dumps(document,indent=2)+'\n')
    print('SOURCE_ADDRESS_DONE',output,flush=True)

if __name__ == '__main__':
    main()
