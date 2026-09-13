"""Fixed 24-entry cross-frame certified source gate cache, paid CPU execution.

History and predicates use actual SRAM/RF. Reference arrays only check outputs.
No new trained function or claimed RTL/PPA result.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
OPEN = HERE.parents[1]
sys.path.insert(0, str(OPEN/'breadth_20260912/hardware/matched_local_chain'))
import run as matched
sys.path.insert(0, str(OPEN/'production_fusions_20260913/encoded_consumer'))
import binding
import kernel

stage = matched.stage
common = matched.common
SOURCE, INPUT = stage.SRC, 90112
HISTORY, STRIDE, CONSTANTS = 114688, 264, 126976
RADII = (64, 256, 1024)
PROGRAM = json.loads((OPEN/'breadth_20260912/source_execution/dense/program.json').read_text())
assert [x['source_t'] for x in PROGRAM[:10]] == list(range(10))
assert [x['dst'] for x in PROGRAM[:10]] == list(range(10))
assert max(x.get('dst', 0) for x in PROGRAM) < 70


class CacheMachine(kernel.QuantMachine):
    def integer_op(self, kind, dst, args):
        a = self.rf[dst].astype(np.int64)
        if kind == 'ICACHE_DIFF':
            x, y = args
            assert self.ready[x] <= self.time and self.ready[y] <= self.time
            v = self.rf[x].astype(np.int64)-self.rf[y].astype(np.int64)
        elif kind == 'ICACHE_ABS':
            v = np.abs(a)
        elif kind == 'ICACHE_MAX':
            assert self.ready[args] <= self.time
            v = np.maximum(a, self.rf[args].astype(np.int64))
        elif kind == 'ICACHE_AND':
            assert self.ready[args] <= self.time
            v = a & self.rf[args].astype(np.int64)
        elif kind == 'ICACHE_REDUCE':
            step, how = args
            v = (np.maximum(a, np.roll(a, step)) if how == 'max'
                 else a & np.roll(a, step))
        elif kind == 'ICACHE_DOT':
            p = args
            assert self.ready[p['reg']] <= self.time
            v = (self.rf[p['reg']].astype(np.int64) << p['shift'])*p['sign']
        elif kind == 'ICACHE_THRESHOLD':
            threshold, direction = args
            v = direction*(a-threshold)
        elif kind == 'ICACHE_MARGIN':
            # Gate is direction*(S-K)>=0. Inclusive safe integer radius.
            v = np.where(a >= 0, a, -a-1)
        elif kind == 'ICACHE_BOUND':
            margin, offset = args
            assert self.ready[margin] <= self.time
            bound = int.from_bytes(self.cword[offset:offset+6], 'little', signed=True)
            v = (self.rf[margin].astype(np.int64) >= bound).astype(np.int64)
        else:
            return super().integer_op(kind, dst, args)
        assert np.all(v >= -(1 << 47)) and np.all(v < (1 << 47)), kind
        return np.asarray(v, np.float64)


def issue(m, kind, dst, args=None):
    m.wait_reg(dst)
    m.advance(op=(kind, dst, args), tag=kind)
    m.wait_reg(dst)


def difference(x, y):
    return common.difference(x, y)


def cache_setup(m, q):
    if m.cache_mode == 'raw' or getattr(m, 'cache_initialized', False):
        return
    m.phase = 'source_cache_cold_metadata'
    for entry in range(24):
        m.advance(write=(HISTORY+entry*STRIDE+256, bytes(8)), tag='cache_valid_clear')
    if m.cache_mode == 'certified':
        row_l1 = np.abs(q['As_q16'].astype(np.int64)).sum(1)
        image = b''.join(b''.join(int(l*d).to_bytes(6, 'little', signed=True)
                                 for d in RADII)+bytes(14) for l in row_l1)
        m.phase = 'source_certificate_constant_cold_fill'
        m.dma_input(image, CONSTANTS, True)
    m.cache_initialized = True


def query(m, entry):
    base = HISTORY+entry*STRIDE
    m.phase = 'source_cache_query'
    word = m.read_word(base+256)
    # Metadata survives later SR responses in the already budgeted RF94.
    issue(m, 'ILOAD', 94, [int.from_bytes(word[:4], 'little'),
                          int.from_bytes(word[4:], 'little')]+[0]*6)
    m.advance(tag='cache_metadata_decode')
    if not int(m.rf[94, 0]):
        return False, None, 0, True
    issue(m, 'ILOAD', 72, [0]*8)
    for t in range(10):
        m.load_i24(70, base+t*24)
        issue(m, 'ICACHE_DIFF', 71, (t, 70))
        issue(m, 'ICACHE_ABS', 71)
        issue(m, 'ICACHE_MAX', 72, 71)
    for step in (1, 2, 4):
        issue(m, 'ICACHE_REDUCE', 72, (step, 'max'))
    maximum = int(m.rf[72, 0])
    radius = int(m.rf[94, 1])
    m.advance(tag='cache_radius_branch')
    hit = maximum <= (radius if m.cache_mode == 'certified' else 0)
    return hit, maximum, radius, False


def begin_certificate(m):
    m.phase = 'source_certificate_initialize'
    for r in (90, 91, 92):
        issue(m, 'ILOAD', r, [1]*8)


def observe_gate(m, ins):
    """Paid arithmetic consumes the actual source dot BEFORE RF reuse."""
    m.phase = 'source_certificate_margin'
    if ins['constant'] >= 0:
        m.advance(tag='source_constant_gate_no_bound')
        return
    assert len(ins['operands']) == 1
    issue(m, 'ICACHE_DOT', 73, ins['operands'][0])
    # External observer only, never used to pick a radius or controller path.
    m.cache_observed_dots.append((m.cache_pixel, m.cache_h, ins['output_t'],
                                  m.rf[73].astype(np.int64).copy()))
    issue(m, 'ICACHE_THRESHOLD', 73, (ins['threshold'], ins['direction']))
    issue(m, 'ICACHE_MARGIN', 73)
    m.coefficient(CONSTANTS+32*ins['output_t'])
    for i in range(3):
        issue(m, 'ICACHE_BOUND', 74, (73, i*6))
        issue(m, 'ICACHE_AND', 90+i, 74)


def finish_certificate(m):
    m.phase = 'source_certificate_reduce'
    for r in (90, 91, 92):
        for step in (1, 2, 4):
            issue(m, 'ICACHE_REDUCE', r, (step, 'and'))
    # At most two RF sources per slot; do not read three vectors for free.
    m.advance(tag='cache_radius_read_pair')
    m.scalar_collector = max([0]+[RADII[i] for i in (0, 1) if int(m.rf[90+i, 0])])
    m.advance(tag='cache_radius_read_final_and_encode')
    return RADII[2] if int(m.rf[92, 0]) else m.scalar_collector


def source(m, data, axis, label):
    start = m.time; before = dict(m.count)
    cache_setup(m, m.source_q)
    identity = data[label+'_I24']; _, _, height, width = identity.shape
    m.sn1_spatial = True
    rows = []
    m.cache_observed_dots = []
    for y in range(height):
        for x in range(width):
            pixel = y*width+x
            m.phase = 'source_I24_input'
            m.dma_input(binding.pack24(identity[:, :, y, x]), INPUT)
            for h in range(0, 96, 8):
                begin = m.time
                selected = pixel < 2 and m.cache_mode != 'raw'
                entry = pixel*12+h//8
                m.cache_pixel, m.cache_h = pixel, h
                m.phase = 'source_compiled_temporal_program'
                # Same physical initial reads for all arms. Cache fallback
                # keeps these ten RFs, so it never repeats the input loads.
                for ins in PROGRAM[:10]:
                    m.count['source_ROM128_fetches'] += 1
                    m.load_i24(ins['dst'], INPUT+(ins['source_t']*96+h)*3)
                hit = False; maximum = None; radius = 0; cold = False
                if selected:
                    hit, maximum, radius, cold = query(m, entry)
                if hit:
                    m.phase = 'source_cache_gate_return'
                    base = HISTORY+entry*STRIDE
                    payload = b''.join(m.read_word(base+240+j) for j in (0, 8))
                    address = SOURCE+(pixel*96+h)*2
                    for j in (0, 8):
                        m.advance(write=(address+j, payload[j:j+8]), tag='source_gate_word_store')
                else:
                    if selected:
                        m.phase = 'source_cache_reference_refresh'
                        for t in range(10):
                            m.store_i24(t, HISTORY+entry*STRIDE+t*24)
                        if m.cache_mode == 'certified':
                            begin_certificate(m)
                    words = np.zeros(8, np.uint16)
                    for ins in PROGRAM[10:]:
                        kind = ins['kind']; m.count['source_ROM128_fetches'] += 1
                        m.phase = 'source_compiled_temporal_program'
                        if kind == 'nop':
                            m.advance(tag='source_scheduled_nop')
                        elif kind == 'commit':
                            payload = words.astype('<u2').tobytes()
                            address = SOURCE+(pixel*96+h)*2
                            for j in (0, 8):
                                m.advance(write=(address+j, payload[j:j+8]), tag='source_gate_word_store')
                        else:
                            for p in ins['operands']:
                                m.wait_reg(p['reg'])
                            if kind == 'gate' and selected and m.cache_mode == 'certified':
                                observe_gate(m, ins)
                                m.phase = 'source_compiled_temporal_program'
                            dst = 95 if kind == 'gate' else ins['dst']
                            m.wait_reg(dst)
                            m.advance(op=('ISOURCE', dst, ins), tag='source_'+kind)
                            if kind == 'gate':
                                m.wait_reg(dst)
                                words |= m.rf[dst].astype(np.uint16) << ins['output_t']
                                m.advance(tag='source_gate_collector_RF_read')
                    if selected:
                        radius = finish_certificate(m) if m.cache_mode == 'certified' else 0
                        m.phase = 'source_cache_gate_reference_commit'
                        payload = words.astype('<u2').tobytes()
                        base = HISTORY+entry*STRIDE
                        for j in (0, 8):
                            m.advance(write=(base+240+j, payload[j:j+8]), tag='cache_gate_reference_write')
                        meta = (1).to_bytes(4, 'little')+radius.to_bytes(4, 'little')
                        m.advance(write=(base+256, meta), tag='cache_reference_valid_commit')
                if pixel < 2:
                    rows.append(dict(entry=entry, hit=hit, cold=cold, delta_max=maximum,
                                     radius=radius, service_slots=m.time-begin,
                                     nonzero_change_hit=bool(hit and maximum > 0)))
    m.drain()
    actual = np.frombuffer(m.state, '<u2', height*width*96, SOURCE).reshape(height, width, 96).copy()
    gate = np.stack([(actual >> t) & 1 for t in range(10)]).transpose(0, 3, 1, 2).astype(bool)
    checks = difference(gate, data[label+'_sn1_gate'])
    assert checks['differences'] == 0, checks
    s_key = label+'_source_S48'
    dot_values = 0
    if s_key in data:
        for pixel, h, t, actual_dot in m.cache_observed_dots:
            yy, xx = divmod(pixel, width)
            assert np.array_equal(actual_dot, data[s_key][t, h:h+8, yy, xx]), (pixel, h, t)
            dot_values += 8
    return dict(service_slots=m.time-start, checks=checks, rows=rows,
                actual_dot_values_checked=dot_values, hits=sum(r['hit'] for r in rows),
                nonzero_change_hits=sum(r['nonzero_change_hit'] for r in rows),
                counts={k:v-before.get(k, 0) for k,v in m.count.items() if v-before.get(k, 0)},
                cache_bytes=24*STRIDE, selected_H8=24, total_H8=height*width*12)


def prepare(path, live, q, label):
    data = common.read_npz(path)
    geo = json.loads(str(data['window_geometry_json']))[label]
    gate = matched.source_gold(data[label+'_I24'], q, 'dense')
    assert np.array_equal(gate, data[label+'_sn1_gate'])
    preview, margin = matched.preview_gold(gate, live, geo)
    expected, _ = matched.oracle.independent_gold(data, q, label, preview['gate'])
    view = dict(data)
    view.update({label+'_preview_Z_shared':preview['z'], label+'_preview_shared_raw':preview['raw'],
                 label+'_preview_BN1_Y':preview['y'], label+'_sn2_gate':preview['gate'],
                 label+'_updated_I24':expected['updated'], label+'_proj_gate':expected['gate'],
                 label+'_continuous_q24':expected['continuous']})
    captured = {key:difference(view[label+'_'+key], data[label+'_'+key])
                for key in ('sn2_gate', 'updated_I24', 'proj_gate', 'continuous_q24')}
    return view, captured, margin


def run(args):
    folder = HERE.parent/'capture_owned'
    q = common.read_npz(folder/'deployed_constants.npz')
    live = common.read_npz(folder/'live_parameters.npz')
    paths = [Path(p) for p in args.frames]
    m = CacheMachine(args.stress); m.cache_mode = args.mode; m.source_q = q
    matched.source_program.run = source
    matched.windows.Machine = lambda stress: m
    matched.windows.build_nrv = stage.preview_directory
    result = dict(mode=args.mode, stress=args.stress, scope='four independent frames; full interior local chain CPU service model',
                  history_bytes=24*STRIDE, cache_state_end=HISTORY+24*STRIDE,
                  frames=[], resource=dict(RF_vectors=96, lanes=8, signed_bits=48, state_bytes=131072,
                                          coefficient_bytes=131072, state_read_bits=64, state_write_bits=64,
                                          coefficient_read_bits=256, extra_ALU_area_unmeasured=True))
    begin_wall = time.monotonic()
    for frame, path in enumerate(paths):
        view, captured, margin = prepare(path, live, q, 'interior')
        start = m.time; before = dict(m.count); prior = dict(m.stages)
        _, prefix, _ = matched.windows.window(view, live, 'interior', False, args.stress, True, 'dense')
        assert all(x['differences'] == 0 for x in prefix['checks'].values()), prefix['checks']
        m.forward_i24 = True
        _, suffix = binding.consumer_run(view, q, 'interior', kernel.make_callback(None, False), machine=m, stress=args.stress)
        record = dict(frame=frame, input=str(path), service_slots=m.time-start,
                      start_phase=start%32, end_phase=m.time%32, source=prefix['source_program'],
                      checks=dict(preview=prefix['checks'], consumer=suffix['checks']),
                      captured_endpoint_comparison=captured, preview_margin=margin,
                      counts={k:v-before.get(k, 0) for k,v in m.count.items() if v-before.get(k, 0)},
                      stages={k:v-prior.get(k, 0) for k,v in m.stages.items() if v-prior.get(k, 0)})
        assert sum(record['stages'].values()) == record['service_slots']
        result['frames'].append(record)
        result['total_service_slots'] = m.time
        result['wall_seconds'] = time.monotonic()-begin_wall
        out = HERE/(args.mode+('_stress' if args.stress else '_ready')+'.json')
        out.write_text(json.dumps(result, indent=2)+'\n')
        print(args.mode, args.stress, frame, record['service_slots'], 'hits',record['source']['hits'], 'wall',round(result['wall_seconds']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('raw', 'exact', 'certified'), required=True)
    p.add_argument('--stress', action='store_true')
    p.add_argument('--frames', nargs='+', required=True)
    run(p.parse_args())
