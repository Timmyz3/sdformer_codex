"""Paid finite NRV/weight frontend replay; not the old RTL's total cycles."""
from pathlib import Path
from collections import Counter
import json
import struct
import numpy as np

HERE = Path(__file__).resolve().parent
OPEN = HERE.parents[1]
ROOT = OPEN.parent
FIXTURE = ROOT / 'psn/rtl/gp_slice/intersection_cases.bin'


def load_cases():
    data = FIXTURE.read_bytes()
    assert data[:4] == b'GPS1'
    pos = 8
    def get(fmt):
        nonlocal pos
        n = struct.calcsize(fmt)
        value = struct.unpack_from(fmt, data, pos)
        pos += n
        return value
    def array(dtype, shape):
        nonlocal pos
        n = int(np.prod(shape))
        v = np.frombuffer(data, dtype, n, pos).copy().reshape(shape)
        pos += v.nbytes
        return v
    cases = []
    for _ in range(struct.unpack_from('<I', data, 4)[0]):
        n, = get('<H')
        name = data[pos:pos+n].decode(); pos += n
        rank, real, temporal = get('<BBB')
        theta = array('<u4', (2,))
        table = array('u1', (8,))
        length, = get('<H')
        program = array('<u2', (length,))
        fields = dict(coeff=array('<i4', (10, 7)), tau=array('<i4', (2, 10)),
                      W=array('i1', (2, 384)), codes=array('u1', (4, 4, 384)),
                      S=array('<i4', (2, 4, 4, 7)), present=array('u1', (2, 4, 4, 7)),
                      gates=array('<u2', (2, 4, 4)))
        cases.append(dict(name=name, rank=rank, real=bool(real), temporal=bool(temporal),
                          theta=theta, table=table, program=program, **fields))
    assert pos == len(data)
    return cases


def source_scan(codes, stress):
    """Four banks: paired 2R32 scan; explicit12bit words span two returns."""
    memory = []
    for k in range(4):
        bits = sum(int(codes[k, p, c]) << (12*c+3*p) for c in range(384) for p in range(4))
        memory.append(bits.to_bytes(576, 'little'))
    decoded = np.zeros_like(codes)
    nrv = [[] for _ in range(4)]
    # Same serial scan/dispatch controller per bank; bank controllers run together.
    cycle = 0; reads = 0; loaded = -1; cache = [0]*4
    for c in range(384):
        needed = (12*c+11)//64
        while loaded < needed:
            while stress and cycle % 32 >= 24: cycle += 1
            loaded += 1
            for k in range(4):
                word = int.from_bytes(memory[k][loaded*8:loaded*8+8], 'little')
                cache[k] = cache[k] | (word << (loaded*64)) if loaded <= 1 else (cache[k] >> 64) | (word << 64)
                assert cache[k].bit_length() <= 128
            reads += 4
            cycle += 2  # request then response; no future source values exposed.
        for k in range(4):
            word = (cache[k] >> (c*12-max(loaded-1, 0)*64)) & 4095
            decoded[k, :, c] = [(word >> (3*p)) & 7 for p in range(4)]
            if word: nrv[k].append((c, tuple(int(v) for v in decoded[k, :, c])))
        cycle += 1  # actual row decode/zero predicate.
    assert np.array_equal(decoded, codes)
    # Materialize finite32bit records, two per64bit write. Subsequent NRV
    # requests are explicitly serviced by2R32 ports, one per tile per bank.
    rebuilt = []; nrv_bytes = 0; transfers = []
    for entries in nrv:
        payload = b''.join(struct.pack('<I', col | (sum(code << (3*p) for p, code in enumerate(word)) << 9))
                           for col, word in entries)
        payload += bytes((-len(payload)) % 8)
        words = [payload[i:i+8] for i in range(0, len(payload), 8)]
        readback = b''.join(words)
        new = []
        for i in range(len(entries)):
            value, = struct.unpack_from('<I', readback, 4*i)
            new.append((value & 511, tuple((value >> (9+3*p)) & 7 for p in range(4))))
        assert new == entries
        rebuilt.append((payload, len(entries))); nrv_bytes += len(payload); transfers.append(len(words))
    write_slots = max(transfers, default=0)
    # Store prefix is serial; consumer reads occur when each iterator needs
    # an entry. No free preloaded Python list substitutes for an NRV port.
    cycle += write_slots
    return rebuilt, dict(source_SR64_reads_equivalent=reads, source_SR32_reads=2*reads,
                     nrv_SW64_writes=sum(transfers), nrv_write_slots=write_slots,
                     source_scan_slots=cycle, source_memory_bytes=2304,
                     nrv_buffer_bytes=nrv_bytes, nrv_rows=sum(map(len, nrv)))


def row_memory(weight, indexed):
    if not indexed: return weight.tobytes()+bytes(2048-384)
    nz = np.flatnonzero(weight)
    blob = struct.pack('<H', len(nz))
    for k in nz: blob += struct.pack('<Hb', int(k), int(weight[k]))
    assert len(blob)+92+128 <= 2048  # cache plus conservatively reserved controller state.
    return blob + bytes(2048-len(blob))


def task(nrv_count, indexed, consume):
    if not indexed:
        for i in range(nrv_count):
            col, codes = yield ('nrv', i)
            raw = yield ('read', col)
            consume(col, codes, raw if raw < 128 else raw-256)
        return
    lo = yield ('read', 0)
    hi = yield ('read', 1)
    count = lo + (hi << 8)
    assert count <= 384
    pointer = 0; current = None
    yield ('control', 1)
    for i in range(nrv_count):
        col, codes = yield ('nrv', i)
        while pointer < count:
            if current is None:
                lo = yield ('read', 2+3*pointer)
                hi = yield ('read', 3+3*pointer)
                current = lo + (hi << 8)
                assert 0 <= current < 384
            yield ('control', 1)  # paid index/NRV comparison and cursor decision.
            if current < col:
                pointer += 1; current = None
            else: break
        if pointer == count: break
        if current == col:
            raw = yield ('read', 4+3*pointer)
            consume(col, codes, raw if raw < 128 else raw-256)
            pointer += 1; current = None


def simulate_tile(memory, nrv, indexed, shared, table, rank, stress):
    result = np.zeros((4, 4, 7), np.int64)
    touched = np.zeros_like(result, bool)
    counts = Counter()
    def consume(k):
        def apply(col, codes, w):
            counts['delivered_weight_values'] += 1
            if w:
                for p, code in enumerate(codes):
                    mask = int(table[code])
                    for r in range(rank):
                        if (mask >> r) & 1:
                            result[k, p, r] += w
                            touched[k, p, r] = True
                            counts['required_S_additions_unpriced'] += 1
        return apply
    generators = [task(nrv[k][1], indexed, consume(k)) for k in range(4)]
    waiting = {}; nrv_waiting = {}; done = set(); events = []; cache = {}; stamp = 0
    fill = None; rr = 0; cycle = 0
    def resume(k, data=None, initial=False):
        try:
            kind, val = next(generators[k]) if initial else generators[k].send(data)
            if kind == 'read': waiting[k] = val; counts['logical_byte_requests'] += 1
            elif kind == 'nrv': nrv_waiting[k] = val
            else: events.append((cycle+val, 'control', k, None)); counts['cursor_control_slots'] += val
        except StopIteration: done.add(k)
    for k in range(4): resume(k, initial=True)
    while len(done) < 4 or events or fill is not None:
        assert cycle < 100000
        assert len(waiting) <= 4 and len(nrv_waiting) <= 4 and len(cache) <= 16
        assert not set(waiting).intersection(nrv_waiting)
        due = [e for e in events if e[0] == cycle]
        events = [e for e in events if e[0] != cycle]
        for _, kind, owner, payload in due:
            if kind == 'fill':
                fill['bytes'][owner] = payload
            else: resume(owner, payload)
        # Each source bank has2R32, partitioned one port per tile. Each tile
        # requests at most one32bit entry per bank per slot. Tile traces can
        # therefore execute together with no hidden cross-tile oversubscription.
        if not stress or cycle % 32 < 24:
            for k, i in list(nrv_waiting.items()):
                value, = struct.unpack_from('<I', nrv[k][0], 4*i)
                payload = (value & 511, tuple((value >> (9+3*p)) & 7 for p in range(4)))
                events.append((cycle+1, 'nrv', k, payload))
                del nrv_waiting[k]; counts['physical_NRV32_reads'] += 1
        elif nrv_waiting: counts['blocked_NRV_slots'] += 1
        if fill is not None and len(fill['bytes']) == 4:
            stamp += 1
            if len(cache) == 16: del cache[min(cache, key=lambda b: cache[b][1])]
            block = fill['block']; blob = bytes(fill['bytes'][j] for j in range(4))
            cache[block] = (blob, stamp)
            for k in fill['owners']:
                events.append((cycle+1, 'read', k, blob[fill['offset']]))
            counts['cache_fill_commits'] += 1
            fill = None
            # One controller slot for fill commit; no lookup in the same slot.
            cycle += 1
            continue
        if shared == 'cache':
            if fill is not None:
                if not stress or cycle % 32 < 24:
                    for _ in range(min(2, 4-fill['issued'])):
                        j = fill['issued']; fill['issued'] += 1
                        events.append((cycle+1, 'fill', j, memory[fill['block']*4+j]))
                        counts['physical_W8_reads'] += 1
                else: counts['blocked_memory_slots'] += 1
            elif waiting:
                k = next((rr+i) % 4 for i in range(4) if (rr+i) % 4 in waiting)
                rr = (k+1) % 4; address = waiting[k]
                owners = [q for q, a in waiting.items() if a == address]
                for q in owners: del waiting[q]
                counts['cache_lookup_slots'] += 1
                counts['coalesced_logical_reads'] += len(owners)-1
                block, off = divmod(address, 4)
                if block in cache:
                    stamp += 1; blob, _ = cache[block]; cache[block] = (blob, stamp)
                    for q in owners: events.append((cycle+1, 'read', q, blob[off]))
                    counts['cache_hits'] += 1
                else:
                    fill = dict(block=block, offset=off, owners=owners, issued=0, bytes={})
                    counts['cache_misses'] += 1
        elif waiting:
            if not stress or cycle % 32 < 24:
                owners = [(rr+i) % 4 for i in range(4) if (rr+i) % 4 in waiting]
                issued = 0
                for k in owners:
                    if k not in waiting or issued == 2: continue
                    address = waiting[k]
                    group = [q for q in owners if q in waiting and waiting[q] == address] if shared == 'merge' else [k]
                    for q in group:
                        del waiting[q]
                        events.append((cycle+1, 'read', q, memory[address]))
                    counts['coalesced_logical_reads'] += len(group)-1
                    counts['physical_W8_reads'] += 1; issued += 1
                    rr = (k+1) % 4
            else: counts['blocked_memory_slots'] += 1
        cycle += 1
    return result, touched, dict(counts), cycle


def main():
    rows = []
    for c in load_cases():
        for stress in (False, True):
            nrv, source = source_scan(c['codes'], stress)
            for indexed in (False, True):
                for shared in ('private', 'merge', 'cache'):
                    mode = ('index' if indexed else 'dense') + '_' + shared
                    values = []; touched = []; counts = Counter(); clocks = []
                    for tile in range(2):
                        v, t, cnt, clock = simulate_tile(row_memory(c['W'][tile], indexed), nrv,
                            indexed, shared, c['table'], c['rank'], stress)
                        values.append(v); touched.append(t); counts.update(cnt); clocks.append(clock)
                    s = np.asarray(values); present = np.asarray(touched)
                    assert np.array_equal(s, c['S']) and np.array_equal(present, c['present'])
                    u = np.einsum('tr,mkpr->mkpt', c['coeff'].astype(np.int64), s)
                    gates = np.sum((u >= c['tau'][:, None, None, :]).astype(np.uint16)
                                   << np.arange(10, dtype=np.uint16), axis=-1, dtype=np.uint16)
                    assert np.array_equal(gates, c['gates'])
                    rows.append(dict(name=c['name'], real=c['real'], temporal=c['temporal'], mode=mode,
                        stress=stress, weight_density=float(np.mean(c['W'] != 0)), **source,
                        tile_W_frontend_slots=clocks, source_plus_W_frontend_slots=source['source_scan_slots']+max(clocks),
                        counts=dict(counts), S_values=int(s.size), gate_bits=320, differences=0))
        print('DONE', c['name'], flush=True)
    result = dict(scope=__doc__, fixed_cache=dict(lines=16, line_bytes=4, logical_bytes_per_tile=92),
        finite_controller='Common128B/tile reserved for four ID cursors, current NRV/code/index values, one pending request per ID, two W return pipelines and cache fill/owner state. This bound plus1154B worst row and92B cache fits common2KiB/tile. Fully-associative tag/merge logic is not physically synthesized.',
        modes=['dense_private','dense_merge','dense_cache','index_private','index_merge','index_cache'],
        service_boundary='Actual source decode then weight frontend completion; S/PSN arithmetic is functionally replayed but not included in these frontend slots. Not old RTL timings.',
        common_unpriced='Source external DMA/configuration, S accumulation/PSN/downstream, SRAM implementation; no new AEE or physical PPA.',
        source_buffer='Four4KiB 2R32/1W64 source banks, common to all arms. Each bank retains576B packed code and at most1536B explicit32bit NRV records. Source scan pairs both32bit read ports; during W service each tile gets one32bit port per bank, with actual one-cycle NRV responses. Stores are a serial prefix. This is a new CPU frontend budget, not the old RTL4x1KiB allocation.',
        rows=rows)
    (HERE/'shared_index_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS', len(rows), 'cases')


if __name__ == '__main__': main()
