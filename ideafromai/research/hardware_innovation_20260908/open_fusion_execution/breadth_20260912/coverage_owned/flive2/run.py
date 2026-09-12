"""Exact bounded one-tile, two-output-row F_live scheduling experiment."""
from pathlib import Path
from collections import Counter
import argparse
import json
import re
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared_index_probe import load_cases, ROOT


def common_best_tail(cases):
    """Give both F modes the already established cheaper in-place class tail."""
    plans = {r['block']: r for r in json.loads((ROOT/'psn/class_reconstruct_control.json').read_text())['records']
             if r['variant'] == 'bits3'}
    by_name = {c['name']: c for c in cases}
    for c in cases:
        c['tail'] = 'fixture_direct'
        c['original_program_length'] = len(c['program'])
        if c['temporal'] or not c['real']: continue
        match = re.search(r'_s2b([0-5])_', c['name'])
        if match is None: continue
        plan = plans[int(match[1])]
        if plan['blockwise_choose'] != 'restore_then_A': continue
        time_case = by_name[c['name'].replace('_packed_class', '_packed_time')]
        decode = np.array([[(int(c['table'][code]) >> r) & 1 for code in range(8)] for r in range(7)])
        cache = decode.copy(); tmp = np.zeros(8, dtype=np.int64); preamble = []
        for op in plan['plan']['microprogram']:
            source = op['src_slot']
            if op['op'] == 'load_U':
                tmp = cache[source].copy(); preamble.append((1 << 15) | source)
            else:
                tmp = tmp+cache[source]; cache[op['dst_slot']] = tmp
                preamble.append((1 << 15) | (1 << 10) | source | (op['dst_slot'] << 3))
        aliases = plan['plan']['output_alias_slots']
        effective = time_case['coeff'][:, :len(aliases)] @ cache[aliases]
        assert np.array_equal(effective, c['coeff'] @ decode), c['name']
        program = []
        for instruction in time_case['program']:
            word = int(instruction)
            if not word & (1 << 15): word = (word & ~7) | aliases[word & 7]
            program.append(word)
        assert len(program)+len(preamble) < len(c['program']) and len(program)+len(preamble) <= 256
        c['program'] = np.array(preamble+program, dtype=np.uint16)
        c['tail'] = 'ordinary_inplace_restore_then_A'


class PE:
    def __init__(self, case, k, flive, reduce):
        self.c = case; self.k = k; self.flive = flive; self.reduce = reduce
        self.packets = []; self.active = None; self.sealed = set(); self.finished = set()
        self.S = np.zeros(56, dtype=np.int64); self.valid = np.zeros(56, dtype=bool)
        self.consumer = None; self.out = None; self.count = Counter()
        self.snapshot = np.zeros((2, 4, 7), dtype=np.int64)
        self.present = np.zeros_like(self.snapshot, dtype=bool)
        self.output = np.zeros((2, 4), dtype=np.uint16)
        self.reserved_returns = 0

    def can_fetch(self, f):
        return any(p['f'] == f and p['state'] == 'fill' for p in self.packets) or len(self.packets) < 2

    def reserve(self, f):
        assert self.can_fetch(f)
        if not any(p['f'] == f and p['state'] == 'fill' for p in self.packets):
            self.packets.append(dict(f=f, rows=[], state='fill'))
        self.reserved_returns += 1
        assert self.reserved_returns <= 1
        self.count['NR4_peak'] = max(self.count['NR4_peak'], len(self.packets))

    def receive(self, f, code, w):
        self.reserved_returns -= 1
        p = next(p for p in self.packets if p['f'] == f and p['state'] == 'fill')
        self.count['logical_W_values'] += 1
        if w:
            p['rows'].append((code, w)); self.count['nonzero_W_rows'] += 1
            if len(p['rows']) == 4: p['state'] = 'ready'
        else: self.count['zero_W_rows'] += 1

    def seal(self, f):
        assert not self.reserved_returns
        for p in list(self.packets):
            if p['f'] == f and p['state'] == 'fill':
                if p['rows']: p['state'] = 'ready'
                else: self.packets.remove(p)
        self.sealed.add(f)

    def syn_done(self, f):
        return f in self.sealed and not any(p['f'] == f for p in self.packets)

    def permitted(self, f):
        return self.flive == 2 or f == 0 or 0 in self.finished

    def start_packet(self, p):
        p['state'] = 'exec'; destinations = []
        for pos in range(4):
            for r in range(self.c['rank']):
                values = [w for code, w in p['rows'] if (int(self.c['table'][code[pos]]) >> r) & 1]
                if values: destinations.append((p['f']*28+pos*7+r, values))
        assert destinations
        self.active = dict(packet=p, destinations=destinations, dest=0, member=0, partial=0)
        self.count['NR4_packets'] += 1; self.count['packet_start_slots'] += 1
        self.count['selected_events'] += sum(len(v) for _, v in destinations)
        self.count['distinct_commits'] += len(destinations)

    def start_consumer(self, f):
        values = self.S[f*28:(f+1)*28].reshape(4, 7).copy()
        valid = self.valid[f*28:(f+1)*28].reshape(4, 7).copy()
        self.snapshot[f] = values; self.present[f] = valid
        assert np.array_equal(values, self.c['S'][f, self.k])
        assert np.array_equal(valid, self.c['present'][f, self.k].astype(bool))
        self.consumer = dict(f=f, p=0, phase='load', r=0, cache=[0]*7, pc=0, u=0, gates=0)
        self.count['consumer_start_slots'] += 1

    def acknowledge(self):
        assert self.out is not None
        f, p, gates = self.out; self.output[f, p] = gates; self.out = None
        self.count['output_handshakes'] += 1
        q = self.consumer
        if p == 3:
            self.finished.add(f); self.consumer = None
            self.valid[f*28:(f+1)*28] = False
        else:
            q.update(p=p+1, phase='load', r=0, cache=[0]*7, pc=0, u=0, gates=0)

    def tick(self):
        assert len(self.packets) <= 2
        self.count['S_valid_peak'] = max(self.count['S_valid_peak'], int(self.valid.sum()))
        if self.out is not None:
            self.count['output_wait_slots'] += 1; return
        if self.consumer is not None:
            q = self.consumer
            if q['phase'] == 'load':
                address = q['f']*28+q['p']*7+q['r']
                q['cache'][q['r']] = int(self.S[address]) if self.valid[address] else 0
                self.count['consumer_S_reads'] += 1
                q['r'] += 1
                if q['r'] == self.c['rank']: q['phase'] = 'program'
            else:
                instruction = int(self.c['program'][q['pc']])
                restore = bool(instruction & (1 << 15) and not instruction & (1 << 9))
                if restore:
                    source = instruction & 7
                    if instruction & (1 << 10):
                        q['u'] += q['cache'][source]
                        assert -16384 <= q['u'] <= 16383
                        q['cache'][(instruction >> 3) & 7] = q['u']
                        self.count['restore_add_store_instructions'] += 1
                    else:
                        q['u'] = q['cache'][source]; self.count['restore_load_instructions'] += 1
                elif instruction & (1 << 9):
                    q['u'] = -int(self.c['tau'][q['f'], (instruction >> 11) & 15])
                    self.count['tau_instructions'] += 1
                elif not instruction & (1 << 15):
                    operand = q['cache'][instruction & 7] << ((instruction >> 3) & 31)
                    q['u'] += -operand if instruction & (1 << 8) else operand
                    self.count['CSD_add_instructions'] += 1
                assert -(1 << 23) <= q['u'] < (1 << 23)
                if not restore and instruction & (1 << 10):
                    q['gates'] |= int(q['u'] >= 0) << ((instruction >> 11) & 15)
                q['pc'] += 1; self.count['program_issues'] += 1
                if q['pc'] == len(self.c['program']):
                    self.out = (q['f'], q['p'], q['gates'])
            return
        if self.active is not None:
            a = self.active; address, values = a['destinations'][a['dest']]
            if self.reduce:
                delta = sum(values); commit = True
                self.count['member_groups'] += 1
            else:
                a['partial'] += values[a['member']]; a['member'] += 1
                delta = a['partial']; commit = a['member'] == len(values)
                self.count['scalar_member_slots'] += 1
                assert -512 <= delta <= 511
            if commit:
                old = int(self.S[address]) if self.valid[address] else 0
                new = old + delta; assert -16384 <= new <= 16383
                self.count['S_reads'] += int(self.valid[address])
                self.S[address] = new; self.valid[address] = True
                self.count['S_writes'] += 1
                a['dest'] += 1; a['member'] = 0; a['partial'] = 0
                if a['dest'] == len(a['destinations']):
                    self.packets.remove(a['packet']); self.active = None
            return
        # A consumer and a synaptic packet never issue on the same PE in one slot.
        for f in range(2):
            if f not in self.finished and self.permitted(f) and self.syn_done(f):
                self.start_consumer(f); return
        p = next((p for p in self.packets if p['state'] == 'ready' and self.permitted(p['f'])), None)
        if p is not None: self.start_packet(p)
        else: self.count['issue_idle_slots'] += 1


class Source:
    def __init__(self, codes, flive):
        self.memory = sum(int(codes[p, c]) << (12*c+3*p) for c in range(384) for p in range(4)).to_bytes(576, 'little')
        self.flive = flive; self.scan_f = 0; self.c = 0; self.loaded = -1; self.window = 0
        self.source_event = None; self.bridge = None; self.inflight = False; self.done = False

    def tick(self, now, pe, counts):
        if self.source_event and self.source_event[0] == now:
            _, word_index, value = self.source_event
            self.window = self.window | value << (64*word_index) if word_index <= 1 else (self.window >> 64) | (value << 64)
            assert self.window.bit_length() <= 128
            self.loaded = word_index; self.source_event = None
        if self.bridge and not self.bridge['need'] and not self.inflight: self.bridge = None
        if self.done or self.bridge or self.source_event or self.inflight: return
        if self.c == 384:
            for f in ([0, 1] if self.flive == 2 else [self.scan_f]): pe.seal(f)
            counts['source_end_slots'] += 1
            if self.flive == 1 and self.scan_f == 0:
                # Permit next-f prefetch while previous-f S/consumer is still live.
                self.scan_f = 1; self.c = 0; self.loaded = -1; self.window = 0
            else: self.done = True
            return
        need_word = (12*self.c+11)//64
        if self.loaded < need_word:
            index = self.loaded+1
            value = int.from_bytes(self.memory[index*8:index*8+8], 'little')
            self.source_event = (now+1, index, value); counts['source_R64_requests'] += 1
            return
        word = (self.window >> (12*self.c-max(self.loaded-1, 0)*64)) & 4095
        code = tuple((word >> (3*p)) & 7 for p in range(4))
        live = any(int(pe.c['table'][v]) for v in code)
        counts['source_rows_scanned'] += 1
        if live:
            self.bridge = dict(c=self.c, code=code, need=set([0, 1] if self.flive == 2 else [self.scan_f]), ready=now+1)
            counts['source_live_rows'] += 1
        self.c += 1


def simulate(case, flive, merge, reduce):
    pes = [PE(case, k, flive, reduce) for k in range(4)]
    sources = [Source(case['codes'][k], flive) for k in range(4)]
    # Actual row addresses in one common2KiB tile memory; two resident W rows.
    memory = case['W'].tobytes() + bytes(2048-768)
    events = []; count = Counter(); rr = 0; out_rr = 0; now = 0
    while not all(len(p.finished) == 2 for p in pes):
        assert now < 100000, (case['name'], flive, [(s.c, s.scan_f, s.bridge, s.inflight) for s in sources])
        due = [e for e in events if e[0] == now]; events = [e for e in events if e[0] != now]
        for _, address, owners in due:
            raw = memory[address]; weight = raw if raw < 128 else raw-256
            for k, f, code in owners:
                pes[k].receive(f, code, weight); sources[k].inflight = False
        # One10bit output handshake per slot, common to all4PEs.
        for j in range(4):
            k = (out_rr+j) % 4
            if pes[k].out is not None:
                pes[k].acknowledge(); out_rr = (k+1) % 4; count['output_slots'] += 1; break
        for p in pes: p.tick()
        for k, s in enumerate(sources): s.tick(now, pes[k], count)
        eligible = []
        for j in range(4):
            k = (rr+j) % 4; s = sources[k]
            if s.bridge and s.bridge['ready'] <= now and not s.inflight:
                f = next((f for f in sorted(s.bridge['need']) if pes[k].can_fetch(f)), None)
                if f is not None: eligible.append((k, f, f*384+s.bridge['c']))
                elif s.bridge['need']: count['NR4_admission_stalls'] += 1
        granted = set(); issued = 0
        for k, f, address in eligible:
            if k in granted or issued == 2: continue
            group = [(k, f)]
            if merge: group = [(kk, ff) for kk, ff, a in eligible if kk not in granted and a == address]
            owners = []
            for kk, ff in group:
                s = sources[kk]; pes[kk].reserve(ff); s.inflight = True; s.bridge['need'].remove(ff)
                owners.append((kk, ff, s.bridge['code'])); granted.add(kk)
            events.append((now+1, address, owners)); rr = (k+1) % 4; issued += 1
            count['physical_W8_reads'] += 1; count['merged_logical_W_reads'] += len(group)-1
        count['W_arbiter_denied_requests'] += len(eligible)-len(granted)
        assert len(events) <= 2
        count['W_return_pipeline_peak'] = max(count['W_return_pipeline_peak'], len(events))
        now += 1
    S = np.stack([p.snapshot for p in pes], axis=1)
    present = np.stack([p.present for p in pes], axis=1)
    gates = np.stack([p.output for p in pes], axis=1)
    assert np.array_equal(S, case['S']) and np.array_equal(present, case['present'])
    assert np.array_equal(gates, case['gates'])
    for p in pes:
        for k, v in p.count.items():
            if k.endswith('_peak'): count[k] = max(count[k], v)
            else: count[k] += v
    expanded = np.array([int(case['table'][x]).bit_count() for x in range(8)], dtype=np.int64)
    expected_events = int(np.einsum('kpc,fc->', expanded[case['codes']], (case['W'] != 0).astype(np.int64)))
    assert count['selected_events'] == expected_events
    assert count['distinct_commits'] == count['S_writes']
    assert count['NR4_peak'] <= 2 and count['S_valid_peak'] <= (28 if flive == 1 else 56)
    assert count['program_issues'] == 32*len(case['program'])
    assert count['output_handshakes'] == 32 and count['consumer_S_reads'] == 32*case['rank']
    assert all(not p.packets and not p.reserved_returns for p in pes)
    assert count['source_R64_requests'] == (576 if flive == 1 else 288)
    return dict(name=case['name'], real=case['real'], temporal=case['temporal'], rank=case['rank'],
                flive=flive, merge=merge, reduce=reduce, service_slots=now, tail=case['tail'],
                original_program_length=case['original_program_length'], program_length=len(case['program']),
                counts=dict(count), S_values_checked=int(S.size), gate_bits_checked=320, differences=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=0); args = ap.parse_args()
    cases = load_cases(); common_best_tail(cases)
    cases = cases[:args.limit] if args.limit else cases
    rows = []
    for case in cases:
        for flive in (1, 2):
            for merge in (False, True):
                for reduce in (False, True):
                    result = simulate(case, flive, merge, reduce); rows.append(result)
                    print('DONE', case['name'], flive, merge, reduce, result['service_slots'], flush=True)
        # Different scheduling policies must preserve private packet/event counts.
        subset = [r for r in rows if r['name'] == case['name']]
        for key in ('selected_events', 'NR4_packets', 'S_writes', 'logical_W_values', 'program_issues'):
            assert len({r['counts'].get(key, 0) for r in subset}) == 1, (case['name'], key)
    result = dict(kind=__doc__, common=dict(tiles=1, PEs=4, P=4, output_rows=2,
        S_words_per_PE=56, S_bits=15, NR4_packets_per_PE=2, W_ports=2, W_port_bits=8,
        W_response_slots=1, W_II=1, W_tau_control_bytes=2048, source_banks=4,
        source_bank_bytes=1024, source_read_ports_per_bank=1, source_port_bits=64,
        source_window_bits=128, source_bridge_entries_per_ID=1, W_inflight_per_ID=1,
        program_replicas=4, program_words=256, program_bits=16,
        source_cache_words_per_PE=7, output_holding_bits_per_PE=10, output_port_bits=10),
        boundary='Resident raw code/W/program/tau start through final T10 output handshake; source/W reads, bounded compaction, shared issue, actual S and CSD program execution. One-tile/two-row CPU model, not old two-tile RTL, whole-layer Gustav or PPA.',
        excluded=['source producer','external cold DMA/configuration','FC2/BN2/shortcut','physical SRAM/area/frequency'],
        rows=rows)
    (HERE/('smoke_results.json' if args.limit else 'results.json')).write_text(json.dumps(result, indent=2)+'\n')
    print('PASS', len(rows), 'runs', flush=True)


if __name__ == '__main__': main()
