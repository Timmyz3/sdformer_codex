"""Complete K864/H32 payload service with all54 K16 blocks; no physical PPA.

The two metadata devices are explicit throughput models. Their ordinary
TCAM/dual-bank assumptions and the new column adapter are not physical RTL.
Machine owns all arithmetic, SRAM/coefficient bytes and the finite96-vectorRF.
"""
from pathlib import Path
import argparse
import json
import sys
from collections import Counter, deque

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import compare as common

META = 98304
DIRECTORY = 101376
BLOCKS, KBLOCK, TOKENS = 54, 16, 20


class Detector:
    """One block initiation per source block; two explicitly budgeted banks.

    TCAM:20 row preload cycles, followed by the M+4 processing pipeline.
    Column: one finish cycle, then20 accepted parent outputs. A common
    stable-count dispatcher can sort concurrently within these20 outputs.
    Both emit packed16-bit parent/count entries through the same SW64 port.
    """
    def __init__(self, mode):
        self.mode = mode
        self.keys = [0] * TOKENS
        self.counts = [0] * TOKENS
        self.relation = [(1 << TOKENS)-1] * TOKENS
        self.jobs = []
        self.writes = deque()
        self.stats = Counter()
        self.block_records = []
        self.max_write_queue = 0
        self.max_banks = 0

    def column(self, k, mask):
        kk = k % KBLOCK
        for i in range(TOKENS):
            if mask & (1 << i):
                self.keys[i] |= 1 << kk
                self.counts[i] += 1
            else:
                self.relation[i] &= ~mask
        self.stats['accepted_columns'] += 1

    def finish(self, time, block, start, length, rows):
        # Primary parent rule reads the independently built row supports;
        # the column arm uses only its relation matrix and counts to select.
        gold, gold_counts = common.parent_rule(self.keys)
        assert gold_counts == self.counts
        parents = [-1] * TOKENS
        if self.mode == 'column_relation':
            for i in range(TOKENS):
                best = 0
                if self.counts[i] < 2:
                    continue
                for j in range(TOKENS):
                    if self.relation[i] & (1 << j) and self.counts[j] > best and \
                       (self.counts[j] < self.counts[i] or j < i):
                        parents[i], best = j, self.counts[j]
            assert parents == gold
        else:
            parents = gold
        self.stats['physical_K_blocks'] += 1
        self.stats['parents'] += sum(p >= 0 for p in parents[:rows])
        self.stats['source_occurrences'] += sum(self.counts[:rows])
        removed = sum(self.counts[p] for p in parents[:rows] if p >= 0)
        self.stats['removed_occurrences'] += removed
        self.writes.append((DIRECTORY + block*8,
                            start.to_bytes(4, 'little') + length.to_bytes(4, 'little')))
        if length:
            # The stage time is a source-stated throughput assumption, not a
            # post-synthesis clock or an uncharged instantaneous CAM call.
            first = time + (TOKENS + 4 if self.mode == 'tcam_pr' else 1)
            packet = [(self.counts[i] << 8) | (parents[i]+1) for i in range(TOKENS)]
            self.jobs.append(dict(block=block, first=first, cursor=0,
                                  values=packet, pending=[]))
            self.max_banks = max(self.max_banks, len(self.jobs))
            assert len(self.jobs) <= 2, 'two-bank metadata storage exhausted'
            self.stats['nonempty_K_blocks'] += 1
            self.stats['detector_pipeline_slots'] += TOKENS + (24 if self.mode == 'tcam_pr' else 1)
        self.block_records.append(dict(k_begin=block*KBLOCK, k_end=(block+1)*KBLOCK,
            NRV_start=start, NRV_rows=length, source_occurrences=sum(self.counts[:rows]),
            parents=sum(p >= 0 for p in parents[:rows]), removed_occurrences=removed,
            live_local_rows=sum(c > 0 for c in self.counts[:rows])))
        self.keys = [0] * TOKENS
        self.counts = [0] * TOKENS
        self.relation = [(1 << TOKENS)-1] * TOKENS

    def tick(self, time):
        for job in self.jobs:
            if time < job['first']:
                continue
            # Finite four-entry64-bit metadata queue; emission backpressures.
            if len(self.writes) >= 4:
                self.stats['metadata_output_backpressure_slots'] += 1
                continue
            i = job['cursor']
            job['pending'].append(job['values'][i])
            job['cursor'] += 1
            self.stats['parent_output_slots'] += 1
            if len(job['pending']) == 4:
                payload = b''.join(x.to_bytes(2, 'little') for x in job['pending'])
                self.writes.append((META + job['block']*40 + (i-3)*2, payload))
                job['pending'] = []
        self.jobs = [j for j in self.jobs if j['cursor'] < TOKENS]
        self.max_write_queue = max(self.max_write_queue, len(self.writes))
        assert len(self.writes) <= 4


class Machine(common.Machine):
    def __init__(self, device=None):
        super().__init__(False)
        self.device = device

    def advance(self, read=None, write=None, coef=None, cwrite=None, op=None, tag=''):
        # Base Machine may recursively advance to resolve a writeback clash.
        # Defer device work to the actual issued cycle, not an attempted slot.
        latency = 2 if op and (op[0].startswith('I') or op[0] in
                  ('AAC', 'decode_u8', 'decode_shift', 'compare', 'control')) else 4
        while op and self.time+latency in self.pending:
            self.advance(tag='port_or_writeback_wait')
        if self.device is not None:
            self.device.tick(self.time)
            if write is None and self.device.writes:
                write = self.device.writes.popleft()
                self.count['metadata_SW64_writes'] += 1
                if tag != 'detector_tail_wait':
                    self.count['metadata_writes_overlapped_with_foreground'] += 1
            elif write is not None and self.device.writes:
                self.count['metadata_SW64_conflicts'] += 1
        super().advance(read=read, write=write, coef=coef, cwrite=cwrite, op=op, tag=tag)

    def finish_device(self):
        self.phase = 'metadata_tail_wait'
        while self.device and (self.device.jobs or self.device.writes):
            self.advance(tag='detector_tail_wait')


def build(m, words, source_origin, positions, device):
    """The original source addresses/NRV production, with accepted-mask fanout."""
    m.phase = 'common_build_P2_NRV'
    n = occurrences = live = block_start = 0
    height, width = words.shape[1:]
    for k in range(864):
        c, rem = divmod(k, 9)
        ky, kx = divmod(rem, 3)
        mask = 0
        for ip, (y, x) in enumerate(positions):
            sy, sx = y+ky-1, x+kx-1
            if not (0 <= sy < 240 and 0 <= sx < 320):
                continue
            ly, lx = sy-source_origin[0], sx-source_origin[1]
            assert 0 <= ly < height and 0 <= lx < width
            address = 2*((c*height+ly)*width+lx)
            raw = m.read_word(address)
            word = int.from_bytes(raw[address%8:address%8+2], 'little')
            mask |= word << (ip*10)
        live |= mask
        m.advance(op=('control', 93, [k, mask, live, 0, 0, 0, 0, 0]), tag='NRV_decode')
        m.wait_reg(93)
        if mask:
            payload = int(m.rf[93, 0]).to_bytes(4, 'little') + int(m.rf[93, 1]).to_bytes(4, 'little')
            m.advance(write=(common.NRV+n*8, payload), tag='NRV_write')
            if device is not None:
                device.column(k, mask)
            n += 1
            occurrences += mask.bit_count()
        if k % KBLOCK == KBLOCK-1 and device is not None:
            device.finish(m.time, k//KBLOCK, block_start, n-block_start, len(positions)*10)
            block_start = n
    return n, occurrences, live


def load_metadata(m, block):
    m.phase = 'K16_directory_and_parent_reads'
    raw = m.read_word(DIRECTORY+block*8)
    start, length = int.from_bytes(raw[:4], 'little'), int.from_bytes(raw[4:], 'little')
    m.advance(tag='K16_directory_select')
    if not length:
        return start, length, [], []
    entries = []
    for off in range(0, 40, 8):
        raw = m.read_word(META+block*40+off)
        entries.extend(int.from_bytes(raw[j:j+2], 'little') for j in range(0, 8, 2))
        m.advance(tag='K16_parent_word_decode')
    return start, length, [(x & 255)-1 for x in entries], [x >> 8 for x in entries]


def pr_u(m, n, base, rows):
    """H8 outer:20 global +20 local vectors; no extra80 accumulators."""
    z = np.zeros((rows, 32), np.float32)
    execution = Counter()
    for hg in range(4):
        m.phase = 'K16_global_clear'
        for tp in range(rows):
            m.wait_reg(20+tp)
            m.advance(op=('clear', 20+tp, None), tag='global_clear')
        consumed = 0
        for block in range(BLOCKS):
            start, length, parents, counts = load_metadata(m, block)
            assert start == consumed
            consumed += length
            execution['covered_K_blocks_times_H8'] += 1
            if not length:
                execution['empty_K_blocks_times_H8'] += 1
                continue
            active = [i for i in range(rows) if counts[i]]
            m.phase = 'K16_local_clear'
            for tp in active:
                m.wait_reg(tp)
                m.advance(op=('clear', tp, None), tag='local_clear')
            m.phase = 'K16_residual_GP'
            for record in range(start, start+length):
                raw = m.read_word(common.NRV+record*8)
                k, original = int.from_bytes(raw[:4], 'little'), int.from_bytes(raw[4:], 'little')
                assert k//KBLOCK == block
                mask = original
                # Same registered20-token residual selection for both arms;
                # the ordinary baseline gets this column-oriented privilege.
                for tp in active:
                    p = parents[tp]
                    if p >= 0 and original & (1 << p):
                        assert original & (1 << tp)
                        mask &= ~(1 << tp)
                m.advance(tag='K16_registered_residual_select')
                execution['source_NRV_replays'] += 1
                if not mask:
                    execution['empty_residual_records_times_H8'] += 1
                    continue
                m.coefficient(base['U']+(k*32+hg*8)*4)
                for tp in active:
                    if mask & (1 << tp):
                        m.wait_reg(tp)
                        m.advance(op=('FMA', tp, (None, None)), tag='K16_residual_issue')
                        execution['residual_vector_adds'] += 1
            m.phase = 'K16_parent_and_global_accumulation'
            for tp in sorted(active, key=lambda i: (counts[i], i)):
                if parents[tp] >= 0:
                    m.wait_reg(tp)
                    m.wait_reg(parents[tp])
                    m.advance(op=('add_reg', tp, parents[tp]), tag='K16_parent_add')
                    execution['parent_vector_adds'] += 1
            for tp in active:
                m.wait_reg(tp)
                m.wait_reg(20+tp)
                m.advance(op=('add_reg', 20+tp, tp), tag='K16_global_add')
                execution['global_vector_adds'] += 1
        assert consumed == n
        m.drain()
        m.phase = 'common_Z_output'
        for tp in range(rows):
            z[tp, hg*8:hg*8+8] = m.rf[20+tp]
            m.advance(op=('TF32', 20+tp, None), tag='Z_conversion')
            m.store_reg(20+tp, common.Z+(tp*32+hg*8)*4)
    return z, dict(execution)


def run(data, params, label, mode):
    geo = json.loads(str(data['window_geometry_json']))[label]
    source = data[label+'_sn1_gate']
    words = sum(source[t].astype(np.uint16) << t for t in range(10))
    blob, base, info = common.coefficients(params, False)
    m = Machine()
    m.phase = 'common_coefficient_fill'
    m.dma_input(blob, 0, True)
    m.phase = 'common_source_input'
    m.dma_input(words.astype('<u2').tobytes(), common.SRC)
    outputs, groups = [], []
    detector_total = Counter()
    h, w = geo['gate_shape']
    oy, ox = geo['gate_origin']
    for y in range(h):
        for x in range(0, w, 2):
            positions = [(oy+y, ox+x+i) for i in range(min(2, w-x))]
            rows = len(positions)*10
            begin = m.time
            device = None if mode == 'direct_gp' else Detector(mode)
            m.device = device
            n, occurrences, live = build(m, words, geo['source_origin'], positions, device)
            m.finish_device()
            if mode == 'direct_gp':
                z = common.execute_u(m, n, base, False, len(positions))
                execution = dict(source_NRV_replays=n, residual_vector_adds=occurrences*4,
                                 parent_vector_adds=0, global_vector_adds=0)
            else:
                z, execution = pr_u(m, n, base, rows)
                detector_total.update(device.stats)
            outputs.append(z)
            groups.append(dict(y=y, x=x, logical_tokens=rows, K=864, H=32,
                physical_K_blocks=BLOCKS, H8_passes=4, NRV_rows=n,
                source_occurrences=occurrences, service_slots=m.time-begin,
                execution=execution, block_details=device.block_records if device else [],
                detector=dict(device.stats) if device else {},
                max_completed_blocks_in_detector_pipeline=device.max_banks if device else 0,
                max_metadata_queue_words=device.max_write_queue if device else 0))
    state = dict(SRAM_capacity_bytes=131072, coefficient_capacity_bytes=131072,
        coefficient_used_bytes=info['used_bytes'], source_bytes=words.nbytes,
        maximum_NRV_bytes=max(g['NRV_rows'] for g in groups)*8,
        output_Z_bytes=20*32*4, source_RF_vector_words=96,
        peak_accumulator_vector_words=80 if mode == 'direct_gp' else 40,
        local_vector_words=0 if mode == 'direct_gp' else 20,
        global_vector_words=80 if mode == 'direct_gp' else 20,
        accumulator_spill_reads=0, accumulator_spill_writes=0,
        parent_table_bytes=0 if mode == 'direct_gp' else 54*40,
        block_directory_bytes=0 if mode == 'direct_gp' else 54*8,
        dispatch_parent_count_table_bytes=0 if mode == 'direct_gp' else 40,
        metadata_output_queue_bytes=0 if mode == 'direct_gp' else 32,
        reserved_detector_banks=0 if mode == 'direct_gp' else 2,
        state_address_high_water=(common.Z+20*32*4) if mode == 'direct_gp' else DIRECTORY+54*8,
        same_area_or_clock_proven=False)
    if mode == 'tcam_pr':
        state['additional_detector_logical_state'] = dict(
            TCAM_ternary_cells_two_banks=2*20*16, column_to_row_staging_bits_two_banks=2*20*16,
            row_count_bits_two_banks=2*20*5, parent_index_valid_bits_two_banks=2*20*6,
            physical_bitcell_cost='Unknown; ternary cells are not SRAM bits.')
    elif mode == 'column_relation':
        state['additional_detector_logical_state'] = dict(
            relation_bits_two_banks=2*20*20, row_count_bits_two_banks=2*20*5,
            parent_index_valid_bits_two_banks=2*20*6,
            RTL_scope='Existing single-task relation leaf only; dual-bank K16 adapter and SRAM reload not RTL closed.')
    return np.concatenate(outputs), dict(service_slots=m.time, stages=dict(m.stages),
        counts=dict(m.count), groups=groups, detector_totals=dict(detector_total),
        state=state, coefficient_parameters=info,
        backpressure='always_ready SR64/SW64/CR256; arithmetic RAW/WAW/writeback waits charged',
        complete_scope='All captured P2/T10 groups in this saved window, all54 K16 blocks and4 H8 groups; not a full spatial layer or full network.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    result = dict(scope=__doc__, K=864, H=32, physical_K_blocks=54, H8_passes=4,
        official_replication=False, axes={})
    axes = ('ordinary',) if args.smoke else ('ordinary', 'lifting_raw')
    for axis in axes:
        cap = common.FULL/'capture'/axis
        data = common.read_npz(cap/'000_zurich_city_09_a_0001.npz')
        params = common.read_npz(cap/'live_parameters.npz')
        result['axes'][axis] = {}
        labels = ('corner',) if args.smoke else ('corner', 'interior')
        for label in labels:
            row, ref = {}, None
            for mode in ('direct_gp', 'tcam_pr', 'column_relation'):
                z, report = run(data, params, label, mode)
                if ref is None:
                    ref = z
                report['difference_vs_full_GP'] = common.difference(z, ref)
                assert report['difference_vs_full_GP']['differences'] == 0
                row[mode] = report
                print(axis, label, mode, report['service_slots'],
                      report['difference_vs_full_GP'], flush=True)
            for mode in ('tcam_pr', 'column_relation'):
                row[mode]['net_service_reduction_vs_GP'] = 1-row[mode]['service_slots']/row['direct_gp']['service_slots']
            row['column_relation']['net_service_reduction_vs_tcam_PR'] = 1-row['column_relation']['service_slots']/row['tcam_pr']['service_slots']
            result['axes'][axis][label] = row
    path = HERE/('smoke.json' if args.smoke else 'results.json')
    path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
