"""Bounded, ordered service reference for one P4/H8 physical slice.

Actual 64-bit transfers and dependent vector instructions are issued; scalar
operation totals are never divided by a PE count. This is a declared machine
model, not measured RTL timing. Contexts execute independently and serially.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from source_count_bound_probe import PREFIX, TAIL, N_UP, compile_bounds

HERE = Path(__file__).resolve().parent
GROUPS = (0, 21, 42, 63)
H_GROUPS = tuple(range(12))
MODES = ('full10Y', 'full10Y_time_major', 'prefix_count_bin', 'candidate_time_major')
CANDIDATES = ('prefix_count_bin', 'candidate_time_major')


class Engine:
    def __init__(self, trace=False):
        self.free = dict(state=0, weight=0, conv=0, psn=0, compare=0)
        self.busy = Counter()
        self.events = Counter()
        self.end = 0
        self.trace = [] if trace else None

    def op(self, unit, kind, dep, beats=1, latency=None):
        start = max(int(dep), self.free[unit])
        self.free[unit] = start + beats
        ready = start + (beats if latency is None else latency)
        self.end = max(self.end, ready)
        self.busy[unit] += beats
        self.events[kind] += beats
        if self.trace is not None and len(self.trace) < 64:
            self.trace.append(dict(unit=unit, kind=kind, start=start, issue_beats=beats, ready=ready))
        return ready, start

    def state(self, kind, dep, beats):
        return self.op('state', kind, dep, beats)[0]

    def conv(self, kind, dep):
        return self.op('conv', kind, dep, latency=1)[0]

    def psn(self, kind, dep):
        return self.op('psn', kind, dep, latency=2)

    def compare(self, kind, dep):
        return self.op('compare', kind, dep, latency=1)[0]


class Slice:
    def __init__(self, params, h_group, trace=False):
        self.e = Engine(trace)
        self.h0 = h_group * 8
        self.w = params['weight_int8'].reshape(96, 864)[self.h0:self.h0 + 8].astype(np.int64)
        self.a = params['temporal_int16'].astype(np.int64)
        self.tau = params['threshold_positive'][params['full_entry']][:, self.h0:self.h0 + 8]
        lo, hi = compile_bounds(params['weight_int8'].reshape(96, 864))
        self.lo = lo[:, self.h0:self.h0 + 8]
        self.hi = hi[:, self.h0:self.h0 + 8]
        if self.lo.min() < -32768 or self.hi.max() > 32767:
            raise ValueError('This finite model uses the actual compiled INT16 bound table.')
        self.bound = int(params['Y_abs_bound'].max())
        self.mem_y = np.zeros((10, 4, 8), dtype=np.int64)
        self.valid = np.zeros(10, dtype=bool)
        self.cache_tag = None
        self.cache_y = np.zeros((4, 8), dtype=np.int64)
        self.cache_dirty = False
        self.cache_ready = 0
        self.w_tag = self.mask_tag = self.a_tag = self.bound_tag = self.eta_tag = None
        self.w_ready = self.mask_ready = self.a_ready = self.bound_ready = self.eta_ready = 0
        self.eta = np.zeros((10, 4, 8), dtype=np.int64)
        self.answer = np.zeros((10, 4, 8), dtype=bool)
        self.live = np.zeros((10, 4, 8), dtype=bool)
        self.logic = Counter()

    def mask(self, k, dep):
        line = k // 8  # Eight eight-bit masks per actual 64-bit W-bank word.
        if self.mask_tag != line:
            self.mask_ready = self.e.op('weight', 'W_nonzero_mask_read', dep)[0]
            self.mask_tag = line
        return self.w[:, k] != 0, max(dep, self.mask_ready)

    def weight(self, k, dep):
        if self.w_tag != k:
            self.w_ready = self.e.op('weight', 'W_coefficient_vector_read', dep)[0]
            self.w_tag = k
        return max(dep, self.w_ready)

    def cache(self, t, dep):
        if self.cache_tag == t:
            self.logic['Y_row_cache_hits'] += 1
            return max(dep, self.cache_ready)
        if self.cache_tag is not None and self.cache_dirty:
            dep = self.e.state('Conv_Y_write64', max(dep, self.cache_ready), 12)
            self.mem_y[self.cache_tag] = self.cache_y
            self.valid[self.cache_tag] = True
        if self.valid[t]:
            dep = self.e.state('Conv_Y_read64', dep, 12)
            self.cache_y = self.mem_y[t].copy()
        else:
            dep = self.e.conv('Y_work_row_zero_init', dep)
            self.cache_y.fill(0)
        self.cache_tag, self.cache_dirty, self.cache_ready = t, False, dep
        return dep

    def y_value(self, t, p):
        if self.cache_tag == t:
            return self.cache_y[p]
        return self.mem_y[t, p] if self.valid[t] else np.zeros(8, dtype=np.int64)

    def operand(self, t, p, dep):
        value = self.y_value(t, p).copy()
        if self.cache_tag == t:
            return value, max(dep, self.cache_ready)
        if np.any(value):
            dep = self.e.state('PSN_or_check_Y_vector_read64', dep, 3)
        else:
            self.logic['zero_Y_vector_reads_bypassed'] += 1
        return value, dep

    def coefficient(self, t, s, dep):
        line = (t * 10 + s) // 4  # Four INT16 A entries per 64-bit word.
        if self.a_tag != line:
            self.a_ready = self.e.state('A_coefficient_read64', dep, 1)
            self.a_tag = line
        return max(dep, self.a_ready)

    def temporal(self, t, p, sources, dep, label):
        accum = np.zeros(8, dtype=np.int64)
        sources = [s for s in sources if self.a[t, s] != 0 and np.any(self.y_value(s, p))]
        if not sources:
            self.logic['zero_temporal_vectors_bypassed'] += 1
            return accum, dep
        u_ready, issue = self.e.psn(label + '_U_clear', dep)
        cursor = issue + 1
        for s in sources:
            if self.a[t, s] == 0 or not np.any(self.y_value(s, p)):
                continue
            cursor = self.coefficient(t, s, cursor)
            value, cursor = self.operand(s, p, cursor)
            u_ready, issue = self.e.psn(label + '_MAC8', max(cursor, u_ready))
            cursor = issue + 1  # Next operand read may overlap the two-cycle MAC.
            accum += self.a[t, s] * value
            self.logic[label + '_active_scalar_products'] += int(np.count_nonzero(value))
        return accum, max(cursor, u_ready)

    def update(self, t, src_p, k, dep, wr):
        dep = self.cache(t, dep)
        mask = src_p[:, None] & (self.w[:, k] != 0)[None]
        ready = self.e.conv('Conv_masked_add32', max(dep, wr))
        self.cache_y += mask * self.w[:, k][None]
        self.cache_ready, self.cache_dirty = ready, True
        self.logic['Conv_active_scalar_adds'] += int(mask.sum())
        return ready

    def retire_row_if_dead(self, t):
        if not self.live[t].any():
            self.valid[t] = False
            if self.cache_tag == t:
                self.cache_tag, self.cache_dirty = None, False

    def bound_check(self, requests, remaining, dep, initial=False):
        # One live L/U vector can broadcast across pending t and p of the
        # same count bin. Different groups/independent contexts are not merged.
        bins = np.searchsorted(N_UP, remaining)
        for bin_id in sorted({int(bins[t, p]) for t, p in requests}):
            current = [(t, p) for t, p in requests if bins[t, p] == bin_id and self.live[t, p].any()]
            if not current:
                continue
            if bin_id and self.bound_tag != bin_id:
                # Actual extrema fit signed16; eight L/U pairs use 32 bytes.
                # They are sign-extended before addition to the INT24 Yi.
                self.bound_ready = self.e.state('bound_LU_H8_read64', dep, 4)
                self.bound_tag = bin_id
            if bin_id:
                dep = max(dep, self.bound_ready)
            for t, p in current:
                mask = self.live[t, p].copy()
                if not mask.any():
                    continue
                y, dep = self.operand(t, p, dep)
                if self.eta_tag != (t, p):
                    self.eta_ready = self.e.state('eta_H8_read64', dep, 2)
                    self.eta_tag = (t, p)
                dep = max(dep, self.eta_ready)
                if bin_id:
                    low, high = y + self.lo[bin_id], y + self.hi[bin_id]
                    if np.any(y[mask]):
                        low_ready = self.e.conv('bound_low_add8', dep)
                        low_compare_ready = self.e.compare('bound_low_compare8', low_ready)
                        # The low comparison samples U before the next add
                        # overwrites it; distinct units may issue together.
                        high_ready = self.e.conv('bound_high_add8', low_ready)
                        dep = self.e.compare('bound_high_compare8', max(low_compare_ready, high_ready))
                    else:
                        dep = self.e.compare('bound_low_compare8', dep)
                        dep = self.e.compare('bound_high_compare8', dep)
                else:
                    low = high = y
                    dep = self.e.compare('final_raw_Y_compare8', dep)
                low = np.maximum(low, -self.bound)
                high = np.minimum(high, self.bound)
                if self.a[t, t] > 0:
                    positive, negative = low >= self.eta[t, p], high < self.eta[t, p]
                else:
                    positive, negative = high <= self.eta[t, p], low > self.eta[t, p]
                decided = mask & (positive | negative)
                self.answer[t, p, decided] = positive[decided]
                self.live[t, p, decided] = False
                self.logic['initial_retired_gates' if initial else 'stream_retired_gates'] += int(decided.sum())
                self.logic['check_H8_vectors'] += 1
                self.retire_row_if_dead(t)
        return dep

    def divide(self, t, p, delta, active, dep):
        d = int(self.a[t, t])
        self.eta[t, p] = np.clip(-np.floor_divide(-delta, d) if d > 0 else np.floor_divide(delta, d),
                                 -self.bound - 1, self.bound + 1)
        if not active.any():
            return dep
        # Executable instruction template for the three-product compiler.
        # The same MAC/adder and scratch registers are reused throughout.
        for label in ('divide_sign_adjust', 'divide_save_n', 'divide_mul_low', 'divide_mul_high_shift23'):
            dep = self.e.psn(label, dep)[0]
        # Unsigned reciprocal high-bit compensation is layer-row static.
        absd = abs(d)
        shift = ((self.bound + 2) * absd - 1).bit_length()
        reciprocal = (1 << shift) // absd
        if reciprocal >= 32768:
            dep = self.e.psn('divide_highbit_shift16_add', dep)[0]
        for label in ('divide_shift_to_q', 'divide_restore_n', 'divide_mul_qd_subtract'):
            dep = self.e.psn(label, dep)[0]
        dep = self.e.compare('divide_remainder_compare8', dep)
        dep = self.e.conv('divide_quotient_correction8', dep)
        dep = self.e.conv('divide_sign_restore8', dep)
        self.logic['materialized_eta_scalars'] += int(active.sum())
        self.logic['division_scalar_products'] += int(active.sum()) * 3
        dep = self.e.state('eta_H8_write64', dep, 2)
        self.eta_tag, self.eta_ready = (t, p), dep
        return dep

    def run(self, words, nrvs, initial_counts, mode):
        dep = self.e.state('source_header_read64', 0, 1)
        phase = {}
        ever = np.zeros((10, 4, 8), dtype=bool)
        candidate = mode in CANDIDATES
        prefix_set = tuple(PREFIX) if candidate else tuple(range(10))
        present_times = [t for t in range(10) if np.any(words & (1 << t))]
        if mode == 'candidate_time_major':
            passes = [(t,) for t in prefix_set]
        else:
            passes = [(t,) for t in present_times] if mode == 'full10Y_time_major' else [prefix_set]
        for pass_index, pass_times in enumerate(passes):
            # The first candidate sweep builds private source/W intersections
            # for all tails. It must visit every descriptor even when its own
            # prefix time is absent. Later time sweeps can skip absent sources.
            build_private_metadata = candidate and pass_index == 0
            for k in nrvs:
                dep = self.e.state('source_NRV_read64', dep, 1)
                dep += 1  # Bounded descriptor decode; one descriptor register.
                src = (words[k][None] & (1 << np.arange(10))[:, None]) != 0
                if mode in ('full10Y_time_major', 'candidate_time_major') and not build_private_metadata and not np.any(src[list(pass_times)]):
                    continue
                nz, mr = self.mask(k, dep)
                if build_private_metadata:
                    ever[TAIL] |= src[TAIL, :, None] & nz[None, None]
                    self.logic['private_live_mask_OR_updates'] += 1
                    metadata_ready = mr + 1
                else:
                    metadata_ready = mr
                needed = [t for t in pass_times if np.any(src[t, :, None] & nz[None])]
                if needed:
                    wr = self.weight(k, mr)
                    order = ([self.cache_tag] if self.cache_tag in needed else [])
                    order += [t for t in needed if t not in order]
                    for t in order:
                        dep = self.update(t, src[t], k, max(dep, mr), wr)
                else:
                    dep = max(dep, mr)
                dep = max(dep, metadata_ready)
        phase['Conv_full10_or_prefix_end'] = dep
        if mode.startswith('full10Y'):
            for t in range(10):
                dep = self.e.state('full_tau_H8_read64', dep, 4)
                for p in range(4):
                    value, dep = self.temporal(t, p, range(10), dep, 'full_PSN')
                    dep = self.e.compare('full_gate_compare8', dep)
                    self.answer[t, p] = value >= self.tau[t]
            phase['PSN_end'] = dep
        else:
            remaining = initial_counts.copy()
            dep = self.e.state('initial_count_snapshot_read64', dep, 5)
            for t in range(10):
                dep = self.e.state('full_tau_H8_read64', dep, 4)
                tau_scratch_valid = True
                if t in TAIL:
                    dep = self.e.state('division_row_constants_read64', dep, 2)
                for p in range(4):
                    value, dep = self.temporal(t, p, PREFIX, dep, 'core_PSN')
                    if t in PREFIX:
                        dep = self.e.compare('core_gate_compare8', dep)
                        self.answer[t, p] = value >= self.tau[t]
                        continue
                    if not tau_scratch_valid:
                        dep = self.e.state('tau_reload_after_division_scratch64', dep, 4)
                    delta = self.tau[t] - value
                    dep = self.e.psn('eta_delta_subtract8', dep)[0]
                    dep = self.e.compare('eta_range_low_compare8', dep)
                    dep = self.e.compare('eta_range_high_compare8', dep)
                    limit = abs(int(self.a[t, t])) * self.bound
                    positive, negative = delta <= -limit, delta > limit
                    self.live[t, p] = ever[t, p] & ~(positive | negative)
                    self.answer[t, p] = np.where(ever[t, p], positive, value >= self.tau[t])
                    dep = self.divide(t, p, delta, self.live[t, p], dep)
                    tau_scratch_valid = not self.live[t, p].any()
            phase['eta_and_core_PSN_end'] = dep
            # All three core Yi have now been consumed. Drop dead dirty state
            # rather than forcing an unnecessary final SRAM writeback.
            self.cache_tag, self.cache_dirty = None, False
            self.valid[:] = False
            self.bound_tag = None
            requests = [(int(t), p) for t in TAIL for p in range(4) if self.live[t, p].any()]
            dep = self.bound_check(requests, remaining, dep, initial=True)
            phase['initial_bound_end'] = dep
            tail_passes = [(int(t),) for t in TAIL] if mode == 'candidate_time_major' else [tuple(map(int, TAIL))]
            for pass_times in tail_passes:
                for k in nrvs:
                    if not self.live[list(pass_times)].any():
                        break
                    dep = self.e.state('source_NRV_read64', dep, 1)
                    dep += 1
                    src = (words[k][None] & (1 << np.arange(10))[:, None]) != 0
                    counter_source = np.zeros_like(src)
                    counter_source[list(pass_times)] = src[list(pass_times)] & self.live[list(pass_times)].any(2)
                    if mode == 'candidate_time_major' and not counter_source.any():
                        continue
                    nz, mr = self.mask(k, dep)
                    needed = [t for t in pass_times if np.any(self.live[t] & src[t, :, None] & nz[None])]
                    if needed:
                        wr = self.weight(k, mr)
                        order = ([self.cache_tag] if self.cache_tag in needed else [])
                        order += [t for t in needed if t not in order]
                        for t in order:
                            dep = self.cache(t, max(dep, mr))
                            mask = self.live[t] & src[t, :, None] & nz[None]
                            dep = self.e.conv('Conv_masked_add32', max(dep, wr))
                            self.cache_y += mask * self.w[:, k][None]
                            self.cache_ready, self.cache_dirty = dep, True
                            self.logic['Conv_active_scalar_adds'] += int(mask.sum())
                    else:
                        dep = max(dep, mr)
                    if counter_source.any():
                        dep = self.e.conv('remaining_count_decrement28', dep)
                        remaining -= counter_source
                        drop = counter_source & ((remaining & (remaining - 1)) == 0)
                        requests = [(t, p) for t in pass_times for p in range(4) if drop[t, p] and self.live[t, p].any()]
                        dep = self.bound_check(requests, remaining, dep)
            if self.live.any():
                raise RuntimeError('Finite private-tail slice failed to complete.')
            phase['private_tail_and_checks_end'] = dep
        # Five gate words plus one context/theta payload header; same endpoint.
        dep = self.e.state('output_gate_and_theta_header_write64', dep, 6)
        phase['output_commit_end'] = dep
        return dict(model_ticks=dep, phase_ends=phase, port_and_unit_busy=dict(self.e.busy),
                    ordered_service_counts=dict(self.e.events), logical_work=dict(self.logic),
                    trace_first_64_events=self.e.trace)


def build_nrvs(words, with_counts):
    e = Engine()
    nrvs = []
    counts = np.zeros((10, 4), dtype=np.int64)
    dep = 0
    for k in range(864):
        dep = e.state('dense_source_word_read64', dep, 1)
        dep += 1  # Detect nonempty support and attach k index.
        if np.any(words[k]):
            nrvs.append(k)
            write_ready = e.state('compact_NRV_write64', dep, 1)
            if with_counts:
                bits = (words[k][None] & (1 << np.arange(10))[:, None]) != 0
                if bits[TAIL].any():
                    count_ready = e.conv('initial_private_count_increment28', dep)
                    dep = max(write_ready, count_ready)
                    counts[TAIL] += bits[TAIL]
                else:
                    dep = write_ready
            else:
                dep = write_ready
    if with_counts:
        dep = e.state('initial_count_snapshot_write64', dep, 5)
    dep = e.state('source_header_write64', dep, 1)
    return np.array(nrvs, dtype=np.int64), counts, dict(model_ticks=dep,
        port_and_unit_busy=dict(e.busy), ordered_service_counts=dict(e.events), descriptors=len(nrvs))


def aggregate(rows):
    totals = dict(model_ticks=sum(r['service']['model_ticks'] for r in rows))
    for field in ('port_and_unit_busy', 'ordered_service_counts', 'logical_work'):
        out = Counter()
        for r in rows:
            out.update(r['service'][field])
        totals[field] = dict(out)
    totals['gate_mismatches'] = sum(r['gate_mismatches'] for r in rows)
    totals['checked_gates'] = sum(r['checked_gates'] for r in rows)
    return totals


def main():
    params = dict(np.load(HERE / 'integer_deployment/common3_diagonal_34.npz'))
    result = dict(
        scope='fixed capture groups 0,21,42,63 of four frames, each of 12 H8 slices; 192 independent P4/H8 contexts per control, serial aggregation, not parallel whole-P4 execution',
        groups=list(GROUPS), h_groups=list(H_GROUPS),
        machine=dict(W_port_bits=64, W_port='read-only, one beat/step, one-step return; coefficient and nonzero mask share it',
                     state_port_bits=64, state_port='one read OR write per step, one-step read return; Y, NRV, A/tau/eta/bounds and output all contend',
                     conv='32 INT24 add lanes, one vector issue/step; source count and bound adds reuse these lanes',
                     psn='8 signed24x16/48 MAC lanes, one vector issue/step, two-step dependent result; divide ALU/shift operations reuse them',
                     comparisons='8 lanes, one vector issue/step; shared by exact gates, bounds and division correction',
                     W_pool_bytes=131072, W_bytes=82944, static_W_mask_bytes=10368,
                     state_parameter_source_pool_bytes=24576, payload_reservation_per_context_bytes=2048,
                     source_NRV_reservation_bytes=8192, bound_table_bytes=4608, bound_entry_bits=16, tau32_bytes=3840,
                     source_header='one 64-bit word holds NRV length and the 10-bit OR of source time support, produced by the required source compaction pass',
                     Y_row_work_register_bytes=96, W_row_register_bytes=8, W_mask_register_bytes=8,
                     PSN_U_register_bytes=48, operand_register_bytes=24, eta_register_bytes=16,
                     bound_or_tau_scratch_register_bytes=48, coefficient_word_register_bytes=8,
                     division_scalar_config_register_bytes=16,
                     zero_Y_lane_metadata_register_bytes=40, private_live_register_bytes=28,
                     output_word_and_header_register_bytes=48,
                     NRV_register_bytes=8, source_counter_reservation_bytes=50,
                     Y_layout='p-major within each t row; 32x24bit row=12 beats, eight-h p vector=3 beats',
                     word_addresses='W64=(hg*864+k); mask64=82944/8+hg*108+floor(k/8). State Y byte offset=slot*96+p*24, eta=eta_base+private_index*64+p*16; both vector starts are 64-bit aligned. Candidate prefix/tail slots reuse storage after all core consumers finish.',
                     NRV_layout='40 support bits + 10-bit k + flags in one actual 64-bit record; generated by an in-place ordered scan'),
        policies=['all controls have the same total capacities and units; ordinary full10Y never materializes eta',
                  'ordinary full10Y also receives a time-major schedule: skip header-certified empty t, otherwise reread the actual NRV stream and resident W, retain one Yi row across all K, then consume ordinary A',
                  'candidate_time_major has the same 96-byte Yi work row: three prefix-time sweeps then the existing single-U core/eta generator, followed by one ordered sweep for each live private time; no extra Ucore bank or RF is allocated',
                  'W and static parameters are already resident; no offchip W fill or PPA is inferred',
                  'one dirty Y row can survive across k; current cached t is serviced first when legal; zero vectors bypass ordinary PSN operand reads',
                  'candidate checks block subsequent descriptors; no free negative-latency feedback or speculative next-k requests',
                  'count snapshots are source-global, but each independent H8 has private remaining counters; no unpaid cross-H8 sharing',
                  'candidate private product-empty metadata is built by real NRV/W-mask intersections during its prefix scan',
                  'source compaction is counted once per P4 source group and reused by the 12 serial H8 contexts; ordinary is not forced to count private sources',
                  'fixed 48-bit shift input selection and register routes are resource assumptions, not a mapped-area claim',
                  'Conv2, halos beyond captured im2col support, upstream sn1 production and the network are outside this slice'],
        source_preparation={m: [] for m in MODES}, axes={m: dict(contexts=[]) for m in MODES})
    for path in sorted((HERE / 'integer_valid10').glob('capture_*.npz')):
        with np.load(path) as cap:
            golden = cap['gate_common3_diagonal_34_exact'].reshape(64, 12, 10, 8, 4)
            golden = golden.transpose(0, 1, 2, 4, 3)  # G,HG,T,P,H.
            for g in GROUPS:
                words = cap['source_gate_words'][g].astype(np.int64)
                prepared = {}
                for mode in MODES:
                    nrvs, counts, prep = build_nrvs(words, mode in CANDIDATES)
                    prepared[mode] = (nrvs, counts)
                    result['source_preparation'][mode].append(dict(capture=path.name, group=g, **prep))
                for hg in H_GROUPS:
                    for mode in MODES:
                        nrvs, counts = prepared[mode]
                        sl = Slice(params, hg, trace=(path.name == 'capture_00.npz' and g == GROUPS[0] and hg == 0))
                        service = sl.run(words, nrvs, counts, mode)
                        mismatch = int(np.count_nonzero(sl.answer != golden[g, hg]))
                        if mismatch:
                            raise RuntimeError(f'{path.name} G{g}/H{hg} {mode}: {mismatch} wrong gates')
                        result['axes'][mode]['contexts'].append(dict(capture=path.name, group=g, h_group=hg,
                            service=service, gate_mismatches=mismatch, checked_gates=320))
                print(path.name, 'group', g, 'complete', flush=True)
    for mode in MODES:
        total = aggregate(result['axes'][mode]['contexts'])
        total['source_preparation_ticks'] = sum(x['model_ticks'] for x in result['source_preparation'][mode])
        total['with_source_preparation_ticks'] = total['model_ticks'] + total['source_preparation_ticks']
        result['axes'][mode]['totals'] = total
    base = result['axes']['full10Y']['totals']
    cand = result['axes']['prefix_count_bin']['totals']
    time_base = result['axes']['full10Y_time_major']['totals']
    time_cand = result['axes']['candidate_time_major']['totals']
    result['comparison'] = dict(
        candidate_over_full10Y_service=cand['model_ticks'] / base['model_ticks'],
        candidate_over_full10Y_with_source_prep=cand['with_source_preparation_ticks'] / base['with_source_preparation_ticks'],
        candidate_over_best_ordinary_service=cand['model_ticks'] / min(result['axes'][m]['totals']['model_ticks'] for m in MODES if m.startswith('full10Y')),
        candidate_over_best_ordinary_with_source_prep=cand['with_source_preparation_ticks'] / min(result['axes'][m]['totals']['with_source_preparation_ticks'] for m in MODES if m.startswith('full10Y')),
        candidate_time_major_over_ordinary_time_major_service=time_cand['model_ticks'] / time_base['model_ticks'],
        candidate_time_major_over_ordinary_time_major_with_source_prep=time_cand['with_source_preparation_ticks'] / time_base['with_source_preparation_ticks'],
        candidate_time_major_minus_ordinary_time_major_service=time_cand['model_ticks'] - time_base['model_ticks'],
        candidate_time_major_minus_ordinary_time_major_with_source_prep=time_cand['with_source_preparation_ticks'] - time_base['with_source_preparation_ticks'],
        interpretation='finite model completion steps under this single declared point; not measured RTL cycles, equal mapped area, or a whole-layer accelerator ratio')
    output = HERE / 'finite_service.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['comparison']), flush=True)


if __name__ == '__main__':
    main()
