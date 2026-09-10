"""Same-machine, one-shot strict permission before private Conv production.

Imports the fixed machine operations, never modifies the other reference.
The single current U register is reused for both endpoints. Seven private
threshold-shifted core rows reside in the charged state memory, not a free RF.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from finite_service import GROUPS, H_GROUPS, Slice, aggregate, build_nrvs
from source_count_bound_probe import PREFIX, TAIL, N_UP

HERE = Path(__file__).resolve().parent


class OneShot(Slice):
    def __init__(self, params, h_group, trace=False, core_bits=48):
        super().__init__(params, h_group, trace)
        self.core_bits = core_bits
        self.core_beats = core_bits // 8
        self.core = np.zeros((7, 4, 8), dtype=np.int32 if core_bits == 32 else np.int64)
        self.tail_index = {int(t): i for i, t in enumerate(TAIL)}
        self.core_valid = np.zeros((7, 4), dtype=bool)

    def permission(self, t, p, value, ever, counts, dep):
        # tau32 + L16 occupy the existing 48-byte scratch; the unused eta16
        # register holds H16. This leaves the only U48 register for the MAC.
        c = value - self.tau[t]
        dep = self.e.psn('one_shot_core_minus_tau8', dep)[0]
        known = ~ever[t, p]
        if known.any():
            dep = self.e.compare('product_empty_gate_compare8', dep)
            self.answer[t, p, known] = (c >= 0)[known]
            self.logic['ordinary_product_empty_private_gates'] += int(known.sum())
        self.live[t, p] = ~known
        if not self.live[t, p].any():
            return dep

        ix = self.tail_index[t]
        self.core[ix, p] = c
        if self.core_bits == 32:
            mismatch = int(np.count_nonzero(self.core[ix, p].astype(np.int64) != c))
            self.logic['core_INT32_store_signextend_mismatches'] += mismatch
            assert not mismatch
        self.core_valid[ix, p] = True
        # C is still in the only U register after the write, so the first
        # endpoint needs no read. The second endpoint restores it from state.
        dep = self.e.state('one_shot_core_H8_write64', dep, self.core_beats)
        bin_id = int(np.searchsorted(N_UP, counts[t, p]))
        if self.bound_tag != bin_id:
            self.bound_ready = self.e.state('one_shot_bound_LU_H8_read64', dep, 4)
            self.bound_tag = bin_id
        dep = max(dep, self.bound_ready)
        dep = self.coefficient(t, t, dep)
        d = int(self.a[t, t])
        lower = self.lo[bin_id] if d > 0 else self.hi[bin_id]
        upper = self.hi[bin_id] if d > 0 else self.lo[bin_id]
        mask = self.live[t, p].copy()
        if np.any(lower[mask]):
            dep = self.e.psn('one_shot_low_endpoint_MAC8', dep)[0]
            self.logic['one_shot_endpoint_scalar_products'] += int(np.count_nonzero(lower[mask]))
        dep = self.e.compare('one_shot_low_endpoint_compare8', dep)
        ulo = c + d * lower
        positive = mask & (ulo >= 0)
        self.answer[t, p, positive] = True
        self.live[t, p, positive] = False
        self.logic['one_shot_positive_gates'] += int(positive.sum())
        self.logic['one_shot_H8_permissions'] += 1

        mask = self.live[t, p].copy()
        if mask.any():
            dep = self.e.state('one_shot_second_endpoint_core_H8_read64', dep, self.core_beats)
            c = self.core[ix, p].astype(np.int64)
            if np.any(upper[mask]):
                dep = self.e.psn('one_shot_high_endpoint_MAC8', dep)[0]
                self.logic['one_shot_endpoint_scalar_products'] += int(np.count_nonzero(upper[mask]))
            dep = self.e.compare('one_shot_high_endpoint_compare8', dep)
            uhi = c + d * upper
            negative = mask & (uhi < 0)
            self.answer[t, p, negative] = False
            self.live[t, p, negative] = False
            self.logic['one_shot_negative_gates'] += int(negative.sum())
        if not self.live[t, p].any():
            self.core_valid[ix, p] = False
        return dep

    def run(self, words, nrvs, initial_counts):
        dep = self.e.state('source_header_read64', 0, 1)
        phase = {}
        ever = np.zeros((10, 4, 8), dtype=bool)

        # Same three time-major prefix sweeps and real source/W-mask empty
        # producer as candidate_time_major. The first sweep visits all NRVs.
        for pass_index, t in enumerate(PREFIX):
            for k in nrvs:
                dep = self.e.state('source_NRV_read64', dep, 1)
                dep += 1
                src = (words[k][None] & (1 << np.arange(10))[:, None]) != 0
                if pass_index and not src[t].any():
                    continue
                nz, mr = self.mask(k, dep)
                if pass_index == 0:
                    ever[TAIL] |= src[TAIL, :, None] & nz[None, None]
                    self.logic['private_live_mask_OR_updates'] += 1
                    metadata_ready = mr + 1
                else:
                    metadata_ready = mr
                if np.any(src[t, :, None] & nz[None]):
                    wr = self.weight(k, mr)
                    dep = self.update(t, src[t], k, max(dep, mr), wr)
                else:
                    dep = max(dep, mr)
                dep = max(dep, metadata_ready)
        phase['prefix_Conv_end'] = dep
        dep = self.e.state('initial_count_snapshot_read64', dep, 5)

        for t in range(10):
            dep = self.e.state('full_tau_H8_read64', dep, 4)
            for p in range(4):
                value, dep = self.temporal(t, p, PREFIX, dep, 'core_PSN')
                if t in PREFIX:
                    dep = self.e.compare('core_gate_compare8', dep)
                    self.answer[t, p] = value >= self.tau[t]
                else:
                    dep = self.permission(t, p, value, ever, initial_counts, dep)
        phase['core_PSN_and_one_shot_end'] = dep
        self.logic['private_gates_requiring_full_Conv'] = int(self.live.sum())
        self.logic['private_H8_vectors_requiring_full_Conv'] = int(self.live.any(2).sum())
        self.logic['private_P4_H8_times_requiring_full_Conv'] = int(self.live.any(axis=(1, 2)).sum())

        # Every prefix consumer has finished. Its 288-byte backing allocation
        # and the 96-byte working row can now serve any one private time.
        self.cache_tag, self.cache_dirty = None, False
        self.valid[:] = False
        for t_value in TAIL:
            t = int(t_value)
            if not self.live[t].any():
                continue
            for k in nrvs:
                dep = self.e.state('source_NRV_read64', dep, 1)
                dep += 1
                src_p = (words[k] & (1 << t)) != 0
                if not np.any(src_p & self.live[t].any(1)):
                    continue
                nz, mr = self.mask(k, dep)
                mask = self.live[t] & src_p[:, None] & nz[None]
                if mask.any():
                    wr = self.weight(k, mr)
                    dep = self.cache(t, max(dep, mr))
                    dep = self.e.conv('Conv_masked_add32', max(dep, wr))
                    self.cache_y += mask * self.w[:, k][None]
                    self.cache_ready, self.cache_dirty = dep, True
                    self.logic['Conv_active_scalar_adds'] += int(mask.sum())
                else:
                    dep = max(dep, mr)
            for p in range(4):
                mask = self.live[t, p].copy()
                if not mask.any():
                    continue
                ix = self.tail_index[t]
                assert self.core_valid[ix, p]
                dep = self.e.state('one_shot_final_core_H8_read64', dep, self.core_beats)
                y, dep = self.operand(t, p, dep)
                dep = self.coefficient(t, t, dep)
                if np.any(y[mask]):
                    dep = self.e.psn('one_shot_final_private_MAC8', dep)[0]
                    self.logic['final_private_active_scalar_products'] += int(np.count_nonzero(y[mask]))
                dep = self.e.compare('one_shot_final_gate_compare8', dep)
                final = self.core[ix, p].astype(np.int64) + self.a[t, t] * y
                self.answer[t, p, mask] = (final >= 0)[mask]
                self.live[t, p] = False
                self.core_valid[ix, p] = False
            # No remaining consumer: no useless dirty-row SRAM writeback.
            self.cache_tag, self.cache_dirty = None, False
            self.valid[t] = False
        assert not self.live.any() and not self.core_valid.any()
        phase['private_full_Conv_and_final_PSN_end'] = dep
        dep = self.e.state('output_gate_and_theta_header_write64', dep, 6)
        phase['output_commit_end'] = dep
        return dict(model_ticks=dep, phase_ends=phase,
                    port_and_unit_busy=dict(self.e.busy),
                    ordered_service_counts=dict(self.e.events),
                    logical_work=dict(self.logic), trace_first_64_events=self.e.trace)


def main(core_bits=48):
    params = dict(np.load(HERE / 'integer_deployment/common3_diagonal_34.npz'))
    a = params['temporal_int16']
    for t in range(10):
        assert all(s in PREFIX or s == t for s in np.flatnonzero(a[t]))
    baseline = json.loads((HERE / 'finite_service.json').read_text())
    tau = params['threshold_positive'][params['full_entry']].astype(np.int64)
    c_bound = np.abs(tau) + np.abs(a[:, PREFIX].astype(np.int64)).sum(1)[:, None] * params['Y_abs_bound'][None]
    c_bound_max = int(c_bound.max())
    assert c_bound_max <= np.iinfo(np.int32).max
    core_bytes = 7 * 4 * 8 * core_bits // 8
    result = dict(
        scope='same 192 serial independent P4/H8 contexts as finite_service; complete T10 common3 integer student, not whole-layer timing',
        groups=list(GROUPS), h_groups=list(H_GROUPS),
        algorithm='three shared prefix Y; C=Ucore-tau. Exactly one 12-bin signed private-Y interval permission. Failed private rows execute complete time-major Conv and C+d*Y gate. No remaining-count updates, eta division, or repeated bounds.',
        core_storage_bits=core_bits,
        numeric_compiler=dict(core_bound_formula='abs(tau[t,h])+sum_s_in_PREFIX(abs(Aq[t,s]))*Y_abs_bound[h]',
                              full_legal_source_core_abs_bound_max=c_bound_max,
                              required_signed_bits=c_bound_max.bit_length() + 1,
                              working_U_bits=48, storage_rounding='none; exact signed int32 store and sign-extension when core_bits=32'),
        machine=baseline['machine'],
        state_allocation=dict(
            shared_prefix_Y_bytes=288, private_core_bytes=core_bytes,
            private_live_aligned_bytes=32, output_gate_bytes=40,
            initial_count_aligned_bytes=56, zero_Y_lane_metadata_bytes=40,
            source_header_bytes=8, Y_valid_and_control_bytes=24,
            conservative_payload_total_bytes=488 + core_bytes, payload_limit_bytes=2048,
            lifetime=f'Y[0:288] contains three prefixes until every core is consumed; afterward its first 96 bytes may hold the only current private Y. C[288:{288 + core_bytes}] contains seven private rows of four H8 vectors, {core_bits} bytes/vector. C is released after initial certification or final exact gate.',
            scratch='existing 48-byte bound/tau scratch = tau32 + L16; existing eta16 register = H16. One U48 and one operand24 only. No ten-row U bank.',
            metadata='live/output/zero metadata reuse the original register capacities; they are additionally included in this conservative 2KiB tally. Their bounded updates use existing sideband logic, as in finite_service.'),
        policies=[
            f'first endpoint uses the just-produced C in the single U register; C writes and restoration for the second endpoint are actual {core_bits // 8}-beat state operations',
            f'failed final gates reread C for {core_bits // 8} beats and use the same eight PSN MAC lanes for d*Y+C',
            'L/U and tau remain simultaneously resident only by repartitioning existing scratch/eta registers, with no added register bytes',
            'ordinary source/W product-empty lanes are exact before bounds; all source compaction/counts and W-mask intersections are charged',
            'whole-vector zero bypasses use already-produced Y metadata; future Y values and golden gates never select requests',
            'static W and parameter pools, 64-bit ports, multiplier latency and source endpoint are inherited unchanged',
            'these are finite service-model completion steps, not RTL cycles, isoarea, PPA or system acceleration'],
        source_preparation=[], contexts=[], baselines={})
    for path in sorted((HERE / 'integer_valid10').glob('capture_*.npz')):
        with np.load(path) as cap:
            golden = cap['gate_common3_diagonal_34_exact'].reshape(64, 12, 10, 8, 4).transpose(0, 1, 2, 4, 3)
            for g in GROUPS:
                words = cap['source_gate_words'][g].astype(np.int64)
                nrvs, counts, prep = build_nrvs(words, True)
                result['source_preparation'].append(dict(capture=path.name, group=g, **prep))
                for hg in H_GROUPS:
                    sl = OneShot(params, hg, trace=(path.name == 'capture_00.npz' and g == GROUPS[0] and hg == 0), core_bits=core_bits)
                    service = sl.run(words, nrvs, counts)
                    mismatch = int(np.count_nonzero(sl.answer != golden[g, hg]))
                    if mismatch:
                        raise RuntimeError(f'{path.name} G{g}/H{hg}: {mismatch} wrong gates')
                    result['contexts'].append(dict(capture=path.name, group=g, h_group=hg,
                        service=service, gate_mismatches=mismatch, checked_gates=320))
                print(path.name, 'group', g, 'one-shot complete', flush=True)
    total = aggregate(result['contexts'])
    total['source_preparation_ticks'] = sum(x['model_ticks'] for x in result['source_preparation'])
    total['with_source_preparation_ticks'] = total['model_ticks'] + total['source_preparation_ticks']
    result['totals'] = total
    keys = [(x['capture'], x['group'], x['h_group']) for x in result['contexts']]
    for mode in ('full10Y_time_major', 'candidate_time_major'):
        ref = baseline['axes'][mode]
        assert keys == [(x['capture'], x['group'], x['h_group']) for x in ref['contexts']]
        result['baselines'][mode] = ref['totals']
    ordinary = result['baselines']['full10Y_time_major']
    candidate = result['baselines']['candidate_time_major']
    result['comparison'] = dict(
        one_shot_over_ordinary_service=total['model_ticks'] / ordinary['model_ticks'],
        one_shot_over_ordinary_with_source_prep=total['with_source_preparation_ticks'] / ordinary['with_source_preparation_ticks'],
        candidate_over_one_shot_service=candidate['model_ticks'] / total['model_ticks'],
        candidate_over_one_shot_with_source_prep=candidate['with_source_preparation_ticks'] / total['with_source_preparation_ticks'],
        candidate_minus_one_shot_with_source_prep=candidate['with_source_preparation_ticks'] - total['with_source_preparation_ticks'])
    output = HERE / 'one_shot_service.json'
    if core_bits == 32:
        original = json.loads(output.read_text())
        result['comparison_with_core48'] = dict(
            ticks_saved=original['totals']['with_source_preparation_ticks'] - total['with_source_preparation_ticks'],
            core32_over_core48=total['with_source_preparation_ticks'] / original['totals']['with_source_preparation_ticks'])
        original['compiled_core32'] = result
        original['preferred_compiled_axis'] = 'compiled_core32'
        output.write_text(json.dumps(original, ensure_ascii=False, indent=2) + '\n')
    else:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(totals=total, comparison=result['comparison'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--core-bits', type=int, choices=(32, 48), default=48)
    main(parser.parse_args().core_bits)
