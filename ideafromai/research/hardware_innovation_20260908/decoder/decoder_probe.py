#!/usr/bin/env python3.12
"""Existing ep34 decoder payload: exact support counts and finite-state bounds.

No forward capture, RTL timing, fixed-point claim or PPA.  The primary mapping
uses 96 output channels, 8 banks, 12 FP32 values per bank = THREE 128-bit beats.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import sys
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path('/home/zhumd/work')
HW = WORK / 'sdformer_codex/SDformer/hw_autoresearch_nts07'
CAP = HW / 'results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831'
CKPT = HW / 'system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth'
sys.path.insert(0, str(WORK / 'ideafromai/research/mechanism_rebuild_gh_20260906/scripts'))
from checkpoint_numpy import read_checkpoint


def parameters(stage):
    state = read_checkpoint(CKPT)['model_state_dict']
    pre = f'sttmultires_unet.decoders.{stage}.'
    pred = f'sttmultires_unet.preds.{stage}.'
    names = [pre+'deconv.0.weight', pre+'sn.spiking_neuron.thresh',
             pre+'sn.spiking_neuron.weight', pred+'sn.spiking_neuron.thresh',
             pred+'sn.spiking_neuron.weight', pre+'norm_layer.norm_layer.weight',
             pre+'norm_layer.norm_layer.bias']
    selected = {n: np.array(state[n], copy=True) for n in names}
    del state
    gc.collect()
    w = selected[names[0]]
    info = {'convtranspose_weight_shape': list(w.shape),
            'weight_fp32_bytes': int(w.nbytes),
            'weight_exact_zeros': int(np.count_nonzero(w == 0)),
            'source_theta': selected[names[1]].tolist(),
            'source_theta_shape': list(selected[names[1]].shape),
            'source_PSN_rank': int(np.linalg.matrix_rank(selected[names[2]].astype(np.float64))),
            'consumer_theta': selected[names[3]].tolist(),
            'consumer_theta_shape': list(selected[names[3]].shape),
            'consumer_PSN_rank': int(np.linalg.matrix_rank(selected[names[4]].astype(np.float64))),
            'consumer_PSN_shape': list(selected[names[4]].shape),
            'BN_gamma_negative': int(np.count_nonzero(selected[names[5]] < 0)),
            'parameter_source': str(CKPT)}
    return w, info


def topology(h, w):
    oh, ow = 2*h, 2*w
    p, q = h*w, oh*ow
    predecessors = np.full((q, 4), -1, np.int32)
    tap_ids = np.full((q, 4), -1, np.int8)
    degree = np.zeros(q, np.int8)
    source_degree = np.zeros(p, np.int8)
    for y in range(h):
        for x in range(w):
            i = y*w+x
            for ky in range(3):
                oy = 2*y+ky-1
                if not 0 <= oy < oh:
                    continue
                for kx in range(3):
                    ox = 2*x+kx-1
                    if not 0 <= ox < ow:
                        continue
                    j = oy*ow+ox
                    k = degree[j]
                    predecessors[j, k] = i
                    tap_ids[j, k] = 3*ky+kx
                    degree[j] += 1
                    source_degree[i] += 1
    last = predecessors.max(1)
    return predecessors, tap_ids, degree, source_degree, last


def load_payload(row, theta):
    meta = row['input']
    payload = row['payload']
    shape = meta['shape']
    raw = zlib.decompress((CAP/payload['compressed_fp32']).read_bytes())
    values = np.frombuffer(raw, '<f4').reshape(shape)
    active = np.count_nonzero(values)
    non_theta = 0
    # Check continuous amplitude without sorting a full 149 MB array.
    flat = values.reshape(-1)
    for lo in range(0, flat.size, 1_000_000):
        v = flat[lo:lo+1_000_000]
        non_theta += int(np.count_nonzero((v != 0) & (v != theta)))
    packed = (CAP/payload['support_sign']).read_bytes()
    n = payload['positive_plane_bytes']
    positive = np.unpackbits(np.frombuffer(packed[:n], np.uint8), bitorder='little', count=flat.size)
    negative = np.unpackbits(np.frombuffer(packed[n:], np.uint8), bitorder='little', count=flat.size)
    support = (positive | negative).reshape(shape).astype(bool)
    assert active == meta['active'] == int(support.sum())
    assert np.array_equal(support, values != 0)
    assert non_theta == 0
    return values, support, {'active': int(active), 'nonzero_amplitude': float(theta),
                            'nonzero_values_different_from_theta': non_theta,
                            'payload_FP32_bytes': len(raw),
                            'positive_plane_bytes': n,
                            'negative_count': int(negative.sum()),
                            'full_values_retained': True}


def live_peak(starts, ends, length):
    starts = starts.reshape(-1)
    ends = ends.reshape(-1)
    keep = starts >= 0
    a = np.bincount(starts[keep], minlength=length)
    b = np.bincount(ends[keep], minlength=length)
    # Values live during the entire source-packet service, including a packet
    # that both starts and completes an output. Retire after that packet.
    active = np.cumsum(a) - np.r_[0, np.cumsum(b)[:-1]]
    return int(active.max())


def numeric_spots(values, weights, predecessors, taps, oh, ow):
    src = values[:, 0].reshape(values.shape[0], values.shape[2], -1)
    points = [(0, 0), (0, 1), (1, 0), (1, 1), (oh//2+1, ow//2+1), (oh-1, ow-1)]
    checks, max_error = 0, 0.0
    # Independently enumerate inverse gather coordinates and forward scatter
    # coordinates, but do not claim GPU FP32 order equivalence.
    h, w = values.shape[-2:]
    for y, x in points:
        target = y*ow+x
        a = np.zeros((values.shape[0], weights.shape[1]), np.float64)
        for slot in range(4):
            p = int(predecessors[target, slot])
            if p >= 0:
                ky, kx = divmod(int(taps[target, slot]), 3)
                a += src[:, :, p].astype(np.float64) @ weights[:, :, ky, kx].astype(np.float64)
        b = np.zeros_like(a)
        for ky in range(3):
            for kx in range(3):
                iy_num, ix_num = y+1-ky, x+1-kx
                if iy_num % 2 or ix_num % 2:
                    continue
                iy, ix = iy_num//2, ix_num//2
                if 0 <= iy < h and 0 <= ix < w:
                    b += src[:, :, iy*w+ix].astype(np.float64) @ weights[:, :, ky, kx].astype(np.float64)
        max_error = max(max_error, float(np.max(np.abs(a-b))))
        assert np.allclose(a, b, atol=1e-12, rtol=1e-12)
        checks += a.size
    return {'full_channel_values_checked': checks, 'coordinates': points,
            'max_float64_reordering_abs_error': max_error,
            'claim': 'boundary/coefficient spot check, not full layer numeric replay or frozen FP32 proof'}


def analyze(row, weights, param, bank_bits, external_bytes):
    t, b, ci, h, w = row['input']['shape']
    co = weights.shape[1]
    assert b == 1 and co == 96 and weights.shape == (ci, co, 3, 3)
    theta = float(param['source_theta'])
    values, support, amplitude = load_payload(row, theta)
    preds, taps, degree, source_degree, last = topology(h, w)
    p, q = h*w, 4*h*w
    count = support[:, 0].sum(axis=1, dtype=np.int32).reshape(t, p)
    padded = np.c_[count, np.zeros(t, np.int32)]
    index = np.where(preds >= 0, preds, p)
    contributions = padded[:, index]
    updates = contributions.sum(2, dtype=np.int64)
    scatter_updates = int(np.sum(count.astype(np.int64)*source_degree[None, :]))
    gather_updates = int(updates.sum())
    assert scatter_updates == gather_updates
    valid = preds[None, :, :] >= 0
    active = (contributions > 0) & valid
    first = np.where(active, preds[None, :, :], p).min(2)
    actual_last = np.where(active, preds[None, :, :], -1).max(2)
    first[first == p] = -1
    geometric_end = np.broadcast_to(last, first.shape)
    nonempty = updates > 0
    n_nonempty = int(nonempty.sum())
    last_site_active = count[:, last] > 0
    direct_retire = int(np.count_nonzero(nonempty & last_site_active))
    late_retire = n_nonempty-direct_retire
    vector_bytes = co*4
    beats = math.ceil(co*32/(8*bank_bits))
    frontiers = []
    for block_t in (1, 2, 5, 10):
        peaks = []
        for lo in range(0, t, block_t):
            nt = min(block_t, t-lo)
            ids = np.arange(nt)[:, None]
            start = np.where(first[lo:lo+nt] >= 0, first[lo:lo+nt]*nt+ids, -1)
            end = geometric_end[lo:lo+nt]*nt+ids
            peaks.append(live_peak(start, end, p*nt))
        peak = max(peaks)
        frontiers.append({'interleaved_T': block_t, 'peak_output_vectors': peak,
                          'partial_sum_bytes_FP32': peak*vector_bytes,
                          'fits_128KiB_partial_sum': peak*vector_bytes <= 128*1024,
                          'two_source_rows_packed_bytes': math.ceil(2*w*ci*block_t/8)})
    oracle_peak = max(live_peak(first[i], actual_last[i], p) for i in range(t))
    geom_peak = frontiers[0]['peak_output_vectors']
    # CSR-like nonzero channel indices are NOT assumed to appear for free:
    # the counters below include reading every fixed-width support bitmap.
    source_vector_bytes = math.ceil(ci/8)
    source_descriptors = t*p
    source_lookups_scatter = source_descriptors
    source_lookups_gather = t*int(degree.sum())
    gate_words_per_lookup = math.ceil(ci/128)
    output_vectors = t*q
    y_bytes = output_vectors*vector_bytes
    gate_bytes = math.ceil(support.size/8)
    # Strong input-stationary scatter FIRST reduces all active channels into
    # nine local kernel-contribution vectors, then merges those into the row
    # frontier. Do not manufacture a gain versus scalar-product scatter.
    source_bundle_vectors = int(np.sum((count > 0)*source_degree[None, :]))
    local_bundle_adds = scatter_updates-source_bundle_vectors
    destination_merge_adds = source_bundle_vectors-n_nonempty
    assert local_bundle_adds+destination_merge_adds == scatter_updates-n_nonempty
    psum_reads = source_bundle_vectors-n_nonempty+late_retire
    psum_writes = source_bundle_vectors-direct_retire
    weight_bytes = int(weights.nbytes)
    input_relayout_bytes = 2*gate_bytes
    layout_cost = math.ceil(input_relayout_bytes/external_bytes)
    common_external = gate_bytes+weight_bytes+y_bytes
    # These are stage-service lower bounds, not a pretend cycle-accurate sum.
    compute_beats = scatter_updates*beats
    source_scatter_beats = source_lookups_scatter*gate_words_per_lookup
    source_gather_beats = source_lookups_gather*gate_words_per_lookup
    scatter_bank_read_beats = psum_reads*beats
    scatter_bank_write_beats = psum_writes*beats
    scatter_bound = max(compute_beats, source_scatter_beats,
                        scatter_bank_read_beats, scatter_bank_write_beats,
                        math.ceil(common_external/external_bytes))
    gather_bound = max(compute_beats, source_gather_beats,
                       math.ceil(common_external/external_bytes))
    width_sensitivity = []
    for bits in (128, 384):
        wb = math.ceil(co*32/(8*bits))
        width_sensitivity.append({
            'bank_word_bits': bits, 'beats_per_vector': wb,
            'coefficient_service_beats': scatter_updates*wb,
            '128bit_parallel_word_ports_across_all_banks': 8*math.ceil(bits/128),
            'qualification': '384bit uses three parallel 128bit slices per logical bank; not an isoarea comparison.'})
    completion_burst = np.bincount(last, minlength=p)
    # Completion in coefficient-service work units, without inventing DMA or
    # normalization timestamps. This is a waiting proxy, not observed latency.
    work = count.astype(np.int64)*source_degree[None, :]*beats
    cumulative_work = np.cumsum(work.reshape(-1))
    completion_index = np.arange(t)[:, None]*p+last[None, :]
    waiting_proxy = cumulative_work[-1]-cumulative_work[completion_index]
    # This diagnostic intentionally separates bank-load balance from a legal
    # 8-spatial-lane schedule: replicated/broadcast weights and compaction are
    # not granted, so these cannot be used as speedup figures.
    bank_load = [int(updates[:, np.arange(q) % 8 == bank].sum()) for bank in range(8)]
    phase_max_sum, raster_max_sum = 0, 0
    for it in range(t):
        grid = updates[it].reshape(2*h, 2*w)
        raster_max_sum += int(grid.reshape(-1, 8).max(1).sum())
        for py in range(2):
            for px in range(2):
                phase = grid[py::2, px::2]
                phase_max_sum += int(phase.reshape(-1, 8).max(1).sum())
    # Dynamic BN: each channel sees all T*Hout*Wout values. Complete T values
    # at one spatial coordinate do not close this normalization domain.
    psn_mac = t*t*q*co
    consumer_gate_bytes = math.ceil(t*q*co/8)
    traffic_saved_by_recompute = 2*y_bytes-gate_bytes
    crossover = traffic_saved_by_recompute/compute_beats
    result = {
        'sample_id': row['global_sample_id'], 'sample_key': row['sample_key'],
        'sequence': row['sequence'], 'module': row['name'],
        'input_shape': row['input']['shape'], 'output_shape': [t, b, co, 2*h, 2*w],
        'amplitude_check': amplitude,
        'exact_support_work': {'active_source_scalars': int(count.sum()),
                              'active_source_tap_vector_updates': scatter_updates,
                              'active_scalar_weight_contributions': scatter_updates*co,
                              'dense_valid_scalar_weight_contributions': int(t*ci*co*source_degree.sum()),
                              'cropped_tap_active_vector_updates': int(count.sum())*9-scatter_updates,
                              'output_vectors': output_vectors, 'output_vectors_with_active_input': n_nonempty,
                              'output_all_input_zero_vectors': output_vectors-n_nonempty,
                              'source_site_active_channels_quantiles': np.quantile(count,[0,.1,.5,.9,1]).tolist(),
                              'output_active_contributions_quantiles': np.quantile(updates,[0,.1,.5,.9,1]).tolist()},
        'topology': {'kernel': [3,3], 'stride': [2,2], 'padding': [1,1], 'output_padding': [1,1],
                     'valid_spatial_tap_edges_per_t': int(degree.sum()),
                     'output_input_site_degree_histogram': {str(k):int(np.count_nonzero(degree==k)) for k in [1,2,4]}},
        'scatter_frontier': {'by_T_interleave': frontiers,
                             'active_last_source_oracle_peak_vectors_T1': oracle_peak,
                             'ordinary_geometric_peak_vectors_T1': geom_peak,
                             'last_active_source_metadata_upper_bound_state_saving_vectors': geom_peak-oracle_peak,
                             'direct_final_result_bypass_vectors': direct_retire,
                             'completion_after_zero_source_requires_final_read_vectors': late_retire,
                             'geometric_completion_burst_vectors_histogram': {
                                 str(k):int(np.count_nonzero(completion_burst==k))
                                 for k in np.unique(completion_burst)},
                             'completion_to_BN_seal_coefficient_work_wait_quantiles':
                                 np.quantile(waiting_proxy,[0,.1,.5,.9,1]).tolist(),
                             'waiting_unit': 'coefficient service beats only; not measured elapsed cycles',
                             'oracle_warning': 'Earlier last-active completion needs lookahead or metadata production; not free prediction.'},
        'service_comparison': {
            'scatter': {'organization': 'full_Cin_3x3_local_bundle_then_scatter',
                        'local_kernel_contribution_register_vectors': 9,
                        'register_output_vectors': 0, 'partial_sum_peak_bytes': geom_peak*vector_bytes,
                        'local_bundle_vector_adds': local_bundle_adds,
                        'destination_merge_vector_adds': destination_merge_adds,
                        'source_bundle_scatter_vectors': source_bundle_vectors,
                        'source_bitmap_local_lookups': source_lookups_scatter,
                        'source_bitmap_local_bytes': source_lookups_scatter*source_vector_bytes,
                        'psum_vector_reads': psum_reads, 'psum_vector_writes': psum_writes,
                        'psum_read_write_bytes': (psum_reads+psum_writes)*vector_bytes,
                        'per_bank_read_beats': scatter_bank_read_beats,
                        'per_bank_write_beats': scatter_bank_write_beats,
                        'stage_resource_service_lower_bound': scatter_bound,
                        'unoverlapped_merge_beat_budget': destination_merge_adds*beats,
                        'weak_scalar_push_psum_read_write_bytes_not_the_baseline':
                            (2*scatter_updates-n_nonempty-direct_retire+late_retire)*vector_bytes},
            'gather': {'register_output_vectors': 1, 'partial_sum_peak_bytes': vector_bytes,
                       'source_bitmap_local_lookups': source_lookups_gather,
                       'source_bitmap_local_bytes': source_lookups_gather*source_vector_bytes,
                       'psum_vector_reads': 0, 'psum_vector_writes': 0,
                       'psum_read_write_bytes': 0,
                       'stage_resource_service_lower_bound': gather_bound},
            'common': {'weight_vector_reads': scatter_updates, 'weight_internal_read_bytes': scatter_updates*vector_bytes,
                       'coefficient_service_beats': compute_beats,
                       'packed_source_DRAM_bytes_with_theta_side_information': gate_bytes,
                       'cold_weight_DRAM_bytes': weight_bytes,
                       'full_output_save_DRAM_bytes': y_bytes,
                       'native_TCYX_to_spatial_source_relayout_byte_lower_bound': input_relayout_bytes,
                       'relayout_service_lower_bound': layout_cost,
                       'cold_common_DRAM_bytes': common_external,
                       'primary_bank_spatial_arbitration_conflicts': 0,
                       'bank_conflict_reason': 'Each vector addresses each of eight channel-striped banks once per beat; retirement shares its read port and is counted above.',
                       'common_RF_vector_capacity': 16,
                       'common_RF_bytes': 16*vector_bytes,
                       'output_DMA_queue_vectors_assumed': 4,
                       'output_DMA_queue_bytes': 4*vector_bytes},
            'bank_width_sensitivity': width_sensitivity},
        'alternate_spatial_mapping_diagnostic': {
            'destination_mod8_contribution_load': bank_load,
            'global_bank_load_balance_upper_bound': scatter_updates/(8*max(bank_load)),
            'fixed8_raster_group_arithmetic_utilization': scatter_updates/(8*raster_max_sum),
            'fixed8_phase_group_arithmetic_utilization': scatter_updates/(8*phase_max_sum),
            'claim': 'Actual-address/count diagnostic only. Different lane axis needs priced coefficient broadcasts/replication, queues and source routing.'},
        'consumer_barrier': {
            'BN_module': f'sttmultires_unet.decoders.{row["name"].split(".decoders.")[1].split(".")[0]}.norm_layer.norm_layer',
            'BN_policy': 'no_running', 'BN_elements_per_channel': t*q,
            'BN_fp32_output_state_bytes': y_bytes,
            'Y_save_then_read_bytes': 2*y_bytes,
            'post_BN_fullrank_PSN_scalar_MACs': psn_mac,
            'post_BN_PSN_96lane_arithmetic_issue_lower_bound': t*t*q,
            'post_PSN_gate_bytes_plus_static_theta': consumer_gate_bytes,
            'raw_T_major_width_needed_before_last_T_without_BN_bytes': (t-1)*q*vector_bytes,
            'ordinary_unconditional_retirement_before_BN_seal': False,
            'simple_recompute_extra_conv_coefficient_service_beats': compute_beats,
            'simple_recompute_extra_source_read_bytes': gate_bytes,
            'simple_recompute_saves_Y_write_read_bytes': 2*y_bytes,
            'optimistic_recompute_traffic_vs_compute_crossover_external_bytes_per_beat': crossover,
            'crossover_warning': 'Optimistic necessary bandwidth comparison: overlap, state, normalization and source availability still matter; not a measured break-even.'},
        'numeric_spots': numeric_spots(values, weights, preds, taps, 2*h, 2*w),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='0,10,20,30')
    ap.add_argument('--bank-bits', type=int, default=128)
    ap.add_argument('--external-bytes', type=int, default=16)
    args = ap.parse_args()
    rows = [json.loads(x) for x in (CAP/'unified_ordered_records.jsonl').open()]
    bn = {(r['global_sample_id'],r['name']):r for r in rows if r['category']=='batch_norm'}
    ranked = []
    for stage in range(4):
        group = [r for r in rows if r['category']=='decoder_convtranspose' and f'.decoders.{stage}.' in r['name']]
        proxy = []
        for r in group:
            t,b,ci,h,w = r['input']['shape']
            co = bn[r['global_sample_id'],f'sttmultires_unet.decoders.{stage}.norm_layer.norm_layer']['input']['shape'][2]
            mac = t*b*ci*co*(3*h-1)*(3*w-1)
            proxy.append(mac*r['input']['active']/r['input']['elements'])
        ranked.append({'stage':stage, 'samples':len(group), 'active_MAC_proxy_per_frame':float(np.mean(proxy))})
    stage = max(ranked,key=lambda x:x['active_MAC_proxy_per_frame'])['stage']
    selected = [r for r in rows if r['category']=='decoder_convtranspose' and f'.decoders.{stage}.' in r['name']
                and r['global_sample_id'] in [int(x) for x in args.samples.split(',')]]
    weights,param = parameters(stage)
    print('selected_decoder',stage,'parameter_shapes',param['convtranspose_weight_shape'],flush=True)
    results = []
    for row in selected:
        out = analyze(row, weights, param, args.bank_bits, args.external_bytes)
        results.append(out)
        print('sample',out['sample_id'],'updates',out['exact_support_work']['active_source_tap_vector_updates'],
              'frontier_bytes',out['service_comparison']['scatter']['partial_sum_peak_bytes'],
              'source_gather_bytes',out['service_comparison']['gather']['source_bitmap_local_bytes'],flush=True)
    result = {'status':'REAL_EP34_PAYLOAD_SUPPORT_AND_FINITE_STATE_SERVICE_BOUNDS',
              'selection':'Most expensive decoder by same 40-sample activity-MAC proxy, then first sample of each sequence; no resampling.',
              'ranking':ranked,'selected_stage':stage,'parameters':param,
              'resource_model':{'lanes':96,'channel_striped_banks':8,'bank_word_bits':args.bank_bits,
                                'FP32_channels_per_bank_per_vector':12,
                                'bank_beats_per_vector':math.ceil(384/args.bank_bits),
                                'bank_ports':'1R1W; separate coefficient and partial-sum memories',
                                'partial_sum_capacity_bytes':128*1024,'source_line_scratch_capacity_bytes':16*1024,
                                'coefficient_cache_capacity_bytes':1024*1024,'common_RF_FP32_vector_capacity':16,
                                'external_shared_bytes_per_beat':args.external_bytes,
                                'state_width':'FP32 capacity and service model; no fixed-point or IEEE arithmetic equivalence claim'},
              'samples':results,
              'limits':['No VCS/RTL cycle result, PPA, or full-network speedup.',
                        'Only four full D3 input payloads; other decoder stages ranked from same40 shape/activity proxy.',
                        'No complete captured Y/BN/PSN consumer values or measured arrival times; no consumer accuracy/restore claim.',
                        '294.912MB Y is the full-N96 save format, not a universal minimum: output-channel stripes or source recomputation change residency and supply.',
                        'Post-PSN gate bytes describe optional materialization; final 2-channel head and temporal sum may be fused, so gate writes are not compulsory.',
                        'Coefficient and psum service are explicit lower bounds with overlap, not a cycle-accurate pipeline.',
                        'The input is an already produced theta*g tensor; upstream skip/PSN work is not saved.',
                        'No source indices magically appear: bitmap lookup traffic is counted, but detailed sparse-decode pipeline is unmodeled.',
                        'Output-queue stalls and native-layout burst utilization remain unresolved. Relayout byte cost is separately counted.',
                        'W is dense and theta is read from checkpoint and checked against actual FP32 payload; no unit-amplitude assumption.',
                        'Ordinary phase grouping, row buffers and affine completion are strong baseline behaviors, not claimed innovations.']}
    (HERE/'decoder_probe_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    table=[]
    for sample in results:
        for org in ('scatter','gather'):
            v=sample['service_comparison'][org]
            table.append({'sample':sample['sample_id'],'sequence':sample['sequence'],'organization':org,
                          'vector_updates':sample['exact_support_work']['active_source_tap_vector_updates'],
                          'partial_sum_peak_bytes':v['partial_sum_peak_bytes'],
                          'local_source_bitmap_bytes':v['source_bitmap_local_bytes'],
                          'partial_sum_SRAM_read_write_bytes':v['psum_read_write_bytes'],
                          'resource_service_lower_bound':v['stage_resource_service_lower_bound'],
                          'saved_output_FP32_bytes':sample['consumer_barrier']['BN_fp32_output_state_bytes']})
    with (HERE/'decoder_cost_table.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=table[0]);writer.writeheader();writer.writerows(table)


if __name__ == '__main__':
    main()
