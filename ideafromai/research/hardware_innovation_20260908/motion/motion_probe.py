#!/usr/bin/env python3.12
"""Three causal adjacent-frame pairs; actual support counts, not RTL cycles.

The previous completed full-resolution flow selects one integer displacement
per 16x16 output tile. Exact linear correction uses that same displacement for
the entire tile's 3x3 receptive halo, and caches continuous pre-BN output.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path('/home/zhumd/work')
HW = WORK / 'sdformer_codex/SDformer/hw_autoresearch_nts07'
CAP = HW / 'results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831'
MODULE = 'sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.1.conv1.0'
TIMES = [(58054907646, 58055007601), (58055007601, 58055107643),
         (58055107643, 58055207650), (58055207650, 58055307648)]
TILE = 16


def availability():
    manifest = json.loads((CAP / 'manifest.json').read_text())
    seqs = {}
    for sample in manifest['cohort']['samples']:
        seqs.setdefault(sample['sequence'], []).append(sample['sample_key'])
    ops = json.loads((CAP / 'operator_runtime.json').read_text())
    rank = sorted([x for x in ops if 'patch_embed.residual_encoding.resblocks.' in x['name']
                   and x['operator'] == 'Conv2d'],
                  key=lambda x: x['activity_weighted_macs_proxy'], reverse=True)
    chosen = next(x for x in rank if x['name'] == MODULE)
    shape = chosen['input_shape_first']
    n = int(np.prod(shape))
    return {
        'old_M1458_sample_keys': seqs,
        'old_M1458_is_consecutive': False,
        'old_source_payload_retained': False,
        'old_pred3_sn_payload_retained': False,
        'selected_module': MODULE, 'input_shape': shape,
        'conv': {'kernel': [3, 3], 'stride': [1, 1], 'padding': [1, 1], 'Cout': 96},
        'patch_residual_rank_by_40_frame_activity_MAC_proxy': [
            {'name': x['name'], 'per_frame_proxy': x['activity_weighted_macs_proxy'] / x['calls']}
            for x in rank],
        'state_bytes': {
            'one_previous_source_gate_plane': (n + 7) // 8,
            'one_previous_pre_BN_Y_FP32': n * 4,
            'old_plus_new_Y_if_no_proved_inplace_schedule': n * 8,
            'full_reuse_old_Y_read_plus_new_Y_write_per_transition': n * 8,
            'full_current_Y_read_for_remaining_dynamic_BN_consumer': n * 4,
        },
    }


def load_frame(path):
    with np.load(path, allow_pickle=False) as z:
        shape = tuple(int(v) for v in z['source_shape'])
        theta = float(np.asarray(z['source_theta']).reshape(()))
        source = np.unpackbits(z['source_bits'], bitorder='little', count=int(np.prod(shape))).reshape(shape)
        sample = path.stem.removesuffix('_source')
    flow_path = path.with_name(sample+'_flow.npz')
    flow = np.load(flow_path)['flow'] if flow_path.exists() else None
    return source[:, 0].astype(bool), theta, flow, sample


def patch(a, y0, y1, x0, x1):
    """Zero extension, including shifted receptive halos at image boundaries."""
    h, w = a.shape[-2:]
    out = np.zeros((*a.shape[:2], y1-y0, x1-x0), dtype=a.dtype)
    ay, by, ax, bx = max(y0, 0), min(y1, h), max(x0, 0), min(x1, w)
    if ay < by and ax < bx:
        out[:, :, ay-y0:by-y0, ax-x0:bx-x0] = a[:, :, ay:by, ax:bx]
    return out


def displacements(flow, h, w, duration_ratio):
    flow = np.asarray(flow).reshape(2, flow.shape[-2], flow.shape[-1])
    sy, sx = flow.shape[-2] / h, flow.shape[-1] / w
    out = {}
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            v = flow[:, int(y*sy):int(min(y+TILE, h)*sy), int(x*sx):int(min(x+TILE, w)*sx)]
            mean = v.mean(axis=(1, 2), dtype=np.float64)
            dx = int(np.rint(mean[0] * duration_ratio / sx))
            dy = int(np.rint(mean[1] * duration_ratio / sy))
            out[y, x] = (dy, dx)
    return out


def count_pair(previous, current, theta_old, theta_new, shifts, use_flow):
    t, ci, h, w = current.shape
    co = 96
    totals = dict(baseline_source_tap_contributions=0, correction_source_tap_contributions=0,
                  previous_source_tap_contributions=0, shared_source_tap_contributions=0,
                  fallback_source_tap_contributions=0, base_valid_output_spatial_positions=0,
                  fallback_output_spatial_positions=0, current_source_tile_reads_bits=0,
                  previous_source_tile_reads_bits=0, previous_halo_reads_bits=0,
                  previous_core_reads_bits=0, source_delta_word128_operations=0,
                  tile_count=0, nonzero_shift_tiles=0, delta_source_added=0,
                  delta_source_deleted=0, delta_source_changed_amplitude=0,
                  unaffected_output_time_vectors=0)
    histogram = {}
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            th, tw = min(TILE, h-y), min(TILE, w-x)
            dy, dx = shifts[y, x] if use_flow else (0, 0)
            histogram[f'{dy},{dx}'] = histogram.get(f'{dy},{dx}', 0) + 1
            a = patch(current, y-1, y+th+1, x-1, x+tw+1)
            b = patch(previous, y-dy-1, y-dy+th+1, x-dx-1, x-dx+tw+1)
            cur = a.sum(axis=(0, 1), dtype=np.int64)
            old = b.sum(axis=(0, 1), dtype=np.int64)
            common = (a & b).sum(axis=(0, 1), dtype=np.int64)
            # Identical gates are still a change if the actual theta changed.
            changed = (a != b) if theta_old == theta_new else (a | b)
            difference = changed.sum(axis=(0, 1), dtype=np.int64)
            valid = ((np.arange(y-dy, y-dy+th)[:, None] >= 0) &
                     (np.arange(y-dy, y-dy+th)[:, None] < h) &
                     (np.arange(x-dx, x-dx+tw)[None, :] >= 0) &
                     (np.arange(x-dx, x-dx+tw)[None, :] < w))
            needed = np.zeros((th+2, tw+2), bool)
            touched = np.zeros((t, th, tw), bool)
            for ky in range(3):
                for kx in range(3):
                    v = cur[ky:ky+th, kx:kx+tw]
                    d = difference[ky:ky+th, kx:kx+tw]
                    totals['baseline_source_tap_contributions'] += int(v.sum())
                    totals['previous_source_tap_contributions'] += int(old[ky:ky+th,kx:kx+tw][valid].sum())
                    totals['shared_source_tap_contributions'] += int(common[ky:ky+th,kx:kx+tw][valid].sum())
                    totals['correction_source_tap_contributions'] += int(d[valid].sum())
                    totals['fallback_source_tap_contributions'] += int(v[~valid].sum())
                    needed[ky:ky+th, kx:kx+tw] |= valid
                    touched |= changed[:, :, ky:ky+th, kx:kx+tw].any(axis=1)
            prev_inbounds = ((np.arange(y-dy-1, y-dy+th+1)[:, None] >= 0) &
                             (np.arange(y-dy-1, y-dy+th+1)[:, None] < h) &
                             (np.arange(x-dx-1, x-dx+tw+1)[None, :] >= 0) &
                             (np.arange(x-dx-1, x-dx+tw+1)[None, :] < w))
            actual_old_reads = needed & prev_inbounds
            core = np.zeros_like(needed)
            core[1:-1, 1:-1] = True
            totals['previous_source_tile_reads_bits'] += t*ci*int(actual_old_reads.sum())
            totals['previous_halo_reads_bits'] += t*ci*int((actual_old_reads & ~core).sum())
            totals['previous_core_reads_bits'] += t*ci*int((actual_old_reads & core).sum())
            valid_current_patch = (min(y+th+1,h)-max(y-1,0)) * (min(x+tw+1,w)-max(x-1,0))
            totals['current_source_tile_reads_bits'] += t*ci*valid_current_patch
            totals['source_delta_word128_operations'] += 3*((t*ci*int(needed.sum())+127)//128)
            totals['base_valid_output_spatial_positions'] += int(valid.sum())
            totals['fallback_output_spatial_positions'] += int((~valid).sum())
            totals['unaffected_output_time_vectors'] += int((~touched[:, valid]).sum())
            totals['delta_source_added'] += int(((a & ~b) & needed).sum())
            totals['delta_source_deleted'] += int(((b & ~a) & needed).sum())
            if theta_new != theta_old:
                totals['delta_source_changed_amplitude'] += int(((a & b) & needed).sum())
            totals['tile_count'] += 1
            totals['nonzero_shift_tiles'] += int(bool(dy or dx))
    baseline = totals['baseline_source_tap_contributions']
    candidate = totals['correction_source_tap_contributions'] + totals['fallback_source_tap_contributions']
    old_y = totals['base_valid_output_spatial_positions']*t*co*4
    y_bytes = t*co*h*w*4
    src_bytes = (totals['current_source_tile_reads_bits']+7)//8
    extra_src_bytes = (totals['previous_source_tile_reads_bits']+7)//8
    flow_bytes = 2*(2*h)*(2*w)*4 if use_flow else 0
    totals.update({'source_tap_ratio_to_bit_skip': candidate/baseline,
                   'scalar_weighted_contributions_original': baseline*co,
                   'scalar_weighted_contributions_candidate': candidate*co,
                   'old_pre_BN_Y_read_bytes': old_y,
                   'new_pre_BN_Y_write_bytes': y_bytes,
                   'remaining_dynamic_BN_Y_read_bytes': y_bytes,
                   'extra_previous_source_read_bytes': extra_src_bytes,
                   'previous_flow_read_bytes': flow_bytes,
                   'displacement_histogram_dy_dx': histogram})
    # Illustrative SERIAL arithmetic+service sum: one 96-output add-vector
    # per cycle, one 128-bit logical operation per cycle, 96 flow-add lanes.
    # It does NOT establish a feasible integrated cycle schedule or PPA.
    flow_ops = ((2*(2*h)*(2*w)-2*totals['tile_count']+95)//96 + 2*totals['tile_count']) if use_flow else 0
    totals['serialized_service_sensitivity'] = []
    for bandwidth in (32, 128, 512):
        common_bytes = src_bytes + 2*y_bytes
        ordinary = baseline + (common_bytes+bandwidth-1)//bandwidth
        delta = candidate + totals['source_delta_word128_operations'] + flow_ops
        delta += (common_bytes+extra_src_bytes+old_y+flow_bytes+bandwidth-1)//bandwidth
        totals['serialized_service_sensitivity'].append({
            'external_bytes_per_cycle': bandwidth,
            'baseline_service_units': ordinary, 'candidate_service_units': delta,
            'ratio_candidate_to_baseline': delta/ordinary})
    return totals


def numeric_spots(previous, current, theta_old, theta_new, shifts, weight, bias):
    """Actual coefficients and full Cin/tap checks; no captured-Y/FP32 claim."""
    h, w = current.shape[-2:]
    errors = []
    checked, fallback = 0, 0
    for y, x in [(0, 0), (0, w//2), (h//2, w//2), (h-1, w-1)]:
        dy, dx = shifts[y//TILE*TILE, x//TILE*TILE]
        if not (0 <= y-dy < h and 0 <= x-dx < w):
            fallback += 1
            continue
        a = patch(current, y-1, y+2, x-1, x+2)[0].astype(np.float64)*theta_new
        b = patch(previous, y-dy-1, y-dy+2, x-dx-1, x-dx+2)[0].astype(np.float64)*theta_old
        ref = np.einsum('ockl,ckl->o', weight, a) + bias
        old = np.einsum('ockl,ckl->o', weight, b) + bias
        delta = np.einsum('ockl,ckl->o', weight, a-b)
        errors.append(float(np.max(np.abs(ref-(old+delta)))))
        checked += int(ref.size)
    return {'FP64_coefficient_outputs_checked': checked,
            'out_of_old_grid_full_recompute_points': fallback,
            'max_absolute_error': max(errors, default=0.0),
            'claim': 'linear identity spot check only; no frozen FP32 equivalence'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', type=Path, default=HERE/'capture')
    args = ap.parse_args()
    result = {'status': 'availability_only_pending_adjacent_capture', 'old_capture': availability(),
              'comparison': 'same pair: ordinary bit-skip vs direct delta vs previous-flow tile delta',
              'tile': [TILE, TILE], 'model_change': 'none in real arithmetic; FP32 order not proved',
              'remaining_work': 'original dynamic BN, complete T10 PSN, downstream residual/decoder all remain',
              'base_reuse_equation': 'Ynew(u)=Yold(u-d)+sum_k W[k]*(znew(u+k)-zold(u-d+k)); old output outside grid uses full current convolution',
              'pair_intervals_us': TIMES,
              'limitations': ['no RTL, clock, PPA, or system speedup',
                              'full previous FP32 pre-BN Y is additional required persistent state',
                              'new Y write and BN read overlap costs already present in a store-Y baseline; they are not counted twice',
                              'the service example shares a store-source/store-Y baseline; a stronger streaming or calibrated-BN baseline must not inherit these unnecessary materializations',
                              'previous final prediction must be complete before the next early convolution, restricting cross-frame pipeline overlap',
                              'the serialized service example is neither an integrated timing result nor a mathematical cycle lower bound; resident weights are assumed identically for both modes',
                              'source/weight compaction, cache-line waste and actual accumulator ports remain unmodeled',
                              'continuous theta is retained; bit support suffices only after actual capture confirms a single recoverable nonzero amplitude']}
    paths = sorted(args.capture.glob('*_source.npz'))
    if len(paths) >= 4:
        capture_meta = json.loads((args.capture/'frames.json').read_text())
        frame_meta = capture_meta['frames'][:4]
        with np.load(args.capture/'operator.npz') as z:
            weight, bias = z['weight'].astype(np.float64), z['bias'].astype(np.float64)
            result['actual_operator'] = {'weight_shape': list(weight.shape),
                                        'weight_FP32_bytes': int(weight.size*4),
                                        'stride': z['stride'].tolist(),
                                        'padding': z['padding'].tolist()}
        frames = [load_frame(p) for p in paths[:4]]
        result['status'] = 'three_adjacent_pairs_measured'
        result['capture_metadata'] = capture_meta
        result['sample_keys'] = [f[3] for f in frames]
        result['source_theta_per_frame'] = [f[1] for f in frames]
        result['source_active_per_frame'] = [int(f[0].sum()) for f in frames]
        result['pairs'] = []
        for i in range(3):
            previous, old_theta, flow, _ = frames[i]
            current, new_theta, _, _ = frames[i+1]
            duration = lambda m: m['timestamp_end_us']-m['timestamp_start_us']
            ratio = duration(frame_meta[i+1])/duration(frame_meta[i])
            shifts = displacements(flow, *current.shape[-2:], ratio)
            result['pairs'].append({'previous': frames[i][3], 'current': frames[i+1][3],
                                    'duration_ratio': ratio,
                                    'direct_full_source_delta_nonzeros': int(np.count_nonzero(current != previous)) if old_theta == new_theta else int(np.count_nonzero(current | previous)),
                                    'direct': count_pair(previous,current,old_theta,new_theta,shifts,False),
                                    'causal_flow': count_pair(previous,current,old_theta,new_theta,shifts,True),
                                    'numeric_spots': numeric_spots(previous,current,old_theta,new_theta,shifts,weight,bias)})
        baseline = sum(p['direct']['baseline_source_tap_contributions'] for p in result['pairs'])
        result['aggregate'] = {'baseline_source_tap_contributions': baseline}
        for mode in ['direct','causal_flow']:
            candidate = sum(p[mode]['correction_source_tap_contributions'] +
                            p[mode]['fallback_source_tap_contributions'] for p in result['pairs'])
            shared = sum(p[mode]['shared_source_tap_contributions'] for p in result['pairs'])
            old = sum(p[mode]['previous_source_tap_contributions'] for p in result['pairs'])
            result['aggregate'][mode] = {'candidate_contributions': candidate,
                                         'ratio_to_original': candidate/baseline,
                                         'fraction_of_old_contributions_cancelled_by_shared_gate': shared/old}
        result['screen_decision'] = {
            'expand_four_sequences': False,
            'reason': 'all three prescribed pairs already increase arithmetic before extra continuous-Y state and old-source/flow reads',
            'scope': 'this layer, these first three adjacent pairs, fixed 16x16 causal integer shift and exact linear correction; not a rejection of all motion-aware models',
            'missing_to_claim_FP32_identity': 'full native ordered arithmetic equivalence or an explicitly evaluated AEE Pareto',
        }
    (HERE/'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'status': result['status'], 'result': str(HERE/'result.json'),
                      'pairs': [{k:v for k,v in x.items() if k not in ['direct','causal_flow']} |
                                {m: x[m]['source_tap_ratio_to_bit_skip'] for m in ['direct','causal_flow']}
                                for x in result.get('pairs',[])]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
