"""Three same-student temporal-code routes, with static row folding baseline.

Only reads existing captures/parameters. Reuses the finite physical-word
coefficient service from support_service_model; does not alter its results.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import Counter
import argparse
import json
import math
import time
import numpy as np
from support_service_model import read_torch, schedule_fc1, serializable

HERE = Path(__file__).resolve().parent
ALG = HERE.parent/'algorithm'
CAP = ALG/'temporal_codes'
T, P, C, H, TILE, LANES = 10, 19200, 96, 384, 32, 96


def compile_routes(D, A):
    groups = []
    seen = {}
    zeros = []
    for t in range(T):
        signature = tuple(D[:, t])
        if not any(signature):
            zeros.append(t)
            continue
        if signature not in seen:
            seen[signature] = len(groups)
            groups.append([])
        groups[seen[signature]].append(t)
    representatives = [g[0] for g in groups]
    folded = np.stack([A[:, g].sum(1) for g in groups], axis=1)
    basis = A @ D.astype(np.int64).T
    routes = {
        'direct_T10_zero_skip': (D.T.copy(), A),
        'folded_time_rows': (D[:, representatives].T.copy(), folded),
        'code_basis_B': (np.eye(len(D), dtype=bool)[1:], basis[:, 1:]),
    }
    assert groups == [[0], [1, 4], [2], [3], [5], [6], [7]]
    assert zeros == [8, 9]
    assert np.array_equal(basis, np.load(CAP/'transformed_temporal_int32.npy'))
    for _, coeff in routes.values():
        assert coeff.min() >= -32768 and coeff.max() <= 32767
    return routes, groups, zeros


def psn_lookup(coeff):
    """Zero-source-aware T10 output schedule for R input slots."""
    R = coeff.shape[1]
    out = []
    for pattern in range(1 << R):
        if pattern == 0:
            out.append(dict(beats=1, vector_MACs=0, slot_reads=0,
                            comparisons=0, U_first_products=0))
            continue
        now, last, seen = 1, [-100]*T, set()
        macs, reads = 0, 0
        for s in range(R):
            target = np.flatnonzero(coeff[:, s])
            if not (pattern >> s & 1) or not len(target):
                continue
            reads += 1
            for t in target:
                t = int(t)
                now = max(now, last[t]+4)
                last[t] = now
                now += 1
                macs += 1
                seen.add(t)
        compare_end = 0
        for t in range(T):
            compare_end = max(compare_end, last[t]+4, 0)+1
        finish = max(now, compare_end)+1
        out.append(dict(beats=finish, vector_MACs=macs, slot_reads=reads,
                        comparisons=T, U_first_products=len(seen)))
    return out


def encoder_frontend(positions, output_beats):
    """Two128-byte raw packets, one in encoder and at most one waiting.

    Each p packet contains fullT decisions, not projected codes. Actual raw
    values are unavailable, so the fixed768 Hamming operations/p are paid.
    Six units evaluate8 candidates for each of16 channel bundles:128 beats/p.
    All routes pay the same ten-beat final encoder+masked-row-write tail.
    """
    now, returned, requested, waiting, started = 0, -1, 0, 0, 0
    active, ready, peak, conflicts = False, 0, 0, 0
    while True:
        if active and now >= ready:
            active = False
            if started == positions:
                assert requested == positions and waiting == 0 and returned < 0
                return dict(beats=now+10, raw_packets=requested,
                            packet_peak=peak, bus_conflicts=conflicts)
        if returned == now:
            waiting += 1
            returned = -1
        if not active and waiting:
            waiting -= 1
            started += 1
            active = True
            ready = now+128
        occupied = int(active)+waiting+int(returned >= 0)
        peak = max(peak, occupied)
        if requested < positions and occupied < 2 and returned < 0:
            if now in output_beats:
                conflicts += 1
            else:
                requested += 1
                returned = now+1
                peak = max(peak, occupied+1)
        now += 1
        if returned < 0 and active and now < ready and (requested == positions or int(active)+waiting >= 2):
            now = ready


def pipeline_step(previous, compute, outputs, positions):
    if previous:
        start = previous['compute_start']
        blocked = set(previous['outputs'])
        old_end = previous['end']
    else:
        start, blocked, old_end = 0, set(), 0
    front = encoder_frontend(positions, blocked)
    compute_start = max(old_end, start+front['beats'])
    return dict(compute_start=compute_start, end=compute_start+compute, outputs=outputs,
                compute_idle=previous.get('compute_idle', 0)+compute_start-old_end,
                frontend_service=previous.get('frontend_service', 0)+front['beats'],
                raw_input_packets=previous.get('raw_input_packets', 0)+front['raw_packets'],
                raw_packet_peak=max(previous.get('raw_packet_peak', 0), front['packet_peak']),
                input_output_bus_conflicts=previous.get('input_output_bus_conflicts', 0)+front['bus_conflicts'])


def pool(coeff):
    part = dict(W=C*H, original_tau48=T*H*6, PSN_coeff16_padded=math.ceil(coeff.size*2/16)*16,
                zero_state_gate_template=T*H//8, dictionary_padded=16, theta_context_padded=16)
    size = sum(part.values())
    return dict(parts=part, payload_bytes=size, allocated_bytes=131072,
                cold_fill_beats=math.ceil(size/128), slack_bytes=131072-size)


def numeric_probe(codes, W, routes, params):
    pp = np.array([0, 1, 31, 32, 127, P-1])
    U = {}
    state_bounds = {}
    w = W.astype(np.int64)
    for name, (decode, coeff) in routes.items():
        source = decode[:, codes[pp]].astype(np.int64)
        slots = np.einsum('rpc,hc->rph', source, w)
        U[name] = np.einsum('tr,rph->tph', coeff, slots)
        state_bounds[name] = int(np.abs(slots).max())
    reference = U['direct_T10_zero_skip']
    tau = params['threshold_int64'][:, None, :]
    positive = params['positive_gain'][None, None, :]
    constants = params['constant_channels'][None, None, :]
    const_gate = params['constant_gate'][:, None, :]
    def gates(u):
        return np.where(constants, const_gate, np.where(positive, u >= tau, u <= tau))
    ref_gate = gates(reference)
    mismatch = {name: dict(U=int(np.count_nonzero(u != reference)),
                           gate=int(np.count_nonzero(gates(u) != ref_gate))) for name, u in U.items()}
    assert all(v['U'] == v['gate'] == 0 for v in mismatch.values())
    w_bound = np.abs(w).sum(1)
    bound = {name: int((np.abs(coeff).sum(1)[:, None]*w_bound[None]).max())
             for name, (_, coeff) in routes.items()}
    assert w_bound.max() < (1 << 23) and max(bound.values()) < (1 << 47)
    assert tau.min() >= -(1 << 47) and tau.max() < (1 << 47)
    return dict(positions=pp.tolist(), U_values_each_route=int(reference.size),
                mismatches_against_direct=mismatch, observed_slot_abs=state_bounds,
                INT24_slot_bound=int(w_bound.max()), conservative_INT48_U_bounds=bound,
                original_tau_min=int(tau.min()), original_tau_max=int(tau.max()),
                theta_source=float(params['theta_source']), theta_output=float(params['theta_output']),
                note='Same new integer student. Raw preprojection trajectories and source PSN producer are not numerically reconstructed here; native FP32 identity is not claimed.')


def evaluate(variant, name, routes, params):
    W = read_torch(CAP/f'{variant}_weight_int8.pt')
    with np.load(CAP/f'{variant}_{name}_codes.npz') as captured:
        codes, shape = captured['codes'], tuple(captured['shape'])
    assert shape == (T, P, C) and codes.shape == (P, C)
    assert codes.min() >= 0 and codes.max() < 8
    assert not (W.reshape(4, LANES, C) == 0).all(1).any()
    rows = {key: Counter() for key in routes}
    pipelines = {key: {} for key in routes}
    banks = {key: np.zeros(8, np.int64) for key in routes}
    psn = {key: psn_lookup(v[1]) for key, v in routes.items()}
    resource = json.loads((HERE/'temporal_basis_service_resources.json').read_text())
    shared_other = sum(resource['common_other_raw_bytes'].values())
    for lo in range(0, P, TILE):
        nc = min(TILE, P-lo)
        code = codes[lo:lo+nc]
        same_reads = []
        for key, (decode, coeff) in routes.items():
            ctr = rows[key]
            features = decode[:, code]
            R = len(decode)
            live = features.any(-1)
            mask = features.reshape(R*nc, C)
            counts = mask.sum(0)
            jobs = [(c, int(counts[c])) for c in range(C) if counts[c]]
            service = schedule_fc1(jobs, 16, 0)
            # Each96h block starts at a multiple-of8 physical word boundary.
            patterns = (live.T*(1 << np.arange(R))).sum(1)
            consumers = [psn[key][int(x)] for x in patterns]
            updates = int(mask.sum())*4
            assigned = int(live.sum())*4
            ctr['FC1_vector_updates'] += updates
            ctr['FC1_first_assignments'] += assigned
            ctr['FC1_adds_after_first'] += updates-assigned
            ctr['FC1_beats'] += service['beats']*4
            ctr['FC1_wait_beats'] += service['core_wait_beats']*4
            ctr['W_read_words'] += service['memory_words']*4
            ctr['W_active_read_beats'] += service['memory_active_beats']*4
            ctr['mask_column_reads'] += service['mask_reads']*4
            ctr['slot_write_bytes'] += updates*LANES*3
            ctr['slot_read_for_add_bytes'] += (updates-assigned)*LANES*3
            ctr['prefetch_peak_bytes'] = max(ctr['prefetch_peak_bytes'], service['prefetch_peak_bytes'])
            ctr['prefetch_descriptor_peak'] = max(ctr['prefetch_descriptor_peak'], service['descriptor_peak'])
            banks[key] += np.asarray(service['bank_reads'])*4
            same_reads.append(service['memory_words'])
            elapsed, outputs = 0, []
            for _ in range(4):
                elapsed += service['beats']+46  # original tau+zero-output template
                ctr['tau_read_beats'] += 46
                for consume in consumers:
                    elapsed += consume['beats']
                    outputs.append(elapsed-1)
                    ctr['PSN_beats_including_output'] += consume['beats']
                    ctr['PSN_vector_MAC_issues'] += consume['vector_MACs']
                    ctr['PSN_slot_read_bytes'] += consume['slot_reads']*LANES*3
                    ctr['PSN_U_write_bytes'] += consume['vector_MACs']*LANES*6
                    ctr['PSN_U_read_bytes'] += (consume['vector_MACs']-consume['U_first_products'])*LANES*6
                    ctr['PSN_vector_comparisons'] += consume['comparisons']
            pipelines[key] = pipeline_step(pipelines[key], elapsed, outputs, nc)
        assert len(set(same_reads)) == 1  # no extra W reads on either new route
    modes = {}
    for key, (_, coeff) in routes.items():
        v = dict(rows[key])
        v['pool'] = pool(coeff)
        v['wide_slots'] = coeff.shape[1]
        v['wide_slot_raw_bytes'] = coeff.shape[1]*TILE*LANES*3
        v['common_other_raw_bytes'] = shared_other
        v['W_read_bytes'] = v['W_read_words']*16
        v['PSN_coeff_count'] = int(coeff.size)
        v['PSN_coeff_nonzeros'] = int(np.count_nonzero(coeff))
        v['PSN_coeff_abs_max'] = int(np.abs(coeff).max())
        v['PSN_scalar_MAC_issues'] = v['PSN_vector_MAC_issues']*LANES
        v['PSN_coefficient_register_read_bytes'] = v['PSN_vector_MAC_issues']*2
        v['W_bank_reads'] = banks[key].tolist()
        v['pipeline'] = {k: x for k, x in pipelines[key].items() if k not in ('outputs','compute_start')}
        assert v['pipeline']['raw_packet_peak'] <= 2 and v['pipeline']['raw_input_packets'] == P
        v['raw_source_transport_bytes'] = P*128
        v['output_transport_bytes'] = P*4*128
        v['Hamming_candidate_evaluations'] = P*C*8
        v['Hamming_core_service_beats'] = P*C*8//6
        v['warm_chain_beats'] = pipelines[key]['end']
        v['cold_chain_beats'] = v['warm_chain_beats']+v['pool']['cold_fill_beats']
        modes[key] = v
    baseline = modes['folded_time_rows']
    for key, v in modes.items():
        v['warm_chain_reduction_vs_folded'] = 1-v['warm_chain_beats']/baseline['warm_chain_beats']
        v['cold_chain_reduction_vs_folded'] = 1-v['cold_chain_beats']/baseline['cold_chain_beats']
        v['FC1_reduction_vs_folded'] = 1-v['FC1_beats']/baseline['FC1_beats']
        v['PSN_reduction_vs_folded'] = 1-v['PSN_beats_including_output']/baseline['PSN_beats_including_output']
    return dict(variant=variant,frame=name,code_histogram=np.bincount(codes.ravel(),minlength=8).tolist(),
                numeric=numeric_probe(codes,W,routes,params),modes=modes)


def aggregate(frames):
    out = {}
    for variant in sorted({f['variant'] for f in frames}):
        fs = [f for f in frames if f['variant'] == variant]
        result = {}
        for key in fs[0]['modes']:
            vv = [f['modes'][key] for f in fs]
            names = ('FC1_vector_updates','FC1_adds_after_first','FC1_beats','FC1_wait_beats',
                     'W_read_bytes','slot_write_bytes','slot_read_for_add_bytes',
                     'PSN_beats_including_output','PSN_scalar_MAC_issues','PSN_slot_read_bytes',
                     'PSN_U_read_bytes','PSN_U_write_bytes','PSN_coefficient_register_read_bytes',
                     'raw_source_transport_bytes','output_transport_bytes',
                     'Hamming_candidate_evaluations','Hamming_core_service_beats','warm_chain_beats','cold_chain_beats')
            d = {n:sum(v[n] for v in vv) for n in names}
            d['wide_slots'] = vv[0]['wide_slots']
            d['wide_slot_raw_bytes'] = vv[0]['wide_slot_raw_bytes']
            d['pool'] = vv[0]['pool']
            d['mean_compute_idle'] = sum(v['pipeline']['compute_idle'] for v in vv)/len(vv)
            d['input_output_bus_conflicts'] = sum(v['pipeline']['input_output_bus_conflicts'] for v in vv)
            d['per_frame_warm_reduction_min'] = min(v['warm_chain_reduction_vs_folded'] for v in vv)
            d['per_frame_warm_reduction_max'] = max(v['warm_chain_reduction_vs_folded'] for v in vv)
            result[key] = d
        base = result['folded_time_rows']
        for d in result.values():
            for k in ('warm_chain_beats','cold_chain_beats','FC1_beats','PSN_beats_including_output'):
                d[k+'_reduction_vs_folded'] = 1-d[k]/base[k]
        out[variant] = dict(frames=len(fs),modes=result)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames',type=int,default=10)
    ap.add_argument('--output',default='temporal_basis_service_result.json')
    args = ap.parse_args()
    D = np.load(CAP/'dictionary.npy')
    assert D.shape == (8,10)
    params=read_torch(ALG/'integer_s0_valid825/integer_parameters.pt')['sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.']
    A = params['temporal_int16'].astype(np.int64)
    routes, groups, zeros = compile_routes(D,A)
    parent_frames = json.loads((CAP/'frames.json').read_text())
    start,frames=time.monotonic(),[]
    for variant in ('untrained_code8','trained_code8'):
        names=[Path(f['file']).stem for f in parent_frames if f['variant']==variant][:args.frames]
        for name in names:
            f=evaluate(variant,name,routes,params)
            frames.append(f)
            print(variant,name,{k:round(v['warm_chain_reduction_vs_folded']*100,4) for k,v in f['modes'].items()},flush=True)
            result=dict(kind='finite-resource service schedule, not RTL cycles or PPA',
                        resources='temporal_basis_service_resources.json',
                        baseline='same-student static zero/duplicate-time-row folding, seven wide slots',
                        exact_fold_groups=groups,zero_time_rows=zeros,
                        original_tau_unchanged=True,dictionary_rank=int(np.linalg.matrix_rank(D)),
                        frames=frames,aggregate=aggregate(frames),elapsed_wall_s=time.monotonic()-start)
            (HERE/args.output).write_text(json.dumps(serializable(result),ensure_ascii=False,indent=2)+'\n')
    print('FINISHED',len(frames),time.monotonic()-start,flush=True)


if __name__=='__main__':
    main()
