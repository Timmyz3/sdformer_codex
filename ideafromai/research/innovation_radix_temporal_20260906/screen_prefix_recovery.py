"""A numerical/operation opportunity screen, not an RTL speedup.

Monotone coarse representation of the existing float64 proxy; exact support
checks within that proxy. Only sample0, two FC1 layers, prefixes 8/12/16.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '4'
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

BASE = Path(__file__).resolve().parent
OLD = BASE.parent/'handoff_reconciliation_20260906'
sys.path.insert(0, str(OLD/'scripts'))
from screen_repair_sectors import load_sources, read_checkpoint, CKPT


def monotone_key(x):
    # Floor to FP32's ordered representable set, while x remains the FP64 proxy.
    y = x.astype(np.float32)
    y = np.where(y.astype(np.float64) > x, np.nextafter(y, np.float32(-np.inf)), y)
    y = np.where(y == 0, np.float32(0), y)  # Numeric -0 and +0 compare equally.
    bits = y.view(np.uint32)
    return np.where(bits >> 31, ~bits, bits ^ np.uint32(0x80000000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    out = ap.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source, _ = load_sources()
    state = read_checkpoint(CKPT)['model_state_dict']
    layers = []
    for stage in (0, 3):
        pre = f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp.'
        spec, S = source[pre+'fc1']
        _, archived = source[pre+'fc2']
        N, C = S.shape
        H, T, P, B = spec['output_channels'], 10, N//10, 32
        W = state[pre+'fc1.weight'].astype(np.float64)
        theta_source = float(state[pre+'sn1.spiking_neuron.thresh'])
        gamma = state[pre+'bn1.norm_layer.weight'].astype(np.float64)
        beta = state[pre+'bn1.norm_layer.bias'].astype(np.float64)
        A = state[pre+'sn2.spiking_neuron.weight'].astype(np.float64)
        bias = state[pre+'sn2.spiking_neuron.bias'].astype(np.float64).reshape(T,1)
        center = state[pre+'sn2.spiking_neuron.center'].astype(np.float64).reshape(T,1)
        theta = float(state[pre+'sn2.spiking_neuron.thresh'])
        row_sum = A.sum(1).reshape(T,1)
        S_time = S.reshape(T,P,C)
        patterns = (S_time.astype(np.uint16)*(1 << np.arange(T,dtype=np.uint16))[:,None,None]).sum(0,dtype=np.uint16)
        pattern_bits = ((np.arange(1 << T)[:,None] >> np.arange(T)) & 1).astype(np.float64)
        table = A @ pattern_bits.T * theta_source
        np.save(out/f'stage{stage}_source_patterns.npy',patterns,allow_pickle=False)
        spikes = S_time.sum(2,dtype=np.int64)
        masks = {k:np.zeros((T,P,H),dtype=bool) for k in (8,12,16)}
        stats = {k:dict(known_prefix_mismatches=0, direct_repair_bit_mismatches=0,
                       direct_repair_max_abs_error=0., direct_MAC_terms=0,
                       direct_table_lookups=0) for k in masks}
        archive_mismatch = 0
        A_dense = bool((A != 0).all())
        for lo in range(0,H,32):
            hi = min(H,lo+32)
            Y = (S.astype(np.float64) @ W[lo:hi].T * theta_source).reshape(T,P,hi-lo)
            mu = Y.mean((0,1))
            std = np.sqrt(((Y-mu)**2).mean((0,1))+1e-5)
            U = np.einsum('ts,sph->tph',A,Y,optimize=True)
            tau = mu*row_sum + std/gamma[lo:hi]*(theta+center-bias-beta[lo:hi]*row_sum)
            direction = np.sign(gamma[lo:hi])
            V, boundary = U*direction, tau*direction
            truth = V >= boundary[:,None,:]
            archive_mismatch += int(np.count_nonzero(truth != archived.reshape(T,P,H)[:,:,lo:hi]))
            key_u, key_tau = monotone_key(V), monotone_key(boundary)
            for k, mask in masks.items():
                pu, pt = key_u >> (32-k), key_tau >> (32-k)
                uncertain = pu == pt[:,None,:]
                mask[:,:,lo:hi] = uncertain
                stats[k]['known_prefix_mismatches'] += int(np.count_nonzero(
                    ((pu > pt[:,None,:]) != truth) & ~uncertain))
            # Recompute only the largest set once; narrower prefixes are subsets.
            # No extra dense table per W/h: each nonzero L is a real MAC with W.
            for t in range(T):
                for j,h in enumerate(range(lo,hi)):
                    ids = np.flatnonzero(masks[8][t,:,h])
                    if not len(ids):
                        continue
                    m = patterns[ids]
                    coeff = table[t,m]
                    direct = coeff @ W[h]
                    pred = direct*direction[j] >= boundary[t,j]
                    error = np.abs(direct-U[t,ids,j])
                    nz = np.count_nonzero(coeff,axis=1)
                    for k, mask in masks.items():
                        selected = mask[t,ids,h]
                        if selected.any():
                            stats[k]['direct_repair_bit_mismatches'] += int(np.count_nonzero(
                                pred[selected] != truth[t,ids[selected],j]))
                            stats[k]['direct_repair_max_abs_error'] = max(
                                stats[k]['direct_repair_max_abs_error'], float(error[selected].max()))
                            stats[k]['direct_MAC_terms'] += int(nz[selected].sum())
                            stats[k]['direct_table_lookups'] += int((m[selected] != 0).sum())
            if hi == H or hi % 512 == 0:
                print('stage',stage,'h',hi,'/',H,'seconds',round(time.monotonic()-started,1),flush=True)
        rows = []
        # Necessary Y time rows can be a subset when A contains exact zeros.
        needed_times = np.array([np.any((A != 0)[[(m >> t)&1 == 1 for t in range(T)]],axis=0)
                                 for m in range(1 << T)],dtype=bool)
        for k, mask in masks.items():
            consumers = mask.sum(0,dtype=np.int64)
            touched = consumers > 0
            consumer_mask = (mask.astype(np.uint16)*(1 << np.arange(T,dtype=np.uint16))[:,None,None]).sum(0,dtype=np.uint16)
            y_time_need = needed_times[consumer_mask]
            baseline_adds = int((y_time_need * spikes.T[:,None,:]).sum(dtype=np.int64))
            psn_mac = int(sum(int(mask[t].sum())*int((A[t] != 0).sum()) for t in range(T)))
            direct_cost_per_ph = np.zeros((P,H),dtype=np.int64)
            for t in range(T):
                active_c = (table[t,patterns] != 0).sum(1,dtype=np.int64)
                direct_cost_per_ph += mask[t]*active_c[:,None]
            assert int(direct_cost_per_ph.sum()) == stats[k]['direct_MAC_terms']
            y_add_per_ph = (y_time_need * spikes.T[:,None,:]).sum(2,dtype=np.int64)
            y_psn_per_ph = sum(mask[t].astype(np.int64)*int((A[t] != 0).sum()) for t in range(T))
            hybrid = []
            for mac_add_cost in (1,4,8,16):
                direct_cost = direct_cost_per_ph*mac_add_cost
                y_cost = y_add_per_ph+y_psn_per_ph*mac_add_cost
                choose = touched & (direct_cost < y_cost)
                hybrid.append(dict(MAC_to_ADD_cost=mac_add_cost,
                    direct_ph=int(choose.sum()), shared_Y_ph=int((touched & ~choose).sum()),
                    chosen_ADD_terms=int(y_add_per_ph[~choose].sum()),
                    chosen_MAC_terms=int(direct_cost_per_ph[choose].sum()+y_psn_per_ph[~choose].sum()),
                    arithmetic_cost_only=int(np.minimum(direct_cost,y_cost).sum())))
            nb = (P+B-1)//B
            padded = np.zeros((T,nb*B,H),dtype=bool)
            padded[:,:P] = mask
            sector_jobs = padded.reshape(T,nb,B,H//4,4).any((2,4))
            row = dict(prefix_bits=k, **stats[k], unknown_outputs=int(mask.sum()),
                unknown_output_fraction=float(mask.mean()), unresolved_ph=int(touched.sum()),
                mean_unresolved_T_per_touched_ph=float(mask.sum()/max(1,touched.sum())),
                consumer_count_histogram=np.bincount(consumers.ravel(),minlength=T+1).tolist(),
                full_FC1_binary_ADD_terms=int(S.sum(dtype=np.int64))*H,
                full_PSN_MAC_terms=int((A != 0).sum())*P*H,
                shared_Y_repair_binary_ADD_terms=baseline_adds,
                shared_Y_repair_PSN_MAC_terms=psn_mac,
                direct_repair_LUT_bytes_FP64=int(table.nbytes),
                prefix_storage_bytes_packed=(N*H*k+7)//8,
                prefix_storage_bytes_B32_128bit_aligned=T*nb*H*((B*k+127)//128)*16,
                full_U_proxy_bytes=N*H*8,
                nominal_full_U32_bytes_not_fp32_equivalence=N*H*4,
                touched_t_B32_q4_jobs=int(sector_jobs.sum()),
                hybrid_arithmetic_sensitivity=hybrid)
            rows.append(row)
            if k == 12:
                np.packbits(mask,axis=2,bitorder='little').tofile(out/f'stage{stage}_prefix12_unknown.le.bitpack')
                np.save(out/f'stage{stage}_prefix12_consumer_mask.npy',consumer_mask,allow_pickle=False)
        layers.append(dict(stage=stage,shape_T_P_H=[T,P,H],C=C,theta=theta,
                           source_theta=theta_source,A_nonzero=int((A != 0).sum()),A_dense=A_dense,
                           archive_support_mismatches=archive_mismatch,rows=rows))
        print('LAYER_RESULT',json.dumps(layers[-1]),flush=True)
    result=dict(scope='sample0, stage0/block0 and stage3/block0 FC1; float64 reconstructed U/BN/PSN',
        mechanism='monotone FP32-format floor then ordered-key prefix; unresolved consumers choose Y sharing or direct temporal-pattern contraction',
        theta='Unchanged checkpoint scalar; output value remains theta*gate',
        limitations=['Prefix monotonicity is checked against float64 proxy, not frozen GPU FP32 arithmetic.',
          'Direct temporal contraction reorders floating operations; all observed threshold flips are retained.',
          'A table entry is a multivalued coefficient: W*L is charged as real MAC, never binary weight-add.',
          'Hybrid only compares arithmetic with arbitrary cost ratios; table reads, queues, state and ports are additional.',
          'Full first FC1/PSN and BN statistics are required. Reported recovery terms are additional work.',
          'Full retained S and uncertainty-map storage, conversion, index traffic and commit still require circuit accounting.'],
        layers=layers,seconds=time.monotonic()-started,RTL_speedup=False,PPA=False)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DONE',result['seconds'],flush=True)


if __name__ == '__main__':
    main()
