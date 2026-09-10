"""Source-domain BN moments: algebraic and operation screen, not a cycle model.

Binary source counts/correlations determine FC1's whole-domain BN statistics
before any FC1 output is materialized. Uses the same two-layer float64 proxy.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
import json
from pathlib import Path
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent/'handoff_reconciliation_20260906/scripts'))
from screen_repair_sectors import load_sources, read_checkpoint, CKPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    out = ap.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=False)
    source, _ = load_sources()
    state = read_checkpoint(CKPT)['model_state_dict']
    results = []
    for stage in (0, 3):
        pre = f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp.'
        spec, S = source[pre+'fc1']
        _, archived = source[pre+'fc2']
        N, C = S.shape
        H, T, P = spec['output_channels'], 10, N//10
        W = state[pre+'fc1.weight'].astype(np.float64)
        theta_s = float(state[pre+'sn1.spiking_neuron.thresh'])
        gamma = state[pre+'bn1.norm_layer.weight'].astype(np.float64)
        beta = state[pre+'bn1.norm_layer.bias'].astype(np.float64)
        A = state[pre+'sn2.spiking_neuron.weight'].astype(np.float64)
        bias = state[pre+'sn2.spiking_neuron.bias'].astype(np.float64).reshape(T, 1)
        center = state[pre+'sn2.spiking_neuron.center'].astype(np.float64).reshape(T, 1)
        theta = float(state[pre+'sn2.spiking_neuron.thresh'])
        R = A.sum(1).reshape(T, 1)
        sf = S.astype(np.float64)
        # Products and sums of 0/1 are exactly represented at these counts.
        G = sf.T @ sf
        n = np.diag(G)
        mu_g = theta_s*(W @ n)/N
        second_g = theta_s**2*np.einsum('hc,hc->h', W @ G, W)/N
        var_g = second_g-mu_g**2
        k = S.sum(1, dtype=np.int64)
        upper = np.triu_indices(C)
        upper_nz = int(np.count_nonzero(G[upper]))
        offdiag_updates = int((k*(k-1)//2).sum())
        diag_updates = int(k.sum())
        errors = dict(max_abs_mean=0., max_abs_variance=0., max_rel_variance=0.,
                      max_abs_threshold=0., moment_vs_direct_bit_mismatches=0,
                      moment_vs_archived_bit_mismatches=0,
                      direct_vs_archived_bit_mismatches=0)
        for lo in range(0, H, 32):
            hi = min(H, lo+32)
            Y = (sf @ W[lo:hi].T*theta_s).reshape(T, P, hi-lo)
            mu = Y.mean((0, 1))
            var = ((Y-mu)**2).mean((0, 1))
            tau = mu*R+np.sqrt(var+1e-5)/gamma[lo:hi]*(theta+center-bias-beta[lo:hi]*R)
            tau_g = mu_g[lo:hi]*R+np.sqrt(var_g[lo:hi]+1e-5)/gamma[lo:hi]*(theta+center-bias-beta[lo:hi]*R)
            U = np.einsum('ts,sph->tph', A, Y, optimize=True)
            direction = np.sign(gamma[lo:hi])
            direct = U*direction >= (tau*direction)[:, None, :]
            moment = U*direction >= (tau_g*direction)[:, None, :]
            archive = archived.reshape(T, P, H)[:, :, lo:hi]
            errors['max_abs_mean'] = max(errors['max_abs_mean'], float(np.abs(mu-mu_g[lo:hi]).max()))
            errors['max_abs_variance'] = max(errors['max_abs_variance'], float(np.abs(var-var_g[lo:hi]).max()))
            errors['max_rel_variance'] = max(errors['max_rel_variance'], float((np.abs(var-var_g[lo:hi])/var).max()))
            errors['max_abs_threshold'] = max(errors['max_abs_threshold'], float(np.abs(tau-tau_g).max()))
            errors['moment_vs_direct_bit_mismatches'] += int(np.count_nonzero(moment != direct))
            errors['moment_vs_archived_bit_mismatches'] += int(np.count_nonzero(moment != archive))
            errors['direct_vs_archived_bit_mismatches'] += int(np.count_nonzero(direct != archive))
        full_f = int(k.sum())*H
        cells = C*(C+1)//2
        count_bits = int(N).bit_length()
        # Coefficients W_c*W_d are NOT free: either store a large fixed product
        # table or generate them. A dense WG contraction avoids that table.
        operations = dict(integer_diagonal_increments=diag_updates,
                          integer_offdiagonal_increments=offdiag_updates,
                          symmetric_coefficient_count=H*cells,
                          symmetric_nonzero_count_MAC_terms=H*upper_nz,
                          online_symmetric_coefficient_multiplications=H*upper_nz,
                          dense_WG_and_dot_MAC_terms=H*C*C+H*C,
                          mean_MAC_terms=H*C,
                          full_FC1_binary_ADD_terms=full_f,
                          full_PSN_MAC_terms=P*H*int((A != 0).sum()))
        sensitivity = []
        for mac_cost in (1, 4, 8, 16):
            # Count+quadratic pass is additional to the single main FC1 pass.
            # Both simple models charge counter increments as one ADD, and
            # omit memory/index/control/sqrt. They are not energy or cycles.
            for mode, terms in [('precomputed_products', H*upper_nz+H*C),
                                ('online_symmetric_products', 2*H*upper_nz+H*C),
                                ('dense_WG', H*C*C+2*H*C)]:
                proxy = diag_updates+offdiag_updates+mac_cost*terms
                sensitivity.append(dict(mode=mode, MAC_to_ADD_cost=mac_cost,
                                        extra_arithmetic_cost_over_F=float(proxy/full_f)))
        record = dict(stage=stage, N=N, C=C, H=H, checked_gate_bits=N*H,
                      theta_source=theta_s, theta_output=theta,
                      active_per_source_row_mean=float(k.mean()),
                      active_per_source_row_max=int(k.max()),
                      pair_counters=cells, nonzero_pair_counters=upper_nz,
                      pair_counter_bits_for_whole_domain=count_bits,
                      pair_counter_packed_bytes=(cells*count_bits+7)//8,
                      pair_counter_uint32_bytes=cells*4,
                      source_bit_bytes=(N*C+7)//8,
                      one_hidden_channel_Y_proxy_bytes=N*8,
                      coefficient_product_table_FP64_bytes=H*cells*8,
                      coefficient_product_table_FP32_nominal_bytes=H*cells*4,
                      minimum_computed_variance=float(var_g.min()),
                      errors=errors, operations=operations,
                      arithmetic_sensitivity=sensitivity)
        np.save(out/f'stage{stage}_cooccurrence_uint32.npy', G.astype(np.uint32), allow_pickle=False)
        results.append(record)
        print(json.dumps(record), flush=True)
    result = dict(scope='sample0; stage0/block0 and stage3/block0; float64 proxy',
                  mechanism='Compute FC1 BN moments from binary input counts and exact co-occurrence counts before FC1',
                  limitations=['Real-arithmetic identity; no original FP32 reduction-order proof or full-network AEE.',
                               'Main FC1/PSN work is retained, and source must be revisited after the statistics pass.',
                               'Counting service, counter banks, weight-product production and SRAM traffic are not modeled.',
                               'Channel-tiled wide-Y plus packed-G materialization is a mandatory comparator.',
                               'Online/precomputed/dense cost ratios are hypothetical arithmetic sensitivity, not PPA.'],
                  RTL_speedup=False, PPA=False, layers=results)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
