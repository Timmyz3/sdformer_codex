"""Real-parameter, complete-BN-domain traces for the finite service model.

Reads existing captures; does not invoke the network, train, hash, or modify inputs.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '4'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import struct
import time
import zlib
import numpy as np

ROOT = Path(__file__).resolve().parent
GH = ROOT.parents[1] / 'mechanism_rebuild_gh_20260906'
HW = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07')
CAP = HW/'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901'
sys.path.insert(0, str(GH/'scripts'))
from checkpoint_numpy import read_checkpoint
HEADER = struct.Struct('<8sHH11I')


def load_sources(samples):
    specs = {r['layer_id']: r for r in json.loads((CAP/'layers.json').read_text())['layers']}
    wanted = {k: v for k, v in specs.items() if any(
        f'layers.{s}.swin_blocks.0.mlp.fc' in v['module_name'] for s in (0, 3))}
    chunks = {(sid, lid): [] for sid in samples for lid in wanted}
    with (CAP/'fc_frames.bin').open('rb') as f:
        while raw := f.read(HEADER.size):
            magic, version, hs, lid, sid, fi, start, n, C, br, nnz, rb, cb, _ = HEADER.unpack(raw)
            if sid > max(samples):
                break
            if (sid, lid) not in chunks:
                f.seek(cb, 1)
                continue
            raw = zlib.decompress(f.read(cb))
            bits = np.unpackbits(np.frombuffer(raw[:n*br], np.uint8).reshape(n, br),
                                 axis=1, bitorder='little')[:, :C]
            if start != sum(len(x) for x in chunks[sid, lid]):
                raise ValueError('Noncontiguous selected source frame')
            chunks[sid, lid].append(bits)
    return {sid: {v['module_name']: (v, np.concatenate(chunks[sid, lid]))
                  for lid, v in wanted.items()} for sid in samples}


def layer_trace(sid, stage, state, source):
    pre = f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp.'
    spec, S = source[pre+'fc1']
    _, archived = source[pre+'fc2']
    N, C = S.shape
    T, B, K = 10, 32, 4
    H, P = spec['output_channels'], N//T
    nb = (P+B-1)//B
    sizes = np.minimum(B, P-np.arange(nb)*B)
    W = state[pre+'fc1.weight'].astype(np.float64)
    theta_source = np.asarray(state[pre+'sn1.spiking_neuron.thresh'], dtype=np.float64)
    theta = float(state[pre+'sn2.spiking_neuron.thresh'])
    if theta_source.size != 1:
        raise ValueError('This trace evaluator needs explicit per-channel theta extension')
    theta_source = float(theta_source)
    gamma = state[pre+'bn1.norm_layer.weight'].astype(np.float64)
    beta = state[pre+'bn1.norm_layer.bias'].astype(np.float64)
    A = state[pre+'sn2.spiking_neuron.weight'].astype(np.float64)
    bias = state[pre+'sn2.spiking_neuron.bias'].astype(np.float64).reshape(T, 1)
    center = state[pre+'sn2.spiking_neuron.center'].astype(np.float64).reshape(T, 1)
    if np.any(gamma == 0):
        raise ValueError('Need explicit zero-gamma constant consumer')
    R = A.sum(1).reshape(T, 1)
    S64 = S.astype(np.float64)
    # Binary products and sums <=192000 are exact representable integers here.
    G = (S64.T @ S64).astype(np.uint32)
    k = S.sum(0, dtype=np.uint32)
    if not np.array_equal(np.diag(G), k):
        raise ValueError('Gram diagonal differs from source counts')
    fail = np.zeros((nb, H), bool)
    nc = np.stack([S.reshape(T, P, C)[:, b*B:b*B+n].sum((0, 1), dtype=np.uint16)
                   for b, n in enumerate(sizes)])
    metrics = dict(reference_vs_archive_mismatches=0, shifted_vs_ordered_mismatches=0,
                   moment_vs_centered_bits=0, gram_vs_centered_bits=0,
                   changed_prediction_bits=0, failed_time_packets=0,
                   min_reference_margin=float('inf'), gram_mu_max_abs_error=0.0,
                   gram_var_max_abs_error=0.0)
    for lo in range(0, H, 32):
        hi = min(H, lo+32)
        wh = W[lo:hi]
        Y = (S64 @ wh.T * theta_source).reshape(T, P, hi-lo)
        mu = Y.mean((0, 1))
        var = ((Y-mu)**2).mean((0, 1))
        mmu = Y.sum((0, 1))/N
        mvar = np.maximum((Y*Y).sum((0, 1))/N-mmu*mmu, 0)
        gmu = (wh @ k.astype(np.float64))*theta_source/N
        gvar = np.maximum(np.sum((wh @ G.astype(np.float64))*wh, 1)*theta_source**2/N-gmu*gmu, 0)
        U = np.einsum('ts,sph->tph', A, Y, optimize=True)
        direction = np.sign(gamma[lo:hi])
        def tau(m, v):
            return (m*R+np.sqrt(v+1e-5)/gamma[lo:hi]*(theta+center-bias-beta[lo:hi]*R))*direction
        V = U*direction
        final_tau = tau(mu, var)
        truth = V >= final_tau[:, None, :]
        normalized = gamma[lo:hi]*(Y-mu)/np.sqrt(var+1e-5)+beta[lo:hi]
        ordered = np.einsum('ts,sph->tph', A, normalized, optimize=True)+bias[:, None, :]-center[:, None, :]
        metrics['reference_vs_archive_mismatches'] += int(np.count_nonzero(truth != archived.reshape(T, P, H)[:, :, lo:hi]))
        metrics['shifted_vs_ordered_mismatches'] += int(np.count_nonzero(truth != (ordered >= theta)))
        metrics['moment_vs_centered_bits'] += int(np.count_nonzero(truth != (V >= tau(mmu, mvar)[:, None, :])))
        metrics['gram_vs_centered_bits'] += int(np.count_nonzero(truth != (V >= tau(gmu, gvar)[:, None, :])))
        metrics['min_reference_margin'] = min(metrics['min_reference_margin'], float(np.min(np.abs(V-final_tau[:, None, :]))))
        metrics['gram_mu_max_abs_error'] = max(metrics['gram_mu_max_abs_error'], float(np.max(np.abs(mu-gmu))))
        metrics['gram_var_max_abs_error'] = max(metrics['gram_var_max_abs_error'], float(np.max(np.abs(var-gvar))))
        cs, cq = np.cumsum(Y.sum(0), 0), np.cumsum((Y*Y).sum(0), 0)
        ends = np.minimum((np.arange(nb)+1)*B, P)-1
        denom = ((ends+1)*T)[:, None]
        pmu = cs[ends]/denom
        pvar = np.maximum(cq[ends]/denom-pmu*pmu, 0)
        ptau = (pmu[None]*R[:, None]+np.sqrt(pvar+1e-5)[None]/gamma[lo:hi]
                *(theta+center[:, None]-bias[:, None]-beta[lo:hi]*R[:, None]))*direction
        padded = np.full((T, nb*B, hi-lo), np.nan)
        padded[:, :P] = V
        packed = padded.reshape(T, nb, B, hi-lo)
        valid = np.arange(B)[None, None, :, None] < sizes[None, :, None, None]
        changed = ((packed >= ptau[:, :, None]) != (packed >= final_tau[:, None, None])) & valid
        order = np.argsort(np.where(valid, np.abs(packed-ptau[:, :, None]), np.inf), axis=2, kind='stable')
        by_rank = np.take_along_axis(changed, order, axis=2)
        failed_t = by_rank[:, :, K:].any(2)
        fail[:, lo:hi] = failed_t.any(0)
        metrics['failed_time_packets'] += int(failed_t.sum())
        metrics['changed_prediction_bits'] += int(changed.sum())
        if hi == H or hi % 512 == 0:
            print('NUMERIC', sid, stage, hi, H, flush=True)
    if sid == 0:
        old = ROOT.parents[1]/'handoff_reconciliation_20260906/records/repair_sector_sample0_r1'
        prior = np.unpackbits(np.fromfile(old/f'stage{stage}_B32K4_anyT_h.le.bitpack', np.uint8).reshape(nb, H//8), axis=1, bitorder='little')
        metrics['existing_sample0_failure_mask_mismatches'] = int(np.count_nonzero(prior != fail))
    np.savez_compressed(ROOT/f'trace_s{sid}_stage{stage}.npz',
                        source_packed=np.packbits(S, axis=1, bitorder='little'),
                        failure=fail, source_counts=nc, sizes=sizes,
                        W=W.astype(np.float32), A=A, G=G, k=k, gamma=gamma,
                        beta=beta, bias=bias, center=center)
    stats = dict(sample=sid, stage=stage, module=pre+'fc1', N=N, P=P, C=C, H=H, T=T,
                 B=B, K=K, source_theta=theta_source, output_theta=theta,
                 A_rank=int(np.linalg.matrix_rank(A)), A_nonzero=int(np.count_nonzero(A)),
                 A_negative=int((A<0).sum()), source_active=int(S.sum(dtype=np.int64)),
                 nonempty_source_rows=int(np.any(S, axis=1).sum()),
                 failed_h_tiles=int(fail.sum()), gram_nonzero_fraction=float(np.mean(G != 0)), **metrics)
    (ROOT/f'numeric_s{sid}_stage{stage}.json').write_text(json.dumps(stats, indent=2)+'\n')
    print('DONE', json.dumps(stats), flush=True)
    return stats


def main():
    started = time.time()
    plan = json.loads((ROOT/'resources.json').read_text())
    state = read_checkpoint(HW/'system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth')['model_state_dict']
    sources = load_sources(plan['samples'])
    rows = []
    for sid in plan['samples']:
        for stage in plan['stages']:
            rows.append(layer_trace(sid, stage, state, sources[sid]))
    (ROOT/'numeric_result.json').write_text(json.dumps(dict(
        scope='Float64 evaluation of real checkpoint parameters and complete captured source domains; observed support agreement does not prove all-input FP32 equality',
        layers=rows, wall_seconds=time.time()-started), indent=2)+'\n')


if __name__ == '__main__':
    main()
