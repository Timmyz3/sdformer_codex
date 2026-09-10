"""Predeclared offline opportunity check. Never a frozen-FP or PPA result."""
from pathlib import Path
import hashlib
import json
import struct
import zlib
import numpy as np
from checkpoint_numpy import read_checkpoint

BASE = Path(__file__).resolve().parents[1]
HW = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07')
CAP = HW/'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901'
CKPT = HW/'system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth'
HEADER = struct.Struct('<8sHH11I')
EXPECTED = {'layers.json': 'bd40c213f075ea3198f7145d25e9c96988701f46d5572c1e40d36e008feab08a',
            'fc_frames.bin': 'dceb6c0c80b9c5898d10b4ad813fbcd7683fa80191b54b78eadaadda04a818b1',
            'sample_order.json': 'd4f1f6e140b531b972d53b48aa64e5f0aa5497b79d460616a0b3f89139a4f773'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sources():
    for name, sha in EXPECTED.items():
        assert digest(CAP/name) == sha
    specs = {r['layer_id']: r for r in json.loads((CAP/'layers.json').read_text())['layers']}
    selected = {k: v for k, v in specs.items()
                if any(f'layers.{stage}.swin_blocks.0.mlp.fc' in v['module_name'] for stage in [0, 3])}
    assert len(selected) == 4
    arrays = {k: [] for k in selected}
    counts = {k: 0 for k in selected}
    frames = {k: 0 for k in selected}
    with (CAP/'fc_frames.bin').open('rb') as f:
        while True:
            raw_header = f.read(HEADER.size)
            assert len(raw_header) == HEADER.size
            magic, version, hs, lid, sid, fi, start, n, C, br, nnz, rb, cb, crc = HEADER.unpack(raw_header)
            assert magic == b'M1558F01' and version == 1 and hs == HEADER.size
            if sid > 0:
                break
            assert sid == 0 and lid in specs
            if lid not in selected:
                f.seek(cb, 1)
                continue
            assert fi == frames[lid] and start == counts[lid]
            assert C == selected[lid]['input_channels'] and br == (C + 7)//8
            dec = zlib.decompressobj()
            raw = dec.decompress(f.read(cb)) + dec.flush()
            assert dec.eof and not dec.unused_data and not dec.unconsumed_tail
            assert len(raw) == rb and zlib.crc32(raw) & 0xffffffff == crc
            mb = n * br
            assert rb == 3*mb + 2*n + nnz and not any(raw[mb:3*mb])
            bits = np.unpackbits(np.frombuffer(raw[:mb], dtype=np.uint8).reshape(n, br),
                                 axis=1, bitorder='little')
            assert not bits[:, C:].any()
            bits = bits[:, :C]
            row_counts = np.frombuffer(raw[3*mb:3*mb+2*n], dtype='<u2')
            assert np.array_equal(row_counts, bits.sum(1)) and int(row_counts.sum()) == nnz
            assert np.all(np.frombuffer(raw[3*mb+2*n:], dtype=np.int8) == 1)
            arrays[lid].append(bits)
            counts[lid] += n
            frames[lid] += 1
    result = {}
    for lid, spec in selected.items():
        value = np.concatenate(arrays[lid], axis=0)
        assert value.shape == (spec['tokens_per_call'], spec['input_channels'])
        result[spec['module_name']] = (spec, value)
    return result


def main():
    output = BASE/'records/threshold_packet_sample0_screen.json'
    assert not output.exists(), 'Preserve the existing research result; no silent rerun.'
    plan_path = BASE/'records/threshold_packet_screen_plan.json'
    plan = json.loads(plan_path.read_text())
    assert digest(CKPT) == '4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48'
    state = read_checkpoint(CKPT)['model_state_dict']
    source = sources()
    all_rows = []
    for stage in [0, 3]:
        pre = f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp.'
        spec, S = source[pre+'fc1']
        _, archived_bits = source[pre+'fc2']
        N, C, H, T = len(S), S.shape[1], spec['output_channels'], 10
        P = N//T
        assert spec['input_shape'][0] == T
        W = state[pre+'fc1.weight'].astype(np.float64)
        theta_source = float(state[pre+'sn1.spiking_neuron.thresh'])
        gamma = state[pre+'bn1.norm_layer.weight'].astype(np.float64)
        beta = state[pre+'bn1.norm_layer.bias'].astype(np.float64)
        A = state[pre+'sn2.spiking_neuron.weight'].astype(np.float64)
        bias = state[pre+'sn2.spiking_neuron.bias'].astype(np.float64).reshape(T, 1)
        center = state[pre+'sn2.spiking_neuron.center'].astype(np.float64).reshape(T, 1)
        theta = float(state[pre+'sn2.spiking_neuron.thresh'])
        assert A.shape == (T, T) and W.shape == (H, C)
        assert np.all(gamma != 0), 'Zero gamma needs separate constant-output accounting.'
        R = A.sum(1).reshape(T, 1)
        scalar_terms = int(S.sum(dtype=np.int64)) * H
        tile_data = {}
        grid = {}
        for B in plan['spatial_packet_sizes']:
            nb = (P+B-1)//B
            sizes = np.minimum(B, P-np.arange(nb)*B)
            per_position = S.reshape(T, P, C).sum((0, 2), dtype=np.int64)
            tile_terms = np.add.reduceat(per_position, np.arange(nb)*B)
            tile_data[B] = nb, sizes, tile_terms
            for K in plan['retained_boundary_values']:
                grid[B, K] = {'failed_h_tiles': np.zeros((nb, H), dtype=bool),
                              'failed_packets': 0, 'prediction_bit_changes': 0,
                              'certified_patch_bits': 0, 'extrema_crosschecks': 0}
        mismatch_archive = mismatch_algebra = 0
        reference_on = 0
        for lo in range(0, H, 32):
            hi = min(H, lo+32)
            hc = hi-lo
            Y = (S.astype(np.float64) @ W[lo:hi].T * theta_source).reshape(T, P, hc)
            mu = Y.mean((0, 1))
            var = ((Y-mu)**2).mean((0, 1))
            d = np.sqrt(var+1e-5)
            U = np.einsum('ts,sph->tph', A, Y, optimize=True)
            tau = mu*R + d/gamma[lo:hi]*(theta+center-bias-beta[lo:hi]*R)
            direction = np.sign(gamma[lo:hi])
            V, final_tau = U*direction, tau*direction
            actual = V >= final_tau[:, None, :]
            normalized = gamma[lo:hi]*(Y-mu)/d+beta[lo:hi]
            original = np.einsum('ts,sph->tph', A, normalized, optimize=True)+bias[:, None, :]-center[:, None, :]
            direct = original >= theta
            mismatch_algebra += int(np.count_nonzero(actual != direct))
            mismatch_archive += int(np.count_nonzero(actual != archived_bits.reshape(T, P, H)[:, :, lo:hi]))
            reference_on += int(actual.sum())
            # Prefix moment producer consumes completed T for every spatial packet.
            y_sum = Y.sum(0)
            y_sumsq = (Y*Y).sum(0)
            cs, cq = np.cumsum(y_sum, 0), np.cumsum(y_sumsq, 0)
            for B in plan['spatial_packet_sizes']:
                nb, sizes, _ = tile_data[B]
                end = np.minimum((np.arange(nb)+1)*B, P)-1
                denom = ((end+1)*T)[:, None]
                pmu = cs[end]/denom
                pvar = np.maximum(cq[end]/denom-pmu*pmu, 0)
                ptau = (pmu[None, :, :]*R[:, None, :] + np.sqrt(pvar+1e-5)[None, :, :]/gamma[lo:hi]
                        * (theta+center[:, None, :]-bias[:, None, :]-beta[lo:hi]*R[:, None, :]))*direction
                assert ptau.shape == (T, nb, hc)
                padded = np.full((T, nb*B, hc), np.nan)
                padded[:, :P] = V
                packed = padded.reshape(T, nb, B, hc)
                valid = np.arange(B)[None, None, :, None] < sizes[None, :, None, None]
                predicted = packed >= ptau[:, :, None, :]
                truth = packed >= final_tau[:, None, None, :]
                changed = (predicted != truth) & valid
                distance = np.where(valid, np.abs(packed-ptau[:, :, None, :]), np.inf)
                order = np.argsort(distance, axis=2, kind='stable')
                by_rank = np.take_along_axis(changed, order, axis=2)
                for K in plan['retained_boundary_values']:
                    stats = grid[B, K]
                    fail = by_rank[:, :, K:, :].any(2)
                    stats['failed_h_tiles'][:, lo:hi] = fail.any(0)
                    stats['failed_packets'] += int(fail.sum())
                    stats['prediction_bit_changes'] += int(changed.sum())
                    stats['certified_patch_bits'] += int((by_rank[:, :, :K, :].sum(2)*(~fail)).sum())
                    # Independent explicit a/b and patch check on fixed first/last packets.
                    for tb in sorted({0, nb-1}):
                        nvalid = int(sizes[tb])
                        for t in [0, T-1]:
                            for h in [0, hc-1]:
                                values = packed[t, tb, :nvalid, h]
                                keep = set(order[t, tb, :min(K, nvalid), h].tolist())
                                off = [v for i, v in enumerate(values) if i not in keep and v < ptau[t, tb, h]]
                                on = [v for i, v in enumerate(values) if i not in keep and v >= ptau[t, tb, h]]
                                cert = ((not off or max(off) < final_tau[t, h]) and
                                        (not on or final_tau[t, h] <= min(on)))
                                assert cert == (not fail[t, tb, h])
                                if cert:
                                    decoded = (values >= ptau[t, tb, h]).copy()
                                    for i in keep:
                                        decoded[i] = values[i] >= final_tau[t, h]
                                    assert np.array_equal(decoded, values >= final_tau[t, h])
                                stats['extrema_crosschecks'] += 1
            if hi == H or hi % 512 == 0:
                print('stage', stage, 'hidden', hi, '/', H, flush=True)
        assert mismatch_algebra == 0, 'Algebra/reference mismatch; do not admit this screen.'
        rows = []
        for (B, K), stats in grid.items():
            nb, sizes, tile_terms = tile_data[B]
            failure = stats.pop('failed_h_tiles')
            assert H % 96 == 0
            group_failure = failure.reshape(nb, H//96, 96).any(2)
            fine_terms = int(np.dot(failure.sum(1, dtype=np.int64), tile_terms))
            group_terms = int(np.dot(group_failure.sum(1, dtype=np.int64)*96, tile_terms))
            fine_psn = int(np.dot(failure.sum(1, dtype=np.int64), sizes))*T*T
            group_psn = int(np.dot(group_failure.sum(1, dtype=np.int64)*96, sizes))*T*T
            packets = T*nb*H
            bare = {}
            for width in [32, 64]:
                per_packet = B+2*width+K*(width+(B-1).bit_length())
                bare[str(width)] = packets*per_packet/8
            rows.append({'B': B, 'K': K, **stats, 'packets': packets,
                         'packet_failure_fraction': stats['failed_packets']/packets,
                         'any_T_failure_h_tile_fraction': float(failure.mean()),
                         'fixed96_any_T_h_group_failure_fraction': float(group_failure.mean()),
                         'full_FC1_active_scalar_terms': scalar_terms,
                         'replay_FC1_terms_by_h': fine_terms,
                         'replay_FC1_terms_fixed96': group_terms,
                         'replay_FC1_fraction_by_h': fine_terms/scalar_terms,
                         'replay_FC1_fraction_fixed96': group_terms/scalar_terms,
                         'full_PSN_scalar_terms': T*T*P*H,
                         'replay_PSN_terms_by_h': fine_psn,
                         'replay_PSN_terms_fixed96': group_psn,
                         'bare_payload_bytes_by_assumed_U_width': bare})
        for B in plan['spatial_packet_sizes']:
            series = [r for r in rows if r['B'] == B]
            assert all(series[i]['failed_packets'] >= series[i+1]['failed_packets'] for i in range(len(series)-1))
        layer = {'module': pre+'fc1', 'N_includes_T': N, 'P': P, 'C': C, 'H': H, 'T': T,
                 'fullrank_A_rank': int(np.linalg.matrix_rank(A)), 'source_theta': theta_source,
                 'gamma_negative': int((gamma < 0).sum()), 'BN_eps_assumption': 1e-5,
                 'reference_binary_count': N*H, 'reference_ones': reference_on,
                 'float64_algebra_vs_original_order_mismatches': mismatch_algebra,
                 'float64_proxy_vs_archived_FC2_source_mismatches': mismatch_archive,
                 'float64_proxy_vs_archived_FC2_source_mismatch_fraction': mismatch_archive/(N*H),
                 'rows': rows}
        all_rows.append(layer)
        print('LAYER_DONE',stage,'archive_mismatch',mismatch_archive,'/',N*H,flush=True)
        for row in rows:
            print('B,K',row['B'],row['K'],'packet_fail',round(row['packet_failure_fraction'],5),
                  'FC1_replay_h',round(row['replay_FC1_fraction_by_h'],5),
                  'FC1_replay_fixed96',round(row['replay_FC1_fraction_fixed96'],5),flush=True)
    result = {'status': 'COMPLETED_FLOAT64_OPPORTUNITY_PROXY_ONLY', 'plan': plan,
              'plan_sha256': digest(plan_path), 'checkpoint_sha256': digest(CKPT),
              'source_hashes': EXPECTED, 'layers': all_rows,
              'script_sha256': digest(Path(__file__)),
              'reader_sha256': digest(Path(__file__).with_name('checkpoint_numpy.py')),
              'checks': 'Hashes; selected frame CRC/order/shape/code; independent BN then PSN float64; explicit certificate/patch boundary checks; monotone K failures.',
              'limits': ['Only sample0, two predeclared layers; no S40/valid825 generalization.',
                         'No measured GPU FP32 equality, no fixed-point contract, no new AEE.',
                         'Prefix stats arithmetic, sorting, packet ports, source gathers and replay stalls are not timed.',
                         'Float64 CPU materializes values for audit; proposed circuit must not retain all those values.',
                         'Bare capacities omit descriptors, masks, patch queues, source/scratch, alignment and SRAM macros.',
                         'Full first FC1 and full first PSN are always charged; fractions describe added recomputation only.'],
              'PPA_ADMISSION': 0, 'RTL_SPEEDUP_ADMISSION': 0, 'FROZEN_FP_EQUIVALENCE': 0}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
