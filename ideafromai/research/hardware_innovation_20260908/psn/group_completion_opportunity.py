"""Full-C all-zero gate prevalence, not an early-exit predictor or cycle model.

The training cache contains intact P4 spatial groups and all C384 source codes.
NumPy float64 accumulates bounded integers exactly; a separate int64 class-sum
reference checks fixed complete-C samples. No model forward or fitting occurs.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
from collections import OrderedDict
import io
import json
from pathlib import Path
import pickle
import sys
import time
import zipfile
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def read_torch(path):
    """Local tensor/dict reader, including uint8 source codes; no Torch needed."""
    with zipfile.ZipFile(path) as z:
        prefix = next(n[:-8] for n in z.namelist() if n.endswith('data.pkl'))
        types = {k+'Storage': np.dtype(v) for k, v in
                 [('Byte', 'u1'), ('Char', 'i1'), ('Short', '<i2'), ('Int', '<i4'),
                  ('Long', '<i8'), ('Double', '<f8'), ('Float', '<f4'), ('Bool', '?')]}
        def rebuild(storage, offset, size, stride, *unused):
            dt, raw = storage
            return np.ndarray(size, dtype=dt, buffer=raw, offset=offset*dt.itemsize,
                              strides=tuple(s*dt.itemsize for s in stride))
        class Reader(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch' and name in types:
                    return types[name]
                if module == 'torch._utils' and name in ('_rebuild_tensor', '_rebuild_tensor_v2'):
                    return rebuild
                if (module, name) == ('collections', 'OrderedDict'):
                    return OrderedDict
                raise ValueError((module, name))
            def persistent_load(self, pid):
                _, dt, key, _, _ = pid
                return dt, z.read(prefix+'data/'+key)
        return Reader(io.BytesIO(z.read(prefix+'data.pkl'))).load()


def csd_terms(value):
    value, terms = abs(int(value)), 0
    while value:
        if value & 1:
            value -= 2-(value & 3)
            terms += 1
        value //= 2
    return terms


def ratio(count, denominator):
    return {'count': int(count), 'denominator': int(denominator),
            'fraction': float(count/denominator)}


def main():
    started = time.monotonic()
    alg = ROOT/'algorithm'
    cache = read_torch(alg/'nrv_cost_probe/s2b3_train.pt')
    q = read_torch(alg/'pruning_probe/original_initial.pt')
    prefix = 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.3.mlp.'
    original = read_torch(alg/'stage2_temporal_codes/integer_parameters.pt')[prefix]
    codes = cache['start_codes']
    W, B, E, tau = (np.asarray(q[k], dtype=np.int64) for k in
                   ('weight_int8', 'B_int8', 'E_int8', 'threshold_int32'))
    F, K, P, C = codes.shape
    H, T = W.shape[0], B.shape[0]
    positions = cache['position_indices'].astype(np.int64)
    assert (F, K, P, C, H, T) == (32, 32, 4, 384, 1536, 10)
    assert np.array_equal(W, original['weight_int8'])
    assert np.array_equal(positions, positions[:, :1]+np.arange(P))
    assert np.all(positions[:, 0] % P == 0)
    assert np.array_equal(q['hidden_indices'], np.arange(H))
    assert np.all(q['hidden_keep']) and np.all(q['weight_mask'])
    assert int(codes.min()) >= 0 and int(codes.max()) < 8
    B8 = B @ E.T
    # Bound every multiply and any partial sum, including signed cancellation.
    bound = int(np.abs(W).sum(1).max())*int(np.abs(E).max())*int(np.abs(B).sum(1).max())
    bound += int(np.abs(tau).max())
    assert bound < 2**53
    Wf, Bf, Ef = W.astype(np.float64), B.astype(np.float64), E.astype(np.float64)
    names = ('zero_bits', 'zero_ph_T10', 'zero_p_H8_T10', 'zero_P4_h_T10',
             'zero_P4_H8_T10', 'empty_source_P4')
    totals = {k: 0 for k in names}
    histogram = np.zeros(321, dtype=np.int64)
    ph_histogram = np.zeros(11, dtype=np.int64)
    any_h = np.zeros(H, dtype=bool)
    any_h8 = np.zeros(H//8, dtype=bool)
    per_frame = []
    margin_min, margin_max = None, None
    checked_values = 0
    for f in range(F):
        source = codes[f].reshape(K*P, C)
        # Complete source reduction, followed by the saved complete T10 consumer.
        S = np.transpose(Ef[source], (0, 2, 1)) @ Wf.T
        margin = np.transpose(Bf @ S, (0, 2, 1))-tau.T[None, :, :]
        assert np.all(margin == np.rint(margin))
        if f in (0, F-1):
            selected = np.array([0, 1, K*P//2, K*P-1])
            class_sums = np.stack([(source[selected] == j).astype(np.int64) @ W.T
                                   for j in range(8)], axis=-1)
            reference = class_sums @ B8.T-tau.T[None, :, :]
            assert np.array_equal(margin[selected], reference)
            checked_values += reference.size
        gate = (margin >= 0).reshape(K, P, H, T)
        grouped = gate.reshape(K, P, H//8, 8, T)
        active = grouped.sum(axis=(1, 3, 4))
        record = {
            'frame': cache['frames'][f],
            'zero_bits': int((~gate).sum()),
            'zero_ph_T10': int((~gate.any(3)).sum()),
            'zero_p_H8_T10': int((~grouped.any(axis=(3, 4))).sum()),
            'zero_P4_h_T10': int((~gate.any(axis=(1, 3))).sum()),
            'zero_P4_H8_T10': int((active == 0).sum()),
            'empty_source_P4': int((codes[f] == 0).all(axis=(1, 2)).sum())}
        for key in names:
            totals[key] += record[key]
        histogram += np.bincount(active.ravel(), minlength=321)
        ph_histogram += np.bincount(gate.sum(3).ravel(), minlength=11)
        any_h |= gate.any(axis=(0, 1, 3))
        any_h8 |= grouped.any(axis=(0, 1, 3, 4))
        margin_min = min(margin_min if margin_min is not None else np.inf, int(margin.min()))
        margin_max = max(margin_max if margin_max is not None else -np.inf, int(margin.max()))
        per_frame.append(record)
    denoms = dict(zero_bits=F*K*P*H*T, zero_ph_T10=F*K*P*H,
                  zero_p_H8_T10=F*K*P*(H//8), zero_P4_h_T10=F*K*H,
                  zero_P4_H8_T10=F*K*(H//8), empty_source_P4=F*K)
    result = {
        'scope': 's2b3 integer direct-code student, original unpruned saved W/B/tau; existing 32 training frames and 32 intact P4 blocks per frame; full C384 and H1536; not valid825 prevalence',
        'formula': 'S[r,p,h]=sum_c E[code[p,c],r]*W[h,c]; gate[p,t,h]=(sum_r B[t,r]*S[r,p,h]>=tau[t,h]); z=theta_output*gate',
        'source_file': 'algorithm/nrv_cost_probe/s2b3_train.pt: start_codes (not target)',
        'parameter_file': 'algorithm/pruning_probe/original_initial.pt',
        'weight_identity': 'weight_int8 exactly matches stage2_temporal_codes/integer_parameters.pt s2b3; tau/B are the later trained integer consumer, not original ep34 parameters',
        'theta_source_folded_into_W': float(original['theta_source']),
        'theta_output_retained_separately': float(original['theta_output']),
        'shape': {'frames': F, 'spatial_blocks_per_frame': K, 'P': P, 'C': C, 'H': H, 'T': T},
        'position_indices': positions.tolist(),
        'physical_group': 'same k-ID/P4 source packet across eight output tiles h=8*f+tile; all 320 gates must be zero. This is not an assertion that one W read has no other k-ID consumers.',
        'rates': {k: ratio(totals[k], denoms[k]) for k in names},
        'active_gate_count_per_320_group': {str(i): int(n) for i, n in enumerate(histogram) if n},
        'active_gate_count_per_ph_T10': ph_histogram.tolist(),
        'hidden_rows_never_fire_in_this_sample': np.flatnonzero(~any_h).tolist(),
        'H8_groups_never_fire_in_this_sample': np.flatnonzero(~any_h8).tolist(),
        'numerics': {'full_margin_min': int(margin_min), 'full_margin_max': int(margin_max),
                     'integer_float64_abs_bound': bound, 'independent_int64_checked_values': checked_values,
                     'claim': 'bounded integer-valued float64 arithmetic with exact full-C int64 checks; not original ep34 FP32 equivalence'},
        'checkpoint_C192_cost_boundary': {
            'already_paid': 'First 192 input columns: source-bank reads/NR4 decoding, W requests, GP reduction and S writes; cannot be recovered by accepting the group.',
            'same_PE_checkpoint': 'Must read current S, execute actual consumer CSD/reconstruction with tau initialization and ten gate decisions per p/h, then combine all 320 decisions. Shared GP/PSN adders cannot execute both simultaneously.',
            'unshared_direct_CSD_terms_per_ph': {'B6': sum(csd_terms(x) for x in B.ravel()),
                                                 'onehot7_B': sum(csd_terms(x) for x in B8[:, 1:].ravel())},
            'term_count_limit': 'Raw coefficient terms only; an existing CSE/reconstruction route may be cheaper. These are not cycle predictions or a mandatory new checker architecture.',
            'rejection': 'Keep original S while checking; a failed group executes remaining C and the final consumer again. Reusing prefix U instead requires explicitly maintained wider T10 U state and continued updates; no free incremental reuse is assumed.',
            'state': 'Existing temporary/cache resources may be reused by serialization; a second full S copy is not inherently required. Original S must remain live until the check rejects or the entire group retires safely.',
            'recognition': 'Full-C zero truth is an oracle label. Partial gates at C192 do not prove final gates; a safe suffix certificate or a trained approximate acceptance rule, with its own costs and AEE, is still missing.',
            'sharing': 'One zero P4/H8 group cannot remove a W fetch still needed by another k-ID/PE. Cached source or DRAM data may still be needed by other F groups. No source/W/DMA savings are calculated here.',
            'required_controls': ['Ordinary empty-source and zero-weight skipping with identical compiled consumer optimizations.',
                                  'Static whole-MLP identity/zero-output deletion, evaluated as a different model with real shortcut and BN2 behavior.',
                                  'Static hidden25 width control, with its kept-H count explicitly specified and the same training budget and real FC2 consumers; it can remove complete GP and PSN rows whereas conditional exit retains prefix and checker costs.']},
        'interpretation': 'Measured all-zero output prevalence only. It does not identify those groups early, price recognition, prove a hardware skip, or imply whole-network speedup.',
        'per_frame_counts': per_frame,
        'elapsed_cpu_seconds': time.monotonic()-started}
    target = Path(__file__).with_suffix('.json')
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'rates': result['rates'], 'never_fire_hidden': len(result['hidden_rows_never_fire_in_this_sample']),
                      'never_fire_H8': len(result['H8_groups_never_fire_in_this_sample']),
                      'elapsed_cpu_seconds': result['elapsed_cpu_seconds'], 'output': str(target)}, indent=2))


if __name__ == '__main__':
    main()
