"""Small, predeclared screen: low-dimensional candidate + certified fallback."""
import os
for thread_key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[thread_key] = '4'
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import random
import sys
import numpy as np

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
PRIOR = BASE.parent / 'innovation_radix_temporal_20260906/source_moments_r1'
PARSER = BASE.parent / 'mechanism_rebuild_gh_20260906/scripts'
sys.path.insert(0, str(PARSER))
from screen_threshold_packets import sources, CKPT, EXPECTED
from checkpoint_numpy import read_checkpoint


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rational_check():
    rng = random.Random(1701)
    checked = confirmed = fallback = 0
    # Fixed signed full-rank 3x3 temporal operator; no LIF simplification.
    A = [[2, -1, 1], [1, 3, -2], [-1, 2, 2]]
    assert round(np.linalg.det(np.array(A))) == 25
    for _ in range(80):
        theta = [Fraction(999883 + 17*c, 1000000) for c in range(5)]
        X = [[theta[c] * rng.randrange(2) for c in range(5)] for t in range(3)]
        L = [[sum(A[t][s] * X[s][c] for s in range(3)) for c in range(5)] for t in range(3)]
        for h in range(5):
            W = [Fraction(rng.randrange(-16, 17), 8) for c in range(5)]
            if h == 0:
                W[2:] = [Fraction(0)] * 3
            threshold = Fraction(rng.randrange(-24, 25), 4)
            direction = rng.choice([-1, 1])
            for t in range(3):
                exact = direction * sum(L[t][c] * W[c] for c in range(5))
                candidate = direction * sum(L[t][c] * W[c] for c in range(2))
                bound2 = sum(L[t][c]**2 for c in range(2, 5)) * sum(W[c]**2 for c in range(2, 5))
                assert (exact - candidate)**2 <= bound2
                margin = candidate - threshold
                on = margin >= 0 and margin**2 >= bound2
                off = margin < 0 and margin**2 > bound2
                truth = exact >= threshold
                if on or off:
                    assert truth == on
                    confirmed += 1
                else:
                    fallback += 1
                checked += 1
    return {'cases': 80, 'gate_decisions': checked, 'confirmed': confirmed,
            'fallback': fallback, 'mismatches': 0,
            'scope': 'Exact rational coordinate-subspace theorem check, not frozen W or BN rounding.'}


def main():
    out = BASE / 'projection_certificate_r1.json'
    assert not out.exists(), 'Preserve the predeclared first screen.'
    plan = json.loads((BASE / 'projection_certificate_plan.json').read_text())
    proof = rational_check()
    assert digest(CKPT) == '4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48'
    gp = PRIOR / 'stage0_cooccurrence_uint32.npy'
    assert digest(gp) == plan['prior_G_sha256']
    assert digest(PRIOR / 'result.json') == plan['prior_result_sha256']
    state = read_checkpoint(CKPT)['model_state_dict']
    source = sources()
    pre = 'sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.'
    spec, S = source[pre + 'fc1']
    _, archived = source[pre + 'fc2']
    N, C = S.shape
    H, T, P = spec['output_channels'], 10, N // 10
    G = np.load(gp, allow_pickle=False)
    assert G.shape == (C, C) and G.dtype == np.uint32 and np.array_equal(G, G.T)
    assert np.array_equal(np.diag(G), S.sum(0, dtype=np.int64))
    pairs = [(i, (13*i+7) % C) for i in range(17)]
    for c, d in pairs:
        assert G[c, d] == np.count_nonzero(S[:, c] & S[:, d])
    W = state[pre + 'fc1.weight'].astype(np.float64)
    assert pre + 'fc1.bias' not in state, 'New bias contract required.'
    theta_s = float(state[pre + 'sn1.spiking_neuron.thresh'])
    gamma = state[pre + 'bn1.norm_layer.weight'].astype(np.float64)
    beta = state[pre + 'bn1.norm_layer.bias'].astype(np.float64)
    A = state[pre + 'sn2.spiking_neuron.weight'].astype(np.float64)
    bias = state[pre + 'sn2.spiking_neuron.bias'].astype(np.float64).reshape(T, 1)
    center = state[pre + 'sn2.spiking_neuron.center'].astype(np.float64).reshape(T, 1)
    theta_out = float(state[pre + 'sn2.spiking_neuron.thresh'])
    assert W.shape == (H, C) and A.shape == (T, T) and np.all(gamma != 0)
    assert np.linalg.matrix_rank(A) == T
    mu = theta_s * (W @ np.diag(G)) / N
    var = theta_s**2 * np.einsum('hc,hc->h', W @ G.astype(np.float64), W) / N - mu**2
    assert np.all(var > 0)
    R = A.sum(1).reshape(T, 1)
    direction = np.sign(gamma)
    tau = (mu*R + np.sqrt(var+1e-5)/gamma*(theta_out+center-bias-beta*R)) * direction
    positions = np.linspace(0, P-1, 64, dtype=np.int64)
    assert len(np.unique(positions)) == 64
    X = S.reshape(T, P, C)[:, positions].astype(np.float64) * theta_s
    Y = X @ W.T
    U = np.einsum('ts,sph->tph', A, Y, optimize=True)
    L = np.einsum('ts,spc->tpc', A, X, optimize=True)  # Oracle only; not free hardware.
    assert np.max(np.abs(U - L @ W.T)) < 1e-10
    truth = U * direction >= tau[:, None, :]
    archived_selected = archived.reshape(T, P, H)[:, positions]
    archive_mismatches = int(np.count_nonzero(truth != archived_selected))
    assert archive_mismatches == 0
    local_G = np.einsum('tpc,upc->ptu', X, X, optimize=True)
    norm_L2 = np.einsum('ti,pij,tj->tp', A, local_G, A, optimize=True)
    _, singular, VT = np.linalg.svd(W, full_matrices=False)
    nnz_position = S.reshape(T, P, C)[:, positions].sum((0, 2), dtype=np.int64)
    original_FC1 = int(nnz_position.sum()) * H
    original_PSN = len(positions) * T*T*H
    rows = []
    for rank in plan['ranks']:
        Q = VT[:rank].T
        B = W @ Q
        E = W - B @ Q.T
        Z = np.einsum('ts,spr->tpr', A, X @ Q, optimize=True)
        estimate = Z @ B.T
        gram_Q = Q.T @ Q
        norm_Z2 = (Z*Z).sum(2)
        # General projection expression, including the small floating Q defect.
        norm_perp2 = norm_L2 - 2*norm_Z2 + np.einsum('tpr,rs,tps->tp', Z, gram_Q, Z, optimize=True)
        explicit = L - Z @ Q.T
        norm_error = float(np.max(np.abs(norm_perp2 - (explicit*explicit).sum(2))))
        assert norm_error < 1e-9
        eps_guard = 1e-10 * (1 + np.abs(norm_L2))
        norm_perp = np.sqrt(np.maximum(0, norm_perp2) + eps_guard)
        defect = Q.T @ E.T
        bound = norm_perp[:, :, None] * np.linalg.norm(E, axis=1)
        bound += np.sqrt(norm_Z2)[:, :, None] * np.linalg.norm(defect, axis=0)
        numerical_guard = 1e-9 * (1 + np.abs(estimate) + np.abs(tau[:, None, :]))
        assert np.all(np.abs(U-estimate) <= bound + numerical_guard)
        normalized_estimate = estimate * direction
        on = normalized_estimate - bound >= tau[:, None, :] + numerical_guard
        off = normalized_estimate + bound < tau[:, None, :] - numerical_guard
        certified = on | off
        assert not np.any(on & off)
        assert np.array_equal(on[certified], truth[certified])
        fail_h = (~certified).any(0)
        fail_group = fail_h.reshape(len(positions), H//96, 96).any(2)
        fallback_FC1_h = int(np.dot(nnz_position, fail_h.sum(1, dtype=np.int64)))
        fallback_FC1_96 = int(np.dot(nnz_position, 96*fail_group.sum(1, dtype=np.int64)))
        row = {
            'rank': rank, 'weight_Frobenius_energy_fraction': float((singular[:rank]**2).sum()/(singular**2).sum()),
            'confirmed_gate_bits': int(certified.sum()), 'gate_bits': int(truth.size),
            'confirmed_gate_fraction': float(certified.mean()),
            'all_T_confirmed_position_h_fraction': float((~fail_h).mean()),
            'fixed96_all_T_group_failure_fraction': float(fail_group.mean()),
            'fallback_FC1_contributions_per_h': fallback_FC1_h,
            'fallback_FC1_contributions_fixed96': fallback_FC1_96,
            'fallback_FC1_fraction_per_h': fallback_FC1_h/original_FC1,
            'fallback_FC1_fraction_fixed96': fallback_FC1_96/original_FC1,
            'fallback_fullT_PSN_MAC_terms_per_h': int(fail_h.sum())*T*T,
            'reduced_source_projection_ADD_terms': int(nnz_position.sum())*rank,
            'reduced_PSN_MAC_terms': len(positions)*T*T*rank,
            'candidate_reconstruction_MAC_terms': len(positions)*T*H*rank,
            'local_symmetric_Gram_entries': len(positions)*T*(T+1)//2,
            'local_Gram_64bit_AND_popcount_words': len(positions)*T*(T+1)//2*((C+63)//64),
            'ideal_orthogonal_local_norm_contraction_MAC_terms': len(positions)*T*(T*(T+1)//2),
            'projection_norm_square_terms': len(positions)*T*rank,
            'certificate_norm_products': len(positions)*T*H,
            'general_float_Q_gram_correction_MAC_terms': len(positions)*T*rank*rank,
            'direct_vs_Gram_norm_max_abs_difference': norm_error,
            'certified_wrong_bits': 0,
            'note': 'Counts exclude complete source-Gram service shown separately and all mapped memory/control. '
                    'A generic quantized Q cannot silently assume orthogonality; charge the correction or use an exact structured basis.'
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in ['rank', 'confirmed_gate_fraction',
                         'all_T_confirmed_position_h_fraction', 'fallback_FC1_fraction_fixed96']}), flush=True)
    old_result = json.loads((PRIOR / 'result.json').read_text())['layers'][0]
    result = {
        'date': '2026-09-07', 'plan_sha256': digest(BASE / 'projection_certificate_plan.json'),
        'script_sha256': digest(Path(__file__)), 'checkpoint_sha256': digest(CKPT),
        'capture_sha256': EXPECTED, 'prior_source_Gram_sha256': digest(gp),
        'scope': 'sample0 stage0 block0, fixed 64 spatial positions, all T and all hidden channels; float64 proxy',
        'positions': positions.tolist(), 'shape_T_P_C_H': [T, len(positions), C, H],
        'theta_source': theta_s, 'theta_output': theta_out, 'PSN_rank': int(np.linalg.matrix_rank(A)),
        'archive_selected_mismatches': archive_mismatches,
        'full_FC1_contribution_terms_selected': original_FC1,
        'full_PSN_MAC_terms_selected': original_PSN,
        'prior_whole_domain_source_Gram_operations_reused_not_free': old_result['operations'],
        'rational_certificate_check': proof, 'ranks': rows,
        'claim_boundary': {'frozen_FP32_proof': False, 'rigorous_rounded_certificate': False,
                           'RTL_cycles': False, 'PPA': False, 'new_AEE': False}
    }
    out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
