"""A first necessary-condition screen for the new cross-BN V_sigma proposal."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
PRIOR = BASE.parent / 'mechanism_rebuild_gh_20260906/scripts'
sys.path.insert(0, str(PRIOR))
from screen_threshold_packets import sources, EXPECTED


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = BASE / 'temporal_signatures_r1.json'
    assert not output.exists(), 'Preserve the first predeclared result.'
    plan = json.loads((BASE / 'screen_plan.json').read_text())
    arrays = sources()  # Existing strict frame parser, including complete-file SHA checks.
    rows = []
    T = plan['c2']['T']
    pc = np.array([i.bit_count() for i in range(1 << T)], dtype=np.int64)
    for stage in plan['c2']['stages']:
        key = f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp.fc1'
        spec, S = arrays[key]
        N, C = S.shape
        H, P = spec['output_channels'], N // T
        assert N == T * P and spec['input_shape'][0] == T
        sig = np.tensordot(1 << np.arange(T, dtype=np.int64), S.reshape(T, P, C), axes=(0, 0))
        # Independent reconstruction certifies that flattening/transposition lost no step.
        reconstructed = ((sig[None, :, :] >> np.arange(T)[:, None, None]) & 1).astype(np.uint8)
        assert np.array_equal(reconstructed, S.reshape(T, P, C))
        ordered = np.sort(sig, axis=1)
        first = np.concatenate([np.ones((P, 1), dtype=bool), ordered[:, 1:] != ordered[:, :-1]], axis=1)
        live_unique = first & (ordered != 0)
        R = live_unique.sum(1)
        nonempty_channels = (sig != 0).sum(1)
        live_Y = S.reshape(T, P, C).any(2).sum(0)
        unique_population = (pc[ordered] * live_unique).sum(1)
        original_contributions = int(S.sum(dtype=np.int64))
        restored_contributions = int(unique_population.sum())
        build_adds = int((nonempty_channels - R).sum())
        restore_adds = int((unique_population - live_Y).sum())
        ordinary_adds = original_contributions - int(live_Y.sum())
        nonzero_positions = int(np.count_nonzero(R))
        adaptive_R = np.minimum(R, T)
        rows.append({
            'module': key, 'N_includes_T': N, 'P': P, 'C': C, 'H': H, 'T': T,
            'R_histogram': {str(int(v)): int(n) for v, n in zip(*np.unique(R, return_counts=True))},
            'R_mean': float(R.mean()),
            'R_percentiles_0_25_50_75_90_99_100': np.percentile(R, [0, 25, 50, 75, 90, 99, 100]).tolist(),
            'R_zero': int(np.count_nonzero(R == 0)),
            'R_between_1_and_T_minus_1': int(np.count_nonzero((R > 0) & (R < T))),
            'R_less_than_T_fraction': float(np.mean(R < T)),
            'R_equal_T_fraction': float(np.mean(R == T)),
            'R_greater_than_T_fraction': float(np.mean(R > T)),
            'all_V_payload_over_dense_Y_equal_width_ignoring_metadata': float(R.sum() / (T * P)),
            'adaptive_min_R_T_payload_over_zero_aware_Y_ignoring_metadata':
                float(adaptive_R.sum() / (T * nonzero_positions)) if nonzero_positions else None,
            'full_PSN_coefficient_terms_per_output_channel': T * T * P,
            'V_PSN_coefficient_terms_per_output_channel': int(T * R.sum()),
            'ordinary_source_contribution_occurrences_per_output_channel': original_contributions,
            'V_build_source_contribution_occurrences_per_output_channel': int(nonempty_channels.sum()),
            'V_restore_contribution_occurrences_per_output_channel': restored_contributions,
            'ordinary_binary_adds_per_output_channel': ordinary_adds,
            'V_build_binary_adds_per_output_channel': build_adds,
            'V_restore_binary_adds_per_output_channel': restore_adds,
            'build_plus_restore_over_ordinary_binary_adds':
                (build_adds + restore_adds) / ordinary_adds if ordinary_adds else None,
            'temporal_bit_reconstruction_mismatches': 0,
            'note': 'Arithmetic counts are opportunity counts, not physical weight reads or cycles. '
                    'V_sigma is a rational-linear candidate; no frozen FP32 or theta=1 assumption.'
        })
        print(json.dumps({k: rows[-1][k] for k in ['module', 'R_mean', 'R_less_than_T_fraction',
            'all_V_payload_over_dense_Y_equal_width_ignoring_metadata',
            'build_plus_restore_over_ordinary_binary_adds']}, ensure_ascii=False), flush=True)
    result = {
        'date': '2026-09-07', 'predeclared_plan_sha256': digest(BASE / 'screen_plan.json'),
        'script_sha256': digest(Path(__file__)),
        'parser_sha256': digest(PRIOR / 'screen_threshold_packets.py'),
        'capture_sha256': EXPECTED, 'sample_id': 0, 'layers': rows,
        'claim_boundary': {'new_forward': False, 'rtl_cycles': False, 'ppa': False,
                           'frozen_fp32_equivalence': False, 'full_network': False}
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
