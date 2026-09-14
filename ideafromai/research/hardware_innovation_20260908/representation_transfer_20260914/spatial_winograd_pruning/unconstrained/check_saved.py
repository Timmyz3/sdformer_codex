"""Read the emitted arrays, checking model semantics and endpoint arithmetic."""
from pathlib import Path
import importlib.util
import json
import csv
import numpy as np

H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('phase3_export', H / 'export.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
f = e.read_npz(H / 'factors.npz')
parent = e.read_npz(H.parent.parent / 'spatial_winograd_inputs/factors.npz')
assert str(f['function_type']) == 'phase3' and 'q2' not in f
assert np.array_equal(f['physical_coeff3'], e.transform(parent['q2'].astype(np.int64))[:, :, [0, 1, 3]])
assert np.array_equal(f['output_scale'], parent['output_scale']/2)
assert np.array_equal(f['a_q40'], np.rint(f['output_scale']*f['BN_gain']*2**40).astype(np.int32))
for k in ['q1', 'b_q20', 'theta', 'bias', 'BN_gain', 'BN_offset', 'first_scale', 'z_lower', 'z_upper']:
    assert np.array_equal(f[k], parent[k]), k
assert np.array_equal(f['p_lower'], f['p_phase_lower'].min(0))
assert np.array_equal(f['p_upper'], f['p_phase_upper'].max(0))
summary = {'passed': True, 'function_type': 'phase3', 'q2_absent': True, 'datasets': {}}
for ds, filename, original_path in [
    ('tiles135', 'gold_tiles.npz', H.parent.parent/'spatial_winograd_inputs/gold_tiles.npz'),
    ('sequences36', 'gold_sequences.npz', H.parent.parent/'quality/q11/sequence_tiles.npz')]:
    g = e.read_npz(H / filename)
    p = e.read_npz(original_path)
    for k in ['source_words', 'output_origin_yx', 'identity_fp32_bits', 'J_q20', 'z_halo_int']:
        assert np.array_equal(g[k], p[k]), k
    M = g['physical_M_int'].astype(np.int64)
    raw = np.stack([M[..., 0]+M[..., 1], M[..., 1]-M[..., 2]], axis=4)
    assert np.array_equal(raw, g['p_int'])
    assert np.array_equal(raw.sum(4), 2*p['p_int'].astype(np.int64).sum(4))
    j = e.identity_to_j(g['identity_fp32_bits'].view(np.float32))
    wide = raw*f['a_q40'][None, None, :, None, None] + \
        (j+f['b_q20'][None, None, :, None, None])*(1 << 20)
    assert np.array_equal(j, g['J_q20'])
    assert np.array_equal(wide, g['wide_int64'])
    assert np.array_equal(e.rne_shift(wide, 26, 24), g['i24'])
    assert np.all(abs(wide) <= f['wide_abs_bound'][None, None, :, None, None])
    summary['datasets'][ds] = dict(records=len(raw), each_raw_J_wide_I24=int(raw.size),
                                  pair_sum_differences=0, endpoint_differences=0)
(H / 'saved_artifact_check.json').write_text(json.dumps(summary, indent=2)+'\n')
reports = json.loads((H / 'SUMMARY.json').read_text())
rows = []
for dataset, arms in reports['comparison'].items():
    for arm, a in arms.items():
        rows.append(dict(dataset=dataset, arm=arm, raw_values=a['raw_values'],
            I24_relative_L2=a['i24_error_vs_q11_parent']['relative_L2'],
            I24_RMSE=a['i24_error_vs_q11_parent']['RMSE'],
            BN_scaled_RMSE=a['bn_scaled_output_error']['RMSE'],
            phase0_I24_RMSE=a['phase_errors']['0']['i24_error_vs_q11_parent']['RMSE'],
            phase1_I24_RMSE=a['phase_errors']['1']['i24_error_vs_q11_parent']['RMSE'],
            raw_parent_multiplier=a['raw_parent_multiplier']))
with (H / 'comparison.csv').open('w', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(summary))
