#!/usr/bin/env python3
"""Independent R41 raw-data audit; does not import or call the main runner."""
import hashlib
import json
import os
from pathlib import Path
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
ROOT = ROUND.parents[2]
OUT = HERE / 'RAW_COUNT_AUDIT_V1.json'
assert not OUT.exists(), 'fresh audit output required'
DS = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(2**20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def nonempty_group_count(values, width):
    # Different implementation from the runner's reshape/any: channel sums
    # at explicit boundaries, followed by a count of nonzero groups.
    sums = np.add.reduceat(values, np.arange(0, values.shape[-1], width),
                           axis=-1, dtype=np.int16)
    return np.count_nonzero(sums, axis=-1)


def agree(actual, expected, label):
    if isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
        assert np.array_equal(actual, expected), (label, int(np.count_nonzero(actual != expected)))
    else:
        assert actual == expected, (label, actual, expected)


started = time.monotonic()
assets_path = ROUND / 'ASSETS_V1.json'
assets = json.loads(assets_path.read_text())
admission_path = HERE / 'ADMISSION_V1.json'
admission = json.loads(admission_path.read_text())
results_path = ROUND / 'actual_v1/RESULTS_V1.json'
results = json.loads(results_path.read_text())
rows_path = ROUND / 'actual_v1/ROWS_V1.npz'
assert results['status'] == 'LEGACY_FIXTURE_GROUP_SCREEN_COMPLETE'
assert results['admission_sha256'] == file_sha(admission_path)
assert results['rows_sha256'] == file_sha(rows_path)
for name, expected in admission['sha256'].items():
    assert file_sha(ROUND / name) == expected, name
old_path = ROOT / assets['old_results_path']
assert file_sha(old_path) == assets['old_results_sha256']
old = json.loads(old_path.read_text())
old_q8 = [x for x in old['forward_records'] if x['arm'] == 'Q8serial']
assert len(old_q8) == len(assets['records']) == len(results['records']) == 8

observations = []
array_entries_checked = 0
input_scalar_count = 0
valid_candidate_rows = 0
all_keys = set()
yy, xx = np.indices((30, 40))
flat_y, flat_x = yy.ravel(), xx.ravel()

with np.load(rows_path, allow_pickle=False) as implementation_rows:
    for index, (asset, old_record, record) in enumerate(zip(assets['records'], old_q8, results['records'])):
        tid = asset['target_id']
        assert tid == old_record['target_id'] == record['target_id']
        assert asset['forward_id'] == old_record['forward_id']
        assert asset['FC_execution'] == old_record['target_execution']['FC']
        assert Path(asset['path']).name == old_record['prediction_file']
        assert asset['sha256'] == old_record['prediction_file_sha256']
        path = ROOT / asset['path']
        assert file_sha(path) == asset['sha256']
        combined = dict(ordinary128=0, zero_shift_direct=0, free5d_direct=0)
        identity = {'target_id': tid, 'FC': {}}
        with np.load(path, allow_pickle=False) as archive:
            for payload, fc, cin, cout in [('sn1', 'fc1', 384, 1536), ('sn2', 'fc2', 1536, 384)]:
                source = archive[payload]
                assert source.dtype == np.dtype('float32') and source.shape == (10, 1, 30, 40, cin)
                assert np.isfinite(source).all() and np.logical_or(source == 0, source == 1).all()
                assert hashlib.sha256(source.tobytes(order='C')).hexdigest() == asset['tensors'][payload]['sha256']
                assert asset['tensors'][payload] == old_record['tensor_records'][payload]
                input_scalar_count += source.size
                spikes = source[:, 0].astype(np.int8)
                del source
                trace = old_record['target_execution']['FC'][fc]
                assert trace['mode'] == 'Q8serial' and trace['bias_applications_per_output_element'] == 0
                assert trace['input_rows'] == 12000 and trace['int_mm_call_count'] == 8
                for bit, call in enumerate(trace['int_mm_calls']):
                    assert (call['Cin'], call['Cout'], call['rows']) == (cin, cout, 12000)
                    assert (call['bit'], call['coefficient']) == (bit, [1, 2, 4, 8, 16, 32, 64, -128][bit])
                    assert call['row_scope'] == 'all_spatial_rows'
                    assert call['explicit_padding_rows'] == call['explicit_padding_channels'] == 0

                g0 = nonempty_group_count(spikes, 128)
                g16 = nonempty_group_count(spikes, 16)
                positive = np.full((10, 30, 40, 5), -1, dtype=np.int16)
                negative = np.full_like(positive, -1)
                valid = np.zeros(positive.shape, dtype=bool)
                boundaries = np.arange(0, cin, 128)
                # Coordinate gather rather than runner's rectangular slices.
                # Signed subtraction and max/min rather than boolean masks.
                for t in range(1, 10):
                    current_rows = spikes[t].reshape(1200, cin)
                    for d, (dy, dx) in enumerate(DS):
                        ref_y, ref_x = flat_y + dy, flat_x + dx
                        legal = (ref_y >= 0) & (ref_y < 30) & (ref_x >= 0) & (ref_x < 40)
                        positions = np.flatnonzero(legal)
                        reference_rows = spikes[t - 1, ref_y[legal], ref_x[legal]]
                        difference = current_rows[positions] - reference_rows
                        gp = np.count_nonzero(np.maximum.reduceat(difference, boundaries, axis=-1) > 0, axis=-1)
                        gn = np.count_nonzero(np.minimum.reduceat(difference, boundaries, axis=-1) < 0, axis=-1)
                        positive[t, flat_y[legal], flat_x[legal], d] = gp
                        negative[t, flat_y[legal], flat_x[legal], d] = gn
                        valid[t, flat_y[legal], flat_x[legal], d] = True
                valid_candidate_rows += int(valid.sum())
                delta = positive.astype(np.int64) + negative.astype(np.int64)
                delta[~valid] = 10000
                # argmin's first-index rule independently enforces the fixed tie order.
                candidates = np.concatenate((g0[..., None], delta), axis=-1)
                choices = np.argmin(candidates, axis=-1)
                best = np.take_along_axis(candidates, choices[..., None], axis=-1)[..., 0]
                d0_candidates = np.stack((g0, delta[..., 0]), axis=-1)
                d0_choices = np.argmin(d0_candidates, axis=-1)
                d0 = np.min(d0_candidates, axis=-1)
                expect = {'G0': g0, 'ordinary16': g16, 'GP': positive, 'GN': negative, 'valid': valid,
                          'd0_cost': d0, 'd0_choice': d0_choices, 'free5d_cost': best, 'free5d_choice': choices}
                for name, values in expect.items():
                    key = f'{index:02d}_{fc}_{name}'
                    agree(implementation_rows[key], values, key)
                    all_keys.add(key)
                    array_entries_checked += values.size
                assert np.array_equal(choices[0], np.zeros((30, 40), dtype=np.int64))
                assert np.array_equal(best[0], g0[0])
                factor = len(trace['int_mm_calls']) * trace['int_mm_calls'][0]['Cout']
                fc_result = record['FC'][fc]
                agree(fc_result['factor'], factor, (tid, fc, 'factor'))
                totals = dict(ordinary128=0, zero_shift_direct=0, free5d_direct=0, ordinary16_nonzero_groups=0)
                for t in range(10):
                    service = {'ordinary128': int(g0[t].sum()) * factor,
                               'zero_shift_direct': int(d0[t].sum()) * factor,
                               'free5d_direct': int(best[t].sum()) * factor,
                               'ordinary16_nonzero_groups': int(g16[t].sum())}
                    per_time = {'t': t, 'grouped_service': service,
                                'choice_counts_direct_then_DS': [int(np.count_nonzero(choices[t] == c)) for c in range(6)],
                                'd0_choice_counts': [int(np.count_nonzero(d0_choices[t] == c)) for c in range(2)],
                                'valid_rows_each_d': [int(np.count_nonzero(valid[t, ..., d])) for d in range(5)],
                                'GP_sums_valid_each_d': [int(positive[t, ..., d][valid[t, ..., d]].sum()) for d in range(5)],
                                'GN_sums_valid_each_d': [int(negative[t, ..., d][valid[t, ..., d]].sum()) for d in range(5)],
                                'zero_current_128_groups': int((cin // 128) * 1200 - g0[t].sum())}
                    agree(fc_result['by_time'][t], per_time, (tid, fc, t))
                    for key in totals:
                        totals[key] += service[key]
                agree(fc_result['total'], totals, (tid, fc, 'total'))
                for key in combined:
                    combined[key] += totals[key]
                base = totals['ordinary128']
                potential = (base - totals['free5d_direct']) / base if base else None
                increment = (totals['zero_shift_direct'] - totals['free5d_direct']) / base if base else None
                agree(fc_result['potential'], potential, (tid, fc, 'potential'))
                agree(fc_result['displacement_increment'], increment, (tid, fc, 'increment'))
                identity['FC'][fc] = {'all_rows_exact': True, 'total': totals, 'potential': potential,
                                      'displacement_increment': increment}
        agree(record['combined'], combined, (tid, 'combined'))
        identity['combined'] = combined
        observations.append(identity)
    assert set(implementation_rows.files) == all_keys

summed = {key: sum(row['combined'][key] for row in observations)
          for key in ('ordinary128', 'zero_shift_direct', 'free5d_direct')}
base = summed['ordinary128']
saved = base - summed['free5d_direct']
extra = summed['zero_shift_direct'] - summed['free5d_direct']
aggregate = {'summed_grouped_service': summed, 'mean_grouped_service': {k: v / 8 for k, v in summed.items()},
             'potential': saved / base if base else None, 'displacement_increment': extra / base if base else None,
             'gate1_at_least_10_percent': bool(base and 10 * saved >= base),
             'gate2_at_least_5_percentage_points': bool(base and 20 * extra >= base),
             'ordinary_zero_shift_potential': (base - summed['zero_shift_direct']) / base if base else None}
aggregate['decision'] = ('ELIGIBLE_ONLY_FOR_SAME_OPERATOR_NEW_VERSION_CHECK'
                         if aggregate['gate1_at_least_10_percent'] and aggregate['gate2_at_least_5_percentage_points']
                         else 'STOP_DISPLACEMENT_ROUTE_ON_THIS_LEGACY_FIXTURE')
agree(results['aggregate'], aggregate, 'aggregate including gates')
assert results['model_forwards'] == results['GPU_calls'] == results['GT_reads'] == results['training_steps'] == 0
report = {
    'schema': 'flow_r41_independent_original_spikes_audit_v1', 'status': 'PASS',
    'reviewer': '/root/route40_fresh_adversary',
    'independence': 'No import or call of main runner; signed-difference max/min reduceat and coordinate gather, with argmin tie rule.',
    'audit_source_sha256': file_sha(__file__), 'results_sha256': file_sha(results_path),
    'rows_sha256': file_sha(rows_path), 'assets_sha256': file_sha(assets_path),
    'admission_sha256': file_sha(admission_path), 'old_results_sha256': file_sha(old_path),
    'identity_count': 8, 'payload_count': 16, 'full_row_count': 192000,
    'input_scalar_count': int(input_scalar_count), 'valid_candidate_rows': valid_candidate_rows,
    'row_array_entries_checked': int(array_entries_checked), 'row_array_count': len(all_keys),
    'per_identity_FC_time_summaries_checked': 160, 'aggregate': aggregate, 'records': observations,
    'new_model_forwards': 0, 'new_GPU_calls': 0, 'GT_payload_reads': 0,
    'not_validated': ['integer current replay or BN execution', 'reference cache or causal schedule',
                      'cycles, energy, RTL/DC', 'E0 transfer or consecutive data windows', 'optical-flow-specific mechanism'],
    'elapsed_seconds': time.monotonic() - started,
}
OUT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps({k: report[k] for k in ('status', 'identity_count', 'payload_count', 'row_array_entries_checked',
                                       'elapsed_seconds', 'aggregate')}, sort_keys=True))
