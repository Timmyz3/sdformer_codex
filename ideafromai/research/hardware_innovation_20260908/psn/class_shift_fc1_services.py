"""Existing finite FC1 service on each trained model's own source captures.

P16 flow-shop numbers are optimistic overlap bounds: same two half-tile state
sets fit the seven banks, but state/coeff-bank arbitration and external bus
backpressure are not closed. Ordinary dedicated MAC is an extra resource.
"""
from pathlib import Path
import argparse
import json
import sys
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'bn_state'))
import stage2_temporal_service_model as svc


def pipeline_bound(tasks):
    front, back, completed = 0, 0, []
    for i, (f, p) in enumerate(tasks):
        free = completed[i-2] if i >= 2 else 0
        front = max(front, free)+f
        back = max(back, front)+p
        completed.append(back)
    return back


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=('signed', 'exact_row', 'onehot7'), default='signed')
    args = parser.parse_args()
    folder = ROOT/'algorithm/stage2_class_shift'
    consumers = json.loads((folder/'consumers.json').read_text())
    basis = np.load(ROOT/'algorithm/stage2_temporal_codes/signed_basis.npz')
    books = np.load(ROOT/'algorithm/stage2_temporal_codes/codebooks.npz')
    svc.svc.C, svc.svc.H, svc.svc.LANES = 384, 1536, 96
    report = {'scope': 'finite local FC1 services and ideal P16 dedicated-MAC overlap bounds; no RTL/PPA',
        'route': args.route,
        'limits': 'P16 uses packed half-tile address half*(R*16)+r*16+p. Overlap bound gives independent stages ideal state/coefficient port access; no claim of completed arbitration. Both B formats use correct new per-prefix input captures.',
        'tau24_stripe': {'bytes_padded': 39936, 'bus_beats': 312, 'local_tau_template_load_beats': 24},
        'variants': {}}
    for variant in ('integer_trained', 'power2_trained'):
        captures = sorted((folder/(variant+'_capture')).glob('*.npz'))
        assert len(captures) == 60
        total = Counter()
        records = []
        for index, path in enumerate(captures):
            name = path.stem.rsplit('_', 1)[1]
            coeff = np.asarray(consumers[variant][name]['B_int8'], dtype=np.int64)
            decode = basis[name+'_coordinates_int8'].astype(np.int64).T
            if args.route == 'onehot7':
                coeff = (coeff @ decode)[:, 1:]
                decode = np.eye(8, dtype=np.int64)[1:]
            elif args.route == 'exact_row':
                dictionary = books[name+'_dictionary'].astype(np.int64)
                rows, seen = [], set()
                for t in range(10):
                    key = tuple(dictionary[:, t])
                    if any(key) and key not in seen:
                        rows.append(t)
                        seen.add(key)
                selected = basis[name+'_selected_code_indices']
                time_basis = dictionary[selected].T[rows]
                inverse = np.rint(np.linalg.inv(time_basis)).astype(np.int64)
                row_decode = dictionary[:, rows].T
                row_coeff = coeff @ inverse
                assert np.array_equal(row_coeff @ row_decode, coeff @ decode)
                decode, coeff = row_decode, row_coeff
            codes = np.load(path)['codes']
            result = {'capture': path.name}
            tasks = []
            for size in (32, 16):
                svc.TILE = size
                local = Counter()
                for start in range(0, len(codes), size):
                    metrics, _, _ = svc.evaluate(codes[start:start+size], (decode, coeff), 1, platforms=('L7_striped',))
                    m = metrics['L7_striped']
                    local['fc1'] += m['fc1_bank_service_beats']
                    local['word_psn'] += m['psn_local_service_beats']
                    local['W_vector_requests'] += m['coefficient_vector_requests']
                    local['W_word_reads'] += m['coefficient_word_reads']
                    local['PSN_state_vector_reads'] += m['psn_state_vector_reads']
                    # HalfA+halfB share H96 coefficients; one tau24 preload.
                    if size == 16:
                        tasks.append((m['fc1_bank_service_beats'],
                                      m['psn_local_service_beats']+(24 if (start//size)%2 == 0 else 0)))
                for key, value in local.items():
                    result[f'P{size}_{key}'] = int(value)*16
                result[f'P{size}_serial'] = int(local['fc1']+local['word_psn']+38*24)*16
            # H-inner schedule: repeat each consecutive half-pair for16 H96
            # stripes before moving to the next P32. Last P32 has only16p.
            ordered = []
            for start in range(0, len(tasks), 2):
                for _ in range(16):
                    ordered.extend(tasks[start:start+2])
            result['P16_dedicated_MAC_ideal_two_buffer_bound'] = pipeline_bound(ordered)
            for key, value in result.items():
                if key != 'capture':
                    total[key] += value
            records.append(result)
            print(variant, index+1, name, flush=True)
        report['variants'][variant] = {'aggregate': dict(total),
                                      'per_frame': {k: v/10 for k, v in total.items()},
                                      'captures': records}
    suffix = '' if args.route == 'signed' else '_'+args.route
    (Path(__file__).resolve().parent/('class_shift_fc1_services'+suffix+'.json')).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
