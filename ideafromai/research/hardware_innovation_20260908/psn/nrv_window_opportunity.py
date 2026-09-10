"""Count NR4 source-control opportunities; this is not a cycle model.

Use the complete integer student capture: all 1200 positions and 384 source
channels in each of 10 frames x 6 stage2 blocks. Source NRV removes rows whose
P positions are all code zero, then packs four consecutive retained rows.
The experiment deliberately precedes output-specific zero-weight compaction.
Both representations get the same four-input reduction opportunity.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def time_decode(dictionary):
    rows = []
    for column in dictionary.T:
        if np.any(column) and not any(np.array_equal(column, r) for r in rows):
            rows.append(column)
    return np.stack(rows, axis=1)


def count(codes, decode, positions):
    histogram = np.zeros(5, dtype=np.int64)
    retained_rows = packets = 0
    for start in range(0, len(codes), positions):
        rows = codes[start:start + positions].T
        rows = rows[np.any(rows != 0, axis=1)]
        retained_rows += len(rows)
        if len(rows) == 0:
            continue
        padding = (-len(rows)) % 4
        if padding:
            rows = np.pad(rows, ((0, padding), (0, 0)))
        groups = rows.reshape(-1, 4, rows.shape[1])
        counts = decode[groups].sum(axis=1, dtype=np.int64)
        histogram += np.bincount(counts.ravel(), minlength=5)
        packets += len(groups)
    scalar = int(histogram @ np.arange(5))
    merged = int(histogram[1:].sum())
    return dict(retained_source_rows=retained_rows, nr4_packets=packets,
                scalar_events=scalar, merged_destinations=merged,
                member_histogram=histogram[1:].tolist(),
                event_issue_reduction=1 - merged / scalar,
                single_member_fraction=float(histogram[1] / merged))


def main():
    folder = ROOT / 'algorithm/direct_code_integer/deployment/capture10'
    records = []
    class_decode = np.eye(8, dtype=np.uint8)[:, 1:]
    for filename in sorted(folder.glob('*.npz')):
        with np.load(filename) as data:
            codes = data['codes']
            decodes = {'class': class_decode,
                       'time': time_decode(data['class_to_spikes'])}
            for positions in (4, 8):
                for route, decode in decodes.items():
                    records.append(dict(capture=filename.name,
                                        block=int(filename.stem[-1]),
                                        P=positions, route=route,
                                        **count(codes, decode, positions)))
    totals = []
    for positions in (4, 8):
        for route in ('class', 'time'):
            chosen = [r for r in records if r['P'] == positions and r['route'] == route]
            scalar = sum(r['scalar_events'] for r in chosen)
            merged = sum(r['merged_destinations'] for r in chosen)
            hist = np.asarray([r['member_histogram'] for r in chosen]).sum(axis=0)
            totals.append(dict(P=positions, route=route, captures=len(chosen),
                               scalar_events=scalar, merged_destinations=merged,
                               event_issue_reduction=1 - merged / scalar,
                               single_member_fraction=float(hist[0] / merged),
                               member_histogram=hist.tolist()))
    result = dict(scope='SOURCE_CONTROL_OPPORTUNITY_ONLY',
                  model='new integer bits3 student, not native ep34',
                  shape_per_capture=[1200, 384], unique_captures=len(records)//4,
                  nr=4,
                  excluded=['output-specific W=0 compaction', 'weight supply',
                            'NR4 generation and arbitration cycles',
                            'CSA area and timing', 'PSN consumer',
                            'finite source refill and backpressure'],
                  totals=totals, records=records)
    target = Path(__file__).with_suffix('.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(totals, indent=2))
    print(target)


if __name__ == '__main__':
    main()
