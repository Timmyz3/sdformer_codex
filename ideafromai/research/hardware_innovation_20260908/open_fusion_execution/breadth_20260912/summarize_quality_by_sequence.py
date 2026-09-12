"""Report completed fresh825 students against the same-file NB0; no new inference."""
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows, totals = [], []
for axis in ('dense', 'contiguous34', 'lifting40'):
    path = HERE / 'algorithm' / 'valid825' / axis / 'paired_NB0.json'
    if not path.exists():
        continue
    pairs = json.loads(path.read_text())
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair['file'].rsplit('_', 1)[0]].append(pair)
    axis_rows = []
    for sequence, group in sorted(groups.items()):
        row = dict(axis=axis, sequence=sequence, frames=len(group),
                   valid_pixels=sum(p['valid_pixels'] for p in group),
                   AEE_frame_mean=sum(p['AEE'] for p in group) / len(group),
                   NB0_frame_mean=sum(p['NB0_AEE'] for p in group) / len(group),
                   delta_mean=sum(p['delta'] for p in group) / len(group),
                   frames_better=sum(p['delta'] < 0 for p in group))
        rows.append(row)
        axis_rows.append(row)
    totals.append(dict(axis=axis, frames=len(pairs), sequences=len(groups),
                       frames_better=sum(p['delta'] < 0 for p in pairs),
                       sequences_better=sum(r['delta_mean'] < 0 for r in axis_rows),
                       largest_sequence_delta=max(r['delta_mean'] for r in axis_rows)))
with (HERE / 'quality_by_sequence.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(HERE / 'quality_by_sequence.json').write_text(json.dumps({
    'scope': 'Descriptive paired full825 sequence means, not a new acceptance gate or independent holdout.',
    'rows': totals,
}, indent=2) + '\n')
print(json.dumps(totals))
