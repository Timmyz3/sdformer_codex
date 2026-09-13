"""Exhaustive mathematical check of F(2,3) binary input-transform values.

No model data, training, RTL, timing, or hardware-performance measurement.
Uses row-major S and V indexing and the standard CVPR2016 B transpose.
"""
import json
from pathlib import Path
from collections import Counter

BT = ((1, 0, -1, 0), (0, 1, 1, 0),
      (0, -1, 1, 0), (0, 1, 0, -1))
terms = []
for a in range(4):
    for b in range(4):
        terms.append([(4*i+j, BT[a][i]*BT[b][j])
                      for i in range(4) for j in range(4)
                      if BT[a][i]*BT[b][j]])
histograms = [Counter() for _ in terms]
for mask in range(1 << 16):
    for k, ts in enumerate(terms):
        value = sum(sign*((mask >> pos) & 1) for pos, sign in ts)
        histograms[k][value] += 1

coordinates = []
for k, (ts, hist) in enumerate(zip(terms, histograms)):
    expected = [0, 1, 2, 3, 4] if k == 5 else [-2, -1, 0, 1, 2]
    assert sorted(hist) == expected
    assert sum(hist.values()) == 65536
    coordinates.append({
        'coordinate_zero_based': [k // 4, k % 4],
        'row_major_index': k,
        'source_terms': [{'coordinate_zero_based': [i // 4, i % 4],
                          'sign': sign} for i, sign in ts],
        'values': sorted(hist),
        'histogram_over_all_binary_tiles': dict(sorted(hist.items())),
    })
result = {
    'scope': 'exhaustive algebraic verification only; no target model or hardware experiment',
    'input_count': 65536, 'input_shape': [4, 4], 'input_alphabet': [0, 1],
    'definition': 'V = B_transpose @ S @ B', 'B_transpose': BT,
    'vectorization': 'row-major for recorded source indices',
    'all_coordinate_value_sets_match_prediction': True,
    'coordinates_with_value_positive_3': [r['coordinate_zero_based'] for r in coordinates if 3 in r['values']],
    'coordinates_with_value_negative_3': [r['coordinate_zero_based'] for r in coordinates if -3 in r['values']],
    'coordinates': coordinates,
}
out = Path(__file__).with_name('winograd_binary_exhaustive.json')
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['input_count', 'all_coordinate_value_sets_match_prediction',
      'coordinates_with_value_positive_3', 'coordinates_with_value_negative_3']}))
