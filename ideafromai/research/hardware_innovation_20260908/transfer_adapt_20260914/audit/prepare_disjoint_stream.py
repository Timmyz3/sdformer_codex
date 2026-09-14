"""Additional output blocks with no input-halo overlap with calibration 0..31.

Same frame, not a new sequence or a train/validation split. Original native
source and full-frame integer raw gold are preserved. No intermediate is
provided to RTL.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np

H = Path(__file__).resolve().parent
B = H.parents[1]
spec = importlib.util.spec_from_file_location(
    'native_fixture', B / 'fusion_review_followup_20260914/pair_dictionary/prepare.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
old = B / 'fusion_review_followup_20260914/pair_dictionary/fixtures/real_0'
q = m.readhex(old / 'q1.hex').reshape(864, 8)
v = m.readhex(old / 'q2.hex').reshape(12, 8, 8).transpose(0, 2, 1).reshape(96, 8)
source = np.load(B / 'r8_consumer_fusion_20260914/data/first_source_words.npy', mmap_mode='r')
gold = np.load(B / 'r8_consumer_fusion_20260914/data/raw_p_full.npy', mmap_mode='r')
out = H / 'fixtures/disjoint_4000'
out.mkdir(parents=True, exist_ok=True)
manifest = []
for tile in range(4000, 4064):
    oy, ox = 2 * (tile // 160) - 1, 2 * (tile % 160) - 1
    assert oy > 2  # Calibration 0..31 can read valid input rows 0..2 only.
    s = np.zeros((96, 4, 4), np.int64)
    for y in range(4):
        for x in range(4):
            if 0 <= oy+y < 240 and 0 <= ox+x < 320:
                s[:, y, x] = source[:, oy+y, ox+x]
    e = np.zeros((864, 40), np.int64)
    for k in range(864):
        c, tap = divmod(k, 9)
        for p in range(4):
            e[k, p*10:p*10+10] = (s[c, p//2+tap//3, p%2+tap%3] >> np.arange(10)) & 1
    z = e.T @ q
    raw = np.concatenate([z @ v[og*8:og*8+8].T for og in range(12)])
    reference = gold[tile].reshape(10, 12, 8, 4).transpose(1, 3, 0, 2).reshape(480, 8)
    assert np.array_equal(raw, reference), tile
    d = out / str(tile)
    d.mkdir(exist_ok=True)
    for name, a in [('source', s), ('origin', [oy, ox]), ('gold', raw)]:
        m.writehex(d / (name + '.hex'), a)
    manifest.append(str(d))
(out / 'manifest.txt').write_text('\n'.join(manifest) + '\n')
(H / 'disjoint_input_check.json').write_text(json.dumps({
    'tiles': 64, 'first': 4000, 'last': 4063,
    'independent_raw_values_recomputed': 245760,
    'matches_full_capture': True,
    'same_frame_as_calibration': True,
    'overlap_with_calibration_input_rows': False,
    'calibration': 'Original fixed order from tiles 0..31; no refit on this set.',
    'manifest': str(out / 'manifest.txt'),
}, indent=2) + '\n')
print('Prepared 64 source-disjoint same-frame tiles; all 245760 raw values matched.')
