from pathlib import Path
from collections import Counter
import json
import shutil
import numpy as np

H = Path(__file__).resolve().parent
B = H.parents[1]
OLD = B / 'fusion_ten_trials_20260914/decompositions/q1_bitplanes'

def readhex(p):
    return np.array([int(v, 16) for v in p.read_text().split()], np.uint32).view(np.int32).astype(np.int64)

def writehex(p, a):
    p.write_text(''.join(f'{int(v) & 0xffffffff:08x}\n' for v in np.asarray(a).flat))

def compile_table(q):
    classes = np.zeros((864, 4), np.int64)
    representatives = np.zeros((32, 8), np.int64)
    sizes = []
    unique_counts = []
    for g in range(4):
        keys = [int(row[0] & 7) | (int(row[1] & 7) << 3) for row in q[:, 2*g:2*g+2]]
        frequency = Counter(key for key in keys if key)
        selected = sorted([key for key in frequency if frequency[key] >= 2], key=lambda key: (-frequency[key], key))[:32]
        sizes.append(len(selected))
        unique_counts.append(len(frequency))
        for k, key in enumerate(keys):
            classes[k, g] = 0 if key == 0 else selected.index(key) + 1 if key in selected else 63
        for slot, key in enumerate(selected):
            representatives[slot, 2*g:2*g+2] = q[keys.index(key), 2*g:2*g+2]
    return classes, representatives, sizes, unique_counts

def prepare():
    cases = [(c['name'], OLD / 'fixtures' / c['name']) for c in json.loads((OLD / 'definition.json').read_text())['fixtures']]
    audit = H.parent / 'audit_decompositions/fixtures'
    cases += [(name, audit / name) for name in ['random_padding_poison', 'count_864_signed3']]
    records = []
    for name, source in cases:
        p = H / 'fixtures' / name
        p.mkdir(parents=True, exist_ok=True)
        for f in ['source.hex', 'origin.hex', 'q1.hex', 'q2.hex', 'k_live.hex', 'gold.hex']:
            shutil.copyfile(source / f, p / f)
        q = readhex(p / 'q1.hex').reshape(864, 8)
        assert np.all((q >= -3) & (q <= 3))
        cls, rep, sizes, unique = compile_table(q)
        writehex(p / 'class.hex', np.sum(cls << (6 * np.arange(4)), axis=1))
        writehex(p / 'representative.hex', rep)
        writehex(p / 'ngroups.hex', [max(sizes)])
        records.append(dict(name=name, group_sizes=sizes, nonzero_unique_keys=unique,
                            grouped_pairs=int(np.count_nonzero((cls > 0) & (cls <= 32))),
                            direct_pairs=int(np.count_nonzero(cls == 63))))
    (H / 'fixtures.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))

if __name__ == '__main__':
    prepare()
