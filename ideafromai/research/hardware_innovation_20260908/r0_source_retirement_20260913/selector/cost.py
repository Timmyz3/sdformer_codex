"""Exact no-stall mode3 service estimator used to select masks, not RTL results."""
from pathlib import Path
import json
import numpy as np


def features(bits, origins):
    bits = np.asarray(bits, dtype=np.uint16)
    packed = (bits * (1 << np.arange(10, dtype=np.uint16))[None, :, None, None, None]).sum(axis=1)
    f = len(bits)
    w = np.zeros(24, dtype=np.int64)
    sums = w.copy(); updates = w.copy(); active = w.copy()
    source = w.copy()
    for tile, (oy, ox) in enumerate(origins):
        for cg in range(24):
            for y in range(4):
                for x in range(4):
                    if not (0 <= int(oy)+y < 240 and 0 <= int(ox)+x < 320):
                        continue
                    source[cg] += 4
                    p = sum(0 <= y-py < 3 and 0 <= x-px < 3 for py in range(2) for px in range(2))
                    for k in (0, 2):
                        a = int(packed[tile, cg*4+k, y, x])
                        b = int(packed[tile, cg*4+k+1, y, x])
                        w[cg] += p * (int(a != 0) + int(b != 0))
                        sums[cg] += p * int((a & b) != 0)
                        active[cg] += p * int((a | b) != 0)
                        updates[cg] += p * (a | b).bit_count()
    return dict(tiles=f, source=source, weight=w, sums=sums, updates=updates,
                active=active, execution=3*active+w+sums+3*updates)


def measure(live, feat):
    g = np.asarray(live, dtype=bool).sum(axis=0)
    source_live = g > 0
    f = feat['tiles']
    updates = int(g @ feat['updates'])
    cycles = 1442*f + int(np.where(source_live, 160*f, 2*f).sum()) + int(g @ feat['execution'])
    return dict(cycles=cycles, source_words=int(source_live @ feat['source']),
                weight_words=int(g @ feat['weight']), sum_issues=int(g @ feat['sums']),
                update_issues=updates, psum_reads=updates+480*f, psum_writes=updates+480*f,
                live_source_groups=int(source_live.sum()))


def verify_previous():
    prev = Path(__file__).resolve().parents[2] / 'r0_execution_trials_20260913'
    rows = json.loads((prev/'consumer_enumeration/results.json').read_text())
    checked = 0
    for name in sorted({r['fixture'] for r in rows}):
        d = prev/'native_sparse/fixtures'/name
        read = lambda file: np.array([int(x, 16) for x in (d/file).read_text().split()])
        words = read('source.hex').reshape(96,4,4)
        bits = ((words[None] >> np.arange(10)[:,None,None,None]) & 1)[None]
        origin = read('origin.hex').astype(np.uint32).view(np.int32)[None]
        live = read('mask.hex').reshape(12,24).astype(bool)
        predicted = measure(live, features(bits, origin))
        for row in rows:
            if row['fixture'] != name or row['mode'] != 3: continue
            for k, v in predicted.items():
                if k == 'live_source_groups': continue
                observed = row[k]
                if k == 'cycles': observed -= sum(row[s] for s in ('source_stalls','weight_stalls','output_stalls'))
                assert observed == v, (name, k, observed, v)
            checked += 1
    result = dict(mode3_commands_checked=checked, all_counters_and_cycles_match=True,
                  source='previous real Verilator results; no new timing claimed')
    (Path(__file__).parent/'cost_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    verify_previous()
