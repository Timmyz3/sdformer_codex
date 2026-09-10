"""Actual source events and NRV row counts; no hardware cycle estimate."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'bn_state'))
from support_service_model import read_torch


def main():
    books = np.load(ROOT/'algorithm/stage2_temporal_codes/codebooks.npz')
    params = read_torch(ROOT/'algorithm/stage2_temporal_codes/integer_parameters.pt')
    report = {'scope': 'four compiled FP32 source models and common teacher; ten diverse frames, all six S2 blocks',
        'units': 'event counts and source NRV row counts, never cycles/energy',
        'comparison': 'class-coded single event per active source versus the same model decoded to distinct live time rows',
        'physical_boundary': 'packed-time gets the same compressed source codes and once-per-row W fetch; state, merger, PSN, source coefficient precision and ports still cost resources',
        'variants': {}}
    for variant in ('teacher','bits3','rows3','scalar8','scores8'):
        folder = (ROOT/'algorithm/stage2_class_shift/power2_trained_capture' if variant == 'teacher'
                  else ROOT/'algorithm/direct_code_stage2'/('compiled_'+variant)/'capture')
        files = sorted(folder.glob('*.npz'))
        assert len(files) == 60
        blocks = defaultdict(Counter)
        histograms = defaultdict(lambda: np.zeros(8, dtype=np.int64))
        for filename in files:
            name = filename.stem.rsplit('_', 1)[1]
            block = int(name[-1])
            prefix = f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{block}.mlp.'
            codes = np.load(filename)['codes'].astype(np.int64)
            assert codes.shape == (1200, 384)
            dictionary = books[name+'_dictionary'].astype(np.uint8)
            # Strong comparator removes constant zero and duplicate time rows.
            unique = np.unique(dictionary.T, axis=0)
            decode = unique[np.any(unique, axis=1)]
            counts = decode.sum(0)
            nonzero_w_per_c = (params[prefix]['weight_int8'] != 0).sum(0)
            class_per_c = (codes != 0).sum(0)
            time_per_c = counts[codes].sum(0)
            m = blocks[name]
            m['trajectories'] += codes.size
            m['class_events_before_Wzero'] += int(class_per_c.sum())
            m['time_events_before_Wzero'] += int(time_per_c.sum())
            m['class_adds_all_H_nonzero_W'] += int(class_per_c @ nonzero_w_per_c)
            m['time_adds_all_H_nonzero_W'] += int(time_per_c @ nonzero_w_per_c)
            histograms[name] += np.bincount(codes.ravel(), minlength=8)
            for p in (8,16,32):
                for start in range(0, len(codes), p):
                    tile = codes[start:start+p]
                    m[f'P{p}_packed_NRV_rows'] += int((tile != 0).any(0).sum())
                    # Tick/class plane paths separately scan nonempty row lists.
                    m[f'P{p}_time_plane_NRV_rows'] += sum(int(row[tile].any(0).sum()) for row in decode)
                    m[f'P{p}_class_plane_NRV_rows'] += sum(int((tile == k).any(0).sum()) for k in range(1,8))
        aggregate = Counter()
        records = {}
        for name, counts in blocks.items():
            aggregate.update(counts)
            records[name] = {'totals_ten_frames': dict(counts), 'class_histogram': histograms[name].tolist(),
                'zero_code_fraction': float(histograms[name][0]/histograms[name].sum()),
                'time_to_class_add_ratio': counts['time_adds_all_H_nonzero_W']/counts['class_adds_all_H_nonzero_W']}
        report['variants'][variant] = {'per_frame': {k: v/10 for k,v in aggregate.items()},
            'time_to_class_add_ratio': aggregate['time_adds_all_H_nonzero_W']/aggregate['class_adds_all_H_nonzero_W'],
            'zero_code_fraction': float(sum(h[0] for h in histograms.values())/aggregate['trajectories']), 'modules': records}
        print(variant, json.dumps({k:v for k,v in report['variants'][variant].items() if k != 'modules'}), flush=True)
    (ROOT/'psn/direct_code_activity.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
