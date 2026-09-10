"""C1 support-class capacity opportunity; no cycle or PPA model.

Preselected samples 0/1/2, four bottleneck operators, K partitions 0/216/431;
all 3000 rows in each domain. Larger windows preserve source/weight identity.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

HW = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07')
LEDGER = HW/'results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901/ep34_c1_support16_rows.memh'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    output = ap.parse_args().output
    result = {}
    hist = {m: Counter() for m in (64, 128, 256, 512, 1024)}
    totals = {m: Counter() for m in hist}
    samples = {m: [] for m in hist}
    with LEDGER.open('rb') as stream:
        for sample in (0, 1, 2):
            for operator in range(4):
                for partition in (0, 216, 431):
                    phase = (sample*4+operator)*432+partition
                    stream.seek(phase*3000*9)
                    rows = [int(v, 16) & 65535 for v in stream.read(3000*9).splitlines()]
                    for size in hist:
                        for start in range(0, len(rows), size):
                            block = rows[start:start+size]
                            nontrivial = [m for m in block if m.bit_count() > 1]
                            unique = set(nontrivial)
                            hist[size][len(unique)] += 1
                            samples[size].append(len(unique))
                            totals[size].update(windows=1, logical_rows=len(block),
                                nonzero_rows=sum(m != 0 for m in block),
                                nontrivial_rows=len(nontrivial),
                                unique_nontrivial_supports=len(unique),
                                singleton_or_zero_rows=len(block)-len(nontrivial),
                                capacity64_fits=int(len(unique) <= 64),
                                capacity128_fits=int(len(unique) <= 128))
    for size, t in totals.items():
        row = dict(t)
        row.update(unique_support_mean=statistics.mean(samples[size]),
                   unique_support_median=statistics.median(samples[size]),
                   unique_support_max=max(samples[size]),
                   capacity64_window_fraction=t['capacity64_fits']/t['windows'],
                   capacity128_window_fraction=t['capacity128_fits']/t['windows'],
                   unique_support_histogram=dict(sorted(hist[size].items())))
        result[str(size)] = row
        print(size, {k:v for k,v in row.items() if k != 'unique_support_histogram'}, flush=True)
    data = dict(scope='3 samples x 4 C1 operators x 3 K partitions x all 3000 rows',
                source=str(LEDGER), selected_rows=108000,
                note='Fixed-K grouping changes the legacy native chunk->K service order; source/Acc traffic must be charged separately.',
                counting='One prospective wide vector per distinct support of popcount>1; singleton uses raw weight, zero uses no product. All logical destination updates remain.',
                limitations=['Capacity opportunity only; not minimal liveness, implemented SRAM occupancy, cycle, energy, or PPA.',
                             'Classifying/recording all original rows, destination maps, expansion, final Acc and weights are additional.',
                             'Compare complete Prosperity plus ordinary EM representative sharing and Transitive Array, not only m935.'],
                rows=result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    main()
