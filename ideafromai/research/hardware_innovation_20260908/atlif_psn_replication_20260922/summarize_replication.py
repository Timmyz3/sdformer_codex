"""Paired seed summaries; full valid825 remains the primary outcome."""
import json
from pathlib import Path
from statistics import mean, median

SENSITIVE_FRAME = 'zurich_city_09_a_0541.npy'


def summarize(directory):
    training = json.loads((directory / 'RECOVERY.json').read_text())
    paired = json.loads((directory / 'PAIRED_VALID825.json').read_text())
    assert training['complete'] and paired['complete']
    pairs = [tuple(files) for files in training['train_files']]
    assert len(pairs) == 256 and all(len(pair) == 2 for pair in pairs)
    assert len({name for pair in pairs for name in pair}) == 512
    assert training['trainable_parameters'] == 5247
    assert {a['rule'] for a in training['arms']} == {'official_eps', 'normalized'}
    assert all(a['successful_updates'] == len(a['updates']) == 256 for a in training['arms'])
    reference = paired['arms']['ep34']['rows']
    assert len(reference) == 825 and len({r['file'] for r in reference}) == 825
    for arm in paired['arms'].values():
        assert [(r['file'], r['pixels']) for r in arm['rows']] == [(r['file'], r['pixels']) for r in reference]
    report = dict(seed=training['seed'], frames=825, pixels=sum(r['pixels'] for r in reference),
                  training_pairs=len(pairs), unique_training_frames=len({n for p in pairs for n in p}),
                  primary='normalized minus official_eps; negative AEE difference means improvement',
                  metrics={name: {k: arm[k] for k in ['aee_frame_mean', 'aee_pixel_weighted',
                           'element_weighted_firing', 'zero_firing_modules_observed']}
                           for name, arm in paired['arms'].items()}, comparisons={},
                  training=[{k: arm[k] for k in ['rule', 'successful_updates', 'overflow_attempts',
                            'training_seconds_including_diagnostics', 'peak_allocated_gib_including_diagnostics',
                            'peak_reserved_gib_including_diagnostics']} for arm in training['arms']])
    for candidate, baseline in [('normalized', 'official_eps'), ('official_eps', 'ep34'), ('normalized', 'ep34')]:
        a = paired['arms'][candidate]['rows']
        b = paired['arms'][baseline]['rows']
        deltas = [x['aee'] - y['aee'] for x, y in zip(a, b)]
        sequences = {}
        rest = []
        sensitive = None
        for row, delta in zip(a, deltas):
            name = Path(row['file']).name
            sequences.setdefault(name.rsplit('_', 1)[0], []).append(delta)
            if name == SENSITIVE_FRAME:
                sensitive = delta
            else:
                rest.append(delta)
        assert sensitive is not None and len(rest) == 824
        report['comparisons'][candidate + '_minus_' + baseline] = dict(
            frame_mean_delta=mean(deltas), median_frame_delta=median(deltas),
            improved_frames=sum(d < 0 for d in deltas), worsened_frames=sum(d > 0 for d in deltas),
            unchanged_frames=sum(d == 0 for d in deltas),
            known_sensitive_frame=SENSITIVE_FRAME, known_sensitive_frame_delta=sensitive,
            mean_delta_without_known_sensitive_frame=mean(rest),
            sequence_mean_deltas={n: mean(v) for n, v in sorted(sequences.items())},
            improved_sequences=sum(mean(v) < 0 for v in sequences.values()),
            worsened_sequences=sum(mean(v) > 0 for v in sequences.values()))
    (directory / 'SUMMARY.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report, reference


def combine(root, seeds):
    summaries, references = zip(*(summarize(root / f'seed_{s}') for s in seeds))
    training = [json.loads((root / f'seed_{s}' / 'RECOVERY.json').read_text()) for s in seeds]
    pairs = [sorted(tuple(p) for p in t['train_files']) for t in training]
    assert all(p == pairs[0] for p in pairs[1:])
    first = references[0]
    assert all([(r['file'], r['pixels']) for r in rows] == [(r['file'], r['pixels']) for r in first]
               for rows in references[1:])
    drift = max(abs(r['aee'] - b['aee']) for rows in references[1:] for r, b in zip(rows, first))
    report = dict(complete=True, seeds=seeds, scope='Fixed 512-frame set and 256 updates per arm; two stochastic replicates',
                  baseline_max_frame_aee_difference_between_seeds=drift,
                  baseline_consistency='exact_AEE_match' if drift == 0 else 'investigate_before_pooling_seeds',
                  results=list(summaries),
                  normalized_better_than_official_in_all_seeds=all(
                      s['comparisons']['normalized_minus_official_eps']['frame_mean_delta'] < 0 for s in summaries),
                  normalized_better_outside_known_frame_in_all_seeds=all(
                      s['comparisons']['normalized_minus_official_eps']['mean_delta_without_known_sensitive_frame'] < 0
                      for s in summaries))
    (root / 'REPLICATION_SUMMARY.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    print(json.dumps(combine(Path(__file__).resolve().parent, [20260922, 20260923]), indent=2))
