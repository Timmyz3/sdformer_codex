#!/usr/bin/env python3
"""Descriptive complete-run resource/loss accounting; no model or quality selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert len(args.result) == 4 and not args.output.exists()
    arms = {}
    for path in args.result:
        raw = json.loads(path.read_text())
        assert raw['status'] == 'ROUND12_FAIR_RECOVERY_COMPLETE'
        assert raw['completed_optimizer_steps'] == len(raw['steps']) == 200
        assert raw['arm'] not in arms
        assert [s['target_id'] for s in raw['steps']] == raw['binding']['fit_ids'] * 10
        means = [sum(s['task_loss'] for s in raw['steps'][i:i+20])/20 for i in range(0, 200, 20)]
        assert all(math.isfinite(x) for x in means)
        arms[raw['arm']] = {
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'training_seconds': raw['training_seconds'],
            'entry_wall_seconds': raw['wall_seconds'],
            'full_entry_peak_bytes': raw['full_entry_peak_bytes'],
            'train_peak_allocated_bytes': raw['max_allocated_bytes'],
            'train_peak_reserved_bytes': raw['max_reserved_bytes'],
            'sum_synchronized_step_seconds': sum(s['synchronized_train_step_seconds'] for s in raw['steps']),
            'fit20_ordered_pass_mean_training_loss': means,
            'runtime': raw['runtime'],
        }
    assert set(arms) == {'native', 'FF', 'CC', 'fine_s2b2'}
    result = {
        'schema': 'r12_training_resources_descriptive_v1',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'arms': arms,
        'sum_training_seconds': sum(x['training_seconds'] for x in arms.values()),
        'sum_entry_wall_seconds': sum(x['entry_wall_seconds'] for x in arms.values()),
        'max_full_entry_reserved_bytes': max(x['full_entry_peak_bytes']['reserved'] for x in arms.values()),
        'scope': 'All four fixed endpoints. Wall time excludes inter-arm transfer and local audits. Ten pass means use the same ordered fit20, but changing parameter states; not terminal fit AEE, held-out quality, convergence proof, significance or deployment latency. No tuning or new gate.',
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('sum_training_seconds', 'sum_entry_wall_seconds', 'max_full_entry_reserved_bytes')}))


if __name__ == '__main__':
    main()
