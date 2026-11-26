#!/usr/bin/env python3
"""Post-supplement descriptive k1 replay; no forward, fit, or score search."""
import json
from pathlib import Path

from analyze_interventions import BASE, arm, reg, sha, summarize


def main():
    root = Path(__file__).resolve().parent
    pins = {
        'probe_v1/results.json': 'bfcf633fb09c05520a252e47f57c4b38060ad50f6fba0f7f1b89e06481f2bf5a',
        'supplement_v1/results.json': '9e8bb5e450126f66521232cfe8f6bcc2f981b0a8d7c26b91b7a77878b562f81e',
        'analyze_interventions.py': '0974218298cfb6c7933b85b35d82da0119aabeab3421360a258f417cf256348a',
    }
    for filename, digest in pins.items():
        if sha(root / filename) != digest:
            raise ValueError('Source SHA differs: ' + filename)
    parent = json.loads((root / 'probe_v1/results.json').read_text())
    supplement = json.loads((root / 'supplement_v1/results.json').read_text())
    previous = json.loads((root / 'analysis_v1/analysis.json').read_text())
    frames = parent['frames']
    seqs = sorted({f['sequence'] for f in frames})
    mappings = {
        'k2_tp_official_sdt_tile': None,
        'k2_psn1_activity': 'psn1_activity',
        'k2_psn1_spatial_hamming': 'psn1_spatial_neighbor_hamming',
    }
    results = {}
    for key, old_key in mappings.items():
        choices = {}
        for frame in supplement['frames']:
            scores = frame['scores'][key]['unit_scores']
            chosen = min(scores, key=lambda x: (-x['score'], tuple(x['tile'])))
            choices[frame['target_id']] = arm(tuple(chosen['tile']))
        if set(choices) != {f['target_id'] for f in frames}:
            raise ValueError('Frame membership mismatch')
        entry = {
            'regions': summarize(frames, choices),
            'per_sequence': {
                seq: summarize([f for f in frames if f['sequence'] == seq], choices)
                for seq in seqs
            },
            'selected_arms': choices,
            'all_GT_wins_ties_losses_vs_coarse': {
                'wins': sum(reg(f, choices[f['target_id']])['AEE'] < reg(f, BASE)['AEE'] for f in frames),
                'ties': sum(reg(f, choices[f['target_id']])['AEE'] == reg(f, BASE)['AEE'] for f in frames),
                'losses': sum(reg(f, choices[f['target_id']])['AEE'] > reg(f, BASE)['AEE'] for f in frames),
            },
        }
        if old_key is not None and entry != previous['k1_summaries'][old_key]:
            raise ValueError('Supplement PSN replay differs from frozen original k1: ' + old_key)
        results[key.replace('k2_', 'k1_', 1)] = entry
    output = {
        'schema': 'flow_supplement_k1_descriptive_replay_v1',
        'scope': 'Post-supplement descriptive replay, after seeing k2 results; not a preregistered new experiment. Fixed score descending and lexicographic tie choose an already executed singleton. No new model forward, fit, hyperparameter or score search. Reused dev8 is not independent confirmation.',
        'source_sha256': pins,
        'previous_analysis_sha256': sha(root / 'analysis_v1/analysis.json'),
        'script_sha256': sha(__file__),
        'new_full_forward_count': 0,
        'PSN_controls_exactly_match_frozen_k1_analysis': True,
        'summaries': results,
        'cost_boundary': 'Equal restored producer service only; scorer cost differs and has not been converted into measured net cycles.',
    }
    target = root / 'supplement_k1_replay.json'
    if target.exists():
        raise ValueError('Refuse to overwrite existing descriptive result')
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: {r: v['regions'][r]['frame_mean_AEE'] for r in ('all_valid', 'B', 'L')}
                      for k, v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
