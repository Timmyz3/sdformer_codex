#!/usr/bin/env python3
"""Reproduce the post-result all-member decomposition; never fit or select."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'post_result_diagnostics.json'
OUTPUT = HERE / 'damage_partition_diagnostic.json'
PARTS = ('B_and_L', 'B_only', 'L_only_B_known', 'neither_B_known', 'B_unknown')


def main():
    raw = json.loads(SOURCE.read_text())
    frames = []
    for frame in raw['per_frame_anchors']:
        native, coarse = frame['native_regions'], frame['coarse_regions']
        count = native['all_valid']['valid_pixels']
        assert count > 0 and count == coarse['all_valid']['valid_pixels']
        for regions in (native, coarse):
            assert sum(regions[key]['valid_pixels'] for key in PARTS) == count
        assert all(native[key]['valid_pixels'] == coarse[key]['valid_pixels'] for key in native)
        contributions = {
            key: (coarse[key]['error_sum'] - native[key]['error_sum']) / count
            for key in PARTS
        }
        delta = coarse['all_valid']['AEE'] - native['all_valid']['AEE']
        assert abs(sum(contributions.values()) - delta) < 1e-10
        frames.append({
            'target_id': frame['target_id'], 'sequence': frame['sequence'],
            'allGT_AEE_change': delta,
            'disjoint_region_contribution_to_frame_AEE_change': contributions,
            'all_regions': {key: {
                'native_AEE': native[key]['AEE'],
                'coarse_AEE': coarse[key]['AEE'],
                'valid_pixels': native[key]['valid_pixels'],
            } for key in native},
        })
    assert len(frames) == 8 and len({f['target_id'] for f in frames}) == 8
    result = {
        'schema': 'flow_round6_post_result_partition_diagnostic_v1',
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'scope': 'Result-triggered descriptive decomposition of all8 paired frames; disjoint regions sum to global AEE delta. No selection, fitting or causal attribution. Boundaries with unknown status stay separate.',
        'frames': frames,
        'frame_equal_global_AEE_contributions': {
            key: sum(f['disjoint_region_contribution_to_frame_AEE_change'][key] for f in frames) / len(frames)
            for key in PARTS
        },
        'total_frame_equal_AEE_delta': sum(f['allGT_AEE_change'] for f in frames) / len(frames),
    }
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text())
        # The initial one-off diagnostic used NumPy mean, whose reduction
        # order can differ at the last bit from this stdlib-only sum.
        for key in ('schema', 'source_sha256', 'scope', 'frames'):
            assert existing[key] == result[key], key
        for key in PARTS:
            assert abs(existing['frame_equal_global_AEE_contributions'][key] - result['frame_equal_global_AEE_contributions'][key]) < 1e-14
        assert abs(existing['total_frame_equal_AEE_delta'] - result['total_frame_equal_AEE_delta']) < 1e-14
        print('PASS: existing diagnostic reproduced; original artifact unchanged')
    else:
        OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print('Wrote', OUTPUT.name)


if __name__ == '__main__':
    main()
