#!/usr/bin/env python3
"""Freeze train-only target and stratified legal MLP-tile sampling; no data/model IO."""
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from flow_eval_contract import canonical_rows, auxiliary_roles, CANONICAL_SHA, AUX_ROLES_SHA, sha


def order_key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    old = here.parents[1] / 'flow_story_20261011/preflight/data_protocol/artifacts'
    records = canonical_rows(old / 'common_targets.json')
    roles = auxiliary_roles(old / 'train_auxiliary_roles.csv', records)
    sequences = {}
    for row in records:
        if roles.get(row['target_id']) == 'gate_fit':
            sequences.setdefault(row['sequence'], []).append(row['target_id'])
    for seq in sequences:
        sequences[seq].sort(key=lambda target: (order_key('flow-a-label-v1:' + target), target))
    expanded = [sequences[seq][rank] for rank in range(16) for seq in sorted(sequences) if rank < len(sequences[seq])]
    initial = [sequences[seq][rank] for rank in range(4) for seq in sorted(sequences) if rank < len(sequences[seq])]
    assert expanded[:len(initial)] == initial
    configs = [(120, 160, 96, 2), (60, 80, 192, 2), (30, 40, 384, 6), (15, 20, 768, 2)]
    descriptors = []
    frame_coverage = []
    for target in expanded:
        selected = []
        for stage, (height, width, channels, blocks) in enumerate(configs):
            grid_h, grid_w = (height + 8) // 9, (width + 8) // 9
            for block in range(blocks):
                strata = {'interior_full': [], 'image_edge_or_partial': []}
                for ty in range(grid_h):
                    for tx in range(grid_w):
                        y0, x0 = ty * 9, tx * 9
                        h, w = min(9, height - y0), min(9, width - x0)
                        kind = 'image_edge_or_partial' if (ty in (0, grid_h - 1) or tx in (0, grid_w - 1) or h != 9 or w != 9) else 'interior_full'
                        unit = {'stage': stage, 'block': block, 'tile_y': ty, 'tile_x': tx,
                                'y0': y0, 'x0': x0, 'height': h, 'width': w, 'T': 10, 'C': channels,
                                'feature_shape': [10, 1, channels, height, width], 'stratum': kind,
                                'unit_id': 's%d.b%d.y%d.x%d' % (stage, block, ty, tx),
                                'candidate_PE_service_cycles': None,
                                'service_status': 'must bind actual downstream execution-contract trace; tile count is not cost'}
                        strata[kind].append(unit)
                for kind, candidates in strata.items():
                    candidates.sort(key=lambda unit: order_key('flow-a-tile-v1:' + target + ':' + unit['unit_id']))
                    picked = candidates[:2]
                    selected.extend(picked)
                    frame_coverage.append({'target_id': target, 'stage': stage, 'block': block,
                                           'stratum': kind, 'candidate_units': len(candidates), 'selected_units': len(picked)})
        descriptors.append({'target_id': target, 'units': selected, 'dense_reference_forwards': 1,
                            'single_unit_forwards': len(selected), 'combination_forward_budget': 8})
    initial_set = set(initial)
    initial_forward = sum(1 + len(r['units']) + 8 for r in descriptors if r['target_id'] in initial_set)
    expanded_forward = sum(1 + len(r['units']) + 8 for r in descriptors)
    report = {'schema': 'flow_a_label_sampling_plan_v1', 'status': 'PLAN_ONLY_NOT_EXECUTED',
              'canonical_manifest_sha256': CANONICAL_SHA, 'auxiliary_roles_sha256': AUX_ROLES_SHA,
              'geometry': 'full9', 'legal_unit': 'MLP-only spatial9x9 times complete T10; partial edge extent explicit',
              'target_sampling': 'Per gate-fit sequence target SHA rank; round-robin(rank,sequence), first4 then first16',
              'sequence_count': len(sequences), 'initial_target_ids': initial, 'expanded_target_ids': expanded,
              'initial_is_expanded_execution_prefix': True,
              'initial_target_count': len(initial), 'expanded_target_count': len(expanded),
              'initial_actual_forward_upper_bound': initial_forward,
              'expanded_cumulative_forward_upper_bound': expanded_forward,
              'coarse_contract_bounds': {'initial': 2280, 'expanded': 9120},
              'reduced_count_reason': 'Stage3 2x3 tile grid has no interior-full stratum; do not duplicate edge units to fill slots.',
              'combination_masks_status': '8 per target budgeted, not instantiated until PE-service weights and gate-state/checkpoint hashes are bound',
              'checkpoint_sha256': None, 'D_statistics_sha256': None, 'region_contract_sha256': None,
              'data_sha256': None, 'wall_seconds_per_forward': None,
              'sampled_units': descriptors, 'stratum_coverage': frame_coverage,
              'script_sha256': sha(__file__)}
    path = here / 'label_sampling_plan.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'path': str(path), 'initial_targets': len(initial), 'expanded_targets': len(expanded),
                      'initial_actual_forward_bound': initial_forward, 'expanded_cumulative_forward_bound': expanded_forward,
                      'sha256': sha(path)}))


if __name__ == '__main__': main()
