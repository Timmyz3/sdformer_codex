#!/usr/bin/env python3
"""R13 identity-only split. Standard library; never opens arrays or quality files."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SCHEMA = 'flow_round13_coverage_identity_v1'
MEMBER_SEED = 'flow-r13-coverage-members-20261016:'
CHECK_SEED = 'flow-r13-coverage-check-20261016:'
CHECKPOINT = 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
PINS = {
    'canonical': ('research/flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json', '6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22'),
    'roles': ('research/flow_story_20261011/preflight/data_protocol/artifacts/train_auxiliary_roles.csv', '74fea9aeaeb94cd6399273bf508494989e7c191f8cc4d22ffd9b72c96da4834f'),
    'r12': ('research/flow_story_20261016/ordinary_recovery_round12/check_membership_v1.json', 'cfc84c52158213599717d1aaafb19c73506ff08edbf54831d2b839bfa9da75d5'),
    'r11': ('research/flow_story_20261016/joint_budget_round11/membership_v1.json', 'f1f893714d3d4428f36e029a5ea0cc601ddf948241d3b34668c0b35b887b85c2'),
    'r10': ('research/flow_story_20261015/block2_allocation_round10/membership_v1.json', '28c6dc4ffb2197301c8d8a763e65fa45b6cf753768b15d45750de49e3228a9df'),
    'r9': ('research/flow_story_20261015/fixed_layer_confirmation_round9/membership_v1.json', '794ae30b0783d2e9d17f5bafd029f5bb6bcbd3fb1993da51ad408ecf778c8443'),
    'r6': ('research/flow_story_20261014/damage_predictor_round6/selection_manifest_v1.json', 'c20684f58702034887f6454073f993b6309182d99bea763398fa16ba39b6541a'),
    'fit20': ('research/flow_story_20261013/recovery/manifests/svs_fit20.json', '29ec7ba9c9faca74c62c3e170c9d3c70855a4605a7eab31f9331c519f56f8915'),
    'check64': ('research/flow_story_20261013/recovery/manifests/check64.json', '310d7c55fe5dc625d7b974bd03482513d46227254105501c088c17e580997a92'),
    'dev32': ('research/flow_story_20261013/breadth/gate_dev32.json', '41309684f927a10ef24b479178ec73a473da9d2ee31880011646f5f682d80509'),
}


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_text(text):
    return digest_bytes(text.encode('utf-8'))


def digest_file(path):
    return digest_bytes(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordered_ids(ids, seed):
    return sorted(ids, key=lambda tid: (digest_text(seed + tid), tid))


def sequence(tid):
    return tid.rsplit('_', 1)[0]


def index(tid):
    return int(tid.rsplit('_', 1)[1])


def nearest(tid, others):
    candidates = [other for other in others if sequence(other) == sequence(tid) and other != tid]
    if not candidates:
        return {'minimum_target_index_gap': None, 'nearest_ids': []}
    distance = min(abs(index(tid) - index(other)) for other in candidates)
    return {'minimum_target_index_gap': distance,
            'nearest_ids': sorted(other for other in candidates if abs(index(tid) - index(other)) == distance)}


def build():
    loaded = {}
    for key, (relative, sha) in PINS.items():
        path = ROOT / relative
        require(digest_file(path) == sha, 'Pinned source changed: ' + key)
        if key == 'roles':
            with path.open(newline='') as stream:
                loaded[key] = list(csv.DictReader(stream))
        else:
            loaded[key] = json.loads(path.read_text())
    canonical_rows = loaded['canonical']['records']
    canonical = {row['target_id']: row for row in canonical_rows}
    require(len(canonical) == len(canonical_rows), 'Duplicate canonical target ID')
    role_rows = loaded['roles']
    roles = {row['target_id']: row for row in role_rows}
    require(len(roles) == len(role_rows), 'Duplicate role target ID')
    for tid, row in roles.items():
        require(canonical[tid]['sequence'] == row['sequence'] and canonical[tid]['protocol_split'] == 'train',
                'Role is not canonical train member: ' + tid)
        require(canonical[tid]['of_context_filename'] == row['of_context_filename'] and
                canonical[tid]['v2_target_filename'] == row['target_filename'], 'Canonical filenames differ')

    # Explicit historical audit identity chain. Old fit20 is conservatively excluded too.
    historical_parts = {
        'dev32': set(loaded['dev32']['target_ids']),
        'check64': set(loaded['check64']['target_ids']),
        'fit20': set(loaded['fit20']['target_ids']),
        'r6_check': set(loaded['r6']['groups']['check']),
        'r9_check': set(loaded['r9']['target_ids']),
        'r10_check': set(loaded['r10']['groups']['check']),
        'r11_check': set(loaded['r11']['groups']['check']),
        'r12_check': set(loaded['r12']['groups']['check']),
    }
    excluded = set().union(*historical_parts.values())
    r12_union = set(loaded['r12']['excluded_ids']) | set(loaded['r12']['groups']['fit']) | set(loaded['r12']['groups']['check'])
    require(excluded == r12_union and len(excluded) == 164, 'Historical exclusion chain differs')
    for key in ('r6', 'r10', 'r11', 'r12'):
        require(set(loaded[key]['groups']['fit']) == historical_parts['fit20'], 'Legacy fit set drifted')
        require(set(loaded[key]['excluded_ids']).issubset(excluded), 'Missing historical exclusion')
    require(set(loaded['r9']['excluded_ids']).issubset(excluded), 'Missing R9 historical exclusion')

    all_sequences = sorted({row['sequence'] for row in role_rows if row['auxiliary_role'] == 'gate_fit'},
                           key=lambda name: (digest_text(name), name))
    require(len(all_sequences) == 10, 'Gate-fit sequence population changed')
    check_sequences, train_sequences = all_sequences[:2], all_sequences[2:]
    pools = {seq: [row['target_id'] for row in role_rows if row['sequence'] == seq
                   and row['auxiliary_role'] == 'gate_fit'] for seq in all_sequences}
    require(sum(map(len, pools.values())) == 4573, 'Gate-fit population changed')
    free_pools = {seq: ordered_ids(set(pool) - excluded, MEMBER_SEED) for seq, pool in pools.items()}
    reference_ids, diversity_ids, selector_ids = [], [], []
    for seq in train_sequences:
        require(len(free_pools[seq]) >= 30, 'Need 25 diversity + 5 selector per train sequence')
        reference_ids.extend(free_pools[seq][:5])
        diversity_ids.extend(free_pools[seq][:25])
        selector_ids.extend(free_pools[seq][25:30])
    check_ids = []
    for seq in check_sequences:
        accepted = []
        for tid in ordered_ids(set(pools[seq]) - excluded, CHECK_SEED):
            if all(abs(index(tid) - index(other)) >= 3 for other in accepted):
                accepted.append(tid)
            if len(accepted) == 4:
                break
        require(len(accepted) == 4, 'Need four spaced new check targets')
        check_ids.extend(accepted)

    # Original BN role/ranking is legal provenance; only eight new training sequences enter.
    bn_by_sequence = {}
    for seq in train_sequences:
        pool = [row for row in role_rows if row['sequence'] == seq and row['auxiliary_role'] == 'bn_calibration'
                and row['target_id'] not in excluded]
        pool.sort(key=lambda row: (row['bn_selection_sha256'], row['target_id']))
        require(len(pool) >= 22, 'Insufficient legal BN members')
        require(all(row['bn_selection_sha256'] for row in pool), 'Missing original BN selection hash')
        bn_by_sequence[seq] = [row['target_id'] for row in pool[:22]]
    bn_ids = [bn_by_sequence[seq][visit] for visit in range(22) for seq in train_sequences]

    groups = {'reference': reference_ids, 'diversity': diversity_ids, 'selector': selector_ids,
              'check': check_ids, 'bn': bn_ids}
    require(set(reference_ids).issubset(diversity_ids), 'Reference is not nested in diversity')
    for name, ids in groups.items():
        require(len(ids) == len(set(ids)), 'Duplicate member in ' + name)
        require(not set(ids) & excluded, 'Historical consumed ID used in ' + name)
    exclusive = ('diversity', 'selector', 'check', 'bn')
    for i, name in enumerate(exclusive):
        for other in exclusive[i+1:]:
            require(not set(groups[name]) & set(groups[other]), 'Cross-role identity leakage')
    require({sequence(tid) for tid in bn_ids}.isdisjoint(check_sequences), 'BN check sequence leakage')

    orders = {'reference': [], 'diversity': []}
    for visit in range(25):
        for seq in train_sequences:
            for condition in orders:
                pos = visit % 5 if condition == 'reference' else visit
                row = dict(canonical[free_pools[seq][pos]])
                row.update(step=len(orders[condition]) + 1, sequence_visit_index=visit + 1)
                orders[condition].append(row)
    require([row['sequence'] for row in orders['reference']] == [row['sequence'] for row in orders['diversity']],
            'Sequence curricula differ')
    require(set(collections.Counter(row['target_id'] for row in orders['reference']).values()) == {5},
            'Reference identity multiplicity differs')
    require(set(collections.Counter(row['target_id'] for row in orders['diversity']).values()) == {1},
            'Diversity identity multiplicity differs')

    initial_bn_ids = {row['target_id'] for row in role_rows if row['bn_initial_member'] == 'True'}
    diagnostics = {
        'historical_parts': {key: sorted(value) for key, value in historical_parts.items()},
        'historical_union_count': len(excluded),
        'sequence_exposure': {seq: {
            'all_backbone_training_members': True,
            'historical_consumed_audit_or_fit_ids': sorted(tid for tid in excluded if sequence(tid) == seq),
            'historical_initial_BN176_members': sorted(tid for tid in initial_bn_ids if sequence(tid) == seq),
            'gate_fit_population': len(pools[seq]),
            'available_after_historical_exclusion': len(free_pools[seq]),
            'new_role': 'check_sequence' if seq in check_sequences else 'training_sequence',
        } for seq in all_sequences},
        'per_selected_member': {
            name: {tid: {
                'nearest_historical_consumed': nearest(tid, excluded),
                'nearest_other_same_group': nearest(tid, ids),
                'nearest_new_BN': nearest(tid, bn_ids),
                'nearest_new_diversity': nearest(tid, diversity_ids),
                'nearest_new_selector': nearest(tid, selector_ids),
            } for tid in ids} for name, ids in groups.items()},
        'distance_scope': 'Target filename index gaps only. Canonical timestamp fields are null; no elapsed-time or temporal-independence claim. No minimum gap imposed between BN/recovery/selector identities; role disjointness is by target identity.',
        'new_check_timestamp_fields_all_null': all(canonical[tid]['target_start_us'] is None and canonical[tid]['target_end_us'] is None for tid in check_ids),
        'original_BN176_overlap_with_new_BN176': len(initial_bn_ids & set(bn_ids)),
        'check_sequence_original_BN176_count': sum(sequence(tid) in check_sequences for tid in initial_bn_ids),
    }
    result = {
        'schema': SCHEMA,
        'protocol_id': 'r13_reference40_vs_diversity200_20261016_v1',
        'checkpoint_sha256': CHECKPOINT,
        'source_paths': {key: pair[0] for key, pair in PINS.items()},
        'source_sha256': {key: pair[1] for key, pair in PINS.items()},
        'selector_source_sha256': digest_file(__file__),
        'split_sequences': {'train': train_sequences, 'check': check_sequences},
        'sequence_sha256_order': [{'sequence': seq, 'sha256': digest_text(seq)} for seq in all_sequences],
        'selection': {
            'sequence_rule': 'Sort ten gate_fit sequence names by (SHA256(sequence UTF8), sequence); first two reserved whole for new-pipeline check.',
            'member_seed': MEMBER_SEED, 'check_seed': CHECK_SEED,
            'training_rule': 'After excluding historical164, SHA256(member_seed+target_id), ID tie per train sequence. First5 reference, first25 diversity, next5 selector reserve.',
            'training_order': '25 round-robin visits over eight train sequences in sequence_sha256_order. Reference cycles first5, diversity visits first25 once. Same sequence at every update.',
            'check_rule': 'Per held sequence SHA256(check_seed+target_id), ID tie, greedy first4 with index gap>=3 within new check group, excluding historical164.',
            'BN_rule': 'Eight train sequences only; original bn_calibration role. Per sequence first22 by original bn_selection_sha256, target_id tie; 22 round-robin visits over train sequences. No GT/labels used.',
            'check_opening': 'One terminal endpoint evaluation after deployment/calibration/training and all compared conditions fixed; same IDs for initial and terminal matched anchors. No selection or tuning from check quality.',
            'data_read_during_selection': 'Only pinned identity/role JSON and CSV; no event, GT, prediction, loss or quality arrays/files.',
        },
        'group_counts': {name: len(ids) for name, ids in groups.items()},
        'excluded_ids': sorted(excluded),
        'train_members_reference': [canonical[tid] for tid in reference_ids],
        'train_members_diversity': [canonical[tid] for tid in diversity_ids],
        'selector_members': [canonical[tid] for tid in selector_ids],
        'check_members': [canonical[tid] for tid in check_ids],
        'bn_members': [canonical[tid] for tid in bn_ids],
        'train_order_reference': orders['reference'],
        'train_order_diversity': orders['diversity'],
        'scope': 'New recovery/BN/selector-pipeline sequence holdout only. Original backbone saw all train sequences; old fit20 and old BN176 used held sequences. Not backbone-unseen, research-unseen, temporally independent or official DSEC test. Reload original checkpoint; do not resume R12 or reuse old BN176.',
        'future_selector_scope': '40 reserved identities not yet executed; no head folds/targets/quality claims frozen by this identity manifest. Separate selector protocol required before data-driven fitting.',
        'new_full_pipeline_BN_requirement': 'Reset all 24 private BN buffers then recalibrate from this legal176 before common initial state; retain the 54 external current-stat contract. Calibration algorithm/geometry/numerics must be separately pinned by runner.',
    }
    return result, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    manifest, diagnostics = build()
    outputs = {'manifest_v1.json': manifest, 'identity_diagnostics_v1.json': diagnostics}
    for filename, value in outputs.items():
        path = HERE / filename
        if args.verify:
            require(json.loads(path.read_text()) == value, 'Frozen output differs: ' + filename)
        else:
            require(not path.exists(), 'Refuse to overwrite: ' + filename)
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'VERIFIED' if args.verify else 'CREATED',
                      'manifest_sha256': digest_file(HERE / 'manifest_v1.json'),
                      'diagnostics_sha256': digest_file(HERE / 'identity_diagnostics_v1.json'),
                      'split_sequences': manifest['split_sequences'],
                      'group_counts': manifest['group_counts'],
                      'check_ids': [row['target_id'] for row in manifest['check_members']]}, indent=2))


if __name__ == '__main__':
    main()
