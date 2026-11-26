#!/usr/bin/env python3
"""Independent R13 metadata audit; does not import the selection implementation."""
import collections
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, label):
    if not ok:
        raise ValueError(label)


def rank(ids, seed=''):
    return sorted(ids, key=lambda item: (hashlib.sha256((seed + item).encode()).hexdigest(), item))


def nearest_index(tid, pool):
    seq, idx = tid.rsplit('_', 1)
    other = [int(x.rsplit('_', 1)[1]) for x in pool if x.rsplit('_', 1)[0] == seq and x != tid]
    return min(abs(int(idx) - val) for val in other) if other else None


def main():
    path = HERE / 'manifest_v1.json'
    m = json.loads(path.read_text())
    source = {}
    for key, relative in m['source_paths'].items():
        p = ROOT / relative
        require(sha(p) == m['source_sha256'][key], 'Source digest ' + key)
        source[key] = p
    with source['roles'].open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    roles = {r['target_id']: r for r in rows}
    canonical = {r['target_id']: r for r in json.loads(source['canonical'].read_text())['records']}
    require(m['schema'] == 'flow_round13_coverage_identity_v1', 'Schema')
    require(m['checkpoint_sha256'] == 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30', 'Checkpoint')
    require(sha(HERE / 'build_manifest.py') == m['selector_source_sha256'], 'Builder source')
    hist = {k: json.loads(v.read_text()) for k, v in source.items() if k not in ('roles', 'canonical')}
    historical = set()
    for key in ('fit20', 'dev32', 'check64', 'r9'):
        historical.update(hist[key]['target_ids'])
    for key in ('r6', 'r10', 'r11', 'r12'):
        historical.update(hist[key]['groups']['check'])
        historical.update(hist[key]['groups']['fit'])
    require(historical == set(m['excluded_ids']) and len(historical) == 164, 'Historical exclusion union')
    seqs = rank({r['sequence'] for r in rows if r['auxiliary_role'] == 'gate_fit'})
    require(m['split_sequences'] == {'check': seqs[:2], 'train': seqs[2:]}, 'Sequence SHA split')
    train, check = seqs[2:], seqs[:2]
    fields = ['train_members_reference', 'train_members_diversity', 'selector_members', 'check_members', 'bn_members']
    ids = {}
    for field in fields:
        ids[field] = [r['target_id'] for r in m[field]]
        require(len(ids[field]) == len(set(ids[field])), 'Identity duplication ' + field)
        require(not set(ids[field]) & historical, 'Historical leakage ' + field)
        for r in m[field]:
            require(r == canonical[r['target_id']], 'Noncanonical row ' + field)
            require(r['protocol_split'] == 'train', 'Nontrain row')
            expected_role = 'bn_calibration' if field == 'bn_members' else 'gate_fit'
            require(roles[r['target_id']]['auxiliary_role'] == expected_role, 'Auxiliary role')
            require(r['sequence'] in (check if field == 'check_members' else train), 'Sequence leakage ' + field)
    require([len(m[k]) for k in fields] == [40, 200, 40, 8, 176], 'Group counts')
    for i, f in enumerate(fields[1:]):
        for g in fields[1:][i+1:]:
            require(not set(ids[f]) & set(ids[g]), 'Role overlap')
    require(set(ids[fields[0]]) < set(ids[fields[1]]), 'Nested reference')
    train_pools = {}
    for seq in train:
        pool = [r['target_id'] for r in rows if r['sequence'] == seq and r['auxiliary_role'] == 'gate_fit'
                and r['target_id'] not in historical]
        train_pools[seq] = rank(pool, 'flow-r13-coverage-members-20261016:')
        for field, a, b in ((fields[0], 0, 5), (fields[1], 0, 25), (fields[2], 25, 30)):
            require([r['target_id'] for r in m[field] if r['sequence'] == seq] == train_pools[seq][a:b], 'Within-sequence selection')
    bn_ids = {}
    for seq in train:
        bn = [r for r in rows if r['sequence'] == seq and r['auxiliary_role'] == 'bn_calibration'
              and r['target_id'] not in historical]
        bn.sort(key=lambda r: (r['bn_selection_sha256'], r['target_id']))
        bn_ids[seq] = [r['target_id'] for r in bn[:22]]
    require(ids['bn_members'] == [bn_ids[s][i] for i in range(22) for s in train], 'BN selection/order')
    for seq in check:
        available = [r['target_id'] for r in rows if r['sequence'] == seq and r['auxiliary_role'] == 'gate_fit'
                     and r['target_id'] not in historical]
        selected = []
        for tid in rank(available, 'flow-r13-coverage-check-20261016:'):
            if not selected or nearest_index(tid, selected) >= 3:
                selected.append(tid)
            if len(selected) == 4:
                break
        require(selected == [r['target_id'] for r in m['check_members'] if r['sequence'] == seq], 'Check selection')
    for condition in ('reference', 'diversity'):
        order = m['train_order_' + condition]
        require(len(order) == 200, 'Training step count')
        for step, row in enumerate(order):
            base = dict(row)
            require(base.pop('step') == step + 1, 'Step index')
            visit = base.pop('sequence_visit_index')
            require(visit == step // 8 + 1, 'Sequence visit index')
            seq = train[step % 8]
            member = train_pools[seq][(visit - 1) % 5 if condition == 'reference' else visit - 1]
            require(base == canonical[member], 'Exact paired sequence/member curriculum')
        multiplicities = collections.Counter(r['target_id'] for r in order)
        require(set(multiplicities.values()) == ({5} if condition == 'reference' else {1}), 'Update multiplicity')
    old_bn = [r['target_id'] for r in rows if r['bn_initial_member'] == 'True']
    result = {
        'schema': 'flow_round13_independent_metadata_audit_v1',
        'status': 'PASS', 'manifest_sha256': sha(path), 'auditor_sha256': sha(__file__),
        'source_count': len(source), 'historical_excluded_count': len(historical),
        'complete_canonical_row_comparisons': sum(len(m[k]) for k in fields) + 400,
        'reference40_diversity200_sequence_visits_identical': True,
        'check_sequences_excluded_from_new_BN_recovery_selector': True,
        'BN_recovery_selector_check_target_identities_disjoint': True,
        'BN176_old_overlap': len(set(old_bn) & set(ids['bn_members'])),
        'check_nearest_historical_index_gaps': {tid: nearest_index(tid, historical) for tid in ids['check_members']},
        'check_nearest_old_BN176_index_gaps': {tid: nearest_index(tid, old_bn) for tid in ids['check_members']},
        'check_nearest_new_check_index_gaps': {tid: nearest_index(tid, ids['check_members']) for tid in ids['check_members']},
        'limitations': [
            'This independent code path was written by the same data-protocol agent; it is not an external review.',
            'No arrays, losses or predictions were accessed; content hashes and availability remain runner preflight work.',
            'Original backbone saw all train. Prior fit20 and BN176 used the new check sequences.',
            'Index gaps do not measure elapsed time. New training roles have adjacent indices and no temporal purge.',
            'R12 weights/old CMA176 may not initialize the new deployment state.',
        ],
    }
    out = HERE / 'metadata_audit_v1.json'
    require(not out.exists(), 'Refuse audit overwrite')
    out.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
