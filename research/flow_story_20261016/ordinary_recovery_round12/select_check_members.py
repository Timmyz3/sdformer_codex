#!/usr/bin/env python3
"""Identity-only R12 selection, including all R11 members in exclusions."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEED = 'flow-r12-fair-recovery-check4-per-sequence-20261016:'
SOURCES = {
    'r11': ('research/flow_story_20261016/joint_budget_round11/membership_v1.json',
            'f1f893714d3d4428f36e029a5ea0cc601ddf948241d3b34668c0b35b887b85c2'),
    'canonical': ('research/flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json',
                  '6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22'),
    'roles': ('research/flow_story_20261011/preflight/data_protocol/artifacts/train_auxiliary_roles.csv',
              '74fea9aeaeb94cd6399273bf508494989e7c191f8cc4d22ffd9b72c96da4834f')}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    paths = {}
    for key, (path, digest) in SOURCES.items():
        paths[key] = ROOT/path
        assert sha(paths[key]) == digest, key
    prior = json.loads(paths['r11'].read_text())
    excluded = set(prior['excluded_ids']) | set(prior['groups']['check']) | set(prior['groups']['fit'])
    rows = json.loads(paths['canonical'].read_text())['records']
    with paths['roles'].open(newline='') as f:
        roles = {r['target_id']:r['auxiliary_role'] for r in csv.DictReader(f)}
    selected = []
    distances = {}
    for sequence in ('zurich_city_01_a', 'zurich_city_05_a'):
        pool = [r['target_id'] for r in rows if r['sequence']==sequence and r['protocol_split']=='train'
                and roles.get(r['target_id'])=='gate_development' and r['target_id'] not in excluded]
        ordered = sorted(pool, key=lambda tid:(hashlib.sha256((SEED+tid).encode()).hexdigest(), tid))
        accepted = []
        for tid in ordered:
            idx = int(tid.rsplit('_', 1)[1])
            if all(abs(idx-int(x.rsplit('_', 1)[1]))>=3 for x in accepted):
                accepted.append(tid)
            if len(accepted)==4:
                break
        assert len(accepted)==4
        old = [int(x.rsplit('_', 1)[1]) for x in excluded if x.rsplit('_', 1)[0]==sequence]
        distances.update({tid:min(abs(int(tid.rsplit('_', 1)[1])-x) for x in old) for tid in accepted})
        selected.extend(accepted)
    assert len(selected)==len(set(selected))==8 and not set(selected)&excluded
    result = {'schema':'flow_round12_fair_recovery_identity_selection_v1',
              'created_utc':datetime.now(timezone.utc).isoformat(), 'seed':SEED,
              'groups':{'fit':prior['groups']['fit'], 'check':selected},
              'excluded_ids':sorted(excluded), 'minimum_within_check_sequence_ID_gap':3,
              'nearest_excluded_ID_distance':distances,
              'source_paths_and_sha256':SOURCES, 'selector_sha256':sha(Path(__file__)),
              'rule':'SHA256(seed+ID), ID tie; first4/sequence satisfying within-batch gap>=3; no arrays/GT/quality read.',
              'scope':'8 identity-new train-side members in two historically exposed, backbone-seen sequences. No independent time/sequence or official-test claim.'}
    out = HERE/'check_membership_v1.json'
    assert not out.exists(), 'Refuse overwrite'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'check':selected, 'excluded':len(excluded), 'sha256':sha(out), 'distances':distances}, indent=2))


if __name__=='__main__':
    main()
