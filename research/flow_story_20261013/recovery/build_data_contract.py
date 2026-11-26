#!/usr/bin/env python3
"""Freeze recovery identities from train-only metadata. Never loads data arrays."""
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANONICAL_SHA = '6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22'
ROLES_SHA = '74fea9aeaeb94cd6399273bf508494989e7c191f8cc4d22ffd9b72c96da4834f'
FIT2_SHA = '9a908ea2da02812c771ff42e9a04f96e14388f27526a73660e4e6abb489e4f6f'
DEV32_SHA = '41309684f927a10ef24b479178ec73a473da9d2ee31880011646f5f682d80509'
CHECKPOINT_SHA = 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
FIT_SEED = 'flow-a-svs-gate-fit2-20261013:'
CHECK_SEED = 'flow-recovery-check64-v1-20261013:'
ORDER_SEED = 'flow-recovery-order-v1-20261013:'
AUGMENTATION = {'seed_prefix': 'flow-recovery-augmentation-v1:',
    'formula': "int(SHA256((seed_prefix+target_id).encode('utf-8')).hexdigest()[:16],16) % 2**63",
    'independent_of': ['arm', 'resume segment', 'worker assignment'],
    'scope': 'one-pass per-target augmentation; same input random stream across arms'}


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Input metadata SHA differs: '+str(path))
    return raw


def order_key(seed, target):
    return hashlib.sha256((seed+target).encode('utf-8')).hexdigest(), target


def make_payloads(canonical, roles, fit2, dev32):
    # Canonical is hashed for binding only: no final-valid record or GT is
    # parsed. All actual member selection comes from the train-only roles CSV.
    pinned(canonical, CANONICAL_SHA)
    rows = list(csv.DictReader(io.StringIO(pinned(roles, ROLES_SHA).decode('utf-8'))))
    old_fit = json.loads(pinned(fit2, FIT2_SHA)); old_dev = json.loads(pinned(dev32, DEV32_SHA))
    require(len(rows) == 6000 and len({r['target_id'] for r in rows}) == 6000, 'Unique train6000 roles required')
    require(all(r['backbone_training_member'] == 'True' for r in rows), 'Non-training member in auxiliary roles')
    by_id = {r['target_id']: r for r in rows}
    roles_by_name = {k: [r for r in rows if r['auxiliary_role'] == k]
                     for k in ('gate_fit', 'gate_development', 'bn_calibration')}
    require({k:len(v) for k,v in roles_by_name.items()} ==
            {'gate_fit':4573, 'gate_development':724, 'bn_calibration':703}, 'Role counts differ')
    for old in (old_fit, old_dev):
        require(old['canonical_sha256'] == CANONICAL_SHA and old['roles_sha256'] == ROLES_SHA, 'Old manifest binding differs')
    require(old_fit['seed'] == FIT_SEED and len(old_fit['target_ids']) == 2, 'Original SVS fit rule differs')
    legacy_ids_sha = hashlib.sha256(('\n'.join(old_dev['target_ids'])+'\n').encode('utf-8')).hexdigest()
    require(len(old_dev['target_ids']) == len(set(old_dev['target_ids'])) == 32 and
            old_dev['target_ids_sha256'] == legacy_ids_sha, 'Old dev32 membership differs')
    require(all(by_id[t]['auxiliary_role'] == 'gate_development' for t in old_dev['target_ids']), 'Old dev roles differ')
    ordered_fit = sorted(roles_by_name['gate_fit'], key=lambda r: order_key(FIT_SEED, r['target_id']))
    original = []; seen = set()
    for r in ordered_fit:
        if r['sequence'] not in seen:
            seen.add(r['sequence']); original.append(r['target_id'])
        if len(original) == 2: break
    require(original == old_fit['target_ids'], 'Cannot reconstruct original fit2')
    selected = []; counts = collections.Counter()
    for r in ordered_fit:
        if counts[r['sequence']] < 2:
            selected.append(r); counts[r['sequence']] += 1
    require(len(selected) == 20 and len(counts) == 10 and set(counts.values()) == {2}, 'Expected ten gate-fit sequences, two targets each')
    require(set(original) <= {r['target_id'] for r in selected}, 'fit2 is not nested in fit20')
    remaining = [r for r in roles_by_name['gate_development'] if r['target_id'] not in set(old_dev['target_ids'])]
    dev_counts = collections.Counter(r['sequence'] for r in remaining)
    require(dict(dev_counts) == {'zurich_city_01_a':89,'zurich_city_05_a':603}, 'Dev remainder changed')
    checks = []
    for sequence in sorted(dev_counts):
        pool = sorted((r for r in remaining if r['sequence'] == sequence), key=lambda r: order_key(CHECK_SEED,r['target_id']))
        checks.extend(pool[:32])
    checks.sort(key=lambda r: order_key(CHECK_SEED,r['target_id']))
    recovery = sorted(roles_by_name['gate_fit'], key=lambda r: order_key(ORDER_SEED,r['target_id']))
    source = {'canonical_sha256':CANONICAL_SHA,'roles_sha256':ROLES_SHA,
              'old_fit2_sha256':FIT2_SHA,'old_dev32_sha256':DEV32_SHA}

    def manifest(schema, population, items, seed, rule):
        ids = [r['target_id'] for r in items]
        return {'schema':schema, 'population':population, 'count':len(ids), 'target_ids':ids,
            'target_ids_sha256':digest(ids), 'sequence_counts':dict(sorted(collections.Counter(r['sequence'] for r in items).items())),
            'seed':seed, 'rule':rule, **source, 'checkpoint_sha256':CHECKPOINT_SHA,
            'checkpoint_optimizer_steps':135930, 'final_valid_used':False,
            'scope':'train-side intermediate-checkpoint pilot; not unseen-backbone or final validation'}

    fit_manifest = manifest('flow_recovery_svs_fit20_v1', 'train_gate_fit', selected, FIT_SEED,
        'SHA256(seed+target_id), target_id tiebreak; first2 per each of ten gate-fit sequences; global same-key order')
    fit_manifest.update(nested_fit2_target_ids=original, BN_policy='P_paper', geometry='full9',
                        selection_uses_images_or_GT=False, expected_native_MLPs=12)
    check_manifest = manifest('flow_recovery_check64_v1','train_gate_development_check', checks, CHECK_SEED,
        'Exclude historical dev32; first32 per 01_a/05_a under SHA order, then global same-key order')
    check_manifest.update(excluded_historical_dev32_ids=old_dev['target_ids'],
        allowed_use='Only fixed recovery endpoints; never fit, calibration, optimizer, LR selection or best-checkpoint selection',
        independent_of_backbone_training=False, temporal_independence_claim=False,
        selection_uses_images_or_GT=False, member_data_or_metric_read_during_construction=False)
    recovery_manifest = manifest('flow_recovery_gatefit4573_v1','train_gate_fit_recovery', recovery, ORDER_SEED,
        'All and only gate-fit4573; SHA256(order_seed+target_id), target_id tiebreak; one fixed complete pass')
    recovery_manifest.update(order_seed=ORDER_SEED, augmentation_seed_policy=AUGMENTATION,
        max_target_visits=4573, max_effective_optimizer_steps=4573, repeat_to_replace_empty_crop=False,
        BN_pool_used_for_optimizer=False, check_data_used_for_optimizer=False)
    names = {'svs_fit20':(fit_manifest,selected), 'check64':(check_manifest,checks),
             'gatefit4573_recovery':(recovery_manifest,recovery)}
    payloads = {}
    for name,(m,items) in names.items():
        payloads[name+'.json'] = (json.dumps(m,indent=2,allow_nan=False)+'\n').encode('utf-8')
        stream = io.StringIO(newline=''); fields = ['order_1based','target_id','sequence','target_filename','auxiliary_role','selection_sha256']
        writer = csv.DictWriter(stream,fieldnames=fields,lineterminator='\n'); writer.writeheader()
        for i,r in enumerate(items,1):
            writer.writerow({'order_1based':i,'target_id':r['target_id'],'sequence':r['sequence'],
                'target_filename':r['target_filename'],'auxiliary_role':r['auxiliary_role'],
                'selection_sha256':order_key(m['seed'],r['target_id'])[0]})
        payloads[name+'.csv'] = stream.getvalue().encode('utf-8')
    idsets = {'fit20':{r['target_id'] for r in selected},'check64':{r['target_id'] for r in checks},
              'recovery4573':{r['target_id'] for r in recovery},'BN703':{r['target_id'] for r in roles_by_name['bn_calibration']},
              'historical_dev32':set(old_dev['target_ids'])}
    intersections = {a+'__'+b:len(idsets[a]&idsets[b]) for i,a in enumerate(idsets) for b in list(idsets)[i+1:]}
    require(intersections['fit20__recovery4573'] == 20 and
            all(v == 0 for k,v in intersections.items() if k != 'fit20__recovery4573'), 'Unexpected auxiliary overlap')
    audit = {'schema':'flow_recovery_data_contract_v1','status':'MEMBERS_FROZEN_METADATA_ONLY',
        **source,'builder_sha256':sha(__file__), 'source_metadata_only':True, 'image_or_GT_array_read':False,
        'final_valid_members_parsed':False, 'final_valid_data_or_metrics_read':False,
        'counts':{k:len(v) for k,v in idsets.items()}, 'target_intersections':intersections,
        'fit2_subset_fit20':True,'no_unknown_timestamps_or_GT_hashes_fabricated':True,
        'manifest_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in payloads.items()},
        'scientific_ready':False, 'readiness_scope':'Member contracts ready; data/checkpoint/source acceptance remains mandatory in consumer'}
    payloads['data_contract.json'] = (json.dumps(audit,indent=2,allow_nan=False)+'\n').encode('utf-8')
    return payloads


def publish(payloads, directory, verify_only=False):
    directory = Path(directory)
    for name,raw in payloads.items():
        path = directory/name
        require(not path.exists() or path.read_bytes() == raw, 'Immutable manifest differs: '+str(path))
        require(not verify_only or path.is_file(), 'Missing sealed manifest: '+str(path))
    if not verify_only:
        directory.mkdir(parents=True,exist_ok=True)
        for name,raw in payloads.items():
            path = directory/name
            if not path.exists():
                with path.open('xb') as f: f.write(raw)
    return {'status':'VERIFIED' if verify_only else 'SEALED','files':len(payloads),
            'data_contract_sha256':hashlib.sha256(payloads['data_contract.json']).hexdigest()}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('canonical','roles','fit2','dev32'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=HERE/'manifests');p.add_argument('--verify',action='store_true')
    a=p.parse_args();print(json.dumps(publish(make_payloads(a.canonical,a.roles,a.fit2,a.dev32),a.output_dir,a.verify),indent=2))
