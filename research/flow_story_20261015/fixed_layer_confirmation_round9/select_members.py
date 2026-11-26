#!/usr/bin/env python3
"""Identity-only R9 member confirmation; no model, image or GT imports."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'research/flow_story_20261014/damage_predictor_round6/selection_manifest_v1.json'
CANON=ROOT/'research/flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json'
ROLES=ROOT/'research/flow_story_20261011/preflight/data_protocol/artifacts/train_auxiliary_roles.csv'
SEED='flow-r9-fixed-s2b2-confirm-20261015:'


def select(canonical,roles,prior):
    excluded=set(prior['excluded_ids'])|set(prior['groups']['check'])|set(prior['groups']['fit'])
    selected=[]
    for seq in sorted(prior['check_sequences']):
        pool=[r for r in canonical if r['sequence']==seq and r['protocol_split']=='train'
              and roles.get(r['target_id'])=='gate_development' and r['target_id'] not in excluded]
        ordered=sorted(pool,key=lambda r:(hashlib.sha256((SEED+r['target_id']).encode()).hexdigest(),r['target_id']))
        accepted=[]
        for row in ordered:
            index=int(row['target_id'].rsplit('_',1)[1])
            if all(abs(index-int(other['target_id'].rsplit('_',1)[1]))>=3 for other in accepted):
                accepted.append(row)
            if len(accepted)==8:break
        assert len(accepted)==8,'Insufficient members for fixed identity rule'
        selected+=accepted
    assert len(selected)==16 and len({r['target_id'] for r in selected})==16
    return selected


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prior=json.loads(OLD.read_text());canonical=json.loads(CANON.read_text())['records']
    with ROLES.open() as f:roles={r['target_id']:r['auxiliary_role'] for r in csv.DictReader(f)}
    rows=select(canonical,roles,prior); ids=[r['target_id'] for r in rows]
    for tag,chosen in [('a',ids[:4]+ids[8:12]),('b',ids[4:8]+ids[12:])]:
        # Frozen R6 collector accepts eight identities per preflight. R9 checks
        # the stronger union/exclusion/spacing contract itself, without patching it.
        out=json.loads(json.dumps(prior));out['groups']['check']=chosen
        out['check_seed']=SEED;out['check_rule']='R9 fixed union16 SHA rank, excluding prior R6 too, within-new-group ID spacing >=3; this file is one half of preflight'
        out['scope']='R9 numerical/data preflight batch only; no singleton collection or R6 labels'
        (HERE/('members_'+tag+'.json')).write_text(json.dumps(out,indent=2)+'\n')
    m={'schema':'flow_round9_identity_confirmation_v1','target_ids':ids,'excluded_ids':sorted(set(prior['excluded_ids'])|set(prior['groups']['check'])|set(prior['groups']['fit'])),
       'seed':SEED,'rule':'8 per existing development sequence by SHA(seed+id), greedy within-new-group numeric ID spacing >=3; no GT/input/model outcomes used',
       'ordinary_arm_fixed_before_new_data':'fine_s2b2','source_sha256':{str(p):sha(p) for p in (OLD,CANON,ROLES,Path(__file__))},
       'scope':'New members in historically exposed backbone-seen training sequences; not new sequences, official test or independent statistical samples'}
    (HERE/'membership_v1.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))


if __name__=='__main__':main()
