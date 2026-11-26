#!/usr/bin/env python3
"""Identity/source-metadata only; never opens event/GT/prediction arrays."""
from pathlib import Path
import csv,hashlib,json

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
R13=REPO/'research/flow_story_20261016/coverage_recovery_round13'
SETUP=REPO/'research/flow_story_20261012/baseline/native_setup'
PINS={
 'reserve':(R13/'data_protocol/manifest_v1.json','eba4544fe99c082d7d099f90d03470d7abad22b01b68d3eb4d022e61e78151d1'),
 'H8':(R13/'next_action_headroom/implementation_h0/h8_manifest.json','1083d2a3f9b940aee0ebf271cb30b062eeed7bc4bcbf51429d573172573569f5'),
 'acceptance':(SETUP/'acceptance.json','b1d87de8fd48b710fe5a49dd37d6d64427d2fe111df249efa73ba2299f454540'),
 'deployment':(SETUP/'native_deployment_manifest.csv','fec5a819074b316779e80fdda41096e60963819bd4318ff69b1966049c391124'),
 'verification':(SETUP/'verification_files.json','769671e6e3754f068587eaed576ce51c0d0d6f939aeabf033a006f9aec9e6c5c'),
 'canonical':(REPO/'research/flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json','6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22')}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
seq=lambda tid:tid.rsplit('_',1)[0]
index=lambda tid:int(tid.rsplit('_',1)[1])

def nearest(tid,ids):
    same=[x for x in ids if x!=tid and seq(x)==seq(tid)]
    if not same:return {'minimum_target_index_gap':None,'nearest_ids':[]}
    d=min(abs(index(tid)-index(x)) for x in same)
    return {'minimum_target_index_gap':d,'nearest_ids':[x for x in sorted(same) if abs(index(tid)-index(x))==d]}

def run():
    for _,(p,h) in PINS.items():assert sha(p)==h
    m=read(PINS['reserve'][0]);h8={r['target_id'] for r in read(PINS['H8'][0])['members']}
    canonical={r['target_id']:r for r in read(PINS['canonical'][0])['records']}
    accepted={r['relative_path']:r for r in read(PINS['verification'][0])['files']}
    deploy=list(csv.DictReader(PINS['deployment'][0].open()))
    groups={k:[] for k in ('fit','development','sealed_confirmation')}
    for sequence in m['split_sequences']['train']:
        rows=[r for r in m['selector_members'] if r['sequence']==sequence and r['target_id'] not in h8]
        assert len(rows)==4
        for i,row in enumerate(rows):
            assert row==canonical[row['target_id']]
            role='fit' if i<2 else 'development' if i==2 else 'sealed_confirmation'
            groups[role].append(row)
    ids={r['target_id'] for rows in groups.values() for r in rows};assert len(ids)==32 and not ids&h8
    exposures={'historical164':set(m['excluded_ids']),'H8_development':h8,
      'recovery_diversity200':{r['target_id'] for r in m['train_members_diversity']},
      'BN176':{r['target_id'] for r in m['bn_members']},'R13_check8':{r['target_id'] for r in m['check_members']}}
    assert all(not ids&v for v in exposures.values())
    roleids={k:{r['target_id'] for r in rows} for k,rows in groups.items()}
    records=[]
    for role,rows in groups.items():
        for row in rows:
            tid=row['target_id'];files=[r for r in deploy if r['target']==tid]
            assert len(files)==3 and {r['category'] for r in files}=={'input','gt','mask'}
            for f in files:
                v=accepted[f['relative_path']]
                assert v['passed'] is True and v['all_elements_finite'] is True and v['sha256']==f['sha256'] and v['bytes']==int(f['bytes'])
            records.append({'target_id':tid,'role':role,'canonical':row,'accepted_array_metadata':files,
              'nearest_role_index_gaps':{k:nearest(tid,v) for k,v in roleids.items()},
              'nearest_previous_exposure_index_gaps':{k:nearest(tid,v) for k,v in exposures.items()}})
    timestamps=[r for r in deploy if r['category']=='timestamps' and r['sequence'] in m['split_sequences']['train']]
    assert len(timestamps)==8
    out={'schema':'flow_r20_next_selector_identity_roles_v1','status':'FROZEN_METADATA_ONLY',
      'source_sha256':{k:{'path':str(p),'sha256':h} for k,(p,h) in PINS.items()},'builder_sha256':sha(Path(__file__)),
      'rule':'Exclude the8 H8 IDs; preserve original reserve order within each original8 training sequences; first2 fit, third development, fourth sealed_confirmation.',
      'groups':groups,'group_counts':{k:len(v) for k,v in groups.items()},'records':records,
      'excluded_exposures':{k:sorted(v) for k,v in exposures.items()},'input_root_from_acceptance':read(PINS['acceptance'][0])['root'],
      'timestamp_sources_to_fetch_metadata_only':timestamps,
      'availability_scope':'All96 input/GT/mask files were accepted with fullSHA/header/finite evidence in20261012 source metadata. Current remote existence/bytes must be rechecked; local arrays not read.',
      'exposure_scope':'No intersection with pinned oldselector/recovery/audit identity ledger. Backbone saw train; all3roles share sequences. H8 is exposed development. No claim of globally research-unseen or sequence-independent confirmation.',
      'temporal_scope':'Canonical timestamps null; filename-index gaps reported only. Source timestamps/generation receipts needed for actual interval/neighbor overlap audit; missing time evidence does not become seconds.',
      'sealed_confirmation_scope':'Eight identities sealed for a later frozen small development-screening check, not official or independent-scene confirmation. Do not read their GT in the first16fit+8development pilot.',
      'arrays_or_quality_read':False,'GPU_calls':0,'SSH_calls':0}
    return out

if __name__=='__main__':
    out=run();p=HERE/'SELECTOR_ROLES_V1.json';assert not p.exists()
    p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'sha256':sha(p),'group_counts':out['group_counts']}))
