#!/usr/bin/env python3
"""R27 identity/timestamp-only recovery/development split. No array/model access."""
import argparse,csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
R13=REPO/'research/flow_story_20261016/coverage_recovery_round13'
R20=REPO/'research/flow_story_20261017/action_domain_round20/independent_review'
SETUP=REPO/'research/flow_story_20261012/baseline/native_setup'
PINS={
 'canonical':(REPO/'research/flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json','6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22'),
 'r13_membership':(R13/'data_protocol/manifest_v1.json','eba4544fe99c082d7d099f90d03470d7abad22b01b68d3eb4d022e61e78151d1'),
 'selector_roles':(R20/'SELECTOR_ROLES_V1.json','e6f216f6b017fc2cbba262ec51c70474f342e593e1549058a2200970949e10a1'),
 'acceptance':(SETUP/'acceptance.json','b1d87de8fd48b710fe5a49dd37d6d64427d2fe111df249efa73ba2299f454540'),
 'deployment':(SETUP/'native_deployment_manifest.csv','fec5a819074b316779e80fdda41096e60963819bd4318ff69b1966049c391124'),
 'verification':(SETUP/'verification_files.json','769671e6e3754f068587eaed576ce51c0d0d6f939aeabf033a006f9aec9e6c5c')}
GAP_US=100000
SEED=20261019

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rank(salt,tid):return hashlib.sha256((salt+tid).encode()).hexdigest(),tid
def sequence(tid):return tid.rsplit('_',1)[0]
def run():
 for key,(p,h) in PINS.items():require(sha(p)==h,'Frozen metadata '+key)
 old=read(PINS['selector_roles'][0]);r13=read(PINS['r13_membership'][0]);canon=read(PINS['canonical'][0])['records']
 seqs=r13['split_sequences']['train'];require(len(seqs)==8,'Eight original recovery-training sequences')
 selected_old={r['target_id'] for r in old['records']};sealed={r['target_id'] for r in old['records'] if r['role']=='sealed_confirmation'}
 h8=set(old['excluded_exposures']['H8_development']);require(len(selected_old)==32 and len(sealed)==len(h8)==8,'Preserve all prior roles')
 exposures={k:set(v) for k,v in old['excluded_exposures'].items()};exposures['R22_fit16_dev8_and_sealed8']=selected_old
 consumed=set().union(*exposures.values())
 time_table={};timestamp_sources=[]
 for spec in old['timestamp_sources_to_fetch_metadata_only']:
  p=R20/'timestamp_sources_v1'/spec['relative_path'];require(sha(p)==spec['sha256'],'Original verified timestamp text')
  pairs=[tuple(map(int,line.split(','))) for line in p.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
  require(all(len(v)==2 and v[0]<v[1] for v in pairs),'Positive source intervals')
  require(all(a[0]<=b[0] for a,b in zip(pairs,pairs[1:])),'Ordered source timestamps')
  time_table[spec['sequence']]=pairs;timestamp_sources.append(dict(sequence=spec['sequence'],relative_path=str(p.relative_to(REPO)),sha256=sha(p),interval_count=len(pairs)))
 def interval(tid):
  seq,number=tid.rsplit('_',1);index=int(number)-1;require(0<=index<len(time_table[seq]),'Target timestamp row');return time_table[seq][index]
 def gap(tid,other):
  if sequence(tid)!=sequence(other):return None
  a,b=interval(tid);c,d=interval(other);return max(a,c)-min(b,d)
 def separated(tid,others,minimum=GAP_US):return all(gap(tid,x) is None or gap(tid,x)>=minimum for x in others)
 def nearest(tid,others):
  same=[(x,gap(tid,x)) for x in sorted(others) if x!=tid and gap(tid,x) is not None]
  if not same:return dict(minimum_window_gap_us=None,nearest_ids=[],positive_overlap_count=0)
  minimum=min(g for _,g in same)
  return dict(minimum_window_gap_us=minimum,nearest_ids=[x for x,g in same if g==minimum],positive_overlap_count=sum(g<0 for _,g in same))
 deployment={}
 for f in csv.DictReader(PINS['deployment'][0].open()):
  if f['category'] in ('input','gt','mask'):deployment.setdefault(f['target'],[]).append(f)
 verified={v['relative_path']:v for v in read(PINS['verification'][0])['files']}
 def accepted(row):
  files=deployment.get(row['target_id'],[])
  require(len(files)==3 and {f['category'] for f in files}=={'input','gt','mask'},'Actual accepted three-file contract')
  for f in files:
   v=verified[f['relative_path']];require(v['passed'] is True and v['all_elements_finite'] is True and v['sha256']==f['sha256'] and v['bytes']==int(f['bytes']),'Accepted complete file proof')
  return files
 train=[];dev=[];selection=[]
 for seq in seqs:
  pool=[r for r in canon if r['protocol_split']=='train' and r['sequence']==seq]
  # No score/GT support is consulted. Time strata span available fresh identities.
  candidates=sorted((r for r in pool if r['target_id'] not in consumed and separated(r['target_id'],sealed|h8)),key=lambda r:(interval(r['target_id']),r['target_id']))
  require(len(candidates)>=8,'Eight fresh metadata-only dev candidates')
  bins=[candidates[i*len(candidates)//8:(i+1)*len(candidates)//8] for i in range(8)]
  chosen=[min(b,key=lambda r:rank('flow-r27-dev64-20261019:',r['target_id'])) for b in bins]
  devids={r['target_id'] for r in chosen}
  trainpool=[r for r in pool if r['target_id'] not in selected_old|h8|devids and separated(r['target_id'],devids|sealed|h8)]
  require(len(trainpool)>=32,'32 recovery members after temporal purge')
  picked=sorted(trainpool,key=lambda r:rank('flow-r27-train256-20261019:',r['target_id']))[:32]
  for row in chosen+picked:accepted(row)
  train+=picked;dev+=chosen
  selection.append(dict(sequence=seq,canonical_train_count=len(pool),fresh_development_candidates=len(candidates),temporally_purged_train_candidates=len(trainpool),development_bins=[dict(index=i+1,candidate_ids=[r['target_id'] for r in b],selected_id=chosen[i]['target_id']) for i,b in enumerate(bins)]))
 trainids={r['target_id'] for r in train};devids={r['target_id'] for r in dev}
 require(len(trainids)==256 and len(devids)==64 and not trainids&devids and not (trainids|devids)&(sealed|h8|selected_old),'Fixed disjoint roles/old selector seal')
 require(not devids&consumed,'No pinned prior experiment identity in newdev')
 require(all(separated(t,devids|sealed|h8) for t in trainids),'Every recovery window temporally purged')
 records=[]
 for role,rows in (('train',train),('development',dev)):
  other=devids if role=='train' else trainids
  for row in rows:
   tid=row['target_id'];t0,t1=interval(tid)
   records.append(dict(target_id=tid,sequence=row['sequence'],role=role,canonical=row,t0_us=t0,t1_us=t1,timestamp_data_row_0based=row['target_index_1based']-1,
    source_files=accepted(row),nearest_other_role_window=nearest(tid,other),nearest_sealed_window=nearest(tid,sealed),nearest_H8_window=nearest(tid,h8),
    prior_exposure_groups=[k for k,ids in exposures.items() if tid in ids]))
 order=[]
 for epoch in (1,2):
  for index in range(32):
   for seq in seqs:
    row=[r for r in train if r['sequence']==seq][index];step=len(order)+1
    digest=hashlib.sha256(('flow-r27-crop-20261019:'+str(step)+':'+row['target_id']).encode()).digest()
    order.append(dict(step=step,pass_index=epoch,sequence_visit_index=index+1,target_id=row['target_id'],sequence=seq,
      crop=dict(top=int.from_bytes(digest[:8],'big')%193,left=int.from_bytes(digest[8:16],'big')%257,height=288,width=384),augmentation='fixed_crop_only_no_flip_or_rotation'))
 require(len(order)==512 and all(sum(s['target_id']==tid for s in order)==2 for tid in trainids),'Two complete matched passes')
 all_files=[dict(target_id=r['target_id'],role=r['role'],**f) for r in records for f in r['source_files']]
 require(len(all_files)==len({f['relative_path'] for f in all_files})==960,'Selected320 exact960 assets')
 return dict(schema='flow_r27_recovery_identity_v1',status='FROZEN_METADATA_ONLY',builder_sha256=sha(__file__),seed=SEED,
  source_sha256={k:dict(relative_path=str(p.relative_to(REPO)),sha256=h) for k,(p,h) in PINS.items()},timestamp_sources=timestamp_sources,
  mapping='Source comma/int64 data rows excluding #header; target_index_1based-1 matches accepted native generator. No target array content read.',
  sequences=seqs,train_members=train,dev_members=dev,train_order=order,records=records,source_files=all_files,sealed_ids=sorted(sealed),H8_historical_ids=sorted(h8),
  excluded_development_exposures={k:sorted(v) for k,v in exposures.items()},selection=selection,
  counts=dict(train=256,development=64,sealed_unchanged=8,historical_H8=8,updates_per_arm=512,source_arrays=960),
  temporal_contract=dict(minimum_cross_role_window_gap_us=GAP_US,minimum_observed_train_dev_gap_us=min(r['nearest_other_role_window']['minimum_window_gap_us'] for r in records),train_dev_positive_overlap_pairs=sum(r['nearest_other_role_window']['positive_overlap_count'] for r in records if r['role']=='train'),train_sealed_positive_overlap_pairs=sum(r['nearest_sealed_window']['positive_overlap_count'] for r in records if r['role']=='train')),
  availability='All selected960 files have existing accepted source SHA/finite/header proof. Current remote bytes must be rechecked beforeuse; this builder never opens arrays or sealed file paths.',
  scope='Same8 original backbone-training sequences, all320 seen by backbone training. Newdev64 is only heldout from this recovery and pinned prior diagnostic identities, not official validation or unseen scenes. Temporal100ms purge reduces direct overlap but does not establish statistical independence.',
  protocol='Three native/Q8/Q6 recovery arms use identical512 identities/order/crops/seed and fixed final endpoint. Old R26 failures unchanged; H8 not used for selection. No selector learning.',
  quality_arrays_read=False,model_forwards=0,GPU_calls=0,SSH_calls=0)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();out=run();dest=HERE/'membership_v1.json'
 if a.verify:require(read(dest)==out,'Deterministic metadata reconstruction');print('R27_MEMBERSHIP_METADATA_VERIFY_PASS')
 else:
  require(not dest.exists(),'No overwrite frozen membership');dest.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'sha256':sha(dest),'counts':out['counts'],'temporal':out['temporal_contract']}))
