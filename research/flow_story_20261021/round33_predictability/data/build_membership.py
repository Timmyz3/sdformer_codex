"""Metadata-only, deterministic Round33 selector identities; never opens arrays."""
import csv, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def rank(role,tid): return hashlib.sha256(('flow-r33-20261021:'+role+':'+tid).encode()).hexdigest()
def main():
    r27=ROOT/'research/flow_story_20261019/round27/independent_review/membership_v1.json'
    canon=ROOT/'research/flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json'
    setup=ROOT/'research/flow_story_20261012/baseline/native_setup'
    dep=setup/'native_deployment_manifest.csv'; ver=setup/'verification_files.json'
    pins={r27:'40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b',canon:'6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22',dep:'fec5a819074b316779e80fdda41096e60963819bd4318ff69b1966049c391124',ver:'769671e6e3754f068587eaed576ce51c0d0d6f939aeabf033a006f9aec9e6c5c'}
    for p,h in pins.items(): assert sha(p)==h,p
    old=read(r27); rows=read(canon)['records']; verified={r['relative_path']:r for r in read(ver)['files']}
    exclusions={k:set(v) for k,v in old['excluded_development_exposures'].items()}
    exclusions['R27_R32_train256_dev64']={r['target_id'] for r in old['records']}
    consumed=set().union(*exclusions.values())
    trainseq=old['sequences']; validseq=sorted({r['sequence'] for r in rows if r['protocol_split']=='valid'})
    assert len(trainseq)==8 and len(validseq)==5 and not set(trainseq)&set(validseq)
    times={}; sources=[]
    for seq in trainseq+validseq:
        rel=f'train_optical_flow/{seq}/flow/forward_timestamps.txt'; p=HERE/'timestamps'/rel
        assert sha(p)==verified[rel]['sha256'],rel
        times[seq]=[tuple(map(int,l.split(','))) for l in p.read_text().splitlines() if l.strip() and not l.startswith('#')]
        sources.append(dict(sequence=seq,relative_path=str(p.relative_to(ROOT)),sha256=sha(p)))
    def interval(row): return times[row['sequence']][row['target_index_1based']-1]
    def gap(a,b):
        if a['sequence']!=b['sequence']: return None
        x,y=interval(a);u,v=interval(b);return max(x,u)-min(y,v)
    def choose(pool,role,n):
        ordered=sorted(pool,key=lambda r:(interval(r),r['target_id']))
        assert len(ordered)>=n,(role,len(ordered))
        bins=[ordered[i*len(ordered)//n:(i+1)*len(ordered)//n] for i in range(n)]
        return [min(b,key=lambda r:rank(role,r['target_id'])) for b in bins]
    selected=[]; pools=[]
    for seq in trainseq:
        pool=[r for r in rows if r['sequence']==seq and r['target_id'] not in consumed and r['protocol_split']=='train']
        tune=choose(pool,'tune',2)
        fitpool=[r for r in pool if r not in tune and all(gap(r,t)>=100000 for t in tune)]
        fit=choose(fitpool,'fit',8)
        selected.extend((role,r) for role,rs in [('fit',fit),('tune',tune)] for r in rs)
        pools.append(dict(sequence=seq,fresh_pool=len(pool),fit_pool_after_tune_purge=len(fitpool)))
    for seq in validseq:
        pool=[r for r in rows if r['sequence']==seq and r['target_id'] not in consumed and r['protocol_split']=='valid']
        selected.extend(('confirm',r) for r in choose(pool,'confirm',8))
        pools.append(dict(sequence=seq,fresh_pool=len(pool)))
    deployment={}
    for f in csv.DictReader(dep.open()):
        if f['category'] in ('input','gt','mask'):deployment.setdefault(f['target'],[]).append(f)
    records=[]
    for role,row in selected:
        tid=row['target_id']; files=deployment[tid]
        assert len(files)==3 and {f['category'] for f in files}=={'input','gt','mask'}
        for f in files:
            v=verified[f['relative_path']]
            assert v['passed'] and v['all_elements_finite'] and v['sha256']==f['sha256'] and v['bytes']==int(f['bytes'])
        t0,t1=interval(row)
        records.append(dict(target_id=tid,sequence=row['sequence'],role=role,canonical=row,t0_us=t0,t1_us=t1,source_files=files))
    ids=[r['target_id'] for r in records]; assert len(ids)==len(set(ids))==120 and not set(ids)&consumed
    minimum=min(gap(a,b) for ar,a in selected for br,b in selected if ar=='fit' and br=='tune' and gap(a,b) is not None)
    assert minimum>=100000
    out=dict(schema='flow_r33_selector_membership_v1',status='FROZEN_METADATA_ONLY',builder_sha256=sha(__file__),records=records,
        counts={r:sum(x['role']==r for x in records) for r in ('fit','tune','confirm')},
        source_pins={str(p.relative_to(ROOT)):h for p,h in pins.items()},timestamp_sources=sources,
        excluded_identity_groups={k:sorted(v) for k,v in exclusions.items()},candidate_pools=pools,
        temporal_contract=dict(fit_tune_minimum_gap_us=minimum,required_gap_us=100000,confirmation_sequence_disjoint=True),
        scope='New selector identities excluding pinned prior exposures and R27/R32 train256/dev64. Fit/tune same8 training sequences; confirmation5 repository-local valid sequences, not official DSEC test and no claim backbone-selection-unseen. Historical exposure inventory is pinned but not exhaustive of every earlier repository run.',
        selection='Metadata-only chronological strata and SHA256 ranking; no GT/quality/model arrays opened.',quality_arrays_read=False,model_forwards=0)
    dest=HERE/'MEMBERSHIP_V1.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(sha256=sha(dest),counts=out['counts'],temporal=out['temporal_contract'],pools=pools)))
if __name__=='__main__':main()
