"""Metadata-only R38 exposure union, temporal selection and source admission."""
from pathlib import Path
import hashlib,json,re
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];B=HERE.parent/'round38_frozen_survival';R33=ROOT/'research/flow_story_20261021/round33_predictability'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=HERE/'R38_ENTRY_METADATA_INDEPENDENT_V1.json';assert not out.exists()
    survey=read(B/'data/ENTRY_METADATA_SURVEY_V1.json');assert sha(B/'ENTRY_PROTOCOL_V1.md')==survey['protocol_sha256']
    for p,h in survey['registry_sources'].items():assert sha(ROOT/p)==h
    local=read(B/'data/LOCAL_EXPOSURE_SCAN_V2.json');remote=read(B/'data/REMOTE_EXPOSURE_SCAN_V2.json')
    assert not local['skipped_large_files'] and not remote['skipped_large_files']
    union=set();bytes_scanned=0;rx=re.compile(rb'(?:zurich_city_\d\d_[a-z]|thun_\d\d_[a-z])_\d{4}')
    for entry in local['files']:
        p=ROOT/entry['path'];h=hashlib.sha256();found=set();carry=b'';size=0
        with p.open('rb') as f:
            while True:
                block=f.read(1024*1024)
                if not block:break
                size+=len(block);h.update(block);joined=carry+block;found.update(v.decode() for v in rx.findall(joined));carry=joined[-128:]
        assert h.hexdigest()==entry['sha256'] and size==entry['bytes'],entry['path']
        assert found==set(entry['target_ids']),(entry['path'],found-set(entry['target_ids']),set(entry['target_ids'])-found)
        union.update(found);bytes_scanned+=size
    assert union==set(local['union_ids'])
    ru=set().union(*(set(v['target_ids']) for v in remote['files']));assert ru==set(remote['union_ids']);union|=ru
    m33=read(R33/'data/MEMBERSHIP_V1.json')
    prior=set(r['target_id'] for r in m33['records'])
    for value in m33['excluded_identity_groups'].values():
        if isinstance(value,list):prior.update(value)
        elif isinstance(value,dict):
            for k in ('ids','target_ids'):
                if k in value:prior.update(value[k])
    assert len(prior)==survey['registry_counts']['pinned_history_plus_R33']
    union|=prior;assert union==set(survey['excluded_union']) and len(union)==6058
    canonicalpath=ROOT/'research/flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json';canonical=read(canonicalpath)['records'];byid={r['target_id']:r for r in canonical}
    times={}
    for p,h in survey['timestamp_bindings'].items():
        assert sha(ROOT/p)==h;seq=Path(p).parts[-3];t=np.loadtxt(ROOT/p,delimiter=',',comments='#',dtype=np.int64);assert t.ndim==2 and t.shape[1]==2 and np.all(t[:,1]>t[:,0]);times[seq]=t
    def interval(r):return times[r['sequence']][r['target_index_1based']-1]
    def gap(a,b):
        aa,ab=interval(a);ba,bb=interval(b);return int(max(aa,ba)-min(ab,bb))
    selected=[];findings=[]
    for seq in [s['sequence'] for s in survey['sequences']]:
        rows=[r for r in canonical if r['sequence']==seq and r['protocol_split']=='valid'];excluded=[r for r in rows if r['target_id'] in union]
        pool=[r for r in rows if r['target_id'] not in union and all(gap(r,e)>=100000 for e in excluded)]
        midpoint=lambda r:float(interval(r).sum())/2
        low=min(map(midpoint,rows));high=max(map(midpoint,rows));picks=[]
        for fraction in (1/8,3/8,5/8,7/8):
            goal=low+(high-low)*fraction;candidates=[r for r in pool if r not in picks and all(gap(r,e)>=100000 for e in picks)]
            assert candidates;pick=min(candidates,key=lambda r:(abs(midpoint(r)-goal),r['target_id']));picks.append(pick)
        s=next(s for s in survey['sequences'] if s['sequence']==seq);assert [r['target_id'] for r in picks]==s['selected_ids'];assert len(pool)==s['fresh_after_purge'];assert len(rows)==s['canonical_count']
        for r,z in zip(picks,s['temporal_targets']):assert midpoint(r)==z['selected_midpoint_us']
        findings.append(dict(sequence=seq,selected_ids=[r['target_id'] for r in picks],selected_intervals=[interval(r).tolist() for r in picks],fresh_after_purge=len(pool),min_gap_vs_excluded_us=min(gap(r,e) for r in picks for e in excluded),min_selected_pair_gap_us=min(gap(picks[i],picks[j]) for i in range(4) for j in range(i))))
        selected.extend(picks)
    assert selected==survey['selected_canonical'] and len(selected)==20
    mempath=B/'data/PROPOSED_SOURCE_MEMBERSHIP_V1.json';members=read(mempath);assert members['metadata_survey_sha256']==sha(B/'data/ENTRY_METADATA_SURVEY_V1.json');assert [r['canonical'] for r in members['records']]==selected
    verificationpath=R33/'data/native_verification_files_v1.json';native={r['relative_path']:r for r in read(verificationpath)['files']};remotecheck=read(B/'data/REMOTE_SELECTED_SOURCE_CHECK_V1.json');headers={(r['target_id'],r['category']):r for r in remotecheck['files']};assert remotecheck['status']=='PASS' and len(headers)==60
    for r in members['records']:
        assert r['target_id'] not in union and r['role']=='new_identity_survival';assert len(r['source_files'])==3
        for f in r['source_files']:
            v=native[f['relative_path']];h=headers[(r['target_id'],f['category'])]
            assert v['passed'] and v['all_elements_finite'];assert v['sha256']==h['sha256']==f['sha256'];assert v['bytes']==h['bytes']==int(f['bytes'])
            assert h['shape']==v['npy_header']['shape'] and h['dtype']==v['npy_header']['dtype']
    result=dict(status='PASS_METADATA_ONLY',auditor_sha256=sha(__file__),protocol_sha256=sha(B/'ENTRY_PROTOCOL_V1.md'),survey_sha256=sha(B/'data/ENTRY_METADATA_SURVEY_V1.json'),proposed_membership_sha256=sha(mempath),native_verification_sha256=sha(verificationpath),local_inventory_sha256=sha(B/'data/LOCAL_EXPOSURE_SCAN_V2.json'),remote_inventory_sha256=sha(B/'data/REMOTE_EXPOSURE_SCAN_V2.json'),remote_source_check_sha256=sha(B/'data/REMOTE_SELECTED_SOURCE_CHECK_V1.json'),local_files_rehashed_and_ID_rescanned=len(local['files']),local_bytes_ID_scanned=bytes_scanned,remote_file_receipts_reaggregated=len(remote['files']),excluded_union_count=len(union),selected_count=len(selected),source_headers_checked=60,sequences=findings,new_arrays_decoded=False,new_quality_read=False,GPU_calls=0,scope='Finite available-history exclusion and100msGT-interval purge; new identities in same5sequences, not new-sequence/generalization or certification of backbone/deleted-history absence. Remote exposure inventory reaggregated from pinned receipts, not second remote filesystem replay. Canonical legacy unverified timestamp text is inherited; selection independently uses pinned accepted native forward timestamps.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'sha256':sha(out),'local_files':len(local['files']),'selected':[r['target_id'] for r in selected]}))
if __name__=='__main__':main()
