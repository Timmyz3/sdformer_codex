#!/usr/bin/env python3
"""Fixed selector roles: source timestamps only, no array/quality access."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    role=HERE/'SELECTOR_ROLES_V1.json'; assert sha(role)=='e6f216f6b017fc2cbba262ec51c70474f342e593e1549058a2200970949e10a1'
    m=json.loads(role.read_text()); table={}; source=[]
    for r in m['timestamp_sources_to_fetch_metadata_only']:
        p=HERE/'timestamp_sources_v1'/r['relative_path'];assert sha(p)==r['sha256']
        a=np.loadtxt(p,delimiter=',',dtype=np.int64)
        assert a.ndim==2 and a.shape[1]==2 and (a[:,1]>a[:,0]).all() and (a[1:,0]>=a[:-1,0]).all()
        table[r['sequence']]=a
        source.append(dict(sequence=r['sequence'],path=str(p.relative_to(HERE)),sha256=sha(p),intervals=len(a),first_line=p.read_text().splitlines()[0]))
    def interval(tid):
        seq,raw=tid.rsplit('_',1); i=int(raw)-1; a=table[seq];assert 0<=i<len(a)
        return int(a[i,0]),int(a[i,1])
    def nearest(tid,others):
        same=[x for x in others if x!=tid and x.rsplit('_',1)[0]==tid.rsplit('_',1)[0]]
        if not same:return dict(minimum_gap_us=None,nearest_ids=[],overlap_count=0,positive_overlap_count=0)
        a,b=interval(tid); rows=[]
        for x in same:
            c,d=interval(x);overlap=max(0,min(b,d)-max(a,c));gap=max(0,max(a,c)-min(b,d))
            rows.append(dict(target_id=x,gap_us=gap,overlap_us=overlap,start_difference_us=c-a))
        best=min(v['gap_us'] for v in rows)
        return dict(minimum_gap_us=best,nearest=[v for v in rows if v['gap_us']==best],overlap_count=sum(v['overlap_us']>0 for v in rows),positive_overlap_count=sum(v['overlap_us']>0 for v in rows))
    groups={k:[r['target_id'] for r in rows] for k,rows in m['groups'].items()}
    records=[]
    for r in m['records']:
        tid=r['target_id'];a,b=interval(tid)
        records.append(dict(target_id=tid,role=r['role'],sequence=r['canonical']['sequence'],timestamp_data_row_0based=r['canonical']['target_index_1based']-1,t0_us=a,t1_us=b,duration_us=b-a,
          nearest_role_windows={k:nearest(tid,v) for k,v in groups.items()},nearest_exposed_windows={k:nearest(tid,v) for k,v in m['excluded_exposures'].items()}))
    pairs={}
    for i,k in enumerate(groups):
        for l in list(groups)[i+1:]:
            rows=[{'target_id':r['target_id'],**r['nearest_role_windows'][l]} for r in records if r['role']==k and r['nearest_role_windows'][l]['minimum_gap_us'] is not None]
            best=min(r['minimum_gap_us'] for r in rows)
            pairs[k+'__'+l]=dict(minimum_gap_us=best,pairs=[r for r in rows if r['minimum_gap_us']==best],positive_overlap_pairs=sum(r['positive_overlap_count'] for r in rows))
    sourcefiles=[REPO/'research/flow_story_20261012/data_encoding'/n for n in ('generate_native_inputs.py','verify_native_dataset.py')]
    out=dict(schema='flow_r21_selector_window_metadata_audit_v1',status='PASS_SOURCE_TIMESTAMP_ROWS',role_manifest_sha256=sha(role),auditor_sha256=sha(Path(__file__)),timestamp_sources=source,generation_mapping_source_sha256={str(p.relative_to(REPO)):sha(p) for p in sourcefiles},
      mapping='np.loadtxt comma/int64 ignores # header; generator and acceptance verifier use target_index_1based-1. No off-by-one physical text-line indexing.',
      records=records,cross_role_pairs=pairs,source_scope='Eight timestamp files match prior verified transfer SHA. The accepted native-input contract binds each target to this row. No new source-H5 reconstruction or original-PNG/GT comparison was performed for these32 identities.',
      independence_scope='Same8 backbone-training sequences; identity and interval separation is not sequence-independent validation. H8 and prior exposures remain disclosed. No role replacements based on gaps.',GT_event_or_prediction_arrays_read=False,GPU_calls=0)
    p=HERE/'SELECTOR_WINDOW_AUDIT_V1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'sha256':sha(p),'cross_role_pairs':pairs},indent=2))
if __name__=='__main__':main()
