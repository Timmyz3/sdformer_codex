"""Final source/obligation audit and exact-control reproducibility receipts."""
from pathlib import Path
import json
import runpy
import numpy as np

H=Path(__file__).resolve().parent
B=H.parents[1]
A=H/'adapt_rr'
env=runpy.run_path(str(H/'verify.py'))
runpy.run_path(str(A/'verify.py'))
predict=env['predict']

def read(path):
    return json.loads(path.read_text())

# Existing modes must reproduce every old diagnostic as well as elapsed cycles.
controls_checked=0
for name in ['results.json','results_small.json','results_64.json']:
    old=read(B/'fusion_review_followup_20260914/rr_modular'/name)
    new=read(H/name)
    index={(r.get('fixture'),r.get('first_tile'),r['tiles'],r['mode'],r['stall'],r['command']):r for r in new}
    for row in old:
        key=(row.get('fixture'),row.get('first_tile'),row['tiles'],row['mode'],row['stall'],row['command'])
        newrow=index[key]
        for field,value in row.items():
            if field!='wall_seconds_so_far':
                assert newrow[field]==value,(name,key,field,value,newrow[field])
        controls_checked+=1
old={(r['mode'],r['stall'],r['command']):r for r in read(H/'results_64.json')}
for row in read(A/'results_controls.json'):
    for field,value in old[row['mode'],row['stall'],row['command']].items():
        if field!='wall_seconds_so_far':
            assert row[field]==value,('adapt original mode reproducibility',field,row[field],value)
    controls_checked+=1

# Reconstruct both raw, identity Q20 and I24 independently on each native tile.
D=B/'r8_consumer_fusion_20260914/data'
c=np.load(D/'consumer_coefficients.npz')
q1,q2,a,b=[c[k].astype(np.int64) for k in ['q1','q2','a_q40','b_q20']]
source=np.load(D/'first_source_words.npy',mmap_mode='r')
rawgold=np.load(D/'raw_p_full.npy',mmap_mode='r')
idfp=np.load(D/'identity_fp32_full.npy',mmap_mode='r')
jgold=np.load(D/'identity_q20_full.npy',mmap_mode='r')
igold=np.load(D/'i24_new_full.npy',mmap_mode='r')
native_tiles=sorted(set(range(128,192))|set(range(159,162))|set(range(19197,19200)))
for tile in native_tiles:
    oy,ox=2*(tile//160)-1,2*(tile%160)-1
    word=np.zeros((96,4,4),np.uint16)
    source_words=0
    for y in range(4):
        for x in range(4):
            if 0<=oy+y<240 and 0<=ox+x<320:
                word[:,y,x]=source[:,oy+y,ox+x]
                source_words+=96
    _,raw=predict(word,q1,q2,source_words)
    raw=raw.transpose(1,2,0).reshape(10,96,2,2)
    j=np.clip(np.rint(idfp[tile].astype(np.float64)*(1<<20)),-2**31,2**31-1).astype(np.int64)
    wide=raw*a[None,:,None,None]+(j+b[None,:,None,None])*(1<<20)
    quotient=wide>>26
    rem=wide&((1<<26)-1)
    i=np.clip(quotient+((rem>(1<<25))|((rem==(1<<25))&((quotient&1)!=0))),-2**23,2**23-1)
    assert np.array_equal(raw,rawgold[tile]) and np.array_equal(j,jgold[tile]) and np.array_equal(i,igold[tile]),tile

pairword=np.fromfile(A/'pair_fixture/source.bin','<u2').reshape(96,4,6)
pairpred=None
for tile in range(2):
    pred,raw=predict(pairword[:,:,2*tile:2*tile+4],q1,q2,1536)
    raw=raw.reshape(4,10,12,8).transpose(2,0,1,3).reshape(480,8)
    assert np.array_equal(raw,np.fromfile(A/'pair_fixture/raw.bin','<i4').reshape(2,480,8)[tile])
    if pairpred is None:
        pairpred=dict.fromkeys(pred,0)
    for k,v in pred.items():
        pairpred[k]+=v

allrows=[]
receipt_files=['results.json','results_small.json','results_cross.json','results_64.json','audit_results.json']
for folder in [H,A]:
    for name in receipt_files+(['results_controls.json','pair_results.json'] if folder==A else []):
        allrows += read(folder/name)

# Reconcile every assertion-run row, including the deliberately asymmetric pair.
for folder in [H,A]:
    rows=read(folder/'audit_results.json')+(read(A/'pair_results.json') if folder==A else [])
    for r in rows:
        n,m=r['tiles'],r['mode']
        p=pairpred if r['fixture']=='pair_fixture' else {k:v*n for k,v in env['predictions'][r['fixture']].items()}
        u=p['U1'] if m==1 else p['U2']
        repair=p['repairs'] if m==2 else 0
        norm=10*n if m==2 else 0
        expected=dict(core_first_issues=u,core_mac_issues=p['M'],core_repair_issues=repair,
                      core_normalization_issues=norm,core_z_vector_reads=u+40*n+repair+norm,
                      core_z_writes=u+10*n+repair+norm,core_source_words=p['source'],
                      borrow_grants=u if m>=3 else 0)
        expected['core_cycles']=5427*n+p['K']+2*p['Q']+3*u+p['M']+2*(repair+norm)+sum(r[k] for k in ['core_source_stalls','core_weight_stalls','core_output_stalls','core_arbitration_stalls'])
        for key,value in expected.items():
            assert r[key]==value,(r['fixture'],m,key,r[key],value)
        assert r['core_arbitration_stalls']==r['conflict_cycles']+r['borrow_consumer_stalls']
        assert r['shared_alu_grants']==r['proof_issues']+(0 if m>=3 else u)+p['M']+repair+norm
        assert r['shared_wide_grants']==r['borrow_grants']+r['consumer_add_issues']
        assert r['consumer_cycles']==3385*n+r['consumer_join_wait_cycles']+r['consumer_output_stalls']+r['consumer_wide_waits']
        assert r['window_cycles']==r['consumer_cycles']+n+n//2
        assert r['total_cycles']==r['window_cycles']+r['launch_cycles']+sum(r[k] for k in ['static_words','parameter_stalls','source_load_words','origin_words','source_load_stalls'])+1

summary=dict(passed=True,commands=len(allrows),rtl_values_each_stage=sum(r['outputs'] for r in allrows),
             original_mode_receipts_exactly_reproduced=controls_checked,
             independently_reconstructed_native_tiles=len(native_tiles),independent_native_raw_J_I24_each=len(native_tiles)*3840,
             pair_prediction=pairpred,
             mode4_change='One RR bit arbitrates the existing wide chain between consumer and producer group; no added datapath or memory port.')
(H/'final_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
