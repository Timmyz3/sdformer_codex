"""Independently rebuild native sums, repair obligations and actual FSM cost."""
from pathlib import Path
import json
import numpy as np

H=Path(__file__).resolve().parent
B=H.parents[1]
read=lambda p:np.array([int(x,16) for x in p.read_text().split()],np.uint32).view(np.int32).astype(np.int64)
rows=[]
for stage in ['small','64','disjoint']:
    rows += json.loads((H/f'results_{stage}.json').read_text())
pred={}
for name in sorted({r['fixture'] for r in rows}):
    p=Path(name)
    params=p if (p/'q1.hex').exists() else B/'fusion_review_followup_20260914/pair_dictionary/fixtures/real_0'
    q=read(params/'q1.hex').reshape(864,8)
    v=read(params/'q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
    oy,ox=read(p/'origin.hex')
    words=read(p/'source.hex').reshape(96,4,4)
    valid=np.array([[0<=oy+y<240 and 0<=ox+x<320 for x in range(4)] for y in range(4)])
    words=words*valid
    bits=((words[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
    patches=np.stack([bits[:,:,i//2:i//2+3,i%2:i%2+3].reshape(10,864) for i in range(4)])
    live=np.any(q!=0,axis=1)
    a=patches[:,:,live].astype(bool)
    good=bool(np.all(np.maximum(q,0).sum(0)<=511) and np.all(np.minimum(q,0).sum(0)>=-512))
    u=[int((a[0]|a[1]).sum()+(a[2]|a[3]).sum()),int((a[0]|a[1]|a[2]).sum()+a[3].sum()),int(a.any(0).sum())]
    if not good:u[1]=u[0]
    exact=np.zeros((4,10,8),np.int64);low=exact.copy();high=exact.copy();rep=fields=0
    for k in np.flatnonzero(live):
        delta=patches[:,:,k,None]*q[k]
        exact+=delta;unwrapped=low+delta
        over=(unwrapped < -128)|(unwrapped>127)
        rep+=int(over.any(axis=(0,2)).sum());fields+=int(over.sum())
        high+=(unwrapped>127).astype(np.int64)-(unwrapped < -128).astype(np.int64)
        low=((unwrapped+128)&255)-128
        assert np.array_equal(exact,low+256*high)
        assert np.all(high>=-16) and np.all(high<16)
    canonical=(((high-(low<0))&31)<<8)|(low&255)
    canonical=np.where(canonical>=4096,canonical-8192,canonical)
    assert np.array_equal(canonical,exact)
    raw=(exact@v.T).reshape(4,10,12,8).transpose(2,0,1,3).reshape(480,8)
    assert np.array_equal(raw,read(p/'gold.hex').reshape(480,8))
    vl=np.any(v.reshape(12,8,8)!=0,axis=1);zl=exact!=0
    pred[name]=dict(U=u,range_ok=int(good),repairs=rep,fields=fields,A=int(a.sum()),K=int(live.sum()),Q=int(a.any((0,1)).sum()),V=int((vl&zl.any((0,1))).sum()),M=int((zl.sum((0,1))*vl.sum(0)).sum()),source=96*int(valid.sum()))

comparisons=0
for r in rows:
    p=pred[r['fixture']];m=r['mode'];u=p['U'][m];rep=p['repairs'] if m==2 else 0;norm=10 if m==2 else 0
    expect=dict(first_issues=u,merged_updates=p['A']-u,repair_issues=rep,repair_fields=p['fields'] if m==2 else 0,normalization_issues=norm,z_vector_reads=u+40+rep+norm,z_writes=u+10+rep+norm,z_scalar_reads=p['M'],mac_issues=p['M'],source_words=p['source'],local_source_reads=p['K'],weight_words=p['Q']+p['V'],second_weight_words=p['V'],psum_reads=480,psum_writes=480,range_ok=p['range_ok'],fallback_used=int(m==1 and not p['range_ok']))
    expect['cycles']=5427+p['K']+2*p['Q']+3*u+p['M']+2*(rep+norm)+sum(r[k] for k in ['source_stalls','weight_stalls','output_stalls'])
    for key,value in expect.items():
        assert r[key]==value,(r['fixture'],m,key,r[key],value)
        comparisons+=1
    assert sum(r['state_cycles'])==r['cycles']

# Same data and no external BP: arithmetic/memory transactions reproduce the
# earlier complete-consumer run. That consumer itself stalls raw drain; its
# actual output_stalls must be removed only for this intrinsic-service check.
# Wrapper/consumer totals are not compared or called raw service.
old=json.loads((B/'fusion_review_followup_20260914/modular_packing/results_64.json').read_text())
matched=0
for prior in old:
    if prior['stall']:continue
    a=[r for r in rows if r['case']=='64' and r['mode']==prior['mode'] and not r['stall'] and r['command']//64==prior['command']]
    for key in ['source_words','weight_words','second_weight_words','local_source_reads','z_vector_reads','z_scalar_reads','z_writes','first_issues','merged_updates','repair_issues','repair_fields','normalization_issues','psum_reads','psum_writes','mac_issues']:
        assert sum(r[key] for r in a)==prior['core_'+key],(key,prior['mode'])
        matched+=1
    assert sum(r['cycles'] for r in a)==prior['core_cycles']-prior['core_output_stalls']
    matched+=1
out=dict(passed=True,commands=len(rows),rtl_raw_values=sum(r['outputs'] for r in rows),native_fixtures=len(pred),independently_recomputed_raw_values=3840*len(pred),counter_checks=comparisons,prior_complete_consumer_core_fields_matched=matched,datapath_unchanged=True)
(H/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
