"""Arithmetic/work obligations reconstructed without reading RTL intermediates."""
from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;B=H.parents[1]
read=lambda p:np.array([int(x,16) for x in p.read_text().split()],np.uint32).view(np.int32).astype(np.int64)
rows=json.loads((H/'results.json').read_text())
small_rows=list(rows)
for stage in ['short','64','disjoint']:
    p=H/f'results_{stage}.json'
    if p.exists():rows+=json.loads(p.read_text())
pred={}
for name in sorted({r['fixture'] for r in rows}):
    d=Path(name);params=d if (d/'q1.hex').exists() else B/'fusion_review_followup_20260914/pair_dictionary/fixtures/real_0'
    q=read(params/'q1.hex').reshape(864,8);v=read(params/'q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
    oy,ox=read(d/'origin.hex');words=read(d/'source.hex').reshape(96,4,4)
    valid=np.array([[0<=oy+y<240 and 0<=ox+x<320 for x in range(4)] for y in range(4)])
    words=words*valid;bits=((words[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
    e=np.stack([bits[:,:,p//2:p//2+3,p%2:p%2+3].reshape(10,864) for p in range(4)]).reshape(40,864)
    kl=np.any(q!=0,axis=1);e=e*kl;z=e@q
    raw=np.concatenate([z@v[og*8:og*8+8].T for og in range(12)])
    assert np.array_equal(raw,read(d/'gold.hex').reshape(480,8)),name
    bitmap=e.reshape(40,54,16);n=bitmap.sum(2);live=n>0;single=n==1
    qb=((q[None]>>np.arange(3)[:,None,None])&1).reshape(3,54,16,8)
    ql=qb.any((2,3));counts=np.einsum('pbk,lbkr->plbr',bitmap,qb)
    dz=counts[:,0].sum(1)+2*counts[:,1].sum(1)-4*counts[:,2].sum(1)
    assert np.array_equal(z,dz)
    a=e.reshape(4,10,864).astype(bool);vl=v.reshape(12,8,8).any(1);zl=z!=0
    native=int((a[0]|a[1]).sum()+(a[2]|a[3]).sum())
    info=dict(K=int(kl.sum()),Q=int(e.any(0).sum()),U=native,DU=int((a[0]&a[1]).sum()+(a[2]&a[3]).sum()),V=int((vl&zl.any(0)).sum()),M=int((zl.sum(0)*vl.sum(0)).sum()),source=int(valid.sum())*96,N=int(live.sum()),single=int(single.sum()))
    info['planes']=int((live[:,None,:]&ql).sum())
    info['hybrid_planes']=int(((live&~single)[:,None,:]&ql).sum())
    info['groups']=int(live.any(0).sum())
    info['prefetch_planes']=int((live.any(0)[None,:]&ql).sum())
    for m,mask in [('full',live),('hybrid',live&~single)]:
        info[m+'_overlap']=int((mask[:,None,:]&(ql[1:]&ql[:-1])).sum())
    pred[name]=info
checks=0
for r in rows:
    p=pred[r['fixture']];m=r['mode'];bp=sum(r[k] for k in ['source_stalls','weight_stalls','output_stalls'])
    if m==14:
        expect=dict(cycles=5437+p['K']+2*p['Q']+3*p['U']+p['M']+bp,first_issues=p['U'],weight_words=p['Q']+p['V'],z_vector_reads=p['U']+40,z_writes=p['U']+20,aux_reads=0,aux_writes=0,aux_issues=0,aux_weight_words=0,aux_events=0,bitmap_native_reads=0,bitmap_native_issues=0,dual_updates=p['DU'])
    else:
        plane=p['hybrid_planes'] if m in [8,9] else p['planes'];one=p['single'] if m in [8,9] else 0
        scans=p['N'] if m in [13,8,9] else 2160
        work=(3*p['N']+plane) if m in [15,11] else 4*(p['N']-one)+2*one
        expect=dict(cycles=5517+(864 if m==15 else 0)+scans+work+p['M']+bp,first_issues=plane+one,weight_words=plane+one+p['V'],z_vector_reads=40,z_writes=60,aux_reads=scans,aux_writes=864,aux_issues=plane,aux_weight_words=plane,bitmap_native_reads=one,bitmap_native_issues=one,dual_updates=0)
        if not r['stall']:expect['aux_events']=0 if m in [11,15] else p['hybrid_overlap' if m==9 else 'full_overlap']
    if m in [8,9]:
        expect['cycles']=5491+3*p['N']+4*(p['N']-p['single'])+2*p['single']+p['M']+bp
        expect['z_vector_reads']=40+p['N'];expect['z_writes']=20+p['N']
    if m==8:
        expect['cycles']+=3*p['groups']-(p['N']-p['single'])
        expect['weight_words']=p['prefetch_planes']+p['single']+p['V']
        expect['aux_weight_words']=p['prefetch_planes'];expect['aux_events']=0
    if 'cache_reads' in r:
        expect['cache_reads']=p['hybrid_planes'] if m==8 else 0
        expect['cache_writes']=3*p['groups'] if m==8 else 0
    expect.update(second_weight_words=p['V'],mac_issues=p['M'],z_scalar_reads=p['M'],source_words=p['source'],local_source_reads=p['K'],psum_reads=480,psum_writes=480)
    for k,v in expect.items():assert r[k]==v,(Path(r['fixture']).name,m,k,r[k],v);checks+=1
    assert sum(r['state_cycles'])==r['cycles']
# Original implementations remain exact, including original deterministic BP.
old=json.loads((B/'fusion_ten_trials_20260914/decompositions/q1_bitplanes/results.json').read_text());matched=0
for r in small_rows:
    if r['mode'] not in [14,15] or r['command']>1:continue
    a=[x for x in old if Path(r['fixture']).name==x['fixture'] and r['mode']==x['mode'] and r['stall']==x['stall'] and r['command']==x['command']]
    if a:
        for key,value in a[0].items():
            if key!='fixture':assert r[key]==value,(key,r['fixture']);matched+=1
out=dict(passed=True,commands=len(rows),rtl_raw_values=sum(r['outputs'] for r in rows),independent_native_raw_values=3840*len(pred),counter_checks=checks,original_fields_reproduced=matched,predictors=pred)
(H/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='predictors'}))
