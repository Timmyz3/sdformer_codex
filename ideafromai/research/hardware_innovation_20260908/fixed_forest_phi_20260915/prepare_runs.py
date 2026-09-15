"""Calibrate on separate tiles and serialize raw source/W/static codebooks."""
from pathlib import Path
import numpy as np
import struct,json
H=Path(__file__).resolve().parent
def forest(masks):
    p=np.full(40,-1,np.int8)
    for i,x0 in enumerate(masks):
        x=int(x0)
        if x.bit_count()<2:continue
        valid=[j for j,y0 in enumerate(masks) if j!=i and int(y0)!=0 and
               (int(y0)&~x)==0 and (int(y0)!=x or j<i)]
        if valid:p[i]=min(valid,key=lambda j:(-int(masks[j]).bit_count(),j))
    return p
def delta(m,p):return np.array([int(x)&~(int(m[p[i]]) if p[i]>=0 else 0) for i,x in enumerate(m)],np.uint16)
def book(m):
    values,counts=np.unique(m,return_counts=True)
    keep=np.array([int(x).bit_count()>1 for x in values]);values=values[keep];counts=counts[keep]
    if not len(values):return np.zeros(8,np.uint16)
    order=sorted(range(len(values)),key=lambda i:(-int(counts[i]),int(values[i])))
    c=[int(values[order[0]])]
    while len(c)<min(8,len(values)):
        rest=[i for i in order if int(values[i]) not in c]
        q=max(rest,key=lambda i:(min((int(values[i])^x).bit_count() for x in c)*int(counts[i]),-int(values[i])))
        c.append(int(values[q]))
    for _ in range(8):
        labels=[min(range(len(c)),key=lambda j:(int(x)^c[j]).bit_count()) for x in values]
        n=list(c)
        for j in range(len(c)):
            ids=[i for i,v in enumerate(labels) if v==j]
            if ids:
                total=sum(int(counts[i]) for i in ids)
                n[j]=sum((sum(int(counts[i]) for i in ids if int(values[i])&(1<<b))*2>total)<<b for b in range(16))
        if n==c:break
        c=n
    return np.array((c+[0]*8)[:8],np.uint16)
records=[]
with np.load(H/'cases.npz') as z:
    for prefix,num in [('',8),('full_',54)]:
        for i in range(num):
            cm=z[prefix+'calibration_masks'][i];cp=forest(cm);cd=delta(cm,cp)
            co,cu=book(cm),book(cd)
            vals,cnt=np.unique(cd,return_counts=True)
            freq=sorted([(int(n),int(v)) for v,n in zip(vals,cnt) if int(v).bit_count()>1],key=lambda x:(-x[0],x[1]))
            cs=np.array(([v for n,v in freq][:2]+[0]*8)[:8],np.uint16)
            for split in ['calibration','held']:
                m=z[prefix+split+'_masks'][i];p=forest(m)
                if not np.array_equal(p,z[prefix+split+'_parents'][i]):raise ValueError('parent contract differs')
                records.append(dict(name=str(z[prefix+split+'_case_name'][i]),kind=2 if prefix else 1,
                    held=int(split=='held'),m=m,p=p,w=z[prefix+split+'_W'][i],co=co,cu=cu,cs=cs))
for k in range(5):
    if k==0:m=np.zeros(40,np.uint16)
    elif k==1:m=np.array([1<<(i%16) for i in range(40)],np.uint16)
    elif k==2:m=np.full(40,0xfffe,np.uint16)
    elif k==3:m=np.array([0xffff ^ (1<<(i%16)) ^ (1<<((i+5)%16)) for i in range(40)],np.uint16)
    else:m=np.array([0x00ff if i%2 else 0x0fff for i in range(40)],np.uint16)
    w=np.resize(np.array([-4,3,-1,2,0,-2,1,-3],np.int8),(16,8))
    if k==2:w=np.full((16,8),-4,np.int8)
    p=forest(m);cb=np.array([0xffff,0x0fff,0x00ff,0xff00,0x0f0f,0x3333,0x5555,0],np.uint16)
    records.append(dict(name='diagnostic_'+str(k),kind=0,held=0,m=m,p=p,w=w,co=cb,cu=cb,cs=cb))
with (H/'runs.bin').open('wb') as f:
    f.write(b'FFP1'+struct.pack('<I',len(records)))
    for c in records:
        n=c['name'].encode();f.write(struct.pack('<H',len(n))+n+bytes([c['kind'],c['held']]))
        for key,dtype in [('m','<u2'),('p','i1'),('w','i1'),('co','<u2'),('cu','<u2'),('cs','<u2')]:f.write(np.asarray(c[key],dtype=dtype).tobytes())
(H/'run_cases.json').write_text(json.dumps(dict(cases=len(records),calibration='same-K separate non-overlapping tile; q8 deterministic weighted binary Lloyd; no held fitting',
    directed='five protocol/negative-residual/EM/zero diagnostics, not network gains'),indent=2)+'\n')
print(len(records),'raw cases')
