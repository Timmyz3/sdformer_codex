from pathlib import Path
from collections import Counter
import json
import numpy as np
H=Path(__file__).resolve().parent
B=H.parents[1];S=H.parent/'pair_sparse';OLD=B/'fusion_review_followup_20260914/pair_dictionary'
def readhex(p):return np.array([int(s,16) for s in p.read_text().split()],np.uint32).view(np.int32).astype(np.int64)
def writehex(p,a):p.write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in np.asarray(a).flat))
def compile_table(q):
    classes=np.zeros((864,4),np.int64);rep=np.zeros((32,8),np.int64);sizes=[];bounds=[];excluded=[]
    for g in range(4):
        keys=[tuple(map(int,row)) for row in q[:,2*g:2*g+2]]
        freq=Counter(k for k in keys if k!=(0,0))
        # Same key ordering as old unsigned signed3 encoded key.
        enc=lambda k:(k[0]&7)|((k[1]&7)<<3)
        chosen=sorted((k for k,n in freq.items() if 2<=n<=255),key=lambda k:(-freq[k],enc(k)))[:32]
        for k,key in enumerate(keys):classes[k,g]=0 if key==(0,0) else chosen.index(key)+1 if key in chosen else 63
        for i,key in enumerate(chosen):rep[i,2*g:2*g+2]=key
        sizes.append(len(chosen));bounds.append([freq[key] for key in chosen]);excluded.append(sum(n for key,n in freq.items() if n>255))
    return classes,rep,sizes,bounds,excluded

def native(words,q,origin):
    events=np.zeros((864,40),np.int64)
    for k in range(864):
        c,tap=divmod(k,9)
        for p in range(4):
            y,x=p//2+tap//3,p%2+tap%3
            if 0<=int(origin[0])+y<240 and 0<=int(origin[1])+x<320:
                events[k,p*10:p*10+10]=(words[c,y,x]>>np.arange(10))&1
    return events,events.T@q

def emit(name,words,q,v,origin,old_gold=None,reference_dir=None):
    events,z=native(words,q,origin)
    raw=np.concatenate([z@v[og*8:og*8+8].T for og in range(12)])
    assert np.min(raw)>=-(1<<31) and np.max(raw)<(1<<31)
    if old_gold is not None:assert np.array_equal(raw.ravel(),old_gold)
    cls,rep,sizes,bounds,excluded=compile_table(q)
    d=H/'fixtures'/name;d.mkdir(parents=True,exist_ok=True)
    for n,a in [('source',words),('origin',origin),('q1',q),('q2',v.reshape(12,8,8).transpose(0,2,1)),('k_live',np.any(q!=0,axis=1)),('gold',raw),('class',np.sum(cls<<(6*np.arange(4)),axis=1)),('representative',rep),('ngroups',[max(sizes)])]:writehex(d/(n+'.hex'),a)
    return dict(name=name,reference_dir=str(reference_dir) if reference_dir else None,group_sizes=sizes,class_bounds=bounds,max_class_bound=max([0]+sum(bounds,[])),over255_direct_pairs=sum(excluded),direct_pairs=int(np.count_nonzero(cls==63)),grouped_pairs=int(np.count_nonzero((cls>0)&(cls<=32))),raw_values=3840)

def main():
    records=[]
    for c in json.loads((S/'fixtures.json').read_text()):
        p=S/'fixtures'/c['name']
        if not p.exists():p=OLD/'fixtures'/c['name']
        records.append(emit(c['name'],readhex(p/'source.hex').reshape(96,4,4),readhex(p/'q1.hex').reshape(864,8),readhex(p/'q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8),readhex(p/'origin.hex'),readhex(p/'gold.hex'),p))
    v=readhex(OLD/'fixtures/real_0/q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
    # Mixed >255 direct and <=255 counted classes, with all four P and both T bytes active.
    q=np.zeros((864,8),np.int64)
    for g in range(4):
        q[:255,2*g:2*g+2]=[-4,3]
        q[255:511,2*g:2*g+2]=[-3,2]
        q[511:638,2*g:2*g+2]=[1,-2]
    full=np.full((96,4,4),1023,np.int64)
    records.append(emit('mixed_class_255_256',full,q,v,[17,19]))
    records.append(emit('q2_zero_overlay',full,q,np.zeros_like(v),[17,19]))
    (H/'fixtures.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(dict(fixtures=len(records),original_gold_checked=20*3840,max_group_count=max(r['max_class_bound'] for r in records),real_bounds=records[0]['class_bounds'])))
if __name__=='__main__':main()
