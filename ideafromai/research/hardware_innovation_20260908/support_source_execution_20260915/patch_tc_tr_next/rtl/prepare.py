from pathlib import Path
import json,struct
import numpy as np
P=Path(__file__).resolve().parent;BASE=P.parents[2]
CAP=BASE/'algorithm/patch_probe/partial_completion/integer_valid10'
def main():
    cases=[];small=[]
    for ci,path in enumerate(sorted(CAP.glob('capture_*.npz'))[:4]):
        d=np.load(path);ids=d['group_ids'];w=d['source_gate_words'].astype(np.uint16)
        assert w.shape==(64,864,4)
        for i,(gid,sw) in enumerate(zip(ids,w)):
            idx=len(cases);cases.append((f'capture{ci}_group{int(gid)}',int(gid),1,sw))
            if ci==0 and i%8==0:small.append(idx)
    assert len(cases)==256 and sorted(cases[i][1]//2400 for i in small)==list(range(8))
    for name,w in [('zero',np.zeros((864,4),np.uint16)),('dense',np.full((864,4),1023,np.uint16)),
                   ('alternating',np.where((np.arange(864)[:,None]+np.arange(4)[None,:])%2,341,682).astype(np.uint16))]:
        small.append(len(cases));cases.append(('diagnostic_'+name,0,0,w))
    stats={}
    for name in ['ordinary_r32','regional_r96_a48']:
        d=np.load(P.parent/(name+'.npz'));u=d['u'].astype(np.int64);v=d['v'].astype(np.int64);a=d['a'].astype(np.int64);tau=d['tau'].astype(np.int64);rank=u.shape[1]
        masks=np.array([sum(int(b)<<i for i,b in enumerate(row)) for row in d['masks']],dtype=np.uint32)
        vcode=((d['v_nonzero'].astype(np.uint8)<<5)|((d['v_sign']<0).astype(np.uint8)<<4)|d['aligned_v_shift']).astype(np.uint8)
        dec=np.where(vcode&32,np.where(vcode&16,-1,1)*(np.int64(1)<<((vcode&15).astype(np.int64))),0);assert np.array_equal(dec,v)
        fields={k:[] for k in ['source','z','y','q','uout','gate']};metadata=[]
        for case,gid,real,w in cases:
            x=((w[...,None]>>np.arange(10))&1).transpose(1,2,0).reshape(40,864).astype(np.int64)
            mask=d['masks'][gid//2400].repeat(4)
            z=(x@u)*mask;y=z@v;q=np.einsum('ts,psr->ptr',a,z.reshape(4,10,rank)).reshape(40,rank)
            out=np.einsum('ts,psh->pth',a,y.reshape(4,10,96)).reshape(40,96)
            assert np.array_equal(out,q@v)
            assert abs(z).max()<32768 and abs(y).max()<2**31 and abs(q).max()<2**31 and abs(out).max()<2**47
            gate=(out>=np.tile(tau,(4,1))).astype(np.uint8)
            fields['source'].append(w);fields['z'].append(z.astype(np.int16));fields['y'].append(y.astype(np.int32));fields['q'].append(q.astype(np.int32));fields['uout'].append(out);fields['gate'].append(gate)
            metadata.append(dict(name=case,group=gid,real=real,region=gid//2400))
        with (P/(name+'.bin')).open('wb') as f:
            f.write(b'PATCH001');f.write(struct.pack('<III',rank,len(cases),len(small)));f.write(np.array(small,dtype='<u4').tobytes())
            for arr,typ in [(u,'i1'),(vcode,'u1'),(a,'<i2'),(tau,'<i8'),(masks,'<u4')]:f.write(np.asarray(arr,dtype=typ).tobytes())
            for idx,(case,gid,real,w) in enumerate(cases):
                nm=case.encode();f.write(struct.pack('<III',gid,real,len(nm)));f.write(nm)
                for key,typ in [('source','<u2'),('z','<i2'),('y','<i4'),('q','<i4'),('uout','<i8'),('gate','u1')]:f.write(np.asarray(fields[key][idx],dtype=typ).tobytes())
        stats[name]=dict(rank=rank,cases=len(cases),small_indices=small,full_real_cases=256,all_integer_order_equal=True,
                         maximums={k:int(np.max(np.abs(np.asarray(fields[k],dtype=np.int64)))) for k in ['z','y','q','uout']},cases_metadata=metadata)
    (P/'fixtures.json').write_text(json.dumps(stats,separators=(',',':'))+'\n');print({k:{j:v[j] for j in ['rank','cases','maximums']} for k,v in stats.items()})
if __name__=='__main__':main()
