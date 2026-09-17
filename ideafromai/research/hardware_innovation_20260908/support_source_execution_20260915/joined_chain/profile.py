"""Independent semantic work counts from X, rather than replaying RTL states."""
from pathlib import Path
import csv,json
import numpy as np

HERE=Path(__file__).resolve().parent
SRC=HERE.parent
PREV=SRC.parent/'support_lut_execution_20260915'

def source_requests(raw,nodes,roots,rank):
    total=0
    for pt in range(32):
        for g in range(6):
            states=np.full(10,int(roots[g]),dtype=np.int64)
            while np.any(states>=16):
                live=states>=16
                v=((nodes[states]>>np.uint64(32))&np.uint64(15)).astype(int)
                chosen=min(v[live].tolist(),key=lambda c:int(rank[g,c]))
                update=live&(v==chosen)
                values=nodes[states[update]]
                bit=raw[pt,update,g*16+chosen]
                states[update]=np.where(bit,(values>>np.uint64(16))&np.uint64(65535),values&np.uint64(65535))
                total+=1
    return total

def main():
    z=np.load(SRC/'source_cases.npz');b=np.load(PREV/'cases.npz');q=np.load(SRC/'prefix_tables_entropy.npz')
    D=z['D'].astype(np.int64);A0=z['A_q12'].astype(np.int64);A1=b['A'].astype(np.int64)
    ds=(D*(1<<np.arange(16))).sum(2)
    ids=[np.flatnonzero((b['hblock']==h)&b['is_real'])[0] for h in range(4)]
    functions={'original':(np.concatenate([b['W'][i] for i in ids],axis=1).astype(np.int64),q['response_canonical_code'],q['class_nodes64'],q['roots'][1]),
               'response_class':(np.load(PREV/'response_class_W.npy').T.astype(np.int64),q['response_canonical_code'],q['class_nodes64'],q['roots'][1])}
    adapt=SRC/'source_class_adapt/adapt_tables.npz'
    if adapt.exists():
        aq=np.load(adapt)
        functions['adapt_response_class']=(np.load(adapt.parent/'adapt_W.npy').T.astype(np.int64),aq['canonical'],aq['class_nodes_entropy'],aq['roots'][1])
    cases=[(f'train{i}',z['X_q16'][i].transpose(1,2,0).astype(np.int64)) for i in range(2)]
    cases.append(('diagnostic_zero',np.zeros((32,96,10),dtype=np.int64)))
    p,c,t=np.indices((32,96,10));cases.append(('diagnostic_signed_extreme',np.where((p+c+t)%2,(1<<23)-1,-(1<<23))))
    rank=np.full((6,16),15,dtype=np.uint8)
    for g in range(6):rank[g,q[f'variables_g{g}']]=np.arange(len(q[f'variables_g{g}']))
    records={}
    for cname,X in cases:
        raw=(np.einsum('ts,pcs->ptc',A0,X)>=z['threshold_q28'][None,:,None]).astype(np.uint8)
        distance=np.count_nonzero(raw.reshape(32,10,6,1,16)!=D[None,None,:,:,:],axis=-1)
        code=distance.argmin(-1)
        for fname,(W,canonical,cnodes,croots) in functions.items():
            for mode in ([0,1,2] if fname=='original' else [2,3] if fname.startswith('adapt') else [0,1,2,3]):
                activecode=code.copy()
                if mode==3:
                    for g in range(6):activecode[:,:,g]=canonical[g,code[:,:,g]]
                S=np.stack([D[g,activecode[:,:,g]] for g in range(6)],axis=2).reshape(320,96)
                channels=2048 if mode<2 else source_requests(raw,q['code_nodes64'] if mode==2 else cnodes,q['roots'][0] if mode==2 else croots,rank)
                for nh,bmode in [(nh,bm) for nh in (1,4) for bm in ([0] if mode==0 else [2,4])]:
                    updates=coeff=mac=jobs=0
                    for hb in range(nh):
                        wh=W[:,hb*96:(hb+1)*96];routes={}
                        table=np.einsum('gkc,gch->gkh',D,wh.reshape(6,16,96))
                        local_canonical=np.tile(np.arange(16),(6,1))
                        for g in range(6):
                            for k in range(1,16):
                                local_canonical[g,k]=next(j for j in range(1,k+1) if np.array_equal(table[g,j],table[g,k]))
                        for g in range(6):
                            for r in range(320):
                                word=int(np.dot(S[r,g*16:(g+1)*16],1<<np.arange(16)))
                                if mode and word.bit_count()>=2:
                                    k=int(np.flatnonzero(ds[g]==word)[0])
                                    if bmode==4:k=int(local_canonical[g,k])
                                    routes.setdefault(('lut',g,k),set()).add(r)
                                else:
                                    for j in range(16):
                                        if word&(1<<j):routes.setdefault(('direct',g,j),set()).add(r)
                        touched=np.zeros(320,dtype=bool)
                        for (kind,g,k),rows in routes.items():
                            vector=D[g,k]@wh[g*16:(g+1)*16] if kind=='lut' else wh[g*16+k]
                            coeff+=8 if kind=='lut' else 6;jobs+=1
                            if np.any(vector):updates+=len(rows);touched[list(rows)]=True
                        mac+=int(sum(np.count_nonzero(A1[:,r%10]) for r in np.flatnonzero(touched)))
                    records[(fname,cname,mode,nh,bmode)]={'source_channels':channels,'source_scalar_mac':channels*100,
                        'source_X_words':channels*2,'source_config_words':32*(30 if mode<2 else 21),
                        'backend_vector_updates':updates,'backend_vector_mac':mac,'backend_coeff_words':coeff,
                        'backend_config_words':nh*(382 if bmode==0 else 397 if bmode==4 else 394),'bridge_writes':1920,'bridge_reads':320*nh,
                        'jobs':jobs}
    compared=0
    for csvpath in [HERE/'cycles.csv',HERE/'strong_cycles.csv',HERE/'adapt_strong_cycles.csv']:
        if not csvpath.exists():continue
        for row in csv.DictReader(csvpath.open()):
            bmode=int(row.get('backend_mode',0 if int(row['mode'])==0 else 2))
            key=(row['function'],row['case'],int(row['mode']),int(row['hblocks']),bmode)
            expected=records[key]
            for k,value in expected.items():
                if k!='jobs':assert int(row[k])==value,(key,k,row[k],value)
            compared+=1
    result={'method':'independent X->source PSN->Hamming->D->semantic routes and touched rows; no cycle prediction',
            'observed_csv_records_verified':compared,'status':'PASS',
            'records':[dict(function=k[0],case=k[1],mode=k[2],hblocks=k[3],backend_mode=k[4],**v) for k,v in records.items()]}
    (HERE/'profile.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({'records_verified':compared,'status':'PASS'}))

if __name__=='__main__':main()
