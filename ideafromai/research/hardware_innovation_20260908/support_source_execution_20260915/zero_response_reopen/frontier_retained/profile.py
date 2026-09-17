"""Independent semantic frontier/residency and zero-aware backend work; no cycle oracle."""
from pathlib import Path
import csv,json,struct
import numpy as np
HERE=Path(__file__).resolve().parent
ZERO=HERE.parent
SOURCE=ZERO.parent

def frontier_work(raw,nodes,roots,rank):
    pairs=refs=misses=batches=0
    for p in range(32):
        tags=[-1]*4;fifo=0
        for g in range(6):
            state=[int(roots[g])]*10
            while any(x>=16 for x in state):
                live=[t for t in range(10) if state[t]>=16]
                var={t:(int(nodes[state[t]])>>32)&15 for t in live}
                candidates=set(var.values())
                chosen=sorted(candidates,key=lambda c:(g*16+c not in tags,int(rank[g,c])))[:4]
                reserved={tags.index(g*16+c) for c in chosen if g*16+c in tags}
                for c in chosen:
                    tag=g*16+c
                    if tag in tags:continue
                    empty=[i for i in range(4) if tags[i]<0 and i not in reserved]
                    victim=empty[0] if empty else next((fifo+k)%4 for k in range(4) if (fifo+k)%4 not in reserved)
                    tags[victim]=tag;reserved.add(victim);fifo=(victim+1)%4;misses+=1
                active=[t for t in live if var[t] in chosen]
                pairs+=len(active);refs+=len(chosen);batches+=1
                for t in active:
                    n=int(nodes[state[t]]);state[t]=(n>>16)&65535 if raw[p,t,g*16+var[t]] else n&65535
    return dict(source_produced_pairs=pairs,source_scalar_mac=pairs*10,source_channel_refs=refs,source_X_words=misses*2,batches=batches)

def verify_static_and_cases():
    a=(ZERO/'joined_zero.bin').read_bytes();b=(HERE/'inputs_expanded.bin').read_bytes()
    off=8+1536+200+80+200+3840*8+384+384+3840+2*36864
    for _ in range(2):n=struct.unpack_from('<I',a,off)[0];off+=4+n*8
    off+=24+96+96
    assert a[:off]==b[:off]
    def cases(buf):
        n=struct.unpack_from('<I',buf,off)[0];i=off+4;out={}
        for _ in range(n):
            real,size=struct.unpack_from('<II',buf,i);i+=8;name=buf[i:i+size].decode();i+=size
            out[name]=(real,buf[i:i+32*96*10*4]);i+=32*96*10*4
        assert i==len(buf);return out
    small,large=cases(a),cases(b)
    for k,v in small.items():assert large[k]==v
    return dict(static_parameter_graph_root_bytes_equal=True,first_two_X_and_diagnostics_equal=True)

def main():
    identity=verify_static_and_cases()
    src=np.load(SOURCE/'source_class_adapt/expanded_sources/source_cases.npz')
    q=np.load(SOURCE/'prefix_tables_entropy.npz');cl=np.load(ZERO/'zero_tables.npz')
    D=src['D'].astype(np.int64);W=np.load(ZERO/'Wz.npy').T.astype(np.int64)
    back=np.load(SOURCE.parent/'support_lut_execution_20260915/cases.npz');A=back['A'].astype(np.int64)
    rank=np.full((6,16),15,dtype=int)
    for g in range(6):rank[g,q[f'variables_g{g}']]=np.arange(len(q[f'variables_g{g}']))
    cases=[(f'train{i}',src['X_q16'][i].transpose(1,2,0).astype(np.int64)) for i in range(32)]
    cases.append(('diagnostic_zero',np.zeros((32,96,10),dtype=np.int64)))
    p,c,t=np.indices((32,96,10));cases.append(('diagnostic_signed_extreme',np.where((p+c+t)%2,(1<<23)-1,-(1<<23))))
    records={}
    for name,X in cases:
        raw=(np.einsum('ts,pcs->ptc',src['A_q12'].astype(np.int64),X)>=src['threshold_q28'][None,:,None]).astype(np.uint8)
        code=np.count_nonzero(raw.reshape(32,10,6,1,16)!=D[None,None],axis=-1).argmin(-1)
        for mode in (2,3):
            active=code.copy()
            if mode==3:
                for g in range(6):active[:,:,g]=cl['canonical'][g,code[:,:,g]]
            work=frontier_work(raw,q['code_nodes64'] if mode==2 else cl['class_nodes_entropy'],q['roots'][0] if mode==2 else cl['roots'][1],rank)
            updates=coeff=mac=jobs=zero_routes=0
            for hb in range(4):
                table=np.einsum('gkc,gch->gkh',D,W[:,hb*96:(hb+1)*96].reshape(6,16,96))
                canonical=np.empty((6,16),dtype=int)
                for g in range(6):
                    for k in range(16):canonical[g,k]=next(j for j in range(k+1) if np.array_equal(table[g,j],table[g,k]))
                touched=np.zeros(320,dtype=bool);routes={}
                for g in range(6):
                    for row,k in enumerate(active.reshape(320,6)[:,g]):
                        mapped=int(canonical[g,k])
                        if mapped==0:zero_routes+=k!=0;continue
                        routes.setdefault((g,mapped),set()).add(row)
                jobs+=len(routes);coeff+=len(routes)*8
                for dest in routes.values():updates+=len(dest);touched[list(dest)]=True
                mac+=sum(np.count_nonzero(A[:,r%10]) for r in np.flatnonzero(touched))
            work.update(backend_vector_updates=int(updates),backend_coeff_words=int(coeff),backend_vector_mac=int(mac),
                        backend_config_words=397*4,source_config_words=21,dictionary_words=12,bridge_writes=1920,bridge_reads=1280,
                        jobs=int(jobs),nonzero_support_zero_response_routes_omitted=int(zero_routes))
            records[(name,mode)]=work
    count=0
    for file in ['small.csv','expanded.csv']:
        if not (HERE/file).exists():continue
        for row in csv.DictReader((HERE/file).open()):
            assert int(row['frontier'])==1 and int(row['backend_mode'])==4 and row['function']=='zero_response'
            expected=records[(row['case'],int(row['mode']))]
            for k,v in expected.items():
                if k not in ('batches','jobs','nonzero_support_zero_response_routes_omitted'):assert int(row[k])==v,(file,row['case'],row['mode'],k,row[k],v)
            assert int(row['words'])==sum(int(row[k]) for k in ['dictionary_words','source_config_words','source_X_words','source_graph_words','backend_config_words','backend_coeff_words'])
            assert int(row['bytes'])==16*int(row['words'])
            assert int(row['cycles_to_done'])==sum(int(row[f'state{i}']) for i in range(10))
            count+=1
    for name,_ in cases:
        a,b=records[name,2],records[name,3]
        for k in ['backend_vector_updates','backend_coeff_words','backend_vector_mac','jobs']:assert a[k]==b[k]
    result=dict(status='PASS',records_verified=count,input_identity=identity,
                methodology='independent logical DAG frontier, resident-first ordering and FIFO allocation for references/X; independent downstream exact zero-aware route sets; graph reads/cycles measured, not predicted',
                records=[dict(case=k[0],mode=k[1],**v) for k,v in records.items()])
    (HERE/'PROFILE.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps(dict(status='PASS',records_verified=count)))

if __name__=='__main__':main()
