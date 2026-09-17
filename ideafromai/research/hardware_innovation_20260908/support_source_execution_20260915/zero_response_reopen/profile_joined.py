"""Independent X -> nearest support -> zero-aware routes and whole-T10 work."""
from pathlib import Path
import sys,csv,json
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SRC=HERE.parent
PREV=SRC.parent/'support_lut_execution_20260915'
sys.path.insert(0,str(SRC/'joined_chain'))
from profile import source_requests

def main():
    z=np.load(SRC/'source_cases.npz');b=np.load(PREV/'cases.npz');q=np.load(SRC/'prefix_tables_entropy.npz')
    D=z['D'].astype(np.int64);A0=z['A_q12'].astype(np.int64);A1=b['A'].astype(np.int64)
    ds=(D*(1<<np.arange(16))).sum(2)
    assert np.all(D[:,0]==0) and np.all(D[:,1:].sum(-1)>=2)
    functions={}
    for name,wp,tab in [('zero_response','Wz.npy','zero_tables.npz'),('integer_L2_zero_allowed','integer_l2_W.npy','integer_l2_tables.npz')]:
        a=np.load(HERE/tab)
        functions[name]=(np.load(HERE/wp).T.astype(np.int64),a['canonical'],a['class_nodes_entropy'],a['roots'][1])
    cases=[(f'train{i}',z['X_q16'][i].transpose(1,2,0).astype(np.int64)) for i in range(2)]
    cases.append(('diagnostic_zero',np.zeros((32,96,10),dtype=np.int64)))
    p,c,t=np.indices((32,96,10));cases.append(('diagnostic_signed_extreme',np.where((p+c+t)%2,(1<<23)-1,-(1<<23))))
    rank=np.full((6,16),15,dtype=np.uint8)
    for g in range(6):rank[g,q[f'variables_g{g}']]=np.arange(len(q[f'variables_g{g}']))
    records={}
    for cname,X in cases:
        raw=(np.einsum('ts,pcs->ptc',A0,X)>=z['threshold_q28'][None,:,None]).astype(np.uint8)
        code=np.count_nonzero(raw.reshape(32,10,6,1,16)!=D[None,None],axis=-1).argmin(-1)
        for fname,(W,canonical,cnodes,croots) in functions.items():
            for mode in (2,3):
                ac=code.copy()
                if mode==3:
                    for g in range(6):ac[:,:,g]=canonical[g,code[:,:,g]]
                channels=source_requests(raw,q['code_nodes64'] if mode==2 else cnodes,q['roots'][0] if mode==2 else croots,rank)
                for nh in (1,4):
                    updates=coeff=mac=jobs=zeros=0
                    for hb in range(nh):
                        wh=W[:,hb*96:(hb+1)*96];routes={}
                        table=np.einsum('gkc,gch->gkh',D,wh.reshape(6,16,96))
                        canonical_h=np.empty((6,16),dtype=int)
                        for g in range(6):
                            for k in range(16):canonical_h[g,k]=next(j for j in range(k+1) if np.array_equal(table[g,j],table[g,k]))
                        for g in range(6):
                            for r in range(320):
                                k=int(ac.reshape(320,6)[r,g]);mapped=int(canonical_h[g,k])
                                if mapped==0:
                                    zeros+=k!=0
                                    assert not np.any(table[g,k])
                                    continue
                                routes.setdefault((g,mapped),set()).add(r)
                        touched=np.zeros(320,dtype=bool)
                        for (g,k),rows in routes.items():
                            assert k!=0 and np.any(table[g,k])
                            coeff+=8;jobs+=1;updates+=len(rows);touched[list(rows)]=True
                        mac+=int(sum(np.count_nonzero(A1[:,r%10]) for r in np.flatnonzero(touched)))
                    records[(fname,cname,mode,nh)]={'source_channels':channels,'source_scalar_mac':channels*100,
                        'source_X_words':channels*2,'source_config_words':32*21,
                        'backend_vector_updates':updates,'backend_vector_mac':mac,'backend_coeff_words':coeff,
                        'backend_config_words':nh*397,'bridge_writes':1920,'bridge_reads':320*nh,
                        'jobs':jobs,'nonzero_support_zero_response_routes_omitted':zeros}
    compared=0
    for path in sorted(HERE.glob('joined_*_cycles.csv')):
        for row in csv.DictReader(path.open()):
            assert int(row['backend_mode'])==4
            key=(row['function'],row['case'],int(row['mode']),int(row['hblocks']))
            for k,value in records[key].items():
                if k not in ('jobs','nonzero_support_zero_response_routes_omitted'):assert int(row[k])==value,(key,k,row[k],value)
            compared+=1
    for fname in functions:
        for cname,_ in cases:
            for nh in (1,4):
                a,b=records[(fname,cname,2,nh)],records[(fname,cname,3,nh)]
                for k in ['backend_coeff_words','backend_vector_updates','backend_vector_mac','jobs']:assert a[k]==b[k],(fname,cname,k)
    result={'status':'PASS','observed_csv_records_verified':compared,'method':'independent semantic routes, all zero mappings omitted before jobs, actual coefficient/update/MAC counters checked',
            'records':[dict(function=k[0],case=k[1],mode=k[2],hblocks=k[3],backend_mode=4,**v) for k,v in records.items()]}
    (HERE/'JOINED_PROFILE.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({'status':'PASS','records_verified':compared}))

if __name__=='__main__':main()
