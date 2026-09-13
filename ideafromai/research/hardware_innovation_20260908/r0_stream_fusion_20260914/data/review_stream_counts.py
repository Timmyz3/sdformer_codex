"""Independent stream review: direct input/geometry totals and absolute cycle formulas."""
import json
import numpy as np
from model_access import HERE
RTL=HERE.parent/'stream_rtl'

def predict(first,n,live,mode):
    words=np.load(HERE/'source_words.npy',mmap_mode='r')
    pad=np.pad(words,((0,0),(1,1),(1,1)))
    ids=np.arange(first,first+n);ys=2*(ids//160);xs=2*(ids%160)
    groups=live.sum(0).astype(np.int64);alive=int((groups!=0).sum())
    valid_total=0;service=0;W=0;Jtotal=0;Htotal=0;updates=0;terminal=0
    pc=np.array([i.bit_count() for i in range(1024)],dtype=np.int32)
    for sy in range(4):
        for sx in range(4):
            valid=((ys+sy-1>=0)&(ys+sy-1<240)&(xs+sx-1>=0)&(xs+sx-1<320))
            valid_total+=int(valid.sum())
            P=sum(0<=sy-py<3 and 0<=sx-px<3 for py in range(2) for px in range(2))
            for cg,G in enumerate(groups):
                if not G:continue
                a,b,c,d=pad[cg*4:cg*4+4,ys+sy,xs+sx]
                u=a|b;v=c|d;active=(u|v)!=0
                K=(u!=0).astype(np.int32)+(v!=0).astype(np.int32)
                J=((a&b)!=0).astype(np.int32)+((c&d)!=0).astype(np.int32)
                H=pc[u&v];L=pc[u|v]
                Q=sum((x!=0).astype(np.int32) for x in (a,b,c,d));mult=int(G)*P
                W+=mult*int(Q.sum());Jtotal+=mult*int(J.sum());Htotal+=mult*int(H.sum())
                terminal+=mult*int((K-active).sum())
                if mode==5:
                    service+=mult*int((2*active+Q+J+3*(L+H)+K).sum());updates+=mult*int((L+H).sum())
                else:
                    service+=mult*int((3*active+Q+J+3*L+H).sum());updates+=mult*int(L.sum())
    return dict(core_cycles=n*(1442+112*alive+2*(24-alive))+service,
        core_source_words=4*alive*valid_total,core_weight_words=W,core_sum_issues=Jtotal+(Htotal if mode==6 else 0),
        core_update_issues=updates,core_merge_issues=Htotal if mode==6 else 0,
        core_psum_reads=updates+480*n,core_psum_writes=updates+480*n,
        source_load_words=1536*n,external_source_words=96*valid_total,padding_words=1536*n-96*valid_total,
        origin_words=n,output_beats=480*n,retired_tiles=n,checked_outputs=3840*n),2*Htotal+terminal

def main():
    rows=json.loads((RTL/'results64.json').read_text())+json.loads((RTL/'results_full.json').read_text())
    assert len(rows)==30,'Wait for all six full-frame jobs'
    cache={};checks=0
    for r in rows:
        key=(r['arm'],r['mode'],r['first_tile'],r['tiles'])
        if key not in cache:
            live=np.load(HERE/(r['arm']+'_weights.npz'))['live']
            cache[key]=predict(r['first_tile'],r['tiles'],live,r['mode'])
        p,delta=cache[key]
        for field,expected in p.items():
            observed=r[field]
            if field=='core_cycles':observed-=sum(r[k] for k in ('core_source_stalls','core_weight_stalls','core_output_stalls'))
            assert observed==expected,(key,field,observed,expected);checks+=1
        cold=10656 if r['command']==0 else 0
        stall=sum(r[k] for k in ('parameter_stalls','source_load_stalls','core_source_stalls','core_weight_stalls','core_output_stalls'))
        assert r['total_cycles']==p['core_cycles']+1539*r['tiles']+1+cold+stall;checks+=1
        assert r['static_weight_words']==(10368 if cold else 0) and r['static_mask_words']==(288 if cold else 0);checks+=2
    out=dict(complete=True,RTL_rerun=False,runs=len(rows),checked_outputs=sum(r['checked_outputs'] for r in rows),independent_scalar_checks=checks,
        all_match=True,predictions=[dict(arm=k[0],mode=k[1],first=k[2],tiles=k[3],**v[0],mode5_minus_mode6=v[1]) for k,v in cache.items()])
    (HERE/'review_stream_counts.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='predictions'},indent=2))
if __name__=='__main__':main()
