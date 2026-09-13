"""Independent closed-form replay audit. No imported RTL author's model."""
import json
from pathlib import Path
from model_access import HERE
RTL=HERE.parent/'pair_parent_merge'

def readhex(p):return [int(t,16) for t in p.read_text().split()]
def predict(path):
    source=[v&1023 for v in readhex(path/'source.hex')];live=readhex(path/'mask.hex')
    oy,ox=[v&65535 for v in readhex(path/'origin.hex')]
    oy=oy-65536 if oy&32768 else oy;ox=ox-65536 if ox&32768 else ox
    m={i:dict(cycles=1442,source_words=0,weight_words=0,update_issues=0,sum_issues=0,pattern_copies=0,merge_issues=0) for i in (3,4,5,6)}
    overlap=0;proof_delta=0;merge_delta=0
    for cg in range(24):
        G=sum(bool(live[og*24+cg]) for og in range(12))
        for mode in m:m[mode]['cycles']+=2 if G==0 else (160 if mode==3 else 112)
        if not G:continue
        for s in range(16):
            if not(0<=oy+s//4<240 and 0<=ox+s%4<320):continue
            a,b,c,d=[source[(4*cg+i)*16+s] for i in range(4)]
            for v in m.values():v['source_words']+=4
            if not(a|b|c|d):continue
            P=sum(0<=s//4-p//2<3 and 0<=s%4-p%2<3 for p in range(4));dest=G*P
            u,v=a|b,c|d;K=int(bool(u))+int(bool(v));J=int(bool(a&b))+int(bool(c&d))
            L3=u.bit_count()+v.bit_count();L4=(u|v).bit_count();H=(u&v).bit_count()
            Q=sum(bool(w) for w in (a,b,c,d))
            patterns={sum(((w>>t)&1)<<i for i,w in enumerate((a,b,c,d))) for t in range(10)}-{0}
            D=len(patterns);B=sum(k.bit_count()-1 for k in patterns)
            for mode in (3,4,5,6):
                r=m[mode];r['weight_words']+=dest*Q
                r['update_issues']+=dest*(L4 if mode in (4,6) else L3)
                r['sum_issues']+=dest*(B if mode==4 else J+H if mode==6 else J)
                r['pattern_copies']+=dest*(D if mode==4 else 0)
                r['merge_issues']+=dest*(H if mode==6 else 0)
            m[3]['cycles']+=dest*(3*K+Q+J+3*L3)
            m[5]['cycles']+=dest*(2+Q+J+3*L3+K)
            m[4]['cycles']+=dest*(2+Q+2*D+B+3*L4)
            m[6]['cycles']+=dest*(3+Q+J+3*L4+H)
            overlap+=dest*H;proof_delta+=dest*(3*H+J+K-2*D-B);merge_delta+=dest*(2*H+K-1)
    for r in m.values():
        r['psum_reads']=r['psum_writes']=r['update_issues']+480
        r['control_cycles']=r['cycles']-r['source_words']-r['weight_words']-r['psum_reads']-r['psum_writes']-r['sum_issues']-r['pattern_copies']-480
    assert m[5]['cycles']-m[4]['cycles']==proof_delta
    assert m[5]['update_issues']-m[4]['update_issues']==overlap
    assert m[5]['cycles']-m[6]['cycles']==merge_delta
    return m,overlap,merge_delta

def main():
    rows=json.loads((RTL/'results.json').read_text());cache={};checks=0
    for r in rows:
        path=Path(r['fixture_path'])
        if str(path) not in cache:cache[str(path)]=predict(path)
        pred=cache[str(path)][0][r['mode']]
        for key,expected in pred.items():
            observed=r[key]
            if key=='cycles':observed-=r['source_stalls']+r['weight_stalls']+r['output_stalls']
            assert observed==expected,(r['fixture'],r['mode'],r['stall'],r['command'],key,observed,expected)
            checks+=1
    previous=json.loads((HERE.parent/'rtl/results.json').read_text())
    keys=('fixture','mode','stall','command')
    index={tuple(r[k] for k in keys):r for r in previous}
    retained_checks=0
    for r in rows:
        if r['mode']!=5:continue
        old=index.get(tuple(r[k] for k in keys))
        assert old is not None,('missing_retained_mode5',r['fixture'])
        for key in ('cycles','outputs','source_words','weight_words','update_issues','sum_issues','pattern_copies','psum_reads','psum_writes','control_cycles','source_stalls','weight_stalls','output_stalls'):
            assert old[key]==r[key],('retained_mode5',r['fixture'],key,old[key],r[key])
            retained_checks+=1
    report=dict(complete=True,RTL_rerun=False,fixtures=len(cache),commands=len(rows),checked_outputs=sum(r['outputs'] for r in rows),counter_checks=checks,retained_mode5_counter_checks=retained_checks,
        all_match=True,formula='For eachlive destination: m5-m6=2H+K-1;H=pairunion overlap popcount,K=activepaircount. J ordinary parent sums, H paid merges; no pattern copies. All source/W work equal.',
        rows=[dict(fixture=Path(k).name,prediction=v[0],overlap_updates=v[1],mode5_minus_mode6_core=v[2]) for k,v in cache.items()])
    (HERE/'review_parent_merge_counters.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
