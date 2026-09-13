"""Independent closed-form replay audit. No imported RTL author's model."""
import json
from pathlib import Path
from model_access import HERE
RTL=HERE.parent/'rtl'

def readhex(p):return [int(t,16) for t in p.read_text().split()]
def predict(path):
    source=[v&1023 for v in readhex(path/'source.hex')];live=readhex(path/'mask.hex')
    oy,ox=[v&65535 for v in readhex(path/'origin.hex')]
    oy=oy-65536 if oy&32768 else oy;ox=ox-65536 if ox&32768 else ox
    m={i:dict(cycles=1442,source_words=0,weight_words=0,update_issues=0,sum_issues=0,pattern_copies=0) for i in (3,4,5)}
    overlap=0;proof_delta=0
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
            for mode in (3,4,5):
                r=m[mode];r['weight_words']+=dest*Q
                r['update_issues']+=dest*(L4 if mode==4 else L3)
                r['sum_issues']+=dest*(B if mode==4 else J)
                r['pattern_copies']+=dest*(D if mode==4 else 0)
            m[3]['cycles']+=dest*(3*K+Q+J+3*L3)
            m[5]['cycles']+=dest*(2+Q+J+3*L3+K)
            m[4]['cycles']+=dest*(2+Q+2*D+B+3*L4)
            overlap+=dest*H;proof_delta+=dest*(3*H+J+K-2*D-B)
    for r in m.values():
        r['psum_reads']=r['psum_writes']=r['update_issues']+480
        r['control_cycles']=r['cycles']-r['source_words']-r['weight_words']-r['psum_reads']-r['psum_writes']-r['sum_issues']-r['pattern_copies']-480
    assert m[5]['cycles']-m[4]['cycles']==proof_delta
    assert m[5]['update_issues']-m[4]['update_issues']==overlap
    return m,overlap,proof_delta

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
    report=dict(complete=True,RTL_rerun=False,fixtures=len(cache),commands=len(rows),checked_outputs=sum(r['outputs'] for r in rows),counter_checks=checks,
        all_match=True,formula='For eachlive destination: m5-m4=3H+J+K-2D-B;H=pairunion overlap popcount,J=pairAND-nonzero count,K=activepaircount,D=distinct nonzero4bitcodes,B=sum(popcount(code)-1).All source/W work equal.',
        rows=[dict(fixture=Path(k).name,prediction=v[0],overlap_updates=v[1],mode5_minus_mode4_core=v[2]) for k,v in cache.items()])
    (HERE/'review_c4_counters.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
