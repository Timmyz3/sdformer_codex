"""Independent state-difference audit; reads RTL results without rerunning it."""
import json
from pathlib import Path
from model_access import HERE
NATIVE=HERE.parent/'native_sparse';ENUM=HERE.parent/'consumer_enumeration'

def readhex(path):return [int(t,16) for t in path.read_text().split()]
def expected(fixture):
    d=NATIVE/'fixtures'/fixture
    source=readhex(d/'source.hex');live=readhex(d/'mask.hex');oy,ox=[v&65535 for v in readhex(d/'origin.hex')]
    oy=oy-65536 if oy&32768 else oy;ox=ox-65536 if ox&32768 else ox
    saved=0;only_B_reads=0;active_pairs=0
    for cp in range(48):
        g=sum(bool(live[og*24+cp//2]) for og in range(12))
        if not g:continue
        for s in range(16):
            if not(0<=oy+s//4<240 and 0<=ox+s%4<320):continue
            a,b=source[(cp*2)*16+s],source[(cp*2+1)*16+s]
            if not(a|b):continue
            legal=sum(0<=s//4-p//2<3 and 0<=s%4-p%2<3 for p in range(4))
            active_pairs+=1
            saved+=12+7*g-2*g*legal
            if not a and b:only_B_reads+=g*legal
    return dict(expected_core_cycle_saving_without_stalls=saved,
                extra_W_reads_in_initial_buggy_DEST_PICK=only_B_reads,active_pair_source_words=active_pairs)

def main():
    rs=json.loads((ENUM/'results.json').read_text())
    names=sorted({r['fixture'] for r in rs});out=[]
    for name in names:
        e=expected(name)
        by={(r['mode'],r['stall'],r['command']):r for r in rs if r['fixture']==name}
        r1,r3=by[1,0,0],by[3,0,0]
        delta=r1['cycles']-r3['cycles']
        assert delta==e['expected_core_cycle_saving_without_stalls'],(name,delta,e)
        for stall in (0,1):
            for command in (0,1):
                a,b=by[1,stall,command],by[3,stall,command]
                work=['source_words','weight_words','psum_reads','psum_writes','sum_issues','update_issues','outputs']
                assert all(a[k]==b[k] for k in work),(name,stall,command)
                base=a['cycles']-a['source_stalls']-a['weight_stalls']-a['output_stalls']
                changed=b['cycles']-b['source_stalls']-b['weight_stalls']-b['output_stalls']
                assert base-changed==e['expected_core_cycle_saving_without_stalls'],(name,stall,command)
        out.append(dict(fixture=name,**e,observed_core_cycle_saving_without_stalls=delta))
    result=dict(complete=True,fixtures=len(out),rows=out,
        rule='For each active inbounds pair-source andG live Ogroups, old NATIVE_DEST+ADVANCE=12+7G states; new DEST_PICK+ADVANCE=2G*number_geometric_consumers. Sum differences; all other work equal.',
        all_observed_deltas_match=True,all_work_counters_equal=True,RTL_not_rerun=True)
    (HERE/'review_enumeration_counters.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
