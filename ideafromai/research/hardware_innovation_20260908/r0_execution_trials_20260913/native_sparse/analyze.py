#!/usr/bin/env python3
"""Independent native-coordinate transaction ledger, not a timing substitute.

Results remain measured Verilator cycles. The native domain and finite FSM
service counts independently predict them, including explicit observed stalls.
"""
import json
import argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def readhex(p):
    return np.array([int(s,16) for s in p.read_text().split()],dtype=np.uint32)

def expected(path):
    src=readhex(path/'source.hex').reshape(96,4,4)
    origin=readhex(path/'origin.hex').astype(np.int32)
    valid=((np.arange(4)+origin[0]>=0)&(np.arange(4)+origin[0]<240))[:,None] & ((np.arange(4)+origin[1]>=0)&(np.arange(4)+origin[1]<320))[None,:]
    src=src*valid[None,:,:]
    mask=readhex(path/'mask.hex').reshape(12,24).astype(bool)
    words_out=0;words_weight=0;updates=0;sums_out=0;sums_weight=0
    active_out=0;active_weight=0;source_repeated=0
    for og in range(12):
        for cp in range(48):
            if not mask[og,cp//2]:continue
            for ky in range(3):
                for kx in range(3):
                    aa=[int(src[2*cp,py+ky,px+kx]) for py in range(2) for px in range(2)]
                    bb=[int(src[2*cp+1,py+ky,px+kx]) for py in range(2) for px in range(2)]
                    words_weight+=int(any(aa))+int(any(bb))
                    sums_weight+=int(any(a&b for a,b in zip(aa,bb)))
                    active_weight+=int(any(a|b for a,b in zip(aa,bb)))
                    for a,b in zip(aa,bb):
                        words_out+=int(a!=0)+int(b!=0)
                        sums_out+=int((a&b)!=0)
                        active_out+=int((a|b)!=0)
                        updates+=(a|b).bit_count()
                    source_repeated+=2*sum(valid[py+ky,px+kx] for py in range(2) for px in range(2))
    live_blocks=int(mask.sum());dead_blocks=288-live_blocks
    live_cgroups=int(mask.any(axis=0).sum())
    native_dest_control=0
    for cp in range(48):
        live_og=int(mask[:,cp//2].sum())
        if not live_og:continue
        for y in range(4):
            for x in range(4):
                if not (int(src[2*cp,y,x])|int(src[2*cp+1,y,x])):continue
                valid_p=sum(0<=y-py<3 and 0<=x-px<3 for py in range(2) for px in range(2))
                native_dest_control+=live_og*(8+valid_p)+(12-live_og)
    # 480 clear + INIT + 480 drain-read + 480 send + FINISH.
    base=1442
    common={'psum_reads':updates+480,'psum_writes':updates+480,'update_issues':updates}
    rows=[]
    for mode in range(3):
        r=dict(common)
        if mode==0:
            live_contexts=live_blocks*72
            r.update(source_words=int(source_repeated),weight_words=words_out,sum_issues=sums_out)
            cyc=base+5*live_contexts+active_out+words_out+sums_out+3*updates+2*dead_blocks*4
        elif mode==1:
            source_pairs=live_cgroups*32
            r.update(source_words=int(live_cgroups*4*valid.sum()),weight_words=words_out,sum_issues=sums_out)
            cyc=base+5*source_pairs+native_dest_control+words_out+sums_out+3*updates+2*(24-live_cgroups)
        else:
            live_contexts=live_blocks*18
            r.update(source_words=int(source_repeated),weight_words=words_weight,sum_issues=sums_weight)
            cyc=base+11*live_contexts+4*active_weight+words_weight+sums_weight+3*updates+2*dead_blocks
        r['cycles_without_stalls']=cyc
        rows.append(r)
    return rows

def analyze(result_file):
    results=json.loads((HERE/result_file).read_text());ledger={}
    for row in results:
        fixture=row['fixture']
        if fixture not in ledger:ledger[fixture]=expected(HERE/'fixtures'/fixture)
        pred=ledger[fixture][row['mode']]
        for key in ('source_words','weight_words','sum_issues','update_issues','psum_reads','psum_writes'):
            assert row[key]==pred[key],(fixture,row['mode'],key,row[key],pred[key])
        assert row['cycles']==pred['cycles_without_stalls']+row['source_stalls']+row['weight_stalls']+row['output_stalls'],(fixture,row)
    output=Path(result_file).stem.replace('_results','')+'_ledger.json'
    (HERE/output).write_text(json.dumps({'result_file':result_file,'runs_checked':len(results),'ledger':ledger},indent=2)+'\n')
    print(json.dumps({'ledger':output,'runs_checked':len(results),'all_counts_and_cycles_match':True}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('results',nargs='?',default='control_results.json')
    analyze(parser.parse_args().results)
