#!/usr/bin/env python3.12
"""Fixed subset forest, discardable result slots, zero-root fallback on miss.
Counts arithmetic and storage intents, not cycles. Same forest and traversal
on both backing/no-backing axes. No extra response payload queue is assumed.
"""
import sys
sys.dont_write_bytecode=True
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import screen_reuse as s
from m2260_c1_hot_parent_probe import threaded_order


def run(masks, parent, residual, order, slots, backing, weights):
    remaining=Counter(int(parent[r]) for r in order if parent[r]>=0)
    hot={};memory={};born={};count=Counter()
    for tick,row in enumerate(order):
        p=int(parent[row]);value=np.zeros(96,dtype=np.int64)
        used_parent=p>=0 and (p in hot or (backing and p in memory))
        if used_parent:
            if p in hot:
                value=hot[p].copy();count['hot_reads']+=1
            else:
                value=memory[p].copy();count['backing_reads']+=1
            work=int(residual[row])
        else:
            work=int(masks[row])
            if p>=0:
                count['missing_parent_rows']+=1
                count['extra_source_issues_from_miss']+=work.bit_count()-max(1,int(residual[row]).bit_count())
        count['source_issue_slots']+=max(work.bit_count(),int(used_parent))
        for b in range(16):
            if work>>b&1:value+=weights[b]
        ref=sum((weights[b] for b in range(16) if int(masks[row])>>b&1),np.zeros(96,dtype=np.int64))
        assert np.array_equal(value,ref)
        assert np.all((-2048<=value)&(value<=2032))
        count['checked_outputs']+=96
        count['architectural_row_commits']+=1
        if p>=0:
            remaining[p]-=1
            if not remaining[p] and p in hot:
                del hot[p];del born[p];count['last_consumer_releases']+=1
        if remaining[row]:
            if len(hot)==slots:
                victim=min(born,key=lambda r:(born[r],r))
                if backing and victim not in memory:
                    memory[victim]=hot[victim].copy();count['backing_writes']+=1
                del hot[victim];del born[victim];count['evictions']+=1
            hot[row]=value.copy();born[row]=tick;count['hot_writes']+=1
        assert len(hot)<=slots
    assert not hot and not any(remaining.values())
    count['tiles']=1
    return count


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--all-spatial',action='store_true',help='All 47 spatial chunks in each of the same 120 K phases')
    a=ap.parse_args()
    rng=np.random.default_rng(260906);w=rng.integers(-128,128,(16,96),dtype=np.int64);w[:,0]=-128;w[:,1]=127
    totals={};rows=[]
    with s.c1.LEDGER.open('rb') as f:
        for sample in range(10):
            for op in range(4):
                for k in (0,216,431):
                    phase=(sample*4+op)*432+k
                    for chunk in (range(47) if a.all_spatial else (0,15,31,46)):
                        n=min(64,3000-chunk*64);f.seek((phase*3000+chunk*64)*9)
                        m=np.array([int(v,16)&65535 for v in f.read(n*9).splitlines()],dtype=np.uint16)
                        r,p=s.c1.old.M504.cleanroom_subset(m);stable,_=s.c1.forest_orders(m,p);dfs,build=threaded_order(m,p)
                        pts={}
                        for ordername,order in (('stable',stable),('threaded',dfs)):
                            for slots in (1,2,4):
                                for backing in (False,True):
                                    name=f'{ordername}_{slots}_{"backing" if backing else "recompute"}'
                                    out=run(m,p,r,order,slots,backing,w)
                                    out['dfs_build_reference']=build if ordername=='threaded' else 0
                                    pts[name]=dict(out);totals.setdefault(name,Counter()).update(out)
                        rows.append(dict(sample=sample,operator=op,partition=k,chunk=chunk,points=pts))
            if a.all_spatial:print(f'forest {sample+1}/10 samples',flush=True)
    comparisons={}
    for name,c in totals.items():
        if name.endswith('_recompute'):
            ref=totals[name.replace('_recompute','_backing')]
            comparisons[name]=dict(source_issue_increase=c['source_issue_slots']/ref['source_issue_slots']-1,
                eliminated_backing_accesses=ref['backing_reads']+ref['backing_writes'],
                missing_parent_rows=c['missing_parent_rows'],
                zero_extra_architectural_commits=c['architectural_row_commits']==ref['architectural_row_commits'])
    output=dict(scope='Fixed-forest backing/rematerialization operation-count screen, not cycles/PPA',
        cohort=('5640 ep34 C1 tiles: 10 samples x 4 Conv x 3 preselected K phases x all 47 spatial chunks'
                if a.all_spatial else '480 preselected ep34 C1 tiles'),
        source=str(s.c1.LEDGER),tiles=len(rows),totals=totals,comparisons=comparisons,rows=rows,
        boundaries=['Identical forest, order, FIFO replacement, remaining-consumer policy and slot count per pair',
          'No-backing miss recomputes current output directly from zero, never recursively commits a parent',
          'Hot slots only, no extra response payload queue; port timing, simultaneous use and writeback not validated',
          'DFS and subset-matching work remains required; this is not the no-matcher online-anchor design',
          'Every added source issue also needs weight delivery; count this before energy or throughput claims',
          'Worst-case work returns to direct zero-skip; fixed 16-source task bounds local work',
          'Actual macro removal, control overhead and area/energy require a matched new RTL implementation'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
