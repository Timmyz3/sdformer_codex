#!/usr/bin/env python3.12
"""Second-wave, fixed-cohort screens after negative small-window results.

Bounded online C1 anchors use no backing store and no future row information.
C2 mixed-width merge counts only slots after producer completion, and is an
optimistic service-intent bound until the wider reduction tree is implemented.
"""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import screen_reuse as reuse


def anchors(masks,slots,difference,order,policy='lru',weights=None):
    cache={};count=Counter();vectors={};computed=set()
    for tick,row in enumerate(order):
        m=int(masks[row])
        if not m:
            count['zero_rows']+=1
            continue
        # One port cycle to read an owned partial is conservatively separate.
        # The operation-count view is also retained independently.
        best=(m.bit_count(),0,-1)
        for p in cache:
            pm=int(masks[p])
            if difference or pm&m==pm:
                cost=max(1,(pm^m).bit_count())
                key=(cost+1,1,p)
                if key<best:best=key
        p=best[2]
        work=m.bit_count() if p<0 else max(1,(int(masks[p])^m).bit_count())
        count['source_issue_slots']+=work
        count['serialized_read_plus_issue_slots']+=best[0]
        if p>=0:
            assert p in computed
            count['anchor_hits']+=1
            count['anchor_reads']+=1
            count['negative_corrections']+=(int(masks[p])&~m&65535).bit_count()
            cache[p]=tick
        else:count['zero_root_rows']+=1
        if weights is not None:
            value=np.zeros(96,dtype=np.int64) if p<0 else vectors[p].copy()
            pm=int(masks[p]) if p>=0 else 0
            for b in range(16):
                if pm>>b&1 and not m>>b&1:value-=weights[b]
            for b in range(16):
                if m>>b&1 and not pm>>b&1:value+=weights[b]
            expected=sum((weights[b] for b in range(16) if m>>b&1),np.zeros(96,dtype=np.int64))
            assert np.array_equal(value,expected)
            assert np.all((-2048<=value)&(value<=2032))
            count['checked_outputs']+=96
        if len(cache)==slots:
            if policy=='lru':victim=min(cache,key=lambda r:(cache[r],r))
            else:
                # Preserve a costly-to-recompute anchor; no future information.
                victim=min(cache,key=lambda r:(int(masks[r]).bit_count(),cache[r],r))
            del cache[victim]
            if weights is not None:del vectors[victim]
            count['discarded_anchors']+=1
        cache[row]=tick
        if weights is not None:vectors[row]=value
        computed.add(row)
        count['anchor_writes']+=1
        count['active_rows']+=1
        assert len(cache)<=slots
    count['tiles']=1
    return count


def run_anchors():
    from m2260_c1_hot_parent_probe import threaded_order,serve_hot
    totals={};rows=[]
    rng=np.random.default_rng(260906)
    weights=rng.integers(-128,128,size=(16,96),dtype=np.int64)
    weights[:,0]=-128;weights[:,1]=127
    with reuse.c1.LEDGER.open('rb') as stream:
        for sample in range(10):
            for op in range(4):
                for k in (0,216,431):
                    phase=(sample*4+op)*432+k
                    for chunk in (0,15,31,46):
                        n=min(64,3000-chunk*64)
                        stream.seek((phase*3000+chunk*64)*9)
                        masks=np.array([int(v,16)&65535 for v in stream.read(n*9).splitlines()],dtype=np.uint16)
                        res,par=reuse.c1.old.M504.cleanroom_subset(masks)
                        stable,_=reuse.c1.forest_orders(masks,par)
                        dfs,build=threaded_order(masks,par)
                        pts={}
                        for ordername,order in (('native',list(range(n))),('stable',stable),('threaded',dfs)):
                            for difference in (False,True):
                                for slots in (2,4):
                                    name=f'{ordername}_{"diff" if difference else "subset"}_{slots}'
                                    out=anchors(masks,slots,difference,order,weights=weights)
                                    out['charged_order_build_slots']=build if ordername=='threaded' else (n if ordername=='stable' else 0)
                                    pts[name]=dict(out)
                                    totals.setdefault(name,Counter()).update(out)
                        ref=reuse.c1.serve(masks,res,par,stable)
                        hot=serve_hot(masks,res,par,dfs,2)
                        pts['existing_1rw_service']=ref
                        pts['threaded_hot2_with_backing']=hot
                        totals.setdefault('existing_1rw_service',Counter()).update(ref)
                        totals.setdefault('threaded_hot2_with_backing',Counter()).update(hot)
                        totals.setdefault('zero_skip',Counter()).update(source_issue_slots=sum(int(m).bit_count() for m in masks),active_rows=sum(bool(m) for m in masks))
                        rows.append(dict(sample=sample,operator=op,partition=k,chunk=chunk,points=pts))
    return dict(scope='Bounded online two/four-anchor CPU feasibility screen; no backing SRAM in the algorithm, not a mapped macro-removal result',
        cohort='Same 480 fixed tiles as first wave',totals=totals,rows=rows,
        boundaries=['Native order only reads already arrived rows; stable/threaded variants require separately charged planning',
          'Anchor selection has 2/4 candidates, not free 64-way matching; scalar selector latency still unimplemented',
          'Every active result is admitted, least-recently-used replaced; cost-aware admission remains untested',
          'One extra anchor-read slot is charged; owned-slot write assumed overlaps output finalization, must be implemented',
          'No backing, recursive replay or repeated architectural commits; miss directly computes current row from zero',
          'Two/four payload slots cost 288/576B; cannot claim removal of SRAM area before matched synthesis',
          'Legacy two-credit cycle model is a contextual reference, not cycle-equivalent to the new serial service-intent model',
          'Input/weight fetching, output backpressure, slot timing and new control costs remain open',
          'Diagnostic natural-binary x signed INT8 arithmetic only; no full-network FP equivalence claim'])


def merged_cost(x,selection,codes,fixed_ports=False):
    chosen=set(selection)
    gcount=x.shape[1]//8
    last={m:max(i//8 for i,c in enumerate(codes) if c==m) for m in selection}
    pending={m:set(d for d in range(4) if m>>d&1) for m in selection}
    classes=bypass=merged=0
    for g in range(gcount):
        cg=codes[g*8:g*8+8]
        classes+=len(set(int(c) for c in cg)&chosen)
        # Schedule selected class updates before destination bypass updates.
        # Last class update must be committed before such a merge can issue.
        for d in range(4):
            sources=sum(bool(int(c)>>d&1) and int(c) not in chosen for c in cg)
            if not sources:continue
            bypass+=1
            free=(sum(not (int(cg[b])>>d&1 and int(cg[b]) not in chosen) for b in (0,1))
                  if fixed_ports else 8-sources)
            ready=[m for m in selection if d in pending[m] and last[m]<=g]
            for m in ready[:free]:
                pending[m].remove(d)
                merged+=1
    scatter=sum(len(d) for d in pending.values())
    return classes+bypass+scatter,merged,scatter


def run_merge():
    HW=reuse.HW; fixtures=HW/'tb_m2018/fixtures'
    cp=HW/'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901/layers.json'
    layers={r['layer_id']:r for r in json.loads(cp.read_text())['layers']}
    totals={};rows=[];count=0
    for prefix,extent in (('m2051_ep34_tsbg_full40_s1920',48),('m2067_ep34_fc2_exact_continuation_s960',192)):
        meta=json.loads((fixtures/(prefix+'.json')).read_text())
        words=np.array([int(v,16)&65535 for v in (fixtures/(prefix+'.memh')).read_text().split()],dtype=np.uint16).reshape(len(meta['rows']),4,extent)
        for row in meta['rows']:
            assert row['negative_codes']==0
            target=row['target'] if extent==48 else 'FC2'
            mult=layers[row['layer_id']]['weight_layout']['output_tile_count']
            x=reuse.prior.bits(words[row['slot'],:,:int(row['source_groups'])])
            for window in (16,64,768):
                key=f'{target}_window{window}';c=Counter()
                for start in range(0,x.shape[1],window):
                    xx=x[:,start:start+window]
                    codes=(xx.astype(np.int64)*np.array([1,2,4,8])[:,None]).sum(0)
                    hist=np.bincount(codes,minlength=16)
                    eligible=[m for m in range(1,16) if m.bit_count()>=2 and hist[m]>=2]
                    ordinary=int((xx.reshape(4,-1,8).sum(2)>0).sum())
                    best=(ordinary,(),0,0)
                    points=[]
                    for cap in (1,2):
                        for sel in combinations(eligible,cap):
                            cost,merged,scatter=merged_cost(xx,sel,codes)
                            if (cost,len(sel),sel)<(best[0],len(best[1]),best[1]):best=(cost,sel,merged,scatter)
                        points.append(best)
                    c.update(windows=1,ordinary=ordinary,source_columns=xx.shape[1],bank_groups=xx.shape[1]//8)
                    for cap,b in enumerate(points,1):
                        c.update({f'cap{cap}_intents':b[0],f'cap{cap}_merged_scatters':b[2],f'cap{cap}_remaining_scatters':b[3],f'cap{cap}_selected':len(b[1]),f'cap{cap}_winning_windows':int(b[0]<ordinary)})
                totals.setdefault(key,Counter()).update({k:v*mult for k,v in c.items()})
                rows.append(dict(target=target,window=window,sequence=row['sequence'],slot=row['slot'],output_tiles=mult,counts=c))
            count+=1
            if count%480==0:print(f'merge {count}/2880',flush=True)
    for t in totals.values():
        for cap in (1,2):
            t[f'cap{cap}_intent_reduction']=1-t[f'cap{cap}_intents']/max(1,t['ordinary'])
            t[f'cap{cap}_selector_budget_per_bank_group']=(t['ordinary']-t[f'cap{cap}_intents'])/max(1,t['bank_groups'])
    return dict(scope='Optimistic C2 mixed-width fusion intent screen; not cycles or measured hardware',workloads=count,
        weighted_by_output_tiles=totals,rows=rows,
        boundaries=['Same 2880 frozen workload templates; natural +1 source only',
          'Only 1/2 partials, at most 1/2 Acc24 inputs per bypass reduction, sharing a fixed total of eight input positions',
          'All selected class updates precede same-bank-group bypasses; producer completion precedes merge',
          'A partial can merge only into a later-or-same-group nonempty bypass for its own consumer',
          'No invented bypass issue just to hide scatter; remaining consumers require separate charged scatter',
          'Critical unpriced cost is mixed-width 8-input tree and wider operand ports, in both fair physical axes',
          'Selection is exhaustive over at most 2 classes, not measured causal hardware classifier',
          'No extra SRAM request savings beyond TSBG; no merging across token, layer or continuation commit boundaries',
          'Arithmetic exactness follows disjoint source groups with sufficient Acc24 range; timed numeric RTL validation still required'])


def main():
    p=argparse.ArgumentParser();p.add_argument('axis',choices=('anchors','merge'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run_anchors() if a.axis=='anchors' else run_merge()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
