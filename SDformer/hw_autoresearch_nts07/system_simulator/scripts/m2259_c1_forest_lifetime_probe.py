#!/usr/bin/env python3
"""Fixed-forest scheduling probe: no new subset relation or hardware claim.

Preselect 480 tiles over all 10 samples / four Conv operators, three K
partitions and four spatial positions. Preserve the M504 parent/residual
mapping. Reuse the same two-credit, 1RW service rules and charge a separate
64-cycle successor-order construction sensitivity rather than free scheduling.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import analyze_m505_h67_liveness_aware_single_port_parent_scratch as old

HW=Path(__file__).resolve().parents[2]
LEDGER=HW/'results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901/ep34_c1_support16_rows.memh'


def forest_orders(masks,parent):
    active=[i for i,m in enumerate(masks) if m]
    stable=sorted(active,key=lambda i:(int(masks[i]).bit_count(),i))
    children={i:[] for i in active}
    roots=[]
    for i in active:
        if parent[i]<0: roots.append(i)
        else: children[int(parent[i])].append(i)
    # A deterministic depth-first traversal: children with smaller descendant
    # trees first, large subtree last. Not a claim of optimal register pressure.
    def size(i): return 1+sum(size(j) for j in children[i])
    dfs=[]
    def visit(i):
        dfs.append(i)
        for j in sorted(children[i],key=lambda j:(size(j),j)): visit(j)
    for root in sorted(roots): visit(root)
    return stable,dfs


def live_metrics(order,parent):
    remaining=Counter(int(parent[i]) for i in order if parent[i]>=0)
    live=set(); peak=0; integral=0; born={}; spans=[]
    for tick,row in enumerate(order):
        p=int(parent[row])
        if p>=0:
            assert p in live
            remaining[p]-=1
            if remaining[p]==0:
                live.remove(p); spans.append(tick-born[p])
        if remaining[row]: live.add(row); born[row]=tick
        peak=max(peak,len(live)); integral+=len(live)
    assert not live
    return dict(peak_live_parent_vectors=peak,live_vector_row_steps=integral,
                longest_parent_span=max(spans,default=0))


def serve(masks,residual,parent,order,elide_single_use=False):
    requirements=[int(parent[i]) for i in order if parent[i]>=0]
    consumers=[k for k,i in enumerate(order) if parent[i]>=0]
    uses=Counter(requirements)
    work={i:max(int(residual[i]).bit_count(),int(parent[i]>=0)) for i in order}
    queue=[]; pending=None; request=0; cursor=0; beat=0; written=set()
    count=Counter({key:0 for key in ('cycles','issues','stalls','reads','writes',
        'forwards','deadline_holds','single_use_write_elisions')})
    while cursor<len(order):
        row=order[cursor]; p=int(parent[row]); needed=work[row]
        ready=p<0 or bool(queue and queue[0]==p)
        final=ready and beat+1==needed
        capacity=len(queue)+int(pending is not None)<2
        exists=request<len(requirements)
        asked=requirements[request] if exists else -1
        consumer=consumers[request] if exists else -1
        predicted_forward=final and exists and capacity and asked==row
        elided=int(predicted_forward and elide_single_use and consumer==cursor+1)
        will_write=final and uses[row]>elided
        hold=will_write and exists and capacity and asked in written and asked!=row and consumer==cursor+1
        issue=ready and not hold
        last=issue and beat+1==needed
        forward=last and exists and capacity and asked==row
        elided=int(forward and elide_single_use and consumer==cursor+1)
        write=last and uses[row]>elided
        read=not write and not forward and exists and capacity and asked in written
        if last and p>=0:
            assert queue and queue.pop(0)==p
        if pending is not None: queue.append(pending)
        if forward: queue.append(asked); request+=1
        if read: request+=1
        pending=asked if read else None
        assert len(queue)+int(pending is not None)<=2
        if write: written.add(row)
        count.update(cycles=1,issues=int(issue),stalls=int(not issue),
            reads=int(read),writes=int(write),forwards=int(forward),
            deadline_holds=int(hold),single_use_write_elisions=int(last and not write and uses[row]>0))
        if issue:
            if last: cursor+=1; beat=0
            else: beat+=1
        assert count['cycles']<=sum(work.values())+2*len(order)+8
    assert request==len(requirements) and not queue and pending is None
    assert count['issues']==sum(work.values())
    return dict(count)


def numeric(masks,residual,parent,order):
    weights=np.arange(16*16,dtype=np.int64).reshape(16,16)%255-127
    vectors={}
    for row in order:
        p=int(parent[row]); value=np.zeros(16,dtype=np.int64) if p<0 else vectors[p].copy()
        for bit in range(16):
            if int(residual[row])>>bit&1: value+=weights[bit]
        dense=sum((weights[bit] for bit in range(16) if int(masks[row])>>bit&1),np.zeros(16,dtype=np.int64))
        assert np.array_equal(value,dense)
        vectors[row]=value


def self_test():
    cases=[[1,3,7,15],[3,3,3,3],[1,2,3,4,5,7,15,0]]
    rng=np.random.default_rng(2259)
    cases.extend(rng.integers(0,65536,size=64,dtype=np.uint16) for _ in range(16))
    for raw in cases:
        masks=np.asarray(raw,dtype=np.uint16)
        residual,parent=old.M504.cleanroom_subset(masks)
        stable,dfs=forest_orders(masks,parent)
        for elide in (False,True):
            expected=old.simulate_liveness_task(masks,elide)
            actual=serve(masks,residual,parent,stable,elide)
            for a,b in [('cycles','liveness_cycles'),('issues','ideal_1r1w_issue_cycles'),
                    ('reads','macro_reads'),('writes','macro_writes'),('forwards','forwarded_reads')]:
                assert actual[a]==expected[b],(a,actual,expected)
        numeric(masks,residual,parent,dfs)
    return len(cases)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    tested=self_test(); rows=[]; totals={}
    with LEDGER.open('rb') as stream:
        for sample in range(10):
            for operator in range(4):
                for partition in (0,216,431):
                    phase=(sample*4+operator)*432+partition
                    for chunk in (0,15,31,46):
                        n=min(64,3000-chunk*64)
                        stream.seek((phase*3000+chunk*64)*9)
                        masks=np.array([int(v,16)&65535 for v in stream.read(n*9).splitlines()],dtype=np.uint16)
                        residual,parent=old.M504.cleanroom_subset(masks)
                        stable,dfs=forest_orders(masks,parent)
                        numeric(masks,residual,parent,dfs)
                        bitcap=(n+7)//8
                        pre=max(160,bitcap+sum(int(m).bit_count()>1 for m in masks)+17*bitcap+2)
                        points={}
                        for name,order,elide in [('stable_dead',stable,False),('dfs_dead',dfs,False),
                                ('stable_single',stable,True),('dfs_single',dfs,True)]:
                            point=serve(masks,residual,parent,order,elide)
                            point.update(live_metrics(order,parent))
                            # Isolated-tile bound: no unproved inter-tile overlap.
                            point['isolated_tile_cycles_with_order_build']=pre+8*point['cycles']+2+(n if name.startswith('dfs') else 0)
                            points[name]=point
                            totals.setdefault(name,Counter()).update(point)
                        rows.append(dict(sample=sample,operator=operator,partition=partition,chunk=chunk,points=points))
    base=totals['stable_dead']; compare={}
    for name,t in totals.items():
        compare[name]=dict(service_ratio_vs_stable_dead=base['cycles']/t['cycles'],
            isolated_tile_ratio_with_charged_order_build=base['isolated_tile_cycles_with_order_build']/t['isolated_tile_cycles_with_order_build'],
            parent_access_reduction=1-(t['reads']+t['writes'])/(base['reads']+base['writes']),
            dependency_live_vector_row_step_reduction=1-t['live_vector_row_steps']/base['live_vector_row_steps'],
            max_peak_live_vectors=max(r['points'][name]['peak_live_parent_vectors'] for r in rows))
    result=dict(cohort='10 samples x 4 Conv operators x K partitions 0/216/431 x spatial chunks 0/15/31/46',
        tiles=len(rows),self_test_cases=tested,scalar_mismatches=0,totals=totals,comparisons=compare,rows=rows,
        scope='Same fixed Prosperity-compatible forest; CPU service/liveness probe, not full ledger, RTL or area',
        caveats=['Dependency-live peak assumes a future physical allocator; current scratch never evicts within its 64-row tile',
            'DFS first-child/next-sibling representation and traversal are standard scheduling, not a first claim',
            'Order construction charged at one extra cycle per row; metadata area and memory-port cost not synthesized',
            'Single-use forwarded write elision is the earlier M505 model option, not existing admitted C1 RTL',
            'No SNN timestep or recurrent-neuron order is changed'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','totals')},indent=2))


if __name__=='__main__': main()
