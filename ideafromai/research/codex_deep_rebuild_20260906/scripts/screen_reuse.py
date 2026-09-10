#!/usr/bin/env python3.12
"""Frozen-descriptor opportunity screens. Neither RTL cycles nor ASIC PPA.

C1 differential reconstruction is established prior art (Phi); this screen
asks whether changing the online result graph helps the existing cohort.
C2 selects common subexpressions by bank-preserving service intents, charging
one separate scatter per consumer. Classification and SRAM costs are open.
"""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
HW = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07')
sys.path.insert(0, str(HW/'system_simulator/scripts'))
import numpy as np
import m2259_c1_forest_lifetime_probe as c1
import m2271_destination_signature_screen as prior


def differential(masks, prim=False, allow_negative=True):
    active = [i for i, m in enumerate(masks) if m]
    stable = sorted(active, key=lambda i: (int(masks[i]).bit_count(), i))
    remaining = set(active)
    order = []
    parent = [-1]*len(masks)
    delta = [0]*len(masks)
    best = {r: (int(masks[r]).bit_count(), 0, -1) for r in active}
    while remaining:
        r = min(remaining, key=lambda r: (best[r], r)) if prim else stable[len(order)]
        p = best[r][2]
        parent[r] = p
        delta[r] = int(masks[r]) ^ (int(masks[p]) if p >= 0 else 0)
        order.append(r)
        remaining.remove(r)
        for nxt in remaining:
            if not allow_negative and int(masks[r]) & int(masks[nxt]) != int(masks[r]):
                continue
            key = (max(1, (int(masks[nxt]) ^ int(masks[r])).bit_count()), 1, r)
            if key < best[nxt]:
                best[nxt] = key
    return delta, parent, order


def check_numeric(masks, parent, order, weights):
    values = {}
    checks = 0
    for row in order:
        p = int(parent[row])
        before = int(masks[p]) if p >= 0 else 0
        after = int(masks[row])
        value = np.zeros(96, dtype=np.int64) if p < 0 else values[p].copy()
        # Every intermediate is still a binary subset sum: [-2048,2032].
        for bit in range(16):
            if before >> bit & 1 and not after >> bit & 1:
                value -= weights[bit]
                assert np.all((-2048 <= value) & (value <= 2032))
        for bit in range(16):
            if after >> bit & 1 and not before >> bit & 1:
                value += weights[bit]
                assert np.all((-2048 <= value) & (value <= 2032))
        expected = sum((weights[b] for b in range(16) if after >> b & 1), np.zeros(96,dtype=np.int64))
        assert np.array_equal(expected, value)
        values[row] = value
        checks += 96
    return checks


def run_c1():
    totals = {n: Counter() for n in ('subset_stable','subset_costmatched_stable','subset_costmatched_prim','differential_stable','differential_prim')}
    rows = []
    rng = np.random.default_rng(20260906)
    weights = rng.integers(-128,128,size=(16,96),dtype=np.int64)
    weights[:,0] = -128
    weights[:,1] = 127
    checked = 0
    # Directed corner cases exercise duplicate masks, strict signed deletion,
    # all-zero, dense masks, and roots; not just implementation-mirroring cases.
    for masks in ([0]*64,[65535,32767,65534,1,2,3],[3]*64,[1,2,4,8]):
        for prim in (False,True):
            _,p,o = differential(masks,prim)
            checked += check_numeric(masks,p,o,weights)
    with c1.LEDGER.open('rb') as stream:
        for sample in range(10):
            for operator in range(4):
                for partition in (0,216,431):
                    phase=(sample*4+operator)*432+partition
                    for chunk in (0,15,31,46):
                        n=min(64,3000-chunk*64)
                        stream.seek((phase*3000+chunk*64)*9)
                        masks=np.array([int(v,16)&65535 for v in stream.read(n*9).splitlines()],dtype=np.uint16)
                        res,par=c1.old.M504.cleanroom_subset(masks)
                        order,_=c1.forest_orders(masks,par)
                        axes={'subset_stable':(res,par,order),
                              'subset_costmatched_stable':differential(masks,False,False),
                              'subset_costmatched_prim':differential(masks,True,False),
                              'differential_stable':differential(masks),
                              'differential_prim':differential(masks,True)}
                        points={}
                        for name,(d,p,o) in axes.items():
                            checked += check_numeric(masks,p,o,weights)
                            out=Counter(c1.serve(masks,d,p,o))
                            out.update(c1.live_metrics(o,p))
                            out['negative_source_corrections']=sum((int(masks[p[r]])&~int(masks[r])&65535).bit_count() for r in o if p[r]>=0)
                            out['active_rows']=len(o)
                            out['tiles']=1
                            # Same legacy preprocess charge is only a reference
                            # sensitivity; new matcher timing/area is unpriced.
                            bitcap=(n+7)//8
                            pre=max(160,bitcap+sum(int(m).bit_count()>1 for m in masks)+17*bitcap+2)
                            out['isolated_reference_preprocess_plus_service']=pre+8*out['cycles']+2
                            out['with_one_cycle_per_row_extra_planner']=pre+8*out['cycles']+2+(n if name!='subset_stable' else 0)
                            points[name]=dict(out)
                            totals[name].update(out)
                        rows.append(dict(sample=sample,operator=operator,partition=partition,chunk=chunk,points=points))
    baseline=totals['subset_stable']
    comparison={name:{'service_model_ratio':baseline['cycles']/t['cycles'],
        'source_issue_reduction':1-t['issues']/baseline['issues'],
        'parent_access_reduction':1-(t['reads']+t['writes'])/(baseline['reads']+baseline['writes']),
        'isolated_ratio_with_one_extra_planner_cycle_per_row':baseline['with_one_cycle_per_row_extra_planner']/t['with_one_cycle_per_row_extra_planner'],
        'maximum_dependency_live_vectors':max(r['points'][name]['peak_live_parent_vectors'] for r in rows)} for name,t in totals.items()}
    paired={order:{'service_ratio':totals['subset_costmatched_'+order]['cycles']/totals['differential_'+order]['cycles'],
        'source_issue_reduction':1-totals['differential_'+order]['issues']/totals['subset_costmatched_'+order]['issues']} for order in ('stable','prim')}
    return dict(scope='480 fixed ep34 tiles; CPU arithmetic/parent-service screen, not new RTL or full Conv cycles',
        cohort='10 samples x 4 Conv x K=0/216/431 x spatial=0/15/31/46; same as m2259',
        source=str(c1.LEDGER),tiles=len(rows),checked_outputs=checked,mismatches=0,
        totals=totals,comparison=comparison,negative_edge_only_paired_comparison=paired,rows=rows,
        boundaries=['Phi already supplies signed residual and zero-pattern fallback; no new algebra claim',
          'Only already computed rows are parents; all 64 descriptors are available before planning',
          'Prim is a global within-tile arithmetic-oriented bound, not a cheap online hardware scheduler',
          'Two credits and 1RW parent service reuse m2259 semantics; new sign logic/matcher/selector costs are not timed',
          'Dependency-live counts do not implement a physical allocator or prove removal of nine macros',
          'Natural +1 source only; negative descriptor bridge is outside the 12bit subset-range proof',
          'One extra planner cycle per row is a sensitivity, not synthesized planner latency',
          'INT8 diagnostic weights establish integer reconstruction, not frozen FP32 AEE or full operator binding'])


def signature_options(x):
    codes=(x.astype(np.int64)*np.array([1,2,4,8])[:,None]).sum(0)
    hist=np.bincount(codes,minlength=16)
    candidates=[m for m in range(1,16) if m.bit_count()>=2 and hist[m]>=2]
    groups=[]
    for begin in range(0,x.shape[1],8):
        present=0
        for m in set(int(v) for v in codes[begin:begin+8]):
            if m: present |= 1<<m
        groups.append(present)
    # Compress repeated eight-bank group signatures; still charge multiplicity.
    group_hist=Counter(groups)
    dest_bits=[sum(1<<m for m in range(1,16) if m>>d&1) for d in range(4)]
    ordinary=sum(count*sum(bool(g&db) for db in dest_bits) for g,count in group_hist.items())
    class_cost={m:sum(count for g,count in group_hist.items() if g>>m&1) for m in candidates}

    def cost(selection):
        selected=sum(1<<m for m in selection)
        updates=sum(class_cost[m] for m in selection)
        scatter=sum(m.bit_count() for m in selection)
        bypass=sum(count*sum(bool(g&~selected&db) for db in dest_bits) for g,count in group_hist.items())
        return updates+bypass+scatter

    full=cost(candidates)
    best={0:(ordinary,())}
    for cap in (1,2):
        b=best[cap-1]
        for sel in combinations(candidates,cap):
            v=(cost(sel),sel)
            if v<b: b=v
        best[cap]=b
    # Exact search of all eligible masks is an upper bound on selector benefit.
    opt=best[2]
    for n in range(3,len(candidates)+1):
        for sel in combinations(candidates,n):
            v=(cost(sel),sel)
            if v<opt:opt=v
    result=Counter(windows=1,ordinary=ordinary,all_share=full,
        cap1=best[1][0],cap2=best[2][0],oracle=opt[0],
        cap1_selected=len(best[1][1]),cap2_selected=len(best[2][1]),
        oracle_selected=len(opt[1]),cap1_winning_windows=int(best[1][0]<ordinary),
        cap2_winning_windows=int(best[2][0]<ordinary),oracle_winning_windows=int(opt[0]<ordinary),
        classification_source_columns=x.shape[1],classification_k8_groups=x.shape[1]//8)
    # Actual arithmetic reconstruction for chosen bounded schemes.
    for cap in (1,2):
        result[f'cap{cap}_selected_members']=sum(int(hist[m]) for m in best[cap][1])
    return result,best,opt,codes


def check_signature(x,selection,codes):
    rng=np.random.default_rng(20260906+x.shape[1])
    w=rng.integers(-128,128,size=(x.shape[1],96),dtype=np.int64)
    w[:,0]=-128; w[:,1]=127
    got=np.zeros((4,96),dtype=np.int64)
    for m in range(1,16):
        idx=np.flatnonzero(codes==m)
        if not len(idx):continue
        if m in selection:
            v=w[idx].sum(axis=0)
            for d in range(4):
                if m>>d&1:got[d]+=v
        else:
            for i in idx:
                for d in range(4):
                    if m>>d&1:got[d]+=w[i]
    assert np.array_equal(got,x.astype(np.int64)@w)
    assert np.all((-2**23<=got)&(got<2**23))
    return got.size


def run_c2():
    fixtures=HW/'tb_m2018/fixtures'
    capdir=HW/'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901'
    layers={r['layer_id']:r for r in json.loads((capdir/'layers.json').read_text())['layers']}
    totals={};weighted={};rows=[];checked=0;count=0
    for prefix,extent in (('m2051_ep34_tsbg_full40_s1920',48),('m2067_ep34_fc2_exact_continuation_s960',192)):
        meta=json.loads((fixtures/(prefix+'.json')).read_text())
        words=np.array([int(v,16)&65535 for v in (fixtures/(prefix+'.memh')).read_text().split()],dtype=np.uint16)
        words=words.reshape(len(meta['rows']),4,extent)
        for row in meta['rows']:
            target=row['target'] if extent==48 else 'FC2'
            assert row['negative_codes']==0
            layer=layers[row['layer_id']]
            assert layer['target']==target
            mult=layer['weight_layout']['output_tile_count']
            x=prior.bits(words[row['slot'],:,:int(row['source_groups'])])
            count+=1
            for window in (16,64,768):
                key=f'{target}_window{window}'
                c=Counter()
                for start in range(0,x.shape[1],window):
                    xx=x[:,start:start+window]
                    out,best,opt,codes=signature_options(xx)
                    c.update(out)
                    # Cross-check against independently pre-existing m2271 cost.
                    ref=prior.analyze(xx)
                    assert out['ordinary']==ref['ordinary_nonempty_k8_bundles']
                    assert out['all_share']==ref['bank_preserving_total_update_intents']
                    if row['slot']<2:
                        for sel in (best[1][1],best[2][1],opt[1]):
                            checked+=check_signature(xx,sel,codes)
                totals.setdefault(key,Counter()).update(c)
                weighted.setdefault(key,Counter()).update({k:v*mult for k,v in c.items()})
                rows.append(dict(sequence=row['sequence'],slot=row['slot'],target=target,window=window,output_tiles=mult,counts=c))
            if count%480==0:print(f'C2 {count}/2880 workloads',flush=True)
    def finish(t):
        d=dict(t)
        for scheme in ('all_share','cap1','cap2','oracle'):
            d[scheme+'_intent_reduction']=1-t[scheme]/max(1,t['ordinary'])
            d[scheme+'_max_extra_intents_per_k8_classification_group']=(t['ordinary']-t[scheme])/max(1,t['classification_k8_groups'])
        return d
    return dict(scope='C2 bank-preserving service-intent selection screen; NOT cycles/energy/PPA',
        workloads=count,fc={k:finish(v) for k,v in totals.items()},
        fc_weighted_by_output_tiles={k:finish(v) for k,v in weighted.items()},
        checked_outputs=checked,mismatches=0,rows=rows,
        boundaries=['Same 2880 descriptor workload templates and 16/64/768 windows as m2271; not all tokens',
          'Selection uses arrived full-window descriptors; metadata scan/buffering/selection time is not free in hardware',
          'Empty selection retains ordinary intents, so gain is a selector opportunity bound, not measured speedup',
          'Cap1/2 require at most 1/2 live Acc24 partial vectors (288/576B), in addition to unchanged four output contexts',
          'One class update, bypass update or per-consumer scatter counts as one intent, not equal physical latency',
          'No extra weight-fetch saving is claimed beyond TSBG',
          'All partials use Acc24 and disjoint source sets; natural +1 sources and INT8 diagnostic weights',
          'No free coalescing of a wide partial with an INT8 K8 reduction; mixed-width fusion remains a separate hypothesis',
          'All-share and ordinary counts cross-check against pre-existing m2271 implementation',
          'Oracle enumerates all selections; it is not an implementable constant-time selector'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('axis',choices=('c1','c2'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=run_c1() if args.axis=='c1' else run_c2()
    result['python_version']=sys.version
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','totals')},indent=2))


if __name__=='__main__':main()
