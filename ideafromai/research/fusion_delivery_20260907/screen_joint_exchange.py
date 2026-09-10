"""Bounded joint virtual-parent admission, on an explicitly reused cohort."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import Counter
import itertools
import json
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import screen_virtual_parents as vp


def certificate(baseline, new, n):
    bp, np = baseline[2], new[2]
    old_writers = {int(p) for p in bp if p >= 0}
    new_writers = {int(p) for p in np if p >= 0}
    V = len(new[0])
    q = sum(p >= 0 for p in np[n:])
    r = sum(bp[i] < 0 and np[i] >= 0 for i in range(n))
    D = len(old_writers - new_writers)
    U = len({p for p in new_writers if p < n} - old_writers)
    assert new[-1]['wide_reads'] - baseline[-1]['wide_reads'] == r+q
    assert new[-1]['wide_writes'] - baseline[-1]['wide_writes'] == V+U-D
    return {k:int(v) for k,v in {'V':V,'q':q,'r':r,'D':D,'U':U,'delta_reads':r+q,'delta_writes':V+U-D}.items()}


def optimize(rows, pool, budget=4, shortlist=8):
    current = baseline = vp.forest(rows, [])
    spent = 0
    steps, costs = [], Counter()
    while spent < budget:
        singles = []
        for v in pool:
            if v in current[0]:
                continue
            trial = vp.forest(rows, current[0]+[v])
            costs['single_graph_trials'] += 1
            if trial[-1]['binary_vector_adds'] < current[-1]['binary_vector_adds']:
                singles.append(([v], trial))
        singles.sort(key=lambda t:(t[1][-1]['binary_vector_adds'],t[1][-1]['wide_read_write_events'],t[0]))
        trials = list(singles)
        if spent+2 <= budget:
            selected = [item[0][0] for item in singles[:shortlist]]
            for pair in itertools.combinations(selected,2):
                costs['pair_graph_trials'] += 1
                trials.append((sorted(pair),vp.forest(rows,current[0]+sorted(pair))))
        admitted = []
        for new_masks,trial in trials:
            c,now = trial[-1],current[-1]
            if c['binary_vector_adds'] < now['binary_vector_adds'] and all(c[k] <= now[k] for k in (
                    'source_coefficient_reads','wide_read_write_events','peak_live_parent_vectors')):
                admitted.append((new_masks,trial))
        if not admitted:
            break
        new_masks, best = min(admitted,key=lambda t:(t[1][-1]['binary_vector_adds'],t[1][-1]['wide_read_write_events'],len(t[0]),t[0]))
        cert = certificate(baseline,best,len(rows))
        assert cert['delta_reads']+cert['delta_writes'] <= 0
        steps.append({'new_masks':new_masks,'counts':dict(best[-1]),'certificate_vs_original':cert,
                      'diagnostic_lane_values_verified':vp.verify(best,len(rows))})
        current = best
        spent += len(new_masks)
    return {'steps':steps,'final':dict(current[-1]),'virtual_masks':current[0],
            'spent_insertion_credits':spent,'planner':dict(costs)}


def main():
    out = BASE/'joint_exchange_r1.json'
    assert not out.exists()
    plan_path = BASE/'joint_exchange_plan.json'
    plan = json.loads(plan_path.read_text())
    plan_sha = vp.digest(plan_path)
    assert vp.digest(vp.LEDGER) == vp.LEDGER_SHA
    # a..f are bits0..5, x/y bits6/7. Exactly the review's joint-only counterexample.
    rows = [65,130,79,143,115,179]
    counterexample = optimize(rows,[15,51])
    assert counterexample['final']['binary_vector_adds'] == 12
    assert counterexample['final']['wide_read_write_events'] == 6
    assert counterexample['final']['peak_live_parent_vectors'] == 2
    assert len(counterexample['steps'][0]['new_masks']) == 2
    mp = json.loads((BASE/'materialization_plan.json').read_text())
    total,tiles = Counter(),[]
    with vp.LEDGER.open('rb') as stream:
        for sample,op,part,chunk in itertools.product(mp['evaluation_samples'],mp['operators'],mp['k_partitions'],mp['evaluation_chunks']):
            rows,_ = vp.old.read_rows(stream,sample,op,part,chunk*64,min(64,3000-chunk*64))
            orig = set(rows)
            pool = sorted({a&b for i,a in enumerate(rows) for b in rows[i+1:] if (a&b).bit_count()>=2 and (a&b) not in orig})
            point = optimize(rows,pool,plan['total_new_node_budget'],plan['joint_shortlist'])
            vp.ref.add(total,point['final'])
            total.update(point['planner'])
            total['tiles'] += 1
            total['tiles_with_exchange'] += bool(point['steps'])
            total['accepted_pair_steps'] += sum(len(s['new_masks'])==2 for s in point['steps'])
            total['diagnostic_lane_values_verified'] += sum(s['diagnostic_lane_values_verified'] for s in point['steps'])
            tiles.append({'sample':sample,'operator':op,'k_partition':part,'chunk':chunk,'result':point})
            if len(tiles)%32 == 0:
                print('Joint exchange tiles:',len(tiles),flush=True)
    assert len(tiles)==192 and vp.digest(plan_path)==plan_sha
    report = {'date':'2026-09-07','status':'BOUNDED_JOINT_EXCHANGE_REFERENCE',
              'plan':plan,'plan_sha256':plan_sha,'script_sha256':vp.digest(Path(__file__)),
              'imported_graph_script_sha256':vp.digest(BASE/'screen_virtual_parents.py'),
              'same_cohort_virtual_receipt_sha256':vp.digest(BASE/'virtual_parents_r1.json'),
              'ledger_sha256':vp.LEDGER_SHA,'counterexample':counterexample,'aggregate':dict(total),'tiles':tiles,
              'limits':['Shortlist8 is fixed before this execution; not a global optimum or new blind test.',
                        'The necessary port certificate does not close true allocation, cycle timing or matching cost.',
                        'All original outputs remain; only additional intermediate captures are selected.'],
              'claim_boundary':{'RTL_speedup':False,'PPA':False,'AEE':False,'FP32_equivalence':False}}
    with out.open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(report['aggregate'],indent=2),flush=True)


if __name__=='__main__':
    main()
