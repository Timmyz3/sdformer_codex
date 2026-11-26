"""Stdlib exhaustive/geometry checks for the bounded CPU reference planner."""
import copy
from fractions import Fraction
import itertools
import json
import math
import random
import unittest
import group_policy as gp
import exact_budget_policy as ep


def proposal(costs, scores=None, policy='independent'):
    scores=scores if scores is not None else [[float(i+1) for i in range(len(costs[0]))] for _ in costs]
    flow=[[[3.,1.] for _ in row] for row in costs]
    return gp.propose_pairs(scores,flow,costs,costs,policy=policy)


def manual_groups(p, members):
    p=copy.deepcopy(p);p['policy']='motion';p['groups']=[]
    for i,ids in enumerate(members):
        cost=sum(p['tiles'][j]['slow_service_cost'] for j in ids)
        p['groups'].append({'group_id':i,'members':ids,'valid_points':cost,
                           'slow_service_cost':cost,'score':max(p['tiles'][j]['score'] for j in ids)})
    return p


def brute(p,target):
    options=[]
    for mask in itertools.product((False,True),repeat=len(p['groups'])):
        groups=[g for g,keep in zip(p['groups'],mask) if keep]
        if sum(g['slow_service_cost'] for g in groups)!=target:continue
        score=sum((Fraction(g['score']) for g in groups),Fraction(0))
        members=tuple(sorted(i for g in groups for i in g['members']))
        options.append((score,members))
    return min(options,key=lambda v:(-v[0],v[1])) if options else None


class ExactTests(unittest.TestCase):
    def test_zero_all_negative_and_empty(self):
        p=proposal([[81,36,27,12]],[[-4.,-3.,-2.,-1.]])
        zero=ep.select_exact(p,0);self.assertTrue(zero['feasible']);self.assertEqual(zero['selected_tile_ids'],[])
        all_=ep.select_exact(p,156);self.assertEqual(all_['selected_tile_ids'],[0,1,2,3]);self.assertEqual(all_['objective_score'],-10.)
        empty=proposal([[0,0]],[[None,None]])
        self.assertEqual(ep.reachable_costs(empty,100)['costs'],[0])
        self.assertFalse(ep.select_exact(empty,1)['feasible'])

    def test_pair_never_split_and_explicit_infeasible(self):
        p=manual_groups(proposal([[81,36,27,12]]),[[0,1],[2,3]])
        self.assertEqual(ep.reachable_costs(p,156)['costs'],[0,39,117,156])
        for cost in range(157):
            r=ep.select_exact(p,cost)
            if r['feasible']:
                ids=set(r['selected_tile_ids']);self.assertEqual(0 in ids,1 in ids);self.assertEqual(2 in ids,3 in ids)
            else:
                self.assertIsNone(r['achieved_cost']);self.assertIsNone(r['slow_mask']);self.assertIsNone(r['selected_tile_ids'])

    def test_lexicographic_tie_and_group_order_independence(self):
        p=manual_groups(proposal([[1,1,1,1]],[[1.,1.,1.,1.]]),[[0,3],[1,2]])
        a=ep.select_exact(p,2)
        self.assertEqual(a['selected_tile_ids'],[0,3])
        q=copy.deepcopy(p);q['groups'].reverse()
        self.assertEqual(ep.select_exact(q,2),a)
        p=proposal([[1,1,2]],[[.1,.2,.3]])
        # Exact source float values, not decimal idealization or rounded sum.
        self.assertEqual(ep.select_exact(p,2)['selected_tile_ids'],list(brute(p,2)[1]))

    def test_random_exhaustive_every_target(self):
        rng=random.Random(20261013)
        for n in range(1,9):
            for case in range(8):
                costs=[rng.randint(1,9) for _ in range(n)]
                scores=[rng.choice([-2.,-1.,0.,.1,.2,1.,2.]) for _ in range(n)]
                p=proposal([costs],[scores]);members=[];i=0
                while i<n:
                    size=2 if i+1<n and rng.randrange(2) else 1
                    members.append(list(range(i,i+size)));i+=size
                p=manual_groups(p,members)
                reachable=[]
                for target in range(sum(costs)+1):
                    expected=brute(p,target);actual=ep.select_exact(p,target)
                    self.assertEqual(actual['feasible'],expected is not None)
                    if expected is not None:
                        reachable.append(target)
                        self.assertEqual(actual['selected_tile_ids'],list(expected[1]))
                        exact=actual['objective_score_exact']
                        self.assertEqual(Fraction(int(exact['numerator']),int(exact['denominator'])),expected[0])
                        self.assertEqual(actual['achieved_cost'],target)
                self.assertEqual(ep.reachable_costs(p,sum(costs))['costs'],reachable)

    def test_native_tails_common_actual_cost_and_no_mutation(self):
        costs=[[81,81,81,81,36]]*3+[[27,27,27,27,12]]
        scores=[[float(y*5+x+1) for x in range(5)] for y in range(4)]
        proposals={policy:proposal(costs,scores,policy) for policy in ep.POLICIES}
        before=copy.deepcopy(proposals)
        for cap in (0,300,600,900,1200):
            shared=ep.common_target(proposals,cap)
            target=shared['target_cost'];self.assertLessEqual(target,cap)
            reach=[set(ep.reachable_costs(p,cap)['costs']) for p in proposals.values()]
            self.assertEqual(target,max(set.intersection(*reach)))
            for p in proposals.values():
                selected=ep.select_exact(p,target)
                self.assertTrue(selected['feasible']);self.assertEqual(selected['achieved_cost'],target)
                self.assertEqual(sum(t['valid_points'] for t in p['tiles'] if t['tile_id'] in selected['selected_tile_ids']),target)
                json.dumps(selected,allow_nan=False)
        self.assertEqual(proposals,before)
        self.assertGreater(len({tuple(tuple(g['members']) for g in p['groups']) for p in proposals.values()}),1)

    def test_common_target_never_uses_scores_to_choose_budget(self):
        costs=[[3,5,7],[2,4,6]]
        props={k:proposal(costs,policy=k) for k in ep.POLICIES}
        original=ep.common_target(props,17)
        changed=copy.deepcopy(props)
        # Preserve grouping and identical per-tile scores across arms, but change all scores.
        for p in changed.values():
            for t in p['tiles']:t['score']=-99.
            for g in p['groups']:g['score']=-99.
        self.assertEqual(original['target_cost'],ep.common_target(changed,17)['target_cost'])
        self.assertEqual(original['common_reachable_costs'],ep.common_target(list(changed.values()),17)['common_reachable_costs'])

    def test_counters_and_finite_output_even_score_sum_overflows_float(self):
        p=proposal([[1,1]],[[1.7e308,1.7e308]])
        r=ep.select_exact(p,2);c=r['counters']
        self.assertIsNone(r['objective_score']);self.assertIsNotNone(r['objective_score_exact'])
        self.assertEqual(c['state_visits'],3);self.assertEqual(c['transition_count'],6)
        self.assertEqual(c['include_transitions_attempted'],c['include_transitions_within_cap']+c['include_transitions_over_cap'])
        self.assertEqual(c['reachable_states_after_each_group'],[1,2,3])
        json.dumps(r,allow_nan=False)

    def test_input_guards_and_limits(self):
        p=proposal([[81,36,27,12]])
        for target in (-1,2049,True,1.):
            with self.assertRaises(ValueError):ep.select_exact(p,target)
        for bad in (proposal([[1]*33]),proposal([[2049]])):
            with self.assertRaises(ValueError):ep.reachable_costs(bad,1)
        for mode in ('overlap','missing','score','fake_cost','nan'):
            q=copy.deepcopy(p)
            if mode=='overlap':q['groups'][1]['members']=[0]
            elif mode=='missing':q['groups'].pop()
            elif mode=='score':q['groups'][0]['score']+=1
            elif mode=='fake_cost':q['tiles'][0]['slow_service_cost']+=1
            else:q['tiles'][0]['score']=float('nan')
            with self.assertRaises(ValueError):ep.select_exact(q,0)
        props={k:proposal([[3,5]],policy=k) for k in ep.POLICIES}
        props['motion']['tiles'][0]['score']=99.;props['motion']['groups'][0]['score']=99.
        with self.assertRaises(ValueError):ep.common_target(props,8)
        with self.assertRaises(ValueError):ep.common_target([p]*4,8)


if __name__=='__main__':
    unittest.main(verbosity=2)
