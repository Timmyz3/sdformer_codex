"""Standard-library CPU tests: independent exhaustive oracle and pinned old DP."""
import copy
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import time
import unittest

import select_joint as joint
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'flow_story_20261013/motion_groups/exact_budget_policy.py'
OLD_SHA='9d7ece29230b891a961fe00855a92bfde0d149862de60d1db97a571a848ee519'
assert hashlib.sha256(OLD.read_bytes()).hexdigest()==OLD_SHA
spec=importlib.util.spec_from_file_location('pinned_joint_old_DP',OLD)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def proposal(scores,costs,groups,policy='motion'):
    tiles=[{'tile_id':i,'row':0,'col':i,'score':s,'valid_points':n,'slow_service_cost':n}
           for i,(s,n) in enumerate(zip(scores,costs))]
    return {'schema':'flow_motion_group_proposal_v1','shape':[1,len(scores)],'policy':policy,
        'association_mode':'direction_one_hop','tiles':tiles,
        'groups':[{'group_id':g,'members':list(ids),'score':max(scores[i] for i in ids),
                   'valid_points':sum(costs[i] for i in ids),'slow_service_cost':sum(costs[i] for i in ids)}
                  for g,ids in enumerate(groups)]}


def independent(p):
    q=copy.deepcopy(p);q['policy']='independent'
    q['groups']=[{'group_id':t['tile_id'],'members':[t['tile_id']],'score':t['score'],
                  'valid_points':t['valid_points'],'slow_service_cost':t['slow_service_cost']}
                 for t in q['tiles'] if t['valid_points']]
    return q


def brute(p,cost,rule):
    n=len(p['tiles']);best=None
    for bits in itertools.product((False,True),repeat=n):
        if any(bits[i] and not t['valid_points'] for i,t in enumerate(p['tiles'])):continue
        if sum(t['slow_service_cost'] for i,t in enumerate(p['tiles']) if bits[i])!=cost:continue
        if any(len(g['members'])==2 and
               ((rule=='at_least_one' and not any(bits[i] for i in g['members'])) or
                (rule in ('all_or_none','legacy_group_max') and len({bits[i] for i in g['members']})!=1))
               for g in p['groups']):continue
        ids=tuple(i for i in range(n) if bits[i])
        value=(sum((Fraction(g['score']) for g in p['groups'] if all(bits[i] for i in g['members'])),Fraction(0))
               if rule=='legacy_group_max' else
               sum((Fraction(p['tiles'][i]['score']) for i in ids),Fraction(0)))
        if best is None or value>best[0] or value==best[0] and ids<best[1]:best=(value,ids)
    return best


class JointChecks(unittest.TestCase):
    def test_exhaustive_rules_signed_ties_and_tail_costs(self):
        rng=random.Random(20261014)
        for case in range(24):
            n=rng.randrange(2,8);scores=[rng.choice([-2.,0.,.1,.2,1.,1.]) for _ in range(n)]
            costs=[rng.choice([1,2,3,4,9]) for _ in range(n)]
            shuffled=list(range(n));rng.shuffle(shuffled);groups=[]
            while shuffled:
                k=2 if len(shuffled)>1 and rng.random()<.7 else 1
                groups.append(sorted(shuffled[:k]));del shuffled[:k]
            p=proposal(scores,costs,groups)
            for rule in joint.RULES:
                for target in sorted({0,1,sum(costs)//2,sum(costs)-1,sum(costs)}):
                    with self.subTest(case=case,rule=rule,target=target):
                        got=joint.select_exact(p,target,rule);expected=brute(p,target,rule)
                        self.assertEqual(got['feasible'],expected is not None)
                        if expected:
                            self.assertEqual(got['selected_tile_ids'],list(expected[1]))
                            self.assertEqual(got['objective_score_exact'],{'numerator':str(expected[0].numerator),'denominator':str(expected[0].denominator)})
                            self.assertEqual(got['slow_service_cost'],target)
                        else:self.assertIsNone(got['slow_mask'])

    def test_free_equivalent_to_old_independent_with_crossing_pairs(self):
        for scores in ([.1,.2,.1+.2,0.,-.1,0.], [1.]*6, [-1.]*6):
            p=proposal(scores,[81,36,27,12,81,27],[[0,5],[1,4],[2,3]])
            singleton=independent(p)
            for target in range(265):
                a=joint.select_exact(p,target,'free');b=old.select_exact(singleton,target)
                for key in ('feasible','slow_mask','selected_tile_ids','objective_score_exact','slow_service_cost'):
                    self.assertEqual(a[key],b[key],(target,key))

    def test_at_least_one_allows_optional_singletons_and_partial_pairs(self):
        p=proposal([10.,1.,2.],[81,81,81],[[0,1],[2]])
        a=joint.select_exact(p,81,'at_least_one');b=joint.select_exact(p,81,'all_or_none')
        self.assertEqual(a['selected_tile_ids'],[0]);self.assertEqual(a['pair_state_counts']['10'],1)
        self.assertEqual(b['selected_tile_ids'],[2]);self.assertEqual(b['pair_state_counts']['00'],1)
        self.assertFalse(joint.select_exact(p,0,'at_least_one')['feasible'])
        self.assertTrue(joint.select_exact(p,0,'all_or_none')['feasible'])

    def test_legacy_group_max_is_separate_objective_and_exact_old_result(self):
        p=proposal([6.,6.,10.],[81,81,162],[[0,1],[2]])
        new=joint.select_exact(p,162,'all_or_none');legacy=joint.select_exact(p,162,'legacy_group_max')
        self.assertEqual(new['selected_tile_ids'],[0,1]);self.assertEqual(legacy['selected_tile_ids'],[2])
        expected=old.select_exact(p,162)
        for key in ('slow_mask','selected_tile_ids','objective_score_exact','selected_group_ids'):
            self.assertEqual(legacy[key],expected[key])

    def test_exact900_infeasible_is_not_relaxed_and_input_not_mutated(self):
        p=proposal([1.,2.],[81,81],[[0,1]]);before=copy.deepcopy(p)
        r=joint.select_exact(p,900,'at_least_one')
        self.assertFalse(r['feasible']);self.assertEqual(r['target_cost'],900)
        self.assertIsNone(r['slow_service_cost']);self.assertIsNone(r['slow_mask']);self.assertEqual(p,before)
        p=proposal([1.,2.,3.],[81,36,12],[[0,1],[2]])
        self.assertFalse(joint.select_exact(p,90,'free')['feasible'])

    def test_invalid_overlap_cost_score_and_empty_tile_handling(self):
        p=proposal([1.,2.,None],[3,2,0],[[0,1]])
        r=joint.select_exact(p,3,'free');self.assertEqual(r['slow_mask'],[[True,False,False]])
        mutations=[lambda q:q['groups'].append(copy.deepcopy(q['groups'][0])),
                   lambda q:q['tiles'][0].update(score=float('nan')),
                   lambda q:q['tiles'][0].update(slow_service_cost=4),
                   lambda q:q['groups'][0].update(score=0),
                   lambda q:q['groups'][0].update(members=[1,0])]
        for change in mutations:
            bad=copy.deepcopy(p);change(bad)
            with self.assertRaises(ValueError):joint.select_exact(bad,3,'free')
        for target in (True,-1,2049,3.):
            with self.assertRaises(ValueError):joint.select_exact(p,target)
        with self.assertRaises(ValueError):joint.select_exact(p,3,'unknown')

    def test_group_order_does_not_change_tile_objective_tie(self):
        p=proposal([1.]*5,[1]*5,[[0,4],[1,3],[2]])
        q=copy.deepcopy(p);q['groups'].reverse()
        for rule in joint.RULES:
            self.assertEqual(joint.select_exact(p,3,rule),joint.select_exact(q,3,rule))


if __name__=='__main__':
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(JointChecks))
    report={'schema':'flow_joint_admission_CPUchecks_v1','status':'PASS' if result.wasSuccessful() else 'FAIL',
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'wall_seconds':time.monotonic()-started,
        'source_sha256':hashlib.sha256(Path(joint.__file__).read_bytes()).hexdigest(),
        'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'old_DP_sha256':OLD_SHA,
        'scope':'stdlib synthetic exhaustive masks/legacy equivalence; no model/GT/torch/GPU'}
    (HERE/'cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
