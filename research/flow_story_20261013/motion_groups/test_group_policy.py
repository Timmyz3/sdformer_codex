"""Independent stdlib CPU checks; run python3 -B test_group_policy.py.

These checks establish policy invariants and declared quantization semantics,
not flow accuracy, learned-policy quality, hardware cycles or energy.
"""
import copy
import json
import math
import random
import unittest

from group_policy import propose_pairs, quantize_offset, select_groups


def grid(height,width,value):
    return [[copy.deepcopy(value) for _ in range(width)] for _ in range(height)]


class QuantizationChecks(unittest.TestCase):
    def test_eight_directions_and_rotation(self):
        for dx,dy in ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)):
            self.assertEqual(quantize_offset(dx,dy), (dx,dy))
            self.assertEqual(quantize_offset(-dy,dx), (-dy,dx))
        # Physical 90-degree rotation occurs before anisotropic tile scaling.
        self.assertEqual(quantize_offset(18,9,tile_size_pixels=(18,9)), (1,1))
        self.assertEqual(quantize_offset(-9,18,tile_size_pixels=(18,9)), (0,1))

    def test_stage2_small_motion_is_direction_prior_not_endpoint(self):
        cfg={'tile_size_pixels':(144,144)}
        self.assertEqual(quantize_offset(1,0,**cfg), (1,0))
        self.assertIsNone(quantize_offset(1,0,association_mode='endpoint',**cfg))
        for dx,result in ((71.999,None),(72,(1,0)),(144,(1,0)),(288,(2,0)),(360,None)):
            self.assertEqual(quantize_offset(dx,0,association_mode='endpoint',**cfg),result)
        self.assertIsNone(quantize_offset(288,0,association_mode='endpoint',max_distance_tiles=1,**cfg))

    def test_missing_zero_nonfinite_and_minimum(self):
        for flow in ((None,1),(1,None),(0,0),(float('nan'),1),(float('inf'),0),(True,1)):
            self.assertIsNone(quantize_offset(*flow))
        self.assertIsNone(quantize_offset(0,0,min_motion_pixels=0))
        self.assertIsNone(quantize_offset(.99,0,min_motion_pixels=1))
        self.assertEqual(quantize_offset(1,0,min_motion_pixels=1),(1,0))
        self.assertEqual(quantize_offset(-1e300,1e300),( -1,1))

    def test_bad_quantization_configuration(self):
        for options in ({'tile_size_pixels':(0,9)}, {'tile_size_pixels':(float('nan'),9)},
                        {'max_distance_tiles':0}, {'max_distance_tiles':3},
                        {'max_distance_tiles':True}, {'association_mode':'warp'},
                        {'min_motion_pixels':-1}):
            with self.assertRaises(ValueError):quantize_offset(1,1,**options)


class ProposalChecks(unittest.TestCase):
    def test_conflicting_chain_max_priority_and_boundary_singleton(self):
        # 3->out of grid remains singleton, even though 2 points into it.
        p=propose_pairs([[1,2,9,1]],grid(1,4,(1,0)),grid(1,4,81),grid(1,4,810))
        self.assertEqual(p['pairs'],[[1,2]])
        self.assertEqual([g['members'] for g in p['groups']],[[0],[1,2],[3]])
        self.assertEqual(p['metadata']['conflict_rejected_edges'],[[0,1]])
        self.assertEqual(p['tiles'][3]['association_reason'],'partner_out_of_grid')
        # Equal max scores use IDs, not iteration or random state.
        p=propose_pairs([[1,1,1,1]],grid(1,4,(1,0)),grid(1,4,81),grid(1,4,810))
        self.assertEqual(p['pairs'],[[0,1]])

    def test_local_control_odd_tail_and_empty_tiles(self):
        p=propose_pairs(grid(2,5,1),grid(2,5,(0,3)),
                        [[81,81,81,81,9],[81,81,81,0,0]],
                        [[81,81,81,81,9],[81,81,81,0,0]],policy='horizontal_local')
        self.assertEqual(p['pairs'],[[0,1],[2,3],[5,6]])
        self.assertEqual([g['members'] for g in p['groups']],[[0,1],[2,3],[4],[5,6],[7]])
        self.assertEqual(p['tiles'][4]['association_reason'],'partner_out_of_grid')
        self.assertEqual(p['tiles'][7]['association_reason'],'partner_empty')
        self.assertEqual(p['metadata']['active_tiles'],8)

    def test_unknown_or_stationary_is_not_paired_or_imputed(self):
        flow=[[(1,0),None,(0,0),(1,0),(float('nan'),0),(1,0)]]
        for policy in ('motion','rotated90','horizontal_local'):
            p=propose_pairs(grid(1,6,2),flow,grid(1,6,81),grid(1,6,5),policy=policy)
            for index in (1,2,4):
                self.assertIn([index],[g['members'] for g in p['groups']])
                self.assertFalse(p['tiles'][index]['flow_eligible'])
            self.assertFalse(p['metadata']['GT_used_or_imputed'])
            # Output retains reason codes, not nonfinite sentinel numbers.
            json.dumps(p,allow_nan=False)
        independent=propose_pairs([[9,8]],[[None,(0,0)]],[[4,4]],[[7,7]],policy='independent')
        self.assertEqual(select_groups(independent,7)['selected_tile_ids'],[0])

    def test_scale_conversion_and_rotation_consistency(self):
        scores=[[5,4,3],[2,1,0],[7,6,8]]
        f=[[(1,0),(0,1),(-1,0)],[(1,1),(-1,-1),(0,-1)],[(0,-1),(1,0),(-1,-1)]]
        direct=propose_pairs(scores,f,grid(3,3,81),grid(3,3,3),policy='rotated90',tile_size_pixels=(18,9))
        rotated=[[(-dy,dx) for dx,dy in row] for row in f]
        explicit=propose_pairs(scores,rotated,grid(3,3,81),grid(3,3,3),policy='motion',tile_size_pixels=(18,9))
        self.assertEqual(direct['pairs'],explicit['pairs'])
        self.assertEqual([t['offset_xy'] for t in direct['tiles']],[t['offset_xy'] for t in explicit['tiles']])
        scaled=[[(dx/2.,dy/3.) for dx,dy in row] for row in f]
        converted=propose_pairs(scores,scaled,grid(3,3,81),grid(3,3,3),policy='rotated90',
                                flow_scale_to_pixels=(2.,3.),tile_size_pixels=(18,9))
        self.assertEqual(direct['pairs'],converted['pairs'])
        self.assertEqual([t['offset_xy'] for t in direct['tiles']],[t['offset_xy'] for t in converted['tiles']])

    def test_input_not_mutated_and_invalid_inputs_rejected(self):
        args=([[1,2]],[[(1,0),(-1,0)]],[[81,10]],[[100,12]])
        snapshot=copy.deepcopy(args)
        p=propose_pairs(*args)
        select_groups(p,112)
        self.assertEqual(args,snapshot)
        for scores,flow,valid,cost in (([[None]],[[None]],[[1]],[[1]]),
            ([[1]],[[None]],[[0]],[[1]]), ([[1]],[[None]],[[1]],[[0]]),
            ([[1]],[[None]],[[1]],[[1.5]]), ([[1]],[[None]],[[True]],[[1]]),
            ([[1,2]],[[None]],[[1,1]],[[1,1]])):
            with self.assertRaises(ValueError):propose_pairs(scores,flow,valid,cost)


class BudgetChecks(unittest.TestCase):
    def test_actual_tail_cost_cap_zero_full_and_never_split(self):
        p=propose_pairs([[10,9,1]], [[(1,0),(-1,0),None]], [[81,9,1]], [[7,3,2]])
        self.assertEqual(p['pairs'],[[0,1]])
        self.assertEqual(select_groups(p,0)['slow_mask'],[[False,False,False]])
        partial=select_groups(p,5)
        self.assertEqual(partial['slow_mask'],[[False,False,True]])
        self.assertEqual(partial['slow_service_cost'],2)
        self.assertEqual(partial['unused_slow_cost'],3)
        self.assertEqual(partial['selected_valid_points'],1)
        full=select_groups(p,12)
        self.assertEqual(full['slow_mask'],[[True,True,True]])
        self.assertEqual(full['selected_valid_points'],91)
        self.assertEqual(full['slow_service_cost'],12)
        self.assertEqual(select_groups(p,100)['unused_slow_cost'],88)

    def test_cost_validation_catches_tampering(self):
        original=propose_pairs([[2,1]], [[None,None]], [[81,81]], [[5,7]],policy='independent')
        for alteration in ('overlap','cost','row','coverage','score'):
            p=copy.deepcopy(original)
            if alteration=='overlap':p['groups'][1]['members']=[0]
            elif alteration=='cost':p['groups'][0]['slow_service_cost']=1
            elif alteration=='row':p['tiles'][0]['row']=1
            elif alteration=='coverage':p['groups'].pop()
            elif alteration=='score':p['groups'][0]['score']=999
            with self.assertRaises(ValueError):select_groups(p,12)
        for cap in (-1,True,1.5,float('inf')):
            with self.assertRaises(ValueError):select_groups(original,cap)

    def test_deterministic_randomized_partition_and_exact_budget_invariants(self):
        rng=random.Random(20261013)
        for trial in range(120):
            h,w=rng.randint(1,5),rng.randint(1,7)
            scores=[[rng.randint(-5,5) for _ in range(w)] for _ in range(h)]
            valid=[[rng.choice((0,1,9,81)) for _ in range(w)] for _ in range(h)]
            costs=[[rng.randint(1,100) if n else 0 for n in row] for row in valid]
            flows=[[rng.choice((None,(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(18,0)))
                    for _ in range(w)] for _ in range(h)]
            for policy in ('independent','horizontal_local','motion','rotated90'):
                p=propose_pairs(scores,flows,valid,costs,policy=policy,
                                association_mode='endpoint' if trial%2 else 'direction_one_hop')
                repeated=propose_pairs(scores,flows,valid,costs,policy=policy,
                                association_mode='endpoint' if trial%2 else 'direction_one_hop')
                self.assertEqual(p,repeated)
                members=[i for g in p['groups'] for i in g['members']]
                active=[r*w+c for r in range(h) for c in range(w) if valid[r][c]]
                self.assertEqual(sorted(members),active)
                self.assertEqual(len(members),len(set(members)))
                total=sum(sum(row) for row in costs)
                for cap in (0,total//3,total//2,total,total+1):
                    selection=select_groups(p,cap)
                    mask=selection['slow_mask']
                    actual=sum(costs[r][c] for r in range(h) for c in range(w) if mask[r][c])
                    self.assertEqual(selection['slow_service_cost'],actual)
                    self.assertLessEqual(actual,cap)
                    self.assertEqual(selection['unused_slow_cost'],cap-actual)
                    for g in p['groups']:
                        self.assertEqual(len({mask[i//w][i%w] for i in g['members']}),1)
                    for r in range(h):
                        for c in range(w):
                            if not valid[r][c]:self.assertFalse(mask[r][c])


if __name__=='__main__':
    unittest.main(verbosity=2)
