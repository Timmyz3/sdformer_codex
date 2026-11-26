import copy
import unittest
import reduce_raw_v1 as r
import compare_scores_v1 as compare

class Tests(unittest.TestCase):
    def test_frozen_sources_and_rules(self):
        metric,signed,g0=r.frozen_math()
        self.assertEqual(len(metric.REGIONS),16);self.assertEqual(len(signed.ARMS),15)
        self.assertEqual(len(g0.TILES),12);self.assertEqual(len(g0.SCORES),7)
    def test_plan_repeat_exclusion(self):
        p=r.plan([{'target_id':str(i)} for i in range(8)])
        self.assertEqual(len(p),122);self.assertEqual(sum(v[3] for v in p),120)
        self.assertEqual(p[3],('0','CC_repeat1','compact',False,'CC'))
        self.assertEqual(p[4],('0','CC_repeat2','compact',False,'CC'))
        self.assertEqual(len({(v[0],v[4]) for v in p if v[3]}),120)
    def test_incomplete_rejected(self):
        with self.assertRaises(ValueError):r.validate_complete({'schema':'flow_r19_compact_graph_v1','status':'INCOMPLETE_OR_NUMERICAL_FAILURE'}, {})
    def test_actual_count_and_numerical_population(self):
        a={'schema':'flow_r19_compact_graph_independent_execution_audit_v1','status':'PASS','quality_analysis_allowed':True,
           'counts':dict(full_forwards=122,native=8,compact=114,expanded=0,compact_target_calls=114,quality_endpoints=120,
            determinism_replays=2,FC1_rows=67021520,FC2_rows=67021520,BN_calls=9516,archives=122,head_payloads=488,
            capture_payloads=98,optimizer_steps=0,calibration_forwards=0),
           'numerical_checks':[dict(kind='allfine_vs_native',pass_value=True) for _ in range(8)]+[dict(kind='CC_exact_replay',pass_value=True) for _ in range(2)]}
        for row in a['numerical_checks']:row['pass']=row.pop('pass_value')
        x={'schema':'flow_r19_compact_graph_v1','status':'COMPACT_GRAPH122_COMPLETE','graph_id':r.GRAPH,'binding':{'graph_id':r.GRAPH},
           'full_forward_count':122,'full_forward_attempts':122,'training_performed':False,'optimizer_steps':0,'calibration_forwards':0,
           'all_states_unchanged':True,'all_hooks_removed':True,'original_modules_restored':True}
        r.validate_complete(x,a)
        for key in ('quality_endpoints','capture_payloads','FC1_rows'):
            b=copy.deepcopy(a);b['counts'][key]-=1
            with self.assertRaises(ValueError):r.validate_complete(x,b)
        b=copy.deepcopy(a);b['numerical_checks'][-1]['pass']=False
        with self.assertRaises(ValueError):r.validate_complete(x,b)
        b=copy.deepcopy(a);b['numerical_checks'][-1]['kind']='allfine_vs_native'
        with self.assertRaises(ValueError):r.validate_complete(x,b)
    def test_score_recompute_tolerance_does_not_hide_rank_change(self):
        _,_,g0=r.frozen_math()
        own=[{'tile':list(t),**{n:1. for n in g0.SCORES}} for t in g0.TILES]
        reported=copy.deepcopy(own)
        self.assertEqual(compare.compare_arrays(own,reported,g0),(0.,84))
        reported[-1][g0.SCORES[0]]+=1e-12
        with self.assertRaisesRegex(ValueError,'Exact choice'):compare.compare_arrays(own,reported,g0)

if __name__=='__main__':unittest.main()
