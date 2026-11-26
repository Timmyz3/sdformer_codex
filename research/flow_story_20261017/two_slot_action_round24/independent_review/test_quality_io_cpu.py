import copy,unittest
import reduce_raw_v1 as r

class Checks(unittest.TestCase):
    def test_sources_plan_and_replay_roles(self):
        metric,core=r.frozen_math();self.assertEqual(len(metric.REGIONS),16);self.assertEqual(tuple(core.ARMS),r.ARMS)
        plan=r.plan([{'target_id':str(i)} for i in range(8)])
        self.assertEqual(len(plan),122);self.assertEqual(sum(x[3] for x in plan),120)
        self.assertEqual([x[1] for x in plan[:6]],['native','FF','T2','T2_repeat1','T2_repeat2','fine_0_0'])
        self.assertEqual([x[4] for x in plan if not x[3]],['T2','T2'])
        self.assertEqual(len({(x[0],x[4]) for x in plan if x[3]}),120)
    def test_complete_counts_and_numerical_families(self):
        result={'schema':'flow_r24_two_slot_domain_v1','status':'TWO_SLOT_DOMAIN122_COMPLETE','graph_id':r.GRAPH,'binding':{'graph_id':r.GRAPH},
          'full_forward_attempts':122,'full_forward_count':122,'training_performed':False,'optimizer_steps':0,'calibration_forwards':0,
          'all_states_unchanged':True,'all_hooks_removed':True,'original_modules_restored':True}
        audit={'schema':'flow_r24_two_slot_independent_execution_audit_v1','status':'PASS','quality_analysis_allowed':True,
          'counts':dict(full_forwards=122,native=8,compact=114,expanded=0,quality_endpoints=120,determinism_replays=2,
            FC1_rows=67335120,FC2_rows=67335120,BN_calls=9516,archives=122,head_payloads=488,capture_payloads=98,optimizer_steps=0,calibration_forwards=0),
          'numerical_checks':[{'kind':k,'pass':True} for k in ['allfine_vs_native']*8+['T2_exact_replay']*2]}
        r.validate_complete(result,audit)
        for field in ('full_forwards','quality_endpoints','capture_payloads','FC1_rows','determinism_replays'):
            x=copy.deepcopy(audit);x['counts'][field]-=1
            with self.assertRaises(ValueError):r.validate_complete(result,x)
        x=copy.deepcopy(audit);x['numerical_checks'][-1]['kind']='CC_exact_replay'
        with self.assertRaises(ValueError):r.validate_complete(result,x)
        x=copy.deepcopy(audit);x['numerical_checks'][-1]['pass']=False
        with self.assertRaises(ValueError):r.validate_complete(result,x)
        x=copy.deepcopy(result);x['status']='INCOMPLETE_OR_NUMERICAL_FAILURE'
        with self.assertRaises(ValueError):r.validate_complete(x,audit)

if __name__=='__main__':unittest.main()
