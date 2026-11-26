import copy,unittest
import reduce_raw_v1 as r

class Checks(unittest.TestCase):
    def test_sources_plan_and_replay_roles(self):
        metric,core=r.frozen_math();self.assertEqual(len(metric.REGIONS),16);self.assertEqual(tuple(core.ARMS),r.ARMS)
        plan=r.plan([{'target_id':str(i)} for i in range(8)])
        self.assertEqual(len(plan),44);self.assertEqual(sum(x[3] for x in plan),40)
        self.assertEqual([x[1] for x in plan[:9]],['native','FF','CC','CC_repeat1','CC_repeat2','F12','T8','T8_repeat1','T8_repeat2'])
        self.assertEqual([x[4] for x in plan if not x[3]],['CC','CC','T8','T8'])
        self.assertEqual(len({(x[0],x[4]) for x in plan if x[3]}),40)
    def test_complete_counts_and_numerical_families(self):
        result={'schema':'flow_r20_action_domain_v1','status':'ACTION_DOMAIN44_COMPLETE','graph_id':r.GRAPH,'binding':{'graph_id':r.GRAPH},
          'full_forward_attempts':44,'full_forward_count':44,'training_performed':False,'optimizer_steps':0,'calibration_forwards':0,
          'all_states_unchanged':True,'all_hooks_removed':True,'original_modules_restored':True}
        audit={'schema':'flow_r20_action_domain_independent_execution_audit_v1','status':'PASS','quality_analysis_allowed':True,
          'counts':dict(full_forwards=44,native=8,compact=36,expanded=0,quality_endpoints=40,determinism_replays=4,
            FC1_rows=24343920,FC2_rows=24343920,BN_calls=3432,archives=44,head_payloads=176,capture_payloads=116,optimizer_steps=0,calibration_forwards=0),
          'numerical_checks':[{'kind':k,'pass':True} for k in ['allfine_vs_native']*8+['CC_exact_replay']*2+['T8_exact_replay']*2]}
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
