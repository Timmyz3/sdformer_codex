import copy,unittest
import analysis_core as c

def metric(v,n=2):return dict(valid_pixels=n,error_sum=v*n if n else 0.,AEE=v if n else None)
def fixture():
    return [dict(target_id=str(i),sequence='seq'+str(i),arms={a:{r:metric(2.) for r in c.REGIONS} for a in c.ARMS}) for i in range(8)]
class Checks(unittest.TestCase):
    def test_all_eight_and_signed_context(self):
        f=fixture();f[0]['arms']['T8']['all_valid']=metric(3.);f[0]['arms']['fine_0_0']['all_valid']=metric(4.)
        x=c.analyze(f);self.assertEqual(x['feasibility'][0]['feasible_singleton_count'],11)
        self.assertEqual(x['signed_action_diagnostics'][0]['singleton_signed_gain']['fine_0_0']['all_valid']['signed_AEE'],-1.)
        self.assertTrue(x['all_eight_have_feasible_singleton']);self.assertEqual(x['quality_prediction_count'],120)
    def test_any_new_zero_stops_not_only_old_hardframe(self):
        f=fixture()
        for a in c.SINGLETONS:f[1]['arms'][a]['B']=metric(2.2)
        x=c.analyze(f);self.assertEqual(x['zero_feasible_singleton_frames'],['1']);self.assertEqual(x['resource_decision'],'STOP_THIS_T8_K1_SELECTOR_FAMILY')
    def test_support_and_exact_limits(self):
        f=fixture();f[0]['arms']['fine_0_0']['all_valid']=metric(2.06);f[0]['arms']['fine_0_0']['L']=metric(2.1)
        self.assertTrue(c.gate(f[0]['arms']['fine_0_0'],f[0]['arms']['native'])['all_three_pass'])
        for a in c.ARMS:f[0]['arms'][a]['L']=metric(0.,0)
        x=c.analyze(f);self.assertEqual(x['feasibility'][0]['feasible_singleton_count'],0)
        self.assertIsNone(x['summaries']['native']['seq0']['L']['frame_mean_AEE'])
        f[0]['arms']['native']['B']['error_sum']=-1
        with self.assertRaises(ValueError):c.analyze(f)
if __name__=='__main__':unittest.main()
