import unittest
import audit_actual as a

class Tests(unittest.TestCase):
    def test_partial_rejected(self):
        with self.assertRaises(ValueError):a.validate_header({'schema':'flow_r18_g0_cached_ordinary_v1','status':'RUNNING'}, {})
    def test_uniform_expected_only(self):
        w=[1/12]*12;p={'weights':w,'kind':'expected_uniform_randomized_k1','selected_tile':None,'mask':None,
          'rational_weight':{'numerator':1,'denominator':12},'endpoint_arms':[f'fine_{r}_{c}' for r,c in a.core.TILES]}
        a.check_policy(p,w,'uniform_expectation')
        p['selected_tile']=[0,0]
        with self.assertRaises(ValueError):a.check_policy(p,w,'uniform_expectation')
    def test_real_endpoint_and_tails(self):
        w=[0.]*12;w[11]=1.
        p={'weights':w,'kind':'deterministic_k1','selected_tile':[2,3],'endpoint_arm':'fine_2_3',
          'mask':[[r==2 and c==3 for c in range(5)] for r in range(4)]}
        a.check_policy(p,w,'fixed_2_3');p['mask'][3][4]=True
        with self.assertRaises(ValueError):a.check_policy(p,w,'fixed_2_3')

if __name__=='__main__':unittest.main()
