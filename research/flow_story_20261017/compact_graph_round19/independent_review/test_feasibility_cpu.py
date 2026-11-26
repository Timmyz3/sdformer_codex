import unittest
import diagnose_feasibility_v1 as d

class Tests(unittest.TestCase):
    def test_support_and_conjunction(self):
        n={r:{'AEE':1.,'valid_pixels':10} for r in ('all_valid','B','L')}
        c={r:dict(v) for r,v in n.items()};c['L']['AEE']=1.051
        self.assertFalse(d.gate(c,n)['all_three_pass'])
        c['L']['AEE']=1.05;self.assertTrue(d.gate(c,n)['all_three_pass'])
        for x in (n,c):x['L']={'AEE':None,'valid_pixels':0}
        self.assertFalse(d.gate(c,n)['all_three_pass'])
    def test_full_population(self):
        n={r:{'AEE':1.,'valid_pixels':10} for r in ('all_valid','B','L')}
        bad={r:{'AEE':1.1,'valid_pixels':10} for r in ('all_valid','B','L')}
        frames=[{'target_id':str(i),'sequence':str(i),'arms':{'native':n,'CC':bad,**{a:(n if i and j==0 else bad) for j,a in enumerate(d.SINGLETONS)}}} for i in range(8)]
        out=d.analyze(frames)
        self.assertEqual(out['zero_feasible_singleton_frames'],['0'])
        self.assertEqual([x['feasible_singleton_count'] for x in out['frames']],[0]+[1]*7)
        self.assertTrue(all(len(x['singleton_actions'])==12 for x in out['frames']))

if __name__=='__main__':unittest.main()
