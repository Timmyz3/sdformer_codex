import copy
import unittest
import pilot_quality_core as m

class Tests(unittest.TestCase):
    def fixture(self):
        frames=[]
        for j in range(8):
            arms={}
            for arm in m.ARMS:
                value=j+1+(2 if arm=='CC' else 1 if arm=='fine_0_0' else 3 if arm=='fine_0_1' else 0)
                arms[arm]={r:{'valid_pixels':2,'error_sum':2.*value,'AEE':float(value)} for r in m.REGIONS}
                arms[arm]['B_unknown']={'valid_pixels':0,'error_sum':0.,'AEE':None}
            frames.append({'target_id':f'id{j}','sequence':f'seq{j}','arms':arms})
        return frames

    def test_signed_full_support(self):
        out=m.analyze(self.fixture());v=out['action_values'][0]['regions']['all_valid']
        self.assertEqual(v['CC_added_damage'],2)
        self.assertEqual(v['singleton_gains']['fine_0_0'],1)
        self.assertEqual(v['singleton_gains']['fine_0_1'],-1)
        self.assertEqual((v['positive'],v['negative'],v['minimum'],v['maximum']),(11,1,-1,2))
        self.assertEqual(out['summaries']['native']['overall']['all_valid']['frame_mean_AEE'],4.5)
        self.assertEqual((out['frame_region_cells'],out['summary_cells']),(1920,2160))

    def test_empty_not_gain_or_protection_evidence(self):
        frames=self.fixture()
        for f in frames:
            for a in m.ARMS:f['arms'][a]['B']=copy.deepcopy(f['arms'][a]['B_unknown'])
        out=m.analyze(frames)
        self.assertIsNone(out['action_values'][0]['regions']['B']['CC_added_damage'])
        self.assertIsNone(out['action_values'][0]['regions']['B']['maximum'])
        self.assertFalse(out['action_values'][0]['protection_vs_native']['CC']['diagnostic_all_three_pass'])

    def test_support_and_frame_rejection(self):
        with self.assertRaisesRegex(ValueError,'Complete H8'):m.analyze(self.fixture()[:7])
        frames=self.fixture();frames[0]['arms']['CC']['L']={'valid_pixels':1,'error_sum':3.,'AEE':3.}
        with self.assertRaisesRegex(ValueError,'Same all-output support'):m.analyze(frames)

if __name__=='__main__':unittest.main()
