import copy,unittest
import analysis_core as a

def frames():
    out=[]
    for i in range(8):
        arms={}
        for arm,val in {'native':1.,'FF':1.,'CC':2.,'F12':1.4,'T8':1.7}.items():
            arms[arm]={r:{'AEE':val,'error_sum':val*(i+1),'valid_pixels':i+1} for r in a.REGIONS}
        out.append({'target_id':str(i),'sequence':str(i),'arms':arms})
    return out

class Checks(unittest.TestCase):
    def test_population_sign_and_nonadditivity(self):
        q=a.analyze(frames());self.assertEqual((q['frame_region_cells'],q['summary_cells']),(640,720))
        self.assertAlmostEqual(q['signed_layout_diagnostics']['F12_gain_vs_CC']['overall']['all_valid']['frame_mean_signed_contrast'],.6)
        self.assertAlmostEqual(q['signed_layout_diagnostics']['layout_nonadditivity']['overall']['all_valid']['frame_mean_signed_contrast'],-.1)
        self.assertTrue(q['protection_vs_native']['FF']['overall_and_all_frames']);self.assertFalse(q['protection_vs_native']['T8']['overall_and_all_frames'])
    def test_macro_pixel_and_empty(self):
        rows=[dict(AEE=1.,error_sum=1.,valid_pixels=1),dict(AEE=3.,error_sum=27.,valid_pixels=9),dict(AEE=None,error_sum=0.,valid_pixels=0)]
        q=a.aggregate(rows);self.assertEqual(q['frame_mean_AEE'],2.);self.assertEqual(q['pixel_weighted_AEE'],2.8);self.assertEqual(q['empty_region_frames'],1)
        fs=frames()
        for arm in a.ARMS:fs[0]['arms'][arm]['L']={'AEE':None,'error_sum':0.,'valid_pixels':0}
        q=a.analyze(fs);self.assertFalse(q['protection_vs_native']['FF']['overall_and_all_frames']);self.assertIsNone(q['signed_layout_diagnostics']['layout_nonadditivity']['frames'][0]['regions']['L']['signed_AEE_contrast'])
    def test_reject_wrong_support_and_invalid_metric(self):
        for edit in ('support','sum','finite','missing'):
            f=frames()
            if edit=='support':f[0]['arms']['CC']['all_valid']['valid_pixels']=2
            elif edit=='sum':f[0]['arms']['CC']['all_valid']['error_sum']=0
            elif edit=='finite':f[0]['arms']['CC']['all_valid']['AEE']=float('nan')
            else:del f[0]['arms']['T8']
            with self.assertRaises(ValueError):a.analyze(f)

if __name__=='__main__':unittest.main()
