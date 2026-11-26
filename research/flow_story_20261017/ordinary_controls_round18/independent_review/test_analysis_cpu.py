import unittest
import numpy as np
import analysis_core as a

class Tests(unittest.TestCase):
    def test_pool_shape_and_constant(self):
        u=np.full((10,1,30,40,384),2,dtype=np.float32);s=np.ones_like(u)
        scores=a.independently_score(u,s)
        self.assertEqual(len(scores),12)
        for row in scores:
            self.assertEqual([row[k] for k in a.SCORES],[2,2,0,0,0,1,0])
        self.assertTrue(all(a.select(scores,n)==0 for n in a.SCORES))
    def test_temporal_and_spatial_scores(self):
        u=np.zeros((10,1,30,40,384),dtype=np.float32)
        u+=np.arange(10,dtype=np.float32)[:,None,None,None,None]
        s=np.broadcast_to((np.arange(30)%2)[None,None,:,None,None],u.shape).astype(np.float32)
        r=a.independently_score(u,s)[0]
        self.assertEqual(r['temporal_neighbor_abs_difference'],1)
        self.assertEqual(r['spatial_neighbor_abs_difference'],0)
        self.assertEqual(r['pooling_residual_mse'],0)
        self.assertEqual(r['psn1_spatial_neighbor_hamming'],.5)
        u+=np.arange(30,dtype=np.float32)[None,None,:,None,None]
        r=a.independently_score(u,s)[0]
        self.assertEqual(r['spatial_neighbor_abs_difference'],.5)
        self.assertAlmostEqual(r['pooling_residual_mse'],2/3)
    def test_expected_error_not_ensemble(self):
        ms=[{'valid_pixels':1,'error_sum':float(i),'AEE':float(i)} for i in range(12)]
        self.assertEqual(a.expectation(ms,[1/12]*12)['AEE'],5.5)
        self.assertEqual(a.expectation(ms,[0]*11+[1])['AEE'],11)
        empty=[{'valid_pixels':0,'error_sum':0,'AEE':None}]*12
        self.assertIsNone(a.expectation(empty,[1/12]*12)['AEE'])
        bad=ms.copy();bad[-1]={'valid_pixels':2,'error_sum':22.,'AEE':11.}
        with self.assertRaises(ValueError):a.expectation(bad,[1/12]*12)
    def test_macro_and_pixel_weights(self):
        got=a.aggregate([{'valid_pixels':1,'error_sum':10.,'AEE':10.},{'valid_pixels':9,'error_sum':9.,'AEE':1.}])
        self.assertEqual(got['frame_mean_AEE'],5.5);self.assertEqual(got['pixel_weighted_AEE'],1.9)
    def test_complete_policy_population(self):
        frames=[];weights={}
        score=[{'tile':list(t),**{n:0. for n in a.SCORES}} for t in a.TILES]
        def metrics(v):return {r:{'valid_pixels':1,'error_sum':v,'AEE':v} for r in a.REGIONS}
        for i in range(8):
            tid=f'f{i}';arms={'native':metrics(2.),'CC':metrics(12.)}
            arms.update({f'fine_{r}_{c}':metrics(float(j)) for j,(r,c) in enumerate(a.TILES)})
            frames.append({'target_id':tid,'sequence':f's{i}','arms':arms});weights[tid]=a.policy_weights(score)
        out=a.analyze(frames,weights)
        self.assertEqual(out['frame_region_cells'],2560);self.assertEqual(out['summary_cells'],2880)
        self.assertEqual(out['summaries']['uniform_expectation']['overall']['all_valid']['frame_mean_AEE'],5.5)
        self.assertEqual(out['policies']['uniform_expectation'][0]['gain_vs_CC']['all_valid'],6.5)
        self.assertFalse(out['protection_vs_native']['uniform_expectation']['overall_and_all_frames'])
        self.assertTrue(out['protection_vs_native']['fixed_0_0']['overall_and_all_frames'])

if __name__=='__main__':unittest.main()
