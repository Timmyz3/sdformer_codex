import unittest
import numpy as np
import analysis_core as c
class Checks(unittest.TestCase):
    def test_pixel_and_grid_different_and_empty_preserved(self):
        q=np.zeros((30,40,2),np.float32);gt=np.zeros((2,480,640),np.float32)
        valid=np.zeros((480,640),bool);valid[0,0]=valid[15,15]=True
        gt[:,0,0]=[2.,0.];gt[:,15,15]=[-2.,0.]
        masks={r:valid.copy() for r in c.REGIONS};masks['B']=np.zeros_like(valid)
        out=c.measure(q,gt,masks)
        self.assertEqual(out['all_valid']['pixel']['AEE'],2.)
        self.assertEqual(out['all_valid']['grid']['pixel_weighted_cell_AEE'],0.)
        self.assertIsNone(out['B']['grid']['unweighted_cell_AEE']);self.assertIsNone(out['B']['pixel']['AEE'])
    def test_coordinates_true_tail_and_original_units(self):
        q=np.zeros((30,40,2),np.float32);q[29,39]=[-3.,4.]
        full=c.lift(q);np.testing.assert_array_equal(full[:,479,639],[-3.,4.]);np.testing.assert_array_equal(full[:,479,623],[0.,0.])
        pooled=c.pool3(q);self.assertEqual(float(pooled[29,39,0]),-1.);self.assertEqual(float(pooled[29,39,1]),np.float32(4/3))
    def test_weighted_vs_cell_mean(self):
        q=np.zeros((30,40,2),np.float32);gt=np.zeros((2,480,640),np.float32);valid=np.zeros((480,640),bool)
        valid[:16,:16]=True;valid[0,16]=True;gt[0,:16,:16]=1.;gt[0,0,16]=3.
        out=c.measure(q,gt,{r:valid.copy() for r in c.REGIONS})['L']['grid']
        self.assertEqual(out['nonempty_cells'],2);self.assertEqual(out['unweighted_cell_AEE'],2.)
        self.assertAlmostEqual(out['pixel_weighted_cell_AEE'],259/257)
if __name__=='__main__':unittest.main()
