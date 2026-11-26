import unittest
import numpy as np
import projection_audit_math as a
class Checks(unittest.TestCase):
    def test_original_labels_sparse_invalid(self):
        gt=np.zeros((2,480,640),np.float32);mask=np.zeros((480,640),bool)
        gt[:,1,1]=[2.,-4.];gt[:,15,15]=[-2.,8.];mask[1,1]=mask[15,15]=True;gt[:,0,0]=np.nan
        y,n=a.truth_grid(gt,mask);np.testing.assert_array_equal(y[0,0],[0.,2.]);self.assertEqual(n.sum(),2)
        prior=a.priors(y[None],n[None]);np.testing.assert_array_equal(prior['position_prior'][29,39],[0.,2.])
    def test_ridge_probability_weights_and_folded_state(self):
        rng=np.random.default_rng(11);x=rng.normal(size=(1,30,40,384)).astype(np.float32);w=rng.normal(size=(384,2))/384
        y=x@w+np.array([1.,-2.]);n=np.full((1,30,40),2,np.int64);n[:,0]=0
        state=a.independent_ridge(x,y,n);pred=a.project(state,x)
        self.assertEqual(state['fit_cells'],1160);self.assertLess(float(np.square(pred[n>0]-y[n>0]).mean()),1e-5)
        self.assertEqual(a.verify_model(state,state)['weight']['max_absolute_difference'],0.)
        bad=pred.copy();bad.flat[0]+=1.
        with self.assertRaises(ValueError):a.close(bad,pred)
if __name__=='__main__':unittest.main()
