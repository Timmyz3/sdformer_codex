import json
import numpy as np
import core

def run():
    q,s=core.quantize(np.array([-40000,-2.5,-1.5,-.5,.5,1.5,2.5,40000],np.float32),1)
    assert q.tolist()==[-32767,-2,-2,0,0,2,2,32767] and s.tolist()==[True,False,False,False,False,False,False,True]
    scale=core.power2_scale(np.array([0,1,.03,100]));assert scale[0]==1
    assert np.all(np.log2(scale)==np.rint(np.log2(scale))) and np.all(scale[1:]*32767>=np.array([2,.06,200]))
    zero=core.voxel_features(np.zeros((10,16,16),np.float32));assert zero.dtype==np.float32 and np.all(zero==0)
    rng=np.random.default_rng(34);z=rng.normal(size=(10,16,16)).astype(np.float32);x=core.voxel_features(z)
    assert x.shape==(160,) and np.all(np.isfinite(x))
    ridge=dict(mean=np.zeros(3),scale=np.ones(3),intercept=np.array([.125,-.125]),coef=np.array([[1,-1],[.5,2],[-1,.5]]))
    a=np.tile(np.array([[.25,-.5,1]],np.float32),(64,1));c=core.calibrate_ridge(ridge,a)
    out,t=core.affine_mixed(ridge,a,c);ref,_=core.affine_fp32(ridge,a)
    assert np.array_equal(out,ref) and not t['input_saturation'].any() and not c['weight_saturation'].any()
    assert core.choose([[0,0,1,2,3,4]],True).tolist()==[-1] and core.choose([[0,0,1,2,3,4]]).tolist()==[0]
    return dict(status='PASS',tests=['RNE_ties_and_saturation','power2_margin','FP32_features_zero_and_nonzero','exact_integer_dot_dequant','choice_ties'],GPU_used=False)

if __name__=='__main__':print(json.dumps(run()))
