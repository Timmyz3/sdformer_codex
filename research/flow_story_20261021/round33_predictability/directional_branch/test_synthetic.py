#!/usr/bin/env python3
"""Synthetic-only X160 observability-branch tests, without new quality reads."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import predictors as p

BASE_PATH=Path(__file__).resolve().parent.parent/'algorithm/predictors.py'
assert hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()=='35663dd6b999621652d9d185d72b967e33c380e1064a09ec75540d2759cd61c3'
spec=importlib.util.spec_from_file_location('frozen_base128_for_synthetic',BASE_PATH)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def close(a,b,atol=1e-12):
    np.testing.assert_allclose(a,b,atol=atol,rtol=1e-11)


def manual(z):
    _,h,w=z.shape;out=[]
    for cy in range(4):
        for cx in range(4):
            a=z[:,cy*h//4:(cy+1)*h//4,cx*w//4:(cx+1)*w//4]
            st2=sx2=sy2=sxt=syt=0.
            for t in range(9):
                for y in range(1,a.shape[1]-1):
                    for x in range(1,a.shape[2]-1):
                        dt=a[t+1,y,x]-a[t,y,x]
                        dx=(a[t+1,y,x+1]+a[t,y,x+1]-a[t+1,y,x-1]-a[t,y,x-1])/4
                        dy=(a[t+1,y+1,x]+a[t,y+1,x]-a[t+1,y-1,x]-a[t,y-1,x])/4
                        st2+=dt*dt;sx2+=dx*dx;sy2+=dy*dy;sxt+=dx*dt;syt+=dy*dt
            for numerator,spatial2 in ((sxt,sx2),(syt,sy2)):
                denom=np.sqrt(spatial2*st2)
                out.append(numerator/denom if denom>1e-12 else 0.)
    return np.asarray(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
    rng=np.random.default_rng(33160);z=rng.normal(size=(10,16,20));z[np.abs(z)<.5]=0
    x=p.voxel_features(z)
    assert x.shape==(160,) and np.array_equal(x[:128],base.voxel_features(z))
    close(x[128:],manual(z))
    corr=x[128:].reshape(4,4,2)
    mirrorx=p.directional_couplings(z[:,:,::-1]).reshape(4,4,2)
    close(mirrorx[:,:,0],-corr[:,::-1,0]);close(mirrorx[:,:,1],corr[:,::-1,1])
    mirrory=p.directional_couplings(z[:,::-1,:]).reshape(4,4,2)
    close(mirrory[:,:,0],corr[::-1,:,0]);close(mirrory[:,:,1],-corr[::-1,:,1])
    close(p.directional_couplings(z[::-1]),-p.directional_couplings(z))
    close(p.directional_couplings(-z),p.directional_couplings(z))
    close(p.directional_couplings(3*z),p.directional_couplings(z))
    assert np.array_equal(p.directional_couplings(np.zeros_like(z)),np.zeros(32))
    static=np.repeat(z[:1],10,axis=0)
    assert np.array_equal(p.directional_couplings(static),np.zeros(32))
    t=np.arange(10)[:,None,None];yy=np.arange(16)[None,:,None];xx=np.arange(20)[None,None,:]
    plane=-2*xx+3*yy+4*t
    close(p.directional_couplings(plane).reshape(16,2),np.tile([-1,1],(16,1)))
    assert np.array_equal(p.directional_couplings(plane*1e-10),np.zeros(32))
    # Head/label/selection mathematics are an exact source copy. Only feature
    # dimension/schema and descriptive metadata change outside these functions.
    same_functions=['make_labels','pooled','split_aux','component_products','signed_scores','augmented_features',
        'train_ridge','ridge_predict','crossfit_auxiliary','choose_actions','offline_replacement_scores','loso_development']
    for name in same_functions:assert inspect.getsource(getattr(p,name))==inspect.getsource(getattr(base,name)),name
    data=dict(x=rng.normal(size=(24,128)),delta_aee=rng.normal(scale=.01,size=(24,6)),
        valid_pixels=rng.integers(50,200,size=24),aux_g1=rng.normal(size=(24,14)),aux_g4=rng.normal(size=(24,224)),
        ids=np.asarray([f'synthetic_{i}' for i in range(24)]),groups=np.asarray([f'group_{i//6}' for i in range(24)]))
    base_model=base.fit_suite(data)
    augmented=dict(data);augmented['x']=np.c_[data['x'],np.zeros((24,32))]
    model=p.fit_suite(augmented)
    before=base.predict_suite(base_model,data['x']);after=p.predict_suite(model,augmented['x'])
    for method in p.METHODS:close(before['scores'][method],after['scores'][method],atol=1e-10)
    assert model['models']['direct_aug_g1']['input_width']==186
    assert model['models']['direct_aug_g4']['input_width']==576
    result=dict(schema='flow_r33_directional_synthetic_checks_v1',status='PASS',
        checks=['base128 bitwise preserved','independent scalar coupling formula','x/y mirrors with cell reorder',
                'time reversal','polarity and amplitude scaling','zero and static temporal fields','known linear plane signs',
                'small denominator guard','head/label math unchanged','zero append synthetic head parity','aug dimensions186/576'],
        real_data_read=False,real_data_fitting=False,new_flow_network_forwards=0,GPU_used=False,seed=33160,
        source_sha256={Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (p.__file__,__file__)})
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='PASS',real_data_read=False)))


if __name__=='__main__':main()
