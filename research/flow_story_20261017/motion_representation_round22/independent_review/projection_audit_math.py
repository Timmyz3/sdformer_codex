"""Independent normalized weighted ridge/priors audit; no model-source import."""
import numpy as np
from io_common import require

def truth_grid(gt,valid):
    require(gt.shape==(2,480,640) and valid.shape==(480,640) and np.isin(valid,(0,1)).all(),'Original GT/mask shape/binary')
    mask=valid.astype(bool);require(np.isfinite(gt[:,mask]).all(),'Finite valid original GT')
    counts=mask.reshape(30,16,40,16).sum((1,3),dtype=np.int64)
    sums=np.where(mask[...,None],gt.transpose(1,2,0),0.).astype(np.float64).reshape(30,16,40,16,2).sum((1,3),dtype=np.float64)
    return sums/np.maximum(counts,1)[...,None],counts

def independent_ridge(prefix,truth,counts):
    x=np.asarray(prefix,np.float64);y=np.asarray(truth,np.float64);n=np.asarray(counts,np.float64)
    require(x.shape==(*n.shape,384) and y.shape==(*n.shape,2),'Original384 to2')
    valid=n>0;raw=x[valid];labels=y[valid];weight=n[valid];size=len(weight)
    require(size>384 and np.isfinite(raw).all() and np.isfinite(labels).all(),'Finite adequate fit-only support')
    # Probability weights sum1; alpha/n is equivalent to source mean-weight1.
    probability=weight/weight.sum();mu=np.sum(raw*probability[:,None],axis=0);nu=np.sum(labels*probability[:,None],axis=0)
    sigma=np.sqrt(np.sum(np.square(raw-mu)*probability[:,None],axis=0));sigma=np.where(sigma>1e-6,sigma,1.)
    standardized=(raw-mu)/sigma
    normal=standardized.T@(probability[:,None]*standardized)
    normal.flat[::385]+=10./size
    cross=standardized.T@(probability[:,None]*(labels-nu))
    coefficient=np.linalg.solve(normal,cross)/sigma[:,None];intercept=nu-mu@coefficient
    return dict(weight=coefficient.astype(np.float32),bias=intercept.astype(np.float32),fit_cells=size,fit_valid_pixels=int(n.sum()),alpha=10.)

def priors(truth,counts):
    y=np.asarray(truth,np.float64);n=np.asarray(counts,np.int64)
    require(y.shape==(*n.shape,2) and n.sum()>0,'Finite nonempty fit-only prior support')
    mass=n.sum(0);numerator=(y*n[...,None]).sum(0)
    global_flow=numerator.sum((0,1))/mass.sum();position=numerator/np.maximum(mass,1)[...,None];position[mass==0]=global_flow
    return dict(global_vector=np.broadcast_to(global_flow.astype(np.float32),(30,40,2)).copy(),position_prior=position.astype(np.float32))

def close(actual,reference,*,atol=2e-5,rtol=2e-5):
    a=np.asarray(actual);b=np.asarray(reference);require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'Finite paired shapes')
    d=np.abs(a.astype(np.float64)-b.astype(np.float64));ok=bool(np.all(d<=atol+rtol*np.abs(b.astype(np.float64))))
    require(ok,'Independent motion projection arithmetic differs')
    return dict(max_absolute_difference=float(d.max(initial=0.)),absolute_tolerance=atol,relative_tolerance=rtol,**{'pass':ok})

def verify_model(reported,expected):
    require(set(reported)==set(expected)=={'weight','bias','alpha','fit_cells','fit_valid_pixels'},'Exact ridge state fields')
    require(reported['alpha']==expected['alpha']==10. and reported['fit_cells']==expected['fit_cells'] and reported['fit_valid_pixels']==expected['fit_valid_pixels'],'Fit-only support/recipe')
    return {k:close(np.asarray(reported[k],np.float32),expected[k]) for k in ('weight','bias')}

def project(state,prefix):
    return (np.asarray(prefix,np.float32)@np.asarray(state['weight'],np.float32)+np.asarray(state['bias'],np.float32)).astype(np.float32)
