"""Fixed R34 FP32 features and mixed int16 readout. No labels, fit or GPU API."""
import numpy as np

METHODS = ('direct_raw','signed_g1','signed_g4','direct_aug_g1','direct_aug_g4','direct_residual_g1','direct_residual_g4')
F = np.float32
QMAX = 32767

def sum32(x,axis=None):
    return np.sum(x,axis=axis,dtype=np.float32)

def voxel_features(voxel):
    z=np.asarray(voxel,dtype=np.float32)
    if z.ndim!=3 or z.shape[0]!=10 or z.shape[1]%4 or z.shape[2]%4 or min(z.shape[1:])<12 or not np.all(np.isfinite(z)):
        raise ValueError('finite T10 raw voxel with 4x4 nonempty-interior cells required')
    t,h,w=z.shape; edges=np.linspace(-1,1,t,dtype=np.float32); base=[]; direction=[]
    for cy in range(4):
        for cx in range(4):
            a=np.ascontiguousarray(z[:,cy*h//4:(cy+1)*h//4,cx*w//4:(cx+1)*w//4])
            absolute=np.abs(a);mass_t=sum32(absolute,axis=(1,2));mass=sum32(mass_t)
            temporal=np.abs(sum32(a,axis=0));diff=np.abs(a[1:]-a[:-1])
            base.extend((sum32(np.maximum(a,F(0)))/F(a.size),sum32(np.maximum(-a,F(0)))/F(a.size),
                F(np.count_nonzero(a>0))/F(a.size),F(np.count_nonzero(a<0))/F(a.size),
                (sum32(temporal)/F(temporal.size))/F(t),sum32(diff)/F(diff.size),
                sum32(edges*mass_t)/mass if mass!=0 else F(0),
                sum32((edges*edges)*mass_t)/mass if mass!=0 else F(0)))
            mid=(a[1:]+a[:-1])*F(.5)
            it=a[1:,1:-1,1:-1]-a[:-1,1:-1,1:-1]
            ix=(mid[:,1:-1,2:]-mid[:,1:-1,:-2])*F(.5)
            iy=(mid[:,2:,1:-1]-mid[:,:-2,1:-1])*F(.5)
            it2=sum32(it*it)
            for spatial in (ix,iy):
                numerator=sum32(spatial*it);denom=np.sqrt(F(sum32(spatial*spatial)*it2))
                direction.append(F(numerator/denom) if denom>F(1e-12) else F(0))
    out=np.asarray(base+direction,dtype=np.float32)
    if out.shape!=(160,) or not np.all(np.isfinite(out)):raise ValueError('invalid features')
    return out

def cast_ridge(ridge):
    return {k:np.asarray(ridge[k],dtype=np.float32) for k in ('mean','scale','intercept','coef')}

def standardized(ridge,x):
    r=cast_ridge(ridge);z=(np.asarray(x,dtype=np.float32)-r['mean'])/r['scale']
    if not np.all(np.isfinite(z)):raise ValueError('nonfinite standardization')
    return z,r

def affine_fp32(ridge,x):
    z,r=standardized(ridge,x);out=np.zeros((len(z),len(r['intercept'])),dtype=np.float32)
    for k in range(z.shape[1]):out=np.add(out,np.multiply(z[:,k,None],r['coef'][k,None,:]))
    out=np.add(out,r['intercept'])
    return out,dict(standardized=z,output=out)

def power2_scale(maximum):
    m=np.asarray(maximum,dtype=np.float64)
    exponent=np.ceil(np.log2(np.where(m>0,2*m/QMAX,1)))
    return np.asarray(np.where(m>0,np.exp2(exponent),1),dtype=np.float32)

def quantize(x,scale):
    rounded=np.rint(np.asarray(x,dtype=np.float32)/np.asarray(scale,dtype=np.float32))
    saturation=(rounded < -QMAX)|(rounded > QMAX)
    return np.clip(rounded,-QMAX,QMAX).astype(np.int16),saturation

def calibrate_ridge(ridge,fit_x):
    z,r=standardized(ridge,fit_x)
    sx=power2_scale(np.max(np.abs(z)));sw=power2_scale(np.max(np.abs(r['coef']),axis=0))
    qw,ws=quantize(r['coef'],sw)
    bound=z.shape[1]*QMAX**2
    if bound>=np.iinfo(np.int64).max:raise ValueError('int64 bound')
    return dict(input_scale=sx,weight_scale=sw,weight_q=qw,weight_saturation=ws,
        fit_standardized_absmax=F(np.max(np.abs(z))),coefficient_absmax=np.max(np.abs(r['coef']),axis=0),
        accumulator_absolute_bound=bound,fit_rows=len(z))

def affine_mixed(ridge,x,calibration):
    z,r=standardized(ridge,x);qx,sat=quantize(z,calibration['input_scale'])
    acc=qx.astype(np.int64)@np.asarray(calibration['weight_q'],dtype=np.int64)
    out=acc.astype(np.float32)*F(calibration['input_scale'])
    out=out*np.asarray(calibration['weight_scale'],dtype=np.float32)
    out=out+r['intercept']
    return out,dict(standardized=z,input_q=qx,input_saturation=sat,int64_accumulator=acc,output=out)

def products(aux,grid):
    z=np.asarray(aux,dtype=np.float32);d=2*grid*grid
    return z[:,:d,None].transpose(0,2,1)*z[:,d:].reshape(len(z),6,d)

def signed(aux,grid):
    p=products(aux,grid);out=np.zeros(p.shape[:2],dtype=np.float32)
    for k in range(p.shape[2]):out=out+p[:,:,k]
    return out

def choose(scores,allow=False):
    scores=np.asarray(scores);out=np.argmin(scores,axis=1).astype(np.int64)
    if allow:out[np.min(scores,axis=1)>=0]=-1
    return out

def run_suite(model,x,mode='fp32',calibration=None,calibrate=False):
    x=np.asarray(x,dtype=np.float32)
    if mode not in ('fp32','mixed_int16'):raise ValueError(mode)
    if calibrate and (mode!='mixed_int16' or len(x)!=64):raise ValueError('only fit64 calibration')
    scales={} if calibrate else calibration;trace={}
    def affine(section,name,features):
        key=section+'/'+name;r=model[section][name]
        if mode=='fp32':out,t=affine_fp32(r,features)
        else:
            if calibrate:scales[key]=calibrate_ridge(r,features)
            out,t=affine_mixed(r,features,scales[key])
        trace[key]=t;return out
    scores={'direct_raw':affine('models','direct_raw',x)};auxes={}
    for g in (1,4):
        aux=affine('auxiliary',f'g{g}',x);auxes[f'g{g}']=aux;s=signed(aux,g);scores[f'signed_g{g}']=s
        aug=np.concatenate((x,aux,products(aux,g).reshape(len(x),-1)),axis=1)
        scores[f'direct_aug_g{g}']=affine('models',f'direct_aug_g{g}',aug)
        scores[f'direct_residual_g{g}']=s+affine('models',f'direct_residual_g{g}',aug)
    result=dict(scores=scores,predicted_auxiliary=auxes,fixed_six_choices={m:choose(scores[m]) for m in METHODS},
        allow_Q4_choices={m:choose(scores[m],True) for m in METHODS})
    return result,trace,scales
