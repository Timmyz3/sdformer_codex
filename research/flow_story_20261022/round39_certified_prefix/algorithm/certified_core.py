"""R39 fixed-schedule FP32 enclosure core; no files, GT, fitting or GPU API.

Real bound arithmetic is widened after each operation. Rounded head arithmetic
uses monotone FP32 endpoint operations directly. The two must not be confused.
"""
from fractions import Fraction
import numpy as np

F=np.float32
U=Fraction(1,2**24)
ETA=Fraction(1,2**150)
FMAX=F(np.finfo(np.float32).max)

def rational_endpoint(q,upper):
    q=Fraction(q)
    f=F(float(q))
    if not np.isfinite(f):raise ValueError('nonfinite constant')
    exact=Fraction.from_float(float(f))
    if (upper and exact<q) or (not upper and exact>q):
        f=np.nextafter(f,F(np.inf if upper else -np.inf),dtype=np.float32)
    assert (Fraction.from_float(float(f))>=q if upper else Fraction.from_float(float(f))<=q)
    return f

def constants(n):
    if n<0 or n*U>=Fraction(1,2):raise ValueError('unsupported reduction length')
    return rational_endpoint(n*U/(1-n*U),True),rational_endpoint(n*ETA/(1-n*U),True)

def outward(a,b,op,upper):
    a,b=F(a),F(b)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore',under='ignore'):
        v={'add':np.add,'sub':np.subtract,'mul':np.multiply,'div':np.divide}[op](a,b,dtype=np.float32)
        v=np.nextafter(v,F(np.inf if upper else -np.inf),dtype=np.float32)
    if not np.isfinite(v):raise ValueError('real bound nonfinite; reference fallback required')
    return v

def lo(a,b,op):return outward(a,b,op,False)
def hi(a,b,op):return outward(a,b,op,True)

def sum_bounds_nonnegative(h,n):
    h=F(h)
    if not np.isfinite(h) or h<0:raise ValueError('finite nonnegative sum required')
    g,b=constants(n)
    lower=max(F(0),lo(lo(h,b,'sub'),hi(1,g,'add'),'div'))
    upper=hi(hi(h,b,'add'),lo(1,g,'sub'),'div')
    return F(lower),F(upper)

def rounded_sum_enclosure(lower,upper,n):
    g,b=constants(n)
    lower=max(F(0),lo(lo(lo(1,g,'sub'),lower,'mul'),b,'sub'))
    upper=hi(hi(hi(1,g,'add'),upper,'mul'),b,'add')
    return F(lower),F(upper)

def cheap_cell(a):
    a=np.ascontiguousarray(a,dtype=np.float32)
    if a.ndim!=3 or a.shape[0]!=10 or not np.isfinite(a).all():raise ValueError('finite T10 cell required')
    count=a.size;P=a.shape[1]*a.shape[2]
    if count>=2**24:raise ValueError('exact count conversion required')
    pos=np.sum(np.maximum(a,F(0)),dtype=np.float32)
    neg=np.sum(np.maximum(-a,F(0)),dtype=np.float32)
    mt=np.sum(np.abs(a),axis=(1,2),dtype=np.float32)
    mass=np.sum(mt,dtype=np.float32);edges=np.linspace(-1,1,10,dtype=np.float32)
    x=np.array([pos/F(count),neg/F(count),F(np.count_nonzero(a>0))/F(count),F(np.count_nonzero(a<0))/F(count),
                np.sum(edges*mt,dtype=np.float32)/mass if mass!=0 else F(0),
                np.sum((edges*edges)*mt,dtype=np.float32)/mass if mass!=0 else F(0)],dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(mt).all():raise ValueError('reference numerical domain exceeded')
    return dict(known=x,positive_sum=pos,negative_sum=neg,mass_t=mt,spatial_count=P,time_count=10)

def unknown_bounds(cheap):
    """Accepts only legitimately computed six-feature metadata, never a voxel."""
    P=int(cheap['spatial_count']);T=int(cheap['time_count']);Q=9*P
    if T!=10:raise ValueError('T10 required')
    ap,au=sum_bounds_nonnegative(cheap['positive_sum'],T*P-1)
    bp,bu=sum_bounds_nonnegative(cheap['negative_sum'],T*P-1)
    bounds=[sum_bounds_nonnegative(v,P-1) for v in cheap['mass_t']]
    mhi=F(0)
    for _,upper in bounds:mhi=hi(mhi,upper,'add')
    # Exclude overflow in a subtraction and its full positive reduction, and
    # all signed temporal partial sums, before using their error models.
    if not hi(2,mhi,'mul')<FMAX:raise ValueError('cannot exclude feature overflow')
    ulo=max(F(0),lo(ap,bu,'sub'),lo(bp,au,'sub'))
    g9,b9=constants(9);etime=hi(hi(g9,mhi,'mul'),hi(P,b9,'mul'),'add')
    tlo=max(F(0),lo(ulo,etime,'sub'));thi=hi(mhi,etime,'add')
    tlo,thi=rounded_sum_enclosure(tlo,thi,P-1)
    # Bound the actual rounded, ordered divisions. Endpoint inputs already
    # enclose every possible FP32 reduction output.
    x4=(F(F(tlo/F(P))/F(T)),F(F(thi/F(P))/F(T)))
    dlo,dhi=F(0),F(0)
    for t in range(1,T):
        l,r=bounds[t];pl,pr=bounds[t-1]
        term=max(F(0),lo(l,pr,'sub'),lo(pl,r,'sub'))
        dlo=max(F(0),lo(dlo,term,'add'));dhi=hi(dhi,hi(r,pr,'add'),'add')
    # Constants 1+u and Q*eta are rounded upward from exact rationals.
    qm=rational_endpoint(Q*ETA,True);omu=rational_endpoint(1-U,False);opu=rational_endpoint(1+U,True)
    dlow=max(F(0),lo(lo(omu,dlo,'mul'),qm,'sub'));dhigh=hi(hi(opu,dhi,'mul'),qm,'add')
    dlow,dhigh=rounded_sum_enclosure(dlow,dhigh,Q-1)
    x5=(F(dlow/F(Q)),F(dhigh/F(Q)))
    out=np.array([x4,x5],dtype=np.float32).T
    if not np.isfinite(out).all() or np.any(out[0]>out[1]):raise ValueError('invalid feature interval')
    return out

def complete_cell(a):
    a=np.ascontiguousarray(a,dtype=np.float32)
    temporal=np.abs(np.sum(a,axis=0,dtype=np.float32));diff=np.abs(a[1:]-a[:-1])
    return np.array([(np.sum(temporal,dtype=np.float32)/F(temporal.size))/F(10),np.sum(diff,dtype=np.float32)/F(diff.size)],dtype=np.float32)

def interval(lo_,hi_=None):
    lower=np.asarray(lo_,dtype=np.float32);upper=lower.copy() if hi_ is None else np.asarray(hi_,dtype=np.float32)
    if lower.shape!=upper.shape or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower>upper):raise ValueError('finite ordered endpoints required')
    return lower,upper

def rounded_binary(a,b,op):
    """Enclose the original rounded FP32 graph, not unrounded arithmetic."""
    al,au=a;bl,bu=b
    with np.errstate(over='ignore',invalid='ignore',divide='ignore',under='ignore'):
        if op=='add':lower,upper=al+bl,au+bu
        elif op=='sub':lower,upper=al-bu,au-bl
        elif op=='mul':
            four=np.stack([al*bl,al*bu,au*bl,au*bu]);lower,upper=np.min(four,axis=0),np.max(four,axis=0)
        elif op=='positive_div':
            if np.any(bl<=0) or not np.array_equal(bl,bu):raise ValueError('fixed positive denominator required')
            lower,upper=al/bl,au/bl
        else:raise ValueError(op)
    return interval(lower,upper)

def affine(ridge,x):
    r={k:np.asarray(ridge[k],dtype=np.float32) for k in ('mean','scale','coef','intercept')}
    z=rounded_binary(rounded_binary(x,interval(r['mean']),'sub'),interval(r['scale']),'positive_div')
    out=interval(np.zeros(len(r['intercept']),dtype=np.float32))
    for k in range(len(r['mean'])):
        term=rounded_binary((z[0][k],z[1][k]),interval(r['coef'][k]),'mul');out=rounded_binary(out,term,'add')
    return rounded_binary(out,interval(r['intercept']),'add')

def heads(folded,ordinary,x):
    folded_scores=affine(folded['ridge'],x);f=ordinary['family'];s=ordinary['selection']
    aux=affine(f['auxiliary'],x)
    first=(aux[0][:2][None,:],aux[1][:2][None,:]);responses=(aux[0][2:].reshape(6,2),aux[1][2:].reshape(6,2))
    products=rounded_binary(first,responses,'mul');signed=interval(np.zeros(6,dtype=np.float32))
    for k in range(2):signed=rounded_binary(signed,(products[0][:,k],products[1][:,k]),'add')
    augmented=tuple(np.concatenate([x[i],aux[i],products[i].reshape(-1)]) for i in (0,1))
    residual=affine(f['residual'][str(s['alpha'])],augmented)
    ordinary_scores=rounded_binary(signed,rounded_binary(interval(F(s['shrink'])),residual,'mul'),'add')
    return {'folded128':folded_scores,'ordinary128':ordinary_scores}

def certify(scores):
    lower,upper=interval(*scores)
    if lower.shape!=(6,):raise ValueError('fixed six actions required')
    for a in range(6):
        if all(upper[a]<lower[j] if j<a else upper[a]<=lower[j] for j in range(6) if j!=a):return a
    return None
