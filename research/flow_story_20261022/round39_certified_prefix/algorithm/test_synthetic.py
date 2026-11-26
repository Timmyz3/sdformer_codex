"""Synthetic-only preflight. Reads frozen models/source, never dataset arrays."""
from pathlib import Path
from fractions import Fraction
import importlib.util,json,hashlib,os,time
import numpy as np
import certified_core as c
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
MODELROOT=ROOT/'research/flow_story_20261022/round38_frozen_survival/model_sources'

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def frac(v):return Fraction.from_float(float(v))

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 start=time.monotonic();rng=np.random.default_rng(20261022);numeric=load(MODELROOT/'numeric_frozen.py','r39_frozen_numeric_test')
 fold=json.loads((MODELROOT/'FOLDED_MODEL_V1.json').read_text());ordinary=json.loads((MODELROOT/'ORDINARY_MODEL_128_V1.json').read_text())
 constant_checks=[]
 for n in (0,9,19199,191999,172799):
  g,b=c.constants(n);assert frac(g)>=n*c.U/(1-n*c.U) and frac(b)>=n*c.ETA/(1-n*c.U)
  if n:assert b>0
  constant_checks.append(dict(n=n,gamma_upper=float(g),beta_upper=float(b)))
 primitive_count=0
 vals=[c.F(0),c.F(1),c.F(-1),np.nextafter(c.F(0),c.F(1),dtype=np.float32),c.F(2**-126)]+list(rng.uniform(-10,10,32).astype(np.float32))
 for a in vals:
  for b in vals:
   for op in ('add','sub','mul','div'):
    if op=='div' and b==0:continue
    if op=='add':exact=frac(a)+frac(b)
    elif op=='sub':exact=frac(a)-frac(b)
    elif op=='mul':exact=frac(a)*frac(b)
    else:exact=frac(a)/frac(b)
    try:l=c.lo(a,b,op);u=c.hi(a,b,op)
    except ValueError:continue
    assert frac(l)<=exact<=frac(u),(a,b,op,l,u,exact)
    primitive_count+=1
 shape=(10,120,160);a=np.ones(shape,np.float32);a[:,:,80:]=-1;b=a.copy();b[1::2]*=-1
 small=np.nextafter(c.F(0),c.F(1),dtype=np.float32)
 fixtures={'constant_half_sign':a,'alternating_half_sign':b,'zero':np.zeros(shape,np.float32),'subnormal':np.full(shape,small,np.float32),'random':rng.normal(0,2,shape).astype(np.float32)}
 impulse=np.zeros(shape,np.float32);impulse[4,7,9]=3;fixtures['impulse']=impulse
 checks=[]
 for name,z in fixtures.items():
  cheap=c.cheap_cell(z);bounds=c.unknown_bounds(cheap);actual=c.complete_cell(z)
  assert np.all(actual>=bounds[0]) and np.all(actual<=bounds[1]),(name,bounds,actual)
  checks.append(dict(name=name,bounds=bounds.astype(float).tolist(),actual=actual.astype(float).tolist()))
 ca,cb=c.cheap_cell(a),c.cheap_cell(b)
 assert np.array_equal(ca['known'],cb['known']) and np.array_equal(ca['mass_t'],cb['mass_t']) and ca['positive_sum']==cb['positive_sum'] and ca['negative_sum']==cb['negative_sum']
 assert np.array_equal(c.complete_cell(a),[1,0]) and np.array_equal(c.complete_cell(b),[0,2])
 assert np.array_equal(c.unknown_bounds(ca),c.unknown_bounds(cb))
 # Smaller synthetic whole image exercises all16 cheap/full cell assembly.
 z=rng.normal(size=(10,40,48)).astype(np.float32);old=numeric.voxel_features(z)[:128];assembled=[]
 for cy in range(4):
  for cx in range(4):
   cell=z[:,cy*10:(cy+1)*10,cx*12:(cx+1)*12];v=c.cheap_cell(cell)['known'];late=c.complete_cell(cell);assembled.extend([*v[:4],*late,*v[4:]])
 assert np.array_equal(old,np.asarray(assembled,dtype=np.float32))
 def point_scores(x):
  fs,_=numeric.affine_fp32(fold['ridge'],x[None,:]);f=ordinary['family'];s=ordinary['selection'];aux,_=numeric.affine_fp32(f['auxiliary'],x[None,:]);aug=np.concatenate([x[None,:],aux,numeric.products(aux,1).reshape(1,-1)],axis=1);res,_=numeric.affine_fp32(f['residual'][str(s['alpha'])],aug);os_=numeric.signed(aux,1)+np.float32(s['shrink'])*res
  return dict(folded128=fs[0],ordinary128=os_[0])
 max_width={k:0.0 for k in ('folded128','ordinary128')};head_points=0;certificates=0
 for sample in range(24):
  center=rng.normal(size=128).astype(np.float32);width=np.float32(0 if sample<8 else .01 if sample<16 else .1);lower=(center-width).astype(np.float32);upper=(center+width).astype(np.float32);bound=c.heads(fold,ordinary,c.interval(lower,upper))
  for k,v in bound.items():max_width[k]=max(max_width[k],float(np.max(v[1]-v[0])))
  for j in range(5):
   x=center if j==0 else lower if j==1 else upper if j==2 else (lower+(upper-lower)*rng.random(128).astype(np.float32)).astype(np.float32)
   actual=point_scores(x)
   for name,v in actual.items():
    l,u=bound[name];assert np.all(l<=v) and np.all(v<=u)
    if width==0:assert np.array_equal(l,v) and np.array_equal(u,v)
    winner=c.certify(bound[name])
    if winner is not None:assert winner==int(np.argmin(v));certificates+=1
   head_points+=1
 assert c.certify(c.interval(np.zeros(6,np.float32)))==0
 assert c.certify(c.interval(np.array([1,0,0,2,2,2],np.float32)))==1
 assert c.certify(c.interval(np.zeros(6,np.float32),np.ones(6,np.float32))) is None
 bad=dict(ca);bad['mass_t']=np.full(10,np.finfo(np.float32).max,np.float32)
 try:c.unknown_bounds(bad)
 except ValueError:overflow_refused=True
 else:raise AssertionError('overflow guard did not refuse')
 out=HERE/'SYNTHETIC_PREFLIGHT_V1.json';assert not out.exists()
 write(out,dict(status='PASS_SYNTHETIC_ONLY',constant_checks=constant_checks,exact_rational_outward_operation_checks=primitive_count,fixtures=checks,cheap_collision_proven=True,synthetic_full_feature_exact=True,head_sample_points=head_points,singleton_head_exact=True,certified_sample_checks=certificates,head_max_interval_width=max_width,overflow_refused=overflow_refused,low_id_ties_exact=True,real_input_arrays_read=0,GT_read=False,new_fits=0,GPU_used=False,seconds=time.monotonic()-start,sources={str(p):sha(p) for p in (HERE/'certified_core.py',Path(__file__),MODELROOT/'numeric_frozen.py',MODELROOT/'FOLDED_MODEL_V1.json',MODELROOT/'ORDINARY_MODEL_128_V1.json')}))
 print('SYNTHETIC_ONLY_PASS',primitive_count,head_points,flush=True)
if __name__=='__main__':main()
