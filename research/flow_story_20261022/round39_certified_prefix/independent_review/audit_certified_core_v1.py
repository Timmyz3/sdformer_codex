"""Independent R39 implementation preflight: exact rational and synthetic only."""
from pathlib import Path
from fractions import Fraction as Q
import hashlib,importlib.util,itertools,json,os
import numpy as np
HERE=Path(__file__).resolve().parent;B=HERE.parent;ROOT=HERE.parents[3];F=np.float32
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def frac(x):return Q.from_float(float(x))
def reference_affine(m,x):
 co,mu,sd,bi=[np.asarray(m[k],F) for k in ['coef','mean','scale','intercept']]
 z=(x-mu)/sd;v=np.zeros(len(bi),F)
 for k in range(len(x)):v=np.add(v,np.multiply(z[k],co[k]))
 return np.add(v,bi)
def reference_heads(f,o,x):
 fs=reference_affine(f['ridge'],x);aux=reference_affine(o['family']['auxiliary'],x)
 prod=np.multiply(aux[:2][None,:],aux[2:].reshape(6,2));s=np.zeros(6,F)
 for k in range(2):s=np.add(s,prod[:,k])
 aug=np.concatenate([x,aux,prod.reshape(-1)]);rs=reference_affine(o['family']['residual'][str(o['selection']['alpha'])],aug)
 return {'folded128':fs,'ordinary128':np.add(s,np.multiply(F(o['selection']['shrink']),rs))}
def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='';out=HERE/'CORE_IMPLEMENTATION_PREFLIGHT_V1.json';assert not out.exists()
 c=module(B/'algorithm/certified_core.py','_R39_under_review');freeze=read(B/'algorithm/SYNTHETIC_SOURCE_FREEZE_V1.json')
 # The implementation's synthetic receipt independently pins the same source.
 synt=read(B/'algorithm/SYNTHETIC_PREFLIGHT_V1.json');assert synt['status']=='PASS_SYNTHETIC_ONLY'
 for p,h in synt['sources'].items():assert sha(Path(p))==h
 mr=ROOT/'research/flow_story_20261022/round38_frozen_survival/model_sources';f=read(mr/'FOLDED_MODEL_V1.json');o=read(mr/'ORDINARY_MODEL_128_V1.json')
 assert sha(mr/'FOLDED_MODEL_V1.json')=='c9f05bff6666bab108e6561968859b5261e34e89578e472256bba4afd257946c'
 assert sha(mr/'ORDINARY_MODEL_128_V1.json')=='052d86b89613e0ed30d7165879b8b8909ea452869126048bba8302b9db7cb541'
 numeric=module(mr/'numeric_frozen.py','_old_numeric_reference')
 unit=Q(1,2**24);eta=Q(1,2**150);constants=[]
 for n in [0,1,7,9,119,19199,172799,191999]:
  g,b=c.constants(n);assert frac(g)>=n*unit/(1-n*unit) and frac(b)>=n*eta/(1-n*unit)
  if n:assert b>0
  constants.append(dict(n=n,gamma_upper=float(g),beta_upper=float(b)))
 for value in [eta,9*eta,172800*eta,1+unit,1-unit]:
  lo,hi=c.rational_endpoint(value,False),c.rational_endpoint(value,True);assert frac(lo)<=value<=frac(hi)
 minimum=np.nextafter(F(0),F(1),dtype=F)
 assert minimum>0 and F(minimum/F(2))==0 and F(F(1)+F(2**-24))==1
 rng=np.random.default_rng(390023);vals=np.array([0.,-0.,minimum,-minimum,2**-126,-2**-126,1.,-1.,np.nextafter(F(1),F(2)),1e-20,-1e-20,1e20,-1e20,np.finfo(F).max/4,-np.finfo(F).max/4],F)
 primitive_checks=0;guard_refusals=0
 for a,b in itertools.product(vals,repeat=2):
  for op in ['add','sub','mul','div']:
   if op=='div' and b==0:continue
   exact={'add':lambda:frac(a)+frac(b),'sub':lambda:frac(a)-frac(b),'mul':lambda:frac(a)*frac(b),'div':lambda:frac(a)/frac(b)}[op]()
   try:lo,hi=c.lo(a,b,op),c.hi(a,b,op)
   except ValueError:guard_refusals+=1;continue
   assert frac(lo)<=exact<=frac(hi),(a,b,op,lo,hi,exact);primitive_checks+=1
 # Full-size cells, but entirely synthetic; original FP32 formula is repeated
 # independently instead of using complete_cell as its own reference.
 shape=(10,120,160);base=np.ones(shape,F);base[:,:,80:]=-1
 fixtures={'zero':np.zeros(shape,F),'constant_positive':np.ones(shape,F),'constant_signed':base,
  'alternating_signed':base*np.where(np.arange(10)%2==0,F(1),F(-1))[:,None,None],
  'subnormal_positive':np.full(shape,minimum,F),'subnormal_signed':base*minimum,
  'random':rng.normal(0,3,shape).astype(F),'large_constant':np.full(shape,F(1e30),F)}
 checks=[]
 for name,a in fixtures.items():
  cheap=c.cheap_cell(a);lo,hi=c.unknown_bounds(cheap)
  actual=np.array([F(np.sum(np.abs(np.sum(a,axis=0,dtype=F)),dtype=F)/F(19200))/F(10),np.sum(np.abs(np.subtract(a[1:],a[:-1])),dtype=F)/F(172800)],F)
  assert np.array_equal(actual,c.complete_cell(a)) and np.all(lo<=actual) and np.all(actual<=hi)
  checks.append(dict(name=name,actual=actual.astype(float).tolist(),lower=lo.astype(float).tolist(),upper=hi.astype(float).tolist()))
 # Verify cheap/full decomposition against the untouched primary numeric code.
 z=rng.normal(size=(10,80,96)).astype(F);ref=numeric.voxel_features(z)[:128];features=[]
 for cy,cx in itertools.product(range(4),repeat=2):
  a=z[:,cy*20:(cy+1)*20,cx*24:(cx+1)*24];q=c.cheap_cell(a)['known'];late=c.complete_cell(a);features.extend([*q[:4],*late,*q[4:]])
 assert np.array_equal(ref,np.array(features,F))
 point_checks=0;singleton_checks=0;certified_checks=0
 for width in [0.,.001,.03,.2]:
  for _ in range(5):
   mid=rng.normal(size=128).astype(F);lower=np.subtract(mid,F(width));upper=np.add(mid,F(width));bounds=c.heads(f,o,c.interval(lower,upper))
   for x in [mid,lower,upper,np.where(rng.integers(0,2,128),lower,upper).astype(F)]:
    actual=reference_heads(f,o,x)
    for name,score in actual.items():
     lo,hi=bounds[name];assert np.all(lo<=score) and np.all(score<=hi);point_checks+=1
     if width==0:assert np.array_equal(lo,score) and np.array_equal(hi,score);singleton_checks+=1
     action=c.certify(bounds[name])
     if action is not None:assert action==int(np.argmin(score));certified_checks+=1
 tie_boxes=0
 for case in range(12):
  lo=rng.integers(-2,3,6).astype(F);hi=lo+rng.integers(0,2,6).astype(F);winner=c.certify(c.interval(lo,hi))
  for bits in itertools.product([0,1],repeat=6):
   score=np.where(bits,hi,lo)
   if winner is not None:assert winner==int(np.argmin(score))
  tie_boxes+=1
 for i in range(6):
  scores=np.ones(6,F);scores[i:]=0;assert c.certify(c.interval(scores))==i
 collision=read(HERE/'SYNTHETIC_COUNTEREXAMPLE_V1.json');assert collision['all16_cells_cheap_values_bitwise_equal']
 report=dict(status='PASS_FIXED_CORE_CPU_FEASIBILITY_GO',auditor_sha256=sha(__file__),core_sha256=sha(B/'algorithm/certified_core.py'),entry_protocol_sha256=sha(B/'ENTRY_PROTOCOL_V1.md'),synthetic_freeze_sha256=sha(B/'algorithm/SYNTHETIC_SOURCE_FREEZE_V1.json'),author_synthetic_receipt_sha256=sha(B/'algorithm/SYNTHETIC_PREFLIGHT_V1.json'),constants=constants,
  exact_rational_primitive_checks=primitive_checks,nonfinite_outward_guard_refusals=guard_refusals,independent_full_size_cell_fixtures=checks,original_feature_decomposition_exact=True,independent_head_point_checks=point_checks,singleton_head_checks=singleton_checks,certified_point_checks=certified_checks,exhaustive_binary_tie_boxes=tie_boxes,
  accepted_scope='Execute one source/input-frozen fixed-rule fit64 CPU attainability screen. Mathematical/source preflight plus independent synthetic checks; no measured certification rate, net saving or deployable FPU implementation implied.',
  runner_source_and_membership_freeze_required=True,no_new_RTL_prerequisite_for_this_CPU_screen=True,GT_values_read=0,real_input_arrays_read=0,new_fits=0,GPU_calls=0)
 out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:report[k] for k in ['status','core_sha256','exact_rational_primitive_checks','independent_head_point_checks','singleton_head_checks','certified_point_checks','exhaustive_binary_tie_boxes']}))
if __name__=='__main__':main()
