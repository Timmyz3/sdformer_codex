"""One frozen old-fit64 certification screen; no GT, fitting, or GPU."""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,os,sys,time,traceback
import numpy as np
import certified_core as c
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
METHODS=('folded128','ordinary128')
CHEAP=55877120;LATE=4204844;BAND=144320
BASE={'folded128':113938624+8599,'ordinary128':113938624+26646}

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 return x
def write(p,v):
 assert not p.exists(),p
 p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n')
def imported(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def cells(z):
 for cy in range(4):
  for cx in range(4):yield np.ascontiguousarray(z[:,cy*120:(cy+1)*120,cx*160:(cx+1)*160])

def one_head(method,fold,ordinary,x):
 if method=='folded128':return c.affine(fold['ridge'],x)
 assert method=='ordinary128'
 f=ordinary['family'];s=ordinary['selection'];aux=c.affine(f['auxiliary'],x)
 products=c.rounded_binary((aux[0][:2][None,:],aux[1][:2][None,:]),(aux[0][2:].reshape(6,2),aux[1][2:].reshape(6,2)),'mul')
 signed=c.interval(np.zeros(6,np.float32))
 for k in range(2):signed=c.rounded_binary(signed,(products[0][:,k],products[1][:,k]),'add')
 augmented=tuple(np.concatenate([x[i],aux[i],products[i].reshape(-1)]) for i in (0,1))
 residual=c.affine(f['residual'][str(s['alpha'])],augmented)
 return c.rounded_binary(signed,c.rounded_binary(c.interval(np.float32(s['shrink'])),residual,'mul'),'add')

def runtime(z,fold,ordinary):
 """No cache/reference/GT argument. Fixed natural schedule for both policies.

 Each head interval graph and numerical guard is independent. Shared CPU raw
 storage is not a deployable free cache. The screen sets ALL head costs to zero.
 """
 lower=np.zeros(128,np.float32);upper=np.zeros(128,np.float32);decisions={};stages=[];zeros=0
 try:
  for i,a in enumerate(cells(z)):
   cheap=c.cheap_cell(a);zeros+=int(np.sum(cheap['mass_t'],dtype=np.float32)==0);unknown=c.unknown_bounds(cheap);offset=i*8
   indexes=np.array([0,1,2,3,6,7])+offset
   lower[indexes]=cheap['known'];upper[indexes]=cheap['known'];lower[offset+4:offset+6]=unknown[0];upper[offset+4:offset+6]=unknown[1]
 except ValueError as error:
  return dict(decisions={m:dict(certified=False,stage=0,reason=str(error)) for m in METHODS},stages=[],known_feature_error=False,zero_mass_cells=None)
 completed=iter(cells(z))
 for stage in range(17):
  if stage:
   late=c.complete_cell(next(completed));offset=(stage-1)*8+4;lower[offset:offset+2]=late;upper[offset:offset+2]=late
  row=dict(stage=stage,feature_lower=lower.copy(),feature_upper=upper.copy(),heads={})
  for method in METHODS:
   if method in decisions:continue
   try:bounds=one_head(method,fold,ordinary,c.interval(lower,upper))
   except ValueError as error:
    row.setdefault('numerical_guard',{})[method]=str(error);decisions[method]=dict(certified=False,stage=stage,reason=str(error));continue
   action=c.certify(bounds)
   row['heads'][method]=dict(lower=bounds[0].copy(),upper=bounds[1].copy(),certified_action=action)
   if action is not None:decisions[method]=dict(certified=True,stage=stage,action=int(action),reason=None)
  stages.append(row)
  if len(decisions)==2:break
 for m in METHODS:
  if m not in decisions:decisions[m]=dict(certified=False,stage=16,reason='all cells completed without a certificate')
 return dict(decisions=decisions,stages=stages,known_feature_error=False,zero_mass_cells=zeros)

def full_reference(z,fold,ordinary,num):
 # Verification or explicit fallback only, outside runtime's information path.
 x=num.voxel_features(z)[:128];fs,_=num.affine_fp32(fold['ridge'],x[None,:]);f=ordinary['family'];s=ordinary['selection'];aux,_=num.affine_fp32(f['auxiliary'],x[None,:]);aug=np.concatenate([x[None,:],aux,num.products(aux,1).reshape(1,-1)],axis=1);res,_=num.affine_fp32(f['residual'][str(s['alpha'])],aug);ordinary_score=num.signed(aux,1)+np.float32(s['shrink'])*res
 return x,dict(folded128=fs[0],ordinary128=ordinary_score[0])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-sha256',required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 assert a.execute and sha(a.review)==a.review_sha256
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):assert os.environ.get(k)=='2'
 freeze=read(HERE/'SOURCE_FREEZE_V1.json')
 for name,h in freeze['files'].items():assert sha(HERE/name)==h
 for b in freeze['bound'].values():assert sha(b['path'])==b['sha256']
 manifest=read(HERE/'INPUT_MANIFEST_V1.json');assert len(manifest['records'])==64 and len({r['sequence'] for r in manifest['records']})==8
 for r in manifest['records']:assert r['role']=='fit' and sha(r['path'])==r['sha256'] and Path(r['path']).stat().st_size==r['bytes']
 modelpaths=freeze['bound'];fold=read(modelpaths['folded_model']['path']);ordinary=read(modelpaths['ordinary_model']['path']);num=imported(modelpaths['numeric']['path'],'r39_full_reference')
 ids=[r['target_id'] for r in manifest['records']];assert ids==fold['fit_ids']==ordinary['fit_ids']
 out=HERE/'actual_fit64_v1';assert not out.exists();out.mkdir();(out/'identities').mkdir();start=time.monotonic();results=[];checked_features=0;checked_scores=0
 try:
  for rec in manifest['records']:
   z=np.load(rec['path'],allow_pickle=False);assert z.shape==(10,480,640) and z.dtype==np.float32 and np.isfinite(z).all()
   actual=runtime(z,fold,ordinary)
   # Only now compute complete features/scores for offline containment checks.
   x,scores=full_reference(z,fold,ordinary,num);decisions=actual['decisions']
   for st in actual['stages']:
    l,u=st['feature_lower'],st['feature_upper'];assert np.all(l<=x) and np.all(x<=u),(rec['target_id'],st['stage'],'feature containment');checked_features+=128
    for name,h in st['heads'].items():
     assert np.all(h['lower']<=scores[name]) and np.all(scores[name]<=h['upper']),(rec['target_id'],st['stage'],name,'score containment');checked_scores+=6
   zeros=actual['zero_mass_cells']
   if zeros is None:
    # Feature-domain fallback pays at least its full original prefix. The
    # complete reference here computes this diagnostic after runtime stopped.
    zeros=sum(np.sum(np.sum(np.abs(cell),axis=(1,2),dtype=np.float32),dtype=np.float32)==0 for cell in cells(z));actual['zero_mass_cells']=int(zeros)
   for name,d in decisions.items():
    ref=int(np.argmin(scores[name]));d['reference_action']=ref
    if d['certified']:assert d['action']==ref,(rec['target_id'],name,'false certificate')
    else:d['action']=ref;d['full_reference_fallback']=True
    k=int(d['stage']);baseline=BASE[name]-32*zeros;optimistic=CHEAP-32*zeros+LATE*k+BAND*((k+3)//4)
    # Never assign savings to an uncertified identity. Actual fallback would
    # additionally pay duplicated work; this screening bound favors stopping.
    if not d['certified']:optimistic=max(baseline,optimistic)
    d.update(optimistic_prefix_ticks=optimistic,original_prefix_ticks=baseline,optimistic_saved_fraction=1-optimistic/baseline,bound_head_control_metadata_cost_assumed_zero=True)
   record=dict(target_id=rec['target_id'],sequence=rec['sequence'],input_sha256=rec['sha256'],reference_features=x,reference_scores=scores,**actual)
   write(out/'identities'/(rec['target_id']+'.json'),record);results.append(dict(target_id=rec['target_id'],sequence=rec['sequence'],zero_mass_cells=int(zeros),decisions=decisions));print('FIT_CERTIFIED',len(results),{k:d['stage'] for k,d in decisions.items()},flush=True)
 except BaseException as error:
  write(out/'FAILURE_V1.json',dict(error=str(error),traceback=traceback.format_exc(),completed_identities=len(results),no_gain_claim=True));raise
 summary={};seqs=list(dict.fromkeys(r['sequence'] for r in results))
 for name in METHODS:
  scoped={}
  for scope in ['overall']+seqs:
   ds=[r['decisions'][name] for r in results if scope=='overall' or r['sequence']==scope];mean=sum(d['optimistic_prefix_ticks'] for d in ds)/len(ds);base_mean=sum(d['original_prefix_ticks'] for d in ds)/len(ds)
   scoped[scope]=dict(identities=len(ds),stage_histogram=dict(collections.Counter(str(d['stage']) for d in ds)),mean_stage=float(np.mean([d['stage'] for d in ds])),fallbacks=sum(not d['certified'] for d in ds),original_mean_prefix_ticks=base_mean,optimistic_mean_prefix_ticks=mean,optimistic_mean_saved_fraction=1-mean/base_mean,optimistic_worst_prefix_ticks=max(d['optimistic_prefix_ticks'] for d in ds),optimistic_worst_saved_fraction=min(d['optimistic_saved_fraction'] for d in ds))
  summary[name]=scoped
 go={name:summary[name]['overall']['optimistic_mean_saved_fraction']>=.1 for name in METHODS}
 write(out/'SUMMARY_V1.json',dict(schema='flow_r39_fit64_certification_screen_v1',status='PASS_NUMERICAL_SCREEN_COMPLETE',results=summary,necessary_optimistic_10pct_screen=go,continue_exposed_transport_allowed=any(go.values()),not_deployability_GO=True,source_freeze_sha256=sha(HERE/'SOURCE_FREEZE_V1.json'),input_manifest_sha256=sha(HERE/'INPUT_MANIFEST_V1.json'),review_sha256=a.review_sha256,checked_feature_bounds=checked_features,checked_score_bounds=checked_scores,all_certified_actions_exact=True,GT_read=False,training=False,GPU_used=False,complete_reference_only_for_validation_or_fallback=True,CPU_seconds=time.monotonic()-start,identities=results))
 print('FIT64_SCREEN_COMPLETE',go,round(time.monotonic()-start,3),flush=True)
if __name__=='__main__':main()
