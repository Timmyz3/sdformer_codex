"""Independent synthetic R22 q-fit review. No actual feature/GT arrays read."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse,contextlib,copy,hashlib,importlib.util,io,json,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock
import numpy as np
HERE=Path(__file__).resolve().parent;RUN=HERE.parent;DATE=RUN.parent
FIT=RUN/'analysis/fit_motion_projection.py';CORE=DATE/'motion_structure_round21/design/structure_models.py'
ROLE=DATE/'action_domain_round20/independent_review/SELECTOR_ROLES_V1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
fit=load(FIT,'_independent_fit_review');mathcore=load(CORE,'_independent_math_review')
ROLE_DATA=json.loads(ROLE.read_text());MEASURE=[]
def dump(p,x):p.write_text(json.dumps(x));return {'path':str(p),'sha256':sha(p)}
class ArrayBox:
 def __init__(self,u):self.u=u
 def __enter__(self):return {'u':self.u}
 def __exit__(self,*args):pass

def fixture(folder,change=None,run_arrays=False):
 rows=ROLE_DATA['groups']['fit']+ROLE_DATA['groups']['development'];members=[];records=[];arraymap={}
 for i,row in enumerate(rows):
  tid=row['target_id'];role='fit' if i<16 else 'development'
  archive=folder/(tid+'.npz');archive.write_bytes(('synthetic-native-'+tid).encode())
  u=np.broadcast_to(np.float32(i+1),(10,1,30,40,384))
  arraymap[str(archive)]=('u',u,role)
  rec={'target_id':tid,'selector_role':role,'arm':'native','quality_endpoint':True,'execution':'native',
       'prediction_file':archive.name,'prediction_file_sha256':sha(archive),'tensor_records':{'u':{'sha256':hashlib.sha256(np.ascontiguousarray(u).tobytes()).hexdigest()}},'input_evidence':{}}
  member={'target_id':tid,'sequence':row['sequence'],'role':role,'native_archive':{'path':str(archive),'sha256':sha(archive)}}
  if role=='fit':
   for key,kind in (('fit_GT','gt'),('fit_mask','mask')):
    p=folder/(tid+'_'+kind+'.npy');p.write_bytes(('synthetic-'+kind+'-'+tid).encode())
    member[key]={'path':str(p),'sha256':sha(p)};rec['input_evidence'][kind]=sha(p)
    arraymap[str(p)]=(kind,i+1,role)
  members.append(member);records.append(rec)
 records[1:1]=[dict(records[0],arm='native_repeat1',quality_endpoint=False),dict(records[0],arm='native_repeat2',quality_endpoint=False)]
 graph='r22_W271482_all78current_native_fit16dev8'
 execution={'graph_id':graph,'status':'REPRESENTATION_NATIVE26_COMPLETE','full_forward_count':26,'binding':{'selector_roles_sha256':sha(ROLE),'selector_role_map':{r['target_id']:r['role'] for r in members}},'forward_records':records}
 audit={'schema':'flow_r22_representation_independent_execution_audit_v1','status':'PASS','representation_artifacts_allowed':True,'quality_analysis_allowed':False}
 inp={'schema':'flow_r22_motion_projection_fit_inputs_v1','graph_id':graph,'source_pins':[{'path':str(p),'sha256':sha(p)} for p in (FIT,CORE)],'roles':{'path':str(ROLE),'sha256':sha(ROLE)},'members':members}
 if change:change(inp,execution,audit)
 inp['execution_result']=dump(folder/'execution.json',execution);audit['source_result_sha256']=inp['execution_result']['sha256']
 inp['execution_audit']=dump(folder/'audit.json',audit);manifest=folder/'inputs.json';dump(manifest,inp)
 calls=[];fits=[]
 def fake_load(path,**kwargs):
  require_path=str(Path(path));kind,value,role=arraymap[require_path];calls.append((kind,role,require_path))
  if not run_arrays:raise RuntimeError('ARRAY_BOUNDARY_REACHED')
  if kind=='u':return ArrayBox(value)
  if role!='fit':raise AssertionError('DEV_GT_READ')
  if kind=='gt':return np.full((2,480,640),value,np.float32)
  return np.ones((480,640),bool)
 original_spec=importlib.util.spec_from_file_location
 def spec_hook(name,path,*args,**kwargs):
  spec=original_spec(name,path,*args,**kwargs)
  if name=='_r22_frozen_structure_math':
   original_exec=spec.loader.exec_module
   def exec_module(module):
    original_exec(module)
    def spy_ridge(x,y,n,alpha=10.):
     ids=[int(v) for v in x[:,0,0,0]];fits.append(ids)
     return {'weight':np.zeros((384,2),np.float32),'bias':np.full(2,np.mean(ids),np.float32),
             'fit_cells':int((n>0).sum()),'fit_valid_pixels':int(n.sum()),'alpha':10.}
    module.ridge_motion=spy_ridge
   spec.loader.exec_module=exec_module
  return spec
 argv=['fit','--inputs',str(manifest),'--inputs-sha256',sha(manifest),'--output',str(folder/'out')]
 with mock.patch.object(sys,'argv',argv),mock.patch.object(np,'load',side_effect=fake_load),mock.patch.object(importlib.util,'spec_from_file_location',side_effect=spec_hook),contextlib.redirect_stdout(io.StringIO()):
  fit.main()
 return json.loads((folder/'out/predictions.json').read_text()),calls,fits

class Checks(unittest.TestCase):
 def test_pixel_labels_axes_support_and_units(self):
  yy,xx=np.indices((480,640));gt=np.stack((2.*xx+yy,xx-3.*yy)).astype(np.float32);valid=(xx+yy)%7!=0
  gt[:,~valid]=np.nan
  labels,counts=mathcore.pixel_labels(gt,valid)
  for y,x in ((0,0),(7,17),(29,39)):
   mask=valid[16*y:16*y+16,16*x:16*x+16];block=gt[:,16*y:16*y+16,16*x:16*x+16]
   np.testing.assert_allclose(labels[y,x],block[:,mask].astype(np.float64).mean(1),rtol=0,atol=1e-10)
   self.assertEqual(counts[y,x],int(mask.sum()))
  self.assertEqual(counts.sum(),valid.sum())
  labels2,_=mathcore.pixel_labels(np.full((2,480,640),16,np.float32),np.ones((480,640),bool))
  np.testing.assert_array_equal(labels2,16.)
  empty=np.zeros((480,640),bool);empty[:16,:16]=True
  z,n=mathcore.pixel_labels(np.ones((2,480,640),np.float32),empty)
  self.assertEqual(n[1,1],0);np.testing.assert_array_equal(z[1,1],0.)
 def test_weighted_ridge_independent_augmented_least_squares(self):
  rng=np.random.default_rng(71);x=np.zeros((1,20,30,384));x[...,:3]=rng.normal(size=(1,20,30,3));x[...,3]=7.
  truth=np.stack((2*x[...,0]-.4*x[...,1]+3,-.5*x[...,0]+x[...,2]-2),-1)
  counts=(np.arange(600).reshape(1,20,30)%7+1).astype(float);counts[:,0,0]=0
  model=mathcore.ridge_motion(x,truth,counts)
  keep=counts>0;a=x[keep];y=truth[keep];w=counts[keep];w=w/np.mean(w)
  mean=np.average(a,axis=0,weights=w);std=np.sqrt(np.average((a-mean)**2,axis=0,weights=w));std[std<=1e-6]=1.
  design=np.column_stack((np.ones(len(a)),(a-mean)/std));reg=np.zeros((384,385));reg[:,1:]=np.eye(384)*np.sqrt(10.)
  coef=np.linalg.lstsq(np.vstack((design*np.sqrt(w[:,None]),reg)),np.vstack((y*np.sqrt(w[:,None]),np.zeros((384,2)))),rcond=None)[0]
  wanted=(design@coef).astype(np.float32);got=mathcore.project(model,a)
  np.testing.assert_allclose(got,wanted,rtol=2e-6,atol=2e-6)
  self.assertEqual(model['fit_valid_pixels'],int(counts.sum()));self.assertEqual(model['fit_cells'],599)
  MEASURE.append({'ridge_independent_augmented_LS_max_abs':float(np.max(np.abs(got-wanted)))})
 def test_pool_true_tail(self):
  x=np.arange(2*30*40*2,dtype=np.float32).reshape(2,30,40,2);got=mathcore.pool_expand3(x)
  for y in range(0,30,3):
   for z in range(0,40,3):
    block=x[:,y:y+3,z:z+3];want=block.mean((1,2),dtype=np.float32)
    np.testing.assert_array_equal(got[:,y:y+3,z:z+3],np.broadcast_to(want[:,None,None],block.shape))
 def test_fixed_metadata_refusals_before_arrays(self):
  def swapped(inp,ex,au):
   inp['members'][0]['sequence'],inp['members'][2]['sequence']=inp['members'][2]['sequence'],inp['members'][0]['sequence']
  cases=[('LOO sequence',swapped),
   ('role binding',lambda i,e,a:e['binding'].__setitem__('selector_roles_sha256','0'*64)),
   ('unique original',lambda i,e,a:e['forward_records'].__setitem__(3,copy.deepcopy(e['forward_records'][0]))),
   ('No development',lambda i,e,a:i['members'][16].__setitem__('fit_GT',{'path':'DEV_FORBIDDEN','sha256':'0'*64})),
   ('admission',lambda i,e,a:a.__setitem__('quality_analysis_allowed',True)),
   ('complete native26',lambda i,e,a:e.__setitem__('full_forward_count',25))]
  for expected,change in cases:
   with self.subTest(expected=expected),tempfile.TemporaryDirectory() as temp:
    with self.assertRaisesRegex(ValueError,expected):fixture(Path(temp),change)
  with tempfile.TemporaryDirectory() as temp:
   with self.assertRaisesRegex(RuntimeError,'ARRAY_BOUNDARY_REACHED'):fixture(Path(temp))
  MEASURE.append({'metadata_negative_cases':6,'sequence_swap_preserves_eight_groups_of_two':True,'all_rejected_before_any_array_load':True})
 def test_oof_and_dev_fit_isolation_with_spy_projection(self):
  with tempfile.TemporaryDirectory() as temp:out,calls,fits=fixture(Path(temp),run_arrays=True)
  self.assertEqual(len(fits),9)
  for fold,ids in zip(out['q_folds'],fits[:8]):
   self.assertEqual(len(ids),14);self.assertEqual(len(fold['held_ids']),2)
   mapping={r['target_id']:r['sequence'] for r in ROLE_DATA['groups']['fit']}
   self.assertTrue(all(mapping[t]!=fold['sequence'] for t in fold['train_ids']))
   self.assertTrue(all(mapping[t]==fold['sequence'] for t in fold['held_ids']))
  self.assertEqual(fits[-1],list(range(1,17)))
  self.assertEqual(sum(k=='gt' for k,r,p in calls),16);self.assertEqual(sum(k=='mask' for k,r,p in calls),16)
  self.assertFalse(any(k!='u' and role!='fit' for k,role,path in calls))
  self.assertEqual(len(out['predictions']),24)
  self.assertEqual(out['status'],'FROZEN_BEFORE_DEVELOPMENT_QUALITY')
  MEASURE.append({'scope_test_uses_spy_ridge_not_empirical_fit':True,'OOF_fits':8,'OOF_train_held':[14,2],'final_fit_inputs':16,'dev_prediction_inputs':8,'synthetic_fit_GT_loads':16,'dev_or_sealed_GT_loads':0})

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();assert not args.output.exists()
 start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 out={'schema':'flow_r22_independent_qfit_synthetic_review_v1','status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_sha256':{'fitter':sha(FIT),'math':sha(CORE),'review_script':sha(__file__)},'measurements':MEASURE,'real_feature_or_GT_arrays_read':False,'GPU_used':False,'wall_seconds':time.monotonic()-start}
 args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));raise SystemExit(0 if result.wasSuccessful() else 1)
