#!/usr/bin/env python3
"""Synthetic rejection checks for independent R14 auditor; no new outputs read."""
import copy,hashlib,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
import audit_conversion as a


def fake_members():return {'check_members':[{'target_id':'id'+str(i),'sequence':'seq'+str(i//4)} for i in range(8)]}
def fake_execution():
 ex={'all12_observed_fc_rows':{n:{'fc1_rows':a.G.expected_layer(n,'native')[0],'fc2_rows':a.G.expected_layer(n,'native')[0]} for n in a.G.GEOMETRY},
     'summed_fc1_rows':558000,'summed_fc2_rows':558000,'anchor':'native','mode':'actual','layers':{},
     'head_records':[{'shape':[1,2,480,640],'dtype':'torch.float32','finite':True,'sha256':'a'*64} for _ in range(4)],
     'branch_records':{n:{'shape':list(a.G.GEOMETRY[n]),'input_sha256':'b'*64,'output_sha256':'c'*64} for n in a.G.SELECTED}}
 return ex

def fake_scope():
 names={f'bn{i:02d}' for i in range(78)}
 scope={'scope':'ALL78_CURRENT','BN_count':78,'private_BN_current':24,'external_BN_current':54,'frozen_BN_names':[],
  'current_stat_BN_names':sorted(names),'non_BN_parameters_and_affine_unchanged':True,'source_BN_buffers_removed_before_inference':True,
  'calibration_performed':False,'external_BN_artifact_loaded':False,
  'BN_signature_before_inference':{n:dict(training=False,tracking=False,running_mean=None,running_var=None,num_batches_tracked=None) for n in names}}
 return scope,names

class Tests(unittest.TestCase):
 def test_plan_and_actual16_service_recomputed(self):
  m=fake_members();p=a.schedule(m);d={'full_forward_count':16,'forward_records':[dict(r,execution=fake_execution()) for r in p]}
  counts,lookup=a.verify_records(d,m)
  self.assertEqual((counts['FC1_rows'],counts['FC2_rows'],counts['observed_FC_modules'],len(lookup)),(8928000,8928000,384,16))
  for case in ('missing','wrongorder','wrongW','oneFC','sum','branch','head','wrapper'):
   bad=copy.deepcopy(d);r=bad['forward_records'][0];ex=r['execution']
   if case=='missing':bad['forward_records'].pop()
   elif case=='wrongorder':bad['forward_records'].reverse()
   elif case=='wrongW':r['checkpoint_step']=271482
   elif case=='oneFC':next(iter(ex['all12_observed_fc_rows'].values()))['fc1_rows']-=1
   elif case=='sum':ex['summed_fc2_rows']-=1
   elif case=='branch':ex['branch_records'].pop(next(iter(ex['branch_records'])))
   elif case=='head':ex['head_records'][3]['shape']=[1,2,120,160]
   else:ex['layers']['wrong']={}
   with self.assertRaises(ValueError,msg=case):a.verify_records(bad,m)
 def test_all78_current_scope_rejections(self):
  scope,names=fake_scope();a.verify_scope(scope,names)
  for case in ('missing','tracking','buffer','training','affine','imported','calibration','private'):
   bad=copy.deepcopy(scope);key=sorted(names)[0]
   if case=='missing':bad['BN_signature_before_inference'].pop(key)
   elif case in ('tracking','training'):bad['BN_signature_before_inference'][key][case]=True
   elif case=='buffer':bad['BN_signature_before_inference'][key]['running_mean']='0'*64
   elif case=='affine':bad['non_BN_parameters_and_affine_unchanged']=False
   elif case=='imported':bad['external_BN_artifact_loaded']=True
   elif case=='calibration':bad['calibration_performed']=True
   else:bad['frozen_BN_names']=[key]
   with self.assertRaises(ValueError,msg=case):a.verify_scope(bad,names)
 def test_actual_archive_payload_and_shape(self):
  with tempfile.TemporaryDirectory() as root:
   p=Path(root)/'pred.npz';value=np.ones((2,480,640),np.float32);np.savez_compressed(p,native=value)
   digest=hashlib.sha256(value.tobytes()).hexdigest();a.read_prediction(p,a.sha(p),digest)
   with self.assertRaises(ValueError):a.read_prediction(p,'0'*64,digest)
   with self.assertRaises(ValueError):a.read_prediction(p,a.sha(p),'0'*64)
   np.savez_compressed(p,native=value[:,:2]);
   with self.assertRaises(ValueError):a.read_prediction(p,a.sha(p),hashlib.sha256(value[:,:2].tobytes()).hexdigest())
   with self.assertRaises(ValueError):a.local(Path(root),'../outside.npz')
 def test_unique_fixed_members_and_native_geometry(self):
  m=fake_members();m['check_members'][0]=m['check_members'][1]
  with self.assertRaises(ValueError):a.schedule(m)
  total=sum(t*b*h*w for t,b,h,w,c in a.G.GEOMETRY.values());self.assertEqual(total,558000)
  self.assertEqual(sum(a.G.expected_layer(n,'native')[0] for n in a.G.GEOMETRY),total)
  self.assertEqual([r['checkpoint_step'] for r in a.schedule(fake_members())],[135930]*8+[271482]*8)

if __name__=='__main__':
 stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 out={'schema':'flow_round14_independent_auditor_cpu_v1','status':'PASS' if r.wasSuccessful() else 'FAIL','tests_run':r.testsRun,
  'auditor_sha256':a.sha(Path(a.__file__)),'test_sha256':a.sha(Path(__file__)),'test_log':stream.getvalue(),'scope':'Synthetic16 records, BN scope and actual small archive bytes; no R14 actual results/quality/GPU.'}
 path=Path(__file__).parent/'AUDITOR_CPU_V1.json';assert not path.exists();path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print(stream.getvalue());print(json.dumps({'status':out['status'],'receipt_sha256':a.sha(path)}));raise SystemExit(0 if r.wasSuccessful() else 1)
