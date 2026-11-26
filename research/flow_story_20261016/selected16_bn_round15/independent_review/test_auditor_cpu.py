#!/usr/bin/env python3
"""Independent S8 audit tests; synthetic execution and actual source-state CPU only."""
import copy,hashlib,io,json,unittest
from pathlib import Path
import numpy as np
import audit_selected16 as a

def members():return {'check_members':[{'target_id':'id'+str(i),'sequence':'seq'+str(i//4)} for i in range(8)]}
def execution(fixed,shapes):
 return {'anchor':'native','mode':'actual','layers':{},'summed_fc1_rows':558000,'summed_fc2_rows':558000,
  'all12_observed_fc_rows':{n:dict(fc1_rows=a.G.expected_layer(n,'native')[0],fc2_rows=a.G.expected_layer(n,'native')[0]) for n in a.G.GEOMETRY},
  'head_records':[{'shape':[1,2,480,640],'dtype':'torch.float32','finite':True,'sha256':'a'*64} for _ in range(4)],
  'branch_records':{n:{'shape':list(a.G.GEOMETRY[n]),'input_sha256':'b'*64,'output_sha256':'c'*64} for n in a.G.SELECTED},
  'all78_BN_observations':{n:{'calls':1,'input_shapes':[s],'current_stat':n not in fixed} for n,s in shapes.items()}}
def fixture_scope(fixed,shapes):
 chosen={}
 for n in fixed:
  c=shapes[n][2]
  chosen[n+'.running_mean']=np.zeros(c,np.float32);chosen[n+'.running_var']=np.ones(c,np.float32);chosen[n+'.num_batches_tracked']=np.array(176,np.int64)
 sig={n:{'training':False,'tracking':n in fixed,**{key:hashlib.sha256(chosen[n+'.'+key].tobytes()).hexdigest() if n in fixed else None for key in a.SUFFIX}} for n in shapes}
 scope={'scope':a.SCOPE,'BN_count':78,'private_BN_frozen':16,'private_BN_current':8,'external_BN_current':54,'retained_buffer_count':48,
  'frozen_BN_names':sorted(fixed),'current_stat_BN_names':sorted(set(shapes)-fixed),'non_BN_parameters_and_affine_unchanged':True,
  'own_CMA_source_loaded':True,'source_unselected_BN_buffers_removed_before_inference':True,'calibration_performed':False,'BN_signature_before_inference':sig}
 return scope,chosen

class Tests(unittest.TestCase):
 def test_actual_independent_source_and_own48_reconstruction(self):
  record,chosen,fixed,shapes=a.reconstruct();self.assertEqual(record['selected16_state_sha256'],a.SELECTED)
  self.assertEqual((len(chosen),len(fixed),len(shapes),record['selected16_tensor_count']),(48,16,78,525))
  self.assertEqual(record['selected16_elements'],54944214)
 def test_records_true78_shapes_and_native8_rejections(self):
  fixed,shapes=a.names_and_shapes();m=members();d={'full_forward_count':8,'forward_records':[dict(r,execution=execution(fixed,shapes)) for r in a.schedule(m)]}
  totals,lookup=a.verify_records(d,m,fixed,shapes)
  self.assertEqual((totals['FC1_rows'],totals['FC2_rows'],totals['current_BN_calls'],totals['fixed_BN_calls']),(4464000,4464000,496,128))
  for case in ('missing','extra','order','FC','branch','head','BNmissing','BNshape','BNpolicy','BNcalls'):
   bad=copy.deepcopy(d);ex=bad['forward_records'][0]['execution'];name=next(iter(shapes))
   if case=='missing':bad['forward_records'].pop()
   elif case=='extra':bad['forward_records'].append(bad['forward_records'][0])
   elif case=='order':bad['forward_records'].reverse()
   elif case=='FC':next(iter(ex['all12_observed_fc_rows'].values()))['fc1_rows']-=1
   elif case=='branch':ex['branch_records'].clear()
   elif case=='head':ex['head_records'][0]['finite']=False
   elif case=='BNmissing':ex['all78_BN_observations'].pop(name)
   elif case=='BNshape':ex['all78_BN_observations'][name]['input_shapes'][0][0]+=1
   elif case=='BNpolicy':ex['all78_BN_observations'][name]['current_stat']=not ex['all78_BN_observations'][name]['current_stat']
   else:ex['all78_BN_observations'][name]['calls']=2
   with self.assertRaises(ValueError,msg=case):a.verify_records(bad,m,fixed,shapes)
 def test_scope_precise16fixed62current_rejections(self):
  fixed,shapes=a.names_and_shapes();scope,chosen=fixture_scope(fixed,shapes);a.verify_scope(scope,chosen,fixed,shapes)
  f=sorted(fixed)[0];current=sorted(set(shapes)-fixed)[0]
  for case in ('wrongfixed','stats','currentbuffer','training','tracking','recalibration','source'):
   bad=copy.deepcopy(scope)
   if case=='wrongfixed':bad['frozen_BN_names'][0]=current
   elif case=='stats':bad['BN_signature_before_inference'][f]['running_mean']='0'*64
   elif case=='currentbuffer':bad['BN_signature_before_inference'][current]['running_var']='0'*64
   elif case=='training':bad['BN_signature_before_inference'][current]['training']=True
   elif case=='tracking':bad['BN_signature_before_inference'][current]['tracking']=True
   elif case=='recalibration':bad['calibration_performed']=True
   else:bad['own_CMA_source_loaded']=False
   with self.assertRaises(ValueError,msg=case):a.verify_scope(bad,chosen,fixed,shapes)
 def test_unique_check8_and_complete_geometry(self):
  m=members();m['check_members'][0]=m['check_members'][1]
  with self.assertRaises(ValueError):a.schedule(m)
  fixed,shapes=a.names_and_shapes()
  self.assertEqual(sum(np.prod(s) for n,s in shapes.items() if n not in fixed),1091784960)
  self.assertEqual(sum(a.G.expected_layer(n,'native')[0] for n in a.G.GEOMETRY),558000)

if __name__=='__main__':
 stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 out={'schema':'flow_selected16_independent_auditor_cpu_v1','status':'PASS' if r.wasSuccessful() else 'FAIL','tests_run':r.testsRun,
  'auditor_sha256':a.sha(a.__file__),'test_sha256':a.sha(__file__),'test_log':stream.getvalue(),
  'scope':'Actual original W plus own48 source CPU reconstruction and synthetic8/78hook/BN rejection checks; no GPU/SSH/new quality.'}
 p=Path(__file__).parent/'AUDITOR_CPU_V1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(stream.getvalue());print(json.dumps({'status':out['status'],'receipt_sha256':a.sha(p)}));raise SystemExit(0 if r.wasSuccessful() else 1)
