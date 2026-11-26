"""Synthetic fixed193/raw audit tests. No new actual run/quality read."""
import copy,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit_deployment',HERE/'audit_deployment.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class Tests(unittest.TestCase):
 def test_schedule_geometry(self):
  m=a.read(a.R13/'data_protocol/manifest_v1.json');s=a.schedule(m)
  self.assertEqual(len(s),193);self.assertEqual(s[178]['mode'],'dual_reference')
  self.assertEqual(sum(sum(a.G.expected_layer(n,r['layout'],r['mode']=='dual_reference')[0] for n in a.G.GEOMETRY) for r in s),107745600)
 def records(self):
  m=a.read(a.R13/'data_protocol/manifest_v1.json');old=a.read(a.R13/'raw/experiments_v1/diversity/recover/native/results.json')
  oldff=a.read(a.R13/'raw/experiments_v1/diversity/recover/FF/results.json')
  native=old['evaluations']['initial']['frames'][0]['arms']['native']['execution'];fine=oldff['initial_numerical_pair']['old_actual_execution'];dual=oldff['initial_numerical_pair']['old_dual_execution']
  rows=[];names={'bn_'+str(i) for i in range(78)}
  for want in a.schedule(m):
   if want['phase']=='calibration':
    ex={'all12_observed_fc_rows':{n:{'fc1_rows':a.G.expected_layer(n,'native')[0],'fc2_rows':a.G.expected_layer(n,'native')[0]} for n in a.G.GEOMETRY},'summed_fc1_rows':558000,'summed_fc2_rows':558000,'gradient_enabled':False,'BN_mode':'all78_reset_CMA_train_nonBN_eval','all78_num_batches_tracked':{n:want['forward_id'] for n in names}}
   else:
    ex=copy.deepcopy(dual if want['mode']=='dual_reference' else native if want['layout']=='native' else fine)
    ex.update(anchor=want['layout'],mode=want['mode'],branch_records={n:{'input_sha256':'0'*64,'output_sha256':'1'*64,'shape':list(a.G.GEOMETRY[n])} for n in a.G.SELECTED})
   ex['head_records']=[{'shape':[1,2,480,640],'dtype':'torch.float32','finite':True} for i in range(4)]
   rows.append(dict(want,execution=ex))
  return {'forward_records':rows,'full_forward_count':193},m,names
 def test_recompute_all_records(self):
  d,m,names=self.records();counts,_=a.verify_records(d,m,names)
  self.assertEqual(counts['FC1_rows'],107745600);self.assertEqual(counts['observed_FC_modules'],4632)
  self.assertEqual(counts['wrapped_layers'],72);self.assertEqual(counts['native_calls_in_wrappers'],6240)
 def test_reject_broken_actual_records(self):
  d,m,names=self.records()
  for case in ('missing','counter','row','head','branch','order'):
   bad=copy.deepcopy(d)
   if case=='missing':bad['forward_records'].pop()
   elif case=='counter':bad['forward_records'][30]['execution']['all78_num_batches_tracked'][next(iter(names))]=0
   elif case=='row':bad['forward_records'][40]['execution']['all12_observed_fc_rows'][next(iter(a.G.GEOMETRY))]['fc2_rows']-=1
   elif case=='head':bad['forward_records'][180]['execution']['head_records'].pop()
   elif case=='branch':bad['forward_records'][180]['execution']['branch_records'].pop(next(iter(a.G.SELECTED)))
   else:bad['forward_records'][178]['mode']='actual'
   with self.assertRaises(ValueError,msg=case):a.verify_records(bad,m,names)
 def test_raw20_artifact_and_exact_equality(self):
  original=a.G.GEOMETRY;selected=a.G.SELECTED
  tiny={n:(10,1,1,1,1) for n in sorted(selected)};a.G.GEOMETRY=tiny
  try:
   with tempfile.TemporaryDirectory() as temp:
    root=Path(temp);arrays={f'flow_{i}':np.full((1,2,480,640),i,dtype=np.float32) for i in range(4)}
    arrays.update({f'{kind}_{i}':np.full((10,1,1,1,1),i,dtype=np.float32) for i in range(8) for kind in ('input','branch')})
    records={k:{'shape':list(v.shape),'dtype':'float32','payload_sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in arrays.items()};names=sorted(selected)
    ex={'head_records':[{'sha256':records[f'flow_{i}']['payload_sha256']} for i in range(4)],'branch_records':{n:{'input_sha256':records[f'input_{i}']['payload_sha256'],'output_sha256':records[f'branch_{i}']['payload_sha256']} for i,n in enumerate(names)},'layers':{n:{'input_sha256':records[f'input_{i}']['payload_sha256']} for i,n in enumerate(names)}}
    pair={'actual_execution':ex,'dual_execution':ex,'all4_flow_heads_exact':True,'all8_inputs_exact':True,'all8_branches_exact':True}
    for side in ('actual','dual'):
     p=root/(side+'.npz');np.savez_compressed(p,**arrays);pair[side+'_raw']={'file':p.name,'file_sha256':a.sha(p),'arrays':records,'branch_names_in_index_order':names}
    d={'first_frame_FF_dual':pair};m={'check_members':[{'target_id':'frame'}]};lookup={('evaluation','frame','FF',mode):ex for mode in ('actual','dual_reference')}
    arts,last=a.verify_raw_pair(root,d,m,lookup);self.assertEqual(len(arts),2);self.assertEqual(last,records['flow_3']['payload_sha256'])
    damaged=copy.deepcopy(d);damaged['first_frame_FF_dual']['dual_raw']['file_sha256']='0'*64
    with self.assertRaises(ValueError):a.verify_raw_pair(root,damaged,m,lookup)
    damaged=copy.deepcopy(d);damaged['first_frame_FF_dual']['actual_raw']['arrays'].pop('branch_7')
    with self.assertRaises(ValueError):a.verify_raw_pair(root,damaged,m,lookup)
    arrays['flow_3'][0,0,0,0]+=1;np.savez_compressed(root/'dual.npz',**arrays)
    damaged=copy.deepcopy(d);damaged['first_frame_FF_dual']['dual_raw']['file_sha256']=a.sha(root/'dual.npz')
    with self.assertRaises(ValueError):a.verify_raw_pair(root,damaged,m,lookup)
  finally:a.G.GEOMETRY=original
 def test_source_freezes_and_no_quality_access(self):
  import ast
  pins=a.source_pins();self.assertTrue(pins)
  tree=ast.parse((HERE/'audit_deployment.py').read_text())
  self.assertFalse(any(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value in ('metrics','summaries','AEE','aee') for n in ast.walk(tree)))
if __name__=='__main__':unittest.main()
