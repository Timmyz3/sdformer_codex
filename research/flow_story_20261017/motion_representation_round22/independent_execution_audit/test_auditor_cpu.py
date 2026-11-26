import copy,unittest
import numpy as np
import audit_representation as a
ROWS=[{'target_id':str(i)} for i in range(24)]
ROLES={str(i):'fit' if i<16 else 'development' for i in range(24)}
def fixture():
 _,bn=a.S.A15.names_and_shapes();out=[]
 for item in a.plan(ROWS,ROLES):
  r=dict(item);fcs={k:{'fc1_rows':int(np.prod(v[:-1])),'fc2_rows':int(np.prod(v[:-1]))} for k,v in a.S.G.GEOMETRY.items()}
  r.update(target_execution={},all12_observed_fc_rows=fcs,summed_fc1_rows=558000,summed_fc2_rows=558000,
   native_BN_calls={k:1 for k in bn},native_BN_inputs={k:[{'shape':s,'dtype':'torch.float32','stride':[1]*len(s)}] for k,s in bn.items()},
   state_and_BN_unchanged=True,original_modules_restored_and_hooks_removed=True)
  out.append(r)
 return out,bn
class Checks(unittest.TestCase):
 def test_fixed_native26_counts(self):
  rows,bn=fixture();c=a.check_records(rows,ROWS,ROLES,bn)
  self.assertEqual(c['FC1_rows'],14508000);self.assertEqual(c['FC2_rows'],14508000)
  self.assertEqual(c['BN_calls'],2028);self.assertEqual(c['BN_input_elements'],26*1414344960)
  self.assertEqual(c['native'],26);self.assertEqual(c['quality_endpoints'],24);self.assertEqual(c['determinism_replays'],2)
  self.assertEqual([x['arm'] for x in rows[:4]],['native','native_repeat1','native_repeat2','native'])
  self.assertEqual(a.check_records(rows[:3],ROWS,ROLES,bn)['full_forwards'],3)
 def test_no_compact_or_role_mutation(self):
  original,bn=fixture()
  mutations=[lambda r:r[0].__setitem__('execution','compact'),lambda r:r[0].__setitem__('target_execution',{'representatives':408}),
   lambda r:r[0]['all12_observed_fc_rows'][a.S.TARGET].__setitem__('fc1_rows',4080),
   lambda r:r[2].__setitem__('arm','native_repeat1'),lambda r:r[1].__setitem__('quality_endpoint',True),
   lambda r:r[0].__setitem__('selector_role','development'),lambda r:r[0].__setitem__('state_and_BN_unchanged',False),
   lambda r:r[0]['native_BN_calls'].__setitem__(next(iter(bn)),0)]
  for change in mutations:
   r=copy.deepcopy(original);change(r)
   with self.assertRaises(ValueError):a.check_records(r,ROWS,ROLES,bn)
  with self.assertRaises(ValueError):a.plan(ROWS[:-1]+[ROWS[0]],ROLES)
  with self.assertRaises(ValueError):a.plan(ROWS,{**ROLES,'0':'sealed_confirmation'})
 def test_eight_array_exact_replay(self):
  x={k:np.zeros(2,np.float32) for k in ('head0','head1','head2','head3','u','branch','sn1','sn2')}
  self.assertTrue(a.replay(x,copy.deepcopy(x))['pass'])
  for k in x:
   y=copy.deepcopy(x);y[k][0]=-0.;self.assertFalse(a.replay(y,x)['pass'])
  y=copy.deepcopy(x);del y['sn2']
  with self.assertRaises(ValueError):a.replay(y,x)
 def test_frozen_roles_metadata_only(self):
  rows,roles,evidence,files=a.roles_and_sources(a.RUN/'not_opened_metadata_only',verify_files=False)
  self.assertEqual(len(files),72);self.assertEqual(len(evidence),24)
  self.assertEqual([roles[r['target_id']] for r in rows],['fit']*16+['development']*8)
  self.assertEqual(sum(x['bytes'] for x in files),361276416)
  self.assertEqual(len({x['path'] for x in files}),72)
if __name__=='__main__':unittest.main()
