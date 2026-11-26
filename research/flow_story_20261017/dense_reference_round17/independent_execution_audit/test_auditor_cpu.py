import copy,unittest
import audit_dense as a

class DenseAuditTests(unittest.TestCase):
 def fixture(self):
  _,members,_=a.S.h8_inputs();_,bn=a.S.A15.names_and_shapes();records=[]
  for r in a.plan(members):
   records.append(dict(r,all12_observed_fc_rows={n:{'fc1_rows':a.math.prod(s[:-1]),'fc2_rows':a.math.prod(s[:-1])} for n,s in a.S.G.GEOMETRY.items()},summed_fc1_rows=558000,summed_fc2_rows=558000,
    native_BN_calls={n:1 for n in bn},native_BN_inputs={n:[{'shape':s,'dtype':'torch.float32','stride':[1]*len(s)}] for n,s in bn.items()},
    state_and_BN_unchanged=True,original_modules_restored_and_hooks_removed=True,
    target_execution={} if r['execution']=='native' else {'reference_dense_only':True,'expanded_input_sha256':'0'*64}))
  return members,bn,records
 def test_exact_H8_dense_budget(self):
  members,bn,records=self.fixture();totals=a.check_count_records(records,members,bn)
  self.assertEqual(members[0]['target_id'],'zurich_city_07_a_0095')
  self.assertEqual((totals['native'],totals['expanded'],totals['compact']),(8,112,0))
  self.assertEqual(totals['FC1_rows'],66960000);self.assertEqual(totals['BN_calls'],9360)
  self.assertEqual(sum(r['arm'] in ('native','FF') for r in records)*4,64)
 def test_missing_or_changed_order_rejected(self):
  members,bn,records=self.fixture()
  with self.assertRaises(ValueError):a.check_count_records(records[:-1],members,bn)
  records[2],records[3]=records[3],records[2]
  with self.assertRaises(ValueError):a.check_count_records(records,members,bn)
 def test_actual_count_policy_and_closure_rejections(self):
  members,bn,base=self.fixture()
  def fc(r):r[2]['all12_observed_fc_rows'][a.S.TARGET]['fc1_rows']=1400
  def bncount(r):r[0]['native_BN_calls'][next(iter(bn))]=0
  def compact(r):r[2]['execution']='compact'
  def closure(r):r[5]['state_and_BN_unchanged']=False
  def wrapper(r):r[2]['target_execution']['representatives']=140
  for mutate in (fc,bncount,compact,closure,wrapper):
   records=copy.deepcopy(base);mutate(records)
   with self.assertRaises(ValueError):a.check_count_records(records,members,bn)
 def test_sources_bound_before_results(self):
  pins=a.source_pins();self.assertEqual(pins[str(a.PRIOR)],a.PRIOR_SHA)

if __name__=='__main__':unittest.main()
