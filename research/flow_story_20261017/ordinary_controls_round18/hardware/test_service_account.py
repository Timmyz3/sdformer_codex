import unittest
from dataclasses import replace
import service_account as s

class Checks(unittest.TestCase):
 def test_independent_operations(self):
  p=s.HERE.parent/'implementation/g0_core.py'
  self.assertEqual(s.sha(p),'01d54124dfecec70cffa9efaa2a8e88579ecb14477ae891d46643b140f15216d')
  g=s.load(p,'r18_g0_readonly_counter')
  for t,c in ((2,1),(10,384),(10,7)):
   self.assertEqual(s.independent_tile_work(t,c),g.operation_ledger((t,1,9,9,c))['by_score'])
 def test_all_actual_tail_masks(self):
  for y in range(3):
   for x in range(4):
    p=s.cover((y,x));self.assertEqual(sum(a for a,_ in p),1200);self.assertEqual(len(p),212)
    self.assertEqual(sum(f for _,f in p),81);self.assertEqual(sum(a for a,f in p if not f),1119)
 def test_finite_storage_reject(self):
  m=s.load(s.OLD,'r18_resources_check');cfg=m.Config()
  for n,ops in s.independent_tile_work().items():
   row=s.score_schedule(n,ops,cfg);self.assertLessEqual(row['score_local_peak_bytes'],2*1024*1024)
   with self.assertRaises(ValueError):s.score_schedule(n,ops,replace(cfg,activation_sram_bytes=1024*1024))
 def test_dense_and_compact_not_mixed(self):
  r=s.report();self.assertEqual(r['dense_target']['work']['FC1_rows'],12000)
  self.assertEqual(r['dense_target']['work']['FC2_rows'],12000)
  self.assertEqual(r['hypothetical_compact_separate']['target_FC_rows_each'],2120)
  self.assertEqual(r['hypothetical_compact_separate']['native_dense_BN_elements'],23040000)
  for n,v in r['per_deployed_score'].items():
   self.assertEqual(v['actual_dense_cancelled_FC_rows'],0);self.assertIsNone(v['actual_dense_speedup_claim'])
   self.assertEqual(v['extra_native_score_PSN1'] is not None,n.startswith('psn1'))
 def test_descriptor_materialization_not_free(self):
  m=s.load(s.OLD,'r18_resource_check2');cfg=m.Config();ops=s.independent_tile_work()
  p=s.score_schedule('pooling_residual_mse',ops['pooling_residual_mse'],cfg)
  self.assertEqual(p['pool_repeat_read_write_ticks'],2*12*10*81*384)

if __name__=='__main__':unittest.main()
