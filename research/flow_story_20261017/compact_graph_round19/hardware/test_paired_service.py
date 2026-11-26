import copy,unittest
from dataclasses import replace
import paired_service as p

class Checks(unittest.TestCase):
 def test_true_tail_and_full_domain(self):
  for arm,r in [('FF',1200),('CC',140),('single',212)]:
   cells=p.cover(arm);self.assertEqual(len(cells),r);self.assertEqual(sum(x['area'] for x in cells),1200)
   self.assertTrue(all(1<=x['area']<=9 for x in cells))
   for graph,rows in [('dense_expanded',12000),('compact',r*10)]:
    a,s=p.layer(graph,arm);self.assertEqual(a['work']['FC1_rows'],rows);self.assertEqual(a['work']['FC2_rows'],rows)
    self.assertEqual(a['full_BN_domain_elements'],23040000)
    self.assertEqual(a['external_transactions_bytes']['BN1_Z_read'],3*18432000*4)
    self.assertEqual(a['external_transactions_bytes']['BN2_Z_read'],3*4608000*4)
    self.assertEqual(a['work'].get('BN_area_multiply',0),0)
 def test_finite_capacity_and_weights(self):
  with self.assertRaises(ValueError):p.layer('compact','CC',replace(p.M.Config(),activation_sram_bytes=1024*1024))
  _,s=p.layer('compact','CC')
  self.assertLessEqual(s.peak,2*1024*1024)
  for event in s.events:
   if event.get('resource')=='weight_dma':self.assertLessEqual(event['bytes'],512*1024)
 def test_dependency_and_port_rejections(self):
  _,s=p.layer('compact','CC')
  bad=copy.deepcopy(s);bad.events=[e for e in bad.events if not(e['kind']=='barrier' and e['name']=='BN1_mean')]
  with self.assertRaises(ValueError):bad.validate()
  bad=copy.deepcopy(s);dma=next(e for e in bad.events if e.get('bytes'));dma['end']+=1
  with self.assertRaises(ValueError):bad.validate()
  bad=copy.deepcopy(s);task=next(e for e in bad.events if e['kind']=='task');task['scratch_bytes']=2*1024*1024
  with self.assertRaises(ValueError):bad.validate()
 def test_gather_and_expand_conservation(self):
  for arm,r in [('CC',140),('single',212)]:
   a,_=p.layer('compact',arm);b=a['external_transactions_bytes']
   self.assertEqual(b['FC1_representative_store'],10*r*1536*4)
   self.assertEqual(b['FC1_representative_read'],b['FC1_representative_store'])
   self.assertEqual(b['BN1_gather_read'],10*r*1536*4)
   self.assertEqual(b['BN1_gather_store'],b['BN1_gather_read'])
   self.assertEqual(b['BN1_compact_gather_read'],b['BN1_gather_read'])
   self.assertEqual(a['work']['FC_expand_elements'],23040000)
 def test_local_1rw_port_rejection(self):
  _,s=p.layer('compact','FF')
  pool=next(e for e in s.events if e['name']=='pool/mean')
  self.assertEqual(pool['local_activation_read_bytes'],10*384*4)
  self.assertEqual(pool['local_activation_write_bytes'],10*384*4)
  self.assertEqual(pool['end']-pool['start'],240+8+480)
  p.validate_schedule(s)
  bad=copy.deepcopy(s);pool=next(e for e in bad.events if e['name']=='pool/mean')
  # Preserve timeline but increase claimed traffic above the available slots.
  pool['local_activation_write_bytes']=(pool['end']-pool['start']+1)*64
  with self.assertRaises(ValueError):p.validate_schedule(bad)
 def test_same_score_no_quality_transfer(self):
  r=p.report()
  for pair in r['paired_same_mask'].values():
   self.assertEqual(pair['same_full_BN_domain_elements'],23040000)
   for row in pair['same_score_cases'].values():
    self.assertEqual(row['dense_target_plus_same_score_ticks_assumed']-row['compact_target_plus_same_score_ticks_assumed'],pair['service_tick_difference_dense_minus_compact'])
    self.assertFalse(row['quality_transfer_between_graphs']);self.assertIsNone(row['whole_network_speedup'])
 def test_native_no_score_deployment_comparison(self):
  r=p.report();self.assertEqual(len(r['deployment_vs_native_no_score']),7)
  for name,d in r['deployment_vs_native_no_score'].items():
   self.assertEqual(d['native_no_score_target_service_ticks'],268546374)
   self.assertEqual(d['candidate_target_with_score_service_ticks'],d['compact_target_without_score_ticks']+d['score_plus_selector_ticks'])
   self.assertEqual(d['net_local_service_ticks_saved_vs_native'],d['native_no_score_target_service_ticks']-d['candidate_target_with_score_service_ticks'])
   split=d['score_resource_split_assumed'];self.assertEqual(sum(v for k,v in split.items() if k!='extra_native_PSN_PE_FMA_issue_ticks'),d['score_plus_selector_ticks'])
   self.assertEqual(split['extra_native_PSN_PE_reserved_ticks'],2592000 if name.startswith('psn1') else 0)
   self.assertEqual(d['candidate_PE_reserved_ticks_including_score_PSN'],41619840+(2592000 if name.startswith('psn1') else 0))
   self.assertIsNone(d['whole_network_speedup']);self.assertIsNone(d['precision_or_quality_admission'])
  pool=r['deployment_vs_native_no_score']['pooling_residual_mse']
  self.assertGreater(pool['score_plus_selector_ticks'],pool['compact_target_without_score_ticks'])
  self.assertEqual(pool['candidate_target_with_score_service_ticks'],189403010)

if __name__=='__main__':unittest.main()
