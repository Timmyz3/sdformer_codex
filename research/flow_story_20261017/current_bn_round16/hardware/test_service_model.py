"""Independent invariants for the declared schedule; no neural inference."""
import dataclasses, unittest
from collections import Counter
import service_model as m


class ServiceTests(unittest.TestCase):
 def setUp(self):self.small=m.Config(activation_sram_bytes=8192,reserved_statistics_and_metadata_bytes=4096)
 def test_real_tail_cover(self):
  expected={(9,9):(81,9,81),(11,14):(154,20,100),(3,4):(12,2,12)}
  for (h,w),counts in expected.items():
   for kind,count in zip(('FF','CC','checker'),counts):
    ps=m.points(h,w,m.masks(h,w,kind));self.assertEqual(len(ps),count)
    cover=Counter((y,x) for p in ps for y in range(p['y'],p['y']+p['h']) for x in range(p['x'],p['x']+p['w']))
    self.assertEqual(len(cover),h*w);self.assertEqual(set(cover.values()),{1})
  self.assertEqual(sorted(p['area'] for p in m.points(3,4,[[False]])),[3,9])
 def test_real_geometry_nominal_rows(self):
  result={}
  for arm in ('FF','CC','P'):
   total=0
   for name,(t,b,h,w,c),selected in m.geometry():
    kind='CC' if selected and arm!='FF' else 'FF'
    if arm=='P' and '.layers.2.swin_blocks.2.' in name:kind='FF'
    total+=t*b*len(m.points(h,w,m.masks(h,w,kind)))
   result[arm]=total
  self.assertEqual(result,{'FF':558000,'CC':153600,'P':164200})
 def test_spill_recompute_transaction_conservation(self):
  a,sa=m.layer('tiny',(10,1,3,4,4),[[False]],'spill',self.small,True)
  b,sb=m.layer('tiny',(10,1,3,4,4),[[False]],'recompute',self.small,True)
  self.assertEqual((a['work']['FC1_rows'],a['work']['FC2_rows']),(20,20))
  self.assertEqual((b['work']['FC1_rows'],b['work']['FC2_rows']),(100,60))
  self.assertEqual(a['traffic_bytes']['BN1_write'],20*16*4)
  self.assertEqual(a['traffic_bytes']['BN1_read'],2*20*16*4)
  self.assertFalse(any(k.startswith('BN') for k in b['traffic_bytes']))
  for s in (sa,sb):
   beats=s.validate(expand_bus=True)
   self.assertEqual(sum(x['bytes'] for x in beats),sum(s.traffic.values()))
   self.assertLessEqual(s.peak,self.small.activation_sram_bytes)
   self.assertTrue(all(x['bytes']<=64 for x in beats))
 def test_weight_live_stripe_and_1rw_norm(self):
  _,s=m.layer('tiny',(10,1,3,4,4),[[True]],'spill',self.small,True)
  self.assertTrue(all(e.get('bytes',0)<=self.small.weight_sram_bytes for e in s.events if e.get('resource')=='weight_dma'))
  self.assertGreaterEqual(m.stats_ticks(20,16,'norm',self.small),40)
  with self.assertRaisesRegex(ValueError,'stripe|weight'):
   m.layer('tiny',(10,1,3,4,4),[[True]],'spill',dataclasses.replace(self.small,weight_sram_bytes=16))
 def test_capacity_and_dependence_rejection(self):
  with self.assertRaisesRegex(ValueError,'complete-T'):
   m.layer('tiny',(10,1,3,4,4),[[True]],'spill',dataclasses.replace(self.small,activation_sram_bytes=4200))
  s=m.Schedule(self.small)
  with self.assertRaisesRegex(ValueError,'statistics'):
   s.task('premature_norm',1,requires=('BN1_ready',))
  _,s=m.layer('tiny',(10,1,3,4,4),[[False]],'spill',self.small,True)
  s.events=[e for e in s.events if not(e['kind']=='barrier' and e['name']=='BN1_mean')]
  with self.assertRaises(ValueError):s.validate()
 def test_injected_bus_conflict_rejected(self):
  s=m.Schedule(self.small);s.dma('a',128,'read',0);s.dma('b',128,'read',0)
  s.events[1]['start']=1
  with self.assertRaises(ValueError):s.validate(True)
 def test_same_mask_shared_executor(self):
  a,_=m.layer('ordinary',(10,1,11,14,4),m.masks(11,14,'checker'),'spill',self.small)
  b,_=m.layer('flow',(10,1,11,14,4),m.masks(11,14,'checker'),'spill',self.small)
  for key in ('work','traffic_bytes','modeled_blocking_schedule_ticks'):self.assertEqual(a[key],b[key])
  a,_=m.layer('dense',(10,1,3,4,4),[[True]],'spill',self.small)
  self.assertNotIn('pool_add',a['work']);self.assertNotIn('descriptor_read',a['traffic_bytes'])
 def test_external54_not_a_cycle_baseline(self):
  x=m.external_bn_inventory();self.assertEqual(len(x),54)
  self.assertEqual(sum(r['elements'] for r in x),976584960)
  self.assertFalse(any('cycles' in r or 'ticks' in r for r in x))

if __name__=='__main__':unittest.main()
