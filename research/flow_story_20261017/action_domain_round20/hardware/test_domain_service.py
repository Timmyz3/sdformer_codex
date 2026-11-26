import copy,random,unittest
import domain_service as d
class Checks(unittest.TestCase):
 def test_fixed_geometry(self):
  for arm,reps,full,tail in [('FF',1200,12,8),('CC',140,0,0),('F12',1004,12,0),('T8',336,0,8)]:
   r,s=d.layer(arm);self.assertEqual(r['producer_representatives'],reps);self.assertEqual(r['FC_rows_each'],10*reps)
   self.assertEqual(r['complete_tile_fine_count'],full);self.assertEqual(r['true_tail_tile_fine_count'],tail)
   self.assertEqual(r['full_BN_domain_elements'],23040000);self.assertEqual(r['whole12_FC_rows_each'],546000+10*reps)
   d.P.validate_schedule(s)
 def test_arbitrary_masks_independent_pixel_count(self):
  rng=random.Random(2020)
  for _ in range(20):
   mask=[[bool(rng.getrandbits(1)) for x in range(5)] for y in range(4)]
   cells=d.cover_mask(mask);cover=[[0]*40 for _ in range(30)];n=0
   for gy,row in enumerate(mask):
    for gx,fine in enumerate(row):
     h,w=min(9,30-gy*9),min(9,40-gx*9);n+=h*w if fine else ((h+2)//3)*((w+2)//3)
   for c in cells:
    y,x=c['y'],c['x'];step=1 if c['fine'] else 3
    ey,ex=min(y+step,((y//9)+1)*9,30),min(x+step,((x//9)+1)*9,40)
    for yy in range(y,ey):
     for xx in range(x,ex):cover[yy][xx]+=1
   self.assertEqual(len(cells),n);self.assertTrue(all(v==1 for row in cover for v in row))
 def test_no_mask_or_source_mutation(self):
  mask=d.mask_for('T8');saved=copy.deepcopy(mask);before=d.P.cover
  r,_=d.layer('custom',mask=mask)
  self.assertEqual(mask,saved);self.assertIs(d.P.cover,before);self.assertEqual(d.sha(d.PRIOR),d.PRIOR_SHA)
  r['mask'][0][0]=True;self.assertEqual(mask,saved)
 def test_invalid_masks_rejected(self):
  for bad in [[],[[False]*5]*3,[[False]*4]*4,[[0]*5]*4]:
   with self.assertRaises(ValueError):d.cover_mask(bad)
  with self.assertRaises(ValueError):d.layer('T8',native=True)
 def test_no_score_and_shared_baseline(self):
  report=d.report();self.assertEqual(report['graphs']['native']['modeled_blocking_total_service_ticks'],268546374)
  for r in report['graphs'].values():self.assertEqual(r['score_or_predictor_ticks'],0);self.assertEqual(r['selector_ticks'],0)
  for arm,c in report['versus_native_no_score'].items():
   self.assertEqual(c['net_local_service_ticks_saved'],268546374-c['candidate_no_score_service_ticks'])
   self.assertIsNone(c['whole_network_speedup']);self.assertIsNone(c['precision_or_quality_admission'])
if __name__=='__main__':unittest.main()
