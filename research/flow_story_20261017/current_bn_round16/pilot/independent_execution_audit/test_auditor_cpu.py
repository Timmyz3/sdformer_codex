"""Small independent refusal/geometry tests, no neural model or new raw read."""
import copy,unittest
import numpy as np
import audit_pilot as a

class AuditorTests(unittest.TestCase):
 def test_fixed_plan_totals_and_prefix(self):
  plan=a.plan([{'target_id':str(i)} for i in range(8)])
  self.assertEqual(len(plan),123);self.assertEqual(sum(r['execution']=='expanded' for r in plan),3)
  self.assertEqual(sum(r['execution']=='compact' for r in plan),112)
  self.assertEqual(sum(546000+a.expected_target(r) for r in plan),67600720)
  self.assertEqual(sum(546000+a.expected_target(r) for r in plan[:5]),2779400)
  self.assertEqual([(r['arm'],r['execution']) for r in plan[:5]],[('native','native'),('FF','compact'),('FF','expanded'),('CC','compact'),('CC','expanded')])
 def test_cells_full_T_tail_no_padding(self):
  for arm,n in (('FF',1200),('CC',140),('fine_0_0',212),('fine_2_3',212)):
   cells=a.cells(arm);self.assertEqual(len(cells),n)
   cover=np.zeros((30,40),np.int64)
   for y,yy,x,xx,area in cells:
    self.assertEqual(area,(yy-y)*(xx-x));cover[y:yy,x:xx]+=1
   np.testing.assert_array_equal(cover,np.ones((30,40),np.int64))
 def test_numeric_original_gate_and_report_reject(self):
  x={k:np.zeros((2,),np.float32) for k in ('u','branch','sn1','sn2','head0','head1','head2','head3')};y=copy.deepcopy(x)
  good=a.numerical(x,y);self.assertTrue(good['pass']);a.check_reported(good,good)
  y['head3'][0]=np.float32(2.1e-5);bad=a.numerical(x,y);self.assertFalse(bad['pass']);self.assertEqual(bad['continuous']['head3']['outside_tolerance'],1)
  with self.assertRaises(ValueError):a.check_reported(good,bad)
  y=copy.deepcopy(x);y['sn2'][0]=1;self.assertFalse(a.numerical(x,y)['pass'])
  y=copy.deepcopy(x);y['u'][0]=1e-9;self.assertFalse(a.numerical(x,y)['pass'])
 def test_nonfinite_and_payload_digest(self):
  with self.assertRaises(ValueError):a.continuous(np.array([np.nan],np.float32),np.zeros(1,np.float32))
  x=np.arange(12,dtype=np.float32).reshape(3,4)
  self.assertEqual(a.digest(x),a.digest(x.copy()))
  self.assertNotEqual(a.digest(x),a.digest(x+1))

if __name__=='__main__':unittest.main()
