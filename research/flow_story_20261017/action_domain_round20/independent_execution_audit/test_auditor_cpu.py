import copy,unittest
import numpy as np
import audit_domain as a
M=[{'target_id':'synthetic_'+str(i)} for i in range(8)]

def records():
 _,bn=a.S.A15.names_and_shapes();out=[]
 for item in a.plan(M):
  row=dict(item);arm=item['arm'];cs=[] if arm=='native' else a.cells(arm);r=len(cs);n=12000 if arm=='native' else r*10
  ex={} if arm=='native' else {'cells':cs,'areas':[c[4] for c in cs],'representatives':r,'statistics_barriers':2,
   'FC1_retained_elements':n*1536,'FC2_retained_elements':n*384,'full_T_and_all_channels':True,'native_MLP_shortcut':False,
   'BN_mode':'NATIVE_DENSE_CURRENT','native_BN_calls':2,'full_domain_BN_elements':23040000,'expand_written_elements':23040000,
   'BN_output_dense_elements':23040000,'production_BN1_gather_read_elements':n*1536,'production_BN1_gather_written_elements':n*1536,
   'diagnostic_BN2_gather_read_elements':n*384,'cell_exact_audit_read_elements':46080000,'each_cell_BN1_and_BN2_bitwise_equal':True,
   'packed_input_shape':[10,1,r,1,384],'packed_input_sha256':'a'*64,'diagnostic_moments_full_domain_elements_per_scan':23040000,
   'diagnostic_moments_scans':2,'schedule':'Compact PSN/FC; expand each FC output over original HW; native dense BN; sample all T/B/C at each cell top-left for PSN2. No producer recomputation.'}
  fcs={k:{'fc1_rows':n if k==a.S.TARGET else int(np.prod(s[:-1])),'fc2_rows':n if k==a.S.TARGET else int(np.prod(s[:-1]))} for k,s in a.S.G.GEOMETRY.items()}
  row.update(target_execution=ex,all12_observed_fc_rows=fcs,summed_fc1_rows=sum(v['fc1_rows'] for v in fcs.values()),summed_fc2_rows=sum(v['fc2_rows'] for v in fcs.values()),
   native_BN_calls={k:1 for k in bn},native_BN_inputs={k:[{'shape':s,'dtype':'torch.float32','stride':[1]*len(s)}] for k,s in bn.items()},
   state_and_BN_unchanged=True,original_modules_restored_and_hooks_removed=True)
  out.append(row)
 return out,bn

class Checks(unittest.TestCase):
 def test_independent_plan_totals_and_prefix(self):
  rows,bn=records();out=a.check_count_records(rows,M,bn)
  self.assertEqual(out['FC1_rows'],16*558000+10*547400+8*556040+10*549360)
  self.assertEqual(out['BN_calls'],3432);self.assertEqual(out['quality_endpoints'],40);self.assertEqual(out['determinism_replays'],4)
  self.assertEqual([r['arm'] for r in rows[:9]],['native','FF','CC','CC_repeat1','CC_repeat2','F12','T8','T8_repeat1','T8_repeat2'])
  self.assertEqual(a.check_count_records(rows[:5],M,bn)['full_forwards'],5)
 def test_counter_and_current_domain_rejections(self):
  original,bn=records()
  mutations=[lambda r:r[2]['target_execution'].__setitem__('full_domain_BN_elements',1400*1920),
   lambda r:r[2]['all12_observed_fc_rows'][a.S.TARGET].__setitem__('fc1_rows',12000),
   lambda r:r[2]['native_BN_calls'].__setitem__(next(iter(bn)),0),
   lambda r:r[2].__setitem__('state_and_BN_unchanged',False),
   lambda r:r[3].__setitem__('quality_endpoint',True),
   lambda r:r[4].__setitem__('arm','CC_repeat1'),
   lambda r:r[7].__setitem__('arm','CC_repeat1'),
   lambda r:r[6]['target_execution'].__setitem__('representatives',1004),
   lambda r:r[2]['target_execution'].__setitem__('packed_input_shape',[2,1,140,1,384]),
   lambda r:r[2]['target_execution']['cells'][-1].__setitem__(4,9)]
  for change in mutations:
   rows=copy.deepcopy(original);change(rows)
   with self.assertRaises(ValueError):a.check_count_records(rows,M,bn)
 def test_true_tails_and_spike_expansion(self):
  for arm,n in [('FF',1200),('CC',140),('F12',1004),('T8',336),('T8_repeat1',336)]:
   cs=a.cells(arm);self.assertEqual(len(cs),n);self.assertEqual(sum(c[4] for c in cs),1200)
   packed=np.arange(10*n,dtype=np.float32).reshape(10,1,n,1,1);expanded=a.expand_spikes(packed,cs)
   for i,(y,yy,x,xx,_) in enumerate(cs):self.assertTrue(np.array_equal(expanded[:,:,y:yy,x:xx,:],np.broadcast_to(packed[:,:,i:i+1,:,:],(10,1,yy-y,xx-x,1))))
 def test_bitwise_replay_signed_zero_and_any_head(self):
  keys=('head0','head1','head2','head3','u','branch','sn1','sn2','sn1_packed','sn2_packed')
  original={k:np.zeros(2,dtype=np.float32) for k in keys};self.assertTrue(a.replay(original,copy.deepcopy(original))['pass'])
  for key in keys:
   changed=copy.deepcopy(original);changed[key][0]=-0.0;self.assertFalse(a.replay(changed,original)['pass'])
 def test_old_ff_gate_kept(self):
  z={k:np.zeros(2,dtype=np.float32) for k in ('head0','head1','head2','head3','u','branch','sn1','sn2')}
  self.assertTrue(a.S.numerical(z,copy.deepcopy(z))['pass'])
  for k in ('branch','head0','head1','head2','head3','sn1','sn2'):
   bad=copy.deepcopy(z);bad[k][0]=1;self.assertFalse(a.S.numerical(bad,z)['pass'])

if __name__=='__main__':unittest.main()
