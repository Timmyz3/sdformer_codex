import copy,unittest
import numpy as np
import audit_two_slot as a
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
   'diagnostic_moments_scans':2,'partition_rule':a.PARTITION_RULE,'coarse_representatives_per_complete3x3':2,
   'base_primitive_sha256':a.BASE_PRIMITIVE_SHA,'shared_math_sha256':a.MATH_SHA,'private_globals_overrides':['pack'],'original_globals_mutated':False,'schedule':'Compact PSN/FC; expand each FC output over original HW; native dense BN; sample all T/B/C at each cell top-left for PSN2. No producer recomputation.'}
  fcs={k:{'fc1_rows':n if k==a.S.TARGET else int(np.prod(s[:-1])),'fc2_rows':n if k==a.S.TARGET else int(np.prod(s[:-1]))} for k,s in a.S.G.GEOMETRY.items()}
  row.update(target_execution=ex,all12_observed_fc_rows=fcs,summed_fc1_rows=sum(v['fc1_rows'] for v in fcs.values()),summed_fc2_rows=sum(v['fc2_rows'] for v in fcs.values()),
   native_BN_calls={k:1 for k in bn},native_BN_inputs={k:[{'shape':s,'dtype':'torch.float32','stride':[1]*len(s)}] for k,s in bn.items()},
   state_and_BN_unchanged=True,original_modules_restored_and_hooks_removed=True)
  out.append(row)
 return out,bn

class Checks(unittest.TestCase):
 def test_independent_plan_totals_and_prefix(self):
  rows,bn=records();out=a.check_count_records(rows,M,bn)
  self.assertEqual(out['FC1_rows'],16*558000+10*550440+96*551070)
  self.assertEqual(out['BN_calls'],9516);self.assertEqual(out['quality_endpoints'],120);self.assertEqual(out['determinism_replays'],2)
  self.assertEqual([r['arm'] for r in rows[:6]],['native','FF','T2','T2_repeat1','T2_repeat2','fine_0_0'])
  self.assertEqual(a.check_count_records(rows[:5],M,bn)['full_forwards'],5)
 def test_counter_and_current_domain_rejections(self):
  original,bn=records()
  mutations=[lambda r:r[2]['target_execution'].__setitem__('full_domain_BN_elements',4440*1920),
   lambda r:r[2]['all12_observed_fc_rows'][a.S.TARGET].__setitem__('fc1_rows',12000),
   lambda r:r[2]['native_BN_calls'].__setitem__(next(iter(bn)),0),
   lambda r:r[2].__setitem__('state_and_BN_unchanged',False),
   lambda r:r[3].__setitem__('quality_endpoint',True),
   lambda r:r[4].__setitem__('arm','T2_repeat1'),
   lambda r:r[5]['target_execution'].__setitem__('representatives',212),
   lambda r:r[2]['target_execution'].__setitem__('packed_input_shape',[2,1,444,1,384]),
   lambda r:r[2]['target_execution']['cells'][-1].__setitem__(4,9),
   lambda r:r[2]['target_execution'].__setitem__('coarse_representatives_per_complete3x3',1),
   lambda r:r[2]['target_execution'].__setitem__('original_globals_mutated',True),
   lambda r:r[2]['target_execution'].__setitem__('base_primitive_sha256','0'*64),
   lambda r:r[2]['target_execution'].__setitem__('private_globals_overrides',['pack','native_current_bn']),
   lambda r:r[2]['target_execution'].__setitem__('partition_rule','H2')]
  for change in mutations:
   rows=copy.deepcopy(original);change(rows)
   with self.assertRaises(ValueError):a.check_count_records(rows,M,bn)
 def test_true_tails_and_spike_expansion(self):
  for arm,n in [('FF',1200),('T2',444),('T2_repeat1',444)]+[(f'fine_{y}_{x}',507) for y in range(3) for x in range(4)]:
   cs=a.cells(arm);self.assertEqual(len(cs),n);self.assertEqual(sum(c[4] for c in cs),1200)
   coverage=np.zeros((30,40),dtype=int)
   packed=np.arange(10*n,dtype=np.float32).reshape(10,1,n,1,1);expanded=a.expand_spikes(packed,cs)
   for i,(y,yy,x,xx,area) in enumerate(cs):
    coverage[y:yy,x:xx]+=1
    if y>=27 or x>=36:self.assertEqual(area,1)
    if area>1:
     self.assertEqual(xx-x,3);self.assertIn((yy-y,area),((1,3),(2,6)))
     self.assertEqual(y%3,0 if area==3 else 1)
    self.assertTrue(np.array_equal(expanded[:,:,y:yy,x:xx,:],np.broadcast_to(packed[:,:,i:i+1,:,:],(10,1,yy-y,xx-x,1))))
   self.assertTrue((coverage==1).all())
 def test_independent_per_pixel_H1_equivalence(self):
  # Independent point-to-group map, rather than reusing rectangle construction.
  for arm in a.ARMS[1:]:
   target=tuple(map(int,arm.split('_')[1:])) if arm.startswith('fine_') else None
   groups={}
   for y in range(30):
    for x in range(40):
     fine=arm=='FF' or y>=27 or x>=36 or target==(y//9,x//9)
     key=(y,x) if fine else (y-y%3+(1 if y%3 else 0),x-x%3)
     groups.setdefault(key,set()).add((y,x))
   actual={frozenset((y,x) for y in range(c[0],c[1]) for x in range(c[2],c[3])) for c in a.cells(arm)}
   self.assertEqual(actual,{frozenset(points) for points in groups.values()})
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
