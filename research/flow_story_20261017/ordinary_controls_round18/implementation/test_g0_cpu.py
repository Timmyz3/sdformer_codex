#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
import g0_core as core
import run_g0 as run


class Checks(unittest.TestCase):
    def test_scalar_reference_of_seven_formulas(self):
        x=(np.arange(10*9*9*2,dtype=np.float32).reshape(10,1,9,9,2)%31-15)/8
        s=(x>0).astype(np.float32)
        got=core.tile_scores(x,s)
        values=list(map(float,x.ravel()));n=len(values)
        expected={core.SCORES[0]:math.fsum(abs(v) for v in values)/n,
                  core.SCORES[1]:math.sqrt(math.fsum(v*v for v in values)/n)}
        residual=[];vertical=[];horizontal=[];temporal=[];vh=[];hh=[]
        for t in range(10):
            for i in range(9):
                for j in range(9):
                    for c in range(2):
                        v=float(x[t,0,i,j,c])
                        pooled=math.fsum(float(x[t,0,y,z,c]) for y in range(i//3*3,i//3*3+3)
                                         for z in range(j//3*3,j//3*3+3))/9
                        residual.append((v-pooled)**2)
                        if i<8:
                            vertical.append(abs(v-float(x[t,0,i+1,j,c])));vh.append(s[t,0,i,j,c]!=s[t,0,i+1,j,c])
                        if j<8:
                            horizontal.append(abs(v-float(x[t,0,i,j+1,c])));hh.append(s[t,0,i,j,c]!=s[t,0,i,j+1,c])
                        if t<9:temporal.append(abs(v-float(x[t+1,0,i,j,c])))
        expected.update({core.SCORES[2]:math.fsum(residual)/n,
                         core.SCORES[3]:(math.fsum(vertical)/len(vertical)+math.fsum(horizontal)/len(horizontal))/2,
                         core.SCORES[4]:math.fsum(temporal)/len(temporal),
                         core.SCORES[5]:sum(map(int,s.ravel()))/n,
                         core.SCORES[6]:(sum(vh)/len(vh)+sum(hh)/len(hh))/2})
        for name in core.SCORES:self.assertAlmostEqual(got[name],expected[name],places=12)

    def test_exact_ties_policy_and_no_epsilon(self):
        scores=[dict(tile=list(tile),**{name:1. for name in core.SCORES}) for tile in core.TILES]
        p=core.select(scores,core.SCORES[0]);self.assertEqual(p['selected_tile'],[0,0])
        scores[-1][core.SCORES[0]]=float(np.nextafter(1.,2.))
        p=core.select(scores,core.SCORES[0]);self.assertEqual(p['selected_tile'],[2,3])
        self.assertEqual(sum(map(sum,p['mask'])),1);self.assertFalse(any(p['mask'][3]))
        self.assertFalse(any(row[4] for row in p['mask']))
        uniform=core.policy(uniform=True)
        self.assertIsNone(uniform['mask']);self.assertEqual(len(set(uniform['weights'])),1)
        self.assertEqual(uniform['rational_weight'],dict(numerator=1,denominator=12))

    def test_full_T_and_real_tail_exclusion(self):
        u=np.zeros((10,1,30,40,2),np.float32);s=np.zeros_like(u)
        u[-1,:,:9,:9]=2
        a=core.score_frame(u,s)
        u[:,:,27:]=1000;u[:,:,:,36:]=1000;s[:,:,27:]=1;s[:,:,:,36:]=1
        b=core.score_frame(u,s)
        self.assertEqual(a,b);self.assertGreater(a['scores'][0]['temporal_neighbor_abs_difference'],0)
        self.assertEqual(len(a['policies']),20)
        ledger=core.operation_ledger((10,1,9,9,384))
        self.assertEqual((ledger['input_elements_each'],ledger['representative_elements'],
                          ledger['spatial_edges_each_direction'],ledger['temporal_edges']),
                         (311040,34560,276480,279936))
        self.assertEqual(ledger['by_score'][core.SCORES[2]]['add'],587519)

    def test_invalid_capture_and_score_rejected(self):
        x=np.zeros((10,1,9,9,1),np.float32);s=np.zeros_like(x)
        for bad in ('nan','spike','dtype','short'):
            a=x.copy();b=s.copy()
            if bad=='nan':a[0,0,0,0,0]=np.nan
            elif bad=='spike':b[0,0,0,0,0]=.5
            elif bad=='dtype':a=a.astype(np.float64)
            else:a=a[:1];b=b[:1]
            with self.assertRaises(ValueError):core.tile_scores(a,b)
        rows=[dict(tile=list(t),**{name:0. for name in core.SCORES}) for t in core.TILES]
        rows[0][core.SCORES[0]]=float('nan')
        with self.assertRaises(ValueError):core.select(rows,core.SCORES[0])

    def test_archive_loads_only_two_arrays_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'native.npz'
            u=np.zeros((10,1,30,40,384),np.float32);s=np.zeros_like(u)
            # Loading these unrelated object payloads would fail allow_pickle=False.
            forbidden=np.array([{'must_not_load':True}],dtype=object)
            np.savez_compressed(p,u=u,sn1=s,**{k:forbidden for k in ('head0','head1','head2','head3','branch','sn2')})
            record=dict(prediction_file=p.name,prediction_file_sha256=run.sha(p),
                        tensor_records={name:dict(shape=list(u.shape),sha256=core.payload_sha(x)) for name,x in [('u',u),('sn1',s)]})
            a,b=run.load_capture(p,record);self.assertEqual(a.shape,u.shape);self.assertEqual(b.shape,s.shape)
            bad=copy.deepcopy(record);bad['tensor_records']['u']['sha256']='0'*64
            with self.assertRaises(ValueError):run.load_capture(p,bad)
            bad=copy.deepcopy(record);bad['prediction_file_sha256']='0'*64
            with self.assertRaises(ValueError):run.load_capture(p,bad)

    def test_source_admission_scope_and_common_prefix(self):
        arms=('native','FF','CC')+tuple('fine_%d_%d'%p for p in core.TILES)
        records=[dict(forward_id=i+1,target_id=t,arm=a,tensor_records={'u':{'sha256':t}})
                 for i,(t,a) in enumerate((t,a) for t in core.IDS for a in arms)]
        r=dict(schema='flow_r17_dense_reference_v1',status='DENSE_REFERENCE120_COMPLETE',
               full_forward_count=120,full_forward_attempts=120,
               binding=dict(h8_sha256=run.H8_SHA,BN_mode='REFERENCE_DENSE_CURRENT'),forward_records=records,
               all_states_unchanged=True,all_hooks_removed=True,original_modules_restored=True)
        a=dict(schema='flow_r17_dense_reference_independent_execution_audit_v1',status='PASS',source_result_sha256=run.RESULT_SHA,
               counts=dict(full_forwards=120,archives=120,capture_payloads=64))
        self.assertEqual(len(run.validate_contract(r,a)),8)
        for mutation in ('prefix','duplicate','incomplete','audit'):
            bad=copy.deepcopy(r);bad_a=copy.deepcopy(a)
            if mutation=='prefix':bad['forward_records'][1]['tensor_records']['u']['sha256']='wrong'
            elif mutation=='duplicate':bad['forward_records'][1]['arm']='native'
            elif mutation=='incomplete':bad['full_forward_count']=119
            else:bad_a['status']='FAIL'
            with self.assertRaises(ValueError):run.validate_contract(bad,bad_a)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);a=parser.parse_args()
    start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(schema='flow_r18_g0_cpu_tests_v1',status='PASS' if result.wasSuccessful() else 'FAIL',
                 tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                 source_sha256={n:run.sha(run.HERE/n) for n in run.OWN},
                 wall_seconds=time.monotonic()-start,real_R17_captures_loaded=False,
                 GPU_calls=0,model_forwards=0,GT_loaded=False,quality_loaded=False)
    if a.output:
        core.require(not a.output.exists(),'Fresh CPU receipt');a.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    raise SystemExit(0 if result.wasSuccessful() else 1)
