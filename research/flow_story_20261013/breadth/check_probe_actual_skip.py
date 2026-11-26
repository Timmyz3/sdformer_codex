#!/usr/bin/env python3
"""CPU-only correctness of diagnostic hooks/comparisons, not GPU root cause."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
import torch
import probe_actual_skip as p

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'flow_story_20261011/preflight/cpu_numerics/native_mlp_probe.py'
spec=importlib.util.spec_from_file_location('skip_diag_native_factory',SOURCE)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
MEASUREMENTS=[]


class Tests(unittest.TestCase):
    def test_real_native_stage_hooks_F32_F64_and_tail(self):
        for dtype,tolerance in ((torch.float32,2e-5),(torch.float64,1e-11)):
            mlp=native.make_mlp(8,dtype);state=copy.deepcopy(mlp.state_dict())
            x=torch.randn(10,1,19,20,8,dtype=dtype);views=p.tile_views(19,20,3)
            self.assertEqual(views[-1][1:],[18,19,18,20] if isinstance(views[-1],list) else (18,19,18,20))
            with torch.inference_mode():results,traces=p.local_probe(mlp,x,views)
            self.assertEqual(set(traces),set(p.STAGES))
            for tile in results:
                self.assertEqual(tile['tile_input_shape'][0:2],[10,1])
                for name,stage in tile['stages'].items():
                    self.assertLessEqual(stage['max_abs_error'],tolerance)
            self.assertTrue(all(torch.equal(v,mlp.state_dict()[k]) for k,v in state.items()))
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in mlp.modules()))
            MEASUREMENTS.append({'dtype':str(dtype),'source':'synthetic native MLP, same CPU u','results':results})

    def test_PSN_prehook_is_actual_addmm_before_threshold(self):
        mlp=native.make_mlp(8,torch.float64);x=torch.randn(10,1,3,5,8,dtype=torch.float64)
        with torch.inference_mode():trace=p.trace_mlp(mlp,x,[('all',0,3,0,5)])
        sn1=mlp.sn1.spiking_neuron
        with torch.inference_mode():h=torch.addmm(sn1.bias,sn1.weight,x.flatten(1)).reshape(x.shape)
        self.assertTrue(torch.equal(h,trace['sn1_h']['all']))
        self.assertTrue(torch.equal((h>=0).double(),trace['sn1']['all']))
        bn1=trace['bn1']['all'];sn2=mlp.sn2.spiking_neuron
        with torch.inference_mode():h2=torch.addmm(sn2.bias,sn2.weight,bn1.flatten(1)).reshape(bn1.shape)
        self.assertTrue(torch.equal(h2,trace['sn2_h']['all']))
        self.assertTrue(torch.equal((h2>=0).double(),trace['sn2']['all']))

    def test_sign_disagreement_detects_small_membrane_errors(self):
        a=torch.tensor([-1e-8,0,1e-8,1.]).reshape(1,1,1,1,4)
        b=torch.tensor([1e-8,-1e-8,2e-8,1.]).reshape(1,1,1,1,4)
        result=p.compare_stage(a,b,'sn2_h',(9,18))
        self.assertEqual(result['threshold_sign_disagreements'],2)
        self.assertTrue(result['allclose_atol1e_5_rtol1e_5'])
        self.assertEqual(result['first_sign_disagreements'][0]['global_TBHWC_index'],[0,0,9,18,0])
        spikes=p.compare_stage((a>=0).float(),(b>=0).float(),'sn2')
        self.assertEqual(spikes['spike_disagreements'],2)
        self.assertFalse(spikes['allclose_atol1e_5_rtol1e_5'])

    def test_frozen_evaluator_and_tile_selection(self):
        p.evaluator()
        self.assertEqual(len(p.tile_views(120,160,3)),3)
        self.assertEqual(p.tile_views(8,8,3),[('0,0',0,8,0,8)])


def main():
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    output={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'errors':[str(x) for x in result.errors],
        'failures':[str(x) for x in result.failures],'GPU_initialized':torch.cuda.is_initialized(),
        'scope':'CPU diagnostic implementation checks only; actual trained GPU discrepancy not reproduced here',
        'source_sha256':{f.name:p.sha(f) for f in (Path(p.__file__),Path(__file__),SOURCE)},'measurements':MEASUREMENTS}
    (HERE/'probe_actual_skip_cpu_results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__=='__main__':raise SystemExit(main())
