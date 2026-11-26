#!/usr/bin/env python3
"""CPU contract checks; do not establish trained GPU numerical equivalence."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
import torch
import same_shape_tile_mlp as tile
import run_same_shape_reference as runner
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'flow_story_20261011/preflight/cpu_numerics/native_mlp_probe.py'
spec=importlib.util.spec_from_file_location('same_shape_native_factory',SOURCE)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
MEASUREMENTS=[]


class Tests(unittest.TestCase):
    def test_native_all_mixed_none_tail_and_noncontiguous(self):
        torch.manual_seed(213)
        for dtype in (torch.float32,torch.float64):
            mlp=native.make_mlp(8,dtype);state=copy.deepcopy(mlp.state_dict())
            x=torch.randn(10,1,19,40,8,dtype=dtype)[:,:,:,::2,:]
            self.assertFalse(x.is_contiguous())
            for name,mask in (('all',torch.ones(3,3,dtype=torch.bool)),('none',torch.zeros(3,3,dtype=torch.bool)),
                ('mixed',torch.tensor([[1,0,1],[0,1,0],[1,0,1]],dtype=torch.bool)),
                ('tail',torch.tensor([[0,0,0],[0,0,0],[0,0,1]],dtype=torch.bool))):
                reference=tile.SameShapeTileMLP(mlp,'block.mlp','tiled_all_then_mask',lambda n,u:mask,True)
                with torch.inference_mode():a=reference(x)
                replay=tile.ReplayPolicy(reference.records)
                actual=tile.SameShapeTileMLP(mlp,'block.mlp','tiled_kept_only',replay,True)
                with torch.inference_mode():b=actual(x)
                replay.complete();self.assertTrue(torch.equal(a,b));self.assertEqual(a.stride(),b.stride())
                self.assertEqual(reference.records[0]['observed']['native_mlp_calls'],9)
                self.assertEqual(actual.records[0]['observed']['native_mlp_calls'],int(mask.sum()))
                if name=='tail':self.assertEqual(actual.records[0]['observed']['fc1_rows'],20)
                if name=='all':self.assertEqual(actual.records[0]['observed']['native_mlp_calls'],9)
                self.assertEqual(reference.records[0]['output_sha256'],actual.records[0]['output_sha256'])
                MEASUREMENTS.append({'dtype':str(dtype),'mask':name,'reference':reference.records[0],
                    'actual':actual.records[0],'output':runner.compare(a,b,torch)})
            self.assertTrue(all(torch.equal(value,mlp.state_dict()[key]) for key,value in state.items()))
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in mlp.modules()))

    def test_context_restores_on_success_and_exception(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__();self.cnf='ADD';self.use_checkpoint=False;self.mlp=native.make_mlp(8,torch.float32)
            def forward(self,x):return self.mlp(x)+x
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.first=Block();self.second=Block()
            def forward(self,x):return self.second(self.first(x))
        model=Model().eval();original=(model.first.mlp,model.second.mlp)
        x=torch.randn(10,1,10,11,8);policy=lambda n,u:torch.tensor([[1,0],[0,1]],dtype=torch.bool)
        with torch.inference_mode():
            with tile.same_shape_mlp_model(model,'tiled_all_then_mask',policy,True) as wrappers:
                a=model(x);records=[r for w in wrappers for r in w.records]
            replay=tile.ReplayPolicy(records)
            with tile.same_shape_mlp_model(model,'tiled_kept_only',replay,True):b=model(x)
            replay.complete();self.assertTrue(torch.equal(a,b))
        self.assertEqual(original,(model.first.mlp,model.second.mlp))
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with tile.same_shape_mlp_model(model,'tiled_all_then_mask',policy):raise RuntimeError('injected')
        self.assertEqual(original,(model.first.mlp,model.second.mlp))

    def test_contract_guards(self):
        mlp=native.make_mlp(8,torch.float32);x=torch.randn(10,1,3,5,8)
        w=tile.SameShapeTileMLP(mlp,'a','tiled_kept_only',lambda n,u:torch.ones(1,1,dtype=torch.bool))
        with self.assertRaisesRegex(ValueError,'Inference only'):w(x)
        with torch.inference_mode():
            with self.assertRaisesRegex(ValueError,'T10'):w(x[:5])
        bn=next(m for m in mlp.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))
        bn.track_running_stats=False
        with torch.inference_mode():
            with self.assertRaisesRegex(ValueError,'Current-stat'):w(x)

    def test_replay_rejects_mismatch_missing_duplicate(self):
        r={'layer_name':'a','shape_TBHWC':[10,1,1,1,8],'keep':[[True]]}
        with self.assertRaisesRegex(ValueError,'Duplicate'):tile.ReplayPolicy([r,r])
        p=tile.ReplayPolicy([r]);x=torch.zeros(10,1,1,1,8)
        with self.assertRaisesRegex(ValueError,'Incomplete'):p.complete()
        with self.assertRaisesRegex(ValueError,'geometry'):p('a',x[:,:,:,:,:4])
        p('a',x);p.complete()
        with self.assertRaisesRegex(ValueError,'duplicate'):p('a',x)

    def test_pinned_loader_and_policy(self):
        runner.probe.evaluator()
        self.assertEqual(runner.sha(HERE/'policies.py'),runner.POLICIES_SHA)


def main():
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    record={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'errors':[str(v) for v in result.errors],
        'failures':[str(v) for v in result.failures],'GPU_initialized':torch.cuda.is_initialized(),
        'scope':'synthetic CPU native MLP contract tests; GPU/real checkpoint and AEE not tested',
        'source_sha256':{p.name:runner.sha(p) for p in (Path(__file__),Path(tile.__file__),Path(runner.__file__),SOURCE)},
        'measurements':MEASUREMENTS}
    (HERE/'same_shape_reference_cpu_results.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__=='__main__':raise SystemExit(main())
