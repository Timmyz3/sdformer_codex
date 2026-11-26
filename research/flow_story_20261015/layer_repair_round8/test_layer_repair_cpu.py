#!/usr/bin/env python3
"""Small frozen native-MLP fixture; no full network, data or GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
import run_layer_repair as runner

HERE = Path(__file__).resolve().parent
COLLECTOR = runner.load_collector(); PROBE = COLLECTOR.load_probe(); PARENT = PROBE.load_scope_runner()
FIXTURE_PATH = PROBE.BASE/'scope_extension_v1/test_scope_extension_cpu.py'
sys.path.insert(0, str(FIXTURE_PATH.parent))
spec = importlib.util.spec_from_file_location('_layer_repair_fixture', FIXTURE_PATH)
FIXTURE = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIXTURE)
spec = importlib.util.spec_from_file_location('_frozen_r7', runner.OLD/'repair_scope_round7/run_repair_scope.py')
R7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R7)
MERGE = PARENT.load_frozen_merge()
MEASUREMENTS = []


class Tests(unittest.TestCase):
    def test_all_six_actual_dual_masks_and_real_tails(self):
        model, targets, shapes, x, functional = FIXTURE.build()
        digest = PARENT.source_digest(model); adapter = PARENT.wrapped_model
        for arm, fine in runner.ARMS:
            actual, ex, branch = runner.mixed_forward(PARENT, model, functional, targets, shapes, x, fine, MERGE, torch, capture=True)
            dual, dx, dbranch = runner.mixed_forward(PARENT, model, functional, targets, shapes, x, fine, MERGE, torch, mode='dual_reference', capture=True)
            self.assertEqual(len(ex['layers']), 8)
            self.assertEqual(ex['summed_fc1_rows'], 9100)
            self.assertEqual(ex['summed_fc2_rows'], 9100)
            self.assertEqual(dx['summed_fc1_rows'], 20080)
            self.assertEqual(len(actual), len(dual))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,dual)))
            self.assertEqual(branch.keys(), dbranch.keys())
            self.assertTrue(all(torch.equal(branch[k],dbranch[k]) for k in branch))
            for name, record in ex['layers'].items():
                expected = name == runner.PREFIX+str(fine[0])+'.mlp'
                self.assertTrue(all(v == expected for row in record['fine_mask'] for v in row))
            self.assertIs(PARENT.wrapped_model,adapter)
            self.assertEqual(PARENT.source_digest(model),digest)
            self.assertTrue(all(model.get_submodule(name) is m for name,m in targets.items()))
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        MEASUREMENTS.append({'all6_actual_dual_exact':True,'all8_tiled':True,'toy_each_arm_rows':9100})

    def test_CC_CF_match_frozen_R7_adapter(self):
        model,targets,shapes,x,functional=FIXTURE.build()
        for fine, coarse in [((),(0,2)),(tuple(range(6)),(0,))]:
            actual,ex,_=runner.mixed_forward(PARENT,model,functional,targets,shapes,x,fine,MERGE,torch)
            prior,px,_=R7.mixed_forward(PARENT,model,functional,targets,shapes,x,coarse,MERGE,torch)
            self.assertEqual(len(actual),len(prior))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,prior)))
            self.assertEqual(ex['all12_observed_fc_rows'],px['all12_observed_fc_rows'])
            self.assertEqual(ex['layers'],px['layers'])
        MEASUREMENTS.append({'CC_CF_graph_preserved_on_toy':True})

    def test_failure_cleanup(self):
        model,targets,shapes,x,functional=FIXTURE.build(); adapter=PARENT.wrapped_model
        with patch.object(model,'forward',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
                runner.mixed_forward(PARENT,model,functional,targets,shapes,x,(3,),MERGE,torch)
        self.assertIs(PARENT.wrapped_model,adapter)
        self.assertTrue(all(model.get_submodule(name) is m for name,m in targets.items()))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        MEASUREMENTS.append({'failure_cleanup':True})


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261014)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources=[Path(__file__),Path(runner.__file__),runner.COLLECTOR_PATH,Path(R7.__file__),PROBE.SCOPE_RUNNER,
             PROBE.BASE/'structured_merge_mlp.py',FIXTURE_PATH]
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'torch':str(torch.__version__),
             'GPU_initialized':torch.cuda.is_initialized(),'full_model_run':False,
             'source_sha256':{str(p):runner.sha(p) for p in sources},'measurements':MEASUREMENTS,
             'scope':'Small CPU layer masks, true tails, dual, anchor and failure cleanup; no full model or quality',
             'errors':[str(x) for x in result.errors],'failures':[str(x) for x in result.failures]}
    (HERE/'runner_cpu_results.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
