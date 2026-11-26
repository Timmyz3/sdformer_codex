#!/usr/bin/env python3
"""Small native-MLP fixture only; no full network, data or GPU execution."""
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
import run_repair_scope as runner

HERE = Path(__file__).resolve().parent
COLLECTOR = runner.load_collector(); PROBE = COLLECTOR.load_probe(); PARENT = PROBE.load_scope_runner()
FIXTURE_PATH = PROBE.BASE/'scope_extension_v1/test_scope_extension_cpu.py'
sys.path.insert(0, str(FIXTURE_PATH.parent))
spec = importlib.util.spec_from_file_location('_repair_fixture', FIXTURE_PATH)
FIXTURE = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIXTURE)
MERGE = PARENT.load_frozen_merge()
MEASUREMENTS = []


class Tests(unittest.TestCase):
    def test_mixed_actual_dual_and_masks(self):
        model, targets, shapes, x, functional = FIXTURE.build()
        digest = PARENT.source_digest(model); original_adapter = PARENT.wrapped_model
        for coarse, expected_rows in [((0,), 15800), ((2,), 10440)]:
            actual, ex, branch = runner.mixed_forward(PARENT, model, functional, targets, shapes, x, coarse,
                                                      MERGE, torch, capture=True)
            dual, dx, dbranch = runner.mixed_forward(PARENT, model, functional, targets, shapes, x, coarse,
                                                    MERGE, torch, mode='dual_reference', capture=True)
            self.assertEqual(len(ex['layers']), 8)
            self.assertEqual(ex['summed_fc1_rows'], expected_rows)
            self.assertEqual(ex['summed_fc2_rows'], expected_rows)
            self.assertEqual(dx['summed_fc1_rows'], 20080)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, dual)))
            self.assertTrue(all(torch.equal(branch[k], dbranch[k]) for k in branch))
            for name, r in ex['layers'].items():
                stage = next(s for s in (0, 2) if f'.layers.{s}.' in name)
                self.assertTrue(all(v == (stage not in coarse) for row in r['fine_mask'] for v in row))
            self.assertIs(PARENT.wrapped_model, original_adapter)
            self.assertEqual(PARENT.source_digest(model), digest)
            self.assertTrue(all(model.get_submodule(name) is m for name, m in targets.items()))
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        MEASUREMENTS.append({'mixed_actual_dual_exact': True, 'all8_tiled': True, 'toy_actual_rows': [15800, 10440]})

    def test_anchor_adapter_matches_frozen_parent(self):
        model, targets, shapes, x, functional = FIXTURE.build()
        for coarse, kind in [((), 'fine'), ((0, 2), 'coarse')]:
            actual, ex, _ = runner.mixed_forward(PARENT, model, functional, targets, shapes, x, coarse, MERGE, torch)
            prior, px, _ = PARENT.forward_probe(model, functional, targets, shapes, x, kind, (0, 2), MERGE, torch)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, prior)))
            self.assertEqual(ex['all12_observed_fc_rows'], px['all12_observed_fc_rows'])
            self.assertEqual(ex['layers'], px['layers'])
        MEASUREMENTS.append({'old_coarse_and_fine_graph_preserved_on_toy': True})

    def test_failure_restores_wrappers_hooks_and_adapter(self):
        model, targets, shapes, x, functional = FIXTURE.build(); adapter = PARENT.wrapped_model
        with patch.object(model, 'forward', side_effect=RuntimeError('synthetic forward failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic forward failure'):
                runner.mixed_forward(PARENT, model, functional, targets, shapes, x, (0,), MERGE, torch)
        self.assertIs(PARENT.wrapped_model, adapter)
        self.assertTrue(all(model.get_submodule(name) is m for name, m in targets.items()))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        MEASUREMENTS.append({'failure_cleanup': True})


if __name__ == '__main__':
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(20261014)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources = [Path(__file__), Path(runner.__file__), runner.COLLECTOR_PATH, PROBE.SCOPE_RUNNER,
               PROBE.BASE/'structured_merge_mlp.py', FIXTURE_PATH]
    receipt = {'passed': result.wasSuccessful(), 'tests_run': result.testsRun, 'torch': str(torch.__version__),
               'GPU_initialized': torch.cuda.is_initialized(), 'full_model_run': False,
               'source_sha256': {str(p): runner.sha(p) for p in sources}, 'measurements': MEASUREMENTS,
               'scope': 'Small CPU mixed-stage geometry/dual/anchor/cleanup tests; no quality or full model',
               'errors': [str(x) for x in result.errors], 'failures': [str(x) for x in result.failures]}
    (HERE/'runner_cpu_results.json').write_text(json.dumps(receipt, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
