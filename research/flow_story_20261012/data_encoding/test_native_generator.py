#!/usr/bin/env python3
"""CPU checks against AST-extracted, pinned upstream voxel preprocessing."""
import ast
import json
from pathlib import Path
import types
import unittest
import numpy as np
import torch
import generate_native_inputs as gen

torch.set_num_threads(1)
SOURCE = Path('/home/zhumd/work/sdformer_codex/SDformer/third_party/SDformerFlow')


class NativeGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for rel, expected in gen.SOURCE_PINS.items():
            assert gen.sha(SOURCE / rel) == expected
        rep_ast = ast.parse((SOURCE / 'DSEC_dataloader/event_representations.py').read_text())
        nodes = [n for n in rep_ast.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ('VoxelGrid', 'rectify_events')]
        ns = {'np': np, 'torch': torch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<upstream-representations-AST>', 'exec'), ns)
        cls.rep = types.SimpleNamespace(VoxelGrid=ns['VoxelGrid'], rectify_events=ns['rectify_events'])
        pre_ast = ast.parse((SOURCE / 'DSEC_dataloader/DSEC_dataset_preprocess.py').read_text())
        function = next(n for n in pre_ast.body if isinstance(n, ast.FunctionDef) and n.name == '_load_events')
        clause = next(n for n in ast.walk(function) if isinstance(n, ast.If) and
                      isinstance(n.test, ast.Compare) and any(isinstance(x, ast.Constant) and x.value == 'voxel' for x in n.test.comparators))
        cls.native_code = compile(ast.Module(body=clause.body, type_ignores=[]), '<upstream-voxel-block-AST>', 'exec')
        cls.rectify = np.zeros((480, 640, 2), dtype=np.float32)
        cls.rectify[0, :6, 0] = [0.25, -0.25, 639.75, 640.25, 10.5, 20.25]
        cls.rectify[0, :6, 1] = [0.5, 0.5, 20.75, 21.0, -0.25, 479.75]
        cls.events = {'p': np.array([1, 0, 1, 0, 1, 1], dtype=np.uint8),
                      'x': np.arange(6, dtype=np.uint16), 'y': np.zeros(6, dtype=np.uint16),
                      't': np.array([55614000000, 55614000301, 55614013001, 55614050103, 55614099001, 55614100000], dtype=np.int64)}

    def test_exact_ast_reference_including_outside_rectified_events(self):
        events = self.events
        xy = self.rep.rectify_events(events['x'], events['y'], self.rectify)
        namespace = {'t': events['t'].copy(), 'p': events['p'], 'x_rect': xy[:, 0], 'y_rect': xy[:, 1],
                     'num_frames_per_ts': 10, 'VoxelGrid': self.rep.VoxelGrid, 'torch': torch}
        exec(self.native_code, namespace)
        generated = gen.native_voxel(events, self.rectify, self.rep, np, torch)
        self.assertTrue(np.array_equal(generated, namespace['chunk'].numpy()))

    def test_integer_timestamp_translation_invariance(self):
        before = gen.native_voxel(self.events, self.rectify, self.rep, np, torch)
        shifted = dict(self.events, t=self.events['t'] + 9876543210)
        after = gen.native_voxel(shifted, self.rectify, self.rep, np, torch)
        self.assertTrue(np.array_equal(before, after))

    def test_degenerate_time_rejected_not_linspace_fabricated(self):
        wrong = dict(self.events, t=np.full(6, 55614000000, dtype=np.int64))
        with self.assertRaisesRegex(ValueError, 'undefined'):
            gen.native_voxel(wrong, self.rectify, self.rep, np, torch)

    def test_path_traversal_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            gen.path_under(Path('/tmp/versioned_dataset'), '../old_dataset/test.npy')

    def test_sequence_shards_cover_canonical_disjointly(self):
        manifest = Path(__file__).resolve().parents[2] / 'flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json'
        records = json.loads(manifest.read_text())['records']
        mapping, loads = gen.sequence_shards(records, 4)
        partitions = [{r['target_id'] for r in records if mapping[r['sequence']] == i} for i in range(4)]
        self.assertEqual(sum(loads), 8152)
        self.assertEqual(len(set.union(*partitions)), 8152)
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(partitions[i] & partitions[j])
        self.assertEqual(gen.sequence_shards(records[::-1], 4), (mapping, loads))


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeGeneratorTests))
    gen.atomic_json(Path(__file__).with_name('native_generator_cpu_tests.json'),
                    {'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
                     'GPU_initialized': torch.cuda.is_initialized(), 'torch': str(torch.__version__),
                     'generator_sha256': gen.sha(gen.__file__), 'test_sha256': gen.sha(__file__),
                     'source_pins': gen.SOURCE_PINS,
                     'scope': 'Synthetic CPU AST-reference equality and guards, not real archive/new-host cross-version validation'})
    raise SystemExit(0 if result.wasSuccessful() else 1)
