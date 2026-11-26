#!/usr/bin/env python3
"""R11 own CPU toy execution, frozen-score replay and GT-free decision checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '2'
import ast
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
import numpy as np
import torch
import run_joint as runner
import frozen_policies as policies
import source_contract as contract
import select_members
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(contract.source_path('fixture').parent))
fixture = contract.load_source('fixture', '_r11_cpu_fixture')
MEASUREMENTS = []


def build():
    torch.manual_seed(20261016)
    model, targets, _, _, functional = fixture.build()
    shapes = {name: (10, 1, 30, 40, 4) for name in targets}
    return model, targets, shapes, torch.randn(10, 1, 30, 40, 4), functional


def restored(test, model, targets, before, parent):
    test.assertTrue(all(model.get_submodule(n) is m for n, m in targets.items()))
    test.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
    test.assertEqual(parent.source_digest(model), before)


class Tests(unittest.TestCase):
    def test_actual_joint_k2_k4_not_singleton_sums_and_dual_contract(self):
        collector, probe, parent, r9, r8 = runner.load_runtime()
        model, targets, shapes, x, functional = build(); before = parent.source_digest(model)
        cc = probe.forward_probe(model, functional, targets, shapes, x, (), fixture.merge, torch, capture=True, capture_u=True)
        self.assertEqual(cc[1]['summed_fc1_rows'], 59200)
        u_sha = cc[1]['layers'][contract.TARGET]['input_sha256']
        observed = []
        for selected in (((0, 3), (2, 0)), ((0, 1), (0, 3), (2, 0), (2, 2))):
            actual = probe.forward_probe(model, functional, targets, shapes, x, selected, fixture.merge, torch,
                                         capture=True, capture_u=True, expected_u_sha=u_sha)
            dual = probe.forward_probe(model, functional, targets, shapes, x, selected, fixture.merge, torch,
                                       capture=True, mode='dual_reference', expected_u_sha=u_sha)
            pair = runner.check_pair(actual, dual, torch)
            self.assertEqual(pair['flow_list_length'], 2)
            self.assertEqual(actual[1]['summed_fc1_rows'], 59200+720*len(selected))
            self.assertEqual(dual[1]['summed_fc1_rows'], 155200)
            self.assertTrue(torch.equal(actual[3]['u'], cc[3]['u']))
            for name, layer in actual[1]['layers'].items():
                self.assertEqual(sum(sum(row) for row in layer['fine_mask']), len(selected) if name == contract.TARGET else 0)
                self.assertFalse(any(layer['fine_mask'][3]))
                self.assertFalse(any(row[4] for row in layer['fine_mask']))
                points = 81*len(selected) if name == contract.TARGET else 0
                representatives = 140-9*len(selected) if points else 140
                self.assertEqual(layer['selected_service']['fine_spatial_points'], points)
                self.assertEqual(layer['selected_service']['coarse_representative_spatial_points'], representatives)
                self.assertEqual(layer['executed_fc_rows_per_layer'], 10*(points+representatives))
            observed.append({'k': len(selected), 'rows': actual[1]['summed_fc1_rows'], 'actual_dual_exact': True})
            restored(self, model, targets, before, parent)
        for arm in ('native', 'FF', 'fine_s2b2'):
            actual = runner.anchor_forward(r9, r8, parent, model, functional, targets, shapes, x, arm, fixture.merge, torch, capture=arm != 'native')
            self.assertEqual(actual[1]['summed_fc1_rows'], 69800 if arm == 'fine_s2b2' else 144000)
            if arm == 'native': self.assertEqual(actual[1]['layers'], {})
            else:
                dual = runner.anchor_forward(r9, r8, parent, model, functional, targets, shapes, x, arm, fixture.merge, torch,
                                             mode='dual_reference', capture=True)
                runner.check_pair(actual, dual, torch)
                if arm == 'fine_s2b2': runner.check_prefix(actual[1], cc[1])
                self.assertTrue(all(actual[1]['layers'][contract.TARGET]['fine_mask'][3]))
                with self.assertRaisesRegex(ValueError, 'complete flow list'):
                    runner.check_pair(actual, (dual[0]+[dual[0][-1]], dual[1], dual[2]), torch)
            restored(self, model, targets, before, parent)
        MEASUREMENTS.append({'CPU_toy_only': True, 'all12_shape': [10, 1, 30, 40, 4], 'joint_pairs': observed,
            'anchor_pairs': ['FF', 'fine_s2b2'], 'five_inputs_four_outputs_exact': True, 'flow_list_length': 2,
            'all8_branches_exact': True, 'real_tails_checked': True})

    def test_reject_bad_prefix_mask_and_restore_after_failure(self):
        collector, probe, parent, _, _ = runner.load_runtime()
        model, targets, shapes, x, functional = build(); before = parent.source_digest(model)
        probe.forward_probe(model, functional, targets, shapes, x, (), fixture.merge, torch)
        for selected in (((0, 0),), ((0, 0), (3, 0)), ((0, 0), (0, 0))):
            with self.assertRaises(ValueError): probe.forward_probe(model, functional, targets, shapes, x, selected, fixture.merge, torch)
        with self.assertRaisesRegex(ValueError, 'predecision input'):
            probe.forward_probe(model, functional, targets, shapes, x, ((0, 0), (0, 1)), fixture.merge, torch, expected_u_sha='0'*64)
        raw = collector.load_probe(); raw.TARGET = contract.TARGET
        def corrupt(*args, **kwargs):
            out = raw.forward_probe(*args, **kwargs)
            if args[5]: out[1]['layers'][contract.PREFIX_NAMES[0]]['output_sha256'] = '0'*64
            return out
        guarded = runner.audited_forward(corrupt)
        guarded(model, functional, targets, shapes, x, (), fixture.merge, torch)
        with self.assertRaisesRegex(ValueError, 'Prior-prefix output'):
            guarded(model, functional, targets, shapes, x, ((0, 0), (0, 1)), fixture.merge, torch)
        with patch.object(targets[contract.TARGET].fc2, 'forward', side_effect=RuntimeError('injected failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                probe.forward_probe(model, functional, targets, shapes, x, ((0, 0), (0, 1)), fixture.merge, torch)
        restored(self, model, targets, before, parent)

    def test_production_service_and_logical_alias_budget(self):
        shapes = {'sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp'.format(s, b):
                  (10, 1, h, w, c) for s, depth, h, w, c in
                  ((0, 2, 120, 160, 96), (1, 2, 60, 80, 192), (2, 6, 30, 40, 384), (3, 2, 15, 20, 768))
                  for b in range(depth)}
        for arm in runner.ANCHORS: self.assertEqual(runner.expected_rows(shapes, arm=arm), runner.ROWS[arm])
        weighted = {}
        for k in (2, 4):
            selected = policies.TILES[:k]
            self.assertEqual(runner.expected_rows(shapes, selected), runner.ROWS['k'+str(k)])
            self.assertEqual(runner.expected_rows(shapes, selected, 'dual_reference'), 609600)
            weighted[k] = sum(runner.expected_rows({n: s}, selected)*8*s[-1]**2 for n, s in shapes.items())
        self.assertEqual(weighted, {2: 71415889920, 4: 73114583040})
        self.assertEqual(runner.BUDGET, 8*(4+16*2)+1+4)
        decisions = {name: {'selected': {str(k): [list(t) for t in policies.TILES[:k]] for k in (2, 4)}} for name in policies.POLICIES}
        plan = policies.joint_plan(decisions)
        self.assertEqual(sum(x['alias_of'] is None for x in plan), 2)
        self.assertEqual(plan[2]['alias_of'], 'B5_task_k2')
        self.assertEqual(plan[3]['alias_of'], 'B5_task_k4')
        self.assertNotEqual(plan[0]['selected_tiles'], plan[1]['selected_tiles'])
        # Alias state cannot leak across frame calls.
        self.assertEqual(policies.joint_plan(decisions), plan)
        MEASUREMENTS.append({'production_geometry_arithmetic_only': True, 'each_FC_rows': runner.ROWS,
            'dual_rows': 609600, 'two_FC_weight_products': weighted, 'cycles_claim': False})

    def test_identity_exclusions_replay_and_old_adjacency_reported(self):
        member = select_members.verify_membership(); inputs = select_members.inputs()
        selected, excluded = select_members.select(*inputs)
        self.assertEqual(len(excluded), 148)
        self.assertEqual([r['target_id'] for r in selected], member['groups']['check'])
        self.assertFalse(set(member['groups']['check']) & set(excluded))
        for sequence in select_members.SEQUENCES:
            suffixes = [int(t.rsplit('_', 1)[1]) for t in member['groups']['check'] if t.startswith(sequence+'_')]
            self.assertEqual(len(suffixes), 4)
            self.assertTrue(all(abs(a-b) >= 3 for i, a in enumerate(suffixes) for b in suffixes[i+1:]))
        collector, _, _, _, _ = runner.load_runtime()
        compat = json.loads((HERE/'selection_manifest_v1.json').read_text())
        chosen = collector.select_rows(compat, json.loads(contract.source_path('fit20').read_text()), inputs[0], inputs[1],
                                       set(inputs[2]['excluded_ids'])-set(inputs[2]['groups']['fit']), 'check')
        self.assertEqual([r['target_id'] for _, r in chosen], member['groups']['check'])
        adjacency = {tid: [old for old in excluded if old.rsplit('_', 1)[0] == tid.rsplit('_', 1)[0]
                          and abs(int(old.rsplit('_', 1)[1])-int(tid.rsplit('_', 1)[1])) < 3]
                     for tid in member['groups']['check']}
        MEASUREMENTS.append({'identity_only': True, 'excluded_count': len(excluded), 'selected': member['groups']['check'],
                             'historical_ID_distance_under3': adjacency, 'independent_time_or_sequence_holdout': False})

    def test_frozen_models_exact_origin_old_score_replay_no_fit_and_ties(self):
        models = policies.load_models()
        old = json.loads(contract.source_path('r10_decisions').read_text())
        for key in ('final_heads', 'selector_models', 'position_mean_gain'): self.assertEqual(models[key], old[key])
        self.assertEqual(policies.ranking(np.ones(12, dtype=np.float32)), list(range(12)))
        raw = json.loads(contract.source_path('r10_result').read_text())
        n = 0
        with patch.object(policies.pm, 'ridge_fit', side_effect=AssertionError('Refitting forbidden')), \
             patch.object(policies.pm, 'pca_fit', side_effect=AssertionError('Refitting forbidden')):
            for frame in raw['frames']:
                if frame['group'] != 'check': continue
                tid = frame['target_id']; path = contract.source_path('r10_result').parent/frame['predecision_input_file']
                self.assertEqual(contract.sha(path), frame['predecision_input_file_sha256'])
                with np.load(path, allow_pickle=False) as z: cells = z['no_GT_cell_descriptor'][0]
                inferred = policies.infer(models, cells, frame['ordinary_scores'])
                for name in policies.POLICIES:
                    index = inferred['decisions'][name]['ranking_indices'][0]
                    self.assertEqual('fine_r{}_c{}'.format(*policies.TILES[index]), old['selected_arms'][name][tid])
                    if name in old['scores']:
                        self.assertTrue(np.array_equal(np.asarray(inferred['decisions'][name]['scores']), np.asarray(old['scores'][name][tid])))
                self.assertEqual(inferred['flow_projection'], old['check_flow_prediction'][tid])
                self.assertEqual(inferred['PCA_projection'], old['check_PCA_prediction'][tid])
                for family, value in inferred['features'].items(): self.assertEqual(value, old['check_features'][tid][family])
                n += 1
        self.assertEqual(n, 8)
        self.assertEqual(tuple(inferred['decisions']['fixed_uniform_index3']['ranking_indices']), policies.UNIFORM_ORDER)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'inference_models_v1.json'; changed = copy.deepcopy(models)
            changed['selector_models']['flow11_task']['beta'][0] += 1
            path.write_text(json.dumps(changed))
            with patch.object(policies, 'HERE', Path(directory)):
                with self.assertRaisesRegex(ValueError, 'Compact coefficient bytes'): policies.load_models()
        MEASUREMENTS.append({'old_R10_no_GT_input_replay_frames': n, 'all16_top1_exact': True,
            'ten_learned_score_vectors_exact': True, 'projected_vectors_and_feature_families_exact': True,
            'new_GTs_or_model_forwards_read': False, 'fit_functions_forbidden': True})

    def test_GT_isolation_writer_before_evaluation_and_event_only_loading(self):
        self.assertEqual(tuple(inspect.signature(policies.infer).parameters), ('models', 'cells', 'ordinary'))
        models = policies.load_models(); rng = np.random.default_rng(15)
        cells = rng.normal(size=(10, 14, 384)).astype(np.float32)
        ordinary = [{'tile': list(tile), **{k: float(rng.random()) for k in
                    tuple(policies.pm.BASE_FEATURES)+('psn1_activity', 'psn1_spatial_neighbor_hamming')}} for tile in policies.TILES]
        common = {'cells': cells, 'ordinary': ordinary, 'gt': np.zeros((2, 8, 8)), 'mask': np.ones((8, 8), dtype=bool)}
        first = policies.infer(models, common['cells'], common['ordinary'])
        common['gt'][:] = float('nan'); common['mask'][:] = False
        second = policies.infer(models, common['cells'], common['ordinary'])
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'decision.json'; events = []
            def write(p, value):
                events.append('write'); p.write_text(json.dumps(value))
            digest = runner.write_decisions_before_evaluation(path, first, write)
            self.assertEqual(digest, contract.sha(path)); events.append('evaluate')
            self.assertEqual(events, ['write', 'evaluate'])
            with self.assertRaisesRegex(ValueError, 'overwrite'): runner.write_decisions_before_evaluation(path, second, write)
            target = Path(directory)/'events.npy'; np.save(target, np.ones((10, 480, 640), dtype=np.float32))
            class Native:
                @staticmethod
                def target_files(row): return {'input': 'events.npy', 'gt': 'GT_MUST_NOT_EXIST.npy', 'mask': 'MASK_MUST_NOT_EXIST.npy'}
                @staticmethod
                def checked_path(root, relative): return Path(root)/relative
            events_array, event_sha = runner.event_only(Native, Path(directory), {}, {'events.npy': {'sha256': contract.sha(target)}})
            self.assertEqual(events_array.shape, (10, 480, 640)); self.assertEqual(event_sha, contract.sha(target))
        # Audit the production function ordering as well as the pure inference API.
        source = inspect.getsource(runner.execute)
        self.assertLess(source.index('write_decisions_before_evaluation('), source.index('metrics.load_arrays('))
        self.assertLess(source.index('write_decisions_before_evaluation('), source.index('metrics.partitions('))
        self.assertNotIn('ridge_fit', source); self.assertNotIn('pca_fit', source)

    def test_safe_import_source_paths_and_bound_receipt(self):
        sources = [HERE/name for name in runner.OWN_SOURCES]
        _, _, parent, _, _ = runner.load_runtime()
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory)/'receipt.json'
            data = {'passed': True, 'GPU_initialized': False, 'source_sha256': {p.name: contract.sha(p) for p in sources}}
            receipt.write_text(json.dumps(data)); parent.checked_cpu_receipt(receipt, sources)
            data['source_sha256']['run_joint.py'] = '0'*64; receipt.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'current source'): parent.checked_cpu_receipt(receipt, sources)
        code = "import os,sys;os.environ['CUDA_VISIBLE_DEVICES']='marker';sys.path.insert(0,"+repr(str(HERE))+");import run_joint as r;r.load_runtime();assert os.environ['CUDA_VISIBLE_DEVICES']=='marker';import torch;assert not torch.cuda.is_initialized()"
        child = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(len(contract.verify_all()), 22)


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources = [HERE/name for name in runner.OWN_SOURCES]
    sources += [Path(fixture.__file__), Path(fixture.fixture.__file__), fixture.fixture.FACTORY]
    receipt = {'schema': 'flow_round11_joint_cpu_contract_v1', 'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
        'torch': str(torch.__version__), 'NumPy': np.__version__, 'GPU_initialized': torch.cuda.is_initialized(), 'full_model_run': False,
        'source_sha256': {str(p): contract.sha(p) for p in sources}, 'source_pins_sha256': contract.sha(HERE/'source_pins.json'),
        'membership_sha256': contract.sha(HERE/'membership_v1.json'), 'models_sha256': contract.sha(HERE/'inference_models_v1.json'),
        'measurements': MEASUREMENTS, 'errors': [str(x) for x in result.errors], 'failures': [str(x) for x in result.failures],
        'scope': 'CPU C4 native-MLP toys; frozen R10 old noGT cache score replay, new identity rules, guards. No new dataset GT, full model, GPU or SSH.'}
    (HERE/'runner_cpu_results.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__ == '__main__': raise SystemExit(main())
