"""Bounded synthetic CPU checks; zero full-model, GT or CUDA work."""
import copy, hashlib, importlib.util, json, os, sys, tempfile, time, unittest
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_r30_final_vectors as export
sys.path.insert(0, str(HERE.parents[1]/'implementation'))
import mixed_integer_core as core
import torch
import numpy as np
torch.set_num_threads(2)


def mask(h=30, w=40):
    out = torch.zeros(h, w, dtype=torch.bool)
    out[0:9, 0:9] = True
    return out


def fixture():
    sources = {'members': 'm', 'core': 'c'}
    result = dict(schema='flow_r30_joint_recovery_v1', graph_id=export.GRAPH, arm='joint',
        status='JOINT512_AND256_DEVELOPMENT_PREDICTIONS_COMPLETE', engineering_smoke=False,
        model_forwards=1289, expected_model_forwards=1289, completed_optimizer_steps=512,
        weights_sha256='weights', initial_state_sha256='initial', final_state_sha256='final',
        training_state_sha256='final', final_parity={'M': {
            'training_deployment': {'pass': True}, 'dense_integer': {'pass': True}}},
        sources=sources, recipe={'updates': 512}, joint_contract={'mode': 'M'})
    audit = dict(schema='flow_r30_independent_actual_execution_audit_v1', status='PASS',
        engineering_smoke=False, quality_analysis_allowed=True, graph_id=export.GRAPH,
        auditor_sha256=export.AUDITOR_SHA, source_result_sha256='result',
        checkpoint_sha256='weights', source_mask_contract_sha256=export.CONTRACT_SHA,
        final_state_sha256='final', counts=dict(optimizer_updates=512,
            quality_predictions=256, formal_model_forwards=1289))
    return result, audit


class Tests(unittest.TestCase):
    def test_accepted_scope_and_reject_wrong_source(self):
        r, a = fixture()
        export.validate_completed(r, a, 'result', 'weights', export.CONTRACT_SHA)
        changes = [('schema', 'other'), ('engineering_smoke', True),
                   ('completed_optimizer_steps', 1), ('model_forwards', 5),
                   ('graph_id', 'old')]
        for key, value in changes:
            bad = copy.deepcopy(r); bad[key] = value
            with self.assertRaises(ValueError):
                export.validate_completed(bad, a, 'result', 'weights', export.CONTRACT_SHA)
        for key, value in [('checkpoint_sha256', 'wrong'), ('auditor_sha256', 'wrong'),
                           ('source_result_sha256', 'wrong'), ('status', 'PENDING'),
                           ('final_state_sha256', 'wrong')]:
            bad = copy.deepcopy(a); bad[key] = value
            with self.assertRaises(ValueError):
                export.validate_completed(r, bad, 'result', 'weights', export.CONTRACT_SHA)

    def test_checkpoint_final_only(self):
        r, _ = fixture()
        state = {f'p{i}': torch.ones(1) for i in range(477)}
        cp = dict(schema='flow_r30_joint_checkpoint_v1', arm='joint',
            completed_optimizer_steps=512, model_state_dict=state,
            sources=r['sources'], recipe=r['recipe'], joint_contract=r['joint_contract'],
            initial_state_sha256='initial', final_state_sha256='final')
        self.assertIs(export.checkpoint_state(cp, r, lambda _: 'final'), state)
        for key, value in [('arm', 'Q8'), ('completed_optimizer_steps', 1),
                           ('final_state_sha256', 'other'), ('sources', {})]:
            bad = copy.deepcopy(cp); bad[key] = value
            with self.assertRaises(ValueError):
                export.checkpoint_state(bad, r, lambda _: 'final')

    def test_exact_four_coordinate_classes(self):
        self.assertEqual(export.coordinates(mask()), [
            ('restored_complete_Q8', 4, 4, True), ('low_complete_Q4', 4, 13, False),
            ('right_tail_Q8', 0, 36, True), ('bottom_tail_Q8', 27, 0, True)])
        for broken in [mask(27, 36), torch.zeros(30, 40, dtype=torch.bool)]:
            with self.assertRaises(ValueError):
                export.coordinates(broken)
        broken = mask(); broken[0, 0] = False
        with self.assertRaises(ValueError):
            export.coordinates(broken)

    def test_actual_conv_input_halo_order(self):
        torch.manual_seed(20261020)
        x = torch.randint(0, 2, (10, 1, 8, 30, 40)).float()
        unfold = torch.nn.functional.unfold(x.reshape(10, 8, 30, 40), 3, padding=1)
        for t in (0, 9):
            for _, y, col, _ in export.coordinates(mask()):
                actual = export.binary_row(x, 'conv', t, 0, y, col)
                self.assertTrue(torch.equal(actual, unfold[t, :, y*40+col].to(torch.int8)))
        bad = x.clone(); bad[0, 0, 0, 0, 36] = 0.5
        with self.assertRaises(ValueError):
            export.binary_row(bad, 'conv', 0, 0, 0, 36)

    def test_real_integer_micro_capture_and_exports(self):
        torch.manual_seed(20261020)
        groups = {core.group_name(name): mask() for name in export.TARGETS}
        sources = (torch.nn.Linear(8, 32, bias=False), torch.nn.Conv2d(8, 32, 3, padding=1, bias=False))
        wrappers = {name: core.STEAffine(source, kind, 'M',
                    selected=groups[core.group_name(name)], ste=False)
                    for name, source, kind in zip(export.TARGETS, sources, ('linear', 'conv'))}
        inputs = (torch.randint(0, 2, (10, 1, 30, 40, 8)).float(),
                  torch.randint(0, 2, (10, 1, 8, 30, 40)).float())
        with export.capture_small(wrappers, groups, core) as captures:
            with torch.no_grad():
                for name, x in zip(export.TARGETS, inputs):
                    wrappers[name](x)
        self.assertTrue(all(not w.capture_integer and w.last_integer is None and
                            not w._forward_hooks and not w._forward_pre_hooks
                            for w in wrappers.values()))
        with tempfile.TemporaryDirectory() as tmp:
            result = export.export_cases(Path(tmp), captures)
            self.assertEqual(len(result['cases']), 32)
            vectors = (Path(tmp)/result['vectors_file']).read_text().splitlines()
            expected = (Path(tmp)/result['expected_file']).read_text().splitlines()
            self.assertEqual(len(vectors), 97); self.assertEqual(len(expected), 33)
            with np.load(Path(tmp)/result['payload_file'], allow_pickle=False) as arrays:
                self.assertEqual(len(arrays.files), 128)
                for record in result['cases']:
                    for suffix, desc in record['payloads'].items():
                        self.assertEqual(export.payload(arrays[desc['key']]), desc['sha256'])
            bad = copy.deepcopy(captures)
            bad[export.TARGETS[0]]['cases'][0]['expected'][0] += 1
            with self.assertRaises(ValueError):
                export.export_cases(Path(tmp), bad)

    def test_exception_hook_cleanup(self):
        name = export.TARGETS[0]
        w = core.STEAffine(torch.nn.Linear(8, 32, bias=False), 'linear', 'M',
                          selected=mask(), ste=False)
        with self.assertRaises(ValueError):
            with export.capture_small({name: w}, {core.group_name(name): mask()}, core, (name,)):
                with torch.no_grad():
                    w(torch.full((10, 1, 30, 40, 8), 0.5))
        self.assertIsNone(w.last_integer)
        self.assertFalse(w.capture_integer)
        self.assertFalse(w._forward_hooks or w._forward_pre_hooks)


if __name__ == '__main__':
    destination = Path(sys.argv[1])
    if destination.exists():
        raise FileExistsError(destination)
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    export.write(destination, dict(
        schema='flow_r30_final_vector_export_cpu_v1', status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        seconds=time.monotonic()-started,
        sources={str(p.relative_to(HERE.parents[1])): export.sha(p)
                 for p in (Path(export.__file__), Path(__file__),
                           HERE.parents[1]/'implementation/mixed_integer_core.py')},
        full_model_forwards=0, real_integer_micro_forwards=2,
        GPU_used=False, GT_values_read=False))
    raise SystemExit(not result.wasSuccessful())
