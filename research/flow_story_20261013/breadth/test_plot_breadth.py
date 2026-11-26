import copy
import json
from pathlib import Path
import tempfile
import unittest

import plot_breadth as plot
import test_summary


class PlotTests(unittest.TestCase):
    def fixture(self):
        original = test_summary.SummaryTests().fixture()
        original['cases'][1].update(policy='uniform', keep_fraction=.75, execution='dense_masked')
        original.update(checkpoint_run_contract_sha256='r'*64, data_contract={'encoding_id': 'synthetic'},
                        source_sha256={'model.py': 's'*64}, input_file_sha256={'inputs': 'i'*64},
                        method_implementation_sha256={'policies.py': 'p'*64, 'static_methods.py': 's'*64}, precision='FP32, TF32 off, no autocast',
                        training_performed=False)
        calibrated = copy.deepcopy(original)
        ids = ['cal%d' % i for i in range(176)]
        calibrated['BN_artifact'] = {'sha256': 'b'*64, 'metadata': {
            'schema': 'flow_breadth_bn_calibration_v1', 'status': 'COMPLETE', 'method': 'cma_native_batch_stats_reset',
            'fit_target_ids': ids, 'completed_ids': ids, 'fit_target_ids_sha256': plot.digest(ids),
            'checkpoint_sha256': original['checkpoint_sha256'], 'data_contract_sha256': plot.digest(original['data_contract']),
            'source_sha256': original['source_sha256'], 'geometry': 'full9', 'non_bn_state_unchanged': True,
            'optimizer_steps': 0, 'fit_uses_final_valid': False, 'fit_role': 'bn_calibration'}}
        for entry in calibrated['arms'].values():
            entry['BN_buffer_load'] = {'count_per_BN': 176, 'non_bn_state_unchanged': True}
        return original, calibrated

    def test_shared_scale_paired_values_and_null_missing(self):
        result = plot.prepare(*self.fixture(), 'a'*64, 'b'*64)
        self.assertEqual(result['color_scale']['vmin'], -.5)
        self.assertEqual(result['color_scale']['vmax'], .5)
        for panel in result['panels']:
            self.assertEqual(len(panel['cells']), 24)
            self.assertEqual(panel['cells'][0]['delta_AEE'], .5)
            self.assertIsNone(panel['cells'][1]['delta_AEE'])

    def test_binding_errors_rejected(self):
        for field, replacement in [('checkpoint_sha256', 'x'*64), ('geometry', 'crop_center9'),
                                   ('population', {}), ('source_sha256', {'other': 'x'*64})]:
            original, calibrated = self.fixture()
            calibrated[field] = replacement
            with self.assertRaises((ValueError, KeyError)):
                plot.prepare(original, calibrated, 'a'*64, 'b'*64)
        original, calibrated = self.fixture()
        calibrated['BN_artifact']['metadata']['completed_ids'] = []
        with self.assertRaises(ValueError):
            plot.prepare(original, calibrated, 'a'*64, 'b'*64)

    def test_failed_and_incomplete_not_zero(self):
        original, calibrated = self.fixture()
        original['arms']['candidate']['status'] = 'FAILED'
        calibrated['arms']['candidate']['frames'].pop()
        p1 = plot.panel(original, 'a'*64, 'Original D')
        p2 = plot.panel(calibrated, 'b'*64, 'CMA176 D', True)
        self.assertIsNone(p1['cells'][0]['delta_AEE'])
        self.assertIsNone(p2['cells'][0]['delta_AEE'])
        self.assertEqual(p1['cells'][0]['status'], 'FAILED')
        self.assertEqual(p2['cells'][0]['status'], 'INVALID_EVIDENCE')

    def test_explicit_extended_source_location_and_conflict(self):
        original, calibrated = self.fixture()
        static = calibrated['method_implementation_sha256'].pop('static_methods.py')
        calibrated['extended_implementation_sha256'] = {'static_methods.py': static}
        plot.prepare(original, calibrated, 'a'*64, 'b'*64)
        calibrated['method_implementation_sha256']['static_methods.py'] = 'x'*64
        with self.assertRaises(ValueError):
            plot.prepare(original, calibrated, 'a'*64, 'b'*64)

    def test_cpu_png_and_pdf_render(self):
        result = plot.prepare(*self.fixture(), 'a'*64, 'b'*64)
        with tempfile.TemporaryDirectory(prefix='flow_breadth_plot_test_') as directory:
            out = Path(directory)
            plot.render(result, out)
            self.assertTrue((out/'breadth_delta_aee.png').read_bytes().startswith(b'\x89PNG'))
            self.assertTrue((out/'breadth_delta_aee.pdf').read_bytes().startswith(b'%PDF'))


if __name__ == '__main__':
    run = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PlotTests))
    Path(__file__).with_name('plot_cpu_checks.json').write_text(json.dumps({
        'passed': run.wasSuccessful(), 'tests': run.testsRun, 'GPU_executed': False,
        'scope': 'Synthetic report bindings, missingness and PNG/PDF CPU rendering; no scientific result generated'}, indent=2)+'\n')
    raise SystemExit(0 if run.wasSuccessful() else 1)
