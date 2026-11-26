import copy
from pathlib import Path
import unittest

import numpy as np

from reduce_raw import CHECKPOINT_SHA, METRIC, bind_execution, safe_local_file


def fixture():
    pins = {'execution_auditor_sha256': 'auditor', 'execution_audit_freeze_sha256': 'freeze'}
    result = {'binding': {'checkpoint_sha256': CHECKPOINT_SHA},
              'own_BN_buffers': {'sha256': 'buffers'}, 'own_BN_receipt': {'sha256': 'receipt'}}
    audit = {'schema': 'flow_later_checkpoint_independent_execution_audit_v1', 'status': 'PASS',
        'source_result_sha256': 'actual', 'auditor_sha256': 'auditor', 'pre_result_freeze_sha256': 'freeze',
        'checkpoint_sha256': CHECKPOINT_SHA, 'BN_buffers_sha256': 'buffers', 'BN_receipt_sha256': 'receipt',
        'full_forwards': 193, 'optimizer_updates': 0, 'calibration_forwards': 176,
        'quality_prediction_count': 16, 'quality_archive_count': 8, 'dual_forward_count': 1,
        'source_state_independently_reconstructed': True, 'non_bn_weights_unchanged': True,
        'BN_state_closure': True, 'actual_dual_raw_arrays_exact': True}
    return result, audit, pins


class IOCPU(unittest.TestCase):
    def test_actual_audit_source_and_complete_counts_required(self):
        result, audit, pins = fixture()
        bind_execution(result, audit, 'actual', pins)
        changes = {'schema': 'other', 'status': 'INCOMPLETE', 'source_result_sha256': 'other',
            'auditor_sha256': 'other', 'pre_result_freeze_sha256': 'other', 'checkpoint_sha256': 'old',
            'BN_buffers_sha256': 'old', 'BN_receipt_sha256': 'old', 'full_forwards': 192,
            'optimizer_updates': 1, 'calibration_forwards': 175, 'quality_prediction_count': 15,
            'quality_archive_count': 9, 'dual_forward_count': 0,
            'source_state_independently_reconstructed': False, 'non_bn_weights_unchanged': False,
            'BN_state_closure': False, 'actual_dual_raw_arrays_exact': False}
        for name, value in changes.items():
            with self.subTest(field=name):
                wrong = copy.deepcopy(audit)
                wrong[name] = value
                with self.assertRaises(ValueError):
                    bind_execution(result, wrong, 'actual', pins)

    def test_artifacts_cannot_escape_actual_result_directory(self):
        self.assertEqual(safe_local_file(Path('/tmp/run'), 'pred.npz'), Path('/tmp/run/pred.npz'))
        for name in ('../other.npz', '/tmp/other.npz', '.', '..', 'nested/pred.npz'):
            with self.assertRaises(ValueError):
                safe_local_file(Path('/tmp/run'), name)

    def test_inherited_metric_preserves_all_valid_and_empty_regions(self):
        gt = np.zeros((2, 4, 4), dtype=np.float32)
        valid = np.ones((4, 4), dtype=bool)
        events = np.zeros((10, 4, 4), dtype=np.float32)
        pred = np.stack([np.full((4, 4), 3, dtype=np.float32), np.full((4, 4), 4, dtype=np.float32)])
        masks = METRIC.masks_from_original(gt, valid, events, 1., 1.)
        measured = METRIC.measure(pred, gt, masks)
        self.assertEqual(measured['all_valid'], {'valid_pixels': 16, 'error_sum': 80., 'AEE': 5.})
        self.assertEqual(measured['voxel_zero'], measured['all_valid'])
        self.assertEqual(measured['L'], {'valid_pixels': 0, 'error_sum': 0., 'AEE': None})
        self.assertEqual(len(measured), 16)


if __name__ == '__main__':
    unittest.main()
