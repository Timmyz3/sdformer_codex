"""Synthetic admission tests only; no actual predictions or quality are read."""
import copy
from pathlib import Path
import unittest
from reduce_raw import bind_execution, safe_local_file, STEPS, CHECKPOINT_SHA, SOURCE_SHA, CURRENT_SHA


def fixture():
    ids = ['seq_a_' + str(i) for i in range(8)]
    pins = {'execution_auditor_sha256': 'a'*64, 'execution_audit_freeze_sha256': 'b'*64}
    result = {'binding': {'BN_scope': 'ALL78_CURRENT'}, 'checkpoints': {}}
    audit = {'schema': 'flow_bn_conversion_round14_independent_execution_audit_v1', 'status': 'PASS',
        'source_result_sha256': 'c'*64, 'auditor_sha256': 'a'*64, 'audit_freeze_sha256': 'b'*64,
        'full_forwards': 16, 'optimizer_updates': 0, 'calibration_forwards': 0, 'dual_forwards': 0,
        'quality_prediction_count': 16, 'quality_archive_count': 16, 'executed_FC1_rows': 8928000,
        'executed_FC2_rows': 8928000, 'BN_scope': 'ALL78_CURRENT', 'source_states': {}, 'references': {},
        'quality_predictions': []}
    for key in ('all_states_unchanged', 'all_BN_buffers_removed', 'all_hooks_removed', 'original_modules_restored'):
        audit[key] = result[key] = True
    for step in STEPS:
        result['checkpoints'][step] = {'checkpoint_step': int(step), 'checkpoint_sha256': CHECKPOINT_SHA[step],
            'source_state_sha256': SOURCE_SHA[step], 'initial_allcurrent_state_sha256': CURRENT_SHA[step],
            'final_allcurrent_state_sha256': CURRENT_SHA[step], 'full_forward_count': 8,
            'state_unchanged': True, 'BN_buffers_unchanged': True, 'original_modules_restored': True, 'all_hooks_removed': True}
        audit['source_states'][step] = {'checkpoint_sha256': CHECKPOINT_SHA[step], 'source_state_sha256': SOURCE_SHA[step],
            'allcurrent_state_sha256': CURRENT_SHA[step], 'state_tensors': 477, 'state_elements': 54919238,
            'removed_BN_buffers': 234, 'state_unchanged': True, 'all78_current_verified': True}
        audit['references'][step] = {}
        audit['quality_predictions'] += [{'checkpoint_step': int(step), 'target_id': tid, 'layout': 'native', 'array_sha256': 'd'*64} for tid in ids]
    return result, audit, pins, ids


class AdmissionTest(unittest.TestCase):
    def test_complete_fixed16(self):
        result, audit, pins, ids = fixture()
        self.assertEqual(len(bind_execution(result, audit, 'c'*64, pins, ids)), 16)

    def test_bad_count_source_population_or_closure_rejected(self):
        mutations = [
            lambda r,a: a.update(status='PENDING'),
            lambda r,a: a.update(source_result_sha256='x'),
            lambda r,a: a.update(auditor_sha256='x'),
            lambda r,a: a.update(audit_freeze_sha256='x'),
            lambda r,a: a.update(full_forwards=15),
            lambda r,a: a.update(optimizer_updates=1),
            lambda r,a: a.update(calibration_forwards=176),
            lambda r,a: a.update(dual_forwards=1),
            lambda r,a: a.update(quality_prediction_count=15),
            lambda r,a: a.update(quality_archive_count=8),
            lambda r,a: a.update(executed_FC1_rows=0),
            lambda r,a: a.update(BN_scope='PRIVATE24'),
            lambda r,a: a.update(all_BN_buffers_removed=False),
            lambda r,a: r.update(original_modules_restored=False),
            lambda r,a: a['source_states']['271482'].update(checkpoint_sha256=CHECKPOINT_SHA['135930']),
            lambda r,a: r['checkpoints']['271482'].update(final_allcurrent_state_sha256='x'),
            lambda r,a: a['source_states']['271482'].update(state_tensors=711),
            lambda r,a: a['source_states']['271482'].update(all78_current_verified=False),
            lambda r,a: r['checkpoints']['271482'].update(full_forward_count=7),
            lambda r,a: a['quality_predictions'].__setitem__(1, copy.deepcopy(a['quality_predictions'][0])),
            lambda r,a: a['quality_predictions'][0].update(layout='FF'),
            lambda r,a: a['quality_predictions'][0].update(checkpoint_step=271482),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(mutation=i):
                result, audit, pins, ids = fixture()
                mutate(result, audit)
                with self.assertRaises(ValueError):
                    bind_execution(result, audit, 'c'*64, pins, ids)

    def test_artifacts_are_sibling_files(self):
        self.assertEqual(safe_local_file(Path('/tmp/test'), 'frame.npz'), Path('/tmp/test/frame.npz'))
        for value in ('../frame.npz', '/tmp/frame.npz', '.', '..', '', 'a/b.npz'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_local_file(Path('/tmp/test'), value)


if __name__ == '__main__':
    unittest.main()
