#!/usr/bin/env python3
"""Small CPU tests of native acceptance guards; never reads remote data."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_verify', HERE / 'verify_native_dataset.py')
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
DATA = HERE.parents[1] / 'flow_story_20261011/preflight/data_protocol'


def contract():
    return {'encoding_id': v.ENCODING_ID, 'canonical_manifest_sha256': v.CANONICAL_SHA,
            'source_pins': v.SOURCE_PINS, 'script_sha256': v.GENERATOR_SHA,
            'CPU_threads': 1, 'bins': 10, 'shape': [10, 480, 640], 'dtype': 'float32',
            'time': '(t-t[0]).astype(float32); t/t[-1]; native convert_CHW renormalization',
            'spatial': 'all sliced events; native rectify and tap-wise validity',
            'polarity': 'signed 2p-1; no post-voxel z-score', 'slice': '[t_beg,t_end)',
            'torch': '2.5.1+cu124', 'numpy': '2.2.6', 'h5py': '3.15.1'}


def shard_reports(records, root):
    mapping, loads = v.expected_shards(records)
    reports = []
    for index in range(4):
        ids = [r['target_id'] for r in records if mapping[r['sequence']] == index]
        reports.append({'status': 'SHARD_COMPLETE', 'GPU_initialized': False, 'GT_mask_modified': False,
                        'encoding_id': v.ENCODING_ID, 'canonical_manifest_sha256': v.CANONICAL_SHA,
                        'source_pins': v.SOURCE_PINS, 'script_sha256': v.GENERATOR_SHA,
                        'raw_hash_inventory_sha256': 'a' * 64, 'output_root': str(root),
                        'shard': {'index': index, 'count': 4, 'whole_sequence_assignment': mapping,
                                  'target_loads': loads, 'global_canonical_targets': len(records)},
                        'contract': contract(), 'contract_sha256': v.digest(contract()),
                        'target_ids': ids, 'target_count': len(ids),
                        'records': [{'target_id': t} for t in ids]})
    return reports


class NativeAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = v.canonical_members(DATA / 'artifacts/common_targets.json')
        cls.audit = v.pinned_json(DATA / 'archive_data_audit.json', v.AUDIT_SHA)
        cls.expected = v.source_members(cls.records, cls.audit)

    def test_pinned_real_inventory_and_bytes(self):
        self.assertEqual(len(self.expected), 16362)
        source_bytes = sum(r['bytes'] for r in self.expected.values())
        self.assertEqual(source_bytes, 61749620933)
        self.assertEqual(source_bytes,
                         self.audit['raw_required_left_events_bytes'] +
                         sum(r['bytes'] for r in self.audit['phase1_metadata_transfer']['files']) +
                         8152 * (v.ARRAYS['gt'][2] + v.ARRAYS['mask'][2]))
        self.assertEqual(sum(r['known_sha256'] is not None for r in self.expected.values()), 30)
        self.assertFalse(any('/event_tensors/' in p for p in self.expected))
        self.assertEqual(source_bytes + 8152 * v.ARRAYS['input'][2], 161922440389)

    def test_four_shards_exact_cover(self):
        root = Path('/tmp/new_native_test_root')
        reports = shard_reports(self.records, root)
        self.assertEqual(v.validate_shard_membership(reports, self.records, root, 'a' * 64), contract())
        self.assertEqual([r['target_count'] for r in reports], [2045, 2155, 1985, 1967])

    def test_shard_missing_duplicate_and_mixed_runtime_rejected(self):
        root = Path('/tmp/new_native_test_root')
        base = shard_reports(self.records, root)
        cases = []
        x = copy.deepcopy(base); x[0]['records'].pop(); cases.append(x)
        x = copy.deepcopy(base); x[0]['records'][1] = x[0]['records'][0]; cases.append(x)
        x = copy.deepcopy(base); x[1]['shard']['index'] = 0; cases.append(x)
        x = copy.deepcopy(base); x[1]['contract']['torch'] = '2.2.2'; x[1]['contract_sha256'] = v.digest(x[1]['contract']); cases.append(x)
        x = copy.deepcopy(base); x[1]['raw_hash_inventory_sha256'] = 'b' * 64; cases.append(x)
        x = copy.deepcopy(base); x[1]['status'] = 'BOUNDED_SUBSET_COMPLETE'; cases.append(x)
        for i, reports in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                v.validate_shard_membership(reports, self.records, root, 'a' * 64)

    def test_transport_evidence_and_corruption_rejection(self):
        member = {'relative_path': 'source/events.h5', 'bytes': 11, 'sha256': '1' * 64}
        plan = {'source_root': '/source', 'target_root': '/target'}
        core = dict(schema='flow-dsec-batch-v1', **plan, files=[dict(member, sha256=None)])
        msha = v.digest(core); manifest = dict(core, manifest_sha256=msha, batch_id=msha[:24])
        nested = {'schema': 'flow-dsec-source-hashes-v1', 'manifest_sha256': msha, 'files': [member]}
        receipt = dict(schema='flow-dsec-receipt-v1', status='verified', **plan,
                       manifest_sha256=msha, batch_id=msha[:24], file_count=1, total_bytes=11,
                       files=[member], source_hashes=nested, source_hashes_sha256=v.digest(nested))
        inventory = {member['relative_path']: member}
        self.assertEqual(v.validate_transport_receipt(manifest, receipt, plan, inventory), inventory)
        cases = []
        x = copy.deepcopy(receipt); x['source_hashes_sha256'] = '0' * 64; cases.append(x)
        x = copy.deepcopy(receipt); x['files'][0]['sha256'] = '2' * 64; cases.append(x)
        x = copy.deepcopy(receipt); x['target_root'] = '/wrong'; cases.append(x)
        for x in cases:
            with self.assertRaises(ValueError):
                v.validate_transport_receipt(manifest, x, plan, inventory)

    def test_complete_recovery_metadata_and_missing_member_rejection(self):
        # No large files needed: verifies exact source evidence membership;
        # actual bytes are separately rehashed by verify_file, tested below.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); root = directory / 'DSEC_native_test'
            members = [dict(relative_path=p, bytes=r['bytes'], sha256=r['known_sha256'] or v.digest(p))
                       for p, r in self.expected.items()]
            plan = {'schema': 'flow_a_raw_recovery_v1', 'canonical_sha256': v.CANONICAL_SHA,
                    'audit_sha256': v.AUDIT_SHA, 'source_root': self.audit['remote_root'],
                    'target_root': str(root), 'old_root': str(directory / 'old'), 'old_voxels_included': False,
                    'source_csv_sha256': self.audit['transfer_manifest']['sha256'], 'files': members,
                    'file_count': 16362, 'total_bytes': 61749620933,
                    'raw_events_bytes': self.audit['raw_required_left_events_bytes']}
            status = {'status': v.READY_STATUS, 'failures': [], 'results': [], 'linked_files': 16362,
                      'transferred_verified_files': 0, 'transferred_verified_bytes': 0}
            v.atomic_json(directory / 'recovery_plan.json', plan)
            v.atomic_json(directory / 'recovery_status.json', status)
            v.atomic_json(directory / 'verified_local_links.json', {'files': members, 'count': 16362, 'bytes': 61749620933})
            def write_inventory(rows):
                with (directory / 'source_hashes.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['relative_path', 'bytes', 'sha256']); writer.writeheader(); writer.writerows(rows)
            write_inventory(members)
            inventory, evidence = v.validate_recovery(root, directory, self.expected, self.audit)
            self.assertEqual(len(inventory), 16362)
            self.assertEqual(evidence['source_file_count'], 16362)
            write_inventory(members[:-1])
            with self.assertRaisesRegex(ValueError, 'exact membership'):
                v.validate_recovery(root, directory, self.expected, self.audit)

    def test_real_npy_full_sha_header_finite_and_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'input.npy'
            array = np.zeros(v.ARRAYS['input'][0], dtype=np.float32); array[0, 0, 1] = 3.5
            np.save(path, array, allow_pickle=False)
            row = {'relative_path': path.name, 'category': 'input', 'bytes': path.stat().st_size,
                   'sha256': v.sha(path), 'tensor_sha256': hashlib.sha256(array.tobytes()).hexdigest()}
            result = v.verify_file(root, row)
            self.assertTrue(result['passed'], result)
            self.assertTrue(result['all_elements_finite'])
            with path.open('r+b') as f:
                f.seek(-1, 2); f.write(b'\x01')
            self.assertFalse(v.verify_file(root, row)['passed'])

    def test_nonfinite_rejected_even_when_file_sha_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'gt.npy'
            array = np.zeros(v.ARRAYS['gt'][0], dtype=np.float32); array[0, 0, 0] = np.nan
            np.save(path, array, allow_pickle=False)
            result = v.verify_file(root, {'relative_path': path.name, 'category': 'gt',
                                         'bytes': path.stat().st_size, 'sha256': v.sha(path)})
            self.assertFalse(result['passed']); self.assertIn('nonfinite', result['error'])

    def test_wrong_dtype_and_payload_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'gt.npy'
            np.save(path, np.zeros(v.ARRAYS['gt'][0], dtype=np.int32), allow_pickle=False)
            row = {'relative_path': path.name, 'category': 'gt', 'bytes': path.stat().st_size, 'sha256': v.sha(path)}
            self.assertIn('header', v.verify_file(root, row)['error'])
            np.save(path, np.zeros(v.ARRAYS['gt'][0], dtype=np.float32), allow_pickle=False)
            row.update(sha256=v.sha(path), tensor_sha256='f' * 64)
            self.assertIn('Tensor SHA', v.verify_file(root, row)['error'])

    def test_symlink_and_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'real').write_bytes(b'content'); (root / 'link').symlink_to('real')
            row = {'relative_path': 'link', 'category': 'raw_left_events', 'bytes': 7, 'sha256': v.sha(root / 'real')}
            self.assertIn('Symlink', v.verify_file(root, row)['error'])
            for bad in ('../x', '/absolute', 'a//b', 'a/./b'):
                with self.subTest(path=bad), self.assertRaises(ValueError): v.safe_relative(bad)

    def test_acceptance_never_published_for_subset_or_failed_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'acceptance'
            with self.assertRaises(ValueError):
                v.publish_outputs(output, {}, [], contract(), {}, Path(tmp))
            rows = {str(i): {'category': 'input' if i < 8152 else 'gt' if i < 16304 else 'mask' if i < 24456 else 'raw'}
                    for i in range(24514)}
            results = [{'relative_path': str(i), 'passed': i != 12} for i in range(24514)]
            with self.assertRaises(ValueError):
                v.publish_outputs(output, rows, results, contract(), {}, Path(tmp))
            self.assertFalse((output / 'acceptance.json').exists())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeAcceptanceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {'schema': 'flow-native-validator-cpu-tests-v1', 'tests_run': result.testsRun,
               'passed': result.wasSuccessful(), 'failures': len(result.failures), 'errors': len(result.errors),
               'validator_sha256': v.sha(HERE / 'verify_native_dataset.py'),
               'test_sha256': v.sha(__file__), 'scope': 'Local synthetic rejection tests plus pinned canonical/archive inventory arithmetic; no remote full dataset acceptance.',
               'GPU_used': False}
    v.atomic_json(HERE / 'native_validator_cpu_tests.json', summary)
    raise SystemExit(0 if result.wasSuccessful() else 1)
