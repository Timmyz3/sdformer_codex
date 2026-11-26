"""Stdlib-only receipt/timestamp boundary checks; no arrays, model or GPU load.

The fixture has the real DSEC '# from_timestamp_us, to_timestamp_us' header and
three distinct data intervals. Its second DATA row uses the
already documented 100003us example, not a nominal100000us interval. Receipt
contents and expected answers are literal, not produced by the function under
test or by another implementation of its row-selection algorithm.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import collect_coarse_cache as collector


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TimestampBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='flow_coarse_timestamp_cpu_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.row = {'target_id': 'zurich_fixture_0002', 'sequence': 'zurich_fixture',
                    'target_index_1based': 2, 'protocol_split': 'train'}
        self.input_rel = 'saved_flow_data/train/zurich_fixture/input/zurich_fixture_0002.npy'
        self.ts_rel = 'train_optical_flow/zurich_fixture/flow/forward_timestamps.txt'
        self.ts_path = self.root/self.ts_rel
        self.ts_path.parent.mkdir(parents=True)
        self.ts_path.write_text('# from_timestamp_us, to_timestamp_us\n1000000, 1099990\n55617507634, 55617607637\n90000000000, 90000123457\n')
        self.receipt_path = self.root/'.native_encoding_receipts/zurich_fixture_0002.json'
        self.receipt_path.parent.mkdir()
        # No NPY is needed here: timestamp_binding receives the separately accepted input digest.
        self.input_sha = hashlib.sha256(b'accepted-native-input-fixture').hexdigest()
        self.encoding_sha = hashlib.sha256(b'accepted-native-encoding-contract-fixture').hexdigest()
        self.paths = {'input': self.input_rel}
        self.hashes = {self.input_rel: {'sha256': self.input_sha}}
        self.acceptance = {'encoding_contract_sha256': self.encoding_sha}
        self.receipt = {'schema_version': 'flow_native_input_receipt_v1',
            'target_id': 'zurich_fixture_0002', 'relative_path': self.input_rel,
            'sha256': self.input_sha, 'contract_sha256': self.encoding_sha,
            'timestamp_index_1based': 2, 'timestamp_pair_us': [55617507634, 55617607637],
            'source_hashes': {self.ts_rel: {'sha256': file_sha(self.ts_path)}}}
        self.inventory = {self.input_rel: {'category': 'input', 'sha256': self.input_sha},
                          self.ts_rel: {'category': 'timestamps', 'sha256': file_sha(self.ts_path)}}
        self.write_receipt(reseal=True)

    def write_receipt(self, *, reseal=False):
        self.receipt_path.write_text(json.dumps(self.receipt, sort_keys=True)+'\n')
        if reseal:
            self.inventory[self.input_rel]['generation_receipt_sha256'] = file_sha(self.receipt_path)

    def call(self, **overrides):
        args = dict(root=self.root, row=self.row, paths=self.paths, hashes=self.hashes,
                    inventory=self.inventory, acceptance=self.acceptance)
        args.update(overrides)
        return collector.timestamp_binding(**args)

    def test_accepted_second_row_preserves_actual_non_nominal_interval(self):
        result = self.call()
        self.assertEqual(result['timestamp_pair_us'], [55617507634, 55617607637])
        self.assertEqual(result['interval_us'], 100003)
        self.assertNotEqual(result['interval_us'], 100000)
        self.assertEqual(result['timestamp_index_1based'], 2)
        self.assertEqual(result['timestamp_source_sha256'], self.inventory[self.ts_rel]['sha256'])
        self.assertEqual(result['generation_receipt_sha256'], self.inventory[self.input_rel]['generation_receipt_sha256'])
        self.assertIs(result['per_target_interval_verified'], True)

    def test_receipt_content_changed_without_acceptance_is_rejected(self):
        self.receipt['timestamp_pair_us'][1] += 1
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, 'Receipt SHA differs'):
            self.call()

    def test_comments_blank_and_inline_comments_do_not_shift_target_index(self):
        self.ts_path.write_text('# from_timestamp_us, to_timestamp_us\n\n# sequence annotation\n'
                                '1000000, 1099990 # first target\n \t \n'
                                '55617507634, 55617607637 # second target\n'
                                '# between data rows\n90000000000, 90000123457\n\n')
        new_sha = file_sha(self.ts_path)
        self.inventory[self.ts_rel]['sha256'] = new_sha
        self.receipt['source_hashes'][self.ts_rel]['sha256'] = new_sha
        self.write_receipt(reseal=True)
        result = self.call()
        self.assertEqual(result['timestamp_index_1based'], 2)
        self.assertEqual(result['timestamp_pair_us'], [55617507634, 55617607637])
        self.assertEqual(result['interval_us'], 100003)

    def test_timestamp_file_changed_without_acceptance_is_rejected(self):
        # Even an unchanged selected row cannot excuse tampering elsewhere in the source file.
        self.ts_path.write_text('# from_timestamp_us, to_timestamp_us\n1000000, 1099991\n55617507634, 55617607637\n90000000000, 90000123457\n')
        with self.assertRaisesRegex(ValueError, 'Timestamp source differs'):
            self.call()

    def test_canonical_index_must_match_receipt(self):
        changed = dict(self.row, target_index_1based=1)
        with self.assertRaisesRegex(ValueError, 'Receipt binding differs'):
            self.call(row=changed)

    def test_resealed_receipt_cannot_substitute_neighbor_timestamp_pair(self):
        # Receipt hash is internally valid, but its purported row2 pair actually came from row3.
        self.receipt['timestamp_pair_us'] = [90000000000, 90000123457]
        self.write_receipt(reseal=True)
        with self.assertRaisesRegex(ValueError, 'Timestamp values differ'):
            self.call()

    def test_wrong_input_or_encoding_acceptance_is_rejected(self):
        altered_hashes = copy.deepcopy(self.hashes)
        altered_hashes[self.input_rel]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Receipt binding differs'):
            self.call(hashes=altered_hashes)
        with self.assertRaisesRegex(ValueError, 'Receipt binding differs'):
            self.call(acceptance={'encoding_contract_sha256': '1'*64})
        inventory = copy.deepcopy(self.inventory)
        inventory[self.input_rel]['sha256'] = '2'*64
        with self.assertRaisesRegex(ValueError, 'Receipt binding differs'):
            self.call(inventory=inventory)

    def test_timestamp_receipt_and_inventory_need_same_accepted_source(self):
        self.receipt['source_hashes'][self.ts_rel]['sha256'] = '3'*64
        self.write_receipt(reseal=True)
        with self.assertRaisesRegex(ValueError, 'Timestamp source differs'):
            self.call()

    def test_even_resealed_nonincreasing_interval_is_rejected(self):
        self.ts_path.write_text('# from_timestamp_us, to_timestamp_us\n1000000, 1099990\n55617507634, 55617507634\n90000000000, 90000123457\n')
        new_sha = file_sha(self.ts_path)
        self.inventory[self.ts_rel]['sha256'] = new_sha
        self.receipt['source_hashes'][self.ts_rel]['sha256'] = new_sha
        self.receipt['timestamp_pair_us'] = [55617507634, 55617507634]
        self.write_receipt(reseal=True)
        with self.assertRaisesRegex(ValueError, 'Timestamp values differ'):
            self.call()


if __name__ == '__main__':
    unittest.main(verbosity=2)
