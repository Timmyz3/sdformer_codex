#!/usr/bin/env python3
"""Behavioral and adversarial CPU checks for the reduction address component."""
import copy
import json
from pathlib import Path
import unittest
import address_model as m


class AddressTests(unittest.TestCase):
    def test_odd_even_unit_and_multichannel_integer_golden(self):
        for n in (1, 2, 3, 4, 5, 7, 8, 9, 17):
            for channels in (1, 2, 3, 8, 16):
                result = m.reduce_group(n, channels)
                self.assertEqual(result['output'], result['expected'])
                self.assertEqual(m.validate_trace(result)['status'], 'PASS_INDEPENDENT_TRANSACTION_REPLAY')

    def test_odd_carries_are_explicit_and_compacted(self):
        result = m.reduce_group(5, 3)
        # n=5->3->2->1 has two carry levels, each with all three channels.
        self.assertEqual(result['transaction_counts']['copy'], 6)
        self.assertEqual(result['transaction_counts']['add'], 12)
        carries = [job for job in result['jobs'] if job['kind'] == 'carry']
        self.assertEqual(carries[0]['src'], [12])
        self.assertEqual(carries[0]['dst'], 4096 + 6)
        self.assertEqual(carries[3]['src'], [4096 + 6])
        self.assertEqual(carries[3]['dst'], 3)
        for job in carries:
            events = [e for e in result['trace'] if e['operation'] == job['id']]
            self.assertEqual([e['kind'] for e in events], ['read', 'copy', 'write'])

    def test_collision_backpressure_is_measured_and_more_ports_help(self):
        slow = m.default_config(); slow.update(banks=1, shared_read_write=True)
        fast = dict(slow, banks=16, shared_read_write=False)
        a = m.reduce_group(17, 8, slow); b = m.reduce_group(17, 8, fast)
        self.assertEqual(a['output'], b['output'])
        self.assertGreater(a['synthetic_component_cycles'], b['synthetic_component_cycles'])
        self.assertGreater(a['bank_conflict_requests']['shared_bank_port_requests_denied'], 0)
        m.validate_trace(a); m.validate_trace(b)

    def test_finite_capacity_and_queue(self):
        cfg = m.default_config(); cfg.update(max_active_operations=1, register_words=3)
        a = m.reduce_group(9, 3, cfg); m.validate_trace(a)
        self.assertEqual(a['peak_active_operations'], 1)
        for bad in (dict(cfg, local_capacity_words=8191), dict(cfg, register_words=2)):
            with self.assertRaises(ValueError): m.reduce_group(5, 2, bad)
        with self.assertRaises(ValueError): m.reduce_group(4097, 1, cfg)

    def test_no_input_config_mutation(self):
        cfg = m.default_config(); before = copy.deepcopy(cfg)
        values = [[-8, 7], [3, -4], [5, 9]]; saved = copy.deepcopy(values)
        result = m.reduce_group(3, 2, cfg, values=values)
        self.assertEqual(result['output'], [0, 12]); m.validate_trace(result)
        self.assertEqual(cfg, before); self.assertEqual(values, saved)

    def test_replay_rejects_corrupted_carry_destination_and_value(self):
        result = m.reduce_group(5, 3)
        carry = next(j for j in result['jobs'] if j['kind'] == 'carry')
        bad = copy.deepcopy(result)
        write = next(e for e in bad['trace'] if e['operation'] == carry['id'] and e['kind'] == 'write')
        write['address'] -= 1
        with self.assertRaises(ValueError): m.validate_trace(bad)
        bad = copy.deepcopy(result)
        copy_event = next(e for e in bad['trace'] if e['kind'] == 'copy')
        copy_event['data'][0] += 1
        with self.assertRaises(ValueError): m.validate_trace(bad)

    def test_replay_rejects_early_write_and_false_resource_claims(self):
        result = m.reduce_group(5, 3)
        bad = copy.deepcopy(result)
        write = next(e for e in bad['trace'] if e['kind'] == 'write' and e['phase'] == 'reduce_0')
        write['issue'] = 0; write['retire'] = 1
        with self.assertRaises(ValueError): m.validate_trace(bad)
        bad = copy.deepcopy(result); bad['parameters']['register_words'] = 3
        with self.assertRaises(ValueError): m.validate_trace(bad)
        bad = copy.deepcopy(result); bad['parameters']['banks'] = 1
        with self.assertRaises(ValueError): m.validate_trace(bad)

    def test_actual_ledger_source_and_layout_endpoints(self):
        self.assertEqual(m.sha(m.SOURCE), m.SOURCE_SHA)
        inventory = json.loads(m.SOURCE.read_text())['external_BN']
        self.assertEqual(len(inventory['records']), 54)
        layouts = {}
        for record in inventory['records']:
            layout, group = m.observed_layout(record)
            layouts[layout] = layouts.get(layout, 0) + 1
            end = m.source_address(record, record['reduction_elements_per_channel'] - 1, record['channels'] - 1)
            self.assertEqual(end, record['elements'] - 1)
            self.assertEqual(m.source_address(record, 0, 0), 0)
        self.assertEqual(layouts, {'native_channel_major': 15, 'native_channel_contiguous': 39})


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AddressTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = dict(schema='flow_bn_address_cpu_checks_v1', tests_run=result.testsRun,
                   failures=len(result.failures), errors=len(result.errors), status='PASS' if result.wasSuccessful() else 'FAIL',
                   source_sha256={str(Path(__file__).resolve()):m.sha(Path(__file__)), str(Path(m.__file__).resolve()):m.sha(m.__file__)},
                   native_FP32_equivalence_validated=False, GPU_or_full_model_execution=False)
    output = Path(__file__).parent / 'cpu_checks.json'
    if output.exists(): raise RuntimeError('Preserve existing CPU receipt; use unittest module for rechecks')
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
