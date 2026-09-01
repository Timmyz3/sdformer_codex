#!/usr/bin/env python3
"""Payload-free dual-runtime attacks for the M1628 source successor."""
from __future__ import print_function

import copy
import importlib.util
import json
from pathlib import Path
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "scripts" /
          "build_m1628_decoder_compact_l2_retained_ledger_successor_source.py")


def load_source():
    spec = importlib.util.spec_from_file_location("m1628_source_test",
                                                  str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class M1628Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_source()

    def rejects(self, action):
        with self.assertRaises(self.m.M1628Error):
            action()

    def request(self, miter, kind, output_block, mutator=None):
        m = self.m
        ordinal = miter.next_request_ordinal
        row = m.synthetic_request(miter.configuration, ordinal, kind)
        row["destination"] = miter.next_destination
        row["output_block"] = output_block
        if kind in ("weight_read", "weight_write"):
            row["width_bytes"] = 16
            row["addresses"] = [miter.next_destination % 128]
            row["banks"] = [miter.next_destination % 8]
        elif kind in ("psum_read", "psum_write"):
            row["width_bytes"] = 48
            row["addresses"] = [output_block * 48]
            row["banks"] = [output_block % 6]
        row["port_ready_cycle"] = max(
            miter.derived_port_calendar[m.port_index(kind, bank)]
            for bank in row["banks"])
        row["issue_cycle"] = max(row["earliest_issue_cycle"],
                                 row["dependency_ready_cycle"],
                                 row["port_ready_cycle"])
        row["return_cycle"] = (row["issue_cycle"] +
                               m.B.C.latency_for(m.B.C.kind_index(kind)) +
                               row["beats"] - 1)
        if mutator is not None:
            mutator(row)
        miter.accept_request_pair(row, copy.deepcopy(row))
        return row

    def destination(self, miter, dense=None, last_cycle=None, mutator=None):
        if dense is None:
            dense = miter.configuration == self.m.CONFIGS[0]
        row = self.m.synthetic_state(miter, dense, last_cycle)
        if mutator is not None:
            mutator(row)
        miter.accept_destination_pair(row, copy.deepcopy(row))
        return row

    def populate_destination(self, miter, include_weight=True,
                             include_psum=False):
        if include_weight:
            self.request(miter, "weight_read", 0)
        if include_psum:
            self.request(miter, "psum_write", 0)
        for block in range(self.m.OUTPUT_BLOCKS):
            self.request(miter, "commit", block)

    def completed_miter(self, configuration, address_bias=0):
        m = self.m
        miter = m.CanonicalPrefixMiter(configuration)
        for _destination in range(m.PREFIX_DESTINATIONS):
            self.request(miter, "weight_read", 0)
            for block in range(m.OUTPUT_BLOCKS):
                if address_bias:
                    self.request(
                        miter, "commit", block,
                        lambda row, bias=address_bias:
                            row.update({"addresses": [value + bias
                                                      for value in
                                                      row["addresses"]]}))
                else:
                    self.request(miter, "commit", block)
            self.destination(miter)
        return miter

    def bundle(self, address_biases=(0, 0, 0)):
        return [self.completed_miter(configuration, bias).finish()
                for configuration, bias in zip(self.m.CONFIGS,
                                                address_biases)]

    def test_01_describe_and_static_self_test_are_source_only(self):
        description = self.m.describe()
        self.assertTrue(description["authorization"]["source_only"])
        for field in ("actual_payload", "l2_execution", "l3", "pilot",
                      "production", "cycles", "traffic", "energy",
                      "speedup", "rtl", "eda", "paper_result"):
            self.assertFalse(description["authorization"][field])
        self.assertFalse(description["future_gate"]["review_present"])
        self.assertFalse(description["future_gate"]["release_present"])
        result = self.m.static_self_test()
        self.assertEqual(result["status"],
                         "PASS_M1628_THREE_P1_SOURCE_REPAIR_STATIC_ONLY")
        self.assertEqual(len(result["sessions"]), 3)

    def test_02_all_request_exact_fields_remain_pairwise(self):
        base = self.m.synthetic_request(self.m.CONFIGS[0], 0)
        for field in self.m.REQUEST_FIELDS:
            reference = copy.deepcopy(base)
            compact = copy.deepcopy(base)
            if field == "configuration":
                compact[field] = self.m.CONFIGS[1]
            elif field in ("kind",):
                compact[field] = "external_read"
            elif field in ("addresses", "banks"):
                compact[field] = [reference[field][0] + 1]
            elif field == "packed_event_sha256":
                compact[field] = "f" * 64
            elif isinstance(compact[field], int):
                compact[field] += 1
            else:
                raise AssertionError(field)
            self.rejects(lambda reference=reference, compact=compact:
                         self.m.CanonicalPrefixMiter(
                             self.m.CONFIGS[0]).accept_request_pair(
                                 reference, compact))

    def test_03_survivor_earlier_max_return_cannot_be_lost(self):
        miter = self.m.CanonicalPrefixMiter(self.m.CONFIGS[0])
        for block in range(self.m.OUTPUT_BLOCKS):
            self.request(miter, "commit", block,
                         (lambda row: row.update({"return_cycle": 5000}))
                         if block == 0 else None)
        state = self.m.synthetic_state(miter, True, last_cycle=100)
        self.assertIn(5000, state["outstanding_active_returns"][14])
        state["outstanding_active_returns"] = [[] for _index in range(16)]
        self.rejects(lambda: miter.accept_destination_pair(
            state, copy.deepcopy(state)))

    def test_04_survivor_future_return_cannot_disappear_next_destination(self):
        miter = self.m.CanonicalPrefixMiter(self.m.CONFIGS[0])
        for block in range(self.m.OUTPUT_BLOCKS):
            self.request(miter, "commit", block,
                         (lambda row: row.update({"return_cycle": 5000}))
                         if block == 0 else None)
        self.destination(miter, last_cycle=100)
        for block in range(self.m.OUTPUT_BLOCKS):
            self.request(miter, "commit", block)
        second = self.m.synthetic_state(miter, True, last_cycle=200)
        self.assertIn(5000, second["outstanding_active_returns"][14])
        second["outstanding_active_returns"][14] = []
        self.rejects(lambda: miter.accept_destination_pair(
            second, copy.deepcopy(second)))

    def test_05_survivor_last_psum_write_ready_cannot_move_backward(self):
        miter = self.m.CanonicalPrefixMiter(self.m.CONFIGS[0])
        self.populate_destination(miter, include_psum=True)
        first = self.destination(miter)
        ready = first["numeric_dependency_state"][
            "last_psum_write_ready"][0]
        self.assertGreater(ready, 0)
        self.populate_destination(miter)
        second = self.m.synthetic_state(miter, True)
        second["numeric_dependency_state"]["last_psum_write_ready"][0] = 0
        self.rejects(lambda: miter.accept_destination_pair(
            second, copy.deepcopy(second)))

    def test_06_survivor_cache_clear_and_content_reset_are_rejected(self):
        miter = self.m.CanonicalPrefixMiter(self.m.CONFIGS[0])
        self.populate_destination(miter)
        first = self.destination(miter)
        self.populate_destination(miter)
        cleared = self.m.synthetic_state(miter, True)
        cleared["cache"]["valid_entries"] = 0
        self.rejects(lambda: miter.accept_destination_pair(
            cleared, copy.deepcopy(cleared)))
        unchanged = self.m.synthetic_state(miter, True)
        for field in ("tick", "hits", "misses", "evictions"):
            unchanged["cache"][field] = first["cache"][field]
        self.rejects(lambda: miter.accept_destination_pair(
            unchanged, copy.deepcopy(unchanged)))

    def test_07_new_cache_transition_requires_request_and_predecessor(self):
        miter = self.m.CanonicalPrefixMiter(self.m.CONFIGS[0])
        self.populate_destination(miter)
        self.destination(miter)
        self.populate_destination(miter)
        bad_previous = self.m.synthetic_state(miter, True)
        bad_previous["cache"]["previous_state_sha256"] = "0" * 64
        self.rejects(lambda: miter.accept_destination_pair(
            bad_previous, copy.deepcopy(bad_previous)))
        bad_request = self.m.synthetic_state(miter, True)
        bad_request["cache"]["accepted_weight_request_sha256"] = "1" * 64
        self.rejects(lambda: miter.accept_destination_pair(
            bad_request, copy.deepcopy(bad_request)))

    def test_08_survivor_wrong_request_scope_rejected(self):
        mutations = (("module", 99), ("timestep", 99),
                     ("destination", 41), ("output_block", 99))
        for field, value in mutations:
            miter = self.m.CanonicalPrefixMiter(self.m.CONFIGS[0])
            self.rejects(lambda miter=miter, field=field, value=value:
                         self.request(miter, "commit", 0,
                                      lambda row: row.update({field: value})))

    def test_09_survivor_kind_and_byte_ledgers_cannot_be_reported(self):
        miter = self.m.CanonicalPrefixMiter(self.m.CONFIGS[0])
        self.populate_destination(miter)
        bad = self.m.synthetic_state(miter, True)
        bad["kind_counts"] = {"external_read": bad["request_count"]}
        bad["byte_counts"] = {"external_read": 0}
        self.rejects(lambda: miter.accept_destination_pair(
            bad, copy.deepcopy(bad)))

    def test_10_survivor_address_and_commit_digests_cannot_be_reported(self):
        for field, value in (("packed_transaction_address_sha256", "1" * 64),
                             ("packed_commit_sequence_sha256", "2" * 64)):
            miter = self.m.CanonicalPrefixMiter(self.m.CONFIGS[0])
            self.populate_destination(miter)
            bad = self.m.synthetic_state(miter, True)
            bad[field] = value
            self.rejects(lambda miter=miter, bad=bad:
                         miter.accept_destination_pair(bad,
                                                       copy.deepcopy(bad)))

    def test_11_new_port_calendar_and_bank_provenance_are_derived(self):
        miter = self.m.CanonicalPrefixMiter(self.m.CONFIGS[0])
        self.rejects(lambda: self.request(
            miter, "commit", 0,
            lambda row: row.update({"port_ready_cycle": 99,
                                    "issue_cycle": 99,
                                    "return_cycle": 103})))
        miter = self.m.CanonicalPrefixMiter(self.m.CONFIGS[0])
        self.rejects(lambda: self.request(
            miter, "commit", 0,
            lambda row: row.update({"addresses": row["addresses"] * 2,
                                    "banks": [0, 0]})))

    def test_12_survivor_forged_finish_rows_rejected(self):
        forged = [{"configuration": configuration,
                   "destinations": self.m.PREFIX_DESTINATIONS,
                   "requests": 0,
                   "commits": self.m.EXPECTED_COMMITS_PER_CONFIG,
                   "dense_cache_covered": False,
                   "dense_psum_1rw_covered": False,
                   "final_commit_digest": "f" * 64}
                  for configuration in self.m.CONFIGS]
        self.rejects(lambda: self.m.validate_bundle(forged))

    def test_13_finish_is_incomplete_protected_and_one_shot(self):
        incomplete = self.m.CanonicalPrefixMiter(self.m.CONFIGS[0])
        self.rejects(incomplete.finish)
        self.rejects(lambda: self.m._finish_session(
            incomplete, {"session_identity": incomplete.session_identity}))
        self.rejects(lambda: self.m._new_session(object()))
        miter = self.completed_miter(self.m.CONFIGS[0])
        receipt = miter.finish()
        self.rejects(miter.finish)
        self.rejects(lambda: setattr(receipt, "_tag", "0" * 64))

    def test_14_finish_clone_and_tag_mutation_rejected(self):
        rows = self.bundle()
        original = rows[0]
        clone = object.__new__(type(original))
        object.__setattr__(clone, "_payload_json", original._payload_json)
        object.__setattr__(clone, "_tag", original._tag)
        object.__setattr__(clone, "_locked", True)
        cloned = [clone, rows[1], rows[2]]
        self.rejects(lambda: self.m.validate_bundle(cloned))
        object.__setattr__(original, "_tag", "0" * 64)
        self.rejects(lambda: self.m.validate_bundle(rows))

    def test_15_three_distinct_sessions_and_exact_order_required(self):
        rows = self.bundle()
        self.rejects(lambda: self.m.validate_bundle(
            [rows[0], rows[0], rows[2]]))
        self.rejects(lambda: self.m.validate_bundle(
            [rows[1], rows[0], rows[2]]))

    def test_16_shared_commit_stream_is_genuine_and_required(self):
        rows = self.bundle((0, 0, 4096))
        self.rejects(lambda: self.m.validate_bundle(rows))

    def test_17_valid_bundle_is_consumed_and_replay_rejected(self):
        rows = self.bundle()
        self.assertTrue(self.m.validate_bundle(rows))
        self.rejects(lambda: self.m.validate_bundle(rows))

    def test_18_actual_release_payload_and_claims_stay_closed(self):
        self.rejects(lambda: self.m.actual_prefix_release(
            lambda: True, object()))
        source = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("np.load", "numpy.load", "torch.load", ".npz",
                          ".tar.zst",
                          "m1458_m1434_motion_ep34_live93_unified_hardware_capture"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
