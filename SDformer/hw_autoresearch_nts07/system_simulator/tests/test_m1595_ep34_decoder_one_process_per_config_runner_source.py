#!/usr/bin/env python3
"""Source-only tests for M1595; the actual child entry is never invoked."""
from __future__ import print_function

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/run_m1595_ep34_decoder_one_process_per_config_runner_source.py"
SPEC = importlib.util.spec_from_file_location("m1595_source_test", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def clean_row(config):
    return {
        "configuration": config,
        "resource_manifest_sha256": M.RESOURCE_SHA256,
        "total_cycles": 101,
        "request_count": 3,
        "kind_counts": {"compute": 2, "commit": 1},
        "byte_counts": {"compute": 288, "commit": 3},
        "transaction_address_sha256": "a" * 64,
        "commit_sequence_sha256": "b" * 64,
        "streaming": {"materialized_transaction_list": False,
                      "destinations": 7, "timesteps": 10},
        "schema": "upstream",
        "pilot_call_ordinal": 0,
        "module_ordinal": 0,
        "timesteps": 10,
        "diagnostic_only": True,
        "paper_result": False,
        "product_capture": False,
        "production": False,
        "payload_fd_sha256": "c" * 64,
        "payload_fd_size": 4096,
        "m1573_rss": {"gate_calls": 2,
                      "baseline_current_rss_kib": 100,
                      "baseline_peak_rss_kib": 120,
                      "max_current_rss_kib": 130,
                      "max_peak_rss_kib": 140,
                      "absolute_limit_kib": M.RSS_LIMIT_KIB,
                      "fresh_exec_required": True},
        "fresh_exec_required": True,
    }


def layout(root, name):
    root = Path(root)
    return M.Layout(root / (name + ".result"), root / (name + ".attempt"),
                    root / (name + ".work"), root / (name + ".failure"),
                    root / (name + ".lock"))


def envelope(config, target, parent_pid, nonce, child_pid):
    row = clean_row(config)
    ticket = M.child_ticket(nonce, config, target, parent_pid)
    return {"schema": "m1595_ep34_decoder_child_result_r1_v1",
            "configuration": config, "parent_pid": parent_pid,
            "child_pid": child_pid, "ticket_sha256": ticket,
            "m1583_source_sha256": M.ENGINE_SHA256,
            "result_sha256": M.canonical_sha(row), "result": row}


class M1595Tests(unittest.TestCase):
    def test_01_description_and_authority_are_source_only(self):
        value = M.describe()
        self.assertEqual(value["population"]["configurations"], list(M.CONFIGS))
        self.assertEqual(value["population"]["decoder_stage"], "D0")
        self.assertEqual(value["population"]["call_ordinal"], 0)
        self.assertTrue(value["process_model"]["fresh_python_per_configuration"])
        self.assertTrue(value["attempt"]["failure_permanent"])
        self.assertFalse(value["execution"]["actual"])
        self.assertFalse(value["claim_boundary"]["cycles"])
        authority = M.verify_authorities()
        self.assertEqual(authority["m1583_source_sha256"], M.ENGINE_SHA256)

    def test_02_preflight_is_read_only_and_collision_closed(self):
        with tempfile.TemporaryDirectory(prefix="m1595_preflight.") as root:
            candidate = layout(root, "clean")
            value = M.preflight(candidate)
            self.assertFalse(value["attempt_consumed"])
            self.assertEqual(value["child_processes"], 0)
            self.assertFalse(candidate.attempt.exists())
            candidate.result.mkdir()
            with self.assertRaises(M.M1595Error):
                M.preflight(candidate)

    def test_03_three_config_success_uses_distinct_synthetic_children(self):
        with tempfile.TemporaryDirectory(prefix="m1595_success.") as root:
            candidate = layout(root, "success")
            calls = []

            def fake_launcher(config, target, parent_pid, nonce):
                ordinal = len(calls)
                item = envelope(config, target, parent_pid, nonce,
                                parent_pid + 100 + ordinal)
                M.write_new(target, item)
                calls.append((config, parent_pid, item["child_pid"],
                              item["ticket_sha256"]))
                return item

            result = M.execute_controlled(candidate, fake_launcher)
            self.assertEqual([row[0] for row in calls], list(M.CONFIGS))
            self.assertEqual(len(set(row[2] for row in calls)), 3)
            self.assertEqual(len(set(row[3] for row in calls)), 3)
            self.assertEqual(result["population"]["fresh_child_processes"], 3)
            self.assertEqual([row["configuration"] for row in result["results"]],
                             list(M.CONFIGS))
            self.assertTrue(candidate.attempt.is_file())
            self.assertTrue(candidate.result.is_dir())
            self.assertFalse(candidate.work.exists())
            outer = (candidate.result / M.OUTER).read_text().split()
            self.assertEqual(outer,
                             [M.sha256(candidate.result / M.MANIFEST), M.MANIFEST])

    def test_04_failure_is_permanently_consumed_before_retry(self):
        with tempfile.TemporaryDirectory(prefix="m1595_failure.") as root:
            candidate = layout(root, "failure")
            calls = []

            def failing_launcher(config, target, parent_pid, nonce):
                calls.append(config)
                if len(calls) == 2:
                    raise M.M1595Error("synthetic child failure")
                item = envelope(config, target, parent_pid, nonce,
                                parent_pid + 200 + len(calls))
                M.write_new(target, item)
                return item

            with self.assertRaises(M.M1595Error):
                M.execute_controlled(candidate, failing_launcher)
            self.assertEqual(calls, list(M.CONFIGS[:2]))
            self.assertTrue(candidate.attempt.is_file())
            self.assertTrue(candidate.failure.is_dir())
            self.assertFalse(candidate.result.exists())
            self.assertFalse(candidate.work.exists())
            with self.assertRaises(M.M1595Error):
                M.execute_controlled(candidate, failing_launcher)
            self.assertEqual(calls, list(M.CONFIGS[:2]))

    def test_05_child_envelope_identity_and_result_mutations_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m1595_envelope.") as root:
            target = Path(root) / "child.json"
            parent_pid = 1234; nonce = "1" * 64
            item = envelope(M.CONFIGS[0], target, parent_pid, nonce, 5678)
            ticket = M.child_ticket(nonce, M.CONFIGS[0], target, parent_pid)
            self.assertEqual(M.verify_child_envelope(
                M.CONFIGS[0], item, parent_pid, ticket)["request_count"], 3)
            attacks = []
            for key, value in (("configuration", M.CONFIGS[1]),
                               ("parent_pid", parent_pid + 1),
                               ("child_pid", parent_pid),
                               ("ticket_sha256", "0" * 64),
                               ("m1583_source_sha256", "0" * 64),
                               ("result_sha256", "0" * 64)):
                bad = dict(item); bad[key] = value
                with self.assertRaises(M.M1595Error, msg=key):
                    M.verify_child_envelope(M.CONFIGS[0], bad, parent_pid, ticket)
                attacks.append(key)
            bad = json.loads(json.dumps(item))
            bad["result"]["kind_counts"]["commit"] = 2
            bad["result_sha256"] = M.canonical_sha(bad["result"])
            with self.assertRaises(M.M.M1583Error):
                M.verify_child_envelope(M.CONFIGS[0], bad, parent_pid, ticket)
            attacks.append("request_conservation")
            bad = json.loads(json.dumps(item))
            bad["result"]["m1573_rss"]["gate_calls"] = 0
            bad["result_sha256"] = M.canonical_sha(bad["result"])
            with self.assertRaises(M.M.M1583Error):
                M.verify_child_envelope(M.CONFIGS[0], bad, parent_pid, ticket)
            attacks.append("gate_calls")
            bad = json.loads(json.dumps(item))
            bad["result"]["m1573_rss"]["max_peak_rss_kib"] = M.RSS_LIMIT_KIB
            bad["result_sha256"] = M.canonical_sha(bad["result"])
            with self.assertRaises(M.M.M1583Error):
                M.verify_child_envelope(M.CONFIGS[0], bad, parent_pid, ticket)
            attacks.append("strict_8gib")
            self.assertEqual(len(attacks), 9)

    def test_06_real_subprocess_and_actual_worker_are_not_used_by_tests(self):
        source = SOURCE.read_text()
        self.assertIn("subprocess.run(command", source)
        self.assertIn('"--child-config", config', source)
        self.assertIn("row = M.one_shot_worker_entry(config)", source)
        self.assertIn("for ordinal, config in enumerate(CONFIGS)", source)
        self.assertIn("ATTEMPT_CONSUMED_BEFORE_CHILD", source)
        self.assertNotIn("PRODUCT_CAPTURE_TYPED_K8\", \"--child", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
