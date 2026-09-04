#!/usr/bin/env python3
"""M1227 source-only tests: no GPU, checkpoint load, capture, EDA, or release."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1227_motion_final_checkpoint_unified_hardware_r1.py")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1227_motion_final_checkpoint_unified_capture_source_contract_r1_20260830.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SPEC = importlib.util.spec_from_file_location("m1227_source_under_test", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def static_inventory():
    result = {}
    for category, count in M.EXPECTED_STATIC_COUNTS.items():
        if category == "atlif":
            result[category] = list(M.DEAD_SN_V) + ["live.atlif.{}".format(i) for i in range(93)]
        else:
            result[category] = ["{}.{}".format(category, i) for i in range(count)]
    return result


def live_records(live, samples):
    return [
        {"global_sample_id": sample, "category": category, "name": name}
        for sample in samples for category, names in live.items() for name in names
    ]


class ProfilerFixture(object):
    def __init__(self):
        self.execution_records = [{"sample_id": 0, "kind": "fixture"}]
        self.operator_records = {"op": {"name": "op", "calls": 1}}
        self.atlif_records = {"live": {"calls": 1}}


class M1227SourceTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.static = static_inventory()
        self.live = M.expected_live_inventory(self.static)

    def assert_rejected(self, function, *args):
        with self.assertRaises(M.M1227Error):
            function(*args)

    def test_01_policy_source_test_and_docs_are_exact(self):
        self.assertEqual(self.policy["schema"], M.SOURCE_SCHEMA)
        self.assertEqual(self.policy["status"], M.SOURCE_STATUS)
        self.assertEqual(self.policy["source"]["sha256"], digest(SOURCE))
        self.assertEqual(self.policy["test"]["sha256"], digest(Path(__file__).resolve()))
        self.assertEqual(digest(M.DOCS359), M.DOCS359_SHA256)
        self.assertFalse(self.policy["claim_boundary"]["production_authorized"])

    def test_02_import_is_lazy_and_heavy_dependency_free(self):
        code = (
            "import importlib.util,sys;"
            "p={!r};s=importlib.util.spec_from_file_location('isolated_m1227',p);"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "print(int('torch' in sys.modules),int('numpy' in sys.modules),"
            "int('m1227_sealed_m1174' in sys.modules))"
        ).format(str(SOURCE))
        output = subprocess.check_output([sys.executable, "-c", code]).decode().strip()
        self.assertEqual(output, "0 0 0")

    def test_03_static_and_live_topology_is_exact(self):
        self.assertEqual(sum(M.EXPECTED_STATIC_COUNTS.values()), 259)
        self.assertEqual(M.EXPECTED_STATIC_COUNTS["atlif"], 105)
        self.assertEqual(sum(M.EXPECTED_LIVE_COUNTS.values()), 247)
        self.assertEqual(M.EXPECTED_LIVE_COUNTS["atlif"], 93)
        self.assertEqual(len(M.DEAD_SN_V), 12)
        self.assertFalse(set(M.DEAD_SN_V) & set(self.live["atlif"]))

    def test_04_exact_sample_module_matrix_passes(self):
        records = live_records(self.live, range(40))
        audit = M.audit_call_matrix(records, self.live, range(40))
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["records"], 9880)

    def test_05_missing_live_call_is_rejected(self):
        records = live_records(self.live, [0])
        self.assertEqual(M.audit_call_matrix(records[:-1], self.live, [0])["status"], "FAIL")

    def test_06_duplicate_live_call_is_rejected(self):
        records = live_records(self.live, [0])
        self.assertEqual(M.audit_call_matrix(records + records[:1], self.live, [0])["status"], "FAIL")

    def test_07_dead_sn_v_call_is_rejected(self):
        records = live_records(self.live, [0])
        records.append({"global_sample_id": 0, "category": "atlif", "name": M.DEAD_SN_V[0]})
        audit = M.audit_call_matrix(records, self.live, [0])
        self.assertEqual(audit["status"], "FAIL")
        self.assertTrue(any(item.startswith("dead_module_fired") for item in audit["errors"]))

    def test_08_wrong_category_and_sample_are_rejected(self):
        records = live_records(self.live, [0])
        changed = [dict(row) for row in records]
        changed[0]["category"] = "atlif"
        self.assertEqual(M.audit_call_matrix(changed, self.live, [0])["status"], "FAIL")
        changed = [dict(row) for row in records]
        changed[0]["global_sample_id"] = 9
        self.assertEqual(M.audit_call_matrix(changed, self.live, [0])["status"], "FAIL")

    def test_09_attention_cartesian_population_and_mutations(self):
        records = [{"sample_id": sample, "name": name}
                   for sample in range(2) for name in M.ATTENTION_ALIASES]
        self.assertEqual(M.audit_attention_population(records, samples=2)["records"], 24)
        self.assert_rejected(M.audit_attention_population, records[:-1], 2)
        self.assert_rejected(M.audit_attention_population, records[:-1] + records[:1], 2)
        changed = [dict(row) for row in records]
        changed[0]["name"] = "S9.B9.attn"
        self.assert_rejected(M.audit_attention_population, changed, 2)

    def test_10_atomic_snapshot_is_forensic_and_collision_safe(self):
        records = live_records(self.live, [0])
        audit = M.audit_call_matrix(records, self.live, [0])
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = M.atomic_sample_snapshot(root, 0, records, ProfilerFixture(), audit)
            manifest = M.strict_json(result / "snapshot_manifest.json")
            self.assertEqual(manifest["status"], "SAMPLE_COMPLETE__FORENSIC_ONLY__NOT_CANONICAL")
            self.assertFalse(manifest["claim_boundary"]["canonical"])
            for member, expected in manifest["files"].items():
                self.assertEqual(digest(result / member), expected)
            self.assert_rejected(M.atomic_sample_snapshot, root, 0, records,
                                 ProfilerFixture(), audit)

    def test_11_payload_exact_population_and_delete_mutation(self):
        with tempfile.TemporaryDirectory() as name:
            staging = Path(name)
            payloads = staging / "payloads"
            payloads.mkdir()
            hashes = [hashlib.sha256(item.encode()).hexdigest()[:12]
                      for item in M.C1_TARGETS + M.DECODER_TARGETS]
            for sample in range(40):
                for index, value in enumerate(hashes):
                    for suffix in ("fp32.zlib", "support_sign.le.bitpack"):
                        (payloads / "s{:02d}_o{:05d}_{}.{}".format(
                            sample, index, value, suffix)).write_bytes(b"x")
            self.assertEqual(len(M.validate_payload_population(staging)), 640)
            next(payloads.iterdir()).unlink()
            self.assert_rejected(M.validate_payload_population, staging)

    def test_12_recursive_seal_detects_tamper_and_extra(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "nested").mkdir()
            member = root / "nested/member.txt"
            member.write_text("exact\n", encoding="utf-8")
            M.write_double_seal(root)
            self.assertEqual(len(M.verify_double_seal(root)), 1)
            member.write_text("tampered\n", encoding="utf-8")
            self.assert_rejected(M.verify_double_seal, root)

    def test_13_source_contract_cannot_launch(self):
        self.assert_rejected(M.validate_launch_contract, self.policy, CONTRACT)

    def test_14_namespaces_are_fresh_and_not_ep29_bound(self):
        namespace = self.policy["future_release"]["namespaces"]
        self.assertIn("m1227", namespace["result"])
        self.assertIn("m1227", namespace["attempt_marker"])
        self.assertIn("m1227", namespace["production_log"])
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("== 29", text)
        self.assertNotIn("ep29", text.lower())

    def test_15_m1224_and_substrate_are_exactly_bound(self):
        review = M.validate_m1224()
        self.assertEqual(review["root_cause"]["arithmetic"]
                         ["runtime_live_unified_hook_modules_per_sample"], 247)
        self.assertEqual(digest(M.SUBSTRATE), M.SUBSTRATE_SHA256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
