#!/opt/anaconda3/envs/pytorch310/bin/python3.10
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "hw_autoresearch_nts07/scripts/"
    "hammer_m1455_m1434_motion_ep34_live93_capture_result_source.py")
BASE_TEST = ROOT / (
    "hw_autoresearch_nts07/tests/"
    "test_hammer_m1333_m1327_final_ep34_capture_result_source.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = load("test_m1455_source", SOURCE)
OLD = load("test_m1455_base_fixture", BASE_TEST)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class Fixture:
    def __init__(self):
        self.old = OLD.BaseFixture()
        self.root = self.old.root
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["schema"] = "m1434_motion_ep34_live93_unified_hardware_capture_r1_v1"
        manifest["status"] = (
            "CAPTURE_COMPLETE__FRESH_M1434_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM")
        dead = {"kind": "H60_STATIC_BUT_RUNTIME_BYPASSED", "count": 12,
                "names": list(M.M1434.DEAD_SN2_Q),
                "terminal_lf_sha256": M.M1434.DEAD_SN2_Q_SHA256}
        final = manifest["m1227_runtime_contract"]["final_selection_identity"]
        manifest["m1434_runtime_contract"] = {
            "static_modules": 259, "static_atlif": 105,
            "static_atlif_terminal_lf_sha256": M.M1434.STATIC_ATLIF_SHA256,
            "live_modules_per_sample": 247, "live_atlif": 93,
            "live_atlif_terminal_lf_sha256": M.M1434.LIVE_ATLIF_SHA256,
            "dead_atlif": dead, "dead_calls_per_sample": 0,
            "ordered_records": 9880, "attention_records": 480,
            "payload_files": 640, "final_selection_identity": final}
        manifest["forensic_snapshots"] = {
            "samples": 40, "atomic_per_sample": True,
            "failure_forensic_only": True, "automatic_canonical_promotion": False}
        write_json(manifest_path, manifest)
        (self.root / "m1227_admission.json").unlink()
        admission = {
            "schema": "m1434_final_capture_admission_r1_v1", "status": "PASS",
            "ordered": 9880, "attention": 480, "payload_files": 640,
            "execution": 7360, "operator_rows": 79,
            "atlif_live_rows": 93, "atlif_static": 105,
            "static_atlif_terminal_lf_sha256": M.M1434.STATIC_ATLIF_SHA256,
            "live_atlif_terminal_lf_sha256": M.M1434.LIVE_ATLIF_SHA256,
            "dead_atlif": dead,
            "claim_boundary": {"capture_only": True, "paper_result": False,
                               "cycles": False, "speedup": False,
                               "energy": False, "ppa": False}}
        write_json(self.root / "m1434_admission.json", admission)
        live_names = M.live_inventory()["atlif"]
        write_json(self.root / "atlif_activity.json",
                   [{"name": name, "calls": 40} for name in live_names])
        OLD.seal(self.root)

    def close(self):
        self.old.close()


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = Fixture()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1455_case_")
        self.root = Path(self.temp.name) / "result"
        shutil.copytree(self.fixture.root, self.root)

    def tearDown(self):
        self.temp.cleanup()

    def reseal(self):
        OLD.seal(self.root)

    def reject(self):
        self.reseal()
        with mock.patch.object(M.M1434, "validate_snapshot_population_live93"), \
                mock.patch.object(M.M1401.M1338, "validate_retained_payloads",
                                  return_value=320), \
                mock.patch.object(M.M1401.M1338.OLD,
                                  "validate_attention_geometry",
                                  return_value={"records": 480}), \
                mock.patch.object(M.M1401.M1338,
                                  "validate_attention_exact_archive"):
            with self.assertRaises(M.M1455Error):
                M.validate_result(self.root)

    def test_01_positive_full_live93_fixture(self):
        with mock.patch.object(M.M1434, "validate_snapshot_population_live93") as snapshots, \
                mock.patch.object(M.M1401.M1338, "validate_retained_payloads",
                                  return_value=320) as retained, \
                mock.patch.object(M.M1401.M1338.OLD,
                                  "validate_attention_geometry",
                                  return_value={"records": 480}), \
                mock.patch.object(M.M1401.M1338,
                                  "validate_attention_exact_archive"):
            result = M.validate_result(self.root)
        snapshots.assert_called_once_with(self.root)
        retained.assert_called_once()
        self.assertEqual(result["population"]["ordered"], 9880)
        self.assertEqual(result["population"]["atlif"], 93)
        self.assertEqual(result["population"]["forensic_snapshots"], 40)

    def test_02_complete_graph_duplicate_rejected(self):
        path = self.root / "unified_ordered_records.jsonl"
        rows = path.read_text().splitlines()
        row = json.loads(rows[8000]); row["global_order"] = 7999
        rows[8000] = json.dumps(row, sort_keys=True)
        path.write_text("\n".join(rows) + "\n")
        self.reject()

    def test_03_bool_global_order_rejected(self):
        path = self.root / "unified_ordered_records.jsonl"
        rows = path.read_text().splitlines()
        row = json.loads(rows[4]); row["global_order"] = True
        rows[4] = json.dumps(row, sort_keys=True)
        path.write_text("\n".join(rows) + "\n")
        self.reject()

    def test_04_dead_sn2q_call_rejected(self):
        path = self.root / "unified_ordered_records.jsonl"
        rows = path.read_text().splitlines()
        row = json.loads(rows[0]); row["category"] = "atlif"
        row["name"] = M.M1434.DEAD_SN2_Q[0]
        rows[0] = json.dumps(row, sort_keys=True)
        path.write_text("\n".join(rows) + "\n")
        self.reject()

    def test_05_manifest_schema_rejected(self):
        path = self.root / "manifest.json"; value = json.loads(path.read_text())
        value["schema"] = "wrong"; write_json(path, value); self.reject()

    def test_06_dead_digest_rejected(self):
        path = self.root / "m1434_admission.json"; value = json.loads(path.read_text())
        value["dead_atlif"]["terminal_lf_sha256"] = "0" * 64
        write_json(path, value); self.reject()

    def test_07_claim_promotion_rejected(self):
        path = self.root / "manifest.json"; value = json.loads(path.read_text())
        value["claim_boundary"]["speedup"] = True
        write_json(path, value); self.reject()

    def test_08_recursive_unsealed_symlink_rejected(self):
        os.symlink("absent", self.root / "hidden_link")
        with self.assertRaises(Exception):
            M.validate_result(self.root)

    def test_09_missing_attention_payload_rejected(self):
        (self.root / "attention_qk/manifest.json").unlink()
        self.reject()

    def test_10_source_has_no_capture_or_remote_action(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("subprocess", text)
        self.assertNotIn("torch.cuda", text)
        self.assertNotIn("os.kill", text)
        self.assertNotIn('add_argument("--run"', text)


if __name__ == "__main__":
    unittest.main()
