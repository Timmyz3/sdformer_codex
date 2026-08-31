#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/hammer_m1333_m1327_final_ep34_capture_result_source.py"
M1323_TEST = ROOT / "hw_autoresearch_nts07/system_simulator/tests/test_m1323_ep34_decoder_capture_adapter_source.py"
spec = importlib.util.spec_from_file_location("test_m1333_source", SOURCE)
M = importlib.util.module_from_spec(spec); sys.modules[spec.name] = M; spec.loader.exec_module(M)
helper_spec = importlib.util.spec_from_file_location("test_m1333_m1323_fixture", M1323_TEST)
H = importlib.util.module_from_spec(helper_spec); sys.modules[helper_spec.name] = H
helper_spec.loader.exec_module(H)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def seal(root: Path) -> None:
    for path in (root / M.MANIFEST, root / M.OUTER):
        if path.exists() or path.is_symlink():
            path.unlink()
    # Deliberately emulate ordinary rglob seal creation.  The M1333 verifier,
    # not this fixture helper, must independently catch hidden broken links.
    members = sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                     if path.is_file() and path.name not in {M.MANIFEST, M.OUTER})
    manifest = root / M.MANIFEST
    manifest.write_text("".join("{}  {}\n".format(M.sha256(root / name), name)
                                for name in members), encoding="utf-8")
    (root / M.OUTER).write_text("{}  {}\n".format(M.sha256(manifest), M.MANIFEST),
                               encoding="ascii")


class BaseFixture:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1333_base_")
        self.root = Path(self.temp.name) / "result"
        self.root.mkdir()
        ordered, _inventory, _cohort = H.make_ordered()
        selected = {"candidate_id": "resume_ep34", "epoch": 34,
                    "checkpoint": {"sha256": M.OLD.CHECKPOINT_SHA256},
                    "configuration": {"sha256": M.OLD.CONFIG_SHA256},
                    "profile": {"sha256": M.OLD.PROFILE_SHA256, "samples": 825,
                                "module_counts": {"ATLIFTernaryPSN": 105,
                                                  "ShiftmaxAttention": 12}}}
        final = {"epoch": 34, "checkpoint_sha256": M.OLD.CHECKPOINT_SHA256,
                 "config_sha256": M.OLD.CONFIG_SHA256,
                 "profile_sha256": M.OLD.PROFILE_SHA256,
                 "selection_sha256": M.OLD.SELECTION_SHA256}
        self.manifest = {
            "schema": "m1227_motion_final_checkpoint_unified_hardware_capture_r1_v1",
            "status": "CAPTURE_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
            "identity": {"contract_sha256": M.OLD.RUNTIME_SHA256,
                         "selection": {"selected": selected},
                         "checkpoint_load_audit": {"missing_count": 0,
                                                   "unexpected_count": 0},
                         "module_counts": {"ATLIFTernaryPSN": 105,
                                           "ShiftmaxAttention": 12}},
            "cohort": {"samples": copy.deepcopy(M.OLD.expected_cohort())},
            "m1227_runtime_contract": {"static_modules": 259, "static_atlif": 105,
                "live_modules_per_sample": 247, "live_atlif": 93,
                "dead_sn_v": list(M.M1227.DEAD_SN_V), "dead_calls_per_sample": 0,
                "ordered_records": 9880, "attention_records": 480,
                "payload_files": 640, "final_selection_identity": final},
            "claim_boundary": {"capture_only": True, "accuracy": False,
                "cycles": False, "speedup": False, "system_speedup": False,
                "energy": False, "rtl": False, "ppa": False,
                "fresh_result_hammer_required": True},
        }
        admission = {"schema": "m1227_final_capture_admission_r1_v1",
            "status": "PASS", "ordered": 9880, "attention": 480,
            "payload_files": 640, "execution": 7360, "operator_rows": 79,
            "atlif_live_rows": 93, "atlif_static": 105,
            "dead_sn_v": list(M.M1227.DEAD_SN_V),
            "claim_boundary": {"capture_only": True, "paper_result": False,
                "cycles": False, "speedup": False, "energy": False, "ppa": False}}
        write_json(self.root / "manifest.json", self.manifest)
        write_json(self.root / "m1227_admission.json", admission)
        (self.root / "unified_ordered_records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered),
            encoding="utf-8")
        payload_root = self.root / "payloads"; payload_root.mkdir()
        for row in ordered:
            payload = row["payload"]
            if payload.get("retained") is True:
                for key in ("compressed_fp32", "support_sign"):
                    (self.root / payload[key]).write_bytes(b"x")
        records = []
        attention_root = self.root / "attention_qk"; attention_root.mkdir()
        for sample in range(40):
            for name in M.M1227.ATTENTION_ALIASES:
                filename = "sample{}_{}.npz".format(
                    sample, name.replace(".", "_").replace("/", "_"))
                path = attention_root / filename
                np.savez_compressed(path, q_bits_packed=np.array([1], dtype=np.uint8),
                                    k_bits_packed=np.array([1], dtype=np.uint8),
                                    gate_q17=np.array([1], dtype=np.uint16))
                records.append({"sample_id": sample, "name": name,
                                "file": str(path), "sha256": M.sha256(path)})
        write_json(attention_root / "manifest.json", {"records": records})
        write_json(self.root / "execution_trace.json", [{} for _ in range(7360)])
        write_json(self.root / "operator_runtime.json",
                   [{"name": "op.%d" % index, "calls": 40} for index in range(79)])
        write_json(self.root / "atlif_activity.json",
                   [{"name": "live.%d" % index, "calls": 40} for index in range(93)])
        (self.root / "RUN_COMPLETE.txt").write_text(
            "PASS_M1174_UNIFIED_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM\n")
        seal(self.root)

    def close(self):
        self.temp.cleanup()


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = BaseFixture()

    @classmethod
    def tearDownClass(cls):
        cls.base.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1333_case_")
        self.root = Path(self.temp.name) / "result"
        shutil.copytree(self.base.root, self.root)

    def tearDown(self):
        self.temp.cleanup()

    def rewrite_manifest(self):
        write_json(self.root / "manifest.json", self.base.manifest)

    def reject(self):
        seal(self.root)
        with self.assertRaises(M.M1333Error):
            M.validate_result(self.root)

    def test_01_positive_full_fixture(self):
        result = M.validate_result(self.root)
        self.assertEqual(result["population"], {"ordered": 9880, "attention": 480,
                         "payload": 640, "execution": 7360, "operator": 79, "atlif": 93})

    def test_02_broken_symlink_rejected_even_when_unsealed(self):
        os.symlink(self.root / "absent-target", self.root / "hidden-broken-link")
        with self.assertRaisesRegex(M.M1333Error, "symlink"):
            M.validate_result(self.root)

    def test_03_ordered_missing_global_order_rejected(self):
        path = self.root / "unified_ordered_records.jsonl"
        rows = path.read_text().splitlines(); row = json.loads(rows[0]); row.pop("global_order")
        rows[0] = json.dumps(row, sort_keys=True); path.write_text("\n".join(rows) + "\n")
        self.reject()

    def test_04_ordered_invented_frozen_identity_rejected(self):
        path = self.root / "unified_ordered_records.jsonl"
        rows = path.read_text().splitlines(); row = json.loads(rows[300]); row["name"] = "invented.module"
        rows[300] = json.dumps(row, sort_keys=True); path.write_text("\n".join(rows) + "\n")
        self.reject()

    def test_05_attention_cartesian_duplicate_rejected(self):
        path = self.root / "attention_qk/manifest.json"; value = json.loads(path.read_text())
        value["records"][1]["sample_id"] = value["records"][0]["sample_id"]
        value["records"][1]["name"] = value["records"][0]["name"]
        write_json(path, value); self.reject()

    def test_06_attention_record_sha_rejected(self):
        path = self.root / "attention_qk/manifest.json"; value = json.loads(path.read_text())
        value["records"][0]["sha256"] = "0" * 64
        write_json(path, value); self.reject()

    def test_07_attention_npz_content_rejected(self):
        manifest = json.loads((self.root / "attention_qk/manifest.json").read_text())
        payload = self.root / "attention_qk" / Path(manifest["records"][0]["file"]).name
        np.savez_compressed(payload, q_bits_packed=np.array([1], dtype=np.uint8))
        manifest["records"][0]["sha256"] = M.sha256(payload)
        write_json(self.root / "attention_qk/manifest.json", manifest); self.reject()

    def test_08_checkpoint_missing_key_rejected(self):
        value = copy.deepcopy(self.base.manifest)
        value["identity"]["checkpoint_load_audit"].pop("missing_count")
        write_json(self.root / "manifest.json", value); self.reject()

    def test_09_checkpoint_bool_or_string_rejected(self):
        for bad in (False, "0"):
            value = copy.deepcopy(self.base.manifest)
            value["identity"]["checkpoint_load_audit"]["unexpected_count"] = bad
            write_json(self.root / "manifest.json", value)
            self.reject()
            shutil.rmtree(self.root); shutil.copytree(self.base.root, self.root)

    def test_10_ep34_identity_rejected(self):
        value = copy.deepcopy(self.base.manifest)
        value["identity"]["selection"]["selected"]["epoch"] = 35
        write_json(self.root / "manifest.json", value); self.reject()

    def test_11_cohort_sha_rejected(self):
        value = copy.deepcopy(self.base.manifest)
        value["cohort"]["samples"][0]["sha256"] = "0" * 64
        write_json(self.root / "manifest.json", value); self.reject()

    def test_12_predecessor_failure_chain_exact(self):
        review = M.verify_failed_predecessor()
        self.assertEqual(review["status"], "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED")

    def test_13_missing_canonical_fails_without_creation(self):
        self.assertFalse(M.CANONICAL_RESULT.exists())
        with self.assertRaises(M.M1333Error):
            M.main(["--validate-canonical-result"])
        self.assertFalse(M.CANONICAL_RESULT.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
