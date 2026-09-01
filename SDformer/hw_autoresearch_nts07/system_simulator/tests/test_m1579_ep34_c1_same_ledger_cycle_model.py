#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/run_m1579_ep34_c1_same_ledger_cycle_model.py"


def load_source():
    spec = importlib.util.spec_from_file_location("m1579_under_test", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import M1579")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_source()


class M1579Tests(unittest.TestCase):
    def test_01_source_audit_binds_final_ep34_capture(self):
        value = M.source_audit()
        self.assertEqual(value["status"], "PASS_SOURCE_AUDIT__NO_EXECUTION")
        self.assertEqual(value["checkpoint_sha256"],
                         "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48")
        self.assertEqual(value["geometry"]["source_rows"], 51_840_000)
        self.assertEqual(value["geometry"]["ledger_bytes"], 466_560_000)
        self.assertFalse(value["old_ep35_cycles_reusable"])
        self.assertFalse(value["production"])

    def test_02_release_is_exactly_one_cpu_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "result"
            ledger = output / "rows.memh"
            release = base / "release.json"
            value = {
                "schema": M.RELEASE_SCHEMA,
                "status": M.RELEASE_STATUS,
                "source_sha256": M.sha256(M.SOURCE),
                "output": str(output),
                "ledger": str(ledger),
                "cpu_runs": 1,
                "gpu_runs": 0,
                "eda_runs": 0,
                "maximum_workers": 3,
                "frozen_inputs": {
                    "m1524": M.M1524_SHA256,
                    "m528": M.M528_SHA256,
                    "m505": M.M505_SHA256,
                    "m504": M.M504_SHA256,
                    "docs359": M.DOCS359_SHA256,
                },
            }
            release.write_text(json.dumps(value), encoding="utf-8")
            M.verify_release(release, output, ledger, 3)
            value["cpu_runs"] = 2
            release.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                M.verify_release(release, output, ledger, 3)

    def test_03_all_baselines_share_one_array_recurrence(self):
        _, m528 = M.load_modules()
        shape = (M.SAMPLES, M.OPERATORS, M.CHUNKS, M.PARTITIONS)
        arrays = {name: np.zeros(shape, dtype=np.int32)
                  for name in m528.FIELD_NAMES}
        arrays["row_count"].fill(64)
        arrays["input_nnz"].fill(7)
        arrays["search_rows"].fill(3)
        arrays["residual_nnz"].fill(4)
        arrays["exact_parent_rows"].fill(2)
        arrays["ideal_issue_cycles"].fill(6)
        arrays["m504_cycles"].fill(8)
        arrays["dead_cycles"].fill(7)
        arrays["combined_cycles"].fill(6)
        row = m528.cycle_row(arrays, 0, None)
        self.assertEqual(set(row), {
            "m468_strong_zero_cycles",
            "m473_same_coordinate_bit_cycles",
            "m473_fused_concurrent_1r1w_ceiling_cycles",
            "m504_all_write_1rw_cycles",
            "m505_dead_write_only_1rw_cycles",
            "m505_combined_pvrf_1rw_cycles",
        })
        ratio = m528.ratio_fields(row)["speedup_vs_m468_strong_zero"]
        self.assertEqual(ratio,
                         row["m468_strong_zero_cycles"] /
                         row["m505_dead_write_only_1rw_cycles"])
        self.assertGreater(ratio, 0.0)

    def test_04_claim_boundary_is_explicit_in_source(self):
        text = SOURCE.read_text(encoding="utf-8")
        for token in ("cycle_model", "rtl_cycle", "full_network",
                      "system_speedup", "wall_clock", "multi_sequence"):
            self.assertIn(token, text)
        self.assertNotIn("time.time(", text)
        self.assertNotIn("perf_counter(", text)

    def test_05_no_network_gpu_or_eda_action(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imported = {alias.name.split(".")[0]
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                    for alias in node.names}
        self.assertTrue({"socket", "requests", "paramiko", "torch"}.isdisjoint(imported))
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("dc_shell", text)
        self.assertNotIn("vcs ", text)
        self.assertNotIn("ssh ", text)

    def test_06_protected_document_identity(self):
        self.assertEqual(M.sha256(M.DOCS359), M.DOCS359_SHA256)


if __name__ == "__main__":
    unittest.main()
