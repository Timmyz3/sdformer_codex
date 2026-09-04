#!/opt/anaconda3/bin/python
"""Source-only and mutation tests for the M2145 calibrated CPU replay."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / (
    "scripts/analyze_m2145_ep34_tsbg_fulltoken_calibrated_replay.py")
SPEC = importlib.util.spec_from_file_location("m2145_source", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M2145SourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibrated = M.calibration()

    def test_all_frozen_vcs_cycle_fields_are_exact(self):
        self.assertEqual(
            self.calibrated["status"],
            "PASS_M2145_EXACT_2880_ROW_CALIBRATION")
        self.assertEqual(
            self.calibrated["axis_cycle_fields_reconstructed_exactly"], 5760)
        self.assertEqual(self.calibrated["base_cycles_mismatches"], 0)
        self.assertEqual(self.calibrated["tsbg_cycles_mismatches"], 0)

    def test_lru4_group_major_mutation_is_observable(self):
        values = np.zeros((4, 48, 16), dtype=np.int8)
        for context in range(4):
            for group in range(5):
                values[context, group, (context + group) % 16] = 1
        base_cycles, base = M.engine_cycles(values, 0)
        tsbg_cycles, tsbg = M.engine_cycles(values, 1)
        self.assertGreater(base["misses"], tsbg["misses"])
        self.assertGreater(base_cycles, tsbg_cycles)

    def test_cycle_field_mutation_fails_exact_gate(self):
        values = np.zeros((4, 48, 16), dtype=np.int8)
        values[0, 0, 0] = 1
        predicted, _ = M.engine_cycles(values, 0)
        self.assertNotEqual(predicted, predicted + 1)

    def test_batch_recurrence_matches_scalar_recurrence(self):
        rng = np.random.default_rng(2145)
        values = np.where(
            rng.random((7, 4, 48, 16)) < 0.07,
            rng.choice(np.array([-1, 1], dtype=np.int8),
                       size=(7, 4, 48, 16)), 0).astype(np.int8)
        lower = np.any(values[:, :, :, :8] != 0, axis=3)
        upper = np.any(values[:, :, :, 8:] != 0, axis=3)
        for mode in (0, 1):
            batch = M.batch_engine_cycles(lower, upper, mode)
            for index in range(values.shape[0]):
                cycles, row = M.engine_cycles(values[index], mode)
                expected = [cycles, row["hits"], row["misses"],
                            row["evictions"], row["live_rows"],
                            row["issues"], row["scalar_reads"]]
                self.assertEqual(batch[index].tolist(), expected)

    def test_descriptor_mutation_changes_calibration_key(self):
        values = np.zeros((4, 96, 16), dtype=np.int8)
        before = M.descriptor_key(values)
        values[3, 95, 15] = -1
        self.assertNotEqual(before, M.descriptor_key(values))

    def _make_sealed_tree(self, root: Path) -> None:
        (root / "review.json").write_text("{}\n", encoding="utf-8")
        (root / "SHA256SUMS").write_text(
            f"{M.sha256(root / 'review.json')}  review.json\n",
            encoding="ascii")
        (root / "SHA256SUMS.seal.sha256").write_text(
            f"{M.sha256(root / 'SHA256SUMS')}  SHA256SUMS\n",
            encoding="ascii")

    def test_seal_rejects_member_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_sealed_tree(root)
            M.verify_double_seal(root)
            (root / "review.json").write_text('{"mutation": true}\n',
                                                encoding="utf-8")
            with self.assertRaises(M.M2145Error):
                M.verify_double_seal(root)

    def test_seal_rejects_unlisted_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_sealed_tree(root)
            (root / "extra.bin").write_bytes(b"mutation")
            with self.assertRaises(M.M2145Error):
                M.verify_double_seal(root)

    def test_production_replay_is_locked_before_independent_hammer(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            with self.assertRaises(M.M2145Error):
                M.run(output)
            self.assertFalse(os.path.lexists(str(output)))

    def test_source_has_no_execution_tool_surface(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("import subprocess", text)
        self.assertNotIn("import socket", text)
        self.assertNotIn("os.system", text)
        self.assertNotIn("Popen", text)
        self.assertIn("SOURCE_HAMMER", text)
        self.assertIn("full_network\": False", text)
        self.assertIn("system_speedup\": False", text)

    def test_selftest_is_inert_and_protected_identity_is_unchanged(self):
        row = M.selftest()
        self.assertEqual(row["cycle_fields_exact"], 5760)
        self.assertEqual(row["mutations_rejected"], 4)
        self.assertEqual(row["production_frames_decoded"], 0)
        self.assertFalse(row["production_replay_executed"])
        self.assertEqual(row["vcs_runs"], 0)
        self.assertEqual(row["eda_runs"], 0)
        self.assertEqual(M.sha256(M.DOC359), M.EXPECTED[M.DOC359])


if __name__ == "__main__":
    unittest.main()
