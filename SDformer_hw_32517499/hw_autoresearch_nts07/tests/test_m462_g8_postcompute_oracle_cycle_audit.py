#!/usr/bin/env python3
"""CPU-only semantic and attack tests for the fail-closed M462 analyzer."""

from __future__ import print_function

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "hw_autoresearch_nts07/system_simulator/scripts/"
          "analyze_m462_h67_g8_ffn_postcompute_oracle_cycles.py")


def load_module():
    spec = importlib.util.spec_from_file_location("m462_analyzer_test", str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_module()


class M462CycleAuditTests(unittest.TestCase):

    def test_strict_token_and_t10_site_masks_are_not_interchangeable(self):
        strict = np.ones((10, 1, 2, 2), dtype=np.bool_)
        strict[9, 0, 0, 1] = False
        sites = M.all_t_site_mask(strict)
        self.assertEqual(int(strict.sum()), 39)
        self.assertEqual(int(sites.sum()), 3)
        broadcast = np.broadcast_to(sites[None, ...], strict.shape)
        self.assertEqual(int(broadcast.sum()), 30)

    def test_tau_zero_and_positive_boundary_are_literal(self):
        shape = (10, 1, 1, 1)
        finite = np.ones(shape, dtype=np.bool_)
        exact = np.zeros(shape, dtype=np.bool_)
        exact[0] = True
        rho = np.full(shape, 0.25, dtype=np.float64)
        rho[0] = 0.0
        rho[1] = 0.125
        rho[2] = np.nextafter(np.float64(0.125), np.float64(0.0))
        arrays = {"finite": finite, "f_exact_zero": exact, "rho": rho}
        strict0, equal0, inclusive0 = M.masks_for_tau(arrays, 0.0)
        self.assertTrue(np.array_equal(strict0, exact))
        self.assertTrue(np.array_equal(equal0, exact))
        self.assertTrue(np.array_equal(inclusive0, exact))
        strict, equal, inclusive = M.masks_for_tau(arrays, 0.125)
        self.assertTrue(strict[0, 0, 0, 0])
        self.assertFalse(strict[1, 0, 0, 0])
        self.assertTrue(equal[1, 0, 0, 0])
        self.assertTrue(strict[2, 0, 0, 0])
        self.assertTrue(inclusive[1, 0, 0, 0])

    def test_profile_normalization_is_integer_floor_per_role(self):
        self.assertEqual(M.normalize_profile_cycles(17, 13, 13), 17)
        self.assertEqual(M.normalize_profile_cycles(17, 6, 13), 7)
        with self.assertRaises(RuntimeError):
            M.normalize_profile_cycles(17, 14, 13)
        with self.assertRaises(RuntimeError):
            M.normalize_profile_cycles(17, 0, 0)

    def test_full_profile_and_atlif_mask_invariants(self):
        ledger = M.load_ffn_ledger(
            ROOT / "hw_autoresearch_nts07/results/"
            "motion_ffn_resident_fusion_opportunity_review_r1_20260824/"
            "ffn_pair_ledger.csv")
        denominators = {}
        for index, pair in enumerate(ledger):
            denominators[(pair, "fc1")] = 1000 + index
            denominators[(pair, "fc2")] = 2000 + index
        result = M.full_mask_invariants(ledger, denominators)
        self.assertEqual(result["linear"], 159784111)
        self.assertEqual(result["sn1_atlif"], 9120000)
        self.assertEqual(result["sn2_atlif"], 36480000)
        self.assertEqual(result["ffn_accounted"], 205384111)

    def test_array_receipt_binds_dtype_shape_and_bytes(self):
        value = np.arange(20, dtype=np.int32).reshape(10, 1, 1, 2)
        receipt = M.array_receipt(value)
        self.assertEqual(receipt["dtype"], "<i4")
        self.assertEqual(receipt["shape"], [10, 1, 1, 2])
        self.assertEqual(receipt["elements"], 20)
        self.assertEqual(receipt["bytes"], 80)
        changed = value.copy()
        changed[-1, -1, -1, -1] += 1
        self.assertNotEqual(receipt["logical_sha256"],
                            M.logical_array_sha256(changed))

    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text('{"x": 1, "x": 2}\n', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                M.strict_json(path)
            path.write_text('{"x": NaN}\n', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                M.strict_json(path)

    def test_manifest_rejects_duplicate_and_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            leaf = root / "leaf.bin"
            leaf.write_bytes(b"m462")
            digest = hashlib.sha256(b"m462").hexdigest()
            manifest = root / "manifest.sha256"
            manifest.write_text("{}  leaf.bin\n".format(digest),
                                encoding="utf-8")
            self.assertEqual(M.manifest_entries(root, manifest.name),
                             ["leaf.bin"])
            manifest.write_text("{}  leaf.bin\n{}  leaf.bin\n".format(
                digest, digest), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                M.manifest_entries(root, manifest.name)
            manifest.write_text("{}  ../leaf.bin\n".format(digest),
                                encoding="utf-8")
            with self.assertRaises(RuntimeError):
                M.manifest_entries(root, manifest.name)

    def test_claim_language_keeps_oracle_out_of_executable_system_claim(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"postcompute_oracle_only": True', text)
        self.assertIn('"executable_skip": False', text)
        self.assertIn('"system_speedup": False', text)
        self.assertIn('"delta_aee_available": False', text)
        self.assertNotIn('"system_speedup": True', text)


if __name__ == "__main__":
    unittest.main()
