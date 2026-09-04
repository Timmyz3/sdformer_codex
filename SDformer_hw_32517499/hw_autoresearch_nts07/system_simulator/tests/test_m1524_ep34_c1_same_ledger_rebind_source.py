#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ROOT = HW.parent
SOURCE = HW / "system_simulator/scripts/build_m1524_ep34_c1_same_ledger_rebind_source.py"
OLD_M40 = HW / "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822"
OLD_M40_MANIFEST = OLD_M40 / "m40_bottleneck_packed_source_manifest.json"
OLD_ROWS = HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"


def load_source():
    spec = importlib.util.spec_from_file_location("m1524_source_under_test", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import M1524 source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_source()


class M1524Tests(unittest.TestCase):
    def test_01_default_audit_is_source_only(self):
        value = M.audit()
        self.assertEqual(value["status"], M.STATUS)
        self.assertEqual(value["identity"]["retained_records"], 40)
        self.assertEqual(value["geometry"]["source_rows"], 51_840_000)
        self.assertEqual(value["capacity_coordinate"]["candidate_macro_rounded_bytes"],
                         213_376)
        self.assertFalse(value["rebind_decision"]
                         ["old_ep35_cycle_numerators_or_denominators_reusable"])
        self.assertTrue(value["rebind_decision"]["new_ep34_row_ledger_required"])
        self.assertFalse(value["claim_boundary"]["production"])
        self.assertFalse(value["claim_boundary"]["cycles"])

    def test_02_exact_capture_population_and_payload_seals(self):
        records, entries = M.collect_records()
        self.assertEqual(len(records), 40)
        self.assertEqual({row["name"] for row in records}, set(M.MODULES))
        self.assertEqual([records[index * 4]["sample_key"] for index in range(10)],
                         list(M.EXPECTED_SAMPLE_KEYS))
        for row in records:
            self.assertEqual(entries[row["payload"]["support_sign"]],
                             row["payload"]["support_sign_sha256"])
            self.assertEqual(entries[row["payload"]["compressed_fp32"]],
                             row["payload"]["compressed_sha256"])

    def test_03_support_decode_and_population(self):
        records, _ = M.collect_records()
        for row in records:
            support = M.decode_support(row)
            self.assertEqual(support.shape, (10, 768, 15, 20))
            self.assertEqual(int(support.sum()), row["input"]["active"])

    def test_04_exact_two_code_numeric_binding(self):
        records, _ = M.collect_records()
        audit = M.numeric_codebook_audit(records)
        self.assertEqual(len(audit), 4)
        self.assertTrue(all(row["two_code_exact"] for row in audit))
        self.assertEqual(tuple(row["nonzero_bits_hex"] for row in audit),
                         tuple("{:08x}".format(value) for value in M.NONZERO_BITS))
        self.assertEqual(tuple(row["nonzero_count"] for row in audit),
                         M.NONZERO_COUNTS)

    def test_05_checkpoint_c1_weight_identity(self):
        weights = M.checkpoint_weight_audit()
        self.assertEqual(len(weights), 4)
        self.assertEqual(tuple(row["module"] for row in weights), M.MODULES)
        self.assertEqual(tuple(row["content_sha256"] for row in weights),
                         M.WEIGHT_SHA256)
        self.assertTrue(all(row["shape"] == [768, 768, 3, 3] and
                            row["bias"] is None for row in weights))

    def test_06_synthetic_k3_unfold_geometry(self):
        support = np.zeros((10, 768, 15, 20), dtype=np.bool_)
        support[0, 0, 7, 8] = True
        masks = M.phase_masks(support, 0).reshape(10, 15, 20)
        expected = np.zeros_like(masks)
        for kernel_y in range(3):
            for kernel_x in range(3):
                output_y = 7 - kernel_y + 1
                output_x = 8 - kernel_x + 1
                expected[0, output_y, output_x] |= np.uint16(
                    1 << (kernel_y * 3 + kernel_x))
        np.testing.assert_array_equal(masks, expected)
        lines = M.m528_compatible_lines(masks.reshape(-1))
        self.assertEqual(len(lines), 3000 * 9)
        self.assertTrue(all(len(line) == 8 for line in lines.splitlines()))

    def test_07_mapping_matches_frozen_m410_lower16(self):
        manifest = json.loads(OLD_M40_MANIFEST.read_text())
        record = manifest["records"][0]
        raw = (OLD_M40 / record["packed_file"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), record["packed_file_sha256"])
        plane = raw[:record["positive_plane_bytes"]]
        bits = np.unpackbits(np.frombuffer(plane, dtype=np.uint8), bitorder="little")
        support = bits.reshape(10, 768, 15, 20).astype(np.bool_, copy=False)
        with OLD_ROWS.open("rb") as stream:
            for partition in (0, 1, 27, 215, 431):
                stream.seek(partition * 3000 * 9)
                frozen = np.fromiter(
                    (int(line, 16) & 0xffff for line in stream.read(3000 * 9).splitlines()),
                    dtype=np.uint16, count=3000)
                np.testing.assert_array_equal(M.phase_masks(support, partition), frozen)

    def test_08_source_has_no_production_or_external_action(self):
        tree = ast.parse(SOURCE.read_text())
        imported = {alias.name.split(".")[0]
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                    for alias in node.names}
        self.assertTrue({"subprocess", "socket", "requests", "paramiko"}.isdisjoint(imported))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr,
                                 {"write_text", "write_bytes", "mkdir", "rename",
                                  "replace", "unlink", "touch"})

    def test_09_protected_document_identity(self):
        self.assertEqual(M.sha256(M.DOCS359), M.DOCS359_SHA256)


if __name__ == "__main__":
    unittest.main()
